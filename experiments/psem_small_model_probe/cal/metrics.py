"""Headline metrics + fixed-priority threshold selection (issue #117, Gate 2).

Exactly four headline metrics:

1. ``contamination_s_per_speech_h`` — exclusive non-anchor active seconds per
   active-speech hour. Decoder-dependent current-segment numerator: per
   episode over ``[evaluation_start, first valid CUT source_boundary)`` (full
   window when no valid CUT); denominator is the full-window active-speech
   hour as before. No new headline metric was added by the V2 speaker-change
   repair.
2. ``false_cuts`` — KEEP-expected episodes emitting >= 1 confirmed CUT
   (unchanged: any CUT counts, no transition gating on KEEP).
3. ``missed_rate`` — CUT-expected episodes emitting no *valid* CUT
   (transition-aware: premature-only episodes score missed=true).
4. ``replacement delay p50/p90`` — SPLIT into source-boundary error
   (``source_boundary_time - transition``) vs decision/emission delay
   (``decision_time - source_boundary_time``), over first *valid* CUT per
   detected CUT-expected episode.

Topology views ``A->A+B->A`` (KEEP) vs ``A->A+B->B`` (CUT) are always
reported separately. Frame AUPRC/F1, unbound fraction, and role-flip
agreement are diagnostics only — never headline, never selection inputs.

Fixed-priority threshold rule (single scalar per model x regime, frozen
after; no topology/corpus/episode-specific taus, no model-specific
persistence): (a) zero false cuts on KEEP calibration when achievable,
else fewest; (b) lowest missed clean-transfer rate; (c) lowest median
total replacement delay; deterministic fallthrough to the lowest tau.
"""

from __future__ import annotations

import statistics
from typing import Any

from experiments.psem_small_model_probe.adapter.decoder import CommonPersistenceDecoder
from experiments.psem_small_model_probe.cal.eval_semantics import (
    current_segment_contam_s,
    episode_role,
    split_cuts,
    stratum_of,
)

KEEP_TOPOLOGIES = frozenset({"A", "A->A+B->A", "overlap_return", "A+A+B"})
CUT_TOPOLOGIES = frozenset({"A->A+B->B"})
# C6-derived "A+B" is binding/uncertain: reported, excluded from calibration.
TAU_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05..0.95


def replay_decisions(
    frames: list[dict[str, Any]], tau: float, frame_ms: int
) -> tuple[list[dict[str, Any]], int]:
    """Replay recorded frames through the 500 ms decoder at one tau.

    Returns (confirmed CUT events, 300 ms sensitivity CUT_SENS count).
    """
    decoder = CommonPersistenceDecoder(frame_ms, confirmation_ms=500, sensitivity_ms=300)
    cuts: list[dict[str, Any]] = []
    sens = 0
    for frame in frames:
        out = decoder.update(frame, tau=tau)
        if out["action"] == "CUT":
            cuts.append(dict(out))
        elif out["action"] == "CUT_SENS":
            sens += 1
    return cuts, sens


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def score_episode(
    record: dict[str, Any], tau: float, frame_ms: int
) -> dict[str, Any]:
    """Per-episode outcome at one tau (KEEP/CUT sets by stratum, rev3).

    Role resolves stratum-first via :func:`episode_role` (topology is
    display-only); pre-rev3 records without ``stratum`` keep the exact
    rev2 topology semantics. CUT validity is transition-aware: only CUTs
    with ``source_boundary_time >= authoritative_transition_ms -
    CUT_TOLERANCE_MS`` count toward detection. Episode success := first
    valid CUT exists; premature-only episodes score missed=true +
    premature_cut=true (+ n_premature_cuts diagnostic). KEEP-episode
    false-cut usage is unchanged (any committed CUT = false cut). All
    delays are measured on the first *valid* CUT. ``contam_s`` is the
    decoder-dependent current-segment numerator (full window when no
    valid CUT); legacy records without ``gt_eval`` fall back to the
    stored full-window value.
    """
    cuts, sens = replay_decisions(record["frames"], tau, frame_ms)
    topology = record["topology"]
    transition = record["authoritative_transition_ms"]
    valid, premature = split_cuts(cuts, transition)
    role = episode_role(record, KEEP_TOPOLOGIES, CUT_TOPOLOGIES)
    first = valid[0] if valid else None
    gt_eval = record.get("gt_eval")
    if gt_eval is not None:
        contam = current_segment_contam_s(
            gt_eval, first["source_boundary_time"] if first else None
        )
    else:
        contam = record.get("contam_s", 0.0)
    return {
        "episode_id": record["episode_id"],
        "topology": topology,
        "stratum": stratum_of(record),
        "role": role,
        "tau": tau,
        "n_cuts": len(cuts),
        "n_valid_cuts": len(valid),
        "n_premature_cuts": len(premature),
        "premature_cut": bool(premature),
        "sens_hits": sens,
        "false_cut": role == "KEEP" and bool(cuts),
        "missed": role == "CUT" and first is None,
        "contam_s": contam,
        "src_err_ms": (first["source_boundary_time"] - transition) if first else None,
        "dec_delay_ms": (first["decision_time"] - first["source_boundary_time"])
        if first
        else None,
        "total_delay_ms": (first["decision_time"] - transition) if first else None,
    }


def aggregate(
    records: list[dict[str, Any]], tau: float, frame_ms: int
) -> dict[str, Any]:
    """Aggregate one tau over all episodes of a model x regime cell.

    KEEP/CUT/OTHER partition by per-episode ``role`` (stratum-first,
    topology fallback for pre-rev3 records); C6/OTHER stays headline-
    excluded (diagnostic ``n_other_topology`` count only).
    """
    episodes = [score_episode(r, tau, frame_ms) for r in records]
    keep = [e for e in episodes if e["role"] == "KEEP"]
    cut = [e for e in episodes if e["role"] == "CUT"]
    other = [e for e in episodes if e["role"] == "OTHER"]
    detected = [e for e in cut if not e["missed"]]
    src_err = [e["src_err_ms"] for e in detected if e["src_err_ms"] is not None]
    dec_delay = [e["dec_delay_ms"] for e in detected if e["dec_delay_ms"] is not None]
    total = [e["total_delay_ms"] for e in detected if e["total_delay_ms"] is not None]
    contam_s = sum(e["contam_s"] for e in episodes)
    speech_h = sum(r["active_speech_s"] for r in records) / 3600.0
    # Diagnostics only: frame anchor score vs GT any-speech label.
    labels = [1.0 if f["speech_gt"] else 0.0 for r in records for f in r["frames"]]
    scores = [f["anchor"] for r in records for f in r["frames"]]
    return {
        "tau": tau,
        "n_episodes": len(episodes),
        "n_keep": len(keep),
        "n_cut": len(cut),
        "n_other_topology": len(other),
        "contamination_s_per_speech_h": (contam_s / speech_h) if speech_h > 0 else None,
        "false_cuts": sum(1 for e in keep if e["false_cut"]),
        "missed": sum(1 for e in cut if e["missed"]),
        "missed_rate": (sum(1 for e in cut if e["missed"]) / len(cut)) if cut else None,
        "delay_src_err_p50": _percentile(src_err, 0.5),
        "delay_src_err_p90": _percentile(src_err, 0.9),
        "delay_dec_p50": _percentile(dec_delay, 0.5),
        "delay_dec_p90": _percentile(dec_delay, 0.9),
        "delay_total_p50": _percentile(total, 0.5),
        "delay_total_p90": _percentile(total, 0.9),
        "median_total_delay_ms": statistics.median(total) if total else None,
        "frame_auprc_diag": _average_precision(labels, scores),
        "frame_f1_diag": _f1(labels, [1.0 if s >= tau else 0.0 for s in scores]),
    }


def select_threshold(rows: list[dict[str, Any]]) -> tuple[float, str]:
    """Fixed-priority pick over per-tau aggregates; returns (tau, reason)."""
    attainable_zero = any(r["false_cuts"] == 0 for r in rows)

    def key(r: dict[str, Any]) -> tuple:
        return (
            0 if r["false_cuts"] == 0 else 1,
            r["false_cuts"],
            r["missed_rate"] if r["missed_rate"] is not None else float("inf"),
            r["median_total_delay_ms"]
            if r["median_total_delay_ms"] is not None
            else float("inf"),
            r["tau"],
        )

    best = min(rows, key=key)
    reason = (
        f"a) false_cuts={best['false_cuts']} "
        f"(zero attainable: {attainable_zero}); "
        f"b) missed={best['missed']}/{best['n_cut']} "
        f"(rate={best['missed_rate']}); "
        f"c) median_total_delay={best['median_total_delay_ms']}ms; "
        f"tau={best['tau']} lowest on remaining ties"
    )
    return best["tau"], reason


def split_selection_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Threshold-selection subsets: (KEEP-calibration, C2-clean-transfer).

    The miss objective runs on C2 ONLY, never C4: C4 (overlap-takeover)
    and C6 (uncertain) episodes are excluded from selection inputs.
    Stratum-less pre-rev3 records resolve by legacy topology into KEEP
    only (topology ``A`` conflates C1/C2, so no record without an
    explicit stratum may enter the C2 subset — fail-closed).
    """
    keep = [
        r for r in records
        if episode_role(r, KEEP_TOPOLOGIES, CUT_TOPOLOGIES) == "KEEP"
    ]
    c2 = [r for r in records if stratum_of(r) == "C2"]
    return keep, c2


def selection_aggregate(
    keep_records: list[dict[str, Any]],
    c2_records: list[dict[str, Any]],
    tau: float,
    frame_ms: int,
) -> dict[str, Any]:
    """Per-tau selection row: KEEP false cuts + C2-only miss objective."""
    keep_agg = aggregate(keep_records, tau, frame_ms) if keep_records else None
    c2_agg = aggregate(c2_records, tau, frame_ms) if c2_records else None
    return {
        "tau": tau,
        "keep_n": keep_agg["n_keep"] if keep_agg else 0,
        "keep_false_cuts": keep_agg["false_cuts"] if keep_agg else 0,
        "c2_n": c2_agg["n_cut"] if c2_agg else 0,
        "c2_missed": c2_agg["missed"] if c2_agg else 0,
        "c2_missed_rate": c2_agg["missed_rate"] if c2_agg else None,
        "median_total_delay_ms": c2_agg["median_total_delay_ms"] if c2_agg else None,
    }


def select_threshold_split(rows: list[dict[str, Any]]) -> tuple[float, str]:
    """Fixed-priority pick over split selection rows (rev3 CAL rule)."""
    attainable_zero = any(r["keep_false_cuts"] == 0 for r in rows)

    def key(r: dict[str, Any]) -> tuple:
        return (
            0 if r["keep_false_cuts"] == 0 else 1,
            r["keep_false_cuts"],
            r["c2_missed_rate"] if r["c2_missed_rate"] is not None else float("inf"),
            r["median_total_delay_ms"]
            if r["median_total_delay_ms"] is not None
            else float("inf"),
            r["tau"],
        )

    best = min(rows, key=key)
    reason = (
        f"a) keep_false_cuts={best['keep_false_cuts']} "
        f"(zero attainable: {attainable_zero}); "
        f"b) c2_missed={best['c2_missed']}/{best['c2_n']} "
        f"(rate={best['c2_missed_rate']}); "
        f"c) median_total_delay={best['median_total_delay_ms']}ms; "
        f"tau={best['tau']} lowest on remaining ties"
    )
    return best["tau"], reason


def _average_precision(labels: list[float], scores: list[float]) -> float | None:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    total_pos = sum(labels)
    if total_pos == 0 or not scores:
        return None
    hits = 0.0
    area = 0.0
    for rank, i in enumerate(order, 1):
        if labels[i] > 0:
            hits += 1.0
            area += hits / rank
    return area / total_pos


def _f1(labels: list[float], preds: list[float]) -> float | None:
    tp = sum(1 for l, p in zip(labels, preds) if l > 0 and p > 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p > 0)
    fn = sum(1 for l, p in zip(labels, preds) if l > 0 and p == 0)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else None
