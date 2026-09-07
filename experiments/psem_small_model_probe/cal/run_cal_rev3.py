#!/usr/bin/env python3
"""rev3 CAL freeze: CAL12_rev3 (C1..C6 x2) x FireRed O/C + NeoVAD O/C.

FireRed: reuse valid raw anchor inference where the episode exists
(kept 10); FRESH native inference ONLY for the 2 new C2 episodes
(same frozen spans). NeoVAD: FRESH native inference for ALL CAL cells
(warm-up semantics changed — reuse FORBIDDEN); per-episode headers
must carry warmup_frames 500 (O) / 100 (C) at frame_ms=10, stub=false.

Thresholds: existing TAU_GRID, fixed priority with the miss objective
on the C2-clean-transfer subset ONLY (KEEP false cuts first, then C2
miss rate, then C2 delay). Writes cal/results_repaired_v3/ (rev1/rev2
artifacts untouched).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.adapter.protocol import frame_bytes  # noqa: E402
from experiments.psem_small_model_probe.cal import audio_resolve  # noqa: E402
from experiments.psem_small_model_probe.cal.audio_resolve import SAMPLES_PER_MS  # noqa: E402
from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.metrics import (  # noqa: E402
    TAU_GRID,
    aggregate,
    replay_decisions,
    select_threshold_split,
    selection_aggregate,
    split_selection_records,
)
from experiments.psem_small_model_probe.cal.run_cal import load_gt_index  # noqa: E402

CAL_DIR = Path(__file__).resolve().parent
MANIFEST_REV3 = CAL_DIR.parent / "manifest" / "manifest_rev3.jsonl"
FREEZE_REV3 = CAL_DIR.parent / "manifest" / "dataset_freeze_rev3.json"
CAL12_REV3 = CAL_DIR / "CAL12_rev3.jsonl"
ARCHIVE = CAL_DIR / "results"
OUT = CAL_DIR / "results_repaired_v3"

REVISION = "PSEM-SMALL-MODEL-PROBE-v1-rev3"


class FailClosed(RuntimeError):
    pass


def verify_rev3():
    freeze = json.loads(FREEZE_REV3.read_text(encoding="utf-8"))
    actual = hashlib.sha256(MANIFEST_REV3.read_bytes()).hexdigest()
    if actual != freeze["file_sha256"]:
        raise FailClosed("manifest_rev3 hash mismatch (refusing to run)")
    return freeze


def load_cal12_rev3():
    rows = [json.loads(l) for l in CAL12_REV3.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if len(rows) != 12:
        raise FailClosed(f"CAL12_rev3 rows = {len(rows)}")
    per = Counter(r["stratum"] for r in rows)
    if dict(per) != {"C1": 2, "C2": 2, "C3": 2, "C4": 2, "C5": 2, "C6": 2}:
        raise FailClosed(f"CAL12_rev3 strata {dict(per)}")
    return rows


def load_archive(model, regime):
    """(headers, steps_by_ep) from frozen rev1/rev2 raw steps archive."""
    path = ARCHIVE / f"{model}_{regime}_steps.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [json.loads(l) for l in lines if '"episode_header"' in l]
    steps = [json.loads(l) for l in lines if '"type": "step"' in l]
    for h in headers:
        if h.get("stub_fallback") is not False or h.get("dry_run") not in (False, None):
            raise FailClosed(f"archive {model} {regime}: non-native header {h.get('episode_id')}")
        if h.get("lifecycle") != "BOUND":
            raise FailClosed(f"archive {model} {regime}: lifecycle={h.get('lifecycle')}")
    by_ep: dict[str, list[dict]] = {}
    for s in steps:
        if s.get("model") != model or s.get("regime") != regime:
            raise FailClosed(f"archive {model} {regime}: step cell mismatch")
        by_ep.setdefault(s["episode_id"], []).append(s)
    return headers, by_ep


def rebuild_reused(row, header, ep_steps, gt):
    """Rebuild one record from archived raw anchors (GT gates recomputed)."""
    eval_start, eval_end = int(row["evaluation_start_ms"]), int(row["evaluation_end_ms"])
    anchor = row["anchor_speaker"]
    ep_steps = sorted(ep_steps, key=lambda s: s["source_time_ms"])
    gaps = {b["source_time_ms"] - a["source_time_ms"]
            for a, b in zip(ep_steps, ep_steps[1:])}
    if len(gaps) != 1:
        raise FailClosed(f"{row['episode_id']}: irregular grid {sorted(gaps)}")
    fms = gaps.pop()
    if (eval_end - eval_start) % fms != 0 or len(ep_steps) != (eval_end - eval_start) // fms:
        raise FailClosed(f"{row['episode_id']}: count vs window mismatch")
    unit = SAMPLES_PER_MS * 2 * fms
    frames, out_steps = [], []
    for i, s in enumerate(ep_steps):
        t = eval_start + (i + 1) * fms
        if s["source_time_ms"] != t or s.get("lifecycle") != "BOUND":
            raise FailClosed(f"{row['episode_id']}: grid/lifecycle mismatch at {i}")
        center = eval_start * SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
        any_speech = gt_any_speech(gt, center)
        anchor_speech = gt_anchor_speech(gt, anchor, center)
        frames.append({"speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                       "anchor": float(s["anchor"]), "adapter_speech": None,
                       "lifecycle": "BOUND", "source_time_ms": t})
        out_steps.append({"type": "step", "episode_id": row["episode_id"],
                          "model": s["model"], "regime": s["regime"],
                          "source_time_ms": t, "speech_gt": any_speech,
                          "anchor_speech_gt": anchor_speech,
                          "anchor": float(s["anchor"]), "lifecycle": "BOUND"})
    contam, active = gt_window_stats(gt, anchor, eval_start, eval_end)
    out_header = {**header, "evaluator_revision": 3, "revision": REVISION}
    record = {"episode_id": row["episode_id"], "topology": str(row["topology"]),
              "stratum": row["stratum"],
              "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
              "frames": frames, "contam_s": contam, "active_speech_s": active,
              "lifecycle": "BOUND",
              "gt_eval": compact_gt(gt, anchor, eval_start, eval_end)}
    return record, out_header, out_steps, fms


def run_fresh(adapter, name, regime, row, gt):
    """Fresh native inference for one episode (fail-closed: no stub)."""
    frame_ms = int(adapter.frame_ms)
    eval_start, eval_end = int(row["evaluation_start_ms"]), int(row["evaluation_end_ms"])
    window_ms = eval_end - eval_start
    if window_ms % frame_ms != 0:
        raise FailClosed(f"{row['episode_id']}: window not a {frame_ms}ms multiple")
    n_frames = window_ms // frame_ms
    unit = frame_bytes(frame_ms)
    if regime == "C" and not row.get("causal_bindable", False):
        raise FailClosed(f"{row['episode_id']}: C regime but not causal_bindable")
    span_start, span_end = audio_resolve.span_for_regime(row, regime)
    wav = audio_resolve.resolve_audio(row)
    ref_pcm = audio_resolve.load_span(wav, span_start, span_end)
    ref_sha = audio_resolve.sha256_pcm(ref_pcm)
    eval_pcm = audio_resolve.load_span(wav, eval_start, eval_end)
    if len(ref_pcm) % unit != 0:
        raise FailClosed(f"{row['episode_id']}: bind span not frame-aligned")
    if len(eval_pcm) != n_frames * unit:
        raise FailClosed(f"{row['episode_id']}: eval window byte mismatch")
    adapter.reset()
    adapter.bind(ref_pcm)
    header = {"type": "episode_header", "episode_id": row["episode_id"],
              "model": name, "regime": regime, "topology": str(row["topology"]),
              "stratum": row["stratum"], "stub_fallback": False,
              "fallback_reason": "native weights", "dry_run": False,
              "lifecycle": "BOUND", "bind_span_sha256": ref_sha,
              "evaluator_revision": 3, "revision": REVISION}
    if hasattr(adapter, "episode_header"):
        for k, v in adapter.episode_header().items():
            header.setdefault(k, v)
    if name == "neovad":
        want = (span_end - span_start) // frame_ms
        if header.get("warmup_frames") != want:
            raise FailClosed(f"{row['episode_id']}: warmup_frames="
                             f"{header.get('warmup_frames')} want {want}")
    frames, out_steps = [], []
    for i in range(n_frames):
        out = adapter.step(eval_pcm[i * unit:(i + 1) * unit])
        t = eval_start + (i + 1) * frame_ms
        center = eval_start * SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
        any_speech = gt_any_speech(gt, center)
        anchor_speech = gt_anchor_speech(gt, row["anchor_speaker"], center)
        frames.append({"speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                       "anchor": float(out.anchor), "adapter_speech": out.speech,
                       "lifecycle": "BOUND", "source_time_ms": t})
        out_steps.append({"type": "step", "episode_id": row["episode_id"],
                          "model": name, "regime": regime, "source_time_ms": t,
                          "speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                          "anchor": float(out.anchor), "lifecycle": "BOUND"})
    contam, active = gt_window_stats(gt, row["anchor_speaker"], eval_start, eval_end)
    record = {"episode_id": row["episode_id"], "topology": str(row["topology"]),
              "stratum": row["stratum"],
              "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
              "frames": frames, "contam_s": contam, "active_speech_s": active,
              "lifecycle": "BOUND",
              "gt_eval": compact_gt(gt, row["anchor_speaker"], eval_start, eval_end)}
    return record, header, out_steps, frame_ms


def load_native_adapter(name):
    import importlib

    module = importlib.import_module(
        f"experiments.psem_small_model_probe.adapter.{name}")
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        candidate = getattr(module, attr)
        if (isinstance(candidate, type) and hasattr(candidate, "reset")
                and hasattr(candidate, "bind") and hasattr(candidate, "step")):
            try:
                return candidate()
            except Exception as exc:
                raise FailClosed(f"{name}: native construct failed: {exc}")
    raise FailClosed(f"{name}: no adapter class found (stub FORBIDDEN in rev3)")


def run_model_cell(name, regime, rows, gt_index, fresh_episodes):
    """Reuse archived raws except fresh_episodes (fresh native inference)."""
    arch_headers, arch_steps = load_archive(name, regime)
    arch_by_ep = {h["episode_id"]: h for h in arch_headers}
    adapter = None
    records, out_lines = [], []
    frame_ms = None
    for row in rows:
        ep = row["episode_id"]
        key = (str(row["corpus"]).lower(), row["session_id"])
        gt = gt_index.get(key)
        if gt is None:
            raise FailClosed(f"{ep}: no GT intervals")
        if ep in fresh_episodes:
            if adapter is None:
                adapter = load_native_adapter(name)
            record, header, steps, fms = run_fresh(adapter, name, regime, row, gt)
        else:
            if ep not in arch_steps:
                raise FailClosed(f"{name} {regime}: {ep} needs fresh inference "
                                 f"(not in archive, not flagged fresh)")
            record, header, steps, fms = rebuild_reused(
                row, arch_by_ep[ep], arch_steps[ep], gt)
        frame_ms = fms if frame_ms is None else frame_ms
        if fms != frame_ms:
            raise FailClosed(f"{ep}: frame_ms drift")
        records.append(record)
        out_lines.append(json.dumps(header))
        out_lines.extend(json.dumps(s) for s in steps)
    if len(records) != 12 or any(r["lifecycle"] != "BOUND" for r in records):
        raise FailClosed(f"{name} {regime}: cell incomplete")
    assert frame_ms is not None
    return records, out_lines, frame_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="rev3 CAL freeze")
    parser.add_argument("--results-dir", type=Path, default=OUT)
    args = parser.parse_args()

    freeze = verify_rev3()
    rows = load_cal12_rev3()
    gt_index = load_gt_index()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    new_c2 = {r["episode_id"] for r in rows if r["stratum"] == "C2"}
    if len(new_c2) != 2:
        raise FailClosed(f"new C2 episodes: {sorted(new_c2)}")
    # FireRed: fresh ONLY for the 2 new C2; NeoVAD: fresh for all 12.
    fresh = {"firered": new_c2,
             "neovad": {r["episode_id"] for r in rows}}

    thresholds, summary_cells = [], []
    for name in ("firered", "neovad"):
        for regime in ("O", "C"):
            records, out_lines, frame_ms = run_model_cell(
                name, regime, rows, gt_index, fresh[name])
            bound = [r for r in records if r["lifecycle"] == "BOUND"]
            per_tau = [aggregate(bound, tau, frame_ms) for tau in TAU_GRID]
            keep, c2 = split_selection_records(bound)
            if len(keep) != 6 or len(c2) != 2:
                raise FailClosed(f"{name} {regime}: selection subsets "
                                 f"keep={len(keep)} c2={len(c2)}")
            sel_rows = [selection_aggregate(keep, c2, tau, frame_ms) for tau in TAU_GRID]
            tau, why = select_threshold_split(sel_rows)
            sel = next(r for r in sel_rows if r["tau"] == tau)
            (args.results_dir / f"{name}_{regime}_steps.jsonl").write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8")
            (args.results_dir / f"{name}_{regime}_calibration.jsonl").write_text(
                "\n".join(json.dumps(r) for r in per_tau) + "\n", encoding="utf-8")
            (args.results_dir / f"{name}_{regime}_selection.jsonl").write_text(
                "\n".join(json.dumps(r) for r in sel_rows) + "\n", encoding="utf-8")
            cuts_at_tau = sum(len(replay_decisions(r["frames"], tau, frame_ms)[0])
                              for r in bound)
            sens_at_tau = sum(replay_decisions(r["frames"], tau, frame_ms)[1]
                              for r in bound)
            thresholds.append({"model": name, "regime": regime, "tau": tau,
                               "selection_reason": why,
                               "keep_false_cuts": sel["keep_false_cuts"],
                               "c2_missed": sel["c2_missed"], "c2_n": sel["c2_n"],
                               "c2_missed_rate": sel["c2_missed_rate"]})
            summary_cells.append({"model": name, "regime": regime, "tau": tau,
                                  "agg": next(r for r in per_tau if r["tau"] == tau),
                                  "sel": sel, "cuts_at_tau": cuts_at_tau,
                                  "sens_at_tau": sens_at_tau, "frame_ms": frame_ms})
    (args.results_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# CAL12 rev3 threshold freeze (PSEM-SMALL-MODEL-PROBE-v1-rev3, evaluator_revision=3)",
             "",
             "Fixed priority: (a) zero KEEP false cuts when achievable; "
             "(b) lowest C2-clean-transfer miss rate (C4 NEVER in miss objective); "
             "(c) lowest C2 median total delay.",
             "",
             "| model | regime | tau | keep_false/6 | c2_missed/2 | c2_rate | "
             "cut(all)/4 | contam s/h | cuts@tau | sens@tau |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cell in summary_cells:
        agg, sel = cell["agg"], cell["sel"]
        lines.append(
            f"| {cell['model']} | {cell['regime']} | {cell['tau']} | "
            f"{sel['keep_false_cuts']}/6 | {sel['c2_missed']}/2 | {sel['c2_missed_rate']} | "
            f"{agg['missed']}/{agg['n_cut']} | "
            f"{agg['contamination_s_per_speech_h']} | "
            f"{cell['cuts_at_tau']} | {cell['sens_at_tau']} |")
    (args.results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.results_dir / "provenance.json").write_text(json.dumps({
        "revision": REVISION, "evaluator_revision": 3,
        "manifest": "manifest/manifest_rev3.jsonl",
        "manifest_sha256": hashlib.sha256(MANIFEST_REV3.read_bytes()).hexdigest(),
        "freeze_sha256": freeze["freeze_sha256"],
        "cal_rows": "cal/CAL12_rev3.jsonl",
        "cal_sha256": hashlib.sha256(CAL12_REV3.read_bytes()).hexdigest(),
        "selection_rule": ("fixed-priority KEEP-false-cuts, then C2-only miss "
                           "rate (never C4), then C2 delay; C6 diagnostic-only"),
        "raw_inference": ("firered: REUSED archived raw anchors for kept 10, "
                          "FRESH native inference for 2 new C2; neovad: FRESH "
                          "native inference all 12 (warm-up semantics changed, "
                          "reuse forbidden)"),
        "stub_fallback": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(thresholds, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
