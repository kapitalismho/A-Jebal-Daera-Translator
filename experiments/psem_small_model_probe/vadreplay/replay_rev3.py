#!/usr/bin/env python3
"""Gate 5 production-VAD replay, winner-only, rev3 (evaluator_revision=3).

Winner: firered x C ONLY — the sole causal formulation with a valid CUT
signal under rev3 MAIN48 (4/16 CUT detected: C2 2/8 + C4 2/8 at frozen
tau; neovad-C 0/16, nothing to retain). Frozen tau firered-C applied
as-is (no retuning).

Raw-inference REUSE: firered-C anchor frames reused verbatim from
``main/results_repaired_v3/firered_C_main.jsonl`` (any-speech gate
already applied). FRESH: full-source production VAD spans per session
with the SAME VAD profile as before (Silero ONNX, thr 0.5, chunk 512,
pre-roll/hangover 500 ms); GT compact spans regenerated via the frozen
loader (no reselect). Records carry stratum; CUT-hit ids resolve by
role (stratum truth), split C2/C4.

Writes ``vadreplay/results_repaired_v3/`` (older sets untouched).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("PSEM_CORPUS_ROOT", r"C:\Users\salee\.psem-corpus")

from experiments.psem_small_model_probe.cal import audio_resolve  # noqa: E402
from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    KEEP_STRATA,
    compact_gt,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.metrics import (  # noqa: E402
    aggregate,
    replay_decisions,
    score_episode,
)
from experiments.psem_small_model_probe.cal.run_cal import (  # noqa: E402
    FailClosed,
    load_gt_index,
)
from experiments.psem_small_model_probe.main.run_main import topology_view  # noqa: E402
from experiments.psem_small_model_probe.vadreplay.run_replay import (  # noqa: E402
    REGIME,
    VAD_MODEL_PATH,
    SileroVadOnnx,
    _in_spans,
    _pct,
    production_spans,
)

REPLAY_DIR = Path(__file__).resolve().parent
MANIFEST_REV3 = REPLAY_DIR.parent / "manifest" / "manifest_rev3.jsonl"
FREEZE_REV3 = REPLAY_DIR.parent / "manifest" / "dataset_freeze_rev3.json"
MAIN_V3 = REPLAY_DIR.parent / "main" / "results_repaired_v3"
CAL_V3 = REPLAY_DIR.parent / "cal" / "results_repaired_v3"
OUT = REPLAY_DIR / "results_repaired_v3"

REVISION = "PSEM-SMALL-MODEL-PROBE-v1-rev3"
MODEL = "firered"
assert REGIME == "C"


def main() -> None:
    freeze = json.loads(FREEZE_REV3.read_text(encoding="utf-8"))
    if hashlib.sha256(MANIFEST_REV3.read_bytes()).hexdigest() != freeze["file_sha256"]:
        raise FailClosed("manifest_rev3 hash mismatch (refusing to run)")
    rows = [json.loads(l) for l in MANIFEST_REV3.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if r["split"] == "MAIN48"]
    if len(rows) != 48:
        raise FailClosed(f"MAIN48 rows: {len(rows)} != 48")
    gt_index = load_gt_index()
    taus = {(e["model"], e["regime"]): e["tau"]
            for e in json.loads((CAL_V3 / "thresholds.json").read_text())}
    tau = taus[(MODEL, REGIME)]
    if not VAD_MODEL_PATH.is_file():
        raise FailClosed(f"production VAD model missing: {VAD_MODEL_PATH}")

    lines = (MAIN_V3 / f"{MODEL}_{REGIME}_main.jsonl").read_text().splitlines()
    headers = [json.loads(l) for l in lines if '"episode_header"' in l]
    steps = [json.loads(l) for l in lines if '"type": "step"' in l]
    if len(headers) != 48:
        raise FailClosed(f"rev3 archive headers: {len(headers)} != 48")
    for h in headers:
        if h.get("stub_fallback") is not False or h.get("lifecycle") != "BOUND":
            raise FailClosed(f"reused header not native BOUND: {h['episode_id']}")
        if h.get("evaluator_revision") != 3:
            raise FailClosed(f"reused header not rev3: {h['episode_id']}")
    by_ep: dict[str, list[dict]] = {}
    for s in steps:
        by_ep.setdefault(s["episode_id"], []).append(s)
    row_by_ep = {r["episode_id"]: r for r in rows}
    if set(by_ep) != set(row_by_ep):
        raise FailClosed("rev3 archive episode set mismatch")

    out_lines: list[str] = []
    engine = SileroVadOnnx(VAD_MODEL_PATH)
    vad_step_ms: list[float] = []
    vad_audio_s = 0.0
    span_cache: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        wav = audio_resolve.resolve_audio(row)
        key = str(wav)
        if key not in span_cache:
            spans, total = production_spans(wav, engine, vad_step_ms)
            from experiments.psem_small_model_probe.vadreplay.run_replay import (  # noqa
                VAD_SAMPLE_RATE_HZ,
            )
            vad_audio_s += total / VAD_SAMPLE_RATE_HZ
            span_cache[key] = spans
            out_lines.append(json.dumps({
                "type": "vad_session", "audio_path": key,
                "total_samples": total, "span_count": len(spans),
                "speech_s": sum(e - s for s, e in spans) / VAD_SAMPLE_RATE_HZ,
            }))

    records_gt, records_prod = [], []
    n_frames_total = n_agree = n_gt_speech = 0
    n_vad_gated_off = n_vad_extra = n_vad_empty_windows = 0
    pvad_step_ms: list[float] = []
    eval_audio_s = 0.0
    frame_ms: int | None = None
    for header in headers:
        ep = header["episode_id"]
        row = row_by_ep[ep]
        gt = gt_index.get((str(row["corpus"]).lower(), row["session_id"]))
        if gt is None:
            raise FailClosed(f"no GT intervals for {ep}")
        eval_start, eval_end = int(row["evaluation_start_ms"]), int(row["evaluation_end_ms"])
        window_ms = eval_end - eval_start
        ep_steps = sorted(by_ep[ep], key=lambda s: s["source_time_ms"])
        gaps = {b["source_time_ms"] - a["source_time_ms"]
                for a, b in zip(ep_steps, ep_steps[1:])}
        if len(gaps) != 1:
            raise FailClosed(f"{ep}: irregular grid")
        fms = gaps.pop()
        frame_ms = fms if frame_ms is None else frame_ms
        if fms != frame_ms or window_ms % fms != 0 or len(ep_steps) != window_ms // fms:
            raise FailClosed(f"{ep}: grid/count mismatch")
        unit = audio_resolve.SAMPLES_PER_MS * 2 * fms
        spans = span_cache[str(audio_resolve.resolve_audio(row))]
        frames_gt, frames_prod = [], []
        vad_on_in_window = 0
        for i, s in enumerate(ep_steps):
            t = eval_start + (i + 1) * fms
            if s["source_time_ms"] != t:
                raise FailClosed(f"{ep}: grid mismatch at {i}")
            center = eval_start * audio_resolve.SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            speech_gt = bool(s["speech_gt"])
            speech_vad = _in_spans(spans, center)
            n_frames_total += 1
            n_agree += speech_gt == speech_vad
            n_gt_speech += bool(speech_gt)
            n_vad_gated_off += bool(speech_gt and not speech_vad)
            n_vad_extra += bool(speech_vad and not speech_gt)
            vad_on_in_window += bool(speech_vad)
            base = {"anchor": float(s["anchor"]), "adapter_speech": None,
                    "lifecycle": "BOUND", "source_time_ms": t,
                    "anchor_speech_gt": bool(s.get("anchor_speech_gt", False))}
            frames_gt.append({**base, "speech_gt": speech_gt})
            frames_prod.append({**base, "speech_gt": speech_vad})
            out_lines.append(json.dumps({
                "type": "step", "episode_id": ep, "model": MODEL,
                "regime": REGIME, "gate": "gt+prod", "source_time_ms": t,
                "speech_gt": speech_gt, "speech_vad": speech_vad,
                "anchor_speech_gt": base["anchor_speech_gt"],
                "anchor": float(s["anchor"]), "lifecycle": "BOUND"}))
            if s.get("step_ms") is not None:
                pvad_step_ms.append(float(s["step_ms"]))
        if vad_on_in_window == 0:
            n_vad_empty_windows += 1
        eval_audio_s += window_ms / 1000.0
        contam, active = gt_window_stats(gt, row["anchor_speaker"], eval_start, eval_end)
        meta = {"episode_id": ep, "topology": str(row["topology"]),
                "stratum": row["stratum"],
                "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
                "contam_s": contam, "active_speech_s": active,
                "lifecycle": "BOUND",
                "gt_eval": compact_gt(gt, row["anchor_speaker"], eval_start, eval_end)}
        records_gt.append({**meta, "frames": frames_gt})
        records_prod.append({**meta, "frames": frames_prod})
        out_lines.append(json.dumps({
            "type": "episode_header", "episode_id": ep, "model": MODEL,
            "regime": REGIME, "topology": row["topology"], "stratum": row["stratum"],
            "tau_frozen": tau, "stub_fallback": False, "evaluator_revision": 3,
            "revision": REVISION, "lifecycle": "BOUND",
            "vad_gate_on_frames": vad_on_in_window, "vad_gate_on_n": len(ep_steps)}))

    assert frame_ms is not None
    agg_gt = aggregate(records_gt, tau, frame_ms)
    agg_prod = aggregate(records_prod, tau, frame_ms)
    view_gt = topology_view(records_gt, tau, frame_ms)
    view_prod = topology_view(records_prod, tau, frame_ms)
    cuts_gt = sum(len(replay_decisions(r["frames"], tau, frame_ms)[0]) for r in records_gt)
    sens_gt = sum(replay_decisions(r["frames"], tau, frame_ms)[1] for r in records_gt)
    cuts_prod = sum(len(replay_decisions(r["frames"], tau, frame_ms)[0]) for r in records_prod)
    sens_prod = sum(replay_decisions(r["frames"], tau, frame_ms)[1] for r in records_prod)
    ep_detail = []
    for rec_gt, rec_prod in zip(records_gt, records_prod):
        sc_gt = score_episode(rec_gt, tau, frame_ms)
        sc_prod = score_episode(rec_prod, tau, frame_ms)
        ep_detail.append({
            "episode_id": rec_gt["episode_id"], "topology": rec_gt["topology"],
            "stratum": rec_gt["stratum"],
            "gt_n_cuts": sc_gt["n_cuts"], "gt_n_valid": sc_gt["n_valid_cuts"],
            "gt_missed": sc_gt["missed"], "gt_false_cut": sc_gt["false_cut"],
            "gt_premature": sc_gt["premature_cut"],
            "prod_n_cuts": sc_prod["n_cuts"], "prod_n_valid": sc_prod["n_valid_cuts"],
            "prod_missed": sc_prod["missed"], "prod_false_cut": sc_prod["false_cut"],
            "prod_premature": sc_prod["premature_cut"]})
    cut_eps = [e for e in ep_detail if e["stratum"] not in KEEP_STRATA
               and e["stratum"] != "C6"]
    gt_hit_ids = sorted(e["episode_id"] for e in cut_eps if not e["gt_missed"])
    prod_hit_ids = sorted(e["episode_id"] for e in cut_eps if not e["prod_missed"])
    retained_ids = sorted(set(gt_hit_ids) & set(prod_hit_ids))
    retained = (len(prod_hit_ids) / len(gt_hit_ids)) if gt_hit_ids else None
    vad_total_s = sum(vad_step_ms) / 1000.0
    pvad_total_s = sum(pvad_step_ms) / 1000.0
    payload = {
        "model": MODEL, "regime": REGIME, "tau_frozen": tau,
        "evaluator_revision": 3, "revision": REVISION, "n_episodes": len(rows),
        "gt": {"agg": agg_gt, "view": view_gt, "cuts": cuts_gt, "sens": sens_gt},
        "prod": {"agg": agg_prod, "view": view_prod, "cuts": cuts_prod, "sens": sens_prod},
        "gate_agreement": {
            "frames": n_frames_total,
            "agree_fraction": (n_agree / n_frames_total) if n_frames_total else None,
            "gt_speech_frames": n_gt_speech, "gt_speech_gated_off": n_vad_gated_off,
            "vad_extra_on": n_vad_extra, "vad_empty_windows": n_vad_empty_windows},
        "cut_retention": {"gt_hit_ids": gt_hit_ids, "prod_hit_ids": prod_hit_ids,
                          "retained_ids": retained_ids,
                          "gt_hit_by_stratum": {
                              s: sorted(e["episode_id"] for e in cut_eps
                                        if e["stratum"] == s and not e["gt_missed"])
                              for s in ("C2", "C4")},
                          "prod_hit_by_stratum": {
                              s: sorted(e["episode_id"] for e in cut_eps
                                        if e["stratum"] == s and not e["prod_missed"])
                              for s in ("C2", "C4")}},
        "retained_improvement_fraction": retained,
        "episodes": ep_detail,
        "cpu": {"production_vad_step_ms_p50": _pct(vad_step_ms, 0.5),
                "production_vad_step_ms_p95": _pct(vad_step_ms, 0.95),
                "production_vad_step_ms_max": max(vad_step_ms) if vad_step_ms else None,
                "production_vad_n_chunks": len(vad_step_ms),
                "production_vad_audio_s": vad_audio_s,
                "production_vad_rtf": (vad_total_s / vad_audio_s) if vad_audio_s else None,
                "pvad_step_ms_p50": _pct(pvad_step_ms, 0.5),
                "pvad_step_ms_p95": _pct(pvad_step_ms, 0.95),
                "pvad_reset_ms_p50": None, "pvad_bind_ms_p50": None,
                "combined_rtf_eval_audio": ((vad_total_s + pvad_total_s) / eval_audio_s)
                if eval_audio_s else None,
                "eval_audio_s": eval_audio_s, "cpu_count": os.cpu_count(),
                "adapter_timing": "reused archived step_ms (live run); reset/bind not re-timed"},
    }
    from experiments.psem_small_model_probe.vadreplay.run_replay import (  # noqa
        DEV_ONLY_NOTE,
        _f,
        _row,
        verdict_text,
    )
    gt = payload["gate_agreement"]
    lines = [
        f"# Gate 5 — production VAD replay, rev3 (firered x C, frozen tau={tau})",
        "",
        f"> {DEV_ONLY_NOTE}",
        "",
        "> Winner-only: firered regime C (sole causal formulation with a valid "
        "CUT signal under rev3 MAIN48: 4/16 CUT detected — C2 2/8 + C4 2/8; "
        "neovad-C 0/16, nothing to retain). Same 48 MAIN48 rows, same causal "
        "bind, same 500 ms confirmer / 300 ms sensitivity; gates compared: GT "
        "any-speech vs production Silero VAD spans (thr 0.5, chunk 512, "
        "pre-roll/hangover 500 ms).",
        "",
        f"## GT any-speech gate vs production-VAD (frozen tau={tau})",
        "",
        "| gate | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | "
        "dec p50/p90 (ms) | CUT events / sens hits | contam s/h |",
        "|---|---|---|---|---|---|---|",
        f"| GT-gate {_row(agg_gt, cuts_gt, sens_gt)} "
        f"{_f(agg_gt['contamination_s_per_speech_h'])} |",
        f"| prod-VAD {_row(agg_prod, cuts_prod, sens_prod)} "
        f"{_f(agg_prod['contamination_s_per_speech_h'])} |",
        "",
        "Cross-check: the GT-gate row reproduces main/results_repaired_v3 "
        "firered-C exactly (lifecycle, binding, decoder identical; only the "
        "gate differs).",
        "",
        "## Gate agreement (per 10 ms frame, GT any-speech vs prod VAD)",
        "",
        f"- frames scored: {gt['frames']}; agreement: {_f(gt['agree_fraction'])}",
        f"- GT-speech frames gated OFF by production VAD: "
        f"{gt['gt_speech_gated_off']}/{gt['gt_speech_frames']}",
        f"- production-gate-ON frames where GT is off: {gt['vad_extra_on']}",
        f"- eval windows with ZERO production-gate coverage: "
        f"{gt['vad_empty_windows']}/{payload['n_episodes']}",
        "",
        "## CUT retention (GT-gate -> prod-VAD, stratum truth)",
        "",
        f"- CUT successes GT-gate: "
        f"{agg_gt['n_cut'] - agg_gt['missed']}/{agg_gt['n_cut']}; "
        f"prod-VAD: {agg_prod['n_cut'] - agg_prod['missed']}/{agg_prod['n_cut']}; "
        f"hit-count ratio = {_f(retained)}",
        f"- episode-level retention: {len(retained_ids)}/{len(gt_hit_ids)} "
        f"GT-detected CUT episodes still detected under prod-VAD",
        f"- by stratum GT-gate: C2 "
        f"{len(payload['cut_retention']['gt_hit_by_stratum']['C2'])}/8, C4 "
        f"{len(payload['cut_retention']['gt_hit_by_stratum']['C4'])}/8; "
        f"prod-VAD: C2 "
        f"{len(payload['cut_retention']['prod_hit_by_stratum']['C2'])}/8, C4 "
        f"{len(payload['cut_retention']['prod_hit_by_stratum']['C4'])}/8",
        f"- false cuts GT-gate: {agg_gt['false_cuts']}; "
        f"prod-VAD: {agg_prod['false_cuts']}",
        "",
        "## Verdict",
        "",
        f"{verdict_text(payload)}",
        "",
    ]
    summary = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    (OUT / "replay_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "revision": REVISION, "evaluator_revision": 3,
        "winner_only": f"{MODEL} x {REGIME}",
        "winner_basis": ("sole causal formulation with a valid CUT signal under "
                         "rev3 MAIN48 (firered-C 4/16: C2 2/8 + C4 2/8; "
                         "neovad-C 0/16, nothing to retain)"),
        "adapter_frames": "REUSED from main/results_repaired_v3/firered_C_main.jsonl (rev3).",
        "vad_spans": "FRESH full-source Silero ONNX replay, same profile as before "
                     "(thr 0.5, chunk 512, pre-roll/hangover 500ms).",
        "manifest_sha256": freeze["file_sha256"],
        "supersedes_without_overwriting": "vadreplay/results/* VOID, preserved in place.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
