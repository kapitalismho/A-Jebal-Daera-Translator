#!/usr/bin/env python3
"""rev3 MAIN rescore: manifest_rev3 MAIN48 x frozen CAL-v3 taus.

FireRed O/C: replay raw anchor inference verbatim from
main/results/{model}_{regime}_main.jsonl (same 48 episodes; GT gates
recomputed, gt_eval attached, stratum carried). NeoVAD O/C: FRESH
native inference for all 48 (warm-up semantics changed — reuse
FORBIDDEN); per-episode warmup headers verified. Headlines read
false cuts/24 KEEP, missed/16 CUT, contam s/h, split delays, plus a
stratum breakdown (C2 detected/8 and C4 detected/8 separately, total
CUT/16). Writes main/results_repaired_v3/ (older sets untouched).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.cal import audio_resolve  # noqa: E402
from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    CUT_STRATA,
    KEEP_STRATA,
    OTHER_STRATA,
    compact_gt,
    gt_anchor_speech,
    gt_any_speech,
    gt_window_stats,
)
from experiments.psem_small_model_probe.cal.metrics import (  # noqa: E402
    TAU_GRID,
    aggregate,
    replay_decisions,
    score_episode,
)
from experiments.psem_small_model_probe.cal.run_cal import load_gt_index  # noqa: E402
from experiments.psem_small_model_probe.cal.run_cal_rev3 import (  # noqa: E402
    load_native_adapter,
    run_fresh,
)

MAIN_DIR = Path(__file__).resolve().parent
MANIFEST_REV3 = MAIN_DIR.parent / "manifest" / "manifest_rev3.jsonl"
FREEZE_REV3 = MAIN_DIR.parent / "manifest" / "dataset_freeze_rev3.json"
ARCHIVE = MAIN_DIR / "results"
CAL_V3 = MAIN_DIR.parent / "cal" / "results_repaired_v3"
OUT = MAIN_DIR / "results_repaired_v3"

REVISION = "PSEM-SMALL-MODEL-PROBE-v1-rev3"


class FailClosed(RuntimeError):
    pass


def load_main48():
    rows = [json.loads(l) for l in MANIFEST_REV3.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    main48 = [r for r in rows if r["split"] == "MAIN48"]
    if len(main48) != 48:
        raise FailClosed(f"MAIN48 rows = {len(main48)}")
    per = Counter(r["stratum"] for r in main48)
    if dict(per) != {"C1": 8, "C2": 8, "C3": 8, "C4": 8, "C5": 8, "C6": 8}:
        raise FailClosed(f"MAIN48 strata {dict(per)}")
    keep = sum(1 for r in main48 if r["stratum"] in KEEP_STRATA)
    cut = sum(1 for r in main48 if r["stratum"] in CUT_STRATA)
    other = sum(1 for r in main48 if r["stratum"] in OTHER_STRATA)
    if (keep, cut, other) != (24, 16, 8):
        raise FailClosed(f"MAIN48 KEEP/CUT/OTHER = {keep}/{cut}/{other}")
    return main48


def load_frozen_taus():
    prov = json.loads((CAL_V3 / "provenance.json").read_text(encoding="utf-8"))
    if prov.get("evaluator_revision") != 3 or prov.get("revision") != REVISION:
        raise FailClosed("CAL v3 provenance mismatch")
    taus = {}
    for entry in json.loads((CAL_V3 / "thresholds.json").read_text(encoding="utf-8")):
        taus[(entry["model"], entry["regime"])] = entry["tau"]
    for m, r in (("firered", "O"), ("firered", "C"), ("neovad", "O"), ("neovad", "C")):
        if (m, r) not in taus:
            raise FailClosed(f"missing CAL v3 tau for {m} {r}")
    return taus


def rebuild_firered(model, regime, rows, gt_index):
    """Replay archived FireRed raw anchors for the same 48 episodes."""
    path = ARCHIVE / f"{model}_{regime}_main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [json.loads(l) for l in lines if '"episode_header"' in l]
    steps = [json.loads(l) for l in lines if '"type": "step"' in l]
    if len(headers) != 48:
        raise FailClosed(f"{model} {regime}: {len(headers)} headers != 48")
    for h in headers:
        if h.get("stub_fallback") is not False:
            raise FailClosed(f"{model} {regime}: stub fallback present")
        if h.get("lifecycle") != "BOUND":
            raise FailClosed(f"{model} {regime}: lifecycle={h.get('lifecycle')}")
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for s in steps:
        if s.get("model") != model or s.get("regime") != regime:
            raise FailClosed(f"{model} {regime}: step cell mismatch")
        by_ep[s["episode_id"]].append(s)
    row_by_ep = {r["episode_id"]: r for r in rows}
    if set(by_ep) != set(row_by_ep):
        raise FailClosed(f"{model} {regime}: episode set mismatch")
    records, out_lines = [], []
    frame_ms = None
    for header in headers:
        ep = header["episode_id"]
        row = row_by_ep[ep]
        gt = gt_index.get((str(row["corpus"]).lower(), row["session_id"]))
        if gt is None:
            raise FailClosed(f"{ep}: no GT intervals")
        eval_start, eval_end = int(row["evaluation_start_ms"]), int(row["evaluation_end_ms"])
        anchor = row["anchor_speaker"]
        ep_steps = sorted(by_ep[ep], key=lambda s: s["source_time_ms"])
        gaps = {b["source_time_ms"] - a["source_time_ms"]
                for a, b in zip(ep_steps, ep_steps[1:])}
        if len(gaps) != 1:
            raise FailClosed(f"{ep}: irregular grid")
        fms = gaps.pop()
        frame_ms = fms if frame_ms is None else frame_ms
        if fms != frame_ms:
            raise FailClosed(f"{ep}: frame_ms drift")
        if (eval_end - eval_start) % fms != 0 or len(ep_steps) != (eval_end - eval_start) // fms:
            raise FailClosed(f"{ep}: count vs window mismatch")
        unit = audio_resolve.SAMPLES_PER_MS * 2 * fms
        frames = []
        for i, s in enumerate(ep_steps):
            t = eval_start + (i + 1) * fms
            if s["source_time_ms"] != t or s.get("lifecycle") != "BOUND":
                raise FailClosed(f"{ep}: grid/lifecycle mismatch at {i}")
            center = eval_start * audio_resolve.SAMPLES_PER_MS + (i * unit) // 2 + unit // 4
            any_speech = gt_any_speech(gt, center)
            anchor_speech = gt_anchor_speech(gt, anchor, center)
            frames.append({"speech_gt": any_speech, "anchor_speech_gt": anchor_speech,
                           "anchor": float(s["anchor"]), "adapter_speech": None,
                           "lifecycle": "BOUND", "source_time_ms": t})
            out_lines.append(json.dumps({
                "type": "step", "episode_id": ep, "model": model, "regime": regime,
                "source_time_ms": t, "speech_gt": any_speech,
                "anchor_speech_gt": anchor_speech, "anchor": float(s["anchor"]),
                "lifecycle": "BOUND", "step_ms": s.get("step_ms")}))
        contam, active = gt_window_stats(gt, anchor, eval_start, eval_end)
        out_lines.append(json.dumps({**header, "evaluator_revision": 3,
                                     "revision": REVISION, "stratum": row["stratum"]}))
        records.append({"episode_id": ep, "topology": str(row["topology"]),
                        "stratum": row["stratum"],
                        "authoritative_transition_ms": int(row["authoritative_transition_time_ms"]),
                        "frames": frames, "contam_s": contam, "active_speech_s": active,
                        "lifecycle": "BOUND",
                        "gt_eval": compact_gt(gt, anchor, eval_start, eval_end)})
    assert frame_ms is not None
    return records, out_lines, frame_ms


def run_neovad_fresh(regime, rows, gt_index):
    adapter = load_native_adapter("neovad")
    records, out_lines = [], []
    frame_ms = None
    for row in rows:
        gt = gt_index.get((str(row["corpus"]).lower(), row["session_id"]))
        if gt is None:
            raise FailClosed(f"{row['episode_id']}: no GT intervals")
        record, header, steps, fms = run_fresh(adapter, "neovad", regime, row, gt)
        frame_ms = fms if frame_ms is None else frame_ms
        if fms != frame_ms:
            raise FailClosed(f"{row['episode_id']}: frame_ms drift")
        records.append(record)
        out_lines.append(json.dumps(header))
        out_lines.extend(json.dumps(s) for s in steps)
    if len(records) != 48:
        raise FailClosed(f"neovad {regime}: {len(records)} records != 48")
    assert frame_ms is not None
    return records, out_lines, frame_ms


def stratum_breakdown(records, tau, frame_ms):
    eps = [score_episode(r, tau, frame_ms) for r in records]
    by_s: dict[str, list[dict]] = defaultdict(list)
    for e in eps:
        by_s[e["stratum"]].append(e)
    view = {}
    for s in ("C1", "C2", "C3", "C4", "C5", "C6"):
        group = by_s.get(s, [])
        view[s] = {"n": len(group),
                   "false_cuts": sum(1 for e in group if e["false_cut"]),
                   "missed": sum(1 for e in group if e["missed"]),
                   "detected": sum(1 for e in group
                                   if e["role"] == "CUT" and not e["missed"])}
    return view


def main() -> None:
    freeze = json.loads(FREEZE_REV3.read_text(encoding="utf-8"))
    actual = hashlib.sha256(MANIFEST_REV3.read_bytes()).hexdigest()
    if actual != freeze["file_sha256"]:
        raise FailClosed("manifest_rev3 hash mismatch (refusing to run)")
    rows = load_main48()
    taus = load_frozen_taus()
    gt_index = load_gt_index()
    OUT.mkdir(parents=True, exist_ok=True)

    cells = []
    for name in ("firered", "neovad"):
        for regime in ("O", "C"):
            if name == "firered":
                records, out_lines, frame_ms = rebuild_firered(name, regime, rows, gt_index)
            else:
                records, out_lines, frame_ms = run_neovad_fresh(regime, rows, gt_index)
            tau = taus[(name, regime)]
            bound = [r for r in records if r["lifecycle"] == "BOUND"]
            if len(bound) != 48:
                raise FailClosed(f"{name} {regime}: {len(bound)} BOUND != 48")
            per_tau = [aggregate(bound, t, frame_ms) for t in TAU_GRID]
            agg = next(r for r in per_tau if r["tau"] == tau)
            if (agg["n_keep"], agg["n_cut"], agg["n_other_topology"]) != (24, 16, 8):
                raise FailClosed(f"{name} {regime}: headline sets "
                                 f"{agg['n_keep']}/{agg['n_cut']}/{agg['n_other_topology']}")
            cuts_at_tau = sum(len(replay_decisions(r["frames"], tau, frame_ms)[0])
                              for r in bound)
            sens_at_tau = sum(replay_decisions(r["frames"], tau, frame_ms)[1]
                              for r in bound)
            view = stratum_breakdown(bound, tau, frame_ms)
            (OUT / f"{name}_{regime}_main.jsonl").write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8")
            (OUT / f"{name}_{regime}_calibration.jsonl").write_text(
                "\n".join(json.dumps({**r, "tau_frozen": tau}) for r in per_tau)
                + "\n", encoding="utf-8")
            cells.append({"model": name, "regime": regime, "tau": tau, "agg": agg,
                          "view": view, "cuts_at_tau": cuts_at_tau,
                          "sens_at_tau": sens_at_tau})
    (OUT / "thresholds_frozen.json").write_text(json.dumps(
        [{"model": c["model"], "regime": c["regime"], "tau": c["tau"]} for c in cells],
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# MAIN48 rev3 rescore (PSEM-SMALL-MODEL-PROBE-v1-rev3, evaluator_revision=3)",
             "",
             f"> frozen taus from cal/results_repaired_v3/thresholds.json "
             f"(applied as-is, no MAIN48 retuning)",
             "",
             "## Headlines (500 ms primary; 300 ms sensitivity diagnostic)",
             "",
             "| model | regime | tau | false_cuts/24 | missed/16 | contam s/h | "
             "src p50/p90 | dec p50/p90 | total p50/p90 | cuts@tau | sens@tau |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        a = c["agg"]
        lines.append(
            f"| {c['model']} | {c['regime']} | {c['tau']} | "
            f"{a['false_cuts']}/24 | {a['missed']}/16 | "
            f"{a['contamination_s_per_speech_h']} | "
            f"{a['delay_src_err_p50']}/{a['delay_src_err_p90']} | "
            f"{a['delay_dec_p50']}/{a['delay_dec_p90']} | "
            f"{a['delay_total_p50']}/{a['delay_total_p90']} | "
            f"{c['cuts_at_tau']} | {c['sens_at_tau']} |")
    lines += ["", "## Stratum breakdown (detected = CUT-role episodes with a valid CUT)",
              "",
              "| model | regime | C1 fc/8 | C2 det/8 | C3 fc/8 | C4 det/8 | "
              "C5 fc/8 | C6 n | CUT det/16 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        v = c["view"]
        cut_det = v["C2"]["detected"] + v["C4"]["detected"]
        lines.append(
            f"| {c['model']} | {c['regime']} | {v['C1']['false_cuts']}/8 | "
            f"{v['C2']['detected']}/8 | {v['C3']['false_cuts']}/8 | "
            f"{v['C4']['detected']}/8 | {v['C5']['false_cuts']}/8 | {v['C6']['n']} | "
            f"{cut_det}/16 |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps({
        "revision": REVISION, "evaluator_revision": 3,
        "manifest": "manifest/manifest_rev3.jsonl",
        "manifest_sha256": actual, "freeze_sha256": freeze["freeze_sha256"],
        "frozen_taus_from": "cal/results_repaired_v3/thresholds.json (applied as-is, no MAIN48 retuning)",
        "raw_inference": ("firered: REPLAYED verbatim from main/results/*_main.jsonl; "
                          "neovad: FRESH native inference all 48/ep-regime with "
                          "warm-up headers verified per episode"),
        "stub_fallback": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
