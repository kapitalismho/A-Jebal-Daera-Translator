#!/usr/bin/env python3
"""rev3 stratum-truth unit tests (no pytest; runnable via python).

Covers: C1..C6 role mapping, legacy topology fallback (rev2 preserved),
MAIN48 24/16/8 + 8-each counts, CAL12_rev3 C2x2 + session disjointness,
stratum-first score_episode (topology display-only), and the KEEP vs
C2-only selection split (miss objective never on C4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
    CUT_STRATA,
    KEEP_STRATA,
    OTHER_STRATA,
    compact_gt,
    episode_role,
    stratum_of,
)
from experiments.psem_small_model_probe.cal.metrics import (  # noqa: E402
    CUT_TOPOLOGIES,
    KEEP_TOPOLOGIES,
    aggregate,
    score_episode,
    select_threshold_split,
    selection_aggregate,
    split_selection_records,
)

PROBE = Path(__file__).resolve().parent.parent
MANIFEST_REV3 = PROBE / "manifest" / "manifest_rev3.jsonl"
CAL12_REV3 = PROBE / "cal" / "CAL12_rev3.jsonl"

FRAME_MS = 100
TAU = 0.5


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}" + (f" — {detail}" if detail else ""))


def make_gt(specs):
    intervals = [
        {"start_sample": s * 16, "end_sample": e * 16,
         "active_speakers": list(sp), "masked": m}
        for s, e, sp, m in specs
    ]
    intervals.sort(key=lambda iv: iv["start_sample"])
    return {"intervals": intervals,
            "starts": [iv["start_sample"] for iv in intervals]}


def make_frames(n, speech, anchors, eval_start=0):
    assert len(speech) == len(anchors) == n
    return [
        {"speech_gt": speech[i],
         "anchor_speech_gt": speech[i] and anchors[i] >= TAU,
         "anchor": anchors[i],
         "adapter_speech": True,
         "lifecycle": "BOUND",
         "source_time_ms": eval_start + (i + 1) * FRAME_MS}
        for i in range(n)
    ]


def make_record(episode_id, topology, stratum, transition, frames, anchor="A"):
    gt = make_gt([(0, len(frames) * FRAME_MS, ["A"], False)])
    rec = {"episode_id": episode_id,
           "topology": topology,
           "authoritative_transition_ms": transition,
           "frames": frames,
           "contam_s": 0.0,
           "active_speech_s": sum(1 for f in frames if f["speech_gt"]) * FRAME_MS / 1000.0,
           "lifecycle": "BOUND",
           "gt_eval": compact_gt(gt, anchor, 0, len(frames) * FRAME_MS)}
    if stratum is not None:
        rec["stratum"] = stratum
    return rec


def cut_frames(n=10):
    # Sustained non-anchor speech -> confirmed CUT (500 ms decoder).
    return make_frames(n, [True] * n, [0.0] * n)


def keep_frames(n=10):
    # Sustained anchor speech -> no CUT.
    return make_frames(n, [True] * n, [1.0] * n)


def test_role_mapping():
    cases = [("C1", "KEEP"), ("C2", "CUT"), ("C3", "KEEP"),
             ("C4", "CUT"), ("C5", "KEEP"), ("C6", "OTHER")]
    for stratum, want in cases:
        # Deliberately misleading display topology: stratum must win.
        got = episode_role({"stratum": stratum, "topology": "A"},
                           KEEP_TOPOLOGIES, CUT_TOPOLOGIES)
        check(f"role/{stratum}->{want}", got == want, f"topology=A got={got}")
    check("sets/keep", set(KEEP_STRATA) == {"C1", "C3", "C5"})
    check("sets/cut", set(CUT_STRATA) == {"C2", "C4"})
    check("sets/other", set(OTHER_STRATA) == {"C6"})
    check("stratum_of/known", stratum_of({"stratum": "C2"}) == "C2")
    check("stratum_of/missing", stratum_of({"topology": "A"}) is None)
    check("stratum_of/bogus", stratum_of({"stratum": "CX"}) is None)


def test_legacy_fallback():
    # Pre-rev3 records (no stratum): exact rev2 topology semantics.
    check("legacy/A->KEEP",
          episode_role({"topology": "A"}, KEEP_TOPOLOGIES, CUT_TOPOLOGIES) == "KEEP")
    check("legacy/A->A+B->B->CUT",
          episode_role({"topology": "A->A+B->B"},
                       KEEP_TOPOLOGIES, CUT_TOPOLOGIES) == "CUT")
    check("legacy/A+B->OTHER",
          episode_role({"topology": "A+B"}, KEEP_TOPOLOGIES, CUT_TOPOLOGIES) == "OTHER")
    out = score_episode(make_record("e", "A", None, 0, cut_frames()), TAU, FRAME_MS)
    check("legacy/score-A-cut->false_cut",
          out["false_cut"] is True and out["missed"] is False and out["role"] == "KEEP")


def test_main48_counts():
    rows = [json.loads(line) for line in
            MANIFEST_REV3.read_text(encoding="utf-8").splitlines() if line.strip()]
    main48 = [r for r in rows if r["split"] == "MAIN48"]
    per = {}
    for r in main48:
        per[r["stratum"]] = per.get(r["stratum"], 0) + 1
    check("main48/8-each",
          per == {"C1": 8, "C2": 8, "C3": 8, "C4": 8, "C5": 8, "C6": 8}, str(per))
    keep = sum(1 for r in main48 if r["stratum"] in KEEP_STRATA)
    cut = sum(1 for r in main48 if r["stratum"] in CUT_STRATA)
    other = sum(1 for r in main48 if r["stratum"] in OTHER_STRATA)
    check("main48/24-16-8", (keep, cut, other) == (24, 16, 8),
          f"{keep}/{cut}/{other}")


def test_cal12_rev3():
    rows = [json.loads(line) for line in
            CAL12_REV3.read_text(encoding="utf-8").splitlines() if line.strip()]
    check("cal12/n=12", len(rows) == 12, str(len(rows)))
    per = {}
    for r in rows:
        per[r["stratum"]] = per.get(r["stratum"], 0) + 1
    check("cal12/2-each",
          per == {"C1": 2, "C2": 2, "C3": 2, "C4": 2, "C5": 2, "C6": 2}, str(per))
    c2 = [r for r in rows if r["stratum"] == "C2"]
    check("cal12/C2x2", len(c2) == 2)
    check("cal12/C2-display-A", all(r["topology"] == "A" for r in c2))
    rev3 = [json.loads(line) for line in
            MANIFEST_REV3.read_text(encoding="utf-8").splitlines() if line.strip()]
    other_sess = {(r["corpus"], r["session_id"]) for r in rev3
                  if r["split"] in ("MAIN48", "EXT24")}
    cal_sess = {(r["corpus"], r["session_id"]) for r in rows}
    check("cal12/disjoint", not (cal_sess & other_sess),
          str(cal_sess & other_sess))
    check("cal12/corpora", sorted({r["corpus"] for r in rows}) == ["alimeeting", "ami"])
    for r in rows:
        check(f"cal12/span-{r['episode_id']}",
              r["native_reference_end_ms"] - r["native_reference_start_ms"] == 5000)
        if r["causal_bindable"]:
            check(f"cal12/causal-{r['episode_id']}",
                  r["causal_reference_end_ms"] - r["causal_reference_start_ms"] == 1000
                  and r["causal_reference_end_ms"] <= r["authoritative_transition_time_ms"])
    check("cal12/unique-ids", len({r["episode_id"] for r in rows}) == 12)


def test_score_stratum_truth():
    # C2 displays as "A" (TOPO_OF bug): stratum must still score CUT.
    out = score_episode(make_record("c2", "A", "C2", 0, cut_frames()), TAU, FRAME_MS)
    check("score/C2-cut", out["role"] == "CUT" and out["missed"] is False
          and out["false_cut"] is False, str({k: out[k] for k in ("role", "missed", "false_cut")}))
    out = score_episode(make_record("c2m", "A", "C2", 0, keep_frames()), TAU, FRAME_MS)
    check("score/C2-nocut->missed", out["missed"] is True and out["false_cut"] is False)
    out = score_episode(make_record("c1", "A", "C1", 0, cut_frames()), TAU, FRAME_MS)
    check("score/C1-cut->false_cut", out["false_cut"] is True and out["missed"] is False)
    out = score_episode(make_record("c6", "A", "C6", 0, cut_frames()), TAU, FRAME_MS)
    check("score/C6-excluded", out["role"] == "OTHER"
          and out["false_cut"] is False and out["missed"] is False)
    agg = aggregate([make_record("c1", "A", "C1", 0, keep_frames()),
                     make_record("c2", "A", "C2", 0, keep_frames()),
                     make_record("c6", "A+B", "C6", 0, keep_frames())], TAU, FRAME_MS)
    check("agg/roles", (agg["n_keep"], agg["n_cut"], agg["n_other_topology"]) == (1, 1, 1),
          str({k: agg[k] for k in ("n_keep", "n_cut", "n_other_topology")}))
    check("agg/headlines", agg["false_cuts"] == 0 and agg["missed"] == 1)


def test_selection_split():
    recs = [make_record(f"k{i}", "A", s, 0, keep_frames())
            for i, s in enumerate(["C1", "C3", "C5"])]
    recs += [make_record("c2", "A", "C2", 0, keep_frames()),
             make_record("c4", "A->A+B->B", "C4", 0, keep_frames()),
             make_record("c6", "A+B", "C6", 0, keep_frames()),
             make_record("legacy", "A", None, 0, keep_frames())]
    keep, c2 = split_selection_records(recs)
    check("split/keep", sorted(r["episode_id"] for r in keep) == ["k0", "k1", "k2", "legacy"],
          str([r["episode_id"] for r in keep]))
    check("split/c2-only", [r["episode_id"] for r in c2] == ["c2"],
          "C4/C6/legacy-topology-A must never enter the C2 subset")
    rows = [selection_aggregate(keep, c2, tau, FRAME_MS) for tau in (0.3, 0.5)]
    check("split/row-fields",
          all(set(r) == {"tau", "keep_n", "keep_false_cuts", "c2_n",
                         "c2_missed", "c2_missed_rate", "median_total_delay_ms"}
              for r in rows))
    synth = [{"tau": 0.3, "keep_false_cuts": 0, "c2_missed": 1, "c2_n": 2,
              "c2_missed_rate": 0.5, "median_total_delay_ms": 900.0},
             {"tau": 0.5, "keep_false_cuts": 0, "c2_missed": 0, "c2_n": 2,
              "c2_missed_rate": 0.0, "median_total_delay_ms": 1200.0}]
    tau, reason = select_threshold_split(synth)
    check("split/select-c2-miss", tau == 0.5, reason)


def main():
    test_role_mapping()
    test_legacy_fallback()
    test_main48_counts()
    test_cal12_rev3()
    test_score_stratum_truth()
    test_selection_split()
    print("ALL STRATUM-REV3 TESTS PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
