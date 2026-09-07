#!/usr/bin/env python3
"""rev3 CAL12 rebuild: 6 strata x 2 corpora, GT-only deterministic.

Keeps the 10 frozen CAL12 rows that already cover C1/C3(canonical
A->A+B->A)/C4/C5/C6 x 2 corpora, and displaces the duplicate-C3
``overlap_return``-flavor slot (one per corpus) in favor of C2
(clean direct/silence-gap handoff). New C2 episodes come from the
existing CAL session pool (sessions already in CAL first, then the
rest of the builder's CAL pool), selected model/threshold-blind:
first finalizable C2 candidate in deterministic builder order that
fits (non-overlapping eval window, <=4 CAL rows per session).

Binding spans derive from the same builder finalize() as before.
Selection-only change: episodes. Old CAL rows preserved (frozen
manifest.jsonl untouched).

Writes (old files untouched):
  cal/CAL12_rev3.jsonl
  cal/CAL12_rev3_provenance.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import experiments.psem_small_model_probe.manifest.build_manifest as bm  # noqa: E402

CAL_DIR = Path(__file__).resolve().parent
MANIFEST_REV3 = CAL_DIR.parent / "manifest" / "manifest_rev3.jsonl"
OUT_ROWS = CAL_DIR / "CAL12_rev3.jsonl"
OUT_PROV = CAL_DIR / "CAL12_rev3_provenance.json"

REVISION = "PSEM-SMALL-MODEL-PROBE-v1-rev3"
CAL_CAP_PER_SESSION = 4


class FailClosed(RuntimeError):
    pass


def main() -> None:
    rev3 = [json.loads(line) for line in
            MANIFEST_REV3.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_ep = {r["episode_id"]: r for r in rev3}
    old_cal = [r for r in rev3 if r["split"] == "CAL12"]
    if len(old_cal) != 12:
        raise FailClosed(f"frozen CAL12 rows = {len(old_cal)}")

    # Displace the duplicate-C3 overlap_return-flavor slot (one per corpus);
    # keep the canonical A->A+B->A C3 representative.
    displaced = [r for r in old_cal
                 if r["stratum"] == "C3" and r["topology"] == "overlap_return"]
    kept = [r for r in old_cal if r not in displaced]
    if len(displaced) != 2 or len(kept) != 10:
        raise FailClosed(f"displaced={len(displaced)} kept={len(kept)}")
    if sorted(r["session_id"] for r in displaced) != ["EN2006a", "R1019_M1928"]:
        raise FailClosed("unexpected displaced rows: "
                         + ", ".join(r["episode_id"] for r in displaced))

    # Builder candidate pool for C2 (same helpers, same per-session sort).
    sessions = bm.load_gt()
    by: dict = {}
    for key in sorted(sessions):
        sess = sessions[key]
        for c in bm.gen_candidates(sess):
            c["key"] = key
            by.setdefault((key, c["stratum"]), []).append(c)
    for (key, stratum), v in by.items():
        if stratum in ("C3", "C4"):
            v.sort(key=lambda c: (-c["extra"].get("overlap_a_active_samples", 0),
                                  c["trans_sample"]))
        else:
            v.sort(key=lambda c: c["trans_sample"])

    keys = sorted(sessions)
    cal_pools = {"ami": [k for k in keys if k[0] == "AMI"][:4],
                 "alimeeting": [k for k in keys if k[0] == "AliMeeting"][:4]}

    taken_trans = {(r["corpus"], r["session_id"],
                    r["authoritative_transition_time_ms"]) for r in rev3}
    kept_by_session: dict = {}
    for r in kept:
        kept_by_session.setdefault((r["corpus"], r["session_id"]), []).append(r)
    new_rows = []
    search_order_note = {}

    for corpus_name, corpus_tag in (("ami", "AMI"), ("alimeeting", "AliMeeting")):
        pool = cal_pools[corpus_name]
        kept_pair_set = {(r["corpus"], r["session_id"]) for r in kept}
        # Prefer sessions already in CAL (0) over the rest of the pool (1);
        # sorted-key order within each tier: fully deterministic.
        ordered = sorted(
            pool,
            key=lambda k: (0 if (sessions[k]["corpus"], sessions[k]["session_id"])
                           in kept_pair_set else 1, k),
        )
        picked_row = None
        for key in ordered:
            sess = sessions[key]
            if (sess["corpus"], sess["session_id"]) != (
                    {"AMI": "ami", "AliMeeting": "alimeeting"}[key[0]], key[1]):
                raise FailClosed(f"session key mismatch: {key}")
            n_cal = len(kept_by_session.get((sess["corpus"], sess["session_id"]), []))
            if n_cal >= CAL_CAP_PER_SESSION:
                continue
            for c in by.get((key, "C2"), []):
                trans_ms = c["trans_sample"] // bm.MS
                if (sess["corpus"], sess["session_id"], trans_ms) in taken_trans:
                    continue
                f = bm.finalize(c)
                if f is None:
                    continue
                clash = False
                for x in (kept_by_session.get((sess["corpus"], sess["session_id"]), [])
                          + [r for r in new_rows
                             if (r["corpus"], r["session_id"])
                             == (sess["corpus"], sess["session_id"])]):
                    if not (f["evaluation_end_ms"] <= x["evaluation_start_ms"]
                            or x["evaluation_end_ms"] <= f["evaluation_start_ms"]):
                        clash = True
                        break
                if clash:
                    continue
                suffix = max(int(r["episode_id"].rsplit(":A", 1)[1])
                             for r in rev3 if r["session_id"] == sess["session_id"]) + 1
                row = {
                    "schema_version": bm.SCHEMA,
                    "episode_id": f"{sess['session_id']}:A{suffix:05d}",
                    "corpus": sess["corpus"],
                    "session_id": sess["session_id"],
                    "topology": bm.topo_label(c),
                    "split": "CAL12",
                    "evaluation_start_ms": f["evaluation_start_ms"],
                    "evaluation_end_ms": f["evaluation_end_ms"],
                    "anchor_speaker": f["anchor"],
                    "native_reference_start_ms": f["native_reference_start_ms"],
                    "native_reference_end_ms": f["native_reference_end_ms"],
                    "causal_reference_start_ms": f["causal_reference_start_ms"],
                    "causal_reference_end_ms": f["causal_reference_end_ms"],
                    "causal_bindable": f["causal_bindable"],
                    "authoritative_transition_time_ms": f["authoritative_transition_time_ms"],
                    "ontology_subset": False,
                    "control_subset": False,
                    "stratum": "C2",
                }
                if row["episode_id"] in by_ep:
                    raise FailClosed(f"episode_id collision: {row['episode_id']}")
                if row["topology"] != "A":
                    raise FailClosed(f"C2 display topology {row['topology']}")
                new_rows.append(row)
                taken_trans.add((sess["corpus"], sess["session_id"], trans_ms))
                search_order_note[row["episode_id"]] = (
                    f"pool={corpus_name} session={sess['session_id']} "
                    f"trans_ms={trans_ms} anchor={f['anchor']}")
                picked_row = row
                break
            if picked_row is not None:
                break
        if picked_row is None:
            raise FailClosed(f"CAL12_rev3/C2/{corpus_name}: no candidate")
        print(f"new C2 row: {picked_row['episode_id']} {search_order_note[picked_row['episode_id']]}")

    cal12_rev3 = sorted(kept + new_rows, key=lambda r: r["episode_id"])
    if len(cal12_rev3) != 12:
        raise FailClosed(f"CAL12_rev3 rows = {len(cal12_rev3)}")
    per = Counter(r["stratum"] for r in cal12_rev3)
    if dict(per) != {"C1": 2, "C2": 2, "C3": 2, "C4": 2, "C5": 2, "C6": 2}:
        raise FailClosed(f"CAL12_rev3 stratum counts {dict(per)}")
    # CAL<->MAIN/EXT session disjointness + both corpora present.
    main_sess = {(r["corpus"], r["session_id"]) for r in rev3 if r["split"] == "MAIN48"}
    ext_sess = {(r["corpus"], r["session_id"]) for r in rev3 if r["split"] == "EXT24"}
    cal_sess = {(r["corpus"], r["session_id"]) for r in cal12_rev3}
    if cal_sess & (main_sess | ext_sess):
        raise FailClosed(f"session overlap: {cal_sess & (main_sess | ext_sess)}")
    corpora = sorted({r["corpus"] for r in cal12_rev3})
    if corpora != ["alimeeting", "ami"]:
        raise FailClosed(f"corpora {corpora}")

    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True) + "\n" for r in cal12_rev3)
    OUT_ROWS.write_text(raw, encoding="utf-8", newline="\n")
    prov = {
        "revision": REVISION,
        "note": ("GT-only deterministic rebuild, model/threshold-blind; "
                 "selection-only change (episodes); binding spans derived "
                 "with the same builder finalize(); old CAL rows preserved"),
        "source_manifest": "manifest/manifest_rev3.jsonl",
        "source_manifest_sha256": hashlib.sha256(
            MANIFEST_REV3.read_bytes()).hexdigest(),
        "displaced_duplicate_c3_slot": sorted(r["episode_id"] for r in displaced),
        "kept_episodes": sorted(r["episode_id"] for r in kept),
        "new_c2_episodes": {r["episode_id"]: search_order_note[r["episode_id"]]
                            for r in new_rows},
        "stratum_counts": dict(sorted(per.items())),
        "sessions": sorted(f"{c}/{s}" for c, s in cal_sess),
        "disjoint_from_main_ext": True,
        "corpora": corpora,
        "file_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
    OUT_PROV.write_text(json.dumps(prov, sort_keys=True, indent=1) + "\n",
                        encoding="utf-8")
    print(json.dumps({"rows": str(OUT_ROWS), "stratum_counts": dict(sorted(per.items())),
                      "sessions": prov["sessions"],
                      "disjoint": True, "corpora": corpora}, indent=1))


if __name__ == "__main__":
    main()
