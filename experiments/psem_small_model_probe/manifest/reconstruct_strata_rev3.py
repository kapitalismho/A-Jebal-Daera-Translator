#!/usr/bin/env python3
"""rev3 schema repair: explicit stratum C1..C6 on the frozen 84 episodes.

Replays experiments/psem_small_model_probe/manifest/build_manifest.py
selection logic verbatim (same builder helpers, same pool order, same
take()/fits()/finalize() rules) and captures the in-memory ``stratum``
the builder already assigned but did not emit. Joins to the frozen
manifest.jsonl rows and FAILS CLOSED unless all 84 replayed rows are
field-identical (episode_id/session/transition/spans) to the frozen
rows — i.e. episode selection is unchanged; only the schema gains the
explicit ``stratum`` field.

Writes (old freeze files untouched):
  manifest/manifest_rev3.jsonl
  manifest/dataset_freeze_rev3.json

Provenance: PSEM-SMALL-MODEL-PROBE-v1-rev3 (same episodes/windows/
references, schema repair only).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import experiments.psem_small_model_probe.manifest.build_manifest as bm  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OLD_MANIFEST = OUT_DIR / "manifest.jsonl"
OLD_FREEZE = OUT_DIR / "dataset_freeze.json"
NEW_MANIFEST = OUT_DIR / "manifest_rev3.jsonl"
NEW_FREEZE = OUT_DIR / "dataset_freeze_rev3.json"

REVISION = "PSEM-SMALL-MODEL-PROBE-v1-rev3"


class FailClosed(RuntimeError):
    pass


def replay_selection():
    """Verbatim replay of build_manifest.main() selection; returns picked."""
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

    cache = {}

    def ensure(c):
        k = id(c)
        if k not in cache:
            cache[k] = bm.finalize(c)
        return cache[k]

    def fits(key, f, extra=()):
        for _, x in list(picked) + [("x", g) for g in extra]:
            if x["key"] == key and not (f["evaluation_end_ms"] <= x["evaluation_start_ms"]
                                        or x["evaluation_end_ms"] <= f["evaluation_start_ms"]):
                return False
        return True

    used = set()
    picked = []  # (split, finalized candidate)

    def take(pool_keys, stratum, corpus, n, cap, split):
        got = []
        for key in pool_keys:
            if len(got) >= n:
                break
            if key[0] != corpus:
                continue
            for c in by.get((key, stratum), []):
                if len(got) >= n:
                    break
                if id(c) in used:
                    continue
                if sum(1 for s, x in picked if s == split and x["key"] == key) >= cap:
                    break
                f = ensure(c)
                if f is None or not fits(key, f, got):
                    continue
                used.add(id(c))
                got.append(f)
        if len(got) < n:
            raise FailClosed(f"{split}/{stratum}/{corpus}: {len(got)}/{n}")
        picked.extend((split, f) for f in got)
        return got

    keys = sorted(sessions)
    cal_groups = [("A", "C1", False), ("A+B", "C6", True), ("A+A+B", "C5", True),
                  ("A->A+B->A", "C3", False), ("A->A+B->B", "C4", False),
                  ("overlap_return", "C3", True)]
    cal_pools = {"ami": [k for k in keys if k[0] == "AMI"][:4],
                 "alimeeting": [k for k in keys if k[0] == "AliMeeting"][:4]}
    for grp, stratum, want_ov in cal_groups:
        for corpus in ("ami", "alimeeting"):
            pool = cal_pools[corpus]
            done = False
            for key in pool:
                for c in by.get((key, stratum), []):
                    if id(c) in used:
                        continue
                    if stratum == "C3" and want_ov != (c["v2_topo"] == "overlap_return"):
                        continue
                    if stratum == "C5" and want_ov != bool(c["extra"].get("has_overlap")):
                        continue
                    if stratum == "C6" and not c["extra"].get("has_overlap"):
                        continue
                    if sum(1 for s, x in picked if s == "CAL12" and x["key"] == key) >= 4:
                        continue
                    f = ensure(c)
                    if f is None or not fits(key, f):
                        continue
                    used.add(id(c))
                    f["cal_group"] = grp
                    picked.append(("CAL12", f))
                    done = True
                    break
                if done:
                    break
            if not done:
                raise FailClosed(f"CAL12/{grp}/{corpus}: no candidate")
    cal_keys = {x["key"] for s, x in picked if s == "CAL12"}
    rest_ami = [k for k in keys if k[0] == "AMI" and k not in cal_keys]
    rest_ali = [k for k in keys if k[0] == "AliMeeting" and k not in cal_keys]
    main_pools = {"ami": rest_ami[:6], "alimeeting": rest_ali[:6]}
    for stratum in ("C1", "C2", "C3", "C4", "C5", "C6"):
        take(main_pools["ami"], stratum, "AMI", 4, 6, "MAIN48")
        take(main_pools["alimeeting"], stratum, "AliMeeting", 4, 6, "MAIN48")
    main_keys = {x["key"] for s, x in picked if s == "MAIN48"}
    ext_pools = {"ami": [k for k in rest_ami if k not in main_keys][:3],
                 "alimeeting": [k for k in rest_ali if k not in main_keys][:3]}
    for stratum in ("C1", "C2", "C3", "C4", "C5", "C6"):
        take(ext_pools["ami"], stratum, "AMI", 2, 6, "EXT24")
        take(ext_pools["alimeeting"], stratum, "AliMeeting", 2, 6, "EXT24")
    return picked


def emit_rows(picked):
    counters = {}
    rows = []
    for split, c in picked:
        key = c["key"]
        counters[key] = counters.get(key, 0) + 1
        sess = c["sess"]
        topo = c.get("cal_group") if split == "CAL12" else bm.topo_label(c)
        rows.append({
            "schema_version": bm.SCHEMA,
            "episode_id": f"{sess['session_id']}:A{counters[key]:05d}",
            "corpus": sess["corpus"],
            "session_id": sess["session_id"],
            "topology": topo,
            "split": split,
            "evaluation_start_ms": c["evaluation_start_ms"],
            "evaluation_end_ms": c["evaluation_end_ms"],
            "anchor_speaker": c["anchor"],
            "native_reference_start_ms": c["native_reference_start_ms"],
            "native_reference_end_ms": c["native_reference_end_ms"],
            "causal_reference_start_ms": c["causal_reference_start_ms"],
            "causal_reference_end_ms": c["causal_reference_end_ms"],
            "causal_bindable": c["causal_bindable"],
            "authoritative_transition_time_ms": c["authoritative_transition_time_ms"],
            "ontology_subset": False,  # recomputed below, same rule as builder
            "control_subset": False,
            "stratum": c["stratum"],
        })
    rows.sort(key=lambda r: r["episode_id"])
    return rows


def main() -> None:
    picked = replay_selection()
    replayed = emit_rows(picked)

    old_rows = [json.loads(line) for line in
                OLD_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_by_ep = {r["episode_id"]: r for r in old_rows}
    if len(old_by_ep) != 84 or len(replayed) != 84:
        raise FailClosed(f"row counts replayed={len(replayed)} frozen={len(old_rows)}")

    identical = 0
    for r in replayed:
        old = old_by_ep.get(r["episode_id"])
        if old is None:
            raise FailClosed(f"replayed episode_id missing from freeze: {r['episode_id']}")
        re_cmp = {k: v for k, v in r.items()
                  if k not in ("stratum", "ontology_subset", "control_subset")}
        old_cmp = {k: v for k, v in old.items()
                   if k not in ("ontology_subset", "control_subset")}
        if re_cmp != old_cmp:
            raise FailClosed(f"row drift at {r['episode_id']}:\n replayed={re_cmp}\n frozen  ={old_cmp}")
        if (r["evaluation_start_ms"] != old["evaluation_start_ms"]
                or r["evaluation_end_ms"] != old["evaluation_end_ms"]
                or r["authoritative_transition_time_ms"] != old["authoritative_transition_time_ms"]
                or r["native_reference_start_ms"] != old["native_reference_start_ms"]
                or r["native_reference_end_ms"] != old["native_reference_end_ms"]
                or old["session_id"] != r["session_id"]):
            raise FailClosed(f"span/session drift at {r['episode_id']}")
        identical += 1
    print(f"reconstruction verification: {identical}/84 rows field-identical "
          f"(episode_id/session/transition/spans)")

    # Adopt the frozen ontology/control flags verbatim (selection unchanged).
    rev3_rows = []
    for r in replayed:
        old = old_by_ep[r["episode_id"]]
        rev3_rows.append({**old, "stratum": r["stratum"]})
    rev3_rows.sort(key=lambda r: r["episode_id"])

    # FAIL-CLOSED stratum asserts.
    from experiments.psem_small_model_probe.cal.eval_semantics import (  # noqa: E402
        CUT_STRATA,
        KEEP_STRATA,
        OTHER_STRATA,
    )
    main48 = [r for r in rev3_rows if r["split"] == "MAIN48"]
    per = Counter(r["stratum"] for r in main48)
    if dict(per) != {"C1": 8, "C2": 8, "C3": 8, "C4": 8, "C5": 8, "C6": 8}:
        raise FailClosed(f"MAIN48 stratum counts {dict(per)}")
    keep = sum(1 for r in main48 if r["stratum"] in KEEP_STRATA)
    cut = sum(1 for r in main48 if r["stratum"] in CUT_STRATA)
    other = sum(1 for r in main48 if r["stratum"] in OTHER_STRATA)
    if (keep, cut, other) != (24, 16, 8):
        raise FailClosed(f"MAIN48 KEEP/CUT/OTHER = {keep}/{cut}/{other}")
    print(f"MAIN48 strata: {dict(sorted(per.items()))} "
          f"KEEP={keep} CUT={cut} OTHER={other}")

    raw = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True) + "\n" for r in rev3_rows)
    NEW_MANIFEST.write_text(raw, encoding="utf-8", newline="\n")
    file_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    old_freeze = json.loads(OLD_FREEZE.read_text(encoding="utf-8"))
    counts = dict(old_freeze["counts"])
    stratum_counts = {
        split: dict(Counter(r["stratum"] for r in rev3_rows if r["split"] == split))
        for split in ("CAL12", "MAIN48", "EXT24")
    }
    freeze_payload = {"rows": rev3_rows, "counts": counts}
    freeze_sha = hashlib.sha256(json.dumps(
        freeze_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")).hexdigest()
    new_freeze = dict(old_freeze)
    new_freeze.update({
        "revision": REVISION,
        "provenance": {
            "revision": REVISION,
            "note": ("same episodes/windows/references as "
                     "PSEM-SMALL-MODEL-PROBE-v1, schema repair only: "
                     "explicit stratum C1..C6 per row"),
            "source_manifest": "manifest/manifest.jsonl",
            "source_manifest_sha256": old_freeze["file_sha256"],
            "source_freeze_sha256": old_freeze["freeze_sha256"],
            "reconstruction": "manifest/reconstruct_strata_rev3.py "
                               "(verbatim builder-logic replay, 84/84 field-identical)",
        },
        "counts": counts,
        "stratum_counts": stratum_counts,
        "file_sha256": file_sha,
        "freeze_sha256": freeze_sha,
    })
    NEW_FREEZE.write_text(json.dumps(new_freeze, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(NEW_MANIFEST), "rows": len(rev3_rows),
                      "stratum_counts": stratum_counts,
                      "file_sha256": file_sha, "freeze_sha256": freeze_sha}, indent=1))


if __name__ == "__main__":
    main()
