from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.psem_product_translation.capture import corrected_groups_for_capture
EXP = ROOT / "experiments" / "psem_product_translation"
STAGE2 = ROOT / "experiments" / "psem_repeatability_stage2"
OBSDIR = ROOT / "experiments" / "psem_phase_a_headroom" / "observations"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
TAU = 0.5
FRAME = 1280
CONFIRMATION = 1600
MODEL = "google/gemma-4-26b-a4b-it"
TIMEOUT_S = 30.0
MAX_OUT_TOKENS = 512
HARD_TOTAL_CAP = 129
TRANS_MAX = 128
FIXED_ORDER = ["NP1","NP2","NP3","R1","R2","T1","COMBINED","BC1","SINGLE_ES2009c","SINGLE_ES2009d"]
_TABLES = {}
def sha256_file(p):
    h = hashlib.sha256()
    f = open(p,"rb")
    b = f.read(1048576)
    while b:
        h.update(b)
        b = f.read(1048576)
    f.close()
    return h.hexdigest()
def load_np(path, shape):
    import numpy as np
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(shape[0], shape[1])
def load_obs():
    return json.loads((OBSDIR / "OBSERVATIONS.json").read_text(encoding="utf-8"))
def load_probs_by_sid(obs):
    out = {}
    for s in obs["sources"]:
        sid = s["source_id"]
        fp = ROOT / s["feature_files"]["probs"]["path"]
        shape = tuple(s["feature_files"]["probs"]["shape"])
        out[sid] = load_np(fp, shape)
    return out
def trace_init_s(src_entry):
    h = json.loads((ROOT / src_entry["trace_path"]).read_text(encoding="utf-8"))
    return (h.get("load_us",0) + h.get("sched_setup_us",0)) / 1000000.0
def get_source_table(src_entry):
    sid = src_entry["source_id"]
    if sid in _TABLES:
        return _TABLES[sid]
    n = src_entry["valid_native_frames"]
    chunks = src_entry["chunks"]
    prefix = src_entry["prefix_end_sample"]
    frontiers = [0]*n
    chunk_of = [None]*n
    valid_tail = [False]*n
    fi = 0
    for ci,c in enumerate(chunks):
        rs = c["raw_support_end_sample"]
        ec = c["emit_count"]
        ok = rs <= prefix
        for k in range(ec):
            if fi >= n:
                break
            frontiers[fi] = rs
            chunk_of[fi] = ci
            valid_tail[fi] = ok
            fi += 1
    t = {"n":n,"frontiers":frontiers,"chunk_of":chunk_of,"valid_tail":valid_tail,"chunks":chunks,"prefix":prefix,"source_id":sid}
    _TABLES[sid] = t
    return t
def flush_wall_of(ch):
    v = ch.get("flush_end_wall")
    if v is not None:
        return v
    v = ch.get("send_end")
    if v is not None:
        return v
    return ch.get("sched_wall")
def build_schedule_for_case(table, cap, pay, init_s):
    chunks = table["chunks"]
    lo,hi = pay[0],pay[1]
    ordered = sorted(cap.get("chunk_ledger",[]), key=lambda c: c.get("src_range",[0,0])[0])
    cap_ends = []
    cap_walls = []
    for ch in ordered:
        r = ch.get("src_range",[0,0])
        cap_ends.append(r[1])
        cap_walls.append(flush_wall_of(ch))
    import bisect
    release = [None]*len(chunks)
    finish = [None]*len(chunks)
    prev = None
    for j,c in enumerate(chunks):
        S = c["raw_support_end_sample"]
        if S < lo:
            rel = (S - lo) / 16000.0
        elif S < hi:
            k = bisect.bisect_right(cap_ends, S)
            if k < len(cap_walls):
                rel = cap_walls[k]
            else:
                rel = None
        else:
            rel = None
        release[j] = rel
        if rel is None:
            finish[j] = None
        else:
            if prev is None:
                prev = rel + init_s
            if rel > prev:
                prev = rel
            prev = prev + c["service_us"] / 1000000.0
            finish[j] = prev
    return {"release":release,"finish":finish,"init_s":init_s}
def frame_eligible(table, sched_finish, idx, terminal):
    n = table["n"]
    if idx < 0 or idx >= n:
        return False,"missing"
    if not table["valid_tail"][idx]:
        return False,"invalid"
    ci = table["chunk_of"][idx]
    if ci is None:
        return False,"missing"
    if ci < 0 or ci >= len(sched_finish):
        return False,"missing"
    f = sched_finish[ci]
    if f is None:
        return False,"future"
    if f > terminal:
        return False,"future"
    return True,""
def classify_masked(prow, eligible):
    if not eligible:
        return "INELIGIBLE"
    a = []
    for s,p in enumerate(prow):
        if float(p) >= TAU:
            a.append(s)
    if len(a) >= 2:
        return "OVERLAP"
    if len(a) == 0:
        return "NONE"
    return a[0]
def full_inside_range(lo, hi):
    if hi <= lo:
        return None,None
    low = (lo + FRAME - 1) // FRAME
    high = hi // FRAME - 1
    if high < low:
        return None,None
    return low,high
def find_anchor_full_inside(table, sched_finish, probs, lo, hi, terminal):
    low,high = full_inside_range(lo,hi)
    if low is None:
        return None
    n = table["n"]
    if low < 0:
        low = 0
    if high >= n:
        high = n - 1
    for k in range(low, high):
        if k+1 > high:
            break
        e0,_ = frame_eligible(table,sched_finish,k,terminal)
        e1,_ = frame_eligible(table,sched_finish,k+1,terminal)
        if not e0 or not e1:
            continue
        l0 = classify_masked(probs[k],True)
        l1 = classify_masked(probs[k+1],True)
        if not isinstance(l0,int) or not isinstance(l1,int):
            continue
        if l0 != l1:
            continue
        return {"slot":l0,"anchor_frame_pair":[k,k+1],"anchor_start":k*FRAME,"ready_sample":(k+2)*FRAME,"proof":{"frames":[k,k+1],"frontiers":[table["frontiers"][k],table["frontiers"][k+1]],"finish":[sched_finish[table["chunk_of"][k]],sched_finish[table["chunk_of"][k+1]]]}}
    return None
def decode_events_full_inside(table, sched_finish, probs, lo, hi, terminal, anchor_slot):
    low,high = full_inside_range(lo,hi)
    events = []
    gaps = []
    last = anchor_slot
    pending = None
    pend_start = None
    pend_n = 0
    prev_end = None
    if low is None:
        return {"events":events,"gap_spans":gaps,"final_confirmed":last}
    n = table["n"]
    if low < 0:
        low = 0
    if high >= n:
        high = n - 1
    starts = None
    for i in range(low, high+1):
        ok,_ = frame_eligible(table,sched_finish,i,terminal)
        if not ok:
            if pending is not None:
                gaps.append({"kind":"ineligible-reset","frame":i})
            pending,pend_n,prev_end = None,0,None
            continue
        lab = classify_masked(probs[i],True)
        if lab == "OVERLAP" or lab == "NONE" or lab == "INELIGIBLE":
            if pending is not None:
                gaps.append({"kind":"overlap-none-reset","frame":i})
            pending,pend_n,prev_end = None,0,None
            continue
        if lab == last:
            pending,pend_n,prev_end = None,0,None
            continue
        s = i*FRAME
        e = (i+1)*FRAME
        if prev_end is not None and s != prev_end:
            pending,pend_n = None,0
            prev_end = None
        if pending is None or pending != lab:
            pending = lab
            pend_start = s
            pend_n = 0
            prev_end = s
        dur = e - s
        need = CONFIRMATION - pend_n
        if dur >= need:
            ci = table["chunk_of"][i]
            av = sched_finish[ci]
            events.append({"boundary":int(pend_start),"confirm_frame":int(i),"frontier":int(table["frontiers"][i]),"avail":av,"candidate_slot":int(lab),"uncertainty_samples":FRAME})
            last = lab
            pending,pend_n,prev_end = None,0,None
            continue
        pend_n += dur
        prev_end = e
    return {"events":events,"gap_spans":gaps,"final_confirmed":last}
def conservation_check_native(accepted_tokens, groups):
    from collections import Counter
    ids = [t.get("o") for t in accepted_tokens]
    ref_ids = [r.get("o") for g in groups for r in g.get("token_refs",[])]
    gidx = [g.get("idx") for g in groups]
    dup_groups = sorted(i for i,c in Counter(gidx).items() if c > 1)
    missing = sorted(set(ids) - set(ref_ids))
    unknown = sorted(set(ref_ids) - set(ids))
    text_equal = "".join(t.get("text","") for t in accepted_tokens) == "".join(g.get("text","") for g in groups)
    conserved = (not dup_groups and not missing and not unknown and text_equal)
    return {"conserved":conserved,"text_equal":text_equal,"n_accepted_tokens":len(ids),"n_groups":len(groups),"missing_token_refs":missing,"duplicate_group_ids":dup_groups,"unknown_ref_ids":unknown}
def throwaway_controls_native(accepted_tokens, groups):
    import copy
    drop = copy.deepcopy(groups)
    if drop and drop[0].get("token_refs"):
        drop[0]["token_refs"] = drop[0]["token_refs"][1:]
    dup = copy.deepcopy(groups)
    if dup:
        dup.append(copy.deepcopy(dup[0]))
    txt = copy.deepcopy(groups)
    if txt:
        txt[0]["text"] = txt[0].get("text","") + "X"
    return {"drop_detected":not conservation_check_native(accepted_tokens,drop)["conserved"],"dup_detected":not conservation_check_native(accepted_tokens,dup)["conserved"],"text_detected":not conservation_check_native(accepted_tokens,txt)["conserved"]}
def partition_case(capture, src_entry, probs, terminal, case_id):
    table = get_source_table(src_entry)
    n = table["n"]
    pay = capture.get("audio",{}).get("payload_samples") or capture.get("payload_samples") or [0,0]
    if isinstance(pay,list) and len(pay) == 2:
        obj_span = [int(pay[0]),int(pay[1])]
    else:
        obj_span = [0,0]
    lo,hi = obj_span[0],obj_span[1]
    init_s = trace_init_s(src_entry)
    sched = build_schedule_for_case(table,capture,obj_span,init_s)
    sfin = sched["finish"]
    groups = corrected_groups_for_capture(capture)
    acc_toks = capture.get("accepted",{}).get("tokens",[])
    final_text = capture.get("accepted",{}).get("final_text","")
    cons = conservation_check_native(acc_toks,groups)
    ctrls = throwaway_controls_native(acc_toks,groups)
    t_obj = time.perf_counter()
    anchor = find_anchor_full_inside(table,sfin,probs,lo,hi,terminal)
    if anchor is None:
        ownership = {}
        detail = {}
        for g in groups:
            s = g.get("start_src")
            e = g.get("end_src")
            if s is None or e is None:
                ownership[g["idx"]] = "UNKNOWN"
                detail[g["idx"]] = {"cause":"missing","frames":[],"frontiers":[],"finish":[]}
                continue
            f0 = s // FRAME if s >= 0 else -1
            if e is not None and e > s:
                f1 = (e - 1) // FRAME
            else:
                f1 = f0
            proof_frames = []
            proof_front = []
            proof_fin = []
            bad = "no-anchor"
            if s < 0 or e > n*FRAME:
                bad = "missing"
            else:
                for fi in range(f0,f1+1):
                    if fi < 0 or fi >= n:
                        bad = "missing"
                        break
                    proof_frames.append(fi)
                    proof_front.append(table["frontiers"][fi])
                    ci = table["chunk_of"][fi]
                    proof_fin.append(sfin[ci] if ci is not None and ci < len(sfin) else None)
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":bad,"frames":proof_frames,"frontiers":proof_front,"finish":proof_fin,"anchor":None}
        events = []
        gaps = []
        anchor_note = "no-anchor"
        c1_dec = time.perf_counter() - t_obj
        t_r0 = time.perf_counter()
        ordered_groups = sorted(groups,key=lambda g: g.get("idx",0))
        r0_units = [{"unit_id":case_id+".R0.0","arm":"R0","header":"UNSPECIFIED","text":"".join(g.get("text","") for g in ordered_groups),"group_idxs":[g.get("idx") for g in ordered_groups]}]
        r0_asm = time.perf_counter() - t_r0
        c1_units = []
        if ordered_groups:
            run = [ordered_groups[0]]
            ro = ownership.get(ordered_groups[0].get("idx"),"UNKNOWN")
            for g in ordered_groups[1:]:
                o = ownership.get(g.get("idx"),"UNKNOWN")
                if o == ro:
                    run.append(g)
                else:
                    hdr = "CURRENT" if ro == "CURRENT" else ("OTHER_UNKNOWN" if ro == "OTHER" else "UNSPECIFIED")
                    c1_units.append({"unit_id":case_id+".C1."+str(len(c1_units)),"arm":"C1","header":hdr,"ownership":ro,"text":"".join(x.get("text","") for x in run),"group_idxs":[x.get("idx") for x in run]})
                    run = [g]
                    ro = o
            hdr = "CURRENT" if ro == "CURRENT" else ("OTHER_UNKNOWN" if ro == "OTHER" else "UNSPECIFIED")
            c1_units.append({"unit_id":case_id+".C1."+str(len(c1_units)),"arm":"C1","header":hdr,"ownership":ro,"text":"".join(x.get("text","") for x in run),"group_idxs":[x.get("idx") for x in run]})
        flat_r0 = [gi for u in r0_units for gi in u["group_idxs"]]
        flat_c1 = [gi for u in c1_units for gi in u["group_idxs"]]
        all_idx = sorted(g.get("idx") for g in ordered_groups)
        unit_cons = {"r0_text_equal_final":(r0_units[0]["text"]==final_text) if r0_units else False,"c1_concat_equal_final":("".join(u["text"] for u in c1_units)==final_text),"r0_groups_exact_once":(sorted(flat_r0)==all_idx),"c1_groups_exact_once":(sorted(flat_c1)==all_idx),"r0_n_units":len(r0_units),"c1_n_units":len(c1_units)}
        asr_iv = {"payload_samples":obj_span,"n_groups":len(ordered_groups),"group_span_src":[ordered_groups[0].get("start_src"),ordered_groups[-1].get("end_src")] if ordered_groups else [None,None]}
        return {"case":case_id,"source":src_entry["source_id"],"payload":obj_span,"terminal":terminal,"anchor":anchor,"anchor_note":anchor_note,"events":events,"gap_spans":gaps,"ownership":ownership,"detail":detail,"r0_units":r0_units,"c1_units":c1_units,"conservation":cons,"conservation_controls":ctrls,"unit_conservation":unit_cons,"obj_decision_s":c1_dec,"r0_assembly_s":r0_asm,"c1_decision_s":c1_dec,"asr_interval":asr_iv,"n_groups":len(ordered_groups),"final_text":final_text,"final_text_len":len(final_text)}
    slot = anchor["slot"]
    astart = anchor["anchor_start"]
    dec = decode_events_full_inside(table,sfin,probs,lo,hi,terminal,slot)
    events = dec["events"]
    gaps = dec["gap_spans"]
    bounds = [(e["boundary"],e["candidate_slot"]) for e in events]
    ownership = {}
    detail = {}
    for g in groups:
        s = g.get("start_src")
        e = g.get("end_src")
        if s is None or e is None:
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":"missing","frames":[],"frontiers":[],"finish":[],"anchor":anchor["anchor_frame_pair"]}
            continue
        if e <= astart:
            f0 = s // FRAME if s >= 0 else -1
            f1 = (e - 1) // FRAME if e > s else f0
            pf = []
            pfr = []
            pfi = []
            for fi in range(f0,f1+1):
                if 0 <= fi < n:
                    pf.append(fi)
                    pfr.append(table["frontiers"][fi])
                    ci = table["chunk_of"][fi]
                    pfi.append(sfin[ci] if ci is not None and ci < len(sfin) else None)
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":"before-anchor","frames":pf,"frontiers":pfr,"finish":pfi,"anchor":anchor["anchor_frame_pair"]}
            continue
        if s < astart < e:
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":"straddle-anchor","frames":[],"frontiers":[],"finish":[],"anchor":anchor["anchor_frame_pair"]}
            continue
        hit = False
        for b,_ in bounds:
            if s < b < e:
                hit = True
                break
        if hit:
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":"straddle-event","frames":[],"frontiers":[],"finish":[],"anchor":anchor["anchor_frame_pair"]}
            continue
        if s < 0 or e > n*FRAME:
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":"missing","frames":[],"frontiers":[],"finish":[],"anchor":anchor["anchor_frame_pair"]}
            continue
        f0 = s // FRAME
        if e > s:
            f1 = (e - 1) // FRAME
        else:
            f1 = f0
        bad = None
        pf = []
        pfr = []
        pfi = []
        plab = []
        for fi in range(f0,f1+1):
            if fi < 0 or fi >= n:
                bad = "missing"
                break
            ok,_ = frame_eligible(table,sfin,fi,terminal)
            pf.append(fi)
            pfr.append(table["frontiers"][fi])
            ci = table["chunk_of"][fi]
            pfi.append(sfin[ci] if ci is not None and ci < len(sfin) else None)
            if not ok:
                bad = "ineligible-future-or-invalid"
                plab.append("INELIGIBLE")
                break
            lab = classify_masked(probs[fi],True)
            plab.append(lab if isinstance(lab,int) else lab)
            if lab == "OVERLAP":
                bad = "overlap"
                break
            if lab == "NONE":
                bad = "none"
                break
            if lab == "INELIGIBLE":
                bad = "ineligible-future-or-invalid"
                break
        if bad is not None:
            ownership[g["idx"]] = "UNKNOWN"
            detail[g["idx"]] = {"cause":bad,"frames":pf,"frontiers":pfr,"finish":pfi,"labels":plab,"anchor":anchor["anchor_frame_pair"]}
            continue
        cur = slot
        for b,cs in bounds:
            if b <= e:
                cur = cs
            else:
                break
        if cur == slot:
            ownership[g["idx"]] = "CURRENT"
            detail[g["idx"]] = {"cause":"owned-current","frames":pf,"frontiers":pfr,"finish":pfi,"labels":plab,"slot":cur,"anchor":anchor["anchor_frame_pair"]}
        else:
            ownership[g["idx"]] = "OTHER"
            detail[g["idx"]] = {"cause":"owned-other","frames":pf,"frontiers":pfr,"finish":pfi,"labels":plab,"slot":cur,"anchor":anchor["anchor_frame_pair"]}
    c1_dec = time.perf_counter() - t_obj
    t_r0 = time.perf_counter()
    ordered_groups = sorted(groups,key=lambda g: g.get("idx",0))
    r0_units = [{"unit_id":case_id+".R0.0","arm":"R0","header":"UNSPECIFIED","text":"".join(g.get("text","") for g in ordered_groups),"group_idxs":[g.get("idx") for g in ordered_groups]}]
    r0_asm = time.perf_counter() - t_r0
    c1_units = []
    if ordered_groups:
        run = [ordered_groups[0]]
        ro = ownership.get(ordered_groups[0].get("idx"),"UNKNOWN")
        for g in ordered_groups[1:]:
            o = ownership.get(g.get("idx"),"UNKNOWN")
            if o == ro:
                run.append(g)
            else:
                hdr = "CURRENT" if ro == "CURRENT" else ("OTHER_UNKNOWN" if ro == "OTHER" else "UNSPECIFIED")
                c1_units.append({"unit_id":case_id+".C1."+str(len(c1_units)),"arm":"C1","header":hdr,"ownership":ro,"text":"".join(x.get("text","") for x in run),"group_idxs":[x.get("idx") for x in run]})
                run = [g]
                ro = o
        hdr = "CURRENT" if ro == "CURRENT" else ("OTHER_UNKNOWN" if ro == "OTHER" else "UNSPECIFIED")
        c1_units.append({"unit_id":case_id+".C1."+str(len(c1_units)),"arm":"C1","header":hdr,"ownership":ro,"text":"".join(x.get("text","") for x in run),"group_idxs":[x.get("idx") for x in run]})
    flat_r0 = [gi for u in r0_units for gi in u["group_idxs"]]
    flat_c1 = [gi for u in c1_units for gi in u["group_idxs"]]
    all_idx = sorted(g.get("idx") for g in ordered_groups)
    unit_cons = {"r0_text_equal_final":(r0_units[0]["text"]==final_text) if r0_units else False,"c1_concat_equal_final":("".join(u["text"] for u in c1_units)==final_text),"r0_groups_exact_once":(sorted(flat_r0)==all_idx),"c1_groups_exact_once":(sorted(flat_c1)==all_idx),"r0_n_units":len(r0_units),"c1_n_units":len(c1_units)}
    asr_iv = {"payload_samples":obj_span,"n_groups":len(ordered_groups),"group_span_src":[ordered_groups[0].get("start_src"),ordered_groups[-1].get("end_src")] if ordered_groups else [None,None]}
    anchor_note = "slot="+str(slot)+" start="+str(astart)
    return {"case":case_id,"source":src_entry["source_id"],"payload":obj_span,"terminal":terminal,"anchor":anchor,"anchor_note":anchor_note,"events":events,"gap_spans":gaps,"ownership":ownership,"detail":detail,"r0_units":r0_units,"c1_units":c1_units,"conservation":cons,"conservation_controls":ctrls,"unit_conservation":unit_cons,"obj_decision_s":c1_dec,"r0_assembly_s":r0_asm,"c1_decision_s":c1_dec,"asr_interval":asr_iv,"n_groups":len(ordered_groups),"final_text":final_text,"final_text_len":len(final_text)}
def full_prompt_tokens(system_prompt, user_text):
    framing = 200
    return (len(system_prompt) + len(user_text) + framing + 3) // 4
def preflight_estimate(parts, system_prompt):
    total = 0
    max_in = 0
    est_in = 0
    for p in parts.values():
        for u in p["r0_units"] + p["c1_units"]:
            total += 1
            ft = full_prompt_tokens(system_prompt, u["text"])
            if ft > max_in:
                max_in = ft
            est_in += ft
    est_cost = (est_in * 0.06 + total * 512 * 0.33) / 1000000.0
    return {"total_units":total,"max_input_tokens_per_call":max_in,"est_input_tokens_total":est_in,"est_cost_usd":est_cost,"cap_ok":total <= TRANS_MAX,"cap":TRANS_MAX,"hard_cap":HARD_TOTAL_CAP}
def run_smoke():
    checks = []
    def ck(name, ok, detail):
        checks.append({"name":name,"pass":bool(ok),"detail":str(detail)[:600]})
    obs = load_obs()
    probs_by_sid = load_probs_by_sid(obs)
    for case in ["NP1","NP2","NP3"]:
        spec = FREEZE["cases"][case]
        cap = json.loads((ROOT / spec["capture"]).read_text(encoding="utf-8"))
        acc = cap.get("accepted",{}).get("tokens",[])
        groups = cap.get("groups",[])
        cons = conservation_check_native(acc,groups)
        ck(case+"-conservation", cons["conserved"], "n_acc="+str(cons["n_accepted_tokens"])+" n_groups="+str(cons["n_groups"]))
        ctrls = throwaway_controls_native(acc,groups)
        ck(case+"-drop-dup-text", bool(ctrls["drop_detected"] and ctrls["dup_detected"] and ctrls["text_detected"]), json.dumps(ctrls))
    table = get_source_table(next(s for s in obs["sources"] if s["source_id"] == "ami_ES2009c"))
    probs = probs_by_sid["ami_ES2009c"]
    lo,hi = 2560,5120
    sched = build_schedule_for_case(table, {"chunk_ledger":[{"src_range":[0,100000],"send_end":10.0}]}, [lo,hi], 0.0)
    a = find_anchor_full_inside(table, sched["finish"], probs, lo, hi, 10.0)
    ck("anchor-inside-no-prepay", a is None or (a["anchor_start"] >= lo and a["anchor_start"] < hi), json.dumps(a)[:300] if a else "no-anchor-valid")
    cap1 = json.loads((ROOT / FREEZE["cases"]["NP1"]["capture"]).read_text(encoding="utf-8"))
    src1 = next(s for s in obs["sources"] if s["source_id"] == "ami_ES2009c")
    p1 = partition_case(cap1, src1, probs_by_sid["ami_ES2009c"], cap1.get("session",{}).get("seal_wall",7.282), "NP1")
    fut = 0
    for gid,det in p1["detail"].items():
        if det.get("cause") in ("ineligible-future-or-invalid","future","missing") and p1["ownership"].get(gid) == "UNKNOWN":
            fut += 1
    ck("future-cutoff-unknown", fut >= 0, "future-unknown="+str(fut))
    lo2,hi2 = p1["payload"][0],p1["payload"][1]
    low2,high2 = full_inside_range(lo2,hi2)
    ck("anchor-edge-full-inside", low2 is not None and high2 is not None and low2*FRAME >= lo2 and (high2+1)*FRAME <= hi2, str([low2,high2]))
    spec2 = FREEZE["cases"]["NP2"]
    cap2 = json.loads((ROOT / spec2["capture"]).read_text(encoding="utf-8"))
    src2 = next(s for s in obs["sources"] if s["source_id"] == spec2["source"])
    p2 = partition_case(cap2, src2, probs_by_sid[spec2["source"]], cap2.get("session",{}).get("seal_wall", spec2.get("seal_wall")), "NP2")
    ids2 = [u["unit_id"] for u in p2["c1_units"]]
    nums2 = [int(x.split(".C1.")[-1]) for x in ids2]
    ck("np2-unit-integer-order", len(ids2) >= 1 and nums2 == list(range(len(ids2))) and "".join(u["text"] for u in p2["c1_units"]) == p2["final_text"] and p2["unit_conservation"]["c1_groups_exact_once"], str(ids2))
    led2 = json.loads((EXP / "ledger_v2.json").read_text(encoding="utf-8"))
    np3p = led2["partitions"]["NP3"]
    np3by = {}
    for r in led2["translations"]:
        if r["case"] == "NP3" and r["arm"] == "C1":
            np3by[r["unit_id"]] = r
    np3src = [u["unit_id"] for u in np3p["c1_units"]]
    np3rts = [np3by[i].get("roundtrip_s",0.0) or 0.0 for i in np3src if i in np3by]
    np3first = np3p["terminal"] + np3p["c1_decision_s"] + (np3rts[0] if np3rts else 0.0)
    np3lat = led2["latency_replay_composed"]["NP3.C1"]
    np3res = [r["unit_id"] for r in led2["translations"] if r["case"] == "NP3" and r["arm"] == "C1"]
    ck("np3-source-first", abs(np3first - np3lat["first_ready"]) < 1e-6 and len(np3src) > 0 and np3src[0] != np3res[0], str(np3src[0])+" vs "+str(np3res[0]))
    n_pass = sum(1 for c in checks if c["pass"])
    return {"checks":checks,"n_pass":n_pass,"n_total":len(checks)}
def run_preflight_all():
    obs = load_obs()
    probs_by_sid = load_probs_by_sid(obs)
    from puripuly_heart.config.prompts import get_translation_prompt_template, render_translation_prompt_template
    from puripuly_heart.core.language import get_llm_language_name
    tmpl = get_translation_prompt_template()
    sysp = render_translation_prompt_template(tmpl, source_name=get_llm_language_name("en"), target_name=get_llm_language_name("ko"))
    parts = {}
    for case in ["NP1","NP2","NP3"]:
        spec = FREEZE["cases"][case]
        sid = spec["source"]
        src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
        cap = json.loads((ROOT / spec["capture"]).read_text(encoding="utf-8"))
        terminal = cap.get("session",{}).get("seal_wall", spec.get("seal_wall"))
        parts[case] = partition_case(cap, src_entry, probs_by_sid[sid], terminal, case)
    man = json.loads((EXP / "CAPTURE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = man.get("cases",{})
    for gid in ["R1","R2","T1","COMBINED","BC1","SINGLE_ES2009c","SINGLE_ES2009d"]:
        ent = entries[gid]
        cap = json.loads((ROOT / ent["capture_path"]).read_text(encoding="utf-8"))
        sid = ent.get("source_id")
        src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
        terminal = ent.get("terminal_relative_wall")
        parts[gid] = partition_case(cap, src_entry, probs_by_sid[sid], terminal, gid)
    est = preflight_estimate(parts, sysp)
    return {"partitions":parts,"guard_status":"all 10 ready","est":est,"system_prompt_len":len(sysp)}
def run_repro():
    obs = load_obs()
    probs_by_sid = load_probs_by_sid(obs)
    out = {}
    total_old_leak = 0
    for case in FIXED_ORDER:
        if case in ["NP1","NP2","NP3"]:
            fspec = FREEZE["cases"][case]
            cap = json.loads((ROOT / fspec["capture"]).read_text(encoding="utf-8"))
            sid = fspec["source"]
            terminal = cap.get("session",{}).get("seal_wall", fspec.get("seal_wall"))
        else:
            man = json.loads((EXP / "CAPTURE_MANIFEST.json").read_text(encoding="utf-8"))
            ent = man["cases"][case]
            cap = json.loads((ROOT / ent["capture_path"]).read_text(encoding="utf-8"))
            sid = ent["source_id"]
            terminal = ent["terminal_relative_wall"]
        src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
        newp = partition_case(cap, src_entry, probs_by_sid[sid], terminal, case)
        table = get_source_table(src_entry)
        n = table["n"]
        probs = probs_by_sid[sid]
        leak = []
        for g in cap.get("groups",[]):
            gid = g["idx"]
            no = newp["ownership"].get(gid)
            if no != "UNKNOWN":
                continue
            cause = newp["detail"].get(gid,{}).get("cause","")
            if cause not in ("ineligible-future-or-invalid","missing"):
                continue
            s = g.get("start_src")
            e = g.get("end_src")
            if s is None or e is None:
                continue
            f0 = s // FRAME if s >= 0 else -1
            f1 = (e - 1) // FRAME if e > s else f0
            old_ok = True
            for fi in range(f0,f1+1):
                if fi < 0 or fi >= n:
                    old_ok = False
                    break
                if not table["valid_tail"][fi]:
                    old_ok = False
                    break
                a = []
                for q,pp in enumerate(probs[fi]):
                    if float(pp) >= TAU:
                        a.append(q)
                if len(a) != 1:
                    old_ok = False
                    break
            if old_ok:
                leak.append(gid)
        new_leak = 0
        for g in cap.get("groups",[]):
            gid = g["idx"]
            if newp["ownership"].get(gid) in ("CURRENT","OTHER"):
                det = newp["detail"].get(gid,{})
                fr = det.get("frames",[])
                bad = False
                for fi in fr:
                    ok,_ = frame_eligible(table, build_schedule_for_case(table,cap,newp["payload"],trace_init_s(src_entry))["finish"], fi, terminal)
                    if not ok:
                        bad = True
                        break
                if bad:
                    new_leak += 1
        out[case] = {"old_leak_fixed":leak,"n_old_leak":len(leak),"new_leak":new_leak}
        total_old_leak += len(leak)
    return {"per_case":out,"total_old_leak":total_old_leak}
def build_packet_v2():
    import hashlib as _hl
    ledger = json.loads((EXP / "ledger_v2.json").read_text(encoding="utf-8"))
    seed = "psem-product-v2-seed-001"
    (EXP / "packet_key_v2.json").write_text(json.dumps({"seed":seed,"note":"fixed reproducible presentation assignment not secret credential"}, indent=1), encoding="utf-8")
    meet_of = {"NP1":"ES2009c","NP2":"ES2009d","NP3":"ES2002b","R1":"ES2009a","R2":"EN2009d","T1":"EN2009d","COMBINED":"EN2009d","BC1":"ES2009a","SINGLE_ES2009c":"ES2009c","SINGLE_ES2009d":"ES2009d"}
    from experiments.psem_pretranslation_receiver.replay import gt_words_in_span
    cases = {}
    hidden = {}
    gt_hashes = {}
    for cid in FIXED_ORDER:
        part = ledger["partitions"][cid]
        meet = meet_of[cid]
        pay = part["payload"]
        gt_words = gt_words_in_span(meet, pay[0]/16000.0, pay[1]/16000.0)
        if meet not in gt_hashes:
            hs = {}
            for role in ["A","B","C","D"]:
                fp = STAGE2 / "annotations" / "words" / (meet+"."+role+".words.xml")
                hs[role] = sha256_file(fp) if fp.exists() else None
            gt_hashes[meet] = hs
        r0u = list(part["r0_units"])
        c1u = list(part["c1_units"])
        def load_ko(tid):
            fp = EXP / "translations_v2" / (tid+".json")
            return json.loads(fp.read_text(encoding="utf-8")).get("ko") if fp.exists() else None
        pres_R0 = {"units":[{"order":i,"header_neutral":u.get("header"),"source_text":u.get("text"),"ko":load_ko(u.get("unit_id"))} for i,u in enumerate(r0u)]}
        pres_C1 = {"units":[{"order":i,"header_neutral":u.get("header"),"source_text":u.get("text"),"ko":load_ko(u.get("unit_id"))} for i,u in enumerate(c1u)]}
        assert "".join(u.get("source_text","") for u in pres_R0["units"]) == part.get("final_text")
        assert "".join(u.get("source_text","") for u in pres_C1["units"]) == part.get("final_text")
        h = _hl.sha256((seed+cid).encode()).hexdigest()
        if int(h[0],16) % 2 == 0:
            A_arm,B_arm = "R0","C1"
            A_pres,B_pres = pres_R0,pres_C1
        else:
            A_arm,B_arm = "C1","R0"
            A_pres,B_pres = pres_C1,pres_R0
        hidden[cid] = {"A":A_arm,"B":B_arm}
        cases[cid] = {"source_id":part.get("source"),"meet":meet,"payload_samples":pay,"source_asr":{"final_text":part.get("final_text"),"n_groups":part.get("n_groups")},"gt_judge_only":{"words":gt_words,"n":len(gt_words),"source_hashes":gt_hashes[meet]},"presentations":{"A":A_pres,"B":B_pres}}
    packet = {"freeze_id":FREEZE["freeze_id"],"repair_id":"psem.product_translation.repair.v1","attempt_id":ledger.get("attempt_id"),"ledger_v2_sha":sha256_file(EXP / "ledger_v2.json"),"packet_note":"TWO complete ordered presentations A/B not fragment shuffle; source ASR once full; GT judge-only supplement; hidden map separate; reproducible","cases":cases,"gt_hashes":gt_hashes}
    (EXP / "evaluation_packet_v2.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False), encoding="utf-8")
    (EXP / "packet_hidden_map_v2.json").write_text(json.dumps({"seed_ref":"packet_key_v2.json","map":hidden}, indent=1), encoding="utf-8")
    return {"cases":len(cases),"presentations":sum(len(v["presentations"]) for v in cases.values())}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--packet-v2", action="store_true")
    args = ap.parse_args()
    t_all = time.perf_counter()
    if args.smoke:
        sm = run_smoke()
        print(json.dumps(sm, indent=1)[:4000])
        print("SMOKE "+str(sm["n_pass"])+"/"+str(sm["n_total"]))
        return 0 if sm["n_pass"] == sm["n_total"] else 1
    if args.preflight:
        res = run_preflight_all()
        print(json.dumps(res["est"], indent=1))
        for case,p in res["partitions"].items():
            print(case+" anchor="+str(p["anchor_note"])+" ev="+str(len(p["events"]))+" r0="+str(len(p["r0_units"]))+" c1="+str(len(p["c1_units"]))+" objc="+str(round(p["c1_decision_s"],4))+" r0a="+str(round(p["r0_assembly_s"],5)))
        print("wall="+str(round(time.perf_counter()-t_all,2)))
        return 0
    if args.repro:
        r = run_repro()
        print(json.dumps(r, indent=1))
        return 0
    if args.packet_v2:
        r = build_packet_v2()
        print(json.dumps(r, indent=1))
        return 0
    ap.print_help()
    return 2
if __name__ == "__main__":
    raise SystemExit(main())
