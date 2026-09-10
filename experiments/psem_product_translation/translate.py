from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments" / "psem_product_translation"
FREEZE = json.loads((EXP / "FREEZE.json").read_text(encoding="utf-8"))
FIXED_ORDER = FREEZE["execution_order"]["fixed_case_order"]
MODEL = "google/gemma-4-26b-a4b-it"
TIMEOUT_S = 30.0
MAX_OUT = 512
TRANS_MAX = 128
HARD_CAP = 129
OUT_DIR = EXP / "translations_v2"
LEDGER_V2 = EXP / "ledger_v2.json"
PAID_V2 = EXP / "paid_counter_v2.json"
from experiments.psem_product_translation.probe import load_obs, load_probs_by_sid, partition_case, full_prompt_tokens
def sha256_file(p):
    h = hashlib.sha256()
    f = open(p,"rb")
    b = f.read(1048576)
    while b:
        h.update(b)
        b = f.read(1048576)
    f.close()
    return h.hexdigest()
def load_api_key_safe():
    p = ROOT / ".env.local"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):]
        if "=" not in s:
            continue
        k,v = s.split("=",1)
        if k.strip() == "OPENROUTER_API_KEY":
            v = v.strip().strip('"').strip("'")
            return v or None
    return None
def collect_partitions():
    import json as _j
    obs = load_obs()
    probs_by_sid = load_probs_by_sid(obs)
    parts = {}
    for case in ["NP1","NP2","NP3"]:
        spec = FREEZE["cases"][case]
        sid = spec["source"]
        src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
        cap = _j.loads((ROOT / spec["capture"]).read_text(encoding="utf-8"))
        terminal = cap.get("session",{}).get("seal_wall", spec.get("seal_wall"))
        parts[case] = partition_case(cap, src_entry, probs_by_sid[sid], terminal, case)
    man = _j.loads((EXP / "CAPTURE_MANIFEST.json").read_text(encoding="utf-8"))
    entries = man.get("cases",{})
    for gid in ["R1","R2","T1","COMBINED","BC1","SINGLE_ES2009c","SINGLE_ES2009d"]:
        ent = entries[gid]
        cap = _j.loads((ROOT / ent["capture_path"]).read_text(encoding="utf-8"))
        sid = ent.get("source_id")
        src_entry = next(s for s in obs["sources"] if s["source_id"] == sid)
        terminal = ent.get("terminal_relative_wall")
        parts[gid] = partition_case(cap, src_entry, probs_by_sid[sid], terminal, gid)
    return parts, obs, man
def freeze_runmanifest(parts):
    man = json.loads((EXP / "CAPTURE_MANIFEST.json").read_text(encoding="utf-8"))
    files = {}
    files["FREEZE.json"] = sha256_file(EXP / "FREEZE.json")
    files["probe.py"] = sha256_file(EXP / "probe.py")
    files["translate.py"] = sha256_file(EXP / "translate.py")
    files["OBSERVATIONS.json"] = sha256_file(ROOT / "experiments/psem_phase_a_headroom/observations/OBSERVATIONS.json")
    files["CAPTURE_MANIFEST.json"] = sha256_file(EXP / "CAPTURE_MANIFEST.json")
    for case in ["NP1","NP2","NP3"]:
        spec = FREEZE["cases"][case]
        files[spec["capture"]] = sha256_file(ROOT / spec["capture"])
    for gid,ent in man.get("cases",{}).items():
        files[ent["capture_path"]] = sha256_file(ROOT / ent["capture_path"])
    terms = {cid:parts[cid]["terminal"] for cid in FIXED_ORDER if cid in parts}
    payloads = {cid:parts[cid]["payload"] for cid in FIXED_ORDER if cid in parts}
    return {"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"files":files,"terminals":terms,"payloads":payloads}
def load_original_map():
    import json as _j
    old_ledger = _j.loads((EXP / "ledger.json").read_text(encoding="utf-8"))
    m = {}
    for r in old_ledger.get("translations",[]):
        key = (r.get("case"),r.get("arm"),r.get("source_text"))
        if key not in m:
            tid = r.get("unit_id")
            fp = EXP / "translations" / (tid+".json")
            fh = sha256_file(fp) if fp.exists() else None
            m[key] = {"record":r,"file_sha":fh,"unit_id":tid}
    return m, old_ledger
async def run_live(attempt_id):
    t_start = time.perf_counter()
    if not attempt_id or len(attempt_id) < 4:
        print("need --attempt-id unique >=4 chars")
        return 2
    if LEDGER_V2.exists():
        print("fail-closed ledger_v2 exists no autorerun")
        return 2
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        print("fail-closed translations_v2 non-empty no autorerun")
        return 2
    if PAID_V2.exists():
        print("fail-closed paid_counter_v2 exists no autorerun")
        return 2
    try:
        orig_paid = json.loads((EXP / "paid_counter.json").read_text(encoding="utf-8"))
        orig_count = int(orig_paid.get("count",0))
        orig_auth = int(orig_paid.get("auth",0))
    except Exception:
        print("fail-closed original paid counter corrupt missing")
        return 2
    if orig_count != 50 or orig_auth != 1:
        print("fail-closed original paid unexpected")
        return 2
    parts,obs,man = collect_partitions()
    from puripuly_heart.config.prompts import get_translation_prompt_template, render_translation_prompt_template
    from puripuly_heart.core.language import get_llm_language_name
    tmpl = get_translation_prompt_template()
    sysp = render_translation_prompt_template(tmpl, source_name=get_llm_language_name("en"), target_name=get_llm_language_name("ko"))
    total = sum(len(p["r0_units"])+len(p["c1_units"]) for p in parts.values())
    max_in = 0
    for p in parts.values():
        for u in p["r0_units"]+p["c1_units"]:
            ft = full_prompt_tokens(sysp, u["text"])
            if ft > max_in:
                max_in = ft
    if total > TRANS_MAX:
        print("escalate cap total exceeds")
        return 2
    if max_in > 4000:
        print("escalate input exceeds 4000")
        return 2
    old_map,old_ledger = load_original_map()
    old_usage = [r.get("usage") or {} for r in old_ledger.get("translations",[])]
    old_pt = sum((u.get("prompt_tokens") or 0) for u in old_usage)
    old_ct = sum((u.get("completion_tokens") or 0) for u in old_usage)
    old_cost = sum(float(u.get("cost") or 0) for u in old_usage)
    reuse = []
    need = []
    for cid in FIXED_ORDER:
        p = parts[cid]
        for u in p["r0_units"]:
            key = (cid,"R0",u["text"])
            hit = old_map.get(key)
            if hit and hit["record"].get("arm") == "R0":
                reuse.append((cid,"R0",u,hit))
            else:
                need.append((cid,"R0",u))
        for u in p["c1_units"]:
            key = (cid,"C1",u["text"])
            hit = old_map.get(key)
            if hit and hit["record"].get("arm") == "C1":
                reuse.append((cid,"C1",u,hit))
            else:
                need.append((cid,"C1",u))
    if orig_count + len(need) > HARD_CAP:
        print("fail-closed cap would exceed before execution")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runman = freeze_runmanifest(parts)
    runman["attempt_id"] = attempt_id
    runman["reuse_n"] = len(reuse)
    runman["new_n"] = len(need)
    (OUT_DIR / "_runmanifest_v2.json").write_text(json.dumps(runman, indent=1), encoding="utf-8")
    key = load_api_key_safe()
    if not key:
        print("blocked provider unavailable")
        return 3
    from puripuly_heart.config.settings import OpenRouterRoutingMode, OpenRouterProviderRouting
    from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
    import httpx
    provider = OpenRouterLLMProvider(api_key=key, model=MODEL, models=(MODEL,), routing_mode=OpenRouterRoutingMode.LATENCY, provider_routing=OpenRouterProviderRouting.GEMMA4_26B_31B_LATENCY, max_tokens=MAX_OUT, timeout=TIMEOUT_S)
    last = {}
    orig_post = httpx.AsyncClient.post
    async def rec_post(self, url, *a, **k):
        resp = await orig_post(self, url, *a, **k)
        try:
            d = resp.json()
            last["data"] = {"choices":d.get("choices"),"usage":d.get("usage"),"model":d.get("model"),"status":resp.status_code}
        except Exception as e:
            last["data"] = {"status":resp.status_code,"parse_error":repr(e)[:200]}
        return resp
    httpx.AsyncClient.post = rec_post
    paid = {"count":orig_count,"auth":orig_auth,"units":list(orig_paid.get("units",[])),"v2_new_units":[],"attempt_id":attempt_id}
    def persist():
        PAID_V2.write_text(json.dumps(paid, indent=1), encoding="utf-8")
    persist()
    results = []
    req_log = []
    for cid,arm,u,hit in reuse:
        rec0 = hit.get("record") if isinstance(hit,dict) else None
        if rec0 is None:
            continue
        tid = u.get("unit_id") if isinstance(u,dict) else None
        results.append({"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":u.get("text"),"ko":rec0.get("ko"),"ok":rec0.get("ok"),"error":rec0.get("error"),"roundtrip_s":rec0.get("roundtrip_s"),"usage":rec0.get("usage"),"finish_reason":rec0.get("finish_reason"),"model":rec0.get("model",MODEL),"provenance":{"reused":True,"original_unit_id":rec0.get("unit_id"),"original_file_sha":hit.get("file_sha"),"original_roundtrip_s":rec0.get("roundtrip_s")}})
    need_sorted = sorted(need, key=lambda x: (FIXED_ORDER.index(x[0]), int(x[2]["unit_id"].split(".")[-1])))
    for cid,arm,u in need_sorted:
        tid = u["unit_id"]
        text = u["text"]
        if full_prompt_tokens(sysp, text) > 4000:
            results.append({"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":text,"ko":None,"ok":False,"error":"input exceeds 4000","roundtrip_s":0.0,"provenance":{"reused":False,"attempt_id":attempt_id}})
            continue
        if int(paid.get("count",0)) + 1 > HARD_CAP:
            results.append({"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":text,"ko":None,"ok":False,"error":"cap would exceed","roundtrip_s":0.0,"provenance":{"reused":False,"attempt_id":attempt_id}})
            break
        paid["count"] = int(paid.get("count",0)) + 1
        paid["v2_new_units"].append(tid)
        paid["units"].append(tid)
        persist()
        last.clear()
        uid = uuid.uuid5(uuid.NAMESPACE_URL, tid+"::v2::"+attempt_id)
        t0 = time.perf_counter()
        wall = datetime.now(timezone.utc).isoformat()
        try:
            tr = await provider.translate(utterance_id=uid, text=text, system_prompt=sysp, source_language="en", target_language="ko", context="")
            dt = time.perf_counter() - t0
            ko = tr.text if hasattr(tr,"text") else str(tr)
            data = last.get("data",{})
            ch = (data.get("choices") or [{}])[0] if isinstance(data.get("choices"),list) else {}
            fr = ch.get("finish_reason") if isinstance(ch,dict) else None
            usage = data.get("usage")
            if fr == "length":
                rec = {"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":text,"ko":None,"ok":False,"error":"truncated","roundtrip_s":dt,"usage":usage,"finish_reason":fr,"model":data.get("model",MODEL),"provenance":{"reused":False,"attempt_id":attempt_id,"request_wall":wall,"input_hash":hashlib.sha256(text.encode()).hexdigest()[:16]}}
            else:
                rec = {"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":text,"ko":ko,"ok":True,"error":None,"roundtrip_s":dt,"usage":usage,"finish_reason":fr,"model":data.get("model",MODEL),"provenance":{"reused":False,"attempt_id":attempt_id,"request_wall":wall,"input_hash":hashlib.sha256(text.encode()).hexdigest()[:16]}}
            results.append(rec)
            req_log.append({"unit_id":tid,"attempt_id":attempt_id,"request_sanitized":{"model":MODEL,"max_tokens":MAX_OUT,"timeout_s":TIMEOUT_S,"source":"en","target":"ko","context":"","text_len":len(text),"full_prompt_tok":full_prompt_tokens(sysp,text)},"response_sanitized":{"status":data.get("status"),"finish_reason":fr,"usage":usage}})
            (OUT_DIR / (tid+".json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            dt = time.perf_counter() - t0
            msg = repr(e)[:500]
            rec = {"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":text,"ko":None,"ok":False,"error":msg,"roundtrip_s":dt,"provenance":{"reused":False,"attempt_id":attempt_id,"request_wall":wall}}
            results.append(rec)
            req_log.append({"unit_id":tid,"attempt_id":attempt_id,"request_sanitized":{"model":MODEL},"response_sanitized":{"error":msg[:300]}})
            (OUT_DIR / (tid+".json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        await provider.close()
    except Exception:
        pass
    httpx.AsyncClient.post = orig_post
    for cid,arm,u,hit in reuse:
        tid = u["unit_id"]
        fp = OUT_DIR / (tid+".json")
        if not fp.exists():
            rec0 = hit["record"]
            rec = {"unit_id":tid,"case":cid,"arm":arm,"header":u.get("header"),"source_text":u.get("text"),"ko":rec0.get("ko"),"ok":rec0.get("ok"),"error":rec0.get("error"),"roundtrip_s":rec0.get("roundtrip_s"),"usage":rec0.get("usage"),"finish_reason":rec0.get("finish_reason"),"model":rec0.get("model",MODEL),"provenance":{"reused":True,"original_unit_id":rec0.get("unit_id"),"original_file_sha":hit.get("file_sha"),"original_roundtrip_s":rec0.get("roundtrip_s")}}
            fp.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    new_usage = [r.get("usage") or {} for r in results if r.get("provenance",{}).get("reused") is False and r.get("usage")]
    new_pt = sum((x.get("prompt_tokens") or 0) for x in new_usage)
    new_ct = sum((x.get("completion_tokens") or 0) for x in new_usage)
    new_cost = sum(float(x.get("cost") or 0) for x in new_usage)
    lat = {}
    by_unit = {}
    for r in results:
        by_unit[r.get("unit_id")] = r
    for cid in FIXED_ORDER:
        p = parts[cid]
        asr_fin = p["terminal"]
        c1c = p["c1_decision_s"]
        r0c = p["r0_assembly_s"]
        for arm in ["R0","C1"]:
            src_ids = [u["unit_id"] for u in (p["r0_units"] if arm == "R0" else p["c1_units"])]
            urs = [by_unit[tid] for tid in src_ids if tid in by_unit]
            rts = [r.get("roundtrip_s",0.0) or 0.0 for r in urs]
            ownc = r0c if arm == "R0" else c1c
            lat[cid+"."+arm] = {"asr_fin":asr_fin,"own_arm_cost":ownc,"n_requests":len(urs),"n_ok":sum(1 for r in urs if r.get("ok")),"roundtrip_sum":sum(rts),"first_ready":asr_fin+ownc+(rts[0] if rts else 0.0),"all_ready":asr_fin+ownc+sum(rts),"note":"replay-composed NOT same-time; single trial cache/order biased; reused outputs keep original walls no fresh-trial pretend"}
    full = {"freeze_id":FREEZE["freeze_id"],"repair_id":"psem.product_translation.repair.v1","attempt_id":attempt_id,"generated_at_utc":datetime.now(timezone.utc).isoformat(),"baseline_commit":FREEZE["authority"]["baseline_commit"],"capture_manifest_sha":sha256_file(EXP / "CAPTURE_MANIFEST.json"),"runmanifest":runman,"partitions":{k:{"anchor":v["anchor"],"anchor_note":v["anchor_note"],"events":v["events"],"gap_spans":v["gap_spans"],"ownership":v["ownership"],"detail":v["detail"],"r0_units":v["r0_units"],"c1_units":v["c1_units"],"conservation":v["conservation"],"unit_conservation":v["unit_conservation"],"obj_decision_s":v["obj_decision_s"],"r0_assembly_s":v["r0_assembly_s"],"c1_decision_s":v["c1_decision_s"],"payload":v["payload"],"terminal":v["terminal"],"asr_interval":v["asr_interval"],"n_groups":v["n_groups"],"final_text":v["final_text"]} for k,v in parts.items()},"usage_summary":{"original_paid":{"prompt":old_pt,"completion":old_ct,"cost":old_cost,"n":49},"new_paid":{"prompt":new_pt,"completion":new_ct,"cost":new_cost,"n":len(need_sorted)},"reused_n":len(reuse)},"auth":{"reused_original":True,"new_auth":0},"translations":results,"request_log_sanitized":req_log,"latency_replay_composed":lat,"paid":paid,"translation_config":{"model":MODEL,"routing":"gemma4_26b_31b_latency","source":"en","target":"ko","context":"","max_tokens":MAX_OUT,"timeout_s":TIMEOUT_S,"attempts":1,"order_new":"C1-delta case order only","order_correction":"original R0-first per-case sequential single trial cache/order biased not counterbalanced; preserved contradiction"},"note":"provider actual outputs; semantic benefit awaits Director review; no auto judging"}
    LEDGER_V2.write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
    print("done reuse="+str(len(reuse))+" new="+str(len(need_sorted))+" ok="+str(sum(1 for r in results if r.get("ok")))+" wall="+str(round(time.perf_counter()-t_start,1)))
    return 0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--attempt-id", default="")
    a = ap.parse_args()
    if a.live:
        return asyncio.run(run_live(a.attempt_id))
    ap.print_help()
    return 2
if __name__ == "__main__":
    raise SystemExit(main())
