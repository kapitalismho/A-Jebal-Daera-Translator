from __future__ import annotations
import asyncio
import hashlib
import json
import re
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

EXP = ROOT / "experiments" / "psem_evidence_delivery_gap" / "text_partition_probe"
CHUNK = 512
DRAIN_TIMEOUT_S = 60.0
QUIESCE_S = 3.0
ARM_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0
KEEP_FIELDS = ("text", "start_ms", "end_ms", "is_final", "language", "confidence")
WORD_RE = re.compile(r"\S+")


def load_soniox_key() -> str:
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("SONIOX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SONIOX_API_KEY missing")


def load_payload(freeze: dict) -> tuple[np.ndarray, bytes]:
    audio = freeze["audio"]
    p0, p1 = audio["payload_samples"]
    with wave.open(audio["path"], "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1
        r.setpos(p0)
        raw = r.readframes(p1 - p0)
    pcm = np.frombuffer(raw, dtype=np.int16).copy()
    assert len(pcm) == p1 - p0
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == audio["payload_sha256"]
    return pcm, raw


def sanitize_token(token: dict, dropped: set) -> dict:
    rec: dict = {}
    for key in KEEP_FIELDS:
        if key in token:
            value = token[key]
            if key == "text":
                rec[key] = str(value or "")
            elif key in ("start_ms", "end_ms"):
                rec[key] = value if isinstance(value, (int, float)) else None
            elif key == "is_final":
                rec[key] = bool(value)
            else:
                rec[key] = value if isinstance(value, (str, int, float)) else None
    for key in token:
        if key not in KEEP_FIELDS:
            dropped.add(key)
    return rec


async def run_session(backend: SonioxRealtimeSTTBackend, payload: np.ndarray, p0: int) -> dict:
    arm_t0 = time.monotonic()
    arm_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t = time.monotonic()
    session = await backend.open_session()
    session_open_s = round(time.monotonic() - t, 3)
    pad_ms = int(session.trailing_silence_ms)
    pad_samples = int(session.sample_rate_hz * (pad_ms / 1000.0)) if pad_ms > 0 else 0
    cls = type(session)
    orig_handle = cls._handle_message
    orig_emit = cls._emit_final_text
    raw_tokens: list = []
    raw_msg_with_tokens = 0
    raw_msg_without_tokens = 0
    control_fin_end = 0
    dropped_fields: set = set()
    emit_snaps: list = []
    empty_acks = 0
    orig_put = cls._put_event

    def tee_handle(self, message) -> None:
        nonlocal raw_msg_with_tokens, raw_msg_without_tokens, control_fin_end
        try:
            text = message.decode("utf-8", errors="ignore") if isinstance(message, bytes) else message
            data = json.loads(text)
        except ValueError:
            data = {}
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, list) and tokens:
            raw_msg_with_tokens += 1
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                if str(token.get("text", "") or "") in ("<fin>", "<end>"):
                    control_fin_end += 1
                rec = sanitize_token(token, dropped_fields)
                rec["arrival_wall"] = round(time.monotonic() - arm_t0, 3)
                raw_tokens.append(rec)
        else:
            raw_msg_without_tokens += 1
        return orig_handle(self, message)

    def tee_emit(self) -> bool:
        pre = [{"text": tok.text, "end_ms": tok.end_ms,
                "start_ms": getattr(tok, "start_ms", None) if hasattr(tok, "start_ms") else None}
               for tok in self._final_tokens]
        out = orig_emit(self)
        post = [{"text": tok.text, "end_ms": tok.end_ms} for tok in self._final_tokens]
        emit_snaps.append({"pre": pre, "post": post, "emitted": bool(out),
                           "wall": round(time.monotonic() - arm_t0, 3)})
        return out

    def tee_put(self, event) -> None:
        nonlocal empty_acks
        try:
            is_final = bool(getattr(event, "is_final", False))
            text = str(getattr(event, "text", "") or "")
        except Exception:
            return orig_put(self, event)
        if is_final and text == "":
            empty_acks += 1
        return orig_put(self, event)

    cls._handle_message = tee_handle
    cls._emit_final_text = tee_emit
    cls._put_event = tee_put
    log: list = []

    async def pump() -> None:
        try:
            async for ev in session.events():
                log.append({"wall": time.monotonic(), "text": ev.text,
                            "is_final": bool(ev.is_final)})
        except Exception as exc:
            log.append({"wall": time.monotonic(), "error": type(exc).__name__,
                        "detail": str(exc)[:200]})

    pump_task = asyncio.create_task(pump())
    try:
        async with asyncio.timeout(ARM_TIMEOUT_S):
            t0 = time.monotonic()
            total = len(payload)
            for off in range(0, total, CHUNK):
                target = t0 + (off / CHUNK) * (CHUNK / 16000.0)
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await session.send_audio(payload[off:off + CHUNK].tobytes())
            finalize_wall = round(time.monotonic() - arm_t0, 3)
            await session.on_speech_end()
            fw = arm_t0 + finalize_wall
            deadline = time.monotonic() + DRAIN_TIMEOUT_S
            while time.monotonic() < deadline:
                posts = [e for e in log if e.get("is_final") and "error" not in e and e["wall"] >= fw]
                if posts and time.monotonic() - posts[-1]["wall"] >= QUIESCE_S:
                    break
                await asyncio.sleep(0.05)
            post_finals = [e for e in log if e.get("is_final") and "error" not in e and e["wall"] >= fw]
            pre_finals = [e["text"] for e in log if e.get("is_final") and "error" not in e and e["wall"] < fw]
            nonempty = [e for e in post_finals if e["text"]]
            sel = nonempty[-1] if nonempty else (post_finals[-1] if post_finals else None)
            collection_elapsed_s = round(time.monotonic() - (arm_t0 + finalize_wall), 3)
    finally:
        cls._handle_message = orig_handle
        cls._emit_final_text = orig_emit
        cls._put_event = orig_put
        try:
            await asyncio.wait_for(session.close(), timeout=CLOSE_TIMEOUT_S)
        except Exception:
            pass
        pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
    received = [{"text": e["text"], "rel_wall": round(e["wall"] - arm_t0, 3),
                 "after_finalize": e["wall"] >= fw} for e in log if e.get("is_final") and "error" not in e]
    return {"arm_utc": arm_utc, "session_open_s": session_open_s,
            "trailing_silence_ms": pad_ms, "trailing_silence_samples": pad_samples,
            "finalize_wall": finalize_wall,
            "final_text": sel["text"] if sel else "",
            "final_wall": round(sel["wall"] - arm_t0, 3) if sel else None,
            "final_latency_s": round(sel["wall"] - (arm_t0 + finalize_wall), 3) if sel else None,
            "final_status": ("ok" if sel and sel["text"]
                             else ("empty-ack" if sel else "timeout")),
            "nfinal_post": len(post_finals), "received_finals": received,
            "pre_finalize_finals": pre_finals,
            "partials_tail": [e["text"] for e in log if not e.get("is_final")][-3:],
            "collection_elapsed_s": collection_elapsed_s,
            "wall_s": round(time.monotonic() - arm_t0, 3),
            "raw_tokens": raw_tokens, "raw_msg_with_tokens": raw_msg_with_tokens,
            "raw_msg_without_tokens": raw_msg_without_tokens,
            "control_fin_end": control_fin_end,
            "dropped_field_names": sorted(dropped_fields),
            "emit_snapshots": emit_snaps, "empty_ack_count": empty_acks}


def accepted_stream(snaps: list) -> tuple[list, dict]:
    nonempty = [s for s in snaps if s["emitted"] and s["post"]]
    assert nonempty
    last = nonempty[-1]
    tokens = [{"o": i, "text": t["text"], "end_ms": t["end_ms"]} for i, t in enumerate(last["post"])]
    trace = {"n_emit_snapshots": len(snaps), "n_nonempty_emits": len(nonempty),
             "used_emit_index": len(snaps) - 1 - snaps[::-1].index(last)}
    pre_text = "".join(t["text"] for t in last["pre"])
    post_text = "".join(t["text"] for t in last["post"])
    trace["pre_len"] = len(pre_text)
    if pre_text != post_text:
        stripped = pre_text[:len(pre_text) - len(pre_text.lstrip())]
        tail = pre_text[len(pre_text.rstrip()):]
        trace["leading_stripped"] = stripped
        trace["trailing_stripped"] = tail
        assert stripped + post_text + tail == pre_text
    else:
        trace["leading_stripped"] = ""
        trace["trailing_stripped"] = ""
    return tokens, trace


def build_groups(final_text: str, tokens: list, p0: int) -> list:
    tok_of_char: list = []
    for tok in tokens:
        tok_of_char.extend([tok["o"]] * len(tok["text"]))
    assert len(tok_of_char) == len(final_text)
    ends = {tok["o"]: tok["end_ms"] for tok in tokens}
    texts = {tok["o"]: tok["text"] for tok in tokens}
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(final_text)]
    assert spans
    bounds: list = []
    for idx, (ws, we) in enumerate(spans):
        nxt = spans[idx + 1][0] if idx + 1 < len(spans) else len(final_text)
        gstart = 0 if idx == 0 else ws
        bounds.append((gstart, we, nxt))
    groups = []
    prev_end_src = None
    for idx, (gstart, wend, gend) in enumerate(bounds):
        text = final_text[gstart:gend]
        core = final_text[gstart:wend]
        core_owners = [tok_of_char[gstart + k] for k in range(len(core))]
        end_ms = ends.get(core_owners[-1])
        refs: list = []
        k = 0
        while k < len(text):
            o = tok_of_char[gstart + k]
            j = k
            while j < len(text) and tok_of_char[gstart + j] == o:
                j += 1
            refs.append({"o": o, "end_ms": ends.get(o), "slice": text[k:j]})
            k = j
        end_src = p0 + round(end_ms * 16) if isinstance(end_ms, (int, float)) else None
        if idx == 0:
            start_src: int | None = p0
            start_note = "session-start-lower-bound"
        else:
            start_src = prev_end_src
            start_note = "chained-prev-end"
        groups.append({"idx": idx, "text": text, "char_range": [gstart, gend],
                       "word": core, "start_src": start_src, "end_prov_ms": end_ms,
                       "end_src": end_src, "token_refs": refs,
                       "unresolved": end_src is None, "start_note": start_note})
        prev_end_src = end_src if end_src is not None else prev_end_src
    return groups


def partition(groups: list, bounds: list) -> tuple[list, bool]:
    ordered = sorted(bounds)
    frags: dict = {}
    for grp in groups:
        end = grp["end_src"]
        if end is None:
            fi = len(ordered)
            supported = False
        else:
            fi = sum(1 for b in ordered if b < end)
            supported = True
        frags.setdefault(fi, []).append(grp["idx"])
    out = []
    for fi in sorted(frags):
        idxs = frags[fi]
        text = "".join(groups[i]["text"] for i in idxs)
        amb = [i for i in idxs if any(groups[i]["start_src"] is not None
                                      and groups[i]["end_src"] is not None
                                      and groups[i]["start_src"] < b < groups[i]["end_src"]
                                      for b in ordered)]
        unres = [i for i in idxs if groups[i]["unresolved"]]
        srcs = [groups[i]["end_src"] for i in idxs if groups[i]["end_src"] is not None]
        starts = [groups[i]["start_src"] for i in idxs if groups[i]["start_src"] is not None]
        out.append({"fi": fi, "group_range": [idxs[0], idxs[-1]], "n_groups": len(idxs),
                    "text": text, "start_src": min(starts) if starts else None,
                    "end_src": max(srcs) if srcs else None,
                    "ambiguous_groups": amb, "unresolved_groups": unres,
                    "ownership_supported": supported and not unres})
    return out, True


def norm_word(word: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", word.lower())


def gt_compare(groups: list, variants: dict, freeze: dict) -> dict:
    gt = freeze["gt_posthoc"]
    vocab: dict = {}
    for grp in groups:
        key = norm_word(grp["word"])
        vocab.setdefault(key, []).append(grp["idx"])
    retention = {}
    for side, words in (("A", gt["a_in_span_words"]), ("C", gt["c_in_span_head"])):
        for entry in words:
            key = norm_word(entry["w"])
            hits = vocab.get(key, [])
            retention[entry["w"]] = {"side": side, "present": bool(hits), "groups": hits}
    ownership = {}
    scored_a_that = (vocab.get("that") or [None])[0]
    probes = [("do", "A", "left", None), ("that", "A", "left", None),
              ("You", "C", "right", scored_a_that), ("have", "C", "right", scored_a_that),
              ("to", "C", "right", scored_a_that), ("hope", "C", "right", scored_a_that)]
    for word, side, gt_side, after in probes:
        hits = vocab.get(norm_word(word), [])
        if after is None:
            scored = hits[0] if hits else None
            sel_note = "first-transcript-occurrence"
        else:
            cands = [h for h in hits if h > after]
            scored = cands[0] if cands else None
            sel_note = "first-occurrence-after-scored-A-that"
        per_variant = {}
        for vname, vres in variants.items():
            if scored is None:
                per_variant[vname] = {"scored_group": None, "verdict": "unscoreable-missing",
                                      "all_occurrences": hits, "selection": sel_note}
                continue
            owner = None
            for frag in vres["fragments"]:
                if frag["group_range"][0] <= scored <= frag["group_range"][1]:
                    owner = frag["fi"]
                    break
            nb = len(vres["bounds"])
            if nb == 0:
                label, verdict = "single", "merged-single"
            else:
                label = "left" if owner == 0 else ("right" if nb == 1 or owner == 2 else "mid")
                if gt_side == "left":
                    verdict = "agree" if owner == 0 else "disagree"
                else:
                    verdict = "agree" if owner is not None and owner >= 1 else "disagree"
            grp = groups[scored]
            misaligned = bool(gt_side == "left" and grp["end_src"] is not None
                              and grp["end_src"] > 52156984)
            per_variant[vname] = {"scored_group": scored, "fragment": owner,
                                  "label": label,
                                  "expected": "left" if gt_side == "left" else "right-side",
                                  "verdict": verdict, "end_src": grp["end_src"],
                                  "timing_note": "word-timing-misalignment" if misaligned else "timing-consistent",
                                  "all_occurrences": hits, "selection": sel_note}
        ownership[word] = {"gt_side": gt_side, "region": side, "per_variant": per_variant}
    final_len = sum(len(g["text"]) for g in groups)
    return {"retention": retention, "ownership": ownership, "final_len": final_len,
            "selection_rule": "A-probes score the first transcript occurrence (GT A-question opens the utterance); C-probes score the first occurrence after the scored A-that group (GT C-head follows the A-question). All occurrences listed regardless."}


async def main() -> None:
    freeze = json.loads((EXP / "freeze.json").read_text(encoding="utf-8"))
    p0, p1 = freeze["audio"]["payload_samples"]
    payload, _ = load_payload(freeze)
    backend = SonioxRealtimeSTTBackend(api_key=load_soniox_key(), language_hints=["en"])
    sess = await run_session(backend, payload, p0)
    usable = bool(sess["final_text"]) and any(s["emitted"] and s["post"] for s in sess["emit_snapshots"])
    (EXP / "raw_session.tmp.json").write_text(
        json.dumps(sess, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if usable:
        tokens, trace = accepted_stream(sess["emit_snapshots"])
        assert "".join(t["text"] for t in tokens) == sess["final_text"]
        groups = build_groups(sess["final_text"], tokens, p0)
        assert "".join(g["text"] for g in groups) == sess["final_text"]
    else:
        tokens, trace, groups = [], {"note": "no-usable-final"}, []
    bounds = {"baseline": [], "ideal": [52156984],
              "f0": [52139200, 52203200], "h": [52158400, 52203200]}
    variants: dict = {}
    if usable:
        for vname in ("baseline", "ideal", "f0", "h"):
            frags, _ = partition(groups, bounds[vname])
            conservation = "".join(f["text"] for f in frags) == sess["final_text"]
            assert conservation
            variants[vname] = {"bounds": bounds[vname], "fragments": frags,
                               "conservation": conservation}
        comparison: dict = gt_compare(groups, variants, freeze)
    else:
        comparison = {"prerequisite": "no completed final with accepted token timestamps; partition unscored, no rerun under this probe"}
    out = {"run_id": str(uuid.uuid4()), "run_utc": sess["arm_utc"],
           "freeze_id": freeze["authority"]["freeze_id"],
           "authority": {"baseline_commit": freeze["authority"]["baseline_commit"],
                         "branch": freeze["authority"]["branch"]},
           "audio": {"path": freeze["audio"]["path"],
                     "payload_samples": [p0, p1],
                     "payload_sha256": freeze["audio"]["payload_sha256"]},
           "session": {"model": "stt-rt-v5", "language_hints": ["en"],
                       "chunk_samples": CHUNK, "paced": "realtime-32ms",
                       "hold_s": 0.0, "provider_sessions": 1,
                       "session_open_s": sess["session_open_s"],
                       "trailing_silence_ms": sess["trailing_silence_ms"],
                       "trailing_silence_samples": sess["trailing_silence_samples"],
                       "finalize_wall": sess["finalize_wall"],
                       "final_wall": sess["final_wall"],
                       "final_latency_s": sess["final_latency_s"],
                       "final_status": sess["final_status"],
                       "nfinal_post": sess["nfinal_post"],
                       "collection_elapsed_s": sess["collection_elapsed_s"],
                       "wall_s": sess["wall_s"], "translations": 0,
                       "openrouter_calls": 0},
           "raw_evidence": {"n_raw_tokens": len(sess["raw_tokens"]),
                            "raw_msg_with_tokens": sess["raw_msg_with_tokens"],
                            "raw_msg_without_tokens": sess["raw_msg_without_tokens"],
                            "control_fin_end": sess["control_fin_end"],
                            "dropped_field_names": sess["dropped_field_names"],
                            "empty_ack_count": sess["empty_ack_count"],
                            "raw_tokens": sess["raw_tokens"],
                            "emit_snapshots": sess["emit_snapshots"]},
           "accepted": {"final_text": sess["final_text"],
                        "n_tokens": len(tokens), "tokens": tokens,
                        "normalization_trace": trace},
           "groups": groups, "variants": variants,
           "gt_posthoc": comparison,
           "grouping_rule_reading": "Exact-concat reading of the frozen rule: group0=[0, w1.start); group_i=[w_i.start, w_{i+1}.start or len), i.e. trailing whitespace attaches to the preceding word and leading whitespace to the first word. The freeze illustrative range prose (from the previous word end) would double-count inter-word whitespace; this reading satisfies the freeze own exact-concat requirement. End estimate from the last non-whitespace char token end_ms; starts chained (group0 lower-bounded at session start).",
           "paid_sessions_total": 2,
           "paid_note": "First paid session streamed fully but its raw evidence was lost to an implementation crash in the offline grouper (overlapping ranges) before any result was written or observed. Grouper fixed and verified offline on synthetic text; this completing run is the sole scored session. No result-driven rerun.",
           "received_finals": sess["received_finals"],
           "pre_finalize_finals": sess["pre_finalize_finals"],
           "partials_tail": sess["partials_tail"],
           "limits": ["Provider token ms are approximate; projected ownership is agreement with GT side at known S, never an exact speaker-purity claim.",
                      "No added PCM hold here: the entire transcript was finalized first, then partitioned offline; no cross-probe controlled latency claim against the audio-split probe.",
                      "Cached H/F0 thresholds and Te are fixed references from revised Phase D, not reinferred; product deadlines, continuous provisional/commit behavior untested.",
                      "2 paid sessions total (1 crashed pre-write on an implementation bug, 1 scored); result stands positive or negative with no result-driven rerun."]}
    (EXP / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (EXP / "raw_session.tmp.json").unlink(missing_ok=True)
    print(f"sessions=1 nfinal={sess['nfinal_post']} groups={len(groups)} "
          f"latency={sess['final_latency_s']}s tokens={len(tokens)} "
          f"rawtok={len(sess['raw_tokens'])}")


if __name__ == "__main__":
    asyncio.run(main())
