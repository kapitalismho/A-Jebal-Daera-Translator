from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "psem_product_translation"
FREEZE_PATH = EXP / "CAPTURE_FREEZE.json"

CHUNK = 512
CONNECT_TIMEOUT_S = 10.0
CLOSE_TIMEOUT_S = 10.0
POST_EOS_BOUND_S = 15.0
ARM_TIMEOUT_S = 120.0
TRAIL_SAMPLES = 1600
ZERO_CAP_SAMPLES = 30720
KEEP_FIELDS = ("text", "start_ms", "end_ms", "is_final", "language", "confidence")
BANNED_CONFIG_KEYS = ("speaker", "diarization", "translation", "context", "translate")
CASE_ORDER = ("R1", "R2", "T1", "COMBINED", "BC1", "SINGLE_ES2009c", "SINGLE_ES2009d")
WORD_RE = re.compile(r"\S+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_freeze() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def load_soniox_key() -> str:
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("SONIOX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SONIOX_API_KEY missing")


def scrub(text: str) -> str:
    return re.sub(r"[A-Za-z0-9_\-]{64,}", "<redacted-token>", text)


def journal_path(capdir: Path) -> Path:
    return capdir / "attempt_journal.jsonl"


def journal(capdir: Path, event: dict) -> None:
    capdir.mkdir(parents=True, exist_ok=True)
    with journal_path(capdir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        fh.flush()


def journal_entries(capdir: Path) -> list:
    p = journal_path(capdir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def budget_allows(capdir: Path, freeze: dict, session_id: str) -> tuple[bool, str]:
    es = [e for e in journal_entries(capdir)
          if e.get("freeze_id") == freeze["freeze_id"] and e.get("event") == "attempt-journaled"]
    if len(es) >= 7:
        return False, "budget spent %d/7" % len(es)
    if any(e.get("session_id") == session_id for e in es):
        return False, "%s already attempted no retries" % session_id
    return True, "ok"


def source_identity(audio_path: str) -> tuple[Path, int, str, int]:
    path = Path(audio_path)
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2
        frames = r.getnframes()
    raw_all = None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(1 << 20)
            if not blk:
                break
            digest.update(blk)
    _ = raw_all
    return path, frames, digest.hexdigest(), path.stat().st_size


def load_payload(freeze: dict, case: str) -> tuple[np.ndarray, int, int, str, Path]:
    spec = freeze["cases"][case]
    p0, p1 = int(spec["payload_samples"][0]), int(spec["payload_samples"][1])
    path = Path(spec["audio_path"])
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2
        total = r.getnframes()
        assert 0 <= p0 < p1 <= total, "payload out of range (%d,%d vs %d)" % (p0, p1, total)
        r.setpos(p0)
        raw = r.readframes(p1 - p0)
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == spec["payload_sha256"], "source bytes differ from freeze"
    pcm = np.frombuffer(raw, dtype=np.int16).copy()
    assert len(pcm) == p1 - p0
    return pcm, p0, p1, digest, path


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
            dropped.add(str(key))
    return rec


def valid_end_ms(e) -> bool:
    return (not isinstance(e, bool) and isinstance(e, (int, float))
            and math.isfinite(e) and e > 0)


def provider_token_end(end_ms) -> tuple:
    if isinstance(end_ms, bool) or not isinstance(end_ms, (int, float)):
        return None, "missing-end"
    if not math.isfinite(end_ms):
        return None, "nonfinite-end"
    if end_ms <= 0:
        return None, "degenerate-end"
    return end_ms, None


def provider_token_start(token: dict) -> tuple:
    s = token.get("start_ms")
    e = token.get("end_ms")
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        return None, ("ambiguous-start" if token.get("ambiguous") else "missing-start")
    if not math.isfinite(s):
        return None, "nonfinite-start"
    if s < 0:
        return None, "negative-start"
    if not valid_end_ms(e):
        return None, "degenerate-start"
    if s == 0:
        return 0, None
    if s >= e:
        return None, "degenerate-start"
    return s, None


def _start_eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if a is None or b is None:
        return a is None and b is None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def align_accepted_to_raw_finals(accepted_tokens: list, raw_tokens: list) -> tuple:
    pool = [(i, t) for i, t in enumerate(raw_tokens) if t.get("is_final")]
    n = len(accepted_tokens)
    hits: list = []
    for off in range(len(pool) - n + 1):
        match = True
        for k in range(n):
            a = accepted_tokens[k]
            r = pool[off + k][1]
            if a.get("text") != r.get("text") or a.get("end_ms") != r.get("end_ms"):
                match = False
                break
        if match:
            hits.append(off)
    return pool, hits


def hydrate_accepted_starts(accepted_tokens: list, raw_tokens: list) -> tuple:
    pool, hits = align_accepted_to_raw_finals(accepted_tokens, raw_tokens)
    hydrated: list = []
    provenance_by_ordinal: dict = {}
    for i, a in enumerate(accepted_tokens):
        rec = {"o": a.get("o"), "text": a.get("text"), "end_ms": a.get("end_ms")}
        known = a.get("start_ms")
        missing = (known is None or isinstance(known, bool)
                   or not isinstance(known, (int, float)))
        cands = ([(pool[off + i][0], pool[off + i][1].get("start_ms"))
                  for off in hits] if hits else [])
        if not missing:
            rec["start_ms"] = known
            rec["ambiguous"] = False
            provenance_by_ordinal[rec["o"]] = {"raw_idxs": [r for r, _ in cands],
                                              "runs": list(hits),
                                              "reason": "accepted-kept"}
        elif not hits:
            rec["start_ms"] = None
            rec["ambiguous"] = False
            provenance_by_ordinal[rec["o"]] = {"raw_idxs": [], "runs": [],
                                              "reason": "no-raw-alignment"}
        else:
            first = cands[0][1]
            if all(_start_eq(s, first) for _, s in cands):
                rec["start_ms"] = first
                rec["ambiguous"] = False
            else:
                rec["start_ms"] = None
                rec["ambiguous"] = True
            provenance_by_ordinal[rec["o"]] = {"raw_idxs": [r for r, _ in cands],
                                              "runs": list(hits),
                                              "reason": (None if rec["start_ms"] is not None
                                                         or not rec["ambiguous"]
                                                         else "ambiguous-start")}
        hydrated.append(rec)
    return hydrated, provenance_by_ordinal
def build_groups(final_text: str, tokens: list, p0: int, provenance_by_ordinal: dict | None = None) -> list:
    tok_of_char: list = []
    starts: dict = {}
    reasons: dict = {}
    for tok in tokens:
        tok_of_char.extend([tok["o"]] * len(tok["text"]))
        s, reason = provider_token_start(tok)
        starts[tok["o"]] = s
        reasons[tok["o"]] = reason
    assert len(tok_of_char) == len(final_text)
    ends = {tok["o"]: tok["end_ms"] for tok in tokens}
    tok_spans: dict = {}
    off = 0
    for tok in tokens:
        tok_spans[tok["o"]] = (off, off + len(tok["text"]))
        off += len(tok["text"])
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(final_text)]
    assert spans
    bounds: list = []
    for idx, (ws, we) in enumerate(spans):
        nxt = spans[idx + 1][0] if idx + 1 < len(spans) else len(final_text)
        gstart = 0 if idx == 0 else ws
        bounds.append((gstart, we, nxt))
    prov = provenance_by_ordinal or {}
    groups = []
    for idx, (gstart, wend, gend) in enumerate(bounds):
        text = final_text[gstart:gend]
        core = final_text[gstart:wend]
        core_owners = [tok_of_char[gstart + k] for k in range(len(core))]
        end_ms, end_reason = provider_token_end(ends.get(core_owners[-1]))
        refs: list = []
        k = 0
        while k < len(text):
            o = tok_of_char[gstart + k]
            j = k
            while j < len(text) and tok_of_char[gstart + j] == o:
                j += 1
            refs.append({"o": o, "end_ms": ends.get(o),
                         "start_ms": starts.get(o), "slice": text[k:j]})
            k = j
        if end_ms is None:
            end_src = None
        else:
            end_src = p0 + round(end_ms * 16)
        own_chars: dict = {}
        for k, ch in enumerate(core):
            if ch.strip():
                own_chars.setdefault(tok_of_char[gstart + k], []).append(ch)
        lexical = {o for o, chs in own_chars.items()
                   if any(c.isalnum() for c in chs)}
        bad = sorted(o for o in lexical if starts.get(o) is None)
        if not lexical:
            start_src = None
            start_reason = "punct-only-group"
            start_note = "unknown"
            start_ms = None
            start_owner = None
        elif bad:
            start_src = None
            start_reason = reasons.get(bad[0]) or "missing-start"
            start_note = "unknown"
            start_ms = None
            start_owner = None
        else:
            best_o = min(lexical, key=lambda o: (starts[o], o))
            best = starts[best_o]
            start_ms = best
            start_owner = best_o
            start_src = p0 + round(best * 16)
            t0, t1 = tok_spans.get(best_o, (gstart, gend))
            shared = t0 < gstart or t1 > gend
            start_note = "provider-start-shared-token" if shared else "provider-start"
            if end_src is not None and start_src >= end_src:
                start_src = None
                start_reason = "sample-degenerate"
                start_note = "unknown"
            else:
                start_reason = None
        raw_entry = prov.get(start_owner) if start_owner is not None else None
        groups.append({"idx": idx, "text": text, "char_range": [gstart, gend],
                       "word": core, "start_src": start_src, "start_ms": start_ms,
                       "end_prov_ms": ends.get(core_owners[-1]),
                       "end_src": end_src, "end_reason": end_reason,
                       "token_refs": refs,
                       "unresolved": (start_src is None or end_src is None),
                       "start_note": start_note,
                       "start_reason": start_reason, "start_owner": start_owner,
                       "start_raw": (raw_entry.get("raw_idxs", [None])[0]
                                     if raw_entry else None)})
    return groups


def corrected_groups_for_capture(cap: dict) -> list:
    acc = cap.get("accepted", {})
    tokens = acc.get("tokens", [])
    final_text = acc.get("final_text", "")
    if not final_text:
        return []
    pay = cap.get("audio", {}).get("payload_samples") or cap.get("payload_samples") or [0, 0]
    p0 = int(pay[0]) if isinstance(pay, list) and len(pay) == 2 else 0
    needs_hydration = any(isinstance(t.get("start_ms"), bool)
                           or not isinstance(t.get("start_ms"), (int, float))
                           for t in tokens)
    if needs_hydration:
        raw = cap.get("raw_evidence", {}).get("raw_tokens", [])
        hydrated, prov = hydrate_accepted_starts(tokens, raw)
        return build_groups(final_text, hydrated, p0, prov)
    prov = {t.get("o"): None for t in tokens}
    return build_groups(final_text, tokens, p0, prov)


async def capture_session(case: str, freeze: dict, endpoint: str, capdir: Path) -> Path:
    import websockets

    session_id = case
    ok, reason = budget_allows(capdir, freeze, session_id)
    if not ok:
        raise SystemExit("budget refused for %s: %s" % (session_id, reason))
    payload, p0, p1, digest, path = load_payload(freeze, case)

    journal(capdir, {"freeze_id": freeze["freeze_id"], "session_id": session_id,
                     "event": "attempt-journaled", "utc": utc_now()})
    key = load_soniox_key()
    assert len(key) == freeze["secrets"]["key_length"], "key length mismatch"
    arm_start = time.monotonic()
    arm_utc = utc_now()

    config = {"api_key": key,
              "model": freeze["config"]["model"],
              "audio_format": "pcm_s16le",
              "sample_rate": 16000,
              "num_channels": 1,
              "enable_endpoint_detection": False,
              "language_hints": list(freeze["config"]["language_hints"])}
    del key
    blob = json.dumps(config)
    assert not any(b in blob for b in BANNED_CONFIG_KEYS), "banned config key present"
    redacted = {k: (v if k != "api_key" else "<redacted>") for k, v in config.items()}

    chunks: list = []
    sends: list = []
    msg_log: list = []
    raw_tokens: list = []
    dropped: set = set()
    n_fin_end = 0
    fin_wall: float | None = None
    speaker_seen = False
    server_revision = None
    failure: dict | None = None
    first_send_wall: float | None = None
    finalize_wall: float | None = None
    finalize_delivered_wait_s = 0.0
    eos_send_start = None
    eos_send_end = None
    finished_wall = None
    finished_value = None
    finished_wait_s = None
    post_eos_sends = 0
    zeros_trailing = 0
    zeros_bounded = 0
    status = "unknown"

    try:
        ws = await asyncio.wait_for(
            websockets.connect(endpoint, ping_interval=None,
                               open_timeout=CONNECT_TIMEOUT_S),
            timeout=CONNECT_TIMEOUT_S + 5)
    except Exception as exc:
        journal(capdir, {"freeze_id": freeze["freeze_id"], "session_id": session_id,
                         "event": "pre-connect-failure", "utc": utc_now(),
                         "error": type(exc).__name__, "detail": scrub(str(exc))[:300]})
        dest = capdir / ("%s.failure.json" % session_id)
        dest.write_text(json.dumps({"session_id": session_id, "case": case,
                                    "freeze_id": freeze["freeze_id"],
                                    "captured_at_utc": arm_utc, "status": "pre-connect-failure",
                                    "error": type(exc).__name__,
                                    "detail": scrub(str(exc))[:500]}, indent=1) + "\n",
                        encoding="utf-8")
        raise SystemExit("%s pre-connect FAILED %s -> %s" % (session_id, type(exc).__name__, dest))

    open_wall = round(time.monotonic() - arm_start, 3)
    journal(capdir, {"freeze_id": freeze["freeze_id"], "session_id": session_id,
                     "event": "session-opened", "utc": utc_now(), "open_offset_s": open_wall})
    recv_q: asyncio.Queue = asyncio.Queue()

    async def recv_loop() -> None:
        try:
            async for msg in ws:
                recv_q.put_nowait((time.monotonic(), msg))
        except Exception as exc:
            recv_q.put_nowait((time.monotonic(), exc))
        finally:
            recv_q.put_nowait((time.monotonic(), None))

    def note_message(wall: float, msg) -> None:
        nonlocal n_fin_end, fin_wall, server_revision, speaker_seen, finished_wall, finished_value
        if msg is None or isinstance(msg, Exception):
            msg_log.append({"arrival_from_open": round(wall - arm_start, 3),
                            "arrival_wall": None, "proc_ms": 0.0,
                            "error": type(msg).__name__ if msg else "closed"})
            return
        t0 = time.monotonic()
        try:
            text = msg.decode("utf-8", errors="ignore") if isinstance(msg, bytes) else msg
            data = json.loads(text)
        except ValueError:
            data = {}
        proc_ms = round((time.monotonic() - t0) * 1000, 3)
        if not isinstance(data, dict):
            data = {}
        if server_revision is None:
            for k in ("server_revision", "model_revision", "revision"):
                if k in data:
                    server_revision = str(data[k])
        toks = data.get("tokens")
        rec: dict = {"arrival_from_open": round(wall - arm_start, 3),
                     "arrival_wall": None, "proc_ms": proc_ms,
                     "has_tokens": bool(isinstance(toks, list) and toks),
                     "n_tokens": len(toks) if isinstance(toks, list) else 0,
                     "other_keys": sorted(k for k in data if k != "tokens")}
        if data.get("finished") is True and finished_wall is None:
            finished_wall = round(wall - first_send_wall, 3)
            finished_value = True
            rec["finished"] = True
        elif "finished" in data:
            rec["finished"] = data["finished"]
        if isinstance(toks, list) and toks:
            sanitized = []
            for token in toks:
                if not isinstance(token, dict):
                    continue
                if "speaker" in token:
                    speaker_seen = True
                if str(token.get("text", "") or "") in ("<fin>", "<end>"):
                    n_fin_end += 1
                    if fin_wall is None and finalize_wall is not None:
                        fin_wall = round(wall - first_send_wall, 3)
                srec = sanitize_token(token, dropped)
                srec["arrival_wall"] = (round(wall - first_send_wall, 3)
                                        if first_send_wall is not None else None)
                raw_tokens.append(srec)
                sanitized.append(srec)
            rec["tokens"] = sanitized
        msg_log.append(rec)

    async def drain() -> None:
        while not recv_q.empty():
            wall, msg = recv_q.get_nowait()
            note_message(wall, msg)

    async def send_chunk(frag: bytes, idx: int, src0: int | None, sched: float | None) -> None:
        t0 = time.monotonic()
        await ws.send(frag)
        t1 = time.monotonic()
        if src0 is not None and sched is not None:
            chunks.append({"idx": idx, "src_range": [src0, src0 + len(frag) // 2],
                           "sched_wall": round(sched - t0 + (t0 - first_send_wall), 3) if first_send_wall else round(sched, 3),
                           "send_start": round(t0 - arm_start, 3),
                           "send_end": round(t1 - arm_start, 3)})

    try:
        async with asyncio.timeout(ARM_TIMEOUT_S):
            await ws.send(json.dumps(config))
            recv_task = asyncio.create_task(recv_loop())
            t0 = time.monotonic()
            first_send_wall = t0
            idx = 0

            async def paced(frag: bytes, off: int, src0: int | None) -> None:
                target = t0 + off / 16000.0
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await send_chunk(frag, idx, src0, target)
                if src0 is not None:
                    chunks[-1]["sched_wall"] = round(target - t0, 3)
                await drain()

            n_src = (len(payload) + CHUNK - 1) // CHUNK
            for i in range(n_src):
                off = i * CHUNK
                frag = payload[off:off + CHUNK]
                idx = i
                await paced(frag.tobytes(), off, p0 + off)
            delivered = len(payload)

            trail = np.zeros(TRAIL_SAMPLES, dtype=np.int16)
            base = delivered
            for i in range(0, len(trail), CHUNK):
                frag = trail[i:i + CHUNK]
                idx += 1
                await paced(frag.tobytes(), base + i, None)
            delivered += len(trail)
            zeros_trailing = len(trail)
            sends.append({"kind": "trailing-zeros", "nsamples": len(trail),
                          "wall": round(time.monotonic() - t0, 3),
                          "delivered_off": delivered})

            due = t0 + delivered / 16000.0
            now = time.monotonic()
            if due > now:
                await asyncio.sleep(due - now)
                finalize_delivered_wait_s = round(due - now, 3)
            await drain()
            await ws.send(json.dumps({"type": "finalize"}))
            finalize_wall = round(time.monotonic() - t0, 3)
            sends.append({"kind": "finalize", "wall": finalize_wall,
                          "delivered_off": delivered,
                          "delivered_wait_s": finalize_delivered_wait_s})
            await drain()

            zsent = 0
            zbase = delivered
            zbuf = np.zeros(CHUNK, dtype=np.int16).tobytes()
            while zsent < ZERO_CAP_SAMPLES:
                target = t0 + (zbase + zsent) / 16000.0
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                await ws.send(zbuf)
                zsent += CHUNK
                await drain()
                if fin_wall is not None:
                    break
            zeros_bounded = zsent
            delivered = zbase + zsent
            sends.append({"kind": "post-finalize-zeros", "nsamples": zsent,
                          "wall": round(time.monotonic() - t0, 3),
                          "delivered_off": delivered,
                          "stopped_on_fin": fin_wall is not None})

            s0 = time.monotonic()
            await ws.send("")
            s1 = time.monotonic()
            eos_send_start = round(s0 - t0, 3)
            eos_send_end = round(s1 - t0, 3)
            sends.append({"kind": "eos-text-empty", "frame_type": "text",
                          "send_start": eos_send_start, "send_end": eos_send_end,
                          "delivered_off": delivered})
            deadline = time.monotonic() + POST_EOS_BOUND_S
            eos_bound_start = time.monotonic()
            while time.monotonic() < deadline:
                await drain()
                if finished_value is True:
                    status = "complete" if fin_wall is not None else "incomplete"
                    break
                await asyncio.sleep(0.05)
            else:
                status = "incomplete"
            finished_wait_s = round(time.monotonic() - eos_bound_start, 3)
            await drain()
            if status == "unknown":
                status = "incomplete"
            if speaker_seen:
                failure = {"error": "SpeakerLabelViolation",
                           "detail": "server emitted speaker field with no-dia config"}
                status = "failed"
            if fin_wall is None and status == "complete":
                status = "incomplete"
            if status == "incomplete" and failure is None:
                failure = {"error": "BoundTimeout",
                           "detail": ("fin_wall=%s finished=%s within bounds"
                                      % (fin_wall, finished_value))}
                journal(capdir, {"freeze_id": freeze["freeze_id"], "session_id": session_id,
                                 "event": "post-eos-incomplete", "utc": utc_now(),
                                 "detail": failure["detail"]})
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)
    except Exception as exc:
        failure = {"error": type(exc).__name__, "detail": scrub(str(exc))[:500]}
        status = "failed"
        journal(capdir, {"freeze_id": freeze["freeze_id"], "session_id": session_id,
                         "event": "post-connect-failure", "utc": utc_now(), **failure})

    for m in msg_log:
        if m.get("arrival_from_open") is not None and first_send_wall is not None:
            m["arrival_wall"] = round(m["arrival_from_open"] - (first_send_wall - arm_start), 3)
    try:
        await asyncio.wait_for(ws.close(), timeout=CLOSE_TIMEOUT_S)
        close_note = "ok"
    except Exception as exc:
        close_note = type(exc).__name__
    close_wall = round(time.monotonic() - arm_start, 3)
    sent_audio_s = round((len(payload) + zeros_trailing + zeros_bounded) / 16000.0, 3)

    post_final = [t for t in raw_tokens
                  if t.get("arrival_wall") is not None and finalize_wall is not None
                  and t["arrival_wall"] >= finalize_wall]
    turns: list = []
    cur: list = []
    for t in post_final:
        if str(t.get("text", "") or "") in ("<fin>", "<end>"):
            if cur:
                turns.append(cur)
                cur = []
        elif t.get("is_final"):
            cur.append(t)
    tail_unclosed = list(cur)
    if turns:
        acc_src = turns[-1]
        fin_matched = True
        turn_index = len(turns) - 1
    else:
        acc_src = [t for t in post_final if t.get("is_final")]
        fin_matched = False
        turn_index = -1
    acc_tokens = [{"o": i, "text": t["text"], "start_ms": t.get("start_ms"), "end_ms": t.get("end_ms")} for i, t in enumerate(acc_src)]
    final_text = "".join(t["text"] for t in acc_tokens)
    if final_text:
        groups = build_groups(final_text, acc_tokens, p0)
    else:
        groups = []
    received_finals = []
    for m in msg_log:
        ftoks = [t for t in m.get("tokens", []) if t.get("is_final")]
        if ftoks:
            received_finals.append({"text": "".join(t["text"] for t in ftoks
                                                    if t.get("text") not in ("<fin>", "<end>")),
                                    "rel_wall": m.get("arrival_wall"),
                                    "after_finalize": (m.get("arrival_wall") is not None
                                                       and finalize_wall is not None
                                                       and m["arrival_wall"] >= finalize_wall)})
    pre_finals = [e["text"] for e in received_finals
                  if e["rel_wall"] is not None and finalize_wall is not None
                  and e["rel_wall"] < finalize_wall]
    partials = []
    for m in msg_log:
        for t in m.get("tokens", []):
            if not t.get("is_final") and t.get("text") not in ("<fin>", "<end>"):
                partials.append(t["text"])

    out = {
        "case": case,
        "freeze_id": freeze["freeze_id"],
        "captured_at_utc": arm_utc,
        "status": status if not (failure and status == "complete") else "incomplete",
        "audio": {"path": str(path), "source_id": freeze["cases"][case]["source_id"],
                  "payload_samples": [p0, p1], "payload_sha256": digest},
        "session": {"endpoint": endpoint, "model_requested": "stt-rt-v5",
                    "model_server_revision": server_revision or "unknown",
                    "config_sent_redacted": redacted,
                    "open_offset_s": open_wall,
                    "first_send_offset_s": round(first_send_wall - arm_start, 3),
                    "chunk_samples": CHUNK, "pacing": "realtime-32ms",
                    "trailing_silence_samples": TRAIL_SAMPLES,
                    "post_finalize_zero_cap_samples": ZERO_CAP_SAMPLES,
                    "post_finalize_zero_sent_samples": zeros_bounded,
                    "sent_audio_s": sent_audio_s,
                    "finalize_wall": finalize_wall,
                    "finalize_delivered_wait_s": finalize_delivered_wait_s,
                    "fin_wall": fin_wall,
                    "eos_send_start": eos_send_start, "eos_send_end": eos_send_end,
                    "eos_frame_type": "text", "post_eos_sends": post_eos_sends,
                    "finished_wall": finished_wall, "finished_value": finished_value,
                    "finished_wait_s": finished_wait_s,
                    "n_pre_eos": sum(1 for m in msg_log if m.get("arrival_wall") is not None
                                     and eos_send_end is not None and m["arrival_wall"] <= eos_send_end),
                    "n_post_eos": sum(1 for m in msg_log if m.get("arrival_wall") is not None
                                      and eos_send_end is not None and m["arrival_wall"] > eos_send_end),
                    "close_wall": close_wall, "close_note": close_note,
                    "n_messages": len(msg_log),
                    "n_with_tokens": sum(1 for m in msg_log if m.get("has_tokens")),
                    "n_finished_true": sum(1 for m in msg_log if m.get("finished") is True),
                    "n_fin_end": n_fin_end,
                    "speaker_seen": speaker_seen,
                    "failure": failure},
        "chunk_ledger": chunks,
        "control_sends": sends,
        "flush_note": ("sched_wall is the paced realtime target relative to first audio send; "
                       "send_start/send_end are measured websocket completions relative to arm; "
                       "chunk_ledger covers source payload only; zero padding counted separately; "
                       "all arrival walls relative to first audio send; no frame sent after EOS."),
        "raw_messages": msg_log,
        "raw_evidence": {"n_raw_tokens": len(raw_tokens),
                         "raw_msg_with_tokens": sum(1 for m in msg_log if m.get("has_tokens")),
                         "raw_msg_without_tokens": sum(1 for m in msg_log if not m.get("has_tokens")),
                         "control_fin_end": n_fin_end,
                         "dropped_field_names": sorted(dropped),
                         "raw_tokens": raw_tokens},
        "accepted": {"final_text": final_text, "n_tokens": len(acc_tokens),
                     "tokens": acc_tokens, "turn_index": turn_index,
                     "n_turns_post_finalize": len(turns),
                     "n_unclosed_tail": len(tail_unclosed),
                     "fin_matched": fin_matched,
                     "normalization_trace": {"post_finalize_tokens": len(post_final)}},
        "groups": groups,
        "received_finals": received_finals,
        "pre_finalize_finals": pre_finals,
        "partials_tail": partials[-3:],
    }
    if out["status"] == "complete" and not (fin_wall is not None and finished_value is True):
        out["status"] = "incomplete"
    dest = capdir / ("%s.json" % session_id)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    journal(capdir, {"freeze_id": freeze["freeze_id"], "session_id": session_id,
                     "event": "completed", "utc": utc_now(), "status": out["status"],
                     "n_messages": len(msg_log),
                     "n_with_tokens": out["session"]["n_with_tokens"],
                     "sent_audio_s": sent_audio_s,
                     "failure": bool(failure)})
    print("%s status=%s msgs=%d with_tokens=%d fin_end=%d fin_wall=%s finished=%s "
          "sent_audio_s=%s failure=%s -> %s"
          % (session_id, out["status"], len(msg_log), out["session"]["n_with_tokens"],
             n_fin_end, fin_wall, finished_value, sent_audio_s, failure, dest))
    return dest


def write_freeze() -> Path:
    if FREEZE_PATH.exists():
        raise SystemExit("freeze exists refusing overwrite %s" % FREEZE_PATH)
    cases = {
        "R1": {"source_id": "ami_ES2009a",
               "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/ES2009a/ES2009a.Mix-Headset.wav",
               "payload_samples": [3159968, 3207328]},
        "R2": {"source_id": "ami_EN2009d",
               "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/EN2009d/EN2009d.Mix-Headset.wav",
               "payload_samples": [670592, 701312]},
        "T1": {"source_id": "ami_EN2009d",
               "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/EN2009d/EN2009d.Mix-Headset.wav",
               "payload_samples": [701760, 755520]},
        "COMBINED": {"source_id": "ami_EN2009d",
                     "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/EN2009d/EN2009d.Mix-Headset.wav",
                     "payload_samples": [670592, 755520]},
        "BC1": {"source_id": "ami_ES2009a",
                "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/ES2009a/ES2009a.Mix-Headset.wav",
                "payload_samples": [9119360, 9125280]},
        "SINGLE_ES2009c": {"source_id": "ami_ES2009c",
                           "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/ES2009c/ES2009c.Mix-Headset.wav",
                           "payload_samples": [770080, 789280]},
        "SINGLE_ES2009d": {"source_id": "ami_ES2009d",
                           "audio_path": "C:/Users/salee/.psem-corpus/ami/audio/ES2009d/ES2009d.Mix-Headset.wav",
                           "payload_samples": [723520, 747680]},
    }
    audio_identities = {}
    for case in CASE_ORDER:
        spec = cases[case]
        path, frames, fsha, nbytes = source_identity(spec["audio_path"])
        p0, p1 = spec["payload_samples"]
        assert 0 <= p0 < p1 <= frames, "payload out of range %s" % case
        with wave.open(str(path), "rb") as r:
            r.setpos(p0)
            praw = r.readframes(p1 - p0)
        psha = hashlib.sha256(praw).hexdigest()
        spec["payload_sha256"] = psha
        spec["payload_span_s"] = round((p1 - p0) / 16000.0, 3)
        fid = spec["source_id"]
        if fid not in audio_identities:
            audio_identities[fid] = {"audio_path": spec["audio_path"], "bytes": nbytes,
                                     "frames": frames,
                                     "duration_s": round(frames / 16000.0, 3),
                                     "sha256": fsha}
        else:
            assert audio_identities[fid]["sha256"] == fsha
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    total_payload = sum(s["payload_samples"][1] - s["payload_samples"][0] for s in cases.values())
    worst = (total_payload + 7 * (TRAIL_SAMPLES + ZERO_CAP_SAMPLES)) / 16000.0
    freeze = {
        "freeze_id": "psem.product_translation.capture.v1",
        "frozen_at_utc": utc_now(),
        "authority": {"baseline_branch": "experiment-v2-speaker-change-turn-boundaries-ls",
                      "baseline_commit": "83aaed984b8b245082f3ffe7bb15d71f3242361f",
                      "upstream": "0/0",
                      "authorization": "user explicit execute product comparison authorizes bounded existing services cost; Director freezes 7 guard ASR new captures using existing SONIOX_API_KEY; max 7 sessions one each no retries; audio cap 40s including tails",
                      "mutation_surface": ["experiments/psem_product_translation/capture.py",
                                           "experiments/psem_product_translation/captures/",
                                           "experiments/psem_product_translation/CAPTURE_FREEZE.json",
                                           "experiments/psem_product_translation/CAPTURE_MANIFEST.json"],
                      "no_goals": ["no git mutation", "no prod changes", "no old file edits",
                                   "no new inference", "no translation calls",
                                   "no speaker labels", "no retries"]},
        "budget": {"max_sessions": 7, "audio_cap_s": 40,
                   "worst_case_audio_s": round(worst, 3),
                   "note": "one attempt per case no retries estimated under $0.01 not a billing claim"},
        "cases": cases,
        "audio_identities": audio_identities,
        "config": {"model": "stt-rt-v5", "language_hints": ["en"],
                   "diarization": "omitted server default off",
                   "endpoint": "wss://stt-rt.soniox.com/transcribe-websocket",
                   "chunk_samples": CHUNK, "pacing": "realtime-32ms",
                   "trailing_silence_samples": TRAIL_SAMPLES,
                   "post_finalize_zero_cap_samples": ZERO_CAP_SAMPLES,
                   "eos": "one TEXT empty string frame once then no further sends",
                   "post_eos_finished_bound_s": POST_EOS_BOUND_S,
                   "success": "fin token after finalize plus finished true else incomplete"},
        "clock": {"origin": "first audio send wall",
                  "note": "all arrival finalize fin eos finished walls relative to payload start"},
        "script": {"path": "experiments/psem_product_translation/capture.py",
                   "sha256": script_sha},
        "secrets": {"key_name": "SONIOX_API_KEY", "key_length": 64,
                    "note": "loaded from .env.local never logged values redacted"},
    }
    assert worst < 40, "budget exceeded before start"
    EXP.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(freeze, indent=1) + "\n", encoding="utf-8")
    print("freeze %s cases=7 worst_audio_s=%s -> %s"
          % (freeze["freeze_id"], round(worst, 3), FREEZE_PATH))
    return FREEZE_PATH


def build_manifest(capdir: Path) -> Path:
    freeze = load_freeze()
    out_cases = {}
    for case in CASE_ORDER:
        cap = capdir / ("%s.json" % case)
        if not cap.exists():
            out_cases[case] = {"capture_path": None, "sha256": None,
                               "source_id": freeze["cases"][case]["source_id"],
                               "audio_path": freeze["cases"][case]["audio_path"],
                               "audio_sha": None,
                               "payload_samples": freeze["cases"][case]["payload_samples"],
                               "terminal_relative_wall": None, "complete": False,
                               "errors": ["missing capture"]}
            continue
        data = json.loads(cap.read_text(encoding="utf-8"))
        fid = freeze["cases"][case]["source_id"]
        errs = []
        if data.get("failure"):
            errs.append(str(data["failure"].get("error", "failure")))
        if data.get("status") != "complete":
            errs.append("status=%s" % data.get("status"))
        out_cases[case] = {
            "capture_path": "experiments/psem_product_translation/captures/%s.json" % case,
            "sha256": hashlib.sha256(cap.read_bytes()).hexdigest(),
            "source_id": data.get("audio", {}).get("source_id", fid),
            "audio_path": freeze["cases"][case]["audio_path"],
            "audio_sha": freeze["audio_identities"][fid]["sha256"],
            "payload_samples": freeze["cases"][case]["payload_samples"],
            "terminal_relative_wall": data.get("session", {}).get("fin_wall"),
            "complete": data.get("status") == "complete",
            "errors": errs}
    manifest = {"freeze_id": freeze["freeze_id"],
                "created_at_utc": utc_now(),
                "cases": out_cases,
                "attempt_ledger": journal_entries(capdir)}
    dest = EXP / "CAPTURE_MANIFEST.json"
    dest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("manifest complete=%d/7 -> %s"
          % (sum(1 for v in out_cases.values() if v["complete"]), dest))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--cap-dir", default=None)
    args = ap.parse_args()
    if args.freeze_only:
        write_freeze()
        return
    freeze = load_freeze()
    capdir = Path(args.cap_dir) if args.cap_dir else EXP / "captures"
    endpoint = args.endpoint or freeze["config"]["endpoint"]
    if args.manifest_only:
        build_manifest(capdir)
        return
    wanted = list(CASE_ORDER) if args.all else [args.case]
    if not wanted[0]:
        raise SystemExit("pass --case X or --all")
    for case in wanted:
        if case not in freeze["cases"]:
            raise SystemExit("unknown case %s" % case)
        asyncio.run(capture_session(case, freeze, endpoint, capdir))


if __name__ == "__main__":
    main()
