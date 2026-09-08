from __future__ import annotations
import asyncio
import json
import time
import uuid
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, create_peer_vad_gating
from puripuly_heart.core.vad.silero import SileroVadOnnx
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

EXP = ROOT / "experiments" / "psem_evidence_delivery_gap" / "revised_phase_d"
CHUNK = 512
HOLD_S = 1.5
HOLD_SAMPLES = 24000
STALL_THRESHOLD = 128
DRAIN_TIMEOUT_S = 60.0
ARM_TIMEOUT_S = 300.0
TURN_SAMPLE = 52156984


def load_soniox_key() -> str:
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("SONIOX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SONIOX_API_KEY missing")


def select_target_payload(freeze: dict) -> tuple[np.ndarray, int, list]:
    audio = freeze["audio"]
    want_start, want_end = freeze["vad"]["payload_samples"]
    with wave.open(audio["path"], "rb") as r:
        s0, s1 = audio["excerpt_samples"]
        r.setpos(s0)
        pcm = np.frombuffer(r.readframes(s1 - s0), dtype=np.int16)
    engine = SileroVadOnnx(ROOT / "src/puripuly_heart/data/vad/silero_vad.onnx")
    gating = create_peer_vad_gating(
        engine, sample_rate_hz=16000, ring_buffer_ms=500,
        speech_threshold=0.5, hangover_ms=500)
    spans = []
    active = None
    for off in range(0, len(pcm), CHUNK):
        payload = pcm[off:off + CHUNK]
        orig = len(payload)
        chunk = payload.astype(np.float32)
        if orig < CHUNK:
            chunk = np.pad(chunk, (0, CHUNK - orig))
        events = gating.process_chunk(chunk / 32768.0)
        for event in events:
            if isinstance(event, SpeechStart):
                buffered = 1 + sum(type(v).__name__ == "SpeechChunk" for v in events)
                start = max(0, off - (buffered - 1) * CHUNK - int(np.asarray(event.pre_roll).size))
                active = {"start_rel": start, "uid": str(event.utterance_id)}
            elif isinstance(event, SpeechEnd) and active is not None:
                emission = off + orig
                end = emission
                if event.reason == "silence":
                    end -= int(round(event.trailing_silence_ms * 16))
                active.update(end_rel=max(active["start_rel"], end), reason=event.reason)
                spans.append(active)
                active = None
    if active is not None:
        active.update(end_rel=len(pcm), reason="eos-flush")
        spans.append(active)
    w0 = audio["excerpt_samples"][0]
    hit = None
    for s in spans:
        if w0 + s["start_rel"] <= TURN_SAMPLE < w0 + s["end_rel"]:
            hit = s
    if hit is None:
        raise RuntimeError("no VAD span covers turn")
    if w0 + hit["start_rel"] != want_start or w0 + hit["end_rel"] != want_end:
        raise RuntimeError("target span mismatch freeze")
    payload = pcm[hit["start_rel"]:hit["end_rel"]].copy()
    info = [{"start_abs": w0 + s["start_rel"], "end_abs": w0 + s["end_rel"],
             "reason": s["reason"]} for s in spans]
    return payload, w0 + hit["start_rel"], info


async def pump_events(session, log: list) -> None:
    try:
        async for ev in session.events():
            log.append({"wall": time.monotonic(), "text": ev.text, "is_final": bool(ev.is_final)})
    except Exception as exc:
        log.append({"wall": time.monotonic(), "error": type(exc).__name__,
                    "detail": str(exc)[:200]})


async def run_arm(backend: SonioxRealtimeSTTBackend, arm: str, bounds: list[int],
                  injections: list[tuple[int, int]], payload: np.ndarray, p0: int) -> dict:
    nseg = len(bounds) - 1
    arm_t0 = time.monotonic()
    arm_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sessions = []
    opens = []
    for _ in range(nseg):
        t = time.monotonic()
        sessions.append(await backend.open_session())
        opens.append(round(time.monotonic() - t, 3))
    logs: list[list] = [[] for _ in sessions]
    pumps = [asyncio.create_task(pump_events(s, logs[i])) for i, s in enumerate(sessions)]
    segs = [{"utterance_id": str(uuid.uuid4()), "src_start": bounds[i],
             "src_end": None, "session": i, "open_wall": opens[i]} for i in range(nseg)]
    hold: deque = deque()
    wake = asyncio.Event()
    state = {"emitted": 0, "forwarded": p0, "source_done": False, "cur": 0,
             "splits": [], "max_depth": 0, "stalls": 0}
    inj_log: list[dict] = []
    switches: list[dict] = []
    total = len(payload)
    t0 = time.monotonic()

    async def source() -> None:
        for off in range(0, total, CHUNK):
            target = t0 + (off / CHUNK) * (CHUNK / 16000.0)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            piece = payload[off:off + CHUNK]
            src_pos = p0 + off
            hold.append({"emit_wall": time.monotonic(), "release_wall": time.monotonic() + HOLD_S,
                         "src_pos": src_pos, "data": piece.tobytes(), "nsamp": len(piece)})
            state["emitted"] = off + len(piece)
            state["max_depth"] = max(state["max_depth"], len(hold))
            if len(hold) >= STALL_THRESHOLD:
                state["stalls"] += 1
                await asyncio.sleep(0.005)
            wake.set()
        state["source_done"] = True
        wake.set()

    async def forward() -> None:
        while True:
            if not hold:
                if state["source_done"]:
                    return
                wake.clear()
                await wake.wait()
                continue
            head = hold[0]
            delay = head["release_wall"] - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            hold.popleft()
            pos, data, nsamp = head["src_pos"], head["data"], head["nsamp"]
            cuts = sorted(s for s in state["splits"] if pos < s < pos + nsamp)
            parts = []
            prev = pos
            for c in cuts:
                parts.append((prev, data[(prev - pos) * 2:(c - pos) * 2], c))
                prev = c
            parts.append((prev, data[(prev - pos) * 2:], None))
            for start, raw, cut in parts:
                if raw:
                    await sessions[state["cur"]].send_audio(raw)
                    state["forwarded"] = start + len(raw) // 2
                if cut is not None:
                    old = state["cur"]
                    segs[old]["src_end"] = cut
                    segs[old]["finalize_wall"] = round(time.monotonic() - arm_t0, 3)
                    await sessions[old].on_speech_end()
                    state["cur"] += 1
                    switches.append({"at_sample": cut, "from_seg": old, "to_seg": state["cur"],
                                     "wall": round(time.monotonic() - arm_t0, 3)})
            if not hold and state["source_done"]:
                return
            wake.clear()

    async def inject(s_req: int, te: int) -> None:
        delay = t0 + (te - p0) / 16000.0 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        received = p0 + state["emitted"]
        mark = state["forwarded"]
        rec = {"S": s_req, "Te": te, "inject_wall": round(time.monotonic() - arm_t0, 3),
               "watermark_sent": mark, "received_pos": received}
        if s_req in state["splits"]:
            rec.update(applied=False, reason="duplicate-request")
        elif s_req <= mark:
            rec.update(applied=False, reason="already-sent")
        elif s_req > received:
            rec.update(applied=False, reason="not-yet-received")
        else:
            state["splits"].append(s_req)
            state["splits"].sort()
            rec.update(applied=True, reason="within-unsent-window")
        inj_log.append(rec)

    try:
        async with asyncio.timeout(ARM_TIMEOUT_S):
            tasks = [asyncio.create_task(source()), asyncio.create_task(forward())]
            tasks += [asyncio.create_task(inject(s, te)) for s, te in injections]
            await asyncio.gather(*tasks)
            segs[state["cur"]]["src_end"] = p0 + total
            segs[state["cur"]]["finalize_wall"] = round(time.monotonic() - arm_t0, 3)
            await sessions[state["cur"]].on_speech_end()
            collection_elapsed = []
            for i, sg in enumerate(segs):
                fw = arm_t0 + sg["finalize_wall"]
                deadline = time.monotonic() + DRAIN_TIMEOUT_S
                found = None
                while found is None and time.monotonic() < deadline:
                    cands = [e for e in logs[i] if e.get("is_final") and e["wall"] >= fw]
                    if cands:
                        found = cands[0]
                        break
                    await asyncio.sleep(0.05)
                collection_elapsed.append(round(time.monotonic() - (arm_t0 + sg["finalize_wall"]), 3))
                sg["final_text"] = found["text"] if found else ""
                sg["final_wall"] = round(found["wall"] - arm_t0, 3) if found else None
                sg["final_latency_s"] = round(sg["final_wall"] - sg["finalize_wall"], 3) if found else None
                sg["final_status"] = "ok" if found and found["text"] else ("empty-final" if found else "timeout")
                sg["pre_finalize_finals"] = [e["text"] for e in logs[i]
                                             if e.get("is_final") and e["wall"] < fw]
                sg["partials_tail"] = [e["text"] for e in logs[i] if not e.get("is_final")][-3:]
    finally:
        for s in sessions:
            try:
                await asyncio.wait_for(s.close(), timeout=10.0)
            except Exception:
                pass
        for p in pumps:
            p.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
    spans = [(sg["src_start"], sg["src_end"]) for sg in segs]
    contiguous = all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    gap_dup = spans[0][0] == p0 and spans[-1][1] == p0 + total and contiguous
    if not gap_dup:
        raise RuntimeError(f"{arm} forward accounting failed: {spans}")
    return {"arm": arm, "arm_utc": arm_utc, "bounds": bounds, "segments": segs,
            "injections": inj_log, "switches": switches,
            "forward_accounting": {"target_samples": total, "gap_dup_ok": gap_dup},
            "queue": {"max_depth": state["max_depth"], "stall_threshold": STALL_THRESHOLD,
                      "stalls": state["stalls"],
                      "input_chunks": (total + CHUNK - 1) // CHUNK},
            "provider_sessions": nseg,
            "session_opens_s": opens, "collection_elapsed_s": collection_elapsed,
            "wall_s": round(time.monotonic() - arm_t0, 3)}


async def main() -> None:
    freeze = json.loads((EXP / "freeze.json").read_text(encoding="utf-8"))
    t_run = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload, p0, spans = select_target_payload(freeze)
    backend = SonioxRealtimeSTTBackend(api_key=load_soniox_key(), language_hints=["en"])
    p1 = p0 + len(payload)
    arms = [
        ("baseline", [p0, p1], []),
        ("ideal", [p0, 52156984, p1], [(52156984, 52175224)]),
        ("f0", [p0, 52139200, 52203200, p1], [(52139200, 52156200), (52203200, 52217640)]),
        ("h", [p0, 52158400, 52203200, p1], [(52158400, 52171560), (52203200, 52217640)]),
    ]
    out = {"run_utc": t_run, "freeze_id": freeze["authority"]["freeze_id"],
           "stt_model": "stt-rt-v5", "stt_endpoint": "wss://stt-rt.soniox.com/transcribe-websocket",
           "hold_samples": HOLD_SAMPLES, "payload_samples": [p0, p1],
           "vad_spans": spans, "translations": 0, "results": []}
    errors = 0
    for name, bounds, injs in arms:
        try:
            res = await run_arm(backend, name, bounds, injs, payload, p0)
        except Exception as exc:
            errors += 1
            res = {"arm": name, "error": type(exc).__name__, "detail": str(exc)[:300]}
            if "Balance" in str(exc) or "401" in str(exc) or "403" in str(exc):
                out["results"].append(res)
                break
        out["results"].append(res)
        print(f"arm={name} status={'error' if 'error' in res else 'done'} "
              f"finals={[len(s.get('final_text', '')) for s in res.get('segments', [])]}", flush=True)
    out["errors"] = errors
    (EXP / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"arms={len(out['results'])} errors={errors} sessions="
          f"{sum(len(r.get('segments', [])) for r in out['results'])}")


if __name__ == "__main__":
    asyncio.run(main())
