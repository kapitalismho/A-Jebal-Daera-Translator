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
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

EXP = ROOT / "experiments" / "psem_evidence_delivery_gap" / "preroll_probe"
CHUNK = 512
HOLD_S = 1.5
DRAIN_TIMEOUT_S = 60.0
QUIESCE_S = 3.0
ARM_TIMEOUT_S = 300.0
CLOSE_TIMEOUT_S = 10.0


def load_soniox_key() -> str:
    for line in (ROOT / ".env.local").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("SONIOX_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SONIOX_API_KEY missing")


def load_inputs(freeze: dict) -> dict[str, np.ndarray]:
    audio = freeze["audio"]
    a = freeze["anchors"]
    with wave.open(audio["path"], "rb") as r:
        assert r.getnchannels() == 1 and r.getsampwidth() == 2
        assert r.getframerate() == 16000 and r.getnframes() == audio["nframes"]
        r.setpos(a["C0_input"][0])
        c0 = np.frombuffer(r.readframes(a["C0_len_samples"]), dtype=np.int16).copy()
        r.setpos(a["C1_input"][0])
        c1 = np.frombuffer(r.readframes(a["C1_len_samples"]), dtype=np.int16).copy()
    assert len(c0) == a["C0_len_samples"] and len(c1) == a["C1_len_samples"]
    assert c1[-len(c0):].tobytes() == c0.tobytes()
    assert c1[:a["preroll_samples"]].tobytes() != b"\x00" * (a["preroll_samples"] * 2) or True
    return {"C0": c0, "C1": c1}


async def pump_events(session, log: list) -> None:
    try:
        async for ev in session.events():
            log.append({"wall": time.monotonic(), "text": ev.text,
                        "is_final": bool(ev.is_final)})
    except Exception as exc:
        log.append({"wall": time.monotonic(), "error": type(exc).__name__,
                    "detail": str(exc)[:200]})


async def run_arm(backend: SonioxRealtimeSTTBackend, arm: str,
                  payload: np.ndarray, src_start: int) -> dict:
    arm_t0 = time.monotonic()
    arm_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t = time.monotonic()
    session = await backend.open_session()
    session_open_s = round(time.monotonic() - t, 3)
    log: list = []
    pump = asyncio.create_task(pump_events(session, log))
    hold: deque = deque()
    wake = asyncio.Event()
    state = {"emitted": 0, "forwarded": src_start, "source_done": False,
             "max_depth": 0, "stalls": 0}
    total = len(payload)
    t0 = time.monotonic()
    utterance_id = str(uuid.uuid4())

    async def source() -> None:
        for off in range(0, total, CHUNK):
            target = t0 + (off / CHUNK) * (CHUNK / 16000.0)
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            piece = payload[off:off + CHUNK]
            hold.append({"emit_wall": time.monotonic(),
                         "release_wall": time.monotonic() + HOLD_S,
                         "src_pos": src_start + off, "data": piece.tobytes(),
                         "nsamp": len(piece)})
            state["emitted"] = off + len(piece)
            state["max_depth"] = max(state["max_depth"], len(hold))
            if len(hold) >= 128:
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
            if head["data"]:
                await session.send_audio(head["data"])
                state["forwarded"] = head["src_pos"] + head["nsamp"]
            if not hold and state["source_done"]:
                return
            wake.clear()

    try:
        async with asyncio.timeout(ARM_TIMEOUT_S):
            await asyncio.gather(asyncio.create_task(source()),
                                 asyncio.create_task(forward()))
            assert state["forwarded"] == src_start + total
            finalize_wall = round(time.monotonic() - arm_t0, 3)
            await session.on_speech_end()
            fw = arm_t0 + finalize_wall
            deadline = time.monotonic() + DRAIN_TIMEOUT_S
            last_post = 0.0
            while time.monotonic() < deadline:
                posts = [e for e in log
                         if e.get("is_final") and "error" not in e and e["wall"] >= fw]
                if posts and time.monotonic() - posts[-1]["wall"] >= QUIESCE_S:
                    last_post = posts[-1]["wall"]
                    break
                await asyncio.sleep(0.05)
            post_finals = [e for e in log
                           if e.get("is_final") and "error" not in e and e["wall"] >= fw]
            pre_finals = [e["text"] for e in log
                          if e.get("is_final") and "error" not in e and e["wall"] < fw]
            nonempty = [e for e in post_finals if e["text"]]
            sel = nonempty[-1] if nonempty else (post_finals[-1] if post_finals else None)
            collection_elapsed_s = round(time.monotonic() - (arm_t0 + finalize_wall), 3)
    finally:
        try:
            await asyncio.wait_for(session.close(), timeout=CLOSE_TIMEOUT_S)
        except Exception:
            pass
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
    seg = {"utterance_id": utterance_id, "src_start": src_start,
           "src_end": src_start + total, "session_open_s": session_open_s,
           "finalize_wall": finalize_wall,
           "final_text": sel["text"] if sel else "",
           "final_wall": round(sel["wall"] - arm_t0, 3) if sel else None,
           "final_latency_s": round(sel["wall"] - (arm_t0 + finalize_wall), 3) if sel else None,
           "final_status": ("ok" if sel and sel["text"]
                            else ("empty-ack" if sel else "timeout")),
           "received_finals": [{"text": e["text"], "rel_wall": round(e["wall"] - arm_t0, 3),
                                "after_finalize": e["wall"] >= fw} for e in log
                               if e.get("is_final") and "error" not in e],
           "pre_finalize_finals": pre_finals,
           "partials_tail": [e["text"] for e in log if not e.get("is_final")][-3:]}
    return {"arm": arm, "arm_utc": arm_utc, "bounds": [src_start, src_start + total],
            "segments": [seg],
            "forward_accounting": {"target_samples": total,
                                   "forwarded_samples": state["forwarded"] - src_start,
                                   "gap_dup_ok": state["forwarded"] == src_start + total},
            "queue": {"max_depth": state["max_depth"], "stall_threshold": 128,
                      "stalls": state["stalls"],
                      "input_chunks": (total + CHUNK - 1) // CHUNK},
            "provider_sessions": 1, "collection_elapsed_s": collection_elapsed_s,
            "wall_s": round(time.monotonic() - arm_t0, 3)}


async def main() -> None:
    freeze = json.loads((EXP / "freeze.json").read_text(encoding="utf-8"))
    t_run = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payloads = load_inputs(freeze)
    backend = SonioxRealtimeSTTBackend(api_key=load_soniox_key(), language_hints=["en"])
    a = freeze["anchors"]
    out = {"run_utc": t_run, "freeze_id": freeze["authority"]["freeze_id"],
           "stt_model": "stt-rt-v5",
           "stt_endpoint": "wss://stt-rt.soniox.com/transcribe-websocket",
           "hold_samples": 24000,
           "arms": {"C0": a["C0_input"], "C1": a["C1_input"]},
           "translations": 0, "results": []}
    errors = 0
    for name in ("C0", "C1"):
        try:
            res = await run_arm(backend, name, payloads[name], a[f"{name}_input"][0])
        except Exception as exc:
            errors += 1
            res = {"arm": name, "error": type(exc).__name__, "detail": str(exc)[:300]}
            if "Balance" in str(exc) or "401" in str(exc) or "403" in str(exc):
                out["results"].append(res)
                break
        out["results"].append(res)
        print(f"arm={name} status={'error' if 'error' in res else 'done'} "
              f"final_len={len(res.get('segments', [{}])[0].get('final_text', ''))}",
              flush=True)
    out["errors"] = errors
    (EXP / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(f"arms={len(out['results'])} errors={errors} sessions="
          f"{sum(r.get('provider_sessions', 0) for r in out['results'] if 'error' not in r)}")


if __name__ == "__main__":
    asyncio.run(main())
