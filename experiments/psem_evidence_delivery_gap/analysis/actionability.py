from __future__ import annotations
import hashlib
import json
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))

from puripuly_heart.core.vad.gating import (
    SpeechEnd,
    SpeechStart,
    create_peer_vad_gating,
)
from puripuly_heart.core.vad.silero import SileroVadOnnx

CORPUS = Path("C:/Users/salee/.psem-corpus")
VAD_MODEL = ROOT / "src/puripuly_heart/data/vad/silero_vad.onnx"
EXP = ROOT / "experiments/psem_evidence_delivery_gap"
OUT = EXP / "traces/actionability.json"


def read_window(path: Path, start_ms: float, end_ms: float) -> tuple[np.ndarray, str]:
    with wave.open(str(path), "rb") as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2
        s0 = int(start_ms * 16)
        s1 = int(end_ms * 16)
        r.setpos(s0)
        raw = r.readframes(s1 - s0)
    assert len(raw) == (s1 - s0) * 2
    return np.frombuffer(raw, dtype=np.int16), hashlib.sha256(raw).hexdigest()


def run_vad(pcm: np.ndarray, w0_sample: int) -> list[dict]:
    engine = SileroVadOnnx(VAD_MODEL)
    gating = create_peer_vad_gating(
        engine, sample_rate_hz=16000, ring_buffer_ms=500,
        speech_threshold=0.5, hangover_ms=500)
    segments: list[dict] = []
    open_seg: dict | None = None
    for off in range(0, len(pcm), 512):
        payload = pcm[off:off + 512]
        orig = len(payload)
        chunk = payload.astype(np.float32)
        if orig < 512:
            chunk = np.pad(chunk, (0, 512 - orig))
        chunk = chunk / 32768.0
        events = gating.process_chunk(chunk)
        for event in events:
            if isinstance(event, SpeechStart):
                buffered = 1 + sum(type(v).__name__ == "SpeechChunk" for v in events)
                active = max(0, off - (buffered - 1) * 512 - int(np.asarray(event.pre_roll).size))
                open_seg = {
                    "utterance_id": str(event.utterance_id),
                    "short": str(event.utterance_id)[:8],
                    "start_rel": active,
                    "pre_roll_samples": int(np.asarray(event.pre_roll).size),
                    "start_emission_rel": off + orig,
                }
            elif isinstance(event, SpeechEnd) and open_seg is not None:
                emission = off + orig
                end = emission
                if event.reason == "silence":
                    end -= int(round(event.trailing_silence_ms * 16))
                open_seg.update({
                    "payload_end_rel": max(open_seg["start_rel"], end),
                    "end_emission_rel": emission,
                    "reason": event.reason,
                    "trailing_silence_ms": event.trailing_silence_ms,
                })
                segments.append(open_seg)
                open_seg = None
    if open_seg is not None:
        open_seg.update({
            "payload_end_rel": len(pcm),
            "end_emission_rel": None,
            "reason": "eos-flush",
            "trailing_silence_ms": 0,
        })
        segments.append(open_seg)
    for s in segments:
        s["start_abs"] = w0_sample + s["start_rel"]
        s["payload_end_abs"] = w0_sample + s["payload_end_rel"]
        s["start_emission_abs"] = w0_sample + s["start_emission_rel"]
        s["end_emission_abs"] = (w0_sample + s["end_emission_rel"]
                                 if s["end_emission_rel"] is not None else None)
        s["start_ms"] = round(s["start_abs"] / 16, 1)
        s["payload_end_ms"] = round(s["payload_end_abs"] / 16, 1)
        s["end_emission_ms"] = (round(s["end_emission_abs"] / 16, 1)
                                if s["end_emission_abs"] is not None else None)
    return segments


def evaluate(segments: list[dict], boundary_abs: int, frontier_abs: int) -> dict:
    b = boundary_abs
    f = frontier_abs
    owner = None
    position = "outside"
    for idx, s in enumerate(segments):
        if s["start_abs"] <= b < s["payload_end_abs"]:
            owner = s
            prev_end = segments[idx - 1]["payload_end_abs"] if idx > 0 else None
            if prev_end is not None and b < prev_end:
                position = "inside-payload-pre-roll-overlap"
            else:
                position = "inside-payload"
            break
        if (s["end_emission_abs"] is not None
                and s["payload_end_abs"] <= b < s["end_emission_abs"]):
            owner = s
            position = "in-trailing-hangover"
            break
    if owner is None:
        return {"owner_short": None, "position": position, "verdict": "OUTSIDE-ALL-SEGMENTS"}
    if owner["end_emission_abs"] is None:
        verdict = "OPEN-WITHIN-WINDOW" if f <= owner["payload_end_abs"] else "FRONTIER-BEYOND-WINDOW"
        emission_margin = None
    else:
        verdict = "OPEN" if f < owner["end_emission_abs"] else "CLOSED"
        emission_margin = round((owner["end_emission_abs"] - f) / 16, 1)
    return {
        "owner_short": owner["short"],
        "owner_utterance_id": owner["utterance_id"],
        "position": position,
        "verdict": verdict,
        "payload_margin_ms": round((owner["payload_end_abs"] - f) / 16, 1),
        "emission_margin_ms": emission_margin,
        "evidence_delay_ms": round((f - b) / 16, 1),
        "owner_reason": owner["reason"],
    }


def main() -> None:
    bounds = json.loads((EXP / "traces/emission_bounds.json").read_text(encoding="utf-8"))
    enrichment = json.loads((EXP / "traces/case_enrichment.json").read_text(encoding="utf-8"))
    realcheck = json.loads((EXP / "traces/realcheck.json").read_text(encoding="utf-8"))
    windows = [
        {"id": "G04-excerpt", "wav": str(CORPUS / "ami/audio/ES2009a/ES2009a.Mix-Headset.wav"),
         "span_ms": [559000, 580000], "scope": "same-excerpt-as-realcheck"},
        {"id": "G05-excerpt", "wav": str(CORPUS / "ami/audio/EN2009d/EN2009d.Mix-Headset.wav"),
         "span_ms": [372000, 393000], "scope": "same-excerpt-as-realcheck"},
        {"id": "G05-F0-leading", "wav": str(CORPUS / "ami/audio/EN2009d/EN2009d.Mix-Headset.wav"),
         "span_ms": [360000, 376000], "scope": "same-source-bounded-leading-context-for-F0-only"},
    ]
    plan = [
        ("G04", "H", "G04-excerpt"),
        ("G04", "F0", "G04-excerpt"),
        ("G05", "H", "G05-excerpt"),
        ("G05", "F0", "G05-F0-leading"),
    ]
    window_segments: dict[str, list[dict]] = {}
    window_meta: dict[str, dict] = {}
    for w in windows:
        pcm, digest = read_window(Path(w["wav"]), w["span_ms"][0], w["span_ms"][1])
        w0 = int(w["span_ms"][0] * 16)
        window_meta[w["id"]] = {"samples": len(pcm), "w0_sample": w0, "sha256": digest}
        window_segments[w["id"]] = run_vad(pcm, w0)
    evaluations = []
    for case, layer, wid in plan:
        hit = bounds[case]["bounds"][layer][0]
        gt = enrichment[case]["reference_inside_detail"][0]["boundary_sample"]
        ev = evaluate(window_segments[wid], hit["boundary_sample"], hit["emit_sample"])
        evaluations.append({
            "case": case, "layer": layer, "window": wid,
            "requested_boundary_sample": hit["boundary_sample"],
            "requested_boundary_ms": hit["boundary_ms"],
            "earliest_frontier_sample": hit["emit_sample"],
            "earliest_frontier_ms": hit["emit_ms"],
            "retrospective_gt_contrast_sample": gt,
            "retrospective_gt_contrast_ms": round(gt / 16, 1),
            **ev,
        })
    spot = []
    for rc in realcheck:
        wid = "G04-excerpt" if rc["case"] == "G04" else "G05-excerpt"
        ours = window_segments[wid]
        refs = rc["vad_baseline_spans"]
        diffs = []
        matched = len(ours) == len(refs)
        for a, b in zip(ours, refs):
            diffs.append(max(abs(a["start_rel"] - b["start_sample"]),
                             abs(a["payload_end_rel"] - b["end_sample"])))
            matched = matched and diffs[-1] == 0 and a["reason"] == b["reason"]
        spot.append({"case": rc["case"], "window": wid, "matched": matched,
                     "max_abs_diff_samples": max(diffs) if diffs else None})
    trace = {
        "identity": {
            "baseline_commit": "45956c69895aa3e33c60e9c516fc40ba7d60b954",
            "branch": "experiment-v2-speaker-change-turn-boundaries-ls",
            "script": "experiments/psem_evidence_delivery_gap/analysis/actionability.py",
            "vad_profile": {"threshold": 0.5, "ring_buffer_ms": 500, "hangover_ms": 500,
                            "pre_roll_ms": 500, "max_segment_ms": 7000,
                            "start_debounce_chunks": 3, "start_commit_chunks": 3,
                            "chunk_samples": 512},
            "vad_model": "src/puripuly_heart/data/vad/silero_vad.onnx",
            "canonical_source": "traces/emission_bounds.json frontier_sweep.simulate_episode first-event return",
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "paid_calls": 0,
            "network_inference": False,
        },
        "windows": [{**w, **window_meta[w["id"]]} for w in windows],
        "segments": window_segments,
        "evaluations": evaluations,
        "spot_check_vs_realcheck": spot,
        "limits": [
            "ack receiver absent in current Audio: unsupported acknowledgement recorded, not a measured rejection",
            "earliest frontier is offline first-event lower bound, not observed wallclock arrival",
            "VAD openness is not provider mutability or commit",
            "excerpt-local replay starts VAD cold at window start; production runs continuously",
            "pre-roll overlap is contextual and keeps single-segment ownership, never double-counted",
            "retrospective GT split is contrast only, never the requested boundary",
            "no commit or latency credit taken",
        ],
    }
    OUT.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    for e in evaluations:
        print(f'{e["case"]}/{e["layer"]} {e["window"]} boundary={e["requested_boundary_ms"]} '
              f'frontier={e["earliest_frontier_ms"]} gt={e["retrospective_gt_contrast_ms"]} '
              f'{e["position"]} {e["verdict"]} owner={e.get("owner_short")}')
    for s in spot:
        print(f'spot {s["case"]} matched={s["matched"]} maxdiff={s["max_abs_diff_samples"]}')


if __name__ == "__main__":
    main()
