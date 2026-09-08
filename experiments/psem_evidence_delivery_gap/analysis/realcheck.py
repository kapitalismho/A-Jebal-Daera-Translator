"""Lexical sensitivity probe on two predeclared clips: production VAD spans plus
Soniox whole-excerpt and retrospective-split-half STT passes with translation
attempts. Not a delivery baseline, not an ideal intervention, no runtime
owner or output path exercised. Isolated adapters, keys in-process, no
settings or production writes."""
from __future__ import annotations
import asyncio
import json
import time
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))

from puripuly_heart.config.prompts import (
    load_prompt_for_provider,
    render_translation_prompt_template,
)
from puripuly_heart.core.language import get_llm_language_name
from puripuly_heart.core.vad.gating import (
    SpeechEnd,
    SpeechStart,
    create_peer_vad_gating,
)
from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

CORPUS = Path('C:/Users/salee/.psem-corpus')
VAD_MODEL = ROOT / 'src/puripuly_heart/data/vad/silero_vad.onnx'
OUT = ROOT / 'experiments/psem_evidence_delivery_gap/traces/realcheck.json'
CLIPS = [
    {'case': 'G04', 'wav': CORPUS / 'ami/audio/ES2009a/ES2009a.Mix-Headset.wav',
     'excerpt_ms': [559000, 580000]},
    {'case': 'G05', 'wav': CORPUS / 'ami/audio/EN2009d/EN2009d.Mix-Headset.wav',
     'excerpt_ms': [372000, 393000]},
]


def load_keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / '.env.local').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[len('export '):]
        if '=' in line and not line.startswith('#'):
            name, value = line.split('=', 1)
            out[name.strip()] = value.strip().strip('"').strip("'")
    return out


def read_excerpt(path: Path, start_ms: float, end_ms: float) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as r:
        assert r.getframerate() == 16000 and r.getnchannels() == 1 and r.getsampwidth() == 2, \
            f'audio contract mismatch: {path}'
        s0 = int(start_ms * 16)
        s1 = int(end_ms * 16)
        r.setpos(s0)
        raw = r.readframes(s1 - s0)
    return np.frombuffer(raw, dtype=np.int16), 16000


def vad_spans(pcm: np.ndarray) -> list[dict]:
    engine = SileroVadOnnx(VAD_MODEL)
    gating = create_peer_vad_gating(
        engine, sample_rate_hz=16000, ring_buffer_ms=500,
        speech_threshold=0.5, hangover_ms=500)
    spans: list[dict] = []
    active: int | None = None
    uid: str | None = None
    n = 0
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
                buffered = 1 + sum(type(v).__name__ == 'SpeechChunk' for v in events)
                active = max(0, off - (buffered - 1) * 512 - int(np.asarray(event.pre_roll).size))
                uid = str(event.utterance_id)[:8]
            elif isinstance(event, SpeechEnd) and active is not None:
                end = off + orig
                if event.reason == 'silence':
                    end -= int(round(event.trailing_silence_ms * 16))
                spans.append({'utterance': uid, 'start_sample': active,
                              'end_sample': max(active, end), 'reason': event.reason,
                              'trailing_silence_ms': event.trailing_silence_ms})
                active = None
    if active is not None:
        spans.append({'utterance': uid, 'start_sample': active,
                      'end_sample': len(pcm), 'reason': 'eos-flush',
                      'trailing_silence_ms': 0})
    return spans


async def stt_pass(pcm: np.ndarray, api_key: str, tag: str) -> dict:
    backend = SonioxRealtimeSTTBackend(api_key=api_key, language_hints=['en'])
    t0 = time.perf_counter()
    session = await backend.open_session()
    partials: list[str] = []
    finals: list[str] = []
    try:
        for off in range(0, len(pcm), 9600):
            await session.send_audio(pcm[off:off + 9600].tobytes())
        await session.on_speech_end()
        async with asyncio.timeout(45.0):
            async for event in session.events():
                (finals if event.is_final else partials).append(event.text)
                if event.is_final:
                    break
    finally:
        await session.close()
    return {'tag': tag, 'wall_s': round(time.perf_counter() - t0, 2),
            'partials': partials[-3:], 'finals': finals}


async def main() -> None:
    keys = load_keys()
    assert keys.get('SONIOX_API_KEY'), 'SONIOX_API_KEY missing'
    assert keys.get('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY missing'
    enrichment = json.loads(
        (ROOT / 'experiments/psem_evidence_delivery_gap/traces/case_enrichment.json')
        .read_text(encoding='utf-8'))
    template = load_prompt_for_provider('openrouter')
    system_prompt = render_translation_prompt_template(
        template, source_name=get_llm_language_name('en'),
        target_name=get_llm_language_name('ko'))
    translator = OpenRouterLLMProvider(api_key=keys['OPENROUTER_API_KEY'],
                                       model='google/gemma-4-26b-a4b-it')
    results: list[dict] = []
    try:
        for clip in CLIPS:
            pcm, _ = read_excerpt(clip['wav'], *clip['excerpt_ms'])
            spans = vad_spans(pcm)
            case = enrichment[clip['case']]
            from experiments.psem_state_corrected_adaptation_gate import h_postprocess as hp
            v = hp.load_validated_export(
                ROOT / 'experiments/psem_state_corrected_adaptation_gate/results/issue-121-h7301-persistence-v1/export/gpu_export')
            dev = v['dev'][case['source']]['session']
            refs = sorted(int(e.boundary_source_sample) for e in dev.reference.events)
            s0 = int(case['src_span_ms'][0] * 16)
            e1 = int(case['src_span_ms'][1] * 16)
            inside = [r for r in refs if s0 <= r <= e1]
            assert inside, f'no frozen reference inside {clip["case"]}'
            split_abs = inside[0]
            split_rel = split_abs - int(clip['excerpt_ms'][0] * 16)
            whole = await stt_pass(pcm, keys['SONIOX_API_KEY'], 'whole')
            first = await stt_pass(pcm[:split_rel], keys['SONIOX_API_KEY'], 'first-half')
            second = await stt_pass(pcm[split_rel:], keys['SONIOX_API_KEY'], 'second-half')
            translations: dict[str, str] = {}
            translation_route: dict[str, str] = {}
            for key, transcript in (
                    ('whole', ' '.join(whole['finals'])),
                    ('halves', ' '.join(first['finals']) + ' ||| ' + ' '.join(second['finals']))):
                if not transcript.strip():
                    translations[key] = ''
                    translation_route[key] = 'skipped-empty-transcript'
                    continue
                try:
                    translations[key] = (
                        await translator.translate(
                            utterance_id=uuid4(), text=transcript,
                            system_prompt=system_prompt, source_language='en',
                            target_language='ko')).text
                    translation_route[key] = 'openrouter:google/gemma-4-26b-a4b-it'
                except Exception as exc:
                    translations[key] = ''
                    translation_route[key] = f'openrouter-failed:{type(exc).__name__}:{exc}'
            results.append({
                'case': clip['case'], 'wav': str(clip['wav']),
                'excerpt_ms': clip['excerpt_ms'],
                'frozen_ref_boundary_sample': split_abs,
                'frozen_ref_boundary_excerpt_ms': round(split_rel / 16, 1),
                'vad_baseline_spans': [
                    {**s, 'start_excerpt_ms': round(s['start_sample'] / 16, 1),
                     'end_excerpt_ms': round(s['end_sample'] / 16, 1)} for s in spans],
                'translation_model': 'google/gemma-4-26b-a4b-it',
                'translation_route': translation_route,
                'whole': whole, 'first_half': first, 'second_half': second,
                'translations': translations,
            })
    finally:
        await translator.close()
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'clips={len(results)}')


if __name__ == '__main__':
    asyncio.run(main())
