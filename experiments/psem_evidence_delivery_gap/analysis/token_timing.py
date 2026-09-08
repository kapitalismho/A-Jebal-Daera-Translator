"""Token-timed whole-excerpt STT passes for G04/G05.

Experiment-local tee of actual provider JSON payloads (instance-level
 landed in traces/token_timing.json). No production adapter edits.
"""
from __future__ import annotations
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))

from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend

CORPUS = Path('C:/Users/salee/.psem-corpus')
OUT = ROOT / 'experiments/psem_evidence_delivery_gap/traces/token_timing.json'
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


async def timed_pass(pcm: np.ndarray, api_key: str) -> dict:
    backend = SonioxRealtimeSTTBackend(api_key=api_key, language_hints=['en'])
    session = await backend.open_session()
    payloads: list[dict] = []

    def tee(message):
        try:
            data = json.loads(message if isinstance(message, str)
                              else message.decode('utf-8', errors='ignore'))
        except ValueError:
            data = {}
        if isinstance(data, dict) and isinstance(data.get('tokens'), list):
            for token in data['tokens']:
                if isinstance(token, dict) and bool(token.get('is_final')):
                    payloads.append({
                        'text': str(token.get('text', '') or ''),
                        'end_ms': token.get('end_ms'),
                    })
        return tee.__wrapped__(message)

    tee.__wrapped__ = session._handle_message
    session._handle_message = tee
    finals: list[str] = []
    t0 = time.perf_counter()
    try:
        for off in range(0, len(pcm), 9600):
            await session.send_audio(pcm[off:off + 9600].tobytes())
        await session.on_speech_end()
        async with asyncio.timeout(60.0):
            async for event in session.events():
                if event.is_final:
                    finals.append(event.text)
                    break
    finally:
        await session.close()
    return {'wall_s': round(time.perf_counter() - t0, 2),
            'finals': finals, 'final_tokens': payloads}


async def main() -> None:
    keys = load_keys()
    results = []
    for clip in CLIPS:
        with wave.open(str(clip['wav']), 'rb') as r:
            assert r.getframerate() == 16000 and r.getnchannels() == 1
            r.setpos(int(clip['excerpt_ms'][0] * 16))
            raw = r.readframes(int((clip['excerpt_ms'][1] - clip['excerpt_ms'][0]) * 16))
        pcm = np.frombuffer(raw, dtype=np.int16)
        results.append({'case': clip['case'], 'excerpt_ms': clip['excerpt_ms'],
                        **await timed_pass(pcm, keys['SONIOX_API_KEY'])})
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'clips={len(results)}')


if __name__ == '__main__':
    asyncio.run(main())
