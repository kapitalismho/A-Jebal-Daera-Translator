#!/usr/bin/env python3
"""V2 NeoVAD recurrent warm-up checks (no pytest; runnable via python).

Torch-free mock-state-path proof: bind() must drive the FULL reference
PCM chronologically through the per-frame model-step path (one call per
frame, in order, outputs discarded, warmed state kept) while the eval
clock stays anchored (``_source_time_ms`` untouched, no eval frames).
O (5 s) must warm strictly more frames than C (1 s) through identical
weights/reset/path; warmed-state fingerprint must differ from reset-state.

The real-checkpoint path is attempted when torch loads; otherwise it is
reported SKIP (mock proof stands) — never FAIL.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.psem_small_model_probe.adapter.neovad import NeoVADAdapter  # noqa: E402
from experiments.psem_small_model_probe.adapter.protocol import (  # noqa: E402
    BindingError,
    frame_bytes,
)

FRAME_MS = 10
UNIT = frame_bytes(FRAME_MS)  # 320 bytes


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS {name}" + (f" — {detail}" if detail else ""))


class FakeState:
    def __init__(self):
        self.blob = bytearray(b"\x00" * 8)


class FakeModel:
    """Minimal mock of the recurrent model step path."""

    def __init__(self):
        self.calls: list[bytes] = []

    def init_state(self, *args):
        return FakeState()

    def step(self, chunk, state):
        self.calls.append(bytes(chunk))
        state.blob.append(0x01)
        return None


def install_mock(adapter):
    fake = FakeModel()
    adapter._model = fake
    adapter._fresh_state = lambda model: model.init_state()  # noqa: E731 torch-free

    def fake_raw_probs(chunk):
        fake.step(bytes(chunk), adapter._state)
        return (1.0, 0.0, 0.0)

    adapter._raw_probs = fake_raw_probs
    return fake


def synth_span(ms):
    n = ms * 16 * 2
    return bytes(i % 256 for i in range(n))


def test_contract_preserved():
    a = NeoVADAdapter()
    try:
        a.bind(b"\x00" * UNIT)
        check("warmup/bind-before-reset", False, "no RuntimeError")
    except RuntimeError:
        check("warmup/bind-before-reset", True, "RuntimeError")
    a.reset()
    for label, blob in (("empty", b""), ("short", b"\x00" * UNIT)):
        try:
            a.bind(blob)
            check(f"warmup/bind-{label}", False, "no BindingError")
        except BindingError:
            check(f"warmup/bind-{label}", True, "BindingError")
    try:
        a.bind(b"\x00" * (UNIT + 1))
        check("warmup/bind-nonmultiple", False, "no ValueError")
    except ValueError:
        check("warmup/bind-nonmultiple", True, "ValueError")


def test_o_span():
    a = NeoVADAdapter()
    ref = synth_span(5000)
    a.reset()
    fake = install_mock(a)
    a.bind(ref)
    check("warmup/O-model-steps", len(fake.calls) == 500, f"{len(fake.calls)}")
    check("warmup/O-chronological",
          all(c == ref[i * UNIT:(i + 1) * UNIT] for i, c in enumerate(fake.calls)))
    check("warmup/O-clock-anchored", a._source_time_ms == 0 and a.frames == [])
    h = a.episode_header()
    check("warmup/O-header",
          h["bind_span_hash"] == hashlib.sha256(ref).hexdigest()
          and h["bind_span_ms"] == 5000
          and h["warmup_frames"] == 500
          and h["stub_fallback"] is False
          and h["reset_ok"] is True
          and "VOID" in h["prior_neovad_rows_void"],
          f"span_ms={h['bind_span_ms']} warmup={h['warmup_frames']}")
    return a, h


def test_c_span():
    a = NeoVADAdapter()
    ref = synth_span(1000)
    a.reset()
    fake = install_mock(a)
    a.bind(ref)
    check("warmup/C-model-steps", len(fake.calls) == 100, f"{len(fake.calls)}")
    check("warmup/C-clock-anchored", a._source_time_ms == 0 and a.frames == [])
    h = a.episode_header()
    check("warmup/C-header", h["bind_span_ms"] == 1000 and h["warmup_frames"] == 100)
    return a, h


def test_identical_path_o_vs_c():
    a = NeoVADAdapter()
    sha = a.model_sha
    a.reset()
    fake_o = install_mock(a)
    a.bind(synth_span(5000))
    fp_o = a.state_fingerprint()
    a.reset()
    check("warmup/reset-clears",
          a.state_fingerprint() is None and a.warmup_frames == 0
          and a.bind_span_hash is None and a.bind_span_ms is None
          and a._bound is False)
    fake_c = install_mock(a)
    a.bind(synth_span(1000))
    fp_c = a.state_fingerprint()
    check("warmup/same-weights", a.model_sha == sha)
    check("warmup/O-gt-C", len(fake_o.calls) > len(fake_c.calls),
          f"O={len(fake_o.calls)} C={len(fake_c.calls)}")
    check("warmup/warmed-ne-reset", fp_o is not None and fp_c is not None
          and fp_o != fp_c, f"O={fp_o[:12] if fp_o else None} C={fp_c[:12] if fp_c else None}")
    # Fresh-state fingerprint differs from any warmed fingerprint.
    fresh_state_holder = a._fresh_state(a._model)
    a._state, warmed = fresh_state_holder, a._state
    fp_fresh = a.state_fingerprint()
    a._state = warmed
    check("warmup/fresh-ne-warmed", fp_fresh != fp_o and fp_fresh != fp_c,
          f"fresh={fp_fresh[:12] if fp_fresh else None}")
    # Re-bind reproducibility: same span -> same warmed fingerprint.
    a.reset()
    install_mock(a)
    a.bind(synth_span(5000))
    check("warmup/reproducible", a.state_fingerprint() == fp_o)


def test_eval_clock_after_warmup():
    a = NeoVADAdapter()
    a.reset()
    install_mock(a)
    a.bind(synth_span(5000))
    out = a.step(b"\x00" * UNIT)
    check("warmup/eval-anchored", out.source_time_ms == FRAME_MS
          and a.frames[0]["source_time_ms"] == FRAME_MS
          and len(a.frames) == 1,
          f"first eval frame t={out.source_time_ms}ms (warm-up 5000ms leaked? no)")


def test_real_path_attempt():
    try:
        a = NeoVADAdapter()
    except Exception as exc:
        check("warmup/real-init", False, repr(exc)[:120])
        return
    a.reset()
    try:
        a.bind(synth_span(1000))
    except RuntimeError as exc:
        if "torch" in str(exc):
            print("SKIP warmup/real-bind — no torch in this env; "
                  "mock-state-path proof stands")
            return
        raise
    h = a.episode_header()
    fp_w = a.state_fingerprint()
    a.reset()
    fp_r = a.state_fingerprint()
    check("warmup/real-bind",
          h["warmup_frames"] == 100 and h["bind_span_ms"] == 1000
          and fp_w is not None and fp_w != fp_r)


def main():
    test_contract_preserved()
    test_o_span()
    test_c_span()
    test_identical_path_o_vs_c()
    test_eval_clock_after_warmup()
    test_real_path_attempt()
    print("ALL NEOVAD-WARMUP TESTS PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"FAIL {exc}")
        sys.exit(1)
