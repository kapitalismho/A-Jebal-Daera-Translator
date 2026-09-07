from __future__ import annotations

"""NeoVAD GRU adapter. The recurrent hidden state is short-term acoustic memory
only: it does not enroll or lock first-speaker identity, and reset() clears it,
so no speaker persists across episodes. No hysteresis gate is applied here;
step() returns raw p(nonspeech)/p(primary)/p(secondary)."""

import hashlib
from pathlib import Path

from experiments.psem_small_model_probe.adapter.protocol import (
    BindingError,
    StepOut,
    frame_bytes,
    validate_pcm16_chunk,
)

_VENDOR = Path(__file__).resolve().parent / "vendor"
_WEIGHTS_NAME = "neovad_gru.pt"


def _vad_model_cls():
    """Return upstream VADModel, or None if the neovad package is absent.

    This adapter module is itself named ``neovad``, so when the adapter
    directory sits on ``sys.path`` (running adapter scripts directly puts the
    script's dir there) ``import neovad`` would resolve to this file instead
    of the dependency. Temporarily drop our directory from the path — and any
    shadow entry it already caused — then restore everything.
    """
    import sys

    here = Path(__file__).resolve().parent
    saved_path = sys.path[:]
    saved_mod = sys.modules.pop("neovad", None)
    try:
        sys.path = [p for p in saved_path if Path(p or ".").resolve() != here]
        sys.modules.pop("neovad", None)
        from neovad.models.vad import VADModel

        return VADModel
    except ImportError:
        return None
    finally:
        sys.path[:] = saved_path
        if "neovad" not in sys.modules and saved_mod is not None:
            sys.modules["neovad"] = saved_mod


class NeoVADAdapter:
    sample_rate_hz = 16000

    def __init__(
        self,
        frame_ms: int = 10,
        min_bind_ms: int = 1000,
        weights_dir: str | Path | None = None,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        wdir = Path(weights_dir) if weights_dir is not None else _VENDOR
        self.weights_path = wdir / _WEIGHTS_NAME
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"missing weights {self.weights_path}; see adapter/vendor.json "
                "(artifact neovad_gru_pt) for the pinned URL"
            )
        self.frame_ms = frame_ms
        self.min_bind_ms = min_bind_ms
        self.model_sha = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()
        self._model = None
        self._state = None
        self._reset_called = False
        self._bound = False
        self._source_time_ms = 0
        self.bind_span_hash: str | None = None
        self.bind_span_ms: int | None = None
        self.warmup_frames = 0
        self.frames: list[dict] = []
        self.reset()
        self._reset_called = False

    def _model_or_raise(self):
        if self._model is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "NeoVADAdapter.step() needs torch (CPU) to run "
                    f"{self.weights_path.name}; install it, weights are vendored"
                ) from exc
            VADModel = _vad_model_cls()
            if VADModel is not None:
                # Vendored gru.pt is a portable {config, state_dict} checkpoint.
                self._model = VADModel.load(str(self.weights_path), map_location="cpu")
            else:
                # Legacy fallback: full-pickled module (pre-portable checkpoints).
                self._model = torch.load(
                    str(self.weights_path), map_location="cpu", weights_only=False
                )
            self._model.eval()
        if self._state is None:
            # reset() clears recurrent memory across episodes; lazily rebuild a
            # fresh state here so episode 2+ does not step with state=None.
            self._state = self._fresh_state(self._model)
        return self._model

    def _fresh_state(self, model):
        """One fresh recurrent state for model (post-reset start)."""
        import torch

        return model.init_state(1, "cpu", torch.float32)

    def reset(self) -> None:
        self._state = None
        self._reset_called = True
        self._bound = False
        self._source_time_ms = 0
        self.bind_span_hash = None
        self.bind_span_ms = None
        self.warmup_frames = 0
        self.frames = []

    def _warmup_frame(self, chunk: bytes) -> None:
        """One warm-up step through the native streaming path.

        Outputs are discarded; the eval clock (``_source_time_ms``) and the
        eval frame log (``self.frames``) are untouched, so evaluation stays
        anchored at evaluation_start. Kept separate from step() so warm-up
        timestamps can never leak into eval frames.
        """
        self._raw_probs(chunk)

    def bind(self, reference_pcm16: bytes) -> None:
        if not self._reset_called:
            raise RuntimeError("bind() requires reset() first")
        unit = frame_bytes(self.frame_ms, self.sample_rate_hz)
        if len(reference_pcm16) == 0 or len(reference_pcm16) < self.min_bind_ms * 16 * 2:
            raise BindingError(
                f"reference span too short: {len(reference_pcm16)} bytes "
                f"(need >= {self.min_bind_ms} ms mono 16 kHz int16 LE)"
            )
        if len(reference_pcm16) % unit != 0:
            raise ValueError("reference span must be a frame multiple")
        # Recurrent warm-up: fresh post-reset state, then the FULL reference
        # PCM chronologically through the native streaming step path
        # (identical weights/reset/path for O and C spans; only the
        # reference audio differs). Logits discarded; final state kept.
        # Fail-closed without a loadable model: silent unwarmed bind is the
        # rev1/rev2 O≈C bug, so _model_or_raise() must succeed here.
        self._model_or_raise()
        self._state = None
        self._model_or_raise()  # lazily rebuilds exactly one fresh state
        clock_before = self._source_time_ms
        n = 0
        for i in range(len(reference_pcm16) // unit):
            self._warmup_frame(reference_pcm16[i * unit:(i + 1) * unit])
            n += 1
        if self._source_time_ms != clock_before or self.frames:
            raise RuntimeError("warm-up leaked into eval clock/frame log")
        self.bind_span_hash = hashlib.sha256(reference_pcm16).hexdigest()
        self.bind_span_ms = len(reference_pcm16) // (2 * self.sample_rate_hz // 1000)
        self.warmup_frames = n
        self._bound = True

    def state_fingerprint(self) -> str | None:
        """Stable hash of the recurrent state (None when no state).

        Duck-typed (torch/numpy/bytes-like) so fingerprints work without
        importing torch: used by V2 checks to prove warmed-state !=
        reset-state.
        """
        if self._state is None:
            return None
        h = hashlib.sha256()

        def feed(o) -> None:
            if isinstance(o, (bytes, bytearray, memoryview)):
                h.update(bytes(o))
                return
            numpy = getattr(o, "numpy", None)
            if callable(numpy):
                try:
                    arr = numpy()
                    h.update(str(getattr(arr, "shape", "")).encode())
                    h.update(arr.tobytes())
                    return
                except Exception:
                    pass
            tobytes = getattr(o, "tobytes", None)
            if callable(tobytes):
                try:
                    h.update(tobytes())
                    return
                except Exception:
                    pass
            try:
                namespace = vars(o)
            except TypeError:
                h.update(repr(o).encode())
            else:
                walk(namespace)

        def walk(o) -> None:
            if isinstance(o, (list, tuple)):
                for x in o:
                    walk(x)
            elif isinstance(o, dict):
                for k in sorted(o, key=repr):
                    walk(k)
                    walk(o[k])
            else:
                feed(o)

        walk(self._state)
        return h.hexdigest()

    def _raw_probs(self, chunk: bytes) -> tuple[float, float, float]:
        model = self._model_or_raise()
        import torch
        pcm = torch.frombuffer(bytearray(chunk), dtype=torch.int16)
        x = (pcm.to(torch.float32) / 32768.0).unsqueeze(0)
        with torch.no_grad():
            logits = model.step(x, self._state)
            probs = torch.softmax(logits.reshape(-1, 3), dim=-1)[-1]
        p_non, p_pri, p_sec = (float(v) for v in probs.tolist())
        assert abs((p_non + p_pri + p_sec) - 1.0) < 1e-4
        return p_non, p_pri, p_sec

    def step(self, pcm16_chunk: bytes) -> StepOut:
        if not self._bound:
            raise RuntimeError("step() requires bind() first")
        n = validate_pcm16_chunk(
            pcm16_chunk, frame_ms=self.frame_ms, sample_rate_hz=self.sample_rate_hz
        )
        prev = self._source_time_ms
        out = None
        unit = frame_bytes(self.frame_ms, self.sample_rate_hz)
        for i in range(n):
            p_non, p_pri, p_sec = self._raw_probs(
                pcm16_chunk[i * unit:(i + 1) * unit]
            )
            t = prev + (i + 1) * self.frame_ms
            aux = {
                "p_nonspeech": p_non,
                "p_primary": p_pri,
                "p_secondary": p_sec,
            }
            self.frames.append(
                {
                    "source_time_ms": t,
                    "speech": 1.0 - p_non,
                    "anchor": p_pri,
                    "aux": aux,
                }
            )
            out = StepOut(
                speech=1.0 - p_non, anchor=p_pri, aux=aux, source_time_ms=t
            )
        assert out is not None and out.source_time_ms > prev
        self._source_time_ms = out.source_time_ms
        return out

    def episode_header(self) -> dict:
        return {
            "model": "neovad-gru",
            "model_sha": self.model_sha,
            "onnx_sha": "none",
            "ecapa_sha": "none",
            "reset_ok": self._reset_called,
            "bind_span_hash": self.bind_span_hash,
            "bind_span_ms": self.bind_span_ms,
            "warmup_frames": self.warmup_frames,
            "stub_fallback": False,
            "prior_neovad_rows_void": (
                "rev1/rev2 NeoVAD O≈C rows VOID for branch conclusions: "
                "bind() did not warm recurrent state, so O and C spans "
                "shared the same reset-state start"
            ),
        }
