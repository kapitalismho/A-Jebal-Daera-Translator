# Issue #138 receipt: Gemini Transcribe Live synchronous SDK setup off the audio loop

## Baseline

- Commit `46e37cf173660bcbaaecd911d82a6131eef7cbdd`, branch
  `gemini-transcribe-live-keep-synchronous-sdk-init`.
- Locked SDK `google-genai==2.21.0`, verified interpreter CPython 3.12.10 via `uv`
  (system Python 3.14 is unsupported by the project).
- No live credentials in the environment; live-network smoke was not attempted.

## SDK source evidence (2.21.0, offline inspection)

- `Client.__init__` performs no event-loop binding; the loop-sensitive surface is the
  async handshake (`client.aio.live.connect` async context manager) plus `aclose`.
- `Client.close()` closes only the synchronous client; its own docstring directs async
  cleanup to `Client.aio.aclose()`. `AsyncClient.aclose()` closes only the async side.
  Both closes are required (dual close), in that order.
- `types.HttpOptions` accepts `httpx_client` / `httpx_async_client`, and the SDK leaves
  caller-supplied transports open, so the adapter owns them explicitly and closes both
  around the client closes. Default transport/config behavior is otherwise unchanged.
- Transports are allocated before the `Client` constructor runs, so a later constructor
  failure still closes them (no GC fallback, no private SDK members).

## Change

`src/puripuly_heart/providers/stt/gemini_transcribe.py` (provider only):

- Synchronous preparation (imports, `LiveConnectConfig`, explicit `httpx` transports,
  `Client` construction) runs in a single-thread worker executor, never on the shared
  audio event loop. The worker thread is named `gemini-stt-setup`.
- The async handshake (`connect` enter), send/receive loops, context exit, and async
  close stay on the application loop.
- One strongly-held teardown task per session is the single teardown owner. `start`,
  `stop`, and `close` shield-and-join it, so repeated caller cancellation never aborts
  real cleanup; teardown completion is recorded before the public call returns, and a
  pending caller cancel is re-raised only after the drain.
- Setup results attach only to the exact owning session. Late setup completion after
  cancellation or replacement is reclaimed, never published to a retired session.
- Cancellation, factory failure, handshake timeout/refusal, late completion, repeated
  close/cancel, toggle-OFF, and replacement are all bounded and idempotent; partial
  constructor allocations are closed; original errors are preserved, never masked.
- Timing logs separate worker setup from the async handshake and mark readiness
  start/end. No transcript text, keys, or PCM appear in diagnostics.
- Protocol behavior (final/ACK, manual activity detection, per-turn finalize timeout,
  audio/config handling) is unchanged.

## Validation

Focused suites, all passing (`138 passed`):

- `tests/providers/test_gemini_transcribe_lifecycle.py` — offline dual-close of a real
  SDK `Client`, slow-setup ownership across two channels, handshake-cancel races,
  constructor-failure partial allocation, context-factory/enter failure and timeout,
  cancel-during-setup late completion, repeated close/cancel boundedness, toggle
  OFF/replacement owner generation, gated async-close survival under repeated caller
  cancel, close-waiting-for-gated-setup, no-exit-before-enter-finally, factory sync
  raise release, and a 5-round setup/cancel ready race with exactly-once release.
- `tests/providers/test_gemini_transcribe_ownership.py` — dual-channel ownership,
  replacement isolation, plus a real capture-owner test proving the peer channel's
  capture/decode/final path completes while this channel's setup is worker-gated.
- `tests/providers/test_gemini_transcribe_backend.py`,
  `tests/architecture/test_lifecycle_task_guard.py`, `tests/core/test_stt_controller.py`
  — existing protocol/audio/config coverage unchanged.
- `ruff check` and `black --check` clean on all touched files.
- Actual offline-path smoke (real SDK construction + dual close, no network): session
  ready wall 1156 ms cold, setup confirmed off-loop, both transports observed closed.

## Limits and follow-ups

- Common audio ingress is unchanged (#139 owns that boundary).
- The #126 finalization risk remains open and unresolved here.
- Live-network behavior is untested (no authorized credentials available).
