# Desktop overlay startup recovery

## Recovery admission

A failed desktop startup generation is automatically replaced IFF ALL hold:

1. Desktop target with the current authoritative generation.
2. Startup reached `BOUNDS_CONFIRMED` at least once (final observed bounds
   need NOT still be canonical).
3. Failure classifies as `window_reveal_lost` (hidden with canonical bounds)
   or `window_visibility_unstable` (post-bounds drift or intermittent
   visibility) with corroborated evidence.
4. Window identity is unambiguous and Win32 observation is error-free.
5. Generation A fully closes (`close_completed`) with every non-transfer
   resource reaped (child process, bridge, renderer events, tasks); only
   the inert preserved presenter and diagnostics may remain for transfer.
   Any uncertain close goes terminal.

All other failure classes (`window_identity_failed`,
`window_observation_failed`, `window_bounds_failed`,
`window_native_ready_failed`, unclassified `window_configuration_failed`,
spawn/bridge/manifest/crash failures) surface terminally without automatic
replacement.

## State contract

- The first generation strictly fails; its evidence and instance id stay
  diagnostic and are never rewritten.
- Intent stays ON through recovery. The application reports explicit
  `recovering`, shown as ON in the overlay contract with peer held in
  starting; the existing retry/reopen row stays hidden until a terminal
  outcome.
- The replacement starts strictly after A is reaped, outside A's startup
  task so the preserved presenter detaches exactly once with no overlapping
  authority, then runs the identical strict startup contract: canonical
  hidden bounds, show, visible canonical retention, READY.
- Exactly one immediate replacement per user enable/reopen session, no
  backoff sleep; a repeat failure goes terminal, visible and actionable
  through the existing retry/reopen row and view-details.
- Successful recovery records failed and replacement instance ids plus
  classification alongside the connection.
- Recovery preserves the decided desktop session and never reprobes SteamVR.
- Toggle OFF, rapid ON, and shutdown cancel and drain recovery; stale
  instance events are rejected; no child, PID file, bridge, presenter task,
  or output-sink leak.
- Unrelated native crash recovery policy is unchanged.

## Latency scope

- No speculative Flet preparation.
- Desktop fallback no longer waits on peer dependency refresh before
  starting; the post-connect refresh remains.
- Post-connect peer-refresh failure never kills a strictly connected
  fallback desktop and never leaves its monitor unowned; direct-path
  refresh failure keeps its existing terminal semantics.
- Only provably redundant teardown is skipped (completed-close proof);
  uncertain teardown still blocks start.
- No timer, retention, Win32 write, upstream, or fallback-policy changes.

## Failure copy

- `window_reveal_lost`: captions started but the window stayed hidden.
- `window_visibility_unstable`: captions appeared but did not stay visible.
- `window_identity_failed`: captions found an unexpected window.
- `window_observation_failed`: captions could not verify the window.
- `window_bounds_failed`: captions could not apply the window position.
- `window_native_ready_failed`: captions did not become ready.
- Reveal and drift failures reuse the reopen action; identity, observation,
  bounds, and native-ready failures reuse retry.

## Non-goals

- Fixing or forking Flet upstream.
- Changing the SteamVR fallback product policy.
- Topmost/z-order behavior, OVR reliability, subtitle layout/style.
- Retrying broad `window_configuration_failed` or any non-visibility class.

## Live verification

- Paired controlled runs with an injected 0.8 s peer refresh delay show
  the removed pre-start hop directly: baseline fallback first-owned-visible
  1.886 s / connected 2.494 s vs candidate 1.016 s / 1.625 s (about 0.87 s
  on both, matching the injected delay). The direct path is unchanged and
  the idle-path gain stays near zero.
- On supported Python 3.12 with deterministic reveal-loss injection, both
  direct and fallback sessions complete a strict failed-A to connected-B
  handoff with distinct instance ids, the first failure retained, the
  preserved presenter re-emitting captions on B, and first owned visibility
  bound to B's process and canonical bounds. The fallback session emits
  exactly one VR-to-desktop notice and never reprobes SteamVR.
- Toggle OFF during recovering drains cleanly with no child, PID file,
  bridge, or task leaks; identity mismatch stays terminal without
  attempting recovery. Focused suites pass on 3.12 with zero failures.

## Verification limits

- First-frame pixel rendering is not asserted; live proof is Win32
  visibility plus canonical bounds plus caption content bound to the
  owning process.
