# Desktop overlay startup recovery

## Recovery admission

A failed desktop startup generation is automatically replaced IFF ALL hold:

1. Desktop target with the current authoritative generation, enabled intent,
   available settings, open ingress, and a failed startup; bounded to one
   immediate replacement per user enable/reopen session.
2. Failure reason is one of six window startup classes:
   `window_reveal_lost`, `window_visibility_unstable`,
   `window_observation_failed`, `window_bounds_failed`,
   `window_native_ready_failed`, `window_identity_failed`.
3. The current manager reports authoritative
   `startup_recovery_eligible`. The low-level classifier excludes canonical
   missing/invalid bounds, native-ready non-timeout failures, stale
   binding/closed/unbound generations, and broad init/manifest/spawn/auth/
   crash/unknown classes. The application trusts this bit and does not
   re-require title/HWND/endpoint/bounds forensics, exact cause
   explanation, or desktop-flag/evidence-type matching; rich
   `startup_failure_evidence` is retained as diagnostics even when sparse
   or empty.
4. Generation A fully closes with proven cleanup before B starts: the held
   prior manager reports `desktop_cleanup_complete` (same-instance
   shutdown_complete after inner Flet close AND confirmed outer process
   exit; unknown/forced close without ack stays false) PLUS the held
   runtime reports `close_completed` and `transfer_reap_complete` with
   every non-transfer resource reaped (child process, bridge, renderer
   events, tasks); only the inert preserved presenter and diagnostics may
   remain for transfer. The manager reference is held before teardown;
   failed proof goes terminal with no B, even with empty refs.

All other failure classes (unclassified `window_configuration_failed`,
spawn/bridge/manifest/crash/unknown, native-ready non-timeout, stale or
unbound generations) surface terminally without automatic replacement.

## State contract

- The first generation strictly fails; its evidence and instance id stay
  diagnostic and are never rewritten.
- Intent stays ON through recovery. The application reports explicit
  `recovering`, shown as ON in the overlay contract with peer held in
  starting; the existing retry/reopen row stays hidden until a terminal
  outcome.
- The replacement starts strictly after A is reaped and proven clean,
  outside A's startup task so the preserved presenter detaches exactly
  once with no overlapping authority, then runs the identical strict
  startup contract: canonical hidden bounds, show, visible canonical
  retention, READY.
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

## First-visible vs READY

- `desktop_first_visible` is an explicit UX-only field carried through
  `OverlayPeerPresentationState` into the overlay/peer consumer contract
  and dashboard capture presentation (default false for old constructors).
  It stops the desktop PowerButton spinner as soon as the trusted
  same-instance child `desktop_first_visible` lifecycle event is accepted
  while desktop is starting/recovering, before strict `connected`.
- `effective_enabled` remains strict `connected` only (overlay READY after
  retention). VR and monitor readiness are unaffected; content-ready empty
  captions remain permitted with no subtitle loading window.
- The application reads only the current runtime manager's
  `desktop_first_visible` and republishes presentation from a guarded
  `first_visible_callback` (runtime plus instance check). The replacement
  B starts with first-visible false so the recovering spinner returns;
  stale old visibility cannot stop B's spinner. Duplicate callbacks
  republish idempotently.
- Connected/failed/OFF hide the spinner as before, including the OSC
  Captions secondary ingress path which reuses the same capture
  presentation. No new buttons, strings, settings, timers, or runtime
  policy.

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
- Retrying broad `window_configuration_failed` or any non-window class.
- Treating first-visible as readiness: the spinner is a UX fallback only;
  strict READY still gates `connected`, monitor ownership, and peer refresh.
- Adding buttons, copy, settings, timers, or subtitle loading windows for
  startup.

## Live verification

- Direct success (real Flet, Python 3.12): first Win32 visibility 0.766 s
  (HWND PID-bound, title `PuriPuly Overlay`, canonical bounds, caption
  content) = manager first-visible 0.766 s → PowerButton spinner off
  0.797 s while `effective_enabled` still False → strict `connected`
  1.375 s. The 0.609 s first-visible→connected delta is the unchanged
  ~0.6 s visible-canonical retention, not readiness.
- Fallback success (genuine SteamVR-off): first-visible 1.125 s /
  manager 1.141 s → connected 1.75 s (delta 0.625 s, same retention).
  Exactly one VR-to-desktop notice, no SteamVR reprobe.
- Four window-subtype A→B recoveries, each with distinct instance ids,
  first failure retained, presenter-transferred captions on B, and B
  visibility bound to B's PID and canonical bounds:
  - native-ready: A `window_native_ready_failed` (`native_ready_timeout`,
    pre-bounds `page_configured`, sparse evidence with null HWND still
    eligible) → recovering 4.016 s → B Win32 4.906 s / manager 4.922 s /
    spinner-off 4.953 s (effective False) → connected 5.531 s.
  - observation: A `window_observation_failed` (`port_error`) →
    recovering 1.171 s → manager 2.031 s → connected 2.64 s.
  - bounds: A `window_bounds_failed` (`bounds_not_retained`, valid
    canonical) → recovering 1.094 s → connected 2.562 s. A first attempt
    whose B flaked `reveal_lost` went one-shot terminal with null
    replacement, proving the single-replacement bound; retry clean.
  - identity: A `window_identity_failed` (`pid_file_mismatch`, owner
    44092 vs pidfile 44093) → recovering 0.969 s → connected 2.625 s.
    Identity failure is recoverable when the manager deems it eligible;
    the old identity-always-terminal claim is superseded.
- Terminal proofs with no B and no leaks: canonical-missing/invalid
  (`window_bounds_failed`/`canonical_bounds_missing`, eligible False),
  unknown (`window_configuration_failed`, eligible False), and
  missing-ACK (`reveal_lost` eligible True but `desktop_cleanup_complete`
  False despite outer exit 1 — a cleared process ref is not proof) →
  recovering→failed terminal, null replacement.
- Toggle OFF drains: OFF during recovering → off in 63 ms with recovery
  cancelled before replacement; OFF during first-visible (first 0.828 s
  → OFF 0.875 s, spinner had stopped 0.859 s while starting) → off with
  no child, PID file, bridge, or task leaks. Per-scenario
  `*_first_visible` / `*_connected` screenshots retained in the Temp
  evidence dir.
- Focused suites pass on 3.12 with zero failures.

## Verification limits

- First-frame pixel rendering is not asserted; live proof is Win32
  visibility plus canonical bounds plus caption content bound to the
  owning process (screenshots retained per scenario, pixels not asserted).
- First-visible stops only the spinner via the `first_visible_callback`
  path; `connected`, monitor ownership, and peer refresh still gate on
  strict READY after retention. The two are measured separately above
  and must not be conflated.
