# §11 bounded actionability probe (Child #133) — addendum

Executed the only-authorized measurement in bounded form. Verdict: **retain F**.
VAD openness was observed on every requested boundary; it proves no provider
applicability, so no repair branch is selected and no further timing or
provider research follows.

## Scope and identity

Same G04/G05 source audio only, production peer VAD profile (threshold 0.5,
ring 500 ms, hangover 500 ms, pre-roll 500 ms, max 7000 ms, debounce/commit 3,
512-sample chunks) through `SileroVadOnnx` + `create_peer_vad_gating`, the
`realcheck.py` pattern. Windows: G04 excerpt [559000, 580000],
G05 excerpt [372000, 393000] (both identical to `realcheck.json`), plus
G05-F0-leading [360000, 376000]: same EN2009d Mix-Headset source, bounded
leading context for the F0 frontier only, not a new clip. Requested
boundaries are the canonical first-emit H/F0 boundaries from
`traces/emission_bounds.json`; the retrospective GT splits are contrast only.
Baseline `45956c69`, branch `experiment-v2-speaker-change-turn-boundaries-ls`.
Script `analysis/actionability.py`, trace `traces/actionability.json`,
run 2026-09-08T11:42:13Z. Zero paid calls, no network inference, no new
models, no production edits, no mocks.

## Observations

| Case | Requested | Earliest frontier | GT contrast | Position | Verdict | Payload margin | Emission margin |
| ---- | --------- | ----------------- | ----------- | -------- | ------- | -------------- | --------------- |
| G04 H/F0 | 568600.0 | 569362.5 | 569794.0 | inside-payload | OPEN | 1477.5 ms | 1477.5 ms |
| G05 H | 379700.0 | 380722.5 | 383194.0 | inside-payload | OPEN | 333.5 ms | 845.5 ms |
| G05 F0 | 367800.0 | 368722.5 | 383194.0 | inside-payload | OPEN | 5517.5 ms | 5517.5 ms |

All times ms. Owners: G04 `2729e448` (max_duration), G05-H `140651ab`
(silence, 512 ms trim), G05-F0 `f076121d` (max_duration). SpeechEnd emission
frontier recorded separately from trimmed payload end; pre-roll overlap kept
as contextual single-segment ownership, never double-counted. Excerpt reruns
reproduce `realcheck.json` spans exactly (0-sample diff, both cases).

## Not measured

No source-boundary request/ack receiver exists in current Audio, so provider
applicability is unshowable: unsupported acknowledgement, NOT a measured
rejection. Earliest frontiers are offline first-event lower bounds, never
observed wallclock arrival. VAD openness is not provider mutability or
commit; no commit or latency credit taken. Excerpt-local replay starts VAD
cold at the window start while production runs continuously.

## Next prerequisite

Minimum next step is an Audio-owned application semantics decision: a
source-time applied-boundary receiver with commit/ack reporting (applied
position, application time, or reason-not-applied) on the same support
scope, plus stated provider commit semantics. Not a broad migration, model
repair, or any P3 work.
