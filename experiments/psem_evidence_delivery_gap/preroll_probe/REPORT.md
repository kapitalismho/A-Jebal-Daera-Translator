# Preroll probe — REPORT (#133, second-segment context sensitivity only)

## Identity

EN2009d Mix-Headset, GT turn T = 3259811.5 ms (sample 52156984), span end
3265416 ms (sample 52246656). Freeze `freeze.json`
(psem.preroll_probe.freeze.v1, 2026-09-08T12:53:16Z) predates paid run
(`results.json` run 2026-09-08T12:54:02Z, 0 errors). Script `run.py`, no
comments. Two fresh cold Soniox stt-rt-v5 sessions (`language_hints ["en"]`,
512-sample source-paced chunks, uniform 1500 ms BEFORE-SEND hold both arms,
manual finalize exactly once per arm, source/wallclock distinct). One
segment per arm, no boundary injected at runtime; ideal Te 52175224 kept as
historical timing reference only. 0 translations, 0 OpenRouter calls, 0
Cerebras calls. Baseline HEAD `45956c69`
(branch `experiment-v2-speaker-change-turn-boundaries-ls`).

## Inputs

C0 `[52156984, 52246656)` = 89672 samples (5.6045 s, owned segment only).
C1 `[52148984, 52246656)` = 97672 samples (6.1045 s; first 8000 samples =
500 ms preroll context, previously A-owned). Owned `[S, end)` verified byte-identical offline before any paid call.
Historical first ideal segment `[52126528, S)` final `"Right. How could you possibly"` is REUSED from
revised Phase D, NOT rerun; the stored text is unchanged (it ends at and contains `possibly`; `do/that`
absent from it) and no paired new first transcript is claimed here.

## Observed RAW finals (exact, unmerged)

C0:
"You have to hope that there's language in one of them say something. But
that's a human coding. Right. So they'"

C1:
"We do that. You have to hope that there's language in one of them say
something. But that's a human coding. Right. Yep, so they'"

Each arm received exactly one final event, post-finalize, with zero
pre-finalize finals and zero partials (adapter emits finals only; full
received-final lists in `results.json`). Selection rule (frozen): last
non-empty post-finalize final; here first == last.

Correction (2026-09-08): the first draft of this report stated the pair would carry `do that` twice.
Withdrawn. Phrase membership in the stored outputs shows the reused first final contains no `do/that`
and C1 contains it once; across the pair it appears exactly once, in the wrong-owner segment.

## Answers

1. Omitted `do/that` reappears? YES, once across the historical pair. C1 heads with "We
do that." — `do that` are the last two pre-T A words (GT ends 3259.77 <
T 3259.8115), lexically recovered into the second segment. The reused first final contains `possibly`
but not `do/that`, so across the pair `do that` appears exactly once. `possibly` was never missing
from the first final; C1's leading "We" is at most a clipped-`possibly` substitution [INFERENCE: preroll
starts mid-`possibly` at 3259.31], not a recovery event.
2. C head preserved? YES both arms: "You have to hope that there's
language in one of them say something." verbatim, matching the historical
ideal second final head (uniform "in one of them say" ASR substitution
for GT "and one of them says" carries over, not a split effect).
3. The recovered words sit in the wrong owner under assigned-C: if the entire second final is assigned C, pre-T A-owned
`do that` is delivered inside C-attributed output. The 500 ms preroll is the sole input difference between arms;
the emission mechanism is unproven at n=1, but the delivered words are not duplicated across the pair — they appear once.
Leaving that span in the C-assigned final is misattribution; re-owning it to A is untested here because no text
partition/dedup/overlap merge was allowed in this probe. Pre-run hypothesis (freeze decision Q3) was framed as
exposure/duplication; the duplication half is refuted by the stored texts while the misownership half stands as
measured only conditional on assigned-C, not as tested runtime ownership. The hypothesis remains labeled hypothesis,
not a measured mechanism.

## Cost and timing

Per-arm actual final arrival minus own finalize: C0 0.313 s, C1 0.297 s
(matches revised Phase D 0.25-0.33 s band). Session opens: C0 0.782 s, C1
0.703 s. Arm walls: C0 12.454 s, C1 12.89 s. `collection_elapsed_s`
~3.3 s both arms is the frozen 3 s post-final quiescence drain plus poll
granularity, NOT provider latency. Queue max depth 48, stall threshold
128, 0 stalls; forward accounting gap/dup-ok both arms (176/191 input
chunks). Source cost: 2 sessions, (89672 + 97672)/16000 = 11.709 s audio,
0 translation. The extra 500 ms C1 input is reported cost only, aligned
to no deadline.

## Instability (n = 1 per arm, no invariant claims)

Tail renders differ run-to-run: historical ideal second final "Mm-hm.
Right. So they'", C0 "Right. So they'", C1 "Right. Yep, so they'". C1
"Yep" vs C0 absence is observed single-sample variation; no cause
claimed. Single-sample heads ("We do that.") likewise carry no
generality claim.

## Standing and limits

Second-segment context sensitivity only; no runtime ownership migration
was tested and none is claimed. Single-session-per-arm cold starts;
production multi-segment session behavior untested. Results immutable in
`results.json`. No thresholds, no guard search, no text-partition work
(separate owner later).

## Reproduce

```text
uv run python experiments/psem_evidence_delivery_gap/preroll_probe/run.py
```

Requires SONIOX_API_KEY in .env.local. Keys never printed. GT words
reused from `revised_phase_d/freeze.json`; no new dataset read.
