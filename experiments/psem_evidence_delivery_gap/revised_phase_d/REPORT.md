# Revised Phase D probe — REPORT (#133, supersedes P1 merge scope only)

Correction: the first issue overstated provider drain (5-7 s) from a
collection-elapsed field, overstated H as equal to ideal, and understated
the 9-session methodology. Corrected below from stored walls; results.json
and freeze.json unchanged. Derived values in `derived_metrics.json`.

## Identity

EN2009d Mix-Headset excerpt [3248200, 3272600] ms. Target VAD utterance
[3257908, 3265416] ms (7508 ms, max_duration, rederived from actual SpeechStart
PCM payload). GT turn T = 3259811.5 ms (sample 52156984): A/FEE083 ends
3259815, C/MEE094 starts 3259808, 7 ms overlap holds no word interiors.
V2 handoff [3260300, 3260538] rejected as truth. Later B/A/C overlap from
3262688 exposed, never scored. Freeze `freeze.json`
(psem.revised_phase_d.freeze.v1, 2026-09-08T12:30:17Z) predates paid run
(`results.json` run 12:31:06Z, 0 errors). Script `run.py`, no comments.
One clip, 9 Soniox stt-rt-v5 sessions, 0 translations, 0 OpenRouter calls.
Uniform 1500 ms BEFORE-SEND hold on all arms. Scoped logical STT utterance
delivery only; no UI exercised; explicit real-adapter driver (not full runtime
wiring) with manual finalize per segment, new utterance ID per finalization,
FIFO segment-ordered finals.

Methodology limit: arms used pre-opened per-segment provider sessions
(1/2/3/3 = 9 total, matching the frozen 9-session budget). Single-session
production behavior is untested; nothing here asserts the ordinary production
route end-to-end.

## Observed finals (exact)

Baseline (1 final):
"Right. How could you possibly do that? You have to hope that there's
language in one of them say something. But that's a human coding. Mm-hm.
Right. So they'"

Ideal T-split (2 finals):
1. "Right. How could you possibly"
2. "You have to hope that there's language in one of them say something.
But that's a human coding. Mm-hm. Right. So they'"

F0 splits at 3258700 / 3262700 (3 finals):
1. "R"
2. "How could you possibly do that? You have to hope that there's language
in one of them say something."
3. "But that's a human coding, right? Yep, so they'"

H splits at 3259900 / 3262700 (3 finals):
1. "Right. How could you possibly do"
2. "You have to hope that there's language in one of them say something."
3. "But that's a human coding, right? Yep, so they'"

GT anchors: A "...right. How could you possibly do that?" (ends 3259.77);
C "You have to hope that there's language and one of them says something."
(starts 3259.83). Provider renders "and one of them says" as "in one of
them say" in every arm: uniform ASR substitution, not a split effect.

## Answers

1. Baseline mixes: YES. One final holds the complete disjoint A-question and
C-answer with no owner distinction: sequential cross-speaker merge inside one
delivered logical utterance, on a span the ordinary path does not separate.
2. Ideal recovers attribution (A-tail final 1, C-head final 2, no duplicated
words in the scored phrases) at the cost of "do that?" (2 words, fully
before T) missing from final 1. Cause of the omission unproven: boundary
context vs finalize truncation remain open hypotheses.
3. H vs F0 vs ideal: H lowers observed mixed-phrase delivery relative to F0
(H final 2 is C-only; F0 final 2 still merges A+C; F0 final 1 is destroyed
to "R" by a mid-word cut inside A "right"). H costs "that?" (1 word). H is
NOT equal to ideal overall: H carries an extra secondary split at 3262700
inside the later overlap region, where ideal has one continuous final.
4. Omission/dup/fragmentation/delay: no duplicated words in the scored A/C
phrases; tail overlap unscored. Omitted: ideal "do that?", H "that?", F0
"Right." (fragmented to "R"). Fragmentation: the first H/ideal boundary
reduces the target merge; the shared secondary split effect is unresolved,
and not all F0 extra splits are pure harm either. Delay: +1500 ms designed
hold all arms; session opens 0.58-0.75 s setup; per-segment final arrival
minus own finalize 0.25-0.33 s on every segment (baseline 0.328; ideal
0.297/0.297; F0 0.297/0.281/0.265; H 0.250/0.281/0.312). The earlier 5-7 s
figures were collection elapsed after the whole stream, not provider
latency; latency is not the demonstrated issue. Arm walls 11.39/13.20/
14.81/14.78 s. Queue observed max 48 over finite 235-chunk input; 128 was a
stall threshold, no hard bound enforced, 0 stalls. Forward accounting
sample-exact, all 5/5 requests applied within-unsent window, secondary
3262700 applied in both F0/H with no silent drop; the secondary split sits
in the later overlap region so its consequence is unresolved, not neutral.
5. Earliest supported diagnostic loss: the merge above is now MEASURED on one
clip (was UNKNOWN in P1). Continued bounded ambiguity: tail overlap
attribution; omission mechanism at forced cuts; cost acceptability (no
product delay budget approved).

## One next decision

One bounded content-preserving boundary application check: investigate the
observed tail omission ("do that?" / "that?") before claiming usable
recovery, under the stated 1.5 s experimental hold. Not latency/buffer
sweeps, not model repair. No approved product delay budget, no broad
promotion. The positive narrow result stands: H delivers the C phrase
without the A merge where F0 does not, at one missing word of known-unknown
cause.

## Standing and limits

P1 branch F is preserved as historical and still governs PSEM authority (this
probe grants none; the receiver was experiment-local). The revised result
supersedes only the bounded "unmeasured merge" scope. Correct 100 ms
(1600-sample) reconstruction matched the frozen design exactly: no historical
metric change to flag. Fixed GT-derived reference trajectory with no native
rebind is a carried diagnostic limitation. Ideal Te is declared synthetic
timing, not measured execution. Buffered baseline shifts the original
schedule by the hold; segmentation equivalence is theoretical.

## Reproduce

```text
uv run python experiments/psem_evidence_delivery_gap/revised_phase_d/run.py
```

Requires SONIOX_API_KEY in .env.local. Keys never printed. GT words lived in
OS temp only (provenance + hashes in freeze.json), never committed.
Corrected fields (`final_latency_s`, `collection_elapsed_s`, finally-bound
session close) are in the retained script; stored paid results recomputed
offline in `derived_metrics.json` with no new calls.
