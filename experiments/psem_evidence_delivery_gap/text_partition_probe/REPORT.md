# Text-partition probe — REPORT (#133, separate from preroll)

## Identity

Target VAD payload [52126528, 52246656] (7.508 s, sha `6dce52e8…`), EN2009d
Mix-Headset, same span as revised Phase D. Freeze
`text_partition_probe/freeze.json` (`psem.text_partition_probe.freeze.v1`,
2026-09-08T15:54:46Z) predates all provider calls. Script `run.py`, no
comments. Baseline HEAD `45956c69` preserved; preroll and revised artifacts
untouched. Budget exception: frozen budget was 1 provider session;
2 were used because the first streamed fully but its raw evidence was lost
to a pre-observation implementation crash in the offline grouper
(overlapping ranges) before any result was written or observed; the grouper
was fixed, verified offline on synthetic text, and exactly one completing
run was scored. No result-driven rerun; freeze unchanged. 0 translations, 0
OpenRouter calls. Crash-safe raw dump (`raw_session.tmp.json`) now lands
before any assert and is removed on success.

## Method

One continuous session, `language_hints=["en"]`, 512-sample realtime-paced
chunks, no hold, no split, no restart. Manual `on_speech_end` once after the
full payload; live adapter pad inspected exact: 100 ms / 1600 samples.
Experiment-local class-level tee captured 1046 raw token records (992
interim + 54 final, every token record with `start_ms`/`end_ms`/`text`/
`is_final`; token dicts carried no other fields, nothing secret; the 2
tokenless messages were not field-audited) plus the accepted
token snapshot before flush cleared state. Drain: 60 s bound, 3 s
quiescence; every post-finalize final retained (`nfinal_post=1`,
`pre_finalize_finals=[]`, no interim final events, 0 empty acks, 1 `<fin>`
control token). `final_latency_s=0.281` is event-minus-finalize, not
collection (`collection_elapsed_s=3.328` is quiescence wait; open 0.641 s;
wall 12.688 s). Accepted stream: 50 tokens, 1 emit, no normalization strip
this run, event text equals token concat exactly. Accepted stream ends
`…they'`: the 54 raw final records reduce to 50 accepted because 3
same-`end_ms` 7200 tail tokens (`re`, ` just`, `.`) are dropped by the
live adapter monotonic-end guard before `<fin>` — observed adapter
behavior, flagged as a limitation, not repaired here. Partition
conservation covers the accepted final only, not complete raw provider
text or GT. Source map is provider-ms
to `p0 + ms*16`, estimate only, no rescale.

## Observed final (exact, single continuous session)

"Right. How could you possibly do that? You have to hope that there's
language in one of them say something. But that's a human coding. Mm-hm.
Right. So they'"

Lexically complete on the scored anchors: `do`/`that` present (contrast with
the revised split-arm omissions — noted as an uncontrolled observation, not
a controlled claim; methods differ). Uniform ASR substitution `in one of
them say` for GT `and one of them says`, same as every revised arm: ASR
fidelity, not a split effect. Genuine repetition `Right.` twice retained, no
dedup anywhere.

## Partition (29 whitespace groups, exact conservation all four variants)

Whole-word right-end rule over estimated group ends; starts are chained
previous-end estimates, not measured raw starts, so ambiguous means the
estimated chained interval crosses a boundary, not true word overlap.
Straddlers never split; no unresolved groups (all ends present).

- Baseline (no split): whole final, one fragment. Merge stands.
- Ideal S=52156984: left `Right. How could you possibly do ` / right
  `that? You have … they'`. `that?` (group 6, end 52157248) lands 264
  samples (~16.5 ms) past S under the frozen right-end rule plus provider
  overshoot vs GT (`that` ends 52156320, 664 before S), classified
  word-timing-misalignment (~58 ms). An alternative assignment rule could
  place it differently; not tuned here. No exact
  speaker-purity claim; projected ownership follows provider timing.
- F0 [52139200, 52203200]: left `Right. ` (group 0 ends 52139200−192, a
  12 ms margin — not robust against the observed ~58 ms timing error) /
  mid `How could … something. ` (group 1 estimated interval straddles b1,
  ambiguous) / right `But that's … they'` (group 20 estimated interval
  straddles shared b2, ambiguous). A-question fragmented, A-tail merged
  with C-head in mid.
- H [52158400, 52203200]: left `Right. How could you possibly do that? `
  (group 6 ends 1152 before b1, clean) / mid `You have … something. `
  (group 7 estimated interval straddles b1, ambiguous, right by rule) /
  right same tail (group 20 estimated interval straddles b2, ambiguous).
  First boundary separates A/C phrases at text level with zero
  dropped/duplicated/split characters — projected separation only, no
  acoustic purity or recovery guarantee.

GT retention: all 11 anchor words present. Ownership vs known S side —
`do`: ideal/H agree left, F0 mid disagree; `that`: H agree left, ideal/F0
disagree (timing, see above); C-head `You`/`have`/`to`/`hope` (groups
7–10): right of the first boundary in all split variants (ideal right
fragment; F0/H middle fragment). Later overlap
from 3262688 kept in full output, attribution unscored.

## Conclusion (independent of preroll)

On the completed transcript, boundary choice matters: under the frozen
rule, the H first boundary recovers the A-question/C-head split losslessly
at the text level; F0 fragments the A-question and re-merges the tail; the
ideal sample-true boundary misattributes `that?` rightward on this run's
provider timestamps. Timing estimates and the frozen rule jointly decide
the outcome; neither is claimed the sole bottleneck and no rule tuning was
attempted. No runtime-integration, product-deadline, or
provisional/commit claim: the transcript was finalized first and partitioned
offline, so there is no cross-probe controlled latency claim against the
audio-split probe.

## Corrections disclosed

1. Freeze grouping-rule prose described ranges from the previous word end,
   which double-counts inter-word whitespace; the script implements the
   exact-concat reading the freeze itself requires (trailing whitespace to
   the preceding word), recorded in results as `grouping_rule_reading`.
2. First `gt_posthoc` computation scored C-head `You` at its A-region
   occurrence and mislabeled mid-fragment C-heads; recomputed offline from
   stored evidence with phrase-order occurrence selection and
   first-boundary verdicts (`comparison_note` in results). Raw evidence and
   variants untouched.

## Reproduce

```text
uv run python experiments/psem_evidence_delivery_gap/text_partition_probe/run.py
```

Requires SONIOX_API_KEY in .env.local. Keys never printed. GT words lived
only in the freeze as post-hoc reference, never sent to any provider.
