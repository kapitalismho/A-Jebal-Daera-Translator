# PSEM P1 next decision (Child #133 under #132) — CORRECTED x3

> Correction history: Branch A selected first, rejected by Director (layer-8
> unknowns). Branch F recorded. Second correction: whole/split runs relabeled
> as lexical sensitivity probes; near-zero and no-lexical-damage claims
> withdrawn. F-review repair: pinned AMI annotation reuse (8 segments files,
> sha256-verified) with whole-pass owner alignment; provider token end_ms
> retained experiment-locally (bare text was harness discard); canonical
> emission bounds computed; G01 aggregate-excluded from tally (15 tallied);
> invented technical budgets removed. Amendment records in
> `paired_loss_ledger.json` `correction_amendment`. Contract and
> predeclaration identities preserved.

## 1. Protected harm after ordinary delivery boundaries are accounted for

Sequential cross-speaker merge / wrong attribution inside one delivered logical
utterance. Measured status: on G04/G05, production VAD replay shows the frozen
reference boundary inside a single VAD span (NOT already separated). Whether
that co-location produced actual merge/attribution harm is UNKNOWN: STT text
carries no owner labels, and no attribution-labeled delivery was ever recorded.
For the other 14 cases ordinary-path separation is unmeasured — never assumed
harmless, never assumed harmful. Simultaneous-speech lexical mixing is excluded
from the protected harm (unproven on every clip; overlap GT alone proves no
lexical damage).
## 2. Recoverable share with achievable ideal evidence

UNKNOWN. The whole/split runs are lexical sensitivity probes, not recovery
measurements: retrospective GT splits, independent forward passes, concatenated
translations issued as single requests, no runtime owner or output path
exercised — never called an ideal product test. Exact recorded differences
(`traces/text_diff.json`): G04 whole "They're" vs halves "||| Nice. They" and
whole "anyway" vs halves "any"; G05 whole "whether they were looking" vs
halves "wh ||| Looking" (halves visibly omit "whether they were"), whole "But
because" vs halves "Because", whole "of" vs halves "of.". Direction and cause
are UNKNOWN.
Owner alignment now measured on whole passes (`traces/owner_alignment.json`,
8 pinned AMI segments sha256-verified as reuse, provider token end_ms teed
with zero adapter edits): G04 105 tokens labeled (47 mixed-overlap unassigned,
0 unlabeled), G05 131 (33 mixed, 0 unlabeled); single-owner tokens are C-only
on BOTH sides of each split — the whole-pass transcripts cross NO annotation
owner change. The frozen reference boundary is anchor-relative replacement,
not an AMI speaker turn, so the delivery question is attribution-stability
under reference change. Provider end_ms trust is relative-ordering only.
Nothing here proves added PSEM has no value; P1 cannot quantify its value.

## 3. Earliest material loss

UNRESOLVED. Observation/event diagnostics exist (fragmented true evidence
G03-G05 with a verified single-capture construction mechanism; sustained
unmatched firing G01/G07/G08; projection/capture-order artifacts G14-G16;
F0 frame-layer false firing G09/G10 that H100 does not repeat), but every path
from these diagnostics to delivered harm crosses an unmeasured gap (ordinary
separation, commit/ack, attribution labels). A missing application receiver is
a gap, not evidence of earliest material recoverable loss, and authorizes no
repair. Selecting A, B, C, D, or E now would credit unknowns.

## 4. What the #121 F0/H contrast revealed

Frame observation carries signal (posthoc DEV AP 0.488 vs 0.151; H100 joint
gains; H silence where F0 frame-fires on G09/G10; matched catch G02). This is
an observation-layer diagnostic: AP/proxy gain proves no delivered value and no
evidence sufficiency. The 300/500 ms failure cause remains open. Nothing in the
contrast selects a technical repair.

## 5. Did #117 rev3 change the diagnosis? Caveats?

No branch follows from it. The same-threshold tau=0.05 read (false 15-vs-14,
missed 13-vs-12, contam 746-vs-749) supports LOCAL O/C similarity only. The
headline 13/16-vs-4/16 gap's cause is UNRESOLVED: threshold-confound noted,
reference-lifecycle cause neither proven nor rejected. Other caveats stand:
NeoVAD weak both regimes; VAD replay integration-clean at its layer; stale F0
comparison excluded without rescore; MAIN48 has zero A-to-A+B-to-A rows.

## 6. Harmless event failures?

Unproven for every case. G06/G11/G12/G13 emit no event under any operating
point, but delivered consequence is UNKNOWN (ordinary separation unmeasured),
not "harmless". No inspected failure is banked as harmless.

## 7. Correct-but-too-late frequency in the causal set

Zero classifiable: lateness is unmeasurable (commit times and mutability window
unstated). Canonical emission bounds computed from existing arrays
(`traces/emission_bounds.json`,
`frontier_sweep.simulate_episode` first-event return): G02 H emits
boundary 88300 ms → 89362.5 ms (frontier-bound); G04 H/F0 first-emit at
9097600→9109800 samples, before the case span, so the G04 high stretch gets
NO emission (first-event return already fired — the discard mechanism,
confirmed in canonical code, not a reimplementation); G05 H first-emit at
6075200 samples, also before its span. Commit still UNKNOWN; no latency
credit is taken from these bounds.

## 8. Segmentation-unrepairable lexical errors

None demonstrated, none excluded. Overlap context does not prove lexical
damage, and the recorded whole/halves text differences (including G05's
omitted "whether they were") have UNKNOWN attribution. Owner-labeled
whole-pass tokens show no cross-owner span, but mixed-overlap tokens (47/33)
are unassigned and halves lack token timing, so no "no lexical damage" claim
is made on any clip.

## 9. Beyond reference presence?

No demonstrated need — and no demonstrated sufficiency either. All
attribution-layer questions are unmeasured. No richer ontology is authorized;
none is rejected.

## 10. Single next branch

**F. Under the current evidence, no added PSEM authority is justified; defer
with the explicit reopening condition below. This is an insufficient-evidence
verdict, not proof of no value.**

## 11. Authorized next measurement (only)

Single next missing measurement: Audio-owned causal applicability
acknowledgement on the same support scope (the 2 measured clips, ≤6 total
clips). Audio states, for a causally achievable boundary (available at or
after evidence availability time, never before) on a predeclared clip the
real ordinary-delivery baseline does not separate: whether the downstream
path could have applied it, with applied source position / application time
or the exact reason-not-applied. That acknowledgement is the smallest
actionability measurement; without it no repair branch can be selected.
Optional paid token timing (e.g. split-half passes the same experiment-local
way, same 2 clips, no new clips, no translation reruns) is explicitly
conditional on that acknowledgement requiring paired owner-harm input — not
blanket authorized, and VAD-extension scope is authorized only if genuinely
necessary for the acknowledgement, not automatically.
Measurement budgets (actual small scope, not technical acceptance): same
support scope; bounded calls; zero new models/training; unknown timing
exposed. Technical acceptance numbers are UNSPECIFIED and require a later
Audio decision. No contract scope, no discard-policy removal, no P2 lifecycle
work is authorized — measurement only.
Reopening requires all three: a causally achievable boundary (available at or
after evidence availability time, never before) on a predeclared clip the
actual ordinary-delivery baseline does not separate, plus a paired
owner-labeled word comparison showing reduced merge/attribution harm under
boundary-split inference. An offline GT-split text sensitivity result alone
never reopens.

Addendum: the §11 probe is executed (`ACTIONABILITY.md`, trace `traces/actionability.json`, script `analysis/actionability.py`): every canonical requested boundary is VAD-OPEN at its earliest offline frontier, which proves no provider applicability — F retained, next prerequisite is the Audio-owned application semantics decision above.

## 12. Explicitly unauthorized

New neural training; H7302, T2, TA, or any #121 continuation; reopening #107;
new model-family acquisition or PVAD catalog; reopening #117 EXT24 or CONTROL24;
new large dataset or corpus beyond pinned-hash reuse of already-referenced
annotation files (reuse of V2-referenced files with verified hashes is
permitted and was exercised for 8 AMI segments files); broad
threshold optimization; persistence/decoder architecture sweep; full native
reference lifecycle implementation; full Audio architecture migration;
production writes or production PSEM enablement; distillation or student work;
source separation or speaker-conditioned ASR; changing historical acceptance
gates or rewriting #117/#121 conclusions; using frame AP alone to select a
path; any richer ontology; the withdrawn Branch A contract scope,
single-capture policy removal, and P2 lifecycle/vertical-slice work; any P3
repair before the §11 measurement exists; treating the parent DAG's existence
as authorization for any later child.
