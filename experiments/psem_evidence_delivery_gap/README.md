# PSEM evidence-to-delivery gap audit (Child #133, parent #132)

Lean probe. One owner, five artifacts + three tiny probes and six trace files
plus one pinned-reuse annotation dir. No production edits, no training, no
frontier regeneration, no rescoring, no git mutations.

> Correction history: Branch A selected first, rejected and withdrawn; probes
> relabeled (never baseline/ideal) with exact text differences and UNKNOWN
> attribution. F-review repair: 8 pinned AMI segments files retrieved as
> sha256-verified reuse with whole-pass owner alignment; provider token end_ms
> retained experiment-locally; canonical emission bounds computed; G01
> aggregate-excluded (15 tallied); invented budgets removed. Records in
> ledger `correction_amendment`. Contract (`...contract.v1`, frozen
> 2026-09-08T10:12:12Z) and predeclaration (2026-09-08T10:12:42Z) identities
> preserved unchanged.

## Map

Single implementation owner (provenance, case selection, timing interpretation,
final decision share one ledger/contracts; splitting duplicates decisions).
No sibling mutable nodes.

## Baseline / freeze

- Baseline HEAD `45956c69` (branch `experiment-v2-speaker-change-turn-boundaries-ls`,
  clean; ahead/behind origin/same: 0/0, read-only).
- Ordinary peer path with no PSEM authority: VAD SpeechStart/End → STT finals →
  peer translation turn → projection/output. The path keeps no source-time
  applied-boundary records (gap description only — proves no earliest loss).

## Progression

1. `product_measurement_contract.json` frozen BEFORE any case work (8 layers,
   time semantics, harm categories, causal rules).
2. 16 existing cases predeclarated by mechanical frame-disagreement scan
   BEFORE causal inspection (procedure + operating points in ledger).
   Diagnostic set, not prevalence.
3. `analysis/probe.py` reuses cached #121 export + rev3 curves only; writes
   `traces/case_enrichment.json` (per-case speakers + reference anchor detail),
   `traces/rev3_same_threshold.json`, and `traces/text_diff.json` (exact
   whole/halves word diffs + ±30-frame speaker/overlap/speech/target windows
   around the G04/G05 splits; offline alignment from in-repo timelines).
4. `analysis/realcheck.py` ran a lexical sensitivity probe on G04/G05 (NOT a
   baseline or ideal test): production VAD spans + Soniox stt-rt-v5
   whole/halves forward passes split at the frozen reference boundary +
   translation attempts. Historical trace route (kept untouched):
   DeepSeek 402 Insufficient Balance on 4/4, then-banned Cerebras fallback
   2/4. The retained script was cut over to OpenRouter
   `google/gemma-4-26b-a4b-it` with no fallback — reruns take the new route.
   Isolated adapters, keys in-process, no settings/production writes.
   Trace: `traces/realcheck.json`.
5. `evidence_authority_manifest.json` (Phase A),
   `paired_loss_ledger.json` (Phases B-D, per-case diagnostic-vs-delivered
   split), `PSEM_NEXT_DECISION.md` (branch F + 12 answers +
   only-authorized measurement child).
6. `analysis/actionability.py` ran the §11 bounded VAD-actionability probe (production Silero/gating replay on the same G04/G05 audio, zero paid calls) → `traces/actionability.json` + `ACTIONABILITY.md` (all requested boundaries VAD-OPEN at earliest frontier; F retained).

## Reproduce

```text
uv run python experiments/psem_evidence_delivery_gap/analysis/probe.py
uv run python experiments/psem_evidence_delivery_gap/analysis/realcheck.py
uv run python experiments/psem_evidence_delivery_gap/analysis/token_timing.py
```

## Result

Branch F (bounded defer, insufficient evidence — NOT a no-value proof):
lexical sensitivity probes recorded exact whole/halves text differences with
UNKNOWN attribution; owner-labeled whole-pass tokens cross no annotation owner
change (mixed-overlap tokens unassigned; halves untimed); the protected
attribution harm is unmeasured. Exact remaining blocker: provider commit/ack
semantics unstated, so applicability is unshowable. Only authorized next
step: the single §11 actionability measurement; scope budgets there; technical
acceptance unspecified pending Audio.

## Revised Phase D probe

Follow-on engineering probe with real held boundaries and Soniox split
finals on EN2009d [3248200, 3272600]: `revised_phase_d/REPORT.md`
(freeze `revised_phase_d/freeze.json`, script `revised_phase_d/run.py`,
results `revised_phase_d/results.json`). P1 records above preserved as
historical; the revised result supersedes only the bounded merge scope.

## Preroll probe

Second-segment context-sensitivity control (500 ms preroll in/out, 2 fresh
Soniox sessions, RAW finals only): `preroll_probe/REPORT.md` (freeze
`preroll_probe/freeze.json`, script `preroll_probe/run.py`, results
`preroll_probe/results.json`). Historical first final reused, not rerun.

## Text-partition probe

Offline text-boundary comparison on one completed continuous-session final
over the same target payload (no preroll, no audio hold, no translation):
`text_partition_probe/REPORT.md` (freeze `text_partition_probe/freeze.json`,
script `text_partition_probe/run.py`, results
`text_partition_probe/results.json`). Separate conclusion from preroll.

## Equal-timestamp repair

Pending-final guard fix (`<=` to `<`) with offline replay of the cached
text-partition wire groups (44 groups, 165 chars / 53 tokens, tail
`they're just.` restored): `soniox_equal_timestamp_repair/REPORT.md`
(script `soniox_equal_timestamp_repair/replay.py`, results
`soniox_equal_timestamp_repair/results.json`). Cached evidence untouched.
