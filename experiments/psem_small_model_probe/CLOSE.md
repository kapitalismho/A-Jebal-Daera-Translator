# PSEM small-model probe — close-out (issue #117, evaluator_revision=3)

> REPAIRED-V3 outputs only (`evaluator_revision=3`, `*/results_repaired_v3/`).
> V2 EVAL sessions reused dev-only per program approval — no generalization
> claim; V3 holdout required for any selection claim. Engineering branch
> only: no SOTA, no impossibility, no cross-corpus claims.

## Provenance: why rev1 and rev2 are VOID for the branch decision

- rev1 VOID (anchor-only gate): B-only speech starved the 500 ms decoder,
  so clean A→B handoffs could never accumulate; taus calibrated under it.
- rev2 VOID for the branch decision (two defects): (a) C2 clean handoffs
  displayed as topology `A` and were mis-scored as KEEP, so the CUT
  calibration never saw them and the C2 operating point is unknown;
  (b) NeoVAD `bind()` never warmed recurrent state, so O and C spans
  shared the same reset-state start (O≈C rows carry no warm-up contrast).
- rev3 (this close-out): stratum-aware scoring (KEEP=C1/C3/C5, CUT=C2/C4,
  C6 diagnostic-only; topology display-only), C2 calibration restored
  (miss objective on C2 ONLY, never C4), recurrent warm-up active
  (per-episode headers: warmup_frames 500 O / 100 C at frame_ms=10).
  Frozen inputs: `manifest_rev3` (`file_sha256 c9470d21…`, freeze
  `f6cbeefd…`, 84/84 field-identical to v1), CAL12_rev3 6 strata × 2.

## Gate table (rev3 numbers)

| Gate | Verdict | Key numbers |
|---|---|---|
| 0 manifest | FROZEN / PASS | 84 rows; MAIN48 C1..C6 = 8 each (KEEP 24 / CUT 16 / OTHER 8); CAL↔MAIN↔EXT session-disjoint; old freeze files untouched. |
| 1 adapters | PASS | Native weights, stub=false everywhere; ECAPA receipt pinned; 15/15 sessions mono 16 kHz. |
| 2 CAL12 taus (rev3) | FROZEN | firered O 0.85 (keep 5/6, C2 0/2); firered C 0.05 (keep 4/6, C2 2/2); neovad O 0.05 (keep 1/6, C2 2/2); neovad C 0.05 (keep 0/6 ZERO attainable, C2 2/2). TAU_GRID 0.05–0.95, fixed priority. |
| 3 MAIN48 native O | firered SIGNAL-STRONG/UNUSABLE; neovad WEAK | firered O tau 0.85: 22/24 false, missed 3/16, CUT 13/16 (C2 6/8 + C4 7/8), contam 282.3 s/h. neovad O tau 0.05: 8/24 false, missed 13/16, CUT 3/16 (C2 1/8 + C4 2/8), contam 784.0 s/h. |
| 4 MAIN48 causal C | O/C GAP REAL (firered); neovad COLLAPSED | firered C tau 0.05: 14/24 false, missed 12/16, CUT 4/16 (C2 2/8 + C4 2/8), contam 749.4 s/h. neovad C: 1/24 false, missed 16/16, CUT 0/16, no delays. |
| 5 VAD replay (firered C only) | INTEGRATION CLEAN | GT-gate reproduces MAIN firered-C exactly. Agreement 0.8336 (24k frames); retention 4/4 episodes, hit ratio 1.5 (prod 6/16 vs GT 4/16; C4 2/8→4/8, C2 2/8→2/8); false 14→16. Cost is over-triggering, not hidden signal. |
| 6 ontology | NOT RERUN — prior proxy results stand as marked | `ontology/*`, `compare/*` untouched per workstream freeze. |

Stratum note: MAIN48 carries zero `A->A+B->A` rows (persists from v1);
C3-KEEP rests on `overlap_return`. C6 (n=8) diagnostic-only, excluded
from all headlines.

## Branch answers (rev3, engineering level only)

- FireRed O: signal strong (13/16 CUT incl. C2 6/8 + C4 7/8) but operating
  point unusable (22/24 false cuts) — NOT promotable; policy/threshold
  work is the blocker, not sensitivity.
- FireRed C: 4/16 (C2 2/8 + C4 2/8) vs O 13/16 — the O/C gap is real under
  stratum truth. Candidate: short (1 s) enrollment bottleneck, not the
  observation model; unconfirmed, offered as the next probe, not a finding.
- NeoVAD O 3/16 vs C 0/16: first VALID O/C contrast (warm-up active) shows
  warm-up length matters, but the formulation stays weak in both regimes —
  no promotion case either way.
- VAD replay: production gating preserves all 4 GT-detected CUT episodes;
  nothing in the VAD hides signal.
- EXT24 / CONTROL24: UNOPENED (no boundary trigger: no headline near a
  pass boundary; pooled headlines decide; FireRed unpromotable so the
  CONTROL comparison would not change action).

## Known limitations

- Zero `A->A+B->A` rows in MAIN48 (mandatory KEEP cell 0/0, all cells).
- Ontology/compare not rerun to rev3 (stand as marked, GT-proxy caveats kept).
- EVAL dev-only; V3 fresh holdout needed before any model-selection claim.
- CPU numbers this-machine only (Win11, Ryzen 7 9800X3D); FireRed MAIN
  frames replayed from live-run archives, NeoVAD all-fresh.
- Calibration diagnostics (frame AUPRC/F1, sens streams) diagnostic-only,
  never selection inputs.
