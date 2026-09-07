# Gate 5 — production VAD replay, rev3 (firered x C, frozen tau=0.05)

> V2 EVAL sessions reused as dev-only probe per program approval; no unbiased generalization claim; V3 fresh holdout required for selection claims.

> Winner-only: firered regime C (sole causal formulation with a valid CUT signal under rev3 MAIN48: 4/16 CUT detected — C2 2/8 + C4 2/8; neovad-C 0/16, nothing to retain). Same 48 MAIN48 rows, same causal bind, same 500 ms confirmer / 300 ms sensitivity; gates compared: GT any-speech vs production Silero VAD spans (thr 0.5, chunk 512, pre-roll/hangover 500 ms).

## GT any-speech gate vs production-VAD (frozen tau=0.05)

| gate | false cuts (KEEP-n) | missed (CUT-n) | src_err p50/p90 (ms) | dec p50/p90 (ms) | CUT events / sens hits | contam s/h |
|---|---|---|---|---|---|---|
| GT-gate | 14/24 | 12/16 | 1635/1786 | 500/500 | 44 / 1294 | 749.4 |
| prod-VAD | 16/24 | 10/16 | 1055/1560 | 500/500 | 82 / 2328 | 720.4 |

Cross-check: the GT-gate row reproduces main/results_repaired_v3 firered-C exactly (lifecycle, binding, decoder identical; only the gate differs).

## Gate agreement (per 10 ms frame, GT any-speech vs prod VAD)

- frames scored: 24000; agreement: 0.8336
- GT-speech frames gated OFF by production VAD: 239/18693
- production-gate-ON frames where GT is off: 3754
- eval windows with ZERO production-gate coverage: 0/48

## CUT retention (GT-gate -> prod-VAD, stratum truth)

- CUT successes GT-gate: 4/16; prod-VAD: 6/16; hit-count ratio = 1.5
- episode-level retention: 4/4 GT-detected CUT episodes still detected under prod-VAD
- by stratum GT-gate: C2 2/8, C4 2/8; prod-VAD: C2 2/8, C4 4/8
- false cuts GT-gate: 14; prod-VAD: 16

## Verdict

Both good: integration clean — production gating preserves the GT-gate detections.
