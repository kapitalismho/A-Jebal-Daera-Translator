# Soniox equal-timestamp repair

One-line adapter fix plus offline replay of the cached text-partition wire evidence. No network, no rescoring, no edits to historical freezes or results.

## Plan

1. Add a failing regression test for same-`end_ms` finals (within and across messages, identical repeat, stale skip, interim ignore).
2. Change the pending-final guard in `src/puripuly_heart/providers/stt/soniox.py` from `<=` to `<`.
3. Replay the 44 cached wire groups through the real `_SonioxSession` handler and re-derive partitions with the frozen `text_partition_probe` functions.
4. Keep the focused suites green; record this report.

## Setup

- Baseline `45956c69895aa3e33c60e9c516fc40ba7d60b954`, branch `experiment-v2-speaker-change-turn-boundaries-ls`.
- Source evidence: `text_partition_probe/results.json` run `bb7ee489-7e27-490d-9238-f1b89e55a053` (read-only).
- Repair scope: pending final before `<fin>` only. Equal-`end_ms` finals append in wire order; strictly earlier ones are still skipped; interim path untouched.

## Observations

- New unit test failed pre-fix (`they'` vs `they're just..`), passes post-fix.
- Replay: 44 token-bearing wire groups in arrival order, 1046 raw tokens = 992 interim + 53 speech-final + 1 control. The 2 tokenless messages have no retained bodies so their error/metadata content cannot be replayed; groups were reconstructed by consecutive arrival_wall with count equality to raw_msg_with_tokens. The resolved adapter path is asserted and recorded in results.json.
- Corrected final: 165 chars / 53 tokens ending `they're just.` vs prior 157 chars / 50 tokens ending `they'`. Restored tail `re just.` (8 chars) matches the raw speech-final concat exactly.
- All 4 partition variants (baseline/ideal/F0/H) concatenate exactly the corrected final; scored word ownership (`do/that/You/have/to/hope`) identical to the cached baseline; duplicate `Right` groups `[0, 26]` preserved; `7200` tail order `["'", "re", " just", "."]`; consumer event is a single final with `final_language_runs == ()`.
- No _FinalizeRequest is sent in the replay: it covers the single <fin> accumulation path, not live client finalize lifecycle; existing tests cover the unchanged finalize behavior separately.
- Focused suites `tests/providers/test_soniox_backend.py` + `tests/core/test_soniox_multilingual_release_readiness.py`: 35 passed.

## Decision

Repair accepted for the pending-final scope. The append-equal behavior is not claimed for cross-`<fin>` merge or unsolicited finals; those paths are unchanged.

## Limits

- Conservation is measured at the normalized final against the raw speech-final concat, not ASR accuracy against ground truth.
- Provider `end_ms` values are approximate; restored tail overlap stays unscored.
- No translation or latency claim is made by this replay.

## Reproduce

```text
python -m pytest tests/providers/test_soniox_backend.py::test_soniox_session_appends_equal_end_final_tokens_in_wire_order -q
python -m pytest tests/providers/test_soniox_backend.py tests/core/test_soniox_multilingual_release_readiness.py -q
.venv/Scripts/python -m pytest tests/providers/test_soniox_backend.py tests/core/test_soniox_multilingual_release_readiness.py -q
.venv/Scripts/python experiments/psem_evidence_delivery_gap/soniox_equal_timestamp_repair/replay.py
```

The first two runs used the global interpreter (pre-fix FAIL, post-fix 35 passed); the final confirm and the replay used the venv interpreter. tests/conftest.py inserts src on sys.path, so both interpreters import this worktree's adapter.
