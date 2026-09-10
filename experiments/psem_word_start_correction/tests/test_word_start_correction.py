import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.psem_product_translation.capture import (
    build_groups,
    corrected_groups_for_capture,
    hydrate_accepted_starts,
    provider_token_start,
)


def test_group_zero_uses_provider_start_not_session_start():
    groups = build_groups("Not here", [
        {"o": 0, "text": "Not", "start_ms": 240, "end_ms": 300},
        {"o": 1, "text": " here", "start_ms": 360, "end_ms": 480},
    ], 670592)
    assert groups[0]["start_src"] == 670592 + round(240 * 16)
    assert groups[0]["start_ms"] == 240


def test_later_groups_not_chained_to_prev_end():
    groups = build_groups("Not here", [
        {"o": 0, "text": "Not", "start_ms": 240, "end_ms": 300},
        {"o": 1, "text": " here", "start_ms": 360, "end_ms": 480},
    ], 670592)
    assert groups[1]["start_src"] == 670592 + round(360 * 16)
    assert groups[1]["start_src"] != groups[0]["end_src"]


def test_zero_start_valid_when_end_positive():
    s, reason = provider_token_start({"start_ms": 0, "end_ms": 120})
    assert s == 0
    assert reason is None
    groups = build_groups("The end", [
        {"o": 0, "text": "The", "start_ms": 0, "end_ms": 120},
        {"o": 1, "text": " end", "start_ms": 180, "end_ms": 300},
    ], 1000)
    assert groups[0]["start_src"] == 1000


def test_missing_start_has_reason_and_no_fallback():
    groups = build_groups("Not here", [
        {"o": 0, "text": "Not", "start_ms": 240, "end_ms": 300},
        {"o": 1, "text": " here", "end_ms": 480},
    ], 670592)
    assert groups[1]["start_src"] is None
    assert groups[1]["start_reason"] == "missing-start"


def test_bad_starts_report_explicit_reasons():
    cases = [
        ({"start_ms": 300, "end_ms": 300}, "degenerate-start"),
        ({"start_ms": 400, "end_ms": 300}, "degenerate-start"),
        ({"start_ms": -5, "end_ms": 300}, "negative-start"),
        ({"start_ms": float("nan"), "end_ms": 300}, "nonfinite-start"),
        ({"start_ms": float("inf"), "end_ms": 300}, "nonfinite-start"),
        ({"start_ms": 0, "end_ms": 0}, "degenerate-start"),
        ({"start_ms": 60}, "degenerate-start"),
    ]
    for tok, reason in cases:
        s, got = provider_token_start(tok)
        assert s is None
        assert got == reason


def test_punct_only_group_stays_unknown():
    groups = build_groups("Right.", [
        {"o": 0, "text": "R", "start_ms": 60, "end_ms": 120},
        {"o": 1, "text": "ight", "start_ms": 120, "end_ms": 240},
        {"o": 2, "text": ".", "start_ms": 240, "end_ms": 300},
    ], 5000)
    assert len(groups) == 1
    assert groups[0]["start_src"] == 5000 + round(60 * 16)
    tail = build_groups("Wait .", [
        {"o": 0, "text": "Wait", "start_ms": 60, "end_ms": 300},
        {"o": 1, "text": " ", "start_ms": 300, "end_ms": 360},
        {"o": 2, "text": ".", "start_ms": 360, "end_ms": 420},
    ], 5000)
    assert tail[1]["word"] == "."
    assert tail[1]["start_src"] is None
    assert tail[1]["start_reason"] == "punct-only-group"


def test_punct_owner_excluded_when_lexical_core_exists():
    groups = build_groups("We're", [
        {"o": 0, "text": "We", "start_ms": 60, "end_ms": 120},
        {"o": 1, "text": "'", "start_ms": 30, "end_ms": 180},
        {"o": 2, "text": "re", "start_ms": 120, "end_ms": 240},
    ], 7000)
    assert groups[0]["start_ms"] == 60
    assert groups[0]["end_prov_ms"] == 240
    assert groups[0]["end_src"] == 7000 + round(240 * 16)


def test_trailing_whitespace_does_not_extend_end():
    groups = build_groups("All ", [
        {"o": 0, "text": "All", "start_ms": 600, "end_ms": 660},
        {"o": 1, "text": " ", "start_ms": 660, "end_ms": 840},
    ], 10000)
    assert groups[0]["word"] == "All"
    assert groups[0]["end_prov_ms"] == 660
    assert groups[0]["end_src"] == 10000 + round(660 * 16)
    assert groups[0]["token_refs"][1]["slice"] == " "


def test_shared_multiword_token_flagged_without_new_precision():
    groups = build_groups("All the", [
        {"o": 0, "text": "All", "start_ms": 600, "end_ms": 660},
        {"o": 1, "text": " the", "start_ms": 660, "end_ms": 840},
    ], 10000)
    assert groups[1]["start_ms"] == 660
    assert groups[1]["start_note"] == "provider-start-shared-token"
    assert "".join(g["text"] for g in groups) == "All the"


def test_hydrator_recovers_unique_alignment_with_provenance():
    acc = [{"o": 0, "text": "Not", "end_ms": 300},
           {"o": 1, "text": " here", "end_ms": 480}]
    raw = [{"text": "Not", "start_ms": 240, "end_ms": 300, "is_final": False},
           {"text": "Not", "start_ms": 240, "end_ms": 300, "is_final": True},
           {"text": " here", "start_ms": 360, "end_ms": 480, "is_final": True},
           {"text": "<fin>", "start_ms": 0, "end_ms": 0, "is_final": True}]
    hydrated, prov = hydrate_accepted_starts(acc, raw)
    assert [t["start_ms"] for t in hydrated] == [240, 360]
    assert [t["text"] for t in hydrated] == ["Not", " here"]
    assert [t["end_ms"] for t in hydrated] == [300, 480]
    assert prov[0]["raw_idxs"] == [1]
    assert prov[1]["raw_idxs"] == [2]


def test_hydrator_never_teaches_from_partials():
    acc = [{"o": 0, "text": "Not", "end_ms": 300}]
    raw = [{"text": "Not", "start_ms": 999, "end_ms": 300, "is_final": False},
           {"text": "Not", "start_ms": 240, "end_ms": 300, "is_final": True}]
    hydrated, _ = hydrate_accepted_starts(acc, raw)
    assert hydrated[0]["start_ms"] == 240


def test_hydrator_duplicate_runs_with_differing_starts_stay_unknown():
    acc = [{"o": 0, "text": "Hi", "end_ms": 200}]
    raw = [{"text": "Hi", "start_ms": 60, "end_ms": 200, "is_final": True},
           {"text": "Hi", "start_ms": 120, "end_ms": 200, "is_final": True}]
    hydrated, prov = hydrate_accepted_starts(acc, raw)
    assert hydrated[0]["start_ms"] is None
    assert hydrated[0]["ambiguous"] is True
    assert prov[0]["reason"] == "ambiguous-start"
    groups = build_groups("Hi", hydrated, 0, prov)
    assert groups[0]["start_src"] is None
    assert groups[0]["start_reason"] == "ambiguous-start"


def test_hydrator_agreeing_duplicates_usable_with_all_ids():
    acc = [{"o": 0, "text": "Hi", "end_ms": 200}]
    raw = [{"text": "Hi", "start_ms": 60, "end_ms": 200, "is_final": True},
           {"text": "Hi", "start_ms": 60, "end_ms": 200, "is_final": True}]
    hydrated, prov = hydrate_accepted_starts(acc, raw)
    assert hydrated[0]["start_ms"] == 60
    assert prov[0]["raw_idxs"] == [0, 1]


def test_corrected_groups_preserve_text_end_refs():
    cap = {"audio": {"payload_samples": [670592, 700000]},
           "accepted": {"final_text": "Not here",
                        "tokens": [{"o": 0, "text": "Not", "end_ms": 300},
                                   {"o": 1, "text": " here", "end_ms": 480}]},
           "raw_evidence": {"raw_tokens": [
               {"text": "Not", "start_ms": 240, "end_ms": 300, "is_final": True},
               {"text": " here", "start_ms": 360, "end_ms": 480, "is_final": True}]},
           "groups": []}
    groups = corrected_groups_for_capture(cap)
    assert "".join(g["text"] for g in groups) == "Not here"
    assert [g["idx"] for g in groups] == [0, 1]
    assert sorted({r["o"] for g in groups for r in g["token_refs"]}) == [0, 1]
    assert [g["end_src"] for g in groups] == [670592 + round(300 * 16),
                                             670592 + round(480 * 16)]
    assert groups[0]["start_src"] == 670592 + round(240 * 16)


def test_any_invalid_lexical_owner_makes_word_unknown():
    groups = build_groups("ab", [
        {"o": 0, "text": "a", "end_ms": 100},
        {"o": 1, "text": "b", "start_ms": 50, "end_ms": 100},
    ], 1000)
    assert groups[0]["start_src"] is None
    assert groups[0]["start_reason"] == "missing-start"
    assert groups[0]["unresolved"] is True


def test_reversed_invalid_owner_makes_word_unknown():
    groups = build_groups("ab", [
        {"o": 0, "text": "a", "start_ms": 10, "end_ms": 100},
        {"o": 1, "text": "b", "start_ms": 200, "end_ms": 100},
    ], 1000)
    assert groups[0]["start_src"] is None
    assert groups[0]["start_reason"] == "degenerate-start"
    assert groups[0]["unresolved"] is True


def test_valid_min_preserved_when_all_lexical_valid():
    groups = build_groups("ab", [
        {"o": 0, "text": "a", "start_ms": 80, "end_ms": 100},
        {"o": 1, "text": "b", "start_ms": 50, "end_ms": 120},
    ], 1000)
    assert groups[0]["start_src"] == 1000 + round(50 * 16)
    assert groups[0]["unresolved"] is False


def test_sample_rounding_collision_is_unknown_not_zero_duration():
    groups = build_groups("ab", [
        {"o": 0, "text": "a", "start_ms": 1.01, "end_ms": 1.02},
        {"o": 1, "text": "b", "start_ms": 1.01, "end_ms": 1.03},
    ], 0)
    assert groups[0]["end_src"] is not None
    assert groups[0]["start_src"] is None
    assert groups[0]["start_reason"] == "sample-degenerate"
    assert groups[0]["start_ms"] == 1.01
    assert groups[0]["unresolved"] is True


def test_invalid_end_yields_none_but_preserves_raw():
    groups = build_groups("Right.", [
        {"o": 0, "text": "R", "start_ms": 0, "end_ms": 0},
        {"o": 1, "text": "ight", "start_ms": 0, "end_ms": 0},
        {"o": 2, "text": ".", "start_ms": 0, "end_ms": 0},
    ], 9119360)
    assert groups[0]["end_prov_ms"] == 0
    assert groups[0]["end_src"] is None
    assert groups[0]["end_reason"] == "degenerate-end"
    assert groups[0]["unresolved"] is True


def test_nonfinite_and_bool_ends_do_not_crash():
    for bad_end in (float("nan"), float("inf"), True):
        groups = build_groups("Hi", [
            {"o": 0, "text": "Hi", "start_ms": 60, "end_ms": bad_end},
        ], 500)
        assert groups[0]["end_src"] is None
        assert groups[0]["unresolved"] is True


def test_hydrator_keeps_known_start_despite_raw_candidate():
    acc = [{"o": 0, "text": "Go", "start_ms": 100, "end_ms": 200}]
    raw = [{"text": "Go", "start_ms": 111, "end_ms": 200, "is_final": True}]
    hydrated, prov = hydrate_accepted_starts(acc, raw)
    assert hydrated[0]["start_ms"] == 100
    assert prov[0]["reason"] == "accepted-kept"
    assert prov[0]["raw_idxs"] == [0]


def test_hydrator_keeps_invalid_known_start_for_explicit_reason():
    acc = [{"o": 0, "text": "Go", "start_ms": float("nan"), "end_ms": 200}]
    raw = [{"text": "Go", "start_ms": 111, "end_ms": 200, "is_final": True}]
    hydrated, _ = hydrate_accepted_starts(acc, raw)
    assert hydrated[0]["start_ms"] != hydrated[0]["start_ms"]
    groups = build_groups("Go", hydrated, 0)
    assert groups[0]["start_src"] is None
    assert groups[0]["start_reason"] == "nonfinite-start"


def test_accented_word_counts_as_lexical():
    groups = build_groups("caf\u00e9", [
        {"o": 0, "text": "caf", "start_ms": 60, "end_ms": 180},
        {"o": 1, "text": "\u00e9", "start_ms": 180, "end_ms": 240},
    ], 2000)
    assert groups[0]["start_src"] == 2000 + round(60 * 16)
    assert groups[0]["start_reason"] is None
