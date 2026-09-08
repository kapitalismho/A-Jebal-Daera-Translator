from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from puripuly_heart.providers.stt.soniox import _SonioxSession

EXP = ROOT / "experiments" / "psem_evidence_delivery_gap"
SRC = EXP / "text_partition_probe" / "results.json"
FREEZE = EXP / "text_partition_probe" / "freeze.json"
OUT_DIR = Path(__file__).resolve().parent

BASELINE_BOUNDS: dict[str, list[int]] = {
    "baseline": [],
    "ideal": [52156984],
    "f0": [52139200, 52203200],
    "h": [52158400, 52203200],
}


def load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "text_partition_probe_run", EXP / "text_partition_probe" / "run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wire_groups(raw_tokens: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for token in raw_tokens:
        if groups and token["arrival_wall"] == groups[-1][0]["arrival_wall"]:
            groups[-1].append(token)
        else:
            groups.append([token])
    return groups


def wire_payload(token: dict) -> dict:
    return {
        "text": token["text"],
        "start_ms": token.get("start_ms"),
        "end_ms": token.get("end_ms"),
        "is_final": token["is_final"],
        "language": token.get("language"),
        "confidence": token.get("confidence"),
    }


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    probe = load_probe_module()
    adapter_path = Path(sys.modules[_SonioxSession.__module__].__file__).resolve()
    assert str(adapter_path).startswith(str(ROOT))
    p0 = freeze["audio"]["payload_samples"][0]

    raw_tokens = src["raw_evidence"]["raw_tokens"]
    groups = wire_groups(raw_tokens)
    assert len(groups) == src["raw_evidence"]["raw_msg_with_tokens"] == 44
    assert src["raw_evidence"]["control_fin_end"] == 1

    session = _SonioxSession(
        api_key="replay-offline",
        model="stt-rt-v5",
        endpoint="wss://stt-rt.soniox.com/transcribe-websocket",
        sample_rate_hz=16000,
        language_hints=["en"],
        context_terms=[],
        keepalive_interval_s=10.0,
        trailing_silence_ms=100,
        connect_timeout_s=5.0,
    )
    for group in groups:
        session._handle_message(json.dumps({"tokens": [wire_payload(t) for t in group]}))

    events = []
    while not session._events.empty():
        events.append(session._events.get_nowait())
    assert len(events) == 1
    event = events[0]
    assert bool(event.is_final) and event.text

    interim = [t for t in raw_tokens if not t["is_final"]]
    fins = [t for t in raw_tokens if t["is_final"]]
    speech_fins = [t for t in fins if t["text"] not in ("<fin>", "<end>")]
    controls = [t for t in fins if t["text"] in ("<fin>", "<end>")]
    assert (len(raw_tokens), len(interim), len(speech_fins), len(controls)) == (1046, 992, 53, 1)

    expected = "".join(t["text"] for t in speech_fins)
    assert event.text == expected
    assert event.final_language_runs == ()
    assert event.text == "".join(tok.text for tok in session._final_tokens)

    prior = src["accepted"]
    assert len(prior["final_text"]) == 157 and prior["n_tokens"] == 50
    assert len(event.text) == 165 and len(session._final_tokens) == 53
    assert prior["final_text"] + "re just." == event.text

    tokens = [
        {"o": i, "text": tok.text, "end_ms": tok.end_ms}
        for i, tok in enumerate(session._final_tokens)
    ]
    tape_groups = probe.build_groups(event.text, tokens, p0)
    assert "".join(g["text"] for g in tape_groups) == event.text

    variants = {}
    for name, bounds in BASELINE_BOUNDS.items():
        frags, _ = probe.partition(tape_groups, bounds)
        conservation = "".join(f["text"] for f in frags) == event.text
        assert conservation
        variants[name] = {"bounds": bounds, "fragments": frags, "conservation": conservation}

    comparison = probe.gt_compare(tape_groups, variants, freeze)
    prior_ownership = src["gt_posthoc"]["ownership"]
    for word, record in comparison["ownership"].items():
        for variant, verdict in record["per_variant"].items():
            prior_verdict = prior_ownership[word]["per_variant"][variant]
            assert verdict["scored_group"] == prior_verdict["scored_group"]
            assert verdict["verdict"] == prior_verdict["verdict"]

    right_groups = [g["idx"] for g in tape_groups if probe.norm_word(g["word"]) == "right"]
    assert len(right_groups) == 2
    tail_7200 = [tok.text for tok in session._final_tokens if tok.end_ms == 7200]
    assert tail_7200 == ["'", "re", " just", "."]

    out = {
        "repair": "soniox pending-final equal-end guard <= to <",
        "authority": {
            "baseline_commit": src["authority"]["baseline_commit"],
            "branch": src["authority"]["branch"],
        },
        "source_evidence": {
            "results": "experiments/psem_evidence_delivery_gap/text_partition_probe/results.json",
            "freeze": "experiments/psem_evidence_delivery_gap/text_partition_probe/freeze.json",
            "run_id": src["run_id"],
            "immutable": True,
        },
        "wire_replay": {
            "wire_groups": len(groups),
            "adapter_path": str(adapter_path),
            "tokenless_messages_omitted": src["raw_evidence"]["raw_msg_without_tokens"],
            "tokenless_omission_note": "cached evidence retains no bodies for the 2 tokenless messages, so error/metadata content cannot be replayed; replay covers the 44 token-bearing groups reconstructed by consecutive arrival_wall with count equality to raw_msg_with_tokens",
            "control_fin_included": True,
            "finalize_lifecycle_note": "no _FinalizeRequest sent; replay covers the single <fin> accumulation path, not live client finalize lifecycle; existing tests cover unchanged finalize behavior separately",
            "network": "none; direct _SonioxSession._handle_message in wire order",
            "consumer_events": len(events),
        },
        "counts": {
            "raw_tokens": len(raw_tokens),
            "interim": len(interim),
            "speech_final": len(speech_fins),
            "control_fin": len(controls),
        },
        "prior_accepted_preproof": {
            "final_text": prior["final_text"],
            "n_tokens": prior["n_tokens"],
            "text_len": len(prior["final_text"]),
        },
        "corrected_accepted": {
            "final_text": event.text,
            "n_tokens": len(tokens),
            "text_len": len(event.text),
            "tokens": tokens,
            "restored_tail": "re just.",
            "restored_chars": 8,
        },
        "groups": tape_groups,
        "variants": variants,
        "gt_posthoc": comparison,
        "ownership_parity_with_baseline": True,
        "duplicate_right_groups": right_groups,
        "same_end_7200_wire_order": tail_7200,
        "consumer_language_invariant": "final_language_runs == () with language identification off; event text equals raw speech-final concat",
        "limits": [
            "Content conservation is measured at the normalized final against the raw speech-final concat, not ASR accuracy against ground truth.",
            "Provider token ms are approximate; partition ownership is agreement with GT side at known bounds, never exact speaker purity.",
            "The append-equal guarantee is scoped to the pending final before <fin>; unsolicited-fin merge behavior is unchanged and not generally characterized.",
            "Restored tail tokens keep approximate end_ms 7200; tail overlap remains unscored.",
        ],
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wire_groups={len(groups)} raw={len(raw_tokens)} "
        f"prior={len(prior['final_text'])}/{prior['n_tokens']} "
        f"corrected={len(event.text)}/{len(tokens)} "
        f"tail={tail_7200} variants_conserved={len(variants)}"
    )


if __name__ == "__main__":
    main()
