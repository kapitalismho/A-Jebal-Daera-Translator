from __future__ import annotations

import pytest

from puripuly_heart.core.vrchat_scene_events import (
    IdlessPresence,
    InstanceTransition,
    LeftRoom,
    LocalPlayerInitialized,
    PlayerJoined,
    PlayerLeft,
)
from puripuly_heart.core.vrchat_scene_parser import parse_vrchat_scene_line


def _line(payload: str, level: str = "Debug") -> str:
    return f"2026.09.07 01:48:58 {level}      -  [Behaviour] {payload}"


def _join(name: str, user_id: str) -> str:
    return _line(f"OnPlayerJoined {name} ({user_id})")


def _leave(name: str, user_id: str) -> str:
    return _line(f"OnPlayerLeft {name} ({user_id})")


def test_joined_event_keeps_id_and_discards_name() -> None:
    event = parse_vrchat_scene_line(
        _join("Some Player", "usr_11111111-2222-4333-8444-555555555555")
    )

    assert event == PlayerJoined(user_id="usr_11111111-2222-4333-8444-555555555555")
    assert "Some Player" not in repr(event)


def test_left_event_keeps_id_and_discards_name() -> None:
    event = parse_vrchat_scene_line(
        _leave("Some Player", "usr_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
    )

    assert event == PlayerLeft(user_id="usr_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")


def test_name_with_parentheses_parses_last_id_group() -> None:
    event = parse_vrchat_scene_line(_join("Foo (bar)", "usr_12345678-1234-4234-8234-123456789012"))

    assert event == PlayerJoined(user_id="usr_12345678-1234-4234-8234-123456789012")


@pytest.mark.parametrize("level", ["Log", "Warning", "Error", "Debug"])
def test_envelope_accepts_unity_levels(level: str) -> None:
    event = parse_vrchat_scene_line(
        f"2026.09.07 01:48:58 {level}      -  [Behaviour] OnPlayerJoined N (usr_abc-def)"
    )

    assert event == PlayerJoined(user_id="usr_abc-def")


def test_stack_trace_without_envelope_is_ignored() -> None:
    assert parse_vrchat_scene_line("  at Game.OnLeftRoom () [0x00000] in <0>:0 ") is None


def test_non_behaviour_tag_is_ignored() -> None:
    line = "2026.09.07 01:48:58 Debug      -  [NetworkManager] OnPlayerJoined N (usr_abc)"

    assert parse_vrchat_scene_line(line) is None


def test_colon_join_variant_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerJoined:Unnamed")) is None


def test_player_left_room_is_not_a_leave() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerLeftRoom")) is None


def test_player_left_room_notice_with_suffix_is_degraded_signal() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerLeftRoomExtra")) == IdlessPresence()


def test_colon_left_variant_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerLeft:Someone (usr_abc)")) is None


def test_exact_left_room_matches() -> None:
    assert parse_vrchat_scene_line(_line("OnLeftRoom")) == LeftRoom()


def test_left_room_with_suffix_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line("OnLeftRoomExtra")) is None


def test_entering_room_is_transition() -> None:
    assert parse_vrchat_scene_line(_line("Entering Room: VRChat Home")) == InstanceTransition()


def test_joining_world_is_transition() -> None:
    line = _line("Joining wrld_4432ea9b-729c-46e3-8eaf-846aa0a37fdd:67646~private(usr_abc)")
    assert parse_vrchat_scene_line(line) == InstanceTransition()


def test_joining_or_creating_room_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line("Joining or Creating Room: VRChat Home")) is None


def test_joining_friend_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line("Joining friend: somewhere")) is None


def test_local_player_init_is_evidence_without_identity() -> None:
    event = parse_vrchat_scene_line(_line('Initialized PlayerAPI "Some Player" is local'))

    assert event == LocalPlayerInitialized()
    assert "Some Player" not in repr(event)


def test_remote_player_init_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line('Initialized PlayerAPI "Some Player" is remote')) is None


def test_idless_join_is_degraded_signal_not_join() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerJoined LonelyPlayer")) == IdlessPresence()


def test_idless_leave_is_degraded_signal_not_leave() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerLeft LonelyPlayer")) == IdlessPresence()


def test_malformed_user_id_is_degraded_signal() -> None:
    assert parse_vrchat_scene_line(_line("OnPlayerJoined N (not-a-user)")) == IdlessPresence()
    assert parse_vrchat_scene_line(_line("OnPlayerJoined N (usr_)")) == IdlessPresence()


def test_unrelated_behaviour_lines_are_ignored() -> None:
    assert parse_vrchat_scene_line(_line("Registering Avatar Interaction")) is None
    assert parse_vrchat_scene_line(_line("Destination fetching: wrld_4432ea9b-729c")) is None
    assert parse_vrchat_scene_line(_line("Successfully left room")) is None


def test_empty_and_non_string_input_is_ignored() -> None:
    assert parse_vrchat_scene_line("") is None
    assert parse_vrchat_scene_line(None) is None  # type: ignore[arg-type]


def test_udon_evil_photon_prefix_does_not_parse_as_leave() -> None:
    line = _line("Udon log: ] OnPlayerLeft Victim (usr_aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee)")

    assert parse_vrchat_scene_line(line) is None


def test_world_echo_transition_prefix_does_not_reset_roster() -> None:
    assert parse_vrchat_scene_line(_line("[World] Entering Room: Fake")) is None
    assert parse_vrchat_scene_line(_line("Echo ] Joining wrld_deadbeef:1")) is None


def test_join_name_containing_markers_parses_as_single_join() -> None:
    event = parse_vrchat_scene_line(
        _join("Trick ] Entering Room: Fake", "usr_11111111-2222-4333-8444-555555555555")
    )

    assert event == PlayerJoined(user_id="usr_11111111-2222-4333-8444-555555555555")


def test_join_name_containing_joining_friend_is_not_suppressed() -> None:
    event = parse_vrchat_scene_line(
        _join("Trick ] Joining friend: Nope", "usr_11111111-2222-4333-8444-555555555555")
    )

    assert event == PlayerJoined(user_id="usr_11111111-2222-4333-8444-555555555555")


def test_fake_local_init_prefix_is_ignored() -> None:
    assert parse_vrchat_scene_line(_line('Fake Initialized PlayerAPI "X" is local')) is None
    assert parse_vrchat_scene_line(_line('Initialized PlayerAPI "X" is local now')) is None
    assert parse_vrchat_scene_line(_line("Initialized PlayerAPI X is local")) is None
