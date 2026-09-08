from __future__ import annotations

from puripuly_heart.core.vrchat_scene_tracker import VrchatScenePresenceTracker


def _burst(tracker: VrchatScenePresenceTracker, *user_ids: str, local: str) -> None:
    tracker.on_transition()
    for user_id in user_ids:
        tracker.on_join(user_id)
    tracker.on_local_initialized()
    assert local in user_ids


def test_initial_burst_including_self_settles_ready() -> None:
    tracker = VrchatScenePresenceTracker()
    assert tracker.snapshot().status == "syncing"

    _burst(tracker, "usr_self", "usr_b", "usr_c", local="usr_self")
    assert tracker.snapshot().participant_count is None

    tracker.settle_if_quiet()

    snapshot = tracker.snapshot()
    assert snapshot.status == "ready"
    assert snapshot.participant_count == 3


def test_duplicate_join_does_not_drift_count() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()

    tracker.on_join("usr_b")
    tracker.on_join("usr_self")

    assert tracker.snapshot().participant_count == 2


def test_live_join_and_leave_update_ready_count() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()

    tracker.on_join("usr_c")
    assert tracker.snapshot().participant_count == 3

    tracker.on_leave("usr_b")
    assert tracker.snapshot().participant_count == 2


def test_unknown_leave_degrades_and_stays_sticky() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()

    tracker.on_leave("usr_ghost")

    assert tracker.snapshot().status == "degraded"
    assert tracker.snapshot().participant_count is None

    tracker.on_join("usr_c")
    tracker.on_leave("usr_b")
    assert tracker.snapshot().status == "degraded"


def test_transition_heals_degraded_and_clears_members() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()
    tracker.on_leave("usr_ghost")
    assert tracker.snapshot().status == "degraded"

    tracker.on_transition()
    assert tracker.snapshot().status == "syncing"

    tracker.on_join("usr_self")
    tracker.on_local_initialized()
    tracker.settle_if_quiet()

    snapshot = tracker.snapshot()
    assert snapshot.status == "ready"
    assert snapshot.participant_count == 1


def test_empty_roster_never_becomes_ready() -> None:
    tracker = VrchatScenePresenceTracker()
    tracker.on_transition()
    tracker.on_local_initialized()
    tracker.settle_if_quiet()

    snapshot = tracker.snapshot()
    assert snapshot.status == "syncing"
    assert snapshot.participant_count is None


def test_missing_local_evidence_never_becomes_ready() -> None:
    tracker = VrchatScenePresenceTracker()
    tracker.on_transition()
    tracker.on_join("usr_a")
    tracker.on_join("usr_b")
    tracker.settle_if_quiet()

    snapshot = tracker.snapshot()
    assert snapshot.status == "syncing"
    assert snapshot.participant_count is None


def test_left_room_clears_immediately_and_suppresses_residual_leaves() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()

    tracker.on_left_room()

    snapshot = tracker.snapshot()
    assert snapshot.status == "unavailable"
    assert snapshot.participant_count is None

    tracker.on_leave("usr_self")
    tracker.on_leave("usr_b")
    assert tracker.snapshot().status == "unavailable"

    tracker.on_idless_presence()
    assert tracker.snapshot().status == "unavailable"


def test_join_after_left_room_starts_new_syncing_roster() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()
    tracker.on_left_room()

    tracker.on_join("usr_self")
    tracker.on_local_initialized()
    tracker.settle_if_quiet()

    snapshot = tracker.snapshot()
    assert snapshot.status == "ready"
    assert snapshot.participant_count == 1


def test_idless_presence_degrades_syncing_roster() -> None:
    tracker = VrchatScenePresenceTracker()
    tracker.on_transition()
    tracker.on_join("usr_self")

    tracker.on_idless_presence()

    assert tracker.snapshot().status == "degraded"
    assert tracker.snapshot().participant_count is None


def test_file_failure_withholds_count_without_clearing_to_zero() -> None:
    tracker = VrchatScenePresenceTracker()
    _burst(tracker, "usr_self", "usr_b", local="usr_self")
    tracker.settle_if_quiet()

    tracker.on_file_failure()

    assert tracker.snapshot().status == "degraded"
    assert tracker.snapshot().participant_count is None


def test_removing_last_member_degrades_instead_of_zero() -> None:
    tracker = VrchatScenePresenceTracker()
    tracker.on_transition()
    tracker.on_join("usr_self")
    tracker.on_local_initialized()
    tracker.settle_if_quiet()
    assert tracker.snapshot().participant_count == 1

    tracker.on_leave("usr_self")

    assert tracker.snapshot().status == "degraded"
    assert tracker.snapshot().participant_count is None


def test_event_dispatch_routes_typed_events() -> None:
    from puripuly_heart.core.vrchat_scene_events import (
        IdlessPresence,
        InstanceTransition,
        LeftRoom,
        LocalPlayerInitialized,
        PlayerJoined,
        PlayerLeft,
    )

    tracker = VrchatScenePresenceTracker()
    tracker.on_event(InstanceTransition())
    tracker.on_event(PlayerJoined(user_id="usr_self"))
    tracker.on_event(LocalPlayerInitialized())
    tracker.settle_if_quiet()
    assert tracker.snapshot().participant_count == 1

    tracker.on_event(PlayerLeft(user_id="usr_ghost"))
    assert tracker.snapshot().status == "degraded"

    tracker.on_event(InstanceTransition())
    tracker.on_event(PlayerJoined(user_id="usr_self"))
    tracker.on_event(LocalPlayerInitialized())
    tracker.settle_if_quiet()
    assert tracker.snapshot().participant_count == 1

    tracker.on_event(IdlessPresence())
    assert tracker.snapshot().status == "degraded"

    tracker.on_event(LeftRoom())
    assert tracker.snapshot().status == "unavailable"
