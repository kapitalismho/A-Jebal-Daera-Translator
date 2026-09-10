from __future__ import annotations

import pytest

from puripuly_heart.core.overlay.process import OverlayProcessManager


def _manager(
    instance_id: str = "overlay-test", target: str | None = "desktop"
) -> OverlayProcessManager:
    manager = OverlayProcessManager()
    manager.overlay_instance_id = instance_id
    manager.selected_target = target
    return manager


def _reveal_lost_evidence(instance_id: str = "overlay-test") -> dict[str, object]:
    return {
        "failure_reason": "window_reveal_lost",
        "startup_phase": "bounds_confirmed",
        "target": "desktop",
        "desktop_target": True,
        "generation": 1,
        "canonical_bounds": [320, 720, 1344, 320],
        "observed_bounds": [320, 720, 1344, 320],
        "bounds_drift": False,
        "title_confirmed": True,
        "visible_confirmed": False,
        "bounds_confirmed": True,
        "win32_error": None,
        "port_reason": "visible_bounds_not_retained",
        "hwnd": 4242,
        "hwnd_owner_pid": 4321,
        "owner_pid": 4321,
        "pid_file_pid": 4321,
        "endpoint_identity": "flet://desktop-overlay/1",
        "endpoint_matches": True,
        "overlay_instance_id": instance_id,
    }


def test_recovery_eligible_defaults_to_false() -> None:
    manager = _manager()
    assert manager.startup_failure_evidence is None
    assert manager.startup_recovery_eligible is False


def test_classic_reveal_loss_is_eligible() -> None:
    manager = _manager()
    assert manager._startup_recovery_eligible(_reveal_lost_evidence()) is True  # noqa: SLF001


def test_drifted_visibility_loss_is_eligible() -> None:
    manager = _manager()
    evidence = _reveal_lost_evidence()
    evidence["failure_reason"] = "window_visibility_unstable"
    evidence["observed_bounds"] = [400, 720, 1344, 320]
    evidence["bounds_confirmed"] = False
    evidence["bounds_drift"] = True
    assert manager._startup_recovery_eligible(evidence) is True  # noqa: SLF001


def test_missing_pid_file_remains_eligible() -> None:
    manager = _manager()
    evidence = _reveal_lost_evidence()
    evidence["pid_file_pid"] = None
    assert manager._startup_recovery_eligible(evidence) is True  # noqa: SLF001


def test_malformed_and_incomplete_evidence_rejected() -> None:
    manager = _manager()
    eligible = manager._startup_recovery_eligible  # noqa: SLF001
    assert eligible(None) is False
    assert eligible({}) is False
    assert eligible("window_reveal_lost") is False
    incomplete = _reveal_lost_evidence()
    del incomplete["hwnd"]
    assert eligible(incomplete) is False
    incomplete = _reveal_lost_evidence()
    del incomplete["overlay_instance_id"]
    assert eligible(incomplete) is False


def test_cross_instance_evidence_rejected() -> None:
    manager = _manager()
    assert (
        manager._startup_recovery_eligible(_reveal_lost_evidence("other")) is False
    )  # noqa: SLF001


def test_non_desktop_target_rejected() -> None:
    assert (
        _manager(target="steamvr")._startup_recovery_eligible(_reveal_lost_evidence()) is False
    )  # noqa: SLF001
    assert (
        _manager(target=None)._startup_recovery_eligible(_reveal_lost_evidence()) is False
    )  # noqa: SLF001


def test_nonrecoverable_reasons_rejected() -> None:
    manager = _manager()
    for reason in (
        "window_configuration_failed",
        "window_identity_failed",
        "window_observation_failed",
        "window_bounds_failed",
        "window_native_ready_failed",
        "unknown",
    ):
        evidence = _reveal_lost_evidence()
        evidence["failure_reason"] = reason
        assert manager._startup_recovery_eligible(evidence) is False  # noqa: SLF001


def test_equivocal_visibility_evidence_rejected() -> None:
    manager = _manager()
    cases = [
        {"title_confirmed": False},
        {"win32_error": 5},
        {"port_reason": "binding_changed"},
        {"hwnd": None},
        {"owner_pid": 4321, "hwnd_owner_pid": 9999},
        {"pid_file_pid": 9999},
        {"endpoint_matches": False},
        {"endpoint_matches": None},
        {"observed_bounds": None},
        {"canonical_bounds": [320, 720]},
        {"bounds_drift": None},
    ]
    for override in cases:
        evidence = _reveal_lost_evidence()
        evidence.update(override)
        assert manager._startup_recovery_eligible(evidence) is False  # noqa: SLF001


def test_reveal_lost_with_drift_rejected() -> None:
    manager = _manager()
    evidence = _reveal_lost_evidence()
    evidence["bounds_drift"] = True
    assert manager._startup_recovery_eligible(evidence) is False  # noqa: SLF001


def test_failure_evidence_extraction() -> None:
    manager = OverlayProcessManager()
    assert manager._extract_failure_evidence({"type": "startup_error"}) is None  # noqa: SLF001
    assert manager._extract_failure_evidence({"evidence": "nope"}) is None  # noqa: SLF001
    assert manager._extract_failure_evidence({"evidence": {"a": 1}}) is None  # noqa: SLF001
    evidence = manager._extract_failure_evidence(  # noqa: SLF001
        {"overlay_instance_id": "overlay-test", "evidence": {"a": 1}}
    )
    assert evidence == {"a": 1, "overlay_instance_id": "overlay-test"}
    assert (
        manager._extract_failure_evidence(  # noqa: SLF001
            {"overlay_instance_id": "", "evidence": {"a": 1}},
        )
        is None
    )
    assert (
        manager._extract_failure_evidence(  # noqa: SLF001
            {
                "overlay_instance_id": "overlay-test",
                "evidence": {"a": 1, "overlay_instance_id": "overlay-other"},
            },
        )
        is None
    )


def _live_reveal_lost_wire_body() -> dict[str, object]:
    return {
        "bounds_confirmed": True,
        "bounds_drift": False,
        "canonical_bounds": [0, 0, 1344, 336],
        "desktop_target": True,
        "endpoint_identity": "tcp://localhost:55092",
        "endpoint_matches": True,
        "failure_reason": "window_reveal_lost",
        "generation": 1,
        "hwnd": 24117844,
        "hwnd_owner_pid": 37944,
        "observed_bounds": [0, 0, 1344, 336],
        "owner_pid": 37944,
        "pid_file_pid": 37944,
        "port_reason": "visible_bounds_not_retained",
        "startup_phase": "bounds_confirmed",
        "target": "desktop",
        "title_confirmed": True,
        "visible_confirmed": False,
        "win32_error": None,
    }


def test_live_wire_envelope_without_nested_iid_is_admitted() -> None:
    manager = _manager(instance_id="overlay-33cc8d533427e1fa")
    event = {
        "type": "startup_error",
        "failure_reason": "window_reveal_lost",
        "startup_phase": "bounds_confirmed",
        "overlay_instance_id": "overlay-33cc8d533427e1fa",
        "evidence": _live_reveal_lost_wire_body(),
    }
    stored = manager._extract_failure_evidence(event)  # noqa: SLF001
    assert stored is not None
    assert stored["overlay_instance_id"] == "overlay-33cc8d533427e1fa"
    assert manager._startup_recovery_eligible(stored) is True  # noqa: SLF001


def test_live_wire_envelope_with_foreign_iid_is_rejected() -> None:
    manager = _manager(instance_id="overlay-33cc8d533427e1fa")
    event = {
        "type": "startup_error",
        "failure_reason": "window_reveal_lost",
        "startup_phase": "bounds_confirmed",
        "overlay_instance_id": "overlay-stale",
        "evidence": _live_reveal_lost_wire_body(),
    }
    stored = manager._extract_failure_evidence(event)  # noqa: SLF001
    assert stored is not None
    assert manager._startup_recovery_eligible(stored) is False  # noqa: SLF001


@pytest.mark.asyncio
async def test_renderer_envelope_feeds_manager_eligibility() -> None:
    from puripuly_heart.core.overlay.manifest import (
        OVERLAY_CONTRACT_VERSION,
        OverlayLaunchManifest,
    )
    from puripuly_heart.ui import desktop_overlay
    from puripuly_heart.ui.desktop_overlay_startup import (
        DesktopOverlayStartupCoordinator,
        DesktopOverlayStartupPhase,
    )
    from puripuly_heart.ui.desktop_window_zorder import WindowVisibilityConfirmation

    class _StubVisibilityPort:
        def bind_process(self, pid: int) -> None:
            return None

        async def confirm_window_visible(
            self, expected_title: str, *, x: int, y: int, width: int, height: int
        ) -> WindowVisibilityConfirmation:
            return WindowVisibilityConfirmation(
                confirmed=False,
                reason="visible_bounds_not_retained",
                hwnd=4242,
                title_confirmed=True,
                visible_confirmed=False,
                bounds_confirmed=True,
                observed_bounds=(320, 720, 1344, 320),
                hwnd_owner_pid=4321,
            )

        async def confirm_window_bounds(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("bounds gate is not part of this path")

        async def reassert_topmost_after_click_through(self) -> object:
            raise AssertionError("not part of this path")

        def close(self) -> None:
            return None

    async def _never_runner(target: object) -> None:
        raise AssertionError("app runner is not part of this path")

    class _StubViewOwner:
        process_info = (4321, None)
        endpoint_identity = "flet://desktop-overlay/1"

        async def close(self) -> None:
            return None

    window = desktop_overlay.FletDesktopRendererWindow(
        app_runner=_never_runner,  # type: ignore[arg-type]
        locale="en",
        window_z_order_port=_StubVisibilityPort(),  # type: ignore[arg-type]
        window_process_info_provider=lambda: (4321, None),
        view_process_owner=_StubViewOwner(),  # type: ignore[arg-type]
    )
    window.prime_startup_runtime_controls(
        ({"command": "apply_window_bounds", "x": 320, "y": 720, "width": 1344, "height": 320},)
    )
    coordinator = DesktopOverlayStartupCoordinator(1)
    coordinator.advance(DesktopOverlayStartupPhase.PAGE_CONFIGURED)
    coordinator.advance(DesktopOverlayStartupPhase.NATIVE_READY)
    coordinator.advance(DesktopOverlayStartupPhase.BOUNDS_CONFIRMED)
    window._startup_coordinator = coordinator  # noqa: SLF001
    window._startup_generation = 1  # noqa: SLF001
    window._bind_window_z_order_process()  # noqa: SLF001
    with pytest.raises(desktop_overlay.DesktopOverlayStartupError) as excinfo:
        await window._confirm_window_visible()  # noqa: SLF001
    assert excinfo.value.failure_reason == "window_reveal_lost"
    await window.close()

    manifest = OverlayLaunchManifest(
        contract_version=OVERLAY_CONTRACT_VERSION,
        app_version="test",
        overlay_instance_id="overlay-live",
        bridge_url="ws://127.0.0.1:1",
        session_token="test-session-token",
        parent_pid=1234,
        startup_deadline_ms=1000,
        log_dir="logs",
        log_level="INFO",
        locale="en",
        logging_mode="basic",
    )
    renderer = desktop_overlay.DesktopOverlayRenderer(manifest)
    envelope = renderer._startup_error_event(  # noqa: SLF001
        excinfo.value.failure_reason,
        startup_error=excinfo.value,
        default_phase=None,
    )
    assert "overlay_instance_id" not in (excinfo.value.evidence or {})
    assert envelope["evidence"]["overlay_instance_id"] == "overlay-live"

    manager = _manager(instance_id="overlay-live")
    stored = manager._extract_failure_evidence(envelope)  # noqa: SLF001
    assert stored is not None
    assert manager._startup_recovery_eligible(stored) is True  # noqa: SLF001
