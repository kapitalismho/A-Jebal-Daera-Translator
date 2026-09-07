from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import uuid4

from puripuly_heart.app.services.overlay.overlay_application import (
    OverlayApplicationOwner,
    OverlayApplicationState,
)
from puripuly_heart.app.services.peer_application import PeerApplicationSnapshot
from puripuly_heart.app.services.translation_enable import (
    TranslationEnableOwner,
    TranslationEnableState,
)
from puripuly_heart.app.wiring.wiring_translation_runtime_configuration import (
    replace_translation_runtime_enabled,
)
from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.config.resolved import ResolvedOverlayConfig
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript


async def _noop_async() -> None:
    return None


async def _noop_renderer(queue: Any, overlay_instance_id: str) -> None:
    _ = queue, overlay_instance_id


class RecordingBridge:
    def __init__(self) -> None:
        self.snapshots: list[object] = []

    async def replace_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    async def broadcast_shutdown(self) -> None:
        return None


def _overlay_config(target: str = "steamvr") -> ResolvedOverlayConfig:
    return ResolvedOverlayConfig(
        enabled=True,
        target=target,
        show_translation=True,
        show_peer_original=True,
        calibration={},
        desktop_overlay_options={},
    )


def _peer_snapshot() -> PeerApplicationSnapshot:
    return PeerApplicationSnapshot(
        intent_enabled=False,
        activation_requested=False,
        effective_enabled=False,
        desired_active=False,
        activation_generation=0,
        activation_starting=False,
        model_loading=False,
        process_warning_reason=None,
        runtime_signature=None,
        provider_signature=None,
    )


def make_owner(
    config_owner: TranslationRuntimeConfigurationOwner,
) -> OverlayApplicationOwner:
    return OverlayApplicationOwner(
        state_provider=lambda: OverlayApplicationState(
            settings_available=True,
            overlay_intent_enabled=True,
            configured_target="steamvr",
            locale="en",
        ),
        config_provider=lambda: _overlay_config(),
        overlay_intent_sink=lambda _enabled: None,
        output_provider=lambda: None,
        diagnostics_provider=lambda: None,
        peer_snapshot_provider=_peer_snapshot,
        disable_peer_intent=lambda: None,
        sync_peer_effective=lambda: None,
        cancel_peer_activation=lambda: None,
        refresh_peer_dependencies=_noop_async,
        presentation_sink=lambda _state: None,
        state_sink=lambda _state, _reason: None,
        fallback_notice_sink=lambda _active: None,
        cancel_bounds_persistence=_noop_async,
        clear_bounds_suppressed=lambda: None,
        calibration_provider=lambda: cast(OverlayCalibration, OverlayCalibration()),
        logging_mode_provider=lambda: "basic",
        log_dir_provider=lambda: "",
        desktop_controls_factory=lambda _config: [],
        interaction_mode_sink=lambda _mode: None,
        bounds_control_sink=lambda _control: None,
        renderer_event_consumer=_noop_renderer,
        edit_interaction_mode="edit",
        clock=FakeClock(_now=0.0),
        log_basic=lambda _message, _level: None,
        log_detailed=lambda _message, _level, _exception: False,
        translation_enabled_provider=(lambda: config_owner.snapshot().value.translation_enabled),
    )


def attach_live_presenter(
    owner: OverlayApplicationOwner,
    config_owner: TranslationRuntimeConfigurationOwner,
) -> tuple[Any, OverlayPresenter, RecordingBridge]:
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        translation_enabled=config_owner.snapshot().value.translation_enabled,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    runtime = owner.new_runtime()
    runtime.set_overlay_instance_id("overlay-test")
    runtime.adopt_presenter(cast(Any, presenter))
    presenter.attach_bridge(cast(Any, bridge))
    owner.state = "connected"
    return runtime, presenter, bridge


def make_enable_owner(
    config_owner: TranslationRuntimeConfigurationOwner,
    overlay_owner: OverlayApplicationOwner,
    llm_available: bool,
) -> tuple[TranslationEnableOwner, list[bool]]:
    dashboard_values: list[bool] = []
    state_box = {"llm_available": llm_available}

    def state_provider() -> TranslationEnableState:
        return TranslationEnableState(
            runtime_available=True,
            translation_enabled=config_owner.snapshot().value.translation_enabled,
            llm_available=state_box["llm_available"],
            settings_available=True,
            provider_name="gemini",
            qwen_region=None,
            managed_selected=False,
            managed_china=False,
            managed_local_key_available=False,
            managed_release_service_available=False,
            ingress_frozen=False,
        )

    def runtime_sink(enabled: bool) -> None:
        replace_translation_runtime_enabled(config_owner, enabled)
        overlay_owner.notify_translation_runtime_state_changed()

    async def warmup() -> None:
        return None

    async def teardown() -> None:
        return None

    owner = TranslationEnableOwner(
        state_provider=state_provider,
        managed_prepare=_unexpected_prepare,
        founder_route=_unexpected_founder_route,
        pending_sink=lambda _pending: None,
        runtime_ensurer=_unexpected_ensurer,
        usage_refresh_sink=lambda: None,
        usage_refresh_now=_noop_async,
        runtime_sink=runtime_sink,
        dashboard_sink=dashboard_values.append,
        clear_context=lambda: None,
        warmup=warmup,
        message_sink=lambda _key, _values: None,
        qq_dialog_sink=lambda: None,
        result_sink=lambda _result: None,
        log_basic=lambda _message: None,
        log_detailed=lambda _message: None,
        log_error=lambda _message: None,
        founder_letter_sink=lambda: None,
        teardown=teardown,
    )
    return owner, dashboard_values


async def _unexpected_prepare() -> Any:
    raise AssertionError("managed prepare must not run when managed is not selected")


async def _unexpected_founder_route() -> bool:
    raise AssertionError("founder route must not run in these tests")


async def _unexpected_ensurer(_mode: str) -> bool:
    raise AssertionError("runtime ensurer must not run in these tests")


async def _drain() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def test_disabled_startup_first_snapshot_is_source_primary() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=False)
    )
    owner = make_owner(config_owner)
    request = owner._generation_request()
    assert request.translation_enabled is False
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        translation_enabled=request.translation_enabled,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    live_id = uuid4()
    await presenter.emit(
        adapter.peer_active_update(
            text="peer live caption",
            utterance_id=live_id,
            occupant_key=f"peer:{live_id}",
            source_language="en",
            target_language="ko",
            created_at=10.0,
        )
    )
    live_block = presenter.snapshot().blocks[0]
    assert live_block.block_variant == "active_peer"
    assert live_block.primary_text == "peer live caption"
    assert live_block.secondary_text == ""
    assert live_block.secondary_enabled is False
    final_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=final_id,
                channel="peer",
                text="peer finalized source",
                is_final=True,
                created_at=10.1,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    blocks = {block.id: block for block in presenter.snapshot().blocks}
    final_block = blocks[f"peer:{final_id}"]
    assert final_block.block_variant == "finalized"
    assert final_block.primary_text == "peer finalized source"
    assert final_block.secondary_text == ""
    assert final_block.secondary_enabled is False


async def test_restart_reuse_rebuilds_visible_row_as_source_primary() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    _ = runtime
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source toggle",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation toggle",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    before = presenter.snapshot().blocks[0]
    assert before.primary_text == "peer translation toggle"
    replace_translation_runtime_enabled(config_owner, False)
    request = owner._generation_request()
    assert request.translation_enabled is False
    snapshots_before = len(bridge.snapshots)
    await presenter.update_translation_enabled(request.translation_enabled)
    after = presenter.snapshot().blocks[0]
    assert after.primary_text == "peer source toggle"
    assert after.secondary_text == ""
    assert after.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before + 1


async def test_accepted_toggle_round_trip_rebuilds_without_new_speech() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source toggle",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation toggle",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=True)
    snapshots_before = len(bridge.snapshots)
    assert await enable_owner.set_enabled(False) is False
    await _drain()
    off_block = presenter.snapshot().blocks[0]
    assert off_block.primary_text == "peer source toggle"
    assert off_block.secondary_text == ""
    assert off_block.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before + 1
    assert await enable_owner.set_enabled(True) is True
    await _drain()
    on_block = presenter.snapshot().blocks[0]
    assert on_block.primary_text == "peer translation toggle"
    assert on_block.secondary_text == "peer source toggle"
    assert on_block.secondary_enabled is True


async def test_rejected_enable_keeps_source_primary_without_republish() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=False)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer finalized source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    assert presenter.snapshot().blocks[0].primary_text == "peer finalized source"
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=False)
    snapshots_before = len(bridge.snapshots)
    assert await enable_owner.set_enabled(True) is False
    assert config_owner.snapshot().value.translation_enabled is False
    await _drain()
    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "peer finalized source"
    assert block.secondary_text == ""
    assert block.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before


async def test_managed_exhaustion_rebuilds_visible_row_as_source_primary() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source toggle",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation toggle",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    enable_owner, _dashboard = make_enable_owner(config_owner, owner, llm_available=True)
    snapshots_before = len(bridge.snapshots)
    enable_owner.disable_for_managed_exhaustion(reopen_founder_letter=False)
    assert config_owner.snapshot().value.translation_enabled is False
    await _drain()
    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "peer source toggle"
    assert block.secondary_text == ""
    assert block.secondary_enabled is False
    assert len(bridge.snapshots) == snapshots_before + 1


async def test_rapid_toggles_converge_to_latest_snapshot() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    _runtime, presenter, _bridge = attach_live_presenter(owner, config_owner)
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer finalized source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    replace_translation_runtime_enabled(config_owner, False)
    owner.notify_translation_runtime_state_changed()
    replace_translation_runtime_enabled(config_owner, True)
    owner.notify_translation_runtime_state_changed()
    await _drain()
    assert presenter.translation_enabled is True
    block = presenter.snapshot().blocks[0]
    assert block.primary_text == ""
    assert block.secondary_text == "peer finalized source"
    assert block.secondary_enabled is True


async def test_stale_and_closed_updates_are_ignored() -> None:
    config_owner = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(translation_enabled=True)
    )
    owner = make_owner(config_owner)
    runtime, old_presenter, _bridge = attach_live_presenter(owner, config_owner)
    replace_translation_runtime_enabled(config_owner, False)
    owner.notify_translation_runtime_state_changed()
    new_presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        translation_enabled=True,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    runtime.adopt_presenter(cast(Any, new_presenter))
    runtime.set_overlay_instance_id("overlay-next")
    await _drain()
    assert old_presenter.translation_enabled is True
    assert new_presenter.translation_enabled is True
    await runtime.close(
        preserve_presenter_state=True,
        overlay_sink_detach=None,
        preview_reset=None,
        diagnostics_detach=None,
        emit_shutdown=False,
    )
    replace_translation_runtime_enabled(config_owner, False)
    owner.notify_translation_runtime_state_changed()
    await _drain()
    assert new_presenter.translation_enabled is True
