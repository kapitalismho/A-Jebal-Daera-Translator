from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from puripuly_heart.app.services.settings_transaction_result import SettingsTransactionResultOwner
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
)

from puripuly_heart.app.adapters.self_capture.self_capture_provider import (
    SelfCaptureProviderAdapter,
)
from puripuly_heart.app.adapters.ui_runtime import (
    UiProviderRuntimeAdapter,
    UiSettingsRuntimeAdapter,
)
from puripuly_heart.app.ports.settings_view import ProviderApplyIntent, SelfSttProviderEdit
from puripuly_heart.app.services.canonical_settings_persistence import compose_settings_owner
from puripuly_heart.app.services.capture.self_capture_application import (
    SelfCaptureApplicationOwner,
    SelfCaptureApplicationSettings,
)
from puripuly_heart.app.services.provider.provider_settings import ProviderApplicationOwner
from puripuly_heart.app.services.ui_application import UiApplicationBoundary
from puripuly_heart.app.wiring.wiring_provider_runtime import compose_provider_runtime
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_self_capture_session_config_from_vnext,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmission,
    SelfCaptureAdmissionStatus,
)
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.core.vad.gating import SpeechEnd
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeGpuRuntimeFactory,
    FakeProvisioningPort,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


class _TransportSession:
    def __init__(self) -> None:
        self.closed = asyncio.Event()
        self.speech_ends: list[object] = []

    async def send_audio(self, _pcm16le: bytes) -> None:
        return None

    async def on_speech_end(self, *, trailing_silence_ms=None, reason=None) -> None:
        self.speech_ends.append((trailing_silence_ms, reason))

    async def stop(self) -> None:
        self.closed.set()

    async def close(self) -> None:
        self.closed.set()

    async def events(self):
        await self.closed.wait()
        if False:
            yield STTBackendTranscriptEvent(text="", is_final=True)


class _TransportBackend:
    def __init__(self) -> None:
        self.sessions: list[_TransportSession] = []

    async def open_session(self) -> _TransportSession:
        session = _TransportSession()
        self.sessions.append(session)
        return session


class _ManagedProviderFactory:

    def __init__(self) -> None:
        self.providers: list[ManagedSTTProvider] = []
        self.backends: list[object] = []

    async def create(self, request, *, gpu_runtime, on_terminal_failure=None):
        _ = gpu_runtime
        transport = _TransportBackend()
        self.backends.append(transport)
        if request.provider_id == STTProviderName.ROLLING_FREE.value:
            backend = RollingSTTBackend(
                providers=(
                    RollingProviderDefinition(
                        name=STTProviderName.GEMINI_TRANSCRIBE,
                        build_backend=lambda transport=transport: transport,
                        is_configured=lambda: True,
                    ),
                )
            )
        else:
            backend = transport
        provider = ManagedSTTProvider(
            backend=backend,
            sample_rate_hz=request.config.sample_rate_hz,
            stt_provider_name=STTProviderName(request.provider_id),
            channel=request.channel,
            clock=SystemClock(),
            reset_deadline_s=2.0,
            drain_timeout_s=0.2,
            bridging_ms=200,
            connect_attempts=1,
        )
        self.providers.append(provider)
        if request.session_options is not None:
            await provider.reconfigure_session_options(request.session_options)
        return provider


class _RuntimeFactory:
    def __init__(self) -> None:
        self.provider_factory = _ManagedProviderFactory()
        self.runtime: LocalASRProviderRuntimeOwner | None = None

    def create(self, callbacks: LocalASRProviderRuntimeCallbacks):
        self.runtime = LocalASRProviderRuntimeOwner(
            provider_factory=self.provider_factory,
            gpu_runtime_factory=FakeGpuRuntimeFactory(),
            provisioning=FakeProvisioningPort(),
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
        )
        return self.runtime


class _Admission:
    async def admit(self, _config):
        return SelfCaptureAdmission(SelfCaptureAdmissionStatus.ADMITTED)


class _Source:
    async def close(self) -> None:
        return None


class _ChannelReset:
    async def reset_provider_channel(self, _channel: str) -> None:
        return None


class _Peer:
    last_provider_signature = None
    last_runtime_signature = None
    last_intent_enabled = False
    last_activation_requested = False

    def effective_enabled(self) -> bool:
        return False

    def activation_requested(self, *, intent_enabled: bool, eula_accepted: bool) -> bool:
        return bool(intent_enabled and eula_accepted)


class _ManagedRelease:
    service = None

    async def rebuild(self, **_kwargs) -> None:
        return None


class _NoopPort:
    def __getattr__(self, _name):
        return _noop


async def _noop() -> None:
    return None


def _settings(provider: str) -> AppSettingsVNext:
    base = AppSettingsVNext()
    return replace(
        base,
        intent=replace(
            base.intent,
            stt=replace(base.intent.stt, provider=provider),
            languages=replace(base.intent.languages, source_language="ko"),
        ),
    )


@pytest.mark.asyncio
async def test_provider_apply_intent_full_vertical_rolling_gemini_soniox_reverse_and_speech_end(
    tmp_path: Path,
) -> None:
    settings_owner = compose_settings_owner(tmp_path / "settings.json")
    settings_owner.start()
    settings_owner.canonical = _settings(STTProviderName.ROLLING_FREE.value)
    harness_factory = _RuntimeFactory()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        local_asr_provider_runtime_factory=harness_factory,
    )
    runtime = harness_factory.runtime
    assert runtime is not None
    peer = _Peer()
    llm_runtime = ProviderRuntimeHandle(name="llm", provider=object())
    sources: list[_Source] = []

    async def source_factory(_config):
        source = _Source()
        sources.append(source)
        return source

    async def run_audio_loop(**_kwargs):
        await asyncio.Event().wait()

    def request_factory(config, _warmup):
        current = settings_owner.canonical
        assert current is not None
        return build_self_stt_provider_request_from_vnext(current)

    config_a = build_self_capture_session_config_from_vnext(settings_owner.canonical)
    capture_owner = SelfCaptureSessionOwner(
        admission=_Admission(),
        provider=SelfCaptureProviderAdapter(runtime, _ChannelReset()),
        provider_request_factory=request_factory,
        source_factory=source_factory,
        vad_factory=lambda _config: object(),
        run_audio_loop=run_audio_loop,
        vad_sink=harness.self_owner,
    )

    async def replace_self_stt(smooth: bool) -> None:
        await self_application.replace_provider(smooth_local=smooth)

    self_application = SelfCaptureApplicationOwner(
        settings_provider=lambda: SelfCaptureApplicationSettings(
            config=build_self_capture_session_config_from_vnext(settings_owner.canonical),
            provider_id=settings_owner.canonical.intent.stt.provider,
            qwen_region=None,
        ),
        runtime_available=lambda: True,
        capture_owner=lambda: capture_owner,
        capture_owner_if_created=lambda: capture_owner,
        persist_manual_fallback=lambda: True,
        reset_local_pending=lambda: None,
        clear_gpu_pending=lambda: None,
        overlay_state_provider=lambda: "off",
        mark_promo_eligible=lambda: None,
        dashboard_enabled_sink=lambda _value: None,
        dashboard_needs_key_sink=lambda _value: None,
        dashboard_needs_key=lambda _value: False,
        state_sink=lambda _snapshot: None,
        sync_effective_flags=lambda: None,
        sync_local_notice=lambda: None,
        log_basic=lambda _message: None,
        log_detailed=lambda _message, _level: None,
    )
    await capture_owner.apply_intent(config_a, enabled=True, explicit_toggle_off=False)
    source = capture_owner.source
    loop_task = capture_owner.loop_task
    assert source is sources[0]
    assert loop_task is not None
    initial_provider = runtime.current_provider("self")
    assert initial_provider is not None
    assert initial_provider.stt_provider_name is STTProviderName.ROLLING_FREE
    assert initial_provider._active_session.provider_name is STTProviderName.GEMINI_TRANSCRIBE

    components = compose_provider_runtime(
        config_path=tmp_path / "settings.json",
        settings=settings_owner,
        llm_runtime_provider=lambda: llm_runtime,
        local_asr_runtime_provider=lambda: runtime,
        translation_runtime_configuration_provider=lambda: None,
        self_capture_provider=lambda: capture_owner,
        self_capture_owner=lambda: capture_owner,
        peer=lambda: peer,
        peer_desired=lambda _settings: False,
        canonical_settings=lambda value: value,
        clear_local_pending=lambda: None,
        sync_local_notice=lambda: None,
        managed_pending_sink=lambda _value: None,
        managed_pending_provider=lambda: False,
        dashboard_managed_pending_sink=lambda _value: None,
        sync_effective_flags=lambda _settings: None,
        refresh_overlay=lambda: None,
        refresh_peer_runtime=_noop,
        replace_self_stt=replace_self_stt,
        self_state_sink=lambda _snapshot: None,
        self_availability=lambda snapshot: snapshot.provider_status.value == "ready",
        gpu_recovery=lambda _settings, _plan: _noop(),
        managed_release=lambda: _ManagedRelease(),
        managed_delegate_ready=lambda: None,
        runtime_logging=SimpleNamespace(),
        translation_needs_key_sink=lambda _value: None,
        usage_refresh=_noop,
        failure_sink=lambda _message: None,
        success_sink=lambda _message: None,
        additional_signature_sink=lambda _settings: None,
    )
    components.sync_signatures(settings_owner.canonical)

    results = SettingsTransactionResultOwner()
    provider_application = ProviderApplicationOwner(
        settings=settings_owner,
        runtime=components.runtime,
        merge_settings=copy.deepcopy,
        preserve_before_replace=lambda value: _preserve(value),
        sync_ui=lambda: None,
        order24_patch_provider=lambda _value: None,
        apply_order24=lambda _value: _false(),
        remember_order22=lambda _value: None,
        mutation_service_provider=lambda: None,
        save_failure_sink=lambda _message: None,
        results=results,
        sync_memory=lambda _value: None,
        capture_runtime_signatures=components.capture_signatures_before_canonical_mutation,
        sync_signatures=components.sync_signatures,
        consume_superseded_settings=lambda _value: False,
        active_local_asr_change=lambda _base, _next: False,
        compensate_local_asr=lambda **_kwargs: _noop(),
        llm_retry_pending=lambda: False,
        mark_llm_retry=lambda: None,
    )
    ui_settings = UiSettingsRuntimeAdapter(
        settings=settings_owner,
        projection=SimpleNamespace(),
        application=SimpleNamespace(),
        merge_provider_settings=lambda value: value,
        telemetry_enabled_settings=lambda value, _enabled: value,
    )
    ui_provider = UiProviderRuntimeAdapter(
        settings=settings_owner,
        provider_application=provider_application,
        gpu=SimpleNamespace(),
        managed=SimpleNamespace(),
        credential_verification=SimpleNamespace(),
        provider_settings=SimpleNamespace(),
        build_byok_target_settings=lambda _settings: None,
    )
    boundary = UiApplicationBoundary(
        startup=SimpleNamespace(),
        input_runtime=SimpleNamespace(),
        peer_capture=SimpleNamespace(),
        settings=ui_settings,
        provider=ui_provider,
        microphone=SimpleNamespace(),
        overlay=SimpleNamespace(),
        managed=SimpleNamespace(),
        engagement=SimpleNamespace(),
        diagnostics=SimpleNamespace(),
        state=SimpleNamespace(snapshot=SimpleNamespace()),
        runtime_shutdown=_NoopPort(),
        runtime_logging=SimpleNamespace(),
        settings_secrets=SimpleNamespace(),
        osc_state_publisher=lambda: None,
    )

    soniox_intent = ProviderApplyIntent((SelfSttProviderEdit(STTProviderName.SONIOX),))
    await boundary.apply_provider_intent(soniox_intent)
    assert settings_owner.canonical is not None
    assert settings_owner.canonical.intent.stt.provider == STTProviderName.SONIOX.value
    soniox_provider = runtime.current_provider("self")
    assert soniox_provider is not None
    assert soniox_provider.stt_provider_name is STTProviderName.SONIOX
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.SONIOX.value
    expected_soniox = build_self_capture_session_config_from_vnext(settings_owner.canonical)
    assert capture_owner.snapshot.runtime_signature == expected_soniox.runtime_signature
    assert capture_owner.snapshot.desired_active is True
    assert capture_owner.snapshot.effective_active is True
    assert capture_owner.source is source
    assert capture_owner.loop_task is loop_task

    rolling_intent = ProviderApplyIntent((SelfSttProviderEdit(STTProviderName.ROLLING_FREE),))
    soniox_provider._active_utterance_id = uuid4()
    apply_rolling = asyncio.create_task(boundary.apply_provider_intent(rolling_intent))
    for _ in range(1000):
        await asyncio.sleep(0.001)
        if runtime.snapshot.channel_for("self").pending_handoff or apply_rolling.done():
            break
    assert not apply_rolling.done()
    assert runtime.snapshot.channel_for("self").pending_handoff is True
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.SONIOX.value
    assert capture_owner.snapshot.provider_id == STTProviderName.SONIOX.value
    await harness.self_owner.handle_vad_event(SpeechEnd(uuid4()))
    await apply_rolling

    restored = runtime.current_provider("self")
    assert restored is not None
    assert restored.stt_provider_name is STTProviderName.ROLLING_FREE
    assert isinstance(restored.backend, RollingSTTBackend)
    assert restored.backend.providers[0].name is STTProviderName.GEMINI_TRANSCRIBE
    assert runtime.snapshot.channel_for("self").provider_id == STTProviderName.ROLLING_FREE.value
    assert settings_owner.canonical is not None
    assert settings_owner.canonical.intent.stt.provider == STTProviderName.ROLLING_FREE.value
    expected = build_self_capture_session_config_from_vnext(settings_owner.canonical)
    assert capture_owner.snapshot.runtime_signature == expected.runtime_signature
    assert capture_owner.snapshot.desired_active is True
    assert capture_owner.snapshot.effective_active is True
    assert capture_owner.source is source
    assert capture_owner.loop_task is loop_task

    await capture_owner.close()
    await runtime.close()


async def _preserve(value):
    return value


async def _false() -> bool:
    return False
