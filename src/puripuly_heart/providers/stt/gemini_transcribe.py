"""Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from puripuly_heart.core.speech_boundary import SpeechBoundaryReason, boundary_wait_ms
from puripuly_heart.core.stt.backend import (
    RecoverableSTTSessionError,
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
)

logger = logging.getLogger(__name__)

GEMINI_TRANSCRIBE_STT_MODEL = "gemini-3.5-transcribe-live"
GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ = 16000
GEMINI_TRANSCRIBE_FINALIZE_TIMEOUT_S = 2.0


class GeminiTranscribeFinalizeTimeout(RecoverableSTTSessionError):
    pass


@dataclass(slots=True, eq=False)
class _PendingTurn:
    latest_interim: str = ""
    final_emitted: bool = False
    activity_end_ack: asyncio.Event = field(default_factory=asyncio.Event)
    timeout_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class _StartTurn:
    turn: _PendingTurn


@dataclass(frozen=True, slots=True)
class _EndTurn:
    turn: _PendingTurn


_STOP = object()


@dataclass(slots=True)
class _GeminiClientResources:
    client: Any
    sync_transport: Any
    async_transport: Any
    config: Any


def _build_live_config_sync(language_codes: Sequence[str], custom_vocabulary: Sequence[str]) -> Any:
    from google.genai import types

    transcription_config_kwargs: dict[str, Any] = {"mode": "VERBATIM"}
    if language_codes:
        transcription_config_kwargs["language_codes"] = list(language_codes)
    if custom_vocabulary:
        transcription_config_kwargs["custom_vocabulary"] = list(custom_vocabulary)
    return types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(**transcription_config_kwargs),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
    )


def _create_transports_sync() -> tuple[Any, Any]:
    import httpx

    sync_transport = httpx.Client(timeout=None, follow_redirects=True)
    try:
        async_transport = httpx.AsyncClient(timeout=None, follow_redirects=True)
    except BaseException:
        with contextlib.suppress(Exception):
            sync_transport.close()
        raise
    return sync_transport, async_transport


def _build_http_options_sync(sync_transport: Any, async_transport: Any) -> Any:
    from google.genai import types

    return types.HttpOptions(httpx_client=sync_transport, httpx_async_client=async_transport)


def _create_genai_client_sync(api_key: str, http_options: Any) -> Any:
    from google import genai

    return genai.Client(api_key=api_key, http_options=http_options)


def _prepare_gemini_resources_sync(
    api_key: str, language_codes: Sequence[str], custom_vocabulary: Sequence[str]
) -> _GeminiClientResources:
    config = _build_live_config_sync(language_codes, custom_vocabulary)
    sync_transport, async_transport = _create_transports_sync()
    try:
        http_options = _build_http_options_sync(sync_transport, async_transport)
        client = _create_genai_client_sync(api_key, http_options)
    except BaseException:
        with contextlib.suppress(Exception):
            sync_transport.close()
        with contextlib.suppress(Exception):
            asyncio.run(async_transport.aclose())
        raise
    return _GeminiClientResources(
        client=client,
        sync_transport=sync_transport,
        async_transport=async_transport,
        config=config,
    )


def gemini_transcribe_language_codes(source_language: str | None) -> list[str]:
    if not source_language:
        return []
    from puripuly_heart.core.language import gemini_transcribe_language_hint

    mapped = gemini_transcribe_language_hint(source_language)
    return [mapped] if mapped else []


def _recv_failure_fields(exc: BaseException) -> tuple[str, object, object, str]:
    exception_class = type(exc).__name__
    api_code = getattr(exc, "code", None)
    api_status = getattr(exc, "status", None)
    return exception_class, api_code, api_status, _recv_message_kind(exc, api_code, api_status)


def _recv_message_kind(exc: BaseException, api_code: object, api_status: object) -> str:
    class_name = type(exc).__name__.lower().replace("_", "")
    status_text = str(api_status or "").lower()
    if "goaway" in class_name or "go_away" in status_text:
        return "go_away"
    if (
        "connection" in class_name
        or "closed" in class_name
        or "websocket" in class_name
        or "unavailable" in status_text
    ):
        return "connection_closed"
    if (
        api_code in {400, 422}
        or "invalid" in status_text
        or "validation" in class_name
        or "invalidargument" in class_name
    ):
        return "validation"
    return "other"


@dataclass(slots=True)
class GeminiTranscribeSTTBackend(STTBackend):
    """Gemini 3.5 Transcribe Live STT Backend using the official google-genai SDK."""

    api_key: str
    language_codes: Sequence[str] = ()
    custom_vocabulary: Sequence[str] = ()
    model: str = GEMINI_TRANSCRIBE_STT_MODEL
    sample_rate_hz: int = GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ
    connect_timeout_s: float = 10.0
    finalize_timeout_s: float = GEMINI_TRANSCRIBE_FINALIZE_TIMEOUT_S
    live_connect_factory: Callable[[str, Any], Any] | None = None

    async def open_session(self) -> STTBackendSession:
        if self.sample_rate_hz != GEMINI_TRANSCRIBE_SAMPLE_RATE_HZ:
            raise ValueError(
                "sample_rate_hz must be 16000 for Gemini Transcribe Live transcription"
            )
        if not self.api_key:
            raise ValueError("api_key must be non-empty")
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.finalize_timeout_s <= 0:
            raise ValueError("finalize_timeout_s must be > 0")
        session = _GeminiTranscribeLiveSession(
            api_key=self.api_key,
            language_codes=list(self.language_codes),
            custom_vocabulary=list(self.custom_vocabulary),
            model=self.model,
            sample_rate_hz=self.sample_rate_hz,
            connect_timeout_s=self.connect_timeout_s,
            finalize_timeout_s=self.finalize_timeout_s,
            live_connect_factory=self.live_connect_factory,
        )
        try:
            await session.start()
        except BaseException:
            with contextlib.suppress(BaseException):
                await session.close()
            raise
        return session

    @staticmethod
    async def verify_api_key(api_key: str) -> bool:
        if not api_key:
            return False

        def _check() -> bool:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": api_key},
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status == 200
            except urllib.error.HTTPError as e:
                raise Exception(f"HTTP {e.code}: {e.reason}")
            except Exception as e:
                raise Exception(f"Connection failed: {e}")

        return await asyncio.to_thread(_check)


@dataclass(slots=True)
class _GeminiTranscribeLiveSession(STTBackendSession):
    """Internal session wrapping a google-genai Live API AsyncSession."""

    api_key: str
    language_codes: list[str]
    custom_vocabulary: list[str]
    model: str
    sample_rate_hz: int
    connect_timeout_s: float
    finalize_timeout_s: float
    live_connect_factory: Callable[[str, Any], Any] | None = None

    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False, repr=False
    )
    _send_queue: asyncio.Queue[_StartTurn | _EndTurn | bytes | object] = field(
        init=False, repr=False
    )
    _live_context: Any = field(init=False, default=None, repr=False)
    _live_session: Any = field(init=False, default=None, repr=False)
    _send_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _recv_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stopped: bool = field(init=False, default=False)
    _capture_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    _streaming_turn: _PendingTurn | None = field(init=False, default=None, repr=False)
    _pending_turns: deque[_PendingTurn] = field(init=False, default_factory=deque, repr=False)
    _protocol_failed: bool = field(init=False, default=False)
    _client_resources: _GeminiClientResources | None = field(init=False, default=None, repr=False)
    _setup_future: Any = field(init=False, default=None, repr=False)
    _setup_executor: Any = field(init=False, default=None, repr=False)
    _handshake_task: asyncio.Task[Any] | None = field(init=False, default=None, repr=False)
    _teardown_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _teardown_done: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._events = asyncio.Queue()
        self._send_queue = asyncio.Queue()

    async def start(self) -> None:
        if self._stopped:
            raise RuntimeError("Gemini Transcribe Live session is closed")
        self._teardown_done = False
        self._teardown_task = None
        self._setup_future = None
        self._setup_executor = None
        self._handshake_task = None
        setup_start = time.monotonic()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemini-stt-setup"
        )
        self._setup_executor = executor
        try:
            raw = executor.submit(
                _prepare_gemini_resources_sync,
                self.api_key,
                list(self.language_codes),
                list(self.custom_vocabulary),
            )
        except BaseException:
            self._setup_executor = None
            executor.shutdown(wait=False)
            raise
        self._setup_future = raw
        try:
            resources = await asyncio.wrap_future(raw)
        except BaseException:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise
        setup_elapsed = time.monotonic() - setup_start
        logger.info(
            "[STT] Gemini Transcribe Live setup completed in %.2fs",
            setup_elapsed,
        )
        if self._stopped:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise RuntimeError("Gemini Transcribe Live session closed during setup")
        self._client_resources = resources
        executor.shutdown(wait=False)
        try:
            factory = self.live_connect_factory
            if factory is None:
                factory = resources.client.aio.live.connect
            live_context = factory(model=self.model, config=resources.config)
        except BaseException:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise
        self._live_context = live_context
        handshake_start = time.monotonic()
        handshake_task = asyncio.create_task(live_context.__aenter__())
        self._handshake_task = handshake_task
        try:
            live_session = await asyncio.wait_for(handshake_task, timeout=self.connect_timeout_s)
        except BaseException:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise
        self._handshake_task = None
        handshake_elapsed = time.monotonic() - handshake_start
        logger.info(
            "[STT] Gemini Transcribe Live ready in %.2fs (setup=%.2fs handshake=%.2fs)",
            setup_elapsed + handshake_elapsed,
            setup_elapsed,
            handshake_elapsed,
        )
        if self._stopped:
            try:
                await self._teardown()
            except (asyncio.CancelledError, Exception):
                pass
            raise RuntimeError("Gemini Transcribe Live session closed during connect")
        self._live_session = live_session
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _teardown(self) -> None:
        task = self._teardown_task
        if task is None:
            if self._teardown_done:
                return
            task = asyncio.create_task(self._teardown_body())
            self._teardown_task = task
        if task is asyncio.current_task():
            return
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue

    async def _teardown_body(self) -> None:
        raw = self._setup_future
        if raw is not None and not raw.done():
            try:
                await asyncio.wrap_future(raw)
            except asyncio.CancelledError:
                body_task = asyncio.current_task()
                if body_task is not None and body_task.cancelling():
                    raise
            except Exception:
                pass
        outcome = None
        if raw is not None and raw.done():
            try:
                outcome = raw.result()
            except BaseException:
                outcome = None
        if outcome is not None and outcome is not self._client_resources:
            await self._close_resources(outcome)
        executor, self._setup_executor = self._setup_executor, None
        if executor is not None:
            executor.shutdown(wait=False)
        handshake_task, self._handshake_task = self._handshake_task, None
        if handshake_task is not None:
            if not handshake_task.done():
                handshake_task.cancel()
            try:
                await handshake_task
            except asyncio.CancelledError:
                body_task = asyncio.current_task()
                if body_task is not None and body_task.cancelling():
                    raise
            except Exception:
                pass
        for task in (self._send_task, self._recv_task):
            if task is not None:
                task.cancel()
        pending_tasks = [task for task in (self._send_task, self._recv_task) if task is not None]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._send_task = None
        self._recv_task = None
        live_context, self._live_context = self._live_context, None
        live_session, self._live_session = self._live_session, None
        if live_context is not None:
            with contextlib.suppress(Exception):
                await live_context.__aexit__(None, None, None)
        elif live_session is not None:
            with contextlib.suppress(Exception):
                await live_session.close()
        await self._release_client_resources()
        self._teardown_done = True

    async def _close_resources(self, resources: _GeminiClientResources) -> None:
        with contextlib.suppress(Exception):
            await resources.client.aio.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(resources.client.close)
        with contextlib.suppress(Exception):
            await resources.async_transport.aclose()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(resources.sync_transport.close)

    async def _release_client_resources(self) -> None:
        resources, self._client_resources = self._client_resources, None
        if resources is None:
            return
        await self._close_resources(resources)

    async def _send_loop(self) -> None:
        try:
            while not self._stopped:
                item = await self._send_queue.get()
                if item is _STOP:
                    return
                if isinstance(item, _StartTurn):
                    from google.genai import types

                    self._streaming_turn = item.turn
                    await self._send_realtime(activity_start=types.ActivityStart())
                    logger.info("[STT] Gemini Transcribe Live activityStart sent")
                    continue
                if isinstance(item, _EndTurn):
                    from google.genai import types

                    turn = item.turn
                    self._pending_turns.append(turn)
                    await self._send_realtime(activity_end=types.ActivityEnd())
                    if turn in self._pending_turns:
                        turn.timeout_task = asyncio.create_task(self._finalize_timeout(turn))
                    logger.info("[STT] Gemini Transcribe Live activityEnd sent (finalize)")
                    await turn.activity_end_ack.wait()
                    self._streaming_turn = None
                    if self._protocol_failed:
                        return
                    continue
                if isinstance(item, bytes):
                    await self._send_realtime(
                        audio={
                            "data": item,
                            "mime_type": f"audio/pcm;rate={self.sample_rate_hz}",
                        },
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Gemini Transcribe Live send loop error")
            self._put_event(exc)

    async def _recv_loop(self) -> None:
        try:
            while not self._stopped:
                live_session = self._live_session
                if live_session is None:
                    return
                async for message in live_session.receive():
                    self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            exception_class, api_code, api_status, message_kind = _recv_failure_fields(exc)
            logger.exception(
                "Gemini Transcribe Live recv loop error exception_class=%s "
                "api_code=%s api_status=%s message_kind=%s",
                exception_class,
                api_code,
                api_status,
                message_kind,
            )
            self._put_event(exc)
        finally:
            self._put_event(None)

    def _handle_message(self, message: Any) -> None:
        if self._protocol_failed:
            return
        content = message.server_content
        if content is not None:
            interim = content.interim_input_transcription
            if interim is not None and interim.text:
                text = str(interim.text)
                turn = self._turn_for_interim()
                if turn is not None:
                    turn.latest_interim = text
                logger.debug("[STT] Gemini Transcribe Live interim text_len=%s", len(text))
            final = content.input_transcription
            if final is not None:
                self._handle_final(str(final.text or ""))

        voice_activity = getattr(message, "voice_activity", None)
        activity_type = getattr(voice_activity, "voice_activity_type", None)
        activity_value = getattr(activity_type, "value", activity_type)
        if str(activity_value or "").upper() == "ACTIVITY_END":
            self._handle_activity_end_ack()

    def _turn_for_interim(self) -> _PendingTurn | None:
        return self._streaming_turn

    def _handle_final(self, text: str) -> None:
        turn = next((item for item in self._pending_turns if not item.final_emitted), None)
        if turn is None:
            logger.debug(
                "[STT] Gemini Transcribe Live final ignored without pending finalize text_len=%s",
                len(text),
            )
            return
        self._emit_turn_final(turn, text)

    def _handle_activity_end_ack(self) -> None:
        if not self._pending_turns:
            logger.debug("[STT] Gemini Transcribe Live activityEnd ack without pending finalize")
            return
        turn = self._pending_turns.popleft()
        self._cancel_turn_timeout(turn)
        if not turn.final_emitted:
            self._emit_turn_final(turn, turn.latest_interim)
        turn.activity_end_ack.set()

    def _emit_turn_final(self, turn: _PendingTurn, text: str) -> None:
        if turn.final_emitted:
            return
        turn.final_emitted = True
        if text:
            logger.info("[STT] Transcript final text_len=%s", len(text))
        else:
            logger.debug("[STT] Gemini Transcribe Live empty finalize ack")
        self._put_event(STTBackendTranscriptEvent(text=text, is_final=True))

    def _cancel_turn_timeout(self, turn: _PendingTurn) -> None:
        task = turn.timeout_task
        turn.timeout_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _finalize_timeout(self, turn: _PendingTurn) -> None:
        try:
            await asyncio.sleep(self.finalize_timeout_s)
        except asyncio.CancelledError:
            return
        if self._stopped or self._protocol_failed or turn not in self._pending_turns:
            return
        self._protocol_failed = True
        self._pending_turns.remove(turn)
        self._emit_turn_final(turn, turn.latest_interim)
        turn.activity_end_ack.set()
        logger.warning(
            "[STT] Gemini Transcribe Live finalize timed out after %.2fs; recycling session",
            self.finalize_timeout_s,
        )
        self._put_event(
            GeminiTranscribeFinalizeTimeout(
                f"Gemini Transcribe Live finalize timed out after {self.finalize_timeout_s:.2f}s"
            )
        )

    async def send_audio(self, pcm16le: bytes) -> None:
        if self._stopped or self._protocol_failed:
            return
        if self._capture_turn is None:
            self._capture_turn = _PendingTurn()
            await self._send_queue.put(_StartTurn(self._capture_turn))
        await self._send_queue.put(pcm16le)

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        if self._stopped:
            return
        observed_tail_ms = max(int(trailing_silence_ms or 0), 0)
        wait_ms = boundary_wait_ms(reason, observed_tail_ms=observed_tail_ms)
        logger.info(
            "[STT][Tail] provider=gemini_transcribe boundary_reason=%s observed_tail_ms=%s "
            "boundary_wait_ms=%s",
            reason,
            observed_tail_ms,
            wait_ms,
        )
        if self._capture_turn is not None:
            turn = self._capture_turn
            self._capture_turn = None
            await self._send_queue.put(_EndTurn(turn))

    async def _send_realtime(self, **kwargs: Any) -> None:
        live_session = self._live_session
        if live_session is None:
            return
        await live_session.send_realtime_input(**kwargs)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._send_queue.put(_STOP)
        if self._pending_turns:
            logger.warning(
                "[STT] Gemini Transcribe Live session closed with unresolved finalize requests count=%s",
                len(self._pending_turns),
            )
        for turn in self._pending_turns:
            self._cancel_turn_timeout(turn)
            turn.activity_end_ack.set()
        self._pending_turns.clear()
        self._capture_turn = None
        self._streaming_turn = None
        self._put_event(None)

    async def close(self) -> None:
        await self.stop()
        current_task = asyncio.current_task()
        try:
            await self._teardown()
        except (asyncio.CancelledError, Exception):
            pass
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _put_event(self, event: STTBackendTranscriptEvent | BaseException | None) -> None:
        self._events.put_nowait(event)
