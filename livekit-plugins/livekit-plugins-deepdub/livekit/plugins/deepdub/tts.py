# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import contextlib
import weakref
from dataclasses import dataclass, replace
from typing import Any

from deepdub import DeepdubClient  # type: ignore[import-untyped]
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import TTSDefaultVoiceId, TTSModels

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# LiveKit consumes raw PCM; s16le avoids the per-chunk WAV header Deepdub
# prepends when format is "wav"
AUDIO_FORMAT = "s16le"

# end-stream is a flush, not a terminal command: each flush yields audio ending
# with isFinished, then the socket stays open. The server sends nothing after the
# final flush's audio, so once input is closed and a generation finished we wait
# this long for more before treating the stream as complete.
# ponytail: grace timeout; a distinct terminal signal from the server would remove it.
STREAM_END_GRACE = 1.0


class _StreamConn:
    """A single-use streaming connection, opened (connect + status) ahead of time so the
    ~1s handshake is off the hot path. Still one connection per call: used once, then closed.
    """

    def __init__(self, cm: Any, conn: DeepdubClient) -> None:
        self._cm = cm
        self.conn = conn

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._cm.__aexit__(None, None, None)


@dataclass
class _TTSOptions:
    model: TTSModels | str
    locale: str
    voice_prompt_id: str
    sample_rate: int
    realtime: bool
    accept_emojis: bool
    # one-shot (synthesize) only; the streaming service ignores these
    temperature: float | None
    variance: float | None
    tempo: float | None
    prompt_boost: bool
    accent_base_locale: str | None
    accent_locale: str | None
    accent_ratio: float | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "dd-etts-3.3",
        locale: str = "en-US",
        voice_prompt_id: str = TTSDefaultVoiceId,
        sample_rate: int = 24000,
        realtime: bool = True,
        accept_emojis: bool = False,
        temperature: float | None = None,
        variance: float | None = None,
        tempo: float | None = None,
        prompt_boost: bool = False,
        accent_base_locale: str | None = None,
        accent_locale: str | None = None,
        accent_ratio: float | None = None,
        base_url: str | None = None,
        streaming_url: str | None = None,
    ) -> None:
        """
        Create a new instance of Deepdub TTS.

        ``stream()`` uses Deepdub's streaming text-input API, feeding incremental text
        (e.g. LLM deltas) and flushing each segment. ``synthesize()`` uses the one-shot
        WebSocket TTS API with the full text in a single request. Both go through the
        official ``deepdub`` SDK.

        See https://docs.deepdub.ai/ for more details on the Deepdub API.

        Args:
            api_key (str, optional): The Deepdub API key. Falls back to the DEEPDUB_API_KEY
                environment variable.
            model (TTSModels, optional): The Deepdub TTS model to use. Defaults to "dd-etts-3.3".
            locale (str, optional): The locale for synthesis, e.g. "en-US". Defaults to "en-US".
            voice_prompt_id (str, optional): The Deepdub voice prompt ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            realtime (bool, optional): Request real-time generation priority for lower latency.
                Defaults to True, which suits live voice agents.
            accept_emojis (bool, optional): Whether emojis in the streamed text are voiced.
                Only applies to ``stream()``.
            temperature (float, optional): Sampling temperature. ``synthesize()`` only; the
                streaming service ignores it.
            variance (float, optional): Expressivity variance. ``synthesize()`` only.
            tempo (float, optional): Speed of speech, between 0.5 and 2.0. ``synthesize()`` only.
            prompt_boost (bool, optional): Enhance similarity to the voice prompt.
                ``synthesize()`` only.
            accent_base_locale (str, optional): Base locale for accent control. ``synthesize()``
                only; all three accent parameters must be provided together.
            accent_locale (str, optional): Target accent locale. ``synthesize()`` only.
            accent_ratio (float, optional): Accent strength, between 0.0 and 1.0. ``synthesize()``
                only.
            base_url (str, optional): The one-shot WebSocket TTS URL. Falls back to
                DEEPDUB_BASE_WEBSOCKET_URL or the SDK default.
            streaming_url (str, optional): The streaming text-input WebSocket URL. Falls back to
                DEEPDUB_BASE_WEBSOCKET_STREAMING_URL or the SDK default.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}")

        accent_args = (accent_base_locale, accent_locale, accent_ratio)
        if any(arg is not None for arg in accent_args) and None in accent_args:
            raise ValueError(
                "accent_base_locale, accent_locale and accent_ratio must be provided together"
            )

        # the SDK reads DEEPDUB_API_KEY and picks endpoint defaults; it raises if no key
        self._client = DeepdubClient(
            api_key=api_key,
            base_websocket_url=base_url,
            base_websocket_streaming_url=streaming_url,
        )
        self._opts = _TTSOptions(
            model=model,
            locale=locale,
            voice_prompt_id=voice_prompt_id,
            sample_rate=sample_rate,
            realtime=realtime,
            accept_emojis=accept_emojis,
            temperature=temperature,
            variance=variance,
            tempo=tempo,
            prompt_boost=prompt_boost,
            accent_base_locale=accent_base_locale,
            accent_locale=accent_locale,
            accent_ratio=accent_ratio,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._warm_task: asyncio.Task[_StreamConn] | None = None

    async def _open_stream_conn(self) -> _StreamConn:
        # connect + read the initial status so the socket is ready for stream-config
        cm = self._client.async_connect(streaming_input=True)
        conn: DeepdubClient = await cm.__aenter__()
        try:
            status = await conn._stream_recv_json()
            if status and status.get("action") == "error":
                raise APIError(f"Deepdub connection failed: {status.get('message')}")
        except BaseException:
            await cm.__aexit__(None, None, None)
            raise
        return _StreamConn(cm, conn)

    def _ensure_warm(self) -> None:
        if self._warm_task is None:
            self._warm_task = asyncio.create_task(self._open_stream_conn())

    async def _acquire_stream_conn(self) -> _StreamConn:
        # hand off the prewarmed conn and immediately start opening the next one, so the
        # handshake stays off the hot path while each call still gets its own connection
        self._ensure_warm()
        assert self._warm_task is not None
        task = self._warm_task
        self._warm_task = None
        self._ensure_warm()
        try:
            return await task
        except Exception:
            # prewarm failed (e.g. idle socket dropped); open fresh on the hot path
            return await self._open_stream_conn()

    def prewarm(self) -> None:
        self._ensure_warm()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepdub"

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        locale: NotGivenOr[str] = NOT_GIVEN,
        voice_prompt_id: NotGivenOr[str] = NOT_GIVEN,
        realtime: NotGivenOr[bool] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        variance: NotGivenOr[float | None] = NOT_GIVEN,
        tempo: NotGivenOr[float | None] = NOT_GIVEN,
        prompt_boost: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS configuration. Unset parameters keep their current value; updates
        apply to subsequently created streams.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(locale):
            self._opts.locale = locale
        if is_given(voice_prompt_id):
            self._opts.voice_prompt_id = voice_prompt_id
        if is_given(realtime):
            self._opts.realtime = realtime
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(variance):
            self._opts.variance = variance
        if is_given(tempo):
            self._opts.tempo = tempo
        if is_given(prompt_boost):
            self._opts.prompt_boost = prompt_boost

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._warm_task is not None:
            self._warm_task.cancel()
            with contextlib.suppress(BaseException):
                warm = await self._warm_task
                await warm.aclose()
            self._warm_task = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize the full text in a single request using the one-shot WebSocket TTS API"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._opts
        try:
            async with self._tts._client.async_connect() as conn:
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )
                async for chunk in conn.async_tts(
                    text=self._input_text,
                    voice_prompt_id=opts.voice_prompt_id,
                    model=opts.model,
                    locale=opts.locale,
                    temperature=opts.temperature,
                    variance=opts.variance,
                    tempo=opts.tempo,
                    prompt_boost=opts.prompt_boost,
                    accent_base_locale=opts.accent_base_locale,
                    accent_locale=opts.accent_locale,
                    accent_ratio=opts.accent_ratio,
                    format=AUDIO_FORMAT,
                    sample_rate=opts.sample_rate,
                    realtime=opts.realtime,
                ):
                    output_emitter.push(chunk)
                output_emitter.flush()
        except APITimeoutError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text incrementally using Deepdub's streaming text-input API"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._opts
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        input_started = asyncio.Event()
        input_ended = asyncio.Event()

        async def _input_task(conn: DeepdubClient) -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    # end-stream is a flush: generate the buffered text now, keep the socket
                    await conn.async_stream_end()
                    continue
                if not data:
                    continue
                self._mark_started()
                await conn.async_stream_text(data)
                input_started.set()
            # end_input() already sent a trailing flush sentinel, so the final
            # end-stream has been issued above
            input_ended.set()
            input_started.set()

        async def _recv_task(conn: DeepdubClient) -> None:
            segment_started = False
            generation_finished = False
            await input_started.wait()
            while True:
                try:
                    timeout = (
                        STREAM_END_GRACE
                        if generation_finished and input_ended.is_set()
                        else self._conn_options.timeout
                    )
                    resp = await asyncio.wait_for(conn.async_stream_recv(), timeout)
                except asyncio.TimeoutError:
                    if generation_finished and input_ended.is_set():
                        output_emitter.end_input()
                        return
                    raise
                if resp is None:
                    continue
                if resp.get("action") in ("pong", "status"):
                    continue
                if resp.get("error"):
                    raise APIError(f"Deepdub returned error: {resp}")
                if resp.get("data"):
                    generation_finished = False
                    if not segment_started:
                        segment_started = True
                        output_emitter.start_segment(segment_id=request_id)
                    output_emitter.push(base64.b64decode(resp["data"]))
                if resp.get("isFinished"):
                    generation_finished = True

        stream_conn: _StreamConn | None = None
        try:
            # prewarmed: handshake already done off the hot path; config is sent per-call
            stream_conn = await self._tts._acquire_stream_conn()
            conn = stream_conn.conn
            await conn.async_stream_config(
                model=opts.model,
                locale=opts.locale,
                voice_prompt_id=opts.voice_prompt_id,
                format=AUDIO_FORMAT,
                sample_rate=opts.sample_rate,
                accept_emojis=opts.accept_emojis,
                realtime=opts.realtime,
            )
            tasks = [
                asyncio.create_task(_input_task(conn)),
                asyncio.create_task(_recv_task(conn)),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                # interrupted mid-generation: tell the server to stop
                with contextlib.suppress(Exception):
                    await conn.async_stream_cancel()
                raise
            finally:
                input_started.set()
                await utils.aio.gracefully_cancel(*tasks)
        except (APIError, APITimeoutError, asyncio.CancelledError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if stream_conn is not None:
                await stream_conn.aclose()
