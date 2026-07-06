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
import json
import os
import uuid
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSDefaultVoiceId, TTSModels

API_AUTH_HEADER = "x-api-key"
DEFAULT_BASE_URL = "wss://wsapi.deepdub.ai/open"
DEFAULT_STREAMING_URL = "wss://wss.deepdub.ai/ws"
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# LiveKit consumes raw PCM; s16le avoids the per-chunk WAV header Deepdub
# prepends when format is "wav"
AUDIO_FORMAT = "s16le"

# the streaming endpoint may split the input into several generations, each ending
# with its own isFinished message, and sends nothing after the last one; once input
# is closed and a generation finished, wait this long for another generation to
# start before treating the stream as complete
STREAM_END_GRACE = 1.0


@dataclass
class _TTSOptions:
    model: TTSModels | str
    locale: str
    voice_prompt_id: str
    sample_rate: int
    temperature: float | None
    variance: float | None
    tempo: float | None
    prompt_boost: bool
    realtime: bool
    accept_emojis: bool
    accent_base_locale: str | None
    accent_locale: str | None
    accent_ratio: float | None
    api_key: str
    base_url: str
    streaming_url: str

    def accent_control(self) -> dict[str, Any] | None:
        if (
            self.accent_base_locale is not None
            and self.accent_locale is not None
            and self.accent_ratio is not None
        ):
            return {
                "accentBaseLocale": self.accent_base_locale,
                "accentLocale": self.accent_locale,
                "accentRatio": self.accent_ratio,
            }
        return None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "dd-etts-3.3",
        locale: str = "en-US",
        voice_prompt_id: str = TTSDefaultVoiceId,
        sample_rate: int = 24000,
        temperature: float | None = None,
        variance: float | None = None,
        tempo: float | None = None,
        prompt_boost: bool = False,
        realtime: bool = True,
        accept_emojis: bool = False,
        accent_base_locale: str | None = None,
        accent_locale: str | None = None,
        accent_ratio: float | None = None,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str | None = None,
        streaming_url: str | None = None,
    ) -> None:
        """
        Create a new instance of Deepdub TTS.

        ``stream()`` uses Deepdub's streaming text-input WebSocket API, which accepts
        incremental text (e.g. LLM deltas) and segments it server side. ``synthesize()``
        uses the standard WebSocket TTS API with the full text in a single request.

        See https://docs.deepdub.ai/ for more details on the Deepdub API.

        Args:
            api_key (str, optional): The Deepdub API key. If not provided, it will be read from
                the DEEPDUB_API_KEY environment variable.
            model (TTSModels, optional): The Deepdub TTS model to use. Defaults to "dd-etts-3.3".
            locale (str, optional): The locale for synthesis, e.g. "en-US". Defaults to "en-US".
            voice_prompt_id (str, optional): The Deepdub voice prompt ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            temperature (float, optional): Sampling temperature, between 0.0 and 1.0.
            variance (float, optional): Expressivity variance, between 0.0 and 1.0.
            tempo (float, optional): Speed of speech, between 0.5 and 2.0.
            prompt_boost (bool, optional): Enhance similarity to the voice prompt.
            realtime (bool, optional): Request real-time generation priority for lower latency.
                Defaults to True, which suits live voice agents.
            accept_emojis (bool, optional): Whether emojis in the streamed text are voiced.
                Only applies to ``stream()``.
            accent_base_locale (str, optional): Base locale for accent control, e.g. "en-US".
            accent_locale (str, optional): Target accent locale, e.g. "fr-FR".
            accent_ratio (float, optional): Accent strength, between 0.0 and 1.0. All three
                accent parameters must be provided together.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp
                ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The Deepdub WebSocket TTS URL, used by ``synthesize()``.
                Defaults to the DEEPDUB_BASE_WEBSOCKET_URL environment variable or
                "wss://wsapi.deepdub.ai/open".
            streaming_url (str, optional): The Deepdub streaming text-input WebSocket URL, used
                by ``stream()``. Defaults to the DEEPDUB_BASE_WEBSOCKET_STREAMING_URL environment
                variable or "wss://wss.deepdub.ai/ws".
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        deepdub_api_key = api_key or os.environ.get("DEEPDUB_API_KEY")
        if not deepdub_api_key:
            raise ValueError(
                "Deepdub API key is required, either as argument or set"
                " DEEPDUB_API_KEY environment variable"
            )

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}")

        accent_args = (accent_base_locale, accent_locale, accent_ratio)
        if any(arg is not None for arg in accent_args) and None in accent_args:
            raise ValueError(
                "accent_base_locale, accent_locale and accent_ratio must be provided together"
            )

        self._opts = _TTSOptions(
            model=model,
            locale=locale,
            voice_prompt_id=voice_prompt_id,
            sample_rate=sample_rate,
            temperature=temperature,
            variance=variance,
            tempo=tempo,
            prompt_boost=prompt_boost,
            realtime=realtime,
            accept_emojis=accept_emojis,
            accent_base_locale=accent_base_locale,
            accent_locale=accent_locale,
            accent_ratio=accent_ratio,
            api_key=deepdub_api_key,
            base_url=base_url or os.environ.get("DEEPDUB_BASE_WEBSOCKET_URL") or DEFAULT_BASE_URL,
            streaming_url=streaming_url
            or os.environ.get("DEEPDUB_BASE_WEBSOCKET_STREAMING_URL")
            or DEFAULT_STREAMING_URL,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Deepdub"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, url: str, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(
            session.ws_connect(url, headers={API_AUTH_HEADER: self._opts.api_key}),
            timeout,
        )

    async def _connect_stream_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        # connect and consume the initial status so the socket is ready for stream-config
        ws = await self._connect_ws(self._opts.streaming_url, timeout)
        initial_msg = await ws.receive(timeout=timeout)
        if initial_msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
            await ws.close()
            raise APIConnectionError("Deepdub connection closed before the initial status message")
        status = json.loads(initial_msg.data)
        if status.get("action") == "error":
            await ws.close()
            raise APIError(f"Deepdub connection failed: {status.get('message')}")
        logger.debug(
            "established new Deepdub streaming connection",
            extra={"deepdub_connection_id": status.get("connectionId")},
        )
        return ws

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        locale: NotGivenOr[str] = NOT_GIVEN,
        voice_prompt_id: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        variance: NotGivenOr[float | None] = NOT_GIVEN,
        tempo: NotGivenOr[float | None] = NOT_GIVEN,
        prompt_boost: NotGivenOr[bool] = NOT_GIVEN,
        realtime: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model, locale, voice prompt,
        and generation controls. If any parameter is not provided, the existing value will be
        retained. Updates apply to subsequently created streams.

        Args:
            model (TTSModels, optional): The Deepdub TTS model to use.
            locale (str, optional): The locale for synthesis, e.g. "en-US".
            voice_prompt_id (str, optional): The Deepdub voice prompt ID.
            temperature (float, optional): Sampling temperature, between 0.0 and 1.0.
            variance (float, optional): Expressivity variance, between 0.0 and 1.0.
            tempo (float, optional): Speed of speech, between 0.5 and 2.0.
            prompt_boost (bool, optional): Enhance similarity to the voice prompt.
            realtime (bool, optional): Request real-time generation priority for lower latency.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(locale):
            self._opts.locale = locale
        if is_given(voice_prompt_id):
            self._opts.voice_prompt_id = voice_prompt_id
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(variance):
            self._opts.variance = variance
        if is_given(tempo):
            self._opts.tempo = tempo
        if is_given(prompt_boost):
            self._opts.prompt_boost = prompt_boost
        if is_given(realtime):
            self._opts.realtime = realtime

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


class ChunkedStream(tts.ChunkedStream):
    """Synthesize the full text in a single request using the WebSocket TTS API"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        generation_id = str(uuid.uuid4())
        request = {
            "action": "text-to-speech",
            "generationId": generation_id,
            "targetText": self._input_text,
            "model": self._opts.model,
            "voicePromptId": self._opts.voice_prompt_id,
            "locale": self._opts.locale,
            "temperature": self._opts.temperature,
            "variance": self._opts.variance,
            "tempo": self._opts.tempo,
            "promptBoost": self._opts.prompt_boost,
            "realtime": self._opts.realtime,
            "accentControl": self._opts.accent_control(),
            "format": AUDIO_FORMAT,
            "sampleRate": self._opts.sample_rate,
        }

        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._tts._connect_ws(self._opts.base_url, self._conn_options.timeout)
            await ws.send_str(json.dumps(request))

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                mime_type="audio/pcm",
            )

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Deepdub connection closed unexpectedly",
                        request_id=generation_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                # audio messages are JSON delivered in binary frames
                if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    logger.warning("unexpected Deepdub message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                if data.get("error"):
                    logger.error(
                        "Deepdub returned error",
                        extra={"deepdub_generation_id": generation_id, "error": data},
                    )
                    raise APIError(f"Deepdub returned error: {data}")
                if data.get("generationId") != generation_id:
                    continue
                if data.get("data"):
                    output_emitter.push(base64.b64decode(data["data"]))
                if data.get("isFinished"):
                    output_emitter.flush()
                    break
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                await ws.close()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text incrementally using Deepdub's streaming text-input API"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        input_sent_event = asyncio.Event()
        input_ended = asyncio.Event()

        async def _input_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # Deepdub's streaming endpoint accepts arbitrary text chunks and handles
            # segmentation server side, so LLM deltas are forwarded as-is
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel) or not data:
                    continue

                self._mark_started()
                await ws.send_str(json.dumps({"action": "stream-text", "data": {"text": data}}))
                input_sent_event.set()

            input_ended.set()
            await ws.send_str(json.dumps({"action": "end-stream"}))
            input_sent_event.set()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            segment_started = False
            generation_finished = False
            await input_sent_event.wait()
            while True:
                try:
                    timeout = (
                        STREAM_END_GRACE
                        if generation_finished and input_ended.is_set()
                        else self._conn_options.timeout
                    )
                    msg = await ws.receive(timeout=timeout)
                except asyncio.TimeoutError:
                    if generation_finished and input_ended.is_set():
                        output_emitter.end_input()
                        return
                    raise
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Deepdub connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                # audio messages are JSON delivered in binary frames
                if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    logger.warning("unexpected Deepdub message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                action = data.get("action")
                if action in ("pong", "status"):
                    continue
                if action == "error" or data.get("error"):
                    logger.error(
                        "Deepdub returned error",
                        extra={"error": data},
                    )
                    raise APIError(f"Deepdub returned error: {data}")

                if data.get("data"):
                    generation_finished = False
                    if not segment_started:
                        segment_started = True
                        output_emitter.start_segment(segment_id=request_id)
                    output_emitter.push(base64.b64decode(data["data"]))

                if data.get("isFinished"):
                    generation_finished = True

        config_pkt = {
            "action": "stream-config",
            "config": {
                "model": self._opts.model,
                "locale": self._opts.locale,
                "voicePromptId": self._opts.voice_prompt_id,
                "format": AUDIO_FORMAT,
                "sampleRate": self._opts.sample_rate,
                "acceptEmojis": self._opts.accept_emojis,
                "temperature": self._opts.temperature,
                "variance": self._opts.variance,
                "tempo": self._opts.tempo,
                "promptBoost": self._opts.prompt_boost,
                "realtime": self._opts.realtime,
                "accentControl": self._opts.accent_control(),
            },
        }
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            # one connection per call; the streaming session owns its own lifecycle
            ws = await self._tts._connect_stream_ws(self._conn_options.timeout)
            await ws.send_str(json.dumps(config_pkt))

            tasks = [
                asyncio.create_task(_input_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                # interrupted mid-generation: tell the server to stop generating.
                # cancellation has already been delivered once, so this await runs.
                with contextlib.suppress(Exception):
                    await ws.send_str(json.dumps({"action": "cancel"}))
                raise
            finally:
                input_sent_event.set()
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if ws is not None:
                await ws.close()
