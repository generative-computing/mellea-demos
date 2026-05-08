"""Pipecat STT service that calls a remote vLLM speech endpoint."""

import asyncio
import base64
import io
import json
import os
import time
import wave
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import SegmentedSTTService

from loguru import logger

VLLM_SPEECH_URL = os.environ.get("VLLM_SPEECH_URL", "http://localhost:8083")
VLLM_SPEECH_MODEL = os.environ.get("VLLM_SPEECH_MODEL", "ibm-granite/granite-speech-4.1-2b")
VLLM_SPEECH_BEARER_TOKEN = os.environ.get("VLLM_SPEECH_BEARER_TOKEN", "token-abc123")
VLLM_SPEECH_PATH = os.environ.get("VLLM_SPEECH_PATH", "/v1/chat/completions")
STT_KEYWORD_BIAS = [k.strip() for k in os.environ.get("STT_KEYWORD_BIAS", "Granite,Mellea").split(",") if k.strip()]


class HostedSTTService(SegmentedSTTService):
    """STT service that sends audio to a vLLM speech endpoint for transcription.

    Fires the HTTP request as a background task on VAD stop so frame processing
    (including VADUserStartedSpeakingFrame from a resumed utterance) is never
    blocked on the 300-500ms round-trip. If VAD flips back to speaking before
    the response arrives, the in-flight task is cancelled and any late-arriving
    response is suppressed via an epoch counter — this prevents stale partial
    transcripts from polluting the turn strategy's accumulated text.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session: aiohttp.ClientSession | None = None
        self._endpoint = f"{VLLM_SPEECH_URL.rstrip('/')}{VLLM_SPEECH_PATH}"
        self._active_stt_task: asyncio.Task | None = None
        self._utterance_epoch: int = 0

    def _build_stt_prompt(self) -> str:
        prompt = "transcribe the speech with proper capitalization and punctuation"
        if STT_KEYWORD_BIAS:
            prompt += ". The following terms may appear in the audio: " + ", ".join(STT_KEYWORD_BIAS)
        return prompt

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_SPEECH_BEARER_TOKEN}",
            }
        )

    async def stop(self, frame):
        await self._cancel_active_stt()
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def cancel(self, frame):
        await self._cancel_active_stt()
        await super().cancel(frame)

    async def _cancel_active_stt(self):
        if self._active_stt_task and not self._active_stt_task.done():
            self._active_stt_task.cancel()
            try:
                await self._active_stt_task
            except (asyncio.CancelledError, Exception):
                pass
        self._active_stt_task = None

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        # Bump epoch first so any late response from an in-flight request is suppressed.
        self._utterance_epoch += 1
        await self._cancel_active_stt()
        await super()._handle_user_started_speaking(frame)

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        content = io.BytesIO()
        wav = wave.open(content, "wb")
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.setframerate(self.sample_rate)
        wav.writeframes(self._audio_buffer)
        wav.close()
        content.seek(0)
        self._audio_buffer.clear()
        audio_bytes = content.read()

        await self._cancel_active_stt()

        epoch = self._utterance_epoch
        self._active_stt_task = asyncio.create_task(
            self.process_generator(self.run_stt(audio_bytes, epoch)),
            name=f"{self.name}::stt",
        )

    async def run_stt(
        self, audio: bytes, epoch: int | None = None
    ) -> AsyncGenerator[Frame, None]:
        if epoch is None:
            epoch = self._utterance_epoch

        if not self._session:
            yield ErrorFrame(error="STT session not initialized")
            return

        audio_b64 = base64.b64encode(audio).decode("utf-8")

        payload = {
            "model": VLLM_SPEECH_MODEL,
            "stream": True,
            "temperature": 0.0,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_stt_prompt(),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
        }

        try:
            t0 = time.monotonic()
            audio_kb = len(audio) / 1024
            async with self._session.post(self._endpoint, json=payload) as resp:
                t_response = time.monotonic()
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("STT request failed: {} {}", resp.status, error_text)
                    yield ErrorFrame(error=f"STT failed: {resp.status}")
                    return

                transcribed = ""
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or line_str == "data: [DONE]":
                        continue
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]
                    try:
                        chunk = json.loads(line_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            transcribed += content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

                t_done = time.monotonic()
                transcribed = transcribed.strip()

                if epoch != self._utterance_epoch:
                    logger.debug(
                        "Suppressing stale STT result (epoch {} != {}): {!r}",
                        epoch, self._utterance_epoch, transcribed[:40],
                    )
                    return

                if transcribed:
                    logger.info(
                        "Hosted STT transcription: {!r} | audio={:.0f}KB http={:.3f}s stream={:.3f}s total={:.3f}s",
                        transcribed, audio_kb,
                        t_response - t0, t_done - t_response, t_done - t0,
                    )
                    yield TranscriptionFrame(
                        text=transcribed,
                        user_id=self._user_id or "",
                        timestamp="",
                        language=None,
                    )
                else:
                    logger.warning("Hosted STT returned empty transcription (total={:.3f}s)", t_done - t0)

        except asyncio.CancelledError:
            logger.debug("Hosted STT cancelled (user resumed speaking)")
            raise
        except Exception as e:
            logger.exception("Hosted STT error")
            yield ErrorFrame(error=f"STT error: {e}")
