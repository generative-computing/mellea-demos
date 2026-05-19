"""Pipecat TTS service that calls a remote HTTP TTS server."""

import os
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.tts_service import TTSService

from loguru import logger

HOSTED_TTS_URL = os.environ.get("HOSTED_TTS_URL", "http://localhost:8086")
HOSTED_TTS_PATH = os.environ.get("HOSTED_TTS_PATH", "/synth")
HOSTED_TTS_SAMPLE_RATE = int(os.environ.get("HOSTED_TTS_SAMPLE_RATE", "24000"))


class HostedTTSService(TTSService):
    """TTS service that POSTs text to a remote HTTP TTS endpoint and streams audio back."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=HOSTED_TTS_SAMPLE_RATE,
            **kwargs,
        )
        self._session: aiohttp.ClientSession | None = None
        self._endpoint = f"{HOSTED_TTS_URL.rstrip('/')}{HOSTED_TTS_PATH}"

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not self._session:
            yield ErrorFrame(error="TTS session not initialized")
            return

        payload = {"text": text}

        try:
            await self.start_tts_usage_metrics(text)

            async with self._session.post(self._endpoint, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("TTS request failed: {} {}", resp.status, error_text)
                    yield ErrorFrame(error=f"TTS failed: {resp.status}")
                    return

                async for frame in self._stream_audio_frames_from_iterator(
                    resp.content.iter_chunked(1024),
                    in_sample_rate=HOSTED_TTS_SAMPLE_RATE,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame

        except Exception as e:
            logger.exception("Hosted TTS error")
            yield ErrorFrame(error=f"TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
