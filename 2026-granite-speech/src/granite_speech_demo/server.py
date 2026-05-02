"""FastAPI server with SmallWebRTC transport and pipecat pipeline."""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import numpy as np

import uvicorn
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.services.whisper.stt import WhisperSTTService, Model as WhisperModel

if sys.platform == "darwin":
    from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from granite_speech_demo.mellea_llm import (
    BEST_OF_N,
    GRANITE_SWITCH_ENABLED,
    IVR_REQUIREMENT_LABELS,
    MelleaLLMService,
)

from loguru import logger as loguru_logger

loguru_logger.remove()
loguru_logger.add(sys.stderr, level="INFO")
loguru_logger.add("logs/mellea_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day", retention="7 days")

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


HOST = os.environ.get("HOST", "localhost")
PORT = int(os.environ.get("PORT", "7860"))
_whisper_key = os.environ.get("WHISPER_MODEL", "medium")

if sys.platform == "darwin":
    _WHISPER_MODELS = {
        "tiny": MLXModel.TINY,
        "medium": MLXModel.MEDIUM,
        "large-v3": MLXModel.LARGE_V3,
        "large-v3-turbo": MLXModel.LARGE_V3_TURBO,
        "distil-large-v3": MLXModel.DISTIL_LARGE_V3,
    }
    WHISPER_MODEL = _WHISPER_MODELS.get(_whisper_key, MLXModel.MEDIUM)
else:
    _WHISPER_MODELS = {
        "tiny": WhisperModel.TINY,
        "medium": WhisperModel.MEDIUM,
        "large-v3": WhisperModel.LARGE,
        "large-v3-turbo": WhisperModel.LARGE_V3_TURBO,
        "distil-large-v3": WhisperModel.DISTIL_LARGE_V2,
    }
    WHISPER_MODEL = _WHISPER_MODELS.get(_whisper_key, WhisperModel.MEDIUM)
TTS_VOICE = os.environ.get("TTS_VOICE", "bf_emma")

pcs_map: Dict[str, SmallWebRTCConnection] = {}
active_sessions: Dict[str, Dict[str, Any]] = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]


def _warmup_whisper():
    silence = np.zeros(16000, dtype=np.float32)
    if sys.platform == "darwin":
        import mlx_whisper

        logger.info("Warming up MLX Whisper model (%s)...", WHISPER_MODEL)
        mlx_whisper.transcribe(silence, path_or_hf_repo=WHISPER_MODEL.value)
    else:
        from faster_whisper import WhisperModel as FWModel

        model_name = WHISPER_MODEL if isinstance(WHISPER_MODEL, str) else WHISPER_MODEL.value
        logger.info("Warming up faster-whisper model (%s)...", model_name)
        model = FWModel(model_name, device="auto", compute_type="default")
        model.transcribe(silence)
    logger.info("Whisper warm-up complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_warmup_whisper)
    yield
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


app = FastAPI(lifespan=lifespan)


async def run_bot(webrtc_connection: SmallWebRTCConnection, session_config: dict | None = None):
    logger.info("Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    if sys.platform == "darwin":
        stt = WhisperSTTServiceMLX(model=WHISPER_MODEL)
    else:
        stt = WhisperSTTService(model=WHISPER_MODEL)
    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=TTS_VOICE),
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )

    ivr_validation = (session_config or {}).get("ivr_validation")
    llm = MelleaLLMService(ivr_validation=ivr_validation)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)),
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.2)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/api/ivr/config")
async def ivr_config():
    """Static config the frontend needs before any turn runs — lets it pre-render
    the validation grid with the real requirement labels and sample count."""
    return {
        "requirements": IVR_REQUIREMENT_LABELS,
        "nSamples": BEST_OF_N,
        "available": GRANITE_SWITCH_ENABLED,
    }


@app.post("/api/offer")
async def offer(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    return await _handle_offer(data, background_tasks)


@app.post("/start")
async def rtvi_start(request: Request):
    """RTVI /start endpoint — creates a session ID for the prebuilt UI."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    session_config = request_data.get("body", {})
    if "ivrValidation" in request_data:
        session_config["ivr_validation"] = request_data["ivrValidation"]
    active_sessions[session_id] = session_config

    result = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_request(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """RTVI session proxy — routes /sessions/{id}/api/offer to the offer handler."""
    if session_id not in active_sessions:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    if path.endswith("api/offer"):
        data = await request.json()
        if request.method == "POST":
            session_config = active_sessions.get(session_id, {})
            return await _handle_offer(data, background_tasks, session_config=session_config)
        elif request.method == "PATCH":
            return await _handle_ice_candidate(data)

    return Response(status_code=200)


async def _handle_offer(data: dict, background_tasks: BackgroundTasks, session_config: dict | None = None):
    pc_id = data.get("pc_id")

    if pc_id and pc_id in pcs_map:
        conn = pcs_map[pc_id]
        await conn.renegotiate(
            sdp=data["sdp"],
            type=data["type"],
            restart_pc=data.get("restart_pc", False),
        )
    else:
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=data["sdp"], type=data["type"])

        @conn.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot, conn, session_config)

    answer = conn.get_answer()
    pcs_map[answer["pc_id"]] = conn
    return answer


async def _handle_ice_candidate(data: dict):
    from aiortc.sdp import candidate_from_sdp

    pc_id = data.get("pc_id")
    conn = pcs_map.get(pc_id)
    if not conn:
        return Response(content="Peer connection not found", status_code=404)

    for c in data.get("candidates", []):
        candidate = candidate_from_sdp(c["candidate"])
        candidate.sdpMid = c["sdp_mid"]
        candidate.sdpMLineIndex = c["sdp_mline_index"]
        await conn.add_ice_candidate(candidate)

    return {"status": "success"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mellea Pipecat Voice Server")
    parser.add_argument("--host", default=HOST, help=f"Host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--ssl-certfile", default=None, help="Path to SSL certificate file")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
