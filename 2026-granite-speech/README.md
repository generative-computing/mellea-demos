# granite-speech-demo

Real-time voice conversation demo built on [Granite Speech](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) for transcription, [Granite Switch](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview) for validated LLM generation via [Mellea](https://github.com/generative-computing/mellea)'s `requirement_check` intrinsics, and Pipecat for pipeline orchestration.

```
Browser mic → WebRTC → Silero VAD → Granite Speech STT → Mellea LLM (via Granite Switch) → Kokoro TTS → WebRTC → Browser speaker
```

STT is [IBM Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) served by vLLM. The LLM is [IBM Granite Switch 4.1 3B](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview), also served by vLLM — it exposes `requirement_check` ALoRA intrinsics that power the Best-of-N validation path (any other OpenAI-compatible server works if you don't need that path; point `LLM_URL` / `LLM_MODEL` at it). TTS is Kokoro running locally.

The server ships with a persona of a virtual assistant, configured as the default system prompt. Override via `PROMPT_FILE` or ground with your own docs via `DOCUMENTS_DIR`.

## Generation modes

By default, LLM tokens stream straight through to TTS sentence-by-sentence for low latency.

A **Best-of-N parallel generation** path is also available. It uses Granite Switch's `requirement_check` ALoRA intrinsic to score each candidate against a set of requirements (e.g. "no markdown formatting", "≤50 words", "active voice") and picks the first passing answer. This path is non-streaming because intrinsic validation needs the full answer before scoring. Toggle it at runtime from the frontend.

Barge-in (interrupting the bot mid-response) is handled by Pipecat's `InterruptionFrame` propagation.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Two vLLM servers — one for Granite Speech (STT), one for Granite Switch (LLM). Both require NVIDIA GPUs. See [below](#serving-the-models). The two servers can be on the same host or on separate GPU boxes; set the `*_URL` env vars accordingly.

## Setup

```bash
cp .env.example .env   # edit if your vLLM URLs differ
uv sync
```

### Serving the models

**Granite Speech 4.1 2B (STT)** — [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b):

```bash
vllm serve ibm-granite/granite-speech-4.1-2b \
    --api-key token-abc123 \
    --max-model-len 2048 \
    --port 8083
```

**Granite Switch 4.1 3B (LLM)** — [`ibm-granite/granite-switch-4.1-3b-preview`](https://huggingface.co/ibm-granite/granite-switch-4.1-3b-preview). Adapters (including `requirement_check`) are embedded in the checkpoint; no extra flags needed:

```bash
vllm serve ibm-granite/granite-switch-4.1-3b-preview --port 8000
```

Then make sure `.env` points at them:

```bash
VLLM_SPEECH_URL=http://localhost:8083
LLM_URL=http://localhost:8000/v1
```

If you don't have a second GPU, you can skip the Switch server and point `LLM_URL` / `LLM_MODEL` at any OpenAI-compatible backend — the Best-of-N validation path won't work (it requires Switch's `requirement_check` intrinsic), but the streaming conversation path does.

## Run

### Both backend and frontend (recommended)

```bash
./run.sh
```

Starts the Pipecat backend (http://localhost:7860) and the Next.js frontend (http://localhost:3000) together, with `[backend]` / `[frontend]` line prefixes. Ctrl+C shuts both down cleanly. Bootstraps `.env` from `.env.example` on first run, and runs `npm install` in `frontend/` if `node_modules` is missing.

### Backend only (built-in Pipecat UI)

```bash
uv run uvicorn granite_speech_demo.server:app --host localhost --port 7860
```

Or use the convenience script which loads `.env` and invokes the module entrypoint:

```bash
./start.sh
```

Open http://localhost:7860 — the server redirects to the built-in Pipecat prebuilt UI at `/client/`.

To serve over HTTPS (required for microphone access from non-localhost origins), pass TLS cert paths:

```bash
uv run python -m granite_speech_demo.server --ssl-certfile cert.pem --ssl-keyfile key.pem
```

### With the Next.js frontend

The `frontend/` directory contains a single-page Next.js app (Carbon Design System, IBM Plex fonts) that embeds Pipecat's `voice-ui-kit` and exposes a runtime toggle for IVR validation.

```bash
# Terminal 1 — start the Pipecat backend
uv run uvicorn granite_speech_demo.server:app --host localhost --port 7860

# Terminal 2 — start the frontend
cd frontend
cp .env.example .env.local   # points at http://127.0.0.1:7860 by default
npm install
npm run dev
```

Open http://localhost:3000. The frontend proxies WebRTC signaling to the backend via Next.js API routes (`/api/pipecat/start`, `/api/offer/:sessionId`, `/api/ivr/config`).

## Configuration

All settings are in `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `HOST` | `localhost` | Server bind address |
| `PORT` | `7860` | Server port |
| `LLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible LLM endpoint (default is vLLM serving Granite Switch) |
| `LLM_MODEL` | `ibm-granite/granite-switch-4.1-3b-preview` | Chat model ID |
| `LLM_API_KEY` | _(unset)_ | API key for the LLM endpoint. Not required for local vLLM; set as needed for other backends. |
| `VLLM_SPEECH_URL` | `http://localhost:8083` | vLLM endpoint hosting the Granite Speech model |
| `VLLM_SPEECH_MODEL` | `ibm-granite/granite-speech-4.1-2b` | Speech model ID passed in the chat/completions payload |
| `VLLM_SPEECH_PATH` | `/v1/chat/completions` | Path on the vLLM server for audio-in chat completions |
| `VLLM_SPEECH_BEARER_TOKEN` | `token-abc123` | Bearer token sent to the vLLM speech endpoint |
| `STT_KEYWORD_BIAS` | `Granite,Mellea` | Comma-separated terms appended to the STT prompt to bias transcription |
| `TTS_VOICE` | `bf_emma` | Kokoro voice ID |
| `PROMPT_FILE` | _(unset)_ | Path to a text file whose contents are prepended to the default system/instruct prompts. |
| `DOCUMENTS_DIR` | _(unset)_ | Directory of `.txt` files loaded at import time as Mellea `Document` objects and injected into the system prompt inside `<documents>` tags for grounded answers. |

## Granite Switch and Best-of-N validation

[Granite Switch](https://github.com/generative-computing/granite-switch) is a Granite variant that ships with `requirement_check` ALoRA intrinsics — classifier heads that score a candidate answer against a natural-language requirement. Because Switch is the default LLM, the demo can run a **Best-of-N IVR validation** path that generates several candidates in parallel, scores each against a fixed requirement set, and speaks the first passing answer. The IVR toggle in the frontend turns this on per-session — it's off by default so the baseline turn latency stays low.

### Requirements scored per turn

Defined in `src/granite_speech_demo/mellea_llm.py` as `IVR_REQUIREMENT_SPECS`.

Each requirement has a pass threshold (default `0.5`). A candidate passes only if every requirement clears its threshold. If no candidate passes, a canned fallback answer is used.

### Runtime toggle

The frontend's IVR toggle sends an RTVI `set_ivr_validation` message; the backend switches between streaming and Best-of-N without a reconnect.

### What changes when validation is on

- **Non-streaming.** Each of `BEST_OF_N` (default 3) candidates generates to completion before scoring, so TTS starts later than in streaming mode.
- **System prompt includes requirement instructions.** The same requirements scored by the intrinsic are also embedded as natural-language instructions in the prompt, to nudge generations toward passing.
- **Live validation grid.** The backend pushes `ivr_validation` RTVI server messages (`start` → `sample_start` → `sample_text` → `check` → `done`) so the frontend can render each candidate and its per-requirement scores as they resolve.

### Tuning

Requirement set, labels, instructions, and thresholds all live in `IVR_REQUIREMENT_SPECS` in `mellea_llm.py`. `BEST_OF_N` is a module-level constant in the same file. There are no env vars for these today — edit the file and restart.

## Project structure

```
src/granite_speech_demo/
├── server.py          # FastAPI + SmallWebRTC signaling + pipeline wiring
├── hosted_stt.py      # HostedSTTService — streams audio to the vLLM Granite Speech endpoint
└── mellea_llm.py      # MelleaLLMService — streaming path + Best-of-N IVR validation path, document loading

frontend/              # Next.js app (Carbon Design System, IBM Plex fonts)
├── app/
│   ├── page.tsx             # Single-page demo: top bar + embedded voice UI
│   ├── layout.tsx           # Root layout (fonts, global styles)
│   ├── components/
│   │   └── GraniteSpeechDemo.tsx  # Pipecat voice-ui-kit embed + IVR validation UI
│   └── api/                 # Next.js proxy routes to the Pipecat backend
│       ├── pipecat/start/
│       ├── offer/[sessionId]/
│       └── ivr/config/
├── config.ts                # PIPECAT_BACKEND_URL setting
└── package.json
```

**server.py** sets up the Pipecat pipeline and the FastAPI endpoints (RTVI protocol: `/start`, `/sessions/{id}/api/offer`, plus `/api/ivr/config` for the frontend). Each WebRTC connection spawns its own pipeline:

```
transport.input → HostedSTT (Granite Speech via vLLM) → UserAggregator → MelleaLLM → Kokoro TTS → transport.output → AssistantAggregator
```

**mellea_llm.py** subclasses Pipecat's `LLMService` and has two code paths:

- **Streaming (default):** on each `LLMContextFrame` it extracts the latest user message and streams tokens from the LLM endpoint, pushing `LLMTextFrame`s downstream. TTS is configured with `TextAggregationMode.SENTENCE` so each sentence passes directly to synthesis without additional buffering.
- **Best-of-N (IVR):** runs `BEST_OF_N` (default 3) parallel generations in a thread pool; for each candidate answer it scores every requirement in parallel using Mellea's `requirement_check` ALoRA against the Switch backend, then selects the first passing answer. Intentionally non-streaming — the full answer must be produced before intrinsic validation can score it. Phase events (`start` / `sample_start` / `sample_text` / `check` / `done`) are pushed to the client as RTVI server messages so the UI can render a live validation grid.

The module also loads optional `DOCUMENTS_DIR` `.txt` files into Mellea `Document` objects at import time and embeds them in the system prompt inside `<documents>` tags for RAG-style grounded answers.

## Dependencies

- **[Pipecat AI](https://github.com/pipecat-ai/pipecat)** — pipeline orchestration (WebRTC, Silero VAD, STT/TTS services, SmartTurn)
- **[Mellea](https://github.com/generative-computing/mellea)** — LLM streaming with chunking and requirement-check validation
- **IBM Granite models** — chat (`granite4.1:3b` by default, or a Granite Switch model for IVR validation)
