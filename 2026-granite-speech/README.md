# granite-speech-demo

Real-time voice conversation demo showcasing IBM Granite models with [Pipecat](https://github.com/pipecat-ai/pipecat) for pipeline orchestration and [Mellea](https://github.com/generative-computing/mellea) for validated LLM generation.

```
Browser mic → WebRTC → Silero VAD → Whisper STT → Mellea LLM → Kokoro TTS → WebRTC → Browser speaker
```

Everything runs on-device. STT is Whisper (MLX Whisper on macOS / Apple Silicon, faster-whisper on Linux with CUDA auto-detection), the LLM is served by Ollama over an OpenAI-compatible endpoint, and TTS is Kokoro. Any other OpenAI-compatible server (LM Studio, vLLM, etc.) works by pointing `LLM_URL` / `LLM_MODEL` at it.

The server ships with a persona of a virtual assistant, configured as the default system prompt. Override via `PROMPT_FILE` or ground with your own docs via `DOCUMENTS_DIR`.

## Generation modes

By default, LLM tokens stream straight through to TTS sentence-by-sentence for low latency.

When `GRANITE_SWITCH_ENABLED=true` (i.e. `LLM_MODEL` points at a Granite Switch model), a **Best-of-N parallel generation** path becomes available that uses the `requirement_check` ALoRA intrinsic to score each candidate against a set of requirements (e.g. "no markdown formatting", "≤50 words", "active voice") and picks the first passing answer. This path is non-streaming because intrinsic validation needs the full answer before scoring. Toggle it at runtime from the frontend.

Barge-in (interrupting the bot mid-response) is handled by Pipecat's `InterruptionFrame` propagation.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible LLM server running locally. [Ollama](https://ollama.com/) is the default — install it, then pull the model: `ollama pull granite4.1:3b`. Any other OpenAI-compatible backend works too; point `LLM_URL` / `LLM_MODEL` at it.

## Setup

```bash
cp .env.example .env   # edit if your LLM URL/model differs
uv sync
```

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
| `LLM_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM endpoint (default is Ollama) |
| `LLM_MODEL` | `granite4.1:3b` | Chat model ID |
| `LLM_API_KEY` | `ollama` | API key for the LLM endpoint. Ollama ignores it; set as needed for other backends. |
| `WHISPER_MODEL` | `small` | Whisper model size (`tiny`, `medium`, `large-v3`, `large-v3-turbo`, `distil-large-v3`) |
| `TTS_VOICE` | `bf_emma` | Kokoro voice ID |
| `GRANITE_SWITCH_ENABLED` | `false` | Set to `true` when `LLM_MODEL` is a Granite Switch model. Gates the IVR validation path — flip the frontend toggle to turn it on per-session. |
| `PROMPT_FILE` | _(unset)_ | Path to a text file whose contents are prepended to the default system/instruct prompts. |
| `DOCUMENTS_DIR` | _(unset)_ | Directory of `.txt` files loaded at import time as Mellea `Document` objects and injected into the system prompt inside `<documents>` tags for grounded answers. |

## Using Granite Switch

[Granite Switch](https://github.com/generative-computing/granite-switch) is a Granite variant that exposes `requirement_check` ALoRA intrinsics — classifier heads that score a candidate answer against a natural-language requirement. When `GRANITE_SWITCH_ENABLED=true`, the demo unlocks a **Best-of-N IVR validation** path that generates several candidates in parallel, scores each against a fixed requirement set, and speaks the first passing answer.

### Requirements scored per turn

Defined in `src/granite_speech_demo/mellea_llm.py` as `IVR_REQUIREMENT_SPECS`.

Each requirement has a pass threshold (default `0.5`). A candidate passes only if every requirement clears its threshold. If no candidate passes, a canned fallback answer is used.

### Enabling it

1. Follow the instructions in the [granite-switch repo](https://github.com/generative-computing/granite-switch) to run the model locally on an OpenAI-compatible endpoint that exposes the `requirement_check` intrinsic.
2. Point `LLM_URL` / `LLM_MODEL` at that endpoint.
3. In `.env`:

   ```bash
   GRANITE_SWITCH_ENABLED=true
   ```

4. Restart the server. The Switch gate is armed; flip the IVR toggle in the frontend to turn validation on.

### Runtime toggle

The frontend's IVR toggle sends an RTVI `set_ivr_validation` message; the backend switches between streaming and Best-of-N without a reconnect. The toggle is a no-op if `GRANITE_SWITCH_ENABLED=false`.

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
transport.input → Whisper STT → UserAggregator → MelleaLLM → Kokoro TTS → transport.output → AssistantAggregator
```

**mellea_llm.py** subclasses Pipecat's `LLMService` and has two code paths:

- **Streaming (default):** on each `LLMContextFrame` it extracts the latest user message and streams tokens from the LLM endpoint, pushing `LLMTextFrame`s downstream. TTS is configured with `TextAggregationMode.SENTENCE` so each sentence passes directly to synthesis without additional buffering.
- **Best-of-N (IVR, requires `GRANITE_SWITCH_ENABLED=true`):** runs `BEST_OF_N` (default 3) parallel generations in a thread pool; for each candidate answer it scores every requirement in parallel using Mellea's `requirement_check` ALoRA against the Switch backend, then selects the first passing answer. Intentionally non-streaming — the full answer must be produced before intrinsic validation can score it. Phase events (`start` / `sample_start` / `sample_text` / `check` / `done`) are pushed to the client as RTVI server messages so the UI can render a live validation grid.

The module also loads optional `DOCUMENTS_DIR` `.txt` files into Mellea `Document` objects at import time and embeds them in the system prompt inside `<documents>` tags for RAG-style grounded answers.

## Dependencies

- **[Pipecat AI](https://github.com/pipecat-ai/pipecat)** — pipeline orchestration (WebRTC, Silero VAD, STT/TTS services, SmartTurn)
- **[Mellea](https://github.com/generative-computing/mellea)** — LLM streaming with chunking and requirement-check validation
- **IBM Granite models** — chat (`granite4.1:3b` by default, or a Granite Switch model for IVR validation)
