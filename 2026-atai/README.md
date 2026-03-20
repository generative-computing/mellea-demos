# ATAI Demo

A demo of IBM Granite Libraries + Mellea using Langflow.

## Prerequisites

| Tool | Purpose | Required |
|------|---------|----------|
| Docker or [Colima](https://github.com/abiosoft/colima) | Run containers (8 GB RAM recommended) | Yes |
| Docker Compose v2 | Orchestrate services | Yes |
| [Ollama](https://ollama.com) | Local LLM inference | Yes (unless `--no-ollama`) |
| Git + [Git LFS](https://git-lfs.com/) | Clone LoRA adapter weights | Yes |
| Python 3 | JSON manipulation in load script | Yes |
| [jq](https://jqlang.github.io/jq/) | JSON parsing in shell scripts | Yes |

> **Note:** `start.sh` will auto-install missing tools on macOS via Homebrew and attempt to
> start Docker/Colima if the daemon is not running. On Linux you may need to install missing
> tools manually (instructions printed on failure).

### macOS — install everything at once

```bash
# Homebrew (https://brew.sh) required
brew install colima docker docker-compose ollama jq python git-lfs
git lfs install
colima start --cpu 4 --memory 8   # starts the Docker daemon via Colima
```

Or use [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/) instead of Colima
(requires a free account for personal use).

### Linux (Debian/Ubuntu)

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER" && newgrp docker

# Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Other tools
sudo apt-get update
sudo apt-get install -y jq python3 git git-lfs
git lfs install
```

### Windows

Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with the Ubuntu instructions
above, or install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) and
run `start.sh` from Git Bash or WSL2.

---

## Quick Start

```bash
./start.sh
```

`start.sh` handles everything automatically:
- Verifies (and where possible installs) all prerequisites
- Starts Colima/Docker if the daemon is not running
- Pulls `granite4:micro` from Ollama if not present
- Clones LoRA adapters from Hugging Face if not present
- Starts all containers and waits for health checks
- Loads all demo flows into Langflow via the API

First run takes ~3–5 minutes (model pull + container image download).

When done:

```bash
./stop.sh              # stops everything (data preserved)
./stop.sh --clean      # stops and deletes all data
```

## Demo Flows

| Title | Description |
|-------|-------------|
| [Citation Generation](https://github.com/generative-computing/mellea-demos/tree/main/2026-atai/flows/Citation_Generation) | Demonstrates citation generation and hallucination detection intrinsics to help users verify RAG responses and identify unsupported claims. |
| [Query Clarification — ClapNQ](https://github.com/generative-computing/mellea-demos/tree/main/2026-atai/flows/Clarification-ClapNQ) | Shows how a trained query clarification intrinsic asks follow-up questions for ambiguous queries (e.g. "when did the war end?") over a general-domain corpus, where prompt-based approaches fail to do so. |
| [Query Clarification — Gov](https://github.com/generative-computing/mellea-demos/tree/main/2026-atai/flows/Clarification-Gov) | Demonstrates query clarification over government documents, where missing context (e.g. the user's state for DMV queries) causes the system to ask before answering. |
| [Hallucination Mitigation](https://github.com/generative-computing/mellea-demos/tree/main/2026-atai/flows/Hallucination_Mitigation) | Uses query rewriting and post-generation hallucination detection to identify and remove fabricated content not grounded in the retrieved corpus. |
| [Query Rewrite + Answerability](https://github.com/generative-computing/mellea-demos/tree/main/2026-atai/flows/QueryRewrite_Answerability) | Compares prompt-based vs. intrinsic query rewriting and answerability detection, showing how intrinsics more reliably avoid hallucination and handle multi-turn conversations. |

## Running a Flow

1. Open **Langflow** at http://localhost:7860 (auto-login enabled)
2. Click on the project in the left sidebar to reveal its flows
3. Click on a flow (e.g., "Intrinsic_QR_AD")
4. Click the **Playground** button (bottom right)
5. Enter a query and run it

The flow calls Ollama for inference using the Granite base model and intrinsic LoRA adapters, and retrieves context from ChromaDB (pre-loaded with embeddings).

## Services

| Service | URL | Description |
|---------|-----|-------------|
| Langflow | http://localhost:7860 | Flow design and execution UI |
| ChromaDB | http://localhost:8100 | Vector database with pre-computed embeddings |

## Project Structure

```
├── start.sh / stop.sh        # Lifecycle scripts
├── docker-compose.yml         # All service definitions
├── .env                       # Environment configuration
├── flows/                     # Langflow flow definitions (mounted read-only)
```
