# ATAI Demo

A demo of IBM Granite Libraries + Mellea using Langflow.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (8+ GB memory recommended)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2)
- [Ollama](https://ollama.com) (must be running)
- [Git](https://git-scm.com/) with [Git LFS](https://git-lfs.com/)
- Python 3 (for the start script)

## Quick Start

1. Pull the base model:

```bash
ollama pull granite4:micro
```

2. Download the pre-computed ChromaDB embeddings:

```bash
curl -L -o containers/chromadb/chromadb_data.tar.gz \
  [some-github-url]/chromadb-data-v1/chromadb_data.tar.gz
```

3. Clone the LoRA adapters from Hugging Face (requires [Git LFS](https://git-lfs.com/)):

```bash
git clone https://huggingface.co/ibm-granite/granite-lib-rag-r1.0
```

4. Start the demo:

```bash
./start.sh
```

The start script verifies prerequisites, loads the LoRA adapters into Ollama, starts all containers, waits for health checks, and provisions API keys. First run takes ~2-3 minutes.

5. When done, stop the demo:

```bash
./stop.sh              # stops everything (data preserved)
./stop.sh --clean      # stops and deletes all data
```

## Demo Flows

(document when added)

## Running a Flow

1. Open **Langflow** at http://localhost:7860 (auto-login enabled)
2. Click on a flow (e.g., "Intrinsic_QR_AD")
3. Click the **Playground** button (bottom right)
4. Enter a query and run it

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
