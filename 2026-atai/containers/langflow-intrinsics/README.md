# LangFlow Intrinsics Components

Custom Langflow components for IBM Granite intrinsic operations and ELSER semantic retrieval.

## Components

- **Intrinsic Model** -- Supports intrinsic operations (answerability, hallucination detection, citations, query rewriting, etc.) across three backends: IntrinsicsAPI, vLLM, and Ollama.
- **ChromaDB Search** -- Similarity search against a ChromaDB server using precomputed embeddings.
- **Local Embeddings** -- Runs sentence-transformers models locally to generate embeddings.
- **ELSER Retriever** -- Retrieves documents from Elasticsearch using ELSER semantic search.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd langflow-intrinsics
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

3. Copy the requirements file to the `.langflow` directory so Langflow installs the component dependencies at startup:

   ```bash
   cp requirements.txt ~/.langflow/data/requirements.txt
   ```

## Configuration

Set the `LANGFLOW_COMPONENTS_PATH` environment variable to point to the `components` directory:

```bash
export LANGFLOW_COMPONENTS_PATH=/path/to/langflow-intrinsics/components
```

To make this persistent, add the export to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

## Running

Start Langflow:

```bash
uv run langflow run
```

The custom components will appear in the Langflow sidebar under their respective categories (**tools** for Intrinsic Model, **retrieval** for ELSER Retriever).

### macOS Apple Silicon: PyTorch MPS/Metal crash

`langflow run` uses gunicorn, which forks worker processes via `fork()`. When a component triggers a PyTorch import chain (e.g. `mellea → outlines → torch`), PyTorch calls `torch.backends.mps.is_available()` which initializes the Metal Performance Shaders framework. macOS does not allow Metal/Objective-C framework initialization after `fork()`, causing the worker process to segfault.

The macOS crash report confirms:
- `EXC_BAD_ACCESS (SIGSEGV)` in `at::mps::MPSDevice::MPSDevice()` → `at::mps::is_available()`
- `"*** single-threaded process forked ***"` / `"crashed on child side of fork pre-exec"`

> **Note:** Langflow already sets `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` internally, but this is not sufficient to prevent the Metal framework crash.

**Workaround:** Run uvicorn directly to avoid gunicorn's preforking model:

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
  LANGFLOW_COMPONENTS_PATH=/absolute/path/to/components \
  uvicorn langflow.main:setup_app --factory --host 0.0.0.0 --port 7860 --timeout-keep-alive 300
```

- Use `setup_app` (not `create_app`) to serve the frontend
- `LANGFLOW_COMPONENTS_PATH` must be an absolute path

This only affects macOS with Apple Silicon (MPS backend). Linux deployments using CUDA or CPU-only are not impacted.

## ChromaDB Vector Store

The ChromaDB Search component requires a running ChromaDB server with precomputed embeddings. The `scripts/setup_chromadb.sh` script handles this automatically:

```bash
./scripts/setup_chromadb.sh
```

**How it works (fastest first):**

1. **Reuse existing data** — if `chromadb_data/` already exists, starts the server immediately
2. **Download pre-built data** — downloads a ~1.4GB tarball from the [GitHub Release](insert-url) and extracts it (~1-2 min)
3. **Build from scratch** — clones the [embeddings repo](https://github.com/frreiss/mt-rag-embeddings) and loads parquet files into ChromaDB (~10 min)

The server runs on port **8100** by default.

### Packaging updated data

If you rebuild the ChromaDB data and want to update the pre-built release:

```bash
./scripts/package_chromadb.sh
```

This creates `chromadb_data.tar.gz` and prints the `gh release create` command to upload it.

## Langfuse Integration

Langflow supports [Langfuse](https://langfuse.com/) for tracing and observability. Set the following environment variables before starting Langflow:

```bash
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_HOST=http://localhost:3000
```

Once configured, Langflow will automatically send traces to Langfuse for all flow executions. You can view traces, latencies, and token usage in the Langfuse dashboard.

## OpenShift Deployment

All deployment manifests are in the `openshift/` directory.

### Prerequisites

Install the [IBM Cloud CLI](https://cloud.ibm.com/docs/cli?topic=cli-install-ibmcloud-cli) and the OpenShift CLI:

```bash
# IBM Cloud CLI
curl -fsSL https://clis.cloud.ibm.com/install/osx | sh

# Container Registry plugin
ibmcloud plugin install container-registry

# OpenShift CLI
ibmcloud plugin install container-service
ibmcloud oc init        # downloads the oc binary
```

### Deploy script

The easiest way to build, push, and deploy is the all-in-one deploy script. It automatically bumps the version in `VERSION`, builds and pushes the Docker image to ICR, deploys to OpenShift, and commits/tags the version bump.

```bash
./scripts/deploy.sh            # bump patch  (0.4.0 → 0.4.1)
./scripts/deploy.sh minor      # bump minor  (0.4.1 → 0.5.0)
./scripts/deploy.sh major      # bump major  (0.5.0 → 1.0.0)
```

**Required environment variables** (add to `~/.bash_profile`):

| Variable | Description |
|----------|-------------|
| `ICR_API_KEY` | IBM Cloud API key for Container Registry (`ibmcloud iam api-key-create`) |
| `OC_TOKEN` | OpenShift login token (`oc whoami -t`) |
| `OC_SERVER` | OpenShift API server URL (`oc whoami --show-server`) |

### Individual Docker scripts

For running steps individually:

```bash
./scripts/docker_build.sh      # build image (linux/amd64)
./scripts/docker_push.sh       # push to ICR
./scripts/docker_run.sh        # run locally on port 7860
```

### Langflow manifest

```bash
oc apply -f openshift/langflow-deployment.yaml
```

This deploys Langflow with its own PostgreSQL instance, service, and route at `langflow.intrinsics.vpc-int.res.ibm.com`.

### Langfuse

Langfuse v3 requires four backing services (PostgreSQL, ClickHouse, Redis, MinIO) plus the Langfuse web and worker containers.

1. **Deploy backing services** (official images mirrored to `us.icr.io/intrinsics/`):

   ```bash
   oc apply -f openshift/langfuse-backing-services.yaml
   ```

2. **Install Langfuse via Helm**:

   ```bash
   helm repo add langfuse https://langfuse.github.io/langfuse-k8s
   helm install langfuse langfuse/langfuse -n intrinsics -f openshift/langfuse-values.yaml
   ```

3. **Create API keys**: Sign up at `https://langfuse.intrinsics.vpc-int.res.ibm.com`, create a project under **Settings > API Keys**, then update the Langflow secret:

   ```bash
   oc create secret generic langfuse-secret -n intrinsics \
     --from-literal=LANGFUSE_SECRET_KEY=sk-lf-... \
     --from-literal=LANGFUSE_PUBLIC_KEY=pk-lf-... \
     --dry-run=client -o yaml | oc apply -f -
   oc rollout restart deployment/langflow -n intrinsics
   ```

### Mirrored Images

All images are mirrored to `us.icr.io/intrinsics/` with `--platform linux/amd64` to avoid architecture mismatches from Apple Silicon and to bypass Docker Hub rate limits. The cluster uses the `ris3-all-icr-io` pull secret.

## Importing Flows

Pre-built flows are included in the `flows/` directory. Due to a known Langflow bug, `LANGFLOW_LOAD_FLOWS_PATH` may fail with `AttributeError: 'str' object has no attribute 'hex'`. Instead, import flows manually through the UI:

1. Start Langflow
2. Click the **Import** button (or use the folder icon) on the home screen
3. Select the desired flow JSON file from the `flows/` directory

### Updating flow files after component changes

When component source files (`components/**/*.py`) are modified, the flow JSON files need to be updated to match. The `update_flows.py` script sends each component's code to the running Langflow server, gets back the updated node definition, and patches the flow JSON — preserving user-configured values and fixing edge connections.

```bash
# Requires a running Langflow server (default: http://localhost:7860)
python scripts/update_flows.py

# Update specific flow(s)
python scripts/update_flows.py --flow "flows/patterns/Pattern 2.json"

# Custom server URL
python scripts/update_flows.py --base-url http://localhost:8080
```
