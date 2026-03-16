# LangFlow Trace Visualization

A web-based visualization tool for LangFlow execution traces. This tool has been created to display intuitive visualizations of executions of flows involving intrinsics. The current goal for the project is to support the visualization of flows consisting of components present in the [langflow-intrinsics](../langflow-intrinsics/) repository but the code will also do a best-effort attempt to display other components.

## Pre-requisites

To use the visualization, you need access to:
- A LangFlow deployment with access to the intrinsic LangFlow components and
- A LangFuse deployment connected to the LangFlow deployment

If you do not have access to such deployments, follow the instructions in the [langflow-intrinsics](../langflow-intrinsics/) repository to set them up.

## Setup

### 1. Create a Virtual Environment

```bash
cd langflow-vis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# LangFuse Configuration
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key

# LangFlow Configuration
LANGFLOW_BASE_URL=http://localhost:7860

# Debug Mode (optional)
DEBUG_MODE=false
```

Users must enter their LangFlow API key in the UI. Traces are filtered to only show those belonging to the user associated with the key.

### 4. Run the Server

```bash
bash app/run_dev.sh
```

The server will start at `http://localhost:8005` with auto-reload enabled.

FastAPI auto-generates interactive API docs at `http://localhost:8005/docs`.

## OpenShift Deployment

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
./scripts/deploy.sh            # bump patch  (0.3.0 → 0.3.1)
./scripts/deploy.sh minor      # bump minor  (0.3.1 → 0.4.0)
./scripts/deploy.sh major      # bump major  (0.4.0 → 1.0.0)
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
./scripts/docker_run.sh        # run locally on port 8080
```

### First-time setup

Create the secret and apply the deployment manifest:

```bash
oc create secret generic langflow-vis-secrets \
  --from-literal=LANGFUSE_BASE_URL=<your_langfuse_url> \
  --from-literal=LANGFUSE_PUBLIC_KEY=<your_public_key> \
  --from-literal=LANGFUSE_SECRET_KEY=<your_secret_key> \
  --from-literal=LANGFLOW_BASE_URL=<your_langflow_url> \
  -n intrinsics

oc apply -f openshift/deployment.yaml
```

The application will be available at `https://langflow-vis.intrinsics.vpc-int.res.ibm.com`.

## Usage

1. Open the application URL in your browser
2. Enter your LangFlow API key. The key is saved in your browser's local storage so you only need to enter it once.
3. By default, the visualization will show the most recent execution trace for your user account. Only traces belonging to the user associated with your API key are shown.
4. To visualize a different historical execution trace, select it from the trace selector dropdown.
5. Click on individual components to see their complete outputs together with other properties.

## Contact

- **Yannis Katsis** — [yannis.katsis@ibm.com](mailto:yannis.katsis@ibm.com)
