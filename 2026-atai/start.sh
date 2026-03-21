#!/usr/bin/env bash
# =============================================================================
# ATAI Demo — One-Command Start
# =============================================================================
# Brings up the full ATAI demo stack:
#   Langflow Intrinsics + ChromaDB
#
# Prerequisites:
#   - Docker + Docker Compose
#   - Ollama installed and running (ollama serve)
#   - granite4:micro model pulled (ollama pull granite4:micro)
#
# Usage:
#   ./start.sh                          # Load ALL flows into LangFlow projects (default)
#   ./start.sh --all                    # Same as above (explicit)
#   ./start.sh --demo <name>            # Load a single demo's flows only
#   ./start.sh --no-ollama              # Skip Ollama checks and LoRA loading
#   ./start.sh --no-chromadb            # Skip ChromaDB service
#
# Available demos (for --demo mode):
#   Clarification-ClapNQ      - General domain (ClapNQ) Query Clarification
#   Clarification-Gov         - Government domain (DMV) Query Clarification
#   Hallucination_Mitigation  - Hallucination detection flows
#   QueryRewrite_Answerability - Query rewrite and answerability flows
#   Citation_Generation       - Citation generation flows
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Parse arguments
# =============================================================================
LOAD_ALL=true
DEMO=""
SKIP_OLLAMA=false
SKIP_CHROMADB=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            LOAD_ALL=true
            shift
            ;;
        --demo)
            LOAD_ALL=false
            DEMO="${2:-}"
            if [ -z "$DEMO" ]; then
                error "--demo requires a demo name"
                exit 1
            fi
            shift 2
            ;;
        --no-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --no-chromadb)
            SKIP_CHROMADB=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--all | --demo <name>] [--no-ollama] [--no-chromadb]"
            echo ""
            echo "Options:"
            echo "  --all           Load ALL flows into LangFlow projects (default)"
            echo "  --demo <name>   Load a single demo's flows only"
            echo "  --no-ollama     Skip Ollama checks and LoRA adapter loading"
            echo "  --no-chromadb   Skip ChromaDB service"
            echo ""
            echo "Available demos:"
            echo "  Clarification-ClapNQ       - General domain (ClapNQ) Query Clarification"
            echo "  Clarification-Gov          - Government domain (DMV) Query Clarification"
            echo "  Hallucination_Mitigation   - Hallucination detection flows"
            echo "  QueryRewrite_Answerability - Query rewrite and answerability flows"
            echo "  Citation_Generation        - Citation generation flows"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate demo mode if specified
# =============================================================================
if [ "$LOAD_ALL" = false ]; then
    DEMO_SRC="flows/$DEMO"
    if [ ! -d "$SCRIPT_DIR/$DEMO_SRC" ]; then
        error "Demo not found: $DEMO"
        echo ""
        echo "Available demos:"
        for dir in "$SCRIPT_DIR"/flows/*/; do
            [ -d "$dir" ] && echo "  $(basename "$dir")"
        done
        exit 1
    fi
fi

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================
info "Checking prerequisites..."

# Detect OS for install hints
OS="$(uname -s)"
case "$OS" in
    Darwin) PKG_HINT="brew install" ;;
    Linux)  PKG_HINT="sudo apt-get install -y  # or your distro's package manager" ;;
    *)      PKG_HINT="your package manager" ;;
esac

# Docker — fall back to Colima on macOS/Linux if docker CLI is absent
if ! command -v docker &>/dev/null; then
    warn "docker not found — checking for Colima..."
    if command -v colima &>/dev/null; then
        info "Colima found. Starting Colima..."
        colima start --cpu 4 --memory 8 || {
            error "Failed to start Colima. Please start it manually: colima start"
            exit 1
        }
        ok "Colima started"
    else
        # Try to install Colima via Homebrew on macOS
        if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
            info "Installing Colima and docker CLI via Homebrew..."
            brew install colima docker
            info "Starting Colima..."
            colima start --cpu 4 --memory 8 || {
                error "Failed to start Colima. Please start it manually: colima start"
                exit 1
            }
            ok "Colima installed and started"
        else
            error "Docker is not installed and Colima is not available."
            echo ""
            echo "  Install options:"
            echo "    macOS (recommended): brew install colima docker"
            echo "    macOS (Docker Desktop): https://docs.docker.com/desktop/install/mac-install/"
            echo "    Linux: https://docs.docker.com/engine/install/"
            exit 1
        fi
    fi
else
    # Docker CLI present — make sure the daemon is actually reachable
    if ! docker info &>/dev/null; then
        warn "Docker CLI found but daemon is not running — checking for Colima..."
        if command -v colima &>/dev/null; then
            info "Starting Colima..."
            colima start --cpu 4 --memory 8 || {
                error "Failed to start Colima. Start it manually: colima start"
                exit 1
            }
            ok "Colima started"
        else
            error "Docker daemon is not running."
            echo ""
            echo "  Start options:"
            echo "    Docker Desktop: open the Docker Desktop application"
            echo "    Colima (macOS/Linux): colima start"
            exit 1
        fi
    fi
    ok "Docker found"
fi

# Docker Compose
if ! docker compose version &>/dev/null; then
    error "Docker Compose v2 is not available."
    echo ""
    echo "  Install options:"
    echo "    macOS:  brew install docker-compose  (or update Docker Desktop)"
    echo "    Linux:  https://docs.docker.com/compose/install/"
    exit 1
fi
ok "Docker Compose found"

# jq (required for flow loading)
if ! command -v jq &>/dev/null; then
    if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
        info "Installing jq via Homebrew..."
        brew install jq
    else
        error "jq is required. Please install it:"
        echo "    macOS:  brew install jq"
        echo "    Linux:  sudo apt-get install -y jq  (or equivalent)"
        echo "    Other:  https://jqlang.github.io/jq/download/"
        exit 1
    fi
fi
ok "jq found"

# python3 (required by load-flows.sh for JSON manipulation)
if ! command -v python3 &>/dev/null; then
    error "python3 is required."
    echo ""
    echo "  Install options:"
    echo "    macOS:  brew install python"
    echo "    Linux:  sudo apt-get install -y python3"
    echo "    Other:  https://www.python.org/downloads/"
    exit 1
fi
ok "python3 found"

# git-lfs (required to clone LoRA adapters from Hugging Face)
if ! command -v git-lfs &>/dev/null && ! git lfs version &>/dev/null 2>&1; then
    if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
        info "Installing git-lfs via Homebrew..."
        brew install git-lfs
        git lfs install
    else
        error "git-lfs is required to download the LoRA adapter weights."
        echo ""
        echo "  Install options:"
        echo "    macOS:  brew install git-lfs && git lfs install"
        echo "    Linux:  sudo apt-get install -y git-lfs && git lfs install"
        echo "    Other:  https://git-lfs.com"
        exit 1
    fi
fi
ok "git-lfs found"

# Ollama (skip if --no-ollama)
if [ "$SKIP_OLLAMA" = false ]; then
    if ! command -v ollama &>/dev/null; then
        error "Ollama is not installed."
        echo ""
        echo "  Install options:"
        echo "    macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
        echo "    macOS (Homebrew): brew install ollama"
        echo "    Other: https://ollama.com/download"
        exit 1
    fi
    ok "Ollama found"

    # Ollama running
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        info "Ollama is not running — starting it in the background..."
        ollama serve &>/dev/null &
        # Wait up to 10s for it to start
        OLLAMA_WAIT=0
        while [ $OLLAMA_WAIT -lt 10 ]; do
            if curl -sf http://localhost:11434/api/tags &>/dev/null; then
                break
            fi
            sleep 1
            OLLAMA_WAIT=$((OLLAMA_WAIT + 1))
        done
        if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
            error "Ollama failed to start. Please start it manually: ollama serve"
            exit 1
        fi
        ok "Ollama started"
    fi
    ok "Ollama is running"

    # granite4:micro model
    if ! ollama list 2>/dev/null | grep -q "granite4:micro"; then
        info "granite4:micro model not found — pulling now..."
        ollama pull granite4:micro
    fi
    ok "granite4:micro model available"
else
    warn "Skipping Ollama checks (--no-ollama)"
fi

echo ""

# =============================================================================
# Step 2: Load LoRA adapters (skip if --no-ollama)
# =============================================================================
if [ "$SKIP_OLLAMA" = false ]; then
    info "Loading LoRA adapters for intrinsic operations..."

    GRANITE_LIB="$SCRIPT_DIR/granite-lib-rag-r1.0"

    if [ ! -d "$GRANITE_LIB" ]; then
        info "Downloading granite-lib-rag-r1.0 from Hugging Face..."
        if command -v git &>/dev/null; then
            git clone https://huggingface.co/ibm-granite/granite-lib-rag-r1.0 "$GRANITE_LIB"
        else
            error "git is required to download granite-lib-rag-r1.0"
            exit 1
        fi
    fi

    pushd "$GRANITE_LIB" > /dev/null
    sed -i.bak 's|localhost:55555|localhost:11434|g' run_ollama.sh && rm -f run_ollama.sh.bak
    bash run_ollama.sh
    popd > /dev/null
    ok "LoRA adapters loaded"
else
    warn "Skipping LoRA adapter loading (--no-ollama)"
fi

echo ""

# =============================================================================
# Step 3: Prepare flows for Docker mount (single demo mode only)
# =============================================================================
DEMO_FLOWS_DIR="$SCRIPT_DIR/.demo_flows"
rm -rf "$DEMO_FLOWS_DIR"
mkdir -p "$DEMO_FLOWS_DIR"

if [ "$LOAD_ALL" = false ]; then
    # Single demo mode: copy flows to temp directory for Docker mount
    cp "$SCRIPT_DIR/$DEMO_SRC"/*.json "$DEMO_FLOWS_DIR/" 2>/dev/null || true

    # Also copy from subdirectories if any
    find "$SCRIPT_DIR/$DEMO_SRC" -name "*.json" -type f -exec cp {} "$DEMO_FLOWS_DIR/" \;

    if ! ls "$DEMO_FLOWS_DIR"/*.json &>/dev/null; then
        error "No flow JSON files found in $DEMO_SRC"
        exit 1
    fi
    ok "Loaded $DEMO demo flows: $(ls -1 "$DEMO_FLOWS_DIR"/*.json | wc -l | tr -d ' ') files"
else
    info "All flows mode: flows will be loaded via API after startup"
fi

export DEMO_FLOWS_DIR

echo ""

# =============================================================================
# Step 4: Start Docker Compose
# =============================================================================
info "Starting Docker Compose services..."
# We don't start langflow-vis yet because we need to set the langflow api key first
if [ "$SKIP_CHROMADB" = true ]; then
    # Use --no-deps to skip chromadb dependency
    docker compose up -d --no-deps langflow-intrinsics --scale langflow-vis=0
else
    docker compose up -d --scale langflow-vis=0
fi

echo ""

# =============================================================================
# Step 5: Wait for services to be healthy
# =============================================================================
info "Waiting for services to become healthy..."

wait_for_service() {
    local name="$1"
    local url="$2"
    local timeout="$3"

    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -sf "$url" &>/dev/null; then
            ok "$name is ready (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    warn "$name did not respond within ${timeout}s at $url"
    return 1
}

# ChromaDB (should be fast)
if [ "$SKIP_CHROMADB" = false ]; then
    wait_for_service "ChromaDB" "http://localhost:8100/api/v2/heartbeat" 30
else
    warn "Skipping ChromaDB (--no-chromadb)"
fi

# Langfuse (needs DB migrations on first run)
wait_for_service "Langfuse" "http://localhost:3000" 90

# Langflow (needs migrations + component loading)
wait_for_service "Langflow" "http://localhost:7860/health" 180

echo ""

# =============================================================================
# Step 6: Load flows via API (all mode only)
# =============================================================================
if [ "$LOAD_ALL" = true ]; then
    info "Loading all flows into LangFlow projects..."
    "$SCRIPT_DIR/scripts/load-flows.sh" "$SCRIPT_DIR/flows"
fi

# =============================================================================
# Step 7: Provision Langflow API key for the visualization service
# =============================================================================
provision_langflow_api_key() {
    local langflow_url="http://localhost:7860"

    # Auto-login to get access token (works with LANGFLOW_AUTO_LOGIN=true)
    local login_response
    login_response=$(curl -sf "$langflow_url/api/v1/auto_login") || {
        warn "auto_login request failed" >&2
        return 1
    }

    local token
    token=$(echo "$login_response" \
        | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['access_token'])" 2>/dev/null) || {
        warn "Could not parse access_token from auto_login response" >&2
        warn "Response was: $login_response" >&2
        return 1
    }

    # Create API key
    local key_response
    key_response=$(curl -sf -X POST "$langflow_url/api/v1/api_key/" \
        -H "Authorization: Bearer $token" \
        -H "Content-Type: application/json" \
        -d '{"name":"langflow-vis"}') || {
        warn "API key creation request failed" >&2
        return 1
    }

    local api_key
    api_key=$(echo "$key_response" \
        | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['api_key'])" 2>/dev/null) || {
        warn "Could not parse api_key from response" >&2
        warn "Response was: $key_response" >&2
        return 1
    }

    echo "$api_key"
}

if [ -z "${LANGFLOW_API_KEY:-}" ]; then
    info "Provisioning Langflow API key..."
    LANGFLOW_API_KEY=$(provision_langflow_api_key) || true
    if [ -z "$LANGFLOW_API_KEY" ]; then
        warn "Failed to provision Langflow API key — langflow-vis may not be able to call Langflow"
    else
        export LANGFLOW_API_KEY
        ok "Langflow API key provisioned"
    fi
else
    ok "Using existing LANGFLOW_API_KEY from environment"
fi

echo ""

# =============================================================================
# Step 8: Start langflow-vis with the provisioned API key
# =============================================================================
info "Starting Langflow-Vis..."
docker compose up -d --force-recreate langflow-vis

wait_for_service "Langflow-Vis" "http://localhost:8080" 120

echo ""

# =============================================================================
# Step 9: Print summary
# =============================================================================
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN} ATAI Demo is running!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo -e "  ${BLUE}Langflow${NC}        http://localhost:7860"
echo -e "  ${BLUE}Visualization${NC}   http://localhost:8080"
echo ""
if [ "$LOAD_ALL" = true ]; then
    echo -e "  Loaded projects:"
    for dir in "$SCRIPT_DIR"/flows/*/; do
        [ -d "$dir" ] && echo -e "    - $(basename "$dir")"
    done
    echo ""
fi
echo -e "To stop all services: ${YELLOW}./stop.sh${NC}"
echo -e "To stop and delete data: ${YELLOW}./stop.sh --clean${NC}"
echo ""
