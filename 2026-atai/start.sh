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

# Docker
if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
ok "Docker found"

# Docker Compose
if ! docker compose version &>/dev/null; then
    error "Docker Compose is not available. Please install Docker Compose v2."
    exit 1
fi
ok "Docker Compose found"

# Ollama (skip if --no-ollama)
if [ "$SKIP_OLLAMA" = false ]; then
    if ! command -v ollama &>/dev/null; then
        error "Ollama is not installed. Please install Ollama: https://ollama.com"
        exit 1
    fi
    ok "Ollama found"

    # Ollama running
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        error "Ollama is not running. Please start it with: ollama serve"
        exit 1
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

# jq (required for API-based flow loading)
if [ "$LOAD_ALL" = true ]; then
    if ! command -v jq &>/dev/null; then
        error "jq is required for --all mode. Please install jq: https://jqlang.github.io/jq/download/"
        exit 1
    fi
    ok "jq found"
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
if [ "$SKIP_CHROMADB" = true ]; then
    # Use --no-deps to skip chromadb dependency
    docker compose up -d --no-deps langflow-intrinsics
else
    docker compose up -d
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
# Step 7: Print summary
# =============================================================================
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN} ATAI Demo is running!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo -e "  ${BLUE}Langflow${NC}        http://localhost:7860"
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
