#!/usr/bin/env bash
# =============================================================================
# ATAI Demo — One-Command Start
# =============================================================================
# Brings up the full ATAI demo stack:
#   Langfuse + Langflow Intrinsics + ChromaDB
#
# Prerequisites:
#   - Docker + Docker Compose
#   - Ollama installed and running (ollama serve)
#   - granite4:micro model pulled (ollama pull granite4:micro)
#
# Usage:
#   ./start.sh
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

# Ollama
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

echo ""

# =============================================================================
# Step 2: Load LoRA adapters
# =============================================================================
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

echo ""

# =============================================================================
# Step 3: Verify flow files
# =============================================================================
if [ ! -d "$SCRIPT_DIR/flows" ] || ! ls "$SCRIPT_DIR/flows"/*.json &>/dev/null; then
    error "No flow JSON files found in flows/"
    exit 1
fi
ok "Flow files found"

echo ""

# =============================================================================
# Step 4: Start Docker Compose
# =============================================================================
info "Starting Docker Compose services..."
docker compose up -d

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
wait_for_service "ChromaDB" "http://localhost:8100/api/v2/heartbeat" 30

# Langflow (needs migrations + component loading)
wait_for_service "Langflow" "http://localhost:7860/health" 180

echo ""

# =============================================================================
# Step 6: Print summary
# =============================================================================
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN} ATAI Demo is running!${NC}"
echo -e "${GREEN}=============================================================================${NC}"
echo ""
echo -e "  ${BLUE}Langflow${NC}        http://localhost:7860"
echo ""
echo -e "To stop all services: ${YELLOW}./stop.sh${NC}"
echo -e "To stop and delete data: ${YELLOW}./stop.sh --clean${NC}"
echo ""
