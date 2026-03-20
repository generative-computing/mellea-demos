#!/usr/bin/env bash
# =============================================================================
# Load Flows into LangFlow Projects
# =============================================================================
# Creates a LangFlow project (folder) for each top-level directory in flows/
# and uploads all JSON flow files found recursively within each.
#
# Usage:
#   ./scripts/load-flows.sh [flows_dir]
#   ./scripts/load-flows.sh              # Uses ./flows by default
#   ./scripts/load-flows.sh /path/to/flows
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANGFLOW_URL="${LANGFLOW_URL:-http://localhost:7860}"
FLOWS_DIR="${1:-$SCRIPT_DIR/../flows}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Global token variable
AUTH_TOKEN=""

# =============================================================================
# Get authentication token
# =============================================================================
get_auth_token() {
    info "Authenticating with LangFlow..."

    local response
    response=$(curl -sfL "$LANGFLOW_URL/api/v1/auto_login" 2>/dev/null || echo "")

    if [ -z "$response" ]; then
        error "Failed to authenticate with LangFlow"
        return 1
    fi

    AUTH_TOKEN=$(echo "$response" | jq -r '.access_token' 2>/dev/null || echo "")

    if [ -z "$AUTH_TOKEN" ] || [ "$AUTH_TOKEN" = "null" ]; then
        error "Failed to get access token"
        return 1
    fi

    ok "Authenticated successfully"
    return 0
}

# =============================================================================
# Wait for LangFlow API to be ready
# =============================================================================
wait_for_langflow() {
    local timeout="${1:-120}"
    local elapsed=0

    info "Waiting for LangFlow API to be ready..."
    while [ $elapsed -lt $timeout ]; do
        if curl -sfL "$LANGFLOW_URL/health" &>/dev/null; then
            ok "LangFlow API is ready"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    error "LangFlow API did not become ready within ${timeout}s"
    return 1
}

# =============================================================================
# Create a folder/project in LangFlow
# =============================================================================
create_folder() {
    local folder_name="$1"

    # Check if folder already exists
    local existing_id
    existing_id=$(curl -sfL "$LANGFLOW_URL/api/v1/folders/" \
        -H "Authorization: Bearer $AUTH_TOKEN" \
        -H "Content-Type: application/json" | \
        jq -r ".[] | select(.name == \"$folder_name\") | .id" 2>/dev/null || echo "")

    if [ -n "$existing_id" ] && [ "$existing_id" != "null" ]; then
        echo "$existing_id"
        return 0
    fi

    # Create new folder
    local response
    response=$(curl -sfL -X POST "$LANGFLOW_URL/api/v1/folders/" \
        -H "Authorization: Bearer $AUTH_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$folder_name\"}" 2>/dev/null || echo "")

    if [ -z "$response" ]; then
        error "Failed to create folder: $folder_name"
        return 1
    fi

    echo "$response" | jq -r '.id'
}

# =============================================================================
# Upload a flow JSON file to a folder
# =============================================================================
upload_flow() {
    local flow_file="$1"
    local folder_id="$2"
    local flow_name
    flow_name=$(basename "$flow_file" .json)

    # Use Python to modify JSON and upload (handles special characters better than jq)
    local result
    result=$(python3 << EOF
import json
import sys

try:
    with open("$flow_file", 'r') as f:
        flow = json.load(f)

    # Add folder_id to the JSON body
    flow['folder_id'] = "$folder_id"
    # Remove existing id to let LangFlow assign a new one
    if 'id' in flow:
        del flow['id']

    # Write to stdout as JSON
    print(json.dumps(flow))
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    ) || { warn "Failed to prepare flow: $flow_name"; return 1; }

    # Upload the modified JSON
    local response
    response=$(echo "$result" | curl -sfL -X POST "$LANGFLOW_URL/api/v1/flows/" \
        -H "Authorization: Bearer $AUTH_TOKEN" \
        -H "Content-Type: application/json" \
        -d @- 2>/dev/null || echo "")

    if [ -z "$response" ]; then
        warn "Failed to upload flow: $flow_name"
        return 1
    fi

    ok "Uploaded: $flow_name"
    return 0
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Verify flows directory exists
    if [ ! -d "$FLOWS_DIR" ]; then
        error "Flows directory not found: $FLOWS_DIR"
        exit 1
    fi

    # Wait for LangFlow
    wait_for_langflow 120

    # Get authentication token
    get_auth_token || exit 1

    echo ""
    info "Loading flows from: $FLOWS_DIR"
    echo ""

    local total_flows=0
    local total_projects=0

    # Iterate through each top-level directory (project)
    for project_dir in "$FLOWS_DIR"/*/; do
        [ -d "$project_dir" ] || continue

        local project_name
        project_name=$(basename "$project_dir")

        info "Creating project: $project_name"

        # Create folder in LangFlow
        local folder_id
        folder_id=$(create_folder "$project_name")

        if [ -z "$folder_id" ] || [ "$folder_id" = "null" ]; then
            error "Failed to create/find folder for: $project_name"
            continue
        fi

        total_projects=$((total_projects + 1))

        # Find and upload all JSON files recursively
        while IFS= read -r -d '' flow_file; do
            if upload_flow "$flow_file" "$folder_id"; then
                total_flows=$((total_flows + 1))
            fi
        done < <(find "$project_dir" -name "*.json" -type f -print0)

        echo ""
    done

    echo -e "${GREEN}=============================================================================${NC}"
    echo -e "${GREEN} Flow loading complete!${NC}"
    echo -e "${GREEN}=============================================================================${NC}"
    echo ""
    echo -e "  Projects created: ${BLUE}$total_projects${NC}"
    echo -e "  Flows uploaded:   ${BLUE}$total_flows${NC}"
    echo ""
}

main "$@"
