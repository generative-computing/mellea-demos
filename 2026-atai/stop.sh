#!/usr/bin/env bash
# =============================================================================
# ATAI Demo — Stop
# =============================================================================
# Stops all demo services.
#
# Usage:
#   ./stop.sh          # stop services (data volumes preserved)
#   ./stop.sh --clean  # stop services and delete all data volumes
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ "${1:-}" = "--clean" ]; then
    echo "Stopping ATAI demo services and deleting all data..."
    docker compose down -v --timeout 15
    echo "All services stopped. Data volumes deleted."
else
    echo "Stopping ATAI demo services..."
    docker compose down --timeout 15
    echo "All services stopped. Data volumes preserved."
    echo "To also delete all data: ./stop.sh --clean"
fi
