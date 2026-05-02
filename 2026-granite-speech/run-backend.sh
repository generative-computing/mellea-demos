#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "No .env file found. Copying from .env.example ..."
    cp .env.example .env
    echo "Edit .env if your LLM URL or models differ, then re-run."
    exit 1
fi

cleanup() {
    echo
    echo "[backend] Shutting down ..."
    if [ -n "${child_pid:-}" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    # Best-effort: free the port if anything is still bound
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti ":${PORT:-7860}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

echo "[backend] Starting on http://localhost:${PORT:-7860} ..."
# exec-style via background so the trap can fire on Ctrl-C
uv run python -m granite_speech_demo.server &
child_pid=$!
wait "$child_pid"
