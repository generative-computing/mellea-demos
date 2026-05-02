#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/frontend"

if [ ! -d node_modules ]; then
    echo "[frontend] Installing dependencies ..."
    npm install
fi

cleanup() {
    echo
    echo "[frontend] Shutting down ..."
    if [ -n "${child_pid:-}" ] && kill -0 "$child_pid" 2>/dev/null; then
        # Kill the whole process group so next-server children die too
        kill -TERM -"$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti :3000 2>/dev/null || true)
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

echo "[frontend] Starting on http://localhost:3000 ..."
# Run in its own process group so we can signal all children on cleanup
set -m
npm run dev &
child_pid=$!
set +m
wait "$child_pid"
