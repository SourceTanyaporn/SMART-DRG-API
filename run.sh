#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

uvicorn app.main:app --host "$HOST" --port "$PORT"
