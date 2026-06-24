#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "[WARNING] ANTHROPIC_API_KEY not set — AI chat will be disabled"
  echo "  Set it with: export ANTHROPIC_API_KEY=your_key"
  echo "  Or add it to a .env file in this directory"
fi

echo "[Kroniq] Starting on http://localhost:${FLASK_PORT:-8080}"
exec python3 app.py
