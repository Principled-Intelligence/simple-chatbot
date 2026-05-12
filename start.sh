#!/usr/bin/env bash
set -euo pipefail

# API keys — set these in your environment or uncomment and fill in:
# export OPENAI_API_KEY="sk-..."
# export VOYAGE_API_KEY="pa-..."

DOCS_DIR="${1:-./docs}"

simple-chatbot serve \
  --docs-dir "$DOCS_DIR" \
  --embedding-model "openai/text-embedding-3-small" \
  --chat-model "openai/gpt-5.4-nano" \
  --port 15077 \
  --log-level INFO
