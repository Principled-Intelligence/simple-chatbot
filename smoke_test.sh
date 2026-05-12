#!/usr/bin/env bash
# Invoke smoke_test.py against a running simple-chatbot server.
#
# Usage:
#   ./smoke_test.sh                              # default: http://localhost:15077
#   ./smoke_test.sh http://host:port             # custom base URL (first arg, no leading dash)
#   BASE_URL=http://host:port ./smoke_test.sh
#   ./smoke_test.sh --message "hi"               # flags forwarded to smoke_test.py
#   ./smoke_test.sh http://host:port --timeout 120

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:15077}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  BASE_URL="$1"
  shift
fi

if [[ -x "$HERE/.venv/bin/python" ]]; then
  PYTHON="$HERE/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" "$HERE/smoke_test.py" --base-url "$BASE_URL" "$@"
