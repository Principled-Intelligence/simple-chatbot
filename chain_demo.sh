#!/usr/bin/env bash
# Drive a multi-turn conversation against POST /v1/responses, chaining each
# turn with the previous response's id, then walk the chain via
# GET /v1/responses/{id} and print the full conversation context.
#
# Pairs nicely with scripted offline mode: run the server with
# SIMPLE_CHATBOT_SCRIPTED_LLM=1 and you can exercise the entire chain wiring
# without any API keys.
#
# Usage:
#   ./chain_demo.sh                              # http://localhost:15078
#   ./chain_demo.sh http://localhost:8000        # custom base URL
#   BASE_URL=http://host:port ./chain_demo.sh
#   SIMPLE_CHATBOT_API_KEY=secret ./chain_demo.sh
#
# Requires: curl, jq.

set -euo pipefail

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

BASE_URL="${BASE_URL:-http://localhost:15078}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  BASE_URL="$1"
fi

API_KEY="${SIMPLE_CHATBOT_API_KEY:-}"
AUTH_ARGS=()
[[ -n "$API_KEY" ]] && AUTH_ARGS=(-H "Authorization: Bearer $API_KEY")

# The conversation: one message per turn, sent in order, each chained to the
# prior response. Edit this array to script a different exchange.
MESSAGES=(
  "What is in the docs?"
  "Tell me more about that."
  "Summarise the conversation so far in one sentence."
)

echo "Driving a ${#MESSAGES[@]}-turn conversation against $BASE_URL"
echo

PREV_ID=""
RESPONSE_IDS=()

for i in "${!MESSAGES[@]}"; do
  turn=$((i + 1))
  msg="${MESSAGES[$i]}"

  if [[ -n "$PREV_ID" ]]; then
    body=$(jq -nc --arg input "$msg" --arg prev "$PREV_ID" \
      '{input: $input, previous_response_id: $prev}')
  else
    body=$(jq -nc --arg input "$msg" '{input: $input}')
  fi

  resp=$(curl -sS -X POST "$BASE_URL/v1/responses" "${AUTH_ARGS[@]}" \
    -H 'Content-Type: application/json' -d "$body")

  resp_id=$(jq -r '.id // empty' <<<"$resp")
  if [[ -z "$resp_id" ]]; then
    echo "FAIL — turn $turn did not return a response id:" >&2
    echo "$resp" >&2
    exit 1
  fi

  RESPONSE_IDS+=("$resp_id")
  PREV_ID="$resp_id"
  printf "→ Turn %d sent: %s\n  ⇒ response_id=%s\n" "$turn" "$msg" "$resp_id"
done

echo
echo "=== Full conversation (retrieved via GET /v1/responses/{id}) ==="

CONV_IDS=()
for i in "${!RESPONSE_IDS[@]}"; do
  turn=$((i + 1))
  resp_id="${RESPONSE_IDS[$i]}"
  msg="${MESSAGES[$i]}"

  resp=$(curl -sS "$BASE_URL/v1/responses/$resp_id" "${AUTH_ARGS[@]}")
  CONV_IDS+=("$(jq -r '.conversation_id' <<<"$resp")")

  echo
  echo "── Turn $turn (${resp_id}) ──"
  echo "User:         $msg"
  jq -r '
    .output[] |
    if .type == "function_call"        then "Tool call:    " + .name + "(" + .arguments + ")"
    elif .type == "function_call_output" then "Tool output:  " + .output
    elif .type == "message"             then "Assistant:    " + (.content[0].text // "")
    else "Other:        " + .type
    end
  ' <<<"$resp"
done

echo
# Verify all turns share the same conversation_id (this is the chain invariant).
uniq_count=$(printf '%s\n' "${CONV_IDS[@]}" | sort -u | wc -l)
if [[ "$uniq_count" -ne 1 ]]; then
  echo "FAIL — conversation_id was not stable across turns:" >&2
  printf '  %s\n' "${CONV_IDS[@]}" >&2
  exit 1
fi

echo "Conversation ID (shared across all turns): ${CONV_IDS[0]}"
echo "PASS"
