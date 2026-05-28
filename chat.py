"""Interactive multi-turn REPL for the /v1/responses endpoint.

Drives the server using the OpenAI Python SDK so this also doubles as a
wire-format compatibility check: if the SDK parses our responses cleanly,
the Responses API shape is correct.

Usage:
    uv run python chat.py
    uv run python chat.py --base-url http://localhost:8000/v1
    uv run python chat.py --verbose
    SIMPLE_CHATBOT_API_KEY=secret uv run python chat.py

REPL commands:
    /reset     start a new thread (clear previous_response_id)
    /quit      exit
    Ctrl-D     exit
"""

from __future__ import annotations

import argparse
import os
import sys

from openai import OpenAI


def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _print_response(response, verbose: bool) -> None:
    """Render an OpenAI Response's output items in a compact REPL format."""
    for item in response.output:
        if item.type == "function_call":
            print(f"* tool called: {item.name}({_truncate(item.arguments, 80)})")
        elif item.type == "function_call_output":
            print(f"* tool output: {_truncate(item.output, 200)}")
        elif item.type == "reasoning":
            for s in getattr(item, "summary", []) or []:
                text = getattr(s, "text", None) or str(s)
                print(f"* reasoning: {text}")
        elif item.type == "message":
            for part in item.content:
                if getattr(part, "type", None) == "output_text":
                    print(f"< {part.text}")
        else:
            print(f"* {item.type} (unhandled item type)")

    if verbose:
        usage = getattr(response, "usage", None)
        if usage:
            print(
                f"  [response_id={response.id} "
                f"usage in={usage.input_tokens} out={usage.output_tokens} total={usage.total_tokens}]"
            )
        else:
            print(f"  [response_id={response.id}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:15078/v1")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SIMPLE_CHATBOT_API_KEY", "placeholder"),
        help="API key sent as Authorization: Bearer; the SDK requires one even if the server doesn't enforce auth.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print response id and token usage per turn")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    previous_id: str | None = None

    print(f"Connected to {args.base_url}")
    print("Type your message and press Enter. Commands: /reset, /quit. Ctrl-D exits.")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit", "/q"}:
            return 0
        if user_input == "/reset":
            previous_id = None
            print("(thread reset)")
            continue

        kwargs: dict = {"input": user_input}
        if previous_id is not None:
            kwargs["previous_response_id"] = previous_id

        try:
            response = client.responses.create(**kwargs)
        except Exception as exc:
            print(f"! error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        _print_response(response, verbose=args.verbose)
        previous_id = response.id
        print()


if __name__ == "__main__":
    raise SystemExit(main())
