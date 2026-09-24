"""Interactive multi-turn REPL for the /v1/responses endpoint.

Drives the server using the OpenAI Python SDK so this also doubles as a
wire-format compatibility check: if the SDK parses our responses cleanly,
the Responses API shape is correct.

Usage:
    uv run python chat.py
    uv run python chat.py --base-url http://localhost:8000/v1
    uv run python chat.py --quiet
    SIMPLE_CHATBOT_API_KEY=secret uv run python chat.py

REPL commands:
    /help      show available commands
    /id        print conversation id and log file path (after first turn)
    /reset     start a new thread (clear previous_response_id)
    /verbose   toggle meta line (response id, conversation id, tokens) per turn
    /quit      exit
    Ctrl-D     exit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap

from openai import OpenAI

_SOURCE_BLOCK_RE = re.compile(r"^\[([^\]]+)\]\n?", re.MULTILINE)

COMMANDS = {
    "/help": "show this help",
    "/id": "print conversation id and JSONL log filename",
    "/reset": "start a new thread",
    "/verbose": "toggle meta line after each turn (on by default)",
    "/quit": "exit (also /exit, /q)",
}


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _style(text: str, code: str) -> str:
    if not _use_color():
        return text
    codes = {
        "dim": "\033[2m",
        "bold": "\033[1m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "reset": "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 3].rstrip() + "..."


def _format_tool_args(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(parsed, ensure_ascii=False, indent=None)


def _format_tool_output(output: str, preview_chars: int) -> list[str]:
    """Split RAG tool output into labeled source blocks for readable display."""
    if not output:
        return ["(empty)"]

    blocks = list(_SOURCE_BLOCK_RE.finditer(output))
    if not blocks:
        return [textwrap.indent(_truncate(output, preview_chars), "      ")]

    lines: list[str] = []
    for idx, match in enumerate(blocks):
        source = match.group(1)
        start = match.end()
        end = blocks[idx + 1].start() if idx + 1 < len(blocks) else len(output)
        body = output[start:end].strip()
        preview = _truncate(body.replace("\n", " "), preview_chars)
        lines.append(f"      {_style(f'[{source}]', 'cyan')} {preview}")
    return lines


def _conversation_log_filename(conversation_id: str) -> str:
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:32]
    return f"{digest}.jsonl"


def _extract_conversation_id(response) -> str | None:
    cid = getattr(response, "conversation_id", None)
    if cid:
        return str(cid)
    extra = getattr(response, "model_extra", None) or {}
    cid = extra.get("conversation_id")
    if cid:
        return str(cid)
    return response.model_dump().get("conversation_id")


def _print_conversation_info(
    conversation_id: str, *, heading: str | None = None
) -> None:
    log_name = _conversation_log_filename(conversation_id)
    if heading:
        print(heading)
    print(f"  conversation_id: {conversation_id}")
    print(f"  log file:        conversations/{log_name}")


def _print_help() -> None:
    print("Commands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<8} {desc}")
    print("  Ctrl-D     exit")


def _print_assistant(text: str) -> None:
    print(_style("assistant:", "bold"))
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        print()
        return
    for idx, para in enumerate(paragraphs):
        print(textwrap.fill(para, width=78))
        if idx + 1 < len(paragraphs):
            print()


def _render_meta(response, conversation_id: str | None) -> None:
    parts = [f"response_id={response.id}"]
    if conversation_id:
        parts.append(f"conversation_id={conversation_id}")
    usage = getattr(response, "usage", None)
    if usage:
        parts.append(
            f"tokens in={usage.input_tokens} out={usage.output_tokens} "
            f"total={usage.total_tokens}"
        )
    print(_style(f"  meta: {' '.join(parts)}", "dim"))


def _render_turn(
    response,
    *,
    conversation_id: str | None,
    verbose: bool,
    tool_preview_chars: int,
) -> None:
    """Render one Responses API turn with paired tool call/output blocks."""
    output = response.output
    tool_step = 0
    i = 0

    while i < len(output):
        item = output[i]

        if item.type == "function_call":
            batch: list = []
            while i < len(output) and output[i].type == "function_call":
                batch.append(output[i])
                i += 1

            outputs_by_call_id: dict[str, object] = {}
            while i < len(output) and output[i].type == "function_call_output":
                out_item = output[i]
                outputs_by_call_id[getattr(out_item, "call_id", "")] = out_item
                i += 1

            for call in batch:
                tool_step += 1
                args = _format_tool_args(getattr(call, "arguments", "") or "")
                print(
                    _style(
                        f"  [{tool_step}] tool {call.name}({args})",
                        "dim",
                    )
                )
                out_item = outputs_by_call_id.get(getattr(call, "call_id", ""))
                if out_item is not None:
                    for line in _format_tool_output(
                        getattr(out_item, "output", "") or "",
                        tool_preview_chars,
                    ):
                        print(line)
                else:
                    print(_style("      (no output yet)", "dim"))
            continue

        if item.type == "function_call_output":
            # Orphan output (call_id not matched above); still show it.
            tool_step += 1
            print(_style(f"  [{tool_step}] tool output (unpaired)", "yellow"))
            for line in _format_tool_output(
                getattr(item, "output", "") or "",
                tool_preview_chars,
            ):
                print(line)
            i += 1
            continue

        if item.type == "reasoning":
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None) or str(summary)
                wrapped = textwrap.fill(
                    text, width=78, initial_indent="  ", subsequent_indent="  "
                )
                print(_style(wrapped, "yellow"))
            i += 1
            continue

        if item.type == "message":
            for part in item.content:
                if getattr(part, "type", None) == "output_text":
                    _print_assistant(part.text or "")
            i += 1
            continue

        print(_style(f"  (unhandled output item: {item.type})", "dim"))
        i += 1

    if verbose:
        _render_meta(response, conversation_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:15078/v1")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SIMPLE_CHATBOT_API_KEY", "placeholder"),
        help="API key sent as Authorization: Bearer; the SDK requires one even if the server doesn't enforce auth.",
    )
    parser.add_argument(
        "--conversation-id",
        default=os.environ.get("SIMPLE_CHATBOT_CONVERSATION_ID"),
        help="Optional stable conversation id (sent as X-Conversation-Id on the first turn).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Hide the meta line (response id, conversation id, token usage) after each turn",
    )
    parser.add_argument(
        "--tool-preview-chars",
        type=int,
        default=240,
        help="Max characters per source block in tool output previews (default: 240).",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    previous_id: str | None = None
    conversation_id: str | None = args.conversation_id
    verbose = not args.quiet

    print(f"Connected to {args.base_url}")
    print(
        "Type your message and press Enter. Commands: /help, /id, /reset, /quit. Ctrl-D exits."
    )
    if conversation_id:
        print(f"Using conversation id: {conversation_id}")
    print()

    def _exit_message() -> None:
        if conversation_id:
            print()
            _print_conversation_info(conversation_id, heading="Session conversation:")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _exit_message()
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit", "/q"}:
            _exit_message()
            return 0
        if user_input == "/help":
            _print_help()
            continue
        if user_input == "/id":
            if conversation_id:
                _print_conversation_info(conversation_id)
            else:
                print("(no conversation yet — send a message first)")
            continue
        if user_input == "/reset":
            previous_id = None
            conversation_id = args.conversation_id
            print("(thread reset)")
            continue
        if user_input == "/verbose":
            verbose = not verbose
            print(f"(verbose {'on' if verbose else 'off'})")
            continue

        kwargs: dict = {"input": user_input}
        if previous_id is not None:
            kwargs["previous_response_id"] = previous_id
        if conversation_id:
            kwargs["extra_headers"] = {"X-Conversation-Id": conversation_id}

        try:
            response = client.responses.create(**kwargs)
        except Exception as exc:
            print(f"! error: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        cid = _extract_conversation_id(response)
        if cid:
            conversation_id = cid

        _render_turn(
            response,
            conversation_id=conversation_id,
            verbose=verbose,
            tool_preview_chars=args.tool_preview_chars,
        )
        previous_id = response.id
        print()


if __name__ == "__main__":
    raise SystemExit(main())
