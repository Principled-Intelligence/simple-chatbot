import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path

from loguru import logger


def derive_conversation_id(messages: list[dict]) -> str:
    """Stable id derived from the first user message in the history.

    Clients that don't pass an explicit id still get consistent grouping
    across turns of the same conversation, because the first user message
    is replayed on every subsequent request.
    """
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False, sort_keys=True)
            return hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return "anon"


def _conversation_log_filename(conversation_id: str) -> str:
    digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:32]
    return f"{digest}.jsonl"


def _derive_tool_trace(tool_messages: list[dict]) -> tuple[list[dict], str]:
    """Extract the tool trace from the agent's chat-shape tool_messages.

    Returns (tool_calls, function_call_output) where:
    - tool_calls: [{"name", "arguments"}] for every assistant tool call.
    - function_call_output: the concatenated content of the `tool` role
      messages — the literal function_call_output already formatted as
      "[source]\\n<text>" blocks by the search tool.
    """
    tool_calls: list[dict] = []
    outputs: list[str] = []
    for m in tool_messages:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tool_calls.append(
                    {"name": fn.get("name", ""), "arguments": fn.get("arguments", "") or ""}
                )
        elif role == "tool":
            content = m.get("content")
            if content:
                outputs.append(content)
    return tool_calls, "\n\n".join(outputs)


class ConversationLogger:
    """Append-only, per-conversation JSONL logger safe for concurrent use.

    - One file per conversation so concurrent
      chats never interleave into the same file.
    - Each conversation has its own asyncio.Lock so overlapping turns of
      the same conversation serialize their writes.
    - File I/O is offloaded with asyncio.to_thread so the event loop
      isn't blocked while many requests are in flight.
    - Only the latest user turn and assistant response are stored per
      line; the full history is recoverable by concatenating lines.
    """

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._dir = log_dir
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        logger.bind(path=str(self._dir)).info("Conversation log directory ready")

    async def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock

    async def log(
        self,
        conversation_id: str,
        messages: list[dict],
        response: str,
        chunks: list[dict],
        tool_messages: list[dict] | None = None,
    ) -> None:
        last_user = next(
            (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        tool_calls, function_call_output = _derive_tool_trace(tool_messages or [])
        record = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "turn_index": sum(1 for m in messages if m.get("role") == "user"),
            "last_user_message": last_user,
            "response": response,
            "made_tool_call": bool(tool_calls),
            "tool_calls": tool_calls,
            "function_call_output": function_call_output,
            "retrieved_chunks": chunks,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        path = self._dir / _conversation_log_filename(conversation_id)

        lock = await self._lock_for(conversation_id)
        async with lock:
            try:
                await asyncio.to_thread(self._append, path, line)
            except Exception as e:
                logger.bind(
                    conversation_id=conversation_id,
                    error=str(e),
                ).warning("Failed to write conversation log")

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
