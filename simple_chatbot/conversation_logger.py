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
        misbehavior_injections: list[dict] | None = None,
    ) -> None:
        last_user = next(
            (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        record = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "turn_index": sum(1 for m in messages if m.get("role") == "user"),
            "last_user_message": last_user,
            "response": response,
            "retrieved_chunks": chunks,
        }
        if misbehavior_injections:
            record["misbehavior_injections"] = misbehavior_injections
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
