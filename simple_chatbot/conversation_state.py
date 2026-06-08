"""Conversation-scoped mock state for fixture tools.

Stateful fixture tools (order tracking, customer data, billing) need a small
mutable "world" that persists across a conversation's turns yet stays isolated
across parallel conversations. `ConversationStateStore` maps a conversation id
to one `ConvState` (a plain dict), created lazily and bounded by an LRU cap so
the test harness can't grow without limit. Process-local; resets on restart,
like `ResponseStore`.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict


class ConvState(dict):
    """One conversation's mutable mock world. A thin dict subclass so a tool can
    `state.setdefault("orders", {...})` to lazily seed its own slice."""


class ConversationStateStore:
    """Maps conversation_id -> ConvState, lazily created and LRU-bounded."""

    def __init__(self, max_conversations: int = 512) -> None:
        self._lock = asyncio.Lock()
        self._states: "OrderedDict[str, ConvState]" = OrderedDict()
        self._max = max_conversations

    async def get_or_create(self, conversation_id: str) -> ConvState:
        async with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                state = ConvState()
                self._states[conversation_id] = state
            self._states.move_to_end(conversation_id)
            while len(self._states) > self._max:
                self._states.popitem(last=False)
            return state
