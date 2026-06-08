import asyncio
import unittest

from simple_chatbot.conversation_state import ConversationStateStore, ConvState


def _run(coro):
    return asyncio.run(coro)


class ConversationStateStoreTests(unittest.TestCase):
    def test_get_or_create_returns_same_instance_per_id(self):
        store = ConversationStateStore()
        a = _run(store.get_or_create("conv-1"))
        a["orders"] = {"ORD-1": "open"}
        b = _run(store.get_or_create("conv-1"))
        self.assertIs(a, b)
        self.assertEqual(b["orders"], {"ORD-1": "open"})

    def test_distinct_ids_get_distinct_states(self):
        store = ConversationStateStore()
        a = _run(store.get_or_create("conv-1"))
        b = _run(store.get_or_create("conv-2"))
        self.assertIsNot(a, b)

    def test_returns_convstate_instances(self):
        store = ConversationStateStore()
        self.assertIsInstance(_run(store.get_or_create("conv-1")), ConvState)

    def test_lru_evicts_oldest_beyond_cap(self):
        store = ConversationStateStore(max_conversations=2)
        first = _run(store.get_or_create("conv-1"))
        first["seen"] = True
        _run(store.get_or_create("conv-2"))
        _run(store.get_or_create("conv-3"))  # evicts conv-1 (oldest)
        # conv-1 is recreated fresh, so the prior mutation is gone
        self.assertNotIn("seen", _run(store.get_or_create("conv-1")))

    def test_access_refreshes_recency(self):
        store = ConversationStateStore(max_conversations=2)
        _run(store.get_or_create("conv-1"))
        _run(store.get_or_create("conv-2"))
        _run(store.get_or_create("conv-1"))  # touch conv-1 → conv-2 now oldest
        marked = _run(store.get_or_create("conv-1"))
        marked["kept"] = True
        _run(store.get_or_create("conv-3"))  # evicts conv-2, not conv-1
        self.assertTrue(_run(store.get_or_create("conv-1")).get("kept"))


if __name__ == "__main__":
    unittest.main()
