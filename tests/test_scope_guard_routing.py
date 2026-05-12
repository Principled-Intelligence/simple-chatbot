import unittest
from unittest.mock import patch

from simple_chatbot.cli import _resolve_guard_api_key
from simple_chatbot.config import GuardConfig
from simple_chatbot.guard import ScopeGuardGate


class ScopeGuardRoutingTests(unittest.TestCase):
    def test_env_orbitals_key_is_ignored_when_self_hosted_url_is_set(self):
        with patch.dict("os.environ", {"ORBITALS_API_KEY": "env-key"}):
            key = _resolve_guard_api_key(None, "http://internal-guard")

        self.assertIsNone(key)

    def test_explicit_guard_key_is_kept_with_self_hosted_url(self):
        with patch.dict("os.environ", {"ORBITALS_API_KEY": "env-key"}):
            key = _resolve_guard_api_key("explicit-key", "http://internal-guard")

        self.assertEqual(key, "explicit-key")

    def test_self_hosted_api_url_is_not_replaced_by_api_key(self):
        cfg = GuardConfig(
            enabled=True,
            backend="api",
            api_url="http://internal-guard",
            api_key="explicit-key",
            service_description="Answer only from internal docs.",
        )

        with patch("simple_chatbot.guard.ScopeGuard") as scope_guard:
            ScopeGuardGate(cfg, fallback_description=None)

        scope_guard.assert_called_once_with(
            backend="api",
            api_url="http://internal-guard",
            api_key="explicit-key",
        )

    def test_api_key_without_url_uses_hosted_scope_guard(self):
        cfg = GuardConfig(
            enabled=True,
            backend="api",
            api_key="cloud-key",
            service_description="Answer only from internal docs.",
        )

        with patch("simple_chatbot.guard.ScopeGuard") as scope_guard:
            ScopeGuardGate(cfg, fallback_description=None)

        scope_guard.assert_called_once_with(
            backend="api",
            api_url="https://api.orbitals.principled.app",
            api_key="cloud-key",
        )


if __name__ == "__main__":
    unittest.main()
