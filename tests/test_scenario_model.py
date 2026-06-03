# tests/test_scenario_model.py
import unittest

from simple_chatbot.scenario import Agent, Scenario, tool, ScenarioTool, Call, Route, Final, MalformedCall, UnknownToolCall


class ToolDecoratorTests(unittest.TestCase):
    def test_tool_builds_schema_from_signature(self):
        @tool
        def lookup_invoice(invoice_id: str) -> dict:
            """Look up an invoice by id."""
            return {"invoice_id": invoice_id, "status": "open"}

        self.assertIsInstance(lookup_invoice, ScenarioTool)
        self.assertEqual(lookup_invoice.name, "lookup_invoice")
        self.assertEqual(lookup_invoice.description, "Look up an invoice by id.")
        self.assertEqual(lookup_invoice.parameters["type"], "object")
        self.assertIn("invoice_id", lookup_invoice.parameters["properties"])
        self.assertEqual(
            lookup_invoice.parameters["properties"]["invoice_id"]["type"], "string"
        )
        self.assertEqual(lookup_invoice.parameters["required"], ["invoice_id"])

    def test_optional_param_not_required(self):
        @tool
        def get_time(tz: str = "UTC") -> dict:
            """Return the current time."""
            return {"tz": tz}

        self.assertNotIn("required", get_time.parameters)
        self.assertEqual(get_time.parameters["properties"]["tz"]["type"], "string")

    def test_stub_body_is_callable(self):
        @tool
        def echo(value: str) -> dict:
            """Echo."""
            return {"echoed": value}

        self.assertEqual(echo.func(value="hi"), {"echoed": "hi"})


class StepTypeTests(unittest.TestCase):
    def _t(self):
        @tool
        def lookup_invoice(invoice_id: str) -> dict:
            """Look up an invoice."""
            return {"invoice_id": invoice_id}
        return lookup_invoice

    def test_call_holds_tool_and_args(self):
        t = self._t()
        c = Call(t, {"invoice_id": "INV-1"})
        self.assertIs(c.tool, t)
        self.assertEqual(c.args, {"invoice_id": "INV-1"})
        self.assertFalse(c.irrelevant)

    def test_call_defaults_to_empty_args(self):
        c = Call(self._t())
        self.assertEqual(c.args, {})

    def test_route_holds_target(self):
        self.assertEqual(Route("billing").target, "billing")

    def test_final_flags(self):
        f = Final("done", ignore_retrieval=True)
        self.assertEqual(f.text, "done")
        self.assertTrue(f.ignore_retrieval)

    def test_malformed_and_unknown(self):
        t = self._t()
        self.assertIs(MalformedCall(t).tool, t)
        self.assertEqual(UnknownToolCall("ghost_tool").name, "ghost_tool")


class ScenarioModelTests(unittest.TestCase):
    def _tools(self):
        @tool
        def lookup_invoice(invoice_id: str) -> dict:
            """Look up an invoice."""
            return {"invoice_id": invoice_id}

        @tool
        def get_status(order_id: str) -> dict:
            """Get order status."""
            return {"order_id": order_id, "status": "shipped"}

        return lookup_invoice, get_status

    def _scenario(self):
        inv, status = self._tools()
        return Scenario(
            id="demo",
            entry="dispatcher",
            agents=[
                Agent("dispatcher", routes=["billing", "orders"]),
                Agent("billing", tools=[inv]),
                Agent("orders", tools=[status]),
            ],
        )

    def test_agent_lookup(self):
        s = self._scenario()
        self.assertEqual(s.agent("billing").name, "billing")

    def test_all_tools_is_union_deduped_by_name(self):
        s = self._scenario()
        names = sorted(t.name for t in s.all_tools())
        self.assertEqual(names, ["get_status", "lookup_invoice"])

    def test_route_targets_sorted_unique(self):
        s = self._scenario()
        self.assertEqual(s.route_targets(), ["billing", "orders"])

    def test_validate_rejects_missing_entry(self):
        inv, _ = self._tools()
        with self.assertRaises(ValueError):
            Scenario(id="x", entry="nope", agents=[Agent("a", tools=[inv])])

    def test_validate_rejects_route_to_unknown_agent(self):
        with self.assertRaises(ValueError):
            Scenario(id="x", entry="a", agents=[Agent("a", routes=["ghost"])])

    def test_validate_rejects_duplicate_agent_names(self):
        with self.assertRaises(ValueError):
            Scenario(id="x", entry="a", agents=[Agent("a"), Agent("a")])
