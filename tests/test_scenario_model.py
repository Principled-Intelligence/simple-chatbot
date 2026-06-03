# tests/test_scenario_model.py
import unittest

from simple_chatbot.scenario import tool, ScenarioTool, Call, Route, Final, MalformedCall, UnknownToolCall


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
