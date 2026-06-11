# tests/test_scenario_model.py
import unittest

from simple_chatbot.scenario import Agent, Scenario, tool, ScenarioTool, Call, Route, Final, MalformedCall, UnknownToolCall, Parallel


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

    def test_state_param_excluded_from_schema_and_flagged(self):
        @tool
        def get_order(order_id: str, state) -> dict:
            """Look up an order."""
            return {"order_id": order_id}

        self.assertTrue(get_order.wants_state)
        self.assertNotIn("state", get_order.parameters["properties"])
        self.assertEqual(get_order.parameters["required"], ["order_id"])

    def test_tool_without_state_is_not_flagged(self):
        @tool
        def ping(message: str) -> dict:
            """Echo."""
            return {"message": message}

        self.assertFalse(ping.wants_state)
        self.assertIn("message", ping.parameters["properties"])


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

    def test_validate_rejects_call_to_tool_agent_does_not_own(self):
        inv, _ = self._tools()
        with self.assertRaises(ValueError):
            # agent "a" has no tools but its script calls `inv`
            Scenario(id="x", entry="a", agents=[Agent("a", script=[Call(inv, {"invoice_id": "1"})])])

    def test_validate_rejects_user_tool_named_route(self):
        @tool
        def route(agent: str) -> dict:
            """A tool that collides with the generated handoff tool."""
            return {"agent": agent}

        with self.assertRaises(ValueError):
            Scenario(id="x", entry="a", agents=[Agent("a", tools=[route])])


class ScenarioModeTests(unittest.TestCase):
    def test_mode_defaults_to_deterministic(self):
        s = Scenario(id="m", entry="a", agents=[Agent("a")])
        self.assertEqual(s.mode, "deterministic")

    def test_mode_accepts_live(self):
        s = Scenario(id="m", entry="a", agents=[Agent("a")], mode="live")
        self.assertEqual(s.mode, "live")


class ParallelStepTests(unittest.TestCase):
    def _tool(self):
        @tool
        def lookup(invoice_id: str) -> dict:
            """Look up an invoice."""
            return {"invoice_id": invoice_id}
        return lookup

    def test_parallel_holds_call_family_steps(self):
        lookup = self._tool()
        p = Parallel([Call(lookup, {"invoice_id": "INV-1"}), MalformedCall(lookup)])
        self.assertEqual(len(p.steps), 2)

    def test_valid_parallel_scenario_constructs(self):
        lookup = self._tool()
        s = Scenario(
            id="ok",
            entry="a",
            agents=[Agent("a", tools=[lookup], script=[Parallel([Call(lookup, {"invoice_id": "INV-1"})])])],
        )
        self.assertEqual(s.entry, "a")

    def test_parallel_rejects_route_inside(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([Route("a")])])])

    def test_parallel_rejects_final_inside(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([Final("x")])])])

    def test_parallel_rejects_nested_parallel(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([Parallel([])])])])

    def test_parallel_rejects_empty_steps(self):
        with self.assertRaises(ValueError):
            Scenario(id="bad", entry="a", agents=[Agent("a", script=[Parallel([])])])


class CallTagTests(unittest.TestCase):
    def _tool(self):
        @tool
        def t(x: str) -> dict:
            """t."""
            return {"x": x}
        return t

    def test_tags_default_false(self):
        c = Call(self._tool(), {"x": "1"})
        self.assertFalse(c.redundant)
        self.assertFalse(c.wrong_value)
        self.assertFalse(c.type_mismatch)

    def test_tags_settable(self):
        c = Call(self._tool(), {"x": "1"}, redundant=True, wrong_value=True, type_mismatch=True)
        self.assertTrue(c.redundant)
        self.assertTrue(c.wrong_value)
        self.assertTrue(c.type_mismatch)
