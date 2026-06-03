# simple_chatbot/fixtures/validity_probe.py
"""A single-agent scenario that fires each deterministic validity knob:
unknown tool (→1), malformed args (→2), required-arg violation (→2)."""

from simple_chatbot.scenario import (
    Agent,
    Call,
    Final,
    MalformedCall,
    Scenario,
    UnknownToolCall,
    tool,
)


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""
    return {"invoice_id": invoice_id, "status": "open"}


@tool
def get_order(order_id: str) -> dict:
    """Get an order by id."""
    return {"order_id": order_id, "status": "pending"}


scenario = Scenario(
    id="validity-probe",
    description="Fires unknown-tool, malformed-args, and required-violation knobs.",
    entry="prober",
    agents=[
        Agent(
            "prober",
            system_prompt="A probe agent that deliberately misuses tools.",
            tools=[lookup_invoice, get_order],
            script=[
                UnknownToolCall("ghost_tool"),
                MalformedCall(lookup_invoice),
                Call(get_order),  # omits required order_id → required violation
                Final("Probe complete."),
            ],
        ),
    ],
)
