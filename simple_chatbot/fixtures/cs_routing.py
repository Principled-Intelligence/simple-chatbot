# simple_chatbot/fixtures/cs_routing.py
"""Customer-service dispatcher routing to billing or human escalation."""

from simple_chatbot.scenario import Agent, Call, Final, Route, Scenario, tool


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""
    return {"invoice_id": invoice_id, "amount_due": "42.00", "status": "open"}


@tool
def issue_refund(invoice_id: str, amount: float) -> dict:
    """Issue a refund against an invoice."""
    return {"refunded": True, "invoice_id": invoice_id, "amount": amount}


scenario = Scenario(
    id="cs-routing",
    description="CS dispatcher routing to billing / human escalation.",
    entry="dispatcher",
    agents=[
        Agent(
            "dispatcher",
            system_prompt="Route requests to a specialist; don't answer domain questions yourself.",
            routes=["billing", "human"],
            script=[Route("billing")],
        ),
        Agent(
            "billing",
            system_prompt="You handle billing and refunds.",
            tools=[lookup_invoice, issue_refund],
            script=[
                Call(lookup_invoice, {"invoice_id": "INV-1"}),
                Call(issue_refund, {"invoice_id": "INV-1", "amount": 42.0}),
                Final("I've refunded $42.00 against invoice INV-1."),
            ],
        ),
        Agent(
            "human",
            terminal=True,
            escalation_message="Escalating to a human agent.",
        ),
    ],
)
