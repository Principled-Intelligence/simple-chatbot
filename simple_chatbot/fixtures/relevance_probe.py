# simple_chatbot/fixtures/relevance_probe.py
"""Fires irrelevant, wrong-value, and redundant calls for a billing question."""

from simple_chatbot.scenario import Agent, Call, Final, Scenario, tool


@tool
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by id."""
    return {"invoice_id": invoice_id, "amount_due": "10.00"}


@tool
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    return {"city": city, "forecast": "rainy"}


scenario = Scenario(
    id="relevance-probe",
    description="Deliberately poor tool choices: irrelevant, wrong-value, redundant.",
    entry="agent",
    agents=[
        Agent(
            "agent",
            system_prompt="A billing agent that deliberately makes poor tool choices.",
            tools=[lookup_invoice, get_weather],
            script=[
                Call(get_weather, {"city": "Paris"}, irrelevant=True),  # wrong tool for billing
                Call(lookup_invoice, {"invoice_id": "INV-999"}, wrong_value=True),  # wrong invoice
                Call(lookup_invoice, {"invoice_id": "INV-999"}, redundant=True),  # duplicate
                Final("Your balance is $10.00."),
            ],
        ),
    ],
)
