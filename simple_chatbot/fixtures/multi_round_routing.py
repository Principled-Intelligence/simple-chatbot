# simple_chatbot/fixtures/multi_round_routing.py
"""Front desk routes to tier-1, which routes to tier-2 — two hops, one turn."""

from simple_chatbot.scenario import Agent, Call, Final, Route, Scenario, tool


@tool
def check_status(ticket_id: str) -> dict:
    """Check the status of a support ticket."""
    return {"ticket_id": ticket_id, "status": "escalated"}


scenario = Scenario(
    id="multi-round-routing",
    description="Two-hop handoff (front desk -> tier1 -> tier2) within one user turn.",
    entry="front_desk",
    agents=[
        Agent(
            "front_desk",
            system_prompt="Triage requests and route to first-line support.",
            routes=["tier1"],
            script=[Route("tier1")],
        ),
        Agent(
            "tier1",
            system_prompt="First-line support; escalate hard cases to a specialist.",
            routes=["tier2"],
            script=[Route("tier2")],
        ),
        Agent(
            "tier2",
            system_prompt="Specialist support.",
            tools=[check_status],
            script=[
                Call(check_status, {"ticket_id": "T-9"}),
                Final("Resolved at tier 2."),
            ],
        ),
    ],
)
