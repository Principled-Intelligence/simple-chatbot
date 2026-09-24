# simple_chatbot/fixtures/parallel.py
"""An agent that fans out parallel tool calls in a single round."""

from simple_chatbot.scenario import Agent, Call, Final, Parallel, Scenario, tool


@tool
def get_weather(city: str) -> dict:
    """Get the weather forecast for a city."""
    return {"city": city, "forecast": "sunny"}


@tool
def get_events(city: str) -> dict:
    """List notable events in a city."""
    return {"city": city, "events": ["jazz festival"]}


scenario = Scenario(
    id="parallel",
    description="An agent that gathers facts via parallel tool calls in one round.",
    entry="concierge",
    agents=[
        Agent(
            "concierge",
            system_prompt="Answer travel questions; gather facts in parallel.",
            tools=[get_weather, get_events],
            script=[
                Parallel([
                    Call(get_weather, {"city": "Lisbon"}),
                    Call(get_events, {"city": "Lisbon"}),
                ]),
                Final("Lisbon will be sunny, and there's a jazz festival this weekend."),
            ],
        ),
    ],
)
