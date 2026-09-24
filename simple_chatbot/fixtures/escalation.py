# simple_chatbot/fixtures/escalation.py
"""Dispatcher escalates straight to a human terminal agent."""

from simple_chatbot.scenario import Agent, Route, Scenario


scenario = Scenario(
    id="escalation",
    description="Dispatcher routes directly to a human terminal agent.",
    entry="dispatcher",
    agents=[
        Agent(
            "dispatcher",
            system_prompt="Escalate anything you cannot handle to a human.",
            routes=["human"],
            script=[Route("human")],
        ),
        Agent(
            "human",
            terminal=True,
            escalation_message="A human agent will take over from here.",
        ),
    ],
)
