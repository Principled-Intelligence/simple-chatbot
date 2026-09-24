# simple_chatbot/fixtures/rag_ignore.py
"""Agent retrieves an authoritative passage, then ignores it in its answer."""

from simple_chatbot.scenario import Agent, Call, Final, Scenario, tool


@tool
def search_docs(query: str) -> dict:
    """Search the knowledge base and return matching passages."""
    return {
        "query": query,
        "passages": ["Refunds are allowed within 14 days of purchase."],
    }


scenario = Scenario(
    id="rag-ignore",
    description="Retrieval-as-tool: agent retrieves a passage then contradicts it.",
    entry="support",
    agents=[
        Agent(
            "support",
            system_prompt="Answer using the knowledge base.",
            tools=[search_docs],
            script=[
                Call(search_docs, {"query": "refund window"}),
                Final(
                    "Refunds are available at any time, with no deadline.",
                    ignore_retrieval=True,
                ),
            ],
        ),
    ],
)
