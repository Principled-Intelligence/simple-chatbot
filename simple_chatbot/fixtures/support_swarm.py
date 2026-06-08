"""Customer-support swarm: a router that hands off, statefully, to specialized
subagents (order tracking, customer data, billing) or escalates to a human.

Runs in `live` mode — the LLM drives routing and tool use. The tools are mocks
whose state lives in a per-conversation `ConvState` (injected via the `state`
param), so a change made on one turn is visible on later turns of the same
conversation while staying isolated across parallel conversations. Each tool
lazily seeds its own slice of the world via `state.setdefault(...)`.
"""

from simple_chatbot.scenario import Agent, Scenario, tool


def _orders(state):
    return state.setdefault(
        "orders",
        {"ORD-1": {"order_id": "ORD-1", "item": "Wireless headphones", "status": "open"}},
    )


@tool
def get_order(order_id: str, state) -> dict:
    """Look up an order's current status by id."""
    order = _orders(state).get(order_id)
    if order is None:
        return {"error": f"no order {order_id!r}"}
    return dict(order)


@tool
def cancel_order(order_id: str, state) -> dict:
    """Cancel an order by id."""
    orders = _orders(state)
    order = orders.get(order_id)
    if order is None:
        return {"error": f"no order {order_id!r}"}
    order["status"] = "cancelled"
    return dict(order)


def _customer(state):
    return state.setdefault(
        "customer",
        {"customer_id": "CUST-1", "email": "ada@example.com", "address": "1 Analytical Way"},
    )


@tool
def get_customer(state) -> dict:
    """Return the current customer profile."""
    return dict(_customer(state))


@tool
def update_customer(field: str, value: str, state) -> dict:
    """Update a customer profile field ('email' or 'address')."""
    customer = _customer(state)
    if field not in ("email", "address"):
        return {"error": f"unknown field {field!r}"}
    customer[field] = value
    return dict(customer)


def _invoices(state):
    return state.setdefault(
        "invoices",
        {"INV-1": {"invoice_id": "INV-1", "amount_due": "42.00", "status": "open"}},
    )


@tool
def lookup_invoice(invoice_id: str, state) -> dict:
    """Look up an invoice by id."""
    invoice = _invoices(state).get(invoice_id)
    if invoice is None:
        return {"error": f"no invoice {invoice_id!r}"}
    return dict(invoice)


@tool
def issue_refund(invoice_id: str, amount: float, state) -> dict:
    """Issue a refund against an invoice."""
    invoice = _invoices(state).get(invoice_id)
    if invoice is None:
        return {"error": f"no invoice {invoice_id!r}"}
    invoice["status"] = "refunded"
    return {"refunded": True, "invoice_id": invoice_id, "amount": amount}


scenario = Scenario(
    id="support-swarm",
    description=(
        "Router dispatching to order-tracking / customer-data / billing "
        "subagents, with human escalation."
    ),
    entry="router",
    mode="live",
    agents=[
        Agent(
            "router",
            system_prompt=(
                "You are the front-desk router. Identify the user's intent and hand off to "
                "the right specialist with the `route` tool: order_tracking for order status "
                "or cancellation, customer_data for profile changes, billing for invoices or "
                "refunds, human for anything else. Do not answer domain questions yourself."
            ),
            routes=["order_tracking", "customer_data", "billing", "human"],
        ),
        Agent(
            "order_tracking",
            system_prompt=(
                "You handle order status and cancellations. Use your tools. If the request is "
                "out of scope, route back to the router."
            ),
            tools=[get_order, cancel_order],
            routes=["router"],
        ),
        Agent(
            "customer_data",
            system_prompt=(
                "You handle customer profile reads and updates. Use your tools. If the request "
                "is out of scope, route back to the router."
            ),
            tools=[get_customer, update_customer],
            routes=["router"],
        ),
        Agent(
            "billing",
            system_prompt=(
                "You handle invoices and refunds. Use your tools. If the request is out of "
                "scope, route back to the router."
            ),
            tools=[lookup_invoice, issue_refund],
            routes=["router"],
        ),
        Agent(
            "human",
            terminal=True,
            escalation_message="Escalating to a human agent.",
        ),
    ],
)
