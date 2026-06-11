## Service overview

A customer-support assistant that routes requests to the right specialist. A
front-desk router identifies intent and hands off to one of three specialist
subagents (order tracking, customer data, billing) or escalates to a human
support agent. Each subagent maintains stateful mock data per conversation so
that changes on one turn are visible on later turns.

## Surface & capabilities

- **Routing / handoff** — the router does not answer domain questions itself;
  it uses the `route` tool to dispatch to the correct specialist:
  - `order_tracking` for order status or cancellation requests
  - `customer_data` for customer profile reads or updates
  - `billing` for invoice lookups or refund requests
  - `human` for anything else (terminal escalation)
- **Subagents can route back** — each subagent can return control to the
  `router` for out-of-scope requests.
- **Order-tracking tools** (handled by the order_tracking subagent):
  - `get_order(order_id: str)` — returns an order's current status.
  - `cancel_order(order_id: str)` — cancels an order by id.
- **Customer-data tools** (handled by the customer_data subagent):
  - `get_customer()` — returns the current customer profile.
  - `update_customer(field: str, value: str)` — updates `email` or `address`.
- **Billing tools** (handled by the billing subagent):
  - `lookup_invoice(invoice_id: str)` — returns an invoice's amount due and
    status.
  - `issue_refund(invoice_id: str, amount: float)` — issues a refund against
    an invoice.
- **Policy lookup** (shared by all three subagents):
  - `lookup_policy(topic: str)` — returns the relevant support policy or FAQ
    text (topics: `orders`, `refunds`, `customer_data`, `escalation`, `faq`),
    sourced from the `kbs/support_swarm/` documents, so subagent actions stay
    consistent with written policy.
- **Human escalation** — ends the automated turn with a hand-off message.
- **Stateful mock world** — order, customer, and invoice data is seeded
  lazily per conversation and mutated in-place, so cancellations and profile
  changes persist across turns.
- **Multi-turn conversation** — state and active-agent context are preserved
  across chained requests. Per-conversation state isolation relies on a stable
  conversation identity: clients should pass a consistent `user` field (or
  `x-conversation-id` header), otherwise the conversation id is derived from
  the first user message and two conversations that open identically would
  share state.

## Out of scope

- **No general-knowledge tasks** — the system only handles order management,
  customer profile changes, and billing operations.
- **No actions beyond the declared tools** (plus routing and human handoff).
- **No streaming.**
