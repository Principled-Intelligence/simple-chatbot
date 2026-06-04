## Service overview

An account-operations assistant for invoices, orders, and customer credit limits.

## Surface & capabilities

- **Tools:**
  - `lookup_invoice(invoice_id: str)` — returns an invoice's status.
  - `get_order(order_id: str)` — returns an order's status.
  - `set_credit_limit(limit: int)` — sets a customer's credit limit.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **No refunds, no human handoff, no document search.**
- **Only the three operations above**; no other account or billing actions.
- **No streaming.**
