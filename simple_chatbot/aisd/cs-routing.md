## Service overview

A customer-service assistant for billing matters. A front-line dispatcher triages
each request and hands it off to the right specialist: a billing agent or a human
support agent.

## Surface & capabilities

- **Routing / handoff** — the dispatcher does not answer domain questions itself; it
  routes the conversation to either the `billing` specialist or a `human` agent, and
  once routed the conversation stays with that specialist.
- **Billing tools** (handled by the billing specialist):
  - `lookup_invoice(invoice_id: str)` — returns an invoice's amount due and status.
  - `issue_refund(invoice_id: str, amount: float)` — issues a refund against an
    invoice.
- **Human escalation** — the request can be handed to a human agent, which ends the
  automated turn with a hand-off message.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Billing domain only.** No order management, account/profile changes, shipping,
  or general-knowledge tasks.
- **No actions beyond invoice lookup and refund issuance** (plus human handoff).
- **No streaming.**
