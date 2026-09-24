# Escalation & Triage Guide

## Routing map (router)

The router does not answer domain questions itself. It identifies intent and hands off:

| Intent | Specialist |
| --- | --- |
| Order status, tracking, or cancellation | `order_tracking` |
| Profile changes (email, address) | `customer_data` |
| Invoices, payments, refunds | `billing` |
| Anything else, or an explicit request for a person | `human` |

A specialist that receives an out-of-scope request should route **back to the router**, which can then dispatch to the correct specialist.

## When to escalate to a human

Escalate to the `human` agent when:

- The request falls outside all specialist scopes.
- The customer explicitly asks to speak to a person.
- A policy exception is required (e.g. a refund outside the 30-day window, a non-self-serve data change).
- The customer is frustrated or the automated flow has failed repeatedly.

The `human` agent is **terminal**: once a conversation is escalated, it ends with an escalation message and is handed to a person.
