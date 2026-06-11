## Service overview

A tiered technical-support desk. A front desk triages each request and escalates it
through first-line (tier-1) to specialist (tier-2) support until it can be resolved.

## Surface & capabilities

- **Tiered routing / handoff** — the conversation moves front desk → tier-1 →
  tier-2 as needed, within a single request, until the appropriate tier handles it.
- **Ticket status** (handled by the tier-2 specialist):
  - `check_status(ticket_id: str)` — returns the current status of a support ticket.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **Ticket-status lookups only.** It cannot create, modify, close, or reassign
  tickets.
- **No billing, account, or document-search capabilities.**
- **No streaming.**
