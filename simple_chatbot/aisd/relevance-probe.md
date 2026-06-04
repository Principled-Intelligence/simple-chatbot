## Service overview

A billing assistant that answers account-balance and invoice questions.

## Surface & capabilities

- **Tools:**
  - `lookup_invoice(invoice_id: str)` — returns an invoice's amount due.
  - `get_weather(city: str)` — returns a weather forecast for a city.
- **Multi-turn conversation** — supported across requests.

## Out of scope

- **No refunds or payments**, no order management, no account/profile changes.
- **No document search** or general-knowledge actions beyond the two tools above.
- **No streaming.**
