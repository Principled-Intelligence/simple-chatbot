# Billing & Refund Policy

## Invoice states

- **open** — payment due or partially settled.
- **paid** — settled in full.
- **refunded** — a refund has been issued against the invoice.

## Refund rules

- A refund **must not exceed the invoice's `amount_due`**. Partial refunds (less than the full amount) are allowed; over-refunds are never permitted.
- Refunds are available within **30 days** of the invoice date. Beyond that window, escalate to a human.
- Refunds are returned to the original payment method and typically settle in **5–10 business days**.
- Issuing a refund moves the invoice to the **refunded** state.

## Agent guidance (billing)

- Use `lookup_invoice(invoice_id)` to read the invoice and its `amount_due` before issuing anything.
- Use `issue_refund(invoice_id, amount)` only after confirming `amount` does not exceed `amount_due` and the invoice is within the refund window.
- For order status/cancellation or profile changes, route back to the router.
