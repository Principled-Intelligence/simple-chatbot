# Customer Data Policy

## Editable profile fields

Customers may update the following self-serve fields:

- **email**
- **address**

All other fields — `customer_id`, legal name, date of birth — are **not** self-serve and require a verified identity check handled by a human agent.

## Identity verification

- Before changing contact details (email or address), confirm the customer's identity using information already on file.
- Never read back full sensitive values; confirm only enough to establish identity.

## Data rights

- Customers may request a copy of their data (export) or deletion of their account. These requests are out of scope for self-serve tooling — escalate to a human.

## Agent guidance (customer_data)

- Use `get_customer()` to read the current profile.
- Use `update_customer(field, value)` only for `email` or `address`; any other field returns an error and should be escalated.
- For orders or billing questions, route back to the router.
