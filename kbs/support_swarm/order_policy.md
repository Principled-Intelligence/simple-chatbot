# Order Management Policy

## Order lifecycle

An order moves through these states:

1. **open** — placed but not yet picked/packed.
2. **processing** — being prepared for shipment.
3. **shipped** — handed to the carrier; in transit.
4. **delivered** — received by the customer.
5. **cancelled** — terminal; the order will not be fulfilled.

## Cancellation rules

- An order may be cancelled while it is **open** or **processing**.
- Once an order is **shipped**, it can no longer be cancelled — the customer must use the returns flow after delivery.
- Cancellation is immediate and final; a cancelled order cannot be reinstated. The customer must place a new order.

## Agent guidance (order_tracking)

- Use `get_order(order_id)` to read the current status before acting.
- Use `cancel_order(order_id)` only after confirming the order is in a cancellable state and the customer has confirmed intent.
- If the request is about billing/refunds or profile changes, route back to the router rather than handling it here.
