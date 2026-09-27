# Orders & Checkout — Acme

The **orders-service** owns the order lifecycle; **checkout** is the saga that
creates an order safely across services. Code: `code/orders.py`, `code/checkout.py`.

## Order states

Orders move through a fixed set of states:

```
pending -> paid -> fulfilled -> shipped -> delivered
                \-> cancelled
     paid       \-> refunded
```

- `pending` — order created, payment not yet confirmed.
- `paid` — payment succeeded; inventory reservation committed.
- `fulfilled` — picked & packed in the warehouse.
- `shipped` / `delivered` — carrier updates.
- `cancelled` — before fulfilment, releases stock.
- `refunded` — money returned via payments-service; writes a ledger entry.

Only these values are allowed; `transition()` rejects anything else.

## The checkout saga

1. Compute the cart total (with any coupon) in **cents**.
2. **Reserve** stock for every line item (inventory-service). If any item can't be
   filled, release what was reserved and fail — never oversell.
3. **Charge** the shopper (payments-service). On failure, release the reservation.
4. **Create** the order in `pending`, then emit events.

## Idempotency

Checkout requires an **Idempotency-Key** header. The storefront may retry (flaky
network, double-click), so the same key must return the *same* order rather than
creating a second one. This is tracked in ticket **ORD-405**.
