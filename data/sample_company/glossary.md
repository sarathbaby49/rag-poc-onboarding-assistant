# Glossary — Acme domain terms

- **SKU** — Stock-Keeping Unit. A stable product code (e.g. `TSHIRT-BLK-M`).
- **Cart** — a shopper's in-progress list of items; lives in Redis, 7-day TTL.
- **Checkout** — the saga that turns a cart into a paid order.
- **Order** — a confirmed purchase; moves through pending → paid → … → delivered.
- **Idempotency-Key** — header that makes a retried request safe (no duplicates).
- **Reservation** — stock held during checkout so two shoppers can't buy the last one.
- **Oversell** — selling more units than are in stock; the bug INV-150 prevents.
- **Fulfilment** — picking, packing and shipping an order from the warehouse.
- **GMV** — Gross Merchandise Value: total value of orders placed (before refunds).
- **AOV** — Average Order Value: GMV ÷ number of orders.
- **Conversion rate** — share of sessions that result in a purchase.
- **Cart abandonment** — carts created but never checked out.
- **Refund rate** — share of order value returned to shoppers.
- **Minor units / cents** — how we store money: integers, never floats.
