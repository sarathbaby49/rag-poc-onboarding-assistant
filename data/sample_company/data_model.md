# Data Model — Acme

Each service owns its own PostgreSQL schema. Services never read each other's
tables — they call the owning service or consume its events. All money columns are
**integer cents**; every table has `created_at` / `updated_at`.

## catalog.products
| column | type | notes |
| --- | --- | --- |
| sku | text (PK) | stable, human-readable, never changes |
| title | text | |
| price_cents | integer | never a float |
| category | text | apparel / home / books / … |
| stock | integer | 0 is valid (pre-order) |

## orders.orders
| column | type | notes |
| --- | --- | --- |
| id | text (PK) | `ORD-<uuid>` |
| status | text | one of the order states (see orders.md) |
| total_cents | integer | order total after discounts |
| idempotency_key | text (unique) | prevents duplicate orders |

## orders.order_items
| column | type | notes |
| --- | --- | --- |
| order_id | text (FK) | |
| sku | text | snapshot of the SKU bought |
| qty | integer | |
| unit_price_cents | integer | price at time of purchase (not live price) |

## inventory.stock
| column | type | notes |
| --- | --- | --- |
| sku | text (PK) | |
| available | integer | never goes negative — the oversell guard |
| reserved | integer | held during checkout, released on timeout |

## payments.payment_intents
| column | type | notes |
| --- | --- | --- |
| id | text (PK) | |
| order_id | text | |
| amount_cents | integer | |
| status | text | pending / succeeded / failed / refunded |
