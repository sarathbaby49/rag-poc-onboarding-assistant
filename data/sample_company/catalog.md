# Catalog & Search — Acme

The **catalog-service** owns products and categories; the **search-service**
powers the storefront search box.

## Products

- Every product has a stable **SKU** (e.g. `TSHIRT-BLK-M`) that never changes.
- Prices are stored as **integer cents** (`price_cents`), never floats.
- Categories are flat for now (`apparel`, `home`, `books`, …); nested categories
  are on the roadmap.
- Read path is cached in **Redis** for 60 seconds — the catalog is read far more
  than it's written.

See `code/catalog.py` for `get_product`, `search_products`, and `is_in_stock`.

## Search

- Production search runs on **OpenSearch** with **faceted filters** (category,
  price range, in-stock). The local dev fallback is a simple substring match in
  `search_products`.
- Relevance is tuned with synonyms (e.g. "tee" → "t-shirt"). Add new synonyms in
  the search-service config, not in the catalog.
- Out-of-stock items still appear in search but are marked unbuyable; buyability
  comes from `is_in_stock`.

## Common gotchas

- Don't compute a price by multiplying floats — use integer cents and divide only
  for display.
- A product with `stock = 0` is valid (e.g. a pre-order); it just isn't buyable.
