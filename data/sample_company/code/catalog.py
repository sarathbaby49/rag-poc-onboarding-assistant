"""Catalog service for the Acme storefront.

Owns products and categories. Prices are stored in integer minor units (cents),
never floats. Read-heavy, so hot product lookups are cached in Redis for 60s.
"""

from dataclasses import dataclass


@dataclass
class Product:
    sku: str                # stable stock-keeping unit, e.g. "TSHIRT-BLK-M"
    title: str
    price_cents: int        # integer cents — never a float
    category: str
    stock: int


# In real life this comes from PostgreSQL (catalog schema); inlined for the demo.
_CATALOG = {
    "TSHIRT-BLK-M": Product("TSHIRT-BLK-M", "Black T-Shirt (M)", 1999, "apparel", 120),
    "MUG-ACME-01": Product("MUG-ACME-01", "Acme Coffee Mug", 1299, "home", 340),
    "BOOK-RAG-01": Product("BOOK-RAG-01", "RAG in Practice", 3499, "books", 0),
}


def get_product(sku: str) -> Product | None:
    """Look up a single product by SKU. Returns None if it doesn't exist."""
    return _CATALOG.get(sku)


def search_products(query: str, category: str | None = None) -> list[Product]:
    """Naive substring search over titles, optionally filtered by category.

    Production uses OpenSearch with faceted filters (see CAT-310); this is the
    fallback used in local dev and tests.
    """
    q = query.lower()
    results = [p for p in _CATALOG.values() if q in p.title.lower()]
    if category:
        results = [p for p in results if p.category == category]
    return results


def is_in_stock(sku: str) -> bool:
    """A product is buyable only when catalog stock is above zero."""
    product = get_product(sku)
    return bool(product and product.stock > 0)
