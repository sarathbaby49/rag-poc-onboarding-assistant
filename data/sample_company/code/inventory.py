"""Inventory service for the Acme storefront.

Tracks stock per SKU and prevents overselling. Checkout *reserves* stock before
payment; a successful payment *commits* the reservation, a failure *releases* it.
Reservations expire after 15 minutes so an abandoned checkout frees the stock.
"""

RESERVATION_TTL_SECONDS = 900


class OutOfStock(Exception):
    """Raised when a reservation asks for more units than are available."""


def reserve(stock: dict[str, int], sku: str, qty: int) -> None:
    """Reserve qty units of sku, decrementing available stock.

    Raises OutOfStock rather than letting the balance go negative — this is the
    guard that prevents the oversell bug tracked in INV-150.
    """
    available = stock.get(sku, 0)
    if qty > available:
        raise OutOfStock(f"{sku}: requested {qty}, only {available} left")
    stock[sku] = available - qty


def release(stock: dict[str, int], sku: str, qty: int) -> None:
    """Return reserved units to available stock (payment failed or timed out)."""
    stock[sku] = stock.get(sku, 0) + qty
