"""Cart service for the Acme storefront.

A cart is a short-lived list of line items keyed by SKU. Totals are computed in
integer cents. Carts live in Redis (TTL 7 days); this module holds the pure
logic so it's easy to unit-test.
"""

from dataclasses import dataclass, field

from catalog import get_product


@dataclass
class Cart:
    items: dict[str, int] = field(default_factory=dict)  # sku -> quantity

    def add_item(self, sku: str, qty: int = 1) -> None:
        """Add qty of a SKU. Raises ValueError for an unknown product."""
        if get_product(sku) is None:
            raise ValueError(f"unknown SKU: {sku}")
        self.items[sku] = self.items.get(sku, 0) + qty

    def remove_item(self, sku: str) -> None:
        self.items.pop(sku, None)

    def subtotal_cents(self) -> int:
        """Sum of price * qty across line items, in integer cents."""
        total = 0
        for sku, qty in self.items.items():
            product = get_product(sku)
            if product:
                total += product.price_cents * qty
        return total


def apply_coupon(subtotal_cents: int, code: str) -> int:
    """Return the discounted total in cents. Unknown codes are a no-op.

    Percentages are applied with integer math to avoid rounding drift (see PAY-063
    and the 'money is always integer cents' rule in architecture.md).
    """
    percent = {"WELCOME10": 10, "SUMMER20": 20}.get(code.upper(), 0)
    discount = subtotal_cents * percent // 100
    return subtotal_cents - discount
