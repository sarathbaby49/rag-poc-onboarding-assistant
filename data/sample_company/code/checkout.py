"""Checkout orchestration for the Acme storefront.

Checkout is the saga that ties the services together:

    cart total -> reserve inventory -> take payment -> create order -> emit events

If payment fails, the inventory reservation is released so stock isn't lost. The
whole flow is keyed by an Idempotency-Key so a retried checkout is safe.
"""

from cart import Cart, apply_coupon
from inventory import OutOfStock, release, reserve
from orders import create_order


def checkout(cart: Cart, stock: dict[str, int], idempotency_key: str,
             coupon: str = "", pay=lambda cents: True) -> dict:
    """Run the checkout saga and return the created order.

    `pay` is injected so tests can simulate success/failure without calling the
    real payments-service. It receives the amount in cents and returns a bool.
    """
    total = apply_coupon(cart.subtotal_cents(), coupon)

    # 1. reserve every line item (raises OutOfStock if any item can't be filled)
    reserved: list[tuple[str, int]] = []
    try:
        for sku, qty in cart.items.items():
            reserve(stock, sku, qty)
            reserved.append((sku, qty))
    except OutOfStock:
        for sku, qty in reserved:
            release(stock, sku, qty)
        raise

    # 2. take payment; on failure, release the reservation and abort
    if not pay(total):
        for sku, qty in reserved:
            release(stock, sku, qty)
        raise RuntimeError("payment declined")

    # 3. create the order (idempotent on idempotency_key)
    return create_order(cart.items, total, idempotency_key)
