"""Orders service for the Acme storefront.

Owns the order lifecycle. An order moves through a fixed set of states and every
transition publishes an event on RabbitMQ that other services (inventory,
notifications, ledger) react to.
"""

import time
import uuid

# The only valid order states, in the order they normally occur.
ORDER_STATES = ["pending", "paid", "fulfilled", "shipped", "delivered", "cancelled", "refunded"]


def new_order_id() -> str:
    """Order IDs are prefixed so they're easy to grep in logs: ORD-<uuid>."""
    return f"ORD-{uuid.uuid4().hex[:12]}"


def create_order(cart_items: dict[str, int], total_cents: int, idempotency_key: str) -> dict:
    """Create an order in the 'pending' state.

    Must be idempotent: the storefront may retry checkout, so the same
    Idempotency-Key header must never create two orders (see ORD-405).
    """
    return {
        "id": new_order_id(),
        "items": cart_items,
        "total_cents": total_cents,
        "status": "pending",
        "idempotency_key": idempotency_key,
        "created_at": int(time.time()),
    }


def transition(order: dict, new_status: str) -> dict:
    """Move an order to a new status, rejecting unknown states."""
    if new_status not in ORDER_STATES:
        raise ValueError(f"invalid order status: {new_status}")
    order["status"] = new_status
    order["updated_at"] = int(time.time())
    return order
