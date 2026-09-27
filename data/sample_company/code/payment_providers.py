"""Payment provider registry for Acme Shop.

All providers implement the same PaymentProvider interface, so checkout stays
provider-agnostic. Add a provider by subclassing PaymentProvider and adding it to
PROVIDERS. Amounts are always integer minor units (cents / paise), never floats.
"""

from __future__ import annotations


class PaymentProvider:
    """Common interface every provider implements."""

    name: str = "base"

    def charge(self, amount_cents: int, currency: str, token: str) -> dict:
        raise NotImplementedError

    def refund(self, charge_id: str, amount_cents: int) -> dict:
        raise NotImplementedError

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        raise NotImplementedError


class StripeProvider(PaymentProvider):
    """Primary provider: cards, 3-D Secure, Apple/Google Pay, subscriptions."""

    name = "stripe"


class PayPalProvider(PaymentProvider):
    """Wallet / Express Checkout."""

    name = "paypal"


class RazorpayProvider(PaymentProvider):
    """India: UPI, netbanking, local cards."""

    name = "razorpay"


# The payment providers Acme Shop uses today. Stripe is the default.
PROVIDERS: dict[str, type[PaymentProvider]] = {
    "stripe": StripeProvider,
    "paypal": PayPalProvider,
    "razorpay": RazorpayProvider,
}
DEFAULT_PROVIDER = "stripe"


def get_provider(name: str | None = None) -> PaymentProvider:
    """Return a provider instance by name (falls back to the default: Stripe)."""
    return PROVIDERS[name or DEFAULT_PROVIDER]()


def provider_for_currency(currency: str) -> PaymentProvider:
    """Route by currency: INR -> Razorpay, everything else -> Stripe."""
    return get_provider("razorpay" if currency.upper() == "INR" else "stripe")
