# Payments & Providers — Acme Shop

Acme Shop supports several **payment providers** behind a single
`PaymentProvider` interface (see `code/payment_providers.py`), so checkout code
doesn't care which one runs.

## Providers we use
- **Stripe** — our **primary** provider. Credit/debit cards, 3-D Secure,
  Apple Pay and Google Pay, and subscription billing. Default for US & EU.
- **PayPal** — wallet / Express Checkout, for customers who prefer PayPal.
- **Razorpay** — used in **India**: UPI, netbanking and local cards.

Apple Pay and Google Pay are handled *through Stripe*, not as separate providers.

## How it's wired
- Each provider implements `charge()`, `refund()` and `verify_webhook()`.
- The provider is chosen per order by currency/region — e.g. **INR → Razorpay**,
  **USD/EUR → Stripe**.
- Each provider posts events to `/webhooks/<provider>` (e.g. `/webhooks/stripe`).
- All amounts are stored in integer minor units (cents / paise), never floats.

## Configuration (environment variables)
| Variable | Provider |
| --- | --- |
| `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET` | Stripe |
| `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET` | PayPal |
| `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET` | Razorpay |

## Adding a provider
Implement the `PaymentProvider` interface and register it in the `PROVIDERS` dict
in `code/payment_providers.py`. No checkout code needs to change.
