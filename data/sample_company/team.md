# Teams & Ownership — Acme

Who owns what, and where to ask. Use the Slack channel, not DMs.

| Area | Service(s) | Team | Channel |
| --- | --- | --- | --- |
| Storefront & search | catalog-service, search-service | Discovery | #team-discovery |
| Cart & checkout | cart-service, orders-service, checkout | Purchase | #team-purchase |
| Payments & ledger | payments-service, ledger-service | Payments | #team-payments |
| Stock & warehouse | inventory-service | Fulfilment | #team-fulfilment |
| Infra, CI/CD, access | platform, gateway | Platform | #platform-requests |
| Sales analytics & reporting | data warehouse, dashboards | Data & Analytics | #team-data |

## Notes

- **Deploy access / prod credentials** → Platform team, via #platform-requests.
- **Sales numbers / dashboards** → Data & Analytics (#team-data); reports live in
  `sales/`.
- Every service has an on-call rota; page through #incidents for customer-facing
  breakage.
- Your onboarding buddy is assigned on day one and can route you to the right team.
