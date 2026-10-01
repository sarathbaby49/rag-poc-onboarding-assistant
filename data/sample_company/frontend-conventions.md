# Frontend Conventions — Acme Shop

The storefront is a **React 18** SPA (Vite + TypeScript). These conventions
keep the codebase consistent across the team.

## Structure
- `src/pages/` — route-level components, one file per route.
- `src/components/` — shared UI components (Button, Modal, ProductCard …).
- `src/hooks/` — custom hooks (`useCart`, `useAuth`, `useProducts`).
- `src/api/` — typed API client generated from the OpenAPI spec.

## Style guide
- **Functional components** only; no class components.
- State management via **React Context + useReducer** (no Redux).
- CSS Modules for scoped styles; design tokens in `tokens.css`.
- All user-facing strings go through `i18n.t()` for localisation.

## Testing
- **Vitest** for unit tests; **Playwright** for E2E.
- Minimum coverage target: 80 % lines on `src/hooks/` and `src/api/`.

## PR checklist
1. `npm run lint` passes.
2. `npm run test` passes.
3. No `any` types except in generated API client.
