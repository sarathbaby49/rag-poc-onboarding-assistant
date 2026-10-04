# 💰 Layer 5 — Model Selection, Customization & Cost

Which model runs each step, whether to change the model (prompting vs RAG vs
fine-tuning), and how to pay less for it (token economics, routing, prompt
caching). Hands-on, you build a **cost meter**, a **router**, **per-step model
choice** for the Layer 4 plan graph, and a **cache-friendly prompt**, then watch
your bill in the **Cost Lab**.

## Prerequisites

1. **LLM gateway configured** with access to a cheap, a mid and a strong model:
   ```bash
   python -m checks.check_llm          # must pass
   ```
   Add `CHEAP_MODEL`, `MID_MODEL`, `STRONG_MODEL` to `.env` (see `.env.example`).
   List the names your proxy uses:
   ```bash
   curl $LITELLM_PROXY_API_BASE/v1/models -H "Authorization: Bearer $LITELLM_PROXY_API_KEY"
   ```
2. **Vector index built:** `python -m src.ingest`
3. **Layer 4 is NOT required** — the Cost Lab uses the reference graph.

## Session

- Exercises: [EXERCISES.md](EXERCISES.md) (C1–C4, ~40 min)
- Lab: `streamlit run cost_lab.py`

## Files

| File | Status | What |
|------|--------|------|
| `src/cost.py` | 📝 C1 | price table (done) + `cost_of` |
| `src/models.py` | 📝 C2, C3 | `pick_model`, `model_for_step` |
| `src/caching.py` | 📝 C4 | `build_messages` (cache-hostile until you fix it) |
| `src/cost_helper.py` | ✅ done | gateway calls with usage, usage normalization, per-step graph routing |
| `cost_lab.py` | ✅ done | Streamlit lab: Compare · Router · Plan graph · Caching + session bill |
| `checks/check_cost.py`, `check_routing.py`, `check_caching.py` | ✅ | self-checks |
| `solutions/cost.py`, `models.py`, `caching.py` | ✅ | reference answers |

Prices in `src/cost.py` were checked on 4 Oct 2026 against the Anthropic, OpenAI
and Google pricing pages. Re-check before quoting them; they change often.
