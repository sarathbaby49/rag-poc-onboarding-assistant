"""Layer 6 plumbing for the Production Lab and the checks (done for you).

  - gateway_reachable():        a 5-second probe, so a live demo never hangs on a dead proxy
  - use_reference_solutions():  swap the src/ exercise functions for solutions/ ones
                                (the presenter's "show the finished version" switch)
  - exercise_status():          which Layer 6 exercises are implemented in src/
"""

from __future__ import annotations

import importlib
import inspect
import urllib.error
import urllib.request

from src import config

# module in src/  ->  (solution module, names to swap in)
EXERCISES = {
    "src.eval.run_eval": ("solutions.eval",
                          ["grade_keywords", "retrieval_hit", "judge_faithfulness", "compare_runs"]),
    "src.observability": ("solutions.observability",
                          ["PII_PATTERNS", "redact", "traced_complete", "summarize"]),
}

EXERCISE_IDS = {
    "grade_keywords": "EV1", "retrieval_hit": "EV2", "judge_faithfulness": "EV3", "compare_runs": "EV4",
    "redact": "OB1", "traced_complete": "OB2", "summarize": "OB3",
}


def gateway_reachable(timeout: float = 5.0) -> tuple[bool, str]:
    """(ok, message). Any HTTP answer from the proxy counts as reachable."""
    if not config.LLM_MODEL.startswith("litellm_proxy/"):
        return True, f"using {config.LLM_MODEL} directly"
    if not config.LITELLM_PROXY_API_KEY or not config.LITELLM_PROXY_API_BASE:
        return False, "LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY not set in .env"
    url = config.LITELLM_PROXY_API_BASE.rstrip("/") + "/health/liveliness"
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, f"gateway reachable ({config.LITELLM_PROXY_API_BASE})"
    except urllib.error.HTTPError:
        return True, f"gateway reachable ({config.LITELLM_PROXY_API_BASE})"
    except Exception as exc:  # noqa: BLE001
        return False, (f"can't reach {config.LITELLM_PROXY_API_BASE} ({type(exc).__name__}) — "
                       "on VPN? check with `python -m checks.check_llm`")


def use_reference_solutions() -> None:
    """Point every Layer 6 exercise in src/ at its reference solution."""
    for module_name, (solution_name, names) in EXERCISES.items():
        module = importlib.import_module(module_name)
        solution = importlib.import_module(solution_name)
        for name in names:
            setattr(module, name, getattr(solution, name))


def _is_stub(fn) -> bool:
    try:
        return "raise NotImplementedError" in inspect.getsource(fn)
    except (OSError, TypeError):
        return False


def exercise_status() -> list[tuple[str, str, bool]]:
    """[(exercise id, function name, implemented?)] for the functions currently in src/."""
    rows = []
    for module_name, (_, names) in EXERCISES.items():
        module = importlib.import_module(module_name)
        for name in names:
            if name in EXERCISE_IDS:
                rows.append((EXERCISE_IDS[name], name, not _is_stub(getattr(module, name))))
    return sorted(rows)
