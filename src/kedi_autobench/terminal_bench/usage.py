from __future__ import annotations

from typing import Any

from kedi_autobench.terminal_bench.models import HarborTrialResult, KediTrialResult


def trial_usage(harbor: HarborTrialResult, kedi: KediTrialResult | None) -> dict[str, Any]:
    usage = dict(kedi.usage) if kedi is not None else {}
    agent = harbor.agent_result
    if agent is not None:
        for name, value in (
            ("input_tokens", agent.n_input_tokens),
            ("cache_read_tokens", agent.n_cache_tokens),
            ("output_tokens", agent.n_output_tokens),
            ("cost_usd", agent.cost_usd),
        ):
            if usage.get(name) is None:
                usage[name] = value
    input_tokens = _optional_int(usage.get("input_tokens")) or 0
    output_tokens = _optional_int(usage.get("output_tokens")) or 0
    usage.setdefault("total_tokens", input_tokens + output_tokens)
    return usage


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = ("trial_usage",)
