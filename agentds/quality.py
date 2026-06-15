"""Quality scoring & stratification.

Produces the canonical ``quality`` field: {tier, score, curated, signals}.
Tiers: high / medium / low. The heuristic rewards executable, schema-valid,
multi-turn tool use and penalizes degenerate or malformed trajectories.

Source-provided success/curation signals (SWE `resolved`, CoderForge `reward`,
Toucan quality assessments) are folded in when present via `ext_signals`.
"""
from __future__ import annotations

from .schema import Record


def _json_schema_valid_call(rec: Record) -> tuple[int, int]:
    """(#calls, #calls whose args validate against the declared tool schema)."""
    schema_by_name = {}
    for t in rec.tools:
        fn = t.get("function", {})
        schema_by_name[fn.get("name")] = fn.get("parameters", {})
    total = ok = 0
    for m in rec.messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += 1
            params = schema_by_name.get(fn.get("name"))
            args = fn.get("arguments")
            if params is None:
                continue
            if _args_match(args, params):
                ok += 1
    return total, ok


def _num(v) -> float:
    """Best-effort numeric coercion; non-numeric -> 0.0 (never raises)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _args_match(args, params) -> bool:
    if not isinstance(args, dict):
        return False
    required = params.get("required") or []
    props = params.get("properties") or {}
    if any(r not in args for r in required):
        return False
    # No hallucinated args (allow extras only if schema has no properties listed).
    if props and any(k not in props for k in args):
        return False
    return True


def score(rec: Record, ext_signals: dict | None = None) -> dict:
    ext_signals = ext_signals or {}
    signals: dict = {}
    s = 0.5

    roles = [m.get("role") for m in rec.messages]
    n_turns = len(rec.messages)
    n_assistant = roles.count("assistant")
    n_tool = roles.count("tool")
    has_tools = bool(rec.tool_names)
    n_calls, n_valid = _json_schema_valid_call(rec)

    signals["n_turns"] = n_turns
    signals["n_tool_calls"] = n_calls
    signals["multi_turn"] = n_assistant > 1

    # Multi-turn tool use with observations: the most valuable shape.
    if n_calls and n_tool:
        s += 0.15
    if n_assistant > 1:
        s += 0.05

    # Schema-grounded calls.
    if n_calls:
        valid_ratio = n_valid / n_calls
        signals["valid_arg_ratio"] = round(valid_ratio, 3)
        s += 0.15 * valid_ratio - 0.10 * (1 - valid_ratio)

    # Declared-tool coverage: calls that reference a declared tool.
    if has_tools and n_calls:
        s += 0.05

    # External success / curation signals.
    if ext_signals.get("resolved") is True or _num(ext_signals.get("reward")) >= 1:
        s += 0.15
        signals["verified_success"] = True
    if ext_signals.get("resolved") is False:
        s -= 0.05
    if ext_signals.get("curated"):
        s += 0.1

    # "When-not-to-call" appropriate refusals are valuable but unverifiable.
    if rec.metadata.get("no_call_expected"):
        signals["no_call_expected"] = True
        s += 0.03

    # Penalties for degenerate content.
    total_chars = sum(len(m.get("content") or "") for m in rec.messages if isinstance(m.get("content"), str))
    if total_chars < 40 and not n_calls:
        s -= 0.2
        signals["too_short"] = True
    if n_turns < 2:
        s -= 0.1

    s = max(0.0, min(1.0, s))
    tier = "high" if s >= 0.75 else ("medium" if s >= 0.5 else "low")
    curated = bool(ext_signals.get("curated") or signals.get("verified_success"))
    return {"tier": tier, "score": round(s, 3), "curated": curated, "signals": signals}
