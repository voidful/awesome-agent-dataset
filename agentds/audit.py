"""Data-quality auditor — quantifies defect rates over produced shards.

A standalone quality gate (run `agentds audit`). Loads the parquet shards and
checks every row against the canonical contract, reporting per-defect counts and
percentages so regressions are caught before a push.
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_MARKERS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|start_header_id|>",
            "[INST]", "[/INST]", "<functioncall>", "tool_declare"]
_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _iter_rows(shard_dir: Path):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(str(shard_dir / "*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            yield r


def audit(out_dir: str | Path = "data/expansion") -> dict:
    shard_dir = Path(out_dir) / "data"
    defects: Counter = Counter()
    by_source_defects: dict = defaultdict(Counter)
    by_source_total: Counter = Counter()
    ids: set[str] = set()
    id_collisions = 0
    total = 0
    big_rows = 0
    raw_bytes = 0

    for r in _iter_rows(shard_dir):
        total += 1
        src = r.get("source", "?")
        by_source_total[src] += 1
        rid = r.get("id")
        if rid in ids:
            id_collisions += 1
        ids.add(rid)

        def flag(name):
            defects[name] += 1
            by_source_defects[src][name] += 1

        mstr = r.get("messages") or ""
        raw_bytes += len(mstr)
        if len(mstr) > 1_000_000:
            big_rows += 1
            flag("oversized_row_gt_1MB")
        try:
            msgs = json.loads(mstr)
        except Exception:
            flag("messages_parse_error")
            continue
        try:
            tools = json.loads(r.get("tools") or "[]")
        except Exception:
            flag("tools_parse_error")
            tools = []
        tool_names = set(r.get("tool_names") or [])

        # tools schema sanity
        for t in tools:
            fn = (t or {}).get("function", {})
            params = fn.get("parameters", {})
            if not fn.get("name"):
                flag("tool_missing_name")
            if params and ("type" not in params or "properties" not in params):
                flag("tool_params_not_jsonschema")
            # schema-key-as-arg corruption: the coerce bug turned a dict-schema's own
            # {type,properties,required} keys into args, so all three appear together.
            # (A single legit arg named "type"/"properties" is NOT corruption.)
            props = (params or {}).get("properties", {})
            if isinstance(props, dict) and {"type", "properties", "required"} <= set(props):
                flag("schema_keys_leaked_as_args")

        roles = [m.get("role") for m in msgs]
        if any(x not in _VALID_ROLES for x in roles):
            flag("invalid_role")
        if "assistant" not in roles:
            flag("no_assistant_turn")

        n_calls = n_resp = 0
        assistant_text_chars = 0
        for m in msgs:
            content = m.get("content")
            if isinstance(content, str):
                if "<think>" in content:
                    flag("cot_leak")
                if any(mk in content for mk in _MARKERS):
                    flag("foreign_marker_leak")
                if content.lower().lstrip().startswith(("assistant\n", "assistant ")):
                    flag("role_label_leak")
                if m.get("role") == "assistant":
                    assistant_text_chars += len(content.strip())
                if m.get("role") == "assistant" and "<function=" in content:
                    flag("raw_function_xml")
            for tc in m.get("tool_calls") or []:
                n_calls += 1
                fn = tc.get("function", {})
                if not isinstance(fn.get("arguments"), (dict, list)):
                    flag("args_not_object")
                if tool_names and fn.get("name") not in tool_names:
                    flag("undeclared_tool_call")
            for tr in m.get("tool_responses") or []:
                n_resp += 1
                resp = tr.get("response")
                if isinstance(resp, str) and resp.startswith("OBSERVATION:"):
                    flag("observation_prefix_leak")

        if n_calls and not assistant_text_chars and not n_calls:
            flag("empty_assistant")
        if n_calls and n_resp and (n_calls - n_resp) > 1:
            flag("call_response_imbalance_gt1")

    result = {
        "total_rows": total,
        "id_collisions": id_collisions,
        "oversized_rows_gt_1MB": big_rows,
        "avg_messages_bytes": round(raw_bytes / max(total, 1)),
        "defects": dict(defects.most_common()),
        "defect_rate_pct": {k: round(100 * v / max(total, 1), 3) for k, v in defects.most_common()},
        "worst_sources": {
            s: dict(by_source_defects[s].most_common(3))
            for s in sorted(by_source_defects, key=lambda x: -sum(by_source_defects[x].values()))[:6]
        },
    }
    return result


def print_report(out_dir: str | Path = "data/expansion") -> dict:
    res = audit(out_dir)
    print(f"AUDIT — {res['total_rows']:,} rows | id_collisions={res['id_collisions']} | "
          f"oversized(>1MB)={res['oversized_rows_gt_1MB']} | avg_msg_bytes={res['avg_messages_bytes']:,}")
    if not res["defects"]:
        print("  ✅ no defects detected")
    for name, n in res["defects"].items():
        print(f"  {name:32} {n:>7,}  ({res['defect_rate_pct'][name]}%)")
    return res
