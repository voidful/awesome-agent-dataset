"""Canonical record schema + shared normalization primitives.

The canonical schema mirrors ``voidful/gemma4-agent-sft`` exactly so expansion
shards can be concatenated with the seed dataset:

    id            : str
    source        : str            (e.g. "xlam", "glaive", "swe_zero")
    source_subset : str            (config/split/provenance within the source)
    messages      : str (JSON)     list[{role, content, tool_calls?, tool_responses?}]
    tools         : str (JSON)     list[{type:"function", function:{name, description, parameters}}]
    tool_names    : list[str]
    quality       : str (JSON)     {tier, score, curated, ...}
    metadata      : str (JSON)     {subset, name_map, target_tools, parallel_tool_calls,
                                    domain, filter_reason, ...}

Message shape (OpenAI-style, CoT stripped, no foreign chat-template markers):
    {role: "system"|"user"|"assistant"|"tool", content: str|None,
     tool_calls?:    [{function: {name: str, arguments: dict}}],
     tool_responses?:[{name: str|None, response: str|dict}]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import xxhash

# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def _json_default(o: Any):
    """Make stray non-JSON-serializable values safe (Ellipsis, bytes, sets,
    numpy scalars, etc.) instead of crashing json.dumps."""
    if o is Ellipsis:
        return None
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", "replace")
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    item = getattr(o, "item", None)  # numpy / pandas scalars
    if callable(item):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def _denan(o: Any) -> Any:
    import math
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _denan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_denan(x) for x in o]
    return o


def jdumps(obj: Any) -> str:
    """Compact, deterministic JSON string (UTF-8 preserved). Never raises on odd
    value types, and never emits invalid JSON (bare NaN/Infinity -> null)."""
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                          sort_keys=False, default=_json_default, allow_nan=False)
    except ValueError:  # NaN/Infinity present
        return json.dumps(_denan(obj), ensure_ascii=False, separators=(",", ":"),
                          sort_keys=False, default=_json_default, allow_nan=False)


def jloads_maybe(value: Any) -> Any:
    """json.loads if value is a str that parses; otherwise return value as-is.

    Tolerates the single-quoted python-dict style some datasets emit by falling
    back to a literal-eval-ish repair when strict JSON fails.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        repaired = _repair_pseudo_json(s)
        if repaired is not None:
            return repaired
        return value


def _repair_pseudo_json(s: str) -> Any | None:
    """Best-effort parse of single-quoted / python-literal pseudo-JSON.

    Rejects results containing python Ellipsis (from a literal '...'), so the
    raw string is kept instead of an unserializable object.
    """
    import ast

    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return None if _has_ellipsis(val) else val


def _has_ellipsis(v: Any) -> bool:
    if v is Ellipsis:
        return True
    if isinstance(v, dict):
        return any(_has_ellipsis(x) for x in v.values()) or any(_has_ellipsis(k) for k in v)
    if isinstance(v, (list, tuple, set)):
        return any(_has_ellipsis(x) for x in v)
    return False


# --------------------------------------------------------------------------- #
# Chain-of-thought stripping (seed dataset removes CoT)
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_cot(text: str | None) -> str | None:
    if not text:
        return text
    return _THINK_RE.sub("", text).strip() or None


# --------------------------------------------------------------------------- #
# Foreign chat-template marker stripping
# --------------------------------------------------------------------------- #

_FOREIGN_MARKERS = [
    "<|im_start|>", "<|im_end|>", "<|im_system|>", "<|im_middle|>",
    "<|endoftext|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
    "[INST]", "[/INST]", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|system|>",
    "tool_declare",
]


def strip_markers(text: str | None) -> str | None:
    if not text:
        return text
    for m in _FOREIGN_MARKERS:
        text = text.replace(m, "")
    return text.strip() or None


# --------------------------------------------------------------------------- #
# Tool name normalization (valid identifier, dedup-stable)
# --------------------------------------------------------------------------- #

_NAME_CLEAN_RE = re.compile(r"[^0-9a-zA-Z_]+")


def normalize_tool_name(name: str) -> str:
    """Map an arbitrary tool name to a stable snake_case-ish identifier.

    Examples: "Market Trends API" -> "Market_Trends_API",
              "exa-search-web_search_exa" -> "exa_search_web_search_exa".
    """
    if not name:
        return "unknown_tool"
    cleaned = _NAME_CLEAN_RE.sub("_", name.strip()).strip("_")
    if not cleaned:
        return "unknown_tool"
    if cleaned[0].isdigit():
        cleaned = "fn_" + cleaned
    return cleaned


# --------------------------------------------------------------------------- #
# Parameter schema coercion -> JSON-schema object
# --------------------------------------------------------------------------- #

_PYTYPE_MAP = {
    "str": "string", "string": "string", "text": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "double": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "tuple": "array",
    "dict": "object", "object": "object",
    "any": "string", "none": "null", "null": "null",
}


def _coerce_type(t: Any) -> Any:
    if isinstance(t, list):
        return [_coerce_type(x) for x in t]
    if not isinstance(t, str):
        return t
    base = t.split(",")[0].strip().lower()  # "int, optional" -> "int"
    return _PYTYPE_MAP.get(base, base or "string")


def coerce_parameters(params: Any) -> dict:
    """Return a JSON-schema 'object' parameters dict.

    Accepts already-valid JSON-schema, or xLAM/hermes flat style
    ``{argname: {type, description, default?}}`` and wraps it.
    """
    if not isinstance(params, dict) or not params:
        return {"type": "object", "properties": {}}

    # Already JSON-schema-ish: has a `properties` key, or declares an object-like
    # top-level type (xLAM/ToolACE use "dict"; some use "object"). Detect this BEFORE
    # the flat branch — otherwise a {"type":"dict","properties":{...}} schema gets its
    # own "type"/"properties"/"required" keys treated as argument names (data corruption).
    declared = str(params.get("type", "")).lower()
    if "properties" in params or declared in ("object", "dict"):
        props = params.get("properties") or {}
        out_props = {}
        for k, v in props.items():
            if isinstance(v, dict):
                vv = dict(v)
                if "type" in vv:
                    vv["type"] = _coerce_type(vv["type"])
                if isinstance(vv.get("properties"), dict):  # nested object
                    vv["properties"] = coerce_parameters(vv)["properties"]
                out_props[k] = vv
            else:
                out_props[k] = {"type": "string"}
        schema = {"type": "object", "properties": out_props}
        if isinstance(params.get("required"), list):
            schema["required"] = params["required"]
        return schema

    # Flat style: {argname: {type, description, default}}.
    props: dict[str, Any] = {}
    required: list[str] = []
    for argname, spec in params.items():
        if isinstance(spec, dict):
            prop: dict[str, Any] = {}
            if "type" in spec:
                prop["type"] = _coerce_type(spec["type"])
            else:
                prop["type"] = "string"
            if "description" in spec and spec["description"] is not None:
                prop["description"] = str(spec["description"])
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            if "items" in spec:
                prop["items"] = spec["items"]
            default = spec.get("default", None)
            if default in (None, "", "None") or "optional" in str(spec.get("type", "")).lower():
                pass
            else:
                required.append(argname)
            props[argname] = prop
        else:
            props[argname] = {"type": "string"}
            required.append(argname)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def normalize_tool_def(tool: Any, name_map: dict[str, str]) -> dict | None:
    """Coerce one tool definition into canonical
    ``{type:"function", function:{name, description, parameters}}``.

    Records original->normalized name in ``name_map``.
    """
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    raw_name = fn.get("name") or tool.get("name")
    if not raw_name:
        return None
    norm = normalize_tool_name(raw_name)
    if norm != raw_name:
        name_map[raw_name] = norm
    desc = fn.get("description") or tool.get("description") or ""
    params = fn.get("parameters", fn.get("arguments", {}))
    return {
        "type": "function",
        "function": {
            "name": norm,
            "description": str(desc),
            "parameters": coerce_parameters(params),
        },
    }


# --------------------------------------------------------------------------- #
# Canonical record
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    id: str
    source: str
    source_subset: str
    messages: list[dict]
    tools: list[dict]
    tool_names: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Serialize to the on-disk row (JSON-string fields, matching the seed)."""
        return {
            "id": self.id,
            "source": self.source,
            "source_subset": self.source_subset,
            "messages": jdumps(self.messages),
            "tools": jdumps(self.tools),
            "tool_names": self.tool_names,
            "quality": jdumps(self.quality),
            "metadata": jdumps(self.metadata),
        }

    # -- content hashing for dedup -------------------------------------- #
    def content_hash(self) -> str:
        h = xxhash.xxh64()
        for m in self.messages:
            h.update((m.get("role") or "").encode())
            h.update(_norm_text(m.get("content")).encode())
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                h.update((fn.get("name") or "").encode())
                h.update(jdumps(_sorted(fn.get("arguments"))).encode())
            for tr in m.get("tool_responses") or []:
                h.update(_norm_text(tr.get("response")).encode())
        for t in sorted(self.tool_names):
            h.update(t.encode())
        return h.hexdigest()


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        v = jdumps(v)
    return re.sub(r"\s+", " ", v).strip().lower()


def _sorted(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sorted(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sorted(x) for x in obj]
    return obj


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

VALID_ROLES = {"system", "user", "assistant", "tool"}


def validate_record(rec: Record) -> str | None:
    """Return None if the record is well-formed, else a short reason string."""
    if not rec.messages:
        return "empty_messages"
    roles = [m.get("role") for m in rec.messages]
    if any(r not in VALID_ROLES for r in roles):
        return f"bad_role:{[r for r in roles if r not in VALID_ROLES][:1]}"
    if "assistant" not in roles:
        return "no_assistant_turn"
    # Every assistant tool_call name must resolve to a declared tool (if tools given).
    declared = set(rec.tool_names)
    has_call = False
    for m in rec.messages:
        for tc in m.get("tool_calls") or []:
            has_call = True
            name = tc.get("function", {}).get("name")
            args = tc.get("function", {}).get("arguments")
            if not isinstance(args, (dict, list)):
                return "tool_call_args_not_object"
            if declared and name not in declared:
                return f"undeclared_tool:{name}"
    # Content presence: at least one non-empty assistant content or tool call.
    if not has_call and not any(
        m.get("role") == "assistant" and (m.get("content") or "").strip()
        for m in rec.messages
    ):
        return "empty_assistant"
    return None


def collect_tool_names(tools: Iterable[dict]) -> list[str]:
    names = []
    for t in tools:
        fn = t.get("function", {})
        n = fn.get("name")
        if n and n not in names:
            names.append(n)
    return names
