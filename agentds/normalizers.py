"""Format-family normalizers: raw HF row -> canonical Record(s).

Each normalizer has signature ``fn(row, cfg, ctx) -> Iterator[Record]`` where:
  row : the raw dataset row (dict)
  cfg : per-source column hints from the registry (dict)
  ctx : NormCtx(source, source_subset)

Normalizers set messages/tools/tool_names/metadata and source/source_subset.
The pipeline assigns the final ``id`` (content hash) and ``quality`` tier.

Dispatch is via the NORMALIZERS table at the bottom.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

from .schema import (
    Record,
    coerce_parameters,
    collect_tool_names,
    jloads_maybe,
    normalize_tool_def,
    normalize_tool_name,
    strip_cot,
    strip_markers,
)


@dataclass
class NormCtx:
    source: str
    source_subset: str


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #

_MAX_CONTENT_CHARS = 200_000  # cap pathological dumps (e.g. Excel data in pi sessions)


def _cap(text: str | None) -> str | None:
    if text and len(text) > _MAX_CONTENT_CHARS:
        return text[:_MAX_CONTENT_CHARS] + "\n…[content truncated]"
    return text


def _clean(text: Any) -> str | None:
    if text is None:
        return None
    if not isinstance(text, str):
        return text
    return _cap(strip_markers(strip_cot(text)))


def args_to_obj(raw: Any) -> dict:
    """Coerce a tool-call 'arguments' payload to an object."""
    val = jloads_maybe(raw)
    if isinstance(val, dict):
        return val
    if isinstance(val, list):
        return {"_args": val}
    if val is None:
        return {}
    return {"value": val}


def mk_msg(role: str, content: Any = None, tool_calls=None, tool_responses=None) -> dict:
    m: dict[str, Any] = {"role": role}
    m["content"] = _clean(content) if isinstance(content, str) else content
    if tool_calls:
        m["tool_calls"] = tool_calls
    if tool_responses:
        for tr in tool_responses:
            if isinstance(tr.get("response"), str):
                tr["response"] = _cap(tr["response"])
        m["tool_responses"] = tool_responses
    return m


def mk_call(name: str, arguments: Any) -> dict:
    return {"function": {"name": normalize_tool_name(name), "arguments": args_to_obj(arguments)}}


def clean_tool_response(content: Any) -> Any:
    """SWE-smith etc. wrap observations as a stringified list of {type,text}.
    Flatten to plain text and drop the leading 'OBSERVATION:' marker."""
    val = jloads_maybe(content)
    if isinstance(val, list) and val and isinstance(val[0], dict) and ("text" in val[0]):
        text = "\n".join(str(d.get("text", "")) for d in val if isinstance(d, dict))
    elif isinstance(val, dict) and "text" in val:
        text = str(val["text"])
    elif isinstance(content, str):
        text = content  # plain-string observation (e.g. swe_gym "OBSERVATION:\n…")
    else:
        return content
    return re.sub(r"^OBSERVATION:\s*", "", text).strip()


def synthesize_tools_from_calls(msgs: list[dict]) -> list[dict]:
    """Build minimal tool defs from observed tool_calls when a source declares
    no tools column (e.g. SWE-Zero). Better than an empty namespace for training."""
    seen: dict[str, dict] = {}
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            # Union args across ALL calls of this tool, not just the first — otherwise
            # later calls with more args get scored as schema-invalid downstream.
            props = seen.setdefault(name, {"type": "function", "function": {
                "name": name, "description": "",
                "parameters": {"type": "object", "properties": {}}}})["function"]["parameters"]["properties"]
            for k, v in (fn.get("arguments") or {}).items():
                if k not in props:
                    props[k] = {"type": _infer_type(v)}
    return list(seen.values())


def _infer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"


def tools_from_list(raw_tools: Any, name_map: dict) -> list[dict]:
    raw_tools = jloads_maybe(raw_tools)
    out = []
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        return out
    for t in raw_tools:
        td = normalize_tool_def(t, name_map)
        if td:
            out.append(td)
    return out


def tools_from_strlist(raw: Any, name_map: dict) -> list[dict]:
    """When2Call style: a list of JSON-string tool defs."""
    out = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        td = normalize_tool_def(jloads_maybe(item), name_map)
        if td:
            out.append(td)
    return out


# -- XML-tag (Hermes / smoltalk) parsing ------------------------------------ #

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_TOOLS_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)


def parse_xml_calls(text: str) -> tuple[str | None, list[dict]]:
    """Split assistant content into (prose_without_calls, [tool_calls])."""
    calls = []
    for blob in _TOOL_CALL_RE.findall(text or ""):
        obj = jloads_maybe(blob)
        if isinstance(obj, dict) and obj.get("name"):
            calls.append(mk_call(obj["name"], obj.get("arguments", {})))
    prose = _TOOL_CALL_RE.sub("", text or "").strip() or None
    return prose, calls


def parse_xml_responses(text: str) -> list[dict]:
    out = []
    for blob in _TOOL_RESP_RE.findall(text or ""):
        obj = jloads_maybe(blob)
        if isinstance(obj, dict):
            out.append({"name": normalize_tool_name(obj["name"]) if obj.get("name") else None,
                        "response": obj.get("content", obj.get("response", obj))})
        else:
            out.append({"name": None, "response": blob})
    return out


def parse_xml_tools_block(text: str, name_map: dict) -> list[dict]:
    m = _TOOLS_BLOCK_RE.search(text or "")
    if not m:
        return []
    return tools_from_list(m.group(1), name_map)


# -- SWE-agent function XML: <function=name><parameter=k>v</parameter></function> -- #

_FUNC_XML_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAM_XML_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def parse_function_xml_calls(text: str) -> tuple[str | None, list[dict]]:
    """Parse SWE-agent-style inline function calls; return (prose, [calls])."""
    calls = []
    for name, body in _FUNC_XML_RE.findall(text or ""):
        args = {k: v.strip() for k, v in _PARAM_XML_RE.findall(body)}
        calls.append(mk_call(name, args))
    prose = _FUNC_XML_RE.sub("", text or "").strip() or None
    return prose, calls


# -- BFCL python-call DSL (ToolACE): [Func(k=v, ...), ...] ------------------- #
# Tool names may themselves contain parens/spaces, e.g.
# "User Feed (Video Posts) V2(username=...)". So we split calls on depth-0 commas
# and take the ARGS as the final balanced (...) group, NAME as the prefix.


def _split_top_level(s: str) -> list[str]:
    """Split on commas that sit outside any parentheses/brackets."""
    parts, depth, buf = [], 0, []
    for c in s:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    if buf:
        parts.append("".join(buf))
    return parts


def _final_paren_group(seg: str) -> tuple[str, str] | None:
    """Return (name, args_inside) splitting on the LAST balanced (...) group."""
    end = seg.rfind(")")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        if seg[i] == ")":
            depth += 1
        elif seg[i] == "(":
            depth -= 1
            if depth == 0:
                return seg[:i].strip(), seg[i + 1:end]
    return None


def parse_bfcl_calls(text: str) -> list[dict]:
    s = (text or "").strip()
    if not (s.startswith("[") and s.endswith("]") and "(" in s):
        return []
    inner = s[1:-1]
    calls = []
    for seg in _split_top_level(inner):
        seg = seg.strip()
        if not seg:
            continue
        split = _final_paren_group(seg)
        if not split:
            return []  # not a call list (prose) -> bail
        name, argstr = split
        if not name or not re.match(r"^[A-Za-z_][\w .()-]*$", name):
            return []
        calls.append(mk_call(name, _parse_kv_args(argstr)))
    return calls


_KV_RE = re.compile(r"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\[[^\]]*\]|[^,]+)")


def _parse_kv_args(argstr: str) -> dict:
    import ast

    out = {}
    for k, v in _KV_RE.findall(argstr):
        v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v.strip("'\"")
    return out


# --------------------------------------------------------------------------- #
# Family normalizers
# --------------------------------------------------------------------------- #

_SHAREGPT_ROLES = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "model": "assistant", "bot": "assistant",
    "system": "system",
    "tool": "tool", "observation": "tool", "function": "tool", "tool_response": "tool",
    "function_call": "assistant",  # carries a tool call
    "tool_call": "assistant",
}


def norm_openai_messages(row, cfg, ctx) -> Iterator[Record]:
    """Already OpenAI-style: {role, content, tool_calls[].function.{name,arguments}}.

    arguments may be a JSON string (SWE/native) or object (ToolMind). Tools live
    in a top-level column (cfg['tools_col']) or per-message.
    """
    name_map: dict[str, str] = {}
    raw_msgs = jloads_maybe(row.get(cfg.get("messages_col", "messages")))
    if not isinstance(raw_msgs, list):
        return
    tools = tools_from_list(row.get(cfg.get("tools_col", "tools")), name_map) if cfg.get("tools_col") else []
    clean_resp = cfg.get("clean_tool_content")

    msgs: list[dict] = []
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "function":  # legacy tool result
            resp = m.get("content")
            msgs.append(mk_msg("tool", tool_responses=[{
                "name": normalize_tool_name(m["name"]) if m.get("name") else None,
                "response": clean_tool_response(resp) if clean_resp else resp,
            }]))
            continue
        if role not in ("system", "user", "assistant", "tool"):
            continue
        tool_calls = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if fn.get("name"):
                tool_calls.append(mk_call(fn["name"], fn.get("arguments", {})))
        if not tool_calls and isinstance(m.get("function_call"), dict) and m["function_call"].get("name"):
            tool_calls.append(mk_call(m["function_call"]["name"], m["function_call"].get("arguments", {})))
        if role == "tool":
            resp = m.get("content")
            msgs.append(mk_msg("tool", tool_responses=[{
                "name": normalize_tool_name(m["name"]) if m.get("name") else None,
                "response": clean_tool_response(resp) if clean_resp else resp,
            }]))
            continue
        content = m.get("content")
        # SWE-agent inline function XML (e.g. SWE-smith `tool` split): calls live as
        # <function=bash>…</function> text instead of structured tool_calls.
        if role == "assistant" and not tool_calls and isinstance(content, str) and "<function=" in content:
            prose, xml_calls = parse_function_xml_calls(content)
            if xml_calls:
                tool_calls, content = xml_calls, prose
        msgs.append(mk_msg(role, content, tool_calls or None))

    if not tools:
        tools = _harvest_tools_from_system(msgs, name_map)
    if not tools and cfg.get("synthesize_tools"):
        tools = synthesize_tools_from_calls(msgs)
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map} if name_map else {})


def norm_sharegpt(row, cfg, ctx) -> Iterator[Record]:
    """ShareGPT {from,value}. Handles function_call/observation roles whose value
    is a JSON string (glaive-sharegpt, APIGen-MT)."""
    name_map: dict[str, str] = {}
    conv = row.get(cfg.get("conv_col", "conversations"))
    if not isinstance(conv, list):
        return
    tools = tools_from_list(row.get(cfg.get("tools_col", "tools")), name_map) if cfg.get("tools_col") else []
    system_extra = row.get(cfg.get("system_col")) if cfg.get("system_col") else None

    msgs: list[dict] = []
    if system_extra:
        msgs.append(mk_msg("system", system_extra))
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        frm = turn.get("from") or turn.get("role")
        val = turn.get("value", turn.get("content"))
        role = _SHAREGPT_ROLES.get(frm)
        if role is None:
            continue
        if frm in ("function_call", "tool_call"):
            obj = jloads_maybe(val) or {}
            if isinstance(obj, list):
                calls = [mk_call(o.get("name"), o.get("arguments", {})) for o in obj if isinstance(o, dict) and o.get("name")]
            elif isinstance(obj, dict) and obj.get("name"):
                calls = [mk_call(obj["name"], obj.get("arguments", {}))]
            else:
                calls = []
            if calls:
                msgs.append(mk_msg("assistant", None, calls))
        elif role == "tool":
            msgs.append(mk_msg("tool", tool_responses=[{"name": None, "response": jloads_maybe(val)}]))
        else:
            msgs.append(mk_msg(role, val))

    if not tools and cfg.get("tools_in_system"):
        tools = _harvest_tools_from_system(msgs, name_map)
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map} if name_map else {})


def norm_xlam(row, cfg, ctx) -> Iterator[Record]:
    """xLAM/APIGen single-turn: query + answers(parallel calls) + tools."""
    name_map: dict[str, str] = {}
    tools = tools_from_list(row.get(cfg.get("tools_col", "tools")), name_map)
    answers = jloads_maybe(row.get(cfg.get("answers_col", "answers"))) or []
    calls = [mk_call(a["name"], a.get("arguments", {})) for a in answers if isinstance(a, dict) and a.get("name")]
    query = row.get(cfg.get("query_col", "query"))
    msgs = [mk_msg("user", query)]
    msgs.append(mk_msg("assistant", None, calls or None))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map, "parallel_tool_calls": len(calls) > 1} if name_map or calls else {})


def norm_hermes(row, cfg, ctx) -> Iterator[Record]:
    """Hermes ChatML in ShareGPT wrapper; XML-tag calls/responses; tools col."""
    name_map: dict[str, str] = {}
    conv = row.get(cfg.get("conv_col", "conversations"))
    if not isinstance(conv, list):
        return
    tools = tools_from_list(row.get(cfg.get("tools_col", "tools")), name_map) if row.get(cfg.get("tools_col", "tools")) else []

    msgs: list[dict] = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        frm = turn.get("from") or turn.get("role")
        val = turn.get("value", turn.get("content")) or ""
        role = _SHAREGPT_ROLES.get(frm)
        if role is None:
            continue
        if role == "system":
            if not tools:
                tools = parse_xml_tools_block(val, name_map)
            content = _TOOLS_BLOCK_RE.sub("", val).strip() or None
            msgs.append(mk_msg("system", content))
        elif role == "assistant":
            prose, calls = parse_xml_calls(val)
            msgs.append(mk_msg("assistant", prose, calls or None))
        elif role == "tool":
            resps = parse_xml_responses(val)
            msgs.append(mk_msg("tool", tool_responses=resps or [{"name": None, "response": val}]))
        else:
            msgs.append(mk_msg(role, val))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map} if name_map else {})


def norm_toolace(row, cfg, ctx) -> Iterator[Record]:
    """ToolACE: tools as JSON array buried in system text; calls in BFCL DSL."""
    name_map: dict[str, str] = {}
    conv = row.get("conversations")
    system = row.get("system") or ""
    if not isinstance(conv, list):
        return
    tools = _extract_json_array(system, name_map)

    msgs: list[dict] = []
    sys_clean = _strip_after_anchor(system)
    if sys_clean:
        msgs.append(mk_msg("system", sys_clean))
    for turn in conv:
        frm = turn.get("from")
        val = turn.get("value") or ""
        role = _SHAREGPT_ROLES.get(frm, frm)
        if role == "assistant":
            calls = parse_bfcl_calls(val)
            if calls:
                msgs.append(mk_msg("assistant", None, calls))
            else:
                msgs.append(mk_msg("assistant", val))
        elif role == "tool":
            msgs.append(mk_msg("tool", tool_responses=[{"name": None, "response": jloads_maybe(val)}]))
        elif role == "user":
            msgs.append(mk_msg("user", val))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map} if name_map else {})


def norm_when2call(row, cfg, ctx) -> Iterator[Record]:
    """When2Call SFT: tools=list[str]; messages=[{role,content}] where the
    'correct' answer is often a refusal/clarification (no call). A <TOOLCALL>[...]
    marker may be embedded in content."""
    name_map: dict[str, str] = {}
    tools = tools_from_strlist(row.get("tools"), name_map)
    raw_msgs = row.get("messages") or []
    msgs: list[dict] = []
    had_call = False
    for m in raw_msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant" and "<TOOLCALL>" in content:
            inner = content.split("<TOOLCALL>", 1)[1].split("</TOOLCALL>", 1)[0]
            arr = jloads_maybe(inner) or []
            calls = [mk_call(a["name"], a.get("arguments", {})) for a in arr if isinstance(a, dict) and a.get("name")]
            if calls:
                had_call = True
                msgs.append(mk_msg("assistant", None, calls))
                continue
        if role in ("system", "user", "assistant"):
            msgs.append(mk_msg(role, content))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map, "no_call_expected": not had_call})


def norm_toucan(row, cfg, ctx) -> Iterator[Record]:
    """Toucan: SFT config uses tool_call/tool_response roles; the *-K2/OSS/Qwen3
    configs use legacy function_call(args=string)/role:function."""
    name_map: dict[str, str] = {}
    msgs_raw = jloads_maybe(row.get("messages")) or []
    tools = tools_from_list(row.get("tools") or row.get("available_tools"), name_map)

    msgs: list[dict] = []
    pending_call = None
    for m in msgs_raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if not tools:
                tools = _harvest_tools_from_text(content or "", name_map)
            msgs.append(mk_msg("system", content))
        elif role in ("function_call", "tool_call"):
            obj = jloads_maybe(content) if isinstance(content, str) and content.strip().startswith("{") else None
            fc = m.get("function_call") or m.get("tool_call") or obj or {}
            if fc.get("name"):
                msgs.append(mk_msg("assistant", None, [mk_call(fc["name"], fc.get("arguments", {}))]))
        elif role in ("function", "tool_response", "tool"):
            msgs.append(mk_msg("tool", tool_responses=[{
                "name": normalize_tool_name(m["name"]) if m.get("name") else None,
                "response": content}]))
        elif role == "assistant":
            calls = []
            if isinstance(m.get("function_call"), dict) and m["function_call"].get("name"):
                calls = [mk_call(m["function_call"]["name"], m["function_call"].get("arguments", {}))]
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if fn.get("name"):
                    calls.append(mk_call(fn["name"], fn.get("arguments", {})))
            msgs.append(mk_msg("assistant", content, calls or None))
        elif role == "user":
            msgs.append(mk_msg("user", content))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map, "subset": row.get("subset_name")})


def norm_nemotron_terminal(row, cfg, ctx) -> Iterator[Record]:
    """Terminal transcript: user/assistant alternation. Assistant content carries
    <think>+JSON {analysis,plan,commands:[{keystrokes}]}; user carries
    'New Terminal Output: ...'. Synthesize a single `run_terminal` tool."""
    conv = row.get("conversations")
    if not isinstance(conv, list):
        return
    tool = {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Execute keystrokes in a terminal and return stdout.",
            "parameters": {"type": "object",
                           "properties": {"keystrokes": {"type": "string"}},
                           "required": ["keystrokes"]},
        },
    }
    msgs: list[dict] = []
    used_tool = False
    for turn in conv:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "assistant":
            block = jloads_maybe(strip_cot(content) or "")
            cmds = block.get("commands") if isinstance(block, dict) else None
            if cmds:
                used_tool = True
                # The env returns ONE combined output per turn, so emit ONE call with
                # the concatenated keystrokes (N parallel calls -> 1 response is malformed).
                keys = "".join(c.get("keystrokes", "") for c in cmds if isinstance(c, dict))
                prose = block.get("analysis") if isinstance(block, dict) else None
                msgs.append(mk_msg("assistant", prose, [mk_call("run_terminal", {"keystrokes": keys})]))
            else:
                msgs.append(mk_msg("assistant", content))
        elif role == "user":
            obs = re.sub(r"^New Terminal Output:\s*", "", content)
            if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("tool_calls"):
                msgs.append(mk_msg("tool", tool_responses=[{"name": "run_terminal", "response": obs}]))
            else:
                msgs.append(mk_msg("user", obs))
    tools = [tool] if used_tool else []
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"task": row.get("task"), "run_id": row.get("run_id")})


# -- Web normalizers -------------------------------------------------------- #

_WEB_TOOLS = [
    {"type": "function", "function": {"name": "click", "description": "Click an element by id.",
     "parameters": {"type": "object", "properties": {"element_id": {"type": "string"}}, "required": ["element_id"]}}},
    {"type": "function", "function": {"name": "type", "description": "Type text into an element.",
     "parameters": {"type": "object", "properties": {"element_id": {"type": "string"}, "text": {"type": "string"},
                    "press_enter": {"type": "boolean"}}, "required": ["element_id", "text"]}}},
    {"type": "function", "function": {"name": "scroll", "description": "Scroll the page.",
     "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                    "direction": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "goto", "description": "Navigate to a URL.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "select", "description": "Select an option.",
     "parameters": {"type": "object", "properties": {"element_id": {"type": "string"}, "value": {"type": "string"}}, "required": ["element_id"]}}},
]

_WEB_SYS = ("You are a web navigation agent. Given the page observation and the objective, "
            "issue one browser action via the available tools.")

_MAX_DOM_CHARS = 12000  # cap per-step DOM so multi-step web rows don't bloat to MBs


def _cap_dom(html: str) -> str:
    html = html or ""
    return html if len(html) <= _MAX_DOM_CHARS else html[:_MAX_DOM_CHARS] + "\n…[DOM truncated]"


_WA_ACT = (r"(?:click|type|scroll|goto|hover|press|new_tab|tab_focus|close_tab|"
           r"go_back|go_forward|stop|note|send_msg_to_user)")


def _extract_webarena_action(output: str) -> str | None:
    """Pull the WebArena action out of nnetnav's verbose CoT `output`. The action
    sits after 'the next action I will perform is ```ACTION```' (CoT is discarded)."""
    s = (output or "").strip()
    cands: list[str] = []
    m = re.search(r"next action[^\n]*?\bis\b[:\s]*`{0,3}\s*(" + _WA_ACT + r"\b[^\n`]*)", s, re.I)
    if m:
        cands.append(m.group(1))
    cands += [f.strip() for f in re.findall(r"```\s*([^`]+?)\s*```", s, re.DOTALL)]
    for line in reversed(s.splitlines()):
        t = line.strip().strip("`* ")
        if re.match(r"^" + _WA_ACT + r"\b", t, re.I):
            cands.append(t)
            break
    for c in cands:
        if re.match(r"^" + _WA_ACT + r"\b", c.strip(), re.I):
            return c.strip()
    return None


def norm_nnetnav(row, cfg, ctx) -> Iterator[Record]:
    """nnetnav-live: messages=[system,user(OBSERVATION...)], action in `output`
    column (WebArena grammar, wrapped in CoT). CoT and role labels are stripped;
    the action becomes a structured tool_call (or a final answer for stop/note)."""
    raw_msgs = row.get("messages") or []
    action = _extract_webarena_action(row.get("output") or "")
    if not action:
        return  # pure reasoning, no usable action -> drop
    msgs = [mk_msg("system", _WEB_SYS)]
    for m in raw_msgs:
        if m.get("role") == "user":
            msgs.append(mk_msg("user", m.get("content")))
    low = action.lower()
    if low.startswith(("stop", "note", "send_msg_to_user")):
        ans = re.sub(r"^\w+\s*\[?", "", action).rstrip("]").strip()
        msgs.append(mk_msg("assistant", ans or action))
    else:
        call = _parse_webarena_action(action)
        if call is None:
            return
        msgs.append(mk_msg("assistant", None, [call]))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=list(_WEB_TOOLS), tool_names=[t["function"]["name"] for t in _WEB_TOOLS],
                 metadata={"dataset": row.get("dataset"), "episode_id": row.get("id")})


def norm_weblinx(row, cfg, ctx) -> Iterator[Record]:
    """WebLINX: one step per row. obs = clean_html+candidates+viewport; target=action."""
    action = row.get("action") or ""
    if action.startswith("say("):
        m = re.search(r'utterance="(.*)"\)\s*$', action, re.DOTALL)
        assistant = mk_msg("assistant", m.group(1) if m else action)
    else:
        call = _parse_weblinx_action(action)
        assistant = mk_msg("assistant", None, [call]) if call else mk_msg("assistant", action)
    obs = (f"VIEWPORT: {row.get('viewport','')}\n"
           f"DOM:\n{_cap_dom(row.get('clean_html',''))}\n\nCANDIDATES:\n{(row.get('candidates') or '')[:_MAX_DOM_CHARS]}")
    utt = (row.get("utterances") or "").replace(" ", "")
    user_content = obs if not utt or "Noinstructor" in utt else f"INSTRUCTION: {row.get('utterances')}\n\n{obs}"
    msgs = [mk_msg("system", _WEB_SYS), mk_msg("user", user_content), assistant]
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=list(_WEB_TOOLS), tool_names=[t["function"]["name"] for t in _WEB_TOOLS],
                 metadata={"demo": row.get("demo"), "turn": row.get("turn")})


def norm_mind2web(row, cfg, ctx) -> Iterator[Record]:
    """Mind2Web: one trajectory per row; loop `actions`."""
    task = row.get("confirmed_task") or ""
    actions = row.get("actions") or []
    msgs = [mk_msg("system", _WEB_SYS), mk_msg("user", f"OBJECTIVE: {task}")]
    for act in actions:
        op = (act.get("operation") or {})
        opname = (op.get("op") or "").upper()
        pos = (act.get("pos_candidates") or [{}])[0]
        node = pos.get("backend_node_id")
        obs = f"DOM:\n{_cap_dom(act.get('cleaned_html',''))}"
        msgs.append(mk_msg("user", obs))
        if opname == "TYPE":
            call = mk_call("type", {"element_id": str(node), "text": op.get("value", "")})
        elif opname == "SELECT":
            call = mk_call("select", {"element_id": str(node), "value": op.get("value", "")})
        else:
            call = mk_call("click", {"element_id": str(node)})
        msgs.append(mk_msg("assistant", None, [call]))
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=list(_WEB_TOOLS), tool_names=[t["function"]["name"] for t in _WEB_TOOLS],
                 metadata={"website": row.get("website"), "domain": row.get("domain"),
                           "annotation_id": row.get("annotation_id")})


def norm_smoltalk2(row, cfg, ctx) -> Iterator[Record]:
    """smoltalk2 SFT: messages list; tools in chat_template_kwargs.xml_tools;
    calls/responses XML-embedded in content."""
    name_map: dict[str, str] = {}
    raw_msgs = row.get("messages") or []
    ctk = row.get("chat_template_kwargs") or {}
    tools = []
    for t in (ctk.get("xml_tools") or []):
        td = normalize_tool_def(jloads_maybe(t), name_map)
        if td:
            tools.append(td)
    msgs: list[dict] = []
    for m in raw_msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant" and "<tool_call>" in content:
            prose, calls = parse_xml_calls(content)
            msgs.append(mk_msg("assistant", prose, calls or None))
        elif role == "tool":
            msgs.append(mk_msg("tool", tool_responses=parse_xml_responses(content) or [{"name": None, "response": content}]))
        elif role in ("system", "user", "assistant"):
            msgs.append(mk_msg(role, content))
    if not tools:  # smolagents traces declare tools in prose/python_tools, not xml_tools
        tools = synthesize_tools_from_calls(msgs)
    yield Record(id="", source=ctx.source, source_subset=ctx.source_subset,
                 messages=msgs, tools=tools, tool_names=collect_tool_names(tools),
                 metadata={"name_map": name_map, "smoltalk_source": row.get("source")})


# --------------------------------------------------------------------------- #
# helpers for tool extraction from free text / system prompts
# --------------------------------------------------------------------------- #

def _extract_json_array(text: str, name_map: dict) -> list[dict]:
    """Find the first balanced top-level JSON array in text and normalize it."""
    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    cand = jloads_maybe(text[start:i + 1])
                    if isinstance(cand, list) and cand and isinstance(cand[0], dict) and cand[0].get("name"):
                        return tools_from_list(cand, name_map)
                    break
        start = text.find("[", start + 1)
    return []


_TOOL_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _harvest_tools_from_text(text: str, name_map: dict) -> list[dict]:
    """Glaive-raw style: concatenated JSON tool objects in a system prompt."""
    arr = _extract_json_array(text, name_map)
    if arr:
        return arr
    out = []
    for blob in _TOOL_OBJ_RE.findall(text or ""):
        obj = jloads_maybe(blob)
        if isinstance(obj, dict) and obj.get("name") and ("parameters" in obj or "description" in obj):
            td = normalize_tool_def(obj, name_map)
            if td:
                out.append(td)
    return out


def _harvest_tools_from_system(msgs: list[dict], name_map: dict) -> list[dict]:
    for m in msgs:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            t = parse_xml_tools_block(m["content"], name_map) or _harvest_tools_from_text(m["content"], name_map)
            if t:
                return t
    return []


def _strip_after_anchor(system: str) -> str | None:
    for anchor in ["Here is a list of functions in JSON format",
                   "you can invoke", "access to the following functions"]:
        idx = system.find(anchor)
        if idx != -1:
            return system[:idx].strip() or None
    return system.strip() or None


# -- web action parsers ----------------------------------------------------- #

def _parse_weblinx_action(action: str):
    m = re.match(r"(\w+)\((.*)\)\s*$", action.strip(), re.DOTALL)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    args = _parse_kv_args(argstr)
    if name == "click":
        return mk_call("click", {"element_id": args.get("uid", "")})
    if name == "scroll":
        return mk_call("scroll", {"x": args.get("x", 0), "y": args.get("y", 0)})
    if name in ("text", "change"):
        return mk_call("type", {"element_id": args.get("uid", ""), "text": args.get("value", args.get("text", ""))})
    if name == "submit":
        return mk_call("click", {"element_id": args.get("uid", "")})
    if name == "load":
        return mk_call("goto", {"url": args.get("url", "")})
    return mk_call(name, args)


def _parse_webarena_action(output: str):
    s = (output or "").strip()
    m = re.match(r"(\w+)\s*(.*)$", s)
    if not m:
        return None
    name = m.group(1).lower()
    brackets = re.findall(r"\[([^\]]*)\]", m.group(2))
    if name == "click" and brackets:
        return mk_call("click", {"element_id": brackets[0]})
    if name == "type" and len(brackets) >= 2:
        return mk_call("type", {"element_id": brackets[0], "text": brackets[1],
                                "press_enter": (brackets[2] == "1") if len(brackets) > 2 else False})
    if name == "scroll" and brackets:
        return mk_call("scroll", {"direction": brackets[0]})
    if name == "goto" and brackets:
        return mk_call("goto", {"url": brackets[0]})
    if name in ("stop", "answer"):
        return None  # final answer -> keep as assistant text
    return None


# --------------------------------------------------------------------------- #
# dispatch table
# --------------------------------------------------------------------------- #

NORMALIZERS = {
    "openai_messages": norm_openai_messages,
    "sharegpt": norm_sharegpt,
    "xlam": norm_xlam,
    "hermes": norm_hermes,
    "toolace": norm_toolace,
    "when2call": norm_when2call,
    "toucan": norm_toucan,
    "nemotron_terminal": norm_nemotron_terminal,
    "nnetnav": norm_nnetnav,
    "weblinx": norm_weblinx,
    "mind2web": norm_mind2web,
    "smoltalk2": norm_smoltalk2,
}


def get_normalizer(name: str):
    if name not in NORMALIZERS:
        raise KeyError(f"unknown normalizer '{name}'. known: {sorted(NORMALIZERS)}")
    return NORMALIZERS[name]
