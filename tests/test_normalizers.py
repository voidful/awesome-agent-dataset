"""Offline unit tests for normalizers + dedup, using fixtures from the recon.

Run: .venv/bin/python -m tests.test_normalizers   (or pytest)
"""
from __future__ import annotations

import json

from agentds.normalizers import NormCtx, get_normalizer, parse_bfcl_calls
from agentds.schema import coerce_parameters, normalize_tool_name, validate_record
from agentds.dedup import DeDuper, swe_group_key

CTX = NormCtx("test", "c/s")
_FAILURES: list[str] = []


def check(cond, msg):
    if not cond:
        _FAILURES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def first(norm_name, row, cfg=None):
    recs = list(get_normalizer(norm_name)(row, cfg or {}, CTX))
    assert recs, f"{norm_name} produced no records"
    rec = recs[0]
    rec.tool_names = [t["function"]["name"] for t in rec.tools]
    return rec


def test_coerce_parameters():
    print("\n[coerce_parameters]")
    # xLAM flat style with python types + optional
    flat = {"type": {"description": "d", "type": "str", "default": "game"},
            "count": {"description": "n", "type": "int, optional"}}
    out = coerce_parameters(flat)
    check(out["type"] == "object", "wraps flat -> object")
    check(out["properties"]["type"]["type"] == "string", "str -> string")
    check(out["properties"]["count"]["type"] == "integer", "int -> integer")
    check("count" not in out.get("required", []), "optional not required")
    # already json-schema
    js = {"type": "object", "properties": {"a": {"type": "str"}}, "required": ["a"]}
    out2 = coerce_parameters(js)
    check(out2["properties"]["a"]["type"] == "string", "coerces nested str in json-schema")
    check(out2["required"] == ["a"], "keeps required")


def test_xlam():
    print("\n[xlam]")
    row = {"query": "live giveaways?",
           "answers": json.dumps([{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                                  {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}]),
           "tools": json.dumps([{"name": "live_giveaways_by_type", "description": "d",
                                 "parameters": {"type": {"type": "str", "description": "t"}}}])}
    rec = first("xlam", row, {"query_col": "query", "answers_col": "answers", "tools_col": "tools"})
    calls = rec.messages[1]["tool_calls"]
    check(len(calls) == 2, "two parallel calls")
    check(calls[0]["function"]["arguments"] == {"type": "beta"}, "args object")
    check(rec.metadata.get("parallel_tool_calls") is True, "parallel flag set")
    check(validate_record(rec) is None, "valid record")


def test_glaive_sharegpt():
    print("\n[sharegpt/glaive]")
    row = {"tools": json.dumps([{"name": "get_news_headlines", "description": "d",
                                 "parameters": {"type": "object", "properties": {"country": {"type": "string"}}}}]),
           "conversations": [
               {"from": "human", "value": "latest US news?"},
               {"from": "function_call", "value": json.dumps({"name": "get_news_headlines", "arguments": {"country": "US"}})},
               {"from": "observation", "value": json.dumps({"headlines": ["a", "b"]})},
               {"from": "gpt", "value": "Here are the headlines..."}]}
    rec = first("sharegpt", row, {"conv_col": "conversations", "tools_col": "tools"})
    roles = [m["role"] for m in rec.messages]
    check(roles == ["user", "assistant", "tool", "assistant"], f"roles {roles}")
    check(rec.messages[1]["tool_calls"][0]["function"]["arguments"] == {"country": "US"}, "call args parsed")
    check(rec.messages[2]["tool_responses"][0]["response"] == {"headlines": ["a", "b"]}, "tool response parsed")
    check(validate_record(rec) is None, "valid")


def test_hermes_xml():
    print("\n[hermes]")
    row = {"conversations": [
        {"from": "system", "value": 'You are... <tools>\n[{"type":"function","function":{"name":"get_feed","description":"d","parameters":{"type":"object","properties":{"id":{"type":"string"}}}}}]\n</tools>'},
        {"from": "human", "value": "feed for front_door?"},
        {"from": "gpt", "value": '<tool_call>\n{"name":"get_feed","arguments":{"id":"front_door"}}\n</tool_call>'},
        {"from": "tool", "value": '<tool_response>\n{"name":"get_feed","content":{"url":"x"}}\n</tool_response>'},
        {"from": "gpt", "value": "Done."}], "tools": None}
    rec = first("hermes", row, {"conv_col": "conversations", "tools_col": "tools"})
    check(len(rec.tools) == 1 and rec.tools[0]["function"]["name"] == "get_feed", "tools parsed from <tools>")
    check(rec.messages[2]["tool_calls"][0]["function"]["arguments"] == {"id": "front_door"}, "xml call parsed")
    check(rec.messages[3]["tool_responses"][0]["response"] == {"url": "x"}, "xml response parsed")
    check(validate_record(rec) is None, "valid")


def test_toolace_bfcl():
    print("\n[toolace/bfcl]")
    calls = parse_bfcl_calls('[Market Trends API(trend_type="MARKET_INDEXES", country="us")]')
    check(len(calls) == 1, "one bfcl call")
    check(calls[0]["function"]["name"] == "Market_Trends_API", "name normalized (spaces)")
    check(calls[0]["function"]["arguments"] == {"trend_type": "MARKET_INDEXES", "country": "us"}, "kv args")
    multi = parse_bfcl_calls('[f(a=1), g(b="x")]')
    check(len(multi) == 2, "two bfcl calls")
    check(multi[0]["function"]["arguments"] == {"a": 1}, "int arg")


def test_when2call():
    print("\n[when2call]")
    row = {"tools": [json.dumps({"name": "get_x", "description": "d", "parameters": {"type": "object", "properties": {}}})],
           "messages": [{"role": "user", "content": "trending in NYC?"},
                        {"role": "assistant", "content": "Sorry, I can't provide real-time info."}]}
    rec = first("when2call", row)
    check(rec.metadata.get("no_call_expected") is True, "no-call refusal flagged")
    check(validate_record(rec) is None, "valid refusal record")
    row2 = {"tools": [json.dumps({"name": "get_ico", "description": "d", "parameters": {"type": "object", "properties": {"category": {"type": "string"}}}})],
            "messages": [{"role": "user", "content": "ico calendar?"},
                         {"role": "assistant", "content": '<TOOLCALL>[{"name":"get_ico","arguments":{"category":"defi"}}]</TOOLCALL>'}]}
    rec2 = first("when2call", row2)
    check(rec2.messages[1].get("tool_calls"), "TOOLCALL marker -> call")


def test_swe_native_and_clean():
    print("\n[openai_messages/swe]")
    row = {"trajectory": [
        {"role": "system", "content": "OpenHands"},
        {"role": "user", "content": "<uploaded_files>"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "str_replace_editor", "arguments": json.dumps({"command": "view", "path": "/x"})}}]},
        {"role": "tool", "content": json.dumps([{"type": "text", "text": "OBSERVATION:\ndir listing"}])}],
        "instance_id": "pytorch__vision-6883", "repo": "pytorch/vision"}
    cfg = {"messages_col": "trajectory", "tools_col": None, "synthesize_tools": True, "clean_tool_content": True}
    rec = first("openai_messages", row, cfg)
    check(rec.messages[2]["tool_calls"][0]["function"]["arguments"] == {"command": "view", "path": "/x"}, "string args parsed")
    check(rec.messages[3]["tool_responses"][0]["response"] == "dir listing", "tool content cleaned")
    check(len(rec.tools) == 1, "tools synthesized from calls")
    check(validate_record(rec) is None, "valid")


def test_nemotron_terminal():
    print("\n[nemotron_terminal]")
    block = {"analysis": "list files", "plan": "ls", "commands": [{"keystrokes": "ls -la\n", "duration": "0.1"}]}
    row = {"conversations": [
        {"role": "user", "content": "You are an AI... task"},
        {"role": "assistant", "content": "<think>hmm</think>" + json.dumps(block)},
        {"role": "user", "content": "New Terminal Output: total 0"}],
        "task": "data_task_0717"}
    rec = first("nemotron_terminal", row)
    check(rec.messages[1]["tool_calls"][0]["function"]["name"] == "run_terminal", "terminal tool call")
    check(rec.messages[1]["tool_calls"][0]["function"]["arguments"]["keystrokes"] == "ls -la\n", "keystrokes")
    check(rec.messages[2]["role"] == "tool", "terminal output -> tool")
    check(validate_record(rec) is None, "valid")


def test_web():
    print("\n[web: weblinx/mind2web/nnetnav]")
    wl = first("weblinx", {"action": 'click(uid="abc")', "clean_html": "<html>", "candidates": "(uid=abc)",
                            "viewport": "714h x 1536w", "utterances": "open momondo", "demo": "d1", "turn": 2})
    check(wl.messages[-1]["tool_calls"][0]["function"]["arguments"] == {"element_id": "abc"}, "weblinx click")
    m2 = first("mind2web", {"confirmed_task": "find hotel",
                            "actions": [{"operation": {"op": "CLICK", "value": ""},
                                         "cleaned_html": "<button>",
                                         "pos_candidates": [{"backend_node_id": "123", "tag": "button"}]}]})
    check(m2.messages[-1]["tool_calls"][0]["function"]["arguments"]["element_id"] == "123", "mind2web click node")
    nn = first("nnetnav", {"messages": [{"role": "system", "content": "sys"},
                                        {"role": "user", "content": "OBSERVATION: ... OBJECTIVE: x"}],
                           "output": "type [89] [water chemistry] [1]"})
    args = nn.messages[-1]["tool_calls"][0]["function"]["arguments"]
    check(args == {"element_id": "89", "text": "water chemistry", "press_enter": True}, f"nnetnav type {args}")


def test_smoltalk2():
    print("\n[smoltalk2]")
    row = {"messages": [{"role": "user", "content": "live giveaways?"},
                        {"role": "assistant", "content": '<tool_call>\n{"name":"live_giveaways_by_type","arguments":{"type":"beta"}}\n</tool_call>'}],
           "chat_template_kwargs": {"xml_tools": [json.dumps({"type": "function", "function": {"name": "live_giveaways_by_type", "description": "d", "parameters": {"type": "object", "properties": {"type": {"type": "string"}}}}})]},
           "source": "xlam"}
    rec = first("smoltalk2", row)
    check(len(rec.tools) == 1, "xml_tools parsed")
    check(rec.messages[1]["tool_calls"][0]["function"]["arguments"] == {"type": "beta"}, "tool_call parsed")
    check(validate_record(rec) is None, "valid")


def test_dedup():
    print("\n[dedup]")
    d = DeDuper(near=True)
    row = {"query": "q", "answers": json.dumps([{"name": "f", "arguments": {"a": 1}}]),
           "tools": json.dumps([{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}])}
    cfg = {"query_col": "query", "answers_col": "answers", "tools_col": "tools"}
    r1 = first("xlam", row, cfg)
    r2 = first("xlam", row, cfg)
    check(d.is_dup(r1) is None, "first kept")
    check(d.is_dup(r2) == "exact", "identical -> exact dup")
    # SWE group key — real SWE-bench ids collapse by issue number across datasets
    k1 = swe_group_key({"instance_id": "getmoto__moto-5321"})
    k2 = swe_group_key({"instance_id": "getmoto__moto-5321", "repo": "getmoto/moto"})
    check(k1 == k2, f"same SWE-bench instance -> same key ({k1})")
    check(swe_group_key({"instance_id": "getmoto__moto-9999"}) != k1, "different issue -> different key")
    # Synthetic SWE-smith: same repo+commit, DIFFERENT bug -> DIFFERENT key (was the bug)
    s1 = swe_group_key({"instance_id": "django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__aaa"})
    s2 = swe_group_key({"instance_id": "django-money__django-money.835c1ab8.func_pm_remove_cond__bbb"})
    check(s1 != s2, "synthetic bugs sharing repo+commit are NOT collapsed")
    # CoderForge _runN suffix collapses re-runs of the same synthetic bug
    c1 = swe_group_key({"trajectory_id": "HIPS__autograd_ac044f0d_func_pm_ctrl_invert_if__y2_run1"})
    c2 = swe_group_key({"trajectory_id": "HIPS__autograd_ac044f0d_func_pm_ctrl_invert_if__y2_run2"})
    check(c1 == c2, "same synthetic bug, different run -> same key")
    check(swe_group_key({"source": "x"}) is None, "non-SWE -> no key")


def test_ellipsis_robust():
    print("\n[ellipsis robustness]")
    from agentds.schema import Record, jdumps, jloads_maybe
    # jloads_maybe must NOT turn a literal "..." into python Ellipsis
    v = jloads_maybe("{'a': ...}")
    check(v == "{'a': ...}" or (isinstance(v, str)), "pseudo-json with ... kept as string, not Ellipsis")
    # a record carrying an Ellipsis must still serialize + hash without crashing
    rec = Record(id="", source="s", source_subset="x",
                 messages=[{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "ok"},
                           {"role": "tool", "tool_responses": [{"name": None, "response": Ellipsis}]}],
                 tools=[])
    try:
        rec.content_hash(); row = rec.to_row(); json.loads(row["messages"])
        check(True, "Ellipsis record hashes + serializes")
    except Exception as e:
        check(False, f"Ellipsis crashed: {type(e).__name__}")
    check('"response":null' in jdumps([{"response": Ellipsis}]), "Ellipsis -> null in jdumps")


def test_qa_fixes():
    print("\n[QA regression fixes]")
    from agentds.schema import coerce_parameters, jdumps
    from agentds.normalizers import (parse_function_xml_calls, _extract_webarena_action,
                                     clean_tool_response, synthesize_tools_from_calls)
    # 1. coerce_parameters must NOT corrupt type:"dict" + properties (xLAM/ToolACE)
    d = coerce_parameters({"type": "dict", "properties": {"x": {"type": "str"}}, "required": ["x"]})
    check(d["type"] == "object" and set(d["properties"]) == {"x"}, f"type:dict preserved props ({list(d['properties'])})")
    check(d["properties"]["x"]["type"] == "string", "nested dict type coerced")
    check(d.get("required") == ["x"], "dict required kept")
    check("properties" not in d["properties"] and "type" not in d["properties"], "no schema-key leakage as args")
    # empty object schema
    check(coerce_parameters({"type": "object"})["properties"] == {}, "empty object schema ok")
    # 2. SWE-agent function XML
    prose, calls = parse_function_xml_calls("Let me look.\n<function=bash>\n<parameter=command>ls -la</parameter>\n</function>")
    check(len(calls) == 1 and calls[0]["function"]["name"] == "bash", "function-xml name")
    check(calls[0]["function"]["arguments"] == {"command": "ls -la"}, "function-xml param")
    check(prose == "Let me look.", "function-xml prose extracted")
    # 3. nnetnav action extraction from CoT
    a = _extract_webarena_action("Let's think... In summary, the next action I will perform is ```type [89] [water] [1]```\n")
    check(a == "type [89] [water] [1]", f"nnetnav action extracted ({a})")
    # 4. clean_tool_response strips plain-string OBSERVATION (swe_gym)
    check(clean_tool_response("OBSERVATION:\ndir listing") == "dir listing", "plain OBSERVATION stripped")
    # 5. synthesize merges args across calls
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1, "b": "x"}}}]}]
    tools = synthesize_tools_from_calls(msgs)
    props = tools[0]["function"]["parameters"]["properties"]
    check(set(props) == {"a", "b"}, f"synthesized tool unions args ({set(props)})")
    # 6. jdumps NaN -> null (valid JSON)
    out = jdumps({"x": float("nan"), "y": float("inf")})
    check('"x":null' in out and '"y":null' in out, "NaN/Inf -> null")
    json.loads(out)  # must be valid JSON
    check(True, "jdumps output is valid JSON")
    # 7. unclosed <think> CoT is stripped (hermes_reasoning/toolmind reasoning)
    from agentds.schema import strip_cot
    check(strip_cot("<think>\nlong reasoning with no close tag") is None, "unclosed <think> -> dropped")
    check(strip_cot("<think>reason</think>answer") == "answer", "closed <think> -> answer kept")
    check(strip_cot("<think>r1</think>mid<think>r2 unclosed") == "mid", "mixed closed+unclosed think")
    # 8. content cap on pathological dumps
    from agentds.normalizers import _cap
    big = "x" * 300_000
    check(len(_cap(big)) < 220_000, "oversized content capped")


def main():
    for fn in [test_coerce_parameters, test_xlam, test_glaive_sharegpt, test_hermes_xml,
               test_toolace_bfcl, test_when2call, test_swe_native_and_clean, test_nemotron_terminal,
               test_web, test_smoltalk2, test_dedup, test_ellipsis_robust, test_qa_fixes]:
        fn()
    print("\n" + ("=" * 50))
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} checks")
        for f in _FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
