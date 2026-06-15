"""Push expansion shards to a new HF dataset repo + generate a dataset card."""
from __future__ import annotations

import json
from pathlib import Path

CARD_TEMPLATE = """---
license: other
task_categories:
- text-generation
tags:
- agent
- tool-use
- function-calling
- swe
- web-agent
- gemma4-agent-sft
configs:
- config_name: default
  data_files: data/*.parquet
---

# {repo_id}

Expansion of [voidful/gemma4-agent-sft](https://huggingface.co/datasets/voidful/gemma4-agent-sft),
sharing its **exact canonical schema** so the two can be concatenated.

Built with the [agentds](https://github.com/voidful/awesome-agent-dataset) toolkit:
per-source normalization -> group-level dedup (exact + SWE-provenance + MinHash near-dup,
including dedup against the published seed) -> heuristic quality stratification.

## Schema

| field | type | description |
|---|---|---|
| `id` | str | `{{source}}_{{config}}_{{hash}}` |
| `source` | str | normalized source key |
| `source_subset` | str | `config/split` within the source |
| `messages` | str (JSON) | `list[{{role, content, tool_calls?, tool_responses?}}]` |
| `tools` | str (JSON) | `list[{{type:"function", function:{{name, description, parameters}}}}]` |
| `tool_names` | list[str] | declared tool names |
| `quality` | str (JSON) | `{{tier, score, curated, signals}}` |
| `metadata` | str (JSON) | provenance: hf_id, license, dedup_group, instance_id, ... |

`tool_calls[].function.arguments` are objects; chain-of-thought and foreign
chat-template markers are stripped (matching the seed).

## Composition

Total rows: **{total}**

### By tier
{tier_table}

### By quality tier
{quality_table}

### By source
{source_table}

## Dedup

{dedup_table}

## Provenance & licenses

Each row's `metadata.hf_id` / `metadata.license` records its origin. Licenses are
inherited from upstream sources; review them before downstream use. Sources span
permissive (apache-2.0, mit, cc-by-4.0) and restricted (cc-by-nc-sa-4.0 for WebLINX)
terms.

## Recommended loss mask

```
system / user / tool-schema / tool-observation : 0
assistant natural language / final answer      : 1.0
assistant tool-call JSON                        : 1.5
assistant recovery-after-error action           : 2.0
```
"""


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_card(repo_id: str, report: dict) -> str:
    tier_table = _md_table(["tier", "rows"], list(report.get("tier_counts", {}).items()))
    quality_table = _md_table(["quality", "rows"], list(report.get("quality_tiers", {}).items()))
    source_table = _md_table(
        ["source", "tier", "written"],
        [[k, v["tier"], v["written"]] for k, v in report.get("per_source", {}).items()],
    )
    d = report.get("dedup", {})
    dedup_table = _md_table(
        ["metric", "count"],
        [["candidates seen", d.get("seen", 0)], ["kept", d.get("kept", 0)],
         ["exact dups", d.get("exact_dups", 0)], ["SWE-group dups", d.get("swe_dups", 0)],
         ["near dups", d.get("near_dups", 0)]],
    )
    return CARD_TEMPLATE.format(
        repo_id=repo_id, total=report.get("total_written", 0),
        tier_table=tier_table, quality_table=quality_table,
        source_table=source_table, dedup_table=dedup_table,
    )


def push(out_dir: str | Path, repo_id: str, private: bool = True, dry_run: bool = False) -> str:
    from huggingface_hub import HfApi

    out_dir = Path(out_dir)
    report = json.loads((out_dir / "_report.json").read_text())
    card = build_card(repo_id, report)
    (out_dir / "README.md").write_text(card)

    if dry_run:
        return f"[dry-run] would push {report['total_written']} rows to {repo_id}"

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset",
        allow_patterns=["data/*.parquet", "README.md", "_report.json"],
        delete_patterns=["data/*.parquet"],  # clean overwrite: drop stale shards
        commit_message="agentds expansion batch",
    )
    return f"pushed {report['total_written']} rows to https://huggingface.co/datasets/{repo_id}"
