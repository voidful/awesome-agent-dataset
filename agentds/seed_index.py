"""Build a dedup index from a reference dataset.

Streams a schema-compatible reference dataset (no full download) and recomputes
each row's canonical content_hash, so a run can be deduped against data you've
already trained on / shipped. Default reference is voidful/gemma4-agent-sft, but
any same-schema dataset works (pass --dedup-against). Hashes are cached to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import Record

SEED_ID = "voidful/gemma4-agent-sft"


def _row_to_record(row: dict) -> Record:
    return Record(
        id=row.get("id", ""),
        source=row.get("source", ""),
        source_subset=row.get("source_subset", ""),
        messages=json.loads(row["messages"]) if isinstance(row.get("messages"), str) else (row.get("messages") or []),
        tools=json.loads(row["tools"]) if isinstance(row.get("tools"), str) else (row.get("tools") or []),
        tool_names=row.get("tool_names") or [],
    )


def build_seed_hashes(cache_path: str | Path, seed_id: str = SEED_ID, limit: int | None = None) -> set[str]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return set(cache_path.read_text().split())

    from datasets import load_dataset

    hashes: set[str] = set()
    ds = load_dataset(seed_id, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        try:
            hashes.add(_row_to_record(row).content_hash())
        except Exception:
            continue
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(sorted(hashes)))
    return hashes
