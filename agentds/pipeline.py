"""Streaming ingestion orchestrator: stream -> normalize -> dedup -> quality -> shard.

Big SWE sources are read with ``streaming=True`` and a per-subset row cap, so
they are never fully materialized on disk.
"""
from __future__ import annotations

import json
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import quality
from .dedup import DeDuper
from .normalizers import NormCtx, get_normalizer
from .registry import Registry, Source, Subset
from .schema import Record, collect_tool_names, validate_record

ROW_SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("source_subset", pa.string()),
    ("messages", pa.string()), ("tools", pa.string()),
    ("tool_names", pa.list_(pa.string())),
    ("quality", pa.string()), ("metadata", pa.string()),
])


class ShardWriter:
    def __init__(self, out_dir: Path, rows_per_shard: int = 25000):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_shard = rows_per_shard
        self.buf: list[dict] = []
        self.shard_idx = 0
        self.total = 0

    def add(self, row: dict):
        self.buf.append(row)
        self.total += 1
        if len(self.buf) >= self.rows_per_shard:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        table = pa.Table.from_pylist(self.buf, schema=ROW_SCHEMA)
        path = self.out_dir / f"shard-{self.shard_idx:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self.shard_idx += 1
        self.buf = []

    def close(self):
        self.flush()


@dataclass
class RunReport:
    started: float
    tiers: list[str]
    per_source: dict = field(default_factory=dict)
    dedup: dict = field(default_factory=dict)
    filter_reasons: Counter = field(default_factory=Counter)
    tier_counts: Counter = field(default_factory=Counter)
    quality_tiers: Counter = field(default_factory=Counter)
    errors: dict = field(default_factory=dict)
    total_written: int = 0

    def to_dict(self) -> dict:
        return {
            "tiers": self.tiers,
            "total_written": self.total_written,
            "duration_sec": round(time.time() - self.started, 1),
            "per_source": self.per_source,
            "dedup": self.dedup,
            "tier_counts": dict(self.tier_counts),
            "quality_tiers": dict(self.quality_tiers),
            "filter_reasons": dict(self.filter_reasons.most_common(20)),
            "errors": self.errors,
        }


def _ext_signals(row: dict, source: Source) -> dict:
    sig: dict = {}
    for canonical, col in source.ext.items():
        if col and col in row:
            sig[canonical] = row[col]
    return sig


def _provenance(row: dict, source: Source) -> dict:
    return {c: row[c] for c in source.provenance if c in row and row[c] is not None}


def _make_id(source_key: str, config: str, content_hash: str) -> str:
    # Keep the FULL 16-hex content hash (truncating it risks id collisions); cap the
    # human-readable prefix instead so the total stays ~<=51 chars.
    prefix = f"{source_key}_{config}".replace("/", "_")[:34]
    return f"{prefix}_{content_hash[:16]}"


def run(
    registry: Registry,
    out_dir: str | Path,
    tiers: list[str] | None = None,
    keys: list[str] | None = None,
    seed_hashes: set[str] | None = None,
    near_dedup: bool | None = None,
    limit_per_subset: int | None = None,
    log=print,
) -> RunReport:
    from datasets import load_dataset

    out_dir = Path(out_dir)
    near = registry.defaults.get("near_dedup", True) if near_dedup is None else near_dedup
    thr = registry.defaults.get("near_threshold", 0.85)
    deduper = DeDuper(near=near, threshold=thr)
    if seed_hashes:
        n = deduper.preload_exact(seed_hashes)
        log(f"[seed] preloaded {n} seed content hashes for cross-dedup")

    writer = ShardWriter(out_dir / "data")
    report = RunReport(started=time.time(), tiers=tiers or registry.tiers)

    sources = registry.by_key(keys) if keys else registry.by_tier(tiers)
    log(f"[plan] {len(sources)} sources: {[s.key for s in sources]}")

    for source in sources:
        norm = get_normalizer(source.normalizer)
        src_written = 0
        src_seen = 0
        for sub in source.subsets:
            cap = sub.max_rows
            if limit_per_subset is not None:
                cap = min(cap or limit_per_subset, limit_per_subset)
            label = f"{source.key}:{sub.config}/{sub.split}"
            t0 = time.time()
            try:
                ds = load_dataset(source.hf_id, name=sub.config, split=sub.split,
                                  streaming=source.stream)
            except Exception as e:  # gated / script-only / bad config
                report.errors[label] = f"{type(e).__name__}: {e}"[:300]
                log(f"[skip] {label}: {report.errors[label]}")
                continue

            n_in = n_out = 0
            it = iter(ds)
            while True:
                if cap and n_in >= cap:
                    break
                try:
                    row = next(it)
                except StopIteration:
                    break
                except Exception as e:  # mid-stream network/parquet blip
                    report.errors[f"{label}@{n_in}"] = f"{type(e).__name__}: {e}"[:200]
                    log(f"[warn] {label}: stream ended early at {n_in}: {type(e).__name__}")
                    break
                n_in += 1
                src_seen += 1
                ctx = NormCtx(source=source.key, source_subset=f"{sub.config}/{sub.split}")
                try:
                    recs = list(norm(row, source.cfg, ctx))
                except Exception:
                    report.filter_reasons["normalizer_exception"] += 1
                    continue
                for rec in recs:
                    try:
                        rec.tool_names = collect_tool_names(rec.tools)
                        rec.metadata.update(_provenance(row, source))
                        rec.metadata.setdefault("hf_id", source.hf_id)
                        rec.metadata.setdefault("license", source.license)
                        rec.metadata.setdefault("dedup_group", source.dedup_group)

                        reason = validate_record(rec)
                        if reason:
                            report.filter_reasons[reason] += 1
                            continue

                        dup = deduper.is_dup(rec)
                        if dup:
                            report.filter_reasons[f"dup_{dup}"] += 1
                            continue

                        rec.quality = quality.score(rec, _ext_signals(row, source))
                        rec.id = _make_id(source.key, sub.config, rec.content_hash())
                        writer.add(rec.to_row())
                    except Exception as e:  # never let one bad record kill the run
                        report.filter_reasons[f"finalize_error:{type(e).__name__}"] += 1
                        continue
                    n_out += 1
                    report.tier_counts[source.tier] += 1
                    report.quality_tiers[rec.quality["tier"]] += 1

            src_written += n_out
            log(f"[done] {label}: in={n_in} out={n_out} ({time.time()-t0:.0f}s)")

        report.per_source[source.key] = {"tier": source.tier, "seen": src_seen, "written": src_written}

    writer.close()
    report.total_written = writer.total
    report.dedup = deduper.stats.as_dict()

    (out_dir / "_report.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    log(f"[write] {writer.total} rows in {writer.shard_idx} shards -> {out_dir/'data'}")
    return report
