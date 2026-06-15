"""agentds CLI.

  agentds validate [--tier T ...] [--key K ...] [-n 5]   # normalize live rows, sanity-check
  agentds seed [--limit N]                                 # build seed dedup hash cache
  agentds run --tier T ... [--out DIR] [--limit N] [--no-seed-dedup] [--no-near]
  agentds push --repo voidful/gemma4-agent-sft-v2 [--out DIR] [--dry-run] [--public]
  agentds stats [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import load_registry
from .normalizers import NormCtx, get_normalizer
from .schema import collect_tool_names, validate_record

DEFAULT_OUT = Path("data/expansion")
SEED_CACHE = Path("data/seed_hashes.txt")


def cmd_validate(args):
    reg = load_registry()
    sources = reg.by_key(args.key) if args.key else reg.by_tier(args.tier)
    from datasets import load_dataset

    for source in sources:
        norm = get_normalizer(source.normalizer)
        sub = source.subsets[0]
        print(f"\n=== {source.key}  ({source.hf_id} :: {sub.config}/{sub.split}) ===")
        try:
            ds = load_dataset(source.hf_id, name=sub.config, split=sub.split, streaming=True)
        except Exception as e:
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            continue
        ok = bad = 0
        shown = 0
        for i, row in enumerate(ds):
            if i >= args.n_rows:
                break
            ctx = NormCtx(source=source.key, source_subset=f"{sub.config}/{sub.split}")
            try:
                recs = list(norm(row, source.cfg, ctx))
            except Exception as e:
                print(f"  row {i}: NORMALIZER EXCEPTION {type(e).__name__}: {e}")
                bad += 1
                continue
            for rec in recs:
                rec.tool_names = collect_tool_names(rec.tools)
                reason = validate_record(rec)
                if reason:
                    bad += 1
                    if shown < args.show:
                        print(f"  row {i}: INVALID ({reason})  roles={[m['role'] for m in rec.messages]}")
                        shown += 1
                else:
                    ok += 1
                    if shown < args.show:
                        roles = [m["role"] for m in rec.messages]
                        ncalls = sum(len(m.get("tool_calls") or []) for m in rec.messages)
                        print(f"  row {i}: OK  roles={roles} tools={len(rec.tools)} calls={ncalls}")
                        if args.dump and shown == 0:
                            print(json.dumps(rec.to_row(), ensure_ascii=False, indent=2)[:2000])
                        shown += 1
        print(f"  -> ok={ok} bad={bad}")


def cmd_seed(args):
    from .seed_index import build_seed_hashes

    SEED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    hashes = build_seed_hashes(SEED_CACHE, limit=args.limit)
    print(f"seed hashes: {len(hashes)} -> {SEED_CACHE}")


def cmd_run(args):
    from . import pipeline
    from .seed_index import build_seed_hashes

    reg = load_registry()
    seed_hashes = None
    if not args.no_seed_dedup:
        SEED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        seed_hashes = build_seed_hashes(SEED_CACHE, limit=args.seed_limit)
        print(f"[seed] {len(seed_hashes)} hashes")
    report = pipeline.run(
        reg, out_dir=args.out, tiers=args.tier or None, keys=args.key or None,
        seed_hashes=seed_hashes, near_dedup=(not args.no_near),
        limit_per_subset=args.limit,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


def cmd_push(args):
    from .push import push

    print(push(args.out, args.repo, private=not args.public, dry_run=args.dry_run))


def cmd_stats(args):
    rep = json.loads((Path(args.out) / "_report.json").read_text())
    print(json.dumps(rep, indent=2, ensure_ascii=False))


def cmd_catalog(args):
    from .catalog import write_catalog

    print(write_catalog(args.outfile))


def cmd_audit(args):
    from .audit import print_report

    print_report(args.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="agentds")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="normalize a few live rows per source and report")
    v.add_argument("--tier", action="extend", nargs="*", default=[])
    v.add_argument("--key", action="extend", nargs="*", default=[])
    v.add_argument("-n", "--n-rows", type=int, default=5)
    v.add_argument("--show", type=int, default=3)
    v.add_argument("--dump", action="store_true", help="dump first OK record JSON")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("seed", help="build seed dedup hash cache")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_seed)

    r = sub.add_parser("run", help="run the ingestion pipeline")
    r.add_argument("--tier", action="extend", nargs="*", default=[])
    r.add_argument("--key", action="extend", nargs="*", default=[])
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--limit", type=int, default=None, help="cap rows per subset (override registry)")
    r.add_argument("--seed-limit", type=int, default=None)
    r.add_argument("--no-seed-dedup", action="store_true")
    r.add_argument("--no-near", action="store_true")
    r.set_defaults(func=cmd_run)

    pu = sub.add_parser("push", help="push shards to a new HF dataset repo")
    pu.add_argument("--repo", required=True)
    pu.add_argument("--out", default=str(DEFAULT_OUT))
    pu.add_argument("--public", action="store_true")
    pu.add_argument("--dry-run", action="store_true")
    pu.set_defaults(func=cmd_push)

    st = sub.add_parser("stats", help="print the last run report")
    st.add_argument("--out", default=str(DEFAULT_OUT))
    st.set_defaults(func=cmd_stats)

    cat = sub.add_parser("catalog", help="regenerate CATALOG.md from the registry")
    cat.add_argument("--outfile", default="CATALOG.md")
    cat.set_defaults(func=cmd_catalog)

    au = sub.add_parser("audit", help="quantify data-quality defects in produced shards")
    au.add_argument("--out", default=str(DEFAULT_OUT))
    au.set_defaults(func=cmd_audit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
