"""Generate the awesome-list catalog (CATALOG.md) from the registry.

Single source of truth: configs/registry.yaml. Wired sources (enabled + subsets)
are marked with the row count actually kept in the latest run report; catalog-only
entries (enabled: false) are listed as inventory for contributors to wire up.
"""
from __future__ import annotations

import json
from pathlib import Path

from .registry import Registry, load_registry

TIER_TITLES = {
    "function_calling": "🛠️ Tool / Function Calling",
    "swe_terminal": "💻 SWE / Terminal (environment interaction)",
    "web": "🌐 Web / Browser / GUI",
    "agent_traces": "🧵 Agent Traces (real coding-agent sessions)",
    "general": "💬 General Instruction (retention)",
    "core": "🧩 Core Agent Trajectory Corpora",
    "rl": "🎯 RL / Verifier / Rejection-Sampling",
}
TIER_ORDER = ["function_calling", "agent_traces", "swe_terminal", "web", "core", "rl", "general"]


def _hf_link(hf_id: str) -> str:
    return f"[`{hf_id}`](https://huggingface.co/datasets/{hf_id})"


def _status(src) -> str:
    if src.status:
        return src.status
    return "✅ wired" if src.wired else "📋 listed"


def generate(registry: Registry, report: dict | None = None) -> str:
    # Catalog is deterministic from the registry alone (CI-checkable). Per-source
    # processed counts live in the README composition table + the HF dataset card.
    out: list[str] = []
    out.append("# 📚 Agent-Dataset Catalog\n")
    out.append("Curated inventory of agent / tool-use / SWE / web datasets on the HF Hub, "
               "with normalization status. **✅ wired** = ingested by the pipeline into the "
               "[canonical schema](README.md#-canonical-schema); **📋 listed** = catalogued, "
               "ready to wire up (PRs welcome).\n")
    out.append("> Auto-generated from [`configs/registry.yaml`](configs/registry.yaml) via "
               "`agentds catalog`. Do not edit by hand.\n")

    total = len(registry.sources)
    wired = sum(1 for s in registry.sources if s.wired)
    out.append(f"**{total} datasets catalogued · {wired} wired into the pipeline**\n")

    tiers = [t for t in TIER_ORDER if any(s.tier == t for s in registry.sources)]
    tiers += [t for t in sorted({s.tier for s in registry.sources}) if t not in tiers]

    # table of contents
    out.append("## Contents")
    for t in tiers:
        title = TIER_TITLES.get(t, t)
        anchor = title.lower().replace(" ", "-")
        for ch in "/()🛠️💻🌐🧵💬🧩🎯.":
            anchor = anchor.replace(ch, "")
        anchor = anchor.strip("-").replace("--", "-")
        out.append(f"- [{title}](#{anchor})")
    out.append("")

    for t in tiers:
        srcs = [s for s in registry.sources if s.tier == t]
        srcs.sort(key=lambda s: (not s.wired, s.hf_id.lower()))
        out.append(f"## {TIER_TITLES.get(t, t)}\n")
        out.append("| Dataset | Rows | Tokens | License | Format | Status | Notes |")
        out.append("|---|---|---|---|---|---|---|")
        for s in srcs:
            out.append("| {ds} | {rows} | {tok} | {lic} | {fmt} | {st} | {notes} |".format(
                ds=_hf_link(s.hf_id), rows=s.rows or "—", tok=s.tokens or "—",
                lic=s.license, fmt=s.format or s.normalizer, st=_status(s),
                notes=(s.notes or "").replace("|", "\\|")[:90]))
        out.append("")

    out.append("## Dedup groups\n")
    out.append("Sources sharing a `dedup_group` are deduplicated together (and against the "
               "published seed) so reformatted forks and the same GitHub issue across SWE "
               "datasets don't inflate the count.\n")
    groups: dict[str, list[str]] = {}
    for s in registry.sources:
        groups.setdefault(s.dedup_group, []).append(s.key)
    out.append("| Group | Members |")
    out.append("|---|---|")
    for g, members in sorted(groups.items()):
        out.append(f"| `{g}` | {', '.join(sorted(members))} |")
    out.append("")
    return "\n".join(out)


def write_catalog(out_path: str | Path = "CATALOG.md", report_path: str | Path | None = None) -> str:
    reg = load_registry()
    report = None
    rp = Path(report_path) if report_path else Path("data/expansion/_report.json")
    if rp.exists():
        report = json.loads(rp.read_text())
    md = generate(reg, report)
    Path(out_path).write_text(md)
    return f"wrote {out_path} ({len(reg.sources)} datasets, {sum(s.wired for s in reg.sources)} wired)"
