"""Registry loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "configs" / "registry.yaml"


@dataclass
class Subset:
    config: str
    split: str
    max_rows: int | None = None


@dataclass
class Source:
    key: str
    hf_id: str
    tier: str
    normalizer: str = "openai_messages"
    dedup_group: str = ""
    enabled: bool = True
    license: str = "unknown"
    cfg: dict = field(default_factory=dict)
    ext: dict = field(default_factory=dict)          # canonical signal -> column name
    provenance: list[str] = field(default_factory=list)  # columns -> metadata
    subsets: list[Subset] = field(default_factory=list)
    notes: str = ""
    stream: bool = True     # False -> robust full download then iterate (teich/agent-traces)
    # --- catalog metadata (awesome-list display only) ---
    rows: str = ""          # approx upstream size, e.g. "109k", "12.3M"
    tokens: str = ""        # approx token count if known, e.g. "111B"
    format: str = ""        # upstream format family
    status: str = ""        # override; else derived (wired/listed)

    @property
    def wired(self) -> bool:
        return self.enabled and bool(self.subsets)


@dataclass
class Registry:
    defaults: dict
    sources: list[Source]

    def by_tier(self, tiers: list[str] | None) -> list[Source]:
        out = [s for s in self.sources if s.enabled and s.subsets]
        if tiers:
            out = [s for s in out if s.tier in tiers]
        return out

    def by_key(self, keys: list[str]) -> list[Source]:
        return [s for s in self.sources if s.key in keys]

    @property
    def tiers(self) -> list[str]:
        return sorted({s.tier for s in self.sources})


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> Registry:
    data = yaml.safe_load(Path(path).read_text())
    sources = []
    for s in data.get("sources", []):
        subsets = [Subset(**ss) for ss in (s.get("subsets") or [])]
        sources.append(Source(
            key=s["key"], hf_id=s["hf_id"], tier=s["tier"],
            normalizer=s.get("normalizer", "openai_messages"),
            dedup_group=s.get("dedup_group", s["key"]), enabled=s.get("enabled", True),
            license=s.get("license", "unknown"), cfg=s.get("cfg") or {},
            ext=s.get("ext") or {}, provenance=s.get("provenance") or [],
            subsets=subsets, notes=s.get("notes", ""), stream=s.get("stream", True),
            rows=str(s.get("rows", "")), tokens=str(s.get("tokens", "")),
            format=s.get("format", ""), status=s.get("status", ""),
        ))
    return Registry(defaults=data.get("defaults", {}), sources=sources)
