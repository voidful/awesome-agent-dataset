"""Group-level deduplication.

Three layers, applied in order:
  1. exact      : xxhash of normalized (messages + tool_names) content.
  2. SWE-group  : (repo, base_commit, issue/PR, instance_id) — collapses the same
                  GitHub issue appearing across SWE-Zero / nebius / SWE-Gym /
                  SWE-smith / CoderForge.
  3. near       : MinHash + LSH over assistant action/tool-schema shingles, so
                  reformatted forks (glaive/xlam/toolace variants) don't double-count.

The DeDuper is stateful and used across the whole run, so cross-source duplicates
(e.g. ToolMind rows that re-package xLAM) are caught. Seed hashes can be preloaded
to dedup the expansion against the published gemma4-agent-sft.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from .schema import Record, _norm_text


# --------------------------------------------------------------------------- #
# SWE provenance keying
# --------------------------------------------------------------------------- #

_INSTANCE_NUM_RE = re.compile(r"^(?P<repo>.+?)-(?P<num>\d+)$")
_RUN_SUFFIX_RE = re.compile(r"_run\d+$", re.IGNORECASE)


def swe_group_key(meta: dict) -> str | None:
    """Derive a cross-dataset SWE identity from metadata, or None if not SWE.

    Two id families need different granularity:
      * Real SWE-bench ids "owner__repo-NNNN" (SWE-Zero / nebius / SWE-Gym) — collapse
        by issue number so the same GitHub issue across datasets dedups to one.
      * Synthetic ids "owner__repo.<commit>.bugtype__suffix" (SWE-smith) or
        "owner__repo_<commit>_bugtype__suffix_runN" (CoderForge) — key on the FULL id
        (minus a trailing _runN), because each bug is a DISTINCT task that merely
        shares a repo+commit. (Collapsing on repo+commit would drop ~99% of them.)
    """
    inst = (meta.get("instance_id") or meta.get("trajectory_id") or "").strip()
    repo = meta.get("repo")
    if not inst and not repo:
        return None
    # SWE-bench style: the part before any '.' ends in "-<number>".
    base = inst.split(".")[0]
    if _INSTANCE_NUM_RE.match(base):
        return f"swe::{base.lower()}"
    if inst:
        norm = _RUN_SUFFIX_RE.sub("", inst).lower()
        return f"swe::{norm}"
    return f"swe::repo::{repo.lower()}" if repo else None


# --------------------------------------------------------------------------- #
# Near-dup shingles
# --------------------------------------------------------------------------- #

_MAX_SHINGLES = 1500  # bound MinHash cost on very long trajectories (SWE/agent-traces)


def _shingles(rec: Record, k: int = 5) -> set[str]:
    """Word-level k-shingles over assistant actions + tool schema names.

    Focusing on assistant behavior (not the user prompt or tool observations)
    makes the near-dup signal robust to reformatting and re-templating. Shingles
    are capped (long coding sessions can yield 10k+ shingles, making MinHash the
    run bottleneck); a deterministic stride-sample keeps near-dup quality high.
    """
    parts: list[str] = []
    for m in rec.messages:
        if m.get("role") == "assistant":
            parts.append(_norm_text(m.get("content")))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                parts.append(fn.get("name", ""))
                parts.append(_norm_text(fn.get("arguments")))
    parts.extend(sorted(rec.tool_names))
    text = " ".join(p for p in parts if p)
    toks = text.split()
    if len(toks) < k:
        return {text} if text else set()
    n = len(toks) - k + 1
    if n <= _MAX_SHINGLES:
        return {" ".join(toks[i:i + k]) for i in range(n)}
    stride = n // _MAX_SHINGLES + 1
    return {" ".join(toks[i:i + k]) for i in range(0, n, stride)}


def _minhash(shingles: set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s.encode())
    return mh


# --------------------------------------------------------------------------- #
# DeDuper
# --------------------------------------------------------------------------- #

@dataclass
class DedupStats:
    seen: int = 0
    kept: int = 0
    exact_dups: int = 0
    swe_dups: int = 0
    near_dups: int = 0
    per_reason: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"seen": self.seen, "kept": self.kept, "exact_dups": self.exact_dups,
                "swe_dups": self.swe_dups, "near_dups": self.near_dups}


class DeDuper:
    def __init__(self, near: bool = True, threshold: float = 0.85, num_perm: int = 64):
        self.exact: set[str] = set()
        self.swe: set[str] = set()
        self.near_enabled = near
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if near else None
        self._near_id = 0
        self.stats = DedupStats()

    # -- seed preloading (dedup expansion vs published dataset) ---------- #
    def preload_exact(self, hashes: Iterable[str]) -> int:
        n = 0
        for h in hashes:
            if h not in self.exact:
                self.exact.add(h)
                n += 1
        return n

    def is_dup(self, rec: Record) -> str | None:
        """Return a dup-reason string if rec is a duplicate, else None.
        Side effect: registers the record's signatures when kept."""
        self.stats.seen += 1

        h = rec.content_hash()
        if h in self.exact:
            self.stats.exact_dups += 1
            return "exact"

        sk = swe_group_key(rec.metadata)
        if sk:
            if sk in self.swe:
                self.stats.swe_dups += 1
                return "swe_group"

        if self.near_enabled:
            sh = _shingles(rec)
            if sh:
                mh = _minhash(sh, self.num_perm)
                if self.lsh.query(mh):
                    self.stats.near_dups += 1
                    return "near"

        # Not a duplicate -> register.
        self.exact.add(h)
        if sk:
            self.swe.add(sk)
        if self.near_enabled and sh:
            key = f"n{self._near_id}"
            self._near_id += 1
            self.lsh.insert(key, mh)
        self.stats.kept += 1
        return None
