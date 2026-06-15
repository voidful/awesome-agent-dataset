"""agentds — expansion toolkit for the gemma4-agent-sft canonical schema.

Pipeline: stream HF source -> normalize to canonical schema -> group-level dedup
-> quality stratify -> sharded parquet -> push to HF hub.
"""
import os as _os

# Raise HF read/etag timeouts: the default ~10s read timeout trips on large
# agent-traces downloads, and a single timeout closes the shared HTTP client and
# poisons every later request in the run. Generous timeouts prevent the cascade.
_os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
_os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

__version__ = "0.1.0"
