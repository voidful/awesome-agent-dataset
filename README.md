<div align="center">

# 🤖 awesome-agent-dataset

**A curated catalog of agent-training datasets — *plus* a working toolkit that normalizes, filters, and deduplicates them into one canonical schema.**

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-voidful%2Fagent--sft-orange)](https://huggingface.co/datasets/voidful/agent-sft)

*Most "agent dataset" lists stop at links. This one ships the pipeline too —*
*so you get **clean, deduplicated, schema-unified data**, not a pile of incompatible formats.*

[**📚 Full Catalog**](CATALOG.md) · [**🤗 Output Dataset**](https://huggingface.co/datasets/voidful/agent-sft) · [**🚀 Quickstart**](#-quickstart) · [**🧩 Schema**](#-canonical-schema) · [**🤝 Contributing**](CONTRIBUTING.md)

</div>

---

## Why this exists

For fine-tuning an agent model, **finding tokens is not the bottleneck** — public agent data already far exceeds what a 30B FT needs. The real bottlenecks are:

1. **Format chaos** — every dataset uses a different shape (xLAM `query/answers`, Glaive flat-text, Hermes XML tags, ToolACE Python-call DSL, OpenHands event streams, WebArena action grammars, the new HF `agent-traces` format…).
2. **Massive overlap** — the same GitHub issue appears across 5 SWE datasets; xLAM/Glaive/ToolACE get re-packaged a dozen times. Naively concatenating **severely overestimates** your real data volume.
3. **Quality variance** — valid JSON ≠ a successful task; you need stratification.
4. **Coding-agent skew** — SWE/terminal data is so abundant it drowns general agent ability if unbalanced.

`agentds` solves all four: **one normalizer per format → group-level dedup → quality tiers → balanced mixture**, producing [`voidful/agent-sft`](https://huggingface.co/datasets/voidful/agent-sft) in the same canonical schema as the [`gemma4-agent-sft`](https://huggingface.co/datasets/voidful/gemma4-agent-sft) seed (so they concatenate).

## ✨ Output dataset

[**🤗 voidful/agent-sft**](https://huggingface.co/datasets/voidful/agent-sft) — produced entirely by this repo from the wired sources, deduplicated against the published seed.

<!-- STATS:START -->
**328,451 rows** from 27 wired sources, deduplicated against the published seed.

| Tier | Rows | Share |
|---|--:|--:|
| 🛠️ function_calling | 172,269 | 52% |
| 💻 swe_terminal | 61,153 | 19% |
| 🌐 web | 46,601 | 14% |
| 💬 general | 41,470 | 13% |
| 🧵 agent_traces | 6,958 | 2% |
| **Total** | **328,451** | |

**Quality:** 122,844 high (37%) · 194,695 medium (59%) · 10,912 low (3%)
**Dedup removed 83,829 candidates** — 43,914 SWE-group (same GitHub issue across SWE datasets) · 31,072 near-dup (MinHash) · 8,843 exact, *plus* dedup against the published seed. (E.g. `ansulev/DeepSeek-v4-Pro-Agent` → **0 kept**, fully collapsed into its `TeichAI` twin.)

Coding-heavy data (swe_terminal + agent_traces) is held to **~21%** so general agent ability isn't drowned. See [CATALOG.md](CATALOG.md) for per-source counts.
<!-- STATS:END -->

## 🚀 Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

agentds catalog                       # regenerate CATALOG.md from the registry
agentds validate --tier function_calling -n 30   # normalize LIVE rows, sanity-check
agentds run --tier function_calling --tier agent_traces   # stream → normalize → dedup → shards
agentds stats                         # composition + dedup + quality report
agentds push --repo you/your-agent-sft --public          # to a new HF dataset repo
```

Pick tiers (`function_calling`, `agent_traces`, `swe_terminal`, `web`, `general`) or individual
sources (`--key apigen --key swe_gym`). `--limit N` caps rows/subset for a quick dry run.

## 🧩 Canonical schema

Identical to [`voidful/gemma4-agent-sft`](https://huggingface.co/datasets/voidful/gemma4-agent-sft) so shards concatenate cleanly:

| field | type | description |
|---|---|---|
| `id` | str | `{source}_{config}_{hash16}` |
| `source` | str | normalized source key |
| `source_subset` | str | `config/split` within the source |
| `messages` | str (JSON) | `list[{role, content, tool_calls?, tool_responses?}]` |
| `tools` | str (JSON) | `list[{type:"function", function:{name, description, parameters}}]` |
| `tool_names` | list[str] | declared tool names |
| `quality` | str (JSON) | `{tier, score, curated, signals}` |
| `metadata` | str (JSON) | `{hf_id, license, dedup_group, instance_id, …}` |

- `tool_calls[].function.arguments` are **objects** (string-encoded args parsed).
- Chain-of-thought (`<think>…</think>`, `reasoning_content`) and foreign chat-template markers are stripped.
- `parameters` coerced to a JSON-schema `object` (xLAM/Hermes flat styles wrapped; `str`/`int` → `string`/`integer`).

<details><summary>Example normalized row</summary>

```json
{
  "id": "apigen_mt_dataset_9009cd98a0542977",
  "source": "apigen_mt",
  "source_subset": "dataset/train",
  "messages": "[{\"role\":\"system\",\"content\":\"# Airline Agent Policy…\"},{\"role\":\"user\",\"content\":\"I'd like to cancel a reservation.\"},{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"function\":{\"name\":\"get_reservation_details\",\"arguments\":{\"reservation_id\":\"0U4NPP\"}}}]},{\"role\":\"tool\",\"tool_responses\":[{\"name\":\"get_reservation_details\",\"response\":{\"reservation_id\":\"0U4NPP\",\"status\":\"active\"}}]},{\"role\":\"assistant\",\"content\":\"Your reservation 0U4NPP is active — shall I cancel it?\"}]",
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"get_reservation_details\",\"description\":\"…\",\"parameters\":{\"type\":\"object\",\"properties\":{\"reservation_id\":{\"type\":\"string\"}},\"required\":[\"reservation_id\"]}}}]",
  "tool_names": ["get_reservation_details", "cancel_reservation", "..."],
  "quality": "{\"tier\":\"high\",\"score\":0.9,\"curated\":false,\"signals\":{\"n_turns\":5,\"n_tool_calls\":1,\"multi_turn\":true,\"valid_arg_ratio\":1.0}}",
  "metadata": "{\"hf_id\":\"Salesforce/APIGen-MT-5k\",\"license\":\"cc-by-4.0\",\"dedup_group\":\"xlam_apigen\"}"
}
```
</details>

## 📚 Catalog

**[→ Full catalog of 60+ datasets, by tier, with normalization status (CATALOG.md)](CATALOG.md)**

| Tier | What it teaches | Wired sources (sample) |
|---|---|---|
| 🛠️ **function_calling** | when/how to call tools, schema grounding, when *not* to call | apigen(xLAM), glaive, toolace, when2call, hermes, hermes_reasoning, toolmind |
| 🧵 **agent_traces** | real `claude_code`/`pi` coding-agent sessions (HF `format: agent-traces`, decoded via `teich`) | DeepSeek-v4-Pro, synthtraces, qwen3.7-max-pi, minimax-m3, ml-intern |
| 💻 **swe_terminal** | SWE repair, shell/terminal, long-horizon coding (streamed + sampled) | swe_gym, swe_rebench, swe_zero, swe_smith, coderforge, nemotron-terminal |
| 🌐 **web** | observation→action loops (DOM/AXTree as text) | weblinx, mind2web, nnetnav |
| 💬 **general** | retention — keep natural-answer ability, avoid over-tool-calling | openhermes, smoltalk2 |

Adding a dataset = an entry in [`configs/registry.yaml`](configs/registry.yaml) (+ a normalizer if it's a new format). See [CONTRIBUTING.md](CONTRIBUTING.md).

## ⚙️ How it works

```
                        configs/registry.yaml  (single source of truth)
                                  │
   HF source ──stream──▶ normalize ──▶ validate ──▶ group-dedup ──▶ quality ──▶ parquet shards ──▶ 🤗 push
  (streaming=True,     (per-format)   (schema     (exact + SWE-     (tiers)
   never fully          │             well-formed) provenance +
   downloaded)          │                          MinHash near-dup,
                        │                          incl. vs seed)
                  agentds/normalizers.py     agentds/dedup.py   agentds/quality.py
```

- **Normalizers** ([`agentds/normalizers.py`](agentds/normalizers.py)) — one per format family: xLAM, ShareGPT (incl. Glaive `function_call`/`observation`), Hermes `<tool_call>` XML, ToolACE BFCL `[Func(k=v)]` (paren/space/path-style names), When2Call `<TOOLCALL>` + appropriate-refusal rows, native OpenHands SWE trajectories, Nemotron terminal transcripts, WebLINX/Mind2Web/WebArena action grammars, and the HF `agent-traces` format. Tools are synthesized from observed calls when a source ships no schema.
- **Group-level dedup** ([`agentds/dedup.py`](agentds/dedup.py)) — (1) exact xxhash of normalized content; (2) **SWE-provenance** key so the same GitHub issue across SWE-Zero/nebius/SWE-Gym/SWE-smith/CoderForge collapses to one (real `repo-NNNN` ids by issue number; synthetic ids at full granularity); (3) **MinHash + LSH** near-dup over assistant action/tool-schema shingles. Stateful across the whole run + preloads the published seed's hashes.
- **Quality** ([`agentds/quality.py`](agentds/quality.py)) — `{tier: high|medium|low, score, curated, signals}`; rewards multi-turn, schema-valid, observation-grounded tool use; folds in source success signals (SWE `resolved`, CoderForge `reward`); penalizes degenerate trajectories.

## 🧪 Recommended training recipe

- **Stage A — agentic continued post-training** (10–30B tok): SWE/terminal 55% · tool-use 20% · web 15% · general 10%.
- **Stage B — high-quality agent SFT** (1–3B tok): filter to `quality.tier == "high"` + verified successes.
- **Stage C — RL / rejection sampling**: use executable/verified subsets (`reward==1`, `resolved`).

Recommended loss mask:

```
system / user / tool-schema / tool-observation : 0
assistant natural language / final answer      : 1.0
assistant tool-call JSON                        : 1.5
assistant recovery-after-error action           : 2.0
```

## 🤝 Contributing

PRs that **add datasets** or **wire up catalog-only entries** are the most valuable — see [CONTRIBUTING.md](CONTRIBUTING.md). The bar: it must normalize cleanly (`agentds validate` green) and declare a `dedup_group`.

## 🧪 Tests

```bash
.venv/bin/python -m tests.test_normalizers   # offline, fixture-based
```

## 📄 License & citation

Code: [MIT](LICENSE). **Each dataset keeps its upstream license** — recorded in every row's `metadata.license`; review before downstream use (sources span apache-2.0 / mit / cc-by-4.0 and restricted terms like cc-by-nc-sa-4.0).

```bibtex
@misc{awesome-agent-dataset,
  title  = {awesome-agent-dataset: a catalog and normalization toolkit for agent-training data},
  author = {voidful},
  year   = {2026},
  url    = {https://github.com/voidful/awesome-agent-dataset}
}
```

## 🙏 Acknowledgements

Built on the open datasets catalogued here and the HuggingFace `datasets` / `teich` / `datasketch` ecosystems. Seed format from [`voidful/gemma4-agent-sft`](https://huggingface.co/datasets/voidful/gemma4-agent-sft).
