# Contributing

Thanks for helping grow the catalog and the toolkit! The two highest-value contributions are
**adding a dataset** and **wiring up a catalog-only (📋 listed) entry**.

## Add a dataset to the catalog (listing only)

Append an entry to [`configs/registry.yaml`](configs/registry.yaml) under `sources:` with
`enabled: false`. Minimal fields:

```yaml
  - {key: c_mydata, hf_id: org/my-agent-dataset, tier: function_calling,
     dedup_group: my_family, enabled: false, license: apache-2.0,
     rows: "50k", format: sharegpt, notes: "one-line description"}
```

Then regenerate the catalog and open a PR:

```bash
agentds catalog          # updates CATALOG.md
```

`tier` ∈ `function_calling | agent_traces | swe_terminal | web | general | core | rl`.

## Wire a dataset into the pipeline (✅ wired)

1. **Find its format.** Inspect a few rows:
   ```bash
   curl "https://datasets-server.huggingface.co/first-rows?dataset=org%2Fname&config=default&split=train"
   ```
2. **Reuse or add a normalizer.** If it matches an existing family
   ([`agentds/normalizers.py`](agentds/normalizers.py): `openai_messages`, `sharegpt`, `xlam`,
   `hermes`, `toolace`, `when2call`, `toucan`, `nemotron_terminal`, `weblinx`, `mind2web`,
   `nnetnav`, `smoltalk2`), just point to it via `normalizer:` + `cfg:` column hints. If it's a
   genuinely new shape, add a `norm_<name>(row, cfg, ctx) -> Iterator[Record]` function and
   register it in the `NORMALIZERS` dispatch table.
3. **Fill the registry entry** with `subsets` (config/split + `max_rows` cap), `cfg`, optional
   `ext` (success signals → column) and `provenance` (columns → metadata; required for SWE dedup):
   ```yaml
   - key: mydata
     hf_id: org/my-agent-dataset
     tier: function_calling
     normalizer: sharegpt
     dedup_group: my_family
     enabled: true
     license: apache-2.0
     cfg: {conv_col: conversations, tools_col: tools}
     subsets:
       - {config: default, split: train, max_rows: 20000}
   ```
4. **Validate against live data** — this must be green:
   ```bash
   agentds validate --key mydata -n 30
   ```
5. **Add a fixture test** in [`tests/test_normalizers.py`](tests/test_normalizers.py) if you added
   a normalizer, and run `python -m tests.test_normalizers`.

## Guidelines

- **Always declare a `dedup_group`.** Reformatted forks and re-packaged corpora must share a group
  so they dedup together (this is the whole point — don't inflate the count).
- **Stream big sources.** Anything over a few GB is read with `streaming=True` + a `max_rows` cap;
  never assume the full dataset fits on disk.
- **Respect licenses.** Put the correct upstream `license` in the entry — it's written into every
  row's `metadata.license`. Don't wire gated datasets without noting the gating.
- **Keep coding-agent data balanced.** SWE/terminal/agent-traces data is abundant; cap it so it
  doesn't drown general agent ability.
- **Strip CoT.** Normalizers drop `<think>…</think>` / `reasoning_content` to match the canonical schema.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m tests.test_normalizers
```
