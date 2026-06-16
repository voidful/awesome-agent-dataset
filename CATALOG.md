# 📚 Agent-Dataset Catalog

Curated inventory of agent / tool-use / SWE / web datasets on the HF Hub, with normalization status. **✅ wired** = ingested by the pipeline into the [canonical schema](README.md#-canonical-schema); **📋 listed** = catalogued, ready to wire up (PRs welcome).

> Auto-generated from [`configs/registry.yaml`](configs/registry.yaml) via `agentds catalog`. Do not edit by hand.

**63 datasets catalogued · 27 wired into the pipeline**

## Contents
- [🛠️ Tool / Function Calling](#tool-function-calling)
- [🧵 Agent Traces (real coding-agent sessions)](#agent-traces-real-coding-agent-sessions)
- [💻 SWE / Terminal (environment interaction)](#swe-terminal-environment-interaction)
- [🌐 Web / Browser / GUI](#web-browser-gui)
- [🧩 Core Agent Trajectory Corpora](#core-agent-trajectory-corpora)
- [🎯 RL / Verifier / Rejection-Sampling](#rl-verifier-rejection-sampling)
- [💬 General Instruction (retention)](#general-instruction-retention)

## 🛠️ Tool / Function Calling

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`argilla/apigen-function-calling`](https://huggingface.co/datasets/argilla/apigen-function-calling) | — | — | cc-by-4.0 | xlam | ✅ wired |  |
| [`hiyouga/glaive-function-calling-v2-sharegpt`](https://huggingface.co/datasets/hiyouga/glaive-function-calling-v2-sharegpt) | — | — | apache-2.0 | sharegpt | ✅ wired |  |
| [`interstellarninja/hermes_reasoning_tool_use`](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use) | — | — | apache-2.0 | hermes | ✅ wired |  |
| [`Nanbeige/ToolMind`](https://huggingface.co/datasets/Nanbeige/ToolMind) | — | — | apache-2.0 | openai_messages | ✅ wired |  |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | — | — | apache-2.0 | hermes | ✅ wired |  |
| [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) | — | — | cc-by-4.0 | when2call | ✅ wired |  |
| [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) | — | — | cc-by-4.0 | sharegpt | ✅ wired |  |
| [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | — | — | apache-2.0 | toolace | ✅ wired |  |
| [`amityco/apigen-tau-bench-split-turn`](https://huggingface.co/datasets/amityco/apigen-tau-bench-split-turn) | 46k | — | apache-2.0 | sharegpt | 📋 listed | tau-bench multi-turn splits |
| [`argilla/Synth-APIGen-v0.1`](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1) | 49k | — | apache-2.0 | xlam_query_answers | 📋 listed | subset of apigen-function-calling (wired) |
| [`bytedance-research/ToolHop`](https://huggingface.co/datasets/bytedance-research/ToolHop) | — | — | apache-2.0 | other | 📋 listed | multi-hop tool reasoning w/ executable tools |
| [`fuvty/tau-bench-synthetic`](https://huggingface.co/datasets/fuvty/tau-bench-synthetic) | — | — | mit | other | 📋 listed | customer-service tool-use |
| [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113k | — | apache-2.0 | glaive_flat_text | 📋 listed | raw flat-text; prefer the sharegpt fork (wired) |
| [`nvidia/Nemotron-Agentic-v1`](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | 335k | — | cc-by-4.0 | openai_messages | 📋 listed | interactive agent + tool calling |
| [`nvidia/Nemotron-SFT-Agentic-v2`](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | 992k | — | cc-by-4.0 | openai_messages | 📋 listed | in the gemma4-agent-sft reference; tool-calling SFT main |
| [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) | 9k | — | cc-by-4.0 | openai_messages | 📋 listed | train_pref DPO pairs (when-not-to-call) |
| [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k | — | cc-by-4.0 | xlam_query_answers | 📋 listed | canonical xLAM; GATED — needs HF license + token |
| [`stabletoolbench/ToolEnv2404`](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404) | — | — | apache-2.0 | other | 📋 listed | StableToolBench tool environment |

## 🧵 Agent Traces (real coding-agent sessions)

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`ansulev/DeepSeek-v4-Pro-Agent`](https://huggingface.co/datasets/ansulev/DeepSeek-v4-Pro-Agent) | — | — | other | openai_messages | ✅ wired |  |
| [`armand0e/minimax-m3-claude-code-traces`](https://huggingface.co/datasets/armand0e/minimax-m3-claude-code-traces) | — | — | other | openai_messages | ✅ wired |  |
| [`armand0e/qwen3.7-max-pi-traces`](https://huggingface.co/datasets/armand0e/qwen3.7-max-pi-traces) | — | — | other | openai_messages | ✅ wired |  |
| [`clem/hf-coding-tools-traces`](https://huggingface.co/datasets/clem/hf-coding-tools-traces) | — | — | other | openai_messages | ✅ wired |  |
| [`julien-c/synthtraces`](https://huggingface.co/datasets/julien-c/synthtraces) | — | — | other | openai_messages | ✅ wired |  |
| [`lewtun/ml-intern-sessions`](https://huggingface.co/datasets/lewtun/ml-intern-sessions) | — | — | other | openai_messages | ✅ wired |  |
| [`TeichAI/DeepSeek-v4-Pro-Agent`](https://huggingface.co/datasets/TeichAI/DeepSeek-v4-Pro-Agent) | — | — | other | openai_messages | ✅ wired |  |
| [`thomasmustier/pi-for-excel-sessions`](https://huggingface.co/datasets/thomasmustier/pi-for-excel-sessions) | — | — | other | openai_messages | ✅ wired |  |
| [`jedisct1/security-audits`](https://huggingface.co/datasets/jedisct1/security-audits) | 35k | — | other | agent_traces | 📋 listed | 3.3GB; download times out — wire up via offline shard-sampling |

## 💻 SWE / Terminal (environment interaction)

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`nebius/SWE-rebench-openhands-trajectories`](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | — | — | cc-by-4.0 | openai_messages | ✅ wired |  |
| [`nvidia/Nemotron-Terminal-Corpus`](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | — | — | cc-by-4.0 | nemotron_terminal | ✅ wired |  |
| [`nvidia/SWE-Zero-openhands-trajectories`](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories) | — | — | cc-by-4.0 | openai_messages | ✅ wired |  |
| [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) | — | — | mit | openai_messages | ✅ wired |  |
| [`SWE-Gym/OpenHands-Sampled-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | — | — | mit | openai_messages | ✅ wired |  |
| [`togethercomputer/CoderForge-Preview`](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) | — | — | apache-2.0 | openai_messages | ✅ wired |  |
| [`AlienKevin/SWE-ZERO-12M-trajectories`](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories) | 12.29M | 111B | cc-by-4.0 | openhands_events | 📋 listed | mid-training corpus (NOT verified SFT); stream+sample only |
| [`internlm/SWE-Fixer-Train-110K`](https://huggingface.co/datasets/internlm/SWE-Fixer-Train-110K) | 115k | — | apache-2.0 | raw_swe_instance | 📋 listed | GitHub issue resolution / SWE repair |
| [`nvidia/SWE-Hero-openhands-trajectories`](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories) | 34k | — | cc-by-4.0 | openhands_events | 📋 listed | long-horizon coding |
| [`R2E-Gym/R2E-Gym-V1`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1) | 7478 | — | mit | raw_swe_instance | 📋 listed | real-world repo execution tasks |
| [`SWE-Gym/OpenHands-Verifier-Trajectories`](https://huggingface.co/datasets/SWE-Gym/OpenHands-Verifier-Trajectories) | 5272 | — | mit | openhands_events | 📋 listed | verifier / rejection sampling |
| [`SWE-Gym/SWE-Gym`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 2438 | — | mit | raw_swe_instance | 📋 listed | validated instances; train + eval |
| [`SWE-Gym/SWE-Gym-Raw`](https://huggingface.co/datasets/SWE-Gym/SWE-Gym-Raw) | 64.7k | — | mit | raw_swe_instance | 📋 listed | raw, unverified; environment expansion |

## 🌐 Web / Browser / GUI

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`McGill-NLP/WebLINX`](https://huggingface.co/datasets/McGill-NLP/WebLINX) | — | — | cc-by-nc-sa-4.0 | weblinx | ✅ wired |  |
| [`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web) | — | — | cc-by-4.0 | mind2web | ✅ wired |  |
| [`stanfordnlp/nnetnav-live`](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | — | — | mit | nnetnav | ✅ wired |  |
| [`iMeanAI/Mind2Web-Live`](https://huggingface.co/datasets/iMeanAI/Mind2Web-Live) | 646 | — | cc-by-4.0 | web_action_trajectory | 📋 listed | live web tasks |
| [`McGill-NLP/A3-Synth`](https://huggingface.co/datasets/McGill-NLP/A3-Synth) | — | — | cc-by-4.0 | openai_messages | 📋 listed | screenshot-only observations; needs OCR/captioning before a text model can use it |
| [`OpenGVLab/ScaleCUA-Data`](https://huggingface.co/datasets/OpenGVLab/ScaleCUA-Data) | — | — | apache-2.0 | web_action_trajectory | 📋 listed | cross-platform GUI (Linux/macOS/Win/Android/iOS/Web) |
| [`osunlp/Multimodal-Mind2Web`](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) | 14.2k | — | cc-by-4.0 | web_action_trajectory | 📋 listed | multimodal (screenshots) — needs OCR for text model |
| [`stanfordnlp/nnetnav-wa`](https://huggingface.co/datasets/stanfordnlp/nnetnav-wa) | — | — | mit | web_action_trajectory | 📋 listed | WebArena-style synthetic demos |

## 🧩 Core Agent Trajectory Corpora

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`Agent-Ark/Toucan-1.5M`](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | 1.5M | — | apache-2.0 | toucan | 📋 listed | in gemma4-agent-sft reference; MCP tool trajectories |
| [`CharlieDreemur/OpenManus-RL`](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL) | — | — | apache-2.0 | other | 📋 listed | ReAct-style; AgentInstruct+Agent-FLAN+AgentGym |
| [`neulab/agent-data-collection`](https://huggingface.co/datasets/neulab/agent-data-collection) | — | — | mit | other | 📋 listed | unified multi-source (OpenHands/SWE/Mind2Web/...) |
| [`open-thoughts/AgentTrove`](https://huggingface.co/datasets/open-thoughts/AgentTrove) | 1.70M | — | apache-2.0 | openai_messages | 📋 listed | in gemma4-agent-sft reference; 219-source agent trajectory pool |
| [`open-thoughts/OpenThoughts-Agent-v1-SFT`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT) | 15.2k | — | apache-2.0 | openai_messages | 📋 listed | agent SFT |
| [`Solaris99/AgentBank`](https://huggingface.co/datasets/Solaris99/AgentBank) | 53k | — | apache-2.0 | other | 📋 listed | agent skill diversity |
| [`zai-org/AgentInstruct`](https://huggingface.co/datasets/zai-org/AgentInstruct) | 1866 | — | apache-2.0 | sharegpt | 📋 listed | high-quality starter (ALFWorld/WebShop/KG/OS/DB) |

## 🎯 RL / Verifier / Rejection-Sampling

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1) | — | — | cc-by-4.0 | other | 📋 listed | step-level behavior cloning pivots |
| [`open-thoughts/OpenThoughts-Agent-RL-5K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-RL-5K) | 5k | — | apache-2.0 | other | 📋 listed | executable tasks w/ verifier/reward |
| [`open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K) | 9437 | — | apache-2.0 | openai_messages | 📋 listed | RL cold start |
| [`open-thoughts/TaskTrove`](https://huggingface.co/datasets/open-thoughts/TaskTrove) | 17.2k | — | apache-2.0 | other | 📋 listed | AgentTrove task complement |

## 💬 General Instruction (retention)

| Dataset | Rows | Tokens | License | Format | Status | Notes |
|---|---|---|---|---|---|---|
| [`HuggingFaceTB/smoltalk2`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | — | — | apache-2.0 | smoltalk2 | ✅ wired |  |
| [`teknium/OpenHermes-2.5`](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | — | — | other | sharegpt | ✅ wired |  |
| [`allenai/Dolci-Instruct-SFT-Tool-Use`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use) | — | — | odc-by | openai_messages | 📋 listed | OLMo 3 Instruct tool-use retention |
| [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | 1.1M | — | apache-2.0 | openai_messages | 📋 listed | incl. APIGen-FC 80k subset; overlaps APIGen |

## Dedup groups

Sources sharing a `dedup_group` are deduplicated together (and against the published seed) so reformatted forks and the same GitHub issue across SWE datasets don't inflate the count.

| Group | Members |
|---|---|
| `a3_synth` | a3_synth |
| `agent_traces` | at_deepseek_v4_pro, at_deepseek_v4_pro_mirror, at_hf_coding_tools, at_minimax_m3_cc, at_ml_intern, at_pi_excel, at_qwen37_max_pi, at_security_audits, at_synthtraces |
| `agentbank` | c_agentbank |
| `agentinstruct` | c_agentinstruct |
| `agenttrove` | c_agenttrove, c_tasktrove |
| `dolci` | c_dolci |
| `glaive` | c_glaive_v2_raw, glaive |
| `hermes` | hermes, hermes_reasoning |
| `mind2web` | c_mind2web_live, c_mm_mind2web, mind2web |
| `nemotron` | c_nemotron_rl_swe, c_nemotron_v1, c_nemotron_v2 |
| `nemotron_terminal` | nemotron_terminal |
| `neulab_aggregate` | c_neulab |
| `nnetnav` | c_nnetnav_wa, nnetnav |
| `openhermes` | openhermes |
| `openmanus` | c_openmanus |
| `openthoughts_agent` | c_ot_agent_rl, c_ot_agent_sft, c_ot_coldstart |
| `scalecua` | c_scalecua |
| `smoltalk2` | c_smoltalk, smoltalk2 |
| `swe_openhands` | c_r2egym, c_swefixer, c_swegym, c_swegym_raw, c_swegym_verifier, c_swehero, c_swezero12m, coderforge, swe_gym, swe_rebench, swe_smith, swe_zero |
| `taubench` | c_apigen_tau, c_taubench_synth |
| `toolace` | toolace |
| `toolbench` | c_toolenv2404 |
| `toolhop` | c_toolhop |
| `toolmind` | toolmind |
| `toucan` | c_toucan |
| `weblinx` | weblinx |
| `when2call` | c_when2call_pref, when2call |
| `xlam_apigen` | apigen, apigen_mt, c_synth_apigen, c_xlam60k |
