---
name: GreenTune dataset strategy
description: Two-dataset approach for AMD hackathon fine-tuning — Hermes agent traces as base + synthetic GreenTune domain data mixed in
type: project
---

Use a two-dataset blend for GreenTune fine-tuning:

1. **Base: AdversaLLC/hermes-agent-reasoning-traces (config: "glm-5.1")** — 7,055 samples, ShareGPT format, multi-turn agent reasoning with `<think>` blocks and tool calls. Categories: Agent Tools (2775), Terminal & Coding (2237), Repository Tasks (1022), Browser Automation (639), File Operations (134), Scheduling (104), Planning (92), Multi-Tool (52). Avg 19 turns per conversation. Teaches the model how to reason as an agent and use tools.

2. **Domain: Synthetic GreenTune dataset (~2-3K samples)** — generated via `dataset_builder.py` using Claude. Alpaca format. 7 categories: GPU power diagnosis, energy efficiency, ROCm ops, cost attribution, fine-tuning ops, workload scheduling, AluminatiAI product. Teaches GPU/energy domain knowledge.

**Why:** The Hermes traces give agent reasoning patterns; the synthetic data gives domain specialization. Combined, the model learns *how to think step-by-step* and *what to know about GPU energy*.

**How to apply:** `greentune.py` needs to support both ShareGPT and Alpaca formats, merge them, and shuffle before training. The Hermes dataset uses `from: system/human/gpt/tool` turn structure.
