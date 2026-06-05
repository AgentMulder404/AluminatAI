# NemulAI — H100 SXM Model Test Walkthrough

Full product test: baseline run → NemulAI monitoring + optimization → show improvement.

## Pod Info

```
SSH: ssh root@64.247.201.59 -p 19795 -i ~/.ssh/runpod_nemulai
GPU: NVIDIA H100 80GB HBM3 (700W TDP)
```

---

## Step 0: Install Everything (one time)

SSH in and run:

```bash
# Core deps
pip install nvidia-ml-py rich requests python-dotenv numpy
pip install huggingface_hub transformers accelerate
pip install vllm

# Clone NemulAI and install the agent
cd /workspace
git clone https://github.com/AgentMulder404/NemulAI.git
cd NemulAI/agent
pip install -e ".[all]"

# Verify
nvidia-smi
python3 -c "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetName(h))"
python3 -c "from vllm import LLM; print('vLLM ready')"
python3 -m cli --help
```

---

## Step 1: Download First Test Model

Start with one model to prove the workflow, then scale to all 12.

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

---

## Step 2: BASELINE Run (No NemulAI)

This is the "before" — raw inference, no monitoring, no optimization.

### Terminal 2 — Run the model

```bash
cd /workspace/NemulAI/agent

# Quick baseline: just run vLLM inference and time it manually
python3 -c "
import time, json
from vllm import LLM, SamplingParams

print('Loading Qwen2.5-7B (baseline, no monitoring)...')
llm = LLM(model='Qwen/Qwen2.5-7B-Instruct', dtype='bfloat16', max_model_len=4096)

prompts = [
    'Explain how GPU power consumption relates to compute utilization.',
    'Write a Python function that monitors GPU temperature using pynvml.',
    'Compare the energy efficiency of H100 vs A100 for LLM inference.',
    'What are five ways to reduce the carbon footprint of training LLMs?',
    'Design a monitoring alert system for GPU temperature and power.',
    'Write a REST API endpoint that accepts GPU telemetry data.',
    'Explain the roofline model in high-performance computing.',
    'A company runs 100 req/s on 4 H100s. Calculate monthly energy cost.',
    'What is the difference between FP16, BF16, and FP8 precision?',
    'Write a Dockerfile for a vLLM inference service with CUDA 12.4.',
]

tok = llm.get_tokenizer()
formatted = [tok.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True) for p in prompts]

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

t0 = time.time()
outputs = llm.generate(formatted, params)
elapsed = time.time() - t0

total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
tok_s = total_out / elapsed

print()
print('═' * 60)
print('  BASELINE (no NemulAI)')
print('═' * 60)
print(f'  Model:       Qwen2.5-7B-Instruct')
print(f'  Prompts:     {len(prompts)}')
print(f'  Output toks: {total_out}')
print(f'  Time:        {elapsed:.1f}s')
print(f'  Throughput:  {tok_s:.1f} tok/s')
print(f'  Power:       (unknown — no monitoring)')
print(f'  J/token:     (unknown — no monitoring)')
print(f'  Cost/1M tok: (unknown — no monitoring)')
print('═' * 60)
print()
print('This is what running blind looks like.')
print('You have throughput, but ZERO visibility into power, cost, or efficiency.')
"
```

Write down the throughput number. That's all you get without NemulAI.

---

## Step 3: WITH NemulAI — Agent + Monitoring

Now open **two terminals**.

### Terminal 1 — Start NemulAI Agent

```bash
cd /workspace/NemulAI/agent

# Start the agent daemon in the foreground so you can see live output
ALUMINATAI_TEAM=ml-platform \
ALUMINATAI_MODEL=qwen2.5-7b \
ALUMINATAI_LOG_LEVEL=INFO \
ALUMINATAI_SAMPLE_INTERVAL=2 \
python3 -m cli run
```

You'll see live power samples streaming:
```
[INFO] NemulAI agent started — GPU 0: NVIDIA H100 80GB HBM3
[INFO] Sample: 72.3W | util 0% | temp 31°C | mem 0.1/80.0 GB
[INFO] Sample: 71.8W | util 0% | temp 31°C | mem 0.1/80.0 GB
...
```

Leave this running. When the model loads and runs, you'll see power spike to 400-650W.

### Terminal 2 — Run the Same Model (now monitored)

```bash
cd /workspace/NemulAI/agent

# Run the same inference, but now add NemulAI's benchmark wrapper
python3 -m cli benchmark --gpu 0 --duration 60 --model-tag qwen2.5-7b --framework vllm &

# While benchmark samples power, run the actual inference
python3 -c "
import time
from vllm import LLM, SamplingParams

print('Loading Qwen2.5-7B (WITH NemulAI monitoring)...')
llm = LLM(model='Qwen/Qwen2.5-7B-Instruct', dtype='bfloat16', max_model_len=4096)

prompts = [
    'Explain how GPU power consumption relates to compute utilization.',
    'Write a Python function that monitors GPU temperature using pynvml.',
    'Compare the energy efficiency of H100 vs A100 for LLM inference.',
    'What are five ways to reduce the carbon footprint of training LLMs?',
    'Design a monitoring alert system for GPU temperature and power.',
    'Write a REST API endpoint that accepts GPU telemetry data.',
    'Explain the roofline model in high-performance computing.',
    'A company runs 100 req/s on 4 H100s. Calculate monthly energy cost.',
    'What is the difference between FP16, BF16, and FP8 precision?',
    'Write a Dockerfile for a vLLM inference service with CUDA 12.4.',
]

tok = llm.get_tokenizer()
formatted = [tok.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True) for p in prompts]
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

t0 = time.time()
outputs = llm.generate(formatted, params)
elapsed = time.time() - t0

total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
tok_s = total_out / elapsed
print(f'Throughput: {tok_s:.1f} tok/s in {elapsed:.1f}s')
"
```

**Terminal 1** will now show the agent catching the workload:
```
[INFO] Sample: 623.4W | util 94% | temp 68°C | mem 18.2/80.0 GB
[INFO] Process detected: python3 (PID 1234) — team=ml-platform model=qwen2.5-7b
[INFO] Attribution: 551.6W dynamic (623.4 - 71.8 idle)
```

Now run the **optimize** command to get recommendations:

```bash
python3 -m cli optimize --gpu 0 --duration 30
```

This will output something like:
```
═══════════════════════════════════════════════════
  NemulAI Efficiency Analysis — H100 SXM
═══════════════════════════════════════════════════
  Regime: memory-bound
  Arithmetic Intensity: 85.3 FLOP/byte (ridge: 295.2)

  [P1] Enable Transformer Engine FP8 autocast
       H100 has dedicated TE hardware — 2x throughput, same power
  [P2] Set power cap to 550W
       Memory-bound workload doesn't need full 700W TDP
  [P3] Increase batch size
       GPU util at 78% — larger batches amortize idle power
═══════════════════════════════════════════════════
```

---

## Step 4: APPLY OPTIMIZATIONS — Power Cap

Apply the #1 recommendation that doesn't require code changes: **power capping**.

The H100 runs at 700W TDP but a memory-bound 7B model doesn't need it.
NemulAI can cap the GPU to ~500-550W and maintain the same throughput.

### Terminal 2 — A/B Test: Baseline vs Power-Capped

This is the money shot. NemulAI's `ab` tool runs both configs back-to-back with statistical significance testing.

```bash
cd /workspace/NemulAI/agent

# A/B test: 700W (default) vs 500W power cap
# Uses the same workload command for both phases
python3 -m cli ab \
  --powercap \
  --baseline-watts 700 \
  --optimized-watts 500 \
  --workload "python3 -c \"
import torch, time
device = torch.device('cuda:0')
A = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
B = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
end = time.time() + 30
count = 0
while time.time() < end:
    _ = A @ B
    torch.cuda.synchronize()
    count += 1
tflops = (count * 2 * 8192**3) / 30 / 1e12
print(f'{tflops:.1f} tok/s')
\"" \
  --duration 30 \
  --iterations 3 \
  --gpu 0
```

Expected output:
```
═══════════════════════════════════════════════════════
  NemulAI A/B Experiment — Power Cap Optimization
═══════════════════════════════════════════════════════
  GPU: NVIDIA H100 80GB HBM3

  BASELINE (700W)
    Energy:  18,432 ± 312 J
    Power:   614.4 ± 10.4 W
    Throughput: 892.1 TFLOPS

  OPTIMIZED (500W)
    Energy:  14,128 ± 287 J
    Power:   470.9 ± 8.2 W
    Throughput: 879.3 TFLOPS

  RESULT
    Energy savings:     23.4%  ← statistically significant (p < 0.05)
    Throughput change:  -1.4%  ← NOT significant
    Cost savings:       $0.42/hr
    CO2 savings:        58.2 gCO2e/hr

  Recommendation: ADOPT — 23% energy reduction with negligible throughput loss
═══════════════════════════════════════════════════════
```

---

## Step 5: A/B Test with Real LLM Inference

Now do the same A/B but with actual vLLM model inference:

```bash
python3 -m cli ab \
  --powercap \
  --baseline-watts 700 \
  --optimized-watts 500 \
  --workload "python3 -c \"
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-7B-Instruct', dtype='bfloat16', max_model_len=4096)
tok = llm.get_tokenizer()
prompts = ['Explain GPU power management.', 'Write a Python async web server.', 'Compare TCP and UDP.', 'What is the roofline model?', 'Design a REST API for metrics.']
formatted = [tok.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True) for p in prompts]
outputs = llm.generate(formatted, SamplingParams(temperature=0.7, max_tokens=512))
total = sum(len(o.outputs[0].token_ids) for o in outputs)
print(f'{total} tok/s')
\"" \
  --duration 60 \
  --iterations 3 \
  --gpu 0
```

---

## Step 6: Scale to All Models

Once the workflow is proven with Qwen2.5-7B, test each model family.
Run the same pattern for each — the agent stays running in Terminal 1.

```bash
# Download all models (run in background)
models=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-8B"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "teknium/OpenHermes-2.5-Mistral-7B"
  "NousResearch/Hermes-3-Llama-3.1-8B"
  "google/gemma-2-2b-it"
  "google/gemma-2-9b-it"
)
for m in "${models[@]}"; do
  huggingface-cli download "$m" &
done
wait
```

For each model, repeat:

```bash
# 1. Run with monitoring (Terminal 1 agent catches it)
python3 -m cli benchmark --gpu 0 --duration 60 --model-tag MODEL_NAME --framework vllm

# 2. Get optimization recommendations
python3 -m cli optimize --gpu 0 --duration 30

# 3. A/B test power cap
python3 -m cli ab --powercap --baseline-watts 700 --optimized-watts RECOMMENDED_WATTS \
  --workload "INFERENCE_COMMAND" --duration 30 --iterations 3
```

Or use the automated script:
```bash
python3 /workspace/h100_customer_sim.py --models all --prompts 25
```

---

## What You're Proving

| Without NemulAI | With NemulAI |
|---|---|
| Know throughput only | Know power, energy, cost, CO2 per token |
| No idea if GPU is efficient | Roofline analysis tells you memory-bound vs compute-bound |
| Running at 700W because default | Power capped to 500W — same throughput, 23% less energy |
| Can't compare models on cost | J/token and $/1M tokens for every model side by side |
| No team attribution | Every watt attributed to team + model tag |
| Guessing at optimization | Data-driven P1/P2/P3 recommendations |

---

## Key Power Cap Values to Test per Model Size

| Model Size | Baseline TDP | Recommended Cap | Why |
|---|---|---|---|
| 0.5-0.6B | 700W | 300-350W | Tiny model barely uses compute — massive savings |
| 1.5-2B | 700W | 350-400W | Still memory-bound at this size |
| 4B | 700W | 400-450W | Starting to use more compute |
| 7-9B | 700W | 450-550W | More compute-hungry but still memory-bound on H100 |

The `optimize` command will tell you the exact recommended cap for each model.
