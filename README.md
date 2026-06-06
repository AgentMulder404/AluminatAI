<div align="center">

# ⚡ NemulAI

**Cut your GPU bill by 15-40%. Automatically.**

Self-learning GPU cost intelligence. Per-job attribution, waste detection, and automated optimization for AI teams.

[![PyPI version](https://badge.fury.io/py/nemulai.svg)](https://badge.fury.io/py/nemulai)
[![PyPI Downloads](https://static.pepy.tech/badge/nemulai)](https://pepy.tech/project/nemulai)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/nemulai)](https://pypi.org/project/nemulai)
[![GitHub Stars](https://img.shields.io/github/stars/AgentMulder404/NemulAI?style=social)](https://github.com/AgentMulder404/NemulAI/stargazers)

[Website](https://nemulai.com) · [Docs](https://nemulai.com/docs/agent) · [Dashboard](https://nemulai.com/dashboard) · [Report a Bug](https://github.com/AgentMulder404/NemulAI/issues)

</div>

---

## The Problem

Your A100 burns **$40/hour**. Do you know which of your training jobs was worth it?

Most ML teams can't answer that. `nvidia-smi` shows real-time watts. Cloud providers show a monthly bill. **Neither tells you which specific job was your $800 run.**

The hidden cost of GPU waste compounds fast:
- Training jobs left running overnight when convergence stalled hours ago
- Idle GPUs sitting at 3% utilization, eating full power draw
- No per-team attribution → no accountability → no improvement
- Finance asks "can we cut GPU spend?" and nobody has data to answer with

NemulAI closes that gap — and gets smarter over time. A lightweight Python agent runs on your GPU machines, attributes energy to individual jobs in real time, and streams dollar costs to a dashboard. The self-learning optimizer adapts to your workload patterns and improves its recommendations the longer it runs.

---

## Features

- **Per-job cost attribution** — tracks energy ($) per training run, not just per machine
- **Real-time power monitoring** — samples NVML every 5 seconds via `nvidia-ml-py`
- **Self-learning optimizer** — the agent learns your workload patterns over time; recommendations get better every week
- **Team chargeback** — tag workloads with `ALUMINATAI_TEAM` to split costs by team
- **Waste detection** — idle GPUs flagged automatically, saving 15-40% on compute spend
- **Budget alerts** — get notified before costs spike, not after
- **WAL-backed reliability** — metrics buffer locally during API outages, replay on reconnect
- **Multi-scheduler support** — Kubernetes, Slurm, Run:ai, and manual tagging
- **MLflow & W&B callbacks** — tag experiment runs with energy cost automatically
- **Prometheus endpoint** — expose metrics to your existing Grafana stack (`METRICS_PORT=9100`)
- **Zero infra overhead** — ~0% CPU, ~50 MB RAM, single pip install

---

## Quick Start

### Install

```bash
pip install nemulai
```

### Run

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
nemulai
```

That's it. The agent starts streaming GPU metrics to your dashboard immediately.

Get your API key at [nemulai.com/dashboard](https://nemulai.com/dashboard) — 7-day free trial, no credit card required.

### Docker

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e ALUMINATAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/nemulai-agent:latest
```

---

## Configuration

All settings are environment variables — no config files required.

| Variable | Default | Description |
|---|---|---|
| `ALUMINATAI_API_KEY` | *(required)* | Your API key from the dashboard |
| `ALUMINATAI_API_ENDPOINT` | `https://nemulai.com/v1/metrics/ingest` | Ingest endpoint |
| `SAMPLE_INTERVAL` | `5.0` | Seconds between NVML samples |
| `UPLOAD_INTERVAL` | `60` | Seconds between metric flushes |
| `ALUMINATAI_TEAM` | *(none)* | Team tag for chargeback attribution |
| `ALUMINATAI_MODEL` | *(none)* | Model tag for per-experiment tracking |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `METRICS_PORT` | `9100` | Prometheus scrape port (`0` = disabled) |

---

## Job Attribution

Tag your workloads at launch for per-job cost breakdown:

```bash
ALUMINATAI_TEAM=nlp-team \
ALUMINATAI_MODEL=llama3-finetune \
ALUMINATAI_API_KEY=alum_... \
python train.py
```

Or use the MLflow callback:

```python
from nemulai.integrations.mlflow_callback import NemulMLflowCallback

with mlflow.start_run():
    cb = NemulMLflowCallback()
    trainer.add_callback(cb)
```

Or W&B:

```python
from nemulai.integrations.wandb_callback import NemulWandbCallback

wandb.init(project="my-project")
trainer.add_callback(NemulWandbCallback())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   GPU Machine                           │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │  NVML/NVIDIA │────▶│   Sampler    │  5s interval    │
│  │  Driver      │     │  (nvidia-    │                 │
│  └──────────────┘     │   ml-py)     │                 │
│                       └──────┬───────┘                 │
│                              │                          │
│                       ┌──────▼───────┐                 │
│                       │  Attributor  │ job_id / team   │
│                       │  (process +  │ tagging         │
│                       │  scheduler)  │                 │
│                       └──────┬───────┘                 │
│                              │                          │
│                       ┌──────▼───────┐                 │
│                       │  WAL Buffer  │ survives         │
│                       │  (local)     │ outages          │
│                       └──────┬───────┘                 │
└──────────────────────────────┼──────────────────────────┘
                               │ HTTPS
                               ▼
                    ┌─────────────────────┐
                    │  nemulai.com    │
                    │  /v1/metrics/ingest │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │  Dashboard          │
                    │  watts → $ per job  │
                    │  team attribution   │
                    │  chargeback reports │
                    └─────────────────────┘
```

---

## Deployment

### systemd (recommended for production)

```ini
# /etc/systemd/system/nemulai.service
[Unit]
Description=NemulAI GPU Agent
After=network.target

[Service]
ExecStart=/usr/local/bin/nemulai
Restart=on-failure
RestartSec=10
EnvironmentFile=/etc/nemulai.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now nemulai
```

### Kubernetes DaemonSet

```bash
kubectl apply -f https://raw.githubusercontent.com/AgentMulder404/NemulAI/main/deploy/k8s/daemonset.yaml
```

### Slurm (Prolog/Epilog)

```bash
# /etc/slurm/prolog.d/nemulai.sh
source /etc/nemulai.env
nemulai &
```

Full deployment docs at [nemulai.com/docs/agent](https://nemulai.com/docs/agent).

---

## Why Open Source?

GPU cost optimization should be a solved problem, not a proprietary feature gate.

The GPU monitoring space is full of tools that show you _what's happening_ (`nvidia-smi`, Grafana) or _what happened_ (cloud billing dashboards). NemulAI is the missing link: **what each specific job cost, in real time, in dollars.**

By open-sourcing the agent, anyone can:
- Audit exactly what data is collected (it's just power draw and metadata you tag)
- Run a fully self-hosted stack against your own endpoint
- Contribute integrations for their scheduler, experiment tracker, or cloud provider
- Build on the primitives for their own cost tooling

The hosted dashboard at [nemulai.com](https://nemulai.com) is how the project is sustained. The agent that collects your data will always be free and open.

---

## Self-Hosting

The agent is fully functional without the hosted dashboard. Point it at your own ingest endpoint:

```bash
ALUMINATAI_API_ENDPOINT=https://your-internal-api.com/v1/metrics/ingest \
ALUMINATAI_API_KEY=your_key \
nemulai
```

The ingest API schema is documented at [nemulai.com/docs/api](https://nemulai.com/docs/api).

---

## Contributing

Contributions are welcome. The project follows a standard fork → branch → PR workflow.

**Good first issues:** scheduler integrations, new MLflow/W&B/OTEL hooks, packaging improvements, docs.

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make your changes with tests where applicable
4. Open a PR against `main`

By contributing, you agree your code will be licensed under Apache 2.0 and credited in the NOTICE file.

**Code of conduct:** Be direct, be useful, don't be a jerk.

---

## Citation

If you use NemulAI in research, please cite:

```bibtex
@software{nemulai2026,
  author    = {Kevin},
  title     = {NemulAI: Per-Job GPU Energy Monitoring and Cost Attribution},
  year      = {2026},
  url       = {https://github.com/AgentMulder404/NemulAI},
  version   = {0.2.1}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable format.

---

## Authorship & Credit

NemulAI was created and is maintained by **Kevin**.

- X/Twitter: [@NemulAI_Dev](https://x.com/NemulAI_Dev)
- Website: [nemulai.com](https://nemulai.com)
- GitHub: [@AgentMulder404](https://github.com/AgentMulder404)

Copyright © 2026 Kevin (NemulAI). All rights reserved.

The name **"NemulAI"** is a trademark of the original author. Forks and derivative works are welcome under the Apache 2.0 license, but may not use the NemulAI name or logo to represent their products without written permission.

If you build something with or on top of NemulAI, a mention or link back is appreciated — it helps others find the original project.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

In plain English: use it, fork it, build on it, sell products with it. Keep the copyright notice, don't call your fork "NemulAI", and don't claim you wrote it.

---

<div align="center">
  <sub>Built with obsession by Kevin · Star ⭐ if this saves you money</sub>
</div>
