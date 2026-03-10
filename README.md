<div align="center">

# ⚡ AluminatiAI

**nvidia-smi shows you watts. AluminatiAI shows you dollars.**

Per-job GPU energy monitoring, cost attribution, and waste detection for AI teams.

[![PyPI version](https://badge.fury.io/py/aluminatiai.svg)](https://badge.fury.io/py/aluminatiai)
[![PyPI Downloads](https://static.pepy.tech/badge/aluminatiai)](https://pepy.tech/project/aluminatiai)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/aluminatiai)](https://pypi.org/project/aluminatiai)
[![GitHub Stars](https://img.shields.io/github/stars/AgentMulder404/AluminatAI?style=social)](https://github.com/AgentMulder404/AluminatAI/stargazers)

[Website](https://aluminatiai.com) · [Docs](https://aluminatiai.com/docs/agent) · [Dashboard](https://aluminatiai.com/dashboard) · [Report a Bug](https://github.com/AgentMulder404/AluminatAI/issues)

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

AluminatiAI closes that gap. A lightweight Python agent runs on your GPU machines, attributes energy to individual jobs in real time, and streams dollar costs to a dashboard — so you know what everything costs before the cloud bill arrives.

---

## Features

- **Per-job cost attribution** — tracks energy ($) per training run, not just per machine
- **Real-time power monitoring** — samples NVML every 5 seconds via `nvidia-ml-py`
- **Team chargeback** — tag workloads with `ALUMINATAI_TEAM` to split costs by team
- **Utilization & efficiency metrics** — throughput-per-watt, idle detection, utilization trends
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
pip install aluminatiai
```

### Run

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
aluminatiai
```

That's it. The agent starts streaming GPU metrics to your dashboard immediately.

Get your API key at [aluminatiai.com/dashboard](https://aluminatiai.com/dashboard) — 7-day free trial, no credit card required.

### Docker

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e ALUMINATAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/aluminatai-agent:latest
```

---

## Configuration

All settings are environment variables — no config files required.

| Variable | Default | Description |
|---|---|---|
| `ALUMINATAI_API_KEY` | *(required)* | Your API key from the dashboard |
| `ALUMINATAI_API_ENDPOINT` | `https://aluminatiai.com/v1/metrics/ingest` | Ingest endpoint |
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
from aluminatiai.integrations.mlflow_callback import AluminatiMLflowCallback

with mlflow.start_run():
    cb = AluminatiMLflowCallback()
    trainer.add_callback(cb)
```

Or W&B:

```python
from aluminatiai.integrations.wandb_callback import AluminatiWandbCallback

wandb.init(project="my-project")
trainer.add_callback(AluminatiWandbCallback())
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
                    │  aluminatiai.com    │
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
# /etc/systemd/system/aluminatiai.service
[Unit]
Description=AluminatiAI GPU Agent
After=network.target

[Service]
ExecStart=/usr/local/bin/aluminatiai
Restart=on-failure
RestartSec=10
EnvironmentFile=/etc/aluminatiai.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now aluminatiai
```

### Kubernetes DaemonSet

```bash
kubectl apply -f https://raw.githubusercontent.com/AgentMulder404/AluminatAI/main/deploy/k8s/daemonset.yaml
```

### Slurm (Prolog/Epilog)

```bash
# /etc/slurm/prolog.d/aluminatiai.sh
source /etc/aluminatiai.env
aluminatiai &
```

Full deployment docs at [aluminatiai.com/docs/agent](https://aluminatiai.com/docs/agent).

---

## Why Open Source?

GPU cost visibility should be a solved problem, not a proprietary feature gate.

The GPU monitoring space is full of tools that show you _what's happening_ (`nvidia-smi`, Grafana) or _what happened_ (cloud billing dashboards). AluminatiAI is the missing link: **what each specific job cost, in real time, in dollars.**

By open-sourcing the agent, anyone can:
- Audit exactly what data is collected (it's just power draw and metadata you tag)
- Run a fully self-hosted stack against your own endpoint
- Contribute integrations for their scheduler, experiment tracker, or cloud provider
- Build on the primitives for their own cost tooling

The hosted dashboard at [aluminatiai.com](https://aluminatiai.com) is how the project is sustained. The agent that collects your data will always be free and open.

---

## Self-Hosting

The agent is fully functional without the hosted dashboard. Point it at your own ingest endpoint:

```bash
ALUMINATAI_API_ENDPOINT=https://your-internal-api.com/v1/metrics/ingest \
ALUMINATAI_API_KEY=your_key \
aluminatiai
```

The ingest API schema is documented at [aluminatiai.com/docs/api](https://aluminatiai.com/docs/api).

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

If you use AluminatiAI in research, please cite:

```bibtex
@software{aluminatiai2026,
  author    = {Kevin},
  title     = {AluminatiAI: Per-Job GPU Energy Monitoring and Cost Attribution},
  year      = {2026},
  url       = {https://github.com/AgentMulder404/AluminatAI},
  version   = {0.2.1}
}
```

See [CITATION.cff](CITATION.cff) for the machine-readable format.

---

## Authorship & Credit

AluminatiAI was created and is maintained by **Kevin**.

- X/Twitter: [@AluminatiAi_Dev](https://x.com/AluminatiAi_Dev)
- Website: [aluminatiai.com](https://aluminatiai.com)
- GitHub: [@AgentMulder404](https://github.com/AgentMulder404)

Copyright © 2026 Kevin (AluminatiAI). All rights reserved.

The name **"AluminatiAI"** is a trademark of the original author. Forks and derivative works are welcome under the Apache 2.0 license, but may not use the AluminatiAI name or logo to represent their products without written permission.

If you build something with or on top of AluminatiAI, a mention or link back is appreciated — it helps others find the original project.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

In plain English: use it, fork it, build on it, sell products with it. Keep the copyright notice, don't call your fork "AluminatiAI", and don't claim you wrote it.

---

<div align="center">
  <sub>Built with obsession by Kevin · Star ⭐ if this saves you money</sub>
</div>
