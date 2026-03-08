# AluminatiAI Agent

The open-source GPU monitoring agent for [AluminatiAI](https://aluminatiai.com).

Runs on your GPU machines, samples NVML every 5 seconds, attributes energy to individual jobs, and streams dollar costs to your dashboard.

## Install

```bash
pip install aluminatiai
```

## Run

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
aluminatiai
```

Get your API key at [aluminatiai.com/dashboard](https://aluminatiai.com/dashboard).

## Docker

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e ALUMINATAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/aluminatai-agent:latest
```

## Configuration

All settings are environment variables.

| Variable | Default | Description |
|---|---|---|
| `ALUMINATAI_API_KEY` | *(required)* | Your API key |
| `ALUMINATAI_API_ENDPOINT` | `https://aluminatiai.com/v1/metrics/ingest` | Ingest endpoint |
| `SAMPLE_INTERVAL` | `5.0` | Seconds between NVML samples |
| `UPLOAD_INTERVAL` | `60` | Seconds between metric flushes |
| `ALUMINATAI_TEAM` | *(none)* | Team tag for chargeback |
| `ALUMINATAI_MODEL` | *(none)* | Model tag for experiment tracking |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `METRICS_PORT` | `9100` | Prometheus scrape port (`0` = disabled) |

## Package structure

```
aluminatiai/          (this directory — installed as the aluminatiai package)
├── agent.py          # Daemon entry point / CLI
├── cli.py            # aluminatiai command
├── config.py         # Env-based config
├── collector.py      # NVML sampler (nvidia-ml-py)
├── uploader.py       # HTTPS upload + WAL retry
├── wal.py            # Write-ahead log for offline buffering
├── metrics_server.py # Prometheus /metrics endpoint
├── process_probe.py  # PID → GPU job attribution
├── attribution/      # Scheduler integrations (K8s, Slurm, Run:ai)
├── integrations/     # MLflow, W&B, OTEL callbacks
├── schedulers/       # Minimax scheduler
├── efficiency/       # Throughput-per-watt analysis
├── report/           # Chargeback report generation
└── tests/            # Unit tests + Colab A100 test notebook
```

## Development

```bash
git clone https://github.com/AgentMulder404/AluminatAI.git
cd AluminatAI/agent
pip install -e ".[prometheus]"
python -m pytest tests/
```

## Self-hosting

Point the agent at your own ingest endpoint:

```bash
ALUMINATAI_API_ENDPOINT=https://your-api.internal/v1/metrics/ingest \
ALUMINATAI_API_KEY=your_key \
aluminatiai
```

## License

Apache 2.0 — see [LICENSE](../LICENSE).
