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

## Deployment

### One-line installer (Linux + systemd)

The fastest path to production — installs from PyPI, creates a dedicated system user, writes `/etc/aluminatai/agent.env`, and registers a hardened systemd service:

```bash
curl -sSL https://get.aluminatiai.com | bash
```

Options:

| Flag | Effect |
|------|--------|
| `--local` | Install from local source (dev / air-gapped) |
| `--no-service` | Package only — skip systemd setup |
| `--unattended` / `-y` | Non-interactive; requires `ALUMINATAI_API_KEY` env var |

```bash
# CI / non-interactive
ALUMINATAI_API_KEY=alum_xxx curl -sSL https://get.aluminatiai.com | bash -s -- --unattended

# Already have the package; just set up the service
sudo aluminatiai service install

# Check service health
aluminatiai service status

# Remove the service (keeps data and config)
sudo aluminatiai service uninstall
```

### Manual systemd setup

If you prefer to manage the service yourself:

```bash
pip install aluminatiai

# Create system user and directories
sudo useradd --system --no-create-home --shell /usr/sbin/nologin aluminatai
sudo install -d -m 0700 -o aluminatai -g aluminatai /var/lib/aluminatai
sudo install -d -m 0755 -o aluminatai -g aluminatai /var/log/aluminatai
sudo install -d -m 0750 /etc/aluminatai

# Write the env file (mode 600 — contains your API key)
sudo tee /etc/aluminatai/agent.env > /dev/null <<'EOF'
ALUMINATAI_API_KEY=alum_your_key_here
ALUMINATAI_API_ENDPOINT=https://aluminatiai.com/api/metrics/ingest
SAMPLE_INTERVAL=5.0
UPLOAD_INTERVAL=60
METRICS_PORT=9100
LOG_LEVEL=INFO
EOF
sudo chmod 600 /etc/aluminatai/agent.env

# Install the unit file
sudo cp deploy/aluminatai-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aluminatai-agent

# Verify
sudo systemctl status aluminatai-agent
sudo journalctl -u aluminatai-agent -f
```

The unit file is at `deploy/aluminatai-agent.service` in this repo and includes systemd security hardening (`NoNewPrivileges`, `ProtectSystem=strict`, `PrivateTmp`, system call filtering).

### Useful service commands

```bash
sudo systemctl status aluminatai-agent     # health
sudo journalctl -u aluminatai-agent -f     # live logs
sudo journalctl -u aluminatai-agent --since "1 hour ago"
curl -s localhost:9100/metrics | grep aluminatai  # Prometheus

# Edit config and apply
sudo nano /etc/aluminatai/agent.env
sudo systemctl restart aluminatai-agent
```

### Docker

```bash
docker run --rm --runtime=nvidia --pid=host \
  -e ALUMINATAI_API_KEY=alum_your_key_here \
  ghcr.io/agentmulder404/aluminatai-agent:latest
```

## Configuration

Settings are read in this priority order (highest wins):

1. **Environment variables** (always override everything)
2. **Config file** — JSON or YAML (path via `--config` flag or `ALUMINATAI_CONFIG` env var)
3. **Built-in defaults**

### Config file

```bash
aluminatiai --config /etc/aluminatai.json
# or
ALUMINATAI_CONFIG=/etc/aluminatai.yaml aluminatiai
```

Example `aluminatai.json`:
```json
{
  "api_key": "alum_your_key_here",
  "sample_interval": 2.0,
  "upload_interval": 30,
  "upload_batch_size": 50,
  "metrics_port": 9100,
  "metrics_bind_host": "127.0.0.1",
  "wal_max_mb": 256,
  "log_level": "INFO",
  "log_format": "json"
}
```

YAML config requires `pip install 'aluminatiai[observability]'`.

### All configuration options

| Key (config file) | Env var | Default | Description |
|---|---|---|---|
| `api_key` | `ALUMINATAI_API_KEY` | *(required)* | Your API key |
| `api_endpoint` | `ALUMINATAI_API_ENDPOINT` | `https://…/v1/metrics/ingest` | Ingest endpoint |
| `sample_interval` | `SAMPLE_INTERVAL` | `5.0` | Seconds between NVML samples |
| `upload_interval` | `UPLOAD_INTERVAL` | `60` | Seconds between metric flushes |
| `upload_batch_size` | `UPLOAD_BATCH_SIZE` | `100` | Metrics per HTTP request |
| `upload_max_retries` | `UPLOAD_MAX_RETRIES` | `5` | Max upload retry attempts |
| `heartbeat_interval` | `HEARTBEAT_INTERVAL` | `300` | Seconds between heartbeat pings |
| `scheduler_poll_interval` | `SCHEDULER_POLL_INTERVAL` | `30` | Scheduler poll interval |
| `wal_max_mb` | `WAL_MAX_MB` | `512` | WAL size cap in MB |
| `wal_max_age_hours` | `WAL_MAX_AGE_HOURS` | `24` | WAL TTL in hours |
| `metrics_port` | `METRICS_PORT` | `9100` | Prometheus scrape port (`0` = disabled) |
| `metrics_bind_host` | `METRICS_BIND_HOST` | *(all)* | Bind address for Prometheus |
| `metrics_basic_auth` | `METRICS_BASIC_AUTH` | *(none)* | `user:pass` for Prometheus auth |
| `log_level` | `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `log_format` | `LOG_FORMAT` | `text` | `text` or `json` |
| `dry_run` | `DRY_RUN` | `false` | Collect+attribute but skip uploads |
| `prometheus_only` | `PROMETHEUS_ONLY` | `false` | Local Prometheus only, no cloud |
| `offline_mode` | `OFFLINE_MODE` | `false` | WAL-only, no HTTP uploads |
| `trusted_uids` | `ALUMINATAI_TRUSTED_UIDS` | *(all)* | Comma-separated UIDs for spoofing protection |

### Run modes

| Mode | Flag | Effect |
|---|---|---|
| Normal | *(default)* | Collect → attribute → Prometheus + cloud upload |
| Dry run | `--dry-run` | Collect → attribute → log only (no upload, no WAL) |
| Prometheus only | `--prometheus-only` | Collect → attribute → local Prometheus only |
| Offline | `OFFLINE_MODE=1` | Collect → attribute → WAL only |

```bash
# Test attribution config without sending any data
aluminatiai --dry-run --log-format json

# Air-gapped / firewall — export WAL later
OFFLINE_MODE=1 aluminatiai
aluminatiai replay --output /tmp/metrics.csv --clear

# Prometheus only (no cloud account needed)
aluminatiai --prometheus-only
```

## Structured logging

By default the agent emits human-readable log lines to stderr. Switch to
newline-delimited JSON for log aggregation pipelines (ELK, Grafana Loki, etc.):

```bash
# Via flag
aluminatiai --log-format json

# Via env var
LOG_FORMAT=json aluminatiai

# Via config file
echo '{"log_format": "json", "log_level": "DEBUG"}' > aluminatai.json
aluminatiai --config aluminatai.json
```

Example JSON log line:
```json
{"ts": "2026-03-07T10:00:00+00:00", "level": "INFO", "logger": "aluminatai-agent", "msg": "Uploaded 42 metrics"}
```

## Package structure

```
aluminatiai/          (this directory — installed as the aluminatiai package)
├── agent.py          # Daemon entry point / CLI
├── cli.py            # aluminatiai command
├── config.py         # Env-based config + config file loader
├── collector.py      # NVML sampler (nvidia-ml-py)
├── uploader.py       # HTTPS upload + WAL retry + encryption
├── metrics_server.py # Prometheus /metrics endpoint
├── attribution/      # Scheduler integrations (K8s, Slurm, Run:ai)
├── integrations/     # MLflow, W&B, OTEL callbacks
├── schedulers/       # Scheduler detection + adapters
├── efficiency/       # Throughput-per-watt analysis
├── deploy/           # Deployment files
│   ├── aluminatai-agent.service    # systemd unit (copy to /etc/systemd/system/)
│   └── aluminatai-agent.env.example  # annotated env file template
├── install.sh        # One-line installer (curl | bash)
└── tests/            # Unit tests + Colab A100 test notebook
```

## Development

```bash
git clone https://github.com/AgentMulder404/AluminatAI.git
cd AluminatAI/agent
pip install -e ".[prometheus]"
python -m pytest tests/
```

## Attribution

The agent attributes GPU power to jobs using a 6-step resolution pipeline. The first step that matches wins.

### Resolution pipeline

| Step | Method | Confidence | Score |
|------|--------|------------|-------|
| 1 | `SLURM_JOB_ID` in process env → Slurm adapter | `scheduler` | 0.9 |
| 2 | `RUNAI_JOB_NAME` in process env → Run:ai adapter | `scheduler` | 0.9 |
| 3 | `KUBERNETES_SERVICE_HOST` + cgroup pod UID → K8s adapter | `scheduler` | 0.9 |
| 4 | `ALUMINATAI_TEAM` env var (manual tag) | `tagged` | 1.0 |
| 5 | Custom rules JSON file (operator-defined patterns) | `rules` | 0.6 |
| 6 | Built-in cmdline heuristics (jupyter, vllm, torchserve, …) | `heuristic` | 0.4 |
| — | Unresolved: power split proportionally by GPU memory usage | `memory_split` | 0.2 |
| — | Scheduler poll fallback (`gpu_to_job`) | `scheduler_poll` | 0.7 |
| — | Idle GPU (no processes, `ALUMINATAI_IDLE_TEAM` set) | `idle` | 0.1 |

Confidence scores are exposed via the `aluminatai_attribution_confidence{method, gpu_uuid}` Prometheus gauge.

### Custom attribution rules

Drop an `attribution_rules.json` file to map cmdline patterns to teams:

```json
{
  "rules": [
    { "pattern": "python.*gpt4_train", "team": "llm-infra", "model": "gpt4",     "priority": 10 },
    { "pattern": "vllm.*llama",        "team": "inference",  "model": "llama",    "priority": 5  },
    { "pattern": "jupyter",            "team": "research",   "model": "notebook", "priority": 1  }
  ]
}
```

The agent searches for the config file in order:

1. `ALUMINATAI_ATTRIBUTION_CONFIG` env var (explicit path)
2. `./attribution_rules.json` (working directory)
3. `~/.config/aluminatai/attribution_rules.json`

If no file is found, rules are silently disabled — no behaviour change.

### Env var spoofing protection

On multi-user hosts, set `ALUMINATAI_TRUSTED_UIDS` to a comma-separated list of UIDs
that are allowed to use the `ALUMINATAI_TEAM` manual tag. Processes from other UIDs
claiming the tag will be silently ignored and fall through to rules/heuristics.

```bash
export ALUMINATAI_TRUSTED_UIDS=0,1000   # only root and UID 1000 may self-tag
```

When `ALUMINATAI_TRUSTED_UIDS` is unset (the default), all UIDs are trusted —
existing single-tenant behaviour is preserved.

### Limitations

> **Attribution is best-effort.** Untagged processes are proportionally split by GPU
> memory usage, which may dilute accuracy when multiple unrelated jobs share a GPU
> without MIG isolation. For highest accuracy, tag workloads with `ALUMINATAI_TEAM`
> at launch or use a supported scheduler (Slurm, Kubernetes, Run:ai).

- **Non-Linux** (macOS, Windows, WSL without `/proc`): environ, cmdline, and owner-UID
  reads are silently skipped. Attribution falls back to scheduler-poll or idle only.
- **Process inheritance**: the agent walks up to 3 parent PIDs looking for
  `ALUMINATAI_TEAM`. Deeper process trees (depth > 3) will not inherit tags.
- **MIG**: when MIG slices share a physical GPU index, memory-fraction splitting may
  be inaccurate. Use per-slice NVML handles or a scheduler tag for precision.

## Security

### Environment variable privacy

The agent reads `/proc/<pid>/environ` to attribute GPU jobs to teams. Only a small allowlist of env var keys is retained in memory — all others (including `AWS_SECRET_ACCESS_KEY`, database URLs, tokens, etc.) are dropped immediately after reading:

```
SLURM_JOB_ID, RUNAI_JOB_NAME, KUBERNETES_SERVICE_HOST,
ALUMINATAI_TEAM, ALUMINATAI_MODEL, ALUMINATAI_* (any prefix)
```

**Best practice**: do not pass secrets as env vars to GPU jobs. Use a secrets manager (Vault, AWS Secrets Manager) or mounted secret files instead.

### WAL encryption

The write-ahead log (`data/wal/metrics.wal`) is encrypted automatically when `ALUMINATAI_API_KEY` is set and the `cryptography` package is installed. The encryption key is derived from your API key via SHA-256 (no separate secret needed).

```bash
pip install 'aluminatiai[secure]'
```

If `cryptography` is not installed, the agent logs a one-time warning and falls back to plaintext WAL.

### Prometheus endpoint hardening

By default, the Prometheus `/metrics` endpoint binds to all interfaces (`0.0.0.0:9100`) with no authentication. For production clusters, restrict access:

```bash
# Bind to localhost only (use a TLS-terminating proxy for remote scraping)
export METRICS_BIND_HOST=127.0.0.1

# Optionally require HTTP Basic Auth
export METRICS_BASIC_AUTH=scrape_user:strong_password
```

| Variable | Default | Effect |
|---|---|---|
| `METRICS_BIND_HOST` | `""` (all interfaces) | IP address to bind the Prometheus server |
| `METRICS_BASIC_AUTH` | `""` (no auth) | `user:pass` — enable HTTP Basic Auth |

> **Note**: Basic Auth over plain HTTP is susceptible to eavesdropping. Use a TLS-terminating reverse proxy (nginx, Caddy) in front of the metrics endpoint in production.

### Offline / air-gapped clusters

If the agent cannot reach the AluminatiAI API (air-gapped network, firewall), enable offline mode. All metrics are written to the WAL only — no outbound HTTP requests are made.

```bash
export OFFLINE_MODE=1
aluminatiai
```

When network access is restored (or on a jump host), export and upload the WAL:

```bash
aluminatiai replay --output metrics.csv
# optionally clear the WAL after export
aluminatiai replay --output metrics.csv --clear
```

### Directory permissions

Data, WAL, and log directories are created with mode `0o700` (owner read/write/execute only). For maximum protection, also set `umask 077` in the agent's service unit or container entrypoint.

---

## Self-hosting

Point the agent at your own ingest endpoint:

```bash
ALUMINATAI_API_ENDPOINT=https://your-api.internal/v1/metrics/ingest \
ALUMINATAI_API_KEY=your_key \
aluminatiai
```

## License

Apache 2.0 — see [LICENSE](../LICENSE).
