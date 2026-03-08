# Getting Started

## Prerequisites

- NVIDIA GPU with driver 450.80.02+
- Python 3.8+
- AluminatiAI API key — get one at [aluminatiai.com/dashboard](https://aluminatiai.com/dashboard)

## Install

```bash
pip install aluminatiai
```

## Start monitoring

```bash
export ALUMINATAI_API_KEY=alum_your_key_here
aluminatiai
```

You'll see real-time GPU metrics in the console and on your dashboard.

## Tag workloads for per-job attribution

```bash
ALUMINATAI_TEAM=nlp-team \
ALUMINATAI_MODEL=llama3-finetune \
ALUMINATAI_API_KEY=alum_your_key_here \
python train.py
```

## MLflow / W&B integration

```python
# MLflow
from aluminatiai.integrations.mlflow_callback import AluminatiMLflowCallback
with mlflow.start_run():
    trainer.add_callback(AluminatiMLflowCallback())

# W&B
from aluminatiai.integrations.wandb_callback import AluminatiWandbCallback
wandb.init(project="my-project")
trainer.add_callback(AluminatiWandbCallback())
```

## Production deployment

### systemd

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

### Slurm

```bash
# /etc/slurm/prolog.d/aluminatiai.sh
source /etc/aluminatiai.env
aluminatiai &
```

## Prometheus

The agent exposes a `/metrics` endpoint on port 9100 by default:

```bash
# In your prometheus.yml:
scrape_configs:
  - job_name: aluminatiai
    static_configs:
      - targets: ['gpu-host:9100']
```

Disable with `METRICS_PORT=0`.

## Common issues

### "No NVIDIA GPUs found"

```bash
nvidia-smi   # verify driver is working
```

### "Failed to initialize NVML"

```bash
sudo usermod -a -G video $USER   # add to video group, then re-login
```

### Agent exits immediately

Check `LOG_LEVEL=DEBUG` output and ensure `ALUMINATAI_API_KEY` is set correctly.

## More

Full docs: [aluminatiai.com/docs/agent](https://aluminatiai.com/docs/agent)
