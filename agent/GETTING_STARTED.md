# Getting Started with GPU Energy Agent

## Prerequisites

1. **NVIDIA GPU** with driver 450.80.02 or newer
2. **Python 3.8+**
3. **Linux OS** (Ubuntu, CentOS, etc.)

## Quick Start

### 1. Install Dependencies

```bash
cd agent
pip install -r requirements.txt
```

Expected output:
```
Successfully installed nvidia-ml-py3-7.352.0 psutil-5.9.6 rich-13.7.0 ...
```

### 2. Test GPU Detection

Run a quick test to verify the collector can access your GPUs:

```bash
python collector.py
```

Expected output:
```
Testing GPU Collector...
Found 4 GPUs:
  GPU 0: NVIDIA A100-SXM4-40GB (GPU-abc123...)
  GPU 1: NVIDIA A100-SXM4-40GB (GPU-def456...)
  ...

Collecting 3 samples (2s intervals)...
  GPU 0: 287.4W, 98% util, 76°C, 1437.0J
  ...

✅ Collector test passed!
```

### 3. Run the Agent

Start monitoring all GPUs:

```bash
python main.py
```

You should see real-time metrics updating every 5 seconds:

```
GPU Energy Agent v0.1.0
📊 Monitoring 4 GPUs
⏱️  Sampling interval: 5.0s

┏━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ GPU ┃  Power ┃ Util ┃ Temp ┃ Energy Δ┃ Total kWh┃
┡━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ GPU 0│ 287.4W │  98% │  76°C│   1437J │  0.0012  │
│ GPU 1│ 289.1W │  97% │  77°C│   1446J │  0.0012  │
...
```

Press `Ctrl+C` to stop.

### 4. Export to CSV

Run for 5 minutes and save metrics to CSV:

```bash
python main.py --duration 300 --output data/test_run.csv
```

This creates `data/test_run.csv` with timestamped metrics for analysis.

### 5. Run Tests

#### Test 1: Unit Tests

```bash
python tests/test_collector.py
```

Expected: All tests pass ✅

#### Test 2: Overhead Benchmark

Measure agent CPU/memory impact:

```bash
python tests/benchmark_overhead.py
```

Expected results:
- CPU overhead: <0.1%
- Memory overhead: <50 MB
- Collection latency: <1ms per GPU

#### Test 3: Energy Validation

Validate energy calculations:

```bash
python tests/validate_energy.py --duration 60
```

Expected: Energy error <5% ✅

## Common Issues

### Issue: "No NVIDIA GPUs found"

**Cause**: NVIDIA driver not installed or not working

**Fix**:
```bash
# Check driver
nvidia-smi

# If not working, install driver:
# Ubuntu:
sudo apt install nvidia-driver-535

# Then reboot
sudo reboot
```

### Issue: "Failed to initialize NVML"

**Cause**: Permission issue or driver mismatch

**Fix**:
```bash
# Check driver version
nvidia-smi

# Ensure you're in the right groups
groups

# Add to video group if needed
sudo usermod -a -G video $USER
# Then log out and back in
```

### Issue: "Module 'pynvml' not found"

**Cause**: Dependencies not installed

**Fix**:
```bash
pip install -r requirements.txt
```

## Next Steps

### Week 1-2: Current Focus

- [x] Metrics collection working
- [x] Energy calculations validated
- [x] Overhead benchmarked
- [ ] Run 24-hour stability test
- [ ] Compare against external power meter

### Week 3-4: Attribution

- [ ] Add process tracking (PID → GPU mapping)
- [ ] Parse command lines for model names
- [ ] Implement job start/stop detection

### Week 5-6: Backend Integration

- [ ] Connect to FastAPI backend
- [ ] Upload metrics to database
- [ ] Real-time dashboard

## Testing Checklist

Use this checklist to verify your MVP agent:

```
□ GPU detection works (python collector.py)
□ Agent runs without errors (python main.py)
□ CPU overhead <0.1% (python tests/benchmark_overhead.py)
□ Energy error <5% (python tests/validate_energy.py)
□ CSV export works (python main.py --output data/test.csv)
□ Can run for 1+ hours without crash
□ Energy totals match expectations
```

## Getting Help

If you encounter issues:

1. Check `logs/` directory for error messages
2. Run with verbose logging: `python main.py -v`
3. Review [troubleshooting guide](README.md#troubleshooting)
4. Open issue in main repository

## Performance Tuning

### Reduce Overhead

If CPU usage is too high:

```bash
# Increase sampling interval
python main.py --interval 10

# Disable clock monitoring (edit collector.py):
collector = GPUCollector(collect_clocks=False)
```

### Increase Accuracy

For better energy accuracy:

```bash
# Use faster sampling (1s interval)
python main.py --interval 1

# Trade-off: Slightly higher overhead (~0.05% CPU)
```

## Example Workflows

### Workflow 1: Validate Against Power Meter

```bash
# 1. Start a GPU workload (e.g., training)
python your_training_script.py &

# 2. Run agent for 10 minutes
python main.py --duration 600 --output data/validation.csv

# 3. Compare total kWh with power meter reading
# Note: Meter shows whole system, agent shows GPU only
```

### Workflow 2: Daily Monitoring

```bash
# Run as background service
nohup python main.py --output data/daily_$(date +%Y%m%d).csv > logs/agent.log 2>&1 &

# Check status
tail -f logs/agent.log

# Stop
pkill -f "python main.py"
```

### Workflow 3: Job Energy Profiling

```bash
# Start agent
python main.py --output data/job_profile.csv &
AGENT_PID=$!

# Run your ML job
python train_resnet.py

# Stop agent
kill $AGENT_PID

# Analyze CSV to see energy consumption during job
```

## Success Criteria (Week 1-2)

Your MVP agent is ready when:

- ✅ Runs with <0.1% CPU overhead
- ✅ Collects accurate power metrics (validated against NVML)
- ✅ Energy calculations within 5% of theoretical
- ✅ Can run continuously for 24+ hours
- ✅ CSV export works correctly
- ✅ All unit tests pass

Once these are met, move to Week 3-4 (attribution)!
