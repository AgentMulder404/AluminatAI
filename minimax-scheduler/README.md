# GPU Cost Optimization Agent - Minimax Scheduler

An intelligent job scheduling system that uses minimax algorithm to optimize the trade-off between job completion speed and energy/compute costs.

## Overview

The Cost Optimization Agent makes optimal GPU scheduling decisions by treating scheduling as a two-player game:
- **Speed Player (Maximizer)**: Wants to complete jobs as quickly as possible
- **Cost Player (Minimizer)**: Wants to minimize energy consumption and compute costs

The minimax algorithm with alpha-beta pruning explores the decision tree to find the optimal balance.

## Features

- ⚡ **Minimax Algorithm**: Core decision-making with alpha-beta pruning
- 🎯 **Multi-objective Optimization**: Balances speed vs. cost
- 📊 **Real-time Visualization**: See the decision tree exploration
- 🔌 **Energy-aware**: Considers GPU power consumption in decisions
- 💰 **Cost Tracking**: Shows cost savings vs. naive scheduling

## Architecture

```
minimax-scheduler/
├── backend/
│   ├── models.py          # Job, GPU, State models
│   ├── minimax.py         # Core minimax algorithm
│   ├── evaluator.py       # Heuristic evaluation functions
│   ├── scheduler.py       # Scheduling logic
│   ├── api.py             # FastAPI endpoints
│   └── requirements.txt   # Python dependencies
├── frontend/
│   └── components/
│       ├── MinimaxVisualizer.tsx  # Decision tree viz
│       └── ScheduleComparison.tsx # Results comparison
├── data/
│   └── sample_scenarios.json      # Demo data
└── tests/
    └── test_minimax.py            # Unit tests
```

## Quick Start

### Backend Setup

```bash
cd minimax-scheduler/backend
pip install -r requirements.txt
python api.py
```

### Frontend Integration

The visualization components integrate with the existing AluminatiAI landing page.

## How It Works

### 1. Game Tree Representation

Each node in the tree represents a scheduling state:
- Available GPUs with their current load and power consumption
- Pending jobs with their requirements
- Current time and accumulated cost

### 2. Decision Making

- **Max nodes (Speed)**: Choose actions that complete jobs fastest
- **Min nodes (Cost)**: Choose actions that minimize energy/cost
- **Depth limit**: Configurable lookahead (default: 3-4 levels)

### 3. Evaluation Function

Heuristic scoring based on:
- Job completion time
- Energy consumption (kWh)
- GPU utilization efficiency
- Idle power waste
- Priority weights

### 4. Alpha-Beta Pruning

Optimizes search by pruning branches that won't affect the final decision.

## API Endpoints

### `POST /api/schedule`

Schedule jobs using minimax optimization.

**Request:**
```json
{
  "jobs": [
    {
      "id": "job1",
      "duration": 120,
      "gpu_count": 2,
      "priority": "high",
      "estimated_power": 300
    }
  ],
  "gpus": [
    {
      "id": "gpu1",
      "model": "A100",
      "power_limit": 400,
      "idle_power": 50,
      "cost_per_kwh": 0.12
    }
  ],
  "depth": 4,
  "weights": {
    "speed": 0.6,
    "cost": 0.4
  }
}
```

**Response:**
```json
{
  "schedule": [
    {
      "job_id": "job1",
      "gpu_ids": ["gpu1", "gpu2"],
      "start_time": 0,
      "end_time": 120
    }
  ],
  "metrics": {
    "total_time": 180,
    "total_energy_kwh": 24.5,
    "total_cost": 2.94,
    "naive_cost": 4.20,
    "savings": 1.26
  },
  "decision_tree": {
    "nodes_explored": 142,
    "pruned_branches": 58,
    "max_depth_reached": 4
  }
}
```

## Use Cases

1. **Batch Job Scheduling**: Schedule training jobs overnight
2. **Real-time Inference**: Balance latency vs. cost for inference workloads
3. **Multi-tenant Clusters**: Fair scheduling across teams
4. **Cost Budgeting**: Stay within energy/cost budgets while maximizing throughput

## Performance

- **Search Speed**: ~1000 nodes/second with alpha-beta pruning
- **Decision Time**: < 100ms for typical scenarios (5 GPUs, 10 jobs, depth 4)
- **Cost Savings**: 15-30% vs. naive FIFO scheduling

## Demo

Run the demo to see the agent in action:

```bash
python backend/demo.py
```

This will show:
- Sample scheduling scenario
- Minimax decision process
- Cost comparison
- Recommended schedule

## Built For

AluminatiAI Hackathon - Energy Intelligence for AI Infrastructure
