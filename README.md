# AluminatAI - GPU Energy Intelligence Platform

**Know exactly what your GPUs cost. Every watt, every dollar, every job.**

AluminatAI is an open-source GPU energy monitoring platform that gives AI teams real-time visibility into power consumption, energy costs, and utilization across their GPU fleet. A lightweight Python agent runs on your GPU machines and streams metrics to a cloud dashboard where you can track spending, compare jobs, and optimize workloads.

**Live:** [aluminatiai-landing.vercel.app](https://aluminatiai-landing.vercel.app)

---

## How It Works

```
┌──────────────────┐     HTTPS/JSON      ┌──────────────────────────┐
│   GPU Machine     │ ─────────────────► │   AluminatAI Platform     │
│                    │   every 60s        │                            │
│  ┌──────────────┐ │                     │  ┌──────────┐             │
│  │  Python Agent │ │                     │  │ Next.js  │  Vercel     │
│  │  (pynvml)    │ │                     │  │ API      │             │
│  └──────────────┘ │                     │  └────┬─────┘             │
│                    │                     │       │                    │
│  NVIDIA A100/H100  │                     │  ┌────▼─────┐             │
│  RTX 3090/4090     │                     │  │ Supabase │  PostgreSQL │
│  Any NVIDIA GPU    │                     │  │ Database │  + RLS      │
└──────────────────┘                     │  └────┬─────┘             │
                                          │       │                    │
                                          │  ┌────▼─────┐             │
                                          │  │Dashboard │  React      │
                                          │  │ UI       │  + Recharts │
                                          │  └──────────┘             │
                                          └──────────────────────────┘
```

## Features

- **Real-Time GPU Monitoring** - Power draw, utilization, temperature, memory, and clock speeds sampled every 5 seconds
- **Energy Cost Tracking** - Calculates energy consumption in kWh and converts to dollar costs at your electricity rate
- **Job Attribution** - Track which training jobs consumed how much energy and what they cost
- **Dashboard** - Three views: Today's Cost, Jobs Table, and Utilization vs Power chart
- **Free Trial** - 30-day free trial with auto-generated API keys on signup
- **Lightweight Agent** - <1% CPU, ~50MB RAM overhead on GPU machines
- **Secure** - Row-Level Security, API key auth with `pgcrypto`, rate limiting, server-side validation
- **Minimax Scheduler** - Bonus hackathon project: AI-powered job scheduling that balances speed vs. energy cost

---

## Project Structure

```
AluminatAI/
├── aluminatai-landing/          # Next.js web platform (deployed to Vercel)
│   ├── app/
│   │   ├── api/
│   │   │   ├── metrics/ingest/  # GPU metrics ingestion endpoint
│   │   │   ├── dashboard/       # today-cost, jobs, utilization-chart
│   │   │   ├── user/profile/    # User profile + API key rotation
│   │   │   └── cron/            # Materialized view refresh
│   │   ├── dashboard/           # Protected dashboard UI
│   │   ├── login/               # Auth pages
│   │   └── page.tsx             # Landing page
│   ├── components/              # React components
│   ├── lib/                     # Auth, rate limiting, Supabase clients
│   └── database/migrations/     # SQL migrations (001-005)
│
├── agent/                       # Python GPU monitoring agent
│   ├── main.py                  # Agent entry point
│   ├── collector.py             # NVML-based GPU metrics collector
│   ├── uploader.py              # API upload with retry + local backup
│   ├── config.py                # Environment-based configuration
│   ├── install.sh               # One-line install script
│   └── tests/                   # Test suite + Colab notebook
│
├── minimax-scheduler/           # Hackathon: Minimax GPU job scheduler
│   └── backend/                 # FastAPI + minimax algorithm
│
├── backend/                     # Legacy FastAPI backend (reference)
├── frontend/                    # Legacy React frontend (reference)
├── docker/                      # Docker configs for agent + backend
├── docs/                        # Architecture docs, metrics schema
└── assets/                      # Logo and diagrams
```

---

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- A Supabase account ([supabase.com](https://supabase.com))
- An NVIDIA GPU (for the agent) or Google Colab with GPU runtime

### 1. Clone the Repository

```bash
git clone https://github.com/AgentMulder404/aluminatai-landing.git
cd aluminatai-landing
```

### 2. Set Up the Database (Supabase)

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the migrations in order:

```bash
# Run these SQL files in the Supabase SQL Editor:
database/migrations/002_gpu_monitoring_schema_postgres.sql
database/migrations/003_fix_materialized_view.sql
database/migrations/004_fix_trigger_permissions.sql
database/migrations/005_secure_api_keys_and_constraints.sql
```

This creates:
- `users` table with auto-generated API keys (using `pgcrypto`)
- `gpu_metrics` time-series table with CHECK constraints
- `gpu_jobs` table for job tracking
- `gpu_metrics_hourly` materialized view for fast dashboard queries
- Row-Level Security policies on all tables
- Triggers for user profile auto-creation on signup

### 3. Set Up the Web Platform

```bash
cd aluminatai-landing
npm install
```

Create a `.env.local` file:

```bash
# Supabase (from your project settings > API)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Cron secret (generate with: openssl rand -base64 32)
CRON_SECRET=your-cron-secret
```

Run the development server:

```bash
npm run dev
```

Visit `http://localhost:3000` - you should see the landing page.

### 4. Create an Account

1. Click **"Start Free Trial"** on the landing page
2. Enter your name, email, and password
3. You'll be redirected to the dashboard setup page
4. Copy your API key (starts with `alum_`)

### 5. Install the GPU Agent

On your GPU machine (or Google Colab):

```bash
# Install dependencies
pip install pynvml requests python-dotenv rich

# Set environment variables
export ALUMINATAI_API_KEY="alum_your_key_here"
export ALUMINATAI_API_ENDPOINT="http://localhost:3000/api/metrics/ingest"

# Run the agent
python agent/main.py
```

Options:

```bash
# Custom sampling interval (1 second)
python agent/main.py --interval 1

# Save to CSV + upload
python agent/main.py --output data/metrics.csv

# Run for 5 minutes
python agent/main.py --duration 300

# Quiet mode (no console output)
python agent/main.py --quiet --output data/metrics.csv
```

For production, use the systemd service:

```bash
cd agent
chmod +x install.sh
sudo ./install.sh
```

### 6. Test on Google Colab (A100)

Upload `agent/tests/AluminatAI_A100_Test.ipynb` to Google Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File > Upload notebook** and select the `.ipynb` file
3. **Runtime > Change runtime type** > select **A100 GPU**
4. Paste your API key in Cell 2
5. **Runtime > Run all**

The notebook runs 7 test suites:
- NVML hardware access
- Collector class + energy calculation
- API authentication validation
- End-to-end collect + upload
- Stress test under GPU load (8192x8192 matmul)
- API key security audit
- 60-second continuous monitoring demo

---

## API Reference

### Metrics Ingestion

```
POST /api/metrics/ingest
Header: X-API-Key: alum_your_key_here
```

**Request body** (single metric or array):

```json
[
  {
    "timestamp": "2026-02-06T12:00:00Z",
    "gpu_index": 0,
    "gpu_uuid": "GPU-abc123",
    "gpu_name": "NVIDIA A100-SXM4-40GB",
    "power_draw_w": 285.5,
    "power_limit_w": 400.0,
    "energy_delta_j": 571.0,
    "utilization_gpu_pct": 95,
    "utilization_memory_pct": 60,
    "temperature_c": 72,
    "memory_used_mb": 32000,
    "memory_total_mb": 40960
  }
]
```

**Validation rules:**
- `power_draw_w`: 0-1500W
- `temperature_c`: 0-120C
- `utilization_*_pct`: 0-100
- `timestamp`: valid ISO 8601, not more than 5 minutes in the future
- Max 1000 metrics per request

**Rate limit:** 100 requests/minute per user

### Dashboard APIs

| Endpoint | Method | Auth | Rate Limit | Description |
|---|---|---|---|---|
| `/api/dashboard/today-cost` | GET | Session | 60/min | Today's energy cost |
| `/api/dashboard/jobs` | GET | Session | 60/min | Job history with pagination |
| `/api/dashboard/utilization-chart` | GET | Session | 60/min | Time-series chart data |
| `/api/user/profile` | GET | Session | - | User profile + API key |
| `/api/user/profile` | PATCH | Session | - | Update profile settings |
| `/api/user/profile` | POST | Session | 5/hr | Rotate API key |

### API Key Rotation

```bash
curl -X POST https://aluminatiai-landing.vercel.app/api/user/profile \
  -H "Content-Type: application/json" \
  -H "Cookie: your-session-cookie" \
  -d '{"action": "rotate_api_key"}'
```

---

## Security

- **API Keys**: Generated with `pgcrypto gen_random_bytes()` - 340 bits of entropy
- **Row-Level Security**: Users can only access their own data
- **Rate Limiting**: Per-user limits on all endpoints
- **Input Validation**: Server-side + database CHECK constraints
- **HTTPS**: Enforced by Vercel
- **No ambiguous characters**: API keys exclude `0, O, I, l, 1` to prevent copy errors

---

## Deployment

### Vercel (Web Platform)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd aluminatai-landing
vercel

# Set environment variables in Vercel dashboard
```

### Cron Job (Materialized View Refresh)

Set up a cron job to refresh the hourly metrics view:

- **URL**: `https://your-app.vercel.app/api/cron/refresh-metrics`
- **Method**: POST
- **Header**: `Authorization: Bearer your-cron-secret`
- **Schedule**: Every hour (`0 * * * *`)

You can use [cron-job.org](https://cron-job.org) (free) or Vercel Cron.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Next.js 16 |
| UI | React 19 + Tailwind CSS 4 |
| Charts | Recharts |
| Database | Supabase PostgreSQL |
| Auth | Supabase Auth |
| GPU Agent | Python + pynvml (NVML) |
| Deployment | Vercel |
| Scheduler | Minimax with alpha-beta pruning |

---

## Minimax GPU Scheduler

A bonus hackathon project in `minimax-scheduler/` that uses game theory to optimize GPU job scheduling:

- **Speed Player (Maximizer)**: Wants to complete jobs ASAP
- **Cost Player (Minimizer)**: Wants to minimize energy costs
- **Alpha-Beta Pruning**: Efficiently explores the decision tree
- **Result**: 15-30% cost savings vs. naive FIFO scheduling

```bash
cd minimax-scheduler/backend
pip install -r requirements.txt
python demo.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

---

Built by [@AgentMulder404](https://github.com/AgentMulder404)
