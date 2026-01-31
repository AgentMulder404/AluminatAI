"""
FastAPI server for GPU Cost Optimization Agent
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from models import Job, GPU, ScheduleState, Priority
from minimax import MinimaxScheduler, naive_schedule

app = FastAPI(
    title="GPU Cost Optimization Agent",
    description="Minimax-based intelligent GPU job scheduling",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class JobRequest(BaseModel):
    id: str
    duration: int = Field(..., gt=0, description="Job duration in minutes")
    gpu_count: int = Field(..., gt=0, description="Number of GPUs required")
    priority: str = Field(default="medium", description="Job priority: low, medium, high, critical")
    estimated_power_per_gpu: float = Field(..., gt=0, description="Estimated power consumption per GPU in watts")
    arrival_time: int = Field(default=0, description="When job arrived in queue")
    deadline: Optional[int] = Field(default=None, description="Optional deadline in minutes")


class GPURequest(BaseModel):
    id: str
    model: str = Field(..., description="GPU model name")
    max_power: float = Field(..., gt=0, description="Maximum power consumption in watts")
    idle_power: float = Field(..., gt=0, description="Idle power consumption in watts")
    cost_per_kwh: float = Field(..., gt=0, description="Energy cost in dollars per kWh")


class ScheduleRequest(BaseModel):
    jobs: List[JobRequest]
    gpus: List[GPURequest]
    max_depth: int = Field(default=4, ge=1, le=6, description="Minimax search depth")
    speed_weight: float = Field(default=0.5, ge=0, le=1, description="Weight for speed optimization")
    cost_weight: float = Field(default=0.5, ge=0, le=1, description="Weight for cost optimization")
    include_naive_comparison: bool = Field(default=True, description="Include naive FIFO schedule for comparison")


@app.get("/")
async def root():
    return {
        "message": "GPU Cost Optimization Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/schedule": "POST - Optimize job schedule using minimax",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/schedule")
async def optimize_schedule(request: ScheduleRequest) -> Dict:
    """
    Optimize GPU job scheduling using minimax algorithm.

    Returns optimal schedule that balances speed and cost.
    """
    try:
        # Convert API models to domain models
        jobs = [
            Job(
                id=j.id,
                duration=j.duration,
                gpu_count=j.gpu_count,
                priority=Priority(j.priority),
                estimated_power_per_gpu=j.estimated_power_per_gpu,
                arrival_time=j.arrival_time,
                deadline=j.deadline
            )
            for j in request.jobs
        ]

        gpus = [
            GPU(
                id=g.id,
                model=g.model,
                max_power=g.max_power,
                idle_power=g.idle_power,
                cost_per_kwh=g.cost_per_kwh
            )
            for g in request.gpus
        ]

        # Create initial state
        initial_state = ScheduleState(
            current_time=0,
            pending_jobs=jobs,
            gpus=gpus
        )

        # Run minimax optimization
        scheduler = MinimaxScheduler(
            max_depth=request.max_depth,
            speed_weight=request.speed_weight,
            cost_weight=request.cost_weight
        )

        minimax_result = scheduler.find_optimal_schedule(initial_state)

        # Optionally run naive schedule for comparison
        naive_result = None
        if request.include_naive_comparison:
            naive_result = naive_schedule(initial_state)

        # Build response
        response = {
            "optimal_schedule": minimax_result['schedule'],
            "metrics": minimax_result['metrics'],
            "search_stats": minimax_result['search_stats'],
            "decision_tree": minimax_result['decision_tree'],
            "weights": {
                "speed": request.speed_weight,
                "cost": request.cost_weight
            }
        }

        if naive_result:
            response["naive_schedule"] = naive_result['schedule']
            response["naive_metrics"] = naive_result['metrics']
            response["cost_savings"] = {
                "energy_cost_saved": naive_result['metrics']['total_energy_cost'] - minimax_result['metrics']['total_energy_cost'],
                "time_difference": naive_result['metrics']['total_time'] - minimax_result['metrics']['total_time'],
                "percentage_saved": (
                    (naive_result['metrics']['total_energy_cost'] - minimax_result['metrics']['total_energy_cost']) /
                    naive_result['metrics']['total_energy_cost'] * 100
                ) if naive_result['metrics']['total_energy_cost'] > 0 else 0
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheduling error: {str(e)}")


@app.post("/api/evaluate")
async def evaluate_schedule(request: ScheduleRequest) -> Dict:
    """
    Evaluate different weight combinations to find best balance.
    """
    try:
        # Convert API models
        jobs = [
            Job(
                id=j.id,
                duration=j.duration,
                gpu_count=j.gpu_count,
                priority=Priority(j.priority),
                estimated_power_per_gpu=j.estimated_power_per_gpu,
                arrival_time=j.arrival_time,
                deadline=j.deadline
            )
            for j in request.jobs
        ]

        gpus = [
            GPU(
                id=g.id,
                model=g.model,
                max_power=g.max_power,
                idle_power=g.idle_power,
                cost_per_kwh=g.cost_per_kwh
            )
            for g in request.gpus
        ]

        initial_state = ScheduleState(current_time=0, pending_jobs=jobs, gpus=gpus)

        # Test different weight combinations
        weight_combinations = [
            (1.0, 0.0),  # Pure speed
            (0.75, 0.25),
            (0.5, 0.5),  # Balanced
            (0.25, 0.75),
            (0.0, 1.0),  # Pure cost
        ]

        results = []
        for speed_w, cost_w in weight_combinations:
            scheduler = MinimaxScheduler(
                max_depth=request.max_depth,
                speed_weight=speed_w,
                cost_weight=cost_w
            )
            result = scheduler.find_optimal_schedule(initial_state)
            results.append({
                "weights": {"speed": speed_w, "cost": cost_w},
                "metrics": result['metrics']
            })

        return {"weight_analysis": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
