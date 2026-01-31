"""
Data models for GPU scheduling system
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Job:
    """Represents a GPU job to be scheduled"""
    id: str
    duration: int  # minutes
    gpu_count: int  # number of GPUs required
    priority: Priority
    estimated_power_per_gpu: float  # watts
    arrival_time: int = 0  # when job entered the queue
    deadline: Optional[int] = None  # optional deadline (minutes)

    def __post_init__(self):
        if isinstance(self.priority, str):
            self.priority = Priority(self.priority)

    @property
    def priority_weight(self) -> float:
        """Convert priority to numerical weight"""
        weights = {
            Priority.LOW: 1.0,
            Priority.MEDIUM: 1.5,
            Priority.HIGH: 2.0,
            Priority.CRITICAL: 3.0
        }
        return weights[self.priority]


@dataclass
class GPU:
    """Represents a GPU resource"""
    id: str
    model: str  # e.g., "A100", "H100", "RTX4090"
    max_power: float  # watts
    idle_power: float  # watts when idle
    cost_per_kwh: float  # dollars per kWh
    currently_running: Optional[str] = None  # job_id if busy
    available_at: int = 0  # time (minutes) when GPU becomes available

    @property
    def is_available(self) -> bool:
        """Check if GPU is currently available"""
        return self.currently_running is None

    def energy_cost(self, duration_minutes: int, power_watts: float) -> float:
        """Calculate energy cost for running at given power for duration"""
        kwh = (power_watts * duration_minutes) / (60 * 1000)  # convert to kWh
        return kwh * self.cost_per_kwh


@dataclass
class ScheduleAction:
    """Represents a scheduling decision"""
    job_id: str
    gpu_ids: List[str]
    start_time: int  # minutes from now

    def __hash__(self):
        return hash((self.job_id, tuple(sorted(self.gpu_ids)), self.start_time))


@dataclass
class ScheduleState:
    """Represents the current state of the scheduling system"""
    current_time: int  # current time in minutes
    pending_jobs: List[Job]  # jobs waiting to be scheduled
    gpus: List[GPU]  # all GPUs in the cluster
    completed_jobs: List[str] = field(default_factory=list)  # job IDs
    schedule: List[ScheduleAction] = field(default_factory=list)  # decided schedule
    total_energy_cost: float = 0.0  # accumulated cost

    def copy(self) -> 'ScheduleState':
        """Create a deep copy of the state"""
        import copy
        return copy.deepcopy(self)

    def get_available_gpus(self, at_time: int = None) -> List[GPU]:
        """Get GPUs that are available at the specified time"""
        check_time = at_time if at_time is not None else self.current_time
        return [gpu for gpu in self.gpus if gpu.available_at <= check_time]

    def can_schedule_job(self, job: Job, at_time: int = None) -> bool:
        """Check if there are enough GPUs available for a job"""
        available = self.get_available_gpus(at_time)
        return len(available) >= job.gpu_count

    def apply_action(self, action: ScheduleAction) -> 'ScheduleState':
        """Apply a scheduling action and return new state"""
        new_state = self.copy()

        # Find the job
        job = next((j for j in new_state.pending_jobs if j.id == action.job_id), None)
        if not job:
            return new_state

        # Update GPUs
        for gpu_id in action.gpu_ids:
            gpu = next((g for g in new_state.gpus if g.id == gpu_id), None)
            if gpu:
                gpu.currently_running = job.id
                gpu.available_at = action.start_time + job.duration

                # Calculate energy cost
                energy_cost = gpu.energy_cost(job.duration, job.estimated_power_per_gpu)
                new_state.total_energy_cost += energy_cost

        # Move job from pending to scheduled
        new_state.pending_jobs = [j for j in new_state.pending_jobs if j.id != job.id]
        new_state.schedule.append(action)

        # Advance time to when this job finishes
        new_state.current_time = action.start_time + job.duration

        return new_state

    def get_possible_actions(self, max_actions: int = 5) -> List[ScheduleAction]:
        """
        Generate possible scheduling actions from current state.
        Limited to max_actions for performance.
        """
        actions = []

        for job in sorted(self.pending_jobs, key=lambda j: -j.priority_weight)[:max_actions]:
            # Check if we can schedule this job now
            available_gpus = self.get_available_gpus()

            if len(available_gpus) >= job.gpu_count:
                # Schedule immediately
                selected_gpus = available_gpus[:job.gpu_count]
                action = ScheduleAction(
                    job_id=job.id,
                    gpu_ids=[g.id for g in selected_gpus],
                    start_time=self.current_time
                )
                actions.append(action)
            else:
                # Wait for GPUs to become available
                # Find earliest time when enough GPUs are free
                all_available_times = sorted(set(g.available_at for g in self.gpus))
                for future_time in all_available_times:
                    if future_time <= self.current_time:
                        continue
                    available_future = self.get_available_gpus(future_time)
                    if len(available_future) >= job.gpu_count:
                        selected_gpus = available_future[:job.gpu_count]
                        action = ScheduleAction(
                            job_id=job.id,
                            gpu_ids=[g.id for g in selected_gpus],
                            start_time=future_time
                        )
                        actions.append(action)
                        break  # Only consider first available time

        return actions[:max_actions]  # Limit branching factor

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (no more jobs to schedule)"""
        return len(self.pending_jobs) == 0

    def get_metrics(self) -> Dict:
        """Get current metrics for this state"""
        if not self.schedule:
            return {
                'total_time': 0,
                'total_energy_cost': 0,
                'jobs_completed': 0,
                'avg_wait_time': 0
            }

        # Calculate total makespan (time until all jobs complete)
        max_completion_time = max(
            action.start_time + next(j for j in [*self.pending_jobs, *[
                Job(id=a.job_id, duration=0, gpu_count=0, priority=Priority.LOW, estimated_power_per_gpu=0)
                for a in self.schedule
            ]] if j.id == action.job_id).duration
            for action in self.schedule
        ) if self.schedule else 0

        # Calculate average wait time
        wait_times = []
        for action in self.schedule:
            job = next((j for j in [*self.pending_jobs, *[
                Job(id=a.job_id, duration=0, gpu_count=0, priority=Priority.LOW, estimated_power_per_gpu=0)
                for a in self.schedule
            ]] if j.id == action.job_id), None)
            if job:
                wait_time = action.start_time - job.arrival_time
                wait_times.append(max(0, wait_time))

        return {
            'total_time': max_completion_time,
            'total_energy_cost': self.total_energy_cost,
            'jobs_completed': len(self.schedule),
            'avg_wait_time': sum(wait_times) / len(wait_times) if wait_times else 0
        }
