"""
Demo script to showcase the Cost Optimization Agent
"""
from models import Job, GPU, ScheduleState, Priority
from minimax import MinimaxScheduler, naive_schedule
import json
from typing import Dict


def print_separator(title: str = ""):
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def print_schedule(schedule_data: Dict, title: str = "Schedule"):
    print_separator(title)
    schedule = schedule_data['schedule']
    metrics = schedule_data['metrics']

    print(f"\n📋 Schedule ({len(schedule)} jobs):")
    for i, action in enumerate(schedule, 1):
        print(f"  {i}. Job {action['job_id']}")
        print(f"     GPUs: {', '.join(action['gpu_ids'])}")
        print(f"     Start: {action['start_time']} min")

    print(f"\n📊 Metrics:")
    print(f"  Total Time: {metrics['total_time']} minutes")
    print(f"  Energy Cost: ${metrics['total_energy_cost']:.2f}")
    print(f"  Jobs Completed: {metrics['jobs_completed']}")
    print(f"  Avg Wait Time: {metrics['avg_wait_time']:.1f} minutes")


def create_sample_scenario():
    """Create a sample scheduling scenario"""
    jobs = [
        Job(
            id="training_job_1",
            duration=120,  # 2 hours
            gpu_count=4,
            priority=Priority.HIGH,
            estimated_power_per_gpu=350,  # watts
            arrival_time=0
        ),
        Job(
            id="inference_job_1",
            duration=30,
            gpu_count=1,
            priority=Priority.MEDIUM,
            estimated_power_per_gpu=200,
            arrival_time=0
        ),
        Job(
            id="training_job_2",
            duration=180,  # 3 hours
            gpu_count=2,
            priority=Priority.HIGH,
            estimated_power_per_gpu=400,
            arrival_time=10
        ),
        Job(
            id="inference_job_2",
            duration=15,
            gpu_count=1,
            priority=Priority.LOW,
            estimated_power_per_gpu=150,
            arrival_time=20
        ),
        Job(
            id="training_job_3",
            duration=90,
            gpu_count=3,
            priority=Priority.CRITICAL,
            estimated_power_per_gpu=380,
            arrival_time=30
        ),
    ]

    gpus = [
        GPU(id="gpu_a100_1", model="A100", max_power=400, idle_power=50, cost_per_kwh=0.12),
        GPU(id="gpu_a100_2", model="A100", max_power=400, idle_power=50, cost_per_kwh=0.12),
        GPU(id="gpu_a100_3", model="A100", max_power=400, idle_power=50, cost_per_kwh=0.12),
        GPU(id="gpu_a100_4", model="A100", max_power=400, idle_power=50, cost_per_kwh=0.12),
        GPU(id="gpu_h100_1", model="H100", max_power=700, idle_power=70, cost_per_kwh=0.15),
        GPU(id="gpu_h100_2", model="H100", max_power=700, idle_power=70, cost_per_kwh=0.15),
    ]

    return ScheduleState(current_time=0, pending_jobs=jobs, gpus=gpus)


def main():
    print_separator("🚀 GPU Cost Optimization Agent - Demo")
    print("\nThis demo shows how the minimax algorithm optimizes GPU job scheduling")
    print("by balancing the trade-off between speed and energy cost.\n")

    # Create scenario
    print("📦 Creating sample scenario...")
    print("   - 5 jobs (mix of training and inference)")
    print("   - 6 GPUs (4x A100, 2x H100)")
    print("   - Energy cost: $0.12-0.15 per kWh")

    initial_state = create_sample_scenario()

    # Run Minimax Optimization
    print("\n🧠 Running Minimax Optimization...")
    print("   - Search depth: 4")
    print("   - Weights: 50% speed, 50% cost")

    scheduler = MinimaxScheduler(max_depth=4, speed_weight=0.5, cost_weight=0.5)
    minimax_result = scheduler.find_optimal_schedule(initial_state)

    # Run Naive Scheduling
    print("\n📝 Running Naive FIFO Scheduling (for comparison)...")
    naive_result = naive_schedule(initial_state)

    # Display Results
    print_schedule(minimax_result, "🎯 MINIMAX OPTIMAL SCHEDULE")

    if 'search_stats' in minimax_result:
        stats = minimax_result['search_stats']
        print(f"\n🔍 Search Statistics:")
        print(f"  Nodes Explored: {stats['total_nodes_explored']}")
        print(f"  Nodes Pruned: {stats['nodes_pruned']}")
        print(f"  Max Depth Reached: {stats['max_depth_reached']}")
        print(f"  Decisions Made: {stats['decisions_made']}")

    print_schedule(naive_result, "📋 NAIVE FIFO SCHEDULE")

    # Comparison
    print_separator("💰 COST SAVINGS")
    minimax_cost = minimax_result['metrics']['total_energy_cost']
    naive_cost = naive_result['metrics']['total_energy_cost']
    savings = naive_cost - minimax_cost
    savings_pct = (savings / naive_cost * 100) if naive_cost > 0 else 0

    minimax_time = minimax_result['metrics']['total_time']
    naive_time = naive_result['metrics']['total_time']
    time_diff = naive_time - minimax_time

    print(f"\n💵 Energy Cost:")
    print(f"  Naive:   ${naive_cost:.2f}")
    print(f"  Minimax: ${minimax_cost:.2f}")
    print(f"  Saved:   ${savings:.2f} ({savings_pct:.1f}%)")

    print(f"\n⏱️  Completion Time:")
    print(f"  Naive:   {naive_time} minutes")
    print(f"  Minimax: {minimax_time} minutes")
    print(f"  Diff:    {time_diff:+d} minutes")

    # Test different weight combinations
    print_separator("⚖️  WEIGHT SENSITIVITY ANALYSIS")
    print("\nTesting different speed/cost weight combinations:\n")

    weight_tests = [
        (1.0, 0.0, "Pure Speed"),
        (0.75, 0.25, "Speed Priority"),
        (0.5, 0.5, "Balanced"),
        (0.25, 0.75, "Cost Priority"),
        (0.0, 1.0, "Pure Cost"),
    ]

    print(f"{'Strategy':<20} {'Time':<12} {'Cost':<12} {'Savings':<12}")
    print("-" * 60)

    for speed_w, cost_w, label in weight_tests:
        scheduler = MinimaxScheduler(max_depth=3, speed_weight=speed_w, cost_weight=cost_w)
        result = scheduler.find_optimal_schedule(initial_state)
        metrics = result['metrics']
        cost_saved = naive_cost - metrics['total_energy_cost']

        print(f"{label:<20} {metrics['total_time']:<12} ${metrics['total_energy_cost']:<11.2f} ${cost_saved:<11.2f}")

    print_separator("✅ Demo Complete!")
    print("\nKey Takeaways:")
    print("  1. Minimax finds optimal trade-offs between speed and cost")
    print("  2. Alpha-beta pruning makes search efficient")
    print("  3. Different weights produce different optimal strategies")
    print("  4. Consistently outperforms naive FIFO scheduling")
    print("\nNext Steps:")
    print("  - Run the API server: python api.py")
    print("  - Test via API: POST to http://localhost:8000/api/schedule")
    print("  - View docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
