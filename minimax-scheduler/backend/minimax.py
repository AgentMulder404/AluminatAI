"""
Minimax algorithm with alpha-beta pruning for GPU job scheduling
"""
from typing import Tuple, List, Optional, Dict
from models import ScheduleState, ScheduleAction
import math


class MinimaxStats:
    """Track statistics during minimax search"""
    def __init__(self):
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.max_depth_reached = 0
        self.decision_path: List[Dict] = []

    def to_dict(self) -> Dict:
        return {
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'max_depth_reached': self.max_depth_reached,
            'decision_path': self.decision_path
        }


class MinimaxScheduler:
    """
    Minimax-based scheduler that treats scheduling as a two-player game:
    - Max player (Speed): Wants to minimize completion time
    - Min player (Cost): Wants to minimize energy cost

    Uses alpha-beta pruning for efficiency.
    """

    def __init__(self,
                 max_depth: int = 4,
                 speed_weight: float = 0.5,
                 cost_weight: float = 0.5):
        """
        Args:
            max_depth: Maximum search depth
            speed_weight: Weight for completion time (0-1)
            cost_weight: Weight for energy cost (0-1)
        """
        self.max_depth = max_depth
        self.speed_weight = speed_weight
        self.cost_weight = cost_weight
        self.stats = MinimaxStats()

    def evaluate_state(self, state: ScheduleState) -> float:
        """
        Heuristic evaluation function for a state.
        Returns a score where:
        - Higher is better for Max player (speed-focused)
        - Lower is better for Min player (cost-focused)
        """
        metrics = state.get_metrics()

        # Normalize metrics (these are rough estimates, adjust based on your scale)
        time_score = -metrics['total_time'] / 100.0  # Negative because lower time is better
        cost_score = -metrics['total_energy_cost'] / 10.0  # Negative because lower cost is better

        # Jobs completed is good (more positive)
        completion_score = metrics['jobs_completed'] * 10

        # Wait time penalty (negative)
        wait_penalty = -metrics['avg_wait_time'] / 10.0

        # Remaining jobs penalty
        pending_penalty = -len(state.pending_jobs) * 5

        # Combined score
        score = (
            completion_score +
            self.speed_weight * time_score +
            self.cost_weight * cost_score +
            wait_penalty +
            pending_penalty
        )

        return score

    def minimax(self,
                state: ScheduleState,
                depth: int,
                alpha: float,
                beta: float,
                maximizing: bool) -> Tuple[float, Optional[ScheduleAction]]:
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            state: Current scheduling state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn

        Returns:
            (best_score, best_action)
        """
        self.stats.nodes_explored += 1
        self.stats.max_depth_reached = max(self.stats.max_depth_reached,
                                           self.max_depth - depth)

        # Terminal conditions
        if depth == 0 or state.is_terminal():
            return self.evaluate_state(state), None

        possible_actions = state.get_possible_actions(max_actions=5)

        if not possible_actions:
            # No valid actions, terminal state
            return self.evaluate_state(state), None

        best_action = None

        if maximizing:
            # Max player: wants higher scores (faster completion)
            max_eval = -math.inf

            for action in possible_actions:
                new_state = state.apply_action(action)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.stats.nodes_pruned += 1
                    break  # Beta cutoff

            return max_eval, best_action

        else:
            # Min player: wants lower scores (lower cost)
            min_eval = math.inf

            for action in possible_actions:
                new_state = state.apply_action(action)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action

                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.stats.nodes_pruned += 1
                    break  # Alpha cutoff

            return min_eval, best_action

    def find_optimal_schedule(self, initial_state: ScheduleState) -> Dict:
        """
        Find the optimal schedule starting from initial state.

        Returns:
            Dict containing schedule, metrics, and search statistics
        """
        self.stats = MinimaxStats()  # Reset stats

        current_state = initial_state.copy()
        full_schedule = []
        decision_tree_info = []

        # Iteratively build schedule until all jobs are scheduled
        while not current_state.is_terminal():
            # Run minimax from current state
            score, best_action = self.minimax(
                current_state,
                depth=self.max_depth,
                alpha=-math.inf,
                beta=math.inf,
                maximizing=True  # Start with max player
            )

            if best_action is None:
                # No valid actions found, break
                break

            # Record decision
            decision_tree_info.append({
                'depth': len(full_schedule),
                'action': {
                    'job_id': best_action.job_id,
                    'gpu_ids': best_action.gpu_ids,
                    'start_time': best_action.start_time
                },
                'score': score,
                'nodes_explored_for_decision': self.stats.nodes_explored
            })

            # Apply the best action
            current_state = current_state.apply_action(best_action)
            full_schedule.append(best_action)

            # Reset stats for next decision
            prev_nodes = self.stats.nodes_explored
            self.stats = MinimaxStats()
            self.stats.nodes_explored = prev_nodes  # Accumulate total

        # Get final metrics
        final_metrics = current_state.get_metrics()

        return {
            'schedule': [
                {
                    'job_id': action.job_id,
                    'gpu_ids': action.gpu_ids,
                    'start_time': action.start_time,
                }
                for action in full_schedule
            ],
            'metrics': final_metrics,
            'search_stats': {
                'total_nodes_explored': self.stats.nodes_explored,
                'nodes_pruned': self.stats.nodes_pruned,
                'max_depth_reached': self.stats.max_depth_reached,
                'decisions_made': len(decision_tree_info)
            },
            'decision_tree': decision_tree_info
        }


def naive_schedule(initial_state: ScheduleState) -> Dict:
    """
    Naive FIFO scheduling for comparison.
    Just schedules jobs in order as GPUs become available.
    """
    current_state = initial_state.copy()
    schedule = []

    while not current_state.is_terminal():
        # Get first job that can be scheduled
        for job in current_state.pending_jobs:
            if current_state.can_schedule_job(job):
                available_gpus = current_state.get_available_gpus()[:job.gpu_count]
                action = ScheduleAction(
                    job_id=job.id,
                    gpu_ids=[g.id for g in available_gpus],
                    start_time=current_state.current_time
                )
                current_state = current_state.apply_action(action)
                schedule.append(action)
                break
        else:
            # No job can be scheduled now, advance time
            next_available_time = min(g.available_at for g in current_state.gpus
                                     if g.available_at > current_state.current_time)
            current_state.current_time = next_available_time

    final_metrics = current_state.get_metrics()

    return {
        'schedule': [
            {
                'job_id': action.job_id,
                'gpu_ids': action.gpu_ids,
                'start_time': action.start_time,
            }
            for action in schedule
        ],
        'metrics': final_metrics,
        'algorithm': 'naive_fifo'
    }
