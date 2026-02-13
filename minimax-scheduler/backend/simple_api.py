"""
Simple Flask API for the minimax scheduler (easier to install than FastAPI)
Install: pip3 install flask flask-cors
Run: python3 simple_api.py
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from models import Job, GPU, ScheduleState, Priority
from minimax import MinimaxScheduler, naive_schedule

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

@app.route('/')
def home():
    return jsonify({
        "message": "GPU Cost Optimization Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/schedule": "POST - Optimize job schedule using minimax",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/schedule', methods=['POST'])
def schedule():
    try:
        data = request.json

        # Convert to domain models
        jobs = [
            Job(
                id=j['id'],
                duration=j['duration'],
                gpu_count=j['gpu_count'],
                priority=Priority(j.get('priority', 'medium')),
                estimated_power_per_gpu=j['estimated_power_per_gpu'],
                arrival_time=j.get('arrival_time', 0),
                deadline=j.get('deadline')
            )
            for j in data['jobs']
        ]

        gpus = [
            GPU(
                id=g['id'],
                model=g['model'],
                max_power=g['max_power'],
                idle_power=g['idle_power'],
                cost_per_kwh=g['cost_per_kwh']
            )
            for g in data['gpus']
        ]

        initial_state = ScheduleState(current_time=0, pending_jobs=jobs, gpus=gpus)

        # Run minimax
        scheduler = MinimaxScheduler(
            max_depth=data.get('max_depth', 4),
            speed_weight=data.get('speed_weight', 0.5),
            cost_weight=data.get('cost_weight', 0.5)
        )

        minimax_result = scheduler.find_optimal_schedule(initial_state)

        # Optionally run naive
        naive_result = None
        if data.get('include_naive_comparison', True):
            naive_result = naive_schedule(initial_state)

        response = {
            "optimal_schedule": minimax_result['schedule'],
            "metrics": minimax_result['metrics'],
            "search_stats": minimax_result['search_stats'],
            "decision_tree": minimax_result['decision_tree'],
            "weights": {
                "speed": data.get('speed_weight', 0.5),
                "cost": data.get('cost_weight', 0.5)
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

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting GPU Cost Optimization API on http://localhost:8000")
    print("Visit http://localhost:8000 for API info")
    app.run(host='0.0.0.0', port=8000, debug=True)
