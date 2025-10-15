# FIXED: Applied feedback snippet from simulation_code_iter_1.py
def main():
    # Restore orchestration: parameter registry, simulation init, run, evaluate
    from typing import Dict, Any

    # Example defaults; later load from file or CLI
    defaults: Dict[str, Any] = {
        "population_size": 1000,
        "time_horizon_days": 180,
        "seed": 42,
        "mandate_start_day": 30,
        "mandate_end_day": 120
    }

    # Initialize simulation engine (placeholder call)
    sim = build_simulation(defaults)  # implement build_simulation to construct modules and state
    sim.run(0, defaults["time_horizon_days"] - 1)
    metrics = sim.evaluate(window=(0, defaults["time_horizon_days"] - 1))
    sim.save_results("artifacts/results/simulation_history.json")

Error: Request timed out.

# Execute main for both direct execution and sandbox wrapper invocation
main()