# FIXED: Applied feedback snippet from simulation.py
def main():
    import os
    params = load_params(os.path.join(PROJECT_ROOT, 'parameters.json'))
    # Optional quick-test override
    if os.environ.get('QUICK_TEST', '0') == '1':
        params['population_size'] = 50
        params['simulation_days'] = 10
    params = normalize_params(params)  # new helper to map spec -> internal
    sim = Simulation(params, load_param_definitions(os.path.join(PROJECT_ROOT, 'parameter_definitions.json'), params))
    sim.run(0, sim.params.get('time_steps', 60))
    metrics = sim.evaluate()
    return {
        'final_adoption_rate': metrics.get('final_adoption_rate'),
        'time_to_threshold_70': metrics.get('time_to_threshold_70'),
        'adoption_rate_over_time': metrics.get('adoption_rate_over_time')
    }

Error: Request timed out.

# Execute main for both direct execution and sandbox wrapper invocation
main()