import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Use the same paths as the simulation code
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Set output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/mask_adoption_sim_output_newfeedback_codegeneration")

# Data file paths
test_file = os.path.join(DATA_DIR, "test_data.csv")

# New iter4 result files
baseline_iter4_file = os.path.join(OUTPUT_DIR, "baseline_results_iter4.csv")
high_risk_iter4_file = os.path.join(OUTPUT_DIR, "high_risk_results_iter4.csv")
intervention_iter4_file = os.path.join(OUTPUT_DIR, "intervention_results_iter4.csv")

# Original iter3 result files for comparison
baseline_iter3_file = os.path.join(OUTPUT_DIR, "baseline_results.csv")
high_risk_iter3_file = os.path.join(OUTPUT_DIR, "high_risk_results.csv")
intervention_iter3_file = os.path.join(OUTPUT_DIR, "intervention_results.csv")

def evaluate_model():
    """
    Evaluate the improved iter4 model performance against test data for days 30-39.
    Calculate accuracy, precision, recall, F1 score and other relevant metrics.
    Compare with the previous iter3 model.
    """
    print("\n============ MASK ADOPTION MODEL EVALUATION - ITER4 VS ITER3 (DAYS 30-39) ============\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = pd.read_csv(test_file)
        print(f"Loaded {len(test_data)} test data points")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Filter test data to only include days 30-39
    test_data = test_data[(test_data['day'] >= 30) & (test_data['day'] <= 39)]
    print(f"Filtered to {len(test_data)} test data points for days 30-39")
    
    # Load iter4 model results
    try:
        baseline_iter4 = pd.read_csv(baseline_iter4_file)
        high_risk_iter4 = pd.read_csv(high_risk_iter4_file)
        intervention_iter4 = pd.read_csv(intervention_iter4_file)
        
        # Add scenario column for identification
        baseline_iter4['scenario'] = 'baseline'
        high_risk_iter4['scenario'] = 'high_risk'
        intervention_iter4['scenario'] = 'intervention'
        baseline_iter4['iteration'] = 'iter4'
        high_risk_iter4['iteration'] = 'iter4'
        intervention_iter4['iteration'] = 'iter4'
        
        # Combine results
        all_iter4_results = pd.concat([baseline_iter4, high_risk_iter4, intervention_iter4])
        print(f"Loaded {len(all_iter4_results)} model predictions for iter4")
        
    except Exception as e:
        print(f"Error loading iter4 model predictions: {e}")
        return
    
    # Load iter3 model results for comparison
    try:
        baseline_iter3 = pd.read_csv(baseline_iter3_file)
        high_risk_iter3 = pd.read_csv(high_risk_iter3_file)
        intervention_iter3 = pd.read_csv(intervention_iter3_file)
        
        # Add scenario column for identification
        baseline_iter3['scenario'] = 'baseline'
        high_risk_iter3['scenario'] = 'high_risk'
        intervention_iter3['scenario'] = 'intervention'
        baseline_iter3['iteration'] = 'iter3'
        high_risk_iter3['iteration'] = 'iter3'
        intervention_iter3['iteration'] = 'iter3'
        
        # Combine results
        all_iter3_results = pd.concat([baseline_iter3, high_risk_iter3, intervention_iter3])
        print(f"Loaded {len(all_iter3_results)} model predictions for iter3")
        
    except Exception as e:
        print(f"Error loading iter3 model predictions: {e}")
        all_iter3_results = None
    
    # Extract all days we need to evaluate
    days = sorted(test_data['day'].unique())
    print(f"Evaluating model across days: {', '.join(map(str, days))}")
    
    # Scenarios to evaluate
    scenarios = ['baseline', 'high_risk', 'intervention']
    
    # Dictionary to store aggregate metrics for each model iteration and scenario
    iter4_metrics = {scenario: {} for scenario in scenarios}
    iter3_metrics = {scenario: {} for scenario in scenarios}
    
    # Evaluate iter4 models
    print("\n=== EVALUATING ITER4 MODELS ===")
    for scenario in scenarios:
        print(f"\nEvaluating {scenario} scenario")
        
        # Get results for this scenario
        scenario_results = all_iter4_results[all_iter4_results['scenario'] == scenario]
        
        # Prepare for aggregate metrics
        all_true_values = []
        all_pred_values = []
        
        # Evaluate day by day
        daily_metrics = {}
        for day in days:
            # Get test data for this day
            day_test_data = test_data[test_data['day'] == day]
            
            # Merge with model predictions
            evaluation_data = pd.merge(
                day_test_data[['agent_id', 'wearing_mask']],
                scenario_results[['agent_id', 'mask_wearing_status']],
                on='agent_id',
                how='inner'
            )
            
            if len(evaluation_data) == 0:
                print(f"  No matching data found for day {day}")
                continue
                
            # Convert boolean to int for metrics calculation
            if evaluation_data['wearing_mask'].dtype == bool:
                evaluation_data['wearing_mask'] = evaluation_data['wearing_mask'].astype(int)
            if evaluation_data['mask_wearing_status'].dtype == bool:
                evaluation_data['mask_wearing_status'] = evaluation_data['mask_wearing_status'].astype(int)
            
            # Calculate metrics for this day
            y_true = evaluation_data['wearing_mask']
            y_pred = evaluation_data['mask_wearing_status']
            
            # Add to aggregates
            all_true_values.extend(y_true.tolist())
            all_pred_values.extend(y_pred.tolist())
            
            # Store daily metrics
            daily_metrics[day] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'actual_positive_rate': y_true.mean(),
                'predicted_positive_rate': y_pred.mean()
            }
            
        # Calculate aggregate metrics
        if all_true_values and all_pred_values:
            all_true = np.array(all_true_values)
            all_pred = np.array(all_pred_values)
            
            aggregate = {
                'accuracy': accuracy_score(all_true, all_pred),
                'precision': precision_score(all_true, all_pred, zero_division=0),
                'recall': recall_score(all_true, all_pred, zero_division=0),
                'f1': f1_score(all_true, all_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(all_true, all_pred),
                'actual_positive_rate': all_true.mean(),
                'predicted_positive_rate': all_pred.mean(),
                'total_samples': len(all_true),
                'daily_metrics': daily_metrics
            }
            
            iter4_metrics[scenario] = aggregate
            
            # Print aggregate metrics
            print(f"Aggregate metrics for {scenario} (iter4):")
            print(f"  Accuracy:  {aggregate['accuracy']:.4f}")
            print(f"  Precision: {aggregate['precision']:.4f}")
            print(f"  Recall:    {aggregate['recall']:.4f}")
            print(f"  F1 Score:  {aggregate['f1']:.4f}")
            print(f"  Actual Positive Rate:    {aggregate['actual_positive_rate']:.4f}")
            print(f"  Predicted Positive Rate: {aggregate['predicted_positive_rate']:.4f}")
    
    # Evaluate iter3 models if available
    if all_iter3_results is not None:
        print("\n=== EVALUATING ITER3 MODELS FOR COMPARISON ===")
        for scenario in scenarios:
            print(f"\nEvaluating {scenario} scenario")
            
            # Get results for this scenario
            scenario_results = all_iter3_results[all_iter3_results['scenario'] == scenario]
            
            # Prepare for aggregate metrics
            all_true_values = []
            all_pred_values = []
            
            # Aggregate metrics across all days
            for day in days:
                # Get test data for this day
                day_test_data = test_data[test_data['day'] == day]
                
                # Merge with model predictions
                evaluation_data = pd.merge(
                    day_test_data[['agent_id', 'wearing_mask']],
                    scenario_results[['agent_id', 'mask_wearing_status']],
                    on='agent_id',
                    how='inner'
                )
                
                if len(evaluation_data) == 0:
                    continue
                    
                # Convert boolean to int for metrics calculation
                if evaluation_data['wearing_mask'].dtype == bool:
                    evaluation_data['wearing_mask'] = evaluation_data['wearing_mask'].astype(int)
                if evaluation_data['mask_wearing_status'].dtype == bool:
                    evaluation_data['mask_wearing_status'] = evaluation_data['mask_wearing_status'].astype(int)
                
                # Calculate metrics for this day
                y_true = evaluation_data['wearing_mask']
                y_pred = evaluation_data['mask_wearing_status']
                
                # Add to aggregates
                all_true_values.extend(y_true.tolist())
                all_pred_values.extend(y_pred.tolist())
            
            # Calculate aggregate metrics
            if all_true_values and all_pred_values:
                all_true = np.array(all_true_values)
                all_pred = np.array(all_pred_values)
                
                aggregate = {
                    'accuracy': accuracy_score(all_true, all_pred),
                    'precision': precision_score(all_true, all_pred, zero_division=0),
                    'recall': recall_score(all_true, all_pred, zero_division=0),
                    'f1': f1_score(all_true, all_pred, zero_division=0),
                    'confusion_matrix': confusion_matrix(all_true, all_pred),
                    'actual_positive_rate': all_true.mean(),
                    'predicted_positive_rate': all_pred.mean(),
                    'total_samples': len(all_true)
                }
                
                iter3_metrics[scenario] = aggregate
                
                # Print aggregate metrics
                print(f"Aggregate metrics for {scenario} (iter3):")
                print(f"  Accuracy:  {aggregate['accuracy']:.4f}")
                print(f"  Precision: {aggregate['precision']:.4f}")
                print(f"  Recall:    {aggregate['recall']:.4f}")
                print(f"  F1 Score:  {aggregate['f1']:.4f}")
                print(f"  Actual Positive Rate:    {aggregate['actual_positive_rate']:.4f}")
                print(f"  Predicted Positive Rate: {aggregate['predicted_positive_rate']:.4f}")
    
    # Generate visualizations
    print("\nGenerating evaluation visualizations...")
    
    # Compare iterations for each scenario
    plt.figure(figsize=(15, 10))
    for i, scenario in enumerate(scenarios, 1):
        plt.subplot(3, 1, i)
        
        # Check if both iterations have data for this scenario
        has_iter3 = scenario in iter3_metrics and iter3_metrics[scenario]
        has_iter4 = scenario in iter4_metrics and iter4_metrics[scenario]
        
        if not (has_iter3 and has_iter4):
            continue
        
        # Data for plotting
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        keys = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        iter3_values = [iter3_metrics[scenario][k] for k in keys]
        iter4_values = [iter4_metrics[scenario][k] for k in keys]
        
        plt.bar(x - width/2, iter3_values, width, label='Iter3', color='lightblue')
        plt.bar(x + width/2, iter4_values, width, label='Iter4', color='darkblue')
        
        plt.title(f'{scenario.capitalize()} - Iter3 vs Iter4')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(iter3_values):
            plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
        for i, v in enumerate(iter4_values):
            plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iter3_vs_iter4_metrics.png'))
    
    # Compare actual vs predicted rates
    plt.figure(figsize=(15, 6))
    
    actual_rates = []
    iter3_rates = []
    iter4_rates = []
    
    for scenario in scenarios:
        if scenario in iter3_metrics and scenario in iter4_metrics:
            actual_rates.append(iter3_metrics[scenario]['actual_positive_rate'])  # Same as iter4's actual rate
            iter3_rates.append(iter3_metrics[scenario]['predicted_positive_rate'])
            iter4_rates.append(iter4_metrics[scenario]['predicted_positive_rate'])
    
    scenarios_nice = [s.capitalize() for s in scenarios if s in iter3_metrics and s in iter4_metrics]
    
    x = np.arange(len(scenarios_nice))
    width = 0.25
    
    plt.bar(x - width, actual_rates, width, label='Actual Rate', color='green')
    plt.bar(x, iter3_rates, width, label='Iter3 Predicted', color='lightblue')
    plt.bar(x + width, iter4_rates, width, label='Iter4 Predicted', color='darkblue')
    
    plt.title('Mask Wearing Rates Comparison')
    plt.ylabel('Rate')
    plt.xticks(x, scenarios_nice)
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(actual_rates):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(iter3_rates):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(iter4_rates):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iter3_vs_iter4_rates.png'))
    
    # For baseline only: plot daily metrics over time
    if 'baseline' in iter4_metrics and iter4_metrics['baseline'] and 'daily_metrics' in iter4_metrics['baseline']:
        daily = iter4_metrics['baseline']['daily_metrics']
        
        plt.figure(figsize=(12, 6))
        
        days = sorted(daily.keys())
        actual_rates = [daily[day]['actual_positive_rate'] for day in days]
        predicted_rates = [daily[day]['predicted_positive_rate'] for day in days]
        
        plt.plot(days, actual_rates, 'o-', label='Actual Rate', color='green')
        plt.plot(days, predicted_rates, 's-', label='Iter4 Predicted Rate', color='darkblue')
        
        plt.title('Daily Mask Wearing Rates (Baseline Scenario)')
        plt.xlabel('Day')
        plt.ylabel('Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'baseline_daily_rates.png'))
    
    # Save summary results to CSV
    summary_data = []
    for scenario in scenarios:
        if scenario in iter3_metrics and scenario in iter4_metrics:
            iter3 = iter3_metrics[scenario]
            iter4 = iter4_metrics[scenario]
            
            iter3_error = abs(iter3['actual_positive_rate'] - iter3['predicted_positive_rate'])
            iter4_error = abs(iter4['actual_positive_rate'] - iter4['predicted_positive_rate'])
            error_change = iter3_error - iter4_error
            error_pct = (error_change / iter3_error) * 100 if iter3_error > 0 else 0
            
            summary_data.append({
                'scenario': scenario,
                'actual_rate': iter4['actual_positive_rate'],
                'iter3_predicted_rate': iter3['predicted_positive_rate'],
                'iter4_predicted_rate': iter4['predicted_positive_rate'],
                'iter3_accuracy': iter3['accuracy'],
                'iter4_accuracy': iter4['accuracy'],
                'iter3_precision': iter3['precision'],
                'iter4_precision': iter4['precision'],
                'iter3_recall': iter3['recall'],
                'iter4_recall': iter4['recall'],
                'iter3_f1': iter3['f1'],
                'iter4_f1': iter4['f1'],
                'iter3_error': iter3_error,
                'iter4_error': iter4_error,
                'error_improvement_pct': error_pct
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_summary.csv'), index=False)
    
    print(f"\nEvaluation visualizations and results saved to {OUTPUT_DIR}")
    
    # Print comparison summary
    if all_iter3_results is not None:
        print("\n=== IMPROVEMENT SUMMARY ===")
        for scenario in scenarios:
            if scenario in iter3_metrics and scenario in iter4_metrics:
                iter3 = iter3_metrics[scenario]
                iter4 = iter4_metrics[scenario]
                
                print(f"\n{scenario.upper()} SCENARIO:")
                print(f"  Accuracy:  {iter3['accuracy']:.4f} → {iter4['accuracy']:.4f} ({(iter4['accuracy']-iter3['accuracy'])*100:+.2f}%)")
                print(f"  Precision: {iter3['precision']:.4f} → {iter4['precision']:.4f} ({(iter4['precision']-iter3['precision'])*100:+.2f}%)")
                print(f"  Recall:    {iter3['recall']:.4f} → {iter4['recall']:.4f} ({(iter4['recall']-iter3['recall'])*100:+.2f}%)")
                print(f"  F1 Score:  {iter3['f1']:.4f} → {iter4['f1']:.4f} ({(iter4['f1']-iter3['f1'])*100:+.2f}%)")
                
                # Calculate prediction error (difference from actual)
                iter3_error = abs(iter3['actual_positive_rate'] - iter3['predicted_positive_rate'])
                iter4_error = abs(iter4['actual_positive_rate'] - iter4['predicted_positive_rate'])
                error_change = iter3_error - iter4_error
                error_pct = (error_change / iter3_error) * 100 if iter3_error > 0 else 0
                
                print(f"  Pred Rate: {iter3['predicted_positive_rate']:.4f} → {iter4['predicted_positive_rate']:.4f}")
                print(f"  Pred Error: {iter3_error:.4f} → {iter4_error:.4f} ({error_pct:+.2f}% improvement)")
    
    return iter3_metrics, iter4_metrics

if __name__ == "__main__":
    evaluate_model() 