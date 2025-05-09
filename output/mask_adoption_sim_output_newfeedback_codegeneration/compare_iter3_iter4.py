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

# Iteration 3 result files
baseline_iter3_file = os.path.join(OUTPUT_DIR, "baseline_results.csv")
high_risk_iter3_file = os.path.join(OUTPUT_DIR, "high_risk_results.csv")
intervention_iter3_file = os.path.join(OUTPUT_DIR, "intervention_results.csv")

# Iteration 4 result files
baseline_iter4_file = os.path.join(OUTPUT_DIR, "baseline_results_iter4.csv")
high_risk_iter4_file = os.path.join(OUTPUT_DIR, "high_risk_results_iter4.csv")
intervention_iter4_file = os.path.join(OUTPUT_DIR, "intervention_results_iter4.csv")

def compare_models():
    """
    Compare the performance of iter3 and iter4 models against test data.
    Generate numerical comparisons and visualizations.
    """
    print("\n============= COMPARISON OF ITER3 AND ITER4 MODELS =============\n")
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = pd.read_csv(test_file)
        test_data = test_data[(test_data['day'] >= 30) & (test_data['day'] <= 39)]
        print(f"Loaded {len(test_data)} test data points for days 30-39")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Load iter3 model results
    print("\nLoading iter3 results...")
    try:
        baseline_iter3 = pd.read_csv(baseline_iter3_file)
        high_risk_iter3 = pd.read_csv(high_risk_iter3_file)
        intervention_iter3 = pd.read_csv(intervention_iter3_file)
        
        # Add scenario and iteration columns
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
        return
    
    # Load iter4 model results
    print("\nLoading iter4 results...")
    try:
        baseline_iter4 = pd.read_csv(baseline_iter4_file)
        high_risk_iter4 = pd.read_csv(high_risk_iter4_file)
        intervention_iter4 = pd.read_csv(intervention_iter4_file)
        
        # Add scenario and iteration columns
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
    
    # Scenarios to evaluate
    scenarios = ['baseline', 'high_risk', 'intervention']
    
    # Dictionary to store metrics for each iteration and scenario
    iter3_metrics = {scenario: {} for scenario in scenarios}
    iter4_metrics = {scenario: {} for scenario in scenarios}
    
    # Evaluate iter3 models
    print("\n=== EVALUATING ITER3 MODELS ===")
    for scenario in scenarios:
        print(f"\nEvaluating {scenario} scenario (iter3)")
        
        # Get results for this scenario
        scenario_results = all_iter3_results[all_iter3_results['scenario'] == scenario]
        
        # Prepare for aggregate metrics
        all_true_values = []
        all_pred_values = []
        
        # Aggregate metrics across all days
        for day in range(30, 40):  # Days 30-39
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
    
    # Evaluate iter4 models
    print("\n=== EVALUATING ITER4 MODELS ===")
    for scenario in scenarios:
        print(f"\nEvaluating {scenario} scenario (iter4)")
        
        # Get results for this scenario
        scenario_results = all_iter4_results[all_iter4_results['scenario'] == scenario]
        
        # Prepare for aggregate metrics
        all_true_values = []
        all_pred_values = []
        
        # Aggregate metrics across all days
        for day in range(30, 40):  # Days 30-39
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
            
            iter4_metrics[scenario] = aggregate
            
            # Print aggregate metrics
            print(f"Aggregate metrics for {scenario} (iter4):")
            print(f"  Accuracy:  {aggregate['accuracy']:.4f}")
            print(f"  Precision: {aggregate['precision']:.4f}")
            print(f"  Recall:    {aggregate['recall']:.4f}")
            print(f"  F1 Score:  {aggregate['f1']:.4f}")
            print(f"  Actual Positive Rate:    {aggregate['actual_positive_rate']:.4f}")
            print(f"  Predicted Positive Rate: {aggregate['predicted_positive_rate']:.4f}")
    
    # Generate visualizations for comparison
    print("\nGenerating comparison visualizations...")
    
    # 1. Create metrics comparison by scenario
    for scenario in scenarios:
        if scenario not in iter3_metrics or scenario not in iter4_metrics:
            continue
            
        plt.figure(figsize=(10, 6))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        iter3_values = [iter3_metrics[scenario][m] for m in metrics_to_plot]
        iter4_values = [iter4_metrics[scenario][m] for m in metrics_to_plot]
        
        plt.bar(x - width/2, iter3_values, width, label='Iter3', color='lightblue')
        plt.bar(x + width/2, iter4_values, width, label='Iter4', color='darkblue')
        
        plt.ylabel('Score')
        plt.title(f'{scenario.capitalize()} - Iter3 vs Iter4 Performance Metrics')
        plt.xticks(x, metrics_labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(iter3_values):
            plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
        for i, v in enumerate(iter4_values):
            plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'compare_{scenario}_metrics.png'))
    
    # 2. Create prediction rate comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    actual_rates = [iter3_metrics[s]['actual_positive_rate'] for s in scenarios if s in iter3_metrics]
    iter3_rates = [iter3_metrics[s]['predicted_positive_rate'] for s in scenarios if s in iter3_metrics]
    iter4_rates = [iter4_metrics[s]['predicted_positive_rate'] for s in scenarios if s in iter4_metrics]
    
    plt.bar(x - width, actual_rates, width, label='Actual', color='green')
    plt.bar(x, iter3_rates, width, label='Iter3', color='lightblue')
    plt.bar(x + width, iter4_rates, width, label='Iter4', color='darkblue')
    
    plt.ylabel('Mask Wearing Rate')
    plt.title('Mask Wearing Rates by Scenario')
    plt.xticks(x, [s.capitalize() for s in scenarios])
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'compare_wearing_rates.png'))
    
    # 3. Create F1 score comparison
    plt.figure(figsize=(10, 6))
    
    f1_iter3 = [iter3_metrics[s]['f1'] for s in scenarios if s in iter3_metrics]
    f1_iter4 = [iter4_metrics[s]['f1'] for s in scenarios if s in iter4_metrics]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    plt.bar(x - width/2, f1_iter3, width, label='Iter3', color='lightblue')
    plt.bar(x + width/2, f1_iter4, width, label='Iter4', color='darkblue')
    
    # Calculate and display improvement percentages
    for i, (s, f1_3, f1_4) in enumerate(zip(scenarios, f1_iter3, f1_iter4)):
        improvement = ((f1_4 - f1_3) / f1_3 * 100) if f1_3 > 0 else float('inf')
        plt.text(i, max(f1_3, f1_4) + 0.05, f'{improvement:.1f}% change', ha='center')
    
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison by Scenario')
    plt.xticks(x, [s.capitalize() for s in scenarios])
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(f1_iter3):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(f1_iter4):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'compare_f1_scores.png'))
    
    # 4. Create a summary table as CSV
    summary_data = []
    for scenario in scenarios:
        if scenario in iter3_metrics and scenario in iter4_metrics:
            iter3 = iter3_metrics[scenario]
            iter4 = iter4_metrics[scenario]
            
            iter3_error = abs(iter3['actual_positive_rate'] - iter3['predicted_positive_rate'])
            iter4_error = abs(iter4['actual_positive_rate'] - iter4['predicted_positive_rate'])
            error_improvement = ((iter3_error - iter4_error) / iter3_error * 100) if iter3_error > 0 else 0
            
            f1_improvement = ((iter4['f1'] - iter3['f1']) / iter3['f1'] * 100) if iter3['f1'] > 0 else 0
            
            summary_data.append({
                'scenario': scenario,
                'actual_rate': iter3['actual_positive_rate'],  # Same for both iterations
                'iter3_predicted_rate': iter3['predicted_positive_rate'],
                'iter4_predicted_rate': iter4['predicted_positive_rate'],
                'iter3_accuracy': iter3['accuracy'],
                'iter4_accuracy': iter4['accuracy'],
                'accuracy_improvement': iter4['accuracy'] - iter3['accuracy'],
                'iter3_f1': iter3['f1'],
                'iter4_f1': iter4['f1'],
                'f1_improvement': f1_improvement,
                'iter3_error': iter3_error,
                'iter4_error': iter4_error,
                'error_improvement': error_improvement
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'iter3_iter4_comparison.csv'), index=False)
    
    # Print summary of improvements
    print("\n=== SUMMARY OF IMPROVEMENTS FROM ITER3 TO ITER4 ===")
    for scenario in scenarios:
        if scenario in iter3_metrics and scenario in iter4_metrics:
            iter3 = iter3_metrics[scenario]
            iter4 = iter4_metrics[scenario]
            
            print(f"\nScenario: {scenario.upper()}")
            print(f"  Accuracy:  {iter3['accuracy']:.4f} → {iter4['accuracy']:.4f} " + 
                  f"({(iter4['accuracy']-iter3['accuracy'])*100:+.2f} percentage points)")
            print(f"  F1 Score:  {iter3['f1']:.4f} → {iter4['f1']:.4f} " + 
                  f"({(iter4['f1']-iter3['f1'])*100:+.2f} percentage points)")
            
            # Prediction error (difference from actual rate)
            iter3_error = abs(iter3['actual_positive_rate'] - iter3['predicted_positive_rate'])
            iter4_error = abs(iter4['actual_positive_rate'] - iter4['predicted_positive_rate'])
            error_improvement = ((iter3_error - iter4_error) / iter3_error * 100) if iter3_error > 0 else 0
            
            print(f"  Predicted Rate: {iter3['predicted_positive_rate']:.4f} → {iter4['predicted_positive_rate']:.4f}")
            print(f"  Prediction Error: {iter3_error:.4f} → {iter4_error:.4f} ({error_improvement:+.2f}%)")
    
    print(f"\nComparison visualizations and summary saved to {OUTPUT_DIR}")
    
    return iter3_metrics, iter4_metrics

if __name__ == "__main__":
    compare_models() 