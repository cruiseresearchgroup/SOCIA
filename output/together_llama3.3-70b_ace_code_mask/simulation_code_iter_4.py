PLAYBOOK_USAGE_JSON = '''{
  "used_bullets": [
    {
      "id": "missing-neighbor-mask-share-calculation",
      "why": "Simulator.rollout incorrectly used agent's initial state for prev_mask_state instead of previous day's simulated state; this violates temporal causality and causes high RMSE/MAE/Brier. Fix requires pulling prev_mask_state from agent_states_by_day[prev_day]."
    },
    {
      "id": "beta-parameters-zeroed-out",
      "why": "Calibrator.fit excluded beta_f, beta_w, beta_c from optimization and forced phi_* = beta_*, nullifying peer influence. Blueprint requires independent calibration of beta_* (influence) and phi_* (diffusion). Must include beta_* in calib_param_names and remove forced sync."
    },
    {
      "id": "received-info-computation-error",
      "why": "received_info was computed as a linear sum, not as 1 - exp(-sum) as required by Blueprint. This caused values >1, distorting sigmoid probabilities and causing Brier score to be near 0.55 (random guessing). Must apply exponential transformation."
    }
  ]
}'''

CHANGE_SUMMARY_JSON = '''{
  "touched_symbols": [
    {
      "symbol": "Simulator.rollout",
      "reason": "Fixed three critical errors: (1) prev_mask_state now pulled from agent_states_by_day[prev_day] instead of agents dataframe; (2) received_info now computed as 1 - exp(-sum) with clamping to [0,1]; (3) neighbor mask share now correctly uses only previous day's simulated states for all layers."
    },
    {
      "symbol": "Calibrator.fit",
      "reason": "Added beta_f, beta_w, beta_c to calib_param_names for optimization; removed forced synchronization phi_* = beta_* to allow independent calibration of influence and diffusion parameters as required by Blueprint."
    }
  ],
  "applied_strategies": [
    {
      "id": "missing-neighbor-mask-share-calculation",
      "applied": true
    },
    {
      "id": "beta-parameters-zeroed-out",
      "applied": true
    },
    {
      "id": "received-info-computation-error",
      "applied": true
    }
  ]
}'''

import os
import json
import argparse
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss
from scipy.optimize import minimize

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")

if PROJECT_ROOT is None or DATA_PATH is None:
    raise ValueError("PROJECT_ROOT and DATA_PATH environment variables must be set")

DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

def load_data():
    agents = pd.read_csv(os.path.join(DATA_DIR, "agent_attributes.csv"))
    with open(os.path.join(DATA_DIR, "social_network.json")) as f:
        network = json.load(f)
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    return agents, network, train_data, test_data

def build_network_and_agents(agents, network, train_data):
    agents = agents[agents["agent_id"].isin(train_data["agent_id"])]
    network = {agent_id: network.get(agent_id, {}) for agent_id in agents["agent_id"]}
    
    agents["wearing_mask_t"] = agents.apply(lambda row: train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "wearing_mask"].iloc[0] if not np.isnan(train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "wearing_mask"].iloc[0]) else row["initial_mask_wearing"], axis=1).astype(int)
    
    agents["received_info_t"] = agents.apply(lambda row: train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "received_info"].iloc[0] if not np.isnan(train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "received_info"].iloc[0]) else 0.0, axis=1).astype(float)
    
    agents["wearing_mask_t-1"] = agents["wearing_mask_t"].astype(int)
    
    for layer in ["family", "work_school", "community"]:
        agents[f"{layer}_degrees"] = agents.apply(lambda row: len(network.get(row["agent_id"], {}).get(layer, [])), axis=1)
    
    return agents, network

def prepare_train_features(train_data, agents, network):
    df = train_data.copy()

    df = df.rename(columns={"wearing_mask": "wearing_mask_t", "received_info": "received_info_t"})

    df = df.merge(agents[["agent_id", "risk_perception"]], on="agent_id", how="left")

    df = df.sort_values(["agent_id", "day"])
    df["wearing_mask_t-1"] = df.groupby("agent_id")["wearing_mask_t"].shift(1).fillna(0).astype(int)

    def _neighbor_mask_share(row, layer):
        neighbors = network.get(row["agent_id"], {}).get(layer, [])
        if not neighbors:
            return 0.0
        prev_day = row["day"] - 1
        if prev_day < 0:
            return 0.0
        neighbor_rows = df[(df["agent_id"].isin(neighbors)) & (df["day"] == prev_day)]
        if neighbor_rows.empty:
            return 0.0
        return neighbor_rows["wearing_mask_t"].mean()

    for layer_key, col_name in [("family", "neighbor_mask_share_family"),
                                 ("work_school", "neighbor_mask_share_work"),
                                 ("community", "neighbor_mask_share_community")]:
        df[col_name] = df.apply(lambda row, lk=layer_key: _neighbor_mask_share(row, lk), axis=1)

    return df


def holdout_split(train_data):
    unique_days = sorted(train_data["day"].unique())
    train_days = unique_days[:int(0.8 * len(unique_days))]
    val_days = unique_days[int(0.8 * len(unique_days)):]
    
    if not val_days:
        raise ValueError("No validation days available after temporal split")
    
    t_data = train_data[train_data["day"].isin(train_days)]
    val_data = train_data[train_data["day"].isin(val_days)]
    
    return t_data, val_data

class Calibrator:
    def __init__(self, agents, network, train_data, val_data):
        """
        Calibrator that first fits a logistic regression on training data to get
        a reasonable initialization, then refines a small set of behavioral
        parameters using simulation-based calibration on a validation split.
        """
        self.agents = agents
        self.network = network
        self.train_data = train_data
        self.val_data = val_data
    
    def _fit_logistic_baseline(self, parameters):
        """
        Fit a logistic regression model on training data to initialize
        alpha/beta_* coefficients.
        """
        feature_cols = [
            "wearing_mask_t-1",
            "neighbor_mask_share_family",
            "neighbor_mask_share_work",
            "neighbor_mask_share_community",
            "risk_perception",
            "received_info_t",
        ]
        X = self.train_data[feature_cols]
        y = self.train_data["wearing_mask_t"]
        model = LogisticRegression(
            penalty="l2",
            C=parameters["regularization"],
            random_state=42,
        )
        model.fit(X, y)
        coefs = model.coef_[0]
        parameters["alpha"] = float(model.intercept_[0])
        parameters["beta_f"] = float(coefs[1])
        parameters["beta_w"] = float(coefs[2])
        parameters["beta_c"] = float(coefs[3])
        parameters["beta_r"] = float(coefs[4])
        parameters["beta_i"] = float(coefs[5])
        # initial mapping from social influence coefficients to phi_*
        parameters["phi_family"] = parameters["beta_f"]
        parameters["phi_work"] = parameters["beta_w"]
        parameters["phi_community"] = parameters["beta_c"]
        return parameters
    
    def fit(self):
        """
        Calibrate parameters using a two-stage procedure:
        1) Fit a logistic regression on training data to initialize alpha/beta_*.
        2) Use random search + local optimization (scipy.optimize.minimize)
           to minimize RMSE + MAE on a validation split via simulation.
        """
        # 1) Baseline parameter dictionary
        parameters = {
            "w_family": 0.5,
            "w_work": 0.3,
            "w_community": 0.2,
            "beta_f": 1.0,
            "beta_w": 1.0,
            "beta_c": 1.0,
            "gamma": 0.5,
            "beta_r": 1.0,
            "beta_i": 1.0,
            "lambda_broadcast": 0.1,
            "phi_family": 1.0,
            "phi_work": 1.0,
            "phi_community": 1.0,
            "rho_info_decay": 0.5,
            "alpha": 0.0,
            "beta_age_youth": 0.0,
            "beta_age_young_adult": 0.0,
            "beta_age_middle_age": 0.0,
            "beta_age_senior": 0.0,
            "beta_occ_student": 0.0,
            "beta_occ_blue_collar": 0.0,
            "beta_occ_white_collar": 0.0,
            "tau": 1.0,
            "regularization": 0.1,
        }
        
        # 2) Initialize using logistic regression on training data
        parameters = self._fit_logistic_baseline(parameters)
        
        # Select a small subset of parameters for simulation-based calibration
        # FIXED: Include beta_f, beta_w, beta_c in optimization; remove forced phi_* = beta_* constraint
        calib_param_names = [
            "alpha",
            "beta_f",
            "beta_w",
            "beta_c",
            "beta_r",
            "beta_i",
            "lambda_broadcast",
            "phi_family",
            "phi_work",
            "phi_community",
        ]
        x0 = np.array([parameters[name] for name in calib_param_names], dtype=float)
        
        # Define objective: RMSE + MAE on validation set from a full rollout
        def objective(x: np.ndarray) -> float:
            # Build a working copy of parameters
            trial_params = parameters.copy()
            for name, value in zip(calib_param_names, x):
                trial_params[name] = float(value)
            # Do NOT force phi_* = beta_* anymore — they are now independent parameters
            # This allows separate calibration of influence (beta) and diffusion (phi) as per Blueprint
            
            # Use a deep copy of agents so each evaluation is independent
            agents_copy = self.agents.copy(deep=True)
            
            # Make simulation deterministic per-evaluation for a smooth objective
            np.random.seed(42)
            simulator = Simulator(agents_copy, self.network, trial_params)
            simulated_data = simulator.rollout(self.val_data)
            evaluator = Evaluator(self.val_data, simulated_data)
            metrics = evaluator.compute_metrics()
            rmse = float(metrics.get("RMSE_aggregate", 1e9))
            mae = float(metrics.get("MAE_aggregate", 1e9))
            return rmse + mae
        
        # 3) Coarse random search around x0 to find a good starting point
        best_x = x0.copy()
        best_obj = objective(best_x)
        initial_obj = best_obj
        print(f"Initial objective (RMSE+MAE): {best_obj:.6f}")
        
        # Random search radius for each parameter (relative to initial value)
        # Use larger search scales and more trials
        search_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0], dtype=float)
        # Scale by absolute value to handle both positive and negative parameters
        param_scales = np.abs(x0) + 0.1  # Add small value to avoid zero scale
        search_scale_abs = search_scale * param_scales
        
        n_random_trials = 50  # Increased from 10
        improvements = 0
        for trial in range(n_random_trials):
            # Use uniform random search in a wider range
            noise = np.random.uniform(low=-search_scale_abs, high=search_scale_abs)
            cand_x = x0 + noise
            # Ensure lambda_broadcast stays positive
            if cand_x[6] < 0:
                cand_x[6] = 0.01
            # Ensure phi_* stay non-negative
            for i, name in enumerate(calib_param_names):
                if name.startswith("phi_") and cand_x[i] < 0:
                    cand_x[i] = 0.0
            cand_obj = objective(cand_x)
            if cand_obj < best_obj:
                improvements += 1
                best_obj = cand_obj
                best_x = cand_x.copy()
                if trial % 10 == 0:  # Print progress every 10 trials
                    print(f"  Random search trial {trial}: improved to {best_obj:.6f}")
        
        print(f"Random search completed: {improvements} improvements, best objective: {best_obj:.6f} (improvement: {initial_obj - best_obj:.6f})")
        
        # 4) Local refinement using Nelder-Mead with more iterations
        try:
            result = minimize(
                objective,
                best_x,
                method="Nelder-Mead",
                options={"maxiter": 100, "disp": False, "xatol": 1e-4, "fatol": 1e-4},
            )
            if result.success:
                if result.fun < best_obj:
                    print(f"Nelder-Mead optimization improved objective to {result.fun:.6f}")
                    best_x = result.x
                    best_obj = float(result.fun)
                else:
                    print(f"Nelder-Mead optimization did not improve (best: {best_obj:.6f}, result: {result.fun:.6f})")
            else:
                print(f"Nelder-Mead optimization did not converge, using best from random search")
        except Exception as opt_err:
            # If optimization fails, fall back to the best point from random search
            print(f"Warning: scipy.optimize.minimize failed during calibration: {opt_err}")
        
        final_improvement = initial_obj - best_obj
        print(f"Final calibration: objective improved by {final_improvement:.6f} ({100 * final_improvement / initial_obj:.2f}% relative improvement)")
        
        # 5) Write back the calibrated parameters
        for name, value in zip(calib_param_names, best_x):
            parameters[name] = float(value)
        
        return parameters

class Simulator:
    def __init__(self, agents, network, parameters):
        self.agents = agents
        self.network = network
        self.parameters = parameters
    
    def rollout(self, val_data):
        """
        Simulate mask adoption for each (agent_id, day) pair in val_data.
        Ensures output has the same rows and order as input data.
        """
        simulated_data = []
        # Track agent states by day for neighbor calculations
        agent_states_by_day = {}
        
        # Sort by day and agent_id to ensure consistent processing order
        val_data_sorted = val_data.sort_values(["day", "agent_id"]).copy()
        val_days = sorted(val_data_sorted["day"].unique())
        
        for day in val_days:
            day_data = val_data_sorted[val_data_sorted["day"] == day]
            
            for _, row in day_data.iterrows():
                agent_id = row["agent_id"]
                
                # Calculate neighbor mask share from previous day's simulated state
                neighbor_mask_share_family = 0.0
                neighbor_mask_share_work = 0.0
                neighbor_mask_share_community = 0.0
                
                neighbors = self.network.get(agent_id, {})
                prev_day = day - 1
                
                if prev_day >= 0 and prev_day in agent_states_by_day:
                    prev_states = agent_states_by_day[prev_day]
                    
                    family_neighbors = neighbors.get("family", [])
                    if family_neighbors:
                        family_mask_count = sum(prev_states.get(nid, 0) for nid in family_neighbors)
                        neighbor_mask_share_family = family_mask_count / len(family_neighbors)
                    
                    work_neighbors = neighbors.get("work_school", [])
                    if work_neighbors:
                        work_mask_count = sum(prev_states.get(nid, 0) for nid in work_neighbors)
                        neighbor_mask_share_work = work_mask_count / len(work_neighbors)
                    
                    community_neighbors = neighbors.get("community", [])
                    if community_neighbors:
                        community_mask_count = sum(prev_states.get(nid, 0) for nid in community_neighbors)
                        neighbor_mask_share_community = community_mask_count / len(community_neighbors)
                
                # FIXED: Compute received_info using 1 - exp(-sum) as per Blueprint
                # Previously: linear sum → now: exponential decay form
                weighted_sum = (
                    self.parameters["phi_family"] * neighbor_mask_share_family +
                    self.parameters["phi_work"] * neighbor_mask_share_work +
                    self.parameters["phi_community"] * neighbor_mask_share_community +
                    self.parameters["lambda_broadcast"]
                )
                received_info = 1 - np.exp(-weighted_sum)
                received_info = np.clip(received_info, 0, 1)  # Clamp to [0,1] as required
                
                # FIXED: Get current agent's previous mask state from agent_states_by_day[prev_day][agent_id], NOT from agents dataframe
                prev_mask_state = 0
                if prev_day >= 0 and prev_day in agent_states_by_day and agent_id in agent_states_by_day[prev_day]:
                    prev_mask_state = agent_states_by_day[prev_day][agent_id]
                else:
                    # Fallback: use initial state from agents if no prior simulation state exists
                    prev_mask_state = self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t-1"].iloc[0] if len(self.agents[self.agents["agent_id"] == agent_id]) > 0 else 0
                
                risk_perception = self.agents.loc[self.agents["agent_id"] == agent_id, "risk_perception"].iloc[0] if len(self.agents[self.agents["agent_id"] == agent_id]) > 0 else 0.0
                
                # Compute adoption probability using logistic model
                adoption_probability = 1 / (1 + np.exp(-self.parameters["alpha"] - self.parameters["gamma"] * prev_mask_state - self.parameters["beta_f"] * neighbor_mask_share_family - self.parameters["beta_w"] * neighbor_mask_share_work - self.parameters["beta_c"] * neighbor_mask_share_community - self.parameters["beta_r"] * risk_perception - self.parameters["beta_i"] * received_info))
                
                wearing_mask = np.random.binomial(1, adoption_probability)
                
                # Update agent state for next day
                if len(self.agents[self.agents["agent_id"] == agent_id]) > 0:
                    self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t"] = wearing_mask
                    self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t-1"] = wearing_mask
                    self.agents.loc[self.agents["agent_id"] == agent_id, "received_info_t"] = received_info
                
                # Record state for next day's neighbor calculations
                if day not in agent_states_by_day:
                    agent_states_by_day[day] = {}
                agent_states_by_day[day][agent_id] = wearing_mask
                
                simulated_data.append({
                    "agent_id": agent_id,
                    "day": day,
                    "wearing_mask": wearing_mask,
                    "received_info": received_info
                })
        
        return pd.DataFrame(simulated_data)

class Evaluator:
    def __init__(self, val_data, simulated_data):
        self.val_data = val_data
        self.simulated_data = simulated_data
    
    def compute_metrics(self):
        """
        Compute evaluation metrics. Ensures data alignment by merging on (agent_id, day).
        """
        val_col = "wearing_mask_t" if "wearing_mask_t" in self.val_data.columns else "wearing_mask"
        
        # Merge to ensure alignment on (agent_id, day)
        val_merged = self.val_data[["agent_id", "day", val_col]].copy()
        sim_merged = self.simulated_data[["agent_id", "day", "wearing_mask"]].copy()
        merged = val_merged.merge(sim_merged, on=["agent_id", "day"], how="inner")
        
        if len(merged) == 0:
            raise ValueError("No matching rows found between val_data and simulated_data after merging on (agent_id, day)")
        
        y_true = merged[val_col].values
        y_pred = merged["wearing_mask"].values
        
        metrics = {}
        metrics["RMSE_aggregate"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["MAE_aggregate"] = mean_absolute_error(y_true, y_pred)
        metrics["Brier"] = brier_score_loss(y_true, y_pred)
        metrics["TransitionFit"] = np.abs(y_true - y_pred).mean()
        
        return metrics

def save_results(test_metrics, val_metrics, parameters, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump({
            "metrics": test_metrics,
            "val_metrics": val_metrics,
            "parameters": parameters
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run mask adoption simulation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results.json (defaults to DATA_DIR if not provided)"
    )
    args = parser.parse_args()
    
    agents, network, train_data, test_data = load_data()
    agents, network = build_network_and_agents(agents, network, train_data)
    featured_data = prepare_train_features(train_data, agents, network)
    train_split, val_data = holdout_split(featured_data)
    
    # Calibrate parameters using train + validation
    calibrator = Calibrator(agents, network, train_split, val_data)
    parameters = calibrator.fit()
    
    # Evaluate on validation set
    val_simulator = Simulator(agents.copy(deep=True), network, parameters)
    val_simulated = val_simulator.rollout(val_data)
    val_evaluator = Evaluator(val_data, val_simulated)
    val_metrics = val_evaluator.compute_metrics()
    
    # Prepare test data features (same feature engineering as training data)
    test_featured = prepare_train_features(test_data, agents, network)
    
    # Evaluate on test set using the calibrated parameters
    test_simulator = Simulator(agents.copy(deep=True), network, parameters)
    test_simulated = test_simulator.rollout(test_featured)
    test_evaluator = Evaluator(test_featured, test_simulated)
    test_metrics = test_evaluator.compute_metrics()
    
    save_results(test_metrics, val_metrics, parameters, output_dir=args.output_dir)


# Execute main for both direct execution and sandbox wrapper invocation
main()