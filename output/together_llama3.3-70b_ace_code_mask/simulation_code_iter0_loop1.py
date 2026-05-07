# PLAYBOOK_USAGE_JSON = '''{"used_bullets":[{"id":"strategy-1","why":"reason here"}]}'''

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")

if PROJECT_ROOT is None or DATA_PATH is None:
    raise ValueError("PROJECT_ROOT and DATA_PATH environment variables must be set")

DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

def load_data():
    """
    Load the data from the CSV and JSON files.
    
    Returns:
    - agents (pd.DataFrame): Agent attributes.
    - network (dict): Social network.
    - train_data (pd.DataFrame): Training data.
    """
    try:
        agents = pd.read_csv(os.path.join(DATA_DIR, "agent_attributes.csv"))
        with open(os.path.join(DATA_DIR, "social_network.json")) as f:
            network = json.load(f)
        train_data = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
        return agents, network, train_data
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {e}")

def build_network_and_agents(agents, network, train_data):
    """
    Build the network and agents.
    
    Parameters:
    - agents (pd.DataFrame): Agent attributes.
    - network (dict): Social network.
    - train_data (pd.DataFrame): Training data.
    
    Returns:
    - agents (pd.DataFrame): Updated agent attributes.
    - network (dict): Updated social network.
    """
    # Align populations across files; use intersection of IDs
    agents = agents[agents["agent_id"].isin(train_data["agent_id"])]
    network = {agent_id: network.get(agent_id, {}) for agent_id in agents["agent_id"]}
    
    # Set day 0 mask state per agent from train_data day 0 if available; otherwise fallback to initial_mask_wearing
    agents["wearing_mask_t"] = agents.apply(lambda row: train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "wearing_mask"].iloc[0] if not np.isnan(train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "wearing_mask"].iloc[0]) else row["initial_mask_wearing"], axis=1)
    
    # Derive initial received_info from train_data day 0 if provided; else initialize false
    agents["received_info_t"] = agents.apply(lambda row: train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "received_info"].iloc[0] if not np.isnan(train_data.loc[(train_data["agent_id"] == row["agent_id"]) & (train_data["day"] == 0), "received_info"].iloc[0]) else False, axis=1)
    
    # Precompute per-agent layer degrees and normalization constants
    for layer in ["family", "work_school", "community"]:
        agents[f"{layer}_degrees"] = agents.apply(lambda row: len(network.get(row["agent_id"], {}).get(layer, [])), axis=1)
    
    return agents, network

def holdout_split(train_data):
    """
    Split the training data into training and validation sets.
    
    Parameters:
    - train_data (pd.DataFrame): Training data.
    
    Returns:
    - train_data (pd.DataFrame): Training set.
    - val_data (pd.DataFrame): Validation set.
    """
    unique_days = sorted(train_data["day"].unique())
    train_days = unique_days[:int(0.8 * len(unique_days))]
    val_days = unique_days[int(0.8 * len(unique_days)):]
    
    if not val_days:
        raise ValueError("No validation days available after temporal split")
    
    train_data = train_data[train_data["day"].isin(train_days)]
    val_data = train_data[train_data["day"].isin(val_days)]
    
    return train_data, val_data

class Calibrator:
    def __init__(self, agents, network, train_data):
        """
        Initialize the calibrator.
        
        Parameters:
        - agents (pd.DataFrame): Agent attributes.
        - network (dict): Social network.
        - train_data (pd.DataFrame): Training data.
        """
        self.agents = agents
        self.network = network
        self.train_data = train_data
    
    def fit(self):
        """
        Fit the calibrator.
        
        Returns:
        - parameters (dict): Calibrated parameters.
        """
        try:
            # Define the parameters to be calibrated
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
                "regularization": 0.1
            }
            
            # Calibrate the parameters using logistic regression
            X = self.train_data[["wearing_mask_t-1", "neighbor_mask_share_family", "neighbor_mask_share_work", "neighbor_mask_share_community", "risk_perception", "received_info_t"]]
            y = self.train_data["wearing_mask_t"]
            model = LogisticRegression(penalty="l2", C=parameters["regularization"], random_state=42)
            model.fit(X, y)
            
            # Update the parameters with the calibrated values
            parameters["alpha"] = model.intercept_
            parameters["beta_f"] = model.coef_[1]
            parameters["beta_w"] = model.coef_[2]
            parameters["beta_c"] = model.coef_[3]
            parameters["beta_r"] = model.coef_[4]
            parameters["beta_i"] = model.coef_[5]
            
            return parameters
        except Exception as e:
            raise ValueError(f"Error during calibration: {e}")

class Simulator:
    def __init__(self, agents, network, parameters):
        """
        Initialize the simulator.
        
        Parameters:
        - agents (pd.DataFrame): Agent attributes.
        - network (dict): Social network.
        - parameters (dict): Calibrated parameters.
        """
        self.agents = agents
        self.network = network
        self.parameters = parameters
    
    def rollout(self, val_data):
        """
        Roll out the simulation on the validation data.
        
        Parameters:
        - val_data (pd.DataFrame): Validation data.
        
        Returns:
        - simulated_data (pd.DataFrame): Simulated data.
        """
        simulated_data = []
        for day in range(len(val_data)):
            for agent_id in self.agents["agent_id"]:
                # Compute per-agent neighbor mask share per layer
                neighbor_mask_share_family = self.network.get(agent_id, {}).get("family", []).count(1) / len(self.network.get(agent_id, {}).get("family", [])) if self.network.get(agent_id, {}).get("family", []) else 0
                neighbor_mask_share_work = self.network.get(agent_id, {}).get("work_school", []).count(1) / len(self.network.get(agent_id, {}).get("work_school", [])) if self.network.get(agent_id, {}).get("work_school", []) else 0
                neighbor_mask_share_community = self.network.get(agent_id, {}).get("community", []).count(1) / len(self.network.get(agent_id, {}).get("community", [])) if self.network.get(agent_id, {}).get("community", []) else 0
                
                # Compute received_info via peer and exogenous channels
                received_info = self.parameters["phi_family"] * neighbor_mask_share_family + self.parameters["phi_work"] * neighbor_mask_share_work + self.parameters["phi_community"] * neighbor_mask_share_community + self.parameters["lambda_broadcast"]
                
                # Compute mask adoption probability using a calibrated logistic decision rule with inertia and demographics
                adoption_probability = 1 / (1 + np.exp(-self.parameters["alpha"] - self.parameters["gamma"] * self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t-1"].iloc[0] - self.parameters["beta_f"] * neighbor_mask_share_family - self.parameters["beta_w"] * neighbor_mask_share_work - self.parameters["beta_c"] * neighbor_mask_share_community - self.parameters["beta_r"] * self.agents.loc[self.agents["agent_id"] == agent_id, "risk_perception"].iloc[0] - self.parameters["beta_i"] * received_info))
                
                # Sample the wearing_mask state from the adoption probability
                wearing_mask = np.random.binomial(1, adoption_probability)
                
                # Update the agent's state
                self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t"] = wearing_mask
                self.agents.loc[self.agents["agent_id"] == agent_id, "received_info_t"] = received_info
                
                # Append the simulated data to the list
                simulated_data.append({
                    "agent_id": agent_id,
                    "day": day,
                    "wearing_mask": wearing_mask,
                    "received_info": received_info
                })
        
        return pd.DataFrame(simulated_data)

class Evaluator:
    def __init__(self, val_data, simulated_data):
        """
        Initialize the evaluator.
        
        Parameters:
        - val_data (pd.DataFrame): Validation data.
        - simulated_data (pd.DataFrame): Simulated data.
        """
        self.val_data = val_data
        self.simulated_data = simulated_data
    
    def compute_metrics(self):
        """
        Compute the evaluation metrics.
        
        Returns:
        - metrics (dict): Evaluation metrics.
        """
        try:
            metrics = {}
            metrics["RMSE_aggregate"] = np.sqrt(mean_squared_error(self.val_data["wearing_mask"], self.simulated_data["wearing_mask"]))
            metrics["MAE_aggregate"] = mean_absolute_error(self.val_data["wearing_mask"], self.simulated_data["wearing_mask"])
            metrics["Brier"] = brier_score_loss(self.val_data["wearing_mask"], self.simulated_data["wearing_mask"])
            metrics["TransitionFit"] = np.abs(self.val_data["wearing_mask"] - self.simulated_data["wearing_mask"]).mean()
            
            return metrics
        except Exception as e:
            raise ValueError(f"Error during evaluation: {e}")

def save_results(metrics, parameters):
    """
    Save the results to a file.
    
    Parameters:
    - metrics (dict): Evaluation metrics.
    - parameters (dict): Calibrated parameters.
    """
    try:
        with open("results.json", "w") as f:
            json.dump({
                "metrics": metrics,
                "parameters": parameters
            }, f)
    except Exception as e:
        raise ValueError(f"Error during saving results: {e}")

def main():
    agents, network, train_data = load_data()
    agents, network = build_network_and_agents(agents, network, train_data)
    train_data, val_data = holdout_split(train_data)
    calibrator = Calibrator(agents, network, train_data)
    parameters = calibrator.fit()
    simulator = Simulator(agents, network, parameters)
    simulated_data = simulator.rollout(val_data)
    evaluator = Evaluator(val_data, simulated_data)
    metrics = evaluator.compute_metrics()
    save_results(metrics, parameters)

main()