import os
import json
import argparse
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
    agents = pd.read_csv(os.path.join(DATA_DIR, "agent_attributes.csv"))
    with open(os.path.join(DATA_DIR, "social_network.json")) as f:
        network = json.load(f)
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    return agents, network, train_data

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
    def __init__(self, agents, network, train_data):
        self.agents = agents
        self.network = network
        self.train_data = train_data
    
    def fit(self):
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
        
        feature_cols = ["wearing_mask_t-1", "neighbor_mask_share_family", "neighbor_mask_share_work",
                        "neighbor_mask_share_community", "risk_perception", "received_info_t"]
        X = self.train_data[feature_cols]
        y = self.train_data["wearing_mask_t"]
        model = LogisticRegression(penalty="l2", C=parameters["regularization"], random_state=42)
        model.fit(X, y)
        
        coefs = model.coef_[0]
        parameters["alpha"] = float(model.intercept_[0])
        parameters["beta_f"] = float(coefs[1])
        parameters["beta_w"] = float(coefs[2])
        parameters["beta_c"] = float(coefs[3])
        parameters["beta_r"] = float(coefs[4])
        parameters["beta_i"] = float(coefs[5])
        
        parameters["phi_family"] = parameters["beta_f"]
        parameters["phi_work"] = parameters["beta_w"]
        parameters["phi_community"] = parameters["beta_c"]
        
        return parameters

class Simulator:
    def __init__(self, agents, network, parameters):
        self.agents = agents
        self.network = network
        self.parameters = parameters
    
    def rollout(self, val_data):
        simulated_data = []
        val_days = sorted(val_data["day"].unique())
        for day in val_days:
            for agent_id in self.agents["agent_id"]:
                neighbor_mask_share_family = self.network.get(agent_id, {}).get("family", []).count(1) / len(self.network.get(agent_id, {}).get("family", [])) if self.network.get(agent_id, {}).get("family", []) else 0
                neighbor_mask_share_work = self.network.get(agent_id, {}).get("work_school", []).count(1) / len(self.network.get(agent_id, {}).get("work_school", [])) if self.network.get(agent_id, {}).get("work_school", []) else 0
                neighbor_mask_share_community = self.network.get(agent_id, {}).get("community", []).count(1) / len(self.network.get(agent_id, {}).get("community", [])) if self.network.get(agent_id, {}).get("community", []) else 0
                
                received_info = self.parameters["phi_family"] * neighbor_mask_share_family + self.parameters["phi_work"] * neighbor_mask_share_work + self.parameters["phi_community"] * neighbor_mask_share_community + self.parameters["lambda_broadcast"]
                
                adoption_probability = 1 / (1 + np.exp(-self.parameters["alpha"] - self.parameters["gamma"] * self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t-1"].iloc[0] - self.parameters["beta_f"] * neighbor_mask_share_family - self.parameters["beta_w"] * neighbor_mask_share_work - self.parameters["beta_c"] * neighbor_mask_share_community - self.parameters["beta_r"] * self.agents.loc[self.agents["agent_id"] == agent_id, "risk_perception"].iloc[0] - self.parameters["beta_i"] * received_info))
                
                wearing_mask = np.random.binomial(1, adoption_probability)
                
                self.agents.loc[self.agents["agent_id"] == agent_id, "wearing_mask_t"] = wearing_mask
                self.agents.loc[self.agents["agent_id"] == agent_id, "received_info_t"] = received_info
                
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
        val_col = "wearing_mask_t" if "wearing_mask_t" in self.val_data.columns else "wearing_mask"
        metrics = {}
        metrics["RMSE_aggregate"] = np.sqrt(mean_squared_error(self.val_data[val_col], self.simulated_data["wearing_mask"]))
        metrics["MAE_aggregate"] = mean_absolute_error(self.val_data[val_col], self.simulated_data["wearing_mask"])
        metrics["Brier"] = brier_score_loss(self.val_data[val_col], self.simulated_data["wearing_mask"])
        metrics["TransitionFit"] = np.abs(self.val_data[val_col].values - self.simulated_data["wearing_mask"].values).mean()
        
        return metrics

def save_results(metrics, parameters, output_dir=None):
    if output_dir is None:
        output_dir = DATA_DIR
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump({
            "metrics": metrics,
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
    
    agents, network, train_data = load_data()
    agents, network = build_network_and_agents(agents, network, train_data)
    featured_data = prepare_train_features(train_data, agents, network)
    train_split, val_data = holdout_split(featured_data)
    calibrator = Calibrator(agents, network, train_split)
    parameters = calibrator.fit()
    simulator = Simulator(agents, network, parameters)
    simulated_data = simulator.rollout(val_data)
    evaluator = Evaluator(val_data, simulated_data)
    metrics = evaluator.compute_metrics()
    save_results(metrics, parameters, output_dir=args.output_dir)


# Execute main for both direct execution and sandbox wrapper invocation
main()