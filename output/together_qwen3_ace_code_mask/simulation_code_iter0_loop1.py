from __future__ import annotations

import os
import json
import csv
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.stats import bernoulli
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Set global random seed for determinism
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# Path setup from environment variables
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
if not PROJECT_ROOT or not DATA_PATH:
    raise ValueError("PROJECT_ROOT and DATA_PATH environment variables must be set.")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Define paths to required data files
AGENT_ATTR_PATH = os.path.join(DATA_DIR, "agent_attributes.csv")
NETWORK_PATH = os.path.join(DATA_DIR, "social_network.json")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")

# Validate all input files exist
for path in [AGENT_ATTR_PATH, NETWORK_PATH, TRAIN_DATA_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")

@dataclass
class Agent:
    agent_id: str
    age: int
    age_group: str
    occupation: str
    risk_perception: float
    initial_mask_wearing: bool
    layer_degrees: Dict[str, int]  # family, work_school, community
    wearing_mask: bool = False
    received_info: bool = False
    memory_of_info: float = 0.0
    propensity_score: float = 0.0
    # Precomputed layer adjacency lists (filled during network loading)
    neighbors: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = {"family": [], "work_school": [], "community": []}

class Network:
    def __init__(self):
        self.layers = ["family", "work_school", "community"]
        self.edges = {layer: set() for layer in self.layers}
        self.adjacency = {}  # agent_id -> {layer: [neighbors]}
        self.degrees = {}    # agent_id -> {layer: degree}

    def load_from_json(self, path: str, agent_ids: set):
        """Load multiplex network from JSON, symmetrize edges, deduplicate."""
        with open(path, 'r') as f:
            network_data = json.load(f)

        for agent_id, layers in network_data.items():
            if agent_id not in agent_ids:
                continue
            self.adjacency[agent_id] = {"family": [], "work_school": [], "community": []}
            self.degrees[agent_id] = {"family": 0, "work_school": 0, "community": 0}

            for layer in self.layers:
                if layer not in layers:
                    continue
                neighbors = set(layers[layer])
                # Symmetrize: if i has j, ensure j has i
                for neighbor in neighbors:
                    if neighbor not in agent_ids:
                        continue
                    self.edges[layer].add(tuple(sorted([agent_id, neighbor])))
                    self.adjacency[agent_id][layer].append(neighbor)
                    self.degrees[agent_id][layer] += 1

        # Ensure all agents have initialized adjacency even if no connections
        for agent_id in agent_ids:
            if agent_id not in self.adjacency:
                self.adjacency[agent_id] = {layer: [] for layer in self.layers}
                self.degrees[agent_id] = {layer: 0 for layer in self.layers}

        # Deduplicate and validate
        for layer in self.layers:
            self.edges[layer] = set(sorted(e) for e in self.edges[layer])

    def get_neighbor_mask_share(self, agent_id: str, layer: str, agents: Dict[str, Agent]) -> float:
        """Compute fraction of neighbors wearing mask in given layer."""
        neighbors = self.adjacency.get(agent_id, {}).get(layer, [])
        if not neighbors:
            return 0.0
        mask_wearers = sum(1 for nid in neighbors if agents[nid].wearing_mask)
        return mask_wearers / len(neighbors)

def compute_adoption_probability(agent: Agent, network: Network, params: Dict[str, float]) -> float:
    """Compute probability of wearing mask tomorrow using logistic model."""
    prev_mask = agent.wearing_mask
    risk = agent.risk_perception
    info = agent.received_info or agent.memory_of_info > 0.5  # Binary proxy for memory
    
    # Compute neighbor shares
    share_f = network.get_neighbor_mask_share(agent.agent_id, 'family', agents)
    share_w = network.get_neighbor_mask_share(agent.agent_id, 'work_school', agents)
    share_c = network.get_neighbor_mask_share(agent.agent_id, 'community', agents)
    
    # Normalize layer weights
    w_f = params.get('w_family', 1/3)
    w_w = params.get('w_work', 1/3)
    w_c = params.get('w_community', 1/3)
    total = w_f + w_w + w_c
    if total == 0:
        w_f = w_w = w_c = 1/3
    else:
        w_f /= total
        w_w /= total
        w_c /= total
    
    # Feature vector
    inertia = 1.0 if prev_mask else 0.0
    beta_f = params.get('beta_f', 1.0)
    beta_w = params.get('beta_w', 1.0)
    beta_c = params.get('beta_c', 1.0)
    beta_r = params.get('beta_r', 1.0)
    beta_i = params.get('beta_i', 1.0)
    alpha = params.get('alpha', 0.0)
    gamma = params.get('gamma', 1.0)
    tau = params.get('tau', 1.0)
    
    # Demographic fixed effects
    age_group = agent.age_group
    occupation = agent.occupation
    age_groups = ['youth', 'young_adult', 'middle_age', 'senior']
    occupations = ['student', 'blue_collar', 'white_collar']
    
    feature_sum = 0.0
    # Intercept
    feature_sum += alpha
    # Inertia
    feature_sum += gamma * inertia
    # Layer weights * shares (for normalization effect)
    feature_sum += w_f * share_f + w_w * share_w + w_c * share_c
    # Peer influence slopes
    feature_sum += beta_f * share_f + beta_w * share_w + beta_c * share_c
    # Risk perception
    feature_sum += beta_r * risk
    # Info exposure
    feature_sum += beta_i * (1.0 if info else 0.0)
    
    # Age group effects
    for i, ag in enumerate(age_groups[:-1]):
        beta_age = params.get(f'beta_age_{ag}', 0.0)
        if age_group == ag:
            feature_sum += beta_age
    # Occupation effects
    for i, occ in enumerate(occupations[:-1]):
        beta_occ = params.get(f'beta_occ_{occ}', 0.0)
        if occupation == occ:
            feature_sum += beta_occ
    
    # Sigmoid with temperature
    logit = feature_sum
    tau_safe = max(tau, 1e-6)
    prob = 1.0 / (1.0 + np.exp(-logit / tau_safe))
    return prob

class CalibrationAlgorithm(ABC):
    @abstractmethod
    def fit(self, agents: Dict[str, Agent], network: Network, train_data: pd.DataFrame, 
            train_days: List[str], params: Dict[str, float]) -> Dict[str, float]:
        pass

class GradientBasedCalibrator(CalibrationAlgorithm):
    def __init__(self, regularization: float = 1.0, max_iter: int = 100, tol: float = 1e-4):
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, agents: Dict[str, Agent], network: Network, train_data: pd.DataFrame, 
            train_days: List[str], initial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Calibrate parameters by maximizing likelihood of observed transitions and info receipt.
        Uses gradient-based optimization with L2 regularization.
        """
        # Extract observed transitions and received_info from train_data for train_days
        train_subset = train_data[train_data['day'].isin(train_days)].copy()
        if len(train_subset) == 0:
            raise ValueError("No training data available after temporal split.")

        # Define parameter vector: all calibratable parameters
        param_names = [
            'alpha', 'gamma', 'w_family', 'w_work', 'w_community', 
            'beta_f', 'beta_w', 'beta_c', 'beta_r', 'beta_i', 
            'lambda_broadcast', 'phi_family', 'phi_work', 'phi_community', 
            'rho_info_decay', 'tau'
        ]
        # Age and occupation effects (one baseline per category)
        age_groups = ['youth', 'young_adult', 'middle_age', 'senior']
        occupations = ['student', 'blue_collar', 'white_collar']
        for ag in age_groups[:-1]:  # last is baseline
            param_names.append(f'beta_age_{ag}')
        for occ in occupations[:-1]:  # last is baseline
            param_names.append(f'beta_occ_{occ}')

        # Initialize parameter vector
        x0 = np.array([initial_params.get(p, 0.0) for p in param_names])
        
        # Bounds for each parameter
        bounds = [
            (-5, 5),  # alpha
            (0, 6),   # gamma
            (0, 2),   # w_family
            (0, 2),   # w_work
            (0, 2),   # w_community
            (0, 5),   # beta_f
            (0, 5),   # beta_w
            (0, 5),   # beta_c
            (0, 5),   # beta_r
            (0, 5),   # beta_i
            (0, 0.5), # lambda_broadcast
            (0, 2),   # phi_family
            (0, 2),   # phi_work
            (0, 2),   # phi_community
            (0, 1),   # rho_info_decay
            (0.5, 5), # tau
        ]
        # Age group effects
        for _ in age_groups[:-1]:
            bounds.append((-3, 3))
        # Occupation effects
        for _ in occupations[:-1]:
            bounds.append((-3, 3))

        # Normalize layer weights to sum to 1 at runtime
        def normalize_weights(x):
            w = x[2:5]  # w_family, w_work, w_community
            total = np.sum(w)
            if total == 0:
                w[:] = [1/3, 1/3, 1/3]
            else:
                w[:] = w / total
            return x

        def objective(x):
            # Normalize weights
            x_copy = x.copy()
            normalize_weights(x_copy)
            
            # Extract parameters
            params = dict(zip(param_names, x_copy))
            
            # Compute negative log-likelihood
            nll = 0.0
            for _, row in train_subset.iterrows():
                agent_id = row['agent_id']
                agent = agents[agent_id]
                day = row['day']
                
                # Get previous state (t-1) and current state (t)
                # We assume train_data is sorted by day and agent_id
                # For day 0, previous state is initial_mask_wearing
                if day == 0:
                    prev_mask = agent.initial_mask_wearing
                else:
                    prev_row = train_subset[(train_subset['agent_id'] == agent_id) & (train_subset['day'] == day - 1)]
                    if len(prev_row) == 0:
                        prev_mask = agent.initial_mask_wearing
                    else:
                        prev_mask = bool(prev_row.iloc[0]['wearing_mask'])
                
                curr_mask = bool(row['wearing_mask'])
                
                # Compute received_info target
                received_info_target = bool(row['received_info']) if 'received_info' in row else False
                
                # Compute neighbor mask shares per layer
                share_f = network.get_neighbor_mask_share(agent_id, 'family', agents)
                share_w = network.get_neighbor_mask_share(agent_id, 'work_school', agents)
                share_c = network.get_neighbor_mask_share(agent_id, 'community', agents)
                
                # Compute info receipt probability from peer and broadcast
                phi_f, phi_w, phi_c = params['phi_family'], params['phi_work'], params['phi_community']
                lambda_b = params['lambda_broadcast']
                peer_info_prob = 1 - np.exp(- (phi_f * share_f + phi_w * share_w + phi_c * share_c))
                info_prob = min(1.0, peer_info_prob + lambda_b)
                
                # Compute adoption probability using logistic model
                # Feature vector
                inertia = 1.0 if prev_mask else 0.0
                risk = agent.risk_perception
                info = 1.0 if received_info_target else 0.0  # Use observed for calibration
                # Demographics
                age_group = agent.age_group
                occupation = agent.occupation
                
                # Build feature vector
                features = [
                    1.0,  # intercept
                    inertia,
                    params['w_family'] * share_f,
                    params['w_work'] * share_w,
                    params['w_community'] * share_c,
                    params['beta_f'] * share_f,
                    params['beta_w'] * share_w,
                    params['beta_c'] * share_c,
                    params['beta_r'] * risk,
                    params['beta_i'] * info
                ]
                
                # Age group effects (one baseline omitted)
                age_map = {'youth': 0, 'young_adult': 1, 'middle_age': 2, 'senior': 3}
                for i, ag in enumerate(age_groups[:-1]):
                    if age_group == ag:
                        features.append(params[f'beta_age_{ag}'])
                    else:
                        features.append(0.0)
                # Occupation effects
                occ_map = {'student': 0, 'blue_collar': 1, 'white_collar': 2}
                for i, occ_type in enumerate(occupations[:-1]):
                    if occupation == occ_type:
                        features.append(params[f'beta_occ_{occ_type}'])
                    else:
                        features.append(0.0)
                
                # Compute logits
                logit = params['alpha'] + params['gamma'] * inertia + sum(features[2:])
                tau_safe = max(params['tau'], 1e-6)
                prob = 1.0 / (1.0 + np.exp(-logit / tau_safe))  # Sigmoid with temperature
                
                # Log-likelihood for mask adoption
                if curr_mask:
                    nll -= np.log(prob + 1e-10)
                else:
                    nll -= np.log(1 - prob + 1e-10)
                
                # Log-likelihood for received_info
                if received_info_target:
                    nll -= np.log(info_prob + 1e-10)
                else:
                    nll -= np.log(1 - info_prob + 1e-10)
            
            # L2 regularization
            reg_penalty = self.regularization * np.sum(x[1:]**2)  # Skip intercept
            nll += reg_penalty
            
            return nll

        # Optimization
        result = minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

        if not result.success:
            warnings.warn(f"Calibration optimization failed: {result.message}")

        calibrated_params = dict(zip(param_names, result.x))
        return calibrated_params

class Simulator:
    def __init__(self, agents: Dict[str, Agent], network: Network, params: Dict[str, float]):
        self.agents = agents
        self.network = network
        self.params = params
        self.history = []  # List of dicts: {day: ..., agent_states: {...}}
        self.rho_info_decay = params.get('rho_info_decay', 0.1)

    def _compute_info_receipt(self, agent_id: str) -> float:
        """Compute probability of receiving information today."""
        phi_f = self.params.get('phi_family', 0.5)
        phi_w = self.params.get('phi_work', 0.5)
        phi_c = self.params.get('phi_community', 0.5)
        lambda_b = self.params.get('lambda_broadcast', 0.1)
        
        share_f = self.network.get_neighbor_mask_share(agent_id, 'family', self.agents)
        share_w = self.network.get_neighbor_mask_share(agent_id, 'work_school', self.agents)
        share_c = self.network.get_neighbor_mask_share(agent_id, 'community', self.agents)
        
        peer_info = phi_f * share_f + phi_w * share_w + phi_c * share_c
        info_prob = 1 - np.exp(-peer_info) + lambda_b
        return min(1.0, info_prob)

    def step(self, day: int) -> Dict[str, Any]:
        """Single day simulation step."""
        new_agents = {}
        for agent_id, agent in self.agents.items():
            # Compute received info
            info_prob = self._compute_info_receipt(agent_id)
            received_today = rng.random() < info_prob
            
            # Update memory: decay and add new info
            memory = agent.memory_of_info * (1 - self.rho_info_decay)
            if received_today:
                memory = 1.0
            else:
                memory = max(0.0, memory)
            
            # Compute adoption probability
            prob_mask = compute_adoption_probability(agent, self.network, self.params)
            wear_tomorrow = rng.random() < prob_mask
            
            # Store updated state
            new_agent = Agent(
                agent_id=agent_id,
                age=agent.age,
                age_group=agent.age_group,
                occupation=agent.occupation,
                risk_perception=agent.risk_perception,
                initial_mask_wearing=agent.initial_mask_wearing,
                layer_degrees=agent.layer_degrees,
                wearing_mask=wear_tomorrow,
                received_info=received_today,
                memory_of_info=memory,
                neighbors=agent.neighbors
            )
            new_agents[agent_id] = new_agent
        
        # Update global state
        self.agents = new_agents
        
        # Record state
        aggregate_mask_rate = np.mean([a.wearing_mask for a in self.agents.values()])
        self.history.append({
            'day': day,
            'aggregate_mask_rate': aggregate_mask_rate,
            'agent_states': {aid: {
                'wearing_mask': a.wearing_mask,
                'received_info': a.received_info,
                'memory_of_info': a.memory_of_info
            } for aid, a in self.agents.items()}
        })
        
        return {'day': day, 'aggregate_mask_rate': aggregate_mask_rate}

    def rollout(self, days: List[int]) -> List[Dict[str, Any]]:
        """Run simulation over validation days."""
        for day in days:
            self.step(day)
        return self.history

class Evaluator:
    def __init__(self, agents: Dict[str, Agent], network: Network, train_data: pd.DataFrame, val_days: List[str]):
        self.agents = agents
        self.network = network
        self.train_data = train_data
        self.val_days = val_days

    def compute_metrics(self, simulation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics on validation window."""
        val_data = self.train_data[self.train_data['day'].isin(self.val_days)].copy()
        if len(val_data) == 0:
            raise ValueError("No validation data available after temporal split.")

        # Aggregate metrics
        simulated_daily_rates = [h['aggregate_mask_rate'] for h in simulation_history if h['day'] in self.val_days]
        observed_daily_rates = val_data.groupby('day')['wearing_mask'].mean().reindex(self.val_days).values
        
        if len(simulated_daily_rates) != len(observed_daily_rates):
            raise ValueError("Mismatch in simulation and observation lengths for validation days.")
        
        rmse_agg = np.sqrt(np.mean((np.array(simulated_daily_rates) - observed_daily_rates)**2))
        mae_agg = np.mean(np.abs(np.array(simulated_daily_rates) - observed_daily_rates))
        
        # Brier score: per-agent binary prediction error
        brier = 0.0
        n_samples = 0
        for h in simulation_history:
            if h['day'] not in self.val_days:
                continue
            day_data = val_data[val_data['day'] == h['day']]
            for _, row in day_data.iterrows():
                agent_id = row['agent_id']
                observed = row['wearing_mask']
                # Recompute adoption probability for consistency
                agent = self.agents[agent_id]
                prob = compute_adoption_probability(agent, self.network, self.params)
                brier += (prob - observed)**2
                n_samples += 1
        
        brier /= n_samples if n_samples > 0 else 1

        # TransitionFit: P00, P01, P10, P11
        # We need to reconstruct transitions from observed and simulated data
        transitions_obs = {'00': 0, '01': 0, '10': 0, '11': 0}
        transitions_sim = {'00': 0, '01': 0, '10': 0, '11': 0}
        
        # Sort val_data by agent_id and day
        val_data_sorted = val_data.sort_values(['agent_id', 'day'])
        agents_by_day = defaultdict(list)
        for _, row in val_data_sorted.iterrows():
            agents_by_day[row['day']].append(row)
        
        # For each agent, get transitions
        agent_sequences = defaultdict(list)
        for agent_id in val_data['agent_id'].unique():
            seq = val_data_sorted[val_data_sorted['agent_id'] == agent_id]['wearing_mask'].tolist()
            if len(seq) >= 2:
                for i in range(1, len(seq)):
                    prev, curr = seq[i-1], seq[i]
                    key = f"{int(prev)}{int(curr)}"
                    if key in transitions_obs:
                        transitions_obs[key] += 1
        
        # Simulated transitions
        # We need to reconstruct simulated states per agent per day
        simulated_states = defaultdict(list)
        for h in simulation_history:
            if h['day'] not in self.val_days:
                continue
            for aid, state in h['agent_states'].items():
                simulated_states[aid].append(state['wearing_mask'])
        
        for aid, seq in simulated_states.items():
            if len(seq) >= 2:
                for i in range(1, len(seq)):
                    prev, curr = seq[i-1], seq[i]
                    key = f"{int(prev)}{int(curr)}"
                    if key in transitions_sim:
                        transitions_sim[key] += 1
        
        # Compute TransitionFit as sum of absolute differences
        transition_fit = 0.0
        total_transitions = sum(transitions_obs.values())
        if total_transitions > 0:
            for key in ['00', '01', '10', '11']:
                obs_count = transitions_obs[key]
                sim_count = transitions_sim[key]
                obs_prob = obs_count / total_transitions
                sim_prob = sim_count / total_transitions
                transition_fit += abs(obs_prob - sim_prob)
        
        return {
            "RMSE_aggregate": float(rmse_agg),
            "MAE_aggregate": float(mae_agg),
            "Brier": float(brier),
            "TransitionFit": float(transition_fit),
            "simulated_daily_rates": simulated_daily_rates,
            "observed_daily_rates": observed_daily_rates.tolist(),
            "transitions_observed": transitions_obs,
            "transitions_simulated": transitions_sim
        }

def parse_cli():
    """Optional CLI parser. For now, just return None since we use env vars."""
    return {}

def load_data() -> Tuple[Dict[str, Agent], Network, pd.DataFrame]:
    """Load agents, network, and training data."""
    # Load agent attributes
    agent_df = pd.read_csv(AGENT_ATTR_PATH)
    agent_df = agent_df.dropna(subset=['agent_id'])
    
    # Load network
    with open(NETWORK_PATH, 'r') as f:
        network_json = json.load(f)
    
    # Extract agent_ids from both files
    agent_ids_csv = set(agent_df['agent_id'].astype(str))
    agent_ids_network = set(network_json.keys())
    agent_ids = agent_ids_csv.intersection(agent_ids_network)
    
    if len(agent_ids) == 0:
        raise ValueError("No overlapping agent IDs between agent_attributes.csv and social_network.json")
    
    # Build agents
    agents = {}
    for _, row in agent_df.iterrows():
        aid = str(row['agent_id'])
        if aid not in agent_ids:
            continue
        agent = Agent(
            agent_id=aid,
            age=int(row['age']) if not pd.isna(row['age']) else 0,
            age_group=str(row['age_group']),
            occupation=str(row['occupation']),
            risk_perception=float(row['risk_perception']),
            initial_mask_wearing=bool(row['initial_mask_wearing']),
            layer_degrees={
                'family': int(row.get('family_connections', 0)),
                'work_school': int(row.get('work_school_connections', 0)),
                'community': int(row.get('community_connections', 0))
            }
        )
        agents[aid] = agent
    
    # Build network
    network = Network()
    network.load_from_json(NETWORK_PATH, agent_ids)
    
    # Load train_data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    train_data['day'] = train_data['day'].astype(int)
    train_data['agent_id'] = train_data['agent_id'].astype(str)
    train_data['wearing_mask'] = train_data['wearing_mask'].astype(bool)
    if 'received_info' in train_data.columns:
        train_data['received_info'] = train_data['received_info'].astype(bool)
    else:
        train_data['received_info'] = False
    
    # Align train_data with agent_ids
    train_data = train_data[train_data['agent_id'].isin(agent_ids)]
    
    # Initialize agent states from train_data day 0
    day0_data = train_data[train_data['day'] == 0]
    for _, row in day0_data.iterrows():
        aid = row['agent_id']
        if aid in agents:
            agents[aid].wearing_mask = bool(row['wearing_mask'])
            if 'received_info' in row:
                agents[aid].received_info = bool(row['received_info'])
            else:
                agents[aid].received_info = False
            agents[aid].memory_of_info = 1.0 if agents[aid].received_info else 0.0
    
    # For agents not in day0, use initial_mask_wearing
    for aid in agents:
        if aid not in day0_data['agent_id'].values:
            agents[aid].wearing_mask = agents[aid].initial_mask_wearing
            agents[aid].received_info = False
            agents[aid].memory_of_info = 0.0
    
    return agents, network, train_data

def holdout_split(train_data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split days into training (80%) and validation (20%) using temporal holdout."""
    unique_days = sorted(train_data['day'].unique())
    if len(unique_days) == 0:
        raise ValueError("No days found in train_data.csv")
    
    n_train = int(len(unique_days) * 0.8)
    if n_train == 0:
        raise ValueError("Training set has 0 days after temporal split.")
    
    train_days = unique_days[:n_train]
    val_days = unique_days[n_train:]
    
    if len(val_days) == 0:
        raise ValueError("No validation days available after temporal split.")
    
    return train_days, val_days

def build_network_and_agents() -> Tuple[Dict[str, Agent], Network]:
    """Wrapper to load and build network and agents."""
    agents, network, _ = load_data()
    return agents, network

def calibrate_parameters(agents: Dict[str, Agent], network: Network, train_data: pd.DataFrame, train_days: List[str]) -> Dict[str, float]:
    """Initialize parameters and calibrate using gradient-based algorithm."""
    # Define initial parameters with heuristic defaults
    initial_params = {
        'alpha': 0.0,
        'gamma': 1.0,
        'w_family': 0.33,
        'w_work': 0.33,
        'w_community': 0.34,
        'beta_f': 1.0,
        'beta_w': 1.0,
        'beta_c': 1.0,
        'beta_r': 1.0,
        'beta_i': 1.0,
        'lambda_broadcast': 0.1,
        'phi_family': 0.5,
        'phi_work': 0.5,
        'phi_community': 0.5,
        'rho_info_decay': 0.1,
        'tau': 1.0
    }
    
    # Add age group effects (one baseline omitted)
    age_groups = ['youth', 'young_adult', 'middle_age', 'senior']
    for ag in age_groups[:-1]:
        initial_params[f'beta_age_{ag}'] = 0.0
    
    # Add occupation effects (one baseline omitted)
    occupations = ['student', 'blue_collar', 'white_collar']
    for occ in occupations[:-1]:
        initial_params[f'beta_occ_{occ}'] = 0.0
    
    # Calibrate
    calibrator = GradientBasedCalibrator(regularization=1.0)
    calibrated_params = calibrator.fit(agents, network, train_data, train_days, initial_params)
    return calibrated_params

def simulate_validation(agents: Dict[str, Agent], network: Network, params: Dict[str, float], val_days: List[str]) -> List[Dict[str, Any]]:
    """Run forward simulation on validation window."""
    simulator = Simulator(agents, network, params)
    history = simulator.rollout(val_days)
    return history

def evaluate_simulation(agents: Dict[str, Agent], network: Network, train_data: pd.DataFrame, val_days: List[str], simulation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    evaluator = Evaluator(agents, network, train_data, val_days)
    metrics = evaluator.compute_metrics(simulation_history)
    return metrics

def save_results(calibrated_params: Dict[str, float], simulation_history: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    """Save results to output directory."""
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save calibrated parameters
    with open(os.path.join(OUTPUT_DIR, "calibrated_params.json"), 'w') as f:
        json.dump(calibrated_params, f, indent=2)
    
    # Save simulation history
    with open(os.path.join(OUTPUT_DIR, "simulation_history.json"), 'w') as f:
        json.dump(simulation_history, f, indent=2)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save reproducibility info
    repro_info = {
        "seed": SEED,
        "project_root": PROJECT_ROOT,
        "data_path": DATA_PATH,
        "version": "1.0"
    }
    with open(os.path.join(OUTPUT_DIR, "reproducibility.json"), 'w') as f:
        json.dump(repro_info, f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}")

def main():
    """Orchestrator function that runs the entire simulation pipeline."""
    # Step 1: Parse CLI (optional)
    cli_args = parse_cli()
    
    # Step 2: Load data
    agents, network, train_data = load_data()
    
    # Step 3: Temporal holdout split
    train_days, val_days = holdout_split(train_data)
    
    # Step 4: Calibrate parameters using training data
    calibrated_params = calibrate_parameters(agents, network, train_data, train_days)
    
    # Step 5: Forward simulation on validation window
    simulation_history = simulate_validation(agents, network, calibrated_params, val_days)
    
    # Step 6: Compute evaluation metrics
    metrics = evaluate_simulation(agents, network, train_data, val_days, simulation_history)
    
    # Step 7: Save results
    save_results(calibrated_params, simulation_history, metrics)

# Execute main for both direct execution and sandbox wrapper invocation
main()