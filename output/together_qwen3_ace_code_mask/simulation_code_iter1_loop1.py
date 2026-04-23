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
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Set global random seed for determinism
SEED = 42
np.random.seed(SEED)
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
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")

# Validate all input files exist
for path in [AGENT_ATTR_PATH, NETWORK_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH]:
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
            'alpha', 'gamma', 
            'beta_f', 'beta_w', 'beta_c', 
            'beta_r', 'beta_i', 
            'lambda_broadcast', 
            'phi_family', 'phi_work', 'phi_community', 
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

        def objective(x):
            # Extract parameters
            params = dict(zip(param_names, x))
            
            # Compute negative log-likelihood
            nll = 0.0
            for _, row in train_subset.iterrows():
                agent_id = row['agent_id']
                agent = agents[agent_id]
                day = row['day']
                
                # Get previous state (t-1) and current state (t)
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
                peer_info = phi_f * share_f + phi_w * share_w + phi_c * share_c
                info_prob = 1 - np.exp(-(peer_info + lambda_b))  # Corrected: inside exponential
                info_prob = min(1.0, info_prob)
                
                # Compute adoption probability using logistic model
                # Feature vector
                inertia = 1.0 if prev_mask else 0.0
                risk = agent.risk_perception
                info = 1.0 if received_info_target else 0.0  # Use observed for calibration
                
                # Build feature vector
                features = [
                    1.0,  # intercept
                    inertia,
                    params['beta_f'] * share_f,
                    params['beta_w'] * share_w,
                    params['beta_c'] * share_c,
                    params['beta_r'] * risk,
                    params['beta_i'] * info
                ]
                
                # Age group effects (one baseline omitted)
                age_map = {'youth': 0, 'young_adult': 1, 'middle_age': 2, 'senior': 3}
                for i, ag in enumerate(age_groups[:-1]):
                    if agent.age_group == ag:
                        features.append(params[f'beta_age_{ag}'])
                    else:
                        features.append(0.0)
                # Occupation effects
                occ_map = {'student': 0, 'blue_collar': 1, 'white_collar': 2}
                for i, occ_type in enumerate(occupations[:-1]):
                    if agent.occupation == occ_type:
                        features.append(params[f'beta_occ_{occ_type}'])
                    else:
                        features.append(0.0)
                
                # Compute logits
                logit = params['alpha'] + params['gamma'] * inertia + sum(features[2:])
                prob = 1.0 / (1.0 + np.exp(-logit / params['tau']))  # Sigmoid with temperature
                
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
        info_prob = 1 - np.exp(-(peer_info + lambda_b))  # Corrected: inside exponential
        return min(1.0, info_prob)

    def _compute_adoption_probability(self, agent_id: str) -> float:
        """Compute probability of wearing mask tomorrow using logistic model."""
        agent = self.agents[agent_id]
        prev_mask = agent.wearing_mask
        risk = agent.risk_perception
        info = agent.received_info or agent.memory_of_info > 0.5  # Binary proxy for memory
        
        # Compute neighbor shares
        share_f = self.network.get_neighbor_mask_share(agent_id, 'family', self.agents)
        share_w = self.network.get_neighbor_mask_share(agent_id, 'work_school', self.agents)
        share_c = self.network.get_neighbor_mask_share(agent_id, 'community', self.agents)
        
        # Feature vector
        inertia = 1.0 if prev_mask else 0.0
        beta_f = self.params.get('beta_f', 1.0)
        beta_w = self.params.get('beta_w', 1.0)
        beta_c = self.params.get('beta_c', 1.0)
        beta_r = self.params.get('beta_r', 1.0)
        beta_i = self.params.get('beta_i', 1.0)
        alpha = self.params.get('alpha', 0.0)
        gamma = self.params.get('gamma', 1.0)
        tau = self.params.get('tau', 1.0)
        
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
        # Peer influence slopes (removed w_* terms as per playbook)
        feature_sum += beta_f * share_f + beta_w * share_w + beta_c * share_c
        # Risk perception
        feature_sum += beta_r * risk
        # Info exposure
        feature_sum += beta_i * (1.0 if info else 0.0)
        
        # Age group effects
        for i, ag in enumerate(age_groups[:-1]):
            beta_age = self.params.get(f'beta_age_{ag}', 0.0)
            if age_group == ag:
                feature_sum += beta_age
        # Occupation effects
        for i, occ in enumerate(occupations[:-1]):
            beta_occ = self.params.get(f'beta_occ_{occ}', 0.0)
            if occupation == occ:
                feature_sum += beta_occ
        
        # Sigmoid with temperature
        logit = feature_sum
        prob = 1.0 / (1.0 + np.exp(-logit / tau))
        return prob

    def step(self, day: int) -> Dict[str, Any]:
        """Single day simulation step."""
        new_agents = {}
        for agent_id, agent in self.agents.items():
            # Compute received info
            info_prob = self._compute_info_receipt(agent_id)
            received_today = random.random() < info_prob
            
            # Update memory: decay and add new info
            memory = agent.memory_of_info * (1 - self.rho_info_decay)
            if received_today:
                memory = 1.0
            else:
                memory = max(0.0, memory)
            
            # Compute adoption probability
            prob_mask = self._compute_adoption_probability(agent_id)
            wear_tomorrow = random.random() < prob_mask
            
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
    def __init__(self, agents: Dict[str, Agent], network: Network, eval_data: pd.DataFrame, eval_days: List[int]):
        """
        Evaluator for computing metrics on evaluation data.
        
        Args:
            agents: Dictionary of agents
            network: Network object
            eval_data: DataFrame containing evaluation data (can be validation or test data)
            eval_days: List of days to evaluate on
        """
        self.agents = agents
        self.network = network
        self.eval_data = eval_data
        self.eval_days = eval_days

    def compute_metrics(self, simulation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute evaluation metrics on evaluation window."""
        eval_data_subset = self.eval_data[self.eval_data['day'].isin(self.eval_days)].copy()
        if len(eval_data_subset) == 0:
            raise ValueError(f"No evaluation data available for days {self.eval_days}.")

        # Aggregate metrics
        simulated_daily_rates = [h['aggregate_mask_rate'] for h in simulation_history if h['day'] in self.eval_days]
        observed_daily_rates = eval_data_subset.groupby('day')['wearing_mask'].mean().reindex(self.eval_days).values
        
        if len(simulated_daily_rates) != len(observed_daily_rates):
            raise ValueError(f"Mismatch in simulation and observation lengths for evaluation days {self.eval_days}.")
        
        rmse_agg = np.sqrt(np.mean((np.array(simulated_daily_rates) - observed_daily_rates)**2))
        mae_agg = np.mean(np.abs(np.array(simulated_daily_rates) - observed_daily_rates))
        
        # Brier score: per-agent binary prediction error
        brier = 0.0
        n_samples = 0
        for h in simulation_history:
            if h['day'] not in self.eval_days:
                continue
            day_data = eval_data_subset[eval_data_subset['day'] == h['day']]
            agent_states = h.get('agent_states', {})
            for _, row in day_data.iterrows():
                agent_id = str(row['agent_id'])
                observed = bool(row['wearing_mask'])
                # Get simulated binary outcome from history
                if agent_id in agent_states:
                    predicted = bool(agent_states[agent_id]['wearing_mask'])
                    # Use binary prediction (0 or 1) for Brier score
                    brier += (float(predicted) - float(observed))**2
                    n_samples += 1
                else:
                    # Log warning if agent state missing
                    warnings.warn(f"Missing agent state for {agent_id} on day {h['day']}")
        
        brier /= n_samples if n_samples > 0 else 1

        # TransitionFit: Compute per-agent transition probabilities, then average
        transitions_observed = {'00': 0, '01': 0, '10': 0, '11': 0}
        transitions_simulated = {'00': 0, '01': 0, '10': 0, '11': 0}
        agent_transitions_obs = defaultdict(list)
        agent_transitions_sim = defaultdict(list)
        
        # Sort eval_data by agent_id and day
        eval_data_sorted = eval_data_subset.sort_values(['agent_id', 'day'])
        for _, row in eval_data_sorted.iterrows():
            agent_id = str(row['agent_id'])
            mask_state = bool(row['wearing_mask'])
            agent_transitions_obs[agent_id].append(mask_state)
        
        # Extract simulated transitions per agent
        sorted_history = sorted([h for h in simulation_history if h['day'] in self.eval_days], key=lambda x: x['day'])
        for h in sorted_history:
            agent_states = h.get('agent_states', {})
            for aid, state in agent_states.items():
                aid_str = str(aid)
                agent_transitions_sim[aid_str].append(bool(state['wearing_mask']))
        
        # Compute per-agent transition counts for observed
        for aid, seq in agent_transitions_obs.items():
            if len(seq) >= 2:
                for i in range(1, len(seq)):
                    prev, curr = seq[i-1], seq[i]
                    key = f"{int(prev)}{int(curr)}"
                    if key in transitions_observed:
                        transitions_observed[key] += 1
        
        # Compute per-agent transition counts for simulated
        for aid, seq in agent_transitions_sim.items():
            if len(seq) >= 2:
                for i in range(1, len(seq)):
                    prev, curr = seq[i-1], seq[i]
                    key = f"{int(prev)}{int(curr)}"
                    if key in transitions_simulated:
                        transitions_simulated[key] += 1
        
        # Compute transition probabilities per agent, then average across agents
        # For observed: for each agent with >=2 days, compute their transition probs
        agent_observed_probs = []
        for aid, seq in agent_transitions_obs.items():
            if len(seq) < 2:
                continue
            counts = {'00': 0, '01': 0, '10': 0, '11': 0}
            total = len(seq) - 1
            for i in range(1, len(seq)):
                prev, curr = seq[i-1], seq[i]
                key = f"{int(prev)}{int(curr)}"
                if key in counts:
                    counts[key] += 1
            # Normalize to probabilities
            if total > 0:
                probs = {k: v / total for k, v in counts.items()}
                agent_observed_probs.append(probs)
        
        # For simulated
        agent_simulated_probs = []
        for aid, seq in agent_transitions_sim.items():
            if len(seq) < 2:
                continue
            counts = {'00': 0, '01': 0, '10': 0, '11': 0}
            total = len(seq) - 1
            for i in range(1, len(seq)):
                prev, curr = seq[i-1], seq[i]
                key = f"{int(prev)}{int(curr)}"
                if key in counts:
                    counts[key] += 1
            if total > 0:
                probs = {k: v / total for k, v in counts.items()}
                agent_simulated_probs.append(probs)
        
        # Average transition probabilities across agents
        if len(agent_observed_probs) == 0:
            avg_observed = {'00': 0, '01': 0, '10': 0, '11': 0}
        else:
            avg_observed = {k: np.mean([p.get(k, 0) for p in agent_observed_probs]) for k in ['00','01','10','11']}
        
        if len(agent_simulated_probs) == 0:
            avg_simulated = {'00': 0, '01': 0, '10': 0, '11': 0}
        else:
            avg_simulated = {k: np.mean([p.get(k, 0) for p in agent_simulated_probs]) for k in ['00','01','10','11']}
        
        # TransitionFit: sum of absolute differences between averaged transition probabilities
        transition_fit = sum(abs(avg_observed[k] - avg_simulated[k]) for k in ['00','01','10','11'])
        
        return {
            "RMSE_aggregate": float(rmse_agg),
            "MAE_aggregate": float(mae_agg),
            "Brier": float(brier),
            "TransitionFit": float(transition_fit),
            "simulated_daily_rates": simulated_daily_rates,
            "observed_daily_rates": observed_daily_rates.tolist(),
            "transitions_observed": transitions_observed,
            "transitions_simulated": transitions_simulated
        }

def parse_cli():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Run mask adoption simulation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to PROJECT_ROOT/output if not provided)"
    )
    args = parser.parse_args()
    return vars(args)

def load_data() -> Tuple[Dict[str, Agent], Network, pd.DataFrame, pd.DataFrame]:
    """Load agents, network, training data, and test data."""
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
        
        # Handle missing 'age' column - use default value 0 since it's not used in calculations
        age = int(row['age']) if 'age' in row and not pd.isna(row.get('age')) else 0
        
        # Handle missing 'initial_mask_wearing' column - will be set from train_data day 0 later
        initial_mask_wearing = bool(row['initial_mask_wearing']) if 'initial_mask_wearing' in row else False
        
        agent = Agent(
            agent_id=aid,
            age=age,
            age_group=str(row['age_group']),
            occupation=str(row['occupation']),
            risk_perception=float(row['risk_perception']),
            initial_mask_wearing=initial_mask_wearing,
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
        aid = str(row['agent_id'])
        if aid in agents:
            agents[aid].wearing_mask = bool(row['wearing_mask'])
            # Update initial_mask_wearing from day 0 data if available
            agents[aid].initial_mask_wearing = bool(row['wearing_mask'])
            if 'received_info' in row:
                agents[aid].received_info = bool(row['received_info'])
            else:
                agents[aid].received_info = False
            agents[aid].memory_of_info = 1.0 if agents[aid].received_info else 0.0
    
    # For agents not in day0, use initial_mask_wearing (or default False)
    for aid in agents:
        if aid not in day0_data['agent_id'].values:
            agents[aid].wearing_mask = agents[aid].initial_mask_wearing
            agents[aid].received_info = False
            agents[aid].memory_of_info = 0.0
    
    # Load test_data
    test_data = pd.read_csv(TEST_DATA_PATH)
    test_data['day'] = test_data['day'].astype(int)
    test_data['agent_id'] = test_data['agent_id'].astype(str)
    test_data['wearing_mask'] = test_data['wearing_mask'].astype(bool)
    if 'received_info' in test_data.columns:
        test_data['received_info'] = test_data['received_info'].astype(bool)
    else:
        test_data['received_info'] = False
    
    # Align test_data with agent_ids
    test_data = test_data[test_data['agent_id'].isin(agent_ids)]
    
    return agents, network, train_data, test_data

def holdout_split(train_data: pd.DataFrame) -> Tuple[List[int], List[int]]:
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
    agents, network, _, _ = load_data()
    return agents, network

def calibrate_parameters(agents: Dict[str, Agent], network: Network, train_data: pd.DataFrame, train_days: List[str]) -> Dict[str, float]:
    """Initialize parameters and calibrate using gradient-based algorithm."""
    # Define initial parameters with heuristic defaults
    initial_params = {
        'alpha': 0.0,
        'gamma': 1.0,
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

def simulate_validation(agents: Dict[str, Agent], network: Network, params: Dict[str, float], eval_days: List[int], train_days: List[int]) -> List[Dict[str, Any]]:
    """Run forward simulation on evaluation window."""
    # Advance agents through all training days to reach the last training day state
    # Create a copy of agents for simulation
    sim_agents = {}
    for aid, agent in agents.items():
        sim_agents[aid] = Agent(
            agent_id=agent.agent_id,
            age=agent.age,
            age_group=agent.age_group,
            occupation=agent.occupation,
            risk_perception=agent.risk_perception,
            initial_mask_wearing=agent.initial_mask_wearing,
            layer_degrees=agent.layer_degrees,
            wearing_mask=agent.wearing_mask,
            received_info=agent.received_info,
            memory_of_info=agent.memory_of_info,
            neighbors=agent.neighbors
        )
    
    # Create simulator to advance through training days
    simulator = Simulator(sim_agents, network, params)
    
    # Get sorted training days
    sorted_train_days = sorted(train_days)
    
    # Simulate from day 0 to last training day (inclusive)
    # We need to simulate every day in the training period to reach the final state
    max_train_day = max(sorted_train_days)
    for day in range(max_train_day + 1):
        if day in sorted_train_days:
            simulator.step(day)
    
    # Now sim_agents are at the last training day state
    # Create a new simulator for validation with the updated state
    validation_simulator = Simulator(sim_agents, network, params)
    return validation_simulator.rollout(eval_days)

def main():
    """Orchestrate the full simulation workflow."""
    # Load data
    agents, network, train_data, test_data = load_data()
    
    # Split days into training and validation
    train_days, val_days = holdout_split(train_data)
    
    # Calibrate parameters using training data
    calibrated_params = calibrate_parameters(agents, network, train_data, train_days)
    
    # Advance agent states through all training days to reach the final training state
    # This is critical to avoid state drift in validation
    simulation_history = simulate_validation(agents, network, calibrated_params, val_days, train_days)
    
    # Evaluate on validation set
    evaluator = Evaluator(agents, network, test_data, val_days)
    metrics = evaluator.compute_metrics(simulation_history)
    
    # Output results (to stdout or file)
    print(json.dumps(metrics, indent=2))

# Execute main for both direct execution and sandbox wrapper invocation
main()