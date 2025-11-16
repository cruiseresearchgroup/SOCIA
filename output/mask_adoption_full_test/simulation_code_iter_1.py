def main():
    pass

import os
import json
import math
import random
import traceback
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Optional visualization; gracefully degrade if unavailable
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Optional statsmodels for calibration
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# Optional networkx for cascade metrics (not critical for RMSE)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False


# Path handling instructions (as specified)
import os
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# FIXED: Provide default fallbacks for PROJECT_ROOT and DATA_PATH to ensure the code runs when env vars are unset.
if PROJECT_ROOT is None:
    PROJECT_ROOT = os.getcwd()
if DATA_PATH is None:
    DATA_PATH = "data_fitting/mask_adoption_data"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def sigmoid(x: float) -> float:
    """
    Compute the logistic sigmoid of a value.

    Returns:
        float: The sigmoid value for x within (0,1).
    """
    pass
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp a numeric value into the closed interval [lo, hi].

    Args:
        x (float): Value to clamp.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: Clamped value.
    """
    pass
    return max(lo, min(hi, x))


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across numpy and Python's random.

    Args:
        seed (int): The seed value.

    Returns:
        None
    """
    pass
    np.random.seed(seed)
    random.seed(seed)


MODEL_PLAN: Dict[str, Any] = {
    "model_type": "agent_based",
    "description": "A multiplex networked agent-based simulation calibrated on the first 30 days of data to predict individual mask-wearing behavior for days 30–39.",
    "entities": [
        {"name": "Person"},
        {"name": "Network"},
        {"name": "Location"},
        {"name": "Organization"},
        {"name": "Environment"}
    ],
    "behaviors": [],
    "interactions": [],
    "environment": {
        "type": "network",
        "dimensions": None,
        "time_step": 1,
        "time_unit": "days"
    },
    "parameters": {
        "data_folder": "data_fitting/mask_adoption_data/",
        "num_agents": "infer_from_data",
        "network_topology": "multiplex_small_world_like",
        "avg_degree": "empirical_by_layer",
        "symmetrize_edges": True,
        "rewiring_prob": 0.1,
        "edge_weight_mean": 1.0,
        "layer_weights": {
            "family": 1.0,
            "work_school": 0.6,
            "community": 0.3
        },
        "seed_fraction": 0.05,
        "seed_selection_strategy": "random",
        "influencer_fraction": 0.02,
        "influencer_reach_multiplier": 3.0,
        "adoption_rule": "logistic",
        "beta0_intercept": "estimate_from_train_data",
        "beta_persist": "estimate_from_train_data",
        "beta_info": "estimate_from_train_data",
        "beta_peer_family": "estimate_from_train_data",
        "beta_peer_work": "estimate_from_train_data",
        "beta_peer_comm": "estimate_from_train_data",
        "beta_risk": "estimate_from_train_data",
        "beta_policy": "estimate_from_train_data",
        "beta_normcue": "estimate_from_train_data",
        "dropout_probability": "estimate_from_transitions",
        "fatigue_rate": 0.005,
        "memory_length_days": 7,
        "observation_noise": 0.05,
        "trust_mean": 0.6,
        "trust_std": 0.2,
        "risk_signal": 0.5,
        "misinformation_rate": 0.0,
        "policy_level": "none",
        "policy_start_day": 10,
        "enforcement_strength": 0.0,
        "campaign_intensity": 0.0,
        "campaign_start_day": 30,
        "campaign_end_day": 90,
        "location_counts": {
            "households": 250,
            "workplaces": 50,
            "public_spaces": 10
        },
        "time_horizon_days": 180,
        "update_mode": "synchronous",
        "random_seed": 42,
        "training_days": [0, 29],
        "prediction_days": [30, 39],
        "info_arrival_prob_base": "estimate_from_train_data",
        "social_info_boost": "estimate_from_train_data",
        "info_effect_half_life_days": 3,
        "mandate_threshold_shift": -0.15,
        "noise_std": "calibrate"
    },
    "initialization": {},
    "prediction_period": {
        "start_day": 30,
        "end_day": 39
    },
    "evaluation_metrics": ["RMSE"]
}


@dataclass
class Person:
    """
    Represents an individual agent in the simulation with attributes and behaviors
    related to mask-wearing adoption dynamics.

    Attributes:
        id (int): Unique identifier.
        age (Optional[int]): Age if available; otherwise None.
        demographic_group (Optional[str]): Demographic segment label.
        risk_attitude (float): Baseline propensity impacting adoption (0-1).
        conformity_level (float): Susceptibility to norms (0-1).
        trust_in_authorities (float): Trust (0-1).
        trust_in_peers (float): Trust (0-1).
        perceived_risk (float): Perceived risk (0-1).
        mask_wearing_state (int): Current mask wearing (0 or 1).
        adoption_threshold (float): Threshold for threshold-based models (unused if logistic).
        influence_weight (float): Influence weight used in some rules.
        stubbornness (float): Resistance to change (0-1).
        fatigue_level (float): Accumulated fatigue leading to dropouts (0-1).
        memory_length_days (int): Days considered for norm memory.
        household_id (Optional[int]): Household membership.
        workplace_id (Optional[int]): Workplace membership.
        community_id (Optional[int]): Community membership.
        is_influencer (bool): Influencer status.
        received_info_state (int): Indicator whether info was received at current day.
        last_info_day (Optional[int]): Last day when info was received.
        degree_family (int): Degree in family layer.
        degree_work_school (int): Degree in work/school layer.
        degree_community (int): Degree in community layer.
        norm_history (deque): Rolling history of weighted peer fractions for perceived norm.
        days_since_adopt (int): Number of days since adoption.
    """
    pass
    id: int = -1
    age: Optional[int] = None
    demographic_group: Optional[str] = None
    risk_attitude: float = 0.5
    conformity_level: float = 0.5
    trust_in_authorities: float = 0.6
    trust_in_peers: float = 0.6
    perceived_risk: float = 0.5
    mask_wearing_state: int = 0
    adoption_threshold: float = 0.4
    influence_weight: float = 1.0
    stubbornness: float = 0.1
    fatigue_level: float = 0.0
    memory_length_days: int = 7
    household_id: Optional[int] = None
    workplace_id: Optional[int] = None
    community_id: Optional[int] = None
    is_influencer: bool = False
    received_info_state: int = 0
    last_info_day: Optional[int] = None
    degree_family: int = 0
    degree_work_school: int = 0
    degree_community: int = 0
    norm_history: deque = field(default_factory=lambda: deque(maxlen=7))
    days_since_adopt: int = 0

    def observe_neighbors(self, day: int, peer_fracs_by_layer: Dict[str, float], observation_noise: float) -> Dict[str, float]:
        """
        Observe neighbors' mask wearing fraction per network layer with observation noise.

        Args:
            day (int): Current simulation day.
            peer_fracs_by_layer (Dict[str, float]): Exact peer fractions by layer from the network at t-1.
            observation_noise (float): Gaussian noise std to apply to observation.

        Returns:
            Dict[str, float]: Observed peer fractions after noise and clamping.
        """
        pass
        observed = {}
        for layer, frac in peer_fracs_by_layer.items():
            noise = np.random.normal(0.0, observation_noise)
            observed[layer] = clamp(frac + noise, 0.0, 1.0)
        return observed

    def update_perceived_norm(self, day: int, observed_peer_fracs: Dict[str, float], layer_weights: Dict[str, float], decay_lambda: float = 0.5) -> float:
        """
        Update the perceived social norm using an exponentially decayed memory of weighted peer fractions.

        Args:
            day (int): Current day.
            observed_peer_fracs (Dict[str, float]): Observed peer fractions by layer.
            layer_weights (Dict[str, float]): Layer weights for combining peer fractions.
            decay_lambda (float): Decay rate for memory weighting.

        Returns:
            float: Updated perceived norm value.
        """
        pass
        weighted_value = 0.0
        denom = 0.0
        for layer, w in layer_weights.items():
            weighted_value += w * observed_peer_fracs.get(layer, 0.0)
            denom += w
        weighted_peer = weighted_value / denom if denom > 0 else 0.0
        self.norm_history.appendleft(weighted_peer)

        # Compute exponentially decayed average
        num = 0.0
        den = 0.0
        for k, val in enumerate(self.norm_history, start=1):
            weight = math.exp(-decay_lambda * (k - 1))
            num += weight * val
            den += weight
        norm_value = num / den if den > 0 else 0.0
        return norm_value

    def share_information_or_misinformation(self, day: int, info_arrival_prob_base: float, social_info_boost: float, weighted_peer_frac_prev: float, misinformation_rate: float = 0.0) -> int:
        """
        Sample whether the agent receives information or misinformation.

        Args:
            day (int): Current day.
            info_arrival_prob_base (float): Baseline info arrival probability.
            social_info_boost (float): Multiplier based on social exposure.
            weighted_peer_frac_prev (float): Weighted peer fraction at t-1.
            misinformation_rate (float): Rate of misinformation events.

        Returns:
            int: 1 if info received, else 0 (misinformation treated as not-info for adoption effect).
        """
        pass
        # Probability for info arrival
        p_info = clamp(info_arrival_prob_base + social_info_boost * weighted_peer_frac_prev, 0.0, 1.0)
        received_info = 1 if np.random.rand() < p_info else 0
        # Misinformation: reduces trust in authorities and negates info effect at some rate
        if misinformation_rate > 0.0:
            if np.random.rand() < misinformation_rate:
                # Trust penalty
                self.trust_in_authorities = clamp(self.trust_in_authorities - 0.05, 0.0, 1.0)
                # Treat misinformation as negative info; we do not set received_info_state to 1
                received_info = 0
        if received_info == 1:
            self.received_info_state = 1
            self.last_info_day = day
        else:
            self.received_info_state = 0
        return received_info

    def respond_to_policy_enforcement(self, day: int, policy_start_day: int, enforcement_strength: float, current_intended_state: int) -> int:
        """
        Apply policy enforcement: with some probability, override intended state to wearing.

        Args:
            day (int): Current day.
            policy_start_day (int): Day when policy starts.
            enforcement_strength (float): Probability of enforcement override if in a policy location.
            current_intended_state (int): State decided by behavior model before enforcement.

        Returns:
            int: Possibly overridden mask wearing state.
        """
        pass
        if day >= policy_start_day and enforcement_strength > 0.0:
            if np.random.rand() < enforcement_strength:
                return 1
        return current_intended_state

    def decide_adopt_or_drop_mask(
        self,
        day: int,
        coeffs: Dict[str, float],
        wear_prev: int,
        info_received: int,
        risk: float,
        peer_family_prev: float,
        peer_work_prev: float,
        peer_comm_prev: float,
        policy_t: int,
        norm_prev: float,
        dropout_probability: float,
        fatigue_rate: float,
        info_effect_half_life_days: int,
    ) -> int:
        """
        Decide next-day mask wearing using a logistic model and apply dropout and fatigue.

        Args:
            day (int): Current day t.
            coeffs (Dict[str, float]): Logistic coefficients dict.
            wear_prev (int): Wearing state at t-1.
            info_received (int): Whether info received at t.
            risk (float): Perceived risk.
            peer_family_prev (float): Peer fraction family at t-1.
            peer_work_prev (float): Peer fraction work at t-1.
            peer_comm_prev (float): Peer fraction community at t-1.
            policy_t (int): Policy indicator at t.
            norm_prev (float): Perceived norm at t-1.
            dropout_probability (float): Base dropout probability.
            fatigue_rate (float): Fatigue increment per day of continued wearing.
            info_effect_half_life_days (int): Half-life for info effect.

        Returns:
            int: New wearing state for day t.
        """
        pass
        beta0 = coeffs.get("beta0_intercept", 0.0)
        beta_persist = coeffs.get("beta_persist", 0.0)
        beta_info = coeffs.get("beta_info", 0.0)
        beta_peer_family = coeffs.get("beta_peer_family", 0.0)
        beta_peer_work = coeffs.get("beta_peer_work", 0.0)
        beta_peer_comm = coeffs.get("beta_peer_comm", 0.0)
        beta_risk = coeffs.get("beta_risk", 0.0)
        beta_policy = coeffs.get("beta_policy", 0.0)
        beta_normcue = coeffs.get("beta_normcue", 0.0)

        x = beta0
        x += beta_persist * wear_prev
        x += beta_info * info_received
        x += beta_risk * risk
        x += beta_peer_family * peer_family_prev
        x += beta_peer_work * peer_work_prev
        x += beta_peer_comm * peer_comm_prev
        x += beta_policy * policy_t
        x += beta_normcue * norm_prev

        # Info decay effects if info was received in recent past days
        if self.last_info_day is not None:
            k = max(0, day - self.last_info_day)
            if k >= 0:
                decay = math.exp(-math.log(2) * k / max(1, info_effect_half_life_days))
                x += beta_info * decay

        p = sigmoid(x)
        intended_wear = 1 if np.random.rand() < p else 0

        # Dropout logic if persisted wearing
        if wear_prev == 1 and intended_wear == 1:
            p_drop_eff = min(1.0, dropout_probability + fatigue_rate * max(0, self.days_since_adopt))
            if np.random.rand() < p_drop_eff:
                intended_wear = 0

        # Update fatigue and days since adopt
        if intended_wear == 1:
            self.days_since_adopt = 0 if wear_prev == 0 else self.days_since_adopt + 1
            self.fatigue_level = clamp(self.fatigue_level + fatigue_rate, 0.0, 1.0)
        else:
            self.fatigue_level = clamp(self.fatigue_level - fatigue_rate, 0.0, 1.0)
            if wear_prev == 1:
                self.days_since_adopt = 0

        return intended_wear


@dataclass
class SocialNetwork:
    """
    Represents a multiplex social network with family, work/school, and community layers.

    Attributes:
        adjacency (Dict[str, Dict[int, List[int]]]): Layered adjacency lists.
        edge_weights (Dict[str, Dict[Tuple[int,int], float]]): Edge weights per layer (optional).
        layer_weights (Dict[str, float]): Layer weights for influence aggregation.
    """
    pass
    adjacency: Dict[str, Dict[int, List[int]]] = field(default_factory=lambda: {"family": {}, "work_school": {}, "community": {}})
    edge_weights: Dict[str, Dict[Tuple[int, int], float]] = field(default_factory=lambda: {"family": {}, "work_school": {}, "community": {}})
    layer_weights: Dict[str, float] = field(default_factory=lambda: {"family": 1.0, "work_school": 0.6, "community": 0.3})

    def symmetrize(self) -> None:
        """
        Symmetrize each network layer to ensure undirected influence.
        """
        pass
        for layer, adj in self.adjacency.items():
            new_adj = defaultdict(set)
            for u, nbrs in adj.items():
                for v in nbrs:
                    if u == v:
                        continue
                    new_adj[u].add(v)
                    new_adj[v].add(u)
            # Convert sets back to lists
            self.adjacency[layer] = {u: sorted(list(vs)) for u, vs in new_adj.items()}

    def compute_degrees(self) -> Dict[str, Dict[int, int]]:
        """
        Compute degree per layer for each node.

        Returns:
            Dict[str, Dict[int,int]]: Degree map per layer.
        """
        pass
        degrees = {}
        for layer, adj in self.adjacency.items():
            layer_deg = {u: len(nbrs) for u, nbrs in adj.items()}
            degrees[layer] = layer_deg
        return degrees

    def compute_peer_fractions(self, layer: str, state_by_agent: Dict[int, int]) -> Dict[int, float]:
        """
        Compute peer fraction by agent for a specific layer given current states.

        Args:
            layer (str): Layer name.
            state_by_agent (Dict[int,int]): Current wearing states by agent.

        Returns:
            Dict[int, float]: Fraction of neighbors wearing masks for each agent.
        """
        pass
        result = {}
        adj = self.adjacency.get(layer, {})
        for u, nbrs in adj.items():
            if not nbrs:
                result[u] = 0.0
            else:
                wearing = sum(state_by_agent.get(v, 0) for v in nbrs)
                result[u] = wearing / len(nbrs)
        return result

    def weighted_peer_fraction(self, agent_id: int, peer_fracs_by_layer: Dict[str, float]) -> float:
        """
        Compute weighted peer fraction across layers for an agent.

        Args:
            agent_id (int): Agent ID (unused but available for extensions).
            peer_fracs_by_layer (Dict[str,float]): Peer fraction per layer.

        Returns:
            float: Weighted combined peer fraction.
        """
        pass
        num = 0.0
        den = 0.0
        for layer, w in self.layer_weights.items():
            num += w * peer_fracs_by_layer.get(layer, 0.0)
            den += w
        return num / den if den > 0 else 0.0


@dataclass
class Location:
    """
    Represents a location entity with policy enforcement attributes.

    Attributes:
        id (int): Identifier.
        type (str): Type of location (household, workplace, public_space).
        capacity (int): Capacity.
        mask_policy (bool): Whether mask policy is enforced.
        enforcement_strength (float): Strength of enforcement (0-1).
        signage_or_cue_level (float): Norm cue level.
        open_hours (Tuple[int,int]): Opening hours for effect modeling.
    """
    pass
    id: int = -1
    type: str = "public_space"
    capacity: int = 0
    mask_policy: bool = False
    enforcement_strength: float = 0.0
    signage_or_cue_level: float = 0.0
    open_hours: Tuple[int, int] = (0, 24)

    def enforce_policy(self, person: Person, day: int, policy_start_day: int) -> int:
        """
        Enforce mask policy on a given person.

        Args:
            person (Person): Target person.
            day (int): Current day.
            policy_start_day (int): Policy start day.

        Returns:
            int: Possibly overridden wearing state.
        """
        pass
        if self.mask_policy:
            return person.respond_to_policy_enforcement(day, policy_start_day, self.enforcement_strength, person.mask_wearing_state)
        return person.mask_wearing_state

    def apply_social_norm_cues(self, person: Person) -> None:
        """
        Adjust person's perceived norm via signage or cues.
        """
        pass
        # Simple cue effect: slight increase in conformity
        person.conformity_level = clamp(person.conformity_level + 0.01 * self.signage_or_cue_level, 0.0, 1.0)

    def record_compliance(self, person: Person) -> None:
        """
        Placeholder for recording compliance data for metrics.

        Args:
            person (Person): Target person.

        Returns:
            None
        """
        pass
        # No-op for now


@dataclass
class Organization:
    """
    Represents an organization that broadcasts messages affecting information exposure.

    Attributes:
        id (int): Identifier.
        type (str): Organization type (e.g., 'gov', 'ngo').
        stance (str): Message stance ('pro-mask' or 'anti-mask').
        credibility (float): Perceived credibility (0-1).
        reach (float): Fraction of population reached daily (0-1).
        targeting_strategy (str): Targeting strategy.
        message_schedule (Tuple[int, int]): Start and end day for broadcasting.
        campaign_intensity (float): Intensity (0-1).
    """
    pass
    id: int = 0
    type: str = "gov"
    stance: str = "pro-mask"
    credibility: float = 0.7
    reach: float = 0.1
    targeting_strategy: str = "degree_or_risk_targeting"
    message_schedule: Tuple[int, int] = (30, 90)
    campaign_intensity: float = 0.0

    def broadcast_message(self, day: int, agents: Dict[int, Person]) -> List[int]:
        """
        Broadcast messages to selected agents based on targeting and intensity.

        Args:
            day (int): Current day.
            agents (Dict[int,Person]): Agent registry.

        Returns:
            List[int]: IDs of agents targeted today.
        """
        pass
        start, end = self.message_schedule
        if not (start <= day <= end):
            return []

        # Determine target count based on reach
        N = len(agents)
        target_count = int(self.reach * N)
        if target_count <= 0:
            return []

        # Targeting strategy: prioritize high risk or high degree proxy
        agent_items = list(agents.items())
        if self.targeting_strategy == "degree_or_risk_targeting":
            # Score by risk + degree sum
            scored = []
            for aid, a in agent_items:
                deg_sum = a.degree_family + a.degree_work_school + a.degree_community
                score = 0.7 * a.perceived_risk + 0.3 * (deg_sum / (1 + deg_sum))
                scored.append((aid, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            targets = [aid for aid, _ in scored[:target_count]]
        else:
            # Random fallback
            targets = [aid for aid, _ in random.sample(agent_items, min(target_count, N))]

        # Apply info effect probabilistically with credibility and intensity
        for aid in targets:
            agent = agents[aid]
            p_info = clamp(self.campaign_intensity * self.credibility * agent.trust_in_authorities, 0.0, 1.0)
            if np.random.rand() < p_info:
                agent.received_info_state = 1
                agent.last_info_day = day
        return targets


@dataclass
class Environment:
    """
    Global environment affecting risk signals and policy level.

    Attributes:
        global_risk_signal (float): Global risk factor (0-1).
        misinformation_level (float): Misinformation level (0-1).
        policy_level (str): Policy level (e.g., 'none', 'low', 'high').
        shock_events (List[Dict]): Shock events modifying parameters.
    """
    pass
    global_risk_signal: float = 0.5
    misinformation_level: float = 0.0
    policy_level: str = "none"
    shock_events: List[Dict[str, Any]] = field(default_factory=list)

    def update_risk_signal(self, day: int) -> None:
        """
        Optionally update the global risk signal based on shocks.

        Args:
            day (int): Current day.

        Returns:
            None
        """
        pass
        # Placeholder: No dynamic shocks implemented
        return None

    def trigger_shock_event(self, day: int) -> None:
        """
        Trigger shock events that alter environment.

        Args:
            day (int): Current day.

        Returns:
            None
        """
        pass
        # Placeholder: No dynamic shocks implemented
        return None


class DataLoader:
    """
    Load and validate agent attributes, network, and training data from files.

    Methods:
        load_agents
        load_network
        load_train_data
        build_panel_features
    """
    pass

    def __init__(self, data_dir: str):
        """
        Initialize DataLoader with a base data directory.

        Args:
            data_dir (str): Base data directory path.
        """
        pass
        self.data_dir = data_dir

    def _safe_join(self, filename: str) -> str:
        """
        Join data dir with filename.

        Args:
            filename (str): File name.

        Returns:
            str: Full path to file.
        """
        pass
        return os.path.join(self.data_dir, filename)

    def load_agents(self) -> pd.DataFrame:
        """
        Load agent_attributes.csv with expected columns.

        Returns:
            pd.DataFrame: Agent attributes dataframe (empty if file missing).
        """
        pass
        try:
            agent_file = self._safe_join("agent_attributes.csv")
            if not os.path.exists(agent_file):
                return pd.DataFrame()
            df = pd.read_csv(agent_file)
            # Normalize types
            if "agent_id" in df.columns:
                df["agent_id"] = df["agent_id"].astype(int)
            if "risk_perception" in df.columns:
                # Clamp to [0, 1]
                df["risk_perception"] = df["risk_perception"].astype(float).clip(0, 1)
            return df
        except Exception:
            traceback.print_exc()
            return pd.DataFrame()

    def load_network(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Load social_network.json and return multiplex adjacency dict.

        Returns:
            Dict[str, Dict[int, List[int]]]: Layered adjacency dictionaries keyed by layer name.
        """
        pass
        try:
            net_file = self._safe_join("social_network.json")
            if not os.path.exists(net_file):
                return {"family": {}, "work_school": {}, "community": {}}
            with open(net_file, "r") as f:
                network_data = json.load(f)
            # Build layers
            layers = {"family": {}, "work_school": {}, "community": {}}
            for k, v in network_data.items():
                try:
                    u = int(k)
                except Exception:
                    continue
                for layer in ["family", "work_school", "community"]:
                    nbrs = v.get(layer, [])
                    cleaned = []
                    for n in nbrs:
                        try:
                            cleaned.append(int(n))
                        except Exception:
                            continue
                    layers[layer].setdefault(u, [])
                    layers[layer][u].extend(cleaned)
            return layers
        except Exception:
            traceback.print_exc()
            return {"family": {}, "work_school": {}, "community": {}}

    def load_train_data(self) -> pd.DataFrame:
        """
        Load train_data.csv panel of days and states.

        Returns:
            pd.DataFrame: Training panel (empty if missing).
        """
        pass
        try:
            train_file = self._safe_join("train_data.csv")
            if not os.path.exists(train_file):
                return pd.DataFrame()
            df = pd.read_csv(train_file)
            # Normalize types
            if "agent_id" in df.columns:
                df["agent_id"] = df["agent_id"].astype(int)
            if "day" in df.columns:
                df["day"] = df["day"].astype(int)
            if "wearing_mask" in df.columns:
                df["wearing_mask"] = df["wearing_mask"].astype(int).clip(0, 1)
            if "received_info" in df.columns:
                df["received_info"] = df["received_info"].astype(int).clip(0, 1)
            # Sort
            df = df.sort_values(["agent_id", "day"]).reset_index(drop=True)
            return df
        except Exception:
            traceback.print_exc()
            return pd.DataFrame()

    def build_panel_features(
        self,
        train_df: pd.DataFrame,
        network: SocialNetwork,
        memory_length_days: int = 7,
        decay_lambda: float = 0.5,
        policy_start_day: int = 10
    ) -> pd.DataFrame:
        """
        Build feature matrix for logistic calibration: wear_t, wear_{t-1}, info_t, risk, peer fractions at t-1 by layer, policy_t, norm_{t-1}.

        Args:
            train_df (pd.DataFrame): Training panel data.
            network (SocialNetwork): Multiplex network.
            memory_length_days (int): Memory length for norm computation.
            decay_lambda (float): Decay parameter for norm.
            policy_start_day (int): Policy start day.

        Returns:
            pd.DataFrame: Feature-enriched DataFrame aligned at t rows (dropping t=0).
        """
        pass
        if train_df.empty:
            return pd.DataFrame()

        # If risk_perception exists in train_df; else join from agents
        # Attempt to load agents and merge
        try:
            agents_df = self.load_agents()
            if not agents_df.empty:
                keep_cols = ["agent_id", "risk_perception"]
                for col in ["age_group", "occupation"]:
                    if col in agents_df.columns:
                        keep_cols.append(col)
                agents_df = agents_df[keep_cols].drop_duplicates("agent_id")
                train_df = train_df.merge(agents_df, on="agent_id", how="left")
        except Exception:
            traceback.print_exc()

        # Build dict of wearing by (day, agent)
        wear_map = {(int(r.agent_id), int(r.day)): int(r.wearing_mask) for r in train_df.itertuples()}

        # Precompute peer fractions per layer for t-1 for each (agent, t)
        rows = []
        min_day = train_df["day"].min()
        max_day = train_df["day"].max()

        # Degree maps to fill zeros for missing
        degrees = network.compute_degrees()

        # For each agent, maintain norm history (weighted peer) as we iterate days
        agent_norm_hist: Dict[int, deque] = {}
        for aid in train_df["agent_id"].unique():
            agent_norm_hist[aid] = deque(maxlen=memory_length_days)

        # Prepare adjacency for speed
        adj = network.adjacency

        for t in range(min_day + 1, max_day + 1):
            day_df = train_df[train_df["day"] == t]
            # Build state_by_agent at t-1
            state_prev = {}
            for aid in day_df["agent_id"].values:
                state_prev[aid] = wear_map.get((aid, t - 1), 0)

            # Compute per-layer peer fracs at t-1 for all agents present at t
            layer_fracs_prev = {"family": {}, "work_school": {}, "community": {}}
            for layer in ["family", "work_school", "community"]:
                cur_adj = adj.get(layer, {})
                for aid in day_df["agent_id"].values:
                    nbrs = cur_adj.get(aid, [])
                    if len(nbrs) == 0:
                        layer_fracs_prev[layer][aid] = 0.0
                    else:
                        wearing = sum(wear_map.get((nid, t - 1), 0) for nid in nbrs)
                        layer_fracs_prev[layer][aid] = wearing / len(nbrs)

            # Compute norm_{t-1} via decayed average of weighted peer fractions
            for r in day_df.itertuples():
                aid = int(r.agent_id)
                peer_family_prev = float(layer_fracs_prev["family"].get(aid, 0.0))
                peer_work_prev = float(layer_fracs_prev["work_school"].get(aid, 0.0))
                peer_comm_prev = float(layer_fracs_prev["community"].get(aid, 0.0))

                # Weighted peer fraction using network layer weights
                w = network.layer_weights
                denom = sum(w.values()) if len(w) > 0 else 1.0
                weighted_peer_prev = (
                    w.get("family", 0.0) * peer_family_prev
                    + w.get("work_school", 0.0) * peer_work_prev
                    + w.get("community", 0.0) * peer_comm_prev
                ) / denom if denom > 0 else 0.0

                # Update per-agent norm history
                nh = agent_norm_hist[aid]
                nh.appendleft(weighted_peer_prev)

                # Decayed norm_prev from history
                num = 0.0
                den = 0.0
                for k, val in enumerate(nh, start=1):
                    wt = math.exp(-decay_lambda * (k - 1))
                    num += wt * val
                    den += wt
                norm_prev = num / den if den > 0 else 0.0

                # Features
                wear_prev = int(state_prev.get(aid, 0))
                wear_t = int(getattr(r, "wearing_mask"))
                info_t = int(getattr(r, "received_info", 0))
                risk_val = getattr(r, "risk_perception", np.nan)
                risk_val = 0.5 if (risk_val is None or (isinstance(risk_val, float) and np.isnan(risk_val))) else float(risk_val)
                policy_t = 1 if t >= policy_start_day else 0

                rows.append({
                    "agent_id": aid,
                    "day": int(t),
                    "wear_t": wear_t,
                    "wear_prev": wear_prev,
                    "info_t": info_t,
                    "risk": risk_val,
                    "peer_family_prev": peer_family_prev,
                    "peer_work_prev": peer_work_prev,
                    "peer_comm_prev": peer_comm_prev,
                    "policy_t": policy_t,
                    "norm_prev": float(norm_prev)
                })

        feat_df = pd.DataFrame(rows)
        return feat_df


# Execute main for both direct execution and sandbox wrapper invocation
main()