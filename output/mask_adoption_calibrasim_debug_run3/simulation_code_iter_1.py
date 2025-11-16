import os
import json
import random
import traceback
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Environment Path Handling (as per instructions)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


# -----------------------------------------------------------------------------
# Model Plan (Source of Truth) JSON (parsed at runtime)
# -----------------------------------------------------------------------------
MODEL_PLAN_JSON = r"""
{
  "model_type": "agent_based",
  "description": "A multiplex network-based, agent-centered simulation of mask-wearing adoption that couples information diffusion and social influence with policy and messaging effects. The model is calibrated on the first 30 days of time series data and predicts days 30–39.",
  "parameters": [
    {
      "key": "time_horizon_days",
      "dtype": "int",
      "default": 40,
      "owner_module": "global",
      "frozen": "true"
    },
    {
      "key": "timestep_days",
      "dtype": "int",
      "default": 1,
      "owner_module": "global",
      "frozen": "true"
    },
    {
      "key": "seed",
      "dtype": "int",
      "default": 42,
      "owner_module": "global",
      "frozen": "true"
    },
    {
      "key": "symmetrize_edges",
      "dtype": "bool",
      "default": true,
      "owner_module": "NetworkLayerEngine",
      "frozen": "true"
    },
    {
      "key": "degree_cap_percentile",
      "dtype": "float",
      "default": 0.99,
      "owner_module": "NetworkLayerEngine",
      "frozen": "false"
    },
    {
      "key": "policy_mandate_day",
      "dtype": "int",
      "default": 10,
      "owner_module": "PolicyAndMessaging",
      "frozen": "true"
    },
    {
      "key": "policy_mandate_effect_multiplier",
      "dtype": "float",
      "default": 1.5,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "enforcement_probability",
      "dtype": "float",
      "default": 0.1,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "messaging_effect_size",
      "dtype": "float",
      "default": 0.15,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "misinformation_prevalence",
      "dtype": "float",
      "default": 0.2,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "misinformation_effect_size",
      "dtype": "float",
      "default": 0.3,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "base_messaging_intensity",
      "dtype": "float",
      "default": 0.2,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "base_policy_level",
      "dtype": "float",
      "default": 0.4,
      "owner_module": "PolicyAndMessaging",
      "frozen": "false"
    },
    {
      "key": "info_broadcast_rate_per_day",
      "dtype": "float",
      "default": 0.02,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "peer_info_transmission_prob_family",
      "dtype": "float",
      "default": 0.25,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "peer_info_transmission_prob_work_school",
      "dtype": "float",
      "default": 0.15,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "peer_info_transmission_prob_community",
      "dtype": "float",
      "default": 0.07,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "edge_activation_rate_per_day_family",
      "dtype": "float",
      "default": 1.0,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "edge_activation_rate_per_day_work_school",
      "dtype": "float",
      "default": 0.5,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "edge_activation_rate_per_day_community",
      "dtype": "float",
      "default": 0.2,
      "owner_module": "InformationDiffusion",
      "frozen": "false"
    },
    {
      "key": "base_adoption_rate",
      "dtype": "float",
      "default": 0.005,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "beta_family",
      "dtype": "float",
      "default": 1.2,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "beta_work_school",
      "dtype": "float",
      "default": 0.8,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "beta_community",
      "dtype": "float",
      "default": 0.4,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "beta_info",
      "dtype": "float",
      "default": 2.0,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "beta_risk_perception",
      "dtype": "float",
      "default": 0.8,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "policy_effect_weight",
      "dtype": "float",
      "default": 0.3,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "enforcement_elasticity",
      "dtype": "float",
      "default": 0.5,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "risk_threshold_gamma",
      "dtype": "float",
      "default": 1.0,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "perceived_cost_weight",
      "dtype": "float",
      "default": 0.3,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "social_noise_sigma",
      "dtype": "float",
      "default": 0.15,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "habit_persistence",
      "dtype": "float",
      "default": 0.92,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "abandonment_probability_per_day",
      "dtype": "float",
      "default": 0.005,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "logit_intercept_adoption",
      "dtype": "float",
      "default": -3.0,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    },
    {
      "key": "info_to_adoption_min_lag_days",
      "dtype": "int",
      "default": 1,
      "owner_module": "SocialInfluenceAdoption",
      "frozen": "false"
    }
  ],
  "prediction_period": {
    "start_day": 30,
    "end_day": 39
  }
}
"""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    result = 1.0 / (1.0 + np.exp(-x))
    return result


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def safe_mean(x: Optional[np.ndarray]) -> float:
    if x is None or len(x) == 0:
        return 0.0
    return float(np.mean(x))


# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
class DataLoader:
    """
    Load and validate CSV/JSON data for the simulation.

    This class handles:
    - Loading agent attributes from CSV.
    - Loading social network multiplex from JSON.
    - Loading training time series for initialization and evaluation.
    - Constructing derived mappings and validating data integrity.
    """

    REQUIRED_AGENT_COLS = {"agent_id", "age"}
    REQUIRED_TRAIN_COLS = {"day", "agent_id", "wearing_mask", "received_info"}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.agent_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.network_data: Optional[Dict[str, Dict[str, List[int]]]] = None

    def load_all(self) -> None:
        try:
            agent_file = os.path.join(self.data_dir, "agent_attributes.csv")
            net_file = os.path.join(self.data_dir, "social_network.json")
            train_file = os.path.join(self.data_dir, "train_data.csv")

            if os.path.exists(agent_file):
                self.agent_df = pd.read_csv(agent_file)
                self._validate_and_fix_agent_df()
            else:
                self.agent_df = self._create_synthetic_agents()
            if os.path.exists(net_file):
                with open(net_file, "r") as f:
                    self.network_data = json.load(f)
                self._validate_network_data()
            else:
                self.network_data = self._create_synthetic_network(self.agent_df)

            if os.path.exists(train_file):
                self.train_df = pd.read_csv(train_file)
                self._validate_and_fix_train_df()
            else:
                self.train_df = self._create_synthetic_train(self.agent_df)
        except Exception as e:
            print("Error loading data:", e)
            traceback.print_exc()
            # Fallback to synthetic if anything fails
            self.agent_df = self._create_synthetic_agents()
            self.network_data = self._create_synthetic_network(self.agent_df)
            self.train_df = self._create_synthetic_train(self.agent_df)

    def _validate_and_fix_agent_df(self) -> None:
        df = self.agent_df.copy()
        missing = self.REQUIRED_AGENT_COLS - set(df.columns)
        if missing:
            raise ValueError(f"agent_attributes.csv missing required columns: {missing}")
        # Fill or derive optional columns
        if "age_group" not in df.columns:
            bins = [0, 18, 35, 50, 65, 200]
            labels = ["Youth", "Young Adult", "Middle Age", "Older", "Senior"]
            df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)
        if "occupation" not in df.columns:
            df["occupation"] = "Unknown"
        if "risk_perception" not in df.columns:
            df["risk_perception"] = 0.5
        # Ensure agent_id is int and unique
        df["agent_id"] = df["agent_id"].astype(int)
        if df["agent_id"].duplicated().any():
            raise ValueError("agent_attributes.csv contains duplicate agent_id values")
        df = df.sort_values("agent_id").reset_index(drop=True)
        self.agent_df = df

    def _validate_network_data(self) -> None:
        if not isinstance(self.network_data, dict):
            raise ValueError("social_network.json must be a dictionary mapping agent_id to layer adjacency lists")
        # Ensure required keys in values
        for k, v in self.network_data.items():
            if not isinstance(v, dict):
                raise ValueError(f"Network entry for {k} must be a dict")
            for layer in ["family", "work_school", "community"]:
                if layer not in v:
                    v[layer] = []
                if not isinstance(v[layer], list):
                    raise ValueError(f"Network entry {k}:{layer} must be a list")

    def _validate_and_fix_train_df(self) -> None:
        df = self.train_df.copy()
        missing = self.REQUIRED_TRAIN_COLS - set(df.columns)
        if missing:
            raise ValueError(f"train_data.csv missing required columns: {missing}")
        df["agent_id"] = df["agent_id"].astype(int)
        df["day"] = df["day"].astype(int)
        # Coerce to integers 0/1
        df["wearing_mask"] = df["wearing_mask"].astype(int).clip(0, 1)
        df["received_info"] = df["received_info"].astype(int).clip(0, 1)
        self.train_df = df

    def _create_synthetic_agents(self, n: int = 2000) -> pd.DataFrame:
        ages = np.random.randint(15, 80, size=n)
        bins = [0, 18, 35, 50, 65, 200]
        labels = ["Youth", "Young Adult", "Middle Age", "Older", "Senior"]
        age_groups = pd.cut(ages, bins=bins, labels=labels, right=False).astype(str)
        occupations = np.random.choice(["Student", "Blue Collar", "White Collar", "Healthcare", "Unemployed"], size=n, p=[0.2, 0.3, 0.25, 0.05, 0.2])
        risk_perception = np.clip(np.random.beta(2.0, 4.0, size=n), 0.0, 1.0)
        df = pd.DataFrame({
            "agent_id": np.arange(n),
            "age": ages,
            "age_group": age_groups,
            "occupation": occupations,
            "risk_perception": risk_perception
        })
        return df

    def _create_synthetic_network(self, agent_df: pd.DataFrame, avg_deg_family=4, avg_deg_work=6, avg_deg_comm=8) -> Dict[str, Dict[str, List[int]]]:
        n = int(agent_df.shape[0])

        def sample_undirected_graph(n_nodes: int, avg_deg: int) -> Dict[int, List[int]]:
            m = int(max(0, (n_nodes * avg_deg) // 2))
            edges = set()
            attempts = 0
            max_attempts = max(10000, 5 * m)
            while len(edges) < m and attempts < max_attempts:
                i = int(np.random.randint(0, n_nodes))
                j = int(np.random.randint(0, n_nodes))
                if i == j:
                    attempts += 1
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in edges:
                    attempts += 1
                    continue
                edges.add((a, b))
                attempts += 1
            adj = {i: [] for i in range(n_nodes)}
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            return adj

        family = sample_undirected_graph(n, avg_deg_family)
        work = sample_undirected_graph(n, avg_deg_work)
        community = sample_undirected_graph(n, avg_deg_comm)

        network = {}
        for i in range(n):
            fam = family.get(i, [])
            wor = work.get(i, [])
            com = community.get(i, [])
            network[str(i)] = {
                "family": fam,
                "work_school": wor,
                "community": com,
                "all": list(set(fam + wor + com))
            }
        return network

    def _create_synthetic_train(self, agent_df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        n = agent_df.shape[0]
        records = []
        p0_mask = 0.1
        p0_info = 0.15
        mask_state = (np.random.random(n) < p0_mask)
        info_state = (np.random.random(n) < p0_info)
        for day in range(days):
            info_state = info_state | (np.random.random(n) < 0.02 + 0.002 * day)
            mask_state = mask_state | (np.random.random(n) < 0.01 + 0.002 * day + 0.05 * info_state.astype(float))
            for i in range(n):
                records.append({
                    "day": day,
                    "agent_id": i,
                    "wearing_mask": int(mask_state[i]),
                    "received_info": int(info_state[i])
                })
        df = pd.DataFrame.from_records(records)
        return df


# -----------------------------------------------------------------------------
# Entities
# -----------------------------------------------------------------------------
class Person:
    """
    Person entity representing an individual agent in the simulation.
    """

    def __init__(self, pid: int, age: int, age_group: str, occupation: str, risk_perception: float):
        self.id = pid
        self.age = int(age)
        self.age_group = str(age_group)
        self.occupation = str(occupation)
        self.household_id: Optional[int] = None
        self.workplace_id: Optional[int] = None
        self.mask_adoption_state: int = 0
        self.received_info_state: int = 0
        self.trust_in_authority: float = float(np.clip(np.random.beta(5, 5), 0, 1))
        self.risk_perception: float = float(np.clip(risk_perception, 0, 1))
        self.compliance_trait: float = float(np.clip(np.random.normal(0.5, 0.2), 0, 1))
        self.social_influence_susceptibility: float = float(np.clip(np.random.normal(1.0, 0.2), 0, 2))
        self.perceived_cost_discomfort: float = float(np.clip(np.random.beta(2, 5), 0, 1))
        self.information_exposure_level: float = float(np.clip(np.random.beta(3, 4), 0, 1))


class Household:
    def __init__(self, hid: int, size: int, socioeconomic_status: str = "medium", norm_strength: float = 0.7):
        self.id = hid
        self.size = int(size)
        self.socioeconomic_status = str(socioeconomic_status)
        self.norm_strength = float(np.clip(norm_strength, 0, 1))


class WorkplaceSchool:
    def __init__(self, wid: int, size: int, sector: str = "general", policy_stringency: float = 0.5):
        self.id = wid
        self.size = int(size)
        self.sector = str(sector)
        self.policy_stringency = float(np.clip(policy_stringency, 0, 1))


class PublicLocation:
    def __init__(self, lid: int, type_label: str, average_density: float, enforcement_level: float, signage_messaging: float):
        self.id = lid
        self.type = str(type_label)
        self.average_density = float(np.clip(average_density, 0, 1))
        self.enforcement_level = float(np.clip(enforcement_level, 0, 1))
        self.signage_messaging = float(np.clip(signage_messaging, 0, 1))


class PublicHealthAuthority:
    def __init__(self, policy_level: float = 0.0, messaging_intensity: float = 0.2, campaign_strategy: str = "mandate_on_day_10_then_steady"):
        self.policy_level = float(np.clip(policy_level, 0, 1))
        self.messaging_intensity = float(np.clip(messaging_intensity, 0, 1))
        self.campaign_strategy = str(campaign_strategy)


class InformationChannel:
    def __init__(self, reach: float = 0.8, misinformation_level: float = 0.2, message_valence: float = 0.5):
        self.reach = float(np.clip(reach, 0, 1))
        self.misinformation_level = float(np.clip(misinformation_level, 0, 1))
        self.message_valence = float(np.clip(message_valence, 0, 1))


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
class NetworkLayerEngine:
    """
    Module to construct and manage multiplex network layers and derived mappings.
    """

    def __init__(self, params: Dict[str, Any], network_data: Dict[str, Dict[str, List[int]]], n_agents: int):
        self.params = params
        self.network_data = network_data or {}
        self.n = n_agents
        self.family_adj: List[List[int]] = [[] for _ in range(n_agents)]
        self.work_adj: List[List[int]] = [[] for _ in range(n_agents)]
        self.comm_adj: List[List[int]] = [[] for _ in range(n_agents)]
        self.household_id: np.ndarray = np.full(n_agents, -1, dtype=int)
        self.workplace_id: np.ndarray = np.full(n_agents, -1, dtype=int)
        self.degree_stats: Dict[str, Dict[str, float]] = {}

    def on_init(self) -> None:
        # Parse adjacency
        for k, v in self.network_data.items():
            try:
                i = int(k)
            except Exception:
                continue
            if 0 <= i < self.n:
                self.family_adj[i] = list(sorted(set(int(x) for x in v.get("family", []) if int(x) != i and 0 <= int(x) < self.n)))
                self.work_adj[i] = list(sorted(set(int(x) for x in v.get("work_school", []) if int(x) != i and 0 <= int(x) < self.n)))
                self.comm_adj[i] = list(sorted(set(int(x) for x in v.get("community", []) if int(x) != i and 0 <= int(x) < self.n)))
        # Symmetrize if needed
        if bool(self.params.get("symmetrize_edges", True)):
            self._symmetrize(self.family_adj)
            self._symmetrize(self.work_adj)
            self._symmetrize(self.comm_adj)
        else:
            # Warn and still symmetrize for component identification
            self._symmetrize(self.family_adj)
            self._symmetrize(self.work_adj)
            # Keep community as-is for directed-like influence if desired

        # Cap community degree
        self._cap_degree(self.comm_adj, float(self.params.get("degree_cap_percentile", 0.99)))
        # Derive components
        self.household_id = self._connected_components(self.family_adj)
        self.workplace_id = self._connected_components(self.work_adj)
        # Degree diagnostics
        self.degree_stats = {
            "family": self._degree_stats(self.family_adj),
            "work_school": self._degree_stats(self.work_adj),
            "community": self._degree_stats(self.comm_adj)
        }

    def _symmetrize(self, adj: List[List[int]]) -> None:
        for i in range(self.n):
            for j in list(adj[i]):
                if i not in adj[j]:
                    adj[j].append(i)
        for i in range(self.n):
            adj[i] = list(sorted(set(adj[i])))

    def _cap_degree(self, adj: List[List[int]], percentile: float) -> None:
        degrees = np.array([len(nei) for nei in adj])
        if len(degrees) == 0:
            return
        cap = int(np.quantile(degrees, percentile))
        if cap <= 0:
            return
        for i in range(self.n):
            if len(adj[i]) > cap:
                np.random.shuffle(adj[i])
                adj[i] = adj[i][:cap]

    def _connected_components(self, adj: List[List[int]]) -> np.ndarray:
        visited = np.zeros(self.n, dtype=bool)
        comp_id = np.full(self.n, -1, dtype=int)
        cid = 0
        for i in range(self.n):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                comp_id[i] = cid
                while stack:
                    u = stack.pop()
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            comp_id[v] = cid
                            stack.append(v)
                cid += 1
        return comp_id

    def _degree_stats(self, adj: List[List[int]]) -> Dict[str, float]:
        degrees = np.array([len(nei) for nei in adj], dtype=float)
        if len(degrees) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(degrees)),
            "std": float(np.std(degrees)),
            "min": float(np.min(degrees)),
            "max": float(np.max(degrees))
        }


class PolicyAndMessaging:
    """
    Module controlling policy and messaging dynamics over time.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.state: Dict[str, float] = {
            "policy_level": 0.0,
            "messaging_intensity": float(self.params.get("base_messaging_intensity", 0.2)),
            "enforcement_probability": float(self.params.get("enforcement_probability", 0.1)),
            "broadcast_multiplier": 1.0
        }

    def step(self, day: int) -> Dict[str, float]:
        mandate_day = int(self.params.get("policy_mandate_day", 10))
        base_msg = float(self.params.get("base_messaging_intensity", 0.2))
        base_policy = float(self.params.get("base_policy_level", 0.4))
        multiplier = float(self.params.get("policy_mandate_effect_multiplier", 1.5))
        misinformation = float(self.params.get("misinformation_prevalence", 0.2))
        misinfo_effect = float(self.params.get("misinformation_effect_size", 0.3))
        msg_effect = float(self.params.get("messaging_effect_size", 0.15))
        enforcement_prob = float(self.params.get("enforcement_probability", 0.1))

        if day < mandate_day:
            policy_level = 0.0
            messaging_intensity = base_msg
        else:
            policy_level = min(1.0, base_policy * multiplier)
            messaging_intensity = min(1.0, base_msg * multiplier)

        broadcast_multiplier = (1.0 + msg_effect * messaging_intensity) * (1.0 - misinfo_effect * misinformation)
        broadcast_multiplier = float(clamp(broadcast_multiplier, 0.0, 2.0))

        self.state["policy_level"] = policy_level
        self.state["messaging_intensity"] = messaging_intensity
        self.state["enforcement_probability"] = enforcement_prob
        self.state["broadcast_multiplier"] = broadcast_multiplier
        return dict(self.state)


class InformationDiffusion:
    """
    Module simulating information diffusion via broadcast and peer transmissions.
    """

    def __init__(self, params: Dict[str, Any], n_agents: int, family_adj: List[List[int]], work_adj: List[List[int]], comm_adj: List[List[int]]):
        self.params = params
        self.n = n_agents
        self.family_adj = family_adj
        self.work_adj = work_adj
        self.comm_adj = comm_adj

        self.received_info_state = np.zeros(self.n, dtype=np.int8)
        self.neighbor_frac_family = np.zeros(self.n, dtype=float)
        self.neighbor_frac_work = np.zeros(self.n, dtype=float)
        self.neighbor_frac_comm = np.zeros(self.n, dtype=float)

    def reset_info(self, initial_info_state: np.ndarray) -> None:
        self.received_info_state = initial_info_state.astype(np.int8).copy()

    def step(self, mask_state: np.ndarray, policy_state: Dict[str, float]) -> Dict[str, Any]:
        p_broadcast_base = float(self.params.get("info_broadcast_rate_per_day", 0.02))
        p_broadcast = clamp(p_broadcast_base * float(policy_state.get("broadcast_multiplier", 1.0)), 0.0, 1.0)

        p_peer_family = float(self.params.get("peer_info_transmission_prob_family", 0.25))
        p_peer_work = float(self.params.get("peer_info_transmission_prob_work_school", 0.15))
        p_peer_comm = float(self.params.get("peer_info_transmission_prob_community", 0.07))

        act_family = float(self.params.get("edge_activation_rate_per_day_family", 1.0))
        act_work = float(self.params.get("edge_activation_rate_per_day_work_school", 0.5))
        act_comm = float(self.params.get("edge_activation_rate_per_day_community", 0.2))

        self.neighbor_frac_family = self._compute_neighbor_fraction(mask_state, self.family_adj, act_family)
        self.neighbor_frac_work = self._compute_neighbor_fraction(mask_state, self.work_adj, act_work)
        self.neighbor_frac_comm = self._compute_neighbor_fraction(mask_state, self.comm_adj, act_comm)

        p_peer_total = self._compute_peer_info_probability(self.received_info_state, p_peer_family, p_peer_work, p_peer_comm, act_family, act_work, act_comm)

        p_total = 1.0 - (1.0 - p_broadcast) * (1.0 - p_peer_total)
        draws = np.random.random(self.n)
        new_info = (draws < p_total).astype(np.int8)
        self.received_info_state = np.maximum(self.received_info_state, new_info)

        outputs = {
            "received_info_state": self.received_info_state.copy(),
            "neighbor_frac_family": self.neighbor_frac_family.copy(),
            "neighbor_frac_work": self.neighbor_frac_work.copy(),
            "neighbor_frac_comm": self.neighbor_frac_comm.copy(),
            "info_rate_daily": float(np.mean(self.received_info_state))
        }
        return outputs

    def _compute_neighbor_fraction(self, state: np.ndarray, adj: List[List[int]], activation_rate: float) -> np.ndarray:
        n = self.n
        frac = np.zeros(n, dtype=float)
        for i in range(n):
            neigh = adj[i]
            if not neigh:
                frac[i] = 0.0
                continue
            active = [j for j in neigh if np.random.random() < activation_rate]
            if not active:
                frac[i] = 0.0
                continue
            vals = state[active].astype(float)
            frac[i] = float(np.mean(vals))
        return frac

    def _compute_peer_info_probability(self, info_state: np.ndarray, p_fam: float, p_work: float, p_comm: float,
                                       a_fam: float, a_work: float, a_comm: float) -> np.ndarray:
        n = self.n

        def layer_prob(adj: List[List[int]], p_trans: float, act: float) -> np.ndarray:
            p_layer = np.zeros(n, dtype=float)
            for i in range(n):
                neigh = adj[i]
                if not neigh:
                    p_layer[i] = 0.0
                    continue
                active_informers = [j for j in neigh if (np.random.random() < act and info_state[j] == 1)]
                if not active_informers:
                    p_layer[i] = 0.0
                    continue
                p_fail = (1.0 - p_trans) ** len(active_informers)
                p_layer[i] = 1.0 - p_fail
            return p_layer

        p1 = layer_prob(self.family_adj, p_fam, a_fam)
        p2 = layer_prob(self.work_adj, p_work, a_work)
        p3 = layer_prob(self.comm_adj, p_comm, a_comm)

        p_total = 1.0 - (1.0 - p1) * (1.0 - p2) * (1.0 - p3)
        return p_total


class SocialInfluenceAdoption:
    """
    Module implementing adoption decisions via a logistic-threshold hybrid rule.
    """

    def __init__(self, params: Dict[str, Any], n_agents: int, people: List[Person]):
        self.params = params
        self.n = n_agents
        self.people = people

        self.mask_state = np.zeros(self.n, dtype=np.int8)
        self.time_since_first_info = np.full(self.n, 10**6, dtype=int)

    def reset_mask(self, initial_mask_state: np.ndarray, initial_info_state: np.ndarray) -> None:
        self.mask_state = initial_mask_state.astype(np.int8).copy()
        self.time_since_first_info = np.where(initial_info_state == 1, 0, 10**6)

    def step(self, day: int, received_info_state: np.ndarray,
             neighbor_frac_family: np.ndarray, neighbor_frac_work: np.ndarray, neighbor_frac_comm: np.ndarray,
             policy_state: Dict[str, float]) -> Dict[str, Any]:
        newly_informed = (received_info_state == 1) & (self.time_since_first_info > 5e5)
        self.time_since_first_info = np.where(newly_informed, 0, self.time_since_first_info)
        self.time_since_first_info = np.where(self.time_since_first_info < 5e5, self.time_since_first_info + 1, self.time_since_first_info)

        base_adoption_rate = float(self.params.get("base_adoption_rate", 0.005))
        beta_family = float(self.params.get("beta_family", 1.2))
        beta_work = float(self.params.get("beta_work_school", 0.8))
        beta_comm = float(self.params.get("beta_community", 0.4))
        beta_info = float(self.params.get("beta_info", 2.0))
        beta_risk = float(self.params.get("beta_risk_perception", 0.8))
        policy_weight = float(self.params.get("policy_effect_weight", 0.3))
        enforcement_elasticity = float(self.params.get("enforcement_elasticity", 0.5))
        risk_gamma = float(self.params.get("risk_threshold_gamma", 1.0))
        cost_weight = float(self.params.get("perceived_cost_weight", 0.3))
        noise_sigma = float(self.params.get("social_noise_sigma", 0.15))
        habit_persistence = float(self.params.get("habit_persistence", 0.92))
        abandonment_prob = float(self.params.get("abandonment_probability_per_day", 0.005))
        intercept = float(self.params.get("logit_intercept_adoption", -3.0))
        min_lag = int(self.params.get("info_to_adoption_min_lag_days", 1))

        policy_level = float(policy_state.get("policy_level", 0.0))
        enforcement_prob = float(policy_state.get("enforcement_probability", 0.1))

        risk = np.array([p.risk_perception for p in self.people], dtype=float)
        trust = np.array([p.trust_in_authority for p in self.people], dtype=float)
        cost = np.array([p.perceived_cost_discomfort for p in self.people], dtype=float)
        susc = np.array([p.social_influence_susceptibility for p in self.people], dtype=float)

        wearing = self.mask_state == 1
        if np.any(wearing):
            p_stay = np.clip(habit_persistence - cost_weight * cost, 0.0, 1.0)
            # Combine with baseline abandonment hazard
            p_abandon = 1.0 - (p_stay * (1.0 - abandonment_prob))
            abandon_draw = np.random.random(self.n)
            to_abandon = wearing & (abandon_draw < p_abandon)
            self.mask_state[to_abandon] = 0

        not_wearing = self.mask_state == 0
        if np.any(not_wearing):
            info_term = beta_info * received_info_state.astype(float)
            peer_term = susc * (beta_family * neighbor_frac_family + beta_work * neighbor_frac_work + beta_comm * neighbor_frac_comm)
            risk_term = beta_risk * risk
            policy_term = policy_weight * policy_level * trust
            enforce_term = enforcement_elasticity * enforcement_prob
            threshold = risk_gamma * (1.0 - risk)
            linear_score = intercept + info_term + peer_term + risk_term + policy_term + enforce_term - threshold - cost_weight * cost
            noise = np.random.normal(0.0, noise_sigma, size=self.n)
            p_adopt_full = sigmoid(linear_score + noise)

            # Gate adoption: without info, only base rate allowed; with info, require lag
            p_draw = np.where(received_info_state == 1, p_adopt_full, base_adoption_rate)
            p_draw = np.maximum(p_draw, base_adoption_rate)
            allowed = (self.time_since_first_info >= min_lag) | (received_info_state == 0)
            adopt_draw = np.random.random(self.n)
            to_adopt = not_wearing & allowed & (adopt_draw < p_draw)
            self.mask_state[to_adopt] = 1

        outputs = {
            "mask_adoption_state": self.mask_state.copy(),
            "adoption_rate_daily": float(np.mean(self.mask_state))
        }
        return outputs


class AdoptionAggregator:
    """
    Aggregates daily and subgroup adoption and info series; computes observables.

    Methods
    -------
    - step(mask_state, info_state) -> dict with:
        adoption_rate_daily, info_rate_daily,
        adoption_rate_by_age (dict), adoption_rate_by_occupation (dict)
    """

    def __init__(self, people: List[Person]):
        self.people = people
        self.n = len(people)
        # Precompute group indices
        self.age_groups = sorted(list({p.age_group for p in self.people}))
        self.occ_groups = sorted(list({p.occupation for p in self.people}))
        self.idx_by_age: Dict[str, np.ndarray] = {
            g: np.array([i for i, p in enumerate(self.people) if p.age_group == g], dtype=int) for g in self.age_groups
        }
        self.idx_by_occ: Dict[str, np.ndarray] = {
            g: np.array([i for i, p in enumerate(self.people) if p.occupation == g], dtype=int) for g in self.occ_groups
        }

    def step(self, mask_state: np.ndarray, info_state: np.ndarray) -> Dict[str, Any]:
        adoption_rate_daily = float(np.mean(mask_state)) if len(mask_state) else 0.0
        info_rate_daily = float(np.mean(info_state)) if len(info_state) else 0.0

        adoption_rate_by_age: Dict[str, float] = {}
        for g, idx in self.idx_by_age.items():
            adoption_rate_by_age[g] = float(np.mean(mask_state[idx])) if len(idx) > 0 else 0.0

        adoption_rate_by_occ: Dict[str, float] = {}
        for g, idx in self.idx_by_occ.items():
            adoption_rate_by_occ[g] = float(np.mean(mask_state[idx])) if len(idx) > 0 else 0.0

        return {
            "adoption_rate_daily": adoption_rate_daily,
            "info_rate_daily": info_rate_daily,
            "adoption_rate_by_age": adoption_rate_by_age,
            "adoption_rate_by_occupation": adoption_rate_by_occ
        }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def peak(series: List[float]) -> Tuple[float, int]:
    if not series:
        return 0.0, -1
    arr = np.asarray(series, dtype=float)
    idx = int(np.argmax(arr))
    return float(arr[idx]), idx


def time_to_thresholds(series: List[float], thresholds: List[float]) -> Dict[float, Optional[int]]:
    res: Dict[float, Optional[int]] = {}
    arr = np.asarray(series, dtype=float)
    for th in thresholds:
        idxs = np.where(arr >= th)[0]
        res[float(th)] = int(idxs[0]) if idxs.size > 0 else None
    return res


def variance_across_groups(group_values: Dict[str, float]) -> float:
    if not group_values:
        return 0.0
    vals = np.array(list(group_values.values()), dtype=float)
    return float(np.var(vals))



# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def main():
    # Parse model plan parameters
    try:
        plan = json.loads(MODEL_PLAN_JSON)
    except Exception:
        plan = {}
    params_list = plan.get("parameters", [])
    params_by_key = {p.get("key"): p for p in params_list}

    # Load external parameters from JSON file
    try:
        params_file = os.path.join(PROJECT_ROOT, "output", "mask_adoption_calibrasim_debug_run3", "parameters.json")
        if os.path.exists(params_file):
            with open(params_file, "r", encoding="utf-8") as f:
                external_params = json.load(f)
        else:
            external_params = {}
    except Exception:
        external_params = {}

    def get_param(key: str, default: Any) -> Any:
        # First try external parameters, then fall back to model plan defaults
        if key in external_params:
            return external_params[key]
        return params_by_key.get(key, {}).get("default", default)
        return params_by_key.get(key, {}).get("default", default)

    time_horizon_days = int(get_param("time_horizon_days", 40))
    timestep_days = int(get_param("timestep_days", 1))
    seed = int(get_param("seed", 42))

    # Set seeds for reproducibility
    set_seed(seed)

    # Load data
    loader = DataLoader(DATA_DIR)
    loader.load_all()
    agent_df = loader.agent_df
    train_df = loader.train_df
    network_data = loader.network_data

    if agent_df is None or train_df is None or network_data is None:
        raise RuntimeError("Failed to load required data; aborting.")

    n_agents = agent_df.shape[0]

    # Build Person entities
    people: List[Person] = []
    for _, row in agent_df.iterrows():
        person = Person(
            pid=int(row["agent_id"]),
            age=int(row["age"]),
            age_group=str(row.get("age_group", "Unknown")),
            occupation=str(row.get("occupation", "Unknown")),
            risk_perception=float(row.get("risk_perception", 0.5))
        )
        people.append(person)

    # Map agent_id -> index in arrays
    id_to_index: Dict[int, int] = {p.id: idx for idx, p in enumerate(people)}

    # Initialize NetworkLayerEngine
    net_params = {
        "symmetrize_edges": bool(get_param("symmetrize_edges", True)),
        "degree_cap_percentile": float(get_param("degree_cap_percentile", 0.99))
    }
    net_engine = NetworkLayerEngine(net_params, network_data, n_agents)
    net_engine.on_init()

    # Attach household/workplace IDs to Person
    for i, p in enumerate(people):
        p.household_id = int(net_engine.household_id[i])
        p.workplace_id = int(net_engine.workplace_id[i])

    # Initial states from day 0
    max_day_train = int(train_df["day"].max()) if not train_df.empty else -1
    day0_df = train_df[train_df["day"] == 0] if max_day_train >= 0 else pd.DataFrame(columns=["agent_id", "wearing_mask", "received_info"])
    initial_mask = np.zeros(n_agents, dtype=np.int8)
    initial_info = np.zeros(n_agents, dtype=np.int8)
    if not day0_df.empty:
        for _, r in day0_df.iterrows():
            a_id = int(r["agent_id"])
            idx = id_to_index.get(a_id, None)
            if idx is not None:
                initial_mask[idx] = int(r["wearing_mask"])
                initial_info[idx] = int(r["received_info"])

    # Initialize modules
    policy_params = {
        "policy_mandate_day": int(get_param("policy_mandate_day", 10)),
        "policy_mandate_effect_multiplier": float(get_param("policy_mandate_effect_multiplier", 1.5)),
        "enforcement_probability": float(get_param("enforcement_probability", 0.1)),
        "messaging_effect_size": float(get_param("messaging_effect_size", 0.15)),
        "misinformation_prevalence": float(get_param("misinformation_prevalence", 0.2)),
        "misinformation_effect_size": float(get_param("misinformation_effect_size", 0.3)),
        "base_messaging_intensity": float(get_param("base_messaging_intensity", 0.2)),
        "base_policy_level": float(get_param("base_policy_level", 0.4))
    }
    policy_module = PolicyAndMessaging(policy_params)

    info_params = {
        "info_broadcast_rate_per_day": float(get_param("info_broadcast_rate_per_day", 0.02)),
        "peer_info_transmission_prob_family": float(get_param("peer_info_transmission_prob_family", 0.25)),
        "peer_info_transmission_prob_work_school": float(get_param("peer_info_transmission_prob_work_school", 0.15)),
        "peer_info_transmission_prob_community": float(get_param("peer_info_transmission_prob_community", 0.07)),
        "edge_activation_rate_per_day_family": float(get_param("edge_activation_rate_per_day_family", 1.0)),
        "edge_activation_rate_per_day_work_school": float(get_param("edge_activation_rate_per_day_work_school", 0.5)),
        "edge_activation_rate_per_day_community": float(get_param("edge_activation_rate_per_day_community", 0.2))
    }
    info_module = InformationDiffusion(info_params, n_agents, net_engine.family_adj, net_engine.work_adj, net_engine.comm_adj)
    info_module.reset_info(initial_info)

    adoption_params = {
        "base_adoption_rate": float(get_param("base_adoption_rate", 0.005)),
        "beta_family": float(get_param("beta_family", 1.2)),
        "beta_work_school": float(get_param("beta_work_school", 0.8)),
        "beta_community": float(get_param("beta_community", 0.4)),
        "beta_info": float(get_param("beta_info", 2.0)),
        "beta_risk_perception": float(get_param("beta_risk_perception", 0.8)),
        "policy_effect_weight": float(get_param("policy_effect_weight", 0.3)),
        "enforcement_elasticity": float(get_param("enforcement_elasticity", 0.5)),
        "risk_threshold_gamma": float(get_param("risk_threshold_gamma", 1.0)),
        "perceived_cost_weight": float(get_param("perceived_cost_weight", 0.3)),
        "social_noise_sigma": float(get_param("social_noise_sigma", 0.15)),
        "habit_persistence": float(get_param("habit_persistence", 0.92)),
        "abandonment_probability_per_day": float(get_param("abandonment_probability_per_day", 0.005)),
        "logit_intercept_adoption": float(get_param("logit_intercept_adoption", -3.0)),
        "info_to_adoption_min_lag_days": int(get_param("info_to_adoption_min_lag_days", 1))
    }
    adoption_module = SocialInfluenceAdoption(adoption_params, n_agents, people)
    adoption_module.reset_mask(initial_mask, initial_info)

    aggregator = AdoptionAggregator(people)

    # Run simulation loop
    days = list(range(0, time_horizon_days, timestep_days))
    results_daily = {
        "day": [],
        "adoption_rate": [],
        "info_rate": [],
        "adoption_rate_by_age": [],
        "adoption_rate_by_occupation": []
    }

    for day in days:
        # Current states
        current_mask = adoption_module.mask_state.copy()
        policy_state = policy_module.step(day)
        info_out = info_module.step(current_mask, policy_state)
        adoption_out = adoption_module.step(
            day=day,
            received_info_state=info_out["received_info_state"],
            neighbor_frac_family=info_out["neighbor_frac_family"],
            neighbor_frac_work=info_out["neighbor_frac_work"],
            neighbor_frac_comm=info_out["neighbor_frac_comm"],
            policy_state=policy_state
        )
        agg_out = aggregator.step(adoption_out["mask_adoption_state"], info_out["received_info_state"])

        results_daily["day"].append(day)
        results_daily["adoption_rate"].append(float(agg_out["adoption_rate_daily"]))
        results_daily["info_rate"].append(float(agg_out["info_rate_daily"]))
        results_daily["adoption_rate_by_age"].append(dict(agg_out["adoption_rate_by_age"]))
        results_daily["adoption_rate_by_occupation"].append(dict(agg_out["adoption_rate_by_occupation"]))

    # Compute metrics vs observed for training period (days 0-29 if available)
    obs_adoption_by_day = None
    if not train_df.empty:
        obs_agg = train_df.groupby("day")["wearing_mask"].mean().reset_index()
        obs_adoption_by_day = {int(r["day"]): float(r["wearing_mask"]) for _, r in obs_agg.iterrows()}
    sim_adoption_series = np.array(results_daily["adoption_rate"], dtype=float)
    sim_days = np.array(results_daily["day"], dtype=int)

    # Align series over intersection of days 0..29
    eval_days = [d for d in sim_days if d in range(0, 30)]
    if obs_adoption_by_day is not None and eval_days:
        y_true = np.array([obs_adoption_by_day.get(d, np.nan) for d in eval_days], dtype=float)
        y_pred = np.array([sim_adoption_series[list(sim_days).index(d)] for d in eval_days], dtype=float)
        valid = ~np.isnan(y_true)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        metrics = {
            "RMSE": rmse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "R_squared": r2(y_true, y_pred),
            "PeakAdoption": peak(list(sim_adoption_series)),
            "TimeToThresholds": time_to_thresholds(list(sim_adoption_series), [0.4, 0.6, 0.8])  # based on overall series
        }
    else:
        metrics = {
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "R_squared": float("nan"),
            "PeakAdoption": peak(list(sim_adoption_series)),
            "TimeToThresholds": time_to_thresholds(list(sim_adoption_series), [0.4, 0.6, 0.8])
        }

    # Adoption inequality by age on last simulated day
    last_age_group_values = results_daily["adoption_rate_by_age"][-1] if results_daily["adoption_rate_by_age"] else {}
    metrics["AdoptionInequalityByAge"] = variance_across_groups(last_age_group_values)

    # Extract predictions for days 30-39
    pred_start = int(plan.get("prediction_period", {}).get("start_day", 30))
    pred_end = int(plan.get("prediction_period", {}).get("end_day", 39))
    predictions = []
    for d, val in zip(results_daily["day"], results_daily["adoption_rate"]):
        if pred_start <= d <= pred_end:
            predictions.append({"day": int(d), "predicted_adoption_rate": float(val)})

    # Print summary outputs
    print("Simulation completed.")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("Predictions (days {}-{}):".format(pred_start, pred_end))
    print(json.dumps(predictions, indent=2))

    # Optional: save outputs
    try:
        out_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({
            "day": results_daily["day"],
            "adoption_rate": results_daily["adoption_rate"],
            "info_rate": results_daily["info_rate"]
        }).to_csv(os.path.join(out_dir, "daily_series.csv"), index=False)
        with open(os.path.join(out_dir, "adoption_rate_by_age.json"), "w") as f:
            json.dump({int(d): v for d, v in zip(results_daily["day"], results_daily["adoption_rate_by_age"])}, f)
        with open(os.path.join(out_dir, "adoption_rate_by_occupation.json"), "w") as f:
            json.dump({int(d): v for d, v in zip(results_daily["day"], results_daily["adoption_rate_by_occupation"])}, f)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(out_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f, indent=2)
    except Exception as e:
        print("Warning: Failed to save outputs:", e)



# Execute main for both direct execution and sandbox wrapper invocation
main()