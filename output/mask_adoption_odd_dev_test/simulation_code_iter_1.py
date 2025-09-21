def main():
    pass

import os
import json
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Optional third-party imports with safe fallback
try:
    import pandas as pd
except Exception as e:
    pd = None

try:
    import networkx as nx
except Exception as e:
    nx = None

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

# statsmodels is optional for calibration
try:
    import statsmodels.api as sm
except Exception as e:
    sm = None

# sklearn optional for metrics
try:
    from sklearn.metrics import mean_squared_error
except Exception as e:
    mean_squared_error = None

# ---------------------------------------------------------------------
# Global utilities and constants
# ---------------------------------------------------------------------

def logistic(x: float) -> float:
    """
    Compute a numerically stable logistic function value for the given input.

    Returns:
        float: The logistic (sigmoid) function result in [0, 1].

    Notes:
        The function uses bounds to avoid overflow for extreme values.
    """
    pass
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)
    except Exception:
        return 0.5


def clip01(x: float) -> float:
    """
    Clip a floating-point number to the [0, 1] interval.

    Args:
        x (float): Input value.

    Returns:
        float: Clipped value between 0 and 1.
    """
    pass
    return max(0.0, min(1.0, float(x)))


def poisson(lam: float, rng: np.random.Generator) -> int:
    """
    Draw a sample from a Poisson distribution with parameter lam.

    Args:
        lam (float): The rate parameter lam (lambda).
        rng (np.random.Generator): Numpy random number generator.

    Returns:
        int: A non-negative integer sampled from Poisson(lam).

    Notes:
        Falls back to max(int(lam + normal noise), 0) if numpy is unavailable.
    """
    pass
    try:
        return int(rng.poisson(lam=lam))
    except Exception:
        return int(max(0, lam + rng.normal(0, math.sqrt(max(lam, 1e-9)))))


# Environment variables for path handling
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# ---------------------------------------------------------------------
# Model plan configuration (embedded)
# ---------------------------------------------------------------------

MODEL_PLAN = {
    "model_type": "agent_based",
    "description": "Multiplex network-based agent simulation of mask-wearing behavior with information diffusion, social influence, and policy effects. Calibrated on days 0-29, predicts days 30-39.",
    "environment": {
        "type": "network",
        "time_step": 1,
        "time_unit": "days",
        "network_layers": ["family", "work_school", "community"],
        "edge_directionality": "symmetrized undirected"
    },
    "parameters": {
        "population_size": 1000,
        "time_horizon_days": 40,
        "random_seed": 42,
        "initial_mask_adoption_rate": 0.2,
        "initial_risk_perception_mean": 0.3,
        "initial_risk_perception_std": 0.1,
        "layer_influence_weights": {"family": 1.0, "work_school": 0.6, "community": 0.3},
        "activation_schedule": {"family": "7d/week", "work_school": "5d/week", "community": "3d/week"},
        "risk_update_weight_from_cases": 0.5,
        "message_effect_strength": 0.3,
        "misinformation_effect_strength": -0.25,
        "trust_in_authorities_mean": 0.5,
        "trust_in_authorities_std": 0.15,
        "propensity_to_comply_mean": 0.5,
        "propensity_to_comply_std": 0.2,
        "mask_efficacy_mean": 0.5,
        "mask_efficacy_std": 0.1,
        "mask_quality_levels": ["cloth", "surgical", "respirator"],
        "mask_price": 1.0,
        "subsidy_per_person": 0.0,
        "retailer_initial_stock": 2000,
        "retailer_restock_rate_per_day": 200,
        "supply_reliability": 0.9,
        "policy_start_day": 10,
        "policy_type": "recommendation",
        "enforcement_probability": 0.3,
        "fine_amount": 50.0,
        "messaging_intensity": 0.5,
        "authority_credibility": 0.6,
        "daily_cases_time_series": [],
        "adoption_target_threshold": 0.7,
        "group_multipliers_age": {"Youth": 0.8, "Young Adult": 1.0, "Middle Age": 1.1, "Senior": 1.2},
        "group_multipliers_occupation": {"Student": 0.9, "Blue Collar": 0.85, "White Collar": 1.05, "Healthcare": 1.3},
        "calibrated_coefficients": {
            "alpha0": None,
            "beta_risk": None,
            "beta_info": None,
            "beta_peer_family": None,
            "beta_peer_work": None,
            "beta_peer_community": None,
            "rho_retention": None,
            "theta0": None,
            "theta_family": None,
            "theta_work": None,
            "theta_community": None,
            "theta_msg": None,
            "delta_info": None,
            "epsilon_info": None,
            "beta_policy": None
        },
        # Additional placeholders (not hardcoded usage)
        "mask_requirement_by_layer": {"family": 0.0, "work_school": 1.0, "community": 0.5},  # scenario-level
        "enforcement_strength": 0.8,
        "noncompliance_noise": 0.1,
        "mask_inventory_capacity_per_capita": 2.0,
        "mask_usage_rate_per_wearer_per_day": 0.1
    },
    "prediction_period": {"start_day": 30, "end_day": 39},
    "evaluation_metrics": [
        "RMSE",
        "overall_mask_adoption_rate",
        "time_to_target_adoption",
        "peak_adoption_rate",
        "sustained_adoption_rate",
        "adoption_inequality_index",
        "noncompliance_rate",
        "mask_availability",
        "estimated_transmission_reduction"
    ],
    # Additional fields that may be included but not explicitly used; retained for completeness
    "entities": ["Person", "Location", "PublicHealthAuthority", "Retailer", "InformationChannel"],
    "behaviors": ["decide_mask_use", "share_information", "purchase_masks", "comply_with_policy", "update_risk_perception", "host_interactions"],
    "interactions": ["social_influence", "policy_compliance", "media_exposure", "retail_purchase", "location_interaction", "risk_feedback"]
}

# ---------------------------------------------------------------------
# Config Manager
# ---------------------------------------------------------------------

class ConfigManager:
    """
    Manage configuration derived from the model plan and provide convenient accessors.

    This class stores the model plan, exposes parameter values, and supports
    dynamic retrieval of evaluation metrics and prediction period.

    Attributes:
        model_plan (dict): The loaded or embedded model plan dictionary.
        params (dict): Shorthand reference to model_plan['parameters'].
        env (dict): Shorthand reference to model_plan['environment'].

    Methods:
        get_param(key, default=None): Retrieve a parameter safely.
        get_prediction_period(): Return tuple (start_day, end_day).
        get_evaluation_metrics(): Return the list of metric names to compute.
    """
    def __init__(self, model_plan: Dict[str, Any]):
        """
        Initialize the ConfigManager with the given model plan.

        Args:
            model_plan (dict): The configuration dictionary specifying model details.
        """
        pass
        self.model_plan = model_plan or {}
        self.params = self.model_plan.get("parameters", {})
        self.env = self.model_plan.get("environment", {})

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a parameter value from the model plan parameters.

        Args:
            key (str): Parameter key.
            default (Any): Default value if parameter is missing.

        Returns:
            Any: The parameter value or the default if not found.
        """
        pass
        return self.params.get(key, default)

    def get_prediction_period(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the prediction period window from the model plan.

        Returns:
            tuple: (start_day, end_day) if specified, else (None, None).
        """
        pass
        pp = self.model_plan.get("prediction_period", {})
        return (pp.get("start_day"), pp.get("end_day"))

    def get_evaluation_metrics(self) -> List[str]:
        """
        Retrieve the list of evaluation metric names to compute at evaluation time.

        Returns:
            list[str]: List of metric names.
        """
        pass
        return list(self.model_plan.get("evaluation_metrics", []))


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------

class DataLoader:
    """
    Load and validate data files required for the simulation: agent attributes,
    social network multiplex, and training panel.

    Attributes:
        data_dir (str): Base folder containing data files.
        rng (np.random.Generator): Random number generator for synthetic generation.

    Methods:
        load_agent_attributes(): Load agent attributes dataframe or synthesize if missing.
        load_social_network(): Load multiplex adjacency from JSON; symmetrize and deduplicate.
        load_train_data(): Load training panel dataframe (days 0-29) if available.
        synthesize_data_if_missing(population_size): Create plausible synthetic agents and networks.
        build_neighbors_by_layer(network_json, agent_ids): Return neighbors dict by layer per agent.
    """
    def __init__(self, data_dir: str, rng: np.random.Generator):
        """
        Initialize DataLoader with a data directory and RNG.

        Args:
            data_dir (str): Root directory for data files.
            rng (np.random.Generator): RNG for synthetic data generation.
        """
        pass
        self.data_dir = data_dir
        self.rng = rng

        # Files
        self.agent_file = os.path.join(self.data_dir, "agent_attributes.csv")
        self.network_file = os.path.join(self.data_dir, "social_network.json")
        self.train_file = os.path.join(self.data_dir, "train_data.csv")

    def load_agent_attributes(self, population_size: int) -> pd.DataFrame:
        """
        Load agent attributes CSV or synthesize if unavailable.

        Args:
            population_size (int): Desired population size for synthetic data.

        Returns:
            pandas.DataFrame: Agent attributes with columns: agent_id, age_group, occupation, risk_perception, income_level.
        """
        pass
        if pd is None:
            raise RuntimeError("pandas is required to load or synthesize agent attributes.")

        if os.path.exists(self.agent_file):
            try:
                df = pd.read_csv(self.agent_file)
                required = {"agent_id", "age_group", "occupation", "risk_perception"}
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"Agent attributes file missing required columns: {missing}")
                df["risk_perception"] = df["risk_perception"].clip(0, 1)
                if "income_level" not in df.columns:
                    df["income_level"] = (self.rng.normal(loc=50, scale=20, size=len(df))).clip(5, 100)
                return df
            except Exception as e:
                warnings.warn(f"Failed to load agent_attributes.csv; generating synthetic data. Error: {e}")

        return self._synthesize_agents(population_size)

    def _synthesize_agents(self, population_size: int) -> pd.DataFrame:
        """
        Generate synthetic agent attributes when real data is unavailable.

        Args:
            population_size (int): Number of agents to synthesize.

        Returns:
            pandas.DataFrame: Synthetic agent attributes.
        """
        pass
        if pd is None:
            raise RuntimeError("pandas is required for synthetic agent generation.")

        age_groups = ["Youth", "Young Adult", "Middle Age", "Senior"]
        occupations = ["Student", "Blue Collar", "White Collar", "Healthcare"]
        df = pd.DataFrame({
            "agent_id": np.arange(population_size, dtype=int),
            "age_group": self.rng.choice(age_groups, size=population_size, p=[0.25, 0.35, 0.3, 0.1]),
            "occupation": self.rng.choice(occupations, size=population_size, p=[0.25, 0.35, 0.35, 0.05]),
        })
        # Risk perception with group trends
        base_risk = self.rng.beta(2, 5, size=population_size)
        age_adjust = df["age_group"].map({"Youth": -0.05, "Young Adult": 0.0, "Middle Age": 0.05, "Senior": 0.1}).values
        occ_adjust = df["occupation"].map({"Student": -0.03, "Blue Collar": -0.05, "White Collar": 0.03, "Healthcare": 0.15}).values
        df["risk_perception"] = np.clip(base_risk + age_adjust + occ_adjust, 0, 1)
        df["income_level"] = (self.rng.normal(loc=50, scale=20, size=population_size)).clip(5, 100)
        return df

    def load_social_network(self, agent_ids: List[int], average_degree: Dict[str, int] = None) -> Dict[int, Dict[str, List[int]]]:
        """
        Load social network JSON or build synthetic multiplex if missing.

        Args:
            agent_ids (list[int]): List of agent IDs to include in the network.
            average_degree (dict): Optional mapping of layer to average degree for synthetic graph.

        Returns:
            dict: Mapping agent_id -> {layer: [neighbor_ids]} across layers: family, work_school, community.
        """
        pass
        layers = ["family", "work_school", "community"]
        if os.path.exists(self.network_file):
            try:
                with open(self.network_file, "r") as f:
                    raw = json.load(f)
                # Convert string keys to int; ensure presence of layers
                network = {}
                for k, v in raw.items():
                    try:
                        i = int(k)
                    except Exception:
                        continue
                    network[i] = {L: list(map(int, v.get(L, []))) for L in layers}
                # Symmetrize and deduplicate, remove self-loops
                network = self._symmetrize_network(network, agent_ids, layers=layers)
                return network
            except Exception as e:
                warnings.warn(f"Failed to load social_network.json; generating synthetic network. Error: {e}")

        # Synthesize multiplex
        return self._synthesize_network(agent_ids, average_degree=average_degree or {"family": 3, "work_school": 6, "community": 4})

    def _symmetrize_network(self, network: Dict[int, Dict[str, List[int]]], agent_ids: List[int], layers: List[str]) -> Dict[int, Dict[str, List[int]]]:
        """
        Symmetrize and clean a multiplex network adjacency list.

        Args:
            network (dict): Input adjacency list mapping agent -> layer -> neighbors.
            agent_ids (list[int]): List of valid agent IDs.
            layers (list[str]): Layers to process.

        Returns:
            dict: Cleaned and symmetrized network adjacency.
        """
        pass
        valid = set(agent_ids)
        # Ensure all agent ids present
        for i in agent_ids:
            network.setdefault(i, {L: [] for L in layers})

        # Symmetrize
        for L in layers:
            # Build undirected sets
            for i in agent_ids:
                ni = network.get(i, {}).get(L, [])
                # Remove self-loops and non-valid ids
                ni = [j for j in ni if j in valid and j != i]
                network[i][L] = ni

            for i in agent_ids:
                for j in list(network[i][L]):
                    if i not in network.get(j, {}).get(L, []):
                        network[j][L].append(i)

            # Deduplicate all lists
            for i in agent_ids:
                vals = list(dict.fromkeys(network[i][L]))  # preserve order
                network[i][L] = vals

        return network

    def _synthesize_network(self, agent_ids: List[int], average_degree: Dict[str, int]) -> Dict[int, Dict[str, List[int]]]:
        """
        Synthesize a multiplex network using simple random graph models per layer.

        Args:
            agent_ids (list[int]): Agent identifiers.
            average_degree (dict): Average degree per layer.

        Returns:
            dict: Multiplex network mapping id -> layer -> neighbor list.
        """
        pass
        layers = ["family", "work_school", "community"]
        N = len(agent_ids)
        idx_map = {i: idx for idx, i in enumerate(agent_ids)}

        # If networkx is available, use Watts-Strogatz approximations; else simple random connections
        multiplex = {i: {L: [] for L in layers} for i in agent_ids}

        for L in layers:
            k = max(1, int(average_degree.get(L, 4)))
            if nx is not None and N >= k + 1:
                try:
                    # Rewiring prob different per layer to vary clustering
                    p_rewire = {"family": 0.05, "work_school": 0.1, "community": 0.2}.get(L, 0.1)
                    G = nx.watts_strogatz_graph(n=N, k=min(k + (k % 2 == 0), N - 1), p=p_rewire, seed=int(self.rng.integers(0, 1e9)))
                    for u, v in G.edges():
                        iu = agent_ids[u]
                        iv = agent_ids[v]
                        multiplex[iu][L].append(iv)
                        multiplex[iv][L].append(iu)
                except Exception:
                    # Fallback to random pairs
                    for _ in range(N * k // 2):
                        a = int(self.rng.integers(0, N))
                        b = int(self.rng.integers(0, N))
                        if a != b:
                            ia = agent_ids[a]
                            ib = agent_ids[b]
                            if ib not in multiplex[ia][L]:
                                multiplex[ia][L].append(ib)
                            if ia not in multiplex[ib][L]:
                                multiplex[ib][L].append(ia)
            else:
                # Simple random edges
                for _ in range(N * k // 2):
                    a = int(self.rng.integers(0, N))
                    b = int(self.rng.integers(0, N))
                    if a != b:
                        ia = agent_ids[a]
                        ib = agent_ids[b]
                        if ib not in multiplex[ia][L]:
                            multiplex[ia][L].append(ib)
                        if ia not in multiplex[ib][L]:
                            multiplex[ib][L].append(ia)

        # Deduplicate
        for i in agent_ids:
            for L in layers:
                multiplex[i][L] = list(dict.fromkeys([j for j in multiplex[i][L] if j != i]))

        return multiplex

    def load_train_data(self) -> Optional[pd.DataFrame]:
        """
        Load the training panel data (days 0-29) if available.

        Returns:
            pandas.DataFrame or None: The panel data or None if not present.
        """
        pass
        if pd is None:
            return None
        if os.path.exists(self.train_file):
            try:
                df = pd.read_csv(self.train_file)
                # Basic validation
                required = {"day", "agent_id", "wearing_mask", "received_info"}
                missing = required - set(df.columns)
                if missing:
                    warnings.warn(f"train_data.csv missing columns: {missing}; ignoring file.")
                    return None
                return df
            except Exception as e:
                warnings.warn(f"Failed to load train_data.csv: {e}")
                return None
        return None


# ---------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------

@dataclass
class Person:
    """
    Representation of an individual agent with attributes and state.

    Attributes:
        id (int): Unique identifier.
        age_group (str): Age group category.
        occupation (str): Occupation category.
        socio_group (Optional[str]): Placeholder for socio group; may be None.
        income_level (float): Income proxy.
        risk_perception (float): Perceived risk in [0,1].
        trust_in_authorities (float): Trust in authority in [0,1].
        propensity_to_comply (float): Disposition to comply with policy [0,1].
        mask_status (bool): Whether the agent wears a mask at current day.
        mask_quality (str): Mask quality type.
        health_state (str): Placeholder health state.
        inventory_masks (int): Available mask units.
        neighbors_by_layer (Dict[str, List[int]]): Multiplex adjacency by layer.
        received_info (bool): Whether received pro-mask info recently.
        adoption_probability (float): Last computed adoption probability.
        group_multipliers (Dict[str, float]): Multipliers by group categories.

    Methods:
        decide_mask_use(...): Update mask wearing based on inputs.
        update_risk_perception(...): Update risk perception.
        share_information(...): Update info state via diffusion and messaging.
        purchase_masks(...): Attempt to acquire masks subject to supply.
        comply_with_policy(...): Adjust compliance under enforcement context.
    """
    id: int
    age_group: str
    occupation: str
    socio_group: Optional[str]
    income_level: float
    risk_perception: float
    trust_in_authorities: float
    propensity_to_comply: float
    mask_status: bool
    mask_quality: str
    health_state: str
    inventory_masks: int
    neighbors_by_layer: Dict[str, List[int]]
    received_info: bool = False
    adoption_probability: float = 0.0
    group_multipliers: Dict[str, float] = field(default_factory=dict)

    def decide_mask_use(
        self,
        coeffs: Dict[str, float],
        neighbor_mask_shares: Dict[str, float],
        policy_active: bool,
        rng: np.random.Generator
    ) -> bool:
        """
        Decide whether to wear a mask today based on retention, risk, info, peers, and policy.

        Args:
            coeffs (dict): Coefficients for logistic decision, including alpha0, beta_risk, beta_info, betas by layer, beta_policy, rho_retention.
            neighbor_mask_shares (dict): Layer -> neighbor mask share from t-1.
            policy_active (bool): Whether policy is active today.
            rng (np.random.Generator): Random generator.

        Returns:
            bool: The decided mask status for the day.

        Notes:
            If inventory is zero but adoption is intended, may require purchase to complete.
        """
        pass
        # Retention if previously masked
        rho = float(coeffs.get("rho_retention", 0.95))
        beta_policy = float(coeffs.get("beta_policy", 0.0))
        if self.mask_status:
            # Persistence with probability rho; optionally modulate by policy and info
            stay = rng.random() < rho
            if stay:
                self.adoption_probability = 1.0
                return True

        # Logistic adoption
        x = float(coeffs.get("alpha0", -4.0))
        x += float(coeffs.get("beta_risk", 2.0)) * float(self.risk_perception)
        x += float(coeffs.get("beta_info", 0.5)) * (1.0 if self.received_info else 0.0)
        x += float(coeffs.get("beta_peer_family", 1.5)) * float(neighbor_mask_shares.get("family", 0.0))
        x += float(coeffs.get("beta_peer_work", 0.9)) * float(neighbor_mask_shares.get("work_school", 0.0))
        x += float(coeffs.get("beta_peer_community", 0.4)) * float(neighbor_mask_shares.get("community", 0.0))

        # Policy term
        policy_term = 1.0 if policy_active else 0.0
        x += float(beta_policy) * policy_term * float(self.trust_in_authorities) * float(self.propensity_to_comply)

        # Group multipliers as log-odds offsets (use log of multipliers)
        m_age = self.group_multipliers.get("age", 1.0)
        m_occ = self.group_multipliers.get("occupation", 1.0)
        try:
            x += math.log(max(m_age, 1e-6))
            x += math.log(max(m_occ, 1e-6))
        except Exception:
            x += 0.0

        p = logistic(x)
        self.adoption_probability = p

        # Must have masks in inventory to wear
        if self.inventory_masks <= 0:
            # Will require purchase elsewhere; return based on current inventory status
            return False

        return rng.random() < p

    def update_risk_perception(
        self,
        normalized_cases_today: float,
        message_effect_strength: float,
        misinformation_effect_strength: float,
        risk_update_weight_from_cases: float,
        misinformation_exposure: float = 0.0
    ) -> float:
        """
        Update risk perception combining past risk, case incidence, and information effects.

        Args:
            normalized_cases_today (float): Case incidence normalized to [0,1].
            message_effect_strength (float): Effect of info on risk perception.
            misinformation_effect_strength (float): Negative effect from misinformation exposure.
            risk_update_weight_from_cases (float): Weight for cases in risk update.
            misinformation_exposure (float): Exposure level to misinformation [0,1].

        Returns:
            float: Updated risk perception (clipped to [0,1]).
        """
        pass
        prev = self.risk_perception
        info_term = message_effect_strength * (1.0 if self.received_info else 0.0)
        misinfo_term = misinformation_effect_strength * float(misinformation_exposure)
        w = float(risk_update_weight_from_cases)
        new_val = (1.0 - w) * prev + w * float(normalized_cases_today) + info_term + misinfo_term
        self.risk_perception = clip01(new_val)
        return self.risk_perception

    def share_information(
        self,
        coeffs_info: Dict[str, float],
        neighbor_info_shares: Dict[str, float],
        messaging_intensity: float,
        credibility: float,
        rng: np.random.Generator
    ) -> bool:
        """
        Update the received_info state via logistic information diffusion and decay.

        Args:
            coeffs_info (dict): Coefficients for info diffusion (theta0, theta_* by layer, theta_msg, delta_info, epsilon_info).
            neighbor_info_shares (dict): Layer -> neighbor info share from t-1.
            messaging_intensity (float): Authority messaging intensity.
            credibility (float): Authority credibility weight.
            rng (np.random.Generator): Random number generator.

        Returns:
            bool: New received_info state after diffusion and decay.
        """
        pass
        theta0 = float(coeffs_info.get("theta0", -3.0))
        x = theta0
        x += float(coeffs_info.get("theta_family", 1.0)) * float(neighbor_info_shares.get("family", 0.0))
        x += float(coeffs_info.get("theta_work", 0.6)) * float(neighbor_info_shares.get("work_school", 0.0))
        x += float(coeffs_info.get("theta_community", 0.3)) * float(neighbor_info_shares.get("community", 0.0))
        x += float(coeffs_info.get("theta_msg", 0.5)) * (float(messaging_intensity) * float(credibility))
        p_info = logistic(x)
        epsilon = float(coeffs_info.get("epsilon_info", 0.005))
        delta_info = clip01(float(coeffs_info.get("delta_info", 0.05)))

        new_state = self.received_info
        if rng.random() < p_info or rng.random() < epsilon:
            new_state = True
        else:
            if self.received_info:
                # Decay with probability delta_info
                if rng.random() < delta_info:
                    new_state = False
        self.received_info = new_state
        return new_state

    def purchase_masks(
        self,
        retailer: "Retailer",
        price_per_mask: float,
        supply_reliability: float,
        mask_inventory_capacity: float,
        rng: np.random.Generator
    ) -> bool:
        """
        Attempt to purchase one mask unit from retailer if inventory is below capacity.

        Args:
            retailer (Retailer): The retailer entity.
            price_per_mask (float): Price per mask.
            supply_reliability (float): Probability of successful supply.
            mask_inventory_capacity (float): Max inventory per agent.
            rng (np.random.Generator): Random number generator.

        Returns:
            bool: True if purchase succeeded, else False.
        """
        pass
        if self.inventory_masks >= int(mask_inventory_capacity):
            return False
        if self.income_level < price_per_mask:
            return False
        if retailer.stock_level <= 0:
            return False
        if rng.random() > float(supply_reliability):
            return False
        # Proceed with purchase
        success = retailer.sell_masks(1)
        if success:
            self.inventory_masks += 1
            self.income_level -= price_per_mask
        return success

    def comply_with_policy(
        self,
        enforcement_probability: float,
        enforcement_strength: float,
        noncompliance_noise: float,
        policy_active: bool,
        rng: np.random.Generator
    ) -> bool:
        """
        Determine instantaneous compliance under enforcement at a mandated context.

        Args:
            enforcement_probability (float): Chance of enforcement at an interaction.
            enforcement_strength (float): Strength of enforcement effect on compliance.
            noncompliance_noise (float): Lower bound of noncompliance.
            policy_active (bool): Whether a policy is active.
            rng (np.random.Generator): Random number generator.

        Returns:
            bool: Effective compliance state for an interaction.
        """
        pass
        if not policy_active:
            return self.mask_status
        enforced = rng.random() < enforcement_probability * enforcement_strength
        if enforced:
            return True
        # Minimal noncompliance floor; if below threshold, force masked
        if rng.random() > max(noncompliance_noise, 0.0):
            return self.mask_status or (rng.random() < 0.5)
        return self.mask_status


@dataclass
class Location:
    """
    Representation of a location where interactions occur.

    Attributes:
        id (int): Unique identifier for the

# Execute main for both direct execution and sandbox wrapper invocation
main()
"""