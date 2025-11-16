import os
import json
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import lognorm
import matplotlib.pyplot as plt


# Global Path Handling per instruction
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Silence some warnings for demonstration clarity
warnings.filterwarnings("ignore", category=RuntimeWarning)


def sigmoid(x: float) -> float:
    """
    Compute the sigmoid function.
    Returns a value between 0 and 1.

    This helper function is used for logistic hazard computations across adoption and dropout processes.
    """
    pass
    try:
        if x < -700:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.
    Returns a float in [0, 1]. If values sum to zero, returns 0.

    Notes:
    - This is used to compute inequality_of_adoption across groups.
    """ 
    pass
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.shape[0]
    cum = np.cumsum(arr)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


def ensure_list(value: Any) -> List[Any]:
    """
    Ensure a given value is a list. Wrap a scalar into a single-item list if necessary.
    """
    pass
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def try_int(x: Any) -> Optional[int]:
    """
    Try to cast an object to int. Return None on failure.
    """
    pass
    try:
        return int(x)
    except Exception:
        return None


def rs_normal(mu: float, sigma: float, rng: random.Random) -> float:
    """
    Draw a normal random variable via Python's random.gauss for reproducibility.
    """
    pass
    return rng.gauss(mu, sigma)


def rs_choice(population: List[Any], rng: random.Random) -> Any:
    """
    Draw a random choice from a population using Python's random.Random.
    """
    pass
    if not population:
        return None
    idx = rng.randrange(0, len(population))
    return population[idx]


def rs_uniform(a: float, b: float, rng: random.Random) -> float:
    """
    Draw a uniform random variable on [a, b).
    """
    pass
    return rng.uniform(a, b)


def lognormal_delay(mu_log: float, sigma_log: float, rng: random.Random) -> int:
    """
    Sample a discrete delay from a lognormal distribution with parameters on log scale.
    Returns a non-negative integer number of days.
    """
    pass
    # Use scipy.stats.lognorm to obtain RV; shape = sigma, scale = exp(mu)
    try:
        shape = max(1e-6, float(sigma_log))
        scale = float(math.exp(mu_log))
        rv = lognorm(s=shape, scale=scale)
        delay = rv.rvs(random_state=rng)  # uses numpy random if seedless; but we pass python rng, so approximate with uniform draw
        # To use rng consistently, sample U and inverse CDF
        u = rng.random()
        delay = rv.ppf(u)
        return max(0, int(round(delay)))
    except Exception:
        # Fallback to simple geometric-like integer with mean ~exp(mu)
        m = max(0.0, math.exp(mu_log))
        return max(0, int(round(m)))


def safe_mean(x: List[float]) -> float:
    """
    Safe mean function that returns 0.0 for empty lists.
    """
    pass
    if not x:
        return 0.0
    return float(np.mean(x))


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute Root Mean Square Error between two sequences. Returns NaN if lengths mismatch or zero length.
    """
    pass
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return float("nan")
    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def mae(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute Mean Absolute Error between two sequences. Returns NaN if lengths mismatch or zero length.
    """
    pass
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return float("nan")
    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(arr_true - arr_pred)))


MODEL_PLAN: Dict[str, Any] = {
    "title": "Simulation Task",
    "description": "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks.",
    "simulation_type": "agent_based",
    "prediction_period": {"start_day": 30, "end_day": 39},
    "evaluation_metrics": ["RMSE", "TimeTo50Error", "Rb_MAE", "Churn_MAE"],
    "parameters": {
        "num_agents": 1000,
        "network_topology": "observed_multiplex",
        "avg_degree": 8,
        "small_world_rewiring_prob": 0.1,
        "dynamic_rewire_prob": 0.02,
        "homophily_strength": 0.2,
        "edge_weight_mean": 1.0,
        "initial_adoption_rate": 0.05,
        "threshold_mean": 0.3,
        "threshold_sd": 0.1,
        "influence_weight_peer": 1.0,
        "influence_weight_media": 0.2,
        "influence_weight_policy": 0.5,
        "social_norm_strength": 0.5,
        "adoption_function": "logistic",
        "logistic_beta": 5.0,
        "dropout_base_rate": 0.01,
        "fatigue_rate": 0.005,
        "compliance_cost": 0.2,
        "benefit_perceived": 0.3,
        "stubborn_fraction": 0.1,
        "mandate_start_day": 10,
        "mandate_enforcement_strength": 0.6,
        "penalty_cost": 0.5,
        "messaging_intensity": 0.3,
        "campaign_start_day": 10,
        "observation_noise": 0.0,
        "multi_layer_network": True,
        "household_cluster_size": 4,
        "time_step_length_days": 1,
        "simulation_steps": 40,
        "record_interval": 1,
        "rng_seed": 42,
        "convergence_delta_threshold": 0.001,
        "convergence_lookback": 10,
        "layer_weight_family": 0.6,
        "layer_weight_work": 0.3,
        "layer_weight_community": 0.1,
        "layer_contact_prob_family": 0.9,
        "layer_contact_prob_work": 0.6,
        "layer_contact_prob_community": 0.25,
        "multiplex_overlap_multiplier": 1.3,
        "exposure_window": 3,
        "edge_directionality": "undirected_symmetrized",
        "policy_odds_multiplier": 1.5,
        "message_credibility": 0.7,
        "media_reach": 0.8,
        "message_bias": 0.0,
        "message_frequency": 1.0,
        "initial_informed_rate": 0.2,
        "info_hazard_base": 0.05,
        "info_peer_effect_per_adopting_neighbor": 0.02,
        "info_external_rate": 0.01,
        "adoption_logit_alpha": -2.0,
        "adoption_beta_neighbors": 3.0,
        "adoption_beta_neighbors_sq": 1.5,
        "adoption_gamma_info": 1.2,
        "adoption_gamma_risk": 0.8,
        "adoption_gamma_risk_x_neighbors": 0.5,
        "adoption_gamma_layer_family": 0.5,
        "adoption_gamma_layer_work": 0.3,
        "adoption_gamma_layer_community": 0.1,
        "adoption_delay_mu_log": 0.0,
        "adoption_delay_sigma_log": 0.75,
        "adoption_threshold_lambda": 2.0,
        "drop_logit_intercept": -4.0,
        "drop_beta_one_minus_neighbor_frac": 2.0,
        "drop_beta_one_minus_risk": 1.5,
        "dropout_min_duration_days": 2,
        "dropout_probability_cap": 0.5,
        "Rb_window": 3,
        "inequality_group_field": "age_group",
        "prediction_start_day": 30,
        "prediction_end_day": 39
    },
    "data_files": {
        "agent_attributes.csv": "agent_attributes.csv",
        "social_network.json": "social_network.json",
        "train_data.csv": "train_data.csv"
    },
    "entities": ["Person", "SocialNetwork", "PublicHealthAuthority", "MediaChannel", "SimulationEnvironment"]
}


@dataclass
class Person:
    """
    Person entity representing an agent in the simulation with socio-demographic attributes and behavioral states.

    Attributes:
    - id: Unique agent identifier.
    - age_group: Categorical age group label.
    - occupation: Categorical occupation label.
    - risk_perception: Float in [0, 1] representing risk perception.
    - adoption_state: Boolean indicating mask-wearing status.
    - informed: Boolean indicating whether the agent has received mask-related information.
    - propensity_to_adopt: Optional propensity parameter for alternative models.
    - threshold: Exposure threshold (used if adoption_function is 'threshold').
    - influenceability: Optional sensitivity to social influence (not explicitly used; placeholder).
    - stubborn: Boolean indicating resistance to peer influence.
    - social_trust: Placeholder for trust in information sources.
    - conformity: Placeholder parameter for social norm adherence.
    - compliance_cost_sensitivity: Placeholder; individual sensitivity to compliance costs.
    - fatigue: Accumulated fatigue contributing to dropout propensity.
    - time_since_adoption: Number of days since last adoption.
    - dropout_probability: Cached dropout probability (diagnostic).
    - degree: Total degree across all layers.
    - neighbors_family: Set of neighbor IDs for the family layer.
    - neighbors_work_school: Set of neighbor IDs for the work/school layer.
    - neighbors_community: Set of neighbor IDs for the community layer.
    - cumulative_exposures: Cumulative weighted exposures used for threshold adoption model.
    - exposures_before_adoption: Count of exposures accumulated before first adoption (for metrics).
    - first_adoption_day: Day index of first adoption.
    - pending_adoption_delay: Remaining days before a scheduled adoption activation; -1 if none scheduled.
    """
    id: int
    age_group: str = "Unknown"
    occupation: str = "Unknown"
    risk_perception: float = 0.5
    adoption_state: bool = False
    informed: bool = False
    propensity_to_adopt: float = 0.0
    threshold: float = 2.0
    influenceability: float = 1.0
    stubborn: bool = False
    social_trust: float = 0.5
    conformity: float = 0.5
    compliance_cost_sensitivity: float = 1.0
    fatigue: float = 0.0
    time_since_adoption: int = 0
    dropout_probability: float = 0.0
    degree: int = 0
    neighbors_family: set = field(default_factory=set)
    neighbors_work_school: set = field(default_factory=set)
    neighbors_community: set = field(default_factory=set)
    cumulative_exposures: float = 0.0
    exposures_before_adoption: float = 0.0
    first_adoption_day: Optional[int] = None
    pending_adoption_delay: int = -1

    def observe_neighbors(self):
        """
        Placeholder method for adherence to interface; observation is handled by SocialNetworkEngine.
        """
        pass

    def update_belief(self):
        """
        Placeholder method for adherence to interface; belief update is handled by InfoDiffusion module.
        """
        pass

    def decide_adopt(self):
        """
        Placeholder method for adherence to interface; adoption decision is handled by SocialInfluenceAdoption module.
        """
        pass

    def wear_mask(self):
        """
        Set the agent into adoption (mask-wearing) state and reset relevant counters.
        """
        pass
        self.adoption_state = True
        self.time_since_adoption = 0
        if self.first_adoption_day is None:
            self.first_adoption_day = 0  # Will be set by simulation to current day

    def drop_mask(self):
        """
        Set the agent into non-adoption state and reset relevant counters.
        """
        pass
        self.adoption_state = False
        self.time_since_adoption = 0

    def share_information(self):
        """
        Placeholder for information sharing logic; handled implicitly in InfoDiffusion.
        """
        pass

    def comply_with_mandate(self):
        """
        Placeholder for mandate compliance; actual effect captured in hazard modifiers.
        """
        pass

    def rewire_social_ties(self):
        """
        Placeholder for rewiring operation; implemented in SocialNetworkEngine per ego.
        """
        pass


@dataclass
class SocialNetwork:
    """
    SocialNetwork entity holding multiplex adjacency and computing diagnostics.

    Attributes:
    - topology_type: String specifying topology source or type.
    - adjacency_family: Dict mapping agent_id -> set of neighbor ids in family layer.
    - adjacency_work_school: Dict mapping agent_id -> set of neighbor ids in work/school layer.
    - adjacency_community: Dict mapping agent_id -> set of neighbor ids in community layer.
    - adjacency_all: Dict mapping agent_id -> set of neighbor ids across all layers.
    - avg_degree: Float average degree across all layers.
    - clustering_coefficient: Placeholder for clustering coefficient.
    - homophily_strength: Preference for same-status/group ties in rewiring.
    - edge_weight_distribution: Placeholder for edge weight distribution.
    """
    topology_type: str = "observed_multiplex"
    adjacency_family: Dict[int, set] = field(default_factory=dict)
    adjacency_work_school: Dict[int, set] = field(default_factory=dict)
    adjacency_community: Dict[int, set] = field(default_factory=dict)
    adjacency_all: Dict[int, set] = field(default_factory=dict)
    avg_degree: float = 0.0
    clustering_coefficient: float = 0.0
    homophily_strength: float = 0.2
    edge_weight_distribution: Optional[Any] = None

    def generate_network(self):
        """
        Placeholder for network generation logic; actual implementation is provided in SocialNetworkEngine if needed.
        """
        pass

    def rewire_edges(self):
        """
        Placeholder for network rewiring; actual implementation is provided in SocialNetworkEngine.
        """
        pass

    def propagate_influence(self):
        """
        Placeholder for influence propagation; actual computation in SocialNetworkEngine.
        """
        pass


@dataclass
class PublicHealthAuthority:
    """
    PublicHealthAuthority entity producing policy and messaging signals.

    Attributes:
    - mandate_active: Boolean indicating if mandate is active currently.
    - mandate_start_day: Day index when mandate is scheduled to start.
    - enforcement_strength: Scale of enforcement pressure on drop hazard reduction.
    - penalty_cost: Perceived penalty for non-compliance.
    - messaging_intensity: Messaging intensity parameter.
    - credibility: Credibility of public authority and messaging.
    - campaign_start_day: Day index for the start of messaging campaign.
    - policy_schedule: Placeholder for more complex policy schedules.
    """
    mandate_active: bool = False
    mandate_start_day: int = 10
    enforcement_strength: float = 0.6
    penalty_cost: float = 0.5
    messaging_intensity: float = 0.3
    credibility: float = 0.7
    campaign_start_day: int = 10
    policy_schedule: Optional[Dict[str, Any]] = None

    def issue_mandate(self, current_day: int):
        """
        Update mandate status based on current day.
        """
        pass
        self.mandate_active = current_day >= int(self.mandate_start_day)

    def broadcast_message(self):
        """
        Placeholder for broadcasting messages; actual effect handled in PolicyAndMessaging and InfoDiffusion modules.
        """
        pass

    def adjust_policy(self, current_day: int):
        """
        Placeholder for adjusting policy over time.
        """
        pass
        # For now, no dynamic adjustment beyond mandate start


@dataclass
class MediaChannel:
    """
    MediaChannel entity representing media broadcasting properties.

    Attributes:
    - message_bias: Bias affecting messaging effect.
    - reach: Proportion of population reached daily.
    - message_frequency: Frequency multiplier for messaging.
    - noise_level: Noise in message signal.
    - credibility: Perceived credibility of the media channel.
    """
    message_bias: float = 0.0
    reach: float = 0.8
    message_frequency: float = 1.0
    noise_level: float = 0.0
    credibility: float = 0.7

    def broadcast_message(self):
        """
        Placeholder; effective messaging signal computed in PolicyAndMessaging based on this channel.
        """
        pass


@dataclass
class SimulationEnvironment:
    """
    SimulationEnvironment manages global simulation clock and termination.

    Attributes:
    - current_day: Current day index in the simulation.
    - max_steps: Maximum number of steps (days) to simulate.
    - rng_seed: Random seed for reproducibility.
    - time_step_length_days: Length of one time step in days.
    - termination_condition: Optional callable deciding termination.
    """
    current_day: int = 0
    max_steps: int = 40
    rng_seed: int = 42
    time_step_length_days: int = 1
    termination_condition: Optional[Any] = None

    def initialize(self):
        """
        Initialize environment at day 0. No additional logic required here.
        """
        pass
        self.current_day = 0

    def step(self):
        """
        Advance the environment by one time step.
        """
        pass
        self.current_day += 1

    def collect_metrics(self):
        """
        Placeholder for environment-level metrics (unused).
        """
        pass

    def terminate_if_converged(self, recent_adoption_rates: List[float], delta_threshold: float, lookback: int) -> bool:
        """
        Determine if simulation can terminate early based on convergence of adoption rates over a lookback window.

        Returns True if converged; False otherwise.
        """
        pass
        if len(recent_adoption_rates) < max(2, lookback):
            return False
        recent = recent_adoption_rates[-lookback:]
        diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return all(d < delta_threshold for d in diffs)


class SocialNetworkEngine:
    """
    SocialNetworkEngine maintains multiplex adjacency, computes daily exposures and layer-specific neighbor adoption fractions,
    and performs homophily-based rewiring.

    Key parameters are taken from the configuration dictionary, including:
    - layer weights and contact probabilities
    - dynamic rewiring probability
    - homophily strength
    - multiplex overlap multiplier
    - exposure window for rolling aggregation

    Methods:
    - build_or_load_network: Load from JSON if available, else generate synthetic network.
    - compute_daily_exposures: For each agent, sample layer contacts and compute weighted neighbor adoption fractions.
    - rewire_step: With a probability per ego, rewire one tie towards homophily.
    - assortativity_by_adoption: Compute assortativity coefficient for the combined graph based on adoption state.
    """
    def __init__(self, agents: Dict[int, Person], config: Dict[str, Any], rng: random.Random):
        """
        Initialize the SocialNetworkEngine with agents and configuration.

        Args:
        - agents: Dictionary mapping agent ids to Person objects.
        - config: Configuration parameters including network and layer properties.
        - rng: Random number generator for reproducible stochastic operations.
        """
        pass
        self.agents = agents
        self.config = config
        self.rng = rng
        self.layer_weights = {
            "family": float(config.get("layer_weight_family", 0.6)),
            "work_school": float(config.get("layer_weight_work", 0.3)),
            "community": float(config.get("layer_weight_community", 0.1)),
        }
        self.layer_contact_probs = {
            "family": float(config.get("layer_contact_prob_family", 0.9)),
            "work_school": float(config.get("layer_contact_prob_work", 0.6)),
            "community": float(config.get("layer_contact_prob_community", 0.25)),
        }
        self.dynamic_rewire_prob = float(config.get("dynamic_rewire_prob", 0.02))
        self.homophily_strength = float(config.get("homophily_strength", 0.2))
        self.overlap_multiplier = float(config.get("multiplex_overlap_multiplier", 1.3))
        self.exposure_window = int(config.get("exposure_window", 3))
        self.multi_layer_network = bool(config.get("multi_layer_network", True))

        # Internal state
        self.network = SocialNetwork()
        self.overlap_counts: Dict[Tuple[int, int], int] = {}
        self.degrees: Dict[int, int] = {}

        # Rolling memory for exposures
        self.buffer_frac_overall: List[Dict[int, float]] = []

    def _symmetrize(self, adj: Dict[int, set]) -> Dict[int, set]:
        """
        Symmetrize a given adjacency dictionary to ensure undirected edges.

        Returns a new adjacency dictionary with mutual links.
        """
        pass
        out = {i: set(neigh) for i, neigh in adj.items()}
        for i, neigh in adj.items():
            for j in neigh:
                out.setdefault(j, set()).add(i)
                out.setdefault(i, set()).add(j)
        return out

    def build_or_load_network(self, data_files: Dict[str, str]):
        """
        Load observed multiplex network from JSON if available; otherwise, generate a small-world inspired synthetic network
        and randomly assign edges to layers.

        The resulting adjacency is stored in self.network, and overlap counts are precomputed.
        """
        pass
        json_path = os.path.join(DATA_DIR, data_files.get("social_network.json", "social_network.json"))
        loaded = False
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                # Convert keys to ints and lists to sets while deduplicating
                fam, wor, com, all_adj = {}, {}, {}, {}
                for k_str, v in data.items():
                    i = try_int(k_str)
                    if i is None:
                        continue
                    fam[i] = set(try_int(x) for x in v.get("family", []) if try_int(x) is not None)
                    wor[i] = set(try_int(x) for x in v.get("work_school", []) if try_int(x) is not None)
                    com[i] = set(try_int(x) for x in v.get("community", []) if try_int(x) is not None)
                    all_set = set()
                    all_set.update(fam[i] if i in fam else set())
                    all_set.update(wor[i] if i in wor else set())
                    all_set.update(com[i] if i in com else set())
                    all_adj[i] = all_set
                fam = self._symmetrize(fam)
                wor = self._symmetrize(wor)
                com = self._symmetrize(com)
                all_adj = self._symmetrize(all_adj)

                self.network = SocialNetwork(
                    topology_type="observed_multiplex",
                    adjacency_family=fam,
                    adjacency_work_school=wor,
                    adjacency_community=com,
                    adjacency_all=all_adj,
                    homophily_strength=self.homophily_strength
                )
                loaded = True
            except Exception as e:
                print("Error loading social network JSON; falling back to synthetic generation:", e)

        if not loaded:
            # Synthetic generation: small-world on all agents; assign edges to layers randomly based on Dirichlet-proportions approximating layer weights
            num_agents = len(self.agents)
            avg_degree = int(max(2, int(self.config.get("avg_degree", 8))))
            k = max(2, avg_degree - (avg_degree % 2))
            p_rewire = float(self.config.get("small_world_rewiring_prob", 0.1))
            G = nx.watts_strogatz_graph(n=num_agents, k=k, p=p_rewire, seed=self.config.get("rng_seed", 42))
            adjacency_all = {i: set() for i in self.agents.keys()}
            for u, v in G.edges():
                adjacency_all[u].add(v)
                adjacency_all[v].add(u)

            # Randomly assign each undirected edge to one or more layers with probabilities approximating weights
            fam, wor, com = {i: set() for i in self.agents}, {i: set() for i in self.agents}, {i: set() for i in self.agents}
            for u, vs in adjacency_all.items():
                for v in vs:
                    if v < u:
                        continue
                    r = self.rng.random()
                    if r < 0.33:
                        fam[u].add(v); fam[v].add(u)
                    elif r < 0.66:
                        wor[u].add(v); wor[v].add(u)
                    else:
                        com[u].add(v); com[v].add(u)
                    # Some overlap chance
                    if self.rng.random() < 0.1:
                        extra = rs_choice(["family", "work_school", "community"], self.rng)
                        if extra == "family":
                            fam[u].add(v); fam[v].add(u)
                        elif extra == "work_school":
                            wor[u].add(v); wor[v].add(u)
                        else:
                            com[u].add(v); com[v].add(u)

            self.network = SocialNetwork(
                topology_type="small_world",
                adjacency_family=fam,
                adjacency_work_school=wor,
                adjacency_community=com,
                adjacency_all=adjacency_all,
                homophily_strength=self.homophily_strength
            )

        # Compute overlap counts
        self.overlap_counts.clear()
        for i, agent in self.agents.items():
            nf = self.network.adjacency_family.get(i, set())
            nw = self.network.adjacency_work_school.get(i, set())
            nc = self.network.adjacency_community.get(i, set())
            all_union = set().union(nf, nw, nc)
            for j in all_union:
                layers = int((j in nf)) + int((j in nw)) + int((j in nc))
                self.overlap_counts[tuple(sorted((i, j)))] = layers
        # Degree
        self.degrees = {i: len(self.network.adjacency_all.get(i, set())) for i in self.agents}
        for i, deg in self.degrees.items():
            self.agents[i].degree = deg

    def compute_daily_exposures(self) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]], Dict[int, float]]:
        """
        Compute per-agent overall neighbor adoption fractions and per-layer fractions based on sampled contacts.

        Returns:
        - frac_overall: dict agent_id -> overall weighted adoption fraction among contacted neighbors.
        - frac_by_layer: dict agent_id -> dict layer_name -> layer-specific adoption fraction among contacts.
        - exposures_weighted_adopting: dict agent_id -> total weighted adopting contacts for exposure accounting.
        """
        pass
        frac_overall: Dict[int, float] = {}
        frac_by_layer: Dict[int, Dict[str, float]] = {}
        exposures_weighted_adopting: Dict[int, float] = {}

        # Precompute adopter states for speed
        adopter_state = {i: int(agent.adoption_state) for i, agent in self.agents.items()}

        for i, agent in self.agents.items():
            contacted_layers: Dict[str, set] = {
                "family": set(),
                "work_school": set(),
                "community": set(),
            }
            # Sample contacts per layer
            for layer, adj in [
                ("family", self.network.adjacency_family),
                ("work_school", self.network.adjacency_work_school),
                ("community", self.network.adjacency_community),
            ]:
                neighs = adj.get(i, set())
                p_contact = self.layer_contact_probs.get(layer, 0.0)
                for j in neighs:
                    if self.rng.random() < p_contact:
                        contacted_layers[layer].add(j)

            # Compute per-layer fractions
            by_layer = {}
            total_weighted = 0.0
            total_weighted_adopt = 0.0

            # For exposures counting unique neighbors across layers with overlap multiplier
            contacted_all_unique = set().union(*contacted_layers.values())
            weighted_adopting_contacts = 0.0

            # Compute per-layer fractions naive
            for layer_name in ["family", "work_school", "community"]:
                contacts = list(contacted_layers[layer_name])
                if contacts:
                    adopters = sum(adopter_state.get(j, 0) for j in contacts)
                    by_layer[layer_name] = adopters / float(len(contacts))
                else:
                    by_layer[layer_name] = 0.0

            # Combine layers with weights and overlap multiplier applied at the individual neighbor level
            for j in contacted_all_unique:
                layers_present = [L for L in ["family", "work_school", "community"] if j in contacted_layers[L]]
                if not layers_present:
                    continue
                weight_sum = sum(self.layer_weights[L] for L in layers_present)
                # Apply overlap multiplier if multi-layer contact today
                if len(layers_present) >= 2:
                    weight_sum *= self.overlap_multiplier
                total_weighted += weight_sum
                total_weighted_adopt += weight_sum * adopter_state.get(j, 0)
                weighted_adopting_contacts += weight_sum * adopter_state.get(j, 0)

            overall = (total_weighted_adopt / total_weighted) if total_weighted > 0 else 0.0
            frac_overall[i] = overall
            frac_by_layer[i] = by_layer
            exposures_weighted_adopting[i] = weighted_adopting_contacts

        # Maintain rolling window if needed (not explicitly used but kept for extensibility)
        self.buffer_frac_overall.append(frac_overall.copy())
        if len(self.buffer_frac_overall) > self.exposure_window:
            self.buffer_frac_overall.pop(0)

        return frac_overall, frac_by_layer, exposures_weighted_adopting

    def rewire_step(self):
        """
        Perform dynamic homophily-driven rewiring with probability per ego (dynamic_rewire_prob).
        Prefers replacement ties that match adoption state and group attributes (age_group/occupation).
        """
        pass
        if self.dynamic_rewire_prob <= 0.0:
            return

        for i, agent in self.agents.items():
            if self.rng.random() >= self.dynamic_rewire_prob:
                continue
            # Pick a random neighbor j from combined adjacency
            neighbors = list(self.network.adjacency_all.get(i, set()))
            if not neighbors:
                continue
            j = rs_choice(neighbors, self.rng)
            # If adoption status same, skip rewiring
            if self.agents.get(j) and (self.agents[j].adoption_state == agent.adoption_state):
                continue

            # Candidate pool: nodes not connected to i and not i
            all_nodes = list(self.agents.keys())
            neighbors_set = self.network.adjacency_all.get(i, set())
            candidate_k = [k for k in all_nodes if k != i and k not in neighbors_set]
            if not candidate_k:
                continue

            # Homophily score: 1 + strength * (1{same adoption} + 0.5*1{same group attributes})
            scores = []
            for k in candidate_k:
                a_same = int(self.agents[k].adoption_state == agent.adoption_state)
                g_same = 0
                g_same += int(self.agents[k].age_group == agent.age_group)
                g_same += int(self.agents[k].occupation == agent.occupation)
                g_same = 1 if g_same > 0 else 0
                score = 1.0 + self.homophily_strength * (a_same + 0.5 * g_same)
                scores.append(score)
            # Normalize
            total = sum(scores) if scores else 1.0
            probs = [s / total for s in scores]
            # Sample k proportional to scores
            r = self.rng.random()
            cum = 0.0
            k_selected = candidate_k[-1]
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    k_selected = candidate_k[idx]
                    break

            # Rewire edge (i, j) -> (i, k_selected) across all layers where (i,j) exists
            for layer_name, adj in [
                ("family", self.network.adjacency_family),
                ("work_school", self.network.adjacency_work_school),
                ("community", self.network.adjacency_community)
            ]:
                if j in adj.get(i, set()):
                    # Remove old
                    adj[i].discard(j)
                    adj[j].discard(i)
                    # Add new
                    adj.setdefault(i, set()).add(k_selected)
                    adj.setdefault(k_selected, set()).add(i)
            # Update combined adjacency
            # Rebuild i's combined neighbors
            all_i = set().union(
                self.network.adjacency_family.get(i, set()),
                self.network.adjacency_work_school.get(i, set()),
                self.network.adjacency_community.get(i, set()),
            )
            self.network.adjacency_all[i] = all_i
            # Rebuild for old j and new k
            all_j = set().union(
                self.network.adjacency_family.get(j, set()),
                self.network.adjacency_work_school.get(j, set()),
                self.network.adjacency_community.get(j, set()),
            )
            self.network.adjacency_all[j] = all_j
            all_k = set().union(
                self.network.adjacency_family.get(k_selected, set()),
                self.network.adjacency_work_school.get(k_selected, set()),
                self.network.adjacency_community.get(k_selected, set()),
            )
            self.network.adjacency_all[k_selected] = all_k

        # Recompute overlap counts after rewiring
        self.overlap_counts.clear()
        for i, agent in self.agents.items():
            nf = self.network.adjacency_family.get(i, set())
            nw = self.network.adjacency_work_school.get(i, set())
            nc = self.network.adjacency_community.get(i, set())
            all_union = set().union(nf, nw, nc)
            for j in all_union:
                layers = int((j in nf)) + int((j in nw)) + int((j in nc))
                self.overlap_counts[tuple(sorted((i, j)))] = layers

    def assortativity_by_adoption(self) -> float:
        """
        Compute assortativity coefficient of the combined network with respect to adoption_state attribute.

        Returns:
        - Newman assortativity coefficient r in [-1, 1].
        """
        pass
        try:
            G = nx.Graph()
            for i in self.agents.keys():
                G.add_node(i, adoption=int(self.agents[i].adoption_state))
            for i, neighs in self.network.adjacency_all.items():
                for j in neighs:
                    if j > i:
                        G.add_edge(i, j)
            if G.number_of_edges() == 0:
                return 0.0
            r = nx.attribute_assortativity_coefficient(G, "adoption")
            if math.isnan(r):
                return 0.0
            return float(r)
        except Exception:
            return 0.0


class PolicyAndMessaging:
    """
    PolicyAndMessaging module that activates mandates and computes effective messaging intensity signals.

    Exposed signals:
    - policy_active: bool
    - policy_odds_multiplier: float
    - messaging_intensity_effective: float
    """
    def __init__(self, pha: PublicHealthAuthority, media: MediaChannel, config: Dict[str, Any]):
        """
        Initialize with public health authority, media, and configuration.
        """
        pass
        self.pha = pha
        self.media = media
        self.config = config
        # Read parameters
        self.policy_odds_multiplier_cfg = float(config.get("policy_odds_multiplier", 1.0))

    def step(self, current_day: int) -> Tuple[bool, float, float]:
        """
        Update policy and messaging signals for the current day.

        Returns:
        - policy_active
        - policy_odds_multiplier
        - messaging_intensity_effective
        """
        pass
        self.pha.issue_mandate(current_day)
        self.pha.adjust_policy(current_day)
        policy_active = bool(self.pha.mandate_active)
        policy_odds_mult = self.policy_odds_multiplier_cfg if policy_active else 1.0

        msg_intensity = float(self.pha.messaging_intensity)
        media_reach = float(self.media.reach)
        msg_cred = float(self.media.credibility)
        msg_bias = float(self.media.message_bias)
        msg_freq = float(self.media.message_frequency)
        messaging_intensity_effective = msg_intensity * media_reach * msg_cred * (1.0 + 0.1 * msg_bias) * msg_freq
        return policy_active, policy_odds_mult, messaging_intensity_effective


class InfoDiffusion:
    """
    InfoDiffusion module updates the 'informed' flag for each agent based on hazards:
    - base hazard
    - peer effect from adopting neighbors (approximate via overall neighbor adoption fraction)
    - external messaging effect (from PolicyAndMessaging signals)
    """
    def __init__(self, agents: Dict[int, Person], config: Dict[str, Any], rng: random.Random):
        """
        Initialize InfoDiffusion with agents, configuration, and RNG.
        """
        pass
        self.agents = agents
        self.config = config
        self.rng = rng

    def step(self, frac_overall: Dict[int, float], messaging_intensity_effective: float,
             mean_contact_prob: float) -> Dict[int, bool]:
        """
        Perform a single-day update of informed states.

        Args:
        - frac_overall: dict agent_id -> overall adoption fraction among neighbors from SocialNetworkEngine.
        - messaging_intensity_effective: effective intensity of external messaging.
        - mean_contact_prob: approximate mean contact probability used to scale peer exposures.

        Returns:
        - informed_flags: dict agent_id -> bool indicating informed status at end of step.
        """
        pass
        base = float(self.config.get("info_hazard_base", 0.05))
        peer_eff = float(self.config.get("info_peer_effect_per_adopting_neighbor", 0.02))
        external_rate = float(self.config.get("info_external_rate", 0.01))

        # Compute degree-based expected adopting contacts approximation
        informed_flags: Dict[int, bool] = {}
        for i, agent in self.agents.items():
            if agent.informed:
                informed_flags[i] = True
                continue
            frac = float(frac_overall.get(i, 0.0))
            degree = max(0, agent.degree)
            expected_contacts_adopting = frac * degree * mean_contact_prob
            # Peer term using per-contact hazard approximation: 1 - (1 - peer_eff)^k
            peer_term = 1.0 - ((1.0 - peer_eff) ** max(0.0, expected_contacts_adopting))
            msg_term = external_rate * max(0.0, messaging_intensity_effective)
            p_info = 1.0 - (1.0 - base) * (1.0 - peer_term) * (1.0 - msg_term)
            if self.rng.random() < p_info:
                informed_flags[i] = True
            else:
                informed_flags[i] = False

        # Update agents
        for i, flag in informed_flags.items():
            if flag:
                self.agents[i].informed = True

        return informed_flags


class SocialInfluenceAdoption:
    """
    SocialInfluenceAdoption module decides on adoption for non-adopting agents, using logistic or threshold models.
    """
    def __init__(self, agents: Dict[int, Person], config: Dict[str, Any], rng: random.Random):
        """
        Initialize with agents, configuration, and RNG. Pre-select stubborn agents.
        """
        pass
        self.agents = agents
        self.config = config
        self.rng = rng

        # Adoption model coefficients
        self.alpha = float(config.get("adoption_logit_alpha", -2.0))
        self.b1 = float(config.get("adoption_beta_neighbors", 3.0))
        self.b2 = float(config.get("adoption_beta_neighbors_sq", 1.5))
        self.gI = float(config.get("adoption_gamma_info", 1.2))
        self.gR = float(config.get("adoption_gamma_risk", 0.8))
        self.gXR = float(config.get("adoption_gamma_risk_x_neighbors", 0.5))
        self.gF = float(config.get("adoption_gamma_layer_family", 0.5))
        self.gW = float(config.get("adoption_gamma_layer_work", 0.3))
        self.gC = float(config.get("adoption_gamma_layer_community", 0.1))

        self.compliance_cost = float(config.get("compliance_cost", 0.2))
        self.benefit_perceived = float(config.get("benefit_perceived", 0.3))

        self.adoption_function = str(config.get("adoption_function", "logistic"))
        self.delay_mu = float(config.get("adoption_delay_mu_log", 0.0))
        self.delay_sigma = float(config.get("adoption_delay_sigma_log", 0.75))
        self.threshold_lambda = float(config.get("adoption_threshold_lambda", 2.0))

        # Build stubborn set
        stubborn_fraction = float(config.get("stubborn_fraction", 0.1))
        ids = list(self.agents.keys())
        n_stubborn = int(round(stubborn_fraction * len(ids)))
        # Sample inversely proportional to risk_perception
        weights = []
        for i in ids:
            w = max(1e-6, 1.0 - float(self.agents[i].risk_perception))
            weights.append(w)
        total = sum(weights) if weights else 1.0
        probs = [w / total for w in weights]
        chosen = set()
        # Weighted sampling without replacement
        while len(chosen) < n_stubborn and ids:
            r = self.rng.random()
            cum = 0.0
            sel_idx = len(ids) - 1
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    sel_idx = idx
                    break
            chosen.add(ids[sel_idx])
            # Remove and renormalize
            ids.pop(sel_idx)
            probs.pop(sel_idx)
            if probs:
                s = sum(probs)
                probs = [p / s for p in probs]
        for i in chosen:
            self.agents[i].stubborn = True

        # Track scheduled adoptions: agent_id -> days_remaining
        self.scheduled: Dict[int, int] = {}

    def step(self, frac_overall: Dict[int, float], frac_by_layer: Dict[int, Dict[str, float]],
             policy_active: bool, policy_odds_multiplier: float,
             exposures_weighted_adopting: Dict[int, float], current_day: int) -> Dict[int, bool]:
        """
        Execute adoption decisions for the day, including scheduled activations.

        Args:
        - frac_overall: dict agent_id -> overall fraction adopting among neighbors.
        - frac_by_layer: dict agent_id -> dict layer_name -> fraction adopting among contacts in that layer.
        - policy_active: whether a mandate is active.
        - policy_odds_multiplier: odds multiplier applied when policy is active.
        - exposures_weighted_adopting: dict agent_id -> weighted adopting contacts for exposure accounting.
        - current_day: current simulation day.

        Returns:
        - adoption_flags: dict agent_id -> bool indicating if agent is adopting at end of step.
        """
        pass
        adoption_flags: Dict[int, bool] = {}

        # Decrease scheduled delays and activate adoptions
        to_activate = []
        for aid, days_rem in list(self.scheduled.items()):
            days_rem -= 1
            self.scheduled[aid] = days_rem
            if days_rem <= 0:
                to_activate.append(aid)
        for aid in to_activate:
            agent = self.agents.get(aid)
            if agent and not agent.adoption_state:
                agent.wear_mask()
                agent.first_adoption_day = current_day
            self.scheduled.pop(aid, None)

        for i, agent in self.agents.items():
            if agent.adoption_state:
                adoption_flags[i] = True
                continue

            x = float(frac_overall.get(i, 0.0))
            x2 = x * x
            info = 1.0 if agent.informed else 0.0
            risk = float(agent.risk_perception)
            layer_terms = 0.0
            layers = frac_by_layer.get(i, {})
            layer_terms += self.gF * float(layers.get("family", 0.0))
            layer_terms += self.gW * float(layers.get("work_school", 0.0))
            layer_terms += self.gC * float(layers.get("community", 0.0))
            logit = self.alpha + self.b1 * x + self.b2 * x2 + self.gI * info + self.gR * risk + self.gXR * risk * x + layer_terms
            # Policy odds multiplier
            if policy_active:
                logit += math.log(max(1.0, float(policy_odds_multiplier)))
            # Utility difference
            U = float(self.benefit_perceived) - float(self.compliance_cost)
            logit += U

            # Stubbornness: can still adopt if policy strongly increases odds; here we allow but reduce probability if no policy
            if agent.stubborn and not policy_active:
                # Reduce logit substantially for stubborn absent policy
                logit -= 2.0

            if self.adoption_function == "logistic":
                p_adopt = sigmoid(logit)
                if self.rng.random() < p_adopt:
                    # Schedule with delay
                    delay = lognormal_delay(self.delay_mu, self.delay_sigma, self.rng)
                    if delay <= 0:
                        agent.wear_mask()
                        agent.first_adoption_day = current_day
                    else:
                        self.scheduled[i] = delay
            else:
                # Threshold model
                agent.cumulative_exposures += float(exposures_weighted_adopting.get(i, 0.0))
                K = max(0.5, float(self.threshold_lambda))
                if agent.cumulative_exposures >= K:
                    delay = lognormal_delay(self.delay_mu, self.delay_sigma, self.rng)
                    if delay <= 0:
                        agent.wear_mask()
                        agent.first_adoption_day = current_day
                    else:
                        self.scheduled[i] = delay

            adoption_flags[i] = bool(agent.adoption_state)

            # Track exposures before adoption for metrics
            if not agent.adoption_state:
                agent.exposures_before_adoption += float(exposures_weighted_adopting.get(i, 0.0))

        return adoption_flags


class DropoutAndFatigue:
    """
    DropoutAndFatigue module computes daily dropout decisions among adopters based on:
    - baseline dropout
    - local neighbor adoption fraction (1 - n)
    - low risk (1 - risk_perception)
    - fatigue accumulation
    - policy reduces dropout hazard via enforcement
    """
    def __init__(self, agents: Dict[int, Person], config: Dict[str, Any], rng: random.Random):
        """
        Initialize with agents, configuration, and RNG.
        """
        pass
        self.agents = agents
        self.config = config
        self.rng = rng
        self.base = float(config.get("dropout_base_rate", 0.01))
        self.fatigue_rate = float(config.get("fatigue_rate", 0.005))
        self.intercept = float(config.get("drop_logit_intercept", -4.0))
        self.bN = float(config.get("drop_beta_one_minus_neighbor_frac", 2.0))
        self.bR = float(config.get("drop_beta_one_minus_risk", 1.5))
        self.min_duration = int(config.get("dropout_min_duration_days", 2))
        self.cap = float(config.get("dropout_probability_cap", 0.5))

    def step(self, frac_overall: Dict[int, float], policy_active: bool, enforcement_strength: float) -> Dict[int, bool]:
        """
        Execute dropout decisions for adopters with sufficient duration.

        Args:
        - frac_overall: dict agent_id -> overall neighbor adoption fraction.
        - policy_active: bool; whether mandate active reduces dropout odds.
        - enforcement_strength: float scaling of dropout reduction when policy is active.

        Returns:
        - dropout_flags: dict agent_id -> bool indicating whether the agent has dropped at this step.
        """
        pass
        dropout_flags: Dict[int, bool] = {}
        for i, agent in self.agents.items():
            if not agent.adoption_state:
                dropout_flags[i] = False
                continue
            agent.time_since_adoption += 1
            # Fatigue accumulation
            agent.fatigue += self.fatigue_rate
            if agent.time_since_adoption < self.min_duration:
                dropout_flags[i] = False
                continue

            n = float(frac_overall.get(i, 0.0))
            logit_drop = self.intercept + self.bN * (1.0 - n) + self.bR * (1.0 - float(agent.risk_perception)) + (agent.fatigue)
            if policy_active:
                logit_drop -= math.log(1.0 + max(0.0, enforcement_strength))
            p_drop = min(self.cap, sigmoid(logit_drop) + self.base)
            agent.dropout_probability = p_drop
            if self.rng.random() < p_drop:
                agent.drop_mask()
                dropout_flags[i] = True
            else:
                dropout_flags[i] = False
        return dropout_flags


class AdoptionAggregator:
    """
    AdoptionAggregator module computes daily observables and helper metrics for evaluation:
    - adoption_rate_daily
    - final_adoption_rate
    - time_to_50_percent_adoption
    - Rb_series (windowed ratio)
    - churn_rate_daily
    - mean_exposures_before_adoption
    - assortativity_by_adoption
    - inequality_of_adoption (difference and optionally Gini)
    - policy_effect_size (difference-in-differences slope around mandate)
    - info_rate_daily
    - peak_adoption_rate
    """
    def __init__(self, agents: Dict[int, Person], network_engine: SocialNetworkEngine, config: Dict[str, Any]):
        """
        Initialize with agents, network engine for assortativity, and configuration for window sizes and group fields.
        """
        pass
        self.agents = agents
        self.network_engine = network_engine
        self.config = config

        self.Rb_window = int(config.get("Rb_window", 3))
        self.inequality_group_field = str(config.get("inequality_group_field", "age_group"))
        self.mandate_start_day = int(config.get("mandate_start_day", 10))

        # Time series
        self.adoption_rate_daily: List[float] = []
        self.info_rate_daily: List[float] = []
        self.churn_rate_daily: List[float] = []
        self.Rb_series: List[float] = []
        self.assortativity_daily: List[float] = []
        self.inequality_daily: List[float] = []
        self.policy_effect_size_value: Optional[float] = None
        self.peak_adoption_rate_value: Optional[float] = None
        self.time_to_50_value: Optional[int] = None
        self.final_adoption_rate_value: Optional[float] = None
        self.mean_exposures_before_adoption_value: Optional[float] = None

        # For Rb and churn
        self.new_adopters_history: List[int] = []
        self.active_adopters_history: List[int] = []
        self.drops_history: List[int] = []

    def step(self, current_day: int, dropout_flags: Dict[int, bool]):
        """
        Collect daily observables at the end of the step.

        Args:
        - current_day: int day index.
        - dropout_flags: dict agent_id -> bool flags of drop that occurred today.
        """
        pass
        # Adoption and info rates
        adopters = sum(int(a.adoption_state) for a in self.agents.values())
        infos = sum(int(a.informed) for a in self.agents.values())
        N = max(1, len(self.agents))
        adoption_rate = adopters / float(N)
        info_rate = infos / float(N)
        self.adoption_rate_daily.append(adoption_rate)
        self.info_rate_daily.append(info_rate)

        # New adopters today
        new_adopters = sum(1 for a in self.agents.values() if a.first_adoption_day == current_day)
        self.new_adopters_history.append(new_adopters)
        self.active_adopters_history.append(adopters)

        # Drops today
        drops_today = sum(1 for k, v in dropout_flags.items() if v)
        # Churn rate: drops / adopters yesterday (avoid div by zero)
        prev_active = self.active_adopters_history[-2] if len(self.active_adopters_history) >= 2 else adopters
        churn_rate = (drops_today / float(prev_active)) if prev_active > 0 else 0.0
        self.churn_rate_daily.append(churn_rate)
        self.drops_history.append(drops_today)

        # Rb estimation: new adopters in window / active adopters lagged by 1
        w = self.Rb_window
        if len(self.new_adopters_history) >= w + 1:
            num = sum(self.new_adopters_history[-w:])
            denom = self.active_adopters_history[-w - 1] if len(self.active_adopters_history) >= w + 1 else 0
            Rb = (num / float(max(1, denom))) if denom > 0 else 0.0
            self.Rb_series.append(Rb)
        else:
            self.Rb_series.append(0.0)

        # Assortativity by adoption
        r = self.network_engine.assortativity_by_adoption()
        self.assortativity_daily.append(r)

        # Inequality across groups (difference between max and min group adoption rates)
        group_field = self.inequality_group_field
        groups: Dict[Any, List[int]] = {}
        for a in self.agents.values():
            g = getattr(a, group_field, "Unknown")
            groups.setdefault(g, []).append(int(a.adoption_state))
        group_rates = [float(np.mean(vals)) if vals else 0.0 for vals in groups.values()]
        if group_rates:
            inequality = max(group_rates) - min(group_rates)
        else:
            inequality = 0.0
        self.inequality_daily.append(inequality)

    def finalize(self):
        """
        Compute end-of-run summary metrics after all steps:
        - final adoption rate
        - time to 50 percent adoption
        - mean exposures before adoption
        - policy effect size (difference in slopes pre/post around mandate)
        - peak adoption rate
        """
        pass
        if self.adoption_rate_daily:
            self.final_adoption_rate_value = self.adoption_rate_daily[-1]
            self.peak_adoption_rate_value = max(self.adoption_rate_daily)
            # Time to 50%
            self.time_to_50_value = None
            for t, rate in enumerate(self.adoption_rate_daily):
                if rate >= 0.5:
                    self.time_to_50_value = t
                    break
        # Mean exposures before adoption among adopters
        exposures = [a.exposures_before_adoption for a in self.agents.values() if a.first_adoption_day is not None]
        self.mean_exposures_before_adoption_value = float(np.mean(exposures)) if exposures else 0.0

        # Policy effect size: slope post - slope pre around mandate start
        m = int(self.mandate_start_day)
        pre_end = max(1, m - 1)
        pre_start = max(0, pre_end - 4)
        post_start = m
        post_end = min(len(self.adoption_rate_daily) - 1, post_start + 4)
        def slope(y, s, e):
            if e <= s:
                return 0.0
            xs = list(range(s, e + 1))
            ys = y[s:e + 1]
            x_mean = np.mean(xs)
            y_mean = np.mean(ys)
            num = sum((x - x_mean) * (yy - y_mean) for x, yy in zip(xs, ys))
            den = sum((x - x_mean) ** 2 for x in xs)
            return float(num / den) if den > 0 else 0.0
        pre_slope = slope(self.adoption_rate_daily, pre_start, pre_end)
        post_slope = slope(self.adoption_rate_daily, post_start, post_end)
        self.policy_effect_size_value = post_slope - pre_slope

    def results_dataframe(self) -> pd.DataFrame:
        """
        Build a DataFrame containing key time series observables for the simulation.

        Returns:
        - pd.DataFrame with columns: day, adoption_rate, info_rate, churn_rate, Rb, assortativity, inequality
        """
        pass
        days = list(range(len(self.adoption_rate_daily)))
        data = {
            "day": days,
            "adoption_rate": self.adoption_rate_daily,
            "info_rate": self.info_rate_daily,
            "churn_rate": self.churn_rate_daily,
            "Rb": self.Rb_series,
            "assortativity": self.assortativity_daily,
            "inequality": self.inequality_daily,
        }
        return pd.DataFrame(data)


class DataIO:
    """
    DataIO provides data loading and preprocessing utilities for agents, social network, and time series.
    It adheres to the path handling instructions using environment variables PROJECT_ROOT and DATA_PATH.

    If files are missing or malformed, synthetic data is generated to keep the simulation executable.
    """
    def __init__(self, config: Dict[str, Any], rng: random.Random):
        """
        Initialize data IO with configuration and RNG.
        """
        pass
        self.config = config
        self.rng = rng

    def load_agents(self, data_files: Dict[str, str]) -> Dict[int, Person]:
        """
        Load agents from agent_attributes.csv. If unavailable, generate synthetic agents.

        Returns:
        - Dict[int, Person]
        """
        pass
        path = os.path.join(DATA_DIR, data_files.get("agent_attributes.csv", "agent_attributes.csv"))
        agents: Dict[int, Person] = {}
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "agent_id" not in df.columns:
                    raise ValueError("agent_attributes.csv missing 'agent_id'")
                df["agent_id"] = df["agent_id"].astype(int)
                # Ensure required fields
                if "age_group" not in df.columns:
                    df["age_group"] = "Unknown"
                if "occupation" not in df.columns:
                    df["occupation"] = "Unknown"
                if "risk_perception" not in df.columns:
                    df["risk_perception"] = 0.5

                for _, row in df.iterrows():
                    pid = int(row["agent_id"])
                    agents[pid] = Person(
                        id=pid,
                        age_group=str(row.get("age_group", "Unknown")),
                        occupation=str(row.get("occupation", "Unknown")),
                        risk_perception=float(row.get("risk_perception", 0.5))
                    )
                return agents
            except Exception as e:
                print("Error loading agent_attributes.csv; generating synthetic agents:", e)

        # Synthetic agents
        N = int(self.config.get("num_agents", 1000))
        age_groups = ["Youth", "Young Adult", "Middle Age", "Senior"]
        occupations = ["Student", "Blue Collar", "White Collar", "Retired"]
        for i in range(N):
            agents[i] = Person(
                id=i,
                age_group=rs_choice(age_groups, self.rng),
                occupation=rs_choice(occupations, self.rng),
                risk_perception=min(1.0, max(0.0, rs_uniform(0.2, 0.8, self.rng)))
            )
        return agents

    def load_timeseries(self, data_files: Dict[str, str], agents: Dict[int, Person]) -> Optional[pd.DataFrame]:
        """
        Load time series from train_data.csv. Returns DataFrame or None if unavailable.

        The expected columns: day, agent_id, wearing_mask, received_info.
        """
        pass
        path = os.path.join(DATA_DIR, data_files.get("train_data.csv", "train_data.csv"))
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            if "day" not in df.columns or "agent_id" not in df.columns:
                raise ValueError("train_data.csv must include 'day' and 'agent_id'")
            # Normalize booleans
            for col in ["wearing_mask", "received_info"]:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
                    df[col] = df[col].fillna(False).astype(bool)
                else:
                    df[col] = False
            df["agent_id"] = df["agent_id"].astype(int)
            df["day"] = df["day"].astype(int)
            # Filter to known agents only
            df = df[df["agent_id"].isin(agents.keys())].copy()
            return df
        except Exception as e:
            print("Error loading train_data.csv; ignoring time series:", e)
            return None


class Simulation:
    """
    Simulation coordinates entity initialization, module updates, and the simulation loop. It supports:
    - Data-driven initialization where possible
    - Running the daily loop with network exposures, info diffusion, adoption decisions, dropout, and rewiring
    - Collecting observables and evaluating specified metrics
    - Visualization and results export
    """
    def __init__(self, model_plan: Dict[str, Any]):
        """
        Prepare Simulation with provided model plan JSON-like dictionary.

        Args:
        - model_plan: Dictionary containing parameters, data file references, and evaluation metrics.
        """
        pass
        # FIXED: Ensure we copy and sanitize configuration dictionary
        self.model_plan = model_plan
        self.params: Dict[str, Any] = dict(model_plan.get("parameters", {}))
        self.data_files: Dict[str, str] = dict(model_plan.get("data_files", {}))
        self.evaluation_metrics: List[str] = list(model_plan.get("evaluation_metrics", []))

        # Random seed and RNG
        seed = int(self.params.get("rng_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        self.rng = random.Random(seed)

        # Entities
        self.agents: Dict[int, Person] = {}
        self.network_engine: Optional[SocialNetworkEngine] = None
        self.pha: PublicHealthAuthority = PublicHealthAuthority(
            mandate_active=False,
            mandate_start_day=int(self.params.get("mandate_start_day", 10)),
            enforcement_strength=float(self.params.get("mandate_enforcement_strength", 0.6)),
            penalty_cost=float(self.params.get("penalty_cost", 0.5)),
            messaging_intensity=float(self.params.get("messaging_intensity", 0.3)),
            credibility=float(self.params.get("message_credibility", 0.7)),
            campaign_start_day=int(self.params.get("campaign_start_day", 10)),
        )
        self.media: MediaChannel = MediaChannel(
            message_bias=float(self.params.get("message_bias", 0.0)),
            reach=float(self.params.get("media_reach", 0.8)),
            message_frequency=float(self.params.get("message_frequency", 1.0)),
            noise_level=float(self.params.get("observation_noise", 0.0)),
            credibility=float(self.params.get("message_credibility", 0.7)),
        )
        self.environment: SimulationEnvironment = SimulationEnvironment(
            current_day=0,
            max_steps=int(self.params.get("simulation_steps", 40)),
            rng_seed=seed,
            time_step_length_days=int(self.params.get("time_step_length_days", 1))
        )

        # Modules
        self.policy_module: Optional[PolicyAndMessaging] = None
        self.info_module: Optional[InfoDiffusion] = None
        self.adoption_module: Optional[SocialInfluenceAdoption] = None
        self.dropout_module: Optional[DropoutAndFatigue] = None
        self.aggregator: Optional[AdoptionAggregator] = None

        # Data I/O
        self.data_io = DataIO(self.params, self.rng)
        self.timeseries_df: Optional[pd.DataFrame] = None
        self.observed_adoption_series: Optional[List[float]] = None  # for RMSE over training window

    def initialize(self):
        """
        Initialize simulation: load data, build network, initialize states, and set up modules.
        """
        pass
        # Load agents
        agents = self.data_io.load_agents(self.data_files)
        self.agents = agents

        # Load timeseries
        ts = self.data_io.load_timeseries(self.data_files, self.agents)
        self.timeseries_df = ts

        # Build network
        self.network_engine = SocialNetworkEngine(self.agents, self.params, self.rng)
        self.network_engine.build_or_load_network(self.data_files)

        # Initialize day 0 states
        initial_adoption_rate = float(self.params.get("initial_adoption_rate", 0.05))
        initial_informed_rate = float(self.params.get("initial_informed_rate", 0.2))
        if ts is not None and not ts.empty:
            min_day = int(ts["day"].min())
            day0 = ts[ts["day"] == min_day]
            # Set adoption and informed states from day0 if available
            day0_grouped = day0.groupby("agent_id", as_index=False).agg({
                "wearing_mask": "max",
                "received_info": "max"
            })
            mask_map = {int(r.agent_id): bool(r.wearing_mask) for _, r in day0_grouped.iterrows()}
            info_map = {int(r.agent_id): bool(r.received_info) for _, r in day0_grouped.iterrows()}
            for i, agent in self.agents.items():
                agent.adoption_state = bool(mask_map.get(i, False))
                agent.informed = bool(info_map.get(i, False))
        else:
            # Synthetic initialization
            for i, agent in self.agents.items():
                agent.informed = (self.rng.random() < initial_informed_rate)
                # Risk-driven adoption at day 0
                p0 = initial_adoption_rate + 0.2 * (agent.risk_perception - 0.5)
                p0 = min(1.0, max(0.0, p0))
                agent.adoption_state = (self.rng.random() < p0)

        # Set adoption thresholds for threshold model
        for a in self.agents.values():
            a.threshold = float(self.params.get("adoption_threshold_lambda", 2.0))

        # Setup modules
        self.policy_module = PolicyAndMessaging(self.pha, self.media, self.params)
        self.info_module = InfoDiffusion(self.agents, self.params, self.rng)
        self.adoption_module = SocialInfluenceAdoption(self.agents, self.params, self.rng)
        self.dropout_module = DropoutAndFatigue(self.agents, self.params, self.rng)
        self.aggregator = AdoptionAggregator(self.agents, self.network_engine, self.params)

        # Prepare observed adoption series for evaluation (training window: first 30 days if available)
        self.observed_adoption_series = None
        if ts is not None and not ts.empty:
            # Compute mean wearing_mask per day across all agents in ts
            series = ts.groupby("day")["wearing_mask"].mean().sort_index().tolist()
            self.observed_adoption_series = series

    def step(self):
        """
        Perform a single simulation step: network exposures, policy/messaging, info diffusion,
        adoption decisions, dropout, rewiring, and aggregation.
        """
        pass
        if self.network_engine is None or self.policy_module is None or self.info_module is None \
                or self.adoption_module is None or self.dropout_module is None or self.aggregator is None:
            raise RuntimeError("Simulation modules are not initialized. Call initialize() first.")

        current_day = self.environment.current_day

        # Exposures and fractions
        frac_overall, frac_by_layer, exposures_weighted_adopting = self.network_engine.compute_daily_exposures()

        # Policy and messaging
        policy_active, policy_odds_multiplier, messaging_intensity_effective = self.policy_module.step(current_day)

        # Info diffusion
        mean_contact_prob = np.mean(list(self.network_engine.layer_contact_probs.values())) if self.network_engine.layer_contact_probs else 0.5
        informed_flags = self.info_module.step(frac_overall, messaging_intensity_effective, mean_contact_prob)

        # Adoption decisions and scheduled activations
        adoption_flags = self.adoption_module.step(frac_overall, frac_by_layer, policy_active, policy_odds_multiplier,
                                                   exposures_weighted_adopting, current_day)

        # Dropout
        dropout_flags = self.dropout_module.step(frac_overall, policy_active, float(self.params.get("mandate_enforcement_strength", 0.6)))

        # Rewiring
        self.network_engine.rewire_step()

        # Aggregation
        self.aggregator.step(current_day, dropout_flags)

        # Advance environment
        self.environment.step()

    def run(self):
        """
        Run the simulation loop for up to the configured number of steps, with early termination on convergence.
        """
        pass
        self.environment.initialize()
        max_steps = int(self.params.get("simulation_steps", 40))
        delta_threshold = float(self.params.get("convergence_delta_threshold", 0.001))
        lookback = int(self.params.get("convergence_lookback", 10))
        record_interval = int(self.params.get("record_interval", 1))

        for step_idx in range(max_steps):
            self.step()
            # Convergence check on recorded adoption rate series
            if self.aggregator and self.environment.current_day % record_interval == 0:
                if self.environment.terminate_if_converged(self.aggregator.adoption_rate_daily, delta_threshold, lookback):
                    break

        # Finalize aggregator metrics
        if self.aggregator:
            self.aggregator.finalize()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the simulation according to evaluation_metrics specified in the model plan.
        Supported metrics:
        - RMSE: between observed (if available) and simulated adoption_rate_daily over overlapping range
        - TimeTo50Error: absolute error in time to reach 50% adoption compared to observed (if available)
        - Rb_MAE: MAE between observed and simulated Rb_series (if observed available; else NaN)
        - Churn_MAE: MAE between observed and simulated churn_rate_daily (if observed available; else NaN)

        Returns:
        - Dictionary mapping metric name to computed value (float).
        """
        pass
        results: Dict[str, float] = {}
        metrics_list = ensure_list(self.evaluation_metrics)

        # Simulated series
        sim_adoption = self.aggregator.adoption_rate_daily if self.aggregator else []
        sim_Rb = self.aggregator.Rb_series if self.aggregator else []
        sim_churn = self.aggregator.churn_rate_daily if self.aggregator else []

        observed = self.observed_adoption_series

        for metric in metrics_list:
            if metric == "RMSE":
                if observed:
                    # Use overlapping range
                    L = min(len(observed), len(sim_adoption))
                    y_true = observed[:L]
                    y_pred = sim_adoption[:L]
                    results["RMSE"] = rmse(y_true, y_pred)
                else:
                    results["RMSE"] = float("nan")
            elif metric == "TimeTo50Error":
                if observed:
                    t_obs = None
                    for t, v in enumerate(observed):
                        if v >= 0.5:
                            t_obs = t
                            break
                    t_sim = self.aggregator.time_to_50_value if self.aggregator else None
                    if t_obs is None or t_sim is None:
                        results["TimeTo50Error"] = float("nan")
                    else:
                        results["TimeTo50Error"] = abs(int(t_obs) - int(t_sim))
                else:
                    results["TimeTo50Error"] = float("nan")
            elif metric == "Rb_MAE":
                # No observed Rb by default; return NaN
                results["Rb_MAE"] = float("nan")
            elif metric == "Churn_MAE":
                # No observed churn by default; return NaN
                results["Churn_MAE"] = float("nan")
            else:
                # Unknown metric placeholder
                results[metric] = float("nan")

        return results

    def visualize(self):
        """
        Visualize key time series: adoption rate, info rate, churn, and Rb.
        """
        pass
        if not self.aggregator:
            print("No aggregator found; nothing to visualize.")
            return

        df = self.aggregator.results_dataframe()
        if df.empty:
            print("No results to visualize.")
            return
        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(df["day"], df["adoption_rate"], label="Adoption Rate")
        ax1.set_title("Adoption Rate Over Time")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Rate")
        ax1.grid(True)

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(df["day"], df["info_rate"], label="Info Rate", color="orange")
        ax2.set_title("Information Rate Over Time")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Rate")
        ax2.grid(True)

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(df["day"], df["churn_rate"], label="Churn", color="red")
        ax3.set_title("Churn Rate Over Time")
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Rate")
        ax3.grid(True)

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(df["day"], df["Rb"], label="Rb", color="green")
        ax4.set_title("Behavior Reproduction Number (Rb)")
        ax4.set_xlabel("Day")
        ax4.set_ylabel("Rb")
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str):
        """
        Save simulation results to a CSV file at the given filename.

        Args:
        - filename: The path to the output CSV file.
        """
        pass
        if not self.aggregator:
            print("No aggregator results to save.")
            return
        df = self.aggregator.results_dataframe()
        try:
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print("Error saving results:", e)


def main():
    """
    Main function demonstrating initialization, running, visualization, and saving of the simulation.
    """
    pass
    # Instantiate Simulation with the embedded MODEL_PLAN
    sim = Simulation(MODEL_PLAN)
    sim.initialize()
    sim.run()
    metrics = sim.evaluate()
    print("Evaluation metrics:", metrics)
    # Visualization (optional; will display if environment supports)
    try:
        sim.visualize()
    except Exception as e:
        print("Visualization error:", e)
    # Save results
    output_path = os.path.join(PROJECT_ROOT, "results.csv")
    sim.save_results(output_path)


# Execute main for both direct execution and sandbox wrapper invocation
main()