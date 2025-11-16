def main():
    pass

import os
import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# Global path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or "."
DATA_PATH = os.environ.get("DATA_PATH") or ""
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

AGENTS_FILE = os.path.join(DATA_DIR, "agent_attributes.csv")
NETWORK_FILE = os.path.join(DATA_DIR, "social_network.json")
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.csv")


def sigmoid(x: float) -> float:
    """
    Compute the sigmoid/logistic function of x.

    Returns a value in (0,1), robust to large magnitudes of x.
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
        # Fallback for extreme overflows (unlikely)
        return 1.0 if x > 0 else 0.0


def clip01(p: float) -> float:
    """
    Clip a floating value to the [0,1] interval.

    Ensures probability bounds.
    """
    pass
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide a by b, returning default if b is zero.

    Prevents ZeroDivisionError in sparse cases.
    """
    pass
    if b == 0:
        return default
    return a / b


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Compute Root Mean Square Error between two sequences of floats.

    Returns NaN if sequences are empty or lengths mismatch.
    """
    pass
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return float("nan")
    errors = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    return math.sqrt(sum(errors) / len(errors))


def linear_slope(x: List[float], y: List[float]) -> float:
    """
    Compute slope of a simple linear regression y ~ a + b*x.

    Returns 0 if insufficient data or zero variance in x.
    """
    pass
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return 0.0
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom == 0:
        return 0.0
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    return num / denom


# Model plan embedded as single source of truth (essential fields)
MODEL_PLAN: Dict[str, Any] = {
    "title": "Simulation Task",
    "description": "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks.",
    "data_folder": "data_fitting/mask_adoption_data/",
    "data_files": {
        "agent_attributes.csv": "Contains demographic and behavioral attributes of each agent, including age, occupation, risk perception, and social connection counts",
        "social_network.json": "Contains structured data representing a social network by layers",
        "train_data.csv": "Time series data for the first 30 days, used for training the model"
    },
    "entities": [
        {
            "name": "Person",
            "attributes": [
                "id", "group_id", "mask_wearing", "belief_strength", "adoption_threshold",
                "susceptibility", "stubbornness", "risk_perception", "perceived_cost",
                "trust_in_sources", "degree"
            ],
            "behaviors": [
                "observe_neighbors", "update_beliefs", "decide_adopt_or_abandon",
                "share_signal", "rewire_ties", "comply_with_policy"
            ]
        },
        {
            "name": "Tie",
            "attributes": [
                "node_u", "node_v", "weight", "contact_frequency", "trust",
                "homophily_score", "last_interaction_time"
            ],
            "behaviors": [
                "transmit_influence", "update_weight_decay_or_strengthen"
            ]
        },
        {
            "name": "Group",
            "attributes": ["id", "size", "cohesion", "norm_level", "policy_stringency"],
            "behaviors": ["broadcast_group_norm", "update_norm_from_members"]
        },
        {
            "name": "InformationSource",
            "attributes": [
                "id", "credibility", "bias", "reach", "message_type",
                "broadcast_frequency", "effect_size"
            ],
            "behaviors": ["broadcast_message"]
        },
        {
            "name": "PolicyAuthority",
            "attributes": [
                "id", "mandate_active", "enforcement_probability", "penalty",
                "messaging_strength", "communication_frequency"
            ],
            "behaviors": [
                "issue_or_update_policy", "enforce_policy", "broadcast_guidance"
            ]
        }
    ],
    "interactions": [
        {"name": "peer_influence", "description": "Influence weighted by tie strength", "entities_involved": ["Person", "Tie", "Person"]},
        {"name": "information_broadcast", "description": "Messages update beliefs", "entities_involved": ["InformationSource", "Person"]},
        {"name": "policy_enforcement", "description": "Mandates and penalties", "entities_involved": ["PolicyAuthority", "Person"]},
        {"name": "group_norm_feedback", "description": "Group norms affect thresholds", "entities_involved": ["Group", "Person"]},
        {"name": "tie_rewiring", "description": "Dynamic rewiring based on homophily", "entities_involved": ["Person", "Tie", "Person"]}
    ],
    "parameters": {
        "population_size": 500,
        "simulation_steps": 180,
        "time_step": 1,
        "seed": 42,
        "initial_adoption_rate": 0.1,
        "adoption_target": 0.7,
        "network_topology": "watts_strogatz",
        "average_degree": 8,
        "rewiring_probability": 0.05,
        "homophily_strength": 0.3,
        "tie_weight_mean": 1.0,
        "tie_weight_std": 0.3,
        "social_influence_weight": 0.5,
        "adoption_threshold_mean": 0.5,
        "adoption_threshold_std": 0.15,
        "stubborn_agent_fraction": 0.05,
        "abandonment_rate": 0.01,
        "forgetting_rate": 0.005,
        "external_message_frequency": 0.1,
        "message_effect_size": 0.2,
        "policy_enforcement_prob": 0.3,
        "policy_effect_size": 0.25,
        "observation_window": 30,
        "noise_std": 0.05,
        "adoption_function": "threshold_logistic"
    },
    "evaluation_metrics": {
        "RMSE": {
            "description": "Root Mean Square Error measuring overall predictive accuracy of mask adoption rates",
            "interpretation": "Directly interpretable as percentage deviations due to the 0-1 range",
            "formula": "RMSE = sqrt(sum((predicted_rate - actual_rate)^2) / n)"
        }
    },
    "metrics": [
        {"name": "adoption_rate", "description": "Fraction of agents wearing masks at each step"},
        {"name": "time_to_threshold", "description": "Steps to reach adoption_target"},
        {"name": "steady_state_adoption", "description": "Mean adoption over final window"},
        {"name": "diffusion_speed", "description": "Slope during growth phase"},
        {"name": "assortativity_of_adoption", "description": "Clustering by state"},
        {"name": "group_adoption_gap", "description": "Gap across groups"},
        {"name": "behavioral_R0", "description": "New adoptions caused by one adopter early"},
        {"name": "exposures_to_adoption", "description": "Exposures before first adoption"}
    ],
    "validation_criteria": [
        {"name": "convergence_criterion", "description": "Change over window below 0.005"},
        {"name": "seed_robustness", "description": "Metrics vary <5% across seeds"},
        {"name": "limiting_cases_sanity", "description": "Bounds under extreme params"},
        {"name": "bounds_check", "description": "All probabilities in [0,1]"}
    ],
    "prediction_period": {
        "start_day": None,
        "end_day": None
    },
    # Placeholder for additional fields not explicitly handled
    "additional_fields": {}
}


@dataclass
class Tie:
    """
    Representation of a tie between two nodes in a given layer.

    Stores influence weight, contact frequency, trust, homophily, and recency.
    """
    node_u: int
    node_v: int
    weight: float
    contact_frequency: float
    trust: float
    homophily_score: float
    last_interaction_time: int

    def transmit_influence(self) -> float:
        """
        Compute influence transmission potential for this tie.

        Returns weight scaled by trust and contact frequency.
        """
        pass
        return self.weight * self.trust * self.contact_frequency

    def update_weight_decay_or_strengthen(self, interacted: bool) -> None:
        """
        Update the tie strength over time via decay or strengthen if interaction occurred.

        Simple decay model with optional strengthening.
        """
        pass
        if interacted:
            self.weight = clip01(self.weight + 0.01)
            self.last_interaction_time = self.last_interaction_time + 1
        else:
            self.weight = max(0.0, self.weight - 0.005)
            self.last_interaction_time = self.last_interaction_time + 1


class Person:
    """
    An agent representing a person in the simulation with mask-wearing behavior.

    Attributes include susceptibility, risk perception, thresholds, and social connections.
    """
    def __init__(self, agent_id: int, group_id: int, risk_perception: float, stubborn: bool = False):
        """
        Initialize a Person with base attributes.

        Sets default dynamic states and placeholders for decision variables.
        """
        pass
        self.id = agent_id
        self.group_id = group_id
        self.mask_wearing: bool = False
        self.belief_strength: float = 0.0
        self.adoption_threshold: float = 0.5
        self.susceptibility: float = 1.0
        self.stubbornness: bool = stubborn
        self.risk_perception: float = clip01(risk_perception)
        self.perceived_cost: float = 0.0
        self.trust_in_sources: float = 0.5
        self.degree: int = 0

        # Dynamic states
        self.received_info: bool = False
        self.last_adoption_day: Optional[int] = None
        self.exposures_before_adoption: int = 0
        self.has_ever_adopted: bool = False

    def observe_neighbors(self, peer_mask_rate_by_layer: Dict[str, float]) -> float:
        """
        Observe neighbors' mask wearing rates aggregated across layers.

        Returns a weighted aggregate peer mask rate in [0,1].
        """
        pass
        # Combine layer rates with equal weights if not otherwise specified at call
        if not peer_mask_rate_by_layer:
            return 0.0
        return clip01(sum(peer_mask_rate_by_layer.values()) / len(peer_mask_rate_by_layer))

    def update_beliefs(self, info_signal: float, peer_pressure: float, group_norm: float, noise_std: float = 0.0) -> None:
        """
        Update internal belief strength based on info, peer pressure, and group norm.

        Adds Gaussian noise controlled by noise_std.
        """
        pass
        noise = random.gauss(0.0, noise_std) if noise_std > 0 else 0.0
        delta = 0.4 * info_signal + 0.5 * peer_pressure + 0.3 * group_norm + noise
        self.belief_strength = clip01(self.belief_strength + delta * self.susceptibility)

    def decide_adopt_or_abandon(self,
                                t: int,
                                risk_weight: float,
                                peer_weight: float,
                                peer_rate_agg: float,
                                info_effect: float,
                                has_policy: bool,
                                policy_effect: float,
                                intercept: float,
                                quadratic_peer: float = 0.0,
                                abandonment_prob: float = 0.01,
                                compliance_ceiling: float = 0.95) -> None:
        """
        Decide whether to adopt or abandon mask-wearing based on logistic hazard.

        Uses a sigmoid over risk, peer rate (with optional quadratic), info, and policy signals.
        """
        pass
        if self.stubbornness:
            # Stubborn agents adopt with very low probability unless policy active
            stubborn_bias = -3.0
        else:
            stubborn_bias = 0.0

        peer_term = peer_rate_agg
        if quadratic_peer != 0.0:
            peer_term = peer_term + quadratic_peer * (peer_rate_agg ** 2)

        info_flag = 1.0 if self.received_info else 0.0
        policy_flag = 1.0 if has_policy else 0.0

        linear = (
            intercept
            + risk_weight * self.risk_perception
            + peer_weight * peer_term
            + info_effect * info_flag
            + policy_effect * policy_flag
            + stubborn_bias
            - self.perceived_cost
        )

        p_adopt = min(compliance_ceiling, max(0.0, sigmoid(linear)))
        # Adoption decision
        if not self.mask_wearing:
            if random.random() < p_adopt:
                self.mask_wearing = True
                self.has_ever_adopted = True
                if self.last_adoption_day is None:
                    self.last_adoption_day = t
        else:
            # Abandonment with small probability modulated by low peer pressure
            peer_factor = (1.0 - peer_rate_agg)
            p_abandon = clip01(abandonment_prob * peer_factor)
            if random.random() < p_abandon:
                self.mask_wearing = False

    def share_signal(self, base_share_prob: float) -> bool:
        """
        Decide to share information to neighbors.

        Returns True if sharing occurs.
        """
        pass
        p = clip01(base_share_prob * (0.5 + 0.5 * self.trust_in_sources))
        return random.random() < p

    def rewire_ties(self) -> None:
        """
        Placeholder for tie rewiring at the individual level.

        The actual rewiring is handled by the network-level function; this remains for API completeness.
        """
        pass
        # No-op at the agent level; rewiring is applied in Simulation._rewire_network

    def comply_with_policy(self, enforcement_prob: float) -> None:
        """
        Respond to policy enforcement with higher adoption probability.

        May flip to adoption if not already wearing a mask.
        """
        pass
        if not self.mask_wearing:
            if random.random() < enforcement_prob:
                self.mask_wearing = True
                self.has_ever_adopted = True


class Group:
    """
    Social group capturing norms and policy stringency.

    Provides group-level broadcast and norm updating.
    """
    def __init__(self, group_id: int, member_ids: List[int]):
        """
        Initialize Group with a set of members.

        Sets default cohesion, norm_level, and policy stringency.
        """
        pass
        self.id = group_id
        self.member_ids: List[int] = list(member_ids)
        self.size: int = len(member_ids)
        self.cohesion: float = 0.5
        self.norm_level: float = 0.0
        self.policy_stringency: float = 0.0

    def broadcast_group_norm(self) -> float:
        """
        Provide a broadcast value representing the group's current norm.

        Return the norm_level to influence members.
        """
        pass
        return clip01(self.norm_level)

    def update_norm_from_members(self, agents: Dict[int, Person]) -> None:
        """
        Update group norm based on average adoption among members.

        Sets norm_level to mean mask_wearing of members.
        """
        pass
        if not self.member_ids:
            self.norm_level = 0.0
            return
        adoption = [1.0 if agents[a].mask_wearing else 0.0 for a in self.member_ids if a in agents]
        self.norm_level = statistics.mean(adoption) if adoption else 0.0
        self.size = len(self.member_ids)


class InformationSource:
    """
    External information source broadcasting messages to agents.

    Characterized by credibility, bias, reach, and effect size.
    """
    def __init__(self, src_id: int, credibility: float = 0.7, bias: float = 0.0, reach: float = 0.5,
                 message_type: str = "pro-mask", broadcast_frequency: float = 0.1, effect_size: float = 0.2):
        """
        Initialize an information source with broadcast parameters.

        frequency is the probability to broadcast on a given day; reach is the fraction of agents targeted.
        """
        pass
        self.id = src_id
        self.credibility = clip01(credibility)
        self.bias = bias
        self.reach = clip01(reach)
        self.message_type = message_type
        self.broadcast_frequency = clip01(broadcast_frequency)
        self.effect_size = effect_size

    def broadcast_message(self, t: int, agents: Dict[int, Person]) -> Set[int]:
        """
        Broadcast message to a subset of agents.

        Returns set of agent IDs who receive the message at time t.
        """
        pass
        recipients: Set[int] = set()
        if random.random() > self.broadcast_frequency:
            return recipients
        # Determine number of recipients
        n = len(agents)
        k = int(self.reach * n)
        if k <= 0:
            return recipients
        targets = random.sample(list(agents.keys()), k)
        for aid in targets:
            recipients.add(aid)
        return recipients


class PolicyAuthority:
    """
    Policy authority that issues mandates and enforces compliance.

    Supports policy activation starting from a given day with enforcement.
    """
    def __init__(self, authority_id: int, enforcement_probability: float = 0.3,
                 penalty: float = 0.1, messaging_strength: float = 0.2, communication_frequency: float = 0.1,
                 start_day: int = 10):
        """
        Initialize policy authority with default parameters.

        start_day configures the onset of policy activity.
        """
        pass
        self.id = authority_id
        self.mandate_active: bool = False
        self.enforcement_probability = clip01(enforcement_probability)
        self.penalty = penalty
        self.messaging_strength = messaging_strength
        self.communication_frequency = clip01(communication_frequency)
        self.start_day = start_day

    def issue_or_update_policy(self, t: int) -> None:
        """
        Activate policy mandate when simulation time reaches start_day.

        May remain active thereafter.
        """
        pass
        if t >= self.start_day:
            self.mandate_active = True

    def enforce_policy(self, agents: Dict[int, Person]) -> None:
        """
        Enforce policy by increasing adoption among non-wearers.

        Applies stochastic enforcement across the population.
        """
        pass
        if not self.mandate_active:
            return
        for agent in agents.values():
            if not agent.mask_wearing:
                if random.random() < self.enforcement_probability:
                    agent.comply_with_policy(self.enforcement_probability)

    def broadcast_guidance(self, agents: Dict[int, Person]) -> Set[int]:
        """
        Broadcast policy guidance messages to agents.

        Returns set of agents receiving policy message.
        """
        pass
        recipients: Set[int] = set()
        if not self.mandate_active:
            return recipients
        if random.random() > self.communication_frequency:
            return recipients
        n = len(agents)
        k = max(1, int(0.2 * n))
        targets = random.sample(list(agents.keys()), k)
        for aid in targets:
            recipients.add(aid)
        return recipients


class Simulation:
    """
    Main simulation class coordinating agents, network, and dynamics.

    Loads data, initializes entities, runs the loop, computes metrics, evaluates, visualizes, and saves results.
    """
    def __init__(self, model_plan: Dict[str, Any]):
        """
        Initialize the simulation from the provided model_plan.

        Parses parameters, sets random seed, and prepares placeholders for components.
        """
        pass
        self.model_plan = model_plan
        self.params = model_plan.get("parameters", {})
        self.metrics_spec = model_plan.get("metrics", [])
        self.eval_spec = model_plan.get("evaluation_metrics", {})
        self.prediction_period = model_plan.get("prediction_period", {"start_day": None, "end_day": None})

        # Parameters and defaults
        self.population_size = int(self.params.get("population_size", 500))
        self.simulation_steps = int(self.params.get("simulation_steps", 180))
        self.time_step = int(self.params.get("time_step", 1))
        self.seed = int(self.params.get("seed", 42))
        self.initial_adoption_rate = float(self.params.get("initial_adoption_rate", 0.1))
        self.adoption_target = float(self.params.get("adoption_target", 0.7))
        self.average_degree = int(self.params.get("average_degree", 8))
        self.rewiring_probability = float(self.params.get("rewiring_probability", 0.05))
        self.homophily_strength = float(self.params.get("homophily_strength", 0.3))
        self.tie_weight_mean = float(self.params.get("tie_weight_mean", 1.0))
        self.tie_weight_std = float(self.params.get("tie_weight_std", 0.3))
        self.social_influence_weight = float(self.params.get("social_influence_weight", 0.5))
        self.adoption_threshold_mean = float(self.params.get("adoption_threshold_mean", 0.5))
        self.adoption_threshold_std = float(self.params.get("adoption_threshold_std", 0.15))
        self.stubborn_agent_fraction = float(self.params.get("stubborn_agent_fraction", 0.05))
        self.abandonment_rate = float(self.params.get("abandonment_rate", 0.01))
        self.forgetting_rate = float(self.params.get("forgetting_rate", 0.005))
        self.external_message_frequency = float(self.params.get("external_message_frequency", 0.1))
        self.message_effect_size = float(self.params.get("message_effect_size", 0.2))
        self.policy_enforcement_prob = float(self.params.get("policy_enforcement_prob", 0.3))
        self.policy_effect_size = float(self.params.get("policy_effect_size", 0.25))
        self.observation_window = int(self.params.get("observation_window", 30))
        self.noise_std = float(self.params.get("noise_std", 0.05))
        self.adoption_function = self.params.get("adoption_function", "threshold_logistic")

        # Additional calibrated coefficients (from data_summary/simulation_parameters)
        self.alpha_risk_weight = 2.0
        self.beta_peer_influence = 3.0
        self.delta_info_effect = 1.0
        self.gamma_policy_effect = 1.5
        self.theta_intercept = -2.0
        self.nonlinearity_peer_quadratic = -1.0
        self.compliance_ceiling = 0.95

        # Information diffusion parameters (defaults)
        self.layer_names = ["family", "work_school", "community"]
        self.layer_weights = {"family": 0.5, "work_school": 0.3, "community": 0.2}
        self.info_transmission_per_layer = {"family": 0.3, "work_school": 0.15, "community": 0.05}
        self.exogenous_broadcast_rate = 0.01
        self.info_decay_prob = 0.02

        # Random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Containers
        self.agents: Dict[int, Person] = {}
        self.groups: Dict[int, Group] = {}
        self.info_sources: List[InformationSource] = []
        self.policy = PolicyAuthority(0, enforcement_probability=self.policy_enforcement_prob, start_day=10)

        # Network per layer adjacency
        self.layer_adj: Dict[str, Dict[int, Set[int]]] = {ln: {} for ln in self.layer_names}
        self.ties: Dict[str, Dict[Tuple[int, int], Tie]] = {ln: {} for ln in self.layer_names}
        self.all_nodes: Set[int] = set()

        # Results
        self.time_series: Dict[str, List[float]] = {
            "adoption_rate": [],
            "assortativity_of_adoption": [],
        }
        self.group_time_series: Dict[int, List[float]] = {}
        self.additional_outputs: Dict[str, Any] = {}

        # Tracking for metrics
        self.first_adoption_day: Dict[int, int] = {}
        self.daily_new_adoptions: List[Set[int]] = []
        self.per_agent_exposure_days_before_adoption: Dict[int, int] = {}

        # Load data and initialize
        self._initialize_from_data_or_synthetic()

    def _initialize_from_data_or_synthetic(self) -> None:
        """
        Load agents and network from files if available, otherwise synthetic generation.

        Also initializes initial states, groups, and information sources.
        """
        pass
        agent_df = self._load_agents_dataframe(AGENTS_FILE)
        if agent_df is None:
            # Synthetic agents
            agent_df = self._generate_synthetic_agents(self.population_size)
        else:
            # Ensure correct population size from data
            self.population_size = int(len(agent_df))

        # Network
        loaded_network = self._load_social_network(NETWORK_FILE)
        if loaded_network is None:
            loaded_network = self._generate_synthetic_network(agent_df["agent_id"].tolist(), self.average_degree)

        # Build adjacency and tie objects
        self._build_layer_adjacency(loaded_network)

        # Assign groups by age_group code (or occupation fallback)
        age_map = {aid: grp for aid, grp in zip(agent_df["agent_id"], agent_df["age_group_code"])}
        for grp_id in sorted(set(age_map.values())):
            members = [aid for aid, gid in age_map.items() if gid == grp_id]
            self.groups[grp_id] = Group(grp_id, members)

        # Initialize agents
        stubborn_set = set(random.sample(agent_df["agent_id"].tolist(),
                                        int(self.stubborn_agent_fraction * self.population_size)))
        for idx, row in agent_df.iterrows():
            aid = int(row["agent_id"])
            group_id = int(row["age_group_code"])
            risk = float(row["risk_perception"])
            stubborn = aid in stubborn_set
            person = Person(aid, group_id, risk, stubborn)
            # Thresholds drawn
            person.adoption_threshold = clip01(random.gauss(self.adoption_threshold_mean, self.adoption_threshold_std))
            person.susceptibility = clip01(0.8 + random.random() * 0.4)
            person.trust_in_sources = clip01(0.4 + random.random() * 0.4)
            person.perceived_cost = clip01(max(0.0, random.gauss(0.1, 0.05)))
            # Degree
            degree_all = 0
            for ln in self.layer_names:
                degree_all += len(self.layer_adj[ln].get(aid, set()))
            person.degree = degree_all
            self.agents[aid] = person

        self.all_nodes = set(self.agents.keys())
        # Seed initial adoption, possibly from train data day 0 if available
        train_df = self._load_train_data(TRAIN_FILE)
        if train_df is not None and "day" in train_df.columns and (train_df["day"] == 0).any():
            day0 = train_df[train_df["day"] == 0]
            mask_map = {int(r["agent_id"]): bool(r["wearing_mask"]) for _, r in day0.iterrows()}
            info_map = {int(r["agent_id"]): bool(r.get("received_info", False)) for _, r in day0.iterrows()}
            for aid, agent in self.agents.items():
                agent.mask_wearing = bool(mask_map.get(aid, random.random() < self.initial_adoption_rate))
                agent.received_info = bool(info_map.get(aid, random.random() < 0.15))
        else:
            for agent in self.agents.values():
                agent.mask_wearing = random.random() < self.initial_adoption_rate
                agent.received_info = random.random() < 0.15

        # Initialize info sources
        self.info_sources = [
            InformationSource(1, credibility=0.8, reach=0.4, broadcast_frequency=0.15, effect_size=self.message_effect_size)
        ]

        # Initialize group norms
        for group in self.groups.values():
            group.update_norm_from_members(self.agents)

        # Initialize tracking dicts
        for aid, agent in self.agents.items():
            if agent.mask_wearing:
                agent.has_ever_adopted = True
                self.first_adoption_day[aid] = 0
            self.per_agent_exposure_days_before_adoption[aid] = 0

    def _load_agents_dataframe(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load agent attributes from CSV file if present and valid.

        Returns a DataFrame or None if file is missing or invalid.
        """
        pass
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            # Validate and preprocess
            if "agent_id" not in df.columns or "risk_perception" not in df.columns:
                return None
            # Encode age_group and occupation to integers
            if "age_group" in df.columns:
                df["age_group_code"] = df["age_group"].astype("category").cat.codes
            else:
                df["age_group_code"] = 0
            if "occupation" in df.columns:
                df["occupation_code"] = df["occupation"].astype("category").cat.codes
            else:
                df["occupation_code"] = 0
            # Clip risk_perception
            df["risk_perception"] = df["risk_perception"].clip(0.0, 1.0)
            # Ensure integer id
            df["agent_id"] = df["agent_id"].astype(int)
            return df
        except Exception:
            return None

    def _generate_synthetic_agents(self, n: int) -> pd.DataFrame:
        """
        Generate a synthetic agents DataFrame with demographics and risk perception.

        Returns a DataFrame with agent_id, age_group_code, occupation_code, and risk_perception.
        """
        pass
        age_groups = ["Youth", "Young Adult", "Middle Age", "Senior"]
        occupations = ["Student", "White Collar", "Blue Collar", "Service"]
        age_codes = list(range(len(age_groups)))
        occ_codes = list(range(len(occupations)))
        rows = []
        for i in range(n):
            age = random.choice(age_codes)
            occ = random.choice(occ_codes)
            # Risk perception beta-like distribution
            a, b = (2.0, 3.0) if age in (2, 3) else (1.5, 3.5)
            risk = np.random.beta(a, b)
            rows.append({
                "agent_id": i,
                "age_group_code": age,
                "occupation_code": occ,
                "risk_perception": float(clip01(risk))
            })
        return pd.DataFrame(rows)

    def _load_social_network(self, filepath: str) -> Optional[Dict[str, Dict[int, List[int]]]]:
        """
        Load multiplex social network from JSON if present.

        Returns dictionary mapping layer name -> adjacency dict[user -> neighbors list], or None if unavailable.
        """
        pass
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Data format: top-level dict of user_id -> {layer: [neighbors], ...}
            nodes = set()
            for k in data.keys():
                nodes.add(int(k))
                for ln in ["family", "work_school", "community"]:
                    for nid in data[str(k)].get(ln, []):
                        nodes.add(int(nid))
            # Build layer dicts
            layers = {ln: {int(k): set(map(int, v.get(ln, []))) for k, v in data.items()} for ln in ["family", "work_school", "community"]}
            # Symmetrize
            for ln in self.layer_names:
                # Ensure all nodes present
                for n in nodes:
                    layers[ln].setdefault(n, set())
                for u, nbrs in list(layers[ln].items()):
                    for v in list(nbrs):
                        layers[ln][v].add(u)
            # Convert to list values
            return {ln: {u: list(vs) for u, vs in layers[ln].items()} for ln in self.layer_names}
        except Exception:
            return None

    def _generate_synthetic_network(self, node_ids: List[int], k: int) -> Dict[str, Dict[int, List[int]]]:
        """
        Generate a synthetic multiplex network for layers: family, work_school, community.

        Returns dict[layer][node_id] = list(neighbors)
        """
        n = len(node_ids)
        if n <= 1:
            return {ln: {uid: [] for uid in node_ids} for ln in self.layer_names}

        # Helper to build WS graph with safe parameters
        def watts_strogatz_safe(num_nodes: int, k_layer: int, beta: float) -> nx.Graph:
            k_use = max(2, min(k_layer, num_nodes - 1))
            if k_use % 2 == 1:
                k_use += 1
            if k_use >= num_nodes:
                k_use = num_nodes - 1 if (num_nodes - 1) % 2 == 0 else num_nodes - 2
                k_use = max(2, k_use)
            return nx.watts_strogatz_graph(num_nodes, k_use, beta, seed=self.seed)

        index_to_id = {i: node_ids[i] for i in range(n)}

        # Layer-specific average degrees (sum roughly ~ k)
        k_fam = max(2, k // 4)
        k_work = max(2, k // 3)
        k_comm = max(2, max(2, k - k_fam - k_work))

        graphs: Dict[str, nx.Graph] = {}
        graphs["family"] = watts_strogatz_safe(n, k_fam, beta=0.05)
        graphs["work_school"] = watts_strogatz_safe(n, k_work, beta=0.1)
        graphs["community"] = watts_strogatz_safe(n, k_comm, beta=0.2)

        layer_adj: Dict[str, Dict[int, List[int]]] = {}
        for ln in self.layer_names:
            G = graphs[ln]
            adj: Dict[int, List[int]] = {}
            for u in G.nodes():
                uid = index_to_id[u]
                nbrs = [index_to_id[v] for v in G.neighbors(u)]
                adj[uid] = nbrs
            layer_adj[ln] = adj

        return layer_adj

    def _build_layer_adjacency(self, loaded_network: Dict[str, Dict[int, List[int]]]) -> None:
        """
        Build internal adjacency sets and tie objects from loaded network.
        """
        for ln in self.layer_names:
            self.layer_adj[ln] = {}
            self.ties[ln] = {}
            adj_list = loaded_network.get(ln, {})
            # Initialize sets
            for u, nbrs in adj_list.items():
                self.layer_adj[ln].setdefault(u, set())
                for v in nbrs:
                    if u == v:
                        continue
                    self.layer_adj[ln].setdefault(v, set())
                    self.layer_adj[ln][u].add(v)
                    self.layer_adj[ln][v].add(u)
                    key = (min(u, v), max(u, v))
                    if key not in self.ties[ln]:
                        weight = max(0.0, random.gauss(self.tie_weight_mean, self.tie_weight_std))
                        contact = clip01(random.gauss(0.6 if ln == "family" else 0.4 if ln == "work_school" else 0.2, 0.1))
                        trust = clip01(random.gauss(0.75 if ln == "family" else 0.65, 0.15))
                        homo = clip01(random.random())
                        last_t = random.randint(0, 10)
                        self.ties[ln][key] = Tie(
                            node_u=key[0],
                            node_v=key[1],
                            weight=weight,
                            contact_frequency=contact,
                            trust=trust,
                            homophily_score=homo,
                            last_interaction_time=last_t
                        )

    def _load_train_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load training time-series data if available.
        """
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            # Expected columns include at least: day, agent_id, wearing_mask (optional)
            if "day" not in df.columns or "agent_id" not in df.columns:
                return None
            if "wearing_mask" not in df.columns:
                # If not present, assume missing; create with NaNs/False
                df["wearing_mask"] = False
            return df
        except Exception:
            return None


# Execute main for both direct execution and sandbox wrapper invocation
main()