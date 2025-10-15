def main():
    pass

import os
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Path Handling Instructions - Environment variables
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# FIXED: This iteration aligns entities/parameters with spec, implements adherence dynamics and fatigue, handles null policy dates,
# computes required metrics (masked_interaction_fraction, supply_adequacy, policy_effect_size, final_adoption_rate,
# time_to_50_percent_adoption, average_adherence, sustained_adoption_rate), removes duplicate summarize_metrics,
# adds price elasticity in purchasing, hardens network neighbor mapping, and caps contacts per agent with workplace multiplier.


def clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    """
    Clamp a numeric value between provided low and high bounds.

    Args:
        x (float): Value to clamp.
        low (float): Lower bound.
        high (float): Upper bound.

    Returns:
        float: Clamped value.
    """
    return max(low, min(high, x))
    pass


def safe_normal(mu: float, sigma: float, low: float = None, high: float = None, rng: Optional[np.random.Generator] = None) -> float:
    """
    Draw a sample from a normal distribution and optionally clamp to [low, high].

    Args:
        mu (float): Mean.
        sigma (float): Standard deviation.
        low (float): Optional lower bound.
        high (float): Optional upper bound.
        rng (np.random.Generator): Optional RNG for reproducibility.

    Returns:
        float: Sampled value possibly clamped.
    """
    if rng is None:
        g = np.random.normal(mu, sigma)
    else:
        g = rng.normal(mu, sigma)
    if low is not None or high is not None:
        g = clamp(g, low if low is not None else -np.inf, high if high is not None else np.inf)
    return g
    pass


def triangular(low: float, mode: float, high: float, rng: Optional[np.random.Generator] = None) -> float:
    """
    Sample from a triangular distribution.

    Args:
        low (float): Lower bound.
        mode (float): Mode of the distribution.
        high (float): Upper bound.
        rng (np.random.Generator): Optional RNG.

    Returns:
        float: Sampled value from triangular distribution.
    """
    if rng is None:
        return random.triangular(low, high, mode)
    else:
        u = rng.random()
        c = (mode - low) / (high - low)
        if u < c:
            return low + math.sqrt(u * (high - low) * (mode - low))
        else:
            return high - math.sqrt((1 - u) * (high - low) * (high - mode))
    pass


def poisson(lmbda: float, rng: Optional[np.random.Generator] = None) -> int:
    """
    Sample from a Poisson distribution with rate parameter lambda.

    Args:
        lmbda (float): Rate parameter.
        rng (np.random.Generator): Optional RNG.

    Returns:
        int: Sampled count.
    """
    if lmbda <= 0:
        return 0
    if rng is None:
        return int(np.random.poisson(lmbda))
    return int(rng.poisson(lmbda))
    pass


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists; if not, create it.

    Args:
        path (str): Directory path.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    pass


# Embedded Model Plan with aligned parameters and defaults
MODEL_PLAN: Dict[str, Any] = {
    "title": "Mask Adoption Dynamics Simulation",
    "description": "Simulate how face mask usage is adopted and sustained within a population under social influence, information campaigns, and policy mandates.",
    "simulation_type": "agent_based",
    "entities": [],
    "interactions": [],
    "parameters": {
        # FIXED: Align parameter names to spec-like variants and maintain backward compatibility via normalization.
        "random_seed": 42,
        "population_size": 5000,
        "simulation_days": 180,
        "time_step_days": 1,
        "network_topology": "small_world",
        "average_degree": 10,
        "rewiring_prob": 0.1,
        "homophily": 0.3,

        "initial_adoption_rate": 0.15,
        "social_influence_weight": 0.5,
        "authority_trust_weight": 0.3,
        "risk_signal_level": 0.5,
        "decision_noise": 0.1,

        "peer_influence_weight": 0.4,
        "policy_influence_weight": 0.3,
        "media_influence_weight": 0.2,
        "observation_effect_weight": 0.2,

        "habit_formation_rate": 0.05,
        "abandonment_rate": 0.02,

        # Adherence dynamics
        "adherence_decay_rate": 0.01,

        # Retail pricing and supply
        "mask_cost": 2.0,
        "retailer_count": 50,
        "initial_mask_supply": 10000,
        "restock_rate_per_day": 500,
        "price_markup_mean": 0.25,
        "price_markup_sd": 0.1,
        "price_adjustment_sensitivity": 0.5,
        "price_elasticity_of_demand": -0.8,

        # Information channels
        "information_channel_count": 8,
        "channel_credibility_mean": 0.6, "channel_credibility_sd": 0.15,
        "channel_reach_mean": 0.4, "channel_reach_sd": 0.2,
        "channel_bias_mean": 0.0, "channel_bias_sd": 0.3,
        "info_exposure_mean": 0.5, "info_exposure_sd": 0.2,
        "media_campaign_intensity": 0.3, "misinformation_level": 0.1,

        # Locations
        "locations_count_by_type": {
            "workplaces": 100, "schools": 30, "retail": 80, "transit": 10, "parks": 20
        },
        "location_capacity_mean": 50, "location_capacity_sd": 20,
        "open_hours_start": 8, "open_hours_end": 20,
        "daily_prob_home": 0.5, "daily_prob_work": 0.25,
        "daily_prob_school": 0.1, "daily_prob_retail": 0.1, "daily_prob_other": 0.05,
        "contact_rate_per_location_mean": 5.0, "contact_rate_sd": 2.0,
        "max_contacts_per_day": 20,  # FIXED: Cap per agent contacts
        "workplace_contact_multiplier": 1.2,  # FIXED: Apply multiplier at workplaces

        # Policy defaults with None-safe mandate dates
        "enforcement_level": 0.5,
        "fine_amount": 50.0,
        "messaging_intensity": 0.3,
        "subsidy_rate": 0.0,
        "mandate_start_day": None,  # FIXED: Allow None
        "mandate_end_day": None,    # FIXED: Allow None
        "default_location_enforcement_level": 0.2,
        "location_fine_amount": 0.0,

        # Free mask distribution rate per day
        "free_mask_distribution_rate": 0.0,  # FIXED: Added parameter

        # Policy distribution schedule
        "distribution_interval_days": 14,
        "distribution_quantity_per_capita": 2,
        "targeting_rules_strength": 0.7,

        # Threshold target
        "adoption_threshold_target": 0.7,

        # Social network seed
        "network_seed": 43,
    },
    "modules": [],
    "prediction_period": {"start_day": 91, "end_day": 120},
    "evaluation_metrics": ["adoption_rate_over_time", "time_to_threshold"]
}


@dataclass
class Person:
    """
    Agent representing an individual in the population with demographic, attitudinal, and behavioral attributes.

    Attributes:
        id (int): Unique identifier.
        age (int): Age of the person.
        gender (str): Gender category.
        household_id (int): Household identifier.
        income (float): Annual income.
        socioeconomic_status (int): SES quintile (1-5).
        risk_perception (float): Perceived risk level (0-1).
        trust_in_authorities (float): Trust in authorities (0-1).
        social_influence_susceptibility (float): Susceptibility to peer influence (0-1).
        information_exposure_level (float): Exposure to information channels (0-1).
        mask_attitude (float): Pro-mask attitude (0-1).
        mask_ownership (int): Number of masks owned.
        mask_use_state (int): 1 if wearing/using mask today, else 0.
        compliance_threshold (float): Threshold for adopting mask use (0-1).
        adherence_prob (float): Probability of adhering on a day when intent exists.
        fatigue_level (float): Fatigue [0,1] reducing adherence.
        mask_quality (float): Quality [0,1] placeholder.
        network_neighbors (List[int]): Neighbor IDs in social network.
        daily_schedule_profile (Dict[str, float]): Probabilities for activities.
        location_home_zone (str): Home zone.
        health_status_optional (Optional[str]): Placeholder for health status.
        last_observed_compliance (float): Last observed compliance at a location (lagged).
        prompted_today (bool): Whether the agent was prompted/fined (affects next day).
        workplace_id (Optional[int]): Optional workplace location id for multiplier semantics.
        daily_contacts (int): Counter of contacts generated today (transient).
    """
    id: int
    age: int
    gender: str
    household_id: int
    income: float
    socioeconomic_status: int
    risk_perception: float
    trust_in_authorities: float
    social_influence_susceptibility: float
    information_exposure_level: float
    mask_attitude: float
    mask_ownership: int
    mask_use_state: int
    compliance_threshold: float
    adherence_prob: float = 0.8  # FIXED: Added adherence probability
    fatigue_level: float = 0.0   # FIXED: Added fatigue state
    mask_quality: float = 0.7    # FIXED: Added mask quality placeholder
    network_neighbors: List[int] = field(default_factory=list)
    daily_schedule_profile: Dict[str, float] = field(default_factory=dict)
    location_home_zone: str = "Z1"
    health_status_optional: Optional[str] = None
    last_observed_compliance: float = 0.0
    prompted_today: bool = False
    workplace_id: Optional[int] = None
    daily_contacts: int = 0

    pass


@dataclass
class Household:
    """
    Represents a household unit aggregating members, budget, and mask inventory.

    Attributes:
        id (int): Household ID.
        members (List[int]): List of Person IDs in the household.
        budget (float): Budget available.
        mask_inventory (int): Household shared mask inventory reserve.
        household_norms (float): Pro-mask norm strength (0-1).
        home_location (str): Zone location of home.
    """
    id: int
    members: List[int]
    budget: float
    mask_inventory: int
    household_norms: float
    home_location: str

    pass


@dataclass
class Location:
    """
    Represents a physical location where contacts occur and policies may apply.

    Attributes:
        id (int): Location ID.
        type (str): Type of location (e.g., workplaces, schools, retail, transit, parks).
        capacity (int): Capacity of the location.
        mask_policy (str): 'required' or 'recommended'.
        enforcement_level (float): Enforcement intensity (0-1).
        fine_amount (float): Fine amount for noncompliance.
        open_hours (Tuple[int, int]): Opening hours (start, end).
        zone (str): Zone identifier.
        attendees (List[int]): Transient list of attendees for the day.
        compliance_rate_today (float): Computed compliance rate for today.
    """
    id: int
    type: str
    capacity: int
    mask_policy: str
    enforcement_level: float
    fine_amount: float
    open_hours: Tuple[int, int]
    zone: str

    # Transient daily fields
    attendees: List[int] = field(default_factory=list)
    compliance_rate_today: float = 0.0

    pass


@dataclass
class PolicyCampaign:
    """
    Represents a policy and campaign configuration with mandate status and parameters.

    Attributes:
        id (int): Policy ID.
        mandate_active (bool): Whether mandate is active.
        mandate_start_day (Optional[int]): Start day for mandate or None.
        mandate_end_day (Optional[int]): End day for mandate or None.
        messaging_intensity (float): Messaging intensity (0-1).
        enforcement_level (float): Enforcement intensity (0-1).
        fine_amount (float): Fine amount for noncompliance.
        subsidy_rate (float): Subsidy rate on mask prices (0-1).
        distribution_schedule (Dict[str, Any]): Distribution schedule configuration.
    """
    id: int
    mandate_active: bool
    mandate_start_day: Optional[int]
    mandate_end_day: Optional[int]
    messaging_intensity: float
    enforcement_level: float
    fine_amount: float
    subsidy_rate: float
    distribution_schedule: Dict[str, Any]

    pass


@dataclass
class InformationChannel:
    """
    Represents a media or information channel with credibility, reach, bias, and misinformation properties.

    Attributes:
        id (int): Channel ID.
        channel_type (str): Type such as TV, Radio, SocialMedia, Newspaper, NGO.
        credibility (float): Credibility (0-1).
        reach (float): Reach as fraction of population (0-1).
        bias (float): Ideological bias (-1 to 1).
        misinformation_rate (float): Misinformation rate (0-1).
        targeting_rules (Optional[Dict[str, Any]]): Optional targeting rules.
    """
    id: int
    channel_type: str
    credibility: float
    reach: float
    bias: float
    misinformation_rate: float
    targeting_rules: Optional[Dict[str, Any]] = None

    pass


@dataclass
class Retailer:
    """
    Represents a retailer that sells masks with stock and pricing dynamics.

    Attributes:
        id (int): Retailer ID.
        stock_level (int): Units in stock.
        price (float): Current price per unit.
        restock_rate (int): Daily restock amount.
        location (str): Zone.
    """
    id: int
    stock_level: int
    price: float
    restock_rate: int
    location: str

    pass


class SocialNetworkBuilder:
    """
    Module: SocialNetworkBuilder
    Constructs and maintains the social network (small-world by default) and assigns neighbors to Person agents.

    Methods:
        __init__: Initialize with configuration and RNG.
        build: Build the network and assign neighbors to agents.
    """

    def __init__(self, params: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize the SocialNetworkBuilder with parameters and RNG.

        Args:
            params (Dict[str, Any]): Parameter dictionary.
            rng (np.random.Generator): RNG for reproducibility.
        """
        self.params = params
        self.rng = rng
        pass

    def build(self, agents: List[Person]) -> nx.Graph:
        """
        Build the network according to specified type and assign agent neighbors.

        Args:
            agents (List[Person]): List of Person agents.

        Returns:
            nx.Graph: Constructed graph.
        """
        n = len(agents)
        # FIXED: Use normalized parameter names
        avg_degree = int(self.params.get("avg_degree", self.params.get("average_degree", 8)))
        network_type = self.params.get("social_network_type", self.params.get("network_topology", "small_world"))
        rewiring = float(self.params.get("rewiring_prob", 0.1))
        seed = int(self.params.get("network_seed", 43))

        if n < 2:
            G = nx.Graph()
            for a in agents:
                G.add_node(a.id)
            return G

        if network_type == "small_world":
            k = max(2, avg_degree - (avg_degree % 2))  # ensure even for WS
            G = nx.watts_strogatz_graph(n, k, rewiring, seed=seed)
        elif network_type == "erdos_renyi":
            p = avg_degree / max(1, (n - 1))
            G = nx.erdos_renyi_graph(n, p, seed=seed)
        elif network_type == "scale_free":
            m = max(1, avg_degree // 2)
            G = nx.barabasi_albert_graph(n, m, seed=seed)
        else:
            k = max(2, avg_degree - (avg_degree % 2))
            G = nx.watts_strogatz_graph(n, k, rewiring, seed=seed)

        # Ensure connectedness by regenerating if necessary
        largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)
        attempts = 0
        while largest_cc < 0.95 * n and attempts < 5:
            attempts += 1
            if network_type == "small_world":
                G = nx.watts_strogatz_graph(n, k, rewiring, seed=seed + attempts)
            elif network_type == "erdos_renyi":
                p = avg_degree / max(1, (n - 1))
                G = nx.erdos_renyi_graph(n, p, seed=seed + attempts)
            elif network_type == "scale_free":
                m = max(1, avg_degree // 2)
                G = nx.barabasi_albert_graph(n, m, seed=seed + attempts)
            largest_cc = max((len(c) for c in nx.connected_components(G)), default=0)

        # FIXED: Map graph node indices to agent IDs to avoid ID==index assumption
        idx_to_id = {idx: agent.id for idx, agent in enumerate(agents)}
        for idx, agent in enumerate(agents):
            neighbors_idx = list(G.neighbors(idx))
            neighbors_ids = [idx_to_id[n] for n in neighbors_idx]
            agent.network_neighbors = neighbors_ids

        return G
        pass


class PolicyCampaignModule:
    """
    Module: PolicyCampaignModule
    Applies mandates, enforcement, subsidies, and periodic mask distribution. Produces a daily policy pressure signal.

    Methods:
        __init__: Initialize module with policy campaign object and registries.
        update: Update policy state for the current day and emit signals.
    """

    def __init__(self, policy: PolicyCampaign, locations: List[Location], agents: List[Person], households: List[Household], params: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize the module.

        Args:
            policy (PolicyCampaign): Policy campaign object.
            locations (List[Location]): Locations registry.
            agents (List[Person]): Agents registry.
            households (List[Household]): Households registry.
            params (Dict[str, Any]): Parameter dictionary.
            rng (np.random.Generator): RNG.
        """
        self.policy = policy
        self.locations = locations
        self.agents = agents
        self.households = households
        self.params = params
        self.rng = rng

        # FIXED: Build person_id -> household mapping for O(1) use in distributions.
        self.person_to_household: Dict[int, Household] = {}
        for hh in households:
            for pid in hh.members:
                self.person_to_household[pid] = hh
        pass

    def update(self, day: int, signals: Dict[str, Any]) -> None:
        """
        Update policy state for the current day and emit relevant signals.

        Args:
            day (int): Current simulation day.
            signals (Dict[str, Any]): Signals dictionary to populate.

        Returns:
            None
        """
        # FIXED: Handle None for mandate dates safely
        start = self.policy.mandate_start_day
        end = self.policy.mandate_end_day
        mandate_active = bool((start is not None) and (end is not None) and (start <= day <= end))
        self.policy.mandate_active = mandate_active

        enforcement_level = float(self.params.get("enforcement_level", 0.5))
        fine_amount = float(self.params.get("fine_amount", 50.0))
        default_enforcement = float(self.params.get("default_location_enforcement_level", 0.2))
        default_fine = float(self.params.get("location_fine_amount", 0.0))

        # Update locations
        if mandate_active:
            for L in self.locations:
                L.mask_policy = "required"
                L.enforcement_level = enforcement_level
                L.fine_amount = fine_amount
        else:
            for L in self.locations:
                L.mask_policy = "recommended"
                L.enforcement_level = default_enforcement
                L.fine_amount = default_fine

        # Subsidy broadcast
        subsidy_rate = float(self.params.get("subsidy_rate", 0.0))
        self.policy.subsidy_rate = subsidy_rate

        # Distribution events
        distribution_events = []
        interval_days = int(self.params.get("distribution_interval_days", 14))
        qty_per_capita = int(self.params.get("distribution_quantity_per_capita", 2))
        targeting_strength = float(self.params.get("targeting_rules_strength", 0.7))

        if (interval_days > 0) and (day % interval_days == 0):
            # Target: prioritize low SES and high info exposure combination
            scored_agents = [(a.id, (1.0 / max(1, a.socioeconomic_status)) * (0.5 + 0.5 * a.information_exposure_level)) for a in self.agents]
            scored_agents.sort(key=lambda x: x[1], reverse=True)
            top_k = int(len(scored_agents) * min(1.0, max(0.1, targeting_strength)))
            target_ids = set([aid for (aid, _) in scored_agents[:top_k]])
            for a in self.agents:
                if a.id in target_ids:
                    a.mask_ownership += qty_per_capita
                    distribution_events.append({"agent_id": a.id, "qty": qty_per_capita})
                    hh = self.person_to_household.get(a.id, None)
                    if hh is not None:
                        hh.mask_inventory += qty_per_capita

        # FIXED: Free mask distribution rate support (fraction of population gets 1 free mask per day)
        free_rate = float(self.params.get("free_mask_distribution_rate", 0.0))
        if free_rate > 0.0:
            give_n = int(round(free_rate * len(self.agents)))
            if give_n > 0:
                sample_ids = self.rng.choice([a.id for a in self.agents], size=min(give_n, len(self.agents)), replace=False)
                for pid in sample_ids:
                    agent = next((x for x in self.agents if x.id == pid), None)
                    if agent is not None:
                        agent.mask_ownership += 1
                        hh = self.person_to_household.get(pid, None)
                        if hh is not None:
                            hh.mask_inventory += 0  # no HH transfer for single use, placeholder

        # Emit signals
        signals.setdefault("policy", {})
        signals["policy"]["state"] = {
            "mandate_active": mandate_active,
            "day": day,
            "enforcement_level": enforcement_level if mandate_active else default_enforcement,
            "fine_amount": fine_amount if mandate_active else default_fine,
            "messaging_intensity": float(self.params.get("messaging_intensity", 0.3)),
        }
        signals["policy"]["enforcement_pressure"] = enforcement_level if mandate_active else 0.0
        signals["policy"]["subsidy"] = subsidy_rate
        signals["policy"]["distribution_events"] = distribution_events
        pass


class InformationChannelModule:
    """
    Module: InformationChannelModule
    Simulates media and information flow, producing daily media influence signals considering credibility, reach, and misinformation.

    Methods:
        __init__: Initialize with channels and parameters.
        update: Compute media influence per person and summary intensity.
    """

    def __init__(self, channels: List[InformationChannel], agents: List[Person], params: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize the information module.

        Args:
            channels (List[InformationChannel]): Registered channels.
            agents (List[Person]): Agents registry.
            params (Dict[str, Any]): Parameters dict.
            rng (np.random.Generator): RNG.
        """
        self.channels = channels
        self.agents = agents
        self.params = params
        self.rng = rng
        pass

    def update(self, day: int, signals: Dict[str, Any]) -> None:
        """
        Compute media influence for the day and populate signals.

        Args:
            day (int): Simulation day.
            signals (Dict[str, Any]): Signals dictionary.

        Returns:
            None
        """
        messaging_intensity = float(self.params.get("messaging_intensity", 0.3))
        media_campaign_intensity = float(self.params.get("media_campaign_intensity", 0.3))
        misinformation_level = float(self.params.get("misinformation_level", 0.1))
        media_influence_weight = float(self.params.get("media_influence_weight", 0.2))
        info_exposure_mean = float(self.params.get("info_exposure_mean", 0.5))
        info_exposure_sd = float(self.params.get("info_exposure_sd", 0.2))

        # Combine channels
        media_signal = 0.0
        for ch in self.channels:
            effective_message = messaging_intensity * ch.credibility * ch.reach * (1 - ch.misinformation_rate) + (media_campaign_intensity * ch.credibility * ch.reach)
            effective_bias = ch.bias * ch.reach
            media_signal += effective_message - (misinformation_level * effective_bias)
        if len(self.channels) > 0:
            media_signal = media_signal / len(self.channels)

        # Per-person influence
        influence_by_person = {}
        for p in self.agents:
            exposure_factor = clamp(safe_normal(info_exposure_mean, info_exposure_sd, 0.0, 1.0, self.rng), 0.0, 1.0)
            exposure = exposure_factor * clamp(p.information_exposure_level, 0.0, 1.0)
            person_media_influence = media_influence_weight * exposure * media_signal
            influence_by_person[p.id] = person_media_influence

        signals.setdefault("media", {})
        signals["media"]["influence_by_person"] = influence_by_person
        signals["media"]["summary_intensity"] = media_signal
        pass


class MobilityAndLocationModule:
    """
    Module: MobilityAndLocationModule
    Generates daily schedules, assigns agents to locations, simulates contacts, and computes location-level compliance and enforcement actions.

    Methods:
        __init__: Initialize with registries and parameters.
        update: Assign activities, compute compliance, enforce policies, and generate contacts.
    """

    def __init__(self, locations: List[Location], agents: List[Person], households: List[Household], params: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize the mobility module.

        Args:
            locations (List[Location]): Registered locations.
            agents (List[Person]): Agents registry.
            households (List[Household]): Households registry for enforcement/budget updates.
            params (Dict[str, Any]): Parameters dict.
            rng (np.random.Generator): RNG.
        """
        self.locations = locations
        self.agents = agents
        self.households = households
        self.params = params
        self.rng = rng

        # Index locations by type and zone
        self.locations_by_type: Dict[str, List[Location]] = {}
        for loc in self.locations:
            self.locations_by_type.setdefault(loc.type, []).append(loc)

        # Build id maps for O(1) lookup
        self.id_to_agent: Dict[int, Person] = {a.id: a for a in self.agents}
        self.person_to_household: Dict[int, Household] = {}
        for hh in households:
            for pid in hh.members:
                self.person_to_household[pid] = hh
        pass

    def _draw_activity(self, profile: Dict[str, float]) -> str:
        """
        Draw an activity based on a daily schedule probability profile.

        Args:
            profile (Dict[str, float]): Activity probabilities.

        Returns:
            str: Selected activity key.
        """
        activities = list(profile.keys())
        probs = np.array([profile[a] for a in activities], dtype=float)
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()
        idx = self.rng.choice(len(activities), p=probs)
        return activities[idx]
        pass

    def _choose_location(self, activity: str, zone: str) -> Optional[Location]:
        """
        Choose a location belonging to the given activity type.

        Args:
            activity (str): Activity key mapping to location type.
            zone (str): Person's home zone.

        Returns:
            Optional[Location]: A location for the activity, or None if not applicable.
        """
        type_map = {
            "home": None, "work": "workplaces", "school": "schools", "retail": "retail", "other": "parks"
        }
        loc_type = type_map.get(activity, None)
        if loc_type is None:
            return None
        candidates = self.locations_by_type.get(loc_type, [])
        if not candidates:
            return None
        # Prefer same zone if available
        same_zone = [L for L in candidates if L.zone == zone]
        pool = same_zone if same_zone else candidates
        L = pool[self.rng.integers(0, len(pool))]
        return L
        pass

    def update(self, day: int, signals: Dict[str, Any]) -> None:
        """
        Assign agents to locations based on activities, compute location compliance, enforce policies, and generate contacts.

        Args:
            day (int): Current day.
            signals (Dict[str, Any]): Signals dict to populate.

        Returns:
            None
        """
        # Reset attendees and daily_contacts counters
        for L in self.locations:
            L.attendees = []
        for a in self.agents:
            a.daily_contacts = 0

        # Assign activities and locations
        for p in self.agents:
            profile = p.daily_schedule_profile if p.daily_schedule_profile else {
                "home": float(self.params.get("daily_prob_home", 0.5)),
                "work": float(self.params.get("daily_prob_work", 0.25)),
                "school": float(self.params.get("daily_prob_school", 0.1)),
                "retail": float(self.params.get("daily_prob_retail", 0.1)),
                "other": float(self.params.get("daily_prob_other", 0.05)),
            }
            activity = self._draw_activity(profile)
            L = self._choose_location(activity, p.location_home_zone)
            if L is not None:
                if len(L.attendees) < L.capacity:
                    L.attendees.append(p.id)

        # Compute compliance per location using today's mask_use_state with O(1) lookup
        location_compliance = {}
        for L in self.locations:
            if not L.attendees:
                L.compliance_rate_today = 0.0
            else:
                wearing = sum(1 for pid in L.attendees if (self.id_to_agent.get(pid) is not None and self.id_to_agent[pid].mask_use_state == 1))
                L.compliance_rate_today = wearing / max(1, len(L.attendees))
            location_compliance[L.id] = L.compliance_rate_today

        # Enforcement with fines deducted from household budgets and prompt flag for next day
        enforcement_events = []
        for L in self.locations:
            if L.mask_policy == "required" and L.attendees:
                for pid in L.attendees:
                    person = self.id_to_agent.get(pid)
                    if person is None or person.mask_use_state == 1:
                        continue
                    prob_prompt = clamp(L.enforcement_level * float(self.params.get("compliance_sensitivity", 0.6)), 0.0, 1.0)
                    if self.rng.random() < prob_prompt:
                        evt = {"day": day, "location_id": L.id, "agent_id": pid, "event": "prompt_or_fine"}
                        # Apply fine with probability equal to enforcement level
                        if self.rng.random() < L.enforcement_level and L.fine_amount > 0:
                            hh = self.person_to_household.get(pid, None)
                            if hh is not None:
                                old_budget = hh.budget
                                hh.budget = max(0.0, hh.budget - L.fine_amount)
                                evt["fine"] = L.fine_amount
                                evt["budget_before"] = old_budget
                                evt["budget_after"] = hh.budget
                        enforcement_events.append(evt)
                        person.prompted_today = True

        # Contacts: sample random pairs in each location with per-agent cap and workplace multiplier
        contact_rate_mean = float(self.params.get("contact_rate_per_location_mean", 5.0))
        contact_rate_sd = float(self.params.get("contact_rate_sd", 2.0))
        max_contacts = int(self.params.get("max_contacts_per_day", 20))  # FIXED: Cap contacts per agent
        work_mult = float(self.params.get("workplace_contact_multiplier", 1.0))  # FIXED: Multiplier
        contacts = []
        contacts_per_agent: Dict[int, int] = {}

        for L in self.locations:
            if not L.attendees:
                continue
            base_rate = max(0.0, contact_rate_mean + safe_normal(0.0, contact_rate_sd, rng=self.rng))
            if L.type == "workplaces":
                base_rate *= work_mult  # FIXED: Apply workplace multiplier
            expected_contacts = poisson(base_rate, rng=self.rng)
            att = L.attendees.copy()
            if len(att) >= 2 and expected_contacts > 0:
                for _ in range(expected_contacts):
                    # Attempt to form a contact respecting per-agent cap
                    for _attempt in range(5):
                        i, j = self.rng.choice(att), self.rng.choice(att)
                        if i == j:
                            continue
                        ci = contacts_per_agent.get(i, 0)
                        cj = contacts_per_agent.get(j, 0)
                        if ci >= max_contacts or cj >= max_contacts:
                            continue
                        contacts.append({"a": i, "b": j, "location_id": L.id})
                        contacts_per_agent[i] = ci + 1
                        contacts_per_agent[j] = cj + 1
                        break

        # Emit signals
        signals.setdefault("location", {})
        signals["location"]["compliance_rates"] = location_compliance
        signals.setdefault("enforcement", {})
        signals["enforcement"]["events"] = enforcement_events
        signals["contacts"] = contacts

        # Update last_observed_compliance for next-day behavior (lagged signal)
        person_loc_map = {}
        for L in self.locations:
            for pid in L.attendees:
                person_loc_map[pid] = L.id
        for p in self.agents:
            loc_id = person_loc_map.get(p.id, None)
            if loc_id is not None:
                p.last_observed_compliance = location_compliance.get(loc_id, 0.0)
        pass


class RetailerSupplyModule:
    """
    Module: RetailerSupplyModule
    Manages retailer stocks, pricing, subsidies, and sales to agents and households.

    Methods:
        __init__: Initialize with retailers and parameters.
        update: Restock, adjust prices, and process purchase intents.
    """

    def __init__(self, retailers: List[Retailer], agents: List[Person], households: List[Household], params: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize the retailer supply module.

        Args:
            retailers (List[Retailer]): Retailer registry.
            agents (List[Person]): Agents registry.
            households (List[Household]): Households registry.
            params (Dict[str, Any]): Parameters.
            rng (np.random.Generator): RNG.
        """
        self.retailers = retailers
        self.agents = agents
        self.households = households
        self.params = params
        self.rng = rng

        # Build maps for O(1) household lookup
        self.id_to_agent: Dict[int, Person] = {a.id: a for a in self.agents}
        self.person_to_household: Dict[int, Household] = {}
        for hh in households:
            for pid in hh.members:
                self.person_to_household[pid] = hh
        pass

    def update(self, day: int, signals: Dict[str, Any]) -> None:
        """
        Update retailer stock, price, and process purchase intents.

        Args:
            day (int): Current day.
            signals (Dict[str, Any]): Signals dictionary with purchase intents.

        Returns:
            None
        """
        mask_cost = float(self.params.get("mask_cost", 2.0))
        restock_rate_per_day = int(self.params.get("restock_rate_per_day", 500))
        price_markup_mean = float(self.params.get("price_markup_mean", 0.25))
        price_markup_sd = float(self.params.get("price_markup_sd", 0.1))
        price_adjustment_sensitivity = float(self.params.get("price_adjustment_sensitivity", 0.5))
        subsidy_rate = float(signals.get("policy", {}).get("subsidy", self.params.get("subsidy_rate", 0.0)))

        initial_mask_supply = int(self.params.get("initial_mask_supply", 10000))
        retailer_count = max(1, int(self.params.get("retailer_count", 50)))
        baseline_stock = initial_mask_supply / retailer_count

        # Restock and price adjust
        for r in self.retailers:
            r.stock_level = max(0, r.stock_level + restock_rate_per_day)
            desired_price = mask_cost * (1.0 + safe_normal(price_markup_mean, price_markup_sd, rng=self.rng))
            scarcity_term = price_adjustment_sensitivity * (1.0 - (r.stock_level / max(1.0, baseline_stock)))
            r.price = max(0.2, desired_price + scarcity_term - subsidy_rate * mask_cost)

        # Process purchase intents
        intents = signals.get("person", {}).get("purchase_intents", [])
        transactions = []
        for intent in intents:
            agent_id = intent.get("agent_id")
            qty = int(intent.get("qty", 0))
            agent = self.id_to_agent.get(agent_id, None)
            if agent is None or qty <= 0:
                continue
            # Choose retailer in agent zone or random
            zone = agent.location_home_zone
            candidates = [r for r in self.retailers if r.location == zone and r.stock_level > 0]
            if not candidates:
                candidates = [r for r in self.retailers if r.stock_level > 0]
            if not candidates:
                continue
            rsel = candidates[self.rng.integers(0, len(candidates))]
            units = min(qty, rsel.stock_level)
            household = self.person_to_household.get(agent_id, None)
            budget = household.budget if household else 0.0
            cost = rsel.price * units
            if units > 0 and budget >= cost:
                # Execute
                rsel.stock_level -= units
                agent.mask_ownership += units
                if household:
                    household.budget -= cost
                    household.mask_inventory += max(0, units // 2)  # share some into household reserve
                transactions.append({"day": day, "agent_id": agent_id, "retailer_id": rsel.id, "qty": units, "price": rsel.price, "zone": rsel.location})

        # Emit signals for retail
        signals.setdefault("retail", {})
        signals["retail"]["transactions"] = transactions
        # Average price by zone
        price_by_zone: Dict[str, float] = {}
        zones = set([r.location for r in self.retailers])
        for z in zones:
            rz = [r.price for r in self.retailers if r.location == z]
            price_by_zone[z] = float(np.mean(rz)) if rz else float(np.nan)
        signals["retail"]["price_by_zone"] = price_by_zone
        signals["retail"]["stock_levels"] = {r.id: r.stock_level for r in self.retailers}
        pass


class BehaviorDecisionModule:
    """
    Module: BehaviorDecisionModule
    Updates attitudes via peer/media/policy/observation, decides mask-wearing, and generates purchase intents.

    Methods:
        __init__: Initialize with agents, network, households, and parameters.
        set_prev_avg_price: Set previous day's average price.
        update: Perform behavior updates, adoption decisions with adherence, and generate purchase intents with price elasticity.
    """

    def __init__(self, agents: List[Person], households: List[Household], G: nx.Graph, params: Dict[str, Any], rng: np.random.Generator):
        """
        Initialize the behavior module.

        Args:
            agents (List[Person]): Agents registry.
            households (List[Household]): Households registry.
            G (nx.Graph): Social network graph.
            params (Dict[str, Any]): Parameters dict.
            rng (np.random.Generator): RNG.
        """
        self.agents = agents
        self.households = households
        self.G = G
        self.params = params
        self.rng = rng
        self.prev_avg_price: Optional[float] = None

        # Build O(1) maps
        self.id_to_agent: Dict[int, Person] = {a.id: a for a in self.agents}
        self.person_to_household: Dict[int, Household] = {}
        for hh in households:
            for pid in hh.members:
                self.person_to_household[pid] = hh
        pass

    def set_prev_avg_price(self, value: float) -> None:
        """
        Set previous day's average price.

        Args:
            value (float): Average price value.

        Returns:
            None
        """
        self.prev_avg_price = value
        pass

    def update(self, day: int, signals: Dict[str, Any]) -> None:
        """
        Update behavior state: attitudes and mask use decisions; produce purchase intents.

        Args:
            day (int): Current day.
            signals (Dict[str, Any]): Signals dict to read inputs and write outputs.

        Returns:
            None
        """
        peer_influence_weight = float(self.params.get("peer_influence_weight", 0.4))
        policy_influence_weight = float(self.params.get("policy_influence_weight", 0.3))
        observation_effect_weight = float(self.params.get("observation_effect_weight", 0.2))
        media_influence_weight = float(self.params.get("media_influence_weight", 0.2))
        habit_formation_rate = float(self.params.get("habit_formation_rate", 0.05))
        abandonment_rate = float(self.params.get("abandonment_rate", 0.02))
        compliance_threshold_mean = float(self.params.get("compliance_threshold_mean", 0.5))
        compliance_threshold_sd = float(self.params.get("compliance_threshold_sd", 0.15))
        adoption_noise = float(self.params.get("adoption_noise", 0.05))
        mask_discomfort_cost_mean = float(self.params.get("mask_discomfort_cost_mean", 0.2))
        mask_discomfort_cost_sd = float(self.params.get("mask_discomfort_cost_sd", 0.1))
        risk_perception_weight = float(self.params.get("risk_perception_weight", 0.6))
        trust_weight = float(self.params.get("trust_weight", 0.3))
        wealth_weight_on_purchase = float(self.params.get("wealth_weight_on_purchase", 0.2))
        mask_cost = float(self.params.get("mask_cost", 2.0))
        price_markup_mean = float(self.params.get("price_markup_mean", 0.25))
        price_elasticity = float(self.params.get("price_elasticity_of_demand", -0.8))  # FIXED: Price elasticity used

        fatigue_decay_rate = float(self.params.get("adherence_decay_rate", 0.01))  # FIXED: Adherence decay

        enforcement_pressure = float(signals.get("policy", {}).get("enforcement_pressure", 0.0))
        media_influence_by_person = signals.get("media", {}).get("influence_by_person", {})

        # Household norms map for quick access
        hh_norms = {}
        hh_budget = {}
        for h in self.households:
            for pid in h.members:
                hh_norms[pid] = h.household_norms
                hh_budget[pid] = h.budget

        purchase_intents = []
        adoption_events = []

        expected_retail_price = self.prev_avg_price
        if expected_retail_price is None or (isinstance(expected_retail_price, float) and np.isnan(expected_retail_price)):
            expected_retail_price = mask_cost * (1.0 + price_markup_mean)

        id_to_agent = self.id_to_agent

        for p in self.agents:
            # Calculate peer compliance
            if p.network_neighbors:
                peer_states = [id_to_agent[nid].mask_use_state for nid in p.network_neighbors if nid in id_to_agent]
                peer_compliance = float(np.mean(peer_states)) if peer_states else 0.0
            else:
                peer_compliance = 0.0
            # Observation signal (lagged from previous day)
            observation_signal = float(p.last_observed_compliance) if hasattr(p, "last_observed_compliance") else 0.0

            # Attitude dynamics
            delta_attitude = 0.0
            delta_attitude += peer_influence_weight * p.social_influence_susceptibility * (peer_compliance - p.mask_attitude)
            delta_attitude += media_influence_weight * media_influence_by_person.get(p.id, 0.0)
            delta_attitude += policy_influence_weight * enforcement_pressure
            delta_attitude += observation_effect_weight * (observation_signal - p.mask_attitude)
            # Household norm influence
            delta_attitude += 0.1 * (hh_norms.get(p.id, 0.5) - p.mask_attitude)

            p.mask_attitude = clamp(p.mask_attitude + delta_attitude, 0.0, 1.0)

            # Habit formation vs abandonment
            if p.mask_use_state == 1:
                p.mask_attitude = clamp(p.mask_attitude + habit_formation_rate * (1.0 - p.mask_attitude), 0.0, 1.0)
            else:
                p.mask_attitude = clamp(p.mask_attitude - abandonment_rate * p.mask_attitude, 0.0, 1.0)

            # Utility-based decision
            utility = 0.0
            utility += p.mask_attitude * risk_perception_weight * clamp(p.risk_perception, 0.0, 1.0)
            utility += trust_weight * clamp(p.trust_in_authorities, 0.0, 1.0)
            utility += peer_influence_weight * peer_compliance
            utility += policy_influence_weight * enforcement_pressure
            utility += observation_effect_weight * observation_signal
            if getattr(p, "prompted_today", False):
                utility += 0.1

            # Cost
            cost = clamp(safe_normal(mask_discomfort_cost_mean, mask_discomfort_cost_sd, 0.0, 1.0, self.rng), 0.0, 1.0)
            affordability = min(1.0, (p.income / 365.0) / max(0.1, expected_retail_price))
            decision_score = utility - cost + wealth_weight_on_purchase * affordability + adoption_noise * self.rng.normal()

            threshold = clamp(p.compliance_threshold + self.rng.normal(compliance_threshold_mean - 0.5, compliance_threshold_sd), 0.0, 1.0)

            # Gate wearing on inventory; attempt household allocation; otherwise purchase intent for future
            can_wear = p.mask_ownership > 0
            if not can_wear:
                hh = self.person_to_household.get(p.id, None)
                if hh is not None and hh.mask_inventory > 0:
                    hh.mask_inventory -= 1
                    p.mask_ownership += 1
                    can_wear = True

            will_wear_intent = (decision_score >= threshold) and can_wear

            # FIXED: Adherence with fatigue dynamics (Bernoulli on adherence probability)
            adherence_effective = clamp(p.adherence_prob * (1.0 - p.fatigue_level), 0.0, 1.0)
            wore_today = 1 if (will_wear_intent and (self.rng.random() < adherence_effective)) else 0

            # Fixed fatigue update dynamics
            # Increase fatigue slightly when intending to wear; decay otherwise
            if will_wear_intent:
                p.fatigue_level = clamp(p.fatigue_level + fatigue_decay_rate * 0.5, 0.0, 1.0)
            else:
                p.fatigue_level = clamp(p.fatigue_level - fatigue_decay_rate * 0.5, 0.0, 1.0)
            # Optional habit recovery when actually wearing
            if wore_today:
                p.adherence_prob = clamp(p.adherence_prob + 0.05 * (1 - p.adherence_prob), 0.0, 1.0)

            if (decision_score >= threshold) and not can_wear:
                # FIXED: Use price elasticity to set purchase quantity
                base_qty = 5
                price_ratio = max(0.1, expected_retail_price / max(0.1, mask_cost))
                qty_adj = base_qty * (price_ratio ** price_elasticity)
                qty = int(max(0, round(qty_adj)))
                if qty > 0:
                    purchase_intents.append({"agent_id": p.id, "qty": qty})

            new_state = wore_today
            if new_state != p.mask_use_state:
                adoption_events.append({"day": day, "agent_id": p.id, "from": p.mask_use_state, "to": new_state})
                p.mask_use_state = new_state

            # Clear consumed prompt flag after decision (mobility sets for next day)
            p.prompted_today = False

        signals.setdefault("person", {})
        signals["person"]["mask_use_state"] = {p.id: p.mask_use_state for p in self.agents}
        signals["person"]["adoption_events"] = adoption_events
        signals["person"]["purchase_intents"] = purchase_intents
        pass


class AdoptionAggregator:
    """
    Module: AdoptionAggregator
    Computes daily observables and summary statistics including adoption rate, churn, disparity, location compliance, prices,
    masked interaction fraction, supply adequacy, and adherence.

    Methods:
        __init__: Initialize with registries and parameters.
        update: Aggregate observables for the current day and store time series.
        to_dataframe: Convert records to DataFrame.
        summarize_metrics: Compute required summary metrics.
    """

    def __init__(self, agents: List[Person], households: List[Household], locations: List[Location], retailers: List[Retailer], params: Dict[str, Any]):
        """
        Initialize the aggregator.

        Args:
            agents (List[Person]): Agents registry.
            households (List[Household]): Households registry.
            locations (List[Location]): Locations registry.
            retailers (List[Retailer]): Retailers registry.
            params (Dict[str, Any]): Parameter dict.
        """
        self.agents = agents
        self.households = households
        self.locations = locations
        self.retailers = retailers
        self.params = params

        self.daily_records: List[Dict[str, Any]] = []
        pass

    def update(self, day: int, signals: Dict[str, Any]) -> None:
        """
        Compute daily metrics and append to internal records.

        Args:
            day (int): Current day.
            signals (Dict[str, Any]): Signals dict.

        Returns:
            None
        """
        mask_use_map = signals.get("person", {}).get("mask_use_state", {})
        adoption_rate = float(np.mean(list(mask_use_map.values()))) if mask_use_map else 0.0

        adoption_events = signals.get("person", {}).get("adoption_events", [])
        population_size = len(self.agents)
        churn_rate = 0.0
        if population_size > 0:
            churn_events = [e for e in adoption_events if e.get("from") == 1 and e.get("to") == 0]
            churn_rate = len(churn_events) / population_size

        # Disparity index between top SES quintile (5) and bottom (1)
        ses_top = [mask_use_map.get(a.id, a.mask_use_state) for a in self.agents if a.socioeconomic_status == 5]
        ses_bottom = [mask_use_map.get(a.id, a.mask_use_state) for a in self.agents if a.socioeconomic_status == 1]
        adopt_high = float(np.mean(ses_top)) if ses_top else 0.0
        adopt_low = float(np.mean(ses_bottom)) if ses_bottom else 0.0
        adoption_disparity_index = adopt_high - adopt_low

        # Location compliance rate
        location_compliance_rates = signals.get("location", {}).get("compliance_rates", {})
        location_compliance_rate = float(np.mean(list(location_compliance_rates.values()))) if location_compliance_rates else 0.0

        # Average retailer price
        avg_price = float(np.mean([r.price for r in self.retailers])) if self.retailers else float("nan")

        # FIXED: Compute masked interaction fraction
        contacts = signals.get("contacts", [])
        mask_map = mask_use_map
        masked_contacts = 0
        for c in contacts:
            ai, bi = c.get("a"), c.get("b")
            if mask_map.get(ai, 0) == 1 or mask_map.get(bi, 0) == 1:
                masked_contacts += 1
        masked_interaction_fraction = (masked_contacts / len(contacts)) if contacts else 0.0

        # FIXED: Supply adequacy metric per day from retailer stock > 0
        stock_levels = signals.get("retail", {}).get("stock_levels", {})
        daily_supply_fraction = float(np.mean([1.0 if v > 0 else 0.0 for v in stock_levels.values()])) if stock_levels else 0.0

        # FIXED: Average adherence among adopters (those wearing today)
        adopters = [a for a in self.agents if mask_map.get(a.id, a.mask_use_state) == 1]
        avg_adherence = float(np.mean([getattr(a, "adherence_prob", 0.0) for a in adopters])) if adopters else 0.0

        rec = {
            "day": day,
            "adoption_rate": adoption_rate,
            "churn_rate": churn_rate,
            "adoption_disparity_index": adoption_disparity_index,
            "location_compliance_rate": location_compliance_rate,
            "avg_price": avg_price,
            "masked_interaction_fraction": masked_interaction_fraction,  # FIXED
            "daily_supply_fraction": daily_supply_fraction,  # FIXED
            "average_adherence": avg_adherence,  # FIXED
        }
        self.daily_records.append(rec)
        pass

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert daily records to a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame of daily observables.
        """
        df = pd.DataFrame(self.daily_records)
        return df
        pass

    def summarize_metrics(self) -> Dict[str, Any]:
        """
        Compute summary metrics required by spec:
        - peak_adoption, final_adoption_rate, time_to_50_percent_adoption, sustained_adoption_rate,
          policy_effect_size (14-day diff), supply_adequacy (mean), average_adherence (mean).
        Handles None for mandate dates.

        Returns:
            Dict[str, Any]: Summary metrics dictionary.
        """
        df = self.to_dataframe()
        out: Dict[str, Any] = {}
        if df.empty:
            return {
                "peak_adoption": 0.0,
                "final_adoption_rate": 0.0,
                "time_to_50_percent_adoption": None,
                "sustained_adoption_rate": None,
                "policy_effect_size": None,
                "supply_adequacy": None,
                "average_adherence": None
            }
        out["peak_adoption"] = float(df["adoption_rate"].max())
        out["final_adoption_rate"] = float(df["adoption_rate"].iloc[-1])
        over50 = df[df["adoption_rate"] >= 0.5]
        out["time_to_50_percent_adoption"] = int(over50["day"].iloc[0]) if not over50.empty else None

        # Sustained adoption after mandate end (if defined)
        mandate_end = self.params.get("mandate_end_day", None)
        if mandate_end is not None:
            try:
                mend = int(mandate_end)
                post = df[df["day"] > mend]
                out["sustained_adoption_rate"] = float(post["adoption_rate"].mean()) if not post.empty else None
            except Exception:
                out["sustained_adoption_rate"] = None
        else:
            out["sustained_adoption_rate"] = None

        # Policy effect size: 14-day before vs 14-day after start (if defined)
        start = self.params.get("mandate_start_day", None)
        if start is not None:
            try:
                s = int(start)
                pre = df[(df["day"] >= max(0, s - 14)) & (df["day"] < s)]
                post = df[(df["day"] > s) & (df["day"] <= s + 14)]
                out["policy_effect_size"] = float(post["adoption_rate"].mean() - pre["adoption_rate"].mean()) if (not pre.empty and not post.empty) else None
            except Exception:
                out["policy_effect_size"] = None
        else:
            out["policy_effect_size"] = None

        # Supply adequacy across days
        out["supply_adequacy"] = float(df["daily_supply_fraction"].mean()) if "daily_supply_fraction" in df.columns and not df["daily_supply_fraction"].isna().all() else None
        # Average adherence across period
        out["average_adherence"] = float(df["average_adherence"].mean()) if "average_adherence" in df.columns and not df["average_adherence"].isna().all() else None
        return out
        pass


class Simulation:
    """
    Main Simulation class coordinating initialization, modules, and the run loop.

    Methods:
        __init__: Setup the simulation.
        _normalize_params: Normalize and map parameter keys to internal names.
        initialize_entities: Create agents, households, locations, retailers, channels, policy.
        initialize_network: Build social network and assign neighbors.
        initialize_modules: Initialize all modules.
        run: Execute the simulation loop.
        evaluate: Compute evaluation metrics dynamically from model plan and summary metrics.
        visualize: Plot key results.
        save_results: Save results to CSV.
    """

    def __init__(self, model_plan: Dict[str, Any]):
        """
        Initialize the Simulation with a given model plan.

        Args:
            model_plan (Dict[str, Any]): Model plan JSON-like dictionary.

        Returns:
            None
        """
        self.model_plan = model_plan
        self.params = model_plan.get("parameters", {}).copy()
        self._normalize_params()  # FIXED: Normalize parameters to internal keys

        self.seed = int(self.params.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        # Registries
        self.agents: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []
        self.retailers: List[Retailer] = []
        self.channels: List[InformationChannel] = []
        self.policy: Optional[PolicyCampaign] = None

        # Network
        self.G: Optional[nx.Graph] = None

        # Modules
        self.network_builder: Optional[SocialNetworkBuilder] = None
        self.policy_module: Optional[PolicyCampaignModule] = None
        self.info_module: Optional[InformationChannelModule] = None
        self.mobility_module: Optional[MobilityAndLocationModule] = None
        self.retail_module: Optional[RetailerSupplyModule] = None
        self.behavior_module: Optional[BehaviorDecisionModule] = None
        self.aggregator: Optional[AdoptionAggregator] = None

        # Signals
        self.signals: Dict[str, Any] = {}

        # Time horizon
        self.time_horizon_days = int(self.params.get("time_horizon_days", 120))
        self.timestep_days = float(self.params.get("timestep_days", 1.0))

        # Results
        self.results_df: Optional[pd.DataFrame] = None

        # Initialize entities and modules
        self.initialize_entities()
        self.initialize_network()
        self.initialize_modules()
        pass

    def _normalize_params(self) -> None:
        """
        Normalize parameter names to support both spec-like and previous iteration keys.

        Returns:
            None
        """
        # FIXED: Parameter normalization mapping for compatibility
        if "random_seed" in self.params and "seed" not in self.params:
            self.params["seed"] = self.params["random_seed"]
        if "simulation_days" in self.params and "time_horizon_days" not in self.params:
            self.params["time_horizon_days"] = self.params["simulation_days"]
        if "time_step_days" in self.params and "timestep_days" not in self.params:
            self.params["timestep_days"] = self.params["time_step_days"]
        if "network_topology" in self.params and "social_network_type" not in self.params:
            self.params["social_network_type"] = self.params["network_topology"]
        if "average_degree" in self.params and "avg_degree" not in self.params:
            self.params["avg_degree"] = self.params["average_degree"]
        pass

    def initialize_entities(self) -> None:
        """
        Initialize agents, households, locations, retailers, channels, and policy from model plan.

        Returns:
            None
        """
        N = int(self.params.get("population_size", 5000))
        initial_adoption_rate = float(self.params.get("initial_adoption_rate", 0.15))
        zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]

        # Generate household sizes
        hh_size_probs = [0.25, 0.35, 0.25, 0.1, 0.05]
        hh_sizes = []
        remaining = N
        while remaining > 0:
            size_cat = self.rng.choice([1, 2, 3, 4, 5], p=hh_size_probs)
            size = size_cat if size_cat < 5 else 5
            size = min(size, remaining)
            hh_sizes.append(size)
            remaining -= size

        # Build households with SES
        self.households = []
        pid_counter = 0
        for hid, size in enumerate(hh_sizes):
            members = list(range(pid_counter, pid_counter + size))
            pid_counter += size
            hh_ses = int(self.rng.integers(1, 6))
            budget_scale_by_ses = [0.8, 1.0, 1.2, 1.5, 2.0]
            scale = budget_scale_by_ses[hh_ses - 1]
            base_budget = 2000.0
            budget = base_budget * scale
            mask_inventory = max(0, int(self.rng.normal(0.5, 1.0)))
            home_zone = zones[self.rng.integers(0, len(zones))]
            hh = Household(id=hid, members=members, budget=budget, mask_inventory=mask_inventory, household_norms=clamp(self.rng.normal(0.4, 0.2), 0.0, 1.0), home_location=home_zone)
            setattr(hh, "_ses", hh_ses)  # store SES for coherence
            self.households.append(hh)

        # Build person_id -> household_id mapping
        person_to_household_id: Dict[int, int] = {}
        for hh in self.households:
            for pid in hh.members:
                person_to_household_id[pid] = hh.id

        self.agents = []
        genders = ["male", "female", "other"]
        gender_probs = [0.49, 0.49, 0.02]

        income_means = [20000, 35000, 50000, 75000, 120000]
        income_sds = [5000, 7000, 9000, 13000, 20000]

        for pid in range(N):
            hh_id = person_to_household_id.get(pid, 0)
            hh = next((h for h in self.households if h.id == hh_id), None)
            ses = getattr(hh, "_ses", int(self.rng.integers(1, 6)))
            income = max(0.0, self.rng.normal(income_means[ses - 1], income_sds[ses - 1]))
            age = int(max(0, min(85, triangular(0, 35, 85, self.rng))))
            gender = self.rng.choice(genders, p=gender_probs)
            rp = clamp(self.rng.normal(0.45, 0.2), 0.0, 1.0)
            trust = clamp(self.rng.normal(0.5, 0.25), 0.0, 1.0)
            sus = clamp(self.rng.normal(0.5, 0.15), 0.0, 1.0)
            exposure = clamp(self.rng.normal(0.5, 0.2), 0.0, 1.0)
            attitude = clamp(self.rng.normal(0.3, 0.25), 0.0, 1.0)
            ownership = int(self.rng.binomial(1, initial_adoption_rate)) * self.rng.integers(1, 6)
            use_state = int(self.rng.random() < initial_adoption_rate)
            compliance_threshold = clamp(self.rng.normal(0.5, 0.15), 0.0, 1.0)
            home_zone = hh.home_location if hh is not None else zones[self.rng.integers(0, len(zones))]
            daily_profile = {
                "home": float(self.params.get("daily_prob_home", 0.5)),
                "work": float(self.params.get("daily_prob_work", 0.25)),
                "school": float(self.params.get("daily_prob_school", 0.1)),
                "retail": float(self.params.get("daily_prob_retail", 0.1)),
                "other": float(self.params.get("daily_prob_other", 0.05)),
            }
            p = Person(
                id=pid, age=age, gender=gender, household_id=hh_id, income=float(income),
                socioeconomic_status=ses, risk_perception=rp, trust_in_authorities=trust,
                social_influence_susceptibility=sus, information_exposure_level=exposure,
                mask_attitude=attitude, mask_ownership=ownership, mask_use_state=use_state,
                compliance_threshold=compliance_threshold, daily_schedule_profile=daily_profile,
                location_home_zone=home_zone, adherence_prob=clamp(self.rng.normal(0.8, 0.1), 0.0, 1.0),
                fatigue_level=clamp(self.rng.normal(0.0, 0.05), 0.0, 1.0),
                mask_quality=clamp(self.rng.normal(0.7, 0.1), 0.0, 1.0)
            )
            self.agents.append(p)

        # Locations
        self.locations = []
        loc_id_counter = 0
        open_hours = (int(self.params.get("open_hours_start", 8)), int(self.params.get("open_hours_end", 20)))
        default_enforcement = float(self.params.get("default_location_enforcement_level", 0.2))
        default_fine = float(self.params.get("location_fine_amount", 0.0))
        capacity_mean = int(self.params.get("location_capacity_mean", 50))
        capacity_sd = int(self.params.get("location_capacity_sd", 20))

        type_counts = self.params.get("locations_count_by_type", None)
        if type_counts is None:
            type_counts = {
                "workplaces": 100,
                "schools": 30,
                "retail": 80,
                "transit": 10,
                "parks": 20,
            }
        for loc_type, count in type_counts.items():
            for _ in range(int(count)):
                capacity = max(5, int(self.rng.normal(capacity_mean, max(1, capacity_sd))))
                zone = zones[self.rng.integers(0, len(zones))]
                L = Location(
                    id=loc_id_counter, type=loc_type, capacity=capacity, mask_policy="recommended",
                    enforcement_level=default_enforcement, fine_amount=default_fine, open_hours=open_hours, zone=zone
                )
                loc_id_counter += 1
                self.locations.append(L)

        # Retailers
        self.retailers = []
        retailer_count = int(self.params.get("retailer_count", 50))
        initial_mask_supply = int(self.params.get("initial_mask_supply", 10000))
        restock_rate_per_day = int(self.params.get("restock_rate_per_day", 500))
        base_price = float(self.params.get("mask_cost", 2.0)) * (1.0 + float(self.params.get("price_markup_mean", 0.25)))
        per_retailer_stock = max(0, initial_mask_supply // max(1, retailer_count))
        for rid in range(retailer_count):
            zone = zones[self.rng.integers(0, len(zones))]
            r = Retailer(id=rid, stock_level=per_retailer_stock, price=base_price, restock_rate=restock_rate_per_day, location=zone)
            self.retailers.append(r)

        # Information channels
        self.channels = []
        channel_types = ["TV", "Radio", "SocialMedia", "Newspaper", "NGO"]
        info_channel_count = int(self.params.get("information_channel_count", 8))
        for cid in range(info_channel_count):
            ch_type = channel_types[cid % len(channel_types)]
            credibility = clamp(self.rng.normal(float(self.params.get("channel_credibility_mean", 0.6)), float(self.params.get("channel_credibility_sd", 0.15))), 0.0, 1.0)
            reach = clamp(self.rng.normal(float(self.params.get("channel_reach_mean", 0.4)), float(self.params.get("channel_reach_sd", 0.2))), 0.0, 1.0)
            bias = clamp(self.rng.normal(float(self.params.get("channel_bias_mean", 0.0)), float(self.params.get("channel_bias_sd", 0.3))), -1.0, 1.0)
            misinformation_rate = clamp(float(self.params.get("misinformation_level", 0.1)), 0.0, 1.0)
            ch = InformationChannel(id=cid, channel_type=ch_type, credibility=credibility, reach=reach, bias=bias, misinformation_rate=misinformation_rate)
            self.channels.append(ch)

        # Policy campaign
        start = self.params.get("mandate_start_day", None)
        end = self.params.get("mandate_end_day", None)
        self.policy = PolicyCampaign(
            id=1, mandate_active=False,
            mandate_start_day=start if start is None else int(start),
            mandate_end_day=end if end is None else int(end),
            messaging_intensity=float(self.params.get("messaging_intensity", 0.3)),
            enforcement_level=float(self.params.get("enforcement_level", 0.5)),
            fine_amount=float(self.params.get("fine_amount", 50.0)),
            subsidy_rate=float(self.params.get("subsidy_rate", 0.0)),
            distribution_schedule={
                "interval_days": int(self.params.get("distribution_interval_days", 14)),
                "quantity_per_capita": int(self.params.get("distribution_quantity_per_capita", 2))
            }
        )
        pass

    def initialize_network(self) -> None:
        """
        Build the social network and assign neighbors to agents.

        Returns:
            None
        """
        self.network_builder = SocialNetworkBuilder(self.params, self.rng)
        self.G = self.network_builder.build(self.agents)
        pass

    def initialize_modules(self) -> None:
        """
        Initialize all modules used in the simulation.

        Returns:
            None
        """
        self.policy_module = PolicyCampaignModule(self.policy, self.locations, self.agents, self.households, self.params, self.rng)
        self.info_module = InformationChannelModule(self.channels, self.agents, self.params, self.rng)
        self.mobility_module = MobilityAndLocationModule(self.locations, self.agents, self.households, self.params, self.rng)
        self.retail_module = RetailerSupplyModule(self.retailers, self.agents, self.households, self.params, self.rng)
        self.behavior_module = BehaviorDecisionModule(self.agents, self.households, self.G, self.params, self.rng)
        self.aggregator = AdoptionAggregator(self.agents, self.households, self.locations, self.retailers, self.params)
        pass

    def run(self) -> None:
        """
        Execute the simulation loop across the time horizon.

        Returns:
            None
        """
        self.signals = {}
        avg_price_prev_day = float(np.mean([r.price for r in self.retailers])) if self.retailers else float("nan")
        if self.behavior_module:
            self.behavior_module.set_prev_avg_price(avg_price_prev_day)

        for day in range(self.time_horizon_days):
            # Reset signals for the day
            self.signals = {}

            # Policy update
            self.policy_module.update(day, self.signals)

            # Information update
            self.info_module.update(day, self.signals)

            # Behavior update uses prior-day avg price if available
            if self.behavior_module is not None and self.aggregator is not None and self.aggregator.daily_records:
                avg_price_prev_day = self.aggregator.daily_records[-1].get("avg_price", avg_price_prev_day)
                self.behavior_module.set_prev_avg_price(avg_price_prev_day)

            # 1) Decide behavior first (today's wearing)
            self.behavior_module.update(day, self.signals)

            # 2) Mobility/enforcement uses today's decisions
            self.mobility_module.update(day, self.signals)

            # 3) Retailer supply and transactions
            self.retail_module.update(day, self.signals)

            # 4) Aggregate daily observables
            self.aggregator.update(day, self.signals)

        # Store results
        self.results_df = self.aggregator.to_dataframe()
        pass

    def evaluate(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics specified in the model plan and add additional summary metrics.

        Returns:
            Dict[str, Any]: Dictionary of metric name to computed value or time series.
        """
        if self.results_df is None or self.results_df.empty:
            return {}

        metrics_to_compute = self.model_plan.get("evaluation_metrics", [])
        metrics: Dict[str, Any] = {}

        # Metric: adoption_rate_over_time (time series)
        if "adoption_rate_over_time" in metrics_to_compute:
            metrics["adoption_rate_over_time"] = self.results_df[["day", "adoption_rate"]].copy()

        # Metric: time_to_threshold
        if "time_to_threshold" in metrics_to_compute:
            threshold = float(self.params.get("adoption_threshold_target", 0.7))
            df = self.results_df
            over = df[df["adoption_rate"] >= threshold]
            ttt = int(over["day"].iloc[0]) if not over.empty else None
            metrics["time_to_threshold"] = ttt

        # Additional required summary metrics
        summary = self.aggregator.summarize_metrics()
        metrics.update({
            "peak_adoption": summary.get("peak_adoption"),
            "final_adoption_rate": summary.get("final_adoption_rate"),
            "time_to_50_percent_adoption": summary.get("time_to_50_percent_adoption"),
            "sustained_adoption_rate": summary.get("sustained_adoption_rate"),
            "policy_effect_size": summary.get("policy_effect_size"),
            "supply_adequacy": summary.get("supply_adequacy"),
            "average_adherence": summary.get("average_adherence")
        })

        return metrics
        pass

    def visualize(self) -> None:
        """
        Visualize key simulation results such as adoption rate over time.

        Returns:
            None
        """
        if self.results_df is None or self.results_df.empty:
            print("No results to visualize.")
            return

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results_df["day"], self.results_df["adoption_rate"], label="Adoption Rate")
            if "location_compliance_rate" in self.results_df.columns:
                plt.plot(self.results_df["day"], self.results_df["location_compliance_rate"], label="Location Compliance")
            if "masked_interaction_fraction" in self.results_df.columns:
                plt.plot(self.results_df["day"], self.results_df["masked_interaction_fraction"], label="Masked Interaction Fraction")
            plt.xlabel("Day")
            plt.ylabel("Rate")
            plt.title("Mask Adoption and Related Metrics Over Time")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")

# Execute main for both direct execution and sandbox wrapper invocation
main()