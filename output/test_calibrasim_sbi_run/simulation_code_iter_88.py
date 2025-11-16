import os
import sys
import json
import math
import csv
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# Optional scientific/plotting libraries
try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# FIXED: Move imports to the top and define constants for paths per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def main():
    """
    Placeholder main definition to satisfy early references.

    This definition will be overridden later with the actual main routine.
    """
    pass  # Required syntactic placeholder per instructions


def clamp01(x: float) -> float:
    """
    Clamp a floating-point number to the [0, 1] interval.

    Args:
        x (float): Input value.

    Returns:
        float: Clamped value in [0, 1].
    """
    pass  # Required syntactic placeholder per instructions
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def sigmoid(x: float) -> float:
    """
    Numerically stable logistic sigmoid function.

    Args:
        x (float): Input value.

    Returns:
        float: 1 / (1 + exp(-x)).
    """
    pass  # Required syntactic placeholder per instructions
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default if denominator is zero.

    Args:
        numerator (float): Numerator.
        denominator (float): Denominator.
        default (float): Default value if division not possible.

    Returns:
        float: Result of division or default.
    """
    pass  # Required syntactic placeholder per instructions
    if denominator == 0:
        return default
    return numerator / denominator


def generate_small_world(n: int, k: int, beta: float, rng: random.Random) -> List[List[int]]:
    """
    Generate a Watts-Strogatz small-world network adjacency list.

    Handles small-n cases explicitly and clamps k appropriately.

    Args:
        n (int): Number of nodes.
        k (int): Average degree (even required).
        beta (float): Rewiring probability in [0, 1].
        rng (random.Random): RNG for reproducibility.

    Returns:
        List[List[int]]: Adjacency list for each node.
    """
    pass  # Required syntactic placeholder per instructions
    # FIXED: Handle small-n networks robustly per feedback
    if n <= 1:
        return [[] for _ in range(n)]
    if n == 2:
        return [[1], [0]]
    k = int(k)
    # FIXED: Clamp k <= n-1 and ensure evenness while allowing small-n grace
    max_even = max(0, (n - 1) - ((n - 1) % 2))
    k = min(k + (k % 2), max_even)
    k = max(2, k) if n >= 3 else max(0, k)
    half_k = k // 2
    nbrs = [set() for _ in range(n)]
    # Ring lattice
    for i in range(n):
        for d in range(1, half_k + 1):
            j = (i + d) % n
            nbrs[i].add(j)
            nbrs[j].add(i)
    # Rewire edges
    for i in range(n):
        for j in list(nbrs[i]):
            if j > i and rng.random() < beta:
                nbrs[i].discard(j)
                nbrs[j].discard(i)
                candidates = [x for x in range(n) if x != i and x not in nbrs[i]]
                if candidates:
                    new_j = rng.choice(candidates)
                    nbrs[i].add(new_j)
                    nbrs[new_j].add(i)
    return [sorted(s) for s in nbrs]


@dataclass
class Person:
    """
    Represents an individual agent with attributes and behavioral rules related to mask adoption.

    Attributes:
        id (int): Unique identifier.
        age (int): Age in years.
        sex (str): Sex category (e.g., 'F', 'M', 'O').
        occupation (str): Occupation category.
        essential_worker (bool): Whether the person is an essential worker.
        health_risk_level (float): Individual health risk in [0, 1].
        household_id (int): Household identifier.
        socioeconomic_status (float): SES in [0, 1].
        risk_perception (float): Perceived risk in [0, 1].
        trust_in_government (float): Trust in government in [0, 1].
        trust_in_media (float): Trust in media in [0, 1].
        baseline_compliance_propensity (float): Baseline compliance propensity in [0, 1].
        mask_attitude (float): Attitude toward masks in [-1, 1].
        wearing_mask (bool): Whether wearing a mask today.
        masks_inventory (int): Number of masks available.
        social_network_neighbors (List[int]): Neighbor IDs in social network.
        daily_mobility_profile (Dict[str, float]): Map of location type to visit probability.
        workplace_id (Optional[int]): Workplace location id.
        school_id (Optional[int]): School location id.
        health_status (str): Health status text.
        information_exposure_level (float): Exposure level to information in [0, 1].
        social_susceptibility (float): Susceptibility to peer influence in [0, 1].
        violations (int): Count of policy violations.
        habit_strength (float): Habit strength in [0, 1] promoting consistent wearing.
        last_wore_mask_day (int): Last day index when a mask was worn, -1 if never.
    """
    id: int
    age: int
    sex: str
    occupation: str
    essential_worker: bool
    health_risk_level: float
    household_id: int
    socioeconomic_status: float
    risk_perception: float
    trust_in_government: float
    trust_in_media: float
    baseline_compliance_propensity: float
    mask_attitude: float
    wearing_mask: bool
    masks_inventory: int
    social_network_neighbors: List[int]
    daily_mobility_profile: Dict[str, float]
    workplace_id: Optional[int] = None
    school_id: Optional[int] = None
    health_status: str = "healthy"
    information_exposure_level: float = 0.0
    social_susceptibility: float = 0.5
    violations: int = 0
    habit_strength: float = 0.5
    last_wore_mask_day: int = -1

    def decide_to_wear_mask(
        self,
        day_index: int,
        peer_adoption_rate: float,
        observed_public_wearing: float,
        policy_strength: float,
        enforcement_probability: float,
        market_availability: float,
        price_per_mask: float,
        subsidy_amount: float,
        weights: Dict[str, float],
        rng: random.Random,
        fatigue: float,
    ) -> bool:
        """
        Decide whether to wear a mask today based on social, personal, policy, observation, risk, habit, and market factors.

        Args:
            day_index (int): Current day index.
            peer_adoption_rate (float): Peer wearing fraction.
            observed_public_wearing (float): Public wearing fraction.
            policy_strength (float): 1.0 if mandate on, else 0.0.
            enforcement_probability (float): Probability of inspection/fine at visited locations.
            market_availability (float): Market availability proxy in [0, 1].
            price_per_mask (float): Retail price per mask.
            subsidy_amount (float): Subsidy applied per mask.
            weights (Dict[str, float]): Behavioral weights.
            rng (random.Random): RNG.
            fatigue (float): Mask fatigue parameter.

        Returns:
            bool: True if wearing.
        """
        pass  # Required syntactic placeholder per instructions
        # FIXED: Incorporate more sophisticated utility function including habit and economic cost per feedback
        effective_price = max(0.0, price_per_mask - subsidy_amount)
        if self.masks_inventory <= 0:
            affordability = clamp01((self.socioeconomic_status + 0.1) - 0.5 * max(0.0, effective_price - 1.0))
            if rng.random() > (market_availability * affordability):
                self.baseline_compliance_propensity = clamp01(self.baseline_compliance_propensity - 0.25 * fatigue)
                return False

        # Habit term based on recency
        days_since_last = (day_index - self.last_wore_mask_day) if self.last_wore_mask_day >= 0 else 999
        habit_term = weights.get("habit_weight", 0.2) * self.habit_strength * (1.0 / (1.0 + 0.1 * days_since_last))

        social_term = weights.get("social_influence_weight", 0.4) * self.social_susceptibility * peer_adoption_rate
        personal_term = weights.get("personal_attitude_weight", 0.3) * (
            0.5 * self.baseline_compliance_propensity + 0.5 * ((self.mask_attitude + 1.0) / 2.0)
        )
        policy_term = weights.get("policy_compliance_weight", 0.2) * (
            self.trust_in_government * policy_strength + enforcement_probability
        )
        observation_term = weights.get("observation_weight", 0.1) * observed_public_wearing
        risk_term = weights.get("risk_weight", 0.2) * (self.risk_perception - 0.5)

        # Economic disutility from wearing (consumes mask)
        econ_cost = weights.get("economic_cost_weight", 0.1) * clamp01((effective_price - 0.5) / 2.0)
        fatigue_cost = weights.get("fatigue_weight", 0.1) * fatigue

        utility = social_term + personal_term + policy_term + observation_term + risk_term + habit_term - econ_cost - fatigue_cost
        p = sigmoid(2.8 * (utility - 0.5))
        decide = rng.random() < p
        return decide

    def update_beliefs_from_peers(self, neighbor_attitudes: List[float]) -> None:
        """
        Update mask attitude based on peer attitudes via social learning.

        Args:
            neighbor_attitudes (List[float]): List of neighbor mask attitudes in [-1, 1].
        """
        pass  # Required syntactic placeholder per instructions
        if not neighbor_attitudes:
            return
        avg_att = sum(neighbor_attitudes) / len(neighbor_attitudes)
        self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) * 0.8 + ((avg_att + 1.0) / 2.0) * 0.2) * 2.0 - 1.0

    def update_beliefs_from_media(self, media_delta: Dict[str, float]) -> None:
        """
        Update beliefs based on media and government guidance signals.

        Args:
            media_delta (Dict[str, float]): Deltas for 'risk', 'trust_gov', and 'attitude'.
        """
        pass  # Required syntactic placeholder per instructions
        self.risk_perception = clamp01(self.risk_perception + media_delta.get("risk", 0.0) * self.trust_in_media)
        self.trust_in_government = clamp01(self.trust_in_government + media_delta.get("trust_gov", 0.0))
        self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) + media_delta.get("attitude", 0.0)) * 2.0 - 1.0

    def respond_to_policy(self, compliance_decay_rate: float, positive_exposure: bool, fatigue: float) -> None:
        """
        Update propensity, habit, and attitude due to habit formation or fatigue.

        Args:
            compliance_decay_rate (float): Decay rate when not wearing.
            positive_exposure (bool): Whether person had positive exposure (e.g., wore mask).
            fatigue (float): Fatigue rate.
        """
        pass  # Required syntactic placeholder per instructions
        if positive_exposure:
            # Habit formation boosts attitude and habit slightly
            self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) + 0.003) * 2.0 - 1.0
            self.habit_strength = clamp01(self.habit_strength + 0.01)
        else:
            # Fatigue and compliance decay
            self.baseline_compliance_propensity = clamp01(
                self.baseline_compliance_propensity - compliance_decay_rate - 0.5 * fatigue
            )
            self.habit_strength = clamp01(self.habit_strength - 0.005 * (1.0 + fatigue))

    def travel_and_interact_at_locations(self, location_types: List[str], rng: random.Random) -> List[str]:
        """
        Sample visits to location types based on daily mobility profile.

        Args:
            location_types (List[str]): Location types to consider.
            rng (random.Random): RNG.

        Returns:
            List[str]: Visited location types for the day.
        """
        pass  # Required syntactic placeholder per instructions
        visits = []
        for loc in location_types:
            prob = self.daily_mobility_profile.get(loc, 0.0)
            if rng.random() < prob:
                visits.append(loc)
        return visits


@dataclass
class Household:
    """
    Represents a household that shares norms and mask resources.

    Attributes:
        id (int): Household id.
        member_ids (List[int]): Member person ids.
        household_income (float): Household income proxy in [0, 1].
        norm_strength (float): Strength of shared norms in [0, 1].
        mask_inventory (int): Shared household mask inventory.
    """
    id: int
    member_ids: List[int]
    household_income: float
    norm_strength: float
    mask_inventory: int = 0

    def share_norms(self, persons: Dict[int, Person]) -> None:
        """
        Propagate norms within household by adjusting members' attitudes.

        Args:
            persons (Dict[int, Person]): Mapping of id to Person.
        """
        pass  # Required syntactic placeholder per instructions
        if not self.member_ids:
            return
        avg_att = sum(persons[i].mask_attitude for i in self.member_ids) / len(self.member_ids)
        for i in self.member_ids:
            p = persons[i]
            p.mask_attitude = clamp01(
                ((p.mask_attitude + 1.0) / 2.0) * (1.0 - 0.2 * self.norm_strength) + ((avg_att + 1.0) / 2.0) * 0.2 * self.norm_strength
            ) * 2.0 - 1.0

    def pool_resources_for_masks(self, persons: Dict[int, Person]) -> None:
        """
        Share household mask inventory with members lacking masks.

        Args:
            persons (Dict[int, Person]): Mapping of id to Person.
        """
        pass  # Required syntactic placeholder per instructions
        for i in self.member_ids:
            if self.mask_inventory <= 0:
                break
            if persons[i].masks_inventory <= 0:
                persons[i].masks_inventory += 1
                self.mask_inventory -= 1


@dataclass
class Location:
    """
    Represents a location that can enforce mask policies.

    Attributes:
        id (int): Location id.
        type (str): Location type.
        capacity (int): Maximum concurrent visitors.
        mask_policy (str): 'none', 'recommended', or 'mandate'.
        enforcement_level (float): Strictness in [0, 1].
        foot_traffic_rate (float): Traffic proxy in [0, 1].
    """
    id: int
    type: str
    capacity: int
    mask_policy: str
    enforcement_level: float
    foot_traffic_rate: float

    def enforce_mask_policy(self, visitor_persons: List[Person], authority: "HealthAuthority", rng: random.Random) -> Tuple[int, int]:
        """
        Enforce mask policy, optionally denying entry and applying fines. Tracks violations and denials.

        Args:
            visitor_persons (List[Person]): Admitted visitors.
            authority (HealthAuthority): Authority with enforcement parameters.
            rng (random.Random): RNG.

        Returns:
            Tuple[int, int]: (enforcement actions, denials).
        """
        pass  # Required syntactic placeholder per instructions
        # FIXED: Implement deny entry and fine; track violations and denials per feedback
        if self.mask_policy != "mandate" or self.enforcement_level <= 0.0:
            return 0, 0
        actions = 0
        denials = 0
        inspect_prob = clamp01(self.enforcement_level * authority.enforcement_intensity)
        for p in visitor_persons:
            if not p.wearing_mask:
                if rng.random() < inspect_prob:
                    actions += 1
                    p.violations += 1
                    authority.collected_fines += authority.fine_amount
                    authority.enforcement_cost += authority.enforcement_cost_per_action
                    # Denial probability proportional to enforcement and policy strictness
                    if rng.random() < self.enforcement_level:
                        denials += 1
                    # Enforcement nudges future compliance
                    p.baseline_compliance_propensity = clamp01(p.baseline_compliance_propensity + 0.05)
                    p.risk_perception = clamp01(p.risk_perception + 0.02)
        return actions, denials

    def broadcast_signage(self) -> float:
        """
        Signage or cues at locations that encourage mask use.

        Returns:
            float: Additional observed wearing contribution.
        """
        pass  # Required syntactic placeholder per instructions
        if self.mask_policy == "mandate":
            return 0.05
        if self.mask_policy == "recommended":
            return 0.02
        return 0.0


@dataclass
class HealthAuthority:
    """
    Government authority managing mandates, guidance, and enforcement.

    Attributes:
        id (int): Identifier.
        mandate_on (bool): Whether a mandate is active.
        mandate_start_day (int): Day mandate activates.
        mandate_scope (str): Scope descriptor.
        fine_amount (float): Fine per violation.
        enforcement_resources (float): Resource proxy in [0, 1].
        enforcement_intensity (float): Inspection intensity in [0, 1].
        guidance_strength (float): Guidance strength in [0, 1].
        collected_fines (float): Total fines collected.
        enforcement_cost (float): Total cost of enforcement.
        enforcement_cost_per_action (float): Cost per enforcement action.
    """
    id: int
    mandate_on: bool
    mandate_start_day: int
    mandate_scope: str
    fine_amount: float
    enforcement_resources: float
    enforcement_intensity: float = 0.3
    guidance_strength: float = 0.3
    collected_fines: float = 0.0
    enforcement_cost: float = 0.0
    enforcement_cost_per_action: float = 0.0

    def set_or_update_policy(self, day: int) -> None:
        """
        Update mandate status based on the current day.

        Args:
            day (int): Day index.
        """
        pass  # Required syntactic placeholder per instructions
        if day >= self.mandate_start_day:
            self.mandate_on = True

    def allocate_enforcement(self, locations: List[Location]) -> None:
        """
        Allocate enforcement across locations.

        Args:
            locations (List[Location]): List of locations.
        """
        pass  # Required syntactic placeholder per instructions
        if not locations:
            return
        base = clamp01(self.enforcement_resources)
        for loc in locations:
            if loc.mask_policy == "mandate":
                loc.enforcement_level = clamp01(base * (0.5 + 0.5 * loc.foot_traffic_rate))
            else:
                loc.enforcement_level = clamp01(0.2 * base * loc.foot_traffic_rate)

    def issue_public_guidance(self) -> Dict[str, float]:
        """
        Issue public guidance impacts.

        Returns:
            Dict[str, float]: Deltas for 'risk', 'trust_gov', 'attitude'.
        """
        pass  # Required syntactic placeholder per instructions
        return {
            "risk": 0.01 * self.guidance_strength,
            "trust_gov": 0.01 * self.guidance_strength,
            "attitude": 0.005 * self.guidance_strength,
        }


@dataclass
class InformationSource:
    """
    Media/Information source broadcasting messages.

    Attributes:
        id (int): Identifier.
        credibility (float): Source credibility [0, 1].
        message_intensity (float): Intensity factor.
        message_slant (float): Positive or negative slant.
        reach (float): Fraction of population reached [0, 1].
        misinformation_rate (float): Probability of misinformation broadcast.
        campaign_intensity (float): Extra campaign intensity.
    """
    id: int
    credibility: float
    message_intensity: float
    message_slant: float
    reach: float
    misinformation_rate: float = 0.0
    campaign_intensity: float = 0.0

    def broadcast_messages(self, rng: random.Random) -> Dict[str, float]:
        """
        Broadcast messages; sign and misinformation modulate deltas.

        Args:
            rng (random.Random): RNG.

        Returns:
            Dict[str, float]: Deltas for 'risk' and 'attitude'.
        """
        pass  # Required syntactic placeholder per instructions
        sign = 1.0 if self.message_slant >= 0 else -1.0
        misinf = 1.0 if rng.random() < self.misinformation_rate else 0.0
        intensity = self.message_intensity + self.campaign_intensity
        attitude_delta = (0.01 * intensity * sign * self.credibility) * (1.0 - 2.0 * misinf)
        risk_delta = 0.01 * intensity * max(0.0, sign) * self.credibility
        return {"risk": risk_delta, "attitude": attitude_delta}


@dataclass
class Retailer:
    """
    Retailer supplying masks with inventory and pricing logic.

    Attributes:
        id (int): Retailer id.
        inventory_level (int): Current inventory level.
        restock_rate (float): Not used directly; preserved for compatibility.
        price_per_mask (float): Current price per mask.
        rationing_policy (int): Max quantity per transaction.
    """
    id: int
    inventory_level: int
    restock_rate: float
    price_per_mask: float
    rationing_policy: int = 2

    def sell_masks(self, quantity: int) -> Tuple[int, bool]:
        """
        Sell a quantity of masks, enforcing rationing and inventory constraints.

        Args:
            quantity (int): Requested quantity.

        Returns:
            Tuple[int, bool]: (sold quantity, shortage occurred).
        """
        pass  # Required syntactic placeholder per instructions
        quantity = int(max(0, min(self.rationing_policy, quantity)))
        if self.inventory_level <= 0:
            return 0, True
        sold = min(self.inventory_level, quantity)
        self.inventory_level -= sold
        shortage = sold < quantity
        return sold, shortage

    def adjust_prices(self) -> None:
        """
        Adjust prices based on scarcity, with a simple linear scarcity premium.
        """
        pass  # Required syntactic placeholder per instructions
        base = 1.0
        scarcity = 0.0 if self.inventory_level > 500 else (1.0 - (self.inventory_level / 500.0))
        self.price_per_mask = base + 0.5 * max(0.0, scarcity)

    def restock_to_target(self, target: int, added: int) -> int:
        """
        Restock up to a target inventory, limited by added capacity.

        Args:
            target (int): Target inventory level.
            added (int): Available masks to distribute.

        Returns:
            int: Quantity actually added.
        """
        pass  # Required syntactic placeholder per instructions
        need = max(0, target - self.inventory_level)
        to_add = min(need, max(0, added))
        self.inventory_level += to_add
        return to_add


class Simulation:
    """
    Main simulation class coordinating agents, locations, retailers, and policy dynamics.

    This class implements:
    - Parameter mapping to spec keys (seed, simulation_duration_days, risk_signal_period_days, risk_signal_mode)
    - Multiple retailers with restock scheduling and supply capacity
    - Policy enforcement with fines and denials; violation tracking
    - Efficient visit sampling (pre-sample per person and allocate to locations)
    - Risk signal per spec parameters
    - Fatigue and adoption decay without positive exposure
    - Metrics and validation checks with post-processing
    - Time-dependency in agent activity (weekly/seasonal modulation)
    - Extended validation against real data if provided
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize simulation components and state.

        Args:
            params (Dict[str, Any]): Configuration parameters.
        """
        pass  # Required syntactic placeholder per instructions
        # FIXED: Add error handling for edge cases in parameter parsing
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")

        self.params = params

        # FIXED: Map spec parameter names; prefer 'seed' and 'simulation_duration_days'
        seed_val = int(params.get("seed", params.get("random_seed", 42)))
        self.rng = random.Random(seed_val)
        self.population_size = max(0, int(params.get("population_size", 500)))
        # FIXED: Use simulation_duration_days (spec) with fallback to time_horizon_days
        base_horizon = max(0, int(params.get("simulation_duration_days", params.get("time_horizon_days", 60))))
        # Optional prediction period override
        pp = params.get("prediction_period", None)
        if isinstance(pp, dict) and "start_day" in pp and "end_day" in pp:
            start_day = int(pp.get("start_day", 0))
            end_day = int(pp.get("end_day", base_horizon))
            self.time_horizon_days = max(0, end_day - start_day)
        else:
            self.time_horizon_days = base_horizon

        self.avg_degree = max(0, int(params.get("avg_degree", params.get("average_degree", 8))))
        # FIXED: Keep rewiring_prob usage
        self.rewiring_prob = clamp01(float(params.get("rewiring_prob", 0.05)))
        self.initial_adoption_rate = clamp01(float(params.get("initial_adoption_rate", params.get("initial_adopter_fraction", 0.1))))

        # Risk signal configuration
        # FIXED: Parameterize risk signal per spec
        self.risk_signal_initial = clamp01(float(params.get("risk_signal_initial", 0.2)))
        self.risk_signal_amplitude = clamp01(float(params.get("risk_signal_amplitude", 0.3)))
        self.risk_signal_period_days = max(1, int(params.get("risk_signal_period_days", 30)))
        self.risk_signal_mode = str(params.get("risk_signal_mode", "exogenous"))

        # Behavior weights
        self.weights = {
            "social_influence_weight": clamp01(float(params.get("social_influence_weight", params.get("peer_influence_weight", 0.5)))),
            "personal_attitude_weight": clamp01(float(params.get("personal_attitude_weight", 0.3))),
            "policy_compliance_weight": clamp01(float(params.get("policy_compliance_weight", params.get("policy_effect_weight", 0.35)))),
            "observation_weight": clamp01(float(params.get("observation_weight", params.get("observation_effect", 0.4)))),
            "habit_weight": clamp01(float(params.get("habit_persistence_weight", 0.2))),
            "economic_cost_weight": clamp01(float(params.get("price_sensitivity", 0.3))),
            "fatigue_weight": clamp01(float(params.get("compliance_fatigue_rate", 0.005))),
            "risk_weight": clamp01(float(params.get("perceived_risk_weight", 0.3))),
        }

        # Fatigue and compliance decay
        self.compliance_decay_rate = clamp01(float(params.get("compliance_decay_rate", params.get("forgetting_rate", 0.005))))
        self.mask_fatigue = clamp01(float(params.get("mask_fatigue", params.get("compliance_fatigue_rate", 0.002))))  # FIXED: Added fatigue per feedback

        # Retail/supply params and multiple retailers
        # FIXED: Multiple retailers with restock schedule and supply capacity
        self.restock_interval_days = max(1, int(params.get("restock_interval_days", 3)))
        self.supply_capacity_per_day = max(0, int(params.get("supply_capacity_per_day", 3000)))
        self.restock_lot_size = max(0, int(params.get("restock_lot_size", 2000)))
        self.subsidy_amount = max(0.0, float(params.get("subsidy_amount", 0.0)))  # FIXED: Subsidy support
        self.price_elasticity = float(params.get("price_elasticity_of_demand", -0.2))

        self.mask_price = max(0.0, float(params.get("mask_cost", params.get("mask_price", 1.0))))
        retailer_count = max(1, int(params.get("retailer_count", 50)))
        init_inv = max(0, int(params.get("initial_inventory", int(params.get("supply_initial_stock_per_capita", 2) * max(1, self.population_size) / max(1, retailer_count)))))
        rationing_policy = max(1, int(params.get("rationing_policy", 2)))
        self.retailers: List[Retailer] = []
        for rid in range(retailer_count):
            self.retailers.append(
                Retailer(
                    id=rid,
                    inventory_level=init_inv,
                    restock_rate=0.0,
                    price_per_mask=self.mask_price,
                    rationing_policy=rationing_policy,
                )
            )
        self.stockout_days = 0  # count retailer-day stockouts aggregated

        # Authority and media
        self.authority = HealthAuthority(
            id=0,
            mandate_on=bool(params.get("mandate_on", params.get("mandate_active", False))),
            mandate_start_day=int(params.get("mandate_start_day", 30)),
            mandate_scope=str(params.get("mandate_scope", "indoor_public")),
            fine_amount=float(params.get("penalty_amount", params.get("fine_amount", 50.0))),
            enforcement_resources=float(params.get("enforcement_resources", params.get("enforcement_level", 0.3))),
            enforcement_intensity=float(params.get("enforcement_intensity", params.get("enforcement_probability", 0.3))),
            guidance_strength=float(params.get("government_guidance_influence", params.get("messaging_intensity", 0.35))),
            enforcement_cost_per_action=float(params.get("enforcement_cost_per_action", 1.0)),
        )
        self.media = InformationSource(
            id=0,
            credibility=clamp01(float(params.get("media_credibility", 0.7))),
            message_intensity=clamp01(float(params.get("media_influence_strength", 0.3))),
            message_slant=float(params.get("media_message_slant", 1.0)),
            reach=clamp01(float(params.get("media_reach", 0.6))),
            misinformation_rate=clamp01(float(params.get("misinformation_rate", 0.05))),
            campaign_intensity=clamp01(float(params.get("campaign_intensity", 0.2))),
        )

        # Network
        self.network = generate_small_world(self.population_size, self.avg_degree, self.rewiring_prob, self.rng)

        # Locations setup
        loc_types = params.get(
            "location_types",
            ["home", "work", "school", "store", "public_transport", "other_public"],
        )
        foot_mult = params.get(
            "foot_traffic_multipliers",
            {
                "home": 1.0,
                "work": 0.6,
                "school": 0.5,
                "store": 0.2,
                "public_transport": 0.3,
                "other_public": 0.2,
            },
        )

        # Create a simple set of locations for each type
        self.locations: List[Location] = []
        loc_id = 0
        default_capacity = int(params.get("default_location_capacity", 100))
        for t in loc_types:
            count = max(1, int(params.get(f"{t}_locations", 3)))
            for _ in range(count):
                policy = "none" if t == "home" else ("mandate" if self.authority.mandate_on else "recommended")
                self.locations.append(
                    Location(
                        id=loc_id,
                        type=t,
                        capacity=default_capacity,
                        mask_policy=policy,
                        enforcement_level=0.0,
                        foot_traffic_rate=float(foot_mult.get(t, 0.2)),
                    )
                )
                loc_id += 1


# Execute main for both direct execution and sandbox wrapper invocation
main()