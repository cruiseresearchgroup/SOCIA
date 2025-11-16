def main():
    pass

import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# Path handling per instruction
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# FIXED: Restore deterministic seeding and simulation utilities
def clamp01(x: float) -> float:
    """
    Clamp a numeric value to the [0, 1] interval.

    Args:
        x: Input numeric value.

    Returns:
        A float in the range [0, 1].
    """
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
    pass


def sigmoid(x: float) -> float:
    """
    Numerically stable logistic sigmoid function.

    Args:
        x: Real-valued input.

    Returns:
        Sigmoid(x) in (0, 1).
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0
    pass


def generate_small_world(n: int, k: int, beta: float, rng: random.Random) -> List[List[int]]:
    """
    Generate a small-world network using a Watts–Strogatz-like algorithm without external dependencies.

    Args:
        n: Number of nodes.
        k: Each node is joined with its k nearest neighbors in a ring topology (k should be even; enforced to >=2).
        beta: The probability of rewiring each edge.
        rng: Random number generator for deterministic behavior.

    Returns:
        Adjacency list representation of the graph as a list of neighbor index lists.
    """
    if n <= 1:
        return [[] for _ in range(n)]
    k = max(2, int(k))
    if k % 2 == 1:
        k += 1
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
    pass


@dataclass
class Person:
    """
    Agent representing an individual in the mask adoption simulation.

    Attributes:
        id: Unique identifier.
        age: Age of the individual (years).
        household_id: Household the person belongs to.
        socioeconomic_status: SES category or score in [0,1], higher means wealthier.
        risk_perception: Perceived risk level in [0,1].
        trust_in_government: Trust in official guidance [0,1].
        trust_in_media: Trust in media sources [0,1].
        baseline_compliance_propensity: Intrinsic compliance [0,1].
        mask_attitude: Pro-mask attitude [-1,1], positive encourages wearing.
        wearing_mask: Whether the person is wearing a mask today.
        masks_inventory: Integer count of masks owned.
        social_network_neighbors: List of neighbor person IDs.
        daily_mobility_profile: Map of location types to fraction of day or visit propensity.
        workplace_id: Optional workplace location ID.
        school_id: Optional school location ID.
        health_status: Simple health status label.
        information_exposure_level: Aggregate exposure [0,1].
        social_susceptibility: Susceptibility to social influence [0,1].
    """
    id: int
    age: int
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

    def decide_to_wear_mask(
        self,
        peer_adoption_rate: float,
        observed_public_wearing: float,
        policy_strength: float,
        enforcement_probability: float,
        market_availability: float,
        price_per_mask: float,
        weights: Dict[str, float],
        rng: random.Random,
    ) -> bool:
        """
        Decide whether to wear a mask today by combining social, personal, policy, and observational influences.

        Args:
            peer_adoption_rate: Fraction of neighbors wearing a mask.
            observed_public_wearing: Observed public wearing rate.
            policy_strength: Policy/mandate scalar [0,1].
            enforcement_probability: Probability of inspection/fine if non-compliant (0-1).
            market_availability: Probability masks are available for purchase.
            price_per_mask: Current price per mask unit.
            weights: Decision weights including social_influence_weight, personal_attitude_weight, policy_compliance_weight, observation_weight.
            rng: Random number generator.

        Returns:
            Boolean decision to wear mask.
        """
        # Economic constraint: if no masks and cannot buy, cannot wear.
        if self.masks_inventory <= 0:
            affordability = clamp01((self.socioeconomic_status + 0.1) - 0.5 * max(0.0, price_per_mask - 1.0))
            if rng.random() > (market_availability * affordability):
                return False

        # Utility composition
        social_term = weights.get("social_influence_weight", 0.4) * self.social_susceptibility * peer_adoption_rate
        personal_term = weights.get("personal_attitude_weight", 0.3) * (0.5 * self.baseline_compliance_propensity + 0.5 * ((self.mask_attitude + 1.0) / 2.0))
        policy_term = weights.get("policy_compliance_weight", 0.2) * (self.trust_in_government * policy_strength + enforcement_probability)
        observation_term = weights.get("observation_weight", 0.1) * observed_public_wearing
        risk_term = 0.3 * (self.risk_perception - 0.5)

        utility = social_term + personal_term + policy_term + observation_term + risk_term
        p = sigmoid(2.5 * (utility - 0.5))
        decision = rng.random() < p

        if decision and self.masks_inventory <= 0:
            # Simulate purchase if decided to wear but inventory missing
            self.masks_inventory += 1

        return decision
        pass

    def update_beliefs_from_peers(self, neighbor_attitudes: List[float]) -> None:
        """
        Update attitude based on neighbors' attitudes using a bounded confidence update.

        Args:
            neighbor_attitudes: List of neighbor mask attitudes in [-1,1].

        Returns:
            None
        """
        if not neighbor_attitudes:
            return
        avg_att = sum(neighbor_attitudes) / len(neighbor_attitudes)
        self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) * 0.8 + ((avg_att + 1.0) / 2.0) * 0.2) * 2.0 - 1.0
        pass

    def update_beliefs_from_media(self, media_delta: Dict[str, float]) -> None:
        """
        Update risk perception, trust, and attitude from media inputs.

        Args:
            media_delta: Dictionary with optional keys: 'risk', 'trust_gov', 'attitude'.

        Returns:
            None
        """
        self.risk_perception = clamp01(self.risk_perception + media_delta.get("risk", 0.0) * self.trust_in_media)
        self.trust_in_government = clamp01(self.trust_in_government + media_delta.get("trust_gov", 0.0))
        self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) + media_delta.get("attitude", 0.0)) * 2.0 - 1.0
        pass

    def respond_to_policy(self, compliance_decay_rate: float) -> None:
        """
        Apply compliance fatigue or habit formation effects each day.

        Args:
            compliance_decay_rate: Small rate shifting baseline compliance over time.

        Returns:
            None
        """
        # Fatigue lowers compliance slightly if wearing; habit increases attitude slightly.
        if self.wearing_mask:
            self.mask_attitude = clamp01(((self.mask_attitude + 1.0) / 2.0) + 0.003) * 2.0 - 1.0
        else:
            self.baseline_compliance_propensity = clamp01(self.baseline_compliance_propensity - compliance_decay_rate)
        pass

    def purchase_masks(self, retailer: "Retailer", max_purchase: int = 2) -> Tuple[int, bool]:
        """
        Attempt to purchase masks from a retailer subject to inventory and price.

        Args:
            retailer: Retailer instance.
            max_purchase: Maximum masks to purchase.

        Returns:
            Tuple of (purchased_count, shortage_flag)
        """
        desired = max(0, max_purchase - self.masks_inventory)
        if desired <= 0:
            return 0, False
        affordable_qty = int(max(0, math.floor((self.socioeconomic_status * 10.0) / max(0.1, retailer.price_per_mask))))
        qty = clamp01(affordable_qty / max(1.0, desired)) * desired
        qty = int(min(desired, max(0, round(qty))))
        purchased, shortage = retailer.sell_masks(qty)
        self.masks_inventory += purchased
        return purchased, shortage
        pass

    def share_opinions(self, neighbor_ids: List[int]) -> None:
        """
        Placeholder for explicit opinion sharing dynamics; the main influence is in update_beliefs_from_peers.

        Args:
            neighbor_ids: IDs of neighbors.

        Returns:
            None
        """
        # Simplified; explicit messaging modeled implicitly via update_beliefs_from_peers
        pass

    def travel_and_interact_at_locations(self, location_types: List[str], rng: random.Random) -> List[str]:
        """
        Select locations to visit for the day based on mobility profile.

        Args:
            location_types: Available location type names.
            rng: Random number generator.

        Returns:
            List of location type strings visited.
        """
        visits = []
        for loc in location_types:
            prob = self.daily_mobility_profile.get(loc, 0.0)
            if rng.random() < prob:
                visits.append(loc)
        return visits
        pass


@dataclass
class Household:
    """
    Household of multiple persons pooling resources and norms.

    Attributes:
        id: Household ID.
        member_ids: List of member person IDs.
        household_income: Aggregate income proxy in [0,1].
        norm_strength: Internal social norm strength [0,1].
        mask_inventory: Shared mask inventory count.
    """
    id: int
    member_ids: List[int]
    household_income: float
    norm_strength: float
    mask_inventory: int = 0

    def share_norms(self, persons: Dict[int, Person]) -> None:
        """
        Influence member attitudes toward the household norm average.

        Args:
            persons: Mapping of person id to Person.

        Returns:
            None
        """
        if not self.member_ids:
            return
        avg_att = sum(persons[i].mask_attitude for i in self.member_ids) / len(self.member_ids)
        for i in self.member_ids:
            p = persons[i]
            p.mask_attitude = clamp01(((p.mask_attitude + 1.0) / 2.0) * (1.0 - 0.2 * self.norm_strength) + ((avg_att + 1.0) / 2.0) * 0.2 * self.norm_strength) * 2.0 - 1.0
        pass

    def pool_resources_for_masks(self, persons: Dict[int, Person]) -> None:
        """
        Move some masks from household inventory to members lacking masks.

        Args:
            persons: Mapping of person id to Person.

        Returns:
            None
        """
        # Simple redistribution: ensure each has at least one if possible.
        for i in self.member_ids:
            if self.mask_inventory <= 0:
                break
            if persons[i].masks_inventory <= 0:
                persons[i].masks_inventory += 1
                self.mask_inventory -= 1
        pass


@dataclass
class Location:
    """
    Physical or virtual location in the simulation.

    Attributes:
        id: Unique ID.
        type: Type string (home, work, school, store, public_transport, other_public).
        capacity: Maximum occupancy.
        mask_policy: 'none', 'recommended', or 'mandate'.
        enforcement_level: Level [0,1] indicating inspection likelihood.
        foot_traffic_rate: Approximate daily traffic factor [0,1].
    """
    id: int
    type: str
    capacity: int
    mask_policy: str
    enforcement_level: float
    foot_traffic_rate: float

    def admit_visitors(self, visitor_ids: List[int]) -> List[int]:
        """
        Admit visitors up to capacity.

        Args:
            visitor_ids: Candidate visitor IDs.

        Returns:
            List of admitted visitor IDs.
        """
        if len(visitor_ids) <= self.capacity:
            return visitor_ids
        return visitor_ids[: self.capacity]
        pass

    def enforce_mask_policy(self, visitor_persons: List[Person], authority: "HealthAuthority", rng: random.Random) -> int:
        """
        Enforce mask policy by inspecting a subset of visitors. Non-compliant individuals may be fined.

        Args:
            visitor_persons: Persons visiting the location.
            authority: HealthAuthority instance.
            rng: Random number generator.

        Returns:
            Count of enforcement actions (e.g., warnings/fines).
        """
        if self.mask_policy != "mandate" or self.enforcement_level <= 0.0:
            return 0
        actions = 0
        inspect_prob = clamp01(self.enforcement_level * authority.enforcement_intensity)
        for p in visitor_persons:
            if not p.wearing_mask and rng.random() < inspect_prob:
                actions += 1
                # Effect: increase baseline compliance and risk perception marginally
                p.baseline_compliance_propensity = clamp01(p.baseline_compliance_propensity + 0.05)
                p.risk_perception = clamp01(p.risk_perception + 0.02)
        return actions
        pass

    def broadcast_signage(self) -> float:
        """
        Return a signage influence boost based on policy.

        Returns:
            A small positive float boost to observation-based adoption.
        """
        if self.mask_policy == "mandate":
            return 0.05
        if self.mask_policy == "recommended":
            return 0.02
        return 0.0
        pass


@dataclass
class HealthAuthority:
    """
    Represents a policy maker or health authority.

    Attributes:
        id: Identifier.
        mandate_on: Boolean flag for mandate status.
        mandate_start_day: Day mandate starts.
        mandate_scope: Scope string for policy.
        fine_amount: Amount of fine for non-compliance.
        enforcement_resources: Resource proxy [0,1].
        enforcement_intensity: Scaling for enforcement probability [0,1].
        guidance_strength: Public guidance effect [0,1].
    """
    id: int
    mandate_on: bool
    mandate_start_day: int
    mandate_scope: str
    fine_amount: float
    enforcement_resources: float
    enforcement_intensity: float = 0.3
    guidance_strength: float = 0.3

    def set_or_update_policy(self, day: int) -> None:
        """
        Update mandate status based on the day.

        Args:
            day: Current day.

        Returns:
            None
        """
        if day >= self.mandate_start_day:
            self.mandate_on = True
        pass

    def allocate_enforcement(self, locations: List[Location]) -> None:
        """
        Distribute enforcement resources across locations by adjusting enforcement levels.

        Args:
            locations: List of Location instances.

        Returns:
            None
        """
        if not locations:
            return
        base = clamp01(self.enforcement_resources)
        for loc in locations:
            if loc.mask_policy == "mandate":
                loc.enforcement_level = clamp01(base * (0.5 + 0.5 * loc.foot_traffic_rate))
            else:
                loc.enforcement_level = clamp01(0.2 * base * loc.foot_traffic_rate)
        pass

    def issue_public_guidance(self) -> Dict[str, float]:
        """
        Create a guidance effect vector for the population.

        Returns:
            Dict with keys affecting beliefs: risk, trust_gov, attitude.
        """
        return {
            "risk": 0.01 * self.guidance_strength,
            "trust_gov": 0.01 * self.guidance_strength,
            "attitude": 0.005 * self.guidance_strength,
        }
        pass


@dataclass
class InformationSource:
    """
    Media or information source broadcasting messages.

    Attributes:
        id: Identifier.
        credibility: How trusted it is [0,1].
        message_intensity: Daily message intensity [0,1].
        message_slant: Pro-mask slant in [-1,1].
        reach: Fraction of population reached daily [0,1].
        misinformation_rate: Probability that a message reduces attitude.
        campaign_intensity: Campaign spending/proxy for intensity [0,1].
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
        Generate a message delta affecting risk and attitudes.

        Args:
            rng: Random number generator.

        Returns:
            Dict with deltas for 'risk' and 'attitude'.
        """
        sign = 1.0 if self.message_slant >= 0 else -1.0
        misinf = 1.0 if rng.random() < self.misinformation_rate else 0.0
        # If misinformation triggers, invert slant impact on attitude
        attitude_delta = (0.01 * self.message_intensity * sign * self.credibility) * (1.0 - 2.0 * misinf)
        risk_delta = 0.01 * self.message_intensity * max(0.0, sign) * self.credibility
        return {"risk": risk_delta, "attitude": attitude_delta}
        pass


@dataclass
class Retailer:
    """
    Retailer that supplies masks.

    Attributes:
        id: Identifier.
        inventory_level: Current inventory.
        restock_rate: Fractional restock rate per day.
        price_per_mask: Current price.
        rationing_policy: Max masks per customer per visit.
    """
    id: int
    inventory_level: int
    restock_rate: float
    price_per_mask: float
    rationing_policy: int = 2

    def sell_masks(self, quantity: int) -> Tuple[int, bool]:
        """
        Sell masks if inventory is available and rationing allows.

        Args:
            quantity: Quantity requested.

        Returns:
            Tuple of (quantity_sold, shortage_flag)
        """
        quantity = int(max(0, min(self.rationing_policy, quantity)))
        if self.inventory_level <= 0:
            return 0, True
        sold = min(self.inventory_level, quantity)
        self.inventory_level -= sold
        shortage = sold < quantity
        return sold, shortage
        pass

    def adjust_prices(self) -> None:
        """
        Adjust price based on inventory using a simple inverse relation.

        Returns:
            None
        """
        base = 1.0
        scarcity = 0.0 if self.inventory_level > 500 else (1.0 - (self.inventory_level / 500.0))
        self.price_per_mask = base * (1.0 + 0.5 * scarcity)
        pass

    def restock(self, capacity: int = 1000) -> None:
        """
        Restock inventory up to a capacity based on restock rate.

        Args:
            capacity: Maximum capacity.

        Returns:
            None
        """
        restock_qty = int(max(0, (capacity - self.inventory_level) * self.restock_rate))
        self.inventory_level += restock_qty
        pass


class Simulation:
    """
    Main simulation class coordinating entities, interactions, and metrics.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the simulation environment and entities from parameters.

        Args:
            params: Configuration parameters.

        Returns:
            None
        """
        # FIXED: Restore minimal but complete simulation configuration
        self.params = params
        self.rng = random.Random(int(params.get("random_seed", 42)))
        self.population_size = int(params.get("population_size", 500))
        self.time_horizon_days = int(params.get("time_horizon_days", 60))
        self.avg_degree = int(params.get("avg_degree", params.get("average_degree", 8)))
        self.rewiring_prob = float(params.get("rewiring_prob", 0.05))
        self.initial_adoption_rate = float(params.get("initial_adoption_rate", 0.1))
        self.mask_availability_initial = float(params.get("mask_availability_initial", params.get("mask_availability", 0.9)))
        self.retailer_restock_rate = float(params.get("retailer_restock_rate", 0.1))
        self.mask_price = float(params.get("mask_price", 1.0))
        self.price_elasticity = float(params.get("price_elasticity_of_demand", -0.2))
        self.risk_signal_initial = float(params.get("risk_signal_initial", 0.2))
        self.risk_signal_amplitude = float(params.get("risk_signal_amplitude", 0.3))
        self.risk_signal_decay = float(params.get("risk_signal_decay", 0.01))
        self.compliance_decay_rate = float(params.get("compliance_decay_rate", 0.005))

        self.weights = {
            "social_influence_weight": float(params.get("social_influence_weight", params.get("social_influence_strength", 0.5))),
            "personal_attitude_weight": float(params.get("personal_attitude_weight", 0.3)),
            "policy_compliance_weight": float(params.get("policy_compliance_weight", params.get("government_guidance_influence", 0.35))),
            "observation_weight": float(params.get("observation_weight", params.get("observation_effect", 0.4))),
        }

        # Entities containers
        self.people: Dict[int, Person] = {}
        self.households: Dict[int, Household] = {}
        self.locations: List[Location] = []
        self.authority = HealthAuthority(
            id=0,
            mandate_on=False,
            mandate_start_day=int(params.get("mandate_start_day", 30)),
            mandate_scope=str(params.get("mandate_scope", "indoor_public")),
            fine_amount=float(params.get("fine_amount", 50.0)),
            enforcement_resources=float(params.get("enforcement_level", 0.3)),
            enforcement_intensity=float(params.get("enforcement_level", 0.3)),
            guidance_strength=float(params.get("government_guidance_influence", 0.35)),
        )
        self.media = InformationSource(
            id=0,
            credibility=float(params.get("media_credibility", 0.7)),
            message_intensity=float(params.get("media_influence_strength", 0.3)),
            message_slant=float(params.get("media_message_slant", 1.0)),
            reach=float(params.get("media_reach", 0.6)),
            misinformation_rate=float(params.get("misinformation_rate", 0.05)),
            campaign_intensity=float(params.get("campaign_intensity", 0.2)),
        )
        self.retailer = Retailer(
            id=0,
            inventory_level=int(params.get("initial_inventory", 1000)),
            restock_rate=self.retailer_restock_rate,
            price_per_mask=self.mask_price,
            rationing_policy=int(params.get("rationing_policy", 2)),
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
            {"home": 1.0, "work": 0.6, "school": 0.5, "store": 0.2, "public_transport": 0.15, "other_public": 0.3},
        )
        loc_id = 0
        for t in loc_types:
            # Create a few instances per type
            count = {"home": self.population_size // 4, "work": self.population_size // 10, "school": self.population_size // 15}.get(t, 3)
            for _ in range(max(1, count)):
                policy = "none"
                self.locations.append(
                    Location(
                        id=loc_id,
                        type=t,
                        capacity=max(10, int(self.population_size * 0.1)),
                        mask_policy=policy,
                        enforcement_level=0.0,
                        foot_traffic_rate=float(foot_mult.get(t, 0.3)),
                    )
                )
                loc_id += 1

        # People and households
        self._initialize_people_and_households()

        # Metrics
        self.metrics: Dict[str, Any] = {
            "adoption_rate_over_time": [],
            "compliance_over_time": [],
            "churn_over_time": [],
            "shortage_incidents": 0,
            "enforcement_actions_count": 0,
        }
        pass

    def _initialize_people_and_households(self) -> None:
        """
        Initialize the population and households with heterogeneity and initial states.

        Returns:
            None
        """
        n = self.population_size
        hetero_var = float(self.params.get("heterogeneity_variance", 0.2))
        # Households
        hh_id = 0
        i = 0
        while i < n:
            size = max(1, min(5, int(self.rng.gauss(2.5, 1.0))))
            member_ids = list(range(i, min(n, i + size)))
            income = clamp01(self.rng.random() * (0.7 + 0.3 * self.rng.random()))
            norm_strength = clamp01(self.rng.random())
            self.households[hh_id] = Household(hh_id, member_ids, income, norm_strength, mask_inventory=self.rng.randint(0, 3))
            for pid in member_ids:
                if pid >= n:
                    break
                age = max(0, int(self.rng.gauss(40, 18)))
                ses = clamp01(0.5 + hetero_var * (self.rng.random() - 0.5))
                risk = clamp01(self.risk_signal_initial + hetero_var * (self.rng.random() - 0.5))
                trust_gov = clamp01(0.5 + hetero_var * (self.rng.random() - 0.5))
                trust_media = clamp01(0.5 + hetero_var * (self.rng.random() - 0.5))
                compliance = clamp01(0.5 + hetero_var * (self.rng.random() - 0.5))
                attitude = clamp01(0.5 + hetero_var * (self.rng.random() - 0.5)) * 2.0 - 1.0
                has_mask = self.rng.random() < self.mask_availability_initial
                wearing = has_mask and (self.rng.random() < self.initial_adoption_rate)
                inv = self.rng.randint(0, 2) + (1 if has_mask else 0)
                mobility = {
                    "home": 1.0,
                    "work": 0.6 if age >= 22 and age < 65 else 0.1,
                    "school": 0.5 if age < 22 else 0.0,
                    "store": 0.2,
                    "public_transport": 0.15 if ses < 0.5 else 0.05,
                    "other_public": 0.3,
                }
                susceptibility = clamp01(0.5 + hetero_var * (self.rng.random() - 0.5))
                self.people[pid] = Person(
                    id=pid,
                    age=age,
                    household_id=hh_id,
                    socioeconomic_status=ses,
                    risk_perception=risk,
                    trust_in_government=trust_gov,
                    trust_in_media=trust_media,
                    baseline_compliance_propensity=compliance,
                    mask_attitude=attitude,
                    wearing_mask=wearing,
                    masks_inventory=inv,
                    social_network_neighbors=self.network[pid],
                    daily_mobility_profile=mobility,
                    social_susceptibility=susceptibility,
                )
            i += size
            hh_id += 1
        pass

    def step(self, day: int) -> None:
        """
        Advance the simulation by one day: policy update, media broadcast, purchases, decisions, enforcement, and metrics.

        Args:
            day: Current day index.

        Returns:
            None
        """
        # Policy updates
        self.authority.set_or_update_policy(day)  # FIXED: Implement policy progression

        # Update location policies based on mandate
        for loc in self.locations:
            if self.authority.mandate_on and loc.type in ("work", "school", "store", "public_transport", "other_public"):
                loc.mask_policy = "mandate"
            else:
                loc.mask_policy = "recommended" if loc.type != "home" else "none"

        # Allocate enforcement
        self.authority.allocate_enforcement(self.locations)

        # Media and guidance
        media_effect = self.media.broadcast_messages(self.rng)
        guidance_effect = self.authority.issue_public_guidance()
        combined_effect = {
            "risk": media_effect.get("risk", 0.0) + guidance_effect.get("risk", 0.0),
            "trust_gov": guidance_effect.get("trust_gov", 0.0),
            "attitude": media_effect.get("attitude", 0.0) + guidance_effect.get("attitude", 0.0),
        }

        # Apply belief updates
        for p in self.people.values():
            # Only a fraction reached by media
            if self.rng.random() < self.media.reach:
                p.update_beliefs_from_media(combined_effect)

        # Retailer operations
        self.retailer.adjust_prices()
        self.retailer.restock(capacity=int(self.params.get("retailer_capacity", 2000)))

        # Observations
        previous_wearing = [self.people[i].wearing_mask for i in range(self.population_size)]
        prev_adoption = sum(previous_wearing) / max(1, self.population_size)

        # Household norms and pooling
        for hh in self.households.values():
            hh.share_norms(self.people)
            hh.pool_resources_for_masks(self.people)

        # Purchases for those who want stock (simple heuristic)
        daily_shortage = False
        for p in self.people.values():
            if p.masks_inventory <= 0 and self.rng.random() < 0.5:
                _, shortage = p.purchase_masks(self.retailer, max_purchase=2)
                daily_shortage = daily_shortage or shortage

        if daily_shortage:
            self.metrics["shortage_incidents"] += 1

        # Social influence: compute peer rates
        wearing_flags = {pid: self.people[pid].wearing_mask for pid in self.people}
        peer_rates = {}
        for pid, p in self.people.items():
            neigh = p.social_network_neighbors
            if not neigh:
                peer_rates[pid] = prev_adoption
            else:
                rate = sum(wearing_flags[n] for n in neigh) / max(1, len(neigh))
                peer_rates[pid] = rate

        # Observed public wearing approximate
        observed = prev_adoption

        # Decisions
        new_wearing = {}
        for pid, p in self.people.items():
            policy_strength = 1.0 if self.authority.mandate_on else 0.3
            # Enforcement probability approximated from average across locations
            avg_enforce = sum(loc.enforcement_level for loc in self.locations if loc.mask_policy == "mandate")
            avg_enforce /= max(1, len([1 for loc in self.locations if loc.mask_policy == "mandate"]))
            enforcement_probability = clamp01(avg_enforce)
            market_availability = clamp01(self.retailer.inventory_level / max(1.0, float(self.params.get("retailer_capacity", 2000))))
            decision = p.decide_to_wear_mask(
                peer_adoption_rate=peer_rates[pid],
                observed_public_wearing=observed + 0.02,  # signage minor boost
                policy_strength=policy_strength,
                enforcement_probability=enforcement_probability,
                market_availability=market_availability,
                price_per_mask=self.retailer.price_per_mask,
                weights=self.weights,
                rng=self.rng,
            )
            p.wearing_mask = decision
            # consume a mask if wearing
            if decision and p.masks_inventory > 0 and self.rng.random() < 0.7:
                p.masks_inventory -= 1
            new_wearing[pid] = decision

        # Enforcement at locations: simulate visits and inspections
        visitors_by_location: Dict[int, List[int]] = {}
        loc_types = list({loc.type for loc in self.locations})
        # map type to ids
        loc_ids_by_type: Dict[str, List[int]] = {}
        for loc in self.locations:
            loc_ids_by_type.setdefault(loc.type, []).append(loc.id)

        for pid, p in self.people.items():
            visits = p.travel_and_interact_at_locations(loc_types, self.rng)
            for v_type in visits:
                ids = loc_ids_by_type.get(v_type, [])
                if not ids:
                    continue
                chosen = self.rng.choice(ids)
                visitors_by_location.setdefault(chosen, []).append(pid)

        enforcement_actions = 0
        for loc in self.locations:
            visitor_ids = visitors_by_location.get(loc.id, [])
            admitted_ids = loc.admit_visitors(visitor_ids)
            visitor_persons = [self.people[i] for i in admitted_ids]
            enforcement_actions += loc.enforce_mask_policy(visitor_persons, self.authority, self.rng)
        self.metrics["enforcement_actions_count"] += enforcement_actions

        # Peer belief updates based on attitudes
        for pid, p in self.people.items():
            neighbor_attitudes = [self.people[n].mask_attitude for n in p.social_network_neighbors]
            p.update_beliefs_from_peers(neighbor_attitudes)
            p.respond_to_policy(self.compliance_decay_rate)

        # Metrics per day
        adoption = sum(1 for p in self.people.values() if p.wearing_mask) / max(1, self.population_size)
        self.metrics["adoption_rate_over_time"].append(adoption)
        if self.authority.mandate_on:
            self.metrics["compliance_over_time"].append(adoption)
        switches = sum(1 for i in range(self.population_size) if previous_wearing[i] != new_wearing[i])
        churn = switches / max(1, self.population_size)
        self.metrics["churn_over_time"].append(churn)
        pass

    def run(self) -> Dict[str, Any]:
        """
        Run the full simulation loop over the configured time horizon.

        Returns:
            Dictionary

# Execute main for both direct execution and sandbox wrapper invocation
main()
"""