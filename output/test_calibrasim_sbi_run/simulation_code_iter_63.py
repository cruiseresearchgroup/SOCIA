def main():
    pass

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Set

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT if PROJECT_ROOT else "", DATA_PATH if DATA_PATH else "")


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a value x to the closed interval [lo, hi].

    Args:
        x: Input value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        float: Clamped value.
    """
    pass
    val = max(lo, min(hi, x))
    return val


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.

    Args:
        n: Numerator.
        d: Denominator.
        default: Value to return when denominator equals zero.

    Returns:
        float: Result of division or default.
    """
    pass
    if d == 0:
        return default
    return n / d


def sigmoid(x: float) -> float:
    """
    Numerically stable logistic sigmoid.

    Args:
        x: Input value.

    Returns:
        float: Sigmoid(x) in [0,1].
    """
    pass
    try:
        if x >= 0:
            z = math.exp(-x)
            val = 1.0 / (1.0 + z)
            return val
        else:
            z = math.exp(x)
            val = z / (1.0 + z)
            return val
    except OverflowError:
        val = 0.0 if x < 0 else 1.0
        return val


def ring_small_world(n: int, k: int, p: float, rng: random.Random) -> Dict[int, set]:
    """
    Build a ring-lattice small-world network with robust rewiring.

    Args:
        n: Number of nodes.
        k: Average degree target (even integer preferred).
        p: Rewiring probability.
        rng: Random state.

    Returns:
        dict: Adjacency sets for each node.
    """
    pass
    # FIXED: Early-return for degenerate networks to avoid spurious edges/self-loops.
    if n <= 1 or k <= 0:
        return {i: set() for i in range(max(0, n))}
    # Base ring lattice
    adj = {i: set() for i in range(n)}
    half = max(1, min(k // 2, (n - 1) // 2))
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
    # FIXED: Guard rewiring so it only applies to existing edges and avoids unintended degree inflation.
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rng.random() < p:
                if j not in adj[i]:
                    continue
                adj[i].discard(j)
                adj[j].discard(i)
                for _ in range(20):
                    u = rng.randrange(n)
                    if u != i and u not in adj[i]:
                        adj[i].add(u)
                        adj[u].add(i)
                        break
                else:
                    adj[i].add(j)
                    adj[j].add(i)
    return adj


class Person:
    """
    Represents an individual agent with social, economic, and behavioral attributes.

    Attributes:
        pid (int): Unique person identifier.
        age (int): Age in years.
        income (float): Annual income proxy.
        ses_quintile (int): Socioeconomic status quintile [1..5].
        risk (float): Perceived disease risk [0,1].
        trust (float): Trust in institutions/authority [0,1].
        compliance_trait (float): Compliance propensity trait [0,1].
        attitude (float): Attitudinal leaning toward adoption [-1,1].
        threshold (float): Social threshold for adoption [0,1].
        baseline (float): Baseline adoption propensity [0,1].
        fatigue (float): Fatigue from sustained adoption [0,1].
        habit (float): Habit strength [0,1].
        adopting (bool): Whether currently adopting the behavior.
        mask_inventory (int): On-hand inventory of masks (legacy aggregate).
        media_subscribed (bool): High exposure to legacy media outlet.
        fines_paid (float): Cumulative fines paid.
        disease_state (str): One of 'S','E','I','R'.
        days_in_state (int): Days elapsed in current disease state.
        household_id (int): Household identifier.
        mask_type (Optional[str]): Preferred mask type ('cloth','surgical','N95').
        mask_inventory_by_type (Dict[str,int]): Inventory per mask type.
        cost_sensitivity (float): Price sensitivity [0,1].
        media_consumption_profile (Dict[str,float]): InformationSource weights.
    """
    pass

    def __init__(
        self,
        pid: int,
        age: int,
        income: float,
        risk: float,
        trust: float,
        compliance_trait: float,
        attitude: float,
        threshold: float,
        baseline: float,
        media_subscribed: bool,
        household_id: int,
        cost_sensitivity: float,
        media_consumption_profile: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize a Person.

        Args:
            pid: Person id.
            age: Age.
            income: Income proxy.
            risk: Risk perception in [0,1].
            trust: Trust in [0,1].
            compliance_trait: Compliance trait [0,1].
            attitude: Attitude [-1,1].
            threshold: Social threshold [0,1].
            baseline: Baseline adoption propensity [0,1].
            media_subscribed: If true, high media exposure.
            household_id: Household membership id.
            cost_sensitivity: Price sensitivity [0,1].
            media_consumption_profile: Mapping source_id -> weight [0,1].

        Returns:
            None
        """
        pass
        self.pid = pid
        self.age = age
        self.income = income
        self.ses_quintile = 3  # will be assigned after population creation
        self.risk = clamp(risk, 0.0, 1.0)
        self.trust = clamp(trust, 0.0, 1.0)
        self.compliance_trait = clamp(compliance_trait, 0.0, 1.0)
        self.attitude = clamp(attitude, -1.0, 1.0)
        self.threshold = clamp(threshold, 0.0, 1.0)
        self.baseline = clamp(baseline, 0.0, 1.0)
        self.fatigue = 0.0
        self.habit = 0.0
        self.adopting = False
        self.mask_inventory = 0
        self.media_subscribed = media_subscribed
        self.fines_paid = 0.0
        # Disease module fields
        self.disease_state = 'S'
        self.days_in_state = 0
        # Extended per spec
        self.household_id = household_id
        self.mask_type: Optional[str] = None
        self.mask_inventory_by_type: Dict[str, int] = {"cloth": 0, "surgical": 0, "N95": 0}
        self.cost_sensitivity = clamp(cost_sensitivity, 0.0, 1.0)
        self.media_consumption_profile = media_consumption_profile or {}


class Household:
    """
    Household entity capturing shared inventory and norms.

    Attributes:
        hid (int): Household id.
        member_ids (List[int]): Members.
        norm_strength (float): Normative influence weight [0,1].
        shared_inventory_by_type (Dict[str,int]): Shared stock per type.
    """
    pass

    def __init__(self, hid: int, member_ids: List[int], norm_strength: float = 0.5):
        """
        Initialize Household.

        Args:
            hid: Household id.
            member_ids: List of member person IDs.
            norm_strength: Influence weight [0,1].

        Returns:
            None
        """
        pass
        self.hid = hid
        self.member_ids = list(member_ids)
        self.norm_strength = clamp(norm_strength, 0.0, 1.0)
        self.shared_inventory_by_type: Dict[str, int] = {"cloth": 0, "surgical": 0, "N95": 0}


class Location:
    """
    A location that agents may visit; may enforce policy via fines or entry refusal.

    Attributes:
        name (str): Location identifier.
        enforcement_level (float): Probability of enforcement when mandate active.
        mandate_sensitive (bool): Whether enforcement ties to mandate.
        entry_refusal_if_unmasked (bool): If True, may refuse entry to unmasked visitors.
        observed_norms (float): Share of masked among recent visitors.
        ltype (str): Location type label (e.g., 'Transit','Workplace','Park').
        capacity (int): Maximum simultaneous visitors per day snapshot.
        mask_policy (bool): Whether location policy requires masks.
        foot_traffic_rate (float): Relative rate for visit sampling.
    """
    pass

    def __init__(
        self,
        name: str,
        enforcement_level: float = 0.0,
        mandate_sensitive: bool = True,
        entry_refusal_if_unmasked: bool = False,
        ltype: str = "generic",
        capacity: int = 50,
        mask_policy: bool = False,
        foot_traffic_rate: float = 0.25,
    ):
        """
        Initialize a Location.

        Args:
            name: Name of the location.
            enforcement_level: Probability of enforcement when mandate active.
            mandate_sensitive: True if enforcement applies during mandates.
            entry_refusal_if_unmasked: If True, location may refuse entry to unmasked visitors.
            ltype: Location type label.
            capacity: Capacity of the location per day snapshot.
            mask_policy: Whether masks are required at this location.
            foot_traffic_rate: Relative rate for visit sampling.

        Returns:
            None
        """
        pass
        # FIXED: Added capacity, mask_policy, and foot_traffic_rate per feedback.
        self.name = name
        self.enforcement_level = clamp(enforcement_level, 0.0, 1.0)
        self.mandate_sensitive = mandate_sensitive
        self.entry_refusal_if_unmasked = entry_refusal_if_unmasked
        self.observed_norms = 0.0
        self.ltype = ltype
        self.capacity = max(1, int(capacity))
        self.mask_policy = bool(mask_policy)
        self.foot_traffic_rate = clamp(foot_traffic_rate, 0.0, 1.0)


class PolicyAuthority:
    """
    PolicyAuthority controlling policy schedule, enforcement, and communication.

    Attributes:
        days (List[int]): Days on which state changes.
        states (List[str]): Policy states corresponding to days.
        enforcement_level_default (float): Base enforcement level.
        communication_strategy (str): Strategy label.
        fine_amount (float): Fine per violation.
        subsidy_amount (float): Per-mask subsidy reducing price.
    """
    pass

    def __init__(
        self,
        policy_schedule: Dict[str, List[Any]],
        enforcement_level_default: float = 0.4,
        communication_strategy: str = "neutral",
        fine_amount: float = 50.0,
        subsidy_amount: float = 0.0,
    ):
        """
        Initialize PolicyAuthority with a policy schedule.

        Args:
            policy_schedule: Dict with keys 'day' and 'state'; state in {'none','recommendation','mandate'}.
            enforcement_level_default: Default enforcement intensity.
            communication_strategy: Messaging strategy name.
            fine_amount: Fine per violation.
            subsidy_amount: Per-unit subsidy on masks.

        Returns:
            None
        """
        pass
        self.days = list(policy_schedule.get("day", []))
        self.states = list(policy_schedule.get("state", []))
        self.enforcement_level_default = clamp(enforcement_level_default, 0.0, 1.0)
        self.communication_strategy = communication_strategy
        self.fine_amount = max(0.0, float(fine_amount))
        self.subsidy_amount = max(0.0, float(subsidy_amount))

    def current_state(self, day: int) -> str:
        """
        Get current policy state for a given day.

        Args:
            day: Day index.

        Returns:
            str: Current state label.
        """
        pass
        idx = -1
        for i, d in enumerate(self.days):
            if d <= day:
                idx = i
            else:
                break
        return self.states[idx] if idx >= 0 else "none"

    def policy_intensity(self, day: int) -> float:
        """
        Map policy state to continuous intensity.

        Args:
            day: Day index.

        Returns:
            float: Intensity in [0,1].
        """
        pass
        state = self.current_state(day)
        return 1.0 if state == "mandate" else 0.5 if state == "recommendation" else 0.0

    def enforcement_level(self, day: int) -> float:
        """
        Enforcement intensity for given day.

        Args:
            day: Day index.

        Returns:
            float: Enforcement level in [0,1].
        """
        pass
        state = self.current_state(day)
        return self.enforcement_level_default if state in ("mandate", "recommendation") else 0.0

    def first_mandate_day(self) -> Optional[int]:
        """
        Return the first day when state becomes 'mandate', if any.

        Returns:
            Optional[int]: Day index or None.
        """
        pass
        for d, s in zip(self.days, self.states):
            if s == "mandate":
                return d
        return None


class InformationSource:
    """
    Information source broadcasting messages with credibility and reach.

    Attributes:
        sid (str): Source id.
        message_type (str): 'pro_mask' or 'anti_mask'.
        credibility (float): Credibility weight [0,1].
        reach (float): Reach probability [0,1].
        misinformation_prob (float): Probability of anti-mask/misleading spin.
    """
    pass

    def __init__(self, sid: str, message_type: str, credibility: float, reach: float, misinformation_prob: float = 0.0):
        """
        Initialize InformationSource.

        Args:
            sid: Source id.
            message_type: 'pro_mask' or 'anti_mask'.
            credibility: Credibility in [0,1].
            reach: Reach probability in [0,1].
            misinformation_prob: Probability of misinformation.

        Returns:
            None
        """
        pass
        self.sid = sid
        self.message_type = message_type
        self.credibility = clamp(credibility, 0.0, 1.0)
        self.reach = clamp(reach, 0.0, 1.0)
        self.misinformation_prob = clamp(misinformation_prob, 0.0, 1.0)

    def broadcast_message(self, day: int, persons: List["Person"], rng: random.Random, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast messages to persons; update perceived risk and potentially trust.

        Args:
            day: Day index.
            persons: List of Person.
            rng: Random state.
            params: Global parameter dict.

        Returns:
            Dict with exposure statistics including exposed_ids set.
        """
        pass
        exposures = 0
        exposed_ids: Set[int] = set()
        pro_sign = 1.0 if self.message_type == "pro_mask" else -1.0
        msg_effect_size = float(params.get("message_effect_size", 0.3))
        for p in persons:
            weight = float(p.media_consumption_profile.get(self.sid, 0.0))
            if weight <= 0:
                continue
            if rng.random() < self.reach * weight:
                exposures += 1
                exposed_ids.add(p.pid)
                misinformation_hit = rng.random() < self.misinformation_prob
                sign = pro_sign if not misinformation_hit else -pro_sign
                delta = msg_effect_size * sign * self.credibility * (0.5 + 0.5 * p.trust)
                p.risk = clamp(p.risk + delta, 0.0, 1.0)
                # Slight trust adjustment: consistent pro messages increase trust, anti decrease
                p.trust = clamp(p.trust + 0.01 * sign * self.credibility, 0.0, 1.0)
        return {"source_id": self.sid, "exposures": exposures, "exposed_ids": exposed_ids}


class MaskMarket:
    """
    Mask market with type-specific inventory, pricing, and rationing.

    Attributes:
        types_available (List[str]): Types available.
        price_by_type (Dict[str,float]): Price per type.
        inventory_by_type (Dict[str,int]): Units in stock per type.
        restock_rate_by_type (Dict[str,int]): Daily restock per type.
        rationing_rules (Dict[str,int]): Per-type per-transaction limit.
    """
    pass

    def __init__(
        self,
        types_available: List[str],
        price_by_type: Dict[str, float],
        initial_supply_by_type: Dict[str, int],
        restock_rate_by_type: Dict[str, int],
        rationing_rules: Dict[str, int],
        price_floor: float = 0.1,
        price_ceiling: float = 10.0,
    ):
        """
        Initialize MaskMarket.

        Args:
            types_available: Mask types.
            price_by_type: Starting price per type.
            initial_supply_by_type: Initial inventory per type.
            restock_rate_by_type: Daily restock rate per type.
            rationing_rules: Max units sold per purchase per type.
            price_floor: Min price clamp.
            price_ceiling: Max price clamp.

        Returns:
            None
        """
        pass
        self.types_available = list(types_available)
        self.price_by_type = {t: max(price_floor, float(price_by_type.get(t, 1.0))) for t in self.types_available}
        self.inventory_by_type = {t: max(0, int(initial_supply_by_type.get(t, 0))) for t in self.types_available}
        self.restock_rate_by_type = {t: max(0, int(restock_rate_by_type.get(t, 0))) for t in self.types_available}
        self.rationing_rules = {t: max(1, int(rationing_rules.get(t, 5))) for t in self.types_available}
        self.price_floor = float(price_floor)
        self.price_ceiling = float(price_ceiling)
        self.outage_days_by_type: Dict[str, int] = {t: 0 for t in self.types_available}

    def restock(self) -> None:
        """
        Restock inventory by restock_rate for each type.

        Returns:
            None
        """
        pass
        for t in self.types_available:
            self.inventory_by_type[t] += int(self.restock_rate_by_type.get(t, 0))

    def sell_masks(self, mask_type: str, desired_qty: int) -> int:
        """
        Sell up to desired quantity of a given type, limited by rationing and inventory.

        Args:
            mask_type: Type requested.
            desired_qty: Desired units.

        Returns:
            int: Units sold.
        """
        pass
        inv = int(self.inventory_by_type.get(mask_type, 0))
        limit = int(self.rationing_rules.get(mask_type, desired_qty))
        qty = max(0, min(desired_qty, limit, inv))
        self.inventory_by_type[mask_type] = inv - qty
        return qty

    def consumer_price(self, mask_type: str, subsidy_amount: float = 0.0) -> float:
        """
        Get the consumer price after subsidy.

        Args:
            mask_type: Type.
            subsidy_amount: Per-unit subsidy.

        Returns:
            float: Effective consumer price.
        """
        pass
        base = float(self.price_by_type.get(mask_type, 1.0))
        price = max(self.price_floor, base - max(0.0, subsidy_amount))
        return price

    def end_of_day_outages(self) -> int:
        """
        Count out-of-stock types and increment outage days per type.

        Returns:
            int: Number of types out of stock today.
        """
        pass
        count_types = 0
        for t in self.types_available:
            if self.inventory_by_type.get(t, 0) <= 0:
                self.outage_days_by_type[t] = self.outage_days_by_type.get(t, 0) + 1
                count_types += 1
        return count_types


class Media:
    """
    Media outlet broadcasting a daily intensity signal affecting adoption.

    Attributes:
        baseline_intensity (float): Base intensity.
        shock_day (int): Day of media shock.
        shock_magnitude (float): Additive magnitude at shock day.
    """
    pass

    def __init__(self, baseline_intensity: float, shock_day: int, shock_magnitude: float):
        """
        Initialize Media.

        Args:
            baseline_intensity: Baseline signal intensity.
            shock_day: Day of a one-time shock.
            shock_magnitude: Added intensity at shock day.

        Returns:
            None
        """
        pass
        self.baseline_intensity = clamp(baseline_intensity, 0.0, 2.0)
        self.shock_day = max(0, shock_day)
        self.shock_magnitude = clamp(shock_magnitude, 0.0, 2.0)

    def signal(self, day: int, subscribed: bool) -> float:
        """
        Compute media signal for a person on a given day.

        Args:
            day: Current day.
            subscribed: Whether the person is highly exposed/subscribed.

        Returns:
            float: Media signal intensity.
        """
        pass
        base = self.baseline_intensity
        if day == self.shock_day:
            base += self.shock_magnitude
        intensity = base if subscribed else 0.5 * base
        return intensity


def adoption_probability(
    person: Person,
    peer_share: float,
    policy_intensity: float,
    media_signal: float,
    params: Dict[str, float],
    rng: random.Random,
) -> float:
    """
    Compute adoption probability for a person using parameterized weights.

    Args:
        person: Person instance.
        peer_share: Fraction of neighbors adopting.
        policy_intensity: Policy intensity [0,1].
        media_signal: Media signal intensity.
        params: Parameter dict with weights.
        rng: Random number generator for reproducibility.

    Returns:
        float: Adoption probability in [0,1].
    """
    pass
    # FIXED: Parameterized adoption probability per feedback; replaces hardcoded weights.
    # FIXED: Ensured reproducibility by removing global random and using rng parameter.
    w_social = float(params.get("base_influence_strength", 0.3))
    w_risk = float(params.get("risk_perception_weight", 0.4))
    w_policy = float(params.get("policy_effect_strength", 0.5))
    w_media = float(params.get("info_campaign_intensity", 0.2))
    fatigue_rate = float(params.get("fatigue_rate", 0.01))
    noise = float(params.get("compliance_noise", 0.1))
    linear = (
        w_social * (peer_share - person.threshold)
        + w_policy * (person.trust * policy_intensity)
        + w_media * media_signal
        + w_risk * (person.risk - 0.5)
        + 0.5 * (person.baseline + 0.4 * person.compliance_trait + 0.4 * person.attitude)
        + 0.6 * person.habit
        - fatigue_rate * person.fatigue
    )
    return clamp(sigmoid(3.0 * linear + rng.uniform(-noise, noise)), 0.0, 1.0)


class SocialAdoptionSimulation:
    """
    Main simulation class coordinating agents, households, information, market, policy, and metrics.

    This class implements:
    - Information broadcast (multiple sources with credibility/misinformation)
    - Media broadcast (legacy continuous signal)
    - Mask market with typed supply/restock/pricing and rationing
    - Household norms and shared inventory
    - Peer influence via a small-world network
    - Parameterized adoption decision and backsliding
    - Visits/enforcement with fines and entry refusal
    - Optional SEIR-like disease dynamics with type-specific efficacy
    - Metrics aggregation and validation
    """
    pass

    def __init__(
        self,
        population_size: int,
        time_horizon_days: int,
        random_seed: int = 42,
        include_disease_module: bool = False,
        config_params: Optional[Dict[str, Any]] = None,
        network_mean_degree: int = 8,
        network_rewiring_prob: float = 0.05,
    ):
        """
        Initialize the simulation.

        Args:
            population_size: Number of agents.
            time_horizon_days: Simulation length in days.
            random_seed: RNG seed.
            include_disease_module: Whether to run epidemiological module.
            config_params: Global configuration dict to override defaults.
            network_mean_degree: Mean degree for the small-world network.
            network_rewiring_prob: Rewiring probability for the small-world network.

        Returns:
            None
        """
        pass
        # Defaults per spec-aligned semantics; can be overridden by config_params
        self.params: Dict[str, Any] = {
            # Adoption weights
            "base_influence_strength": 0.3,
            "risk_perception_weight": 0.4,
            "policy_effect_strength": 0.5,
            "info_campaign_intensity": 0.2,
            "compliance_noise": 0.1,
            "fatigue_rate": 0.01,
            "forgetting_rate": 0.03,
            # Information
            "message_effect_size": 0.25,
            "num_information_sources": 3,
            "misinformation_fraction": 0.1,
            "info_source_mean_credibility": 0.7,
            "info_source_std_credibility": 0.2,
            # Households
            "mean_household_size": 3.0,
            "within_household_influence_weight": 0.6,
            "household_norm_enforcement_strength": 0.5,
            "share_rate": 0.5,
            "minimum_inventory_to_wear": 1,
            # Market types
            "mask_types": ["cloth", "surgical", "N95"],
            "mask_price_by_type": {"cloth": 0.5, "surgical": 0.8, "N95": 1.5},
            "initial_supply_by_type": {"cloth": 800, "surgical": 600, "N95": 300},
            "restock_rate_by_type": {"cloth": 100, "surgical": 80, "N95": 40},
            "rationing_rules": {"cloth": 10, "surgical": 8, "N95": 5},
            "price_floor": 0.1,
            "price_ceiling": 10.0,
            "mask_efficacy_by_type": {"cloth": 0.3, "surgical": 0.5, "N95": 0.8},
            # Retailer dynamics FIXED: restock interval and supply shock
            "restock_interval_days": 7,  # FIXED: Restock only on this interval
            "supply_shock_day": -1,      # FIXED: No shock by default
            "supply_shock_multiplier": 1.0,  # FIXED: 1.0 means no change
            "purchase_limit_per_visit": 10,  # FIXED: Per-person, per-type daily limit
            # Policy schedule defaults
            "policy_schedule": {"day": [0], "state": ["none"]},
            "enforcement_level_default": 0.3,
            "fine_amount": 50.0,
            "subsidy_amount": 0.0,
            # Enforcement capacity/budget FIXED
            "enforcement_capacity_per_day": 0,  # 0 = unlimited
            "enforcement_budget": float("inf"),
            # Media legacy
            "media_baseline_intensity": 0.2,
            "media_shock_day": 0,
            "media_shock_magnitude": 0.0,
            # Locations parameters (FIXED: parameterized locations)
            "num_locations": 30,
            "mask_policy_in_locations_fraction": 0.5,
            "location_capacity_mean": 50,
            "location_mix": {"home": 0.4, "workplace": 0.3, "school": 0.1, "transit": 0.1, "retail": 0.1},  # FIXED
            # Target adoption threshold (FIXED)
            "target_adoption_rate": 0.7,
            # Replacement intervals (partial use in coverage metric)
            "replacement_interval_days_by_type": {"cloth": 7, "surgical": 5, "N95": 10},
            # Location compliance context reporting
            "report_contextual_compliance": True,
        }
        if config_params:
            # FIXED: Add configuration ingestion to map provided parameters into the simulation.
            for k, v in config_params.items():
                self.params[k] = v

        # FIXED: Accept alias key 'mask_effectiveness_by_type' and map 'respirator' -> 'N95'
        if "mask_effectiveness_by_type" in self.params and "mask_efficacy_by_type" not in self.params:
            self.params["mask_efficacy_by_type"] = dict(self.params.get("mask_effectiveness_by_type", {}))
        # Map type aliases in lists and dicts
        if "mask_types" in self.params and any(t == "respirator" for t in self.params["mask_types"]):
            self.params["mask_types"] = ["N95" if t == "respirator" else t for t in self.params["mask_types"]]
        for dkey in ("mask_price_by_type", "initial_supply_by_type", "restock_rate_by_type", "rationing_rules", "mask_efficacy_by_type"):
            if dkey in self.params and "respirator" in self.params[dkey]:
                m = dict(self.params[dkey])
                m["N95"] = m.pop("respirator")
                self.params[dkey] = m

        self.population_size = max(1, int(population_size))
        self.days = int(time_horizon_days)
        self.rng = random.Random(int(random_seed))
        self.include_disease_module = include_disease_module

        # Entities
        self.people: List[Person] = []
        self.households: List[Household] = []
        self.network: Dict[int, set] = {}

        # Policy authority with schedule
        self.policy = PolicyAuthority(
            policy_schedule=self.params.get("policy_schedule", {"day": [0], "state": ["none"]}),
            enforcement_level_default=float(self.params.get("enforcement_level_default", 0.3)),
            communication_strategy=str(self.params.get("communication_strategy", "neutral")),
            fine_amount=float(self.params.get("fine_amount", 50.0)),
            subsidy_amount=float(self.params.get("subsidy_amount", 0.0)),
        )

        # Information sources
        self.info_sources: List[InformationSource] = []
        self._initialize_information_sources()

        # Legacy Media (optional additional signal)
        self.media = Media(
            baseline_intensity=float(self.params.get("media_baseline_intensity", 0.2)),
            shock_day=int(self.params.get("media_shock_day", max(1, self.days // 4))),
            shock_magnitude=float(self.params.get("media_shock_magnitude", 0.5)),
        )

        # Mask Market
        self.mask_market = MaskMarket(
            types_available=list(self.params.get("mask_types", ["cloth", "surgical", "N95"])),
            price_by_type=dict(self.params.get("mask_price_by_type", {"cloth": 0.5, "surgical": 0.8, "N95": 1.5})),
            initial_supply_by_type=dict(self.params.get("initial_supply_by_type", {"cloth": 800, "surgical": 600, "N95": 300})),
            restock_rate_by_type=dict(self.params.get("restock_rate_by_type", {"cloth": 100, "surgical": 80, "N95": 40})),
            rationing_rules=dict(self.params.get("rationing_rules", {"cloth": 10, "surgical": 8, "N95": 5})),
            price_floor=float(self.params.get("price_floor", 0.1)),
            price_ceiling=float(self.params.get("price_ceiling", 10.0)),
        )

        # FIXED: Instantiate locations from location_mix spec by type
        self.locations = []
        loc_mix = dict(self.params.get("location_mix", {"home": 0.4, "workplace": 0.3, "school": 0.1, "transit": 0.1, "retail": 0.1}))
        cap_mean = int(self.params.get("location_capacity_mean", 50))
        num_locs_total = int(self.params.get("num_locations", 30))
        for ltype, share in loc_mix.items():
            count = max(1, int(round(share * num_locs_total)))
            for i in range(count):
                cap = max(10, int(self.rng.gauss(cap_mean, max(5, cap_mean * 0.2))))
                ft = clamp(self.rng.random() * 0.5 + 0.1, 0.0, 1.0)
                if ltype in ("transit", "retail"):
                    enforcement = 0.5
                    mandate_sensitive = True
                    entry_refusal = True
                elif ltype in ("workplace", "school"):
                    enforcement = 0.3
                    mandate_sensitive = True
                    entry_refusal = False
                else:  # home
                    enforcement = 0.0
                    mandate_sensitive = False
                    entry_refusal = False
                self.locations.append(
                    Location(
                        name=f"{ltype}-{i}",
                        enforcement_level=enforcement,
                        mandate_sensitive=mandate_sensitive,
                        entry_refusal_if_unmasked=entry_refusal,
                        ltype=ltype,
                        capacity=cap,
                        mask_policy=(ltype in ("transit", "retail", "workplace", "school")),
                        foot_traffic_rate=ft,
                    )
                )

        # Network parameters
        self.network_mean_degree = int(network_mean_degree)
        self.network_rewiring_prob = float(network_rewiring_prob)

        # Metrics tracked
        self.metrics: Dict[str, Any] = {}
        self.validation_report: Dict[str, Any] = {}
        self.adoption_rate_over_time: List[float] = []
        self.Rt_over_time: List[float] = []
        self.new_infections_over_time: List[int] = []
        self.counterfactual_infections_over_time: List[int] = []
        self.total_fines_count: int = 0
        self.enforcement_actions_count: int = 0
        self.total_fines_value: float = 0.0
        self.peer_share_cache: List[float] = []
        self.prev_day_norms: float = 0.5
        self.masks_purchased_cumulative: int = 0
        self.masks_purchased_daily: List[int] = []
        # daily_compliance: (masked_visitors, total_visitors)
        self.daily_compliance: List[Tuple[int, int]] = []
        self.current_streak: List[int] = []
        self.completed_streaks: List[int] = []
        self.mask_type_distribution_over_time: List[Dict[str, float]] = []
        self.exposure_reduction_proxy: List[float] = []
        self.average_perceived_risk_over_time: List[float] = []
        self.unmet_demand_ratio_over_time: List[float] = []
        self.avg_price_over_time: List[float] = []
        # FIXED: Track exposure cohorts for message_impact
        self.exposure_log: List[Set[int]] = []
        self.ever_exposed: Set[int] = set()
        # FIXED: Track daily adopter sets to compute cohort adoption metrics
        self.daily_adopters: List[Set[int]] = []
        # FIXED: Track contextual compliance (required/recommended/optional)
        self.contextual_compliance_ts: List[Dict[str, Optional[float]]] = []
        # Supply shock applied flag
        self._shock_applied: bool = False

    def _initialize_information_sources(self) -> None:
        """
        Initialize information sources based on params.

        Returns:
            None
        """
        pass
        n = int(self.params.get("num_information_sources", 3))
        mis_frac = float(self.params.get("misinformation_fraction", 0.1))
        mean_cred = float(self.params.get("info_source_mean_credibility", 0.7))
        std_cred = float(self.params.get("info_source_std_credibility", 0.2))
        for i in range(n):
            anti = (i < max(1, int(n * mis_frac)))
            msg_type = "anti_mask" if anti else "pro_mask"
            cred = clamp(self.rng.gauss(mean_cred, std_cred), 0.0, 1.0)
            reach = clamp(self.rng.uniform(0.1, 0.6), 0.0, 1.0)
            mis_prob = mis_frac if anti else 0.0
            self.info_sources.append(InformationSource(sid=f"S{i}", message_type=msg_type, credibility=cred, reach=reach, misinformation_prob=mis_prob))

    def initialize_population(self) -> None:
        """
        Create agents, households, and network; assign SES quintiles and initial states.

        Returns:
            None
        """
        pass
        # Households: Poisson sizes truncated
        mean_hh = float(self.params.get("mean_household_size", 3.0))
        remaining = self.population_size
        hid = 0
        household_assignments: List[int] = []
        while remaining > 0:
            size = max(1, int(self.rng.expovariate(1.0 / max(1e-6, mean_hh - 1))) + 1)
            size = min(size, remaining)
            start_idx = len(household_assignments)
            for _ in range(size):
                household_assignments.append(hid)
            self.households.append(Household(hid=hid, member_ids=list(range(start_idx, start_idx + size)), norm_strength=clamp(self.rng.random(), 0.2, 0.8)))
            hid += 1
            remaining -= size

        # Person attributes
        ages = [max(18, int(self.rng.gauss(40, 15))) for _ in range(self.population_size)]
        incomes = [self.rng.uniform(15000.0, 150000.0) for _ in range(self.population_size)]
        risks = [clamp(self.rng.random(), 0.0, 1.0) for _ in range(self.population_size)]
        trusts = [clamp(self.rng.random(), 0.0, 1.0) for _ in range(self.population_size)]
        compliances = [clamp(self.rng.random(), 0.0, 1.0) for _ in range(self.population_size)]
        attitudes = [clamp(self.rng.gauss(0.0, 0.5), -1.0, 1.0) for _ in range(self.population_size)]
        thresholds = [clamp(self.rng.gauss(0.5, 0.15), 0.0, 1.0) for _ in range(self.population_size)]
        baselines = [clamp(self.rng.gauss(0.3, 0.15), 0.0, 1.0) for _ in range(self.population_size)]
        subscriptions = [self.rng.random() < 0.7 for _ in range(self.population_size)]
        # Person media profiles across InformationSources
        media_profiles = []
        for _ in range(self.population_size):
            prof = {}
            total = 0.0
            for src in self.info_sources:
                w = clamp(self.rng.random(), 0.0, 1.0)
                prof[src.sid] = w
                total += w
            if total > 0:
                for k in prof:
                    prof[k] = prof[k] / total
            media_profiles.append(prof)

        for i in range(self.population_size):
            cost_sens = clamp(self.rng.random(), 0.0, 1.0)
            p = Person(
                pid=i,
                age=ages[i],
                income=incomes[i],
                risk=risks[i],
                trust=trusts[i],
                compliance_trait=compliances[i],
                attitude=attitudes[i],
                threshold=thresholds[i],
                baseline=baselines[i],
                media_subscribed=subscriptions[i],
                household_id=int(household_assignments[i]),
                cost_sensitivity=cost_sens,
                media_consumption_profile=media_profiles[i],
            )
            self.people.append(p)

        # Assign SES quintiles by income rank
        sorted_incomes = sorted([(p.income, idx) for idx, p in enumerate(self.people)])
        for rank, (_, idx) in enumerate(sorted_incomes):
            q = int(5 * rank / max(1, self.population_size))
            q = min(4, q)  # 0-based 0..4
            self.people[idx].ses_quintile = q + 1

        # Initialize streaks
        self.current_streak = [0] * self.population_size
        self.completed_streaks = []

        # Initial adoption and inventory
        init_frac = float(self.params.get("initial_adoption_rate", 0.12))
        seed_k = min(self.population_size, max(1, int(self.population_size * init_frac)))
        init_ids = set(self.rng.sample(range(self.population_size), seed_k))
        for i in init_ids:
            self.people[i].adopting = True
            self.people[i].habit = 0.35 + 0.3 * self.rng.random()
            # Seed typed inventory and preferred mask type
            mask_types = list(self.mask_market.types_available)
            choice = self.rng.choices(mask_types, weights=[0.5, 0.3, 0.2], k=1)[0]
            self.people[i].mask_type = choice
            # Give some stock of chosen type
            qty = self.rng.randint(1, 5)
            self.people[i].mask_inventory_by_type[choice] += qty
            self.people[i].mask_inventory = sum(self.people[i].mask_inventory_by_type.values())
            self.current_streak[i] = 1

        # Seed household shared inventory
        for hh in self.households:
            for t in self.mask_market.types_available:
                hh.shared_inventory_by_type[t] = self.rng.randint(0, 10)

        # Disease seeds
        if self.include_disease_module:
            infectious_seed = max(1, self.population_size // 50)
            seed_ids = set(self.rng.sample(range(self.population_size), infectious_seed))
            for i in seed_ids:
                self.people[i].disease_state = 'I'
                self.people[i].days_in_state = 0

        # Network
        self.network = ring_small_world(self.population_size, k=self.network_mean_degree, p=self.network_rewiring_prob, rng=self.rng)

    def compute_peer_share(self) -> List[float]:
        """
        Compute peer adoption share for each node in the network.

        Returns:
            list: Peer shares in [0,1] for each person.
        """
        pass
        shares = [0.0] * self.population_size
        for i, _ in enumerate(self.people):
            neighbors = list(self.network.get(i, []))
            if not neighbors:
                shares[i] = 0.0
            else:
                shares[i] = safe_div(sum(1.0 if self.people[j].adopting else 0.0 for j in neighbors), len(neighbors), 0.0)
        self.peer_share_cache = shares
        return shares

    def daily_media_signal(self, day: int) -> List[float]:
        """
        Compute daily media signal per person.

        Args:
            day: Current day.

        Returns:
            list: Media signals.
        """
        pass
        signals = [self.media.signal(day, p.media_subscribed) for p in self.people]
        return signals

    def information_broadcast(self, day: int) -> None:
        """
        Broadcast messages from information sources and update perceived risk.

        Args:
            day: Day index.

        Returns:
            None
        """
        pass
        # FIXED: Track daily exposures and cohorts for DiD message impact metric.
        daily_exposed: Set[int] = set()
        for src in self.info_sources:
            stats = src.broadcast_message(day, self.people, self.rng, self.params)
            exposed_ids = stats.get("exposed_ids", set())
            if exposed_ids:
                daily_exposed |= set(exposed_ids)
        self.exposure_log.append(daily_exposed)
        self.ever_exposed |= daily_exposed
        # Track average perceived risk
        avg_risk = safe_div(sum(p.risk for p in self.people), self.population_size, 0.0)
        self.average_perceived_risk_over_time.append(avg_risk)

    def mask_market_restock(self, day: int) -> None:
        """
        Restock mask market inventories respecting restock interval and supply shocks.

        Args:
            day: Current simulation day.

        Returns:
            None
        """
        pass
        # FIXED: Restock only on configured interval and apply supply shock persistently.
        restock_interval = int(self.params.get("restock_interval_days", 7))
        shock_day = int(self.params.get("supply_shock_day", -1))
        shock_mult = float(self.params.get("supply_shock_multiplier", 1.0))
        if shock_day >= 0 and (day == shock_day) and not self._shock_applied:
            for t in self.mask_market.types_available:
                r = int(self.mask_market.restock_rate_by_type.get(t, 0))
                self.mask_market.restock_rate_by_type[t] = max(0, int(r * shock_mult))
            self._shock_applied = True
        if restock_interval <= 1 or (day % restock_interval == 0):
            self.mask_market.restock()

    def purchase_masks_typed(self, day: int) -> None:
        """
        Typed mask purchases by agents based on need, price sensitivity, income, and efficacy preference.

        Args:
            day: Current day index for policy-aware purchasing.

        Returns:
            None
        """
        pass
        # FIXED: Make purchasing policy-aware by using current day for mandate state.
        # FIXED: Enforce per-visit purchase limit and unmet demand based on units, not attempts.
        daily_sales = 0
        units_requested = 0
        units_sold = 0
        total_price = 0.0
        eff_by_type = dict(self.params.get("mask_efficacy_by_type", {"cloth": 0.3, "surgical": 0.5, "N95": 0.8}))
        price_weight = float(self.params.get("price_sensitivity", 0.5))
        subsidy = float(self.policy.subsidy_amount)
        minimum_inventory = int(self.params.get("minimum_inventory_to_wear", 1))
        mandate_active = (self.policy.current_state(day) == "mandate")
        per_visit_limit = int(self.params.get("purchase_limit_per_visit", 10))
        purchases_today: Dict[Tuple[int, str], int] = {}

        for p in self.people:
            # Determine need based on adoption state and inventory
            total_inv = sum(p.mask_inventory_by_type.values())
            target_stock = 7 if p.adopting else 2
            need = max(0, target_stock - total_inv)
            if need <= 0:
                continue
            remaining = need
            while remaining > 0:
                # Score each type
                best_t = None
                best_score = -1e9
                daily_income = max(1e-6, p.income / 365.0)
                for t in self.mask_market.types_available:
                    price = self.mask_market.consumer_price(t, subsidy)
                    afford_ratio = price / daily_income
                    eff = float(eff_by_type.get(t, 0.3))
                    score = (1.0 + p.compliance_trait) * eff - (price_weight * p.cost_sensitivity) * afford_ratio
                    if score > best_score:
                        best_score = score
                        best_t = t
                if best_t is None:
                    break
                # Attempt purchase of up to ration limit or remaining, respecting per-visit (per day-per type) limit
                desired = min(remaining, self.mask_market.rationing_rules.get(best_t, remaining))
                key = (p.pid, best_t)
                already = purchases_today.get(key, 0)
                allowed = max(0, per_visit_limit - already)
                desired = min(desired, allowed)
                if desired <= 0:
                    break
                units_requested += desired
                sold = self.mask_market.sell_masks(best_t, desired)
                if sold > 0:
                    purchases_today[key] = already + sold
                    p.mask_inventory_by_type[best_t] = p.mask_inventory_by_type.get(best_t, 0) + sold
                    p.mask_type = p.mask_type or best_t
                    remaining -= sold
                    daily_sales += sold
                    units_sold += sold
                    total_price += self.mask_market.consumer_price(best_t, subsidy) * sold
                else:
                    # Could not buy; stop attempting for this loop to avoid infinite attempts when inventory zero
                    break
            p.mask_inventory = sum(p.mask_inventory_by_type.values())
            # If mandate active and inventory below minimum, one additional attempt for cheapest type respecting per-visit limit
            if mandate_active and p.mask_inventory < minimum_inventory:
                tmin = min(self.mask_market.types_available, key=lambda t: self.mask_market.consumer_price(t, subsidy))
                key2 = (p.pid, tmin)
                already2 = purchases_today.get(key2, 0)
                allowed2 = max(0, per_visit_limit - already2)
                if allowed2 > 0:
                    units_requested += 1
                    sold = self.mask_market.sell_masks(tmin, 1)
                    if sold > 0:
                        purchases_today[key2] = already2 + sold
                        p.mask_inventory_by_type[tmin] = p.mask_inventory_by_type.get(tmin, 0) + sold
                        p.mask_inventory = sum(p.mask_inventory_by_type.values())
                        daily_sales += sold
                        units_sold += sold
                        total_price += self.mask_market.consumer_price(tmin, subsidy) * sold
        # Track metrics
        self.masks_purchased_cumulative += daily_sales
        self.masks_purchased_daily.append(daily_sales)
        unmet_ratio = 1.0 - safe_div(units_sold, units_requested, 0.0) if units_requested > 0 else 0.0
        # FIXED: unmet demand now uses units requested vs sold
        self.unmet_demand_ratio_over_time.append(unmet_ratio)
        avg_price = safe_div(total_price, units_sold, 0.0) if units_sold > 0 else 0.0
        self.avg_price_over_time.append(avg_price)

    def apply_household_norms(self) -> None:
        """
        Apply within-household inventory sharing and normative influence.

        Returns:
            None
        """
        pass
        share_rate = float(self.params.get("share_rate", 0.5))
        wh_weight = float(self.params.get("within_household_influence_weight", 0.6))
        min_inv = int(self.params.get("minimum_inventory_to_wear", 1))
        for hh in self.households:
            members = [self.people[i] for i in hh.member_ids]
            if not members:
                continue
            # Normative reinforcement
            adoption_fraction = safe_div(sum(1 for m in members if m.adopting), len(members), 0.0)
            for m in members:
                delta = wh_weight * (adoption_fraction - 0.5) * hh.norm_strength
                m.compliance_trait = clamp(m.compliance_trait + 0.05 * delta, 0.0, 1.0)
                m.threshold = clamp(m.threshold - 0.03 * delta, 0.0, 1.0)
            # Share inventory from shared pool
            for m in members:
                total_inv = sum(m.mask_inventory_by_type.values())
                if total_inv < min_inv:
                    # Try to pull from shared inventory by type, prioritize higher efficacy
                    for t in sorted(self.mask_market.types_available, key=lambda tt: self.params.get("mask_efficacy_by_type", {}).get(tt, 0.0), reverse=True):
                        if hh.shared_inventory_by_type.get(t, 0) <= 0:
                            continue
                        transfer = min(int(math.ceil(share_rate * hh.shared_inventory_by_type[t])), (min_inv - total_inv))
                        if transfer <= 0:
                            continue
                        hh.shared_inventory_by_type[t] -= transfer
                        m.mask_inventory_by_type[t] = m.mask_inventory_by_type.get(t, 0) + transfer
                        m.mask_inventory = sum(m.mask_inventory_by_type.values())
                        total_inv = m.mask_inventory
                        if m.mask_type is None:
                            m.mask_type = t
                        if total_inv >= min_inv:
                            break

    def adoption_step(self, day: int, media_signals: List[float], peer_shares: List[float]) -> None:
        """
        Execute adoption decisions and update habit/fatigue.

        Args:
            day: Current day index.
            media_signals: Per-person media signals.
            peer_shares: Per-person peer adoption share.

        Returns:
            None
        """
        pass
        policy_intensity = self.policy.policy_intensity(day)
        norms_effect = 0.3 * (self.prev_day_norms - 0.5)  # simple norms term centered at 0.5

        for i, p in enumerate(self.people):
            access_ok = sum(p.mask_inventory_by_type.values()) > 0
            base_prob = adoption_probability(
                person=p,
                peer_share=peer_shares[i],
                policy_intensity=policy_intensity,
                media_signal=media_signals[i],
                params=self.params,
                rng=self.rng,  # FIXED: routed rng for reproducibility
            )
            prob = clamp(base_prob + norms_effect, 0.0, 1.0)
            if not p.adopting:
                if access_ok and self.rng.random() < prob:
                    p.adopting = True
                    self.current_streak[i] = 1
                else:
                    if self.current_streak[i] > 0:
                        self.completed_streaks.append(self.current_streak[i])
                        self.current_streak[i] = 0
            else:
                # Allow backsliding
                drop = clamp(sigmoid(4.0 * (-prob + 0.25 + 0.6 * p.fatigue)), 0.0, 1.0)
                if self.rng.random() < drop:
                    if self.current_streak[i] > 0:
                        self.completed_streaks.append(self.current_streak[i])
                    p.adopting = False
                    self.current_streak[i] = 0
                else:
                    self.current_streak[i] += 1
            # Update habit and fatigue
            if p.adopting:
                p.habit = clamp(p.habit + 0.08 * (1.0 - p.habit), 0.0, 1.0)
                p.fatigue = clamp(p.fatigue + 0.015, 0.0, 1.0)
            else:
                p.habit = clamp(p.habit - 0.03, 0.0, 1.0)
                p.fatigue = clamp(p.fatigue - 0.015, 0.0, 1.0)
            # Forgetting effect
            if not p.adopting:
                fr = float(self.params.get("forgetting_rate", 0.03))
                p.compliance_trait = clamp(p.compliance_trait * (1.0 - fr), 0.0, 1.0)

    def _compute_daily_visits(self) -> Dict[str, List[int]]:
        """
        Compute and return a snapshot of visitors per location for the day.

        Returns:
            Dict[str, List[int]]: Mapping from location name to list of visitor person IDs.
        """
        pass
        # FIXED: Respect location foot_traffic_rate and capacity with weighted sampling.
        visitors_by_location: Dict[str, List[int]] = {loc.name: [] for loc in self.locations}
        weights = [loc.foot_traffic_rate for loc in self.locations]
        total_w = sum(weights) or 1.0
        probs = [w / total_w for w in weights]
        for p in self.people:
            visits = 0
            if self.rng.random() < 0.6:
                visits = 1 + (1 if self.rng.random() < 0.3 else 0)
            for _ in range(visits):
                loc = self.rng.choices(self.locations, weights=probs, k=1)[0]
                if len(visitors_by_location[loc.name]) < loc.capacity:
                    visitors_by_location[loc.name].append(p.pid)
        return visitors_by_location

    def visits_and_enforcement(self, day: int, visitors_by_location: Dict[str, List[int]]) -> None:
        """
        Simulate daily visits to locations and apply enforcement for non-compliance.

        Args:
            day: Current day index.
            visitors_by_location: Precomputed mapping of location to visitor IDs.

        Returns:
            None
        """
        pass
        state = self.policy.current_state(day)
        mandate_active = (state == "mandate")
        remaining_capacity = int(self.params.get("enforcement_capacity_per_day", 0))
        remaining_capacity = None if remaining_capacity == 0 else remaining_capacity  # None means unlimited
        remaining_budget = float(self.params.get("enforcement_budget", float("inf")))

        # Compute observed norms and daily compliance; count effective masked in mandated locations
        norms_sum, norms_count = 0.0, 0
        masked_total, visitors_total = 0, 0

        # FIXED: Contextual compliance tracking by requirement level
        req_masked, req_total = 0, 0
        rec_masked, rec_total = 0, 0
        opt_masked, opt_total = 0, 0

        def loc_policy_level(loc: Location) -> str:
            if state == "mandate":
                if loc.ltype in ("transit", "retail"):
                    return "required"
                elif loc.ltype in ("workplace", "school"):
                    return "recommended"
                else:
                    return "optional"
            elif state == "recommendation":
                return "recommended" if loc.mask_policy else "optional"
            else:
                return "optional"

        for loc in self.locations:
            visitors = visitors_by_location.get(loc.name, [])
            if visitors:
                # compute effective masked for norms and compliance
                eff_masked_flags = []
                level = loc_policy_level(loc)
                for pid in visitors:
                    p = self.people[pid]
                    apply_enforcement = (level == "required") and loc.mandate_sensitive
                    enforcement_prob = clamp(self.policy.enforcement_level(day) * loc.enforcement_level, 0.0, 1.0)
                    situational = False
                    if apply_enforcement and not p.adopting:
                        situational_prob = 0.2 + 0.6 * enforcement_prob
                        situational = (self.rng.random() < situational_prob)
                    eff_masked_flags.append(1.0 if (p.adopting or situational) else 0.0)
                frac_masked = safe_div(sum(eff_masked_flags), len(visitors), 0.0)
                loc.observed_norms = frac_masked
                norms_sum += frac_masked
                norms_count += 1
                if mandate_active and loc.mandate_sensitive and loc_policy_level(loc) == "required":
                    visitors_total += len(visitors)
                    masked_total += int(sum(eff_masked_flags))
                # Contextual buckets
                if level == "required":
                    req_total += len(visitors)
                    req_masked += int(sum(eff_masked_flags))
                elif level == "recommended":
                    # Recommended: count adopters only, not situational
                    rec_total += len(visitors)
                    rec_masked += sum(1 for pid in visitors if self.people[pid].adopting)
                else:
                    opt_total += len(visitors)
                    opt_masked += sum(1 for pid in visitors if self.people[pid].adopting)
            else:
                loc.observed_norms = 0.0

        # FIXED: Smooth norms effect via EMA to reduce volatility.
        alpha = 0.4
        observed = (norms_sum / norms_count) if norms_count else 0.5
        self.prev_day_norms = alpha * observed + (1 - alpha) * self.prev_day_norms

        # Track daily compliance for mandated locations (required)
        if mandate_active:
            self.daily_compliance.append((masked_total, visitors_total))
        else:
            self.daily_compliance.append((0, 0))

        # Store contextual compliance timeseries
        if bool(self.params.get("report_contextual_compliance", True)):
            self.contextual_compliance_ts.append({
                "required": safe_div(req_masked, req_total, None) if req_total > 0 else None,
                "recommended": safe_div(rec_masked, rec_total, None) if rec_total > 0 else None,
                "optional": safe_div(opt_masked, opt_total, None) if opt_total > 0 else None,
            })

        # Enforcement: fines and entry refusal with situational masking, respecting capacity and budget
        for loc in self.locations:
            visitors = visitors_by_location.get(loc.name, [])
            if not visitors:
                continue
            refused_ids: List[int] = []
            for pid in visitors:
                if remaining_capacity is not None and remaining_capacity <= 0:
                    break
                p = self.people[pid]
                # Consume masks when visiting while adopting: reduce inventory of chosen type if available, else highest available
                if p.adopting:
                    chosen = None
                    if p.mask_type and p.mask_inventory_by_type.get(p.mask_type, 0) > 0:
                        chosen = p.mask_type
                    else:
                        # fallback to highest efficacy type available
                        types_sorted = sorted(self.mask_market.types_available, key=lambda t: self.params.get("mask_efficacy_by_type", {}).get(t, 0.0), reverse=True)
                        for t in types_sorted:
                            if p.mask_inventory_by_type.get(t, 0) > 0:
                                chosen = t
                                break
                    if chosen is not None and self.rng.random() < 0.2:
                        p.mask_inventory_by_type[chosen] -= 1
                        p.mask_inventory = sum(p.mask_inventory_by_type.values())
                apply_enforcement = mandate_active and loc.mandate_sensitive and (loc_policy_level(loc) == "required")
                enforcement_prob = clamp(self.policy.enforcement_level(day) * loc.enforcement_level, 0.0, 1.0)
                # Situational compliance if enforcement present (for behavioral display only)
                situational = False
                if apply_enforcement and not p.adopting:
                    situational_prob = 0.2 + 0.6 * enforcement_prob
                    situational = self.rng.random() < situational_prob
                effective_masked = p.adopting or situational
                if apply_enforcement and not effective_masked:
                    if self.rng.random() < enforcement_prob:
                        if remaining_capacity is not None:
                            remaining_capacity -= 1
                        self.enforcement_actions_count += 1
                        if getattr(loc, "entry_refusal_if_unmasked", False) and self.rng.random() < 0.5:
                            refused_ids.append(pid)
                        else:
                            if remaining_budget >= self.policy.fine_amount:
                                self.total_fines_count += 1
                                self.total_fines_value += self.policy.fine_amount
                                remaining_budget -= self.policy.fine_amount
                                p.fines_paid += self.policy.fine_amount
                                p.risk = clamp(p.risk + 0.02, 0.0, 1.0)
                                p.compliance_trait = clamp(p.compliance_trait + 0.02, 0.0, 1.0)
            if refused_ids:
                visitors_by_location[loc.name] = [pid for pid in visitors if pid not in set(refused_ids)]

    def disease_step(self, day: int, visitors_by_location: Dict[str, List[int]]) -> None:
        """
        Execute SEIR-lite dynamics with simple location-based mixing and mask type efficacy.

        Args:
            day: Current day index.
            visitors_by_location: Precomputed mapping of location to visitor IDs.

        Returns:
            None
        """
        pass
        if not self.include_disease_module:
            return

        beta = 0.06
        incubation_days = 3
        infectious_days = 7
        eff_by_type = dict(self.params.get("mask_efficacy_by_type", {"cloth": 0.3, "surgical": 0.5, "N95": 0.8}))

        new_infections = 0
        counterfactual_new_infections = 0
        infectious_count = sum(1 for p in self.people if p.disease_state == 'I')

        for loc in self.locations:
            visitors = visitors_by_location.get(loc.name, [])
            if not visitors:
                continue
            infectious_visitors = [pid for pid in visitors if self.people[pid].disease_state == 'I']
            susceptible_visitors = [pid for pid in visitors if self.people[pid].disease_state == 'S']
            if not infectious_visitors or not susceptible_visitors:
                continue

            def eff_for(pid: int) -> float:
                person = self.people[pid]
                t = person.mask_type
                if t and person.mask_inventory_by_type.get(t, 0) > 0 and person.adopting:
                    return float(eff_by_type.get(t, 0.0))
                # fallback to best available
                best = 0.0
                if person.adopting:
                    for tt, q in person.mask_inventory_by_type.items():
                        if q > 0:
                            best = max(best, float(eff_by_type.get(tt, 0.0)))
                return best

            # Effective infectious pressure reduced by masks on either side
            mask_reduction_inf = 1.0
            if infectious_visitors:
                mask_reduction_inf = 1.0 - safe_div(sum(eff_for(pid) for pid in infectious_visitors), len(infectious_visitors), 0.0)
            mask_reduction_sus = 1.0
            if susceptible_visitors:
                mask_reduction_sus = 1.0 - safe_div(sum(eff_for(pid) for pid in susceptible_visitors), len(susceptible_visitors), 0.0)

            effective_pressure = len(infectious_visitors) * (mask_reduction_inf * mask_reduction_sus)
            cf_pressure = len(infectious_visitors)

            risk = beta * safe_div(effective_pressure, max(1, len(visitors)), 0.0)
            cf_risk = beta * safe_div(cf_pressure, max(1, len(visitors)), 0.0)

            for pid in susceptible_visitors:
                p = self.people[pid]
                if self.rng.random() < risk:
                    p.disease_state = 'E'
                    p.days_in_state = 0
                    new_infections += 1
                if self.rng.random() < cf_risk:
                    counterfactual_new_infections += 1

        # Progress disease
        for p in self.people:
            if p.disease_state == 'E':
                p.days_in_state += 1
                if p.days_in_state >= incubation_days:
                    p.disease_state = 'I'
                    p.days_in_state = 0
            elif p.disease_state == 'I':
                p.days_in_state += 1
                if p.days_in_state >= infectious_days:
                    p.disease_state = 'R'
                    p.days_in_state = 0

        self.new_infections_over_time.append(new_infections)
        self.counterfactual_infections_over_time.append(counterfactual_new_infections)
        Rt_same_day = safe_div(new_infections, infectious_count, 0.0)
        serial = 5
        if len(self.new_infections_over_time) > serial:
            denom = self.new_infections_over_time[-serial - 1]
            Rt_gen = safe_div(new_infections, denom, 0.0)
        else:
            Rt_gen = Rt_same_day
        self.Rt_over_time.append(Rt_gen)

    def _update_daily_metrics(self, day: int) -> None:
        """
        Update daily metrics after all steps: outages, adoption, type distribution, exposure proxy.

        Args:
            day: Day index.

        Returns:
            None
        """
        pass
        # Count outages (types out of stock)
        self.mask_market.end_of_day_outages()
        # Adoption rate aggregation
        adopt_rate = safe_div(sum(1 for p in self.people if p.adopting), self.population_size, 0.0)
        self.adoption_rate_over_time.append(adopt_rate)
        # Track daily adopter IDs for cohort metrics (FIXED)
        adopters_set = {p.pid for p in self.people if p.adopting}
        self.daily_adopters.append(adopters_set)
        # Mask type distribution among adopters
        adopters = [p for p in self.people if p.adopting]
        dist: Dict[str, float] = {}
        if adopters:
            for t in self.mask_market.types_available:
                count_t = sum(1 for p in adopters if (p.mask_type == t and p.mask_inventory_by_type.get(t, 0) > 0))
                dist[t] = safe_div(count_t, len(adopters), 0.0)
        else:
            for t in self.mask_market.types_available:
                dist[t] = 0.0
        self.mask_type_distribution_over_time.append(dist)
        # Exposure reduction proxy: adoption-weighted efficacy
        eff_by_type = self.params.get("mask_efficacy_by_type", {"cloth": 0.3, "surgical": 0.5, "N95": 0.8})
        if adopters:
            eff_sum = 0.0
            for p in adopters:
                t = p.mask_type
                if t and p.mask_inventory_by_type.get(t, 0) > 0:
                    eff_sum += float(eff_by_type.get(t, 0.0))
                else:
                    eff_sum += 0.0
            exposure_proxy = safe_div(eff_sum, self.population_size, 0.0)
        else:
            exposure_proxy = 0.0
        self.exposure_reduction_proxy.append(exposure_proxy)

    def step(self, day: int) -> None:
        """
        Perform a single simulation day in sequence:
        information -> media -> market restock -> typed purchases -> household norms -> peer exposure -> adoption -> visits/enforcement -> disease -> metrics update.

        Args:
            day: Current day index.

        Returns:
            None
        """
        pass
        # 1) Information broadcast
        self.information_broadcast(day)
        # 2) Media signals (legacy or additional)
        media_signals = self.daily_media_signal(day)
        # 3) Market restock and typed purchases
        self.mask_market_restock(day)  # FIXED: interval restock and supply shock
        self.purchase_masks_typed(day)  # FIXED: pass day for policy-aware purchasing
        # 4) Household norms influence
        self.apply_household_norms()
        # 5) Peer exposure and adoption
        peer_shares = self.compute_peer_share()
        self.adoption_step(day, media_signals, peer_shares)
        # 6) Mobility, enforcement
        visits_by_loc = self._compute_daily_visits()
        self.visits_and_enforcement(day, visits_by_loc)
        # 7) Disease (optional)
        self.disease_step(day, visits_by_loc)
        # 8) Metrics accumulation
        self._update_daily_metrics(day)

    def run(self) -> None:
        """
        Run the simulation across all days.

        Returns:
            None
        """
        pass
        self.initialize_population()
        for day in range(self.days):
            self.step(day)
        self.compute_metrics()
        self.validation_report = self.validate()

    def _group_rates_by_age(self) -> Dict[str, float]:
        """
        Helper to compute adoption by age group at final day.

        Returns:
            Dict[str,float]: Adoption rate by age group.
        """
        pass
        def age_group(age: int) -> str:
            if age < 25:
                return "<25"
            elif age < 45:
                return "25-44"
            elif age < 65:
                return "45-64"
            return "65+"

        by_age: Dict[str, float] = {}
        for label in ["<25", "25-44", "45-64", "65+"]:
            group = [p for p in self.people if age_group(p.age) == label]
            by_age[label] = safe_div(sum(1 for p in group if p.adopting), len(group), 0.0)
        return by_age

    def _group_rates_by_ses(self) -> Dict[str, float]:
        """
        Helper to compute adoption by SES quintile at final day.

        Returns:
            Dict[str,float]: Adoption rate by SES quintile as strings.
        """
        pass
        by_ses: Dict[str, float] = {}
        for q in range(1, 6):
            group = [p for p in self.people if p.ses_quintile == q]
            by_ses[str(q)] = safe_div(sum(1 for p in group if p.adopting), len(group), 0.0)
        return by_ses

    def compute_metrics(self) -> None:
        """
        Compute required metrics, including equity and information campaign impact metrics.

        Returns:
            None
        """
        pass
        adoption_series = self.adoption_rate_over_time[:]
        final_rate = adoption_series[-1] if adoption_series else 0.0

        # FIXED: Target adoption threshold parameterized and correct metric

# Execute main for both direct execution and sandbox wrapper invocation
main()