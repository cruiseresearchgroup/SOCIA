def main():
    pass

import os
import sys
import json
import math
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

try:
    import numpy as np
except Exception:
    np = None  # Minimal fallback; code uses random if numpy is missing

# Optional networkx import for network generation
try:
    import networkx as nx
except Exception:
    nx = None

# Path Handling Instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def clip01(x: float) -> float:
    """
    Clamp a float to the interval [0, 1].

    Returns:
        float: The clamped value.
    """
    pass
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def safe_np_rand(seed: int) -> Any:
    """
    Create a deterministic numpy RNG if numpy is available; otherwise return None.

    Args:
        seed (int): Seed for RNG.

    Returns:
        Any: numpy Generator or None.
    """
    pass
    if np is not None:
        rng = np.random.default_rng(seed)
        return rng
    return None


def gini(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.
    Returns 0 for empty or all-equal lists.

    Args:
        values (List[float]): Input non-negative values.

    Returns:
        float: Gini coefficient in [0, 1].
    """
    pass
    if not values:
        return 0.0
    vals = [max(0.0, float(v)) for v in values]
    s = sum(vals)
    if s == 0:
        return 0.0
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    cum = 0.0
    for i, v in enumerate(vals_sorted, start=1):
        cum += i * v
    g = (2 * cum) / (n * s) - (n + 1) / n
    return float(g)


@dataclass
class RiskEnvironment:
    """
    Broadcasts an aggregate risk signal from exogenous schedules or endogenous prevalence.
    Supports simple exogenous Gaussian bump, AR(1)-like process, or coupling to disease prevalence.

    Attributes:
        base_risk_index (float): Baseline risk level.
        volatility (float): Volatility for AR(1)-like mode.
        smoothing (float): Smoothing factor for rolling updates.
        mode (str): Mode of risk generation ('exogenous', 'ar1', 'endogenous').
        peak_day (int): Peak day for exogenous Gaussian bump.
        peak_value (float): Peak value at peak_day.
        width (float): Width (standard deviation) of Gaussian bump.
        last_value (float): Last emitted risk signal.
        rng (random.Random): RNG.
    """
    pass
    base_risk_index: float = 0.3
    volatility: float = 0.05
    smoothing: float = 0.7
    mode: str = "exogenous"  # 'exogenous', 'ar1', 'endogenous'
    peak_day: int = 30
    peak_value: float = 0.8
    width: float = 10.0
    last_value: float = 0.3
    rng: random.Random = field(default_factory=lambda: random.Random(42))

    def step(self, day: int, disease: Optional["DiseaseProcess"] = None) -> float:
        """
        Advance the risk environment by one day and return today's risk signal in [0, 1].

        Args:
            day (int): Day index.
            disease (Optional[DiseaseProcess]): Optional disease process for endogenous coupling.

        Returns:
            float: Risk signal in [0, 1].
        """
        pass
        if self.mode == "exogenous":
            base = float(self.base_risk_index)
            val = base + (self.peak_value - base) * math.exp(-0.5 * ((day - self.peak_day) / max(1e-6, self.width)) ** 2)
            self.last_value = clip01(self.smoothing * self.last_value + (1.0 - self.smoothing) * val)
        elif self.mode == "endogenous":
            # Couple risk to prevalence (share infectious)
            if disease is not None and disease.n > 0:
                prev = sum(1 for s in disease.state if s == "I") / float(disease.n)
            else:
                prev = 0.0
            target = clip01(self.base_risk_index + prev)
            self.last_value = clip01(self.smoothing * self.last_value + (1.0 - self.smoothing) * target)
        else:
            # AR(1)-like random walk bounded in [0,1]
            shock = self.rng.gauss(0.0, self.volatility)
            raw = clip01(self.last_value + shock)
            self.last_value = clip01(self.smoothing * self.last_value + (1.0 - self.smoothing) * raw)
        return self.last_value


@dataclass
class Person:
    """
    Represents an individual in the simulation with attributes and decision rules for mask adoption.

    Attributes:
        id (int): Person ID.
        age (int): Age.
        gender (int): Binary gender code.
        household_id (int): Household membership ID.
        occupation (str): Occupation category.
        health_risk_level (str): Health risk category.
        demographic_group (str): Demographic grouping for homophily and inequality metrics.  # FIXED: Added demographic_group
        trust_in_authorities (float): Baseline trust in authorities.
        trust_in_media (float): Baseline trust in media.
        susceptibility_to_social_influence (float): Individual susceptibility to peer effects.
        risk_perception (float): Perceived risk (0-1).
        exposure_to_misinformation (float): Misinformation exposure (0-1).  # FIXED: Added misinformation exposure
        compliance_fatigue (float): Compliance fatigue (0-1).  # FIXED: Added compliance fatigue
        perceived_mask_cost (float): Perceived cost of wearing mask.
        mask_access (bool): Whether masks are accessible.
        mask_stock (int): Current mask inventory.
        is_wearing_mask (bool): Wearing state for current day.
        compliance_probability (float): Latent compliance propensity.
        belief_state (float): Placeholder belief value.
        information_exposure_history (List[Dict[str, Any]]): History of exposures.
        social_ties (List[int]): Social network neighbor IDs.
        income (float): Income.
        last_consume_day (int): Last day a mask was consumed/replaced.
    """
    pass
    id: int = 0
    age: int = 30
    gender: int = 0  # 0 male, 1 female
    household_id: int = -1
    occupation: str = "non_essential"  # 'essential', 'non_essential', 'student', 'retired'
    health_risk_level: str = "low"
    demographic_group: str = "general"  # FIXED: Added demographic_group
    trust_in_authorities: float = 0.5
    trust_in_media: float = 0.5
    susceptibility_to_social_influence: float = 0.5
    risk_perception: float = 0.2
    exposure_to_misinformation: float = 0.0  # FIXED: Added exposure_to_misinformation
    compliance_fatigue: float = 0.0  # FIXED: Added compliance_fatigue
    perceived_mask_cost: float = 0.2
    mask_access: bool = True
    mask_stock: int = 2
    is_wearing_mask: bool = False
    compliance_probability: float = 0.6
    belief_state: float = 0.0
    information_exposure_history: List[Dict[str, Any]] = field(default_factory=list)
    social_ties: List[int] = field(default_factory=list)
    income: float = 30000.0
    last_consume_day: int = -1

    def decide_wear_mask(
        self,
        social_norm: float,
        mandate_active: bool,
        enforcement_level: float,
        household_norm: float,
        household_weight: float,
        rng: random.Random,
        peer_w: float,
        media_w: float,
        authority_w: float,
        conformity_threshold: float,
        social_w: Optional[float] = None,  # FIXED: Map spec weights override
        policy_w: Optional[float] = None,  # FIXED: Map spec weights override
        risk_w: Optional[float] = None,  # FIXED: Map spec weights override
        adoption_threshold: Optional[float] = None,  # FIXED: Explicit threshold param
    ) -> None:
        """
        Decide whether to wear a mask today based on social, media/risk, authority cues,
        misinformation exposure, and compliance fatigue. Updates internal wearing state but
        does not consume inventory (handled by replace_mask).

        Args:
            social_norm (float): Neighbor mask wearing rate.
            mandate_active (bool): Whether a mandate is active.
            enforcement_level (float): Effective enforcement level [0,1].
            household_norm (float): Household mask wearing rate.
            household_weight (float): Weight for household norms.
            rng (random.Random): Random generator.
            peer_w (float): Baseline peer influence weight.
            media_w (float): Baseline media/risk influence weight.
            authority_w (float): Baseline authority influence weight.
            conformity_threshold (float): Baseline conformity threshold.
            social_w (Optional[float]): Optional override for social influence weight.
            policy_w (Optional[float]): Optional override for policy pressure weight.
            risk_w (Optional[float]): Optional override for risk perception weight.
            adoption_threshold (Optional[float]): Optional override for adoption utility threshold.
        """
        pass
        # FIXED: Scale peer influence by individual susceptibility_to_social_influence and allow override
        eff_peer_w = (social_w if social_w is not None else peer_w) * clip01(self.susceptibility_to_social_influence)  # FIXED
        # Weighted norm perception combining peers and household norms
        norm = clip01(eff_peer_w * social_norm + household_weight * household_norm)
        # Risk perception channel with misinformation dampening and optional override
        risk_weight = (risk_w if risk_w is not None else media_w)
        # Misinformation reduces effective risk signal
        risk_channel = clip01(risk_weight * self.risk_perception * (1.0 - 0.5 * clip01(self.exposure_to_misinformation)))
        # Authority channel with optional override
        authority_weight = (policy_w if policy_w is not None else authority_w)
        authority_channel = authority_weight * (enforcement_level if mandate_active else 0.0)
        # Conformity threshold influences the required utility to adopt
        base_threshold = (adoption_threshold if adoption_threshold is not None else (0.5 + conformity_threshold))  # FIXED
        # Habit persistence: makes switching harder
        habit_bias = -0.1 if self.is_wearing_mask else 0.1
        # FIXED: Fatigue penalty reduces utility when fatigued
        fatigue_penalty = 0.3 * clip01(self.compliance_fatigue)
        utility = norm + risk_channel + authority_channel - self.perceived_mask_cost + habit_bias - fatigue_penalty

        # Combine utility with compliance if a mandate
        if mandate_active:
            comp = clip01(self.compliance_probability + 0.2 * enforcement_level - 0.05 * self.perceived_mask_cost)
        else:
            comp = clip01(self.compliance_probability * 0.95 + 0.05 * (1.0 if self.is_wearing_mask else 0.0))
        self.compliance_probability = comp

        adopt = utility >= base_threshold
        # Apply access/stock constraints
        if adopt and self.mask_access and self.mask_stock > 0:
            self.is_wearing_mask = True
        elif mandate_active and comp > rng.random() and self.mask_access and self.mask_stock > 0:
            self.is_wearing_mask = True
        else:
            self.is_wearing_mask = False

    def update_fatigue(self, wearing: bool, fatigue_rate: float, recovery_rate: float) -> None:
        """
        Update compliance fatigue based on whether the person wore a mask today.

        Args:
            wearing (bool): Whether the agent wore a mask today.
            fatigue_rate (float): Increment when wearing [0,1].
            recovery_rate (float): Decrement when not wearing [0,1].
        """
        pass
        if wearing:
            self.compliance_fatigue = clip01(self.compliance_fatigue + max(0.0, fatigue_rate))
        else:
            self.compliance_fatigue = clip01(self.compliance_fatigue - max(0.0, recovery_rate))

    def purchase_mask(
        self,
        retailers: List["Retailer"],
        threshold: int,
        rng: random.Random,
        rationing_limit_per_person: int = 10,
    ) -> int:
        """
        Attempt to purchase masks when stock falls below a threshold.
        Returns the number of masks purchased.

        Args:
            retailers (List[Retailer]): Available retailers.
            threshold (int): Stock threshold to trigger purchasing.
            rng (random.Random): RNG.
            rationing_limit_per_person (int): Max units allowed per purchase.

        Returns:
            int: Units purchased.
        """
        pass
        if self.mask_stock >= threshold or not self.mask_access:
            return 0
        if not retailers:
            return 0
        # Affordability: willing to spend up to 0.5% of monthly income per purchase
        budget = max(1.0, 0.005 * self.income / 30.0)
        purchased = 0
        # FIXED: Avoid in-place shuffling of retailers by iterating over a shuffled copy
        for r in rng.sample(retailers, len(retailers)):  # FIXED
            if r.inventory <= 0:
                continue
            max_afford = int(budget // max(0.1, r.price))
            if max_afford <= 0:
                continue
            qty_desired = min(rationing_limit_per_person, threshold + 3 - self.mask_stock, max_afford)
            if qty_desired <= 0:
                continue
            got = r.sell(qty_desired)
            purchased += got
            self.mask_stock += got
            if purchased >= qty_desired:
                break
        return purchased

    def replace_mask(self, day: int, replacement_interval_days: int) -> int:
        """
        Consume one mask every replacement_interval_days if currently wearing a mask.
        Returns masks consumed today (0 or 1).

        Args:
            day (int): Day index.
            replacement_interval_days (int): Replacement interval.

        Returns:
            int: Units consumed (0 or 1).
        """
        pass
        consumed = 0
        if self.is_wearing_mask and self.mask_stock > 0:
            # Consume at day multiples
            if replacement_interval_days <= 1:
                self.mask_stock -= 1
                consumed = 1
            else:
                if self.last_consume_day < 0 or (day - self.last_consume_day) >= replacement_interval_days:
                    self.mask_stock -= 1
                    self.last_consume_day = day
                    consumed = 1
        return consumed


@dataclass
class Household:
    """
    Household grouping people with shared norms and income.

    Attributes:
        id (int): Household ID.
        member_ids (List[int]): Members of the household.
        income (float): Household income.
        norms (float): Household norm level (adoption average).
        mask_stock (int): Shared household mask stock (not used directly here).
        home_location_id (int): Location ID representing the home.
    """
    pass
    id: int = 0
    member_ids: List[int] = field(default_factory=list)
    income: float = 40000.0
    norms: float = 0.0
    mask_stock: int = 3
    home_location_id: int = -1

    def share_norms(self, people: List[Person]) -> None:
        """
        Update household norms as the average of members' adoption states.

        Args:
            people (List[Person]): Population list.
        """
        pass
        if not self.member_ids:
            self.norms = 0.0
            return
        adopters = sum(1 for pid in self.member_ids if 0 <= pid < len(people) and people[pid].is_wearing_mask)
        self.norms = adopters / float(len(self.member_ids))


@dataclass
class Location:
    """
    Represents a physical location for attendance and enforcement.

    Attributes:
        id (int): Location ID.
        type (str): Type: 'home', 'workplace', 'school', 'public_space'.
        capacity (int): Capacity constraint.
        risk_level (float): Risk level for disease (if enabled).
        mask_requirement (bool): Whether masks are required today.
        enforcement_level (float): Local enforcement intensity (0-1).
    """
    pass
    id: int = 0
    type: str = "home"  # 'home', 'workplace', 'school', 'public_space'
    capacity: int = 10
    risk_level: float = 0.2
    mask_requirement: bool = False
    enforcement_level: float = 0.0

    def enforce_mask_policy(self) -> None:
        """
        Placeholder for enforcement updates at location-level.
        """
        pass
        # In this simple model, enforcement level remains fixed per day
        return


@dataclass
class PublicHealthAuthority:
    """
    Public health authority controlling mandates, enforcement probability, and campaigns.

    Attributes:
        mandate_active (bool): Whether a mandate is currently active.
        policy_strength (float): Policy strength multiplier [0,1].
        enforcement_probability (float): Probability of enforcement action [0,1].
        campaign_active (bool): Whether an information campaign is active.
        campaign_effect_size (float): Magnitude of campaign effect on perceptions and misinformation [0,1].
        message_frequency_per_day (float): Frequency of broadcast opportunities [0,1].
        mandate_start_day (int): Day index when mandate activates.
        campaign_start_day (int): Campaign start day.
        campaign_end_day (int): Campaign end day.
    """
    pass
    mandate_active: bool = False
    policy_strength: float = 0.7
    enforcement_probability: float = 0.2
    campaign_active: bool = False
    campaign_effect_size: float = 0.15
    message_frequency_per_day: float = 1.0
    mandate_start_day: int = 20
    campaign_start_day: int = 10
    campaign_end_day: int = 10**9

    def step(self, day: int) -> Dict[str, Any]:
        """
        Compute today's policy signals and campaign status.

        Args:
            day (int): Day index.

        Returns:
            Dict[str, Any]: Policy signal bundle.
        """
        pass
        self.mandate_active = day >= int(self.mandate_start_day)
        self.campaign_active = (int(self.campaign_start_day) <= day <= int(self.campaign_end_day))
        return {
            "mandate_active": self.mandate_active,
            "policy_strength": float(self.policy_strength),
            "enforcement_probability": float(self.enforcement_probability),
            "campaign_active": self.campaign_active,
            "campaign_effect_size": float(self.campaign_effect_size),
            "message_frequency_per_day": float(self.message_frequency_per_day),
        }


@dataclass
class MediaOutlet:
    """
    Represents a media source broadcasting risk-related messages to the population.

    Attributes:
        id (int): Outlet ID.
        credibility (float): Credibility factor in [0,1].
        message_bias (float): Bias alignment (-1 to 1).
        misinformation_rate (float): Rate of misinformation content [0,1].
        reach (float): Share of population reached in a broadcast [0,1].
        schedule (List[Dict[str, Any]]): List of active windows with start/end_day.
    """
    pass
    id: int = 0
    credibility: float = 0.7
    message_bias: float = 0.0
    misinformation_rate: float = 0.0
    reach: float = 0.5
    schedule: List[Dict[str, Any]] = field(default_factory=list)

    def broadcast(self, people: List[Person], effect_weight: float, rng: random.Random, day: int, message_frequency_per_day: float = 1.0) -> None:
        """
        Broadcast messages affecting risk perception, trust, and misinformation exposure for a subset of people.

        Args:
            people (List[Person]): Population.
            effect_weight (float): Weight of message effect.
            rng (random.Random): RNG.
            day (int): Day index.
            message_frequency_per_day (float): Chance to broadcast on a given day [0,1].
        """
        pass
        # FIXED: Trigger based on frequency
        if rng.random() > clip01(message_frequency_per_day):
            return
        active = any(win.get("start_day", 0) <= day <= win.get("end_day", 10**9) for win in self.schedule)
        if not active:
            return
        for p in people:
            if rng.random() > self.reach:
                continue
            # Adjust trust and risk perception based on credibility and misinformation
            net_signal = self.credibility * (1.0 - self.misinformation_rate) - self.credibility * self.misinformation_rate
            delta_risk = effect_weight * net_signal * (0.1 + abs(self.message_bias))
            # Heterogeneous uptake based on trust
            gain = clip01(0.05 + 0.3 * p.trust_in_media)
            p.risk_perception = clip01(p.risk_perception + gain * delta_risk)
            # FIXED: Update misinformation exposure from media
            p.exposure_to_misinformation = clip01(
                p.exposure_to_misinformation + self.misinformation_rate * 0.2 - (1.0 - self.misinformation_rate) * 0.05
            )
            # Update trust in authorities via media bias alignment
            align = 1.0 - abs(self.message_bias)  # neutral messages slightly boost trust
            p.trust_in_authorities = clip01(p.trust_in_authorities * (1.0 - 0.01) + 0.01 * align)


@dataclass
class Retailer:
    """
    Retailer providing masks; manages inventory and dynamic pricing.

    Attributes:
        id (int): Retailer ID.
        inventory (int): Current inventory.
        restock_rate (int): Expected daily restock.
        price (float): Current price per mask.
        rationing_limit_per_person (int): Max units per person per day.
        markup (float): Price markup factor for baseline drift.
        price_elasticity_of_demand (float): Elasticity parameter (not fully used).
        supply_variability_cv (float): Coefficient of variation for restock randomness.
        _demand_today (int): Observed demand units today.
    """
    pass
    id: int = 0
    inventory: int = 200
    restock_rate: int = 50
    price: float = 1.0
    rationing_limit_per_person: int = 10
    markup: float = 0.2
    price_elasticity_of_demand: float = -0.5
    supply_variability_cv: float = 0.1
    _demand_today: int = 0

    def sell(self, quantity: int) -> int:
        """
        Sell up to 'quantity' masks; returns actual sold units.

        Args:
            quantity (int): Desired purchase quantity.

        Returns:
            int: Quantity sold.
        """
        pass
        if quantity <= 0:
            return 0
        sold = min(quantity, self.inventory)
        self.inventory -= sold
        self._demand_today += sold
        return sold

    def restock_daily(self, rng: Optional[random.Random] = None) -> int:
        """
        Restock inventory with variability.

        Args:
            rng (Optional[random.Random]): RNG.

        Returns:
            int: Quantity restocked.
        """
        pass
        if rng is None:
            rng = random
        mean = float(self.restock_rate)
        sd = self.supply_variability_cv * mean
        add = max(0, int(rng.gauss(mean, sd)))
        self.inventory += add
        return add

    def adjust_price(self) -> float:
        """
        Adjust price based on excess demand proxy.

        Returns:
            float: New price.
        """
        pass
        # Excess demand ratio approximation
        capacity = max(1.0, float(self.restock_rate))
        demand = float(self._demand_today)
        excess_ratio = max(0.0, (demand - capacity) / capacity)
        self.price = max(0.1, self.price * (1.0 + self.markup * 0.02) * (1.0 + 0.5 * excess_ratio))
        self._demand_today = 0
        return self.price


class DiseaseProcess:
    """
    Minimal disease dynamics for optional integration of infections.

    Attributes:
        n (int): Population size.
        state (List[str]): Disease state per person.
        base_transmission_prob (float): Baseline transmission probability.
        mask_effectiveness (float): Effectiveness of mask on transmission.
        rng (random.Random): RNG.
        series_I (List[int]): Time series of infectious counts.
        series_new_infections (List[int]): Time series of new infections.
    """
    pass

    def __init__(self, n: int, base_transmission_prob: float = 0.02, mask_effectiveness: float = 0.5, rng: Optional[random.Random] = None) -> None:
        """
        Initialize disease states for 'n' agents.

        Args:
            n (int): Population size.
            base_transmission_prob (float): Baseline transmission probability.
            mask_effectiveness (float): Mask effectiveness parameter.
            rng (Optional[random.Random]): RNG.
        """
        pass
        self.n = n
        self.state = ["S"] * n
        self.base_transmission_prob = base_transmission_prob
        self.mask_effectiveness = mask_effectiveness
        self.rng = rng or random
        # Seed a handful of infectious individuals
        for i in range(min(5, n)):
            self.state[self.rng.randrange(0, n)] = "I"
        self.series_I = []
        self.series_new_infections = []

    def transmit_given_contact(self, mask_src: bool, mask_dst: bool, location_type: str) -> float:
        """
        Compute probability of infection given a contact, accounting for masks.

        Args:
            mask_src (bool): Source masked.
            mask_dst (bool): Destination masked.
            location_type (str): Location type for risk modifications.

        Returns:
            float: Probability of transmission.
        """
        pass
        p = self.base_transmission_prob
        # Location risk modifier
        if location_type == "home":
            p *= 0.8
        elif location_type in ("workplace", "school"):
            p *= 1.0
        else:
            p *= 1.2
        # Mask protection multiplicative reductions
        if mask_src:
            p *= (1.0 - self.mask_effectiveness * 0.6)
        if mask_dst:
            p *= (1.0 - self.mask_effectiveness)
        return clip01(p)

    def apply_infections(self, newly_infected: List[int]) -> int:
        """
        Set newly infected individuals to 'I' state.

        Args:
            newly_infected (List[int]): Indices of newly infected.

        Returns:
            int: Count of new infections applied.
        """
        pass
        count = 0
        for idx in newly_infected:
            if 0 <= idx < self.n and self.state[idx] == "S":
                self.state[idx] = "I"
                count += 1
        return count

    def record_daily_metrics(self, I_prev: int, new_infections: int) -> None:
        """
        Record daily disease metrics.

        Args:
            I_prev (int): Previous infectious count.
            new_infections (int): New infections today.
        """
        pass
        self.series_I.append(I_prev)
        self.series_new_infections.append(new_infections)


class Simulation:
    """
    Coordinates environment, agents, policies, media, retailers, and metrics collection.
    """
    pass

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Initialize the simulation with the provided configuration.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary.
        """
        pass
        self.cfg = dict(cfg or {})
        self._normalize_config()  # FIXED: Normalize config and map spec keys
        seed = int(self.cfg.get("random_seed", 42))
        self.rng = random.Random(seed)
        self.np_rng = safe_np_rand(seed)
        self.people: List[Person] = []
        self.households: List[Household] = []
        self.locations_by_type: Dict[str, List[Location]] = {"home": [], "workplace": [], "school": [], "public_space": []}
        self.loc_index: Dict[str, Dict[int, Location]] = {k: {} for k in self.locations_by_type}
        self.retailers: List[Retailer] = []
        self.media_outlets: List[MediaOutlet] = []
        # FIXED: Implement PublicHealthAuthority replacing Government
        self.policy = PublicHealthAuthority(  # FIXED
            mandate_start_day=int(self.cfg.get("mandate_day", 30)),
            policy_strength=float(self.cfg.get("policy_strength", self.cfg.get("authority_influence_strength", 0.2))),
            enforcement_probability=float(self.cfg.get("enforcement_probability", self.cfg.get("policy_enforcement_intensity", 0.6))),
            campaign_effect_size=float(self.cfg.get("campaign_effect_size", 0.15)),
            message_frequency_per_day=float(self.cfg.get("message_frequency_per_day", 1.0)),
            campaign_start_day=int(self.cfg.get("campaign_start_day", 10)),
            campaign_end_day=int(self.cfg.get("campaign_end_day", self.cfg.get("simulation_days", 90))),
        )
        self.with_disease: bool = bool(self.cfg.get("with_disease", False))
        self.with_supply: bool = True
        self.disease: Optional[DiseaseProcess] = None
        self.series: Dict[str, List[Any]] = {}
        self._states_history: List[List[bool]] = []
        self._today_enforcement_events: int = 0
        # FIXED: Introduce RiskEnvironment and integrate
        self.risk_env = RiskEnvironment(
            base_risk_index=float(self.cfg.get("risk_signal_baseline", 0.3)),
            volatility=float(self.cfg.get("risk_signal_volatility", 0.05)),
            smoothing=float(self.cfg.get("risk_signal_smoothing", 0.7)),
            mode=str(self.cfg.get("risk_signal_mode", "exogenous")),
            peak_day=int(self.cfg.get("risk_signal_peak_day", 30)),
            peak_value=float(self.cfg.get("risk_signal_peak_value", 0.8)),
            width=float(self.cfg.get("risk_signal_peak_width", 10.0)),
            rng=self.rng,
        )

        # Build environment
        self._build_population()
        self._build_households()
        self._assign_households()
        self._build_locations()
        self._assign_work_and_school()
        self._build_retailers()
        self._build_media()

        if self.with_disease:
            self.disease = DiseaseProcess(n=len(self.people), base_transmission_prob=float(self.cfg.get("base_transmission_prob", 0.02)), mask_effectiveness=float(self.cfg.get("mask_efficacy", 0.5)), rng=self.rng)
        self._init_series()

    def _normalize_config(self) -> None:
        """
        Normalize configuration keys and set defaults, mapping spec to internal keys. Also
        sets defaults required to align with spec and feedback improvements.

        Notes:
            - Maps aliases like n_agents -> population_size, rewire_prob -> rewiring_prob_small_world.
            - Sets default influence weights and thresholds.
            - Provides parameters for misinformation diffusion and compliance fatigue.
        """
        pass
        cfg = self.cfg
        # FIXED: Map legacy/spec keys to internal keys
        if "n_agents" in cfg and "population_size" not in cfg:
            cfg["population_size"] = cfg["n_agents"]
        if "network_type" in cfg and "network_topology" not in cfg:
            nt = str(cfg["network_type"])
            cfg["network_topology"] = "small_world" if "watt" in nt else nt
        if "avg_degree" in cfg and "average_degree" not in cfg:
            cfg["average_degree"] = cfg["avg_degree"]
        if "mandate_start_day" in cfg and "mandate_day" not in cfg:
            cfg["mandate_day"] = cfg["mandate_start_day"]
        if "observation_period_days" in cfg and "simulation_days" not in cfg:
            cfg["simulation_days"] = cfg["observation_period_days"]
        if "include_disease_dynamics" in cfg:
            cfg["with_disease"] = bool(cfg["include_disease_dynamics"])
        # FIXED: Map rewire_prob alias to rewiring_prob_small_world
        if "rewire_prob" in cfg and "rewiring_prob_small_world" not in cfg:
            cfg["rewiring_prob_small_world"] = cfg["rewire_prob"]
        # FIXED: Map daily contacts spec params into contact_rate_by_location
        contact_map = {}
        if "daily_contacts_home" in cfg:
            contact_map["home"] = int(cfg["daily_contacts_home"])
        if "daily_contacts_work" in cfg:
            contact_map["workplace"] = int(cfg["daily_contacts_work"])
        if "daily_contacts_community" in cfg:
            contact_map["public_space"] = int(cfg["daily_contacts_community"])
        if contact_map:
            cfg["contact_rate_by_location"] = {**cfg.get("contact_rate_by_location", {}), **contact_map}
        # FIXED: Map spec weights into behavior parameters
        if "social_influence_weight" not in cfg and "peer_influence_strength" in cfg:
            cfg["social_influence_weight"] = cfg["peer_influence_strength"]
        if "risk_perception_weight" not in cfg and "media_influence_strength" in cfg:
            cfg["risk_perception_weight"] = cfg["media_influence_strength"]
        if "policy_pressure_weight" not in cfg and "authority_influence_strength" in cfg:
            cfg["policy_pressure_weight"] = cfg["authority_influence_strength"]
        if "adoption_threshold" not in cfg and "conformity_threshold" in cfg:
            cfg["adoption_threshold"] = 0.5 + float(cfg["conformity_threshold"])

        # Defaults
        cfg.setdefault("population_size", 1000)
        cfg.setdefault("simulation_days", 90)
        cfg.setdefault("initial_adoption_rate", 0.15)
        cfg.setdefault("network_topology", "small_world")
        cfg.setdefault("average_degree", 8)
        cfg.setdefault("rewiring_prob_small_world", 0.05)
        cfg.setdefault("peer_influence_strength", 0.5)
        cfg.setdefault("media_influence_strength", 0.3)
        cfg.setdefault("authority_influence_strength", 0.2)
        cfg.setdefault("conformity_threshold", 0.1)
        cfg.setdefault("adoption_threshold", 0.5 + cfg.get("conformity_threshold", 0.1))  # FIXED
        cfg.setdefault("household_norm_weight", 0.3)
        cfg.setdefault("risk_signal_mode", "exogenous")
        cfg.setdefault("risk_signal_baseline", 0.3)
        cfg.setdefault("risk_signal_peak_day", 30)
        cfg.setdefault("risk_signal_peak_value", 0.8)
        cfg.setdefault("risk_signal_to_perception_gain", 0.6)
        cfg.setdefault("mask_price", 1.0)
        cfg.setdefault("mask_replacement_interval_days", 7)
        cfg.setdefault("purchase_threshold", 1)
        cfg.setdefault("rationing_limit_per_person", 10)
        cfg.setdefault("policy_enforcement_intensity", 0.6)
        cfg.setdefault("enforcement_probability", cfg["policy_enforcement_intensity"])  # FIXED
        cfg.setdefault("policy_strength", cfg.get("authority_influence_strength", 0.2))  # FIXED
        cfg.setdefault("compliance_penalty", 50.0)
        cfg.setdefault("restock_interval_days", 1)
        cfg.setdefault("public_space_attendance_base_prob", 0.3)
        cfg.setdefault("mobility_reduction_under_mandate", 0.1)
        cfg.setdefault("homophily_level", 0.0)  # FIXED: Homophily control
        cfg.setdefault("homophily_amplification", 0.3)  # FIXED: Misinformation homophily amplification
        # FIXED: Misinformation diffusion parameters
        cfg.setdefault("misinformation_spread_rate", 0.2)
        cfg.setdefault("misinformation_decay_rate", 0.05)
        # FIXED: Compliance fatigue parameters
        cfg.setdefault("fatigue_rate_per_day", 0.02)
        cfg.setdefault("fatigue_recovery_per_day", 0.015)
        # FIXED: Media/authority campaign knobs
        cfg.setdefault("campaign_effect_size", 0.15)
        cfg.setdefault("message_frequency_per_day", 1.0)

    def _build_population(self) -> None:
        """
        Create persons with heterogeneous attributes, including demographic groups, initial
        misinformation exposure, and compliance fatigue.

        Notes:
            - Demographic groups are assigned for homophily and inequality metrics.
            - Initial misinformation exposure sampled from [0, 0.2].
        """
        pass
        n = int(self.cfg.get("population_size", 1000))
        init_rate = float(self.cfg.get("initial_adoption_rate", 0.1))
        mask_price = float(self.cfg.get("mask_price", 1.0))
        people: List[Person] = []
        rng = self.rng

        demo_groups = ["group_A", "group_B", "group_C"]  # FIXED: Demographic groups
        demo_weights = [0.4, 0.4, 0.2]

        # Income distribution (lognormal-ish using random.gauss on log)
        for i in range(n):
            age = max(18, int(rng.gauss(40, 15)))
            gender = 1 if rng.random() < 0.5 else 0
            occ = rng.choices(["essential", "non_essential", "student", "retired"], weights=[0.2, 0.5, 0.2, 0.1])[0]
            risk = rng.choices(["low", "medium", "high"], weights=[0.6, 0.3, 0.1])[0]
            # Lognormal-like income
            log_income = rng.gauss(10.2, 0.5)  # meanlog, sdlog
            income = math.exp(log_income)
            trust_auth = min(1.0, max(0.0, rng.betavariate(2, 2))) if hasattr(rng, "betavariate") else clip01(rng.random())
            trust_media = clip01(trust_auth + rng.uniform(-0.2, 0.2))
            soc_sus = clip01((rng.betavariate(2, 3) if hasattr(rng, "betavariate") else rng.random()))
            risk_perc = clip01(rng.uniform(0.1, 0.4))
            perceived_cost = max(0.0, rng.gauss(mask_price, 0.2 * mask_price))
            mask_access = rng.random() < 0.85
            stock = max(0, int(rng.expovariate(1 / 2.0))) if hasattr(rng, "expovariate") else rng.randrange(0, 4)
            adoption = rng.random() < init_rate
            comp_prob = rng.uniform(0.4, 0.8)
            demo = rng.choices(demo_groups, weights=demo_weights)[0]  # FIXED
            p = Person(
                id=i,
                age=age,
                gender=gender,
                household_id=-1,
                occupation=occ,
                health_risk_level=risk,
                demographic_group=demo,  # FIXED
                trust_in_authorities=trust_auth,
                trust_in_media=trust_media,
                susceptibility_to_social_influence=soc_sus,
                risk_perception=risk_perc,
                exposure_to_misinformation=clip01(rng.uniform(0.0, 0.2)),  # FIXED
                compliance_fatigue=0.0,  # FIXED
                perceived_mask_cost=perceived_cost,
                mask_access=mask_access,
                mask_stock=stock,
                is_wearing_mask=adoption and stock > 0 and mask_access,
                compliance_probability=comp_prob,
                belief_state=rng.gauss(0.0, 1.0),
                social_ties=[],
                income=income,
            )
            people.append(p)
        self.people = people
        self._build_social_network()

    def _build_social_network(self) -> None:
        """
        Generate social network ties based on config topology with optional homophily-biased rewiring.

        Notes:
            - Initial graph generated by networkx if available; otherwise ring lattice with rewiring.
            - A homophily-level parameter applies biased rewiring to increase within-group connections.
        """
        pass
        n = len(self.people)
        topology = str(self.cfg.get("network_topology", "small_world"))
        avg_k = int(self.cfg.get("average_degree", 8))
        avg_k = max(2, min(avg_k, n - 1))
        if avg_k % 2 == 1:
            avg_k += 1  # FIXED: Ensure even average degree for small-world ring lattice

        # Attempt to use networkx for better generation if available
        if nx is not None:
            try:
                if "small_world" in topology:
                    beta = float(self.cfg.get("rewiring_prob_small_world", 0.05))
                    G = nx.watts_strogatz_graph(n, avg_k, beta, seed=int(self.cfg.get("random_seed", 42)))
                elif "scale_free" in topology:
                    m = max(1, avg_k // 2)
                    G = nx.barabasi_albert_graph(n, m, seed=int(self.cfg.get("random_seed", 42)))
                else:
                    p = avg_k / (n - 1)
                    G = nx.erdos_renyi_graph(n, p, seed=int(self.cfg.get("random_seed", 42)))
                for u, v in G.edges:
                    self.people[u].social_ties.append(v)
                    self.people[v].social_ties.append(u)
                # FIXED: Deduplicate social ties to prevent inflated norms
                for p in self.people:
                    p.social_ties = list(set(p.social_ties))
            except Exception:
                # Fallback to simple method below
                pass

        if not any(p.social_ties for p in self.people):
            # Fallback: ring lattice + random rewirings
            for i in range(n):
                # Ring neighbors
                for d in range(1, avg_k // 2 + 1):
                    j = (i + d) % n
                    self.people[i].social_ties.append(j)
                    self.people[j].social_ties.append(i)
            # Light rewiring to introduce randomness
            beta = float(self.cfg.get("rewiring_prob_small_world", 0.05))
            m_rewire = int(beta * n * avg_k // 2)
            for _ in range(m_rewire):
                a = self.rng.randrange(0, n)
                b = self.rng.randrange(0, n)
                if b != a and b not in self.people[a].social_ties:
                    self.people[a].social_ties.append(b)
                    self.people[b].social_ties.append(a)
            # FIXED: Deduplicate ties
            for p in self.people:
                p.social_ties = list(set(p.social_ties))

        # FIXED: Apply homophily-biased rewiring
        hom = float(self.cfg.get("homophily_level", 0.0))
        if hom > 0.0:
            # Attempt limited rewiring steps proportional to hom * n
            steps = int(hom * n)
            for _ in range(steps):
                a = self.rng.randrange(0, n)
                if not self.people[a].social_ties:
                    continue
                b = self.rng.choice(self.people[a].social_ties)
                if self.people[a].demographic_group != self.people[b].demographic_group:
                    c = self.rng.randrange(0, n)
                    # Prefer same-group c not already connected
                    if c != a and self.people[c].demographic_group == self.people[a].demographic_group and c not in self.people[a].social_ties:
                        # Remove a-b
                        try:
                            self.people[a].social_ties.remove(b)
                        except ValueError:
                            pass
                        try:
                            self.people[b].social_ties.remove(a)
                        except ValueError:
                            pass
                        # Add a-c
                        self.people[a].social_ties.append(c)
                        self.people[c].social_ties.append(a)
        # Deduplicate again
        for p in self.people:
            p.social_ties = list(set(p.social_ties))

    def _build_households(self) -> None:
        """
        Build households with sizes and incomes.

        Notes:
            - Household income is the average of member incomes.
        """
        pass
        n = len(self.people)
        sizes = []
        # Probabilities for sizes {1,2,3,4,5}
        probs = [0.25, 0.30, 0.20, 0.15, 0.10]
        options = [1, 2, 3, 4, 5]
        remaining = n
        while remaining > 0:
            s = self.rng.choices(options, weights=probs)[0]
            s = min(s, remaining)
            sizes.append(s)
            remaining -= s
        hh_list: List[Household] = []
        pid = 0
        for hid, s in enumerate(sizes):
            member_ids = list(range(pid, pid + s))
            pid += s
            # Income as average of members' incomes
            income = sum(self.people[m].income for m in member_ids) / float(s or 1)
            hh = Household(id=hid, member_ids=member_ids, income=income, norms=0.0, mask_stock=max(0, int(self.rng.gauss(3, 1))), home_location_id=hid)
            hh_list.append(hh)
        self.households = hh_list

    def _assign_households(self) -> None:
        """
        Assign people to households by updating their household_id.
        """
        pass
        for hh in self.households:
            for pid in hh.member_ids:
                if 0 <= pid < len(self.people):
                    self.people[pid].household_id = hh.id

    def _build_locations(self) -> None:
        """
        Build home locations (one per household), workplaces, schools, and public spaces.
        """
        pass
        # Homes
        for hh in self.households:
            loc = Location(id=hh.home_location_id, type="home", capacity=len(hh.member_ids), risk_level=0.1, mask_requirement=False, enforcement_level=0.0)
            self.locations_by_type["home"].append(loc)
            self.loc_index["home"][loc.id] = loc

        # Workplaces
        num_workplaces = int(self.cfg.get("num_workplaces", max(1, len(self.people) // 25)))
        for wid in range(num_workplaces):
            cap = max(5, int(self.rng.gauss(20, 7)))
            loc = Location(id=wid, type="workplace", capacity=cap, risk_level=0.3, mask_requirement=False, enforcement_level=self.rng.uniform(0.3, 0.9))
            self.locations_by_type["workplace"].append(loc)
            self.loc_index["workplace"][loc.id] = loc

        # Schools
        num_schools = max(1, len(self.people) // 200)
        for sid in range(num_schools):
            cap = max(10, int(self.rng.gauss(50, 20)))
            loc = Location(id=sid, type="school", capacity=cap, risk_level=0.35, mask_requirement=False, enforcement_level=self.rng.uniform(0.3, 0.9))
            self.locations_by_type["school"].append(loc)
            self.loc_index["school"][loc.id] = loc

        # Public spaces
        num_public = max(5, len(self.people) // 80)
        for cid in range(num_public):
            cap = max(5, int(self.rng.gauss(30, 10)))
            loc = Location(id=cid, type="public_space", capacity=cap, risk_level=0.25, mask_requirement=False, enforcement_level=self.rng.uniform(0.3, 0.9))
            self.locations_by_type["public_space"].append(loc)
            self.loc_index["public_space"][loc.id] = loc

    def _assign_work_and_school(self) -> None:
        """
        Assign working-age persons to workplaces and students to schools by simple round-robin.

        Notes:
            - Ensures capacities are respected approximately.
        """
        pass
        workplaces = self.locations_by_type["workplace"]
        schools = self.locations_by_type["school"]
        i_w, i_s = 0, 0
        cap_w = [w.capacity for w in workplaces]
        cap_s = [s.capacity for s in schools]
        assigned_work: Dict[int, int] = {}
        assigned_school: Dict[int, int] = {}
        for p in self.people:
            if p.occupation in ("essential", "non_essential"):
                attempts = 0
                while workplaces and attempts < len(workplaces):
                    idx = i_w % len(workplaces)
                    if cap_w[idx] > 0:
                        assigned_work[p.id] = workplaces[idx].id
                        cap_w[idx] -= 1
                        i_w += 1
                        break
                    i_w += 1
                    attempts += 1
            elif p.occupation == "student":
                attempts = 0
                while schools and attempts < len(schools):
                    idx = i_s % len(schools)
                    if cap_s[idx] > 0:
                        assigned_school[p.id] = schools[idx].id
                        cap_s[idx] -= 1
                        i_s += 1
                        break
                    i_s += 1
                    attempts += 1
        self._assigned_work = assigned_work
        self._assigned_school = assigned_school

    def _build_retailers(self) -> None:
        """
        Build retailers with initial inventory and pricing.
        """
        pass
        num_stores = int(self.cfg.get("num_stores", max(5, len(self.people) // 100)))
        price = float(self.cfg.get("mask_price", 1.0))
        restock_rate = int(self.cfg.get("supply_capacity_per_day", 5000) // max(1, num_stores))
        limit = int(self.cfg.get("rationing_limit_per_person", 10))
        markup = float(self.cfg.get("retailer_markup", 0.2))
        elast = float(self.cfg.get("price_elasticity_of_demand", -0.5))
        cv = float(self.cfg.get("supply_variability_cv", 0.1))
        retailers = []
        for rid in range(num_stores):
            inv = max(0, int(self.rng.gauss(200, 50)))
            r = Retailer(id=rid, inventory=inv, restock_rate=restock_rate, price=price, rationing_limit_per_person=limit, markup=markup, price_elasticity_of_demand=elast, supply_variability_cv=cv)
            retailers.append(r)
        self.retailers = retailers

    def _build_media(self) -> None:
        """
        Build media outlets per config schedule and reach.

        Notes:
            - Single outlet for simplicity with configurable misinformation_rate and reach.
        """
        pass
        campaign_start_day = int(self.cfg.get("campaign_start_day", 10))
        days = int(self.cfg.get("simulation_days", 90))
        intensity = float(self.cfg.get("campaign_intensity", 0.6))
        message_bias = float(self.cfg.get("message_bias", 0.0))
        outlets = [
            MediaOutlet(
                id=1,
                credibility=0.7,
                message_bias=message_bias,
                misinformation_rate=float(self.cfg.get("misinformation_rate", 0.1)),
                reach=intensity,
                schedule=[{"start_day": campaign_start_day, "end_day": days - 1}],
            )
        ]
        self.media_outlets = outlets

    def _init_series(self) -> None:
        """
        Initialize time series for metrics collection.

        Notes:
            - Tracks adoption rates, both-masked contact share, enforcement actions, churn.
            - Tracks stockouts, mandate flags, campaign intensity, misinformation exposure, and fatigue.
        """
        pass
        self.series = {}
        for key in ["overall_adoption_rate_over_time", "both_masked_contact_share", "enforcement_actions_per_day", "adoption_churn_rate"]:
            self.series[key] = []
        # Adoption by location types
        for ltype in ("home", "workplace", "school", "public_space"):
            self.series[f"adoption_by_location_type_{ltype}"] = []
        # Additional bookkeeping for policy effect (DiD)
        self.series["adoption_high_enforcement"] = []
        self.series["adoption_low_enforcement"] = []
        # FIXED: Track supply shortages (daily flag)
        self.series["retailer_any_stockout"] = []
        # FIXED: Track mandate flag per day for evaluation of noncompliance
        self.series["mandate_active"] = []
        # FIXED: Track campaign intensity series for information elasticity
        self.series["campaign_intensity_series"] = []
        # FIXED: Track inequality indices
        self.series["adoption_inequality_index_demog"] = []  # by demographic group
        self.series["adoption_inequality_index_income"] = []  # by income (legacy)
        # FIXED: Track average misinformation exposure and fatigue
        self.series["avg_misinformation_exposure"] = []
        self.series["avg_compliance_fatigue"] = []

    def _update_risk_signal(self, day: int) -> None:
        """
        Update individuals' risk perceptions using RiskEnvironment broadcast with smoothing.

        Args:
            day (int): Day index.
        """
        pass
        # FIXED: Use RiskEnvironment instead of ad-hoc update
        env_val = self.risk_env.step(day, self.disease if self.with_disease else None)
        gain = float(self.cfg.get("risk_signal_to_perception_gain", 0.6))
        for p in self.people:
            p.risk_perception = clip01(p.risk_perception + gain * (env_val - p.risk_perception))

    def _spread_misinformation(self, day: int, campaign_active: bool, campaign_effect_size: float) -> None:
        """
        Spread misinformation across social ties with homophily amplification and decay.
        Also adjusts risk perception and trust_in_authorities based on exposure.

        Args:
            day (int): Day index.
            campaign_active (bool): Whether a counter-campaign is active.
            campaign_effect_size (float): Strength of counter-campaign effect.
        """
        pass
        spread_rate = float(self.cfg.get("misinformation_spread_rate", 0.2))
        decay_rate = float(self.cfg.get("misinformation_decay_rate", 0.05))
        hom_amp = float(self.cfg.get("homophily_amplification", 0.3))
        new_exposures = [0.0] * len(self.people)
        for i, p in enumerate(self.people):
            ties = p.social_ties
            if not ties:
                neighbor_avg = 0.0
            else:
                num = 0.0
                den = 0.0
                for j in ties:
                    w = 1.0 + (hom_amp if self.people[j].demographic_group == p.demographic_group else 0.0)
                    den += w
                    num += w * clip01(self.people[j].exposure_to_misinformation)
                neighbor_avg = (num / den) if den > 0 else 0.0
            # Update exposure with decay and diffusion
            e = (1.0 - decay_rate) * clip01(p.exposure_to_misinformation) + spread_rate * neighbor_avg
            # Campaign counteracts misinformation
            if campaign_active:
                e = clip01(e - 0.2 * campaign_effect_size * (0.5 * (p.trust_in_authorities + p.trust_in_media)))
            new_exposures[i] = clip01(e)

        # Apply updated exposures and adjust perceptions/trust
        for i, p in enumerate(self.people):
            p.exposure_to_misinformation = new_exposures[i]
            # Decrease risk perception and trust proportional to exposure
            p.risk_perception = clip01(p.risk_perception - 0.1 * spread_rate * new_exposures[i] + (0.05 * campaign_effect_size if campaign_active else 0.0))
            p.trust_in_authorities = clip01(p.trust_in_authorities - 0.05 * new_exposures[i] + (0.03 * campaign_effect_size if campaign_active else 0.0))

    def _neighbor_mask_norm(self, pid: int, prev_states: List[bool]) -> float:
        """
        Compute neighbor mask adoption norm for a person given previous states.

        Args:
            pid (int): Person ID.
            prev_states (List[bool]): Previous mask wearing states.

        Returns:
            float: Neighbor adoption rate.
        """
        pass
        ties = self.people[pid].social_ties
        if not ties:
            return 0.0
        return sum(1 for j in ties if prev_states[j]) / float(len(ties))

    def _household_mask_norm(self, pid: int, prev_states: List[bool]) -> float:
        """
        Compute household-specific mask norm.

        Args:
            pid (int): Person ID.
            prev_states (List[bool]): Previous mask wearing states.

        Returns:
            float: Household adoption rate.
        """
        pass
        hh_id = self.people[pid].household_id
        if hh_id < 0 or hh_id >= len(self.households):
            return 0.0
        members = self.households[hh_id].member_ids
        if not members:
            return 0.0
        return sum(1 for j in members if prev_states[j]) / float(len(members))

    def _attend_locations_today(self, mandate_active: bool) -> Dict[str, Dict[int, List[int]]]:
        """
        Create attendance lists per location type and location ID for today.
        Respects mandate-aware public-space mobility and capacity.

        Args:
            mandate_active (bool): Whether a mandate is active.

        Returns:
            Dict[str, Dict[int, List[int]]]: Attendance mapping by location type and IDs.
        """
        pass
        attendance: Dict[str, Dict[int, List[int]]] = {"home": {}, "workplace": {}, "school": {}, "public_space": {}}
        # Homes
        for hh in self.households:
            attendance["home"][hh.home_location_id] = list(hh.member_ids)

        # Work and school
        for p in self.people:
            if p.occupation in ("essential", "non_essential") and p.id in self._assigned_work:
                wid = self._assigned_work[p.id]
                attendance["workplace"].setdefault(wid, []).append(p.id)
            elif p.occupation == "student" and p.id in self._assigned_school:
                sid = self._assigned_school[p.id]
                attendance["school"].setdefault(sid, []).append(p.id)

        # Public spaces: sample each person with some chance attenuated under mandates
        base_prob = float(self.cfg.get("public_space_attendance_base_prob", 0.3))
        mobility_reduction = float(self.cfg.get("mobility_reduction_under_mandate", 0.1))
        # FIXED: Use actual mandate_active flag to reduce public-space attendance under mandates
        prob = base_prob * (1.0 - mobility_reduction if mandate_active else 1.0)
        pub_locs = self.locations_by_type["public_space"]
        for p in self.people:
            if self.rng.random() < prob and pub_locs:
                # FIXED: Respect capacity by trying a few random locations
                for _ in range(3):
                    loc = self.rng.choice(pub_locs)
                    lst = attendance["public_space"].setdefault(loc.id, [])
                    if len(lst) < loc.capacity:
                        lst.append(p.id)
                        break

        return attendance

    def _mask_gini_by_income_deciles(self, states: List[bool]) -> float:
        """
        Compute Gini of adoption across income deciles.

        Args:
            states (List[bool]): Adoption states.

        Returns:
            float: Gini coefficient across income deciles.
        """
        pass
        n = len(self.people)
        if n == 0:
            return 0.0
        # Pair income with state
        pairs = [(self.people[i].income, 1.0 if states[i] else 0.0) for i in range(n)]
        pairs.sort(key=lambda x: x[0])
        # Split into 10 deciles
        deciles = []
        size = max(1, n // 10)
        for i in range(0, n, size):
            chunk = pairs[i:i + size]
            if not chunk:
                continue
            adoption = sum(v for _, v in chunk) / float(len(chunk))
            deciles.append(adoption)
        return gini(deciles)

    def _adoption_gini_by_demog(self, states: List[bool]) -> float:
        """
        Compute Gini of adoption across demographic groups.

        Args:
            states (List[bool]): Adoption states.

        Returns:
            float: Gini coefficient across demographic group adoption rates.
        """
        pass
        by_group: Dict[str, List[int]] = {}
        for i, p in enumerate(self.people):
            by_group.setdefault(p.demographic_group, []).append(1 if states[i] else 0)
        group_rates = [sum(v) / float(len(v)) for v in by_group.values() if v]
        return gini(group_rates)

    def _host_contacts_and_transmissions(self, attendance: Dict[str, Dict[int, List[int]]], mandate_active: bool) -> Tuple[int, Dict[str, float], float]:
        """
        Host contacts at attended locations and simulate transmissions if enabled.
        Returns (new_infections_today, adoption_by_loc_type, both_masked_contact_share).

        Args:
            attendance (Dict[str, Dict[int, List[int]]]): Attendance mapping.
            mandate_active (bool): Whether a mandate is active.

        Returns:
            Tuple[int, Dict[str, float], float]: New infections, adoption by location type, both masked share.
        """
        pass
        states_mask = [p.is_wearing_mask for p in self.people]
        adoption_loc: Dict[str, float] = {}
        both_masked_total = 0
        contacts_total = 0

        contact_rates = self.cfg.get("contact_rate_by_location", {"home": 3, "workplace": 6, "school": 8, "public_space": 4})  # FIXED: standardize usage

        # FIXED: Apply enforcement filtering even when disease is disabled and standardize contact sampling
        if not self.with_disease or self.disease is None:
            for ltype, locs in attendance.items():
                allow_map: Dict[int, List[int]] = {}
                for loc_id, lst in locs.items():
                    loc_obj = self.loc_index.get(ltype, {}).get(loc_id)
                    enforced = getattr(loc_obj, "enforcement_level", 0.0) if loc_obj else 0.0
                    allowed = []
                    for pid in lst:
                        if mandate_active and loc_obj and getattr(loc_obj, "mask_requirement", False):
                            if not states_mask[pid] and (self.rng.random() < enforced):
                                self._today_enforcement_events += 1  # FIXED: Count enforcement actions in non-disease mode
                                continue
                        allowed.append(pid)
                    allow_map[loc_id] = allowed
                attendees = [pid for _, lst in allow_map.items() for pid in lst]
                adoption_loc[ltype] = (sum(1 for pid in attendees if states_mask[pid]) / float(len(attendees))) if attendees else 0.0
                # Sample contacts to estimate both-masked share using contact rates
                for _, lst in allow_map.items():
                    m = len(lst)
                    if m < 2:
                        continue
                    loc_contacts = max(1, int(contact_rates.get(ltype, 2)))  # FIXED
                    for _ in range(loc_contacts):
                        a, b = self.rng.sample(lst, 2)
                        contacts_total += 1
                        both_masked_total += 1 if (states_mask[a] and states_mask[b]) else 0
            bm_share = (both_masked_total / float(max(1, contacts_total)))
            return 0, adoption_loc, bm_share

        new_infections_set: set = set()
        for ltype, locs in attendance.items():
            # Apply enforcement by possibly denying entry to unmasked if required
            allow_map: Dict[int, List[int]] = {}
            for loc_id, lst in locs.items():
                loc_obj = self.loc_index.get(ltype, {}).get(loc_id)
                enforced = getattr(loc_obj, "enforcement_level", 0.0) if loc_obj else 0.0
                allowed = []
                for pid in lst:
                    if mandate_active and loc_obj and getattr(loc_obj, "mask_requirement", False):
                        if not states_mask[pid] and (self.rng.random() < enforced):
                            self._today_enforcement_events += 1  # FIXED: Count enforcement actions
                            continue
                    allowed.append(pid)
                allow_map[loc_id] = allowed

            attendees = [pid for _, lst in allow_map.items() for pid in lst]
            adoption_loc[ltype] = (sum(1 for pid in attendees if states_mask[pid]) / float(len(attendees))) if attendees else 0.0

            # Sample contacts and simulate transmissions
            for _, lst in allow_map.items():
                m = len(lst)
                if m < 2:
                    continue
                loc_contacts = max(1, int(contact_rates.get(ltype, 2)))
                for _ in range(loc_contacts):
                    a, b = self.rng.sample(lst, 2)
                    contacts_total += 1
                    both_masked_total += 1 if (states_mask[a] and states_mask[b]) else 0
                    ia, ib = self.disease.state[a] == "I", self.disease.state[b] == "I"
                    if ia != ib:
                        src = a if ia else b
                        dst = b if ia else a
                        if self.disease.state[dst] == "S":
                            p_tx = self.disease.transmit_given_contact(states_mask[src], states_mask[dst], ltype)
                            if self.rng.random() < p_tx:
                                new_infections_set.add(dst)

        new_infections = self.disease.apply_infections(list(new_infections_set))
        bm_share = (both_masked_total / float(max(1, contacts_total)))
        return int(new_infections), adoption_loc, bm_share

    def _compute_policy_effect_size(self) -> float:
        """
        Compute a simple DiD-like policy effect using high vs low enforcement adoption rates.

        Returns:
            float: DiD effect size (post-pre difference-in-differences).
        """
        pass
        series_high = self.series.get("adoption_high_enforcement", [])
        series_low = self.series.get("adoption_low_enforcement", [])
        if not series_high or not series_low:
            return 0.0
        day0 = int(self.cfg.get("mandate_day", 30))
        pre_h = [v for i, v in enumerate(series_high) if i < day0]
        pre_l = [v for i, v in enumerate(series_low) if i < day0]
        post_h = [v for i, v in enumerate(series_high) if i >= day0]
        post_l = [v for i, v in enumerate(series_low) if i >= day0]
        if not pre_h or not pre_l or not post_h or not post_l:
            return 0.0
        pre_diff = sum(pre_h) / len(pre_h) - sum(pre_l) / len(pre_l)
        post_diff = sum(post_h) / len(post_h) - sum(post_l) / len(post_l)
        return float(post_diff - pre_diff)

    def run(self) -> Dict[str, Any]:
        """
        Execute the simulation loop and return results including metrics time series and summary values.

        Returns:
            Dict[str, Any]: Results including time series and summary statistics.
        """
        pass
        days = int(self.cfg.get("simulation_days", 90))
        self._states_history = [[p.is_wearing_mask for p in self.people]]
        time_to_50 = None
        time_to_70 = None  # FIXED: Track time-to-70
        time_to_90 = None  # FIXED: Track time-to-90
        sustained_threshold = float(self.cfg.get("ttt_level_70", 0.7))
        sustained_above = 0

        # Identify high vs low enforcement locations (non-home)
        thresh = float(self.cfg.get("high_enforcement_threshold", 0.7))
        high_locs = set()
        low_locs = set()
        for ltype in ("workplace", "school", "public_space"):
            for loc in self.locations_by_type.get(ltype, []):
                if loc.enforcement_level >= thresh:
                    high_locs.add((ltype, loc.id))
                else:
                    low_locs.add((ltype, loc.id))

        # Restock interval
        restock_interval = int(self.cfg.get("restock_interval_days", 1))

        for t in range(days):
            self._today_enforcement_events = 0
            policy = self.policy.step(t)  # FIXED: Use PublicHealthAuthority
            mandate_active = bool(policy.get("mandate_active", False))
            self.series["mandate_active"].append(1 if mandate_active else 0)

            # Apply mandate to locations (non-home)
            for ltype in ("workplace", "school", "public_space"):
                for loc in self.locations_by_type.get(ltype, []):
                    loc.mask_requirement = mandate_active

            # Media broadcasts with message frequency per day
            todays_intensity = 0.0
            for m in self.media_outlets:
                m.broadcast(
                    self.people,
                    float(self.cfg.get("media_influence_strength", 0.3)),
                    self.rng,
                    t,
                    message_frequency_per_day=float(policy.get("message_frequency_per_day", self.cfg.get("message_frequency_per_day", 1.0))),
                )  # FIXED: pass message frequency
                # Aggregate intensity if active today
                for win in m.schedule:
                    if win.get("start_day", 0) <= t <= win.get("end_day", 10**9):
                        todays_intensity = max(todays_intensity, m.reach)
            self.series["campaign_intensity_series"].append(todays_intensity)

            # Update network misinformation diffusion and counter-campaign effects
            self._spread_misinformation(t, campaign_active=bool(policy.get("campaign_active", False)), campaign_effect_size=float(policy.get("campaign_effect_size", 0.0)))  # FIXED

            # Update risk via RiskEnvironment
            self._update_risk_signal(t)

            # Household norms update
            for hh in self.households:
                hh.share_norms(self.people)

            # FIXED: Replace-mask first, then purchases, then decisions to avoid underestimation
            for p in self.people:
                p.replace_mask(t, int(self.cfg.get("mask_replacement_interval_days", 7)))

            # Purchases after consumption
            unmet_demand_today = 0  # FIXED: Track unmet demand to compute supply shortage days
            if self.with_supply:
                for p in self.people:
                    if p.mask_stock < int(self.cfg.get("purchase_threshold", 1)):
                        desired_before = min(int(self.cfg.get("rationing_limit_per_person", 10)), int(self.cfg.get("purchase_threshold", 1)) + 3 - p.mask_stock)
                        inv_before = sum(r.inventory for r in self.retailers)
                        purchased = p.purchase_mask(
                            self.retailers,
                            int(self.cfg.get("purchase_threshold", 1)),
                            self.rng,
                            int(self.cfg.get("rationing_limit_per_person", 10)),
                        )
                        if desired_before > purchased and inv_before > 0:
                            unmet_demand_today += (desired_before - purchased)

                # Retailers maintenance with restock cadence
                for r in self.retailers:
                    # FIXED: Honor restock interval days
                    if restock_interval <= 1 or (t % restock_interval) == 0:
                        r.restock_daily(self.rng)
                    r.adjust_price()
                # Mark stockout day if unmet demand or any retailer hits zero inventory
                any_stockout = (unmet_demand_today > 0) or any(r.inventory <= 0 for r in self.retailers)
                self.series["retailer_any_stockout"].append(bool(any_stockout))
            else:
                self.series["retailer_any_stockout"].append(False)

            # Decisions based on norms and policy after replenishment
            prev_states = self._states_history[-1]
            for p in self.people:
                social_norm = self._neighbor_mask_norm(p.id, prev_states)
                household_norm = self._household_mask_norm(p.id, prev_states)
                # Effective enforcement level combines global enforcement probability and policy strength
                enforcement_level = float(policy.get("enforcement_probability", 0.0)) * float(policy.get("policy_strength", 0.0))
                p.decide_wear_mask(
                    social_norm=social_norm,
                    mandate_active=mandate_active,
                    enforcement_level=enforcement_level,
                    household_norm=household_norm,
                    household_weight=float(self.cfg.get("household_norm_weight", 0.3)),
                    rng=self.rng,
                    peer_w=float(self.cfg.get("peer_influence_strength", 0.5)),
                    media_w=float(self.cfg.get("media_influence_strength", 0.3)),
                    authority_w=float(self.cfg.get("authority_influence_strength", 0.2)),
                    conformity_threshold=float(self.cfg.get("conformity_threshold", 0.1)),
                    social_w=float(self.cfg.get("social_influence_weight", self.cfg.get("peer_influence_strength", 0.5))),  # FIXED: Map spec weights
                    policy_w=float(self.cfg.get("policy_pressure_weight", self.cfg.get("authority_influence_strength", 0.2))),  # FIXED
                    risk_w=float(self.cfg.get("risk_perception_weight", self.cfg.get("media_influence_strength", 0.3))),  # FIXED
                    adoption_threshold=float(self.cfg.get("adoption_threshold", 0.6)),  # FIXED
                )

            # FIXED: Update compliance fatigue after decisions
            for p in self.people:
                p.update_fatigue(
                    wearing=p.is_wearing_mask,
                    fatigue_rate=float(self.cfg.get("fatigue_rate_per_day", 0.02)),
                    recovery_rate=float(self.cfg.get("fatigue_recovery_per_day", 0.015)),
                )

            # Attendance and contacts
            attendance = self._attend_locations_today(mandate_active)  # FIXED: Pass mandate_active into attendance

            # Compute adoption splits high vs low enforcement (for DiD)
            states_mask = [p.is_wearing_mask for p in self.people]

            def adoption_for_locset(locset: set) -> float:
                ids = []
                for ltype, loc_id in locset:
                    ids += attendance.get(ltype, {}).get(loc_id, [])
                return (sum(1 for pid in ids if states_mask[pid]) / float(len(ids))) if ids else 0.0

            high_adopt = adoption_for_locset(high_locs)
            low_adopt = adoption_for_locset(low_locs)
            self.series["adoption_high_enforcement"].append(high_adopt)
            self.series["adoption_low_enforcement"].append(low_adopt)

            # Host contacts and optional disease
            new_inf, adoption_by_loc, bm_share = self._host_contacts_and_transmissions(attendance, mandate_active)
            if self.with_disease and self.disease is not None:
                I_prev = sum(1 for s in self.disease.state if s == "I")
                self.disease.record_daily_metrics(I_prev, new_inf)

            # Daily metrics
            today_states = [p.is_wearing_mask for p in self.people]
            adoption = sum(1 for s in today_states if s) / float(len(today_states) or 1)
            self.series["overall_adoption_rate_over_time"].append(adoption)
            self.series["both_masked_contact_share"].append(bm_share)
            self.series["enforcement_actions_per_day"].append(self._today_enforcement_events)
            for ltype, rate in adoption_by_loc.items():
                self.series[f"adoption_by_location_type_{ltype}"].append(rate)
            # FIXED: Inequality metrics alignment
            self.series["adoption_inequality_index_demog"].append(self._adoption_gini_by_demog(today_states))  # FIXED
            self.series["adoption_inequality_index_income"].append(self._mask_gini_by_income_deciles(today_states))  # FIXED
            flips = sum(1 for i in range(len(today_states)) if today_states[i] != prev_states[i])
            self.series["adoption_churn_rate"].append(flips / float(len(today_states) or 1))
            # FIXED: Track averages for misinformation exposure and fatigue
            self.series["avg_misinformation_exposure"].append(sum(p.exposure_to_misinformation for p in self.people) / float(len(self.people) or 1))
            self.series["avg_compliance_fatigue"].append(sum(p.compliance_fatigue for p in self.people) / float(len(self.people) or 1))
            self._states_history.append(today_states)

            # Threshold metrics
            if time_to_50 is None and adoption >= 0.5:
                time_to_50 = t
            if time_to_70 is None and adoption >= 0.7:
                time_to_70 = t  # FIXED: record time to 70%
            if time_to_90 is None and adoption >= 0.9:
                time_to_90 = t  # FIXED: record time to 90%
            sustained_above += 1 if adoption >= sustained_threshold else 0

        # FIXED: Results alignment with spec naming and aliases
        overall_series = self.series.get("overall_adoption_rate_over_time", [])
        final_adoption = overall_series[-1] if overall_series else 0.0
        avg_adoption = sum(overall_series) / float(len(overall_series) or 1)
        peak_adoption = max(overall_series) if overall_series else 0.0
        peak_day = int(overall_series.index(peak_adoption)) if overall_series else None
        policy_effect = self._compute_policy_effect_size()
        supply_shortage_days = int(sum(1 for v in self.series.get("retailer_any_stockout", []) if v))

        disease_out = None
        if self.with_disease and self.disease is not None:
            total_new_infections = int(sum(self.disease.series_new_infections))
            final_I = int(self.disease.series_I[-1]) if self.disease.series_I else 0
            disease_out = {
                "series_I": self.disease.series_I,
                "series_new_infections": self.disease.series_new_infections,
                "total_new_infections": total_new_infections,
                "final_I": final_I,
            }

        results = {
            "series": self.series,
            "summary": {
                "final_adoption_rate": float(final_adoption),
                "average_adoption_rate": float(avg_adoption),
                "peak_adoption_rate": float(peak_adoption),
                "peak_adoption_day": peak_day,
                "time_to_50": time_to_50,
                "time_to_70": time_to_70,
                "time_to_90": time_to_90,
                "sustained_days_above_threshold": sustained_above,
                "policy_effect_size_did": float(policy_effect),
                "supply_shortage_days": supply_shortage_days,
                "final_avg_misinformation_exposure": float(self.series["avg_misinformation_exposure"][-1]) if self.series["avg_misinformation_exposure"] else 0.0,
                "final_avg_compliance_fatigue": float(self.series["avg_compliance_fatigue"][-1]) if self.series["avg_compliance_fatigue"] else 0.0,
            },
            "disease": disease_out,
        }
        return results


# Execute main for both direct execution and sandbox wrapper invocation
main()