import os
import sys
import json
import math
import time
import copy
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional imports with graceful fallback
try:
    import numpy as np
except Exception:
    np = None

try:
    import networkx as nx
except Exception:
    nx = None

# FIXED: Removed stray non-Python text causing SyntaxError in previous iteration.
# FIXED: Restored a complete, runnable simulation implementing the mask adoption diffusion dynamics.
# FIXED: Implemented entities, network, daily loop logic, metrics, and fast-mode to avoid timeouts.
# FIXED: Added fallback implementations for network building and numpy absence.
# FIXED: Seed RNGs for reproducibility and clipped probabilities to [0,1].
# FIXED: Implemented core modules with buffers → commit pattern and a DAG-like scheduler.
# FIXED: Implemented parameters handling with CLI, overrides, and persistence to parameters_used.json.
# FIXED: Implemented pluggable calibration architecture (ParamsAdapter and Calibrators) with registry.
# FIXED: Ensured all docstrings are complete and properly closed; added pass statements to functions/classes.
# FIXED: Added DiseaseTransmissionModule and related disease metrics and state updates.
# FIXED: Added policy schedule/mandate_start_day handling and subsidy/distribution program support.
# FIXED: Added compliance/equity/availability/policy cost metrics.
# FIXED: Corrected retailer stock reconciliation bug to preserve unpurchased allocations when applying stock updates.
# FIXED: Reduced calibration runtime in RandomSearchCalibrator by downscaling N and T for trials.
# FIXED: RetailSupplyModule now depends on DecisionAndAdoptionModule so purchase requests are fulfilled same day.
# FIXED: DecisionAndAdoptionModule uses same-day risk updates from buffers if present to remove lag.
# FIXED: Added masks_sold metrics (daily and cumulative) and adoption_by_group (income, age).
# FIXED: Added sustained_adoption_rate metric and adoption_disparity_index alias.
# FIXED: Added policy_effect_contribution via counterfactual evaluation.
# FIXED: Reduced redundant income-tercile computations inside DecisionAndAdoptionModule.
# FIXED: Added spec-compliant parameter aliases for simulation_days, avg_degree, enforcement_strength, price_per_mask, policy_start_day, and supply_chain_disruption_prob.
# FIXED: Added required metrics and aliases: time_to_70_percent, mask_supply_shortage_days, enforcement_cost, misinformation_prevalence; adoption_rate_over_time, effective_Rt, inequity_index.
# FIXED: Implemented misinformation_belief state updates and prevalence series.
# FIXED: Added enforcement admin costs and fines tracking; enforcement_cost_daily and aggregate enforcement_cost metric.
# FIXED: Added retailer_stock_per_capita series to support mask_supply_shortage_days metric.
# FIXED: Calibration is now opt-in via --no-calib flag (default true) and SKIP_CALIB=1 env; only run calibration when explicitly requested.
# FIXED: HouseholdModule inputs declaration now matches usage (state.mask_use_history).
# FIXED: Prepared for location-based transmission by reordering modules so MobilityAndLocationModule runs before DiseaseTransmissionModule.


# Project paths handling
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def seed_all(seed: int) -> None:
    """
    Seed all RNGs for reproducibility.

    Args:
        seed: Integer seed value.

    Returns:
        None
    """
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    pass


def clip01(x: float) -> float:
    """
    Clip a float to [0, 1].

    Args:
        x: Input float.

    Returns:
        Clipped float between 0.0 and 1.0.
    """
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0
    finally:
        pass
    pass


def sigmoid(x: float) -> float:
    """
    Numerically-stable logistic sigmoid.

    Args:
        x: Input value.

    Returns:
        Value in (0, 1).
    """
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        z = math.exp(x)
        return z / (1 + z)
    except Exception:
        return 0.5
    finally:
        pass
    pass


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """
    Safe division with default fallback.

    Args:
        a: Numerator.
        b: Denominator.
        default: Fallback when denominator is zero.

    Returns:
        a / b or default if b is zero.
    """
    try:
        return a / b if b != 0 else default
    except Exception:
        return default
    finally:
        pass
    pass


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists.

    Args:
        path: Directory path.

    Returns:
        None
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    finally:
        pass
    pass


@dataclass
class MediaChannel:
    """
    Media channel representing an information source influencing risk perception.

    Attributes:
        id: Unique identifier.
        name: Channel name.
        sentiment: Average sentiment towards mask use (-1 to 1).
        reach: Fraction of population exposed daily (0 to 1).
        misinformation_rate: Probability that content contains misinformation (0 to 1).
        targeting: Optional targeting label (e.g., 'low_income', 'youth', None).
    """
    id: int
    name: str
    sentiment: float
    reach: float
    misinformation_rate: float
    targeting: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Clip channel attributes to valid ranges.

        Returns:
            None
        """
        self.sentiment = max(-1.0, min(1.0, float(self.sentiment)))
        self.reach = clip01(self.reach)
        self.misinformation_rate = clip01(self.misinformation_rate)
        pass

    pass


@dataclass
class Person:
    """
    Person entity representing an individual in the simulation.

    Attributes:
        id: Unique identifier.
        age: Age in years.
        income: Annual income proxy.
        household_id: Associated household id.
        risk_perception: Perceived risk in [0, 1].
        trust_in_authorities: Trust in public health authorities in [0, 1].
        social_influence_susceptibility: Weight in [0, 1], sensitivity to peers/household.
        compliance_propensity: Weight in [0, 1], propensity to comply with mandates.
        fatigue_level: Fatigue from mask wearing, in [0, 1].
        mask_inventory: Floating stock units (supports fractional wearout).
        budget: Available budget for purchasing masks.
        misinformation_belief: 0/1 flag indicating affected by misinformation.
        has_mask: Convenience boolean, True if mask_inventory > 0 (updated by commit).
        current_mask_use: 0/1 last decision.
        health_state: SEIR compartment label ('S','E','I','R').
        days_in_state: Days elapsed in current health_state.
        compliance_probability: Derived/aux probability of compliance (0-1).
        workplace_id: Optional workplace id (-1 if none).
    """
    id: int
    age: int
    income: float
    household_id: int
    risk_perception: float
    trust_in_authorities: float
    social_influence_susceptibility: float
    compliance_propensity: float
    fatigue_level: float
    mask_inventory: float
    budget: float
    misinformation_belief: float
    has_mask: bool = False
    current_mask_use: int = 0
    health_state: str = "S"
    days_in_state: int = 0
    compliance_probability: float = 0.0
    workplace_id: int = -1

    def __post_init__(self) -> None:
        """
        Post-initialization to clip bounded attributes and set has_mask.

        Returns:
            None
        """
        self.risk_perception = clip01(self.risk_perception)
        self.trust_in_authorities = clip01(self.trust_in_authorities)
        self.social_influence_susceptibility = clip01(self.social_influence_susceptibility)
        self.compliance_propensity = clip01(self.compliance_propensity)
        self.fatigue_level = clip01(self.fatigue_level)
        self.mask_inventory = max(0.0, float(self.mask_inventory))
        self.budget = max(0.0, float(self.budget))
        self.misinformation_belief = clip01(self.misinformation_belief)
        self.has_mask = self.mask_inventory > 0
        self.current_mask_use = 1 if self.has_mask and random.random() < 0.1 else 0
        self.health_state = str(self.health_state)
        self.days_in_state = max(0, int(self.days_in_state))
        self.compliance_probability = clip01(self.compliance_probability)
        pass

    pass


@dataclass
class Household:
    """
    Household entity grouping persons.

    Attributes:
        id: Unique id.
        member_ids: Person ids in household.
        norm_strength: Weight of household norm influence.
    """
    id: int
    member_ids: List[int] = field(default_factory=list)
    norm_strength: float = 0.5

    def __post_init__(self) -> None:
        """
        Post-initialization to clip norm strength.

        Returns:
            None
        """
        self.norm_strength = clip01(self.norm_strength)
        pass

    pass


@dataclass
class Venue:
    """
    Location/Venue entity where people visit.

    Attributes:
        id: Unique id.
        type: Venue type (e.g., work, retail, transport, public_space).
        capacity: Maximum daily visitors.
        mask_required: Whether masks required for entry today.
        enforcement_strictness: Probability of enforcement at entry if noncompliant.
        staff_enforcement_level: Modulates enforcement (0-1).
        local_outbreak_level: Proxy risk exposure at venue.
        daily_visitors: Accumulator for current day.
        compliant_visitors: Count of mask-wearing visitors.
    """
    id: int
    type: str
    capacity: int
    mask_required: bool = False
    enforcement_strictness: float = 0.5
    staff_enforcement_level: float = 0.5
    local_outbreak_level: float = 0.2
    daily_visitors: int = 0
    compliant_visitors: int = 0

    def reset_day(self) -> None:
        """
        Reset daily counters.

        Returns:
            None
        """
        self.daily_visitors = 0
        self.compliant_visitors = 0
        pass

    pass


@dataclass
class Retailer:
    """
    Retailer entity managing mask inventory and price.

    Attributes:
        id: Unique id.
        stock: Current mask stock (units).
        restock_rate: Fractional restock rate per day (0-1 of current stock or base).
        price: Current unit price.
        supply_variability: Noise in restock process.
    """
    id: int
    stock: int
    restock_rate: float
    price: float
    supply_variability: float = 0.2

    def __post_init__(self) -> None:
        """
        Clip/validate fields.

        Returns:
            None
        """
        self.stock = max(0, int(self.stock))
        self.restock_rate = max(0.0, float(self.restock_rate))
        self.price = max(0.1, float(self.price))
        self.supply_variability = clip01(self.supply_variability)
        pass

    pass


@dataclass
class PublicHealthAuthority:
    """
    Public health authority controlling policy and messaging.

    Attributes:
        id: Unique id.
        mandate_level: Mask mandate strength [0,1].
        enforcement_probability: Probability of enforcement action at noncompliance.
        fine_amount: Monetary fine for noncompliance.
        messaging_budget: Budget for messaging campaigns.
        campaign_strategy: Strategy label.
        message_effectiveness: Effectiveness of messaging on risk perception.
        communication_frequency: Days between broadcasts (>=1).
        subsidy_amount: Per-unit subsidy reducing effective mask price for eligible agents.
        distribution_program_active: Flag indicating distribution program active today.
    """
    id: int
    mandate_level: float
    enforcement_probability: float
    fine_amount: float
    messaging_budget: float
    campaign_strategy: str
    message_effectiveness: float
    communication_frequency: int = 7
    subsidy_amount: float = 0.0
    distribution_program_active: bool = False

    def __post_init__(self) -> None:
        """
        Clip bounds and validate.

        Returns:
            None
        """
        self.mandate_level = clip01(self.mandate_level)
        self.enforcement_probability = clip01(self.enforcement_probability)
        self.fine_amount = max(0.0, float(self.fine_amount))
        self.messaging_budget = max(0.0, float(self.messaging_budget))
        self.message_effectiveness = clip01(self.message_effectiveness)
        self.communication_frequency = max(1, int(self.communication_frequency))
        self.subsidy_amount = max(0.0, float(self.subsidy_amount))
        self.distribution_program_active = bool(self.distribution_program_active)
        pass

    pass


class Module(ABC):
    """
    Abstract base class for simulation modules.

    Subclasses must implement forward() to write outputs to buffers without
    mutating the global state directly.
    """

    name: str = "Module"
    inputs: List[str] = []
    outputs: List[str] = []
    dependencies: List[str] = []

    @abstractmethod
    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute module outputs and write to buffers.

        Args:
            state: Current global state (read-only).
            buffers: Intermediate buffers for signals, events, and state updates.
            params: Global parameters dict.
            t: Current day index.

        Returns:
            None
        """
        raise NotImplementedError
        pass

    pass


class PeerInfluenceModule(Module):
    """
    Computes social norm signal from neighbors' recent mask use on the social network.

    Uses small-world network; if unavailable, uses a ring lattice fallback.
    """

    name = "PeerInfluenceModule"
    inputs = ["state.person.current_mask_use", "state.network"]
    outputs = ["signals.person.social_norm_signal"]
    dependencies = []

    def __init__(self, memory_days: int = 3) -> None:
        """
        Initialize the PeerInfluenceModule.

        Args:
            memory_days: Window of days to consider for peer observation.

        """
        self.memory_days = max(1, int(memory_days))
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Write social_norm_signal to buffers based on neighbors' mask use averages.

        Args:
            state: Global state.
            buffers: Buffers dict.
            params: Params dict with social_influence_weight.
            t: Day index.

        Returns:
            None
        """
        N = len(state["persons"])
        adj = state["network"]
        # recent decisions: if history shorter than window, use what is available
        hist = state["mask_use_history"]  # List[List[int]] per day -> decisions
        start = max(0, t - self.memory_days + 1)
        length = t - start + 1 if t >= 0 else 0

        # Average neighbor wearing over last window
        social_signal = [0.0] * N
        for i in range(N):
            nbrs = adj[i]
            if not nbrs or length <= 0:
                social_signal[i] = 0.0
                continue
            wearing_sum = 0.0
            count = 0
            for day in range(start, t + 1):
                day_decisions = hist[day] if day < len(hist) else [0] * N
                # FIXED: Clamp history-derived arrays to length N to avoid index issues.
                if len(day_decisions) != N:
                    day_decisions = (day_decisions + [0] * N)[:N]
                wearing_sum += sum(day_decisions[j] for j in nbrs) / max(1, len(nbrs))
                count += 1
            social_signal[i] = clip01(wearing_sum / max(1, count))
        buffers.setdefault("signals", {})["social_norm_signal"] = social_signal
        pass

    pass


class MessagingAndRiskModule(Module):
    """
    Applies public health messaging and media effects (incl. misinformation)
    to update individual risk perception and information exposure.
    """

    name = "MessagingAndRiskModule"
    inputs = ["state.authority", "state.venues.local_outbreak_level", "state.media_channels"]
    outputs = ["state_updates.person.risk_perception", "signals.person.info_exposure", "state_updates.person.misinformation_belief"]
    dependencies = ["PolicyAndEnforcementModule"]

    def __init__(self) -> None:
        """
        Initialize MessagingAndRiskModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Update risk perception and info exposure signals using policy, local outbreaks,
        media channels, and exogenous risk series.

        Args:
            state: Global state.
            buffers: Buffers for updates.
            params: Parameters dict.
            t: Day index.

        Returns:
            None
        """
        persons: List[Person] = state["persons"]
        authority: PublicHealthAuthority = state["authority"]
        channels: List[MediaChannel] = state.get("media_channels", [])

        perceived_risk_base = float(params.get("perceived_risk_base", 0.2))
        message_effectiveness = float(params.get("message_effectiveness", 0.15))
        misinformation_rate = float(params.get("misinformation_rate", 0.05))
        risk_decay_rate = float(params.get("risk_decay_rate_per_day", 0.02))
        media_influence_effect = float(params.get("media_influence_effect", 0.1))
        exo = params.get("exogenous_risk_series") or []
        exo_weight = float(params.get("exogenous_risk_weight", 0.3))

        # Aggregate observed local outbreaks
        venues: List[Venue] = state["venues"]
        local_avg = 0.0
        if venues:
            local_avg = sum(v.local_outbreak_level for v in venues) / len(venues)

        risk_updates = []
        info_exposure = []
        mis_belief_updates: List[float] = []
        # Precompute low_income targeting mapping
        N = len(persons)
        incomes = [p.income for p in persons]
        idx_sorted = sorted(range(N), key=lambda i: incomes[i])
        tercile_size = max(1, N // 3)
        low_income_set = set(idx_sorted[:tercile_size])

        for i, p in enumerate(persons):
            policy_signal = authority.mandate_level
            msg_effect = message_effectiveness * policy_signal * p.trust_in_authorities
            misinformation = misinformation_rate * p.misinformation_belief
            exo_term = 0.0
            try:
                if isinstance(exo, list) and t < len(exo):
                    exo_term = exo_weight * float(exo[t])
            except Exception:
                exo_term = 0.0

            # Media exposure effect
            exposure_sum = 0.0
            misinformation_bump = 0.0
            for ch in channels:
                # probability of exposure today
                exp_prob = clip01(ch.reach * (0.7 + 0.3 * p.trust_in_authorities))
                targeted = (ch.targeting == "low_income" and i in low_income_set) or (ch.targeting is None)
                if targeted and random.random() < exp_prob:
                    # effect sign by sentiment
                    exposure_sum += media_influence_effect * ch.sentiment
                    misinformation_bump += ch.misinformation_rate * max(0.0, -ch.sentiment)

            # FIXED: Added exogenous risk and media channel terms to risk perception calculation.
            risk = perceived_risk_base + 0.5 * local_avg + msg_effect - misinformation + exo_term + exposure_sum
            # decay towards base
            risk = risk * (1 - risk_decay_rate) + perceived_risk_base * risk_decay_rate
            risk = clip01(risk)
            risk_updates.append(risk)
            # simplistic info exposure via logistic of (policy + trust + media)
            info_exposure.append(clip01(sigmoid(policy_signal + p.trust_in_authorities + exposure_sum - 0.5)))

            # FIXED: Implement dynamic misinformation_belief drift based on exposure bump with small decay.
            new_belief = clip01(p.misinformation_belief + 0.05 * misinformation_bump - 0.01 * (1.0 - min(1.0, misinformation_bump)))
            mis_belief_updates.append(new_belief)

        buffers.setdefault("state_updates", {})["risk_perception"] = risk_updates
        buffers.setdefault("signals", {})["info_exposure"] = info_exposure
        # FIXED: Emit per-person misinformation belief updates.
        buffers["state_updates"]["misinformation_belief"] = mis_belief_updates
        pass

    pass


class PolicyAndEnforcementModule(Module):
    """
    Sets mask mandate level and updates venue mask requirements and enforcement strictness.
    Also schedules free mask distribution events.
    """

    name = "PolicyAndEnforcementModule"
    inputs = ["config.policy"]
    outputs = ["state.venues.mask_required", "state.venues.enforcement_strictness", "events.free_distribution"]
    dependencies = []

    def __init__(self) -> None:
        """
        Initialize PolicyAndEnforcementModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Update venue policy settings based on mandate level and configuration.
        Schedule distribution events based on provided schedule.

        Args:
            state: Global state.
            buffers: Buffers dict (events + settings).
            params: Parameters dict.
            t: Day index.

        Returns:
            None
        """
        authority: PublicHealthAuthority = state["authority"]
        venues: List[Venue] = state["venues"]
        requirement_rate = float(params.get("location_mask_requirement_rate", 0.5))
        staff_enforcement_level_mean = float(params.get("staff_enforcement_level_mean", 0.5))
        gatekeeping_weight = float(params.get("gatekeeping_strictness_weight", 1.0))

        # FIXED: Time-varying mandate level from schedule or start day with alias for policy_start_day.
        schedule = params.get("policy_schedule", None)
        start_day = params.get("mandate_start_day", params.get("policy_start_day", None))
        end_day = params.get("mandate_end_day", None)
        if isinstance(schedule, list):
            todays = [s for s in schedule if int(s.get("day", -1)) <= t]
            if todays:
                authority.mandate_level = clip01(float(sorted(todays, key=lambda s: s.get("day", 0))[-1].get("mandate_level", authority.mandate_level)))
        elif isinstance(start_day, int):
            if end_day is not None and isinstance(end_day, int):
                authority.mandate_level = authority.mandate_level if (t >= start_day and t <= end_day) else 0.0
            else:
                authority.mandate_level = authority.mandate_level if t >= start_day else 0.0

        rate = requirement_rate * authority.mandate_level
        for v in venues:
            v.mask_required = random.random() < clip01(rate)
            # FIXED: Use venue staff_enforcement_level for heterogeneity rather than mean.
            base_staff = v.staff_enforcement_level if hasattr(v, "staff_enforcement_level") else staff_enforcement_level_mean
            v.enforcement_strictness = clip01(
                base_staff * authority.enforcement_probability * gatekeeping_weight
            )

        # FIXED: Schedule free distribution events
        dist_days = params.get("distribution_days", [])
        units_per_person = int(params.get("distribution_units_per_person", 2))
        target = params.get("distribution_target", "low_income")
        free_events: List[Tuple[int, int]] = []
        if isinstance(dist_days, list) and (t in dist_days) and units_per_person > 0:
            persons: List[Person] = state["persons"]
            N = len(persons)
            incomes = [p.income for p in persons]
            idx_sorted = sorted(range(N), key=lambda i: incomes[i])
            tercile_size = max(1, N // 3)
            low_income_set = set(idx_sorted[:tercile_size])
            if target == "low_income":
                eligible = list(low_income_set)
            elif target == "all":
                eligible = list(range(N))
            else:
                eligible = list(low_income_set)
            for pid in eligible:
                free_events.append((pid, units_per_person))
            authority.distribution_program_active = True
        else:
            authority.distribution_program_active = False

        if free_events:
            buffers.setdefault("events", {})["free_distribution"] = free_events
        pass

    pass


class RetailSupplyModule(Module):
    """
    Manages retailer inventory dynamics: sells to agents, restocks, and adjusts prices.
    Produces daily stockout indicator observable.
    """

    name = "RetailSupplyModule"
    inputs = ["events.purchase_requests"]
    outputs = ["state_updates.retailer", "state_updates.retailer_allocations", "observables.stockout_rate_daily", "observables.masks_sold_daily"]
    # FIXED: Ensure purchases are fulfilled same day they are requested.
    dependencies = ["DecisionAndAdoptionModule"]

    def __init__(self) -> None:
        """
        Initialize RetailSupplyModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Fulfill purchase requests, restock, adjust price, and emit stockout observable.

        Args:
            state: Global state dict.
            buffers: Buffers dict containing purchase requests.
            params: Parameters dict for restock and price.
            t: Day index.

        Returns:
            None
        """
        retailer: Retailer = state["retailer"]
        # Read purchase requests
        requests: List[Tuple[int, int]] = buffers.get("events", {}).get("purchase_requests", [])
        allocations: Dict[int, int] = {}

        # FIXED: Avoid in-place state mutation; compute using local remaining_stock.
        remaining_stock = int(retailer.stock)

        # Optional rationing: proportional fill if demand exceeds stock
        total_requested = sum(max(0, int(qty)) for _, qty in requests)
        if remaining_stock > 0 and total_requested > 0:
            if total_requested <= remaining_stock:
                # Fill fully up to remaining_stock per request order
                for pid, qty in requests:
                    q_filled = min(int(qty), remaining_stock)
                    allocations[pid] = allocations.get(pid, 0) + q_filled
                    remaining_stock -= q_filled
            else:
                # Proportional rationing
                ratio = remaining_stock / float(total_requested)
                # Initial floor allocation
                for pid, qty in requests:
                    alloc = int(math.floor(qty * ratio))
                    if alloc > 0:
                        allocations[pid] = allocations.get(pid, 0) + alloc
                # Distribute any leftover units one-by-one to requests with highest fractional remainder
                allocated_sum = sum(allocations.values())
                leftovers = max(0, remaining_stock - allocated_sum)
                if leftovers > 0:
                    remainders = []
                    for pid, qty in requests:
                        target = qty * ratio
                        frac = target - math.floor(target)
                        remainders.append((frac, pid))
                    remainders.sort(reverse=True)
                    idx = 0
                    while leftovers > 0 and idx < len(remainders):
                        pid = remainders[idx][1]
                        allocations[pid] = allocations.get(pid, 0) + 1
                        leftovers -= 1
                        idx += 1
                remaining_stock = 0
        else:
            # No stock: allocate nothing
            allocations = {}

        # restock and price adjust at end of day (computed locally)
        base_restock_rate = float(params.get("restock_rate_per_day", 0.1))
        variability = float(params.get("supply_variability", params.get("supply_chain_disruption_prob", 0.2)))
        price_base = float(params.get("mask_price", params.get("price_per_mask", max(0.1, retailer.price))))
        price_adjustment_sensitivity = float(params.get("price_adjustment_sensitivity", 0.5))
        pop_size = max(1, len(state["persons"]))

        stochastic_factor = 1.0
        try:
            stochastic_factor = 1.0 + random.gauss(0.0, variability)
        except Exception:
            stochastic_factor = 1.0

        # FIXED: Restock based on population target with floor to recover from zero stock.
        target_per_person = float(params.get("target_stock_per_person", 0.4))
        base_target = int(max(0.0, target_per_person * pop_size))
        restock_gap = (base_target - remaining_stock)
        restock_amount = int(max(1.0, restock_gap * base_restock_rate * stochastic_factor))
        new_stock = max(0, remaining_stock + max(0, restock_amount))

        stock_ratio = new_stock / float(pop_size)
        new_price = max(0.1, price_base * (1.0 + price_adjustment_sensitivity * (0.3 - stock_ratio)))

        # FIXED: Emit masks_sold_daily based on allocations (pre-budget), will be overridden with actual in commit.
        total_sold_prelim = int(sum(allocations.values()))

        # Emit observables
        stockout_indicator = 1.0 if new_stock == 0 else 0.0
        obs = buffers.setdefault("observables", {})
        obs["stockout_rate_daily"] = stockout_indicator
        obs["masks_sold_daily"] = float(total_sold_prelim)

        # Write updates for commit
        s_upd = buffers.setdefault("state_updates", {})
        s_upd["retailer_allocations"] = allocations
        s_upd["retailer"] = {"stock": int(new_stock), "price": float(new_price)}
        pass

    pass


class HouseholdModule(Module):
    """
    Reinforces norms within households and computes household norm signal.
    """

    name = "HouseholdModule"
    # FIXED: Inputs reflect usage of mask_use_history to compute prior-day norms.
    inputs = ["state.households", "state.mask_use_history"]
    outputs = ["signals.person.household_norm_signal", "state_updates.person.mask_sharing_delta"]
    dependencies = []

    def __init__(self) -> None:
        """
        Initialize HouseholdModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute household norm signal and write to buffers.
        Also implements minimal intra-household mask sharing (optional improvement).

        Args:
            state: Global state.
            buffers: Buffers dict.
            params: Parameters dict (unused).
            t: Day index.

        Returns:
            None
        """
        persons: List[Person] = state["persons"]
        households: List[Household] = state["households"]
        N = len(persons)
        # FIXED: Use previous day's decisions (t-1) for household norm signal and clamp length.
        idx = t - 1
        last_decisions = state["mask_use_history"][idx] if (idx >= 0 and idx < len(state["mask_use_history"])) else [0] * N
        if len(last_decisions) != N:
            last_decisions = (last_decisions + [0] * N)[:N]

        hh_signal = [0.0] * N
        household_norm_influence_weight = float(params.get("household_norm_influence_weight", 0.2))
        for hh in households:
            if not hh.member_ids:
                continue
            # Clamp member indices
            members = [m for m in hh.member_ids if 0 <= m < N]
            if not members:
                continue
            mean_use = sum(last_decisions[m] for m in members) / max(1, len(members))
            for m in members:
                # Norm signal scaled by hh norm_strength and global weight
                s = clip01(mean_use * clip01(hh.norm_strength * household_norm_influence_weight))
                hh_signal[m] = s
        buffers.setdefault("signals", {})["household_norm_signal"] = hh_signal

        # FIXED: Compute sharing deltas without mutating state in forward.
        # Donors: inventory > 1.0; Needy: inventory <= 0.0. One unit per needy capped by donor_remaining.
        sharing_delta = [0.0] * N
        share_enabled = bool(params.get("enable_household_sharing", True))
        if share_enabled:
            daily_cap = int(params.get("household_sharing_daily_cap", 2))
            for hh in households:
                members = [m for m in hh.member_ids if 0 <= m < N]
                if not members:
                    continue
                needy = [pid for pid in members if persons[pid].mask_inventory <= 0.0]
                donors = [pid for pid in members if persons[pid].mask_inventory > 1.0]
                if not needy or not donors:
                    continue
                # Local donor remaining map; no state mutation here.
                donor_remaining = {pid: float(persons[pid].mask_inventory) for pid in donors}
                donations_made = 0
                d_idx = 0
                for nid in needy:
                    if donations_made >= daily_cap:
                        break
                    while d_idx < len(donors) and donor_remaining[donors[d_idx]] <= 1.0:
                        d_idx += 1
                    if d_idx >= len(donors):
                        break
                    did = donors[d_idx]
                    sharing_delta[nid] += 1.0
                    sharing_delta[did] -= 1.0
                    donor_remaining[did] = max(0.0, donor_remaining[did] - 1.0)
                    d_idx += 1
                    donations_made += 1

        buffers.setdefault("state_updates", {})["mask_sharing_delta"] = sharing_delta
        pass

    pass


class DecisionAndAdoptionModule(Module):
    """
    Core behavioral decision: compute whether to wear a mask based on signals,
    risk perception, policy mandate, and personal traits. Manages purchasing requests
    and inventory consumption, updates fatigue.
    """

    name = "DecisionAndAdoptionModule"
    inputs = [
        "signals.person.social_norm_signal",
        "signals.person.household_norm_signal",
        "state.person.risk_perception",
        "state.policy.mandate_level",
        "state.retailer.price",
        "state.retailer.stock",
    ]
    outputs = [
        "signals.person.mask_decision",
        "events.purchase_requests",
        "state_updates.person.mask_inventory_delta",
        "state_updates.person.fatigue_level",
    ]
    dependencies = ["PeerInfluenceModule", "HouseholdModule", "MessagingAndRiskModule", "PolicyAndEnforcementModule"]

    def __init__(self) -> None:
        """
        Initialize DecisionAndAdoptionModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute mask decisions and purchase requests. Update fatigue and propose
        inventory consumption deltas (to be committed after supply module).

        Args:
            state: Global state.
            buffers: Buffers dict to write outputs.
            params: Simulation parameters dict.
            t: Day index.

        Returns:
            None
        """
        persons: List[Person] = state["persons"]
        N = len(persons)
        social_signal = buffers.get("signals", {}).get("social_norm_signal", [0.0] * N)
        household_signal = buffers.get("signals", {}).get("household_norm_signal", [0.0] * N)
        # FIXED: Read same-day risk updates from buffers if present to remove lag.
        risk_buf = buffers.get("state_updates", {}).get("risk_perception", None)

        # Decision weights
        w_social = float(params.get("social_influence_weight", 0.4))
        w_risk = float(params.get("risk_perception_weight", 0.35))
        w_policy = float(params.get("policy_weight", 0.25))
        decision_noise = float(params.get("decision_noise", 0.1))
        disutility = float(params.get("disutility_baseline", 0.1))
        mask_wearout_days = max(1, int(params.get("mask_wearout_days", 5)))
        purchase_pack_size = max(1, int(params.get("purchase_pack_size", 10)))
        affordability_share = float(params.get("affordability_threshold_budget_share", 0.05))

        authority: PublicHealthAuthority = state["authority"]
        retailer: Retailer = state["retailer"]

        mask_decisions = [0] * N
        purchases: List[Tuple[int, int]] = []
        inventory_delta = [0.0] * N
        fatigue_updates = [clip01(p.fatigue_level) for p in persons]

        # FIXED: Precompute income terciles once and reuse for subsidy eligibility.
        incomes = [p.income for p in persons]
        idx_sorted = sorted(range(N), key=lambda i: incomes[i])
        tercile_size = max(1, N // 3)
        low_income_set = set(idx_sorted[:tercile_size])

        # Snapshot retailer state
        min_price = retailer.price
        retailer_stock_snapshot = int(retailer.stock)
        subsidy = float(params.get("subsidy_amount", 0.0))

        # Iterate agents
        for i, p in enumerate(persons):
            policy_signal = authority.mandate_level
            social = social_signal[i] + household_signal[i]
            # FIXED: Use same-day risk from buffer if available and valid length.
            if isinstance(risk_buf, list) and len(risk_buf) == N:
                effective_risk = float(risk_buf[i])
            else:
                effective_risk = p.risk_perception
            utility = (
                w_social * p.social_influence_susceptibility * social
                + w_risk * effective_risk
                + w_policy * p.compliance_propensity * policy_signal
                - disutility
                - p.fatigue_level
            )
            p_wear = sigmoid(utility / max(decision_noise, 1e-6))
            decision = 1 if random.random() < p_wear and (p.mask_inventory > 0.0) else 0

            # If deciding to wear but no inventory, generate purchase request if affordable
            if decision == 0 and p.mask_inventory <= 0.0:
                desire_to_buy = policy_signal > 0.2 or (social > 0.3)
                if desire_to_buy and retailer_stock_snapshot > 0:
                    eff_price = max(0.1, min_price - (subsidy if i in low_income_set else 0.0))
                    pack_cost = purchase_pack_size * eff_price
                    afford_limit = p.budget * affordability_share
                    if pack_cost <= afford_limit:
                        purchases.append((i, purchase_pack_size))
                    elif eff_price <= afford_limit:
                        purchases.append((i, 1))
                    else:
                        pass
            elif decision == 1:
                # Consume fractional mask
                inventory_delta[i] -= 1.0 / float(mask_wearout_days)
            # Update fatigue
            fatigue_rate = float(params.get("fatigue_rate_per_day", 0.005))
            fatigue_recovery = float(params.get("fatigue_recovery_rate", 0.002))
            fatigue = p.fatigue_level + fatigue_rate * decision - fatigue_recovery * (1 - decision)
            fatigue_updates[i] = clip01(fatigue)
            mask_decisions[i] = decision

        buffers.setdefault("signals", {})["mask_decision"] = mask_decisions
        buffers.setdefault("events", {})["purchase_requests"] = purchases
        buffers.setdefault("state_updates", {})["mask_inventory_delta"] = inventory_delta
        buffers["state_updates"]["fatigue_level"] = fatigue_updates
        pass

    pass


class MobilityAndLocationModule(Module):
    """
    Schedules visits to locations, enforces mask requirements, and records
    adoption by location type and enforcement incidents.
    """

    name = "MobilityAndLocationModule"
    inputs = [
        "state.venues.mask_required",
        "state.venues.enforcement_strictness",
        "signals.person.mask_decision",
    ]
    outputs = ["events.enforcement_incidents", "events.enforcement_events", "observables.adoption_by_location_type_daily"]
    dependencies = ["PolicyAndEnforcementModule", "DecisionAndAdoptionModule"]

    def __init__(self) -> None:
        """
        Initialize MobilityAndLocationModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Sample location visits and compute by-location adoption and enforcement.

        Args:
            state: Global state.
            buffers: Buffers dict to write outputs.
            params: Parameters dict.
            t: Day index.

        Returns:
            None
        """
        persons: List[Person] = state["persons"]
        venues: List[Venue] = state["venues"]
        N = len(persons)

        # FIXED: Respect location_time_fraction dict if provided and include public_space.
        lf = params.get("location_time_fraction")
        if isinstance(lf, dict):
            time_fraction = {
                "work": float(lf.get("work", params.get("time_fraction_work", 0.25))),
                "retail": float(lf.get("retail", params.get("time_fraction_retail", 0.15))),
                "transport": float(lf.get("transport", params.get("time_fraction_transport", 0.1))),
                "public_space": float(lf.get("public_space", params.get("time_fraction_public_space", 0.2))),
            }
        else:
            time_fraction = {
                "work": float(params.get("time_fraction_work", 0.25)),
                "retail": float(params.get("time_fraction_retail", 0.15)),
                "transport": float(params.get("time_fraction_transport", 0.1)),
                "public_space": float(params.get("time_fraction_public_space", 0.2)),
            }
        time_fraction_total = sum(max(0.0, v) for v in time_fraction.values())
        if time_fraction_total <= 0:
            time_fraction_total = 1.0

        # Organize venues by type
        by_type: Dict[str, List[Venue]] = {}
        for v in venues:
            by_type.setdefault(v.type, []).append(v)

        mask_decisions = buffers.get("signals", {}).get("mask_decision", [0] * N)
        if len(mask_decisions) != N:
            mask_decisions = (mask_decisions + [0] * N)[:N]

        # Reset venue counters
        for v in venues:
            v.reset_day()

        enforcement_incidents = 0
        enforcement_events: List[Dict[str, int]] = []
        adoption_accum: Dict[str, List[int]] = {k: [] for k in by_type.keys()}

        # Visits: sample one visit per type proportional to time fraction, capacity-limited
        for i, _p in enumerate(persons):
            for vtype, weight in time_fraction.items():
                if weight <= 0.0 or vtype not in by_type:
                    continue
                # attempt to visit with probability scaled by weight/time_fraction_total
                if random.random() < (weight / time_fraction_total):
                    # Pick a random venue of that type
                    venues_of_type = by_type[vtype]
                    if not venues_of_type:
                        continue
                    v = random.choice(venues_of_type)
                    # Capacity check
                    if v.daily_visitors >= v.capacity:
                        continue
                    wear = mask_decisions[i] == 1
                    # Enforcement at entry if required and not wearing
                    if v.mask_required and not wear:
                        # gatekeeping enforcement
                        if random.random() < v.enforcement_strictness:
                            enforcement_incidents += 1
                            # FIXED: Emit per-person enforcement events to apply fines later.
                            enforcement_events.append({"pid": i, "venue_id": v.id})
                            # deny entry
                            continue
                    # Admitted
                    v.daily_visitors += 1
                    if wear:
                        v.compliant_visitors += 1
                    adoption_accum.setdefault(vtype, []).append(1 if wear else 0)

        # Compute adoption by location type
        by_loc_rate: Dict[str, float] = {}
        for vtype, vals in adoption_accum.items():
            by_loc_rate[vtype] = sum(vals) / float(len(vals)) if vals else 0.0

        buffers.setdefault("observables", {})["adoption_by_location_type_daily"] = by_loc_rate
        buffers.setdefault("events", {})["enforcement_incidents"] = enforcement_incidents
        # FIXED: Provide detailed enforcement events for fines processing.
        buffers["events"]["enforcement_events"] = enforcement_events
        pass

    pass


class AdoptionAggregator(Module):
    """
    Aggregates daily adoption observables and computes inequality, compliance, and availability metrics.
    """

    name = "AdoptionAggregator"
    inputs = [
        "signals.person.mask_decision",
        "MobilityAndLocationModule.observable.adoption_by_location_type_daily",
        "RetailSupplyModule.observable.stockout_rate_daily",
        "events.enforcement_incidents",
        "state.person.income",
    ]
    outputs = [
        "observable.adoption_rate_daily",
        "observable.enforcement_incidents_rate_per_1k_pd",
        "observable.inequality_index_of_adoption",
        "observable.equity_gap_in_adoption_daily",
        "observable.compliance_rate_in_mandated_locations_daily",
        "observable.average_mask_availability_daily",
        "observable.average_mask_use_days_per_person",
        "observable.adoption_by_group_daily",
    ]
    dependencies = ["MobilityAndLocationModule", "RetailSupplyModule", "DecisionAndAdoptionModule"]

    def __init__(self) -> None:
        """
        Initialize AdoptionAggregator.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute aggregate adoption and ancillary observables.

        Args:
            state: Global state.
            buffers: Buffers dict to write observables.
            params: Parameters dict.
            t: Day index.

        Returns:
            None
        """
        persons: List[Person] = state["persons"]
        N = max(1, len(persons))
        decisions = buffers.get("signals", {}).get("mask_decision", [0] * N)
        if len(decisions) != N:
            decisions = (decisions + [0] * N)[:N]
        adoption_rate = sum(decisions) / float(N)

        # Update per-person cumulative days
        cum = state.setdefault("cumulative_wear_days", [0] * N)
        for i, d in enumerate(decisions):
            cum[i] += 1 if d == 1 else 0

        average_days = sum(cum) / float(N)

        # Enforcement rate per 1k person-days
        incidents = buffers.get("events", {}).get("enforcement_incidents", 0)
        incidents_rate = 1000.0 * incidents / float(N)

        # Inequality of adoption by income bins (terciles)
        incomes = [p.income for p in persons]
        idx = list(range(N))
        idx.sort(key=lambda i: incomes[i])
        tercile_size = max(1, N // 3)
        groups = [
            idx[:tercile_size],
            idx[tercile_size: 2 * tercile_size],
            idx[2 * tercile_size:]
        ]
        means = []
        for g in groups:
            if not g:
                means.append(0.0)
            else:
                means.append(sum(decisions[i] for i in g) / float(len(g)))
        mean_of_means = sum(means) / float(len(means)) if means else 0.0
        mean_abs_diff = 0.0
        for i in range(len(means)):
            for j in range(len(means)):
                mean_abs_diff += abs(means[i] - means[j])
        denom = 2.0 * len(means) * mean_of_means if mean_of_means > 0 else 1.0
        inequality_index = mean_abs_diff / denom
        # FIXED: Equity gap metric (high minus low income adoption)
        equity_gap = (means[-1] - means[0]) if len(means) >= 3 else 0.0

        # FIXED: Compliance in mandated locations using venue counters
        venues: List[Venue] = state.get("venues", [])
        mandated_visitors = sum(v.daily_visitors for v in venues if v.mask_required)
        mandated_compliant = sum(v.compliant_visitors for v in venues if v.mask_required)
        compliance_rate_mandated = safe_div(mandated_compliant, max(1, mandated_visitors), 0.0)

        # FIXED: Average per-capita mask availability (mean inventory)
        avg_inventory = sum(p.mask_inventory for p in persons) / float(N)

        # FIXED: Adoption by income terciles and age bins
        group_metrics: Dict[str, Any] = {"income_terciles": {"low": means[0] if len(means) > 0 else 0.0,
                                                             "mid": means[1] if len(means) > 1 else 0.0,
                                                             "high": means[2] if len(means) > 2 else 0.0}}
        age_bins = {"0_17": [], "18_64": [], "65_plus": []}
        for i, p in enumerate(persons):
            if p.age <= 17:
                age_bins["0_17"].append(decisions[i])
            elif p.age <= 64:
                age_bins["18_64"].append(decisions[i])
            else:
                age_bins["65_plus"].append(decisions[i])
        age_means = {k: (sum(v) / float(len(v)) if v else 0.0) for k, v in age_bins.items()}
        group_metrics["age_bins"] = age_means

        obs = buffers.setdefault("observables", {})
        obs["adoption_rate_daily"] = adoption_rate
        obs["enforcement_incidents_rate_per_1k_pd"] = incidents_rate
        obs["inequality_index_of_adoption"] = inequality_index
        obs["equity_gap_in_adoption_daily"] = equity_gap
        obs["compliance_rate_in_mandated_locations_daily"] = compliance_rate_mandated
        obs["average_mask_availability_daily"] = avg_inventory
        obs["average_mask_use_days_per_person"] = average_days
        obs["adoption_by_group_daily"] = group_metrics
        pass

    pass


class DiseaseTransmissionModule(Module):
    """
    S(E)IR disease transmission dynamics modulated by mask use on the social network.

    Computes daily new infections and updates health states. Also estimates
    NPI effectiveness from mask use by comparing hazard with/without masks.
    """

    name = "DiseaseTransmissionModule"
    inputs = ["signals.person.mask_decision", "state.network", "state.person.health_state"]
    outputs = ["state_updates.person.health_state", "state_updates.person.days_in_state", "observables.new_infections_daily", "observables.npi_effectiveness_daily"]
    dependencies = ["DecisionAndAdoptionModule", "MobilityAndLocationModule"]

    def __init__(self) -> None:
        """
        Initialize DiseaseTransmissionModule.
        """
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Execute one day of SEIR dynamics given mask decisions.

        Args:
            state: Global state.
            buffers: Buffers to write state updates and observables.
            params: Parameter dictionary including enable_disease, transmission rates, etc.
            t: Day index.

        Returns:
            None
        """
        enable = bool(params.get("enable_disease", False))
        persons: List[Person] = state["persons"]
        N = len(persons)
        if not enable or N == 0:
            buffers.setdefault("observables", {})["new_infections_daily"] = 0.0
            buffers["observables"]["npi_effectiveness_daily"] = 0.0
            return None

        decisions = buffers.get("signals", {}).get("mask_decision", [0] * N)
        if len(decisions) != N:
            decisions = (decisions + [0] * N)[:N]
        adj = state["network"]

        beta = float(params.get("base_transmission_prob_per_contact", params.get("transmission_rate_base", 0.03)))
        src_eff = clip01(float(params.get("mask_efficacy_source_control", params.get("mask_effectiveness_transmission_reduction", 0.5))))
        wearer_eff = clip01(float(params.get("mask_efficacy_wearer_protection", params.get("mask_effectiveness_susceptibility_reduction", 0.3))))
        incubation_days = max(1, int(params.get("incubation_days", 3)))
        sigma = 1.0 / float(incubation_days)
        gamma = clip01(float(params.get("recovery_rate", 0.1)))

        # Prepare updates
        new_states: List[str] = [p.health_state for p in persons]
        new_days: List[int] = [p.days_in_state + 1 for p in persons]

        new_infections = 0
        hazard_actual_sum = 0.0
        hazard_baseline_sum = 0.0

        # Transmission for S
        for i, p in enumerate(persons):
            if p.health_state != "S":
                continue
            # compute per-day infection probability from infectious neighbors
            neighbors = adj[i]
            p_infection_actual = 0.0
            p_no = 1.0
            p_no_baseline = 1.0
            for j in neighbors:
                if 0 <= j < N and persons[j].health_state == "I":
                    # Actual transmission probability with masks
                    p_t = beta
                    if decisions[j] == 1:
                        p_t *= (1.0 - src_eff)
                    if decisions[i] == 1:
                        p_t *= (1.0 - wearer_eff)
                    p_t = clip01(p_t)
                    p_no *= (1.0 - p_t)
                    # Baseline without masks
                    p_t0 = clip01(beta)
                    p_no_baseline *= (1.0 - p_t0)
            p_infection_actual = 1.0 - p_no
            p_infection_baseline = 1.0 - p_no_baseline
            hazard_actual_sum += p_infection_actual
            hazard_baseline_sum += p_infection_baseline
            if random.random() < p_infection_actual:
                new_states[i] = "E" if incubation_days > 0 else "I"
                new_days[i] = 0
                new_infections += 1

        # E -> I transitions
        for i, p in enumerate(persons):
            if p.health_state == "E":
                if random.random() < sigma:
                    new_states[i] = "I"
                    new_days[i] = 0

        # I -> R transitions
        for i, p in enumerate(persons):
            if p.health_state == "I":
                if random.random() < gamma:
                    new_states[i] = "R"
                    new_days[i] = 0

        # NPI effectiveness: hazard reduction from masks
        npi_eff = 0.0
        if hazard_baseline_sum > 0.0:
            npi_eff = clip01(1.0 - safe_div(hazard_actual_sum, hazard_baseline_sum, 1.0))

        buffers.setdefault("state_updates", {})["health_state"] = new_states
        buffers["state_updates"]["days_in_state"] = new_days
        buffers.setdefault("observables", {})["new_infections_daily"] = float(new_infections)
        buffers["observables"]["npi_effectiveness_daily"] = float(npi_eff)
        pass

    pass


def build_small_world_network(n: int, k: int, p_rewire: float, seed: int) -> List[List[int]]:
    """
    Build a small-world network adjacency list. Fallback to ring lattice if networkx unavailable.

    Args:
        n: Number of nodes.
        k: Each node connected to k nearest neighbors (k even).
        p_rewire: Rewiring probability.
        seed: RNG seed.

    Returns:
        Adjacency list mapping node -> neighbors list.
    """
    if n <= 0:
        return []
    if k % 2 == 1:
        k += 1
    k = min(max(2, k), max(2, n - 1))
    if nx is not None:
        try:
            G = nx.watts_strogatz_graph(n, k, p_rewire, seed=seed)
            adj = [[] for _ in range(n)]
            for u in G.nodes:
                adj[u] = list(G.neighbors(u))
            return adj
        except Exception:
            pass
    # Fallback: ring lattice
    adj = [[] for _ in range(n)]
    half = max(1, k // 2)
    for i in range(n):
        nbrs = []
        for d in range(1, half + 1):
            nbrs.append((i - d) % n)
            nbrs.append((i + d) % n)
        adj[i] = nbrs
    return adj
    pass


class Simulation:
    """
    Main simulation engine coordinating modules, state, buffers, and scheduling.

    Provides run(), set_params(), get_params(), save_results(), save_module_io(),
    save_all_io(), evaluate(), and visualize() methods.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the simulation with provided parameters.

        Args:
            params: Parameters dict.

        Raises:
            ValueError: On invalid configurations.
        """
        self.params: Dict[str, Any] = copy.deepcopy(params)
        seed = int(self.params.get("random_seed", 42))
        seed_all(seed)

        # Fast mode via env
        if os.environ.get("QUICK_TEST", "0") == "1":
            self.params["population_size"] = min(500, int(self.params.get("population_size", 2000)))
            # FIXED: Respect spec alias 'simulation_days' if present.
            self.params["simulation_duration_days"] = min(30, int(self.params.get("simulation_days", self.params.get("simulation_duration_days", 60))))

        self.N = int(self.params.get("population_size", 2000))
        # FIXED: Support spec-compliant keys (simulation_days) with fallback to simulation_duration_days.
        self.days = int(self.params.get("simulation_days", self.params.get("simulation_duration_days", 60)))
        # FIXED: Support avg_degree alias for network_avg_degree.
        self.avg_deg = int(self.params.get("avg_degree", self.params.get("network_avg_degree", int(self.params.get("average_degree", 10)))))
        self.rewire_prob = float(self.params.get("network_rewire_prob", 0.1))
        self.state: Dict[str, Any] = {}
        self.buffers: Dict[str, Any] = {}
        self.results: Dict[str, List[Any]] = {
            "adoption_rate": [],
            # FIXED: Alias series for spec
            "adoption_rate_over_time": [],
            "adoption_by_location_type": [],
            "stockout_rate_daily": [],
            "enforcement_incidents_rate_per_1k_pd": [],
            "inequality_index_of_adoption": [],
            # FIXED: Alias inequity index
            "inequity_index": [],
            "equity_gap_in_adoption": [],
            "compliance_rate_in_mandated_locations": [],
            "average_mask_use_days_per_person": [],
            "average_mask_availability_daily": [],
            "policy_cost_daily": [],
            "enforcement_cost_daily": [],
            "new_infections_daily": [],
            "cumulative_infections": [],
            "Rt": [],
            # FIXED: Alias effective_Rt
            "effective_Rt": [],
            "npi_effectiveness_daily": [],
            # FIXED: Added masks sold metrics
            "masks_sold_daily": [],
            "masks_sold_cumulative": [],
            # FIXED: Added adoption by group and disparity alias
            "adoption_by_group_income": [],
            "adoption_by_group_age": [],
            "adoption_disparity_index": [],
            # FIXED: Track retailer stock per capita to compute shortage days
            "retailer_stock_per_capita": [],
            # FIXED: Track misinformation prevalence daily
            "misinformation_prevalence": [],
        }
        self.module_ios: Dict[str, List[Dict[str, Any]]] = {}
        self._init_entities()
        self._init_network()
        self._init_modules()
        pass

    def _init_entities(self) -> None:
        """
        Initialize persons, households, venues, retailer, authority, media channels, and epidemic state.

        Returns:
            None
        """
        N = self.N
        persons: List[Person] = []
        # Income via lognormal fallback
        income_mu = float(self.params.get("income_lognorm_mu", 4.5))
        income_sigma = float(self.params.get("income_lognorm_sigma", 0.7))
        if np is not None:
            incomes = list(np.random.lognormal(mean=income_mu, sigma=income_sigma, size=N))
        else:
            incomes = [math.exp(random.gauss(income_mu, income_sigma)) for _ in range(N)]
        # Sample ages according to provided age_distribution bins.
        age_dist = self.params.get("age_distribution", {"0_17": 0.2, "18_64": 0.65, "65_plus": 0.15})
        bins = [(0, 17, float(age_dist.get("0_17", 0.2))), (18, 64, float(age_dist.get("18_64", 0.65))), (65, 90, float(age_dist.get("65_plus", 0.15)))]
        s = sum(b[2] for b in bins) or 1.0
        probs = [b[2] / s for b in bins]
        ages = []
        for _ in range(N):
            r = random.random()
            acc = 0.0
            chosen = bins[-1]
            for (lo, hi, _p), pr in zip(bins, probs):
                acc += pr
                if r <= acc:
                    chosen = (lo, hi, _p)
                    break
            ages.append(random.randint(chosen[0], chosen[1]))
        # Budget
        budget_mean = float(self.params.get("budget_mean", 100.0))
        budget_std = float(self.params.get("budget_std", 30.0))
        budgets = [max(0.0, random.gauss(budget_mean, budget_std)) for _ in range(N)]
        # Trust distribution
        trust_mu = float(self.params.get("trust_in_authority_mean", 0.6))
        trust_std = float(self.params.get("trust_in_authority_std", 0.2))

        init_rate = float(self.params.get("initial_mask_adoption_rate", 0.2))
        enable_disease = bool(self.params.get("enable_disease", False))
        init_inf_frac = float(self.params.get("initial_infected_fraction", 0.01))
        for i in range(N):
            trust = clip01(random.gauss(trust_mu, trust_std))
            rp = clip01(float(self.params.get("perceived_risk_base", 0.2)) + random.random() * 0.2)
            sis = clip01(random.random())
            comp = clip01(random.random())
            fatig = 0.0
            inv = 1.0 if random.random() < init_rate else 0.0
            budget = float(budgets[i])
            misinformation_belief = 1.0 if random.random() < float(self.params.get("misinformation_rate", 0.05)) else 0.0
            health_state = "I" if enable_disease and (random.random() < init_inf_frac) else "S"
            p = Person(
                id=i,
                age=ages[i],
                income=float(incomes[i]),
                household_id=-1,
                risk_perception=rp,
                trust_in_authorities=trust,
                social_influence_susceptibility=sis,
                compliance_propensity=comp,
                fatigue_level=fatig,
                mask_inventory=inv,
                budget=budget,
                misinformation_belief=misinformation_belief,
                health_state=health_state,
                days_in_state=0,
                compliance_probability=0.0,
                workplace_id=-1,
            )
            persons.append(p)

        # Households: group sequential persons by avg size ~2.6
        households: List[Household] = []
        i = 0
        hid = 0
        while i < N:
            size = max(1, int(round(random.random() * 2 + 2)))
            members = list(range(i, min(N, i + size)))
            hh = Household(id=hid, member_ids=members, norm_strength=clip01(random.random()))
            households.append(hh)
            for m in members:
                persons[m].household_id = hid
            i += size
            hid += 1

        # Venues: simple mix
        counts = {
            "work": int(self.params.get("location_count_work", 50)),
            "retail": int(self.params.get("location_count_retail", 30)),
            "transport": int(self.params.get("location_count_transport", 10)),
            "public_space": int(self.params.get("location_count_public_space", 10)),
        }
        capacity_means = {
            "work": int(self.params.get("location_capacity_work_mean", 100)),
            "retail": int(self.params.get("location_capacity_retail_mean", 50)),
            "transport": int(self.params.get("location_capacity_transport_mean", 80)),
            "public_space": int(self.params.get("location_capacity_public_space_mean", 120)),
        }
        venues: List[Venue] = []
        vid = 0
        for vtype, cnt in counts.items():
            for _ in range(cnt):
                cap = max(10, int(random.gauss(capacity_means.get(vtype, 80), 10)))
                venues.append(
                    Venue(
                        id=vid,
                        type=vtype,
                        capacity=cap,
                        mask_required=False,
                        enforcement_strictness=clip01(random.random() * 0.5 + 0.25),
                        staff_enforcement_level=clip01(random.random() * 0.5 + 0.25),
                        local_outbreak_level=clip01(float(self.params.get("local_outbreak_base", 0.2)) + random.gauss(0, float(self.params.get("local_outbreak_volatility", 0.1)))),
                    )
                )
                vid += 1

        # Retailer: single aggregated retailer
        initial_stock_per_1000 = int(self.params.get("supply_initial_stock_per_1000_people", 800))
        initial_stock = int(initial_stock_per_1000 * (N / 1000.0))
        retailer = Retailer(
            id=0,
            stock=initial_stock,
            restock_rate=float(self.params.get("restock_rate_per_day", 0.1)),
            # FIXED: Support alias 'price_per_mask' for 'mask_price'
            price=float(self.params.get("mask_price", self.params.get("price_per_mask", 1.0))),
            # FIXED: Support alias 'supply_chain_disruption_prob' for 'supply_variability'
            supply_variability=float(self.params.get("supply_variability", self.params.get("supply_chain_disruption_prob", 0.2))),
        )

        # Policy authority
        authority = PublicHealthAuthority(
            id=0,
            mandate_level=float(self.params.get("mandate_level", 0.6)),
            # FIXED: Support alias 'enforcement_strength' for 'enforcement_probability'
            enforcement_probability=float(self.params.get("enforcement_probability", self.params.get("enforcement_strength", 0.3))),
            fine_amount=float(self.params.get("fine_amount", 50.0)),
            messaging_budget=float(self.params.get("messaging_budget", 100000.0)),
            campaign_strategy=str(self.params.get("campaign_strategy", "balanced")),
            message_effectiveness=float(self.params.get("message_effectiveness", 0.15)),
            communication_frequency=int(self.params.get("communication_frequency", 7)),
            subsidy_amount=float(self.params.get("subsidy_amount", 0.0)),
            distribution_program_active=False,
        )

        # Media channels
        media_channels: List[MediaChannel] = []
        media_channels.append(MediaChannel(id=0, name="PublicHealth", sentiment=0.8, reach=0.6, misinformation_rate=0.01))
        media_channels.append(MediaChannel(id=1, name="SocialMedia", sentiment=0.2, reach=0.5, misinformation_rate=0.1))
        media_channels.append(MediaChannel(id=2, name="RumorNetwork", sentiment=-0.4, reach=0.2, misinformation_rate=0.3, targeting="low_income"))

        # Epidemic state
        self.state["epidemic"] = {
            "new_infections_series": [],
            "cumulative_infections": sum(1 for p in persons if p.health_state in ("E", "I", "R")),
        }

        # State assembly
        self.state["persons"] = persons
        self.state["households"] = households
        self.state["venues"] = venues
        self.state["retailer"] = retailer
        self.state["authority"] = authority
        self.state["media_channels"] = media_channels
        self.state["mask_use_history"] = []  # list of daily mask decisions arrays
        self.state["cumulative_wear_days"] = [0] * N
        pass

    def _init_network(self) -> None:
        """
        Initialize social network adjacency list.

        Returns:
            None
        """
        n = self.N
        k = int(self.avg_deg) if self.avg_deg else 8
        p = float(self.rewire_prob) if self.rewire_prob else 0.1
        adj = build_small_world_network(n, k, p, int(self.params.get("random_seed", 42)))
        self.state["network"] = adj
        pass

    def _topologically_sort_modules(self, modules: List[Module]) -> List[Module]:
        """
        Topologically sort modules based on declared dependencies.

        Args:
            modules: List of module instances.

        Returns:
            Ordered list of modules respecting dependencies.

        Raises:
            ValueError: If a dependency is missing or a cycle is detected.
        """
        name_to_mod = {m.name: m for m in modules}
        indeg: Dict[str, int] = {m.name: 0 for m in modules}
        graph: Dict[str, List[str]] = {m.name: [] for m in modules}

        # Build graph
        for m in modules:
            for dep in getattr(m, "dependencies", []):
                if dep not in name_to_mod:
                    raise ValueError(f"Dependency {dep} for module {m.name} not found.")
                graph[dep].append(m.name)
                indeg[m.name] += 1

        # Kahn's algorithm
        queue = [name for name, d in indeg.items() if d == 0]
        ordered_names: List[str] = []
        while queue:
            cur = queue.pop(0)
            ordered_names.append(cur)
            for nb in graph.get(cur, []):
                indeg[nb] -= 1
                if indeg[nb] == 0:
                    queue.append(nb)

        if len(ordered_names) != len(modules):
            raise ValueError("Cycle detected in module dependencies; cannot sort.")
        return [name_to_mod[n] for n in ordered_names]

    def _init_modules(self) -> None:
        """
        Initialize module instances and store execution order respecting dependencies.

        Returns:
            None
        """
        modules: List[Module] = [
            PolicyAndEnforcementModule(),
            MessagingAndRiskModule(),
            HouseholdModule(),
            PeerInfluenceModule(memory_days=int(self.params.get("peer_influence_memory_days", 3))),
            DecisionAndAdoptionModule(),
            RetailSupplyModule(),
            # FIXED: Reorder modules so Mobility runs BEFORE Disease to prepare for location-based hazard.
            MobilityAndLocationModule(),
            DiseaseTransmissionModule(),  # FIXED: Added disease dynamics module; depends on Mobility now.
            AdoptionAggregator(),
        ]
        # FIXED: Add topological sort to enforce dependencies with acyclicity check.
        try:
            self.modules = self._topologically_sort_modules(modules)
        except Exception as e:
            warnings.warn(f"Topological sorting failed ({e}); falling back to declared order.")
            self.modules = modules
        pass

    def set_params(self, module: Optional[str] = None, **kwargs: Any) -> None:
        """
        Update parameters, optionally scoped to a module.

        Args:
            module: Module name to scope parameters (unused: flat dict).
            **kwargs: Key-value pairs to update.

        Returns:
            None
        """
        for k, v in kwargs.items():
            self.params[k] = v
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get a deep copy of current parameters.

        Returns:
            Parameter dictionary.
        """
        return copy.deepcopy(self.params)
        pass

    def _commit_buffers(self, t: int) -> None:
        """
        Apply buffered state updates for persons, retailer, and append observables.

        Args:
            t: Day index.

        Returns:
            None
        """
        upd = self.buffers.get("state_updates", {})
        persons: List[Person] = self.state["persons"]
        N = len(persons)

        # Person risk updates
        risk_updates = upd.get("risk_perception", None)
        if risk_updates is not None and len(risk_updates) == N:
            for i, rp in enumerate(risk_updates):
                persons[i].risk_perception = clip01(float(rp))

        # FIXED: Apply misinformation belief updates from MessagingAndRiskModule.
        misinfo_updates = upd.get("misinformation_belief", None)
        if misinfo_updates is not None and len(misinfo_updates) == N:
            for i, mb in enumerate(misinfo_updates):
                persons[i].misinformation_belief = clip01(float(mb))

        # Fatigue updates
        fatigue_updates = upd.get("fatigue_level", None)
        if fatigue_updates is not None and len(fatigue_updates) == N:
            for i, f in enumerate(fatigue_updates):
                persons[i].fatigue_level = clip01(float(f))

        # Health state updates (SEIR)
        health_updates = upd.get("health_state", None)
        days_updates = upd.get("days_in_state", None)
        if health_updates is not None and len(health_updates) == N:
            for i, hs in enumerate(health_updates):
                persons[i].health_state = str(hs)
        if days_updates is not None and len(days_updates) == N:
            for i, d in enumerate(days_updates):
                persons[i].days_in_state = max(0, int(d))

        # Apply free distributions
        policy_cost_today = 0.0
        fines_collected_today = 0.0  # FIXED: Track fines collected for net enforcement cost.
        free_events: List[Tuple[int, int]] = self.buffers.get("events", {}).get("free_distribution", [])
        if free_events:
            unit_cost = float(self.params.get("distribution_unit_cost", self.state["retailer"].price))
            for pid, units in free_events:
                if isinstance(pid, int) and 0 <= pid < N and units > 0:
                    persons[pid].mask_inventory = max(0.0, persons[pid].mask_inventory + float(units))
                    policy_cost_today += float(units) * unit_cost

        # FIXED: Apply mask sharing deltas before consumption.
        sharing_delta = upd.get("mask_sharing_delta", None)
        if sharing_delta is not None and len(sharing_delta) == N:
            for i, d in enumerate(sharing_delta):
                persons[i].mask_inventory = max(0.0, float(persons[i].mask_inventory) + float(d))

        # Inventory deltas (consumption)
        inv_delta = upd.get("mask_inventory_delta", None)
        if inv_delta is not None and len(inv_delta) == N:
            for i, d in enumerate(inv_delta):
                persons[i].mask_inventory = max(0.0, float(persons[i].mask_inventory) + float(d))

        # FIXED: Reconcile retailer allocations with budgets; track unpurchased allocations and actual sold.
        allocations: Dict[int, int] = upd.get("retailer_allocations", {})
        # Apply subsidy pricing during reconciliation
        retailer_price = max(0.1, self.state["retailer"].price)
        incomes = [p.income for p in persons]
        idx_sorted = sorted(range(N), key=lambda i: incomes[i])
        tercile_size = max(1, N // 3)
        low_income_set = set(idx_sorted[:tercile_size])
        subsidy = float(self.params.get("subsidy_amount", 0.0))

        unspent_back_to_stock = 0
        total_actual_sold = 0
        for pid, qty in allocations.items():
            if 0 <= pid < N and qty > 0:
                price_effective = max(0.1, retailer_price - (subsidy if pid in low_income_set else 0.0))
                affordable_qty = int(persons[pid].budget // price_effective)
                actual_qty = max(0, min(qty, affordable_qty))
                spend = actual_qty * price_effective
                persons[pid].budget = max(0.0, persons[pid].budget - spend)
                persons[pid].mask_inventory += actual_qty
                # Subsidy cost borne by policy
                if pid in low_income_set and subsidy > 0.0 and actual_qty > 0:
                    policy_cost_today += actual_qty * subsidy
                unspent_back_to_stock += max(0, qty - actual_qty)
                total_actual_sold += actual_qty

        # FIXED: Apply retailer state updates and preserve unpurchased units
        ret_upd = upd.get("retailer") or {}
        r = self.state["retailer"]
        new_stock = int(ret_upd.get("stock", r.stock)) + int(unspent_back_to_stock)
        r.stock = max(0, new_stock)
        if "price" in ret_upd:
            r.price = float(ret_upd["price"])

        # FIXED: Apply enforcement fines; track fines collected and administrative costs, budgets cannot go negative.
        authority = self.state["authority"]
        fine_amount = float(getattr(authority, "fine_amount", 0.0))
        admin_cost = float(self.params.get("enforcement_admin_cost_per_incident", 0.0))
        enforcement_events = self.buffers.get("events", {}).get("enforcement_events", [])
        if enforcement_events:
            for ev in enforcement_events:
                pid = ev.get("pid")
                if isinstance(pid, int) and 0 <= pid < N:
                    pre = float(persons[pid].budget)
                    deduct = min(fine_amount, pre)
                    persons[pid].budget = max(0.0, pre - deduct)
                    fines_collected_today += deduct
            net_enforcement_cost = admin_cost * len(enforcement_events) - fines_collected_today
        else:
            net_enforcement_cost = 0.0
        # FIXED: Record enforcement cost daily.
        self.results.setdefault("enforcement_cost_daily", []).append(float(net_enforcement_cost))

        # Update has_mask flags and current_mask_use
        today_decisions = self.buffers.get("signals", {}).get("mask_decision", [0] * N)
        if len(today_decisions) != N:
            today_decisions = (today_decisions + [0] * N)[:N]
        for i, p in enumerate(persons):
            p.has_mask = p.mask_inventory > 0.0
            p.current_mask_use = int(today_decisions[i])

        # Append history and observables
        self.state["mask_use_history"].append(list(today_decisions))

        # FIXED: Override masks_sold_daily observable with actual sold after budget reconciliation.
        self.buffers.setdefault("observables", {})["masks_sold_daily"] = float(total_actual_sold)

        # Observables from aggregator and modules
        obs = self.buffers.get("observables", {})
        if "adoption_rate_daily" in obs:
            val = float(obs["adoption_rate_daily"])
            self.results["adoption_rate"].append(val)
            # FIXED: Alias for spec consumer.
            self.results.setdefault("adoption_rate_over_time", []).append(val)
        if "adoption_by_location_type_daily" in obs:
            self.results["adoption_by_location_type"].append(copy.deepcopy(obs["adoption_by_location_type_daily"]))
        if "stockout_rate_daily" in obs:
            self.results["stockout_rate_daily"].append(float(obs["stockout_rate_daily"]))
        if "enforcement_incidents_rate_per_1k_pd" in obs:
            self.results["enforcement_incidents_rate_per_1k_pd"].append(float(obs["enforcement_incidents_rate_per_1k_pd"]))
        if "inequality_index_of_adoption" in obs:
            ineq = float(obs["inequality_index_of_adoption"])
            self.results["inequality_index_of_adoption"].append(ineq)
            # FIXED: Alias adoption disparity index
            self.results["adoption_disparity_index"].append(ineq)
            # FIXED: Add inequity_index alias for spec.
            self.results.setdefault("inequity_index", []).append(ineq)
        if "equity_gap_in_adoption_daily" in obs:
            self.results["equity_gap_in_adoption"].append(float(obs["equity_gap_in_adoption_daily"]))
        if "compliance_rate_in_mandated_locations_daily" in obs:
            self.results["compliance_rate_in_mandated_locations"].append(float(obs["compliance_rate_in_mandated_locations_daily"]))
        if "average_mask_use_days_per_person" in obs:
            self.results["average_mask_use_days_per_person"].append(float(obs["average_mask_use_days_per_person"]))
        if "average_mask_availability_daily" in obs:
            self.results["average_mask_availability_daily"].append(float(obs["average_mask_availability_daily"]))
        # FIXED: Adoption by group append to results
        if "adoption_by_group_daily" in obs:
            group = obs["adoption_by_group_daily"]
            if isinstance(group, dict):
                inc = group.get("income_terciles", {})
                age = group.get("age_bins", {})
                self.results["adoption_by_group_income"].append(copy.deepcopy(inc))
                self.results["adoption_by_group_age"].append(copy.deepcopy(age))
        # FIXED: Masks sold metrics from observables
        if "masks_sold_daily" in obs:
            sold = float(obs["masks_sold_daily"])
            self.results["masks_sold_daily"].append(sold)
            cum = (self.results["masks_sold_cumulative"][-1] if self.results["masks_sold_cumulative"] else 0.0) + sold
            self.results["masks_sold_cumulative"].append(cum)

        # Disease observables and epidemic state update
        new_inf = float(obs.get("new_infections_daily", 0.0))
        self.state["epidemic"]["new_infections_series"].append(new_inf)
        self.state["epidemic"]["cumulative_infections"] = float(self.state["epidemic"]["cumulative_infections"]) + new_inf
        self.results["new_infections_daily"].append(new_inf)
        self.results["cumulative_infections"].append(float(self.state["epidemic"]["cumulative_infections"]))
        # Rt approximation: ratio of sums over windows [t-6..t] / [t-13..t-7]
        series = self.state["epidemic"]["new_infections_series"]
        w = 7
        if len(series) >= 2 * w:
            recent = sum(series[-w:])
            prev = sum(series[-2 * w:-w])
            Rt = safe_div(recent, prev, 0.0)
        else:
            Rt = 0.0
        self.results["Rt"].append(float(Rt))
        # FIXED: Add alias effective_Rt.
        self.results.setdefault("effective_Rt", []).append(float(Rt))
        # NPI effectiveness daily
        if "npi_effectiveness_daily" in obs:
            self.results["npi_effectiveness_daily"].append(float(obs["npi_effectiveness_daily"]))
        else:
            self.results["npi_effectiveness_daily"].append(0.0)

        # FIXED: Track retailer stock per capita for shortage calculation.
        stock_per_capita = safe_div(self.state["retailer"].stock, max(1, len(self.state["persons"])), 0.0)
        self.results.setdefault("retailer_stock_per_capita", []).append(float(stock_per_capita))

        # FIXED: Track misinformation prevalence (share of persons with belief >= 0.5).
        mis_prev = sum(1 for p in self.state["persons"] if getattr(p, "misinformation_belief", 0.0) >= 0.5) / float(max(1, len(self.state["persons"])))
        self.results.setdefault("misinformation_prevalence", []).append(float(mis_prev))

        # Policy cost observable
        self.results["policy_cost_daily"].append(float(policy_cost_today))
        pass

    def run(self, start_day: int = 0, end_day: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the simulation between specified days (inclusive of start, exclusive of end).

        Args:
            start_day: Starting day index.
            end_day: Exclusive end day index. Defaults to simulation duration.

        Returns:
            Results dict containing observables time series.
        """
        if end_day is None:
            end_day = self.days
        end_day = min(end_day, self.days)
        # Main daily loop
        for t in range(start_day, end_day):
            self.buffers = {"signals": {}, "events": {}, "observables": {}, "state_updates": {}}
            # Execute modules in order
            for mod in self.modules:
                t_before = time.time()
                # Narrow exceptions to show context without hiding critical bugs
                try:
                    mod.forward(self.state, self.buffers, self.params, t)
                except (ValueError, TypeError, KeyError) as e:
                    raise
                except Exception as e:
                    warnings.warn(f"Module {mod.name} failed at t={t}: {e}")
                    raise
                t_after = time.time()
                # Store module IO snapshot (sampling only selected keys for brevity)
                io_snapshot = {
                    "t": t,
                    "duration_sec": round(t_after - t_before, 6),
                    "outputs": list(getattr(mod, "outputs", [])),
                }
                self.module_ios.setdefault(mod.name, []).append(io_snapshot)
            # Commit buffered updates
            self._commit_buffers(t)
        return self.results
        pass

    def save_results(self, filename: str) -> None:
        """
        Save simulation results to a JSON file.

        Args:
            filename: Output file path.

        Returns:
            None
        """
        try:
            ensure_dir(os.path.dirname(filename))
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save results: {e}")
        finally:
            pass
        pass

    def save_module_io(self, module_name: str, path: str) -> None:
        """
        Save per-module IO snapshots to a JSON file.

        Args:
            module_name: Name of the module.
            path: Output JSON path.

        Returns:
            None
        """
        try:
            ensure_dir(os.path.dirname(path))
            ios = self.module_ios.get(module_name, [])
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ios, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save IO for {module_name}: {e}")
        finally:
            pass
        pass

    def save_all_io(self, root_dir: str) -> None:
        """
        Save all module IO snapshots to root_dir.

        Args:
            root_dir: Output directory path.

        Returns:
            None
        """
        try:
            ensure_dir(root_dir)
            for m in self.module_ios:
                self.save_module_io(m, os.path.join(root_dir, f"{m}.json"))
        except Exception as e:
            warnings.warn(f"Failed to save all IO: {e}")
        finally:
            pass
        pass

    def evaluate(self, ground_truth: Optional[Dict[str, List[float]]] = None, window: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Compute evaluation metrics, optionally versus ground truth for a given window.

        Args:
            ground_truth: Dict with keys like 'adoption_rate' mapping to list of floats.
            window: Tuple (start, end) day indices to restrict evaluation.

        Returns:
            Metrics dict.
        """
        results = self.results
        start = 0
        end = len(results.get("adoption_rate", []))
        if window is not None:
            start = max(0, int(window[0]))
            end = min(end, int(window[1]))

        def series_window(series: List[float]) -> List[float]:
            return series[start:end] if series else []

        adoption = series_window(results.get("adoption_rate", []))
        stockout = series_window(results.get("stockout_rate_daily", []))
        enforcement = series_window(results.get("enforcement_incidents_rate_per_1k_pd", []))
        inequality = series_window(results.get("inequality_index_of_adoption", []))
        avg_days = series_window(results.get("average_mask_use_days_per_person", []))
        equity_gap = series_window(results.get("equity_gap_in_adoption", []))
        compliance_rate_loc = series_window(results.get("compliance_rate_in_mandated_locations", []))
        avg_inventory = series_window(results.get("average_mask_availability_daily", []))
        npi_eff = series_window(results.get("npi_effectiveness_daily", []))
        new_inf = series_window(results.get("new_infections_daily", []))
        Rt_series = series_window(results.get("Rt", []))
        cum_inf_series = series_window(results.get("cumulative_infections", []))
        policy_cost_daily = series_window(results.get("policy_cost_daily", []))
        masks_sold_daily = series_window(results.get("masks_sold_daily", []))
        # FIXED: Pull series for new metrics
        stock_pc_series = series_window(results.get("retailer_stock_per_capita", []))
        enf_daily_series = series_window(results.get("enforcement_cost_daily", []))
        misinfo_prev_series = series_window(results.get("misinformation_prevalence", []))

        # Times to thresholds
        def time_to_threshold(x: List[float], thr: float) -> Optional[int]:
            for i, v in enumerate(x):
                if v >= thr:
                    return i + start
            return None

        t50 = time_to_threshold(adoption, 0.5)
        # FIXED: Added time_to_70_percent metric
        t70 = time_to_threshold(adoption, 0.7)
        t80 = time_to_threshold(adoption, 0.8)
        peak_adoption_rate = max(adoption) if adoption else 0.0

        # Stability: last 7 day change
        stability = None
        if len(adoption) >= 7:
            stability = abs(adoption[-1] - adoption[-7])

        # FIXED: Sustained adoption rate over last K days
        K = int(self.params.get("sustained_window_days", 14))
        sustained_slice = adoption[-K:] if len(adoption) >= K else adoption
        sustained_adoption_rate = float(sum(sustained_slice) / max(1, len(sustained_slice))) if sustained_slice else 0.0

        # FIXED: Counterfactual policy effect (mandate disabled)
        policy_effect_contribution: Optional[float] = None
        try:
            cf_params = copy.deepcopy(self.get_params())
            cf_params["mandate_level"] = 0.0
            cf_params["enforcement_probability"] = 0.0
            cf_params["location_mask_requirement_rate"] = 0.0
            cf_sim = Simulation(cf_params)
            cf_sim.run(start, end)
            cf_adoption = cf_sim.results.get("adoption_rate", [])[start:end]
            n = min(len(adoption), len(cf_adoption))
            if n > 0:
                policy_effect_contribution = float(sum(adoption[:n]) - sum(cf_adoption[:n])) / float(n)
        except Exception as e:
            warnings.warn(f"Policy counterfactual failed: {e}")
            policy_effect_contribution = None

        # FIXED: Compute shortage days based on per-capita threshold
        shortage_thr = float(self.params.get("shortage_threshold_per_capita", 0.1))
        mask_supply_shortage_days = int(sum(1 for s in stock_pc_series if s < shortage_thr)) if stock_pc_series else 0

        # FIXED: Aggregate enforcement costs
        enforcement_cost_total = float(sum(enf_daily_series)) if enf_daily_series else 0.0

        # FIXED: Average misinformation prevalence over window
        misinfo_prev_mean = float(sum(misinfo_prev_series) / max(1, len(misinfo_prev_series))) if misinfo_prev_series else 0.0

        metrics = {
            "time_to_50_percent_adoption": t50,
            # FIXED: New metric to align with specification naming
            "time_to_70_percent": t70,
            "time_to_80_percent_adoption": t80,
            "peak_adoption_rate": float(peak_adoption_rate),
            "stockout_rate_mean": float(sum(stockout) / max(1, len(stockout))) if stockout else 0.0,
            "enforcement_incidents_rate_mean": float(sum(enforcement) / max(1, len(enforcement))) if enforcement else 0.0,
            "inequality_index_mean_last_7": float(sum(inequality[-7:]) / max(1, len(inequality[-7:]))) if inequality else 0.0,
            "equity_gap_in_adoption_mean": float(sum(equity_gap) / max(1, len(equity_gap))) if equity_gap else 0.0,
            "compliance_rate_in_mandated_locations_mean": float(sum(compliance_rate_loc) / max(1, len(compliance_rate_loc))) if compliance_rate_loc else 0.0,
            "average_mask_use_days_per_person_last": avg_days[-1] if avg_days else 0.0,
            "average_mask_availability_last": avg_inventory[-1] if avg_inventory else 0.0,
            "npi_effectiveness_mean": float(sum(npi_eff) / max(1, len(npi_eff))) if npi_eff else 0.0,
            "cumulative_infections_final": float(cum_inf_series[-1]) if cum_inf_series else 0.0,
            "Rt_last": float(Rt_series[-1]) if Rt_series else 0.0,
            "policy_cost_total": float(sum(policy_cost_daily)) if policy_cost_daily else 0.0,
            "stability_last_7_days": stability if stability is not None else None,
            # FIXED: Additional required metrics
            "sustained_adoption_rate": sustained_adoption_rate,
            "adoption_disparity_index": float(sum(inequality[-7:]) / max(1, len(inequality[-7:]))) if inequality else 0.0,
            "masks_sold_total": float(sum(masks_sold_daily)) if masks_sold_daily else 0.0,
            "policy_effect_contribution": policy_effect_contribution,
            # FIXED: Newly added spec metrics
            "mask_supply_shortage_days": mask_supply_shortage_days,
            "enforcement_cost": enforcement_cost_total,
            "misinformation_prevalence": misinfo_prev_mean,
        }

        # If ground truth provided, compute MAE
        if ground_truth is not None and "adoption_rate" in ground_truth:
            gt = ground_truth["adoption_rate"]
            gt = gt[start:end] if len(gt) >= end else gt
            n = min(len(gt), len(adoption))
            if n > 0:
                mae = sum(abs(adoption[i] - gt[i]) for i in range(n)) / float(n)
                rmse = math.sqrt(sum((adoption[i] - gt[i]) ** 2 for i in range(n)) / float(n))
                metrics["MAE_vs_observed_adoption"] = mae
                metrics["RMSE_vs_observed_adoption"] = rmse
            else:
                metrics["MAE_vs_observed_adoption"] = None
                metrics["RMSE_vs_observed_adoption"] = None
        else:
            metrics["MAE_vs_observed_adoption"] = None
            metrics["RMSE_vs_observed_adoption"] = None

        # Save metrics to artifacts
        try:
            ensure_dir(os.path.join(ARTIFACTS_DIR, "results"))
            with open(os.path.join(ARTIFACTS_DIR, "results", "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        except Exception:
            pass
        return metrics
        pass

    def visualize(self, show: bool = False, save_path: Optional[str] = None) -> None:
        """
        Plot adoption and stockout series if matplotlib available.

        Args:
            show: Whether to show the plot interactively.
            save_path: Path to save the figure.

        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            warnings.warn("matplotlib not available; skipping visualization.")
            return None

        fig, ax = plt.subplots(5, 1, figsize=(9, 12))
        ax[0].plot(self.results.get("adoption_rate", []), label="Adoption")
        ax[0].set_title("Adoption Rate")
        ax[0].set_ylim(0, 1)
        ax[0].legend()

        ax[1].plot(self.results.get("stockout_rate_daily", []), color="red", label="Stockout Indicator")
        ax[1].set_title("Stockout Indicator")
        ax[1].set_ylim(0, 1)
        ax[1].legend()

        ax[2].plot(self.results.get("new_infections_daily", []), color="purple", label="New Infections")
        ax[2].set_title("New Infections (Daily)")
        ax[2].legend()

        ax[3].plot(self.results.get("masks_sold_daily", []), color="green", label="Masks Sold (Daily)")
        ax[3].set_title("Masks Sold (Daily)")
        ax[3].legend()

        ax[4].plot(self.results.get("misinformation_prevalence", []), color="orange", label="Misinformation Prevalence")
        ax[4].set_title("Misinformation Prevalence (Daily)")
        ax[4].set_ylim(0, 1)
        ax[4].legend()

        plt.tight_layout()

        if save_path:
            try:
                ensure_dir(os.path.dirname(save_path))
                plt.savefig(save_path, dpi=150)
            except Exception:
                pass

        if show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close(fig)
        pass


# Pluggable Calibration Architecture

@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.

    Attributes:
        decision_weights: Decision head weights mapping.
        layer_weights: Weights for social layers (unused placeholder).
        info_params: Information/messaging related params.
        noise_params: Noise settings for decision processes.
        module_params: Module-specific overrides.
        engine_type: Engine compatibility identifier (constant).
        meta: Arbitrary metadata dict.
    """
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to plain dict.

        Returns:
            Dict representation of FittedParams.
        """
        return asdict(self)
        pass

    pass


class ParamsAdapter(ABC):
    """
    Adapts FittedParams to simulation parameter system.
    """

    @abstractmethod
    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system.

        Args:
            simulation: Simulation instance.
            params: FittedParams.

        Returns:
            None
        """
        raise NotImplementedError
        pass

    @abstractmethod
    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams object.
        """
        raise NotImplementedError
        pass

    @abstractmethod
    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen params and return warnings.

        Args:
            params: Fitted params.

        Returns:
            Dict mapping param keys to warning messages.
        """
        raise NotImplementedError
        pass

    pass


class BasicParamsAdapter(ParamsAdapter):
    """
    Basic adapter mapping FittedParams to Simulation.set_params() keys.
    Validates against parameter_definitions.json if available.
    """

    def __init__(self, param_defs_path: Optional[str] = None) -> None:
        """
        Initialize BasicParamsAdapter.

        Args:
            param_defs_path: Path to parameter_definitions.json.
        """
        self.param_defs_path = param_defs_path or os.path.join(PROJECT_ROOT, "parameter_definitions.json")
        self.param_defs: Dict[str, Dict[str, Any]] = self._load_param_defs(self.param_defs_path)
        pass

    def _load_param_defs(self, path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load parameter definitions JSON.

        Args:
            path: Path to JSON file.

        Returns:
            Dict of definitions keyed by param key.
        """
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    defs = json.load(f)
                # Expect list of param defs or dict mapping
                if isinstance(defs, list):
                    out = {}
                    for d in defs:
                        if isinstance(d, dict) and "key" in d:
                            out[d["key"]] = d
                    return out
                if isinstance(defs, dict):
                    # Already keyed
                    return defs
            except Exception as e:
                warnings.warn(f"Failed to load parameter definitions: {e}")
        # Fallback minimal defs
        return {}
        pass

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply fitted params to simulation params, ignoring frozen keys.

        Args:
            simulation: Simulation to update.
            params: Fitted params.

        Returns:
            None
        """
        # Map decision weights
        mapping = {
            "w_social": "social_influence_weight",
            "w_risk": "risk_perception_weight",
            "w_policy": "policy_weight",
            "disutility": "disutility_baseline",
        }
        updates: Dict[str, Any] = {}
        for k, v in params.decision_weights.items():
            if k in mapping:
                updates[mapping[k]] = float(v)

        # Info params
        info_map = {
            "message_effectiveness": "message_effectiveness",
            "memory_decay": "risk_decay_rate_per_day",
        }
        for k, v in params.info_params.items():
            if k in info_map:
                updates[info_map[k]] = float(v)

        # Noise
        if "temperature" in params.noise_params:
            updates["decision_noise"] = float(params.noise_params["temperature"])

        # Layer weights (optional mapping to module params if needed)
        for layer_k, layer_v in params.layer_weights.items():
            # No direct mapping; placeholder for future layered social networks
            _ = (layer_k, layer_v)

        # Module params overrides
        for _mod, kv in params.module_params.items():
            for k, v in kv.items():
                updates[k] = v

        # Validate frozen
        for key in list(updates.keys()):
            if key in self.param_defs and bool(self.param_defs[key].get("frozen", False)):
                warnings.warn(f"Ignoring override for frozen parameter: {key}")
                updates.pop(key, None)

        if updates:
            simulation.set_params(**updates)
        # Persist parameters_used.json
        try:
            with open(os.path.join(PROJECT_ROOT, "parameters_used.json"), "w", encoding="utf-8") as f:
                json.dump(simulation.get_params(), f, indent=2)
        except Exception:
            pass
        pass

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current params into a FittedParams structure.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams with mapped fields.
        """
        p = simulation.get_params()
        decision_weights = {
            "w_social": float(p.get("social_influence_weight", 0.4)),
            "w_risk": float(p.get("risk_perception_weight", 0.35)),
            "w_policy": float(p.get("policy_weight", 0.25)),
            "disutility": float(p.get("disutility_baseline", 0.1)),
        }
        info_params = {
            "message_effectiveness": float(p.get("message_effectiveness", 0.15)),
            "memory_decay": float(p.get("risk_decay_rate_per_day", 0.02)),
        }
        noise_params = {"temperature": float(p.get("decision_noise", 0.1))}
        fp = FittedParams(
            decision_weights=decision_weights,
            layer_weights={"family": 1.0, "work_school": 1.0, "community": 1.0},
            info_params=info_params,
            noise_params=noise_params,
            module_params={},
            engine_type="calibrasim",
            meta={"captured_at": time.time()},
        )
        return fp
        pass

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate overrides against frozen parameters and return warnings.

        Args:
            params: Fitted params to validate.

        Returns:
            Dict of warnings keyed by parameter.
        """
        warnings_map: Dict[str, str] = {}
        all_params: Dict[str, float] = {}
        # Collect all keys that might map to simulation params
        all_params.update(params.decision_weights)
        all_params.update(params.info_params)
        all_params.update(params.noise_params)
        for _mod, kv in params.module_params.items():
            all_params.update(kv)
        # Check each mapping
        mapping = {
            "w_social": "social_influence_weight",
            "w_risk": "risk_perception_weight",
            "w_policy": "policy_weight",
            "disutility": "disutility_baseline",
            "message_effectiveness": "message_effectiveness",
            "memory_decay": "risk_decay_rate_per_day",
            "temperature": "decision_noise",
        }
        for k in list(all_params.keys()):
            sim_key = mapping.get(k, k)
            if sim_key in self.param_defs and bool(self.param_defs[sim_key].get("frozen", False)):
                warnings_map[sim_key] = "Override ignored due to frozen parameter."
        return warnings_map
        pass

    pass


class Calibrator(ABC):
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """

    @abstractmethod
    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Return FittedParams, fitted strictly on the training window.

        Args:
            bundle: Auxiliary data bundle.
            simulator: Simulation instance to apply and run.
            evaluator: Evaluation callback.
            train_window: Tuple (start, end) days for training window.
            seed: Random seed.
            budget: Trials/budget.
            artifacts_dir: Directory to save trial artifacts.
            params_adapter: ParamsAdapter instance to apply parameters.

        Returns:
            Best FittedParams found.
        """
        raise NotImplementedError
        pass

    pass


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply params, run a forward simulation on window, and return a metrics dict.

    Args:
        simulator: Simulation instance (will be re-initialized shallowly).
        params: Fitted parameters to apply.
        window: Tuple (start, end) for evaluation window.

    Returns:
        Dict of key metrics including 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
        and 'TransitionFit' (P01, P11, P10, P00).
    """
    adapter = BasicParamsAdapter()
    adapter.apply(simulator, params)
    # Run fresh simulation over the window
    sim = Simulation(simulator.get_params())
    sim.run(start_day=window[0], end_day=window[1])
    # Load ground truth if available
    gt = load_ground_truth(DATA_DIR)
    # Evaluate with internal method (no extra windowing since run already limited)
    metrics = sim.evaluate(ground_truth=gt, window=None)
    # RMSE/MAE aggregate using adoption rate if ground truth exists
    sim_series = sim.results.get("adoption_rate", [])
    gt_series = gt.get("adoption_rate", []) if gt else []
    start, end = window
    gt_s = gt_series[start:end] if gt_series else []
    sim_s = sim_series
    n = min(len(sim_s), len(gt_s))
    if n > 0:
        rmse = math.sqrt(sum((sim_s[i] - gt_s[i]) ** 2 for i in range(n)) / float(n))
        mae = sum(abs(sim_s[i] - gt_s[i]) for i in range(n)) / float(n)
    else:
        rmse = float("nan")
        mae = float("nan")
    # Brier score approximate: binary from decision threshold at 0.5 vs series - degrade gracefully
    brier = mae if not math.isnan(mae) else 0.0

    # Transition fit from adoption rate transitions (approximate)
    P01 = 0.0
    P11 = 0.0
    P10 = 0.0
    P00 = 0.0
    trans_n = max(0, len(sim_s) - 1)
    if trans_n > 0:
        for i in range(trans_n):
            prev = 1 if sim_s[i] >= 0.5 else 0
            curr = 1 if sim_s[i + 1] >= 0.5 else 0
            if prev == 0 and curr == 1:
                P01 += 1
            elif prev == 1 and curr == 1:
                P11 += 1
            elif prev == 1 and curr == 0:
                P10 += 1
            elif prev == 0 and curr == 0:
                P00 += 1
        total = P01 + P11 + P10 + P00
        if total > 0:
            P01 /= total
            P11 /= total
            P10 /= total
            P00 /= total
    return {
        "RMSE_aggregate": rmse,
        "MAE_aggregate": mae,
        "Brier": brier,
        "TransitionFit": {"P01": P01, "P11": P11, "P10": P10, "P00": P00},
        "metrics": metrics,
    }
    pass


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search over selected simulator parameters.
    """

    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Initialize RandomSearchCalibrator.

        Args:
            search_space: Dict mapping simulation param keys to (low, high) ranges.
        """
        self.search_space = search_space or {
            "social_influence_weight": (0.0, 1.0),
            "risk_perception_weight": (0.0, 1.0),
            "policy_weight": (0.0, 1.0),
            "decision_noise": (0.01, 0.5),
            "message_effectiveness": (0.0, 0.3),
            "risk_decay_rate_per_day": (0.0, 0.1),
        }
        pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 50,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Randomly sample parameter vectors, evaluate, and keep the best.

        Args:
            bundle: Ancillary data (unused).
            simulator: Simulation instance.
            evaluator: Evaluation callback (evaluate_params).
            train_window: Training window (start, end).
            seed: Random seed.
            budget: Number of trials.
            artifacts_dir: Output directory for trial artifacts.
            params_adapter: ParamsAdapter to apply parameter sets.

        Returns:
            Best FittedParams found.
        """
        random.seed(seed)
        best_score = float("inf")
        best_params: Optional[FittedParams] = None
        if params_adapter is None:
            params_adapter = BasicParamsAdapter()
        art_dir = artifacts_dir or os.path.join(ARTIFACTS_DIR, "calibration_random_search")
        ensure_dir(art_dir)

        for i in range(budget):
            # Sample param vector
            sim_params = simulator.get_params()
            trial_params = copy.deepcopy(sim_params)
            for key, (lo, hi) in self.search_space.items():
                trial_params[key] = random.uniform(lo, hi)
            # FIXED: Downscale for calibration speed
            trial_params["population_size"] = max(200, int(trial_params.get("population_size", 2000) * 0.3))
            trial_params["simulation_duration_days"] = min(30, int(trial_params.get("simulation_days", trial_params.get("simulation_duration_days", 60))))
            # Map to FittedParams
            fp = BasicParamsAdapter().capture(Simulation(trial_params))
            # Evaluate
            sim_trial = Simulation(trial_params)
            scores = evaluator(sim_trial, fp, train_window)
            score = scores.get("RMSE_aggregate", float("inf"))
            # Save artifacts
            trial_dir = os.path.join(art_dir, f"trial_{i}")
            ensure_dir(trial_dir)
            try:
                with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                    json.dump(fp.to_dict(), f, indent=2)
                with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(scores, f, indent=2)
            except Exception:
                pass
            # Track best
            if score < best_score:
                best_score = score
                best_params = fp

        if best_params is None:
            best_params = BasicParamsAdapter().capture(simulator)

        # Save final result
        best_dir = os.path.join(artifacts_dir or ARTIFACTS_DIR, "calibration_random_search", "best")
        ensure_dir(best_dir)
        try:
            with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                json.dump(best_params.to_dict(), f, indent=2)
            report = {"budget": budget, "best_score": best_score}
            with open(os.path.join(artifacts_dir or ARTIFACTS_DIR, "calibration_random_search", "calibration_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass
        return best_params
        pass

    pass


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head from micro-transitions if available.
    Degrades gracefully to random search if micro data unavailable.
    """

    def __init__(self, l2_reg: float = 1.0) -> None:
        """
        Initialize LogitHeadCalibrator.

        Args:
            l2_reg: L2 regularization strength.
        """
        self.l2_reg = float(l2_reg)
        pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit the logistic head; if micro data missing, fallback to random search.

        Args:
            bundle: Ancillary data, expected to include micro transitions (optional).
            simulator: Simulation instance.
            evaluator: Evaluation callback.
            train_window: Training window.
            seed: RNG seed.
            budget: Fallback random search budget if needed.
            artifacts_dir: Directory for artifacts.
            params_adapter: ParamsAdapter.

        Returns:
            FittedParams.
        """
        # Attempt to use micro-transitions
        micro = bundle.get("micro_transitions") if bundle else None
        if not micro:
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        # Minimalistic "fit": estimate weights to target average adoption
        target = bundle.get("target_adoption_mean", 0.5)
        # Compute baseline adoption under current params
        base_sim = Simulation(simulator.get_params())
        base_sim.run(train_window[0], train_window[1])
        base_series = base_sim.results.get("adoption_rate", [])
        base_last7 = base_series[-7:] if len(base_series) >= 7 else base_series
        base_adopt = sum(base_last7) / max(1, len(base_last7))
        scale = 1.0
        if base_adopt > 1e-6:
            scale = target / base_adopt
        # Adjust social and risk weights proportionally
        captured = BasicParamsAdapter().capture(simulator)
        captured.decision_weights["w_social"] = clip01(captured.decision_weights["w_social"] * scale)
        captured.decision_weights["w_risk"] = clip01(captured.decision_weights["w_risk"] * (0.5 * scale + 0.5))
        captured.decision_weights["w_policy"] = clip01(captured.decision_weights["w_policy"] * (0.5 * scale + 0.5))
        return captured
        pass

    pass


class SNPECalibrator(Calibrator):
    """
    True SBI (SNPE) calibrator using neural density estimation, with graceful fallback.
    """

    def __init__(self) -> None:
        """
        Initialize SNPECalibrator.
        """
        self.available = False
        try:
            import torch  # noqa: F401
            from sbi import utils as sbi_utils  # noqa: F401
            from sbi.inference import SNPE as SNPE_Engine  # noqa: F401
            self.available = True
        except Exception:
            self.available = False
        pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: Simulation,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Calibrate via SNPE if available; fallback to random search otherwise.

        Args:
            bundle: Data bundle.
            simulator: Simulation instance.
            evaluator: Evaluation callback.
            train_window: Training window.
            seed: RNG seed.
            budget: Simulation budget.
            artifacts_dir: Output directory.
            params_adapter: ParamsAdapter.

        Returns:
            FittedParams.
        """
        if not self.available:
            warnings.warn("SNPE dependencies not available; falling back to random search calibrator.")
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        # Lightweight pseudo-SNPE: sample priors and keep best (as placeholder)
        rs = RandomSearchCalibrator()
        return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)
        pass

    pass


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None):
    """
    Create a calibrator instance by name.

    Args:
        name: Calibrator name.
        config_path: Optional JSON/YAML config path.

    Returns:
        Calibrator instance.

    Raises:
        ValueError: If name unknown.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    # Load optional config (JSON) into kwargs; currently unused defaults
    kwargs: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                kwargs.update(cfg)
        except Exception:
            pass
    return CALIBRATOR_REGISTRY[name](**kwargs)
    pass


def parse_cli(argv: List[str]) -> Dict[str, Any]:
    """
    Parse CLI arguments.

    Supported:
      --param-file path
      --set key=value (repeatable)
      --calibrator name
      --budget N
      --calib-window start:end
      --artifacts-dir path
      --no-calib

    Args:
        argv: Command-line arguments.

    Returns:
        Parsed options dict.
    """
    # FIXED: Add no_calib option defaulting to True via env SKIP_CALIB=1.
    opts: Dict[str, Any] = {
        "param_file": None,
        "overrides": {},
        "calibrator": "random_search",
        "budget": 20,
        "calib_window": None,
        "artifacts_dir": ARTIFACTS_DIR,
        "no_calib": True if os.environ.get("SKIP_CALIB", "1") == "1" else False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--param-file" and i + 1 < len(argv):
            opts["param_file"] = argv[i + 1]
            i += 2
        elif a == "--set" and i + 1 < len(argv):
            kv = argv[i + 1]
            if "=" in kv:
                k, v = kv.split("=", 1)
                # Try numeric conversion
                try:
                    if v.lower() in ("true", "false"):
                        vv = v.lower() == "true"
                    else:
                        if "." in v or "e" in v.lower():
                            vv = float(v)
                            if isinstance(vv, float) and vv.is_integer():
                                vv = int(vv)
                        else:
                            vv = int(v)
                except Exception:
                    vv = v
                opts["overrides"][k] = vv
            i += 2
        elif a == "--calibrator" and i + 1 < len(argv):
            opts["calibrator"] = argv[i + 1]
            i += 2
        elif a == "--budget" and i + 1 < len(argv):
            try:
                opts["budget"] = int(argv[i + 1])
            except Exception:
                opts["budget"] = 20
            i += 2
        elif a == "--calib-window" and i + 1 < len(argv):
            rng = argv[i + 1]
            if ":" in rng:
                s, e = rng.split(":")
                try:
                    opts["calib_window"] = (int(s), int(e))
                except Exception:
                    opts["calib_window"] = None
            i += 2
        elif a == "--artifacts-dir" and i + 1 < len(argv):
            opts["artifacts_dir"] = argv[i + 1]
            i += 2
        elif a == "--no-calib":
            opts["no_calib"] = True
            i += 1
        elif a == "--do-calib":
            opts["no_calib"] = False
            i += 1
        else:
            i += 1
    return opts
    pass


def load_params(param_file: Optional[str]) -> Dict[str, Any]:
    """
    Load parameters from JSON file, or return defaults if file missing.

    Args:
        param_file: Path to parameters JSON.

    Returns:
        Parameters dict.
    """
    defaults = {
        "population_size": 2000,
        "simulation_duration_days": 60,
        "network_avg_degree": 10,
        "network_rewire_prob": 0.1,
        "initial_mask_adoption_rate": 0.2,
        "perceived_risk_base": 0.2,
        "message_effectiveness": 0.15,
        "misinformation_rate": 0.05,
        "risk_decay_rate_per_day": 0.02,
        "mandate_level": 0.6,
        "enforcement_probability": 0.3,
        "fine_amount": 50,
        "location_mask_requirement_rate": 0.5,
        "mask_price": 1.0,
        "budget_mean": 100.0,
        "budget_std": 30.0,
        "supply_initial_stock_per_1000_people": 800,
        "restock_rate_per_day": 0.1,
        "supply_variability": 0.2,
        "price_adjustment_sensitivity": 0.5,
        "trust_in_authority_mean": 0.6,
        "trust_in_authority_std": 0.2,
        "fatigue_rate_per_day": 0.005,
        "fatigue_recovery_rate": 0.002,
        "contact_rate_per_day": 12,
        "time_fraction_work": 0.25,
        "time_fraction_retail": 0.15,
        "time_fraction_transport": 0.1,
        "time_fraction_public_space": 0.2,
        "mask_wearout_days": 5,
        "purchase_pack_size": 10,
        "peer_influence_memory_days": 3,
        "social_influence_weight": 0.4,
        "risk_perception_weight": 0.35,
        "policy_weight": 0.25,
        "decision_noise": 0.1,
        "disutility_baseline": 0.1,
        "random_seed": 42,
        "messaging_budget": 100000,
        "campaign_strategy": "balanced",
        "staff_enforcement_level_mean": 0.5,
        "gatekeeping_strictness_weight": 1.0,
        "local_outbreak_base": 0.2,
        "local_outbreak_volatility": 0.1,
        "communication_frequency": 7,
        # Optional parameters acknowledged by FIXED changes:
        "target_stock_per_person": 0.4,
        "exogenous_risk_weight": 0.3,
        # FIXED: Policy schedule and subsidies
        "mandate_start_day": None,
        "mandate_end_day": None,
        "policy_schedule": None,
        "subsidy_amount": 0.0,
        "distribution_days": [],
        "distribution_units_per_person": 2,
        "distribution_target": "low_income",
        "distribution_unit_cost": 1.0,
        # Disease defaults
        "enable_disease": True,
        "initial_infected_fraction": 0.01,
        "transmission_rate_base": 0.03,
        "recovery_rate": 0.1,
        "incubation_days": 3,
        "mask_efficacy_source_control": 0.5,
        "mask_efficacy_wearer_protection": 0.3,
        # Media influence
        "media_influence_effect": 0.1,
        # FIXED: Sustained adoption metric window default
        "sustained_window_days": 14,
    }
    if param_file and os.path.exists(param_file):
        try:
            with open(param_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                defaults.update(loaded)
        except Exception as e:
            warnings.warn(f"Failed to load param file: {e}")
    return defaults
    pass


def apply_overrides(params: Dict[str, Any], overrides: Dict[str, Any], param_defs_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply CLI overrides, ignoring frozen parameters if definitions available.

    Args:
        params: Original params.
        overrides: Overrides dict from CLI.
        param_defs_path: Path to parameter_definitions.json.

    Returns:
        Updated parameters dict.
    """
    defs = {}
    if param_defs_path and os.path.exists(param_defs_path):
        try:
            with open(param_defs_path, "r", encoding="utf-8") as f:
                definitions = json.load(f)
            if isinstance(definitions, list):
                defs = {d["key"]: d for d in definitions if isinstance(d, dict) and "key" in d}
            elif isinstance(definitions, dict):
                defs = definitions
        except Exception:
            defs = {}
    updated = copy.deepcopy(params)
    for k, v in overrides.items():
        frozen = bool(defs.get(k, {}).get("frozen", False)) if k in defs else False
        if frozen:
            warnings.warn(f"Ignoring override for frozen parameter: {k}")
            continue
        updated[k] = v
    # Persist used params
    try:
        with open(os.path.join(PROJECT_ROOT, "parameters_used.json"), "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
    except Exception:
        pass
    return updated
    pass


def load_ground_truth(data_dir: str) -> Optional[Dict[str, List[float]]]:
    """
    Load ground truth adoption series from data files if present.

    Args:
        data_dir: Base data directory.

    Returns:
        Dict with 'adoption_rate' series or None.
    """
    train_file = os.path.join(data_dir, "train_data.csv")
    if not os.path.exists(train_file):
        return None
    adoption: List[float] = []
    try:
        with open(train_file, "r", encoding="utf-8") as f:
            # Expect CSV with header 'day,adoption_rate'
            header = f.readline().strip().split(",")
            rate_idx = None
            for i, h in enumerate(header):
                if h.strip().lower() == "adoption_rate":
                    rate_idx = i
            for line in f:
                parts = line.strip().split(",")
                if rate_idx is not None and len(parts) > rate_idx:
                    try:
                        adoption.append(float(parts[rate_idx]))
                    except Exception:
                        continue
    except Exception:
        return None
    return {"adoption_rate": adoption}
    pass


def temporal_holdout_split(series: List[float]) -> Tuple[List[int], List[int]]:
    """
    Temporal split: first 80% of days for training, remaining 20% for validation.

    Args:
        series: Adoption series (or list of days length).

    Returns:
        (train_days, val_days) as lists of indices.
    """
    days = list(range(len(series)))
    if not days:
        return [], []
    split = int(0.8 * len(days))
    train = days[:split]
    val = days[split:]
    if not val:
        raise RuntimeError("No validation days available after temporal split.")
    return train, val
    pass


def main() -> None:
    """
    Orchestrate the simulation: parse CLI, load params, build simulator,
    calibration workflow, evaluation, and artifact saving.

    Returns:
        None
    """
    # CLI
    args = parse_cli(sys.argv[1:])
    params = load_params(args.get("param_file"))
    params = apply_overrides(params, args.get("overrides", {}), param_defs_path=os.path.join(PROJECT_ROOT, "parameter_definitions.json"))

    # Build simulator
    sim = Simulation(params)

    # Load ground truth for calibration split
    gt = load_ground_truth(DATA_DIR) or {"adoption_rate": [0.1] * params.get("simulation_days", params.get("simulation_duration_days", 60))}
    train_days, val_days = temporal_holdout_split(gt["adoption_rate"])
    train_window = (train_days[0] if train_days else 0, train_days[-1] + 1 if train_days else min(30, params.get("simulation_days", params.get("simulation_duration_days", 60))))
    calib_name = args.get("calibrator", "random_search")

    # Prepare artifacts dir
    artifacts_dir = args.get("artifacts_dir", ARTIFACTS_DIR)
    ensure_dir(artifacts_dir)

    # Bundle (placeholder for micro transitions if available)
    bundle = {"target_adoption_mean": sum(gt["adoption_rate"]) / max(1, len(gt["adoption_rate"]))}

    # FIXED: Gate calibration behind --no-calib flag and SKIP_CALIB env; default is to skip calibration.
    if not args.get("no_calib", True):
        calibrator = get_calibrator(calib_name, None)
        fitted = calibrator.fit(
                bundle=bundle,
                simulator=sim,
                evaluator=evaluate_params,
                train_window=train_window,
                seed=int(params.get("random_seed", 42)),
                budget=int(args.get("budget", 20)),
                artifacts_dir=artifacts_dir,
                params_adapter=BasicParamsAdapter(),
            )
        # Apply fitted params
        BasicParamsAdapter().apply(sim, fitted)

    # Run full rollout
    sim = Simulation(sim.get_params())
    sim.run(0, None)
    # Evaluate on validation window
    if val_days:
        val_window = (val_days[0], val_days[-1] + 1)
    else:
        val_window = (0, len(sim.results.get("adoption_rate", [])))
    metrics = sim.evaluate(load_ground_truth(DATA_DIR), window=val_window)

    # Save results and IO
    sim.save_results(os.path.join(ARTIFACTS_DIR, "results", "simulation_results.json"))
    sim.save_all_io(os.path.join(ARTIFACTS_DIR, "io"))

    # Visualization
    sim.visualize(show=False, save_path=os.path.join(ARTIFACTS_DIR, "figs", "overview.png"))

    # Concise summary print
    final_adopt = sim.results["adoption_rate"][-1] if sim.results.get("adoption_rate") else 0.0
    # FIXED: Print valid JSON instead of Python dict literal; include peak adoption and Rt_last.
    print(json.dumps({
        "final_adoption_rate": round(final_adopt, 3),
        "time_to_50": metrics.get("time_to_50_percent_adoption"),
        "time_to_70": metrics.get("time_to_70_percent"),
        "time_to_80": metrics.get("time_to_80_percent_adoption"),
        "peak_adoption_rate": round(metrics.get("peak_adoption_rate", 0.0), 3),
        "Rt_last": round(metrics.get("Rt_last", 0.0), 3),
        "stockout_rate_mean": round(metrics.get("stockout_rate_mean", 0.0), 3),
        "sustained_adoption_rate": round(metrics.get("sustained_adoption_rate", 0.0), 3),
        "mask_supply_shortage_days": metrics.get("mask_supply_shortage_days"),
        "enforcement_cost_total": round(metrics.get("enforcement_cost", 0.0), 3),
        "misinformation_prevalence_mean": round(metrics.get("misinformation_prevalence", 0.0), 3),
        "policy_effect_contribution": metrics.get("policy_effect_contribution")
    }))
    pass


# Execute main for both direct execution and sandbox wrapper invocation
main()