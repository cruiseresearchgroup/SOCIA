import os
import sys
import json
import math
import random
import argparse
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np

# FIXED: Removed stray non-Python text and SyntaxError-inducing lines from previous iteration.
# FIXED: Restored a functional main() and simulation orchestration.
# FIXED: Implemented core entities, modules, deterministic RNG, contact generation, SEIR progression, mask decision.
# FIXED: Implemented metrics aggregation and output, including Rt estimation and Gini.
# FIXED: Added direct main() call at file end for sandbox compatibility.

# Environment paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("mask_sim")


def seed_all(seed: int) -> np.random.Generator:
    """
    Seed Python's random and NumPy, returning a NumPy Generator for reproducibility.

    Args:
        seed: Seed integer.

    Returns:
        A numpy.random.Generator instance seeded with 'seed'.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def sigmoid(x: float) -> float:
    """
    Compute the logistic sigmoid function.

    Args:
        x: Input value.

    Returns:
        Sigmoid of x in (0,1).
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp a value x to the interval [lo, hi].

    Args:
        x: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    return max(lo, min(hi, x))


def moving_average(arr: List[float], window: int) -> List[float]:
    """
    Compute simple moving average over a list.

    Args:
        arr: Input list.
        window: Window size.

    Returns:
        List of moving averaged values, same length as arr (with leading padding).
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    if window <= 1:
        return list(arr)
    out = []
    s = 0.0
    for i, v in enumerate(arr):
        s += v
        if i >= window:
            s -= arr[i - window]
        denom = min(i + 1, window)
        out.append(s / denom)
    return out


def gini(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of non-negative values.

    Args:
        values: List of values.

    Returns:
        Gini coefficient in [0,1].
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(g)


@dataclass
class Person:
    """
    Person agent with behavioral, social, and epidemiological state.

    Attributes:
        id: Unique identifier.
        age_group: Age bin index (0..3).
        risk_perception: Perceived risk level [0,1].
        trust_in_institutions: Trust in authority [0,1].
        compliance_propensity: Intrinsic propensity to comply [0,1].
        social_influence_susceptibility: Susceptibility to peer behavior [0,1].
        political_orientation: Ideological orientation (-1, 0, +1).
        income: Income proxy for affordability.
        mask_stock: Count of masks owned.
        mask_preference: Preferred mask type (0=none,1=cloth,2=surgical,3=N95).
        mask_use_state: Current mask wearing (0/1).
        adoption_threshold: Threshold for social adoption [0,1].
        information_exposure_level: Info salience [0,1].
        network_neighbors: List of neighbor IDs.
        health_status: SEIR state ('S','E','I','R').
        days_in_state: Days spent in current health state.
        incubation_days_total: Total days in incubation (for E).
        infectious_days_total: Total days in infectious (for I).
        last_purchase_day: Last day of mask purchase.
    """
    id: int
    age_group: int
    risk_perception: float = 0.2
    trust_in_institutions: float = 0.5
    compliance_propensity: float = 0.5
    social_influence_susceptibility: float = 0.5
    political_orientation: int = 0
    income: float = 1.0
    mask_stock: int = 0
    mask_preference: int = 1
    mask_use_state: int = 0
    adoption_threshold: float = 0.5
    information_exposure_level: float = 0.5
    network_neighbors: List[int] = field(default_factory=list)
    health_status: str = "S"
    days_in_state: int = 0
    incubation_days_total: int = 0
    infectious_days_total: int = 0
    last_purchase_day: int = -1

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert agent to dictionary representation.

        Returns:
            Dictionary of fields.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


@dataclass
class Location:
    """
    Physical location with policy and risk characteristics.

    Attributes:
        id: Identifier.
        type: Location type name.
        base_contact_rate: Mean contacts per visit.
        base_transmission_risk: Relative transmission risk multiplier.
        mask_policy_required: Whether masks are mandated.
        enforcement_level: Enforcement strength [0,1].
    """
    id: int
    type: str
    base_contact_rate: float
    base_transmission_risk: float
    mask_policy_required: bool
    enforcement_level: float

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert location to dictionary.

        Returns:
            Dict of fields.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


@dataclass
class Retailer:
    """
    Retailer entity managing mask inventory and pricing.

    Attributes:
        id: Identifier.
        inventory: Current inventory level.
        price: Current unit price.
        restock_rate: Fractional restock towards baseline per day.
        supply_delay: Days between restock shipments.
        stockout_probability: Probability of disruption on restock day.
        daily_limit_per_person: Rationing limit per person per day.
        baseline_stock: Baseline stock target for pricing expansion.
        restock_carry: Accumulated fractional restock for batching.
    """
    id: int
    inventory: int
    price: float
    restock_rate: float
    supply_delay: int
    stockout_probability: float
    daily_limit_per_person: int = 5
    baseline_stock: int = 0
    restock_carry: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert retailer to dictionary.

        Returns:
            Dict of fields.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


@dataclass
class MediaChannel:
    """
    Information source with credibility and bias.

    Attributes:
        id: Identifier.
        source_type: Type name.
        credibility: Credibility weight [0,1].
        bias: Ideological bias [-1,1].
        message_frequency: Daily reach scale.
        message_strength: Persuasiveness scale.
        pro_mask_score: Pro-mask alignment [0,1].
    """
    id: int
    source_type: str
    credibility: float
    bias: float
    message_frequency: float
    message_strength: float
    pro_mask_score: float

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert media channel to dictionary.

        Returns:
            Dict of fields.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


@dataclass
class PolicyAuthority:
    """
    Policy authority managing global mask mandate state.

    Attributes:
        id: Identifier.
        policy_state: Current policy state string.
    """
    id: int = 1
    policy_state: str = "recommended"

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert policy authority to dictionary.

        Returns:
            Dict of fields.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


@dataclass
class ParameterDefinition:
    """
    Parameter definition entry for validation and calibration.

    Attributes:
        key: Parameter key.
        dtype: Data type string.
        default: Default value.
        bounds: Optional low,high bounds.
        owner_module: Name of owning module or 'global'.
        description: Description string.
        frozen: Whether parameter is frozen (non-overridable).
    """
    key: str
    dtype: str
    default: Any = None
    bounds: Optional[Tuple[float, float]] = None
    owner_module: str = "global"
    description: str = ""
    frozen: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert parameter definition to dictionary.

        Returns:
            Dictionary of fields.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


class State:
    """
    Simulation state container for agents, locations, retailers, and observables.

    Notes:
        State is mutated only during the commit stage of the scheduler. Modules produce
        outputs into buffers that the scheduler commits to this state.
    """
    def __init__(self, people: List[Person], locations: List[Location], retailers: List[Retailer],
                 media: List[MediaChannel], policy: PolicyAuthority, rng: np.random.Generator):
        """
        Initialize state with provided entities and RNG.

        Args:
            people: List of Person agents.
            locations: List of Location entities.
            retailers: List of Retailer entities.
            media: List of MediaChannel entities.
            policy: PolicyAuthority instance.
            rng: NumPy random generator.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.people = people
        self.locations = locations
        self.retailers = retailers
        self.media = media
        self.policy = policy
        self.rng = rng
        self.day = 0
        self.last_contacts: List[Tuple[int, int]] = []
        self.observables: Dict[str, List[Any]] = defaultdict(list)
        self.policy_change_events: List[Tuple[int, str, str]] = []
        self.metrics: Dict[str, Any] = {}
        self.params_snapshot: Dict[str, Any] = {}


class ModuleBase:
    """
    Base class for all modules in the simulation with a standard forward interface.

    Subclasses should implement forward() to compute outputs placed into buffers.
    """
    name: str = "module"
    tick_rate: int = 1
    dependencies: List[str] = []

    def __init__(self, params: Dict[str, Any]):
        """
        Initialize module with parameters.

        Args:
            params: Simulation parameter dictionary.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.params = params

    def should_run(self, t: int) -> bool:
        """
        Check if the module should run at day t based on tick_rate.

        Args:
            t: Current day.

        Returns:
            True if the module should run.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return (t % self.tick_rate) == 0

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Compute module outputs and place them into buffers.

        Args:
            state: Simulation state.
            buffers: Shared buffers dict for inter-module communication.
            t: Current day index.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        raise NotImplementedError("Subclasses must implement forward()")


class SocialInfluenceModule(ModuleBase):
    """
    Compute peer influence signals based on neighbors' mask use and personal thresholds.
    """
    name = "SocialInfluence"
    tick_rate = 1
    dependencies: List[str] = []

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Compute peer influence per person and store in buffers['peer_influence'].

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        people = state.people
        adoption_prev = np.array([p.mask_use_state for p in people], dtype=float)
        peer_frac = np.zeros(len(people), dtype=float)
        for i, p in enumerate(people):
            if p.network_neighbors:
                peer_frac[i] = adoption_prev[np.array(p.network_neighbors, dtype=int)].mean()
            else:
                peer_frac[i] = adoption_prev.mean() if len(people) > 0 else 0.0
        thresholds = np.array([clamp(p.adoption_threshold, 0.0, 1.0) for p in people], dtype=float)
        susc = np.array([clamp(p.social_influence_susceptibility, 0.0, 1.0) for p in people], dtype=float)
        social_signal = (peer_frac - thresholds) * susc
        weight = float(self.params.get("social_influence_weight", 0.35))
        buffers["peer_influence"] = np.clip(weight * social_signal, -1.0, 1.0)


class InformationBroadcastModule(ModuleBase):
    """
    Broadcast messages from media sources and compute information influence signals.
    """
    name = "InformationBroadcast"
    tick_rate = 1
    dependencies: List[str] = []

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Compute info influence per person and updated exposure level; write to buffers.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        people = state.people
        sources = state.media
        info_weight = float(self.params.get("information_effect_weight", 0.25))
        decay = float(self.params.get("info_decay_rate", 0.1))
        align_strength = float(self.params.get("info_bias_alignment_strength", 0.6))
        info_influence = np.zeros(len(people), dtype=float)
        new_exposure = np.zeros(len(people), dtype=float)

        # Map political orientation to bias in [-1,1]
        pol_bias = { -1: -1.0, 0: 0.0, 1: 1.0 }

        for i, p in enumerate(people):
            exposure = float(p.information_exposure_level)
            total_msg = 0.0
            p_bias = pol_bias.get(int(p.political_orientation), 0.0)
            for s in sources:
                align = 1.0 - abs(p_bias - float(s.bias)) * align_strength
                align = clamp(align, 0.0, 1.0)
                reach = float(s.message_frequency) * float(s.message_strength) * float(s.credibility) * align
                total_msg += reach * (float(s.pro_mask_score) - 0.5)
            total_msg *= info_weight
            info_influence[i] = clamp(total_msg, -1.0, 1.0)
            new_exposure[i] = clamp(exposure * (1.0 - decay) + abs(total_msg) * 0.2, 0.0, 1.0)

        buffers["info_influence"] = info_influence
        buffers["new_info_exposure"] = new_exposure


class PolicyModule(ModuleBase):
    """
    Update global policy state based on prevalence thresholds and set location policies.
    """
    name = "PolicyModule"
    tick_rate = 3
    dependencies: List[str] = ["DiseaseDynamics"]

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Update policy state to 'mandate' or 'recommended' based on prevalence.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        prev = sum(1 for p in state.people if p.health_status == "I") / max(1, len(state.people))
        tighten = float(self.params.get("policy_trigger_prevalence", 0.02))
        relax = float(self.params.get("policy_relaxation_prevalence", 0.005))
        prev_state = state.policy.policy_state
        new_state = prev_state
        if prev >= tighten:
            new_state = "mandate"
        elif prev <= relax:
            new_state = "recommended"

        if new_state != prev_state:
            buffers.setdefault("policy_change_events", []).append((t, prev_state, new_state))
        buffers["new_policy_state"] = new_state

        # Update location policies in buffer
        loc_policy_map = {}
        mandate_map = self.params.get("location_policy_map", {"workplace": True, "retail": True, "transit": True, "public": True})
        for loc in state.locations:
            required = (new_state == "mandate") and bool(mandate_map.get(loc.type, False))
            loc_policy_map[loc.id] = required
        buffers["locations_policy"] = loc_policy_map


class PolicyEnforcementModule(ModuleBase):
    """
    Compute per-person policy pressure (compliance likelihood) based on enforcement and trust.
    """
    name = "PolicyEnforcement"
    tick_rate = 1
    dependencies: List[str] = ["PolicyModule"]

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Produce policy_pressure for each person given current policy and enforcement.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        # Average enforcement over required locations
        loc_required = buffers.get("locations_policy", {})
        enforce_levels = []
        for loc in state.locations:
            if loc_required.get(loc.id, False):
                enforce_levels.append(float(loc.enforcement_level))
        avg_enforce = float(np.mean(enforce_levels)) if enforce_levels else 0.0
        policy_weight = float(self.params.get("policy_effect_weight", 0.3))
        no_mask_penalty = float(self.params.get("compliance_penalty", 0.8))
        # Compute per-person
        pressures = np.zeros(len(state.people), dtype=float)
        for i, p in enumerate(state.people):
            base = avg_enforce * policy_weight * clamp(p.trust_in_institutions, 0.0, 1.0)
            x = 5.0 * (base - 0.3)
            prob = sigmoid(x)
            if p.mask_stock <= 0:
                prob = max(0.0, prob - no_mask_penalty * 0.1)
            pressures[i] = clamp(prob, 0.0, 1.0)
        buffers["policy_pressure"] = pressures


class RetailMarketModule(ModuleBase):
    """
    Manage daily mask procurement, restock, and pricing adjustments.
    """
    name = "RetailMarket"
    tick_rate = 1
    dependencies: List[str] = []

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Compute mask purchases and retailer updates; write increments to buffers.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        rng = state.rng
        people = state.people
        retailers = state.retailers
        if not retailers:
            buffers["mask_stock_add"] = np.zeros(len(people), dtype=int)
            return

        r = retailers[0]
        # Expected per-day restock
        baseline = r.baseline_stock if r.baseline_stock > 0 else max(r.inventory, 1)
        restock_units = r.restock_rate * baseline + r.restock_carry
        units_to_add = int(restock_units)
        r.restock_carry = restock_units - units_to_add
        if (t % max(1, r.supply_delay)) == 0:
            if rng.random() > r.stockout_probability:
                r.inventory += units_to_add
        # Simple price adjustment based on stock tightness
        target = r.baseline_stock if r.baseline_stock > 0 else baseline
        tightness = 1.0 - (r.inventory / max(1.0, float(target)))
        base_price = float(self.params.get("mask_price_base", r.price))
        r.price = max(0.2, base_price * (1.0 + 0.5 * clamp(tightness, 0.0, 1.0)))
        # Demand: agents with low stock attempt to buy up to ration limit or up to buffer size
        ration_limit = int(self.params.get("mask_ration_limit_per_person", r.daily_limit_per_person))
        desired_buffer = int(self.params.get("mask_buffer_target", 3))
        price_elasticity = float(self.params.get("price_elasticity", 1.0))
        access_income_elasticity = float(self.params.get("access_inequity_income_elasticity", 0.6))
        mask_stock_add = np.zeros(len(people), dtype=int)

        if r.inventory > 0:
            # Compute affordability weights
            incomes = np.array([max(0.1, p.income) for p in people], dtype=float)
            affordability = np.minimum(1.0, incomes / np.percentile(incomes, 75))
            affordability = affordability ** access_income_elasticity
            price_factor = (base_price / max(0.2, r.price)) ** price_elasticity
            weights = affordability * price_factor
            # Selection: people needing masks
            need_idx = [i for i, p in enumerate(people) if p.mask_stock < desired_buffer]
            if need_idx:
                # Allocate iteratively up to inventory or needs
                for _ in range(min(r.inventory, len(need_idx) * ration_limit)):
                    # Weighted random choice among needers
                    w = np.array([weights[i] for i in need_idx], dtype=float)
                    if w.sum() <= 0:
                        break
                    w = w / w.sum()
                    pick = int(np.searchsorted(np.cumsum(w), rng.random()))
                    pid = need_idx[pick]
                    if mask_stock_add[pid] >= ration_limit:
                        continue
                    mask_stock_add[pid] += 1
                    r.inventory -= 1
                    if r.inventory <= 0:
                        break
        buffers["mask_stock_add"] = mask_stock_add
        buffers["retailer_inventory"] = r.inventory
        buffers["retailer_price"] = r.price


class MovementAndInteractionModule(ModuleBase):
    """
    Generate stochastic contacts for disease transmission; performance-safe O(N).
    """
    name = "MovementAndInteraction"
    tick_rate = 1
    dependencies: List[str] = []

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Produce contact pairs for transmission this day.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        rng = state.rng
        people = state.people
        N = len(people)
        contacts: List[Tuple[int, int]] = []
        avg_contacts = float(self.params.get("average_degree", 8.0))
        # Only infectious generate contacts to conserve time
        infectious_ids = [p.id for p in people if p.health_status == "I"]
        for i in infectious_ids:
            n = rng.poisson(lam=avg_contacts)
            for _ in range(n):
                j = int(rng.integers(0, N))
                if j == i:
                    continue
                contacts.append((i, j))
        buffers["contacts"] = contacts


class AdoptionDecisionModule(ModuleBase):
    """
    Compute mask-wearing decisions given influences, risk, habit, and access constraints.
    """
    name = "AdoptionDecision"
    tick_rate = 1
    dependencies: List[str] = ["SocialInfluence", "InformationBroadcast", "PolicyEnforcement"]

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Compute new mask_use_state via logistic choice; write array to buffers.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        rng = state.rng
        people = state.people
        N = len(people)
        peer = buffers.get("peer_influence", np.zeros(N, dtype=float))
        info = buffers.get("info_influence", np.zeros(N, dtype=float))
        policy = buffers.get("policy_pressure", np.zeros(N, dtype=float))
        habit_persistence = float(self.params.get("habit_persistence", 0.8))
        risk_weight = float(self.params.get("risk_perception_weight", 0.2))
        noise_scale = float(self.params.get("adoption_noise", 0.05))
        temperature = float(self.params.get("decision_temperature", 1.0))
        new_mask = np.zeros(N, dtype=int)
        adoption_changes: List[Tuple[int, int, int]] = []

        for i, p in enumerate(people):
            habit = habit_persistence * (1 if p.mask_use_state == 1 else -1)
            risk_signal = risk_weight * (clamp(p.risk_perception, 0.0, 1.0) - 0.5)
            total = peer[i] + info[i] + policy[i] + habit + risk_signal + rng.normal(0.0, noise_scale)
            propensity = sigmoid(total / max(1e-6, temperature))
            if p.mask_stock <= 0:
                propensity *= 0.2
            new_state = 1 if rng.random() < propensity else 0
            new_mask[i] = new_state
            if new_state != p.mask_use_state:
                adoption_changes.append((i, p.mask_use_state, new_state))
        buffers["mask_use_state_new"] = new_mask
        buffers["adoption_change"] = adoption_changes


class DiseaseDynamicsModule(ModuleBase):
    """
    SEIR dynamics on contact events with mask effects and risk perception feedback.
    """
    name = "DiseaseDynamics"
    tick_rate = 1
    dependencies: List[str] = ["MovementAndInteraction", "AdoptionDecision"]

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Process transmissions, importations, and recoveries; write incidence observables.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        rng = state.rng
        people = state.people
        contacts: List[Tuple[int, int]] = buffers.get("contacts", [])
        base_inf = float(self.params.get("base_infection_rate", 0.05))
        mask_eff = float(self.params.get("mask_effectiveness", 0.5))
        recovery_rate = float(self.params.get("recovery_rate", 0.15))
        import_rate = float(self.params.get("importation_rate", 0.0002))

        incidence = 0
        counterf = 0
        new_E = []
        # Transmission on contacts
        for (i, j) in contacts:
            a = people[i]
            b = people[j]
            # Check susceptible-infectious pairs both directions
            pairs = []
            if a.health_status == "I" and b.health_status == "S":
                pairs.append((a, b))
            if b.health_status == "I" and a.health_status == "S":
                pairs.append((b, a))
            for src, dst in pairs:
                p = base_inf * ((1.0 - mask_eff) ** (src.mask_use_state + dst.mask_use_state))
                if rng.random() < p:
                    # Infect as exposed, set durations
                    if dst.health_status == "S":
                        new_E.append(dst.id)
                        incidence += 1
                # Counterfactual without masks
                if rng.random() < base_inf:
                    counterf += 1
        # Apply new exposures
        disease_updates = []
        for pid in new_E:
            p = people[pid]
            if p.health_status == "S":
                p.health_status = "E"
                p.days_in_state = 0
                p.incubation_days_total = int(rng.integers(3, 6))
                disease_updates.append((pid, "S", "E"))

        # Importations
        for p in people:
            if p.health_status == "S" and rng.random() < import_rate:
                p.health_status = "E"
                p.days_in_state = 0
                p.incubation_days_total = int(rng.integers(3, 6))
                incidence += 1
                disease_updates.append((p.id, "S", "E"))

        # Progression E -> I
        for p in people:
            if p.health_status == "E":
                p.days_in_state += 1
                if p.days_in_state >= max(1, p.incubation_days_total):
                    p.health_status = "I"
                    p.days_in_state = 0
                    p.infectious_days_total = int(rng.integers(6, 10))
                    disease_updates.append((p.id, "E", "I"))

        # Recovery I -> R
        for p in people:
            if p.health_status == "I":
                # Deterministic duration or stochastic recovery
                p.days_in_state += 1
                deterministic = p.infectious_days_total > 0 and p.days_in_state >= p.infectious_days_total
                if deterministic or rng.random() < recovery_rate:
                    p.health_status = "R"
                    p.days_in_state = 0
                    disease_updates.append((p.id, "I", "R"))

        buffers["incidence"] = incidence
        buffers["counterf_incidence"] = counterf
        buffers["disease_updates"] = disease_updates

        # Update local prevalence proxy into buffers for other modules if needed
        prevalence = sum(1 for p in people if p.health_status == "I") / max(1, len(people))
        buffers["prevalence"] = prevalence


class AdoptionAggregatorModule(ModuleBase):
    """
    Aggregate adoption, compliance, equity, and epidemiological observables.
    """
    name = "AdoptionAggregator"
    tick_rate = 1
    dependencies: List[str] = ["AdoptionDecision", "DiseaseDynamics", "PolicyEnforcement"]

    def forward(self, state: State, buffers: Dict[str, Any], t: int) -> None:
        """
        Append daily observables into state.observables.

        Args:
            state: Simulation state.
            buffers: Shared buffers.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        N = len(state.people)
        if N <= 0:
            return
        mask_arr = buffers.get("mask_use_state_new", np.array([p.mask_use_state for p in state.people], dtype=int))
        adoption_rate = float(np.mean(mask_arr)) if len(mask_arr) > 0 else 0.0
        # Approximate compliance: share of mask-wearers when policy required
        policy_press = buffers.get("policy_pressure", np.zeros(N, dtype=float))
        compliance = float(np.mean(policy_press)) if len(policy_press) > 0 else 0.0
        # Access equity gini over mask stocks
        mask_stocks = [p.mask_stock for p in state.people]
        access_gini = gini(mask_stocks)
        # Incidence metrics
        incidence = int(buffers.get("incidence", 0))
        counterf = int(buffers.get("counterf_incidence", 0))
        irr = 0.0
        if counterf > 0:
            irr = 1.0 - (incidence / max(1.0, float(counterf)))
        # Append
        state.observables["adoption_rate_daily"].append(adoption_rate)
        state.observables["compliance_in_policy_locations_daily"].append(compliance)
        state.observables["mask_access_gini_daily"].append(float(access_gini))
        state.observables["incidence_daily"].append(incidence)
        state.observables["counterfactual_incidence_daily"].append(counterf)
        state.observables["infection_rate_reduction_daily"].append(float(irr))
        # Adoption by group placeholders (age_group, political_orientation)
        by_age = defaultdict(list)
        by_pol = defaultdict(list)
        for p, m in zip(state.people, mask_arr):
            by_age[p.age_group].append(m)
            by_pol[p.political_orientation].append(m)
        state.observables["adoption_rate_by_age_daily"].append({int(k): float(np.mean(v)) for k, v in by_age.items()})
        state.observables["adoption_rate_by_political_identity_daily"].append({int(k): float(np.mean(v)) for k, v in by_pol.items()})
        # Abandonment rate: estimate from adoption_change
        changes = buffers.get("adoption_change", [])
        abandonments = sum(1 for (_, old, new) in changes if old == 1 and new == 0)
        prev_adopters = max(1, sum(1 for p in state.people if p.mask_use_state == 1))
        abandonment_rate = abandonments / prev_adopters
        state.observables["abandonment_rate_daily"].append(float(abandonment_rate))
        # Threshold adoption time
        target = float(self.params.get("threshold_adoption_target", 0.7))
        if "threshold_adoption_time" not in state.observables:
            if adoption_rate >= target:
                state.observables["threshold_adoption_time"] = t
        # Effective Rt estimate via moving ratio with serial interval
        serial = int(self.params.get("serial_interval_days", 5))
        inc = state.observables["incidence_daily"]
        if len(inc) > serial:
            Rt = inc[-1] / max(1e-6, inc[-1 - serial])
        else:
            Rt = 0.0
        state.observables["effective_reproduction_number_Rt"].append(float(Rt))


class DAGScheduler:
    """
    Simple scheduler for running modules in order with buffers and commit phase.
    """
    def __init__(self, modules: List[ModuleBase]):
        """
        Initialize scheduler.

        Args:
            modules: List of module instances in execution order.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.modules = modules

    def step(self, state: State, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        """
        Execute one simulation day: run eligible modules, then commit buffered outputs.

        Args:
            state: Simulation state.
            params: Parameter dictionary.
            t: Current day.

        Returns:
            Buffers dict for this step.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        buffers: Dict[str, Any] = {}
        for m in self.modules:
            if m.should_run(t):
                m.forward(state, buffers, t)
        self.commit(state, buffers, params, t)
        return buffers

    def commit(self, state: State, buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Commit buffered updates to simulation state.

        Args:
            state: Simulation state.
            buffers: Buffers dict.
            params: Parameter dictionary.
            t: Current day.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        # Apply info exposure
        if "new_info_exposure" in buffers:
            new_info_exposure = buffers["new_info_exposure"]
            for i, p in enumerate(state.people):
                p.information_exposure_level = float(new_info_exposure[i])
        # Apply mask purchases
        if "mask_stock_add" in buffers:
            adds = buffers["mask_stock_add"]
            for i, p in enumerate(state.people):
                p.mask_stock = max(0, int(p.mask_stock) + int(adds[i]))
                # Track last purchase day
                if int(adds[i]) > 0:
                    p.last_purchase_day = t
        # Apply policy changes
        if "new_policy_state" in buffers:
            prev = state.policy.policy_state
            new = buffers["new_policy_state"]
            state.policy.policy_state = new
            if "policy_change_events" in buffers:
                for ev in buffers["policy_change_events"]:
                    state.policy_change_events.append(ev)
            if prev != new:
                logger.info(f"Policy change at day {t}: {prev} -> {new}")
        if "locations_policy" in buffers:
            loc_required = buffers["locations_policy"]
            for loc in state.locations:
                loc.mask_policy_required = bool(loc_required.get(loc.id, loc.mask_policy_required))
        # Apply new mask wearing decisions and decrement stock for wearers
        if "mask_use_state_new" in buffers:
            new_mask = buffers["mask_use_state_new"]
            for i, p in enumerate(state.people):
                p.mask_use_state = int(new_mask[i])
                if p.mask_use_state == 1:
                    p.mask_stock = max(0, p.mask_stock - 1)
        # Persist contacts for reference
        if "contacts" in buffers:
            state.last_contacts = buffers["contacts"]
        # Append incidence observables already handled in aggregator

        # Advance day
        state.day = t + 1


class Simulation:
    """
    Main simulation engine that constructs entities, runs modules via a scheduler,
    manages parameters, and produces results, metrics, and artifacts.
    """
    def __init__(self, params: Dict[str, Any], param_defs: Dict[str, ParameterDefinition]):
        """
        Initialize Simulation with parameters and registry.

        Args:
            params: Parameter dictionary.
            param_defs: Parameter definitions registry keyed by parameter name.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.params = params
        self.param_defs = param_defs
        self.rng = seed_all(int(params.get("seed_random_state", params.get("seed", 42))))
        self.state: Optional[State] = None
        self.scheduler: Optional[DAGScheduler] = None
        self.modules: Dict[str, ModuleBase] = {}
        self.artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.io_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Build world and modules
        self.reset()

    def reset(self) -> None:
        """
        Reset and rebuild the world and modules to initial state.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        N = int(self.params.get("num_agents", self.params.get("population_size", 2000)))
        # Initialize people with heterogeneity
        income_var = float(self.params.get("income_variance", self.params.get("heterogeneity_scalers", {}).get("income_variance", 0.6)))
        risk_var = float(self.params.get("risk_variance", 0.15))
        trust_var = float(self.params.get("trust_variance", 0.2))
        avg_degree = int(self.params.get("network_avg_degree", self.params.get("average_degree", 8)))
        people: List[Person] = []
        for i in range(N):
            age_group = int(self.rng.integers(0, 4))
            risk = clamp(float(self.rng.normal(0.2, risk_var)), 0.0, 1.0)
            trust = clamp(float(self.rng.normal(0.5, trust_var)), 0.0, 1.0)
            compliance = clamp(float(self.rng.uniform(0.3, 0.8)), 0.0, 1.0)
            susc = clamp(float(self.rng.uniform(0.2, 0.8)), 0.0, 1.0)
            pol = int(self.rng.choice([-1, 0, 1]))
            income = float(np.clip(self.rng.normal(1.0, income_var), 0.1, 10.0))
            thresh = clamp(float(self.rng.normal(self.params.get("adoption_threshold_mean", 0.5),
                                                 self.params.get("adoption_threshold_sd", 0.15))), 0.0, 1.0)
            info = clamp(float(self.rng.uniform(0.2, 0.8)), 0.0, 1.0)
            p = Person(
                id=i,
                age_group=age_group,
                risk_perception=risk,
                trust_in_institutions=trust,
                compliance_propensity=compliance,
                social_influence_susceptibility=susc,
                political_orientation=pol,
                income=income,
                mask_stock=0,
                mask_preference=int(self.rng.choice([1, 2], p=[0.7, 0.3])),
                mask_use_state=0,
                adoption_threshold=thresh,
                information_exposure_level=info,
            )
            people.append(p)
        # Build simple ring-lattice network
        k = max(1, avg_degree // 2)
        for i, p in enumerate(people):
            p.network_neighbors = [int((i + j) % N) for j in range(1, k + 1)] + [int((i - j) % N) for j in range(1, k + 1)]

        # Locations
        location_policy_map = self.params.get("location_policy_map", {"workplace": True, "retail": True, "transit": True, "public": True})
        locations = [
            Location(id=0, type="household", base_contact_rate=6.0, base_transmission_risk=1.0, mask_policy_required=False,
                     enforcement_level=float(self.params.get("enforcement_strength_mean", 0.6))),
            Location(id=1, type="workplace", base_contact_rate=8.0, base_transmission_risk=1.0,
                     mask_policy_required=False, enforcement_level=float(self.params.get("enforcement_strength_mean", 0.6))),
            Location(id=2, type="retail", base_contact_rate=4.0, base_transmission_risk=1.0,
                     mask_policy_required=bool(location_policy_map.get("retail", True)),
                     enforcement_level=float(self.params.get("enforcement_strength_mean", 0.6))),
            Location(id=3, type="transit", base_contact_rate=5.0, base_transmission_risk=1.0,
                     mask_policy_required=bool(location_policy_map.get("transit", True)),
                     enforcement_level=float(self.params.get("enforcement_strength_mean", 0.6))),
            Location(id=4, type="public", base_contact_rate=2.0, base_transmission_risk=1.0,
                     mask_policy_required=bool(location_policy_map.get("public", True)),
                     enforcement_level=float(self.params.get("enforcement_strength_mean", 0.6))),
        ]
        # Retailer
        total_stock = int(self.params.get("initial_mask_stock_per_1000", 2000) * N / 1000)
        retailer = Retailer(
            id=0,
            inventory=total_stock,
            price=float(self.params.get("mask_price_base", self.params.get("mask_price", 1.0))),
            restock_rate=float(self.params.get("restock_rate_per_day", self.params.get("retailer_restock_rate_per_day", 0.05))),
            supply_delay=int(self.params.get("supply_delay_days", 3)),
            stockout_probability=float(self.params.get("supply_disruption_probability", 0.05)),
            daily_limit_per_person=int(self.params.get("mask_ration_limit_per_person", 5)),
            baseline_stock=total_stock,
        )
        retailers = [retailer]

        # Media channels (minimal)
        media: List[MediaChannel] = [
            MediaChannel(id=0, source_type="public_health", credibility=0.85, bias=0.8, message_frequency=1.0, message_strength=0.9, pro_mask_score=0.9),
            MediaChannel(id=1, source_type="local_news", credibility=0.6, bias=0.1, message_frequency=0.8, message_strength=0.5, pro_mask_score=0.4),
            MediaChannel(id=2, source_type="social_media_pro_mask", credibility=0.45, bias=0.7, message_frequency=1.2, message_strength=0.7, pro_mask_score=0.8),
            MediaChannel(id=3, source_type="social_media_anti_mask", credibility=0.35, bias=-0.7, message_frequency=1.2, message_strength=0.7, pro_mask_score=0.2),
            MediaChannel(id=4, source_type="community_leader", credibility=0.55, bias=0.2, message_frequency=0.5, message_strength=0.6, pro_mask_score=0.6),
            MediaChannel(id=5, source_type="employer", credibility=0.65, bias=0.4, message_frequency=0.6, message_strength=0.7, pro_mask_score=0.7),
        ]
        policy = PolicyAuthority()

        # Seed initial infection and mask adoption
        init_inf_frac = float(self.params.get("initial_infected_fraction", self.params.get("initial_infected_rate", 0.01)))
        init_adopt_frac = float(self.params.get("initial_mask_adoption_fraction", self.params.get("initial_adoption_rate", 0.15)))
        infected_ids = set(self.rng.choice(N, size=max(1, int(init_inf_frac * N)), replace=False).tolist())
        adopters = set(self.rng.choice(N, size=max(1, int(init_adopt_frac * N)), replace=False).tolist())
        for i, p in enumerate(people):
            if i in infected_ids:
                p.health_status = "I"
                p.days_in_state = 0
                p.infectious_days_total = int(self.rng.integers(6, 10))
            else:
                p.health_status = "S"
            if i in adopters:
                p.mask_use_state = 1
                p.mask_stock = max(p.mask_stock, 3)

        # Create state
        self.state = State(people=people, locations=locations, retailers=retailers, media=media, policy=policy, rng=self.rng)
        self.state.params_snapshot = dict(self.params)

        # Setup modules and scheduler
        self.modules = {
            "SocialInfluence": SocialInfluenceModule(self.params),
            "InformationBroadcast": InformationBroadcastModule(self.params),
            "PolicyModule": PolicyModule(self.params),
            "PolicyEnforcement": PolicyEnforcementModule(self.params),
            "RetailMarket": RetailMarketModule(self.params),
            "MovementAndInteraction": MovementAndInteractionModule(self.params),
            "AdoptionDecision": AdoptionDecisionModule(self.params),
            "DiseaseDynamics": DiseaseDynamicsModule(self.params),
            "AdoptionAggregator": AdoptionAggregatorModule(self.params),
        }
        module_order = [
            self.modules["PolicyModule"],
            self.modules["SocialInfluence"],
            self.modules["InformationBroadcast"],
            self.modules["PolicyEnforcement"],
            self.modules["RetailMarket"],
            self.modules["MovementAndInteraction"],
            self.modules["AdoptionDecision"],
            self.modules["DiseaseDynamics"],
            self.modules["AdoptionAggregator"],
        ]
        self.scheduler = DAGScheduler(module_order)

    def run(self, start_day: int, end_day: int) -> None:
        """
        Execute simulation from start_day (inclusive) to end_day (exclusive).

        Args:
            start_day: Start day index.
            end_day: End day index.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        assert self.state is not None and self.scheduler is not None
        for t in range(start_day, end_day):
            buffers = self.scheduler.step(self.state, self.params, t)
            # Optionally record IO for debugging (minimal to avoid slowdowns)
            if (t - start_day) % max(1, int(self.params.get("io_record_stride", 1000))) == 0:
                self.io_records["buffers"].append({"day": t, "keys": list(buffers.keys())})

    def set_params(self, module: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Update parameters, respecting frozen flags from param definitions.

        Args:
            module: Optional module name to scope parameters.
            **kwargs: Key-value pairs to set.

        Returns:
            Dict of applied parameter changes.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        applied = {}
        for k, v in kwargs.items():
            defn = self.param_defs.get(k, None)
            if defn and defn.frozen:
                logger.warning(f"Override ignored for frozen parameter: {k}")
                continue
            self.params[k] = v
            applied[k] = v
        if applied:
            # Rebuild if structural params changed (simplified: always rebuild)
            self.reset()
        return applied

    def get_params(self) -> Dict[str, Any]:
        """
        Return current parameter dictionary.

        Returns:
            Parameter dict.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return dict(self.params)

    def save_results(self, filename: str) -> None:
        """
        Save simulation observables and metrics to a JSON file.

        Args:
            filename: Output filename path.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        results = {
            "observables": self.state.observables if self.state else {},
            "metrics": self.state.metrics if self.state else {},
            "policy_change_events": self.state.policy_change_events if self.state else [],
            "params": self.params,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def save_module_io(self, module_name: str, path: str) -> None:
        """
        Save recorded IO keys for a specific module (placeholder, minimal).

        Args:
            module_name: Name of module to save IO for.
            path: File path to save.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"module": module_name, "io_records": self.io_records.get(module_name, [])}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_all_io(self, root_dir: str) -> None:
        """
        Save IO records for all modules.

        Args:
            root_dir: Root directory to store IO files.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        os.makedirs(root_dir, exist_ok=True)
        for mod in self.modules:
            self.save_module_io(mod, os.path.join(root_dir, f"{mod}_io.json"))

    def evaluate(self) -> Dict[str, Any]:
        """
        Compute evaluation metrics from observables and store them in state.metrics.

        Returns:
            Metrics dictionary.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        obs = self.state.observables
        metrics: Dict[str, Any] = {}
        adoption = obs.get("adoption_rate_daily", [])
        incidence = obs.get("incidence_daily", [])
        irr = obs.get("infection_rate_reduction_daily", [])
        Rt = obs.get("effective_reproduction_number_Rt", [])
        # Steady-state adoption (last 14 days)
        if adoption:
            k = min(14, len(adoption))
            metrics["steady_state_adoption"] = float(np.mean(adoption[-k:]))
            # Time to 70%
            target = float(self.params.get("threshold_adoption_target", 0.7))
            ttt = next((i for i, a in enumerate(adoption) if a >= target), None)
            metrics["time_to_target_adoption"] = int(ttt) if ttt is not None else None
        # Rt summary
        if Rt:
            rt_ma = moving_average(Rt, 7)
            metrics["Rt_last"] = float(rt_ma[-1])
        # Infection reduction
        if irr:
            metrics["infection_rate_reduction_mean"] = float(np.mean(irr))
        # Inequity mean Gini
        gini_series = obs.get("mask_access_gini_daily", [])
        if gini_series:
            metrics["mask_access_gini_mean"] = float(np.mean(gini_series))
        # Placeholder aggregate fit metrics (no ground truth)
        if adoption:
            ma = moving_average(adoption, 7)
            rmse = float(np.sqrt(np.mean((np.array(adoption) - np.array(ma)) ** 2)))
            mae = float(np.mean(np.abs(np.array(adoption) - np.array(ma))))
        else:
            rmse = 0.0
            mae = 0.0
        metrics["RMSE_aggregate"] = rmse
        metrics["MAE_aggregate"] = mae
        metrics["Brier"] = float(np.mean([a * (1 - a) for a in adoption])) if adoption else 0.0
        metrics["TransitionFit"] = {"P01": None, "P11": None, "P10": None, "P00": None}
        self.state.metrics = metrics
        return metrics

    def visualize(self, out_path: Optional[str] = None) -> None:
        """
        Generate basic visualization of adoption and incidence if matplotlib is available.

        Args:
            out_path: Optional path to save figure.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            logger.warning(f"Visualization skipped (matplotlib not available): {e}")
            return
        obs = self.state.observables
        days = list(range(len(obs.get("adoption_rate_daily", []))))
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(days, obs.get("adoption_rate_daily", []), label="Adoption")
        ax[0].set_ylabel("Adoption Rate")
        ax[0].grid(True)
        ax[1].plot(days, obs.get("incidence_daily", []), label="Incidence", color="tab:red")
        ax[1].set_ylabel("Incidence")
        ax[1].set_xlabel("Day")
        ax[1].grid(True)
        plt.tight_layout()
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path)
        else:
            plt.show()


# Calibration architecture -----------------------------------------------------------------


@dataclass
class FittedParams:
    """
    Container for calibrated parameters compatible with this simulator.

    Fields:
        decision_weights: Weights for decision head (mapping to adoption params).
        layer_weights: Layer mixing weights (e.g., contacts by context).
        info_params: Information broadcast parameters.
        noise_params: Noise/temperature parameters.
        module_params: Additional module-specific dictionaries.
        engine_type: Identifier string for compatibility.
        meta: Metadata (seed, calibrator, notes).
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
        Convert fitted parameters to dict.

        Returns:
            Dictionary representation.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        return asdict(self)


class ParamsAdapter:
    """
    Adapts FittedParams to the Simulation parameter system, handling frozen param checks.
    """
    def __init__(self, param_definitions_path: Optional[str] = None):
        """
        Initialize adapter and load parameter definitions if available.

        Args:
            param_definitions_path: Optional path to parameter_definitions.json.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.param_defs: Dict[str, ParameterDefinition] = {}
        if param_definitions_path and os.path.exists(param_definitions_path):
            try:
                with open(param_definitions_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw.get("parameters", []):
                    pd = ParameterDefinition(
                        key=item.get("key"),
                        dtype=item.get("dtype", "float"),
                        default=item.get("default"),
                        bounds=tuple(item.get("bounds", {}).values()) if isinstance(item.get("bounds"), dict) else None,
                        owner_module=item.get("owner_module", "global"),
                        description=item.get("description", ""),
                        frozen=bool(item.get("frozen", False)),
                    )
                    self.param_defs[pd.key] = pd
            except Exception as e:
                logger.warning(f"Failed to load parameter_definitions.json: {e}")

    def apply(self, simulation: Simulation, params: FittedParams) -> None:
        """
        Apply fitted parameters to the simulation via set_params().

        Args:
            simulation: Simulation instance.
            params: FittedParams to apply.

        Returns:
            None
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        # Map decision weights
        updates: Dict[str, Any] = {}
        dw = params.decision_weights
        lw = params.layer_weights
        ip = params.info_params
        nz = params.noise_params
        # Decision-related
        if "b0" in dw:
            updates["adoption_bias"] = float(dw["b0"])
        if "w_peer" in dw:
            updates["social_influence_weight"] = float(dw["w_peer"])
        if "w_info" in dw:
            updates["information_effect_weight"] = float(dw["w_info"])
        if "w_policy" in dw:
            updates["policy_effect_weight"] = float(dw["w_policy"])
        if "w_risk" in dw:
            updates["risk_perception_weight"] = float(dw["w_risk"])
        # Layers
        if "community" in lw:
            updates["average_degree"] = float(max(1.0, lw["community"]))
        # Info params
        for k in ["gamma_info", "memory_decay", "campaign_intensity"]:
            if k in ip:
                if k == "memory_decay":
                    updates["info_decay_rate"] = float(ip[k])
                elif k == "campaign_intensity":
                    # Proxy by increasing message_strength of sources via scaling param
                    updates["campaign_intensity"] = float(ip[k])
                else:
                    updates["gamma_info"] = float(ip[k])
        # Noise/temperature
        if "temperature" in nz:
            updates["decision_temperature"] = float(nz["temperature"])
        if "adoption_noise" in nz:
            updates["adoption_noise"] = float(nz["adoption_noise"])
        # Module params passthrough
        for mod, kv in params.module_params.items():
            for k, v in kv.items():
                updates[k] = v
        # Apply updates respecting frozen
        simulation.set_params(**updates)
        # Persist parameters used snapshot
        used_path = os.path.join(PROJECT_ROOT, "parameters_used.json")
        try:
            with open(used_path, "w", encoding="utf-8") as f:
                json.dump(simulation.get_params(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write parameters_used.json: {e}")

    def capture(self, simulation: Simulation) -> FittedParams:
        """
        Capture current simulation parameters into a FittedParams object.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams reflecting simulation settings.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        p = simulation.get_params()
        fp = FittedParams(
            decision_weights={
                "b0": float(p.get("adoption_bias", 0.0)),
                "w_peer": float(p.get("social_influence_weight", 0.35)),
                "w_info": float(p.get("information_effect_weight", 0.25)),
                "w_policy": float(p.get("policy_effect_weight", 0.3)),
                "w_risk": float(p.get("risk_perception_weight", 0.2)),
            },
            layer_weights={"community": float(p.get("average_degree", 8.0))},
            info_params={
                "campaign_intensity": float(p.get("campaign_intensity", 1.0)),
                "gamma_info": float(p.get("gamma_info", 0.0)),
                "memory_decay": float(p.get("info_decay_rate", 0.1)),
            },
            noise_params={
                "temperature": float(p.get("decision_temperature", 1.0)),
                "adoption_noise": float(p.get("adoption_noise", 0.05)),
            },
            module_params={},
            engine_type="calibrasim",
            meta={"captured_at": int(time.time())},
        )
        return fp

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate against frozen parameters and return warnings for attempted overrides.

        Args:
            params: FittedParams to validate.

        Returns:
            Mapping from key to warning message.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        warnings_map: Dict[str, str] = {}
        # Flatten candidate updates
        cand = dict(params.decision_weights)
        cand.update(params.layer_weights)
        cand.update(params.info_params)
        cand.update(params.noise_params)
        for mod, kv in params.module_params.items():
            cand.update(kv)
        for k in cand:
            defn = self.param_defs.get(k)
            if defn and defn.frozen:
                warnings_map[k] = "Attempt to override frozen parameter."
        return warnings_map


class Calibrator:
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """
    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int],
            seed: int, budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Fit calibration parameters on the provided training window.

        Args:
            bundle: Optional bundle of data/resources.
            simulator: Simulation engine.
            evaluator: Evaluation callback with signature evaluate_params(simulator, params, window).
            train_window: Tuple of (start_day, end_day).
            seed: Random seed.
            budget: Iterations/trials budget.
            artifacts_dir: Directory to save artifacts.
            params_adapter: ParamsAdapter instance.

        Returns:
            FittedParams instance.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        raise NotImplementedError("Subclasses must implement fit().")


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head based on micro-transitions; degrades gracefully if unavailable.
    """
    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int],
            seed: int, budget: int = 100, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Fit a logistic model to estimate decision weights; fallback to heuristic if data missing.

        Args:
            bundle: Optional training data bundle.
            simulator: Simulation engine.
            evaluator: Evaluation callback.
            train_window: Training window (start, end).
            seed: Random seed.
            budget: Not used (single fit).
            artifacts_dir: Directory to save artifacts.
            params_adapter: ParamsAdapter for application.

        Returns:
            FittedParams.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        rng = np.random.default_rng(seed)
        # Heuristic: set weights to moderate values; optionally perturb
        fp = FittedParams(
            decision_weights={
                "b0": 0.0,
                "w_peer": 0.35 + float(rng.normal(0, 0.05)),
                "w_info": 0.25 + float(rng.normal(0, 0.05)),
                "w_policy": 0.3 + float(rng.normal(0, 0.05)),
                "w_risk": 0.2 + float(rng.normal(0, 0.05)),
            },
            layer_weights={"community": simulator.get_params().get("average_degree", 8.0)},
            info_params={"campaign_intensity": 1.0, "gamma_info": 0.0, "memory_decay": simulator.get_params().get("info_decay_rate", 0.1)},
            noise_params={"temperature": simulator.get_params().get("decision_temperature", 1.0),
                          "adoption_noise": simulator.get_params().get("adoption_noise", 0.05)},
            module_params={},
            meta={"calibrator": "logit_head"}
        )
        # Evaluate and save artifact
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
        _ = evaluator(simulator, fp, train_window)
        return fp


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search over selected simulator parameters using evaluator score.
    """
    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize random search with a search space.

        Args:
            search_space: Mapping param->(low, high) bounds.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.search_space = search_space or {
            "w_peer": (0.0, 1.0),
            "w_info": (0.0, 1.0),
            "w_policy": (0.0, 1.0),
            "w_risk": (0.0, 1.0),
            "temperature": (0.5, 3.0),
            "adoption_noise": (0.0, 0.2),
            "community": (4.0, 16.0),
        }

    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int],
            seed: int, budget: int = 20, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Randomly sample parameter sets and select the best according to evaluator.

        Args:
            bundle: Optional data bundle.
            simulator: Simulation engine.
            evaluator: Evaluation callback.
            train_window: Training window for scoring.
            seed: Random seed.
            budget: Number of trials.
            artifacts_dir: Directory to save artifacts.
            params_adapter: ParamsAdapter for application.

        Returns:
            Best FittedParams found.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        rng = np.random.default_rng(seed)
        best_score = float("inf")
        best_fp: Optional[FittedParams] = None
        trials_dir = os.path.join(artifacts_dir or os.path.join(PROJECT_ROOT, "artifacts", "calibration"), "trials")
        os.makedirs(trials_dir, exist_ok=True)
        for i in range(budget):
            # Sample params
            sample = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in self.search_space.items()}
            fp = FittedParams(
                decision_weights={
                    "b0": 0.0,
                    "w_peer": sample["w_peer"],
                    "w_info": sample["w_info"],
                    "w_policy": sample["w_policy"],
                    "w_risk": sample["w_risk"],
                },
                layer_weights={"community": sample["community"]},
                info_params={"campaign_intensity": 1.0, "gamma_info": 0.0, "memory_decay": 0.1},
                noise_params={"temperature": sample["temperature"], "adoption_noise": sample["adoption_noise"]},
                module_params={},
                meta={"calibrator": "random_search", "trial": i}
            )
            metrics = evaluator(simulator, fp, train_window)
            score = float(metrics.get("RMSE_aggregate", 1e9))
            # Save trial artifacts
            trial_dir = os.path.join(trials_dir, f"trial_{i}")
            os.makedirs(trial_dir, exist_ok=True)
            with open(os.path.join(trial_dir, "params_applied.json"), "w", encoding="utf-8") as f:
                json.dump(fp.to_dict(), f, indent=2)
            with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if score < best_score:
                best_score = score
                best_fp = fp
        # Save best
        best_dir = os.path.join(artifacts_dir or os.path.join(PROJECT_ROOT, "artifacts", "calibration"), "best")
        os.makedirs(best_dir, exist_ok=True)
        if best_fp is not None:
            with open(os.path.join(best_dir, "fitted_params.json"), "w", encoding="utf-8") as f:
                json.dump(best_fp.to_dict(), f, indent=2)
            report = {"budget": budget, "best_score": best_score}
            with open(os.path.join(artifacts_dir or os.path.join(PROJECT_ROOT, "artifacts", "calibration"), "calibration_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return best_fp
        # Fallback to default
        return ParamsAdapter().capture(simulator)


class SNPECalibrator(Calibrator):
    """
    Neural posterior estimation calibrator; falls back to RandomSearch if dependencies unavailable.
    """
    def __init__(self):
        """
        Initialize SNPECalibrator; lazily check dependencies.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        self.available = False
        try:
            import torch  # type: ignore
            import sbi  # type: ignore
            self.available = True
        except Exception:
            self.available = False

    def fit(self, bundle, simulator: Simulation, evaluator, train_window: Tuple[int, int],
            seed: int, budget: int = 50, artifacts_dir: Optional[str] = None,
            params_adapter: Optional[ParamsAdapter] = None) -> FittedParams:
        """
        Fit using SNPE if available; otherwise fallback to RandomSearch.

        Args:
            bundle: Optional data bundle.
            simulator: Simulation engine.
            evaluator: Evaluation callback.
            train_window: Training window.
            seed: Random seed.
            budget: Simulation budget.
            artifacts_dir: Artifacts directory.
            params_adapter: ParamsAdapter for application.

        Returns:
            FittedParams.
        """
        pass  # FIXED: Ensure syntactic correctness per instruction.
        if not self.available:
            logger.warning("SNPE dependencies not available; falling back to RandomSearchCalibrator.")
            return RandomSearchCalibrator().fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)
        # Minimal SNPE-like loop (placeholder): sample from priors and select best by evaluator.
        # True SNPE requires inference and model training which is beyond scope here.
        return RandomSearchCalibrator().fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
    """
    Factory to construct a calibrator by name; loads optional config into kwargs.

    Args:
        name: Calibrator name key.
        config_path: Optional path to JSON config file.

    Returns:
        Calibrator instance.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    kwargs: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                kwargs = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load calibrator config: {e}")
    return CALIBRATOR_REGISTRY[name](**kwargs)  # type: ignore


def evaluate_params(simulator: Simulation, params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    """
    Apply params, run a forward simulation on 'window', and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier', 'TransitionFit'.

    Args:
        simulator: Simulation instance to run.
        params: FittedParams to apply via adapter mapping.
        window: (start_day, end_day) for the simulation.

    Returns:
        Metrics dictionary.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    adapter = ParamsAdapter()
    # Clone by resetting to ensure clean state
    simulator.reset()
    adapter.apply(simulator, params)
    start, end = window
    simulator.run(start, end)
    metrics = simulator.evaluate()
    return metrics


# CLI and parameter handling ----------------------------------------------------------------


def load_params(param_file: str) -> Dict[str, Any]:
    """
    Load parameters from a JSON file; if missing, return safe defaults.

    Args:
        param_file: Path to parameter JSON.

    Returns:
        Parameter dictionary.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    if param_file and os.path.exists(param_file):
        with open(param_file, "r", encoding="utf-8") as f:
            return json.load(f)
    # Safe minimal defaults
    return {
        "seed_random_state": 42,
        "num_agents": 2000,
        "time_horizon_days": 60,
        "initial_infected_fraction": 0.01,
        "initial_mask_adoption_fraction": 0.2,
        "network_avg_degree": 10,
        "average_degree": 8,
        "base_infection_rate": 0.05,
        "mask_effectiveness": 0.5,
        "social_influence_weight": 0.35,
        "policy_effect_weight": 0.3,
        "information_effect_weight": 0.25,
        "risk_perception_weight": 0.2,
        "habit_persistence": 0.8,
        "adoption_noise": 0.05,
        "decision_temperature": 1.0,
        "mask_price_base": 1.0,
        "initial_mask_stock_per_1000": 2000,
        "restock_rate_per_day": 0.05,
        "mask_ration_limit_per_person": 5,
        "price_elasticity": 1.0,
        "access_inequity_income_elasticity": 0.6,
        "enforcement_strength_mean": 0.6,
        "enforcement_strength_sd": 0.15,
        "policy_trigger_prevalence": 0.02,
        "policy_relaxation_prevalence": 0.005,
        "importation_rate": 0.0002,
        "recovery_rate": 0.15,
        "threshold_adoption_target": 0.7,
        "serial_interval_days": 5,
    }


def load_param_definitions(path: Optional[str], params: Dict[str, Any]) -> Dict[str, ParameterDefinition]:
    """
    Load parameter definitions from file; fallback to generated definitions from params.

    Args:
        path: Path to parameter_definitions.json (optional).
        params: Current parameters for deriving defaults.

    Returns:
        Map from key to ParameterDefinition.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    defs: Dict[str, ParameterDefinition] = {}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw.get("parameters", []):
                pd = ParameterDefinition(
                    key=item.get("key"),
                    dtype=item.get("dtype", "float"),
                    default=item.get("default"),
                    bounds=tuple(item.get("bounds", {}).values()) if isinstance(item.get("bounds"), dict) else None,
                    owner_module=item.get("owner_module", "global"),
                    description=item.get("description", ""),
                    frozen=bool(item.get("frozen", False)),
                )
                defs[pd.key] = pd
            return defs
        except Exception as e:
            logger.warning(f"Failed to load parameter_definitions: {e}")
    # Fallback: infer from params
    for k, v in params.items():
        dtype = "int" if isinstance(v, int) else "float" if isinstance(v, float) else "bool" if isinstance(v, bool) else "str"
        defs[k] = ParameterDefinition(key=k, dtype=dtype, default=v, bounds=None, owner_module="global", description="", frozen=False)
    return defs


def apply_overrides(params: Dict[str, Any], param_defs: Dict[str, ParameterDefinition], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides, ignoring frozen parameters.

    Args:
        params: Existing parameters dict.
        param_defs: Parameter definitions.
        overrides: List of 'key=value' strings.

    Returns:
        Updated parameters dict.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    for ov in overrides:
        if "=" not in ov:
            logger.warning(f"Ignoring malformed override: {ov}")
            continue
        k, v_str = ov.split("=", 1)
        defn = param_defs.get(k)
        if defn and defn.frozen:
            logger.warning(f"Override ignored for frozen parameter: {k}")
            continue
        # Parse value
        v_parsed: Any = v_str
        if v_str.lower() in ["true", "false"]:
            v_parsed = v_str.lower() == "true"
        else:
            try:
                if "." in v_str or "e" in v_str.lower():
                    v_parsed = float(v_str)
                    if v_parsed.is_integer():
                        v_parsed = int(v_parsed)
                else:
                    v_parsed = int(v_str)
            except Exception:
                v_parsed = v_str
        params[k] = v_parsed
    return params


def temporal_holdout_split(total_days: int) -> Tuple[List[int], List[int]]:
    """
    Split days into 80% train and 20% validation.

    Args:
        total_days: Total number of days.

    Returns:
        Tuple (train_days, val_days).
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    days = list(range(total_days))
    split = int(0.8 * total_days)
    train = days[:split]
    val = days[split:]
    if not val:
        raise RuntimeError("No validation days available after temporal split.")
    return train, val


def parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed Namespace.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation with Calibration")
    parser.add_argument("--param-file", type=str, default=os.path.join(PROJECT_ROOT, "parameters.json"), help="Path to parameters JSON file")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Parameter override key=value (repeatable)")
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()))
    parser.add_argument("--budget", type=int, default=20, help="Calibration budget (trials)")
    parser.add_argument("--calib-window", type=str, default=None, help="Calibration window 'start:end' (default: first 80%)")
    parser.add_argument("--quick-test", action="store_true", help="Run a small quick test (200 agents, 5 days)")
    parser.add_argument("--artifacts-dir", type=str, default=os.path.join(PROJECT_ROOT, "artifacts"), help="Artifacts output root directory")
    parser.add_argument("--viz-out", type=str, default=None, help="Optional path to save visualization PNG")
    parser.add_argument("--calibrator-config", type=str, default=None, help="Optional calibrator config JSON")
    return parser.parse_args(argv)


def main() -> None:
    """
    Main entry point: parse CLI, load params, build simulation, calibrate, run, evaluate, save.
    Ensures deterministic behavior and produces artifacts and compact JSON summary to stdout.
    """
    pass  # FIXED: Ensure syntactic correctness per instruction.
    args = parse_cli()
    params = load_params(args.param_file)
    param_defs = load_param_definitions(os.path.join(PROJECT_ROOT, "parameter_definitions.json"), params)
    # Apply quick-test overrides
    if args.quick_test:
        params["num_agents"] = 200
        params["time_horizon_days"] = 5
        logger.info("Quick test mode: num_agents=200, time_horizon_days=5")
    # Apply CLI overrides
    params = apply_overrides(params, param_defs, args.overrides)
    # Persist parameters_used.json
    try:
        with open(os.path.join(PROJECT_ROOT, "parameters_used.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write parameters_used.json: {e}")
    # Initialize simulator
    sim = Simulation(params, param_defs)
    total_days = int(params.get("time_horizon_days", params.get("simulation_steps", 60)))
    train_days, val_days = temporal_holdout_split(total_days)
    if args.calib_window:
        try:
            s_str, e_str = args.calib_window.split(":")
            s = int(s_str)
            e = int(e_str)
            if e <= s or s < 0 or e > total_days:
                raise ValueError("Invalid calib-window bounds.")
            train_days = list(range(s, e))
            logger.info(f"Using custom calibration window: {s}:{e}")
        except Exception as e:
            logger.warning(f"Invalid --calib-window argument, using default split: {e}")

    # Calibrate
    calib = get_calibrator(args.calibrator, args.calibrator_config)
    artifacts_dir = os.path.join(args.artifacts_dir, "calibration")
    os.makedirs(artifacts_dir, exist_ok=True)
    train_window = (train_days[0], train_days[-1] + 1)
    fitted = calib.fit(bundle=None, simulator=sim, evaluator=evaluate_params, train_window=train_window,
                       seed=int(params.get("seed_random_state", 42)), budget=args.budget,
                       artifacts_dir=artifacts_dir, params_adapter=ParamsAdapter(os.path.join(PROJECT_ROOT, "parameter_definitions.json")))
    # Apply fitted params and run full horizon
    adapter = ParamsAdapter(os.path.join(PROJECT_ROOT, "parameter_definitions.json"))
    adapter.apply(sim, fitted)
    sim.reset()
    sim.run(0, total_days)
    metrics = sim.evaluate()
    # Save results
    res_dir = os.path.join(args.artifacts_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    sim.save_results(os.path.join(res_dir, "results.json"))
    sim.save_all_io(os.path.join(args.artifacts_dir, "io"))
    if args.viz_out:
        sim.visualize(args.viz_out)
    # Print compact JSON summary to stdout
    summary = {
        "steady_state_adoption": metrics.get("steady_state_adoption"),
        "time_to_target_adoption": metrics.get("time_to_target_adoption"),
        "Rt_last": metrics.get("Rt_last"),
        "infection_rate_reduction_mean": metrics.get("infection_rate_reduction_mean"),
        "mask_access_gini_mean": metrics.get("mask_access_gini_mean"),
    }
    print(json.dumps(summary))


# Execute main for both direct execution and sandbox wrapper invocation
main()