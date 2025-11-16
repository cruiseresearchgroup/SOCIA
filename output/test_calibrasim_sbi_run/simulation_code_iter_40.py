def main():
    pass

import csv
import json
import logging
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional visualization
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mask_sim")

# Path handling as per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else None
SAFE_PROJECT_ROOT = PROJECT_ROOT if PROJECT_ROOT else os.getcwd()
SAFE_DATA_DIR = DATA_DIR if DATA_DIR else SAFE_PROJECT_ROOT


def safe_load_json(maybe_path: str, strict: bool = True) -> Dict[str, Any]:
    """
    Load JSON from a file path if it exists; otherwise parse the string as JSON.
    Ensures robust loading from either a file system path or a JSON string.
    """
    try:
        if os.path.isfile(maybe_path):
            with open(maybe_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return json.loads(maybe_path)
    except Exception as e:
        if strict:
            raise
        logger.warning(f"Failed to load JSON: {e}")
        return {}
    pass


DEFAULT_CONFIG: Dict[str, Any] = {
    "population_size": 200,
    "days": 60,
    "max_time_steps": None,
    "initial_adoption_rate": 0.1,
    "peer_influence": 0.3,
    "base_influence_weight": 0.3,
    "workplace_influence_weight": 0.2,
    "household_influence_weight": 0.1,
    "policy_signal": 0.0,
    "seed": 42,
    "mandate_enabled": False,
    "mandate_start_day": 5,
    "mandate_end_day": 8,
    "enforcement_probability": 0.1,
    "fine_amount": 50.0,
    "fine_effect_on_future_compliance": 0.3,
    "campaign_intensity": 0.0,
    "risk_perception_baseline": 0.2,
    "risk_perception_update_rate": 0.1,
    "case_prevalence_effect_on_risk_perception": 0.4,
    # FIXED: Added risk_perception_weight per spec mapping.
    "risk_perception_weight": 0.3,
    "perceived_efficacy_update_rate": 0.08,
    "trust_update_rate": 0.05,
    "forgetting_rate": 0.01,
    "adoption_threshold_noise": 0.1,
    "affordability_sensitivity": 0.4,
    "mask_price": 2.0,
    "supply_initial_stock": 300,
    "daily_supply_restock": 60,
    "retailer_count": 3,
    "with_supply": False,
    # FIXED: Added explicit supply model switch as per feedback.
    "with_supply_model": "retailer",  # 'retailer' or 'central'
    # FIXED: Added supply shock probability for retailers.
    "supply_shock_probability": 0.01,
    "with_epi": False,
    "average_degree": 4,
    "social_network_type": "small_world",
    "rewiring_probability": 0.1,
    "homophily_by_attitude": 0.0,
    "time_step_unit": "day",
    # Household/Workplace distributions
    # FIXED: Added distributional parameters per feedback.
    "household_size_distribution": [1, 2, 3, 4],  # simplistic sizes; equal weight
    "workplace_size_distribution": [5, 10, 20, 50],  # sample sizes; equal weight
    "workplace_count": 5,
    "workplace_policy_strictness": 0.0,
    "workplace_enforcement_capacity": 0.0,
    "workplace_adoption_visibility": 0.5,
    "visibility_effect_strength": 0.3,
    "misinformation_prevalence": 0.1,
    "affordability_threshold": 0.05,
    "price_elasticity_of_demand": -0.8,
    "location_mask_requirement_fraction": 0.5,
    "location_counts": {"stores": 10, "transit": 5, "public_venues": 8},
    "consistency_window": 3,
    "evaluation_metrics": [
        "adoption_rate_over_time",
        "time_to_50_percent_adoption",
        "peak_adoption",
        "sustained_adoption_duration",
        "subgroup_disparity_index",
        "policy_enforcement_events",
        "mask_inventory_outage_days",
        "masks_sold",
        "campaign_cost_effectiveness",
        "effective_reproduction_number_Rt",
        "infections_averted",
        # Optional calibration metrics if observed series provided:
        "calibration_rmse",
        "calibration_mae",
        "calibration_corr",
        # FIXED: Added spec metrics.
        "average_adoption_rate",
        "time_to_reach_threshold",
        "sustained_adoption_days",
        "policy_compliance_rate",
        "adoption_variance",
        "misinformation_impact",
        "cumulative_person_days_masked",
    ],
    "habit_formation_rate": 0.1,
    "habit_decay_rate": 0.02,
    "activity_pattern": "weekly_peaks",
    "activity_weekday_peak": [2, 3, 4],
    "activity_peak_multiplier": 1.2,
    "activity_base": 0.8,
    "observed_series_path": None,
    "calibration_rmse_threshold": 0.05,
    "target_adoption_threshold": 0.7,
    # FIXED: Added explicit adoption_threshold for evaluate mapping.
    "adoption_threshold": 0.7,
    # Epi coupling params
    "disease_R0": 2.5,
    "mask_efficacy_source_control": 0.5,
    "mask_efficacy_wearer_protection": 0.3,
    # Media message effect size for efficacy update
    "message_effect_size": 0.2,
    # Optional k-hop visibility of compliance
    "observation_radius_hops": 1,
    # FIXED: Added stochastic mandate probability per day.
    "mandate_probability_per_day": 0.0,
    # FIXED: Added misinformation_reach for broadcast events.
    "misinformation_reach": 0.3,
    # FIXED: Added fatigue parameters per spec.
    "fatigue_rate": 0.01,
    "recovery_from_fatigue_rate": 0.005,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a float between lo and hi.
    Ensures all probabilities and ratios remain bounded.
    """
    return max(lo, min(hi, x))
    pass


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """
    Compute Pearson correlation between two lists.
    Returns 0.0 if insufficient variance or empty inputs.
    """
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den_x = math.sqrt(sum((xs[i] - x_mean) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ys[i] - y_mean) ** 2 for i in range(n)))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)
    pass


def _gini(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of non-negative values.
    Returns a value in [0,1].
    """
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cum = 0.0
    for i, x in enumerate(sorted_vals, start=1):
        cum += i * x
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cum) / (n * total) - (n + 1) / n
    return _clamp(gini, 0.0, 1.0)
    pass


def _sample_from_distribution(rng: random.Random, distribution: Any, min_val: int = 1, max_val: int = 100) -> int:
    """
    Sample an integer size from a provided distribution spec.
    Supported formats:
    - list of sizes (equal probability)
    - dict {'sizes': [...], 'weights': [...]} with same length
    - dict {'weights_by_size': {size: weight, ...}}
    Falls back to uniform in [min_val, max_val] if unrecognized.
    """
    try:
        if isinstance(distribution, list):
            val = int(rng.choice(distribution))
            return max(min_val, min(max_val, val))
        if isinstance(distribution, dict):
            if "sizes" in distribution and "weights" in distribution:
                sizes = list(map(int, distribution["sizes"]))
                weights = list(map(float, distribution["weights"]))
                val = rng.choices(sizes, weights=weights, k=1)[0]
                return max(min_val, min(max_val, int(val)))
            if "weights_by_size" in distribution and isinstance(distribution["weights_by_size"], dict):
                items = list(distribution["weights_by_size"].items())
                sizes = [int(k) for k, _ in items]
                weights = [float(v) for _, v in items]
                val = rng.choices(sizes, weights=weights, k=1)[0]
                return max(min_val, min(max_val, int(val)))
    except Exception:
        pass
    return int(rng.randint(min_val, max_val))
    pass


@dataclass
class MediaChannel:
    """
    Minimal media channel that can broadcast a signal affecting risk perception and trust.
    Includes misinformation parameters and a simple salience level.
    """
    id: int = 0
    misinformation_rate: float = 0.0
    salience: float = 0.0

    def broadcast(self, base_signal: float) -> Tuple[float, float]:
        """
        Compute deltas to risk perception and trust based on media properties.
        Returns (delta_risk, delta_trust).
        """
        delta_risk = 0.02 * self.salience - 0.03 * self.misinformation_rate
        delta_trust = 0.01 * self.salience - 0.02 * self.misinformation_rate
        return delta_risk, delta_trust
        pass
    pass


@dataclass
class Location:
    """
    Represents a public location with mask policies, enforcement, and visibility.
    Tracks simple parameters relevant for enforcement and observation.
    """
    id: int = 0
    loc_type: str = "stores"
    capacity: int = 50
    crowding_level: float = 0.5
    # FIXED: Spec-aligned naming (mask_required, enforcement_level, contact_rate) with backward compatibility.
    mask_required: bool = False
    mandate_policy: Optional[str] = None
    enforcement_level: float = 0.1
    contact_rate: float = 0.5
    # Backward-compatibility alias fields
    mask_requirement: bool = field(default=False, repr=False)
    enforcement_strictness: float = field(default=0.1, repr=False)
    visibility_of_mask_use: float = 0.6
    signage_strength: float = 0.4

    def __post_init__(self) -> None:
        """
        Post-initialization to honor backward compatibility for field names.
        """
        # FIXED: Align aliases for mask requirement and enforcement strictness.
        if self.mask_requirement and not self.mask_required:
            self.mask_required = self.mask_requirement
        if self.enforcement_strictness is not None and self.enforcement_level == 0.1:
            self.enforcement_level = self.enforcement_strictness
        return
        pass

    def enforce_policy(self, compliant: bool, rng: random.Random) -> bool:
        """
        Apply enforcement when non-compliant; returns True if enforcement triggers action.
        This is a simplified Bernoulli process based on enforcement_level.
        """
        if compliant:
            return False
        if rng.random() < self.enforcement_level:
            return True
        return False
        pass

    def record_compliance(self, is_compliant: bool) -> None:
        """
        Placeholder to record compliance at this location, possibly to richer logs.
        Currently a no-op.
        """
        return
        pass

    def allow_or_deny_entry(self, compliant: bool, rng: random.Random) -> bool:
        """
        Allow or deny entry based on policy and compliance; simplified to allow entry.
        Returns always True for simplicity.
        """
        return True
        pass
    pass


@dataclass
class Individual:
    """
    Represents an individual agent in the simulation with behavioral attributes.
    Holds state related to beliefs, compliance, inventory, and social network.
    """
    id: int = 0
    age: int = 30
    income: float = 50000.0
    education_level: int = 2
    health_risk_factor: float = 1.0
    risk_perception: float = 0.2
    trust_in_authority: float = 0.5
    social_norm_sensitivity: float = 0.5
    perceived_mask_efficacy: float = 0.5
    cost_sensitivity: float = 0.5
    compliance_propensity: float = 0.5
    influenceability: float = 0.5
    # FIXED: Added spec-aligned attributes.
    risk_aversion: float = 0.5
    misinformation_susceptibility: float = 0.5
    social_influence_susceptibility: float = 0.5
    fatigue_level: float = 0.0
    current_mask_use: bool = False
    mask_inventory: int = 0
    fines_count: int = 0
    cumulative_spend: float = 0.0
    household_id: Optional[int] = None
    workplace_id: Optional[int] = None
    network_neighbors: List[int] = field(default_factory=list)
    habit_strength: float = 0.0
    mobility_level: float = 1.0
    misinformation_exposure: float = 0.0

    def evaluate_mask_use_decision(
        self,
        peer_share: float,
        policy_signal: float,
        price: float,
        mandate_active: bool,
        rng: random.Random,
        habit_weight: float = 0.5,
        fatigue_level: float = 0.0,
        risk_weight: float = 0.3,
        peer_weight: float = 0.3,
    ) -> bool:
        """
        Compute the probability to adopt mask use and sample new decision.
        Uses a logistic transformation of utility driven by social norms, risk, policy,
        habit, and cost considerations. Includes fatigue penalty and configurable weights.
        """
        # FIXED: Respect risk_weight and peer_weight with susceptibility multipliers.
        peer_util = peer_weight * self.social_norm_sensitivity * peer_share * (0.5 + 0.5 * self.social_influence_susceptibility)
        risk_util = risk_weight * self.perceived_mask_efficacy * _clamp(self.risk_perception, 0.0, 1.0) * (0.5 + 0.5 * self.risk_aversion)
        policy_util = self.trust_in_authority * policy_signal
        cost_penalty = self.cost_sensitivity * (price / max(0.1, price + 1.0))
        mandate_boost = self.compliance_propensity * (1.0 if mandate_active else 0.0)
        habit_boost = habit_weight * _clamp(self.habit_strength, 0.0, 1.0)
        fatigue_penalty = 0.4 * _clamp(fatigue_level, 0.0, 1.0)
        util = (
            0.45 * peer_util
            + 0.25 * policy_util
            + 0.5 * risk_util
            + 0.35 * mandate_boost
            + 0.3 * habit_boost
            - 0.5 * cost_penalty
            - fatigue_penalty
        )
        util = _clamp(util, -5.0, 5.0)
        noise = rng.uniform(-0.1, 0.1)
        prob = 1.0 / (1.0 + math.exp(-(util + noise)))
        if self.mask_inventory <= 0:
            prob *= 0.3
        decision = rng.random() < prob
        return decision
        pass

    def update_beliefs_from_social_influence(self, peer_share: float, update_rate: float) -> None:
        """
        Update perceived efficacy and risk perception based on peer share.
        Implements a bounded adjustment toward observed peer behavior.
        """
        self.perceived_mask_efficacy = _clamp(
            self.perceived_mask_efficacy + update_rate * (peer_share - self.perceived_mask_efficacy), 0.0, 1.0
        )
        self.risk_perception = _clamp(
            self.risk_perception + 0.5 * update_rate * (peer_share - self.risk_perception), 0.0, 1.0
        )
        return
        pass

    def respond_to_policies_and_campaigns(self, campaign_intensity: float) -> None:
        """
        Adjust trust and efficacy slightly in response to campaigns.
        Modulated by misinformation exposure (lower exposure increases update).
        """
        scale = _clamp(1.0 - self.misinformation_exposure, 0.0, 1.0)
        self.trust_in_authority = _clamp(self.trust_in_authority + scale * 0.05 * campaign_intensity, 0.0, 1.0)
        self.perceived_mask_efficacy = _clamp(self.perceived_mask_efficacy + scale * 0.04 * campaign_intensity, 0.0, 1.0)
        return
        pass

    def purchase_masks_from_retailer(
        self,
        price: float,
        inventory: Dict[str, Any],
        rng: random.Random,
        affordability_threshold: float = 0.05,
        price_elasticity: float = -0.8,
    ) -> int:
        """
        Attempt to purchase masks with affordability and price elasticity and update inventory.
        Returns the quantity purchased (0 or 1 in this simple model).
        """
        if inventory.get("stock", 0) <= 0:
            return 0
        income_daily = max(1.0, self.income / 365.0)
        if (price / income_daily) > affordability_threshold:
            return 0
        base_p = _clamp(0.6 * (1.0 - self.cost_sensitivity), 0.0, 1.0)
        price_factor = _clamp((1.0 + price) ** price_elasticity, 0.0, 1.0)
        buy_prob = _clamp(base_p * price_factor, 0.0, 1.0)
        desired_qty = 1 if rng.random() < buy_prob else 0
        qty = min(desired_qty, inventory.get("stock", 0))
        inventory["stock"] -= qty
        self.mask_inventory += qty
        self.cumulative_spend += qty * price
        return qty
        pass

    def share_opinion_with_neighbors(self) -> None:
        """
        Placeholder for opinion dynamics where individuals may share pro/anti-mask messages.
        No-op in this simplified version. Actual information sharing is handled in Simulation.step for efficiency.
        """
        return
        pass

    def attend_work_or_school(self) -> None:
        """
        Placeholder for local policy effect at workplaces/schools.
        Could be used to adjust mask usage propensity based on local policies.
        """
        return
        pass

    def consume_media_messages(self, intensity: float, misinformation_rate: float = 0.0) -> None:
        """
        Adjust perceptions due to media messages, modulated by personal misinformation exposure.
        Affects risk perception and trust in authorities.
        """
        scale = _clamp(1.0 - self.misinformation_exposure, 0.0, 1.0)
        self.risk_perception = _clamp(
            self.risk_perception + scale * (0.02 * intensity - 0.03 * misinformation_rate), 0.0, 1.0
        )
        self.trust_in_authority = _clamp(
            self.trust_in_authority + scale * (0.01 * intensity - 0.02 * misinformation_rate), 0.0, 1.0
        )
        return
        pass

    def comply_with_mandate(self, enforcement_strength: float, location_required: bool, rng: random.Random) -> bool:
        """
        Compute whether the individual will comply with a mandate in a given context.
        Returns True if they decide to comply given trust, compliance propensity, and enforcement strength.
        """
        if not location_required:
            return False
        base = 0.5 * self.compliance_propensity + 0.5 * self.trust_in_authority
        p = _clamp(base + 0.5 * enforcement_strength, 0.0, 1.0)
        return rng.random() < p
        pass
    pass


@dataclass
class Household:
    """
    Represents a household for intra-household dynamics and mask sharing.
    Encodes norms and shared stock dynamics.
    """
    id: int = 0
    member_ids: List[int] = field(default_factory=list)
    household_income: float = 0.0
    household_norms: float = 0.5
    shared_mask_stock: int = 0

    def intra_household_influence(self, individuals: List[Individual], weight: float = 0.1) -> None:
        """
        Adjust household norms to average of member mask use and nudge members toward the new norm.
        """
        if not self.member_ids:
            return
        avg_use = sum(1.0 if individuals[i].current_mask_use else 0.0 for i in self.member_ids) / len(self.member_ids)
        self.household_norms = _clamp(0.7 * self.household_norms + 0.3 * avg_use, 0.0, 1.0)
        for i in self.member_ids:
            ind = individuals[i]
            ind.perceived_mask_efficacy = _clamp(
                ind.perceived_mask_efficacy + weight * (self.household_norms - ind.perceived_mask_efficacy),
                0.0,
                1.0,
            )
        return
        pass

    def share_masks_among_members(self, individuals: List[Individual]) -> None:
        """
        Share masks if some members lack masks, using a simple donor-recipient exchange within the household.
        """
        total = sum(individuals[i].mask_inventory for i in self.member_ids) + self.shared_mask_stock
        if total <= 0:
            return
        need = [i for i in self.member_ids if individuals[i].mask_inventory == 0]
        have = [i for i in self.member_ids if individuals[i].mask_inventory > 1]
        for needy in need:
            if have:
                donor = have.pop()
                individuals[donor].mask_inventory -= 1
                individuals[needy].mask_inventory += 1
        return
        pass
    pass


@dataclass
class WorkplaceSchool:
    """
    Represents a workplace or school with a local policy governing mask use and enforcement.
    Enforcement impacts individuals' compliance propensity.
    """
    id: int = 0
    size: int = 0
    policy_strictness: float = 0.0
    enforcement_capacity: float = 0.0
    adoption_visibility: float = 0.5

    def set_local_mask_policy(self, strictness: float) -> None:
        """
        Set the local policy strictness factor in [0,1], determining the strength of local enforcement and norms.
        """
        self.policy_strictness = _clamp(strictness, 0.0, 1.0)
        return
        pass

    def enforce_policy_on_attendees(self, individuals: List[Individual], rng: random.Random) -> int:
        """
        Enforce policy on attendees; returns number of enforcement actions.
        Simplified loop uses enforcement_capacity scaled passes to probabilistically enforce.
        """
        actions = 0
        if self.policy_strictness <= 0.0 or self.size <= 0:
            return actions
        for _ in range(int(self.enforcement_capacity * max(1, self.size))):
            if rng.random() < 0.02:
                actions += 1
        return actions
        pass

    def communicate_guidelines(self, individuals: List[Individual]) -> None:
        """
        Communicate guidelines to attendees (placeholder).
        Could increase perceived efficacy or trust slightly among members.
        """
        return
        pass
    pass


@dataclass
class Retailer:
    """
    Simplified retailer with inventory and pricing.
    Handles sales and inventory restocking dynamics with endogenous price adjustment.
    """
    id: int = 0
    inventory: int = 0
    price: float = 2.0
    restock_rate: int = 0
    supply_allocation_quota: int = 0

    def sell_masks_to_individuals(self, individuals: List[Individual], rng: random.Random) -> int:
        """
        Sell masks to customers; returns total sold.
        Not used directly in main loop where individuals purchase from selected retailers.
        """
        sold = 0
        for ind in individuals:
            if self.inventory <= 0:
                break
            if rng.random() < 0.2:
                self.inventory -= 1
                ind.mask_inventory += 1
                ind.cumulative_spend += self.price
                sold += 1
        return sold
        pass

    def restock_inventory(self) -> None:
        """
        Restock inventory by restock_rate. Used with central allocation complementarily.
        """
        self.inventory += self.restock_rate
        return
        pass

    def adjust_price_based_on_inventory(self) -> None:
        """
        Adjust price upward if low inventory, downward if abundant.
        Keeps price bounded in [0.5, 10.0].
        """
        if self.inventory <= max(1, int(self.supply_allocation_quota * 0.3)):
            self.price = min(10.0, self.price * 1.05)
        else:
            self.price = max(0.5, self.price * 0.98)
        return
        pass
    pass


@dataclass
class Government:
    """
    Represents government policy and campaign actions, including mandate control and supply allocation.
    Extended with incentive and sanction parameters for richer policy experiments.
    """
    id: int = 0
    mandate_enabled: bool = False
    mandate_status: bool = False
    mandate_start_day: int = 0
    mandate_end_day: Optional[int] = None
    enforcement_probability: float = 0.0
    fine_amount: float = 50.0
    campaign_intensity: float = 0.0
    campaign_targeting_strategy: str = "broadcast"
    budget: float = 0.0
    # FIXED: Extended government with incentives/sanctions fields per feedback.
    incentive_amount: float = 0.0
    sanction_level: float = 0.0
    enforcement_budget: float = 0.0

    def issue_or_lift_mandate(self, day: int) -> None:
        """
        Issue or lift mandate based on schedule (None end means open-ended).
        Updates mandate_status boolean for the current day.
        """
        if not self.mandate_enabled:
            self.mandate_status = False
            return
        self.mandate_status = (day >= self.mandate_start_day) and (
            self.mandate_end_day is None or day <= self.mandate_end_day
        )
        return
        pass

    def adjust_enforcement(self, new_prob: float) -> None:
        """
        Adjust enforcement probability used for mandate enforcement actions.
        """
        self.enforcement_probability = _clamp(new_prob, 0.0, 1.0)
        return
        pass

    def run_public_health_campaign(self, intensity: float) -> None:
        """
        Run a public health campaign by setting the campaign intensity for the day.
        """
        self.campaign_intensity = _clamp(intensity, 0.0, 1.0)
        return
        pass

    def allocate_supply_to_retailers(self, retailers: List[Retailer], amount_total: int) -> None:
        """
        Allocate central supply evenly to retailers. Increases inventory and updates quotas.
        """
        if not retailers or amount_total <= 0:
            return
        per = amount_total // len(retailers)
        for r in retailers:
            r.inventory += per
            r.supply_allocation_quota = per
        return
        pass
    pass


@dataclass
class RegionEnvironment:
    """
    Regional environment for prevalence and risk signals.
    Includes a simple seasonal signal generator and mobility/seasonality modifiers.
    """
    id: int = 0
    baseline_prevalence_indicator: float = 0.1
    mobility_level: float = 1.0
    seasonality_factor: float = 1.0

    def update_prevalence_signal(self, day: int) -> float:
        """
        Update a simple prevalence signal with mild weekly seasonality.
        Returns value in [0,1].
        """
        seasonal = 0.1 * math.sin(day / 14.0 * 2.0 * math.pi)
        signal = _clamp(self.baseline_prevalence_indicator + seasonal, 0.0, 1.0)
        return signal
        pass

    def modulate_risk_perception_signal(self, base_signal: float) -> float:
        """
        Modulate risk perception based on mobility and seasonality settings. Returns value in [0,1].
        """
        return _clamp(base_signal * self.mobility_level * self.seasonality_factor, 0.0, 1.0)
        pass
    pass


class Simulation:
    """
    Main simulation class coordinating agents, environment, policies, and metrics.
    Orchestrates initialization, daily steps, series tracking, evaluation, and optional visualization.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize the Simulation with the provided configuration dictionary.
        Applies a model plan if provided and constructs containers for agents and series.
        """
        self.cfg: Dict[str, Any] = {**DEFAULT_CONFIG, **(cfg or {})}
        self._apply_model_plan(self.cfg.get("model_plan", None))
        self.rng = random.Random(int(self.cfg.get("seed", 42)))
        self.N = max(1, int(self.cfg.get("population_size", 200)))
        val = self.cfg.get("max_time_steps")
        self.days = int(self.cfg.get("days", 10)) if val in (None, "", 0, "0", "None", "null") else int(val)

        # Supply model selection
        self.with_supply = bool(self.cfg.get("with_supply", False))
        # FIXED: Introduced supply model switch per feedback.
        self.with_supply_model = str(self.cfg.get("with_supply_model", "retailer")).lower()

        self.with_epi = bool(self.cfg.get("with_epi", False))
        self.people: List[Individual] = []
        self.households: List[Household] = []
        self.workplaces: List[WorkplaceSchool] = []
        self.locations: List[Location] = []
        self.retailers: List[Retailer] = []
        self.supply: Dict[str, Any] = {"stock": 0, "price": float(self.cfg.get("mask_price", 2.0))}
        self.daily_restock = int(self.cfg.get("daily_supply_restock", 0))

        # Supply instantiation per model
        if self.with_supply and self.with_supply_model == "retailer":
            retailer_count = max(1, int(self.cfg.get("retailer_count", 1)))
            total_initial_stock = max(0, int(self.cfg.get("supply_initial_stock", 0)))
            per_retailer_stock = total_initial_stock // max(1, retailer_count)
            per_restock = int(self.cfg.get("daily_supply_restock", 0)) // max(1, retailer_count)
            self.retailers = [
                Retailer(
                    id=r,
                    inventory=per_retailer_stock,
                    price=float(self.cfg.get("mask_price", 2.0)),
                    restock_rate=per_restock,
                    supply_allocation_quota=per_restock,
                )
                for r in range(retailer_count)
            ]
        elif self.with_supply and self.with_supply_model == "central":
            # FIXED: Enable central supply pool branch when requested.
            self.supply["stock"] = int(self.cfg.get("supply_initial_stock", 0))
            self.daily_restock = int(self.cfg.get("daily_supply_restock", 0))
            self.retailers = []

        me = self.cfg.get("mandate_end_day", 8)
        self.government = Government(
            mandate_enabled=bool(self.cfg.get("mandate_enabled", False)),
            mandate_status=False,
            mandate_start_day=int(self.cfg.get("mandate_start_day", 5)),
            mandate_end_day=(None if me in (None, "None", "null") else int(me)),
            enforcement_probability=float(self.cfg.get("enforcement_probability", 0.1)),
            fine_amount=float(self.cfg.get("fine_amount", 50.0)),
            campaign_intensity=float(self.cfg.get("campaign_intensity", 0.0)),
            incentive_amount=float(self.cfg.get("incentive_amount", 0.0)),
            sanction_level=float(self.cfg.get("sanction_level", 0.0)),
            enforcement_budget=float(self.cfg.get("enforcement_budget", 0.0)),
        )
        self.environment = RegionEnvironment(
            baseline_prevalence_indicator=float(self.cfg.get("risk_perception_baseline", 0.2)),
            mobility_level=1.0,
            seasonality_factor=1.0,
        )

        self.series: Dict[str, List[float]] = {
            "adoption_rate": [],
            "mandate_active": [],
            "retailer_stockout_share": [],
            "neighbor_corr": [],
            "daily_gini": [],
            "retention_rate": [],
            # FIXED: Track any stockout per day for outage metrics.
            "any_stockout": [],
        }
        self.metrics: Dict[str, Any] = {}
        self.enforcement_actions: int = 0
        self.policy_cost_total: float = 0.0
        self.stockout_days: int = 0
        self.retailer_stockout_days: int = 0
        self._violations_under_mandate: int = 0
        self._visits_under_mandate: int = 0
        self.total_masks_sold: int = 0
        self.media = MediaChannel(
            id=1,
            misinformation_rate=float(self.cfg.get("misinformation_prevalence", 0.1)),
            salience=float(self.cfg.get("campaign_intensity", 0.0)),
        )
        self._observed_series: Optional[List[float]] = None
        self.workers_by_wp: Dict[int, List[int]] = {}
        self.visits_total_by_type: Dict[str, int] = {}
        self.visits_compliant_by_type: Dict[str, int] = {}
        self._precomputed_deciles: Optional[List[List[int]]] = None
        # FIXED: Store misinformation event days for impact metric.
        self._misinfo_events: List[int] = []
        pass

    def _apply_model_plan(self, model_plan_input: Optional[Any]) -> None:
        """
        Apply settings from a model plan (dict or JSON path) to the configuration.
        Handles dynamic update of core parameters and evaluation metrics as provided.
        """
        if model_plan_input is None:
            return
        try:
            if isinstance(model_plan_input, str):
                plan = safe_load_json(model_plan_input, strict=False)
            elif isinstance(model_plan_input, dict):
                plan = model_plan_input
            else:
                logger.warning("Model plan provided in unsupported type; ignoring.")
                return
            for k, v in plan.items():
                if k not in ("parameters", "evaluation_metrics", "prediction_period"):
                    self.cfg[k] = v
            params = plan.get("parameters", {})
            if params:
                if "population_size" in params:
                    self.cfg["population_size"] = int(params["population_size"])
                if "time_horizon_days" in params:
                    self.cfg["days"] = int(params["time_horizon_days"])
                    self.cfg["max_time_steps"] = int(params["time_horizon_days"])
                if "random_seed" in params:
                    self.cfg["seed"] = int(params["random_seed"])
                if "network_topology" in params:
                    self.cfg["social_network_type"] = str(params["network_topology"])
                if "avg_degree" in params:
                    self.cfg["average_degree"] = int(params["avg_degree"])
                if "social_influence_weight" in params:
                    self.cfg["base_influence_weight"] = float(params["social_influence_weight"])
                if "observation_norm_weight" in params:
                    self.cfg["workplace_influence_weight"] = float(params["observation_norm_weight"])
                if "habit_formation_rate" in params:
                    self.cfg["habit_formation_rate"] = float(params["habit_formation_rate"])
                if "habit_decay_rate" in params:
                    self.cfg["habit_decay_rate"] = float(params["habit_decay_rate"])
                if "mandate_active" in params:
                    self.cfg["mandate_enabled"] = bool(params["mandate_active"])
                if "mandate_start_day" in params and params["mandate_start_day"] is not None:
                    self.cfg["mandate_start_day"] = int(params["mandate_start_day"])
                if "mandate_end_day" in params:
                    self.cfg["mandate_end_day"] = (
                        params["mandate_end_day"] if params["mandate_end_day"] is None else int(params["mandate_end_day"])
                    )
                if "enforcement_level" in params:
                    self.cfg["enforcement_probability"] = float(params["enforcement_level"])
                if "mask_price" in params:
                    self.cfg["mask_price"] = float(params["mask_price"])
                if "retailer_restock_rate_per_day" in params:
                    pop = int(self.cfg.get("population_size", 200))
                    restock_total = int(max(0, params["retailer_restock_rate_per_day"]) * pop)
                    self.cfg["daily_supply_restock"] = restock_total
                if "initial_mask_stock_per_capita" in params:
                    pop = int(self.cfg.get("population_size", 200))
                    self.cfg["supply_initial_stock"] = int(max(0.0, params["initial_mask_stock_per_capita"]) * pop)
                    self.cfg["with_supply"] = True
                if "target_adoption_threshold" in params:
                    self.cfg["target_adoption_threshold"] = float(params["target_adoption_threshold"])
                if "homophily_by_attitude" in params:
                    self.cfg["homophily_by_attitude"] = float(params["homophily_by_attitude"])
                if "misinformation_rate" in params:
                    self.cfg["misinformation_prevalence"] = float(params["misinformation_rate"])
                if "location_mask_requirement_fraction" in params:
                    self.cfg["location_mask_requirement_fraction"] = float(params["location_mask_requirement_fraction"])
                # Optional supply model override
                if "with_supply_model" in params:
                    self.cfg["with_supply_model"] = str(params["with_supply_model"])
                # FIXED: Spec parameter mappings to internal cfg fields.
                mapping = {
                    "network_type": ("social_network_type", str),
                    "average_degree": ("average_degree", int),
                    "social_influence_strength": ("base_influence_weight", float),
                    "risk_perception_weight": ("risk_perception_weight", float),
                    "perceived_risk_sensitivity_to_prevalence": ("case_prevalence_effect_on_risk_perception", float),
                    "mandate_active_initially": ("mandate_enabled", bool),
                    "mandate_probability_per_day": ("mandate_probability_per_day", float),
                    "enforcement_strength": ("enforcement_probability", float),
                    "information_campaign_intensity": ("campaign_intensity", float),
                    "information_message_frequency_per_day": ("message_frequency", float),
                    "misinformation_reach": ("misinformation_reach", float),
                    "fatigue_rate": ("fatigue_rate", float),
                    "recovery_from_fatigue_rate": ("recovery_from_fatigue_rate", float),
                    "adoption_threshold": ("adoption_threshold", float),
                }
                for sk, (ik, caster) in mapping.items():
                    if sk in params:
                        self.cfg[ik] = caster(params[sk])
            if "evaluation_metrics" in plan:
                self.cfg["evaluation_metrics"] = plan["evaluation_metrics"]
            pred = plan.get("prediction_period", {})
            if isinstance(pred, dict):
                start = pred.get("start_day", None)
                end = pred.get("end_day", None)
                if end is not None:
                    try:
                        self.cfg["days"] = int(end) + 1
                        self.cfg["max_time_steps"] = int(end) + 1
                    except Exception:
                        pass
                self.cfg["prediction_period_start"] = start
        except Exception as e:
            logger.warning(f"Failed to apply model plan: {e}")
        return
        pass

    def initialize(self) -> None:
        """
        Initialize population, network, households, workplaces, locations, and initial states.
        Uses configured distributions for household and workplace sizes.
        """
        self.enforcement_actions = 0
        self.policy_cost_total = 0.0
        self.stockout_days = 0
        self.retailer_stockout_days = 0
        self._violations_under_mandate = 0
        self._visits_under_mandate = 0
        self.total_masks_sold = 0
        self.visits_total_by_type = {}
        self.visits_compliant_by_type = {}
        self.workers_by_wp = {}
        self._misinfo_events = []

        init_rate = float(self.cfg.get("initial_adoption_rate", 0.1))
        avg_degree = max(2, int(self.cfg.get("average_degree", 4)))
        misinfo_prev = float(self.cfg.get("misinformation_prevalence", 0.1))

        # Create population
        self.people = []
        for i in range(self.N):
            adopted = self.rng.random() < init_rate
            income = 30000 + self.rng.random() * 70000
            misinformation_exposure = _clamp(misinfo_prev + self.rng.uniform(-0.1, 0.1), 0.0, 1.0)
            person = Individual(
                id=i,
                age=int(18 + self.rng.random() * 60),
                income=income,
                education_level=int(self.rng.random() * 4),
                health_risk_factor=1.0 + 0.5 * (1 if self.rng.random() < 0.2 else 0),
                risk_perception=float(self.cfg.get("risk_perception_baseline", 0.2)),
                trust_in_authority=_clamp(self.rng.random() * 0.8 + 0.1, 0.0, 1.0),
                social_norm_sensitivity=_clamp(self.rng.random() * 0.8 + 0.1, 0.0, 1.0),
                perceived_mask_efficacy=_clamp(0.3 + 0.4 * self.rng.random(), 0.0, 1.0),
                cost_sensitivity=_clamp(0.3 + 0.4 * self.rng.random(), 0.0, 1.0),
                compliance_propensity=_clamp(0.3 + 0.7 * self.rng.random(), 0.0, 1.0),
                influenceability=_clamp(0.3 + 0.7 * self.rng.random(), 0.0, 1.0),
                # FIXED: Initialize spec attributes with variability.
                risk_aversion=_clamp(0.3 + 0.7 * self.rng.random(), 0.0, 1.0),
                misinformation_susceptibility=_clamp(0.3 + 0.7 * self.rng.random(), 0.0, 1.0),
                social_influence_susceptibility=_clamp(0.3 + 0.7 * self.rng.random(), 0.0, 1.0),
                fatigue_level=0.0,
                current_mask_use=adopted,
                mask_inventory=1 if adopted else 0,
                habit_strength=0.5 if adopted else 0.0,
                mobility_level=_clamp(0.8 + self.rng.uniform(-0.2, 0.3), 0.1, 1.5),
                misinformation_exposure=misinformation_exposure,
            )
            self.people.append(person)

        # Build social network: small-world ring with optional rewiring
        net_type = self.cfg.get("social_network_type", "small_world")
        rewire_p = float(self.cfg.get("rewiring_probability", 0.1))
        avg_k = avg_degree if (avg_degree % 2 == 0) else (avg_degree + 1)
        if self.N >= 2 and net_type == "small_world" and avg_k >= 2:
            ring = [[] for _ in range(self.N)]
            for i in range(self.N):
                for d in range(1, min(avg_k // 2 + 1, self.N)):
                    j2 = (i + d) % self.N
                    if j2 != i:
                        ring[i].append(j2)
                        ring[j2].append(i)
            for i in range(self.N):
                for j in list(ring[i]):
                    if i < j and self.rng.random() < rewire_p:
                        candidates = set(range(self.N)) - {i} - set(ring[i])
                        if candidates:
                            new_j = self.rng.choice(list(candidates))
                            if j in ring[i]:
                                ring[i].remove(j)
                            if i in ring[j]:
                                ring[j].remove(i)
                            ring[i].append(new_j)
                            ring[new_j].append(i)
            for i in range(self.N):
                self.people[i].network_neighbors = list(sorted(set(ring[i])))
        else:
            k = max(2, avg_k)
            for i in range(self.N):
                neighbors = []
                for d in range(1, min(k // 2 + 1, self.N)):
                    neighbors.append((i - d) % self.N)
                    neighbors.append((i + d) % self.N)
                self.people[i].network_neighbors = list(sorted(set(neighbors)))

        # Homophily rewiring
        homophily = float(self.cfg.get("homophily_by_attitude", 0.0))
        if homophily > 0.0:
            for i in range(self.N):
                for j in list(self.people[i].network_neighbors):
                    if i < j and self.rng.random() < homophily:
                        candidates = [k for k in range(self.N) if k != i and k not in self.people[i].network_neighbors]
                        if not candidates:
                            continue
                        target = min(
                            candidates,
                            key=lambda k: abs(
                                self.people[k].perceived_mask_efficacy - self.people[i].perceived_mask_efficacy
                            ),
                        )
                        try:
                            self.people[i].network_neighbors.remove(j)
                            self.people[j].network_neighbors.remove(i)
                        except ValueError:
                            pass
                        self.people[i].network_neighbors.append(target)
                        self.people[target].network_neighbors.append(i)
            for i in range(self.N):
                self.people[i].network_neighbors = list(sorted(set(self.people[i].network_neighbors)))

        # Households using distribution
        # FIXED: Use household_size_distribution to create realistic household clusters.
        self.households = []
        household_id = 0
        i = 0
        hh_dist = self.cfg.get("household_size_distribution", [3])
        while i < self.N:
            size = int(_sample_from_distribution(self.rng, hh_dist, min_val=1, max_val=8))
            size = min(size, self.N - i)
            members = list(range(i, i + size))
            hh_income = sum(self.people[m].income for m in members) / max(1, size)
            hh = Household(
                id=household_id, member_ids=members, household_income=hh_income, household_norms=0.5, shared_mask_stock=0
            )
            self.households.append(hh)
            for m in members:
                self.people[m].household_id = household_id
            household_id += 1
            i += size

        # Workplaces using distribution if provided
        # FIXED: Use workplace_size_distribution for assignments.
        self.workplaces = []
        wp_sizes: List[int] = []
        wp_dist = self.cfg.get("workplace_size_distribution", None)
        if wp_dist:
            remaining = self.N
            while remaining > 0:
                sz = int(_sample_from_distribution(self.rng, wp_dist, min_val=3, max_val=max(3, self.N)))
                sz = min(sz, remaining)
                wp_sizes.append(sz)
                remaining -= sz
        else:
            num_wp = max(1, int(self.cfg.get("workplace_count", 5)))
            wp_sizes = [self.N // num_wp for _ in range(num_wp)]
            remainder = self.N - sum(wp_sizes)
            for r in range(remainder):
                wp_sizes[r % num_wp] += 1
        for w_id, size in enumerate(wp_sizes):
            self.workplaces.append(
                WorkplaceSchool(
                    id=w_id,
                    size=size,
                    policy_strictness=float(self.cfg.get("workplace_policy_strictness", 0.0)),
                    enforcement_capacity=float(self.cfg.get("workplace_enforcement_capacity", 0.0)),
                    adoption_visibility=float(self.cfg.get("workplace_adoption_visibility", 0.5)),
                )
            )
        # Assign individuals sequentially to workplaces per sizes
        idx = 0
        for w in self.workplaces:
            self.workers_by_wp[w.id] = []
            for _ in range(w.size):
                if idx >= self.N:
                    break
                self.people[idx].workplace_id = w.id
                self.workers_by_wp[w.id].append(idx)
                idx += 1

        # Locations
        self.locations = []
        loc_counts = self.cfg.get("location_counts", {"stores": 10, "transit": 5, "public_venues": 8})
        req_frac = float(self.cfg.get("location_mask_requirement_fraction", 0.5))
        loc_id = 0
        for loc_type, count in loc_counts.items():
            c = int(count)
            for _ in range(c):
                if loc_type == "transit":
                    capacity = 100
                    crowd = 0.8
                    vis = 0.8
                    enforce = 0.2
                    signage = 0.7
                elif loc_type == "stores":
                    capacity = 40
                    crowd = 0.5
                    vis = 0.6
                    enforce = 0.15
                    signage = 0.6
                else:
                    capacity = 60
                    crowd = 0.6
                    vis = 0.7
                    enforce = 0.1
                    signage = 0.5
                mask_req = self.rng.random() < req_frac
                # FIXED: Use spec-aligned Location fields.
                self.locations.append(
                    Location(
                        id=loc_id,
                        loc_type=loc_type,
                        capacity=capacity,
                        crowding_level=crowd,
                        mask_required=mask_req,
                        enforcement_level=enforce,
                        contact_rate=_clamp(0.5 + 0.5 * crowd, 0.0, 1.0),
                        visibility_of_mask_use=vis,
                        signage_strength=signage,
                    )
                )
                loc_id += 1

        # Precompute deciles for performance
        # FIXED: Precompute income deciles to avoid daily sorting cost.
        self._precomputed_deciles = self._income_deciles()

        # Initialize series
        self.series["adoption_rate"] = [self._adoption_rate()]
        self.series["mandate_active"] = [1 if self.government.mandate_status else 0]
        self.series["retailer_stockout_share"] = [self._retailer_stockout_share()]
        self.series["neighbor_corr"] = [self._neighbor_influence_corr()]
        self.series["daily_gini"] = [self._adoption_gini_by_income_deciles()]
        self.series["retention_rate"] = [None]
        # Initialize any_stockout flag
        any_out = False
        if self.with_supply and self.with_supply_model == "retailer":
            any_out = any(r.inventory <= 0 for r in self.retailers)
        elif self.with_supply and self.with_supply_model == "central":
            any_out = self.supply["stock"] <= 0
        self.series["any_stockout"] = [1 if any_out else 0]
        return
        pass

    def _adoption_rate(self) -> float:
        """
        Compute current adoption rate as fraction of individuals with current_mask_use True.
        """
        if self.N <= 0:
            return 0.0
        return sum(1 for p in self.people if p.current_mask_use) / float(self.N)
        pass

    def _retailer_stockout_share(self) -> float:
        """
        Compute fraction of retailers stocked out (inventory <= 0).
        Returns 0 if retailer model not in use.
        """
        if not self.retailers:
            return 0.0
        return sum(1 for r in self.retailers if r.inventory <= 0) / len(self.retailers)
        pass

    def _income_deciles(self) -> List[List[int]]:
        """
        Compute indices for income deciles based on initial income ranking.
        Returns a list of lists of indices (up to 10 deciles).
        """
        if self.N <= 0:
            return []
        sorted_indices = sorted(range(self.N), key=lambda idx: self.people[idx].income)
        deciles: List[List[int]] = []
        size = max(1, self.N // 10)
        for i in range(0, self.N, size):
            deciles.append(sorted_indices[i: min(self.N, i + size)])
        if len(deciles) > 10:
            deciles = deciles[:10]
        return deciles
        pass

    def _adoption_gini_by_income_deciles(self) -> float:
        """
        Compute Gini across income decile adoption means using precomputed deciles if available.
        """
        deciles = self._precomputed_deciles if self._precomputed_deciles is not None else self._income_deciles()
        means = []
        for dec in deciles:
            if not dec:
                continue
            mean = sum(1.0 if self.people[i].current_mask_use else 0.0 for i in dec) / len(dec)
            means.append(mean)
        return _gini(means) if means else 0.0
        pass

    def _neighbor_influence_corr(self) -> float:
        """
        Compute correlation between neighbor adoption and individual adoption as a proxy for influence.
        """
        if self.N <= 0:
            return 0.0
        xs: List[float] = []
        ys: List[float] = []
        for p in self.people:
            neigh = p.network_neighbors
            if len(neigh) == 0:
                neigh_adopt = 0.0
            else:
                neigh_adopt = sum(1 for j in neigh if self.people[j].current_mask_use) / max(1, len(neigh))
            xs.append(neigh_adopt)
            ys.append(1.0 if p.current_mask_use else 0.0)
        return _pearson_correlation(xs, ys)
        pass

    def _largest_adopter_component_fraction(self) -> float:
        """
        Fraction of population in the largest connected component among adopters in the social network.
        """
        adopters = {i for i, p in enumerate(self.people) if p.current_mask_use}
        if not adopters or self.N <= 0:
            return 0.0
        visited: set = set()
        max_size = 0
        for node in adopters:
            if node in visited:
                continue
            q = deque([node])
            visited.add(node)
            size = 0
            while q:
                u = q.popleft()
                size += 1
                for v in self.people[u].network_neighbors:
                    if v in adopters and v not in visited:
                        visited.add(v)
                        q.append(v)
            max_size = max(max_size, size)
        return max_size / float(self.N)
        pass

    def _daily_activity_level(self, day: int) -> float:
        """
        Compute a daily activity multiplier to introduce time-dependent activity peaks (weekly pattern).
        """
        pattern = str(self.cfg.get("activity_pattern", "weekly_peaks"))
        base = float(self.cfg.get("activity_base", 0.8))
        peak_mult = float(self.cfg.get("activity_peak_multiplier", 1.2))
        if pattern == "weekly_peaks":
            weekday = day % 7
            peaks = set(int(x) % 7 for x in self.cfg.get("activity_weekday_peak", [2, 3, 4]))
            return peak_mult if weekday in peaks else base
        return _clamp(1.0 + 0.1 * math.sin(2 * math.pi * day / 7.0), 0.6, 1.4)
        pass

    def _k_hop_visible_compliance(self, idx: int, k: int = 1) -> float:
        """
        Compute k-hop visible compliance fraction for a given individual index.
        Returns the average observed mask use among nodes within k hops.
        """
        if k <= 1:
            neigh = self.people[idx].network_neighbors
            return sum(1 for j in neigh if self.people[j].current_mask_use) / max(1, len(neigh))
        visited = {idx}
        frontier = set(self.people[idx].network_neighbors)
        depth = 1
        acc: List[int] = []
        while frontier and depth <= k:
            next_frontier = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                acc.append(node)
                for nb in self.people[node].network_neighbors:
                    if nb not in visited:
                        next_frontier.add(nb)
            frontier = next_frontier
            depth += 1
        if not acc:
            return 0.0
        return sum(1 for j in acc if self.people[j].current_mask_use) / len(acc)
        pass

    def step(self, day: int) -> None:
        """
        Advance the simulation by one day, updating beliefs, behaviors, enforcement, supply, and tracking metrics.
        Implements decoupled risk and efficacy updates and includes supply shock logic.
        """
        # FIXED: Stochastic mandate activation per-day prior to issuing/lifting.
        mandate_prob = float(self.cfg.get("mandate_probability_per_day", 0.0))
        if not self.government.mandate_enabled and self.rng.random() < mandate_prob:
            self.government.mandate_enabled = True
            self.government.mandate_start_day = day
            self.government.mandate_end_day = None
        # Issue or lift mandate based on schedule
        self.government.issue_or_lift_mandate(day)

        # Environment and media
        base_signal = self.environment.update_prevalence_signal(day)
        risk_signal = self.environment.modulate_risk_perception_signal(base_signal)
        self.government.run_public_health_campaign(float(self.cfg.get("campaign_intensity", 0.0)))
        dr_global, dt_global = self.media.broadcast(base_signal)

        # FIXED: Misinformation broadcast events with logging and shock.
        misinfo_p = float(self.cfg.get("misinformation_rate", self.cfg.get("misinformation_prevalence", 0.05)))
        misinfo_reach = float(self.cfg.get("misinformation_reach", 0.3))
        misinfo_event_today = self.rng.random() < misinfo_p
        if misinfo_event_today:
            self._misinfo_events.append(day)
        misinfo_shock = misinfo_reach if misinfo_event_today else 0.0

        # Supply handling
        # FIXED: Clean supply flow by using with_supply_model switch exclusively and align any_stockout series when no supply.
        any_out_today_flagged = 0
        if self.with_supply and self.with_supply_model == "retailer":
            any_out = any(r.inventory <= 0 for r in self.retailers)
            any_out_today_flagged = 1 if any_out else 0
            # Allocate supply from central pool (implicit) via government policy
            self.government.allocate_supply_to_retailers(self.retailers, int(self.cfg.get("daily_supply_restock", 0)))
            # Supply shocks and price adjustments
            shock_p = float(self.cfg.get("supply_shock_probability", 0.01))
            for r in self.retailers:
                if self.rng.random() < shock_p:
                    # FIXED: Supply shock on retailer inventory and price per feedback.
                    r.inventory = max(0, int(r.inventory * 0.5))
                    r.price = min(10.0, r.price * 1.2)
                r.adjust_price_based_on_inventory()
        elif self.with_supply and self.with_supply_model == "central":
            any_out_today_flagged = 1 if self.supply["stock"] <= 0 else 0
            self.supply["stock"] += self.daily_restock
        else:
            any_out_today_flagged = 0
        # FIXED: Ensure any_stockout series appended every day.
        self.series.setdefault("any_stockout", []).append(any_out_today_flagged)

        # Influence weights and habit/fatigue dynamics
        peer_w = float(self.cfg.get("base_influence_weight", self.cfg.get("peer_influence", 0.3)))
        household_w = float(self.cfg.get("household_influence_weight", 0.1))
        workplace_w = float(self.cfg.get("workplace_influence_weight", 0.2))
        habit_form = float(self.cfg.get("habit_formation_rate", 0.1))
        habit_decay = float(self.cfg.get("habit_decay_rate", 0.02))
        fatigue_rate = float(self.cfg.get("fatigue_rate", 0.01))
        recovery_rate = float(self.cfg.get("recovery_from_fatigue_rate", 0.005))
        risk_weight = float(self.cfg.get("risk_perception_weight", 0.3))
        activity = self._daily_activity_level(day)
        peer_w_eff = peer_w * activity

        # For k-hop visible compliance
        k_hops = max(1, int(self.cfg.get("observation_radius_hops", 1)))

        # Prepare per-person visits
        new_states: List[bool] = []
        enforcement_today = 0
        prev_adoption = self.series.get("adoption_rate", [self._adoption_rate()])[-1]
        vis_strength = float(self.cfg.get("visibility_effect_strength", 0.3))
        case_eff = float(self.cfg.get("case_prevalence_effect_on_risk_perception", 0.4))
        all_visits: Dict[int, List[int]] = {}

        # Draw visits; weight expected visits with an average contact_rate of locations
        avg_loc_contact = 0.5
        if self.locations:
            avg_loc_contact = sum(loc.contact_rate for loc in self.locations) / len(self.locations)
        for i, p in enumerate(self.people):
            expected_visits = p.mobility_level * activity * (1.0 + avg_loc_contact)
            visits_count = max(0, int(self.rng.random() + expected_visits))
            if self.locations and visits_count > 0:
                chosen = [self.rng.randrange(0, len(self.locations)) for _ in range(visits_count)]
            else:
                chosen = []
            all_visits[i] = chosen

        # FIXED: Social information sharing pass to affect neighbors' beliefs using susceptibilities.
        risk_deltas = [0.0] * self.N
        eff_deltas = [0.0] * self.N
        base_info_delta = 0.01
        for i, p in enumerate(self.people):
            signal = base_info_delta if p.current_mask_use else -base_info_delta
            for j in p.network_neighbors:
                neigh = self.people[j]
                # Scaling by neighbor susceptibilities and mitigation by their misinformation_susceptibility
                scale = (0.5 + 0.5 * neigh.social_influence_susceptibility) * (1.0 - 0.5 * neigh.misinformation_susceptibility)
                risk_deltas[j] += scale * signal
                eff_deltas[j] += scale * signal
        for j in range(self.N):
            self.people[j].risk_perception = _clamp(self.people[j].risk_perception + risk_deltas[j], 0.0, 1.0)
            self.people[j].perceived_mask_efficacy = _clamp(self.people[j].perceived_mask_efficacy + eff_deltas[j], 0.0, 1.0)

        # Person updates
        for i, p in enumerate(self.people):
            # Compute peer_share first (k-hop visibility optional)
            if k_hops <= 1:
                peer_share = sum(1 for j in p.network_neighbors if self.people[j].current_mask_use) / max(
                    1, len(p.network_neighbors)
                )
            else:
                peer_share = self._k_hop_visible_compliance(i, k=k_hops)

            # risk_perception update from environment and media with weighting
            env_contrib = risk_weight * case_eff * risk_signal
            social_contrib = (1.0 - risk_weight) * peer_share
            p.risk_perception = _clamp(
                p.risk_perception
                + float(self.cfg.get("risk_perception_update_rate", 0.1)) * ((env_contrib + social_contrib) - p.risk_perception),
                0.0,
                1.0,
            )
            # FIXED: Decoupled efficacy update from generic policy signal; now via media effect and trust.
            eff_rate = float(self.cfg.get("perceived_efficacy_update_rate", 0.08))
            media_eff = self.media.salience * float(self.cfg.get("message_effect_size", 0.2))
            trust_scale = _clamp(p.trust_in_authority, 0.0, 1.0)
            p.perceived_mask_efficacy = _clamp(
                p.perceived_mask_efficacy + eff_rate * trust_scale * (media_eff - p.perceived_mask_efficacy),
                0.0,
                1.0,
            )

            # Forgetting on trust
            p.trust_in_authority = _clamp(
                p.trust_in_authority * (1 - float(self.cfg.get("forgetting_rate", 0.01))), 0.0, 1.0
            )
            # Media influence including misinformation shock
            p.consume_media_messages(
                intensity=self.media.salience + dr_global,
                misinformation_rate=self.media.misinformation_rate + misinfo_shock,
            )
            p.trust_in_authority = _clamp(p.trust_in_authority + dt_global, 0.0, 1.0)

            p.update_beliefs_from_social_influence(peer_share, update_rate=peer_w_eff * 0.1 * p.influenceability)
            p.respond_to_policies_and_campaigns(self.government.campaign_intensity)

            # Workplace signal
            wp_signal = 0.0
            if p.workplace_id is not None and 0 <= p.workplace_id < len(self.workplaces):
                wp = self.workplaces[p.workplace_id]
                wp_signal = wp.policy_strictness * workplace_w * activity

            # Mask purchasing attempt when needed
            base_price = float(self.cfg.get("mask_price", self.supply.get("price", 2.0)))
            last_market_price = base_price
            if self.with_supply and p.mask_inventory <= 0 and self.rng.random() < activity:
                if self.with_supply_model == "retailer" and self.retailers:
                    r = self.rng.choice(self.retailers)
                    r_inventory = {"stock": r.inventory}
                    _bought = p.purchase_masks_from_retailer(
                        price=r.price,
                        inventory=r_inventory,
                        rng=self.rng,
                        affordability_threshold=float(self.cfg.get("affordability_threshold", 0.05)),
                        price_elasticity=float(self.cfg.get("price_elasticity_of_demand", -0.8)),
                    )
                    r.inventory = r_inventory["stock"]
                    last_market_price = r.price
                    if _bought > 0:
                        self.total_masks_sold += _bought
                elif self.with_supply_model == "central":
                    qty_bought = p.purchase_masks_from_retailer(price=base_price, inventory=self.supply, rng=self.rng)
                    last_market_price = base_price
                    if qty_bought > 0:
                        self.total_masks_sold += qty_bought

            # Visibility blending with workplace and public visits
            peer_share_visible = peer_share
            if p.workplace_id is not None:
                wp = self.workplaces[p.workplace_id]
                co_workers = self.workers_by_wp.get(wp.id, [])
                if co_workers:
                    loc_rate = sum(1 for j in co_workers if self.people[j].current_mask_use) / len(co_workers)
                else:
                    loc_rate = prev_adoption
                peer_share_visible = (1 - vis_strength * wp.adoption_visibility) * peer_share + (
                    vis_strength * wp.adoption_visibility
                ) * loc_rate

            visits = all_visits.get(i, [])
            for vid in visits:
                loc = self.locations[vid]
                loc_rate_estimate = prev_adoption
                alpha = _clamp(vis_strength * loc.visibility_of_mask_use, 0.0, 1.0)
                peer_share_visible = (1 - alpha) * peer_share_visible + alpha * (
                    (loc_rate_estimate + loc.signage_strength * 0.2) / (1.0 + 0.2)
                )

            # Decision
            mandate_active = bool(self.government.mandate_status)
            # policy_signal is only government campaign intensity now, not pulled for efficacy
            local_policy_signal = float(self.cfg.get("policy_signal", 0.0)) + self.government.campaign_intensity + wp_signal
            decision = p.evaluate_mask_use_decision(
                peer_share=peer_w_eff * peer_share_visible + household_w * self._household_norm_component(p),
                policy_signal=local_policy_signal,
                price=last_market_price,
                mandate_active=mandate_active,
                rng=self.rng,
                habit_weight=0.5,
                fatigue_level=p.fatigue_level,
                risk_weight=risk_weight,
                peer_weight=peer_w,
            )

            # Determine use_today with possible compliance-with-mandate override
            use_today = decision
            # If in a mandated context (global or location-specific), allow explicit compliance to enforce wearing.
            # Compliance decision happens if they initially decided not to wear.
            if not use_today and (mandate_active or any(self.locations[v].mask_required for v in visits)):
                location_required = mandate_active or any(self.locations[v].mask_required for v in visits)
                if p.comply_with_mandate(self.government.enforcement_probability, location_required, self.rng):
                    use_today = True

            # Enforcement: mandate level
            if mandate_active and not use_today:
                if self.rng.random() < self.government.enforcement_probability:
                    enforcement_today += 1
                    p.fines_count += 1
                    fine_boost = float(self.cfg.get("fine_effect_on_future_compliance", 0.3))
                    p.compliance_propensity = _clamp(p.compliance_propensity + fine_boost, 0.0, 1.0)

            # Local workplace enforcement
            if p.workplace_id is not None:
                wp = self.workplaces[p.workplace_id]
                if wp.policy_strictness > 0 and not use_today and self.rng.random() < wp.enforcement_capacity:
                    enforcement_today += 1
                    p.compliance_propensity = _clamp(p.compliance_propensity + 0.05, 0.0, 1.0)

            # Use mask if decided and inventory exists; else no-use under supply constraint
            if self.with_supply:
                if use_today and p.mask_inventory > 0:
                    p.mask_inventory -= 1
                elif use_today and p.mask_inventory <= 0:
                    use_today = False

            # Habit and fatigue updates
            if use_today:
                p.habit_strength = _clamp(p.habit_strength + habit_form * (1.0 - p.habit_strength), 0.0, 1.0)
                # FIXED: Wearing increases fatigue.
                p.fatigue_level = _clamp(p.fatigue_level + fatigue_rate, 0.0, 1.0)
            else:
                p.habit_strength = _clamp(p.habit_strength * (1.0 - habit_decay), 0.0, 1.0)
                # FIXED: Recovery when not wearing.
                p.fatigue_level = _clamp(p.fatigue_level - recovery_rate, 0.0, 1.0)

            # Track compliance and per-visit enforcement
            for vid in visits:
                loc = self.locations[vid]
                self.visits_total_by_type[loc.loc_type] = self.visits_total_by_type.get(loc.loc_type, 0) + 1
                if use_today:
                    self.visits_compliant_by_type[loc.loc_type] = self.visits_compliant_by_type.get(loc.loc_type, 0) + 1
                requirement_active = mandate_active or loc.mask_required
                if requirement_active:
                    self._visits_under_mandate += 1
                    if not use_today:
                        self._violations_under_mandate += 1
                        if loc.enforce_policy(compliant=False, rng=self.rng):
                            enforcement_today += 1
                            p.fines_count += 1
                            p.compliance_propensity = _clamp(p.compliance_propensity + 0.05, 0.0, 1.0)
            new_states.append(use_today)

        # Workplace-level enforce_policy_on_attendees daily
        # FIXED: Explicitly invoke workplace enforcement pass per feedback.
        for w in self.workplaces:
            actions = w.enforce_policy_on_attendees(self.people, self.rng)
            if actions > 0:
                self.enforcement_actions += actions
                enforcement_today += actions

        # Adoption retention
        prev_maskers = sum(1 for p in self.people if p.current_mask_use)
        still_masking = sum(1 for st, p in zip(new_states, self.people) if p.current_mask_use and st)
        retention_today = (still_masking / prev_maskers) if prev_maskers > 0 else None

        # Commit new states
        for p, st in zip(self.people, new_states):
            p.current_mask_use = st

        # Accumulate costs and enforcement
        self.enforcement_actions += enforcement_today
        self.policy_cost_total += 10.0 * self.government.campaign_intensity + 0.5 * enforcement_today

        # Household intradynamics
        for hh in self.households:
            hh.intra_household_influence(self.people, weight=household_w)
            hh.share_masks_among_members(self.people)

        # Series update
        self.series["adoption_rate"].append(self._adoption_rate())
        self.series["mandate_active"].append(1 if self.government.mandate_status else 0)
        self.series["retailer_stockout_share"].append(self._retailer_stockout_share())
        self.series["neighbor_corr"].append(self._neighbor_influence_corr())
        self.series["daily_gini"].append(self._adoption_gini_by_income_deciles())
        self.series["retention_rate"].append(retention_today)
        return
        pass

    def _household_norm_component(self, p: Individual) -> float:
        """
        Compute a small household norm component based on household average adoption for the individual's household.
        """
        if p.household_id is None or not (0 <= p.household_id < len(self.households)):
            return 0.0
        hh = self.households[p.household_id]
        if not hh.member_ids:
            return 0.0
        avg = sum(1.0 if self.people[i].current_mask_use else 0.0 for i in hh.member_ids) / len(hh.member_ids)
        return avg
        pass

    def run(self, days: Optional[int] = None) -> Dict[str, List[Any]]:
        """
        Initialize and run the simulation for the specified number of days.
        Returns the time series dictionary.
        """
        self.initialize()
        total_days = int(days) if days is not None else int(self.days)
        # We already recorded day 0 in initialize; step from day 1 to total_days - 1
        for d in range(1, total_days):
            self.step(d)
        return self.series


# Execute main for both direct execution and sandbox wrapper invocation
main()