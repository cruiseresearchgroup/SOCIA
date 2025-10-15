# FIXED: Applied feedback snippet from simulation.py
def main():
    cfg = DEFAULT_CONFIG.copy()
    sim = Simulation(cfg)
    sim.run()
    metrics = sim.evaluate([
        "mean_adoption",
        "time_to_50_percent_adoption",
        "steady_state_adoption",
        "adoption_disparity_by_income",
        "compliance_in_mandated_locations",
        "masks_sold",
        "network_cascade_size",
        "retention_rate",
    ])
    sim.save_results(os.path.join(SAFE_PROJECT_ROOT, "results.csv"))
    print(json.dumps({"metrics": metrics}))

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mask_sim")

# Path handling as per instructions
# FIXED: Follow exact path handling instructions and provide safe fallbacks for usage sites
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else None  # May be None if env not set
SAFE_PROJECT_ROOT = PROJECT_ROOT if PROJECT_ROOT else os.getcwd()  # For output files
SAFE_DATA_DIR = DATA_DIR if DATA_DIR else SAFE_PROJECT_ROOT  # For data file default lookup

def safe_load_json(maybe_path: str, strict: bool = True) -> Dict[str, Any]:
    """Load JSON from a file path if it exists; otherwise parse the string as JSON."""
    pass
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

# FIXED: Restore a functional minimal simulation core with deterministic defaults
# FIXED: Parameter alignment and additions: homophily_by_attitude, misinformation_prevalence, visibility_effect_strength,
#        affordability_threshold, price_elasticity_of_demand, case_prevalence_effect_on_risk_perception, location counts.
DEFAULT_CONFIG: Dict[str, Any] = {
    "population_size": 200,
    "days": 10,
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
    "campaign_intensity": 0.0,
    "risk_perception_baseline": 0.2,
    "risk_perception_update_rate": 0.1,
    "case_prevalence_effect_on_risk_perception": 0.4,  # FIXED: Scale risk updates by case prevalence effect
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
    "with_epi": False,
    "average_degree": 4,
    "social_network_type": "small_world",
    "rewiring_probability": 0.1,
    "homophily_by_attitude": 0.0,  # FIXED: Implement homophily in network rewiring
    "time_step_unit": "day",
    "workplace_count": 5,
    "workplace_policy_strictness": 0.0,
    "workplace_enforcement_capacity": 0.0,
    "workplace_adoption_visibility": 0.5,
    "visibility_effect_strength": 0.3,  # FIXED: Visibility effect for norms reinforcement
    "misinformation_prevalence": 0.1,  # FIXED: Used to initialize per-person exposure
    "affordability_threshold": 0.05,  # FIXED: Price/income threshold for purchase gating
    "price_elasticity_of_demand": -0.8,  # FIXED: Elastic demand in purchase decision
    "location_mask_requirement_fraction": 0.5,  # Fraction of locations with requirements
    "location_counts": {"stores": 10, "transit": 5, "public_venues": 8},  # FIXED: Minimal location system
    "consistency_window": 3,
    "evaluation_metrics": [
        "adoption_rate_over_time",
        "mean_adoption",
        "time_to_50_percent",
        "peak_adoption",
        "adoption_inequality_index",
        "policy_cost",
        "enforcement_actions_count",
        "stockout_rate",
        "spillover_persistence",
        # Aliases for spec-aligned names supported via mapping
        "overall_adoption_rate",
        "peak_adoption_rate",
        "stockout_frequency",
        "sustained_adoption_rate",
        "violation_rate",
        "policy_effect_size",
        "network_influence_index",
        "adoption_inequality",
        # Validation metrics
        "calibration_rmse",
        "calibration_mae",
        "calibration_corr",
        "calibration_rmse_below_threshold",
        "noncompliance_incidents_per_1000_entries",
        "mask_shortage_days",
        "time_to_threshold",
        # Spec-aligned additional metrics
        "time_to_50_percent_adoption",
        "steady_state_adoption",
        "adoption_disparity_by_income",
        "compliance_in_mandated_locations",
        "masks_sold",
        "network_cascade_size",
        "retention_rate",
    ],
    # FIXED: Habit formation and decay rates to support more realistic behavior persistence
    "habit_formation_rate": 0.1,
    "habit_decay_rate": 0.02,
    # FIXED: Activity pattern parameters to model non-uniform daily activity (peaks)
    "activity_pattern": "weekly_peaks",
    "activity_weekday_peak": [2, 3, 4],  # Tue-Thu peaks
    "activity_peak_multiplier": 1.2,
    "activity_base": 0.8,
    # FIXED: Real data calibration settings
    "observed_series_path": None,  # If None, will attempt DATA_DIR/observed_adoption.csv
    "calibration_rmse_threshold": 0.05,
    # Model-plan mapping defaults (will be overridden by model plan if provided)
    "target_adoption_threshold": 0.7,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a float between lo and hi."""
    pass
    return max(lo, min(hi, x))


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation between two lists."""
    pass
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


def _gini(values: List[float]) -> float:
    """Compute Gini coefficient for non-negative values list."""
    pass
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


@dataclass
class MediaChannel:
    """Minimal media channel that can broadcast a signal affecting risk perception and trust.

    Attributes:
        id: Identifier.
        misinformation_rate: Propensity to introduce negative information.
        salience: Intensity of messaging.
    """
    pass
    id: int = 0
    misinformation_rate: float = 0.0
    salience: float = 0.0

    def broadcast(self, base_signal: float) -> Tuple[float, float]:
        """Compute deltas to risk perception and trust based on media properties.

        Returns:
            Tuple of (delta_risk, delta_trust)
        """
        pass
        delta_risk = 0.02 * self.salience - 0.03 * self.misinformation_rate
        delta_trust = 0.01 * self.salience - 0.02 * self.misinformation_rate
        return delta_risk, delta_trust


@dataclass
class Location:
    """Represents a public location with mask policies, enforcement, and visibility.

    Attributes:
        id: Unique identifier.
        loc_type: Category of location ('stores', 'transit', 'public_venues', etc.).
        capacity: Maximum nominal capacity.
        crowding_level: Effective crowding (0..1).
        mask_requirement: Whether masks are required at this location.
        enforcement_strictness: Probability of enforcement action when non-compliant.
        visibility_of_mask_use: How visible mask use is at this location (0..1).
        signage_strength: Strength of signage/guidelines contributing to perceived norms (0..1).
    """
    pass
    id: int = 0
    loc_type: str = "stores"
    capacity: int = 50
    crowding_level: float = 0.5
    mask_requirement: bool = False
    enforcement_strictness: float = 0.1
    visibility_of_mask_use: float = 0.6
    signage_strength: float = 0.4

    def enforce_policy(self, compliant: bool, rng: random.Random) -> bool:
        """Apply enforcement when non-compliant; returns True if enforcement triggers action."""
        pass
        if compliant:
            return False
        if rng.random() < self.enforcement_strictness:
            return True
        return False

    def record_compliance(self, is_compliant: bool) -> None:
        """Placeholder to record compliance; can be extended to keep per-location stats."""
        pass

    def allow_or_deny_entry(self, compliant: bool, rng: random.Random) -> bool:
        """Allow or deny entry based on policy and compliance; simplified to allow entry."""
        pass
        # Simplified: always allow entry; enforcement handles consequences
        return True


@dataclass
class Individual:
    """Represents an individual agent in the simulation with behavioral attributes.

    Attributes:
        id: Unique identifier.
        age: Age of the individual.
        income: Income level.
        education_level: Education level categorical or numeric.
        health_risk_factor: Baseline health risk multiplier.
        risk_perception: Subjective perception of risk (0..1).
        trust_in_authority: Trust in public authorities (0..1).
        social_norm_sensitivity: Sensitivity to peer influence (0..1).
        perceived_mask_efficacy: Perceived efficacy of masks (0..1).
        cost_sensitivity: Sensitivity to mask cost (0..1).
        compliance_propensity: Propensity to comply with mandates (0..1).
        influenceability: General influenceability (0..1).
        current_mask_use: Whether currently using masks (bool).
        mask_inventory: Integer count of masks available.
        household_id: Household membership id.
        workplace_id: Workplace/school membership id.
        network_neighbors: List of neighbor indices in the population.
        habit_strength: Halo of past behavior increasing future adoption (0..1).
        mobility_level: Propensity to visit locations per day (0..1+).
        misinformation_exposure: Personal susceptibility/exposure to misinformation (0..1).
    """
    pass
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
    current_mask_use: bool = False
    mask_inventory: int = 0
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
    ) -> bool:
        """Compute the probability to adopt mask use and sample new decision.

        The utility combines:
        - Peer influence weighted by social_norm_sensitivity
        - Policy/campaign signal weighted by trust_in_authority
        - Risk perception weighted by perceived efficacy
        - Cost pressure weighted by cost_sensitivity and price
        - Mandate adds a compliance boost
        - Habit strength boosts propensity

        Returns:
            bool indicating whether the individual uses mask after decision.
        """
        pass
        # Utility components
        peer_util = self.social_norm_sensitivity * peer_share
        policy_util = self.trust_in_authority * policy_signal
        risk_util = self.perceived_mask_efficacy * _clamp(self.risk_perception, 0.0, 1.0)
        # FIXED: Use bounded cost penalty from dynamic market price
        cost_penalty = self.cost_sensitivity * (price / max(0.1, price + 1.0))  # bounded [~0, ~0.66]
        mandate_boost = self.compliance_propensity * (1.0 if mandate_active else 0.0)
        habit_boost = habit_weight * _clamp(self.habit_strength, 0.0, 1.0)

        # FIXED: More sophisticated utility model blending components with interpretable weights
        util = (
            0.45 * peer_util
            + 0.25 * policy_util
            + 0.5 * risk_util
            + 0.35 * mandate_boost
            + 0.3 * habit_boost
            - 0.5 * cost_penalty
        )
        util = _clamp(util, -5.0, 5.0)
        # Logistic mapping to probability with slight noise
        noise = rng.uniform(-0.1, 0.1)
        prob = 1.0 / (1.0 + math.exp(-(util + noise)))
        # Inventory constraint: if no masks, reduce prob significantly
        if self.mask_inventory <= 0:
            prob *= 0.3
        return rng.random() < prob or self.current_mask_use

    def update_beliefs_from_social_influence(self, peer_share: float, update_rate: float) -> None:
        """Update perceived efficacy and risk perception based on peer share."""
        pass
        self.perceived_mask_efficacy = _clamp(
            self.perceived_mask_efficacy + update_rate * (peer_share - self.perceived_mask_efficacy), 0.0, 1.0
        )
        self.risk_perception = _clamp(
            self.risk_perception + 0.5 * update_rate * (peer_share - self.risk_perception), 0.0, 1.0
        )

    def respond_to_policies_and_campaigns(self, campaign_intensity: float) -> None:
        """Adjust trust and efficacy slightly in response to campaigns, modulated by misinformation exposure."""
        pass
        # FIXED: Incorporate misinformation exposure to dampen campaign effects
        scale = _clamp(1.0 - self.misinformation_exposure, 0.0, 1.0)
        self.trust_in_authority = _clamp(self.trust_in_authority + scale * 0.05 * campaign_intensity, 0.0, 1.0)
        self.perceived_mask_efficacy = _clamp(self.perceived_mask_efficacy + scale * 0.04 * campaign_intensity, 0.0, 1.0)

    def purchase_masks_from_retailer(self, price: float, inventory: Dict[str, Any], rng: random.Random, affordability_threshold: float = 0.05, price_elasticity: float = -0.8) -> int:
        """Attempt to purchase masks with affordability and elasticity.

        Returns:
            Quantity purchased.
        """
        pass
        if inventory.get("stock", 0) <= 0:
            return 0
        # Affordability gate: price as a share of daily income proxy
        income_daily = max(1.0, self.income / 365.0)
        if (price / income_daily) > affordability_threshold:
            return 0
        # Elastic demand: higher price reduces purchase probability
        base_p = _clamp(0.6 * (1.0 - self.cost_sensitivity), 0.0, 1.0)
        price_factor = _clamp((1.0 + price) ** price_elasticity, 0.0, 1.0)
        buy_prob = _clamp(base_p * price_factor, 0.0, 1.0)
        desired_qty = 1 if rng.random() < buy_prob else 0
        qty = min(desired_qty, inventory.get("stock", 0))
        inventory["stock"] -= qty
        self.mask_inventory += qty
        return qty

    def share_opinion_with_neighbors(self) -> None:
        """Placeholder for opinion dynamics."""
        pass

    def attend_work_or_school(self) -> None:
        """Placeholder for local policy effect at workplaces/schools."""
        pass

    def consume_media_messages(self, intensity: float, misinformation_rate: float = 0.0) -> None:
        """Adjust perceptions due to media messages, modulated by personal misinformation exposure."""
        pass
        scale = _clamp(1.0 - self.misinformation_exposure, 0.0, 1.0)
        self.risk_perception = _clamp(self.risk_perception + scale * (0.02 * intensity - 0.03 * misinformation_rate), 0.0, 1.0)
        self.trust_in_authority = _clamp(self.trust_in_authority + scale * (0.01 * intensity - 0.02 * misinformation_rate), 0.0, 1.0)


@dataclass
class Household:
    """Represents a household for intra-household dynamics."""
    pass
    id: int = 0
    member_ids: List[int] = field(default_factory=list)
    household_income: float = 0.0
    household_norms: float = 0.5
    shared_mask_stock: int = 0

    def intra_household_influence(self, individuals: List[Individual], weight: float = 0.1) -> None:
        """Adjust household norms to average of member mask use and nudge members."""
        pass
        if not self.member_ids:
            return
        avg_use = sum(1.0 if individuals[i].current_mask_use else 0.0 for i in self.member_ids) / len(self.member_ids)
        self.household_norms = _clamp(0.7 * self.household_norms + 0.3 * avg_use, 0.0, 1.0)
        # Nudge perceived efficacy toward household norm
        for i in self.member_ids:
            ind = individuals[i]
            ind.perceived_mask_efficacy = _clamp(
                ind.perceived_mask_efficacy + weight * (self.household_norms - ind.perceived_mask_efficacy),
                0.0,
                1.0,
            )

    def share_masks_among_members(self, individuals: List[Individual]) -> None:
        """Share masks if some members lack masks, using a simple pool."""
        pass
        total = sum(individuals[i].mask_inventory for i in self.member_ids) + self.shared_mask_stock
        if total <= 0:
            return
        need = [i for i in self.member_ids if individuals[i].mask_inventory == 0]
        have = [i for i in self.member_ids if individuals[i].mask_inventory > 1]
        # Give one unit from 'have' to 'need' if possible
        for needy in need:
            if have:
                donor = have.pop()
                individuals[donor].mask_inventory -= 1
                individuals[needy].mask_inventory += 1


@dataclass
# FIXED: Applied feedback snippet from simulation.py
class WorkplaceSchool:
    """Represents a workplace or school with a local policy."""
    id: int = 0
    size: int = 0
    policy_strictness: float = 0.0
    enforcement_capacity: float = 0.0
    adoption_visibility: float = 0.5

    def set_local_mask_policy(self, strictness: float) -> None:
        """Set the local policy strictness factor."""
        pass
        self.policy_strictness = _clamp(strictness, 0.0, 1.0)

    def enforce_policy_on_attendees(self, individuals: List[Individual], rng: random.Random) -> int:
        """Enforce policy on attendees; returns number of enforcement actions.

        Note:
            This method is not directly called in the main loop; inline enforcement
            is applied per person to better integrate with local decision context.
        """
        pass
        actions = 0
        if self.policy_strictness <= 0.0 or self.size <= 0:
            return actions
        # Placeholder: approximate enforcement actions based on capacity
        for _ in range(int(self.enforcement_capacity * self.size)):
            if rng.random() < 0.02:
                actions += 1
        return actions

    def communicate_guidelines(self, individuals: List[Individual]) -> None:
        """Communicate guidelines to attendees."""
        pass


@dataclass
class Retailer:
    """Simplified retailer with inventory and pricing."""
    pass
    id: int = 0
    inventory: int = 0
    price: float = 2.0
    restock_rate: int = 0
    supply_allocation_quota: int = 0

    def sell_masks_to_individuals(self, individuals: List[Individual], rng: random.Random) -> int:
        """Sell masks to customers; returns total sold."""
        pass
        sold = 0
        for ind in individuals:
            if self.inventory <= 0:
                break
            if rng.random() < 0.2:
                self.inventory -= 1
                ind.mask_inventory += 1
                sold += 1
        return sold

    def restock_inventory(self) -> None:
        """Restock inventory by restock_rate."""
        pass
        self.inventory += self.restock_rate

    def adjust_price_based_on_inventory(self) -> None:
        """Adjust price upward if low inventory, downward if abundant."""
        pass
        if self.inventory <= max(1, int(self.supply_allocation_quota * 0.3)):
            self.price = min(10.0, self.price * 1.05)
        else:
            self.price = max(0.5, self.price * 0.98)


@dataclass
class Government:
    """Represents government policy and campaign actions."""
    pass
    id: int = 0
    mandate_enabled: bool = False  # FIXED: Gate mandate activation on this flag
    mandate_status: bool = False
    mandate_start_day: int = 0
    # FIXED: Allow optional end day to represent open-ended mandates
    mandate_end_day: Optional[int] = None
    enforcement_probability: float = 0.0
    fine_amount: float = 50.0
    campaign_intensity: float = 0.0
    campaign_targeting_strategy: str = "broadcast"
    budget: float = 0.0

    def issue_or_lift_mandate(self, day: int) -> None:
        """Issue or lift mandate based on schedule (None end means open-ended)."""
        pass
        # FIXED: Properly gate by mandate_enabled and handle open-ended mandates with None end day
        if not self.mandate_enabled:
            self.mandate_status = False
            return
        self.mandate_status = (day >= self.mandate_start_day) and (
            self.mandate_end_day is None or day <= self.mandate_end_day
        )

    def adjust_enforcement(self, new_prob: float) -> None:
        """Adjust enforcement probability."""
        pass
        self.enforcement_probability = _clamp(new_prob, 0.0, 1.0)

    def run_public_health_campaign(self, intensity: float) -> None:
        """Run a public health campaign."""
        pass
        self.campaign_intensity = _clamp(intensity, 0.0, 1.0)

    def allocate_supply_to_retailers(self, retailers: List[Retailer], amount_total: int) -> None:
        """Allocate supply evenly to retailers."""
        pass
        if not retailers or amount_total <= 0:
            return
        per = amount_total // len(retailers)
        for r in retailers:
            r.inventory += per
            r.supply_allocation_quota = per


@dataclass
class RegionEnvironment:
    """Regional environment for prevalence and risk signals."""
    pass
    id: int = 0
    baseline_prevalence_indicator: float = 0.1
    mobility_level: float = 1.0
    seasonality_factor: float = 1.0

    def update_prevalence_signal(self, day: int) -> float:
        """Update a simple prevalence signal with mild seasonality."""
        pass
        # Sine wave seasonality
        seasonal = 0.1 * math.sin(day / 14.0 * 2.0 * math.pi)
        signal = _clamp(self.baseline_prevalence_indicator + seasonal, 0.0, 1.0)
        return signal

    def modulate_risk_perception_signal(self, base_signal: float) -> float:
        """Modulate risk perception based on mobility and seasonality."""
        pass
        return _clamp(base_signal * self.mobility_level * self.seasonality_factor, 0.0, 1.0)


class Simulation:
    """Main simulation class coordinating agents, environment, policies, and metrics."""
    pass

    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the Simulation with the provided configuration."""
        pass
        # FIXED: Merge user config with defaults safely
        self.cfg: Dict[str, Any] = {**DEFAULT_CONFIG, **(cfg or {})}

        # FIXED: Apply model plan if provided to harmonize configuration and metrics dynamically
        self._apply_model_plan(self.cfg.get("model_plan", None))

        self.rng = random.Random(int(self.cfg.get("seed", 42)))
        self.N = max(1, int(self.cfg.get("population_size", 200)))
        # FIXED: Handle None for max_time_steps to avoid int(None) crash
        # If max_time_steps is None/empty/0, fallback to 'days'
        val = self.cfg.get("max_time_steps")
        # FIXED: Cast defensively when value might be string "None" etc.
        self.days = int(self.cfg.get("days", 10)) if val in (None, "", 0, "0", "None", "null") else int(val)

        # Flags first to be available for conditional initialization
        self.with_supply = bool(self.cfg.get("with_supply", False))
        self.with_epi = bool(self.cfg.get("with_epi", False))  # Placeholder not used

        self.people: List[Individual] = []
        self.households: List[Household] = []
        self.workplaces: List[WorkplaceSchool] = []
        self.locations: List[Location] = []  # FIXED: Implement general locations

        # FIXED: Initialize multiple retailers only when supply is enabled
        self.retailers: List[Retailer] = []
        if self.with_supply:
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
                    supply_allocation_quota=0,
                )
                for r in range(retailer_count)
            ]

        # FIXED: Allow open-ended mandate end day (None) and add mandate_enabled gating
        me = self.cfg.get("mandate_end_day", 8)
        self.government = Government(
            mandate_enabled=bool(self.cfg.get("mandate_enabled", False)),  # FIXED: gate activation
            mandate_status=False,
            mandate_start_day=int(self.cfg.get("mandate_start_day", 5)),
            mandate_end_day=(None if me in (None, "None", "null") else int(me)),
            enforcement_probability=float(self.cfg.get("enforcement_probability", 0.1)),
            fine_amount=float(self.cfg.get("fine_amount", 50.0)),
            campaign_intensity=float(self.cfg.get("campaign_intensity", 0.0)),
        )
        self.environment = RegionEnvironment(
            baseline_prevalence_indicator=float(self.cfg.get("risk_perception_baseline", 0.2)),
            mobility_level=1.0,
            seasonality_factor=1.0,
        )
        # Legacy aggregate supply (disabled with multi-retailers)
        self.supply = {"stock": 0, "price": float(self.cfg.get("mask_price", 2.0))}
        self.daily_restock = int(self.cfg.get("daily_supply_restock", 0))

        # Counters and series
        self.series: Dict[str, List[float]] = {
            "adoption_rate": [],
            "mandate_active": [],
            "retailer_stockout_share": [],
            "neighbor_corr": [],
            "daily_gini": [],
            # FIXED: Add retention rate series for behavioral persistence metric
            "retention_rate": [],
        }
        self.metrics: Dict[str, Any] = {}
        self.enforcement_actions: int = 0
        self.policy_cost_total: float = 0.0
        self.stockout_days: int = 0
        self.retailer_stockout_days: int = 0
        # FIXED: Add violation counters for 'violation_rate' metric
        self._violations_under_mandate: int = 0
        self._visits_under_mandate: int = 0
        # FIXED: Track masks sold for metrics alignment
        self.total_masks_sold: int = 0

        # Optional minimal media channel placeholder
        self.media = MediaChannel(id=1, misinformation_rate=float(self.cfg.get("misinformation_prevalence", 0.1)), salience=float(self.cfg.get("campaign_intensity", 0.0)))

        # FIXED: Placeholder for observed series used in calibration metrics
        self._observed_series: Optional[List[float]] = None

    def _apply_model_plan(self, model_plan_input: Optional[Any]) -> None:
        """Apply settings from a model plan (dict or JSON path) to the configuration.

        Supports key mappings from the task specification to this simulation's config.
        """
        pass
        if model_plan_input is None:
            return
        try:
            # Allow both dict and path
            if isinstance(model_plan_input, str):
                plan = safe_load_json(model_plan_input, strict=False)
            elif isinstance(model_plan_input, dict):
                plan = model_plan_input
            else:
                logger.warning("Model plan provided in unsupported type; ignoring.")
                return

            # Merge top-level keys conservatively
            for k, v in plan.items():
                if k not in ("parameters", "evaluation_metrics", "prediction_period"):
                    # Preserve unknown fields for transparency
                    self.cfg[k] = v

            # Map parameters as per model plan definitions
            params = plan.get("parameters", {})
            if params:
                # FIXED: Map population and horizon
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
                    # Map to workplace influence weight as proxy for observation of norms
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
                    self.cfg["mandate_end_day"] = params["mandate_end_day"] if params["mandate_end_day"] is None else int(params["mandate_end_day"])
                if "enforcement_level" in params:
                    self.cfg["enforcement_probability"] = float(params["enforcement_level"])
                if "mask_price" in params:
                    self.cfg["mask_price"] = float(params["mask_price"])
                if "retailer_restock_rate_per_day" in params:
                    # Interpret as fraction of population; convert to absolute number per day across all retailers
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

            # Evaluation metrics from plan
            if "evaluation_metrics" in plan:
                self.cfg["evaluation_metrics"] = plan["evaluation_metrics"]

            # Prediction period mapping
            pred = plan.get("prediction_period", {})
            if isinstance(pred, dict):
                start = pred.get("start_day", None)
                end = pred.get("end_day", None)
                if end is not None:
                    try:
                        self.cfg["days"] = int(end) + 1  # inclusive end
                        self.cfg["max_time_steps"] = int(end) + 1
                    except Exception:
                        pass
                # 'start_day' could be used for slicing metrics; kept for future use
                self.cfg["prediction_period_start"] = start

        except Exception as e:
            logger.warning(f"Failed to apply model plan: {e}")

    def initialize(self) -> None:
        """Initialize population, network, households, workplaces, locations, and initial states."""
        pass
        # Reset counters for a fresh run
        self.enforcement_actions = 0
        self.policy_cost_total = 0.0
        self.stockout_days = 0
        self.retailer_stockout_days = 0
        self._violations_under_mandate = 0
        self._visits_under_mandate = 0
        self.total_masks_sold = 0  # FIXED: reset counter

        init_rate = float(self.cfg.get("initial_adoption_rate", 0.1))
        avg_degree = max(2, int(self.cfg.get("average_degree", 4)))
        misinfo_prev = float(self.cfg.get("misinformation_prevalence", 0.1))
        # Create individuals with mild heterogeneity
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
                current_mask_use=adopted,
                mask_inventory=1 if adopted else 0,
                habit_strength=0.5 if adopted else 0.0,
                mobility_level=_clamp(0.8 + self.rng.uniform(-0.2, 0.3), 0.1, 1.5),
                misinformation_exposure=misinformation_exposure,
            )
            self.people.append(person)

        # FIXED: Implement Watts–Strogatz small-world network with configurable rewiring probability
        net_type = self.cfg.get("social_network_type", "small_world")
        rewire_p = float(self.cfg.get("rewiring_probability", 0.1))
        avg_k = avg_degree if avg_degree % 2 == 0 else avg_degree + 1
        if self.N >= 2 and net_type == "small_world" and avg_k >= 2:
            # Watts–Strogatz construction
            ring = [[] for _ in range(self.N)]
            for i in range(self.N):
                for d in range(1, min(avg_k // 2 + 1, self.N)):
                    j2 = (i + d) % self.N
                    if j2 != i:
                        ring[i].append(j2)
                        ring[j2].append(i)
            # Rewire edges with probability p
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
                # de-duplicate neighbors
                self.people[i].network_neighbors = list(sorted(set(ring[i])))
        else:
            # Fallback to ring lattice for small N or unsupported types
            k = max(2, avg_k)
            for i in range(self.N):
                neighbors = []
                for d in range(1, min(k // 2 + 1, self.N)):
                    neighbors.append((i - d) % self.N)
                    neighbors.append((i + d) % self.N)
                self.people[i].network_neighbors = list(sorted(set(neighbors)))

        # FIXED: Homophily-based rewiring preferring neighbors with similar attitudes
        homophily = float(self.cfg.get("homophily_by_attitude", 0.0))
        if homophily > 0.0:
            for i in range(self.N):
                for j in list(self.people[i].network_neighbors):
                    if i < j and self.rng.random() < homophily:
                        candidates = [k for k in range(self.N) if k != i and k not in self.people[i].network_neighbors]
                        if not candidates:
                            continue
                        target = min(candidates, key=lambda k: abs(self.people[k].perceived_mask_efficacy - self.people[i].perceived_mask_efficacy))
                        try:
                            self.people[i].network_neighbors.remove(j)
                            self.people[j].network_neighbors.remove(i)
                        except ValueError:
                            pass
                        self.people[i].network_neighbors.append(target)
                        self.people[target].network_neighbors.append(i)
            # deduplicate
            for i in range(self.N):
                self.people[i].network_neighbors = list(sorted(set(self.people[i].network_neighbors)))

        # Initialize households (simple grouping of size ~3)
        self.households = []
        household_id = 0
        i = 0
        while i < self.N:
            size = min(3, self.N - i)
            members = list(range(i, i + size))
            hh_income = sum(self.people[m].income for m in members) / max(1, size)
            hh = Household(id=household_id, member_ids=members, household_income=hh_income, household_norms=0.5, shared_mask_stock=0)
            self.households.append(hh)
            for m in members:
                self.people[m].household_id = household_id
            household_id += 1
            i += size

        # FIXED: Initialize workplaces and assign individuals
        num_wp = max(1, int(self.cfg.get("workplace_count", 5)))
        self.workplaces = [
            WorkplaceSchool(
                id=w,
                size=0,
                policy_strictness=float(self.cfg.get("workplace_policy_strictness", 0.0)),
                enforcement_capacity=float(self.cfg.get("workplace_enforcement_capacity", 0.0)),
                adoption_visibility=float(self.cfg.get("workplace_adoption_visibility", 0.5)),
            )
            for w in range(num_wp)
        ]
        for i, person in enumerate(self.people):
            wid = i % num_wp
            person.workplace_id = wid
        for w in self.workplaces:
            w.size = sum(1 for p in self.people if p.workplace_id == w.id)

        # FIXED: Initialize general public locations
        self.locations = []
        loc_counts = self.cfg.get("location_counts", {"stores": 10, "transit": 5, "public_venues": 8})
        req_frac = float(self.cfg.get("location_mask_requirement_fraction", 0.5))
        loc_id = 0
        for loc_type, count in loc_counts.items():
            c = int(count)
            for _ in range(c):
                # Simple parameterization by type
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
                self.locations.append(Location(
                    id=loc_id,
                    loc_type=loc_type,
                    capacity=capacity,
                    crowding_level=crowd,
                    mask_requirement=mask_req,
                    enforcement_strictness=enforce,
                    visibility_of_mask_use=vis,
                    signage_strength=signage,
                ))
                loc_id += 1

        # Initial series values
        self.series["adoption_rate"] = [self._adoption_rate()]
        self.series["mandate_active"] = [1 if self.government.mandate_status else 0]
        self.series["retailer_stockout_share"] = [self._retailer_stockout_share()]
        self.series["neighbor_corr"] = [self._neighbor_influence_corr()]
        self.series["daily_gini"] = [self._adoption_gini_by_income_deciles()]
        self.series["retention_rate"] = [None]  # No prior-day baseline

    def _adoption_rate(self) -> float:
        """Compute current adoption rate."""
        pass
        if self.N <= 0:
            return 0.0
        return sum(1 for p in self.people if p.current_mask_use) / float(self.N)

    def _retailer_stockout_share(self) -> float:
        """Compute fraction of retailers stocked out (inventory <= 0)."""
        pass
        if not self.retailers:
            return 0.0
        return sum(1 for r in self.retailers if r.inventory <= 0) / len(self.retailers)

    def _income_deciles(self) -> List[List[int]]:
        """Compute indices for income deciles."""
        pass
        if self.N <= 0:
            return []
        sorted_indices = sorted(range(self.N), key=lambda idx: self.people[idx].income)
        deciles: List[List[int]] = []
        size = max(1, self.N // 10)
        for i in range(0, self.N, size):
            deciles.append(sorted_indices[i: min(self.N, i + size)])
        # Ensure exactly 10 groups if possible
        if len(deciles) > 10:
            deciles = deciles[:10]
        return deciles

    def _adoption_gini_by_income_deciles(self) -> float:
        """Compute Gini across income decile adoption means."""
        pass
        deciles = self._income_deciles()
        means = []
        for dec in deciles:
            if not dec:
                continue
            mean = sum(1.0 if self.people[i].current_mask_use else 0.0 for i in dec) / len(dec)
            means.append(mean)
        return _gini(means) if means else 0.0

    def _neighbor_influence_corr(self) -> float:
        """Compute correlation between neighbor adoption and individual adoption (proxy)."""
        pass
        if self.N <= 0:
            return 0.0
        xs: List[float] = []
        ys: List[float] = []
        for p in self.people:
            neigh = p.network_neighbors
            neigh_adopt = sum(1 for j in neigh if self.people[j].current_mask_use) / max(1, len(neigh))
            xs.append(neigh_adopt)
            ys.append(1.0 if p.current_mask_use else 0.0)
        return _pearson_correlation(xs, ys)

    def _largest_adopter_component_fraction(self) -> float:
        """Compute the fraction of the population in the largest connected component among adopters."""
        pass
        # FIXED: Use deque for efficient BFS
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

    def _daily_activity_level(self, day: int) -> float:
        """Compute a daily activity multiplier to introduce time-dependent activity peaks.

        Returns:
            A multiplier typically in [0.6, 1.4] reflecting day-specific activity.
        """
        pass
        pattern = str(self.cfg.get("activity_pattern", "weekly_peaks"))
        base = float(self.cfg.get("activity_base", 0.8))
        peak_mult = float(self.cfg.get("activity_peak_multiplier", 1.2))
        if pattern == "weekly_peaks":
            weekday = day % 7  # 0: Mon, ..., 6: Sun
            peaks = set(int(x) % 7 for x in self.cfg.get("activity_weekday_peak", [2, 3, 4]))
            return peak_mult if weekday in peaks else base
        # Default fallback: mild sinusoid
        return _clamp(1.0 + 0.1 * math.sin(2 * math.pi * day / 7.0), 0.6, 1.4)

    def step(self, day: int) -> None:
        """Advance the simulation by one day, updating behaviors and tracking metrics."""
        pass
        # Update mandate status
        self.government.issue_or_lift_mandate(day)

        # Update environment signal
        base_signal = self.environment.update_prevalence_signal(day)
        risk_signal = self.environment.modulate_risk_perception_signal(base_signal)

        # Campaigns influence public perceptions
        self.government.run_public_health_campaign(float(self.cfg.get("campaign_intensity", 0.0)))
        # For simple model, policy_signal is config plus campaign intensity
        policy_signal_global = float(self.cfg.get("policy_signal", 0.0)) + self.government.campaign_intensity

        # Media channel broadcast (applied per-agent later with misinformation scaling)
        dr_global, dt_global = self.media.broadcast(base_signal)

        # FIXED: Retailer stockouts tracked before restock/allocation
        if self.with_supply and self.retailers:
            self.retailer_stockout_days += sum(1 for r in self.retailers if r.inventory <= 0)
            # Government allocation followed by retailer price adjustment
            self.government.allocate_supply_to_retailers(self.retailers, int(self.cfg.get("daily_supply_restock", 0)))
            for r in self.retailers:
                r.adjust_price_based_on_inventory()

        # Track stockout for legacy path (kept for backward compatibility if retailers not used)
        if self.with_supply and not self.retailers and self.supply["stock"] <= 0:
            self.stockout_days += 1
        # Legacy restock
        if self.with_supply and not self.retailers:
            self.supply["stock"] += self.daily_restock

        # Social influence and decisions
        peer_w = float(self.cfg.get("base_influence_weight", self.cfg.get("peer_influence", 0.3)))
        household_w = float(self.cfg.get("household_influence_weight", 0.1))
        workplace_w = float(self.cfg.get("workplace_influence_weight", 0.2))
        habit_form = float(self.cfg.get("habit_formation_rate", 0.1))
        habit_decay = float(self.cfg.get("habit_decay_rate", 0.02))

        # FIXED: Activity multiplier to reflect time-dependent agent activity levels
        activity = self._daily_activity_level(day)
        peer_w_eff = peer_w * activity

        new_states: List[bool] = []
        enforcement_today = 0

        # Precompute previous adoption for visibility estimates
        prev_adoption = self.series.get("adoption_rate", [self._adoption_rate()])[-1]
        vis_strength = float(self.cfg.get("visibility_effect_strength", 0.3))

        case_eff = float(self.cfg.get("case_prevalence_effect_on_risk_perception", 0.4))  # FIXED: scale risk by case effect

        # Prepare a random visits list per person for visibility and compliance
        all_visits: Dict[int, List[int]] = {}
        for i, p in enumerate(self.people):
            # Determine number of visits based on mobility and activity
            expected_visits = p.mobility_level * activity * 2.0
            visits_count = max(0, int(self.rng.random() + expected_visits))  # Randomized rounding
            if self.locations and visits_count > 0:
                chosen = [self.rng.randrange(0, len(self.locations)) for _ in range(visits_count)]
            else:
                chosen = []
            all_visits[i] = chosen

        for i, p in enumerate(self.people):
            # Update individual risk perception slowly toward environmental signal
            p.risk_perception = _clamp(
                p.risk_perception + float(self.cfg.get("risk_perception_update_rate", 0.1)) * (
                    case_eff * risk_signal - p.risk_perception  # FIXED: Scale by case prevalence effect
                ),
                0.0,
                1.0,
            )
            # Perceived efficacy gradual updates
            p.perceived_mask_efficacy = _clamp(
                p.perceived_mask_efficacy + float(self.cfg.get("perceived_efficacy_update_rate", 0.08)) * (
                    policy_signal_global - p.perceived_mask_efficacy
                ),
                0.0,
                1.0,
            )
            # Forgetting in trust
            p.trust_in_authority = _clamp(p.trust_in_authority * (1 - float(self.cfg.get("forgetting_rate", 0.01))), 0.0, 1.0)
            # Media can slightly adjust trust/risk as well per person with misinformation scaling
            p.consume_media_messages(intensity=self.media.salience + dr_global, misinformation_rate=self.media.misinformation_rate)

            # Peer share from network
            peer_share = sum(1 for j in p.network_neighbors if self.people[j].current_mask_use) / max(1, len(p.network_neighbors))
            # Social influence on beliefs
            p.update_beliefs_from_social_influence(peer_share, update_rate=peer_w_eff * 0.1 * p.influenceability)
            p.respond_to_policies_and_campaigns(self.government.campaign_intensity)

            # Workplace local policy signal
            wp_signal = 0.0
            if p.workplace_id is not None and 0 <= p.workplace_id < len(self.workplaces):
                wp = self.workplaces[p.workplace_id]
                wp_signal = wp.policy_strictness * workplace_w * activity

            # FIXED: Use actual retailer price in decision utility and track a last_market_price fallback
            base_price = float(self.cfg.get("mask_price", self.supply.get("price", 2.0)))
            last_market_price = base_price

            # If supply is active and agent has no inventory, attempt purchase from a retailer (modulated by activity)
            if self.with_supply and p.mask_inventory <= 0 and self.retailers and self.rng.random() < activity:
                r = self.rng.choice(self.retailers)
                r_inventory = {"stock": r.inventory}
                _bought = p.purchase_masks_from_retailer(
                    price=r.price,
                    inventory=r_inventory,
                    rng=self.rng,
                    affordability_threshold=float(self.cfg.get("affordability_threshold", 0.05)),  # FIXED: Added affordability threshold
                    price_elasticity=float(self.cfg.get("price_elasticity_of_demand", -0.8)),     # FIXED: Added price elasticity
                )
                r.inventory = r_inventory["stock"]
                last_market_price = r.price
                if _bought > 0:
                    self.total_masks_sold += _bought  # FIXED: Count masks sold
            elif self.with_supply and p.mask_inventory <= 0 and not self.retailers and self.rng.random() < activity:
                qty_bought = p.purchase_masks_from_retailer(price=base_price, inventory=self.supply, rng=self.rng)
                last_market_price = base_price
                if qty_bought > 0:
                    self.total_masks_sold += qty_bought  # FIXED: Count masks sold in legacy path

            # Visibility-adjusted peer share using workplace visibility and visited locations
            peer_share_visible = peer_share
            if p.workplace_id is not None:
                wp = self.workplaces[p.workplace_id]
                # Estimate workplace local mask rate: fraction among co-workers
                co_workers = [idx for idx in range(self.N) if self.people[idx].workplace_id == wp.id]
                if co_workers:
                    loc_rate = sum(1 for j in co_workers if self.people[j].current_mask_use) / max(1, len(co_workers))
                else:
                    loc_rate = prev_adoption
                peer_share_visible = (1 - vis_strength * wp.adoption_visibility) * peer_share + (vis_strength * wp.adoption_visibility) * loc_rate
            # Blend visibility from visited locations
            visits = all_visits.get(i, [])
            for vid in visits:
                loc = self.locations[vid]
                loc_rate_estimate = prev_adoption  # proxy; could be type-specific later
                alpha = _clamp(vis_strength * loc.visibility_of_mask_use, 0.0, 1.0)
                peer_share_visible = (1 - alpha) * peer_share_visible + alpha * ((loc_rate_estimate + loc.signage_strength * 0.2) / (1.0 + 0.2))

            # Decision
            mandate_active = bool(self.government.mandate_status)
            local_policy_signal = policy_signal_global + wp_signal
            decision = p.evaluate_mask_use_decision(
                peer_share=peer_w_eff * peer_share_visible + household_w * self._household_norm_component(p),
                policy_signal=local_policy_signal,
                price=last_market_price,
                mandate_active=mandate_active,
                rng=self.rng,
                habit_weight=0.5,
            )

            # Enforcement if mandate and non-compliant (global)
            if mandate_active and not decision:
                if self.rng.random() < float(self.cfg.get("enforcement_probability", 0.1)):
                    enforcement_today += 1
                    p.compliance_propensity = _clamp(p.compliance_propensity + 0.1, 0.0, 1.0)

            # Workplace-level enforcement with capacity
            if p.workplace_id is not None:
                wp = self.workplaces[p.workplace_id]
                if wp.policy_strictness > 0 and not decision and self.rng.random() < wp.enforcement_capacity:
                    enforcement_today += 1
                    p.compliance_propensity = _clamp(p.compliance_propensity + 0.05, 0.0, 1.0)

            # FIXED: Enforce daily consumption and disallow use without inventory
            use_today = decision
            if self.with_supply:
                if use_today and p.mask_inventory > 0:
                    p.mask_inventory -= 1
                elif use_today and p.mask_inventory <= 0:
                    use_today = False

            # FIXED: Habit formation and decay dynamics
            if use_today:
                p.habit_strength = _clamp(p.habit_strength + habit_form * (1.0 - p.habit_strength), 0.0, 1.0)
            else:
                p.habit_strength = _clamp(p.habit_strength * (1.0 - habit_decay), 0.0, 1.0)

            # FIXED: Location-level compliance and enforcement accounting (per-visit)
            for vid in visits:
                loc = self.locations[vid]
                # Determine if requirement applies: either global mandate or local requirement
                requirement_active = mandate_active or loc.mask_requirement
                if requirement_active:
                    self._visits_under_mandate += 1
                    if not use_today:
                        # Count a violation event
                        self._violations_under_mandate += 1
                        # Potential enforcement at location
                        if loc.enforce_policy(compliant=False, rng=self.rng):
                            enforcement_today += 1
                            # Improve future compliance slightly
                            p.compliance_propensity = _clamp(p.compliance_propensity + 0.05, 0.0, 1.0)

            new_states.append(use_today)

        # Compute daily retention rate before updating states
        prev_maskers = sum(1 for p in self.people if p.current_mask_use)
        still_masking = sum(1 for st, p in zip(new_states, self.people) if p.current_mask_use and st)
        retention_today = (still_masking / prev_maskers) if prev_maskers > 0 else None

        # Apply decisions
        for p, st in zip(self.people, new_states):
            p.current_mask_use = st

        self.enforcement_actions += enforcement_today
        # Campaign cost (placeholder): linear with intensity and enforcement actions
        self.policy_cost_total += 10.0 * self.government.campaign_intensity + 0.5 * enforcement_today

        # Household dynamics
        for hh in self.households:
            hh.intra_household_influence(self.people, weight=household_w)
            hh.share_masks_among_members(self.people)

        # Update daily series
        self.series["adoption_rate"].append(self._adoption_rate())
        self.series["mandate_active"].append(1 if self.government.mandate_status else 0)
        self.series["retailer_stockout_share"].append(self._retailer_stockout_share())
        self.series["neighbor_corr"].append(self._neighbor_influence_corr())
        self.series["daily_gini"].append(self._adoption_gini_by_income_deciles())
        self.series["retention_rate"].append(retention_today)

    def _household_norm_component(self, p: Individual) -> float:
        """Compute a small household norm component based on household average adoption."""
        pass
        if p.household_id is None or not (0 <= p.household_id < len(self.households)):
            return 0.0
        hh = self.households[p.household_id]
        if not hh.member_ids:
            return 0.0
        avg = sum(1.0 if self.people[i].current_mask_use else 0.0 for i in hh.member_ids) / len(hh.member_ids)
        return avg

    def run(self, days: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the simulation for the specified number of days and return the series."""
        pass
        self.initialize()
        total_days = int(days if days is not None else self.days)
        for day in range(total_days):
            self.step(day)
        return self.series

    def _load_observed_series(self, path_opt: Optional[str] = None) -> Optional[List[float]]:
        """Load observed adoption series from CSV with robust parsing.

        The CSV may have either one column of adoption rates or two columns (day, adoption_rate).
        Values will be clamped to [0,1]. Returns None if unavailable.
        """
        pass
        path = path_opt or self.cfg.get("observed_series_path")
        if not path:
            # Default to DATA_DIR/observed_adoption.csv
            # FIXED: Use safe data dir fallback to avoid None join errors
            path = os.path.join(SAFE_DATA_DIR, "observed_adoption.csv")
        try:
            series: List[float] = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # Detect if header contains column names
                if header and any(h.lower() in ("adoption", "adoption_rate", "rate") for h in header):
                    # Map column index
                    try:
                        idx = next(i for i, h in enumerate(header) if h.lower() in ("adoption", "adoption_rate", "rate"))
                    except StopIteration:
                        idx = 1 if len(header) > 1 else 0
                    for row in reader:
                        if not row:
                            continue
                        try:
                            val = float(row[idx])
                            series.append(_clamp(val, 0.0, 1.0))
                        except Exception:
                            continue
                else:
                    # Treat file as plain numeric per line or two columns (day, rate)
                    if header:
                        # Try parse header as numeric
                        try:
                            if len(header) == 1:
                                series.append(_clamp(float(header[0]), 0.0, 1.0))
                            elif len(header) >= 2:
                                series.append(_clamp(float(header[1]), 0.0, 1.0))
                        except Exception:
                            pass
                    for row in reader:
                        if not row:
                            continue
                        try:
                            if len(row) == 1:
                                series.append(_clamp(float(row[0]), 0.0, 1.0))
                            else:
                                series.append(_clamp(float(row[1]), 0.0, 1.0))
                        except Exception:
                            continue
            if not series:
                logger.warning(f"No data parsed from observed series file: {path}")
                return None
            return series
        except FileNotFoundError:
            logger.warning(f"Observed data file not found at {path}. Skipping calibration metrics.")
            return None
        except Exception as e:
            logger.warning(f"Error reading observed data file: {e}. Skipping calibration metrics.")
            return None

    def _compute_calibration_metrics(self, observed: List[float]) -> Dict[str, Any]:
        """Compute calibration metrics (RMSE, MAE, correlation) against observed series."""
        pass
        sim = self.series.get("adoption_rate", [])
        if not sim or not observed:
            return {"calibration_rmse": None, "calibration_mae": None, "calibration_corr": None, "real_vs_sim_len_mismatch": None}
        n = min(len(sim), len(observed))
        if n <= 0:
            return {"calibration_rmse": None, "calibration_mae": None, "calibration_corr": None, "real_vs_sim_len_mismatch": None}
        diffs = [sim[i] - observed[i] for i in range(n)]
        rmse = math.sqrt(sum(d * d for d in diffs) / n)
        mae = sum(abs(d) for d in diffs) / n
        corr = _pearson_correlation(sim[:n], observed[:n])
        mismatch = abs(len(sim) - len(observed))
        return {
            "calibration_rmse": rmse,
            "calibration_mae": mae,
            "calibration_corr": corr,
            "real_vs_sim_len_mismatch": mismatch,
        }

    def evaluate(self, evaluation_metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the simulation results and compute requested metrics."""
        pass
        metrics = evaluation_metrics if evaluation_metrics is not None else self.cfg.get("evaluation_metrics", [])
        results: Dict[str, Any] = {}
        adoption_series = self.series.get("adoption_rate", [])
        if not adoption_series:
            # If not run yet, return empty metrics
            return results

        # FIXED: Provide alias mapping to spec-aligned metric names
        name_map = {
            "overall_adoption_rate": "adoption_rate_over_time",
            "peak_adoption_rate": "peak_adoption",
            "stockout_frequency": "stockout_rate",
            "adoption_inequality": "adoption_inequality_index",
            "network_influence_index": "neighbor_corr_mean",
            "time_to_target_adoption": "time_to_threshold",
            "noncompliance_incidents_per_1000_entries": "noncompliance_incidents_per_1000_entries",
            "mask_shortage_days": "mask_shortage_days",
            # FIXED: Map spec metric names
            "time_to_50_percent_adoption": "time_to_50_percent",
        }
        # Precompute helper aggregates
        neighbor_corr_series = self.series.get("neighbor_corr", [])
        neighbor_corr_mean = sum(neighbor_corr_series) / len(neighbor_corr_series) if neighbor_corr_series else 0.0

        # FIXED: Load observed series on-demand once for calibration metrics
        if any(k in metrics for k in ("calibration_rmse", "calibration_mae", "calibration_corr", "calibration_rmse_below_threshold")):
            if self._observed_series is None:
                self._observed_series = self._load_observed_series(self.cfg.get("observed_series_path"))
            if self._observed_series is not None:
                cal = self._compute_calibration_metrics(self._observed_series)
            else:
                cal = {"calibration_rmse": None, "calibration_mae": None, "calibration_corr": None, "real_vs_sim_len_mismatch": None}
        else:
            cal = {}

        for raw_name in metrics:
            key = name_map.get(raw_name, raw_name)
            if key == "adoption_rate_over_time":
                results[raw_name] = adoption_series
            elif key == "mean_adoption":
                results[raw_name] = sum(adoption_series) / len(adoption_series)
            elif key == "time_to_50_percent":
                # FIXED: Require consecutive days above threshold
                window = int(self.cfg.get("consistency_window", 3))
                t = None
                for i in range(len(adoption_series) - window + 1):
                    if all(v >= 0.5 for v in adoption_series[i: i + window]):
                        t = i
                        break
                results[raw_name] = t
            elif key == "time_to_threshold" or key == "time_to_target_adoption":
                # Compute time to reach target threshold as per config
                target = float(self.cfg.get("target_adoption_threshold", 0.7))
                t = None
                for i, v in enumerate(adoption_series):
                    if v >= target:
                        t = i
                        break
                results[raw_name] = t
            elif key == "peak_adoption":
                peak_val = max(adoption_series)
                peak_day = adoption_series.index(peak_val)
                # FIXED: If spec asks for peak_adoption_rate, return value only; otherwise return dict
                if raw_name == "peak_adoption_rate":
                    results[raw_name] = peak_val
                else:
                    results[raw_name] = {"value": peak_val, "day": peak_day}
            elif key == "adoption_inequality_index":
                # FIXED: Compute inequality over time and average
                daily_gini = self.series.get("daily_gini", [])
                if daily_gini:
                    results[raw_name] = sum(daily_gini) / len(daily_gini)
                else:
                    results[raw_name] = self._adoption_gini_by_income_deciles()
            elif key == "policy_cost":
                results[raw_name] = self.policy_cost_total
            elif key == "enforcement_actions_count":
                results[raw_name] = self.enforcement_actions
            elif key == "stockout_rate":
                # Share of days where any retailer was stocked out (including initial)
                shares = self.series.get("retailer_stockout_share", [])
                if shares:
                    any_out_days = sum(1 for s in shares if s > 0.0)
                    results[raw_name] = any_out_days / len(shares)
                else:
                    results[raw_name] = 0.0
            elif key == "spillover_persistence":
                # Use average retention rate as persistence proxy
                r = [x for x in self.series.get("retention_rate", []) if x is not None]
                results[raw_name] = (sum(r) / len(r)) if r else None
            elif key == "neighbor_corr_mean":
                results[raw_name] = neighbor_corr_mean
            elif key in ("calibration_rmse", "calibration_mae", "calibration_corr"):
                results[raw_name] = cal.get(key)
            elif key == "calibration_rmse_below_threshold":
                thr = float(self.cfg.get("calibration_rmse_threshold", 0.05))
                rmse = cal.get("calibration_rmse", None)
                results[raw_name] = (rmse is not None and rmse <= thr)
            elif key == "noncompliance_incidents_per_1000_entries":
                visits = self._visits_under_mandate
                viol = self._violations_under_mandate
                rate = (1000.0 * viol / visits) if visits > 0 else None
                results[raw_name] = rate
            elif key == "mask_shortage_days":
                shares = self.series.get("retailer_stockout_share", [])
                results[raw_name] = sum(1 for s in shares if s > 0.0)
            elif key == "steady_state_adoption" or key == "sustained_adoption_rate":
                # Average of last 3 days (or fewer if not enough)
                k = min(3, len(adoption_series))
                results[raw_name] = sum(adoption_series[-k:]) / k if k > 0 else None
            elif key == "adoption_disparity_by_income":
                # Difference between top and bottom income decile adoption
                decs = self._income_deciles()
                if len(decs) >= 2:
                    bottom = decs[0]
                    top = decs[-1]
                    b_rate = sum(1 for i in bottom if self.people[i].current_mask_use) / max(1, len(bottom))
                    t_rate = sum(1 for i in top if self.people[i].current_mask_use) / max(1, len(top))
                    results[raw_name] = t_rate - b_rate
                else:
                    results[raw_name] = 0.0
            elif key == "compliance_in_mandated_locations":
                visits = self._visits_under_mandate
                viol = self._violations_under_mandate
                results[raw_name] = (1.0 - (viol / visits)) if visits > 0 else None
            elif key == "masks_sold":
                results[raw_name] = self.total_masks_sold
            elif key == "network_cascade_size":
                results[raw_name] = self._largest_adopter_component_fraction()
            elif key == "retention_rate":
                # Return average retention across days (excluding None)
                r = [x for x in self.series.get("retention_rate", []) if x is not None]
                results[raw_name] = (sum(r) / len(r)) if r else None
            elif key == "violation_rate":
                visits = self._visits_under_mandate
                viol = self._violations_under_mandate
                results[raw_name] = (viol / visits) if visits > 0 else None
            elif key == "policy_effect_size":
                # End minus start adoption
                results[raw_name] = adoption_series[-1] - adoption_series[0]
            else:
                # Unknown metric: return None for transparency
                results[raw_name] = None

        return results

    def save_results(self, path: str) -> None:
        """Save key series to CSV for external use."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # Header
                w.writerow(["day", "adoption_rate", "mandate_active", "retailer_stockout_share", "neighbor_corr", "daily_gini", "retention_rate"])
                rows = max(len(self.series.get("adoption_rate", [])), len(self.series.get("mandate_active", [])))
                for i in range(rows):
                    w.writerow([
                        i,
                        self.series.get("adoption_rate", [None]*rows)[i] if i < len(self.series.get("adoption_rate", [])) else None,
                        self.series.get("mandate_active", [None]*rows)[i] if i < len(self.series.get("mandate_active", [])) else None,
                        self.series.get("retailer_stockout_share", [None]*rows)[i] if i < len(self.series.get("retailer_stockout_share", [])) else None,
                        self.series.get("neighbor_corr", [None]*rows)[i] if i < len(self.series.get("neighbor_corr", [])) else None,
                        self.series.get("daily_gini", [None]*rows)[i] if i < len(self.series.get("daily_gini", [])) else None,
                        self.series.get("retention_rate", [None]*rows)[i] if i < len(self.series.get("retention_rate", [])) else None,
                    ])
        except Exception as e:
            logger.warning(f"Failed to save results to {path}: {e}")

# Execute main for both direct execution and sandbox wrapper invocation
main()