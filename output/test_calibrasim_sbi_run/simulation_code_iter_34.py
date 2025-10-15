import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Setup logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mask_sim")

# Path handling as per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# FIXED: Restore a functional minimal simulation core with deterministic defaults
# FIXED: Parameter alignment: add max_time_steps, social_network_type, rewiring_probability, retailer_count, workplace params, influence weights.
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
    "time_step_unit": "day",
    "workplace_count": 5,
    "workplace_policy_strictness": 0.0,
    "workplace_enforcement_capacity": 0.0,
    "workplace_adoption_visibility": 0.5,
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
    ],
}

# A safe RNG wrapper
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

    def evaluate_mask_use_decision(
        self,
        peer_share: float,
        policy_signal: float,
        price: float,
        mandate_active: bool,
        rng: random.Random,
    ) -> bool:
        """Compute the probability to adopt mask use and sample new decision.

        The utility combines:
        - Peer influence weighted by social_norm_sensitivity
        - Policy/campaign signal weighted by trust_in_authority
        - Risk perception weighted by perceived efficacy
        - Cost pressure weighted by cost_sensitivity and price
        - Mandate adds a compliance boost

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

        util = (0.5 * peer_util) + (0.3 * policy_util) + (0.6 * risk_util) + (0.4 * mandate_boost) - (0.5 * cost_penalty)
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
        """Adjust trust and efficacy slightly in response to campaigns."""
        pass
        self.trust_in_authority = _clamp(self.trust_in_authority + 0.05 * campaign_intensity, 0.0, 1.0)
        self.perceived_mask_efficacy = _clamp(self.perceived_mask_efficacy + 0.04 * campaign_intensity, 0.0, 1.0)

    def purchase_masks_from_retailer(self, price: float, inventory: Dict[str, Any], rng: random.Random) -> int:
        """Attempt to purchase masks, limited by affordability and available inventory.

        Args:
            price: Price per unit mask.
            inventory: Mutable dict-like with 'stock' key to decrement if purchase succeeds.
            rng: Random generator.

        Returns:
            Quantity purchased.
        """
        pass
        if inventory.get("stock", 0) <= 0:
            return 0
        # Simple affordability model
        budget_factor = _clamp(1.0 - self.cost_sensitivity, 0.0, 1.0)
        desired_qty = 1 if rng.random() < (0.5 + 0.5 * budget_factor) else 0
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

    def consume_media_messages(self, intensity: float) -> None:
        """Adjust perceptions due to media messages."""
        pass
        self.risk_perception = _clamp(self.risk_perception + 0.02 * intensity, 0.0, 1.0)


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
class WorkplaceSchool:
    """Represents a workplace or school with a local policy."""
    pass
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
        # FIXED: Properly handle open-ended mandates with None end day
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
        self.rng = random.Random(int(self.cfg.get("seed", 42)))
        self.N = int(self.cfg.get("population_size", 200))
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

        # FIXED: Initialize multiple retailers only when supply is enabled
        self.retailers: List[Retailer] = []
        if self.with_supply:
            retailer_count = int(self.cfg.get("retailer_count", 1))
            total_initial_stock = int(self.cfg.get("supply_initial_stock", 0))
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

        # FIXED: Allow open-ended mandate end day (None)
        me = self.cfg.get("mandate_end_day", 8)
        self.government = Government(
            mandate_status=bool(self.cfg.get("mandate_enabled", False)),
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
        }
        self.metrics: Dict[str, Any] = {}
        self.enforcement_actions: int = 0
        self.policy_cost_total: float = 0.0
        self.stockout_days: int = 0
        self.retailer_stockout_days: int = 0
        # FIXED: Add violation counters for 'violation_rate' metric
        self._violations_under_mandate: int = 0
        self._visits_under_mandate: int = 0

        # Optional minimal media channel placeholder
        self.media = MediaChannel(id=1, misinformation_rate=0.0, salience=float(self.cfg.get("campaign_intensity", 0.0)))

    def initialize(self) -> None:
        """Initialize population, network, households, workplaces, and initial states."""
        pass
        # Reset counters for a fresh run
        self.enforcement_actions = 0
        self.policy_cost_total = 0.0
        self.stockout_days = 0
        self.retailer_stockout_days = 0
        self._violations_under_mandate = 0
        self._visits_under_mandate = 0

        init_rate = float(self.cfg.get("initial_adoption_rate", 0.1))
        avg_degree = max(2, int(self.cfg.get("average_degree", 4)))
        # Create individuals with mild heterogeneity
        self.people = []
        for i in range(self.N):
            adopted = self.rng.random() < init_rate
            income = 30000 + self.rng.random() * 70000
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
            )
            self.people.append(person)

        # FIXED: Implement Watts–Strogatz small-world network with configurable rewiring probability
        net_type = self.cfg.get("social_network_type", "small_world")
        rewire_p = float(self.cfg.get("rewiring_probability", 0.1))
        avg_k = avg_degree if avg_degree % 2 == 0 else avg_degree + 1
        if net_type == "small_world" and avg_k >= 2:
            # Watts–Strogatz construction
            ring = [[] for _ in range(self.N)]
            for i in range(self.N):
                for d in range(1, avg_k // 2 + 1):
                    j2 = (i + d) % self.N
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
            # Fallback to ring lattice
            k = avg_k
            for i in range(self.N):
                neighbors = []
                for d in range(1, k // 2 + 1):
                    neighbors.append((i - d) % self.N)
                    neighbors.append((i + d) % self.N)
                self.people[i].network_neighbors = list(sorted(set(neighbors)))

        # Initialize households (simple grouping of size ~3)
        self.households = []
        household_id = 0
        i = 0
        while i < self.N:
            size = min(3, self.N - i)
            members = list(range(i, i + size))
            hh_income = sum(self.people[m].income for m in members) / size
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

        # Initial series values
        self.series["adoption_rate"] = [self._adoption_rate()]
        self.series["mandate_active"] = [1 if self.government.mandate_status else 0]
        self.series["retailer_stockout_share"] = [self._retailer_stockout_share()]
        self.series["neighbor_corr"] = [self._neighbor_influence_corr()]
        self.series["daily_gini"] = [self._adoption_gini_by_income_deciles()]

    def _adoption_rate(self) -> float:
        """Compute current adoption rate."""
        pass
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
        xs: List[float] = []
        ys: List[float] = []
        for p in self.people:
            neigh = p.network_neighbors
            neigh_adopt = sum(1 for j in neigh if self.people[j].current_mask_use) / max(1, len(neigh))
            xs.append(neigh_adopt)
            ys.append(1.0 if p.current_mask_use else 0.0)
        return _pearson_correlation(xs, ys)

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

        # Media channel small effect (placeholder)
        dr, dt = self.media.broadcast(base_signal)
        policy_signal_global += dr  # treat as extra salience proxy

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
        new_states: List[bool] = []
        enforcement_today = 0

        for i, p in enumerate(self.people):
            # Update individual risk perception slowly toward environmental signal
            p.risk_perception = _clamp(
                p.risk_perception + float(self.cfg.get("risk_perception_update_rate", 0.1)) * (
                    risk_signal - p.risk_perception
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
            # Media can slightly adjust trust as well
            p.trust_in_authority = _clamp(p.trust_in_authority + dt, 0.0, 1.0)

            # Peer share
            peer_share = sum(1 for j in p.network_neighbors if self.people[j].current_mask_use) / max(1, len(p.network_neighbors))
            # FIXED: Influence beliefs (scale by peer influence weight to support sanity check)
            p.update_beliefs_from_social_influence(peer_share, update_rate=peer_w * 0.1 * p.influenceability)
            p.respond_to_policies_and_campaigns(self.government.campaign_intensity)

            # Workplace local policy signal
            wp_signal = 0.0
            if p.workplace_id is not None and 0 <= p.workplace_id < len(self.workplaces):
                wp = self.workplaces[p.workplace_id]
                wp_signal = wp.policy_strictness * workplace_w

            # FIXED: Use actual retailer price in decision utility and track a last_market_price fallback
            base_price = float(self.cfg.get("mask_price", self.supply.get("price", 2.0)))
            last_market_price = base_price

            # If supply is active and agent has no inventory, attempt purchase from a retailer
            if self.with_supply and p.mask_inventory <= 0 and self.retailers:
                # Choose a random retailer (could extend to choose by price)
                r = self.rng.choice(self.retailers)
                r_inventory = {"stock": r.inventory}
                _bought = p.purchase_masks_from_retailer(price=r.price, inventory=r_inventory, rng=self.rng)
                r.inventory = r_inventory["stock"]
                last_market_price = r.price
            elif self.with_supply and p.mask_inventory <= 0 and not self.retailers:
                p.purchase_masks_from_retailer(price=base_price, inventory=self.supply, rng=self.rng)
                last_market_price = base_price

            # Decision
            mandate_active = bool(self.government.mandate_status)
            local_policy_signal = policy_signal_global + wp_signal
            decision = p.evaluate_mask_use_decision(
                peer_share=peer_w * peer_share + household_w * self._household_norm_component(p),
                policy_signal=local_policy_signal,
                price=last_market_price,
                mandate_active=mandate_active,
                rng=self.rng,
            )

            # Enforcement if mandate and non-compliant
            if mandate_active and not decision:
                if self.rng.random() < float(self.cfg.get("enforcement_probability", 0.1)):
                    enforcement_today += 1
                    p.compliance_propensity = _clamp(p.compliance_propensity + 0.1, 0.0, 1.0)
            # FIXED: Workplace-level enforcement with capacity
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

            # FIXED: Track violations under mandate (visit proxy: one workplace/public visit per day)
            if mandate_active:
                self._visits_under_mandate += 1
                if not use_today:
                    self._violations_under_mandate += 1

            new_states.append(use_today)

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
            "time_to_target_adoption": "time_to_target_adoption",  # handled specially
        }
        # Precompute helper aggregates
        neighbor_corr_series = self.series.get("neighbor_corr", [])
        neighbor_corr_mean = sum(neighbor_corr_series) / len(neighbor_corr_series) if neighbor_corr_series else 0.0

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
            elif key == "time_to_target_adoption":
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
                # FIXED: Use retailer-day stockouts when retailers are active
                if self.with_supply and self.retailers:
                    denom = max(1, len(self.retailers) * max(1, self.days))
                    results[raw_name] = self.retailer_stockout_days / denom
                else:
                    total = max(1, self.days)
                    results[raw_name] = self.stockout_days / total if self.with_supply else 0.0
            elif key == "spillover_persistence":
                # Difference between final adoption and mean adoption during last 2 mandate days
                if self.government.mandate_end_day is not None:
                    end_day = min(self.government.mandate_end_day, len(adoption_series) - 1)
                    start_day = max(0, end_day - 2)
                    during = adoption_series[start_day: end_day + 1]
                    final = adoption_series[-1]
                    results[raw_name] = final - (sum(during) / len(during) if during else 0.0)
                else:
                    results[raw_name] = None
            elif key == "sustained_adoption_rate":
                # mean adoption after mandate end or after day threshold
                end = self.government.mandate_end_day
                start_idx = (end + 1) if (end is not None and (end + 1) < len(adoption_series)) else max(0, len(adoption_series) - 7)
                tail = adoption_series[start_idx:]
                results[raw_name] = (sum(tail) / len(tail)) if tail else None
            elif key == "violation_rate":
                denom = max(1, self._visits_under_mandate)
                results[raw_name] = self._violations_under_mandate / denom
            elif key == "policy_effect_size":
                # FIXED: Simple pre/post difference adjusted by pre-trend
                start = self.government.mandate_start_day
                pre_end = max(0, min(start - 1, len(adoption_series) - 1))
                pre_start = max(0, pre_end - 6)
                post_start = min(len(adoption_series) - 1, start + 1)
                post_end = min(len(adoption_series) - 1, post_start + 6)
                pre = adoption_series[pre_start: pre_end + 1]
                post = adoption_series[post_start: post_end + 1]
                pre_mean = sum(pre) / len(pre) if pre else 0.0
                post_mean = sum(post) / len(post) if post else 0.0
                # Approximate pre-trend slope
                if len(pre) >= 2:
                    slope = (pre[-1] - pre[0]) / (len(pre) - 1)
                else:
                    slope = 0.0
                expected_post = pre[-1] + slope * (len(post) - 1) if pre else 0.0
                results[raw_name] = (post_mean - pre_mean) - (expected_post - (pre[-1] if pre else 0.0))
            elif key == "neighbor_corr_mean":
                results[raw_name] = neighbor_corr_mean
            else:
                results[raw_name] = None  # Unknown metric placeholder

        self.metrics = results
        return results

    def save_results(self, filename: str) -> None:
        """Save time series results to a CSV file."""
        pass
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["day", "adoption_rate", "mandate_active", "retailer_stockout_share", "neighbor_corr", "daily_gini"])
                series_len = len(self.series.get("adoption_rate", []))
                for day in range(series_len):
                    writer.writerow([
                        day,
                        self.series.get("adoption_rate", [None])[day],
                        self.series.get("mandate_active", [None])[day] if day < len(self.series.get("mandate_active", [])) else None,
                        self.series.get("retailer_stockout_share", [None])[day] if day < len(self.series.get("retailer_stockout_share", [])) else None,
                        self.series.get("neighbor_corr", [None])[day] if day < len(self.series.get("neighbor_corr", [])) else None,
                        self.series.get("daily_gini", [None])[day] if day < len(self.series.get("daily_gini", [])) else None,
                    ])
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to {filename}: {e}")

    def visualize(self, out_path: Optional[str] = None) -> Optional[str]:
        """Visualize the adoption rate over time as a plot.

        Returns:
            Path to saved figure if created, else None.
        """
        pass
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            logger.warning(f"Matplotlib not available: {e}")
            return None
        series = self.series.get("adoption_rate", [])
        if not series:
            return None
        plt.figure(figsize=(6, 3))
        plt.plot(series, marker="o")
        plt.title("Mask Adoption Rate Over Time")
        plt.xlabel(f"Time ({self.cfg.get('time_step_unit', 'day')})")
        plt.ylabel("Adoption rate")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        path = out_path or os.path.join(PROJECT_ROOT, "simulation_plot.png")
        try:
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            logger.info(f"Saved plot to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            return None

    def validate(self) -> Dict[str, Any]:
        """Run simple validation scenarios to check core behavioral properties.

        Returns:
            A dict with boolean flags and diagnostic notes for each validation.
        """
        pass
        report: Dict[str, Any] = {}

        # Helper to run a small sim clone
        def run_cfg(base: Dict[str, Any]) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
            sim = Simulation(base)
            results = sim.run()
            metrics = sim.evaluate([
                "time_to_50_percent",
                "mean_adoption",
                "policy_effect_size",
                "violation_rate",
                "sustained_adoption_rate",
            ])
            return results, metrics

        base_cfg = {**DEFAULT_CONFIG, **{
            "population_size": 120,
            "max_time_steps": 40,
            "initial_adoption_rate": 0.05,
            "mandate_enabled": False,
            "with_supply": True,
            "retailer_count": 2,
            "supply_initial_stock": 200,
            "daily_supply_restock": 20,
            "campaign_intensity": 0.0,
        }}

        # 1) S-curve increase under higher peer influence
        cfg_low = {**base_cfg, "base_influence_weight": 0.0}
        cfg_high = {**base_cfg, "base_influence_weight": 0.6}
        _, m_low = run_cfg(cfg_low)
        _, m_high = run_cfg(cfg_high)
        scurve_ok = (m_high.get("time_to_50_percent") is not None) and (
            (m_low.get("time_to_50_percent") is None or m_high["time_to_50_percent"] < m_low["time_to_50_percent"])
        )
        report["s_curve_peer_influence"] = {"ok": bool(scurve_ok), "low": m_low, "high": m_high}

        # 2) Monotonic policy response: higher enforcement increases mean adoption
        cfg_lo_enf = {**base_cfg, "mandate_enabled": True, "mandate_start_day": 5, "mandate_end_day": 20, "enforcement_probability": 0.05}
        cfg_hi_enf = {**cfg_lo_enf, "enforcement_probability": 0.5}
        _, m_lo = run_cfg(cfg_lo_enf)
        _, m_hi = run_cfg(cfg_hi_enf)
        monotonic_ok = m_hi.get("mean_adoption", 0) >= m_lo.get("mean_adoption", 0)
        report["monotonic_policy_response"] = {"ok": bool(monotonic_ok), "low_enf": m_lo, "high_enf": m_hi}

        # 3) No-influence sanity: with zero social/campaign/risk updates, adoption stays near initial
        cfg_no_infl = {
            **base_cfg,
            "base_influence_weight": 0.0,
            "campaign_intensity": 0.0,
            "risk_perception_update_rate": 0.0,
            "perceived_efficacy_update_rate": 0.0,
            "mandate_enabled": False,
        }
        res_no, m_no = run_cfg(cfg_no_infl)
        series_no = res_no.get("adoption_rate", [])
        initial = series_no[0] if series_no else 0.0
        final = series_no[-1] if series_no else 0.0
        no_infl_ok = abs(final - initial) < 0.1
        report["no_influence_sanity_check"] = {"ok": bool(no_infl_ok), "initial": initial, "final": final}

        # 4) Stock dependency: with zero stock/restock, adoption cannot grow beyond initial
        cfg_stock0 = {
            **base_cfg,
            "with_supply": True,
            "supply_initial_stock": 0,
            "daily_supply_restock": 0,
        }
        res_s0, m_s0 = run_cfg(cfg_stock0)
        series_s0 = res_s0.get("adoption_rate", [])
        final_s0 = series_s0[-1] if series_s0 else 0.0
        stock_ok = final_s0 <= (series_s0[0] if series_s0 else 0.0) + 1e-6
        report["stock_dependency"] = {"ok": bool(stock_ok), "series": series_s0, "metrics": m_s0}

        # Summarize
        report["all_passed"] = all(v.get("ok", False) for k, v in report.items() if isinstance(v, dict) and "ok" in v)
        return report


def safe_load_json(path_or_dash: Optional[str], strict: bool = False) -> Dict[str, Any]:
    """Safely load JSON from a file path or stdin ('-') with robust error handling.

    Args:
        path_or_dash: Path to JSON config or '-' to read from stdin. If None, returns {}.
        strict: If True, raise errors on parse issues; otherwise return {} and warn.

    Returns:
        A dict with parsed JSON, or {} if parsing failed in non-strict mode.
    """
    pass
    if not path_or_dash:
        return {}
    try:
        if path_or_dash == "-":
            text = sys.stdin.read()
            return json.loads(text)
        else:
            with open(path_or_dash, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        msg = f"JSON parse error at pos {e.pos}: {e.msg}. Use --strict-json to fail."
        if strict:
            logger.error(msg)
            raise
        logger.warning(msg + " Falling back to defaults.")
        return {}
    except FileNotFoundError:
        if strict:
            raise
        logger.warning("Config file not found; using defaults.")
        return {}
    except Exception as e:
        if strict:
            raise
        logger.warning(f"Unexpected error loading config: {e}. Using defaults.")
        return {}


def docker_available() -> bool:
    """Check whether Docker is available in PATH."""
    pass
    return shutil.which("docker") is not None


def try_docker_version() -> str:
    """Try to get Docker version string if available."""
    pass
    if not docker_available():
        return "not_available"
    import subprocess

    try:
        out = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "error_invoking_docker"


def _self_test() -> Dict[str, Any]:
    """Run a quick self-test simulation to ensure core functionality.

    Returns:
        A dict with sentinel and summary fields.
    """
    pass
    cfg = {
        "population_size": 50,
        "max_time_steps": 5,
        "initial_adoption_rate": 0.05,
        "peer_influence": 0.4,
        "policy_signal": 0.1,
        "seed": 123,
        "with_supply": True,
        "retailer_count": 2,
        "supply_initial_stock": 10,
        "daily_supply_restock": 2,
        "social_network_type": "small_world",
    }
    sim = Simulation(cfg)
    results = sim.run()
    metrics = sim.evaluate(["mean_adoption", "peak_adoption", "stockout_rate"])
    return {
        "status": "SELFTEST_OK",
        "results_summary": {
            "series_len": len(results.get("adoption_rate", [])),
            "final_adoption": results.get("adoption_rate", [None])[-1] if results.get("adoption_rate", []) else None,
            "mean_adoption": metrics.get("mean_adoption"),
            "peak_adoption": metrics.get("peak_adoption"),
            "stockout_rate": metrics.get("stockout_rate"),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the mask adoption simulation.

    Supports:
    - Robust JSON config loading
    - Self-test mode with a clear sentinel
    - Pure-Python execution by default; Docker checks behind a flag
    - Saving results and optional visualization
    - Validation diagnostics behind a flag
    """
    pass
    # FIXED: Restore CLI with argparse and robust options
    parser = argparse.ArgumentParser(description="Mask adoption dynamics simulation (enhanced core).")
    parser.add_argument("--config", default=None, help="Path to JSON config or '-' to read from stdin.")
    parser.add_argument("--strict-json", action="store_true", help="Fail on JSON parse errors.")
    parser.add_argument("--self-test", action="store_true", help="Run a quick self-test and exit 0 on success.")
    parser.add_argument("--use-docker", action="store_true", help="Guarded Docker usage (unused by default).")
    parser.add_argument("--save", default=os.path.join(PROJECT_ROOT, "results.csv"), help="Path to save CSV results.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of the adoption rate.")
    parser.add_argument("--validate", action="store_true", help="Run validation scenarios and print a report.")
    args = parser.parse_args(argv)

    if args.self_test:
        payload = _self_test()
        print(json.dumps(payload))
        return 0

    cfg = safe_load_json(args.config, strict=args.strict_json) if args.config else {}
    if args.use_docker:
        if not docker_available():
            logger.warning("Docker not available; continuing in pure-Python mode.")
        else:
            logger.info(f"Docker is available: {try_docker_version()} (not used by default).")

    # Run simulation
    sim = Simulation(cfg)
    # FIXED: Use max_time_steps if present; fallback to config 'days' and then default, handling None safely
    val = cfg.get("max_time_steps")
    runtime_days = int(cfg.get("days", DEFAULT_CONFIG["days"])) if val in (None, "", 0, "0", "None", "null") else int(val)
    results = sim.run(runtime_days)
    # Evaluate common metrics
    evaluation = sim.evaluate(cfg.get("evaluation_metrics", DEFAULT_CONFIG["evaluation_metrics"]))

    # Output JSON result to stdout
    print(json.dumps({"results": results, "metrics": evaluation}))

    # Demonstrate save_results as required
    sim.save_results(args.save)

    # Optional visualization
    if args.plot:
        sim.visualize()

    # Optional validation
    if args.validate:
        report = sim.validate()
        print(json.dumps({"validation_report": report}))

    return 0


# FIXED: Remove duplicate main() call; ensure a single invocation for sandbox compatibility
# Execute main for both direct execution and sandbox wrapper invocation
main()