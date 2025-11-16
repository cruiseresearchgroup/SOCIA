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
DEFAULT_CONFIG: Dict[str, Any] = {
    "population_size": 200,
    "days": 10,
    "initial_adoption_rate": 0.1,
    "peer_influence": 0.3,
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
    "with_supply": False,
    "with_epi": False,
    "average_degree": 2,
    "time_step_unit": "day",
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
    ],
}

# A safe RNG wrapper
def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a float between lo and hi."""
    pass
    return max(lo, min(hi, x))


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
        cost_penalty = self.cost_sensitivity * (price / max(0.1, price + 1.0))  # bounded [~0, ~0.66]
        mandate_boost = self.compliance_propensity * (1.0 if mandate_active else 0.0)

        util = (0.5 * peer_util) + (0.3 * policy_util) + (0.6 * risk_util) + (0.4 * mandate_boost) - (0.5 * cost_penalty)
        util = _clamp(util, -5.0, 5.0)
        # Logistic mapping to probability with slight noise
        noise = rng.uniform(-0.1, 0.1)
        prob = 1.0 / (1.0 + math.exp(-(util + noise)))
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

    def intra_household_influence(self, individuals: List[Individual]) -> None:
        """Adjust household norms to average of member mask use."""
        pass
        if not self.member_ids:
            return
        avg_use = sum(1.0 if individuals[i].current_mask_use else 0.0 for i in self.member_ids) / len(self.member_ids)
        self.household_norms = _clamp(0.7 * self.household_norms + 0.3 * avg_use, 0.0, 1.0)

    def share_masks_among_members(self, individuals: List[Individual]) -> None:
        """Share masks if some members lack masks."""
        pass
        for i in self.member_ids:
            ind = individuals[i]
            if ind.mask_inventory == 0 and self.shared_mask_stock > 0:
                ind.mask_inventory += 1
                self.shared_mask_stock -= 1


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
        """Enforce policy on attendees; returns number of enforcement actions."""
        pass
        actions = 0
        if self.policy_strictness <= 0.0:
            return actions
        # Placeholder: not integrated with attendance schedules
        for _ in range(int(self.enforcement_capacity * self.size)):
            if rng.random() < 0.05:
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
        if self.inventory <= self.supply_allocation_quota * 0.3:
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
    mandate_end_day: int = 1000000
    enforcement_probability: float = 0.0
    fine_amount: float = 50.0
    campaign_intensity: float = 0.0
    campaign_targeting_strategy: str = "broadcast"
    budget: float = 0.0

    def issue_or_lift_mandate(self, day: int) -> None:
        """Issue or lift mandate based on schedule."""
        pass
        self.mandate_status = self.mandate_start_day <= day <= self.mandate_end_day

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
        if not retailers:
            return
        per = amount_total // len(retailers)
        for r in retailers:
            r.inventory += per


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
        self.days = int(self.cfg.get("days", 10))
        self.people: List[Individual] = []
        self.households: List[Household] = []
        self.workplaces: List[WorkplaceSchool] = []
        self.retailers: List[Retailer] = []
        self.government = Government(
            mandate_status=bool(self.cfg.get("mandate_enabled", False)),
            mandate_start_day=int(self.cfg.get("mandate_start_day", 5)),
            mandate_end_day=int(self.cfg.get("mandate_end_day", 8)),
            enforcement_probability=float(self.cfg.get("enforcement_probability", 0.1)),
            fine_amount=float(self.cfg.get("fine_amount", 50.0)),
            campaign_intensity=float(self.cfg.get("campaign_intensity", 0.0)),
        )
        self.environment = RegionEnvironment(
            baseline_prevalence_indicator=float(self.cfg.get("risk_perception_baseline", 0.2)),
            mobility_level=1.0,
            seasonality_factor=1.0,
        )
        # Supply model (simplified)
        self.supply = {"stock": int(self.cfg.get("supply_initial_stock", 0)), "price": float(self.cfg.get("mask_price", 2.0))}
        self.daily_restock = int(self.cfg.get("daily_supply_restock", 0))
        self.with_supply = bool(self.cfg.get("with_supply", False))
        self.with_epi = bool(self.cfg.get("with_epi", False))  # Placeholder not used

        # Results
        self.series: Dict[str, List[float]] = {"adoption_rate": []}
        self.metrics: Dict[str, Any] = {}
        self.enforcement_actions: int = 0
        self.policy_cost_total: float = 0.0
        self.stockout_days: int = 0

    def initialize(self) -> None:
        """Initialize population, network, and initial states."""
        pass
        init_rate = float(self.cfg.get("initial_adoption_rate", 0.1))
        avg_degree = max(1, int(self.cfg.get("average_degree", 2)))
        # Create individuals with mild heterogeneity
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
        # Build a simple ring network with given average degree (even)
        k = avg_degree if avg_degree % 2 == 0 else avg_degree + 1
        for i in range(self.N):
            neighbors = []
            for d in range(1, k // 2 + 1):
                neighbors.append((i - d) % self.N)
                neighbors.append((i + d) % self.N)
            self.people[i].network_neighbors = neighbors

        # Initialize households (simple grouping of size ~3)
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

        # Initialize workplaces (unused placeholders)
        self.workplaces.append(WorkplaceSchool(id=0, size=self.N, policy_strictness=0.0, enforcement_capacity=0.0, adoption_visibility=0.5))

        # Initial adoption rate
        self.series["adoption_rate"].append(self._adoption_rate())

    def _adoption_rate(self) -> float:
        """Compute current adoption rate."""
        pass
        return sum(1 for p in self.people if p.current_mask_use) / float(self.N)

    def _income_deciles(self) -> List[List[int]]:
        """Compute indices for income deciles."""
        pass
        sorted_indices = sorted(range(self.N), key=lambda idx: self.people[idx].income)
        deciles: List[List[int]] = []
        size = max(1, self.N // 10)
        for i in range(0, self.N, size):
            deciles.append(sorted_indices[i : min(self.N, i + size)])
        return deciles

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
        policy_signal = float(self.cfg.get("policy_signal", 0.0)) + self.government.campaign_intensity

        # Supply restock daily at the start of day
        if self.with_supply:
            self.supply["stock"] += self.daily_restock

        # Track stockout
        if self.with_supply and self.supply["stock"] <= 0:
            self.stockout_days += 1

        # Social influence and decisions
        peer_w = float(self.cfg.get("peer_influence", 0.3))
        new_states: List[bool] = []
        enforcement_today = 0

        for i, p in enumerate(self.people):
            # Update individual risk perception slowly toward environmental signal
            p.risk_perception = _clamp(
                p.risk_perception + float(self.cfg.get("risk_perception_update_rate", 0.1)) * (risk_signal - p.risk_perception),
                0.0,
                1.0,
            )
            # Perceived efficacy gradual updates
            p.perceived_mask_efficacy = _clamp(
                p.perceived_mask_efficacy + float(self.cfg.get("perceived_efficacy_update_rate", 0.08)) * (policy_signal - p.perceived_mask_efficacy),
                0.0,
                1.0,
            )
            # Forgetting
            p.trust_in_authority = _clamp(p.trust_in_authority * (1 - float(self.cfg.get("forgetting_rate", 0.01))), 0.0, 1.0)

            # Peer share
            peer_share = sum(1 for j in p.network_neighbors if self.people[j].current_mask_use) / max(1, len(p.network_neighbors))
            # Influence beliefs
            p.update_beliefs_from_social_influence(peer_share, update_rate=0.1 * p.influenceability)
            p.respond_to_policies_and_campaigns(self.government.campaign_intensity)

            # If supply is active and agent wants to adopt but has no inventory, attempt purchase
            price = float(self.supply["price"])
            if self.with_supply and p.mask_inventory <= 0:
                p.purchase_masks_from_retailer(price=price, inventory=self.supply, rng=self.rng)

            mandate_active = bool(self.government.mandate_status)
            # Decision
            decision = p.evaluate_mask_use_decision(
                peer_share=peer_w * peer_share,
                policy_signal=policy_signal,
                price=price,
                mandate_active=mandate_active,
                rng=self.rng,
            )

            # Enforcement if mandate and non-compliant
            if mandate_active and not decision:
                if self.rng.random() < float(self.cfg.get("enforcement_probability", 0.1)):
                    enforcement_today += 1
                    # Simple behavior: enforcement nudges compliance next time
                    p.compliance_propensity = _clamp(p.compliance_propensity + 0.1, 0.0, 1.0)

            # If adopting and with supply, consume mask inventory
            if decision and self.with_supply and p.mask_inventory > 0 and not p.current_mask_use:
                p.mask_inventory -= 1

            new_states.append(decision)

        # Apply decisions
        for p, st in zip(self.people, new_states):
            p.current_mask_use = st

        self.enforcement_actions += enforcement_today
        # Campaign cost (placeholder): linear with intensity
        self.policy_cost_total += 10.0 * self.government.campaign_intensity + 0.5 * enforcement_today

        # Household dynamics
        for hh in self.households:
            hh.intra_household_influence(self.people)
            hh.share_masks_among_members(self.people)

        # Adoption rate
        self.series["adoption_rate"].append(self._adoption_rate())

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

        for name in metrics:
            if name == "adoption_rate_over_time":
                results[name] = adoption_series
            elif name == "mean_adoption":
                results[name] = sum(adoption_series) / len(adoption_series)
            elif name == "time_to_50_percent":
                t = next((i for i, v in enumerate(adoption_series) if v >= 0.5), None)
                results[name] = t
            elif name == "peak_adoption":
                peak_val = max(adoption_series)
                peak_day = adoption_series.index(peak_val)
                results[name] = {"value": peak_val, "day": peak_day}
            elif name == "adoption_inequality_index":
                # Compute decile adoption means and Gini among deciles
                deciles = self._income_deciles()
                means = []
                for dec in deciles:
                    if not dec:
                        continue
                    mean = sum(1.0 if self.people[i].current_mask_use else 0.0 for i in dec) / len(dec)
                    means.append(mean)
                # Gini for means
                if means:
                    means_sorted = sorted(means)
                    n = len(means_sorted)
                    cum = 0.0
                    for i, x in enumerate(means_sorted, start=1):
                        cum += i * x
                    total = sum(means_sorted)
                    gini = (2 * cum) / (n * total) - (n + 1) / n if total > 0 else 0.0
                    results[name] = _clamp(gini, 0.0, 1.0)
                else:
                    results[name] = 0.0
            elif name == "policy_cost":
                results[name] = self.policy_cost_total
            elif name == "enforcement_actions_count":
                results[name] = self.enforcement_actions
            elif name == "stockout_rate":
                total = max(1, self.days)
                results[name] = self.stockout_days / total if self.with_supply else 0.0
            elif name == "spillover_persistence":
                # Difference between final adoption and mean adoption during last 2 mandate days
                if self.government.mandate_end_day is not None:
                    end_day = min(self.government.mandate_end_day, len(adoption_series) - 1)
                    start_day = max(0, end_day - 2)
                    during = adoption_series[start_day : end_day + 1]
                    final = adoption_series[-1]
                    results[name] = final - (sum(during) / len(during) if during else 0.0)
                else:
                    results[name] = None
            else:
                results[name] = None  # Unknown metric placeholder

        self.metrics = results
        return results

    def save_results(self, filename: str) -> None:
        """Save time series results to a CSV file."""
        pass
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["day", "adoption_rate"])
                series = self.series.get("adoption_rate", [])
                for day, val in enumerate(series):
                    writer.writerow([day, val])
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
        "days": 5,
        "initial_adoption_rate": 0.05,
        "peer_influence": 0.4,
        "policy_signal": 0.1,
        "seed": 123,
        "with_supply": False,
    }
    sim = Simulation(cfg)
    results = sim.run()
    metrics = sim.evaluate(["mean_adoption", "peak_adoption"])
    return {
        "status": "SELFTEST_OK",
        "results_summary": {
            "series_len": len(results.get("adoption_rate", [])),
            "final_adoption": results.get("adoption_rate", [None])[-1] if results.get("adoption_rate", []) else None,
            "mean_adoption": metrics.get("mean_adoption"),
            "peak_adoption": metrics.get("peak_adoption"),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the mask adoption simulation.

    Supports:
    - Robust JSON config loading
    - Self-test mode with a clear sentinel
    - Pure-Python execution by default; Docker checks behind a flag
    - Saving results and optional visualization
    """
    pass
    # FIXED: Restore CLI with argparse and robust options
    parser = argparse.ArgumentParser(description="Mask adoption dynamics simulation (minimal functional core).")
    parser.add_argument("--config", default=None, help="Path to JSON config or '-' to read from stdin.")
    parser.add_argument("--strict-json", action="store_true", help="Fail on JSON parse errors.")
    parser.add_argument("--self-test", action="store_true", help="Run a quick self-test and exit 0 on success.")
    parser.add_argument("--use-docker", action="store_true", help="Guarded Docker usage (unused by default).")
    parser.add_argument("--save", default=os.path.join(PROJECT_ROOT, "results.csv"), help="Path to save CSV results.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG plot of the adoption rate.")
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
    results = sim.run(int(cfg.get("days", DEFAULT_CONFIG["days"])))
    # Evaluate common metrics
    evaluation = sim.evaluate(cfg.get("evaluation_metrics", DEFAULT_CONFIG["evaluation_metrics"]))

    # Output JSON result to stdout
    print(json.dumps({"results": results, "metrics": evaluation}))

    # Demonstrate save_results as required
    sim.save_results(args.save)

    # Optional visualization
    if args.plot:
        sim.visualize()

    return 0


# Execute main for both direct execution and sandbox wrapper invocation
# FIXED: Retain unconditional main() call for sandbox compatibility, return code captured
exit_code = main()

# Execute main for both direct execution and sandbox wrapper invocation
main()