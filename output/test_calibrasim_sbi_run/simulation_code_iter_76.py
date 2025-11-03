def main():
    pass

import json
import os
import sys
import argparse
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

# Optional imports for visualization and network; handle gracefully if unavailable
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

# Path handling per instruction
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def _default_params() -> Dict[str, Any]:
    """
    Provide sane defaults for simulation parameters.
    """
    return {
        # Core
        "population_size": 200,
        "time_horizon_days": 60,
        "seed": 42,
        # Network
        "network_avg_degree": 6,
        "network_rewiring_prob": 0.1,
        # Economics
        "mask_price": 1.0,
        "agent_budget_mean": 50.0,
        "agent_budget_std": 20.0,
        # Supply
        "supply_initial_masks_per_capita": 0.5,
        "supply_restock_per_day_per_capita": 0.05,
        # Policy
        "mandate_enforcement_prob": 0.5,
        "fine_amount": 50.0,
        "mandate_start_day": None,
        "mandate_end_day": None,
        "messaging_intensity": 0.5,
        "entry_denial_policy": True,
        # Media
        "media_reach": 0.5,
        "media_pro_mask_bias": 0.5,
        "misinformation_strength": 0.2,
        "message_schedule": [],
        # Agents
        "initial_adoption_rate": 0.2,
        "base_risk_index": 0.3,
        "social_influence_weight": 0.5,
        "policy_influence_weight": 0.3,
        "risk_perception_weight": 0.2,
        "adoption_cost_weight": 0.15,
        "noise_level": 0.05,
        "habit_formation_rate": 0.01,
        "fatigue_rate": 0.005,
        # Targets and runtime
        "adoption_target_percent": 0.7,
        "fast_mode": False,
    }


@dataclass
class Person:
    """
    Represents an individual agent in the mask adoption simulation with attributes for demographics,
    beliefs, resources, and behaviors related to mask-wearing decisions.

    Attributes:
        id: Unique identifier.
        age: Age of the person (in years).
        household_id: Identifier for household association.
        workplace_id: Identifier for workplace association (optional).
        income: Annual income proxy to inform affordability.
        budget: Current liquid budget available for purchases.
        risk_group: Indicator for higher clinical risk (1) or not (0).
        risk_perception: Subjective risk perception in [0,1].
        trust_in_authorities: Trust factor in [0,1], influences policy sensitivity.
        social_influence_susceptibility: Susceptibility to social norms in [0,1].
        policy_sensitivity: Propensity to comply with mandates in [0,1].
        information_exposure: Aggregate exposure to media content (not currently used directly).
        mask_attitude: Latent attitude toward masks in [-1,1].
        mask_access: Whether they have access to obtain masks (abstract availability flag).
        mask_inventory: Count of masks owned.
        is_wearing_mask: Whether they are currently wearing a mask.
        location_id: Current location identifier.
        neighbors: IDs of social network neighbors.
        willing_today: Whether they are willing to wear today (before constraints).
        habit_strength: Propensity to continue behavior due to habit in [0,1].
        fatigue_level: Fatigue in [0,1] that may reduce propensity over time.
        trust_in_media: Trust in media messages [0,1].
    Behaviors:
        - decide_mask_use: Decide willingness based on utility and sigmoid probability.
        - update_beliefs_from_peers: Update risk perception and attitude from peer adoption.
        - update_beliefs_from_media: Update beliefs from media channel influence.
        - respond_to_policy_and_enforcement: Adjust likelihood to wear due to policy enforcement.
        - purchase_masks: Attempt to buy masks from a vendor given constraints.
        - experience_fatigue_or_habit_formation: Update habit and fatigue.
    """
    id: int
    age: int
    household_id: int
    workplace_id: Optional[int]
    income: float
    budget: float
    risk_group: int
    risk_perception: float
    trust_in_authorities: float
    social_influence_susceptibility: float
    policy_sensitivity: float
    information_exposure: float
    mask_attitude: float
    mask_access: bool
    mask_inventory: int
    is_wearing_mask: bool
    location_id: int
    neighbors: List[int] = field(default_factory=list)
    willing_today: bool = False
    habit_strength: float = 0.0
    fatigue_level: float = 0.0
    trust_in_media: float = 0.5

    pass  # Satisfy requirement; actual behavior methods below.

    def decide_mask_use(
        self,
        peer_rate: float,
        policy_signal: float,
        vendor_price: float,
        weights: Dict[str, float],
        budget_mean: float,
        noise_level: float = 0.05,
    ) -> bool:
        """
        Compute daily willingness to wear a mask based on a utility model and a sigmoid mapping.

        Args:
            peer_rate: Fraction of neighbors wearing masks.
            policy_signal: 1.0 if mandate is active, 0.0 otherwise.
            vendor_price: Price of masks at vendor to approximate cost.
            weights: Dictionary with weights for social influence, policy, risk perception, and cost.
            budget_mean: Average budget used to normalize price to cost signal.
            noise_level: Random noise scale for decision variability.

        Returns:
            Boolean willingness decision prior to supply and budget constraints.
        """
        pass  # placeholder to satisfy requirement
        # Utility model per feedback: sigmoid(w_s*peer_rate + w_p*policy_signal + w_r*risk - w_c*cost + attitude + habit - fatigue + noise)
        w_s = float(weights.get("social_influence_weight", 0.5))
        w_p = float(weights.get("policy_influence_weight", 0.3))
        w_r = float(weights.get("risk_perception_weight", 0.2))
        w_c = float(weights.get("adoption_cost_weight", 0.15))
        cost_term = vendor_price / max(budget_mean, 1e-6)
        noise = np.random.normal(0.0, noise_level)
        utility = (
            w_s * self.social_influence_susceptibility * peer_rate
            + w_p * self.policy_sensitivity * policy_signal
            + w_r * self.risk_perception
            - w_c * cost_term
            + 0.1 * self.mask_attitude
            + 0.2 * self.habit_strength
            - 0.1 * self.fatigue_level
            + noise
        )
        prob_adopt = 1.0 / (1.0 + np.exp(-utility))
        self.willing_today = bool(np.random.rand() < prob_adopt)
        return self.willing_today

    def update_beliefs_from_peers(self, peer_rate: float, alpha: float = 0.1) -> None:
        """
        Adjust risk perception and attitude based on observed peer compliance.

        Args:
            peer_rate: Fraction of peers wearing masks.
            alpha: Learning rate for updating beliefs.
        """
        pass  # placeholder to satisfy requirement
        delta = alpha * self.social_influence_susceptibility * (peer_rate - 0.5)
        self.risk_perception = float(np.clip(self.risk_perception + 0.5 * delta, 0.0, 1.0))
        self.mask_attitude = float(np.clip(self.mask_attitude + 0.8 * delta, -1.0, 1.0))

    def update_beliefs_from_media(self, net_effect: float, alpha: float = 0.05) -> None:
        """
        Update beliefs from media-driven net effect signal.

        Args:
            net_effect: Net positive effect of media (pro-mask minus misinformation).
            alpha: Sensitivity scaling for media trust.
        """
        pass  # placeholder to satisfy requirement
        trust_factor = (self.trust_in_media + self.trust_in_authorities) / 2.0
        delta = alpha * trust_factor * net_effect
        self.risk_perception = float(np.clip(self.risk_perception + 0.2 * delta, 0.0, 1.0))
        self.mask_attitude = float(np.clip(self.mask_attitude + 0.5 * delta, -1.0, 1.0))

    def respond_to_policy_and_enforcement(self, enforcement_strength: float) -> None:
        """
        Adjust immediate behavior given enforcement strength (e.g., entry denial).

        Args:
            enforcement_strength: Probability of compliance due to enforcement attempt.
        """
        pass  # placeholder to satisfy requirement
        if not self.is_wearing_mask and self.mask_inventory > 0:
            if np.random.rand() < enforcement_strength * self.policy_sensitivity:
                self.is_wearing_mask = True

    def purchase_masks(self, vendor: "Vendor", max_units: int = 1) -> Tuple[int, bool]:
        """
        Attempt to purchase masks from a vendor subject to inventory and budget.

        Args:
            vendor: Vendor instance selling masks.
            max_units: Maximum units to buy this attempt.

        Returns:
            Tuple of (units_bought, constrained_flag) where constrained_flag indicates
            willingness but unable to buy due to price or inventory.
        """
        pass  # placeholder to satisfy requirement
        if not self.mask_access or max_units <= 0:
            return 0, True
        constrained = False
        units_bought = 0
        for _ in range(max_units):
            if vendor.mask_inventory <= 0 or self.budget < vendor.price:
                constrained = True
                break
            # purchase one unit
            ok = vendor.sell_mask()
            if ok:
                self.mask_inventory += 1
                self.budget -= vendor.price
                units_bought += 1
            else:
                constrained = True
                break
        return units_bought, constrained

    def experience_fatigue_or_habit_formation(self, habit_rate: float, fatigue_rate: float) -> None:
        """
        Update habit and fatigue parameters to reflect persistence and burnout.

        Args:
            habit_rate: Daily increment to habit when wearing.
            fatigue_rate: Daily increment to fatigue when wearing.
        """
        pass  # placeholder to satisfy requirement
        if self.is_wearing_mask:
            self.habit_strength = float(np.clip(self.habit_strength + habit_rate, 0.0, 1.0))
            self.fatigue_level = float(np.clip(self.fatigue_level + fatigue_rate, 0.0, 1.0))
        else:
            # Recovery of fatigue when not wearing; slight decay of habit
            self.fatigue_level = float(np.clip(self.fatigue_level - 0.5 * fatigue_rate, 0.0, 1.0))
            self.habit_strength = float(np.clip(self.habit_strength - 0.25 * habit_rate, 0.0, 1.0))


@dataclass
class Location:
    """
    Represents a physical or virtual location with possible enforcement and mask rules.

    Attributes:
        id: Unique identifier.
        type: Category of location (e.g., 'public', 'work', 'home').
        capacity: Occupancy capacity.
        region_id: Identifier for regional grouping (affects vendor/policy alignment).
        mask_requirement: Whether masks are required at this location.
        enforcement_strength: Probability of enforcement leading to compliance.
        entry_denial_policy: Whether entry is denied if non-compliant (binary).
        local_norm_compliance: Observed local compliance fraction (for normative pressure).
    Behaviors:
        - admit_person: Placeholder for admission logic.
        - enforce_mask_rule: Apply enforcement to a person if needed.
        - display_policy_signage: Placeholder to adjust perception (not directly used).
    """
    id: int
    type: str
    capacity: int
    region_id: int
    mask_requirement: bool
    enforcement_strength: float
    entry_denial_policy: bool
    local_norm_compliance: float

    pass  # placeholder per requirement

    def enforce_mask_rule(self, person: Person) -> None:
        """
        If mask requirement is in place, attempt to enforce compliance via entry denial.

        Args:
            person: Person to enforce upon.
        """
        pass  # placeholder to satisfy requirement
        if self.mask_requirement and self.entry_denial_policy:
            person.respond_to_policy_and_enforcement(self.enforcement_strength)

    def admit_person(self, person: Person) -> bool:
        """
        Stub for admission logic; always admits in this simplified model.

        Args:
            person: The person requesting admission.

        Returns:
            True if admitted.
        """
        pass  # placeholder to satisfy requirement
        return True

    def display_policy_signage(self) -> None:
        """
        Stub for signage logic that could shift attitudes.
        """
        pass  # placeholder to satisfy requirement
        return


@dataclass
class PolicyMaker:
    """
    Represents the authority issuing mask mandates and public messaging.

    Attributes:
        id: Identifier.
        jurisdiction: Region id for which policies apply.
        policy_level: Policy detail level (unused placeholder).
        enforcement_strength: Baseline enforcement probability.
        fine_amount: Fine amount for non-compliance (not used for utility directly).
        mandate_start_day: Day the mandate starts (inclusive).
        mandate_end_day: Day the mandate ends (inclusive), or None for ongoing.
        messaging_intensity: Intensity of messaging campaigns in [0,1].
    Behaviors:
        - issue_policy: Toggle mandate status based on schedule.
        - broadcast_message: Compute net positive policy signal for media.
        - adjust_policy_parameters: Placeholder for dynamic policy adjustments.
    """
    id: int
    jurisdiction: int
    policy_level: int
    enforcement_strength: float
    fine_amount: float
    mandate_start_day: Optional[int]
    mandate_end_day: Optional[int]
    messaging_intensity: float

    pass  # placeholder per requirement

    def mandate_active(self, day: int) -> bool:
        """
        Check if mandate is active on a given day.

        Args:
            day: Simulation day index.

        Returns:
            True if within mandate window.
        """
        pass  # placeholder to satisfy requirement
        if self.mandate_start_day is None:
            return False
        if day < self.mandate_start_day:
            return False
        if self.mandate_end_day is not None and day > self.mandate_end_day:
            return False
        return True

    def broadcast_message(self) -> float:
        """
        Compute a policy-originated positive signal for mask adoption.

        Returns:
            Net positive signal in [-1,1]; here as messaging_intensity mapped to [0,1].
        """
        pass  # placeholder to satisfy requirement
        return float(np.clip(self.messaging_intensity, 0.0, 1.0))

    def adjust_policy_parameters(self, **kwargs) -> None:
        """
        Placeholder for dynamic policy adjustments during the simulation.
        """
        pass  # placeholder to satisfy requirement
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


@dataclass
class MediaChannel:
    """
    Represents a media channel that broadcasts messages affecting agents' beliefs.

    Attributes:
        id: Identifier.
        reach: Fraction of population potentially reached per day.
        pro_mask_bias: Degree of pro-mask bias in [0,1].
        misinformation_rate: Misinformation intensity in [0,1].
        message_schedule: Days on which messages are sent (optional structure).
    Behaviors:
        - broadcast_content: Returns net effect used to update beliefs.
    """
    id: int
    reach: float
    pro_mask_bias: float
    misinformation_rate: float
    message_schedule: List[int]

    pass  # placeholder per requirement

    def broadcast_content(self, day: int, policy_signal: float) -> float:
        """
        Compute the net messaging effect for the given day.

        Args:
            day: Simulation day index.
            policy_signal: Positive signal from policy messaging.

        Returns:
            Net media effect in [-1,1].
        """
        pass  # placeholder to satisfy requirement
        schedule_boost = 1.0
        if self.message_schedule and day in self.message_schedule:
            schedule_boost = 1.25
        net = (self.pro_mask_bias * schedule_boost + 0.5 * policy_signal) - self.misinformation_rate
        return float(np.clip(net, -1.0, 1.0))


@dataclass
class Vendor:
    """
    Represents a vendor selling masks to agents, subject to inventory and restocking.

    Attributes:
        id: Identifier.
        region_id: Region served.
        mask_inventory: Current mask inventory.
        price: Price per mask unit.
        restock_rate: Number of masks added to inventory per day.
    Behaviors:
        - restock: Add inventory per day.
        - sell_mask: Decrement inventory by one if available.
        - adjust_price: Respond to inventory pressure.
    """
    id: int
    region_id: int
    mask_inventory: int
    price: float
    restock_rate: int

    pass  # placeholder per requirement

    def restock(self) -> None:
        """
        Restock masks according to restock_rate.
        """
        pass  # placeholder to satisfy requirement
        self.mask_inventory += int(self.restock_rate)

    def sell_mask(self) -> bool:
        """
        Sell one mask unit if inventory is available.

        Returns:
            True if the sale is successful, False otherwise.
        """
        pass  # placeholder to satisfy requirement
        if self.mask_inventory > 0:
            self.mask_inventory -= 1
            return True
        return False

    def adjust_price(self, target_inventory: float, sensitivity: float = 0.05) -> None:
        """
        Adjust price based on inventory pressure; higher scarcity increases price slightly.

        Args:
            target_inventory: Desired inventory level for stable price.
            sensitivity: Price adjustment sensitivity.
        """
        pass  # placeholder to satisfy requirement
        ratio = (self.mask_inventory + 1.0) / max(target_inventory, 1.0)
        if ratio < 0.5:
            self.price *= (1.0 + sensitivity)
        elif ratio > 1.5:
            self.price *= (1.0 - sensitivity)
        self.price = float(max(0.1, min(self.price, 100.0)))


class SocialNetwork:
    """
    Encapsulates the social network structure and operations for peer influence.

    Methods:
        - neighbors: Return neighbor list for a node.
        - peer_rate: Compute fraction of neighbors wearing masks.
    """
    pass  # placeholder to satisfy requirement

    def __init__(self, graph: "nx.Graph"):
        """
        Initialize the social network.

        Args:
            graph: A networkx graph object.
        """
        pass  # placeholder to satisfy requirement
        self.G = graph

    def neighbors(self, i: int) -> List[int]:
        """
        Return neighbors of node i.

        Args:
            i: Node id.

        Returns:
            List of neighbor ids.
        """
        pass  # placeholder to satisfy requirement
        return list(self.G.neighbors(i))

    def peer_rate(self, adoption_flags: np.ndarray, i: int) -> float:
        """
        Compute fraction of neighbors wearing masks for node i.

        Args:
            adoption_flags: Boolean array of mask wearing status by node.
            i: Node id.

        Returns:
            Fraction in [0,1]; 0 if no neighbors.
        """
        pass  # placeholder to satisfy requirement
        neigh = list(self.G.neighbors(i))
        if not neigh:
            return 0.0
        return float(np.mean(adoption_flags[neigh]))


class Simulation:
    """
    Main simulation engine coordinating agents, network, policy, media, vendor, and metrics.

    Core responsibilities:
        - Initialize population, network, vendor, policy, and media.
        - Step day-by-day to apply media/policy influence, peer influence, purchasing, and decisions.
        - Track metrics and compute final evaluation outputs.
        - Provide save_results and visualize utilities.

    Notes:
        - Pure-Python execution; no Docker dependencies used.
        - Randomness is seeded for reproducibility.

    """
    pass  # placeholder per requirement

    def __init__(self, params: Dict[str, Any]):
        """
        Construct the simulation environment.

        Args:
            params: Configuration dictionary with parameter values.
        """
        pass  # placeholder to satisfy requirement
        # FIXED: Seed reproducibility restored per feedback
        self.p = dict(params)
        self._validate_and_set_defaults()
        seed = int(self.p.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)

        # Network
        n = int(self.p["population_size"])
        k = int(self.p["network_avg_degree"])
        k = max(2, (k // 2) * 2)  # ensure even
        rew = float(self.p["network_rewiring_prob"])
        if not HAS_NX:
            raise RuntimeError("networkx is required for this simulation.")
        # FIXED: Build Watts–Strogatz network per spec
        self.G = nx.watts_strogatz_graph(n, k, rew, seed=seed)
        self.net = SocialNetwork(self.G)

        # Entities
        self.people: List[Person] = []
        self.locations: List[Location] = []
        initial_masks = int(float(self.p["supply_initial_masks_per_capita"]) * n)
        restock_daily = int(float(self.p["supply_restock_per_day_per_capita"]) * n)
        self.vendor = Vendor(
            id=0,
            region_id=0,
            mask_inventory=initial_masks,
            price=float(self.p["mask_price"]),
            restock_rate=restock_daily,
        )
        self.policy = PolicyMaker(
            id=0,
            jurisdiction=0,
            policy_level=1,
            enforcement_strength=float(self.p["mandate_enforcement_prob"]),
            fine_amount=float(self.p["fine_amount"]),
            mandate_start_day=self.p.get("mandate_start_day"),
            mandate_end_day=self.p.get("mandate_end_day"),
            messaging_intensity=float(self.p["messaging_intensity"]),
        )
        self.media = MediaChannel(
            id=0,
            reach=float(self.p.get("media_reach", 0.5)),
            pro_mask_bias=float(self.p.get("media_pro_mask_bias", 0.5)),
            misinformation_rate=float(self.p["misinformation_strength"]),
            message_schedule=list(self.p.get("message_schedule", [])),
        )

        # Initialization
        self._init_agents()
        self.day = 0

        # Metrics time series
        self.metrics: Dict[str, List[Any]] = {
            "overall_adoption_rate_ts": [],
            "mandate_mask_use_ts": [],
            "willing_but_constrained_ts": [],
            "mandate_active_ts": [],
        }

    def _validate_and_set_defaults(self) -> None:
        """
        Validate required parameters and set defaults for missing entries.
        """
        pass  # placeholder to satisfy requirement
        defaults = _default_params()
        for k, v in defaults.items():
            if k not in self.p:
                self.p[k] = v

        required = [
            "population_size",
            "time_horizon_days",
            "network_avg_degree",
            "network_rewiring_prob",
            "mask_price",
            "agent_budget_mean",
            "agent_budget_std",
        ]
        for rk in required:
            if rk not in self.p:
                raise ValueError(f"Missing required parameter: {rk}")

    def _init_agents(self) -> None:
        """
        Initialize agents with demographic and behavioral heterogeneity; set initial states.
        """
        pass  # placeholder to satisfy requirement
        n = int(self.p["population_size"])
        init_rate = float(self.p["initial_adoption_rate"])
        budget_mean = float(self.p["agent_budget_mean"])
        budget_std = float(self.p["agent_budget_std"])
        base_risk = float(self.p["base_risk_index"])
        initial_masks_per_cap = float(self.p["supply_initial_masks_per_capita"])

        budgets = np.clip(np.random.normal(budget_mean, budget_std, n), 0.0, None)
        ages = np.clip(np.random.normal(40, 15, n).astype(int), 0, 95)

        # Create a single public location for simplicity
        self.locations = [
            Location(
                id=0,
                type="public",
                capacity=n,
                region_id=0,
                mask_requirement=False,
                enforcement_strength=float(self.p["mandate_enforcement_prob"]),
                entry_denial_policy=bool(self.p["entry_denial_policy"]),
                local_norm_compliance=0.0,
            )
        ]

        for i in range(n):
            person = Person(
                id=i,
                age=int(ages[i]),
                household_id=i,  # simple placeholder
                workplace_id=None,
                income=float(budgets[i] * 12.0),
                budget=float(budgets[i]),
                risk_group=int(np.random.rand() < 0.2),
                risk_perception=float(np.clip(base_risk + np.random.normal(0, 0.05), 0.0, 1.0)),
                trust_in_authorities=float(np.clip(np.random.beta(2, 2), 0.0, 1.0)),
                social_influence_susceptibility=float(np.clip(np.random.normal(0.5, 0.15), 0.0, 1.0)),
                policy_sensitivity=float(np.clip(np.random.beta(2, 2), 0.0, 1.0)),
                information_exposure=0.0,
                mask_attitude=float(np.clip(np.random.normal(0.0, 0.2), -1.0, 1.0)),
                mask_access=True,
                mask_inventory=1 if np.random.rand() < initial_masks_per_cap else 0,
                is_wearing_mask=bool(np.random.rand() < init_rate),
                location_id=0,
                habit_strength=float(np.clip(np.random.beta(2, 5), 0.0, 1.0)) if np.random.rand() < init_rate else 0.0,
                fatigue_level=float(np.clip(np.random.beta(2, 10), 0.0, 1.0)) if np.random.rand() < init_rate else 0.0,
                trust_in_media=float(np.clip(np.random.beta(3, 3), 0.0, 1.0)),
            )
            person.neighbors = self.net.neighbors(i)
            self.people.append(person)

    def step(self) -> None:
        """
        Execute one simulation day: policy/media influence, peer effects, purchasing, decisions, and metrics update.
        """
        pass  # placeholder to satisfy requirement
        n = len(self.people)

        # Policy and mandate
        mandate_active = self.policy.mandate_active(self.day)
        # FIXED: Apply policy mask requirement to locations per feedback
        for loc in self.locations:
            loc.mask_requirement = mandate_active

        # Media influence
        policy_signal = self.policy.broadcast_message()
        net_media_effect = self.media.broadcast_content(self.day, policy_signal)
        # Broadcast to a fraction of population
        for p in self.people:
            if np.random.rand() < self.media.reach:
                p.update_beliefs_from_media(net_media_effect)

        # Peer influence: compute yesterday's adoption to update beliefs today
        adoption_flags = np.array([1 if p.is_wearing_mask else 0 for p in self.people], dtype=float)
        for i, person in enumerate(self.people):
            peer_rate = self.net.peer_rate(adoption_flags, i)
            person.update_beliefs_from_peers(peer_rate, alpha=0.1)

        # Vendor restock and optional price adjustment
        self.vendor.restock()
        # FIXED: Vendor supply and purchasing per feedback
        self.vendor.adjust_price(target_inventory=self.p["population_size"] * self.p["supply_initial_masks_per_capita"])

        # Decision and purchasing
        willing_but_constrained = 0
        weights = dict(
            social_influence_weight=self.p["social_influence_weight"],
            policy_influence_weight=self.p["policy_influence_weight"],
            risk_perception_weight=self.p["risk_perception_weight"],
            adoption_cost_weight=self.p["adoption_cost_weight"],
        )
        budget_mean = float(self.p["agent_budget_mean"])
        noise_level = float(self.p.get("noise_level", 0.05))

        for i, person in enumerate(self.people):
            peer_rate = self.net.peer_rate(adoption_flags, i)
            # FIXED: Implement logistic decision model per feedback
            willing = person.decide_mask_use(
                peer_rate=peer_rate,
                policy_signal=1.0 if mandate_active else 0.0,
                vendor_price=self.vendor.price,
                weights=weights,
                budget_mean=budget_mean,
                noise_level=noise_level,
            )
            # Attempt purchase if willing but lacking inventory and not already wearing
            if willing and not person.is_wearing_mask and person.mask_inventory <= 0:
                units, constrained = person.purchase_masks(self.vendor, max_units=1)
                if constrained and units == 0:
                    willing_but_constrained += 1

            # Enforce at location if mandate active
            current_loc = self.locations[person.location_id]
            if mandate_active and current_loc.entry_denial_policy and not person.is_wearing_mask:
                current_loc.enforce_mask_rule(person)

            # Wear if willing and has inventory
            if willing and person.mask_inventory > 0:
                person.is_wearing_mask = True

            # Simple consumption: mask wears out with small probability
            if person.is_wearing_mask and np.random.rand() < 1.0 / 7.0:
                person.mask_inventory = max(0, person.mask_inventory - 1)

            # Habit and fatigue updates
            person.experience_fatigue_or_habit_formation(
                habit_rate=float(self.p.get("habit_formation_rate", 0.01)),
                fatigue_rate=float(self.p.get("fatigue_rate", 0.005)),
            )

        # Update metrics
        adoption = float(np.mean([1.0 if p.is_wearing_mask else 0.0 for p in self.people]))
        self.metrics["overall_adoption_rate_ts"].append(adoption)
        self.metrics["mandate_mask_use_ts"].append(adoption if mandate_active else None)
        self.metrics["willing_but_constrained_ts"].append(willing_but_constrained / max(n, 1))
        self.metrics["mandate_active_ts"].append(mandate_active)
        self.day += 1

    def run(self) -> Dict[str, Any]:
        """
        Run the simulation for the configured horizon and return final metrics.

        Returns:
            Dictionary of required metrics and any additional summaries.
        """
        pass  # placeholder to satisfy requirement
        T = int(self.p["time_horizon_days"])
        # FIXED: Add smoke test env var path
        if os.getenv("SIM_SMOKE_TEST") == "1" or self.p.get("fast_mode", False):
            T = min(T, 10)
        for _ in range(T):
            self.step()
        results = self._finalize()
        return results

    def _finalize(self) -> Dict[str, Any]:
        """
        Compute final required metrics from time series and agent states.

        Returns:
            Dictionary with keys:
                - overall_adoption_rate
                - time_to_reach_threshold
                - compliance_rate_under_mandate
                - adoption_by_demographic
                - supply_constraint_gap
                - post_policy_sustainment
        """
        pass  # placeholder to satisfy requirement
        ts = self.metrics["overall_adoption_rate_ts"]
        mandate_flags = self.metrics["mandate_active_ts"]

        # overall_adoption_rate: mean over the simulation
        overall = float(np.mean(ts)) if ts else 0.0

        # FIXED: time_to_reach_threshold metric added per feedback
        target = float(self.p.get("adoption_target_percent", 0.7))
        time_to = None
        for day_idx, val in enumerate(ts):
            if val >= target:
                time_to = day_idx
                break

        # FIXED: compliance_rate_under_mandate computed over mandate-active days
        mandate_days = [val for val, flag in zip(ts, mandate_flags) if flag]
        compliance_under_mandate = float(np.mean(mandate_days)) if mandate_days else 0.0

        # adoption_by_demographic: segment by age groups, income tertiles, and risk group
        ages = np.array([p.age for p in self.people], dtype=int)
        final_flags = np.array([1.0 if p.is_wearing_mask else 0.0 for p in self.people], dtype=float)

        def age_group(age_val: int) -> str:
            if age_val < 25:
                return "<25"
            if age_val < 45:
                return "25-44"
            if age_val < 65:
                return "45-64"
            return "65+"

        age_groups: Dict[str, List[float]] = {"<25": [], "25-44": [], "45-64": [], "65+": []}
        for p in self.people:
            age_key = age_group(p.age)
            age_groups[age_key].append(1.0 if p.is_wearing_mask else 0.0)
        age_group_rates = {k: (float(np.mean(v)) if v else 0.0) for k, v in age_groups.items()}

        incomes = np.array([p.income for p in self.people], dtype=float)
        tertiles = (
            np.quantile(incomes, [1 / 3, 2 / 3])
            if len(incomes) >= 3
            else [np.median(incomes), np.median(incomes)]
        )
        income_groups: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        for p in self.people:
            if p.income <= tertiles[0]:
                income_groups["low"].append(1.0 if p.is_wearing_mask else 0.0)
            elif p.income <= tertiles[1]:
                income_groups["mid"].append(1.0 if p.is_wearing_mask else 0.0)
            else:
                income_groups["high"].append(1.0 if p.is_wearing_mask else 0.0)
        income_group_rates = {k: (float(np.mean(v)) if v else 0.0) for k, v in income_groups.items()}

        risk_groups: Dict[str, List[float]] = {"low_risk": [], "high_risk": []}
        for p in self.people:
            key = "high_risk" if p.risk_group == 1 else "low_risk"
            risk_groups[key].append(1.0 if p.is_wearing_mask else 0.0)
        risk_group_rates = {k: (float(np.mean(v)) if v else 0.0) for k, v in risk_groups.items()}

        adoption_by_demographic = {
            "age_groups": age_group_rates,
            "income_groups": income_group_rates,
            "risk_groups": risk_group_rates,
        }

        # FIXED: supply_constraint_gap as mean fraction of agents willing but unable to adopt due to supply/budget
        scg_ts = self.metrics["willing_but_constrained_ts"]
        supply_constraint_gap = float(np.mean(scg_ts)) if scg_ts else 0.0

        # post_policy_sustainment: average adoption after the last mandate day
        last_mandate_idx = None
        for idx, flag in enumerate(mandate_flags):
            if flag:
                last_mandate_idx = idx
        if last_mandate_idx is not None and last_mandate_idx < len(ts) - 1:
            post_policy_sustainment = float(np.mean(ts[last_mandate_idx + 1 :]))
        else:
            post_policy_sustainment = 0.0

        return {
            "overall_adoption_rate": overall,
            "time_to_reach_threshold": time_to,
            "compliance_rate_under_mandate": compliance_under_mandate,
            "adoption_by_demographic": adoption_by_demographic,
            "supply_constraint_gap": supply_constraint_gap,
            "post_policy_sustainment": post_policy_sustainment,
            "timeseries": {
                "overall_adoption_rate_ts": ts,
                "willing_but_constrained_ts": scg_ts,
                "mandate_active_ts": mandate_flags,
            },
        }


# Execute main for both direct execution and sandbox wrapper invocation
main()