def main():
    """
    Entry point for running the mask adoption simulation.

    This function demonstrates:
    - Parameter preparation
    - Simulation initialization
    - Running the simulation
    - Visualization of results (if matplotlib is available)
    - Saving results to a CSV file

    It supports reading parameters from a JSON file via command line:
        python simulation.py --params params.json
    """
    pass
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation")
    parser.add_argument("--params", type=str, default=None, help="Path to JSON params file")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (short run)")
    args = parser.parse_args()

    if args.params:
        with open(args.params, "r") as f:
            params = json.load(f)
    else:
        params = {}

    if args.fast:
        params["fast_mode"] = True

    sim = Simulation(params)
    results = sim.run()
    print("Simulation complete. Key metrics:")
    for k in [
        "overall_adoption_rate",
        "time_to_50_percent_adoption",
        "peak_adoption_rate",
        "steady_state_adoption_rate",
        "mandate_effect_size",
        "adoption_inequality_index",
        "noncompliance_under_mandate",
    ]:
        print(f" - {k}: {results.get(k)}")

    # Visualization
    sim.visualize()

    # Save results
    sim.save_results("results.csv")
    print("Results saved to results.csv")


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

    Returns:
        Dictionary containing default parameters for the simulation.
    """
    pass
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
        "purchase_limit_per_person": 3,  # FIXED: Added default daily purchase limit per feedback
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
        "messaging_frequency_per_week": 3,  # FIXED: Added to support media scheduling per feedback
        "message_effect_size": 0.1,         # FIXED: Added to support media effect size per feedback
        "media_credibility": 0.6,           # FIXED: Added credibility attribute per feedback
        "media_bias": 0.0,                  # FIXED: Added bias attribute per feedback
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
        # Counterfactual
        "compute_counterfactual": True,     # FIXED: Added switch to compute mandate effect per feedback
    }


def _map_spec_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map specification parameter names to internal simulation names.

    Args:
        p: Original parameters from user/spec.

    Returns:
        Mapped parameters dictionary.
    """
    pass
    mapped = dict(p)
    # FIXED: Added parameter mapping layer per feedback
    # Network
    if 'average_degree' in p and 'network_avg_degree' not in p:
        mapped['network_avg_degree'] = p['average_degree']
    if 'network_topology' in p and p['network_topology'] == 'small_world':
        mapped.setdefault('network_rewiring_prob', 0.1)

    # Policy/enforcement
    if 'enforcement_probability' in p and 'mandate_enforcement_prob' not in p:
        mapped['mandate_enforcement_prob'] = p['enforcement_probability']
    if 'policy_enforcement_strength' in p and 'mandate_enforcement_prob' not in p:
        mapped['mandate_enforcement_prob'] = p['policy_enforcement_strength']
    if 'fine_amount' in p:
        mapped['fine_amount'] = p['fine_amount']
    if 'mandate_active' in p:
        mapped['mandate_start_day'] = 0 if p['mandate_active'] else None
    if 'mandate_active_initial' in p and 'mandate_start_day' not in p:
        mapped['mandate_start_day'] = 0 if p['mandate_active_initial'] else None
    if 'campaign_intensity' in p and 'messaging_intensity' not in p:
        mapped['messaging_intensity'] = p['campaign_intensity']

    # Supply/Retailer
    if 'mask_supply_per_day' in p and 'supply_restock_per_day_per_capita' not in p and 'population_size' in p:
        mapped['supply_restock_per_day_per_capita'] = p['mask_supply_per_day'] / max(p['population_size'], 1)
    if 'restock_rate_per_day' in p and 'supply_restock_per_day_per_capita' not in p and 'population_size' in p:
        mapped['supply_restock_per_day_per_capita'] = p['restock_rate_per_day'] / max(p['population_size'], 1)
    if 'initial_mask_price' in p and 'mask_price' not in p:
        mapped['mask_price'] = p['initial_mask_price']
    if 'purchase_limit_per_person' in p:
        mapped['purchase_limit_per_person'] = p['purchase_limit_per_person']

    # Costs/influence
    if 'cost_sensitivity' in p and 'adoption_cost_weight' not in p:
        mapped['adoption_cost_weight'] = p['cost_sensitivity']
    if 'social_influence_weight' in p:
        mapped['social_influence_weight'] = p['social_influence_weight']
    if 'policy_pressure_weight' in p and 'policy_influence_weight' not in p:
        mapped['policy_influence_weight'] = p['policy_pressure_weight']
    if 'personal_risk_weight' in p and 'risk_perception_weight' not in p:
        mapped['risk_perception_weight'] = p['personal_risk_weight']

    # Media
    if 'messaging_frequency_per_week' in p:
        mapped['messaging_frequency_per_week'] = p['messaging_frequency_per_week']
    if 'message_effect_size' in p:
        mapped['message_effect_size'] = p['message_effect_size']
    if 'message_trust_weight' in p and 'media_credibility' not in p:
        mapped['media_credibility'] = p['message_trust_weight']

    return mapped


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
        pass
        # Utility model per feedback: sigmoid(w_s*peer_rate + w_p*policy_signal + w_r*risk - w_c*cost + attitude + habit - fatigue + noise)
        w_s = float(weights.get("social_influence_weight", 0.5))
        w_p = float(weights.get("policy_influence_weight", 0.3))
        w_r = float(weights.get("risk_perception_weight", 0.2))
        w_c = float(weights.get("adoption_cost_weight", 0.15))
        # FIXED: Include expected penalty normalized by personal budget per feedback
        personal_budget = max(self.budget, 1e-6)
        price_term = vendor_price / personal_budget
        enforcement_prob = float(weights.get('enforcement_probability', 0.0))
        fine_amount = float(weights.get('fine_amount', 0.0))
        expected_penalty = (policy_signal * enforcement_prob * fine_amount) / personal_budget
        cost_term = price_term + expected_penalty
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
        pass
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
        pass
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
        pass
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
        pass
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
        pass
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

    pass

    def enforce_mask_rule(self, person: Person) -> None:
        """
        If mask requirement is in place, attempt to enforce compliance via entry denial.

        Args:
            person: Person to enforce upon.
        """
        pass
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
        pass
        return True

    def display_policy_signage(self) -> None:
        """
        Stub for signage logic that could shift attitudes.
        """
        pass
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
        fine_amount: Fine amount for non-compliance.
        mandate_start_day: Day the mandate starts (inclusive).
        mandate_end_day: Day the mandate ends (inclusive), or None for ongoing.
        messaging_intensity: Intensity of messaging campaigns in [0,1].
    Behaviors:
        - mandate_active: Toggle mandate status based on schedule.
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

    pass

    def mandate_active(self, day: int) -> bool:
        """
        Check if mandate is active on a given day.

        Args:
            day: Simulation day index.

        Returns:
            True if within mandate window.
        """
        pass
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
        pass
        return float(np.clip(self.messaging_intensity, 0.0, 1.0))

    def adjust_policy_parameters(self, **kwargs) -> None:
        """
        Placeholder for dynamic policy adjustments during the simulation.
        """
        pass
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
        credibility: Perceived credibility weight in [0,1].
        bias: Additional directional bias term.
        message_effect_size: Scaling for net message effect.
        messaging_frequency_per_week: Number of broadcast days per week.
    Behaviors:
        - broadcast_content: Returns net effect used to update beliefs.
    """
    id: int
    reach: float
    pro_mask_bias: float
    misinformation_rate: float
    message_schedule: List[int]
    credibility: float = 0.5
    bias: float = 0.0
    message_effect_size: float = 0.1
    messaging_frequency_per_week: int = 3

    pass

    def broadcast_content(self, day: int, policy_signal: float) -> float:
        """
        Compute the net messaging effect for the given day.

        Args:
            day: Simulation day index.
            policy_signal: Positive signal from policy messaging.

        Returns:
            Net media effect in [-1,1].
        """
        pass
        # FIXED: Added credibility, bias, frequency, and effect size per feedback
        weekly = (day % 7) < int(self.messaging_frequency_per_week)
        schedule_boost = 1.25 if (self.message_schedule and day in self.message_schedule) or weekly else 1.0
        base = (self.pro_mask_bias - self.misinformation_rate + self.bias)
        net = (base * schedule_boost + 0.5 * policy_signal) * (self.credibility * self.message_effect_size)
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

    pass

    def restock(self) -> None:
        """
        Restock masks according to restock_rate.
        """
        pass
        self.mask_inventory += int(self.restock_rate)

    def sell_mask(self) -> bool:
        """
        Sell one mask unit if inventory is available.

        Returns:
            True if the sale is successful, False otherwise.
        """
        pass
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
        pass
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
    pass

    def __init__(self, graph: "nx.Graph"):
        """
        Initialize the social network.

        Args:
            graph: A networkx graph object.
        """
        pass
        self.G = graph

    def neighbors(self, i: int) -> List[int]:
        """
        Return neighbors of node i.

        Args:
            i: Node id.

        Returns:
            List of neighbor ids.
        """
        pass
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
        pass
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
    pass

    def __init__(self, params: Dict[str, Any]):
        """
        Construct the simulation environment.

        Args:
            params: Configuration dictionary with parameter values.
        """
        pass
        # FIXED: Map spec parameters to internal names per feedback
        self.p = _map_spec_params(dict(params))
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
            credibility=float(self.p.get("media_credibility", 0.6)),
            bias=float(self.p.get("media_bias", 0.0)),
            message_effect_size=float(self.p.get("message_effect_size", 0.1)),
            messaging_frequency_per_week=int(self.p.get("messaging_frequency_per_week", 3)),
        )

        # Initialization
        self._init_agents()
        self.day = 0

        # Metrics time series
        self.metrics: Dict[str, Any] = {
            "overall_adoption_rate_ts": [],
            "mandate_active_ts": [],
            "willing_but_constrained_ts": [],
            "price_ts": [],
        }

    def _validate_and_set_defaults(self) -> None:
        """
        Validate required parameters and set defaults for missing entries.
        """
        pass
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
        pass
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
        pass
        n = len

# Execute main for both direct execution and sandbox wrapper invocation
main()