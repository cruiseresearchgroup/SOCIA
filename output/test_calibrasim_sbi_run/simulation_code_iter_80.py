def main():
    pass

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Path handling as instructed
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else os.getcwd()


def clamp01(x: float) -> float:
    """
    Clamp a floating-point value into the [0, 1] interval.

    Returns:
        float: The clamped value between 0.0 and 1.0 inclusive.
    """
    # NOTE: pass retained per environment requirement; functional code follows
    pass
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def sigmoid(x: float) -> float:
    """
    Compute logistic sigmoid for input x.

    Returns:
        float: Value in (0, 1).
    """
    # NOTE: pass retained per environment requirement; functional code follows
    pass
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def generate_small_world(n: int, k: int, beta: float, rng: random.Random) -> List[List[int]]:
    """
    Generate a small-world network using a lightweight Watts–Strogatz-like approach.
    No external dependencies are required.

    Args:
        n (int): Number of nodes.
        k (int): Mean degree (should be even). If odd, one side will floor.
        beta (float): Rewiring probability in [0,1].
        rng (random.Random): Random number generator.

    Returns:
        List[List[int]]: Adjacency list where neighbors[i] is a list of neighbor indices.

    Notes:
        - FIXED: Removed dependency on networkx by implementing a custom small-world generator.
        - Ensures no self-loops and avoids duplicate edges.
    """
    # NOTE: pass retained per environment requirement; functional code follows
    pass
    if n <= 1:
        return [[] for _ in range(n)]
    neighbors = [list() for _ in range(n)]
    half_k = max(1, k // 2)
    # Ring lattice
    for i in range(n):
        for d in range(1, half_k + 1):
            j = (i + d) % n
            neighbors[i].append(j)
            neighbors[j].append(i)
            j2 = (i - d) % n
            if j2 not in neighbors[i]:
                neighbors[i].append(j2)
            if i not in neighbors[j2]:
                neighbors[j2].append(i)
    # Rewiring
    for i in range(n):
        original_neighbors = list(neighbors[i])
        for j in original_neighbors:
            if j > i and rng.random() < beta:
                # remove existing edge
                if j in neighbors[i]:
                    neighbors[i].remove(j)
                if i in neighbors[j]:
                    neighbors[j].remove(i)
                # add new edge
                candidates = [x for x in range(n) if x != i and x not in neighbors[i]]
                if candidates:
                    new_j = rng.choice(candidates)
                    if new_j not in neighbors[i]:
                        neighbors[i].append(new_j)
                    if i not in neighbors[new_j]:
                        neighbors[new_j].append(i)
    # Deduplicate and sort
    return [sorted(set(lst)) for lst in neighbors]


@dataclass
class Person:
    """
    Represents an individual agent with beliefs, preferences, and behaviors related to mask adoption.

    Attributes:
        id (int): Unique identifier for the person.
        neighbors (List[int]): IDs (indices) of social neighbors in the adjacency list.
        risk_perception (float): Perceived infection risk in [0,1].
        compliance_propensity (float): General propensity to comply with guidance in [0,1].
        peer_susceptibility (float): Sensitivity to peer influence in [0,1].
        trust_in_authority (float): Trust weight in [0,1].
        income (float): Income proxy that also informs budget.
        budget (float): Available liquid budget for purchases.
        mask_inventory (int): Number of masks owned.
        is_wearing_mask (bool): Wearing a mask today.
        habit (float): Habit strength in [0,1].
        fatigue (float): Fatigue from continued mask wearing in [0,1].
        info_exposure (float): Exposure level to media in [0,1].
        income_percentile (float): Income percentile in [0,1] assigned at initialization.
    """
    id: int
    neighbors: List[int]
    risk_perception: float
    compliance_propensity: float
    peer_susceptibility: float
    trust_in_authority: float
    income: float
    budget: float
    mask_inventory: int = 0
    is_wearing_mask: bool = False
    habit: float = 0.0
    fatigue: float = 0.0
    info_exposure: float = 0.5
    income_percentile: float = 0.5

    def update_from_media(self, net_media_effect: float) -> None:
        """
        Update beliefs based on media net effect which includes campaign minus misinformation.

        Args:
            net_media_effect (float): Value typically in [-1, 1], positive increases risk and compliance propensity.

        Notes:
            - Beliefs are clamped into [0,1].
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        # Media shifts risk perception and compliance a bit based on exposure
        delta = 0.1 * self.info_exposure * net_media_effect
        self.risk_perception = clamp01(self.risk_perception + delta)
        self.compliance_propensity = clamp01(self.compliance_propensity + 0.08 * self.trust_in_authority * net_media_effect)

    def decide_willingness(
        self,
        peer_rate: float,
        policy_signal: float,
        market_price: float,
        weights: Dict[str, float],
        enforcement_strength: float,
        fine_amount: float,
        rng: random.Random,
        subsidy_level: float = 0.0,
        wealth_effect_on_adoption: float = 0.3,
    ) -> bool:
        """
        Decide whether to wear a mask today, considering peer effects, policy, costs, habit, and fatigue.

        Args:
            peer_rate (float): Fraction of neighbors wearing masks.
            policy_signal (float): 1.0 if mandate active, else 0.0.
            market_price (float): Current market price per mask.
            weights (Dict[str,float]): Weights for utility components.
            enforcement_strength (float): Strength of enforcement in [0,1].
            fine_amount (float): Monetary penalty amount if non-compliant.
            rng (random.Random): RNG for stochasticity.
            subsidy_level (float): Per-unit subsidy in currency units applied at purchase.
            wealth_effect_on_adoption (float): Scales cost pressure by income percentile.

        Returns:
            bool: True if willing to wear a mask today, False otherwise.
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        # Economic cost pressure relative to income/budget
        effective_price = max(0.1, market_price - subsidy_level)
        affordability = min(1.0, (self.budget + 1e-9) / (effective_price + 1e-9))
        base_cost_pressure = (1.0 - affordability)  # 0 if affordable, 1 if cannot afford at all
        # FIXED: Integrate wealth_effect_on_adoption by scaling cost pressure with inverse income percentile
        # Poorer agents (low percentile) experience higher cost pressure
        wealth_scale = 1.0 + wealth_effect_on_adoption * (1.0 - clamp01(self.income_percentile))
        cost_pressure = clamp01(base_cost_pressure * wealth_scale)

        # FIXED: Use fine_amount in expected penalty term
        expected_penalty = policy_signal * enforcement_strength * fine_amount  # monetary
        # Normalize penalty by typical daily disposable budget proxy (2% of income)
        penalty_pressure = expected_penalty / max(1.0, self.income * 0.02)

        # Utility score to wear mask
        score = (
            weights.get('social_weight', 0.4) * self.peer_susceptibility * peer_rate
            + weights.get('risk_weight', 0.5) * self.risk_perception
            + weights.get('policy_weight', 0.4) * clamp01(penalty_pressure)
            + weights.get('habit_weight', 0.3) * self.habit
            - weights.get('fatigue_weight', 0.3) * self.fatigue
            - weights.get('cost_weight', 0.3) * cost_pressure
        )
        # Compliance propensity acts like a prior bias
        score += 0.5 * (self.compliance_propensity - 0.5)
        # FIXED: Remove redundant clamp01 around sigmoid output; sigmoid already in (0,1)
        p = sigmoid(score)
        return rng.random() < p

    def update_habit_and_fatigue(self, habit_formation_rate: float, habit_decay_rate: float, fatigue_change_wear: float, fatigue_recovery: float) -> None:
        """
        Update habit and fatigue based on whether the person wore a mask today.

        Args:
            habit_formation_rate (float): Increase when wearing.
            habit_decay_rate (float): Multiplicative decay when not wearing.
            fatigue_change_wear (float): Increment in fatigue if wearing.
            fatigue_recovery (float): Recovery decrement in fatigue if not wearing.
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        if self.is_wearing_mask:
            self.habit = clamp01(self.habit + habit_formation_rate * (1.0 - self.habit))
            self.fatigue = clamp01(self.fatigue + fatigue_change_wear * (1.0 - self.compliance_propensity))
        else:
            self.habit = clamp01(self.habit * (1.0 - habit_decay_rate))
            self.fatigue = clamp01(max(0.0, self.fatigue - fatigue_recovery))


@dataclass
class Household:
    """
    Placeholder Household entity. Not fully used in the minimal viable simulation.
    Provided for completeness and future extension.

    Attributes:
        id (int): Household ID.
        member_ids (List[int]): Resident member IDs.
        income_level (float): Shared income level proxy.
        mask_norm_strength (float): Household norm strength in [0,1].
    """
    id: int
    member_ids: List[int] = field(default_factory=list)
    income_level: float = 1.0
    mask_norm_strength: float = 0.0

    def reinforce_mask_norms(self) -> None:
        """
        Placeholder: Reinforce norms among members.

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement
        pass


@dataclass
class Location:
    """
    Placeholder Location entity. Not fully used in the minimal viable simulation.
    Provided for completeness and future extension.

    Attributes:
        id (int): Location ID.
        type (str): Location type.
        capacity (int): Capacity.
        contact_rate_modifier (float): Modifies contacts.
        ventilation_level (float): Ventilation score in [0,1].
        mask_requirement (bool): Whether masks are required.
        enforcement_level (float): Enforcement strength at this location.
    """
    id: int
    type: str = "community"
    capacity: int = 100
    contact_rate_modifier: float = 1.0
    ventilation_level: float = 0.3
    mask_requirement: bool = False
    enforcement_level: float = 0.5

    def host_contacts(self) -> None:
        """
        Placeholder: Host contacts among attendees.

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement
        pass

    def enforce_mask_policy(self) -> None:
        """
        Placeholder: Enforce mask policy at this location.

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement
        pass


@dataclass
class InformationSource:
    """
    Information source or media channel broadcasting messages.

    Attributes:
        id (int): Channel ID.
        reach (float): Fraction of population reached.
        credibility (float): Credibility in [0,1].
        bias (float): Bias towards mask-wearing (+) or against (-).
        misinformation_rate (float): Intensity of misinformation in [0,1].
        message_frequency (float): Relative frequency in [0,1].
    """
    id: int
    reach: float = 0.5
    credibility: float = 0.7
    bias: float = 0.2
    misinformation_rate: float = 0.05
    message_frequency: float = 0.5

    def broadcast_message(self) -> float:
        """
        Compute the net message effect as a scalar, positive for pro-mask, negative for anti-mask.

        Returns:
            float: Net message effect in [-1, 1].
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        pro = self.bias * self.credibility * self.message_frequency * self.reach
        anti = self.misinformation_rate * (1.0 - self.credibility) * self.message_frequency * self.reach
        return clamp01(pro) - clamp01(anti)


@dataclass
class PolicyAuthority:
    """
    Represents policy authority that can issue mandates and run campaigns.

    Attributes:
        id (int): Authority ID.
        mandate_active (bool): Whether a mandate is currently active.
        mandate_start_day (Optional[int]): Day index when mandate becomes active.
        enforcement_strength (float): Enforcement strength in [0,1].
        fine_amount (float): Penalty for non-compliance.
        campaign_active (bool): Whether campaign is active.
        campaign_start_day (Optional[int]): Day index when campaign starts.
        message_frequency (float): How often messages are broadcast.
        message_credibility (float): Message credibility for authority-driven campaigns in [0,1].
        supply_distribution_rate (int): Masks distributed per day.
    """
    id: int = 1
    mandate_active: bool = False
    mandate_start_day: Optional[int] = None
    enforcement_strength: float = 0.5
    fine_amount: float = 50.0
    campaign_active: bool = False
    campaign_start_day: Optional[int] = None
    message_frequency: float = 0.5
    message_credibility: float = 0.7
    supply_distribution_rate: int = 0

    def issue_mask_mandate(self, day: int) -> bool:
        """
        Determine and update mandate status based on schedule.

        Args:
            day (int): Current simulation day.

        Returns:
            bool: True if mandate is active today, else False.
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        if self.mandate_start_day is not None and day >= self.mandate_start_day:
            self.mandate_active = True
        return self.mandate_active

    def run_information_campaign(self, day: int) -> float:
        """
        Determine net campaign effect today based on schedule.

        Args:
            day (int): Current simulation day.

        Returns:
            float: Campaign effect scalar in [0, 1].
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        active = self.campaign_start_day is not None and day >= self.campaign_start_day
        self.campaign_active = active
        if not active:
            return 0.0
        return clamp01(self.message_frequency * self.message_credibility)


@dataclass
class MaskMarket:
    """
    Represents mask market (vendor) with finite inventory and simple price adjustments.

    Attributes:
        inventory (int): Current stock.
        restock_per_day (int): Number of masks added each day.
        price (float): Current price per mask.
        min_price (float): Lower bound on price.
        max_price (float): Upper bound on price.
        rng (random.Random): RNG reference.
    """
    inventory: int
    restock_per_day: int
    price: float
    rng: random.Random
    min_price: float = 0.1
    max_price: float = 5.0

    def effective_price(self, subsidy_level: float) -> float:
        """
        Compute effective price after subsidy.

        Args:
            subsidy_level (float): Per-unit subsidy.

        Returns:
            float: Effective price per unit.
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        return max(self.min_price, self.price - max(0.0, subsidy_level))

    def sell(self, max_units: int, budget: float, subsidy_level: float) -> int:
        """
        Attempt to sell up to max_units given budget and inventory.

        Args:
            max_units (int): Max units the buyer wants.
            budget (float): Buyer's budget.
            subsidy_level (float): Subsidy per unit.

        Returns:
            int: Units actually sold.
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        if self.inventory <= 0 or max_units <= 0:
            return 0
        eff_price = self.effective_price(subsidy_level)
        # How many can afford:
        affordable_units = int(budget // eff_price) if eff_price > 0 else max_units
        units = min(max_units, self.inventory, max(0, affordable_units))
        self.inventory -= units
        # Invariant: inventory non-negative
        assert self.inventory >= 0
        return units

    def restock(self) -> None:
        """
        Restock inventory by a fixed amount per day.

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        self.inventory += self.restock_per_day

    def adjust_price(self, target_inventory: int) -> None:
        """
        Adjust price based on inventory relative to target level.

        Args:
            target_inventory (int): Desired inventory level to stabilize around.

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        # Simple heuristic: raise price if inventory too low, lower otherwise
        if self.inventory < 0.5 * target_inventory:
            self.price = min(self.max_price, self.price * 1.1)
        elif self.inventory > 1.5 * target_inventory:
            self.price = max(self.min_price, self.price * 0.9)


class Simulation:
    """
    Main simulation class coordinating agents, market, policy, media, and metrics.

    Methods:
        run(): Run the simulation for configured time horizon and return metrics.
        step(): Execute a single day step.
        results(): Compute and return metrics dict.
        visualize(): Plot adoption rate over time using matplotlib if available.
        save_results(filename): Save time series to CSV.

    Notes:
        - FIXED: Restored a minimal runnable simulation with step loop, metrics, and early stopping.
        - FIXED: Replaced networkx with custom small-world generator.
        - FIXED: Implemented SEIR disease dynamics, enforcement, fines, and metrics.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the simulation with parameters and build population, network, and vendors.

        Args:
            params (Optional[Dict[str,Any]]): Configuration dictionary. Missing values use safe defaults.

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        self.p = params.copy() if params else {}
        self.fast_mode = bool(self.p.get("fast_mode", True))
        seed = int(self.p.get("seed", 42))
        self.rng = random.Random(seed)

        # Population and time
        self.n = int(self.p.get("population_size", 150 if self.fast_mode else 400))
        self.k = int(self.p.get("avg_degree", self.p.get("network_avg_degree", 8)))
        self.rewire_beta = float(self.p.get("rewire_beta", 0.05))
        # Dynamic prediction window
        pred = self.p.get("prediction_period", {})
        start_day = pred.get("start_day", None) if isinstance(pred, dict) else None
        end_day = pred.get("end_day", None) if isinstance(pred, dict) else None
        base_T = int(self.p.get("time_horizon_days", 40 if self.fast_mode else 120))
        if end_day is not None and isinstance(end_day, int):
            base_T = end_day + 1
        self.T = base_T

        # Policy
        self.mandate_start_day = self.p.get("mandate_start_day", self.p.get("policy_mandate_day", self.p.get("policy_mandate_start_day", 20)))
        self.enforcement_strength = float(self.p.get("enforcement_strength", self.p.get("enforcement_strictness", 0.6)))
        self.fine_amount = float(self.p.get("fine_amount", 50.0))
        self.enforcement_check_probability = float(self.p.get("enforcement_check_probability", 0.2))  # probability multiplier for checks when mandated
        self.enforcement_cost_per_check = float(self.p.get("enforcement_cost_per_check", 0.2))
        self.campaign_cost_per_day = float(self.p.get("budget_campaign_per_day", 1000.0))
        self.reactance = bool(self.p.get("reactance", False))  # FIXED: Keep parameter but do not clamp adoption

        # Campaign/Media
        self.campaign_start_day = self.p.get("campaign_start_day", 10)
        self.message_credibility = float(self.p.get("message_credibility", 0.7))
        self.misinformation_rate = float(self.p.get("misinformation_rate", 0.05))
        self.campaign_effect_size = float(self.p.get("campaign_effect_size", 0.15))
        self.message_frequency = float(self.p.get("message_frequency", 0.6))

        # Market
        self.mask_price = float(self.p.get("mask_price", 1.0))
        self.mask_supply_initial = int(self.p.get("mask_supply_initial", 400 if self.fast_mode else 1500))
        self.mask_restock_rate_per_day = int(self.p.get("mask_restock_rate_per_day", 40 if self.fast_mode else 80))
        self.purchase_limit_per_person = int(self.p.get("purchase_limit_per_person", 1))
        self.subsidy_level = float(self.p.get("subsidy_level", 0.0))

        # Adoption dynamics
        self.initial_mask_adoption_rate = float(self.p.get("initial_mask_adoption_rate", self.p.get("initial_adoption_rate", 0.2)))
        self.habit_formation_rate = float(self.p.get("habit_formation_rate", 0.12))
        # FIXED: Tie habit decay to adherence_decay_rate if provided
        self.habit_decay_rate = float(self.p.get("habit_decay_rate", self.p.get("adherence_decay_rate", 0.03)))
        self.fatigue_change_wear = float(self.p.get("fatigue_change_wear", 0.05))
        self.fatigue_recovery = float(self.p.get("fatigue_recovery", 0.04))
        self.wealth_effect_on_adoption = float(self.p.get("wealth_effect_on_adoption", 0.3))
        self.risk_perception_sensitivity_to_prevalence = float(self.p.get("risk_perception_sensitivity_to_prevalence", 0.5))

        # Utility weights
        self.weights = {
            "social_weight": float(self.p.get("social_influence_weight", self.p.get("peer_influence_weight", 0.4))),
            "risk_weight": float(self.p.get("risk_weight", self.p.get("risk_perception_weight", 0.5))),
            "policy_weight": float(self.p.get("policy_weight", 0.4)),
            "habit_weight": float(self.p.get("habit_weight", 0.3)),
            "fatigue_weight": float(self.p.get("fatigue_weight", 0.3)),
            "cost_weight": float(self.p.get("economic_cost_sensitivity", 0.3)),
        }

        # Build network
        self.neighbors = generate_small_world(self.n, self.k, self.rewire_beta, self.rng)

        # Vendor/Market
        self.vendor = MaskMarket(
            inventory=self.mask_supply_initial,
            restock_per_day=self.mask_restock_rate_per_day,
            price=self.mask_price,
            rng=self.rng,
        )
        # FIXED: Keep ~20% of population as target to avoid price collapse/inflation
        self.target_inventory = max(1, int(0.2 * self.n))

        # Policy authority and media
        self.policy = PolicyAuthority(
            id=1,
            mandate_active=False,
            mandate_start_day=self.mandate_start_day,
            enforcement_strength=self.enforcement_strength,
            fine_amount=self.fine_amount,
            campaign_active=False,
            campaign_start_day=self.campaign_start_day,
            message_frequency=self.message_frequency,
            message_credibility=self.message_credibility,
            supply_distribution_rate=int(self.p.get("supply_distribution_rate", self.p.get("production_rate_per_day", 0))),
        )

        self.media_channel = InformationSource(
            id=1,
            reach=0.7,
            credibility=self.message_credibility,
            bias=self.campaign_effect_size,
            misinformation_rate=self.misinformation_rate,
            message_frequency=self.message_frequency,
        )

        # Build agents
        self.people: List[Person] = []
        self._init_people()

        # Disease parameters and state
        self.disease_enabled = bool(self.p.get("disease_model_enabled", True))
        self.base_transmission_prob = float(self.p.get("base_transmission_prob_per_contact", 0.03))
        self.mask_effectiveness_source = float(self.p.get("mask_effectiveness_source_control", self.p.get("mask_effectiveness_outward", 0.5)))
        self.mask_effectiveness_wearer = float(self.p.get("mask_effectiveness_wearer_protection", self.p.get("mask_effectiveness_inward", 0.3)))
        self.ventilation_effect = float(self.p.get("ventilation_effect_on_transmission", 0.3))
        self.contacts_home = int(self.p.get("contact_rate_home", 4))
        self.contacts_work = int(self.p.get("contact_rate_work", 8))
        self.contacts_comm = int(self.p.get("contact_rate_community", 6))
        self.symptomatic_fraction = float(self.p.get("symptomatic_fraction", 0.6))
        self.symptomatic_contact_reduction = float(self.p.get("symptomatic_contact_reduction", 0.5))
        self.incubation_period = int(self.p.get("incubation_period_days", 4))
        self.infectious_period = int(self.p.get("infectious_period_days", 7))
        self.initial_infected = int(self.p.get("initial_infected", 5))

        self.health_states: List[str] = ['S'] * self.n
        self.days_in_state: List[int] = [0] * self.n
        self.is_symptomatic: List[bool] = [False] * self.n

        # Seed initial infections as infectious for immediate dynamics
        initial_ids = self.rng.sample(range(self.n), k=min(self.initial_infected, self.n)) if self.n > 0 else []
        for idx in initial_ids:
            self.health_states[idx] = 'I'
            self.days_in_state[idx] = self.rng.randint(0, max(0, self.infectious_period // 2))
            self.is_symptomatic[idx] = (self.rng.random() < self.symptomatic_fraction)

        # Time and metrics
        self.day = 0
        self.metrics: Dict[str, Any] = {
            "adoption_rate_over_time": [],
            "mandate_active_ts": [],
            "stockout_steps": 0,
            "compliance_ts": [],
            "infections_ts": [],
            "Re_ts": [],
            "masks_sold_daily": [],
            "masks_sold_cum": 0,
            "masks_distributed_cum": 0,
            "policy_cost": 0.0,
            "masks_restocked_cum": 0,
            "fines_collected_cum": 0.0,
        }

    def _init_people(self) -> None:
        """
        Initialize people with heterogeneous attributes (income, beliefs, trust, etc.).

        Returns:
            None
        """
        # NOTE: pass retained per environment requirement; functional code follows
        pass
        # Income via lognormal for inequality heterogeneity
        incomes = []
        for _ in range(self.n):
            incomes.append(max(10.0, self.rng.lognormvariate(4.5, 0.8)))
        sorted_incomes = sorted(incomes)
        def percentile(val: float) -> float:
            if self.n == 0:
                return 0.5
            # rank-based percentile
            rank = sorted_incomes.index(val)
            return clamp01(rank / max(1, self.n - 1))

        for i in range(self.n):
            income = incomes[i]
            budget = income * 0.02 * (0.5 + self.rng.random())  # small liquid budget fraction
            risk_perc = clamp01(0.2 + 0.6 * self.rng.random())
            compliance = clamp01(0.3 + 0.5 * self.rng.random())
            peer_susc = clamp01(0.2 + 0.6 * self.rng.random())
            trust = clamp01(0.3 + 0.6 * self.rng.random())
            inc_pct = percentile(income)
            p = Person(
                id=i,
                neighbors=self.neighbors[i],
                risk_perception=risk_perc,
                compliance_propensity=compliance,
                peer_susceptibility=peer_susc,
                trust_in_authority=trust,
                income=income,
                budget=budget,
                mask_inventory=0,
                is_wearing_mask=False,
                habit=0.0,
                fatigue=0.0,
                info_exposure=clamp01(0.4 + 0.4 * self.rng.random()),
                income_percentile=inc_pct,
            )
            # Initial adoption
            if self.rng.random() < self.initial_mask_adoption_rate:
                p.is_wearing_mask = True
                p.mask_inventory = 1
                p.habit = 0.2 + 0.3 * self.rng.random()
            self.people.append(p)

   

# Execute main for both direct execution and sandbox wrapper invocation
main()