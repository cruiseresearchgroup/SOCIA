import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple


# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT if PROJECT_ROOT else "", DATA_PATH if DATA_PATH else "")


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a value x to the closed interval [lo, hi].

    Args:
        x: Input value.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        float: Clamped value.
    """
    pass
    val = max(lo, min(hi, x))
    return val


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.

    Args:
        n: Numerator.
        d: Denominator.
        default: Value to return when denominator equals zero.

    Returns:
        float: Result of division or default.
    """
    pass
    if d == 0:
        return default
    return n / d


def sigmoid(x: float) -> float:
    """
    Numerically stable logistic sigmoid.

    Args:
        x: Input value.

    Returns:
        float: Sigmoid(x) in [0,1].
    """
    pass
    try:
        if x >= 0:
            z = math.exp(-x)
            val = 1.0 / (1.0 + z)
            return val
        else:
            z = math.exp(x)
            val = z / (1.0 + z)
            return val
    except OverflowError:
        val = 0.0 if x < 0 else 1.0
        return val


def ring_small_world(n: int, k: int, p: float, rng: random.Random) -> Dict[int, set]:
    """
    Build a ring-lattice small-world network with robust rewiring.

    Args:
        n: Number of nodes.
        k: Average degree target (even integer preferred).
        p: Rewiring probability.
        rng: Random state.

    Returns:
        dict: Adjacency sets for each node.
    """
    pass
    # FIXED: Early-return for degenerate networks to avoid spurious edges/self-loops.
    if n <= 1 or k <= 0:
        return {i: set() for i in range(max(0, n))}
    # Base ring lattice
    adj = {i: set() for i in range(n)}
    half = max(1, min(k // 2, (n - 1) // 2))
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
    # FIXED: Guard rewiring so it only applies to existing edges and avoids unintended degree inflation.
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if rng.random() < p:
                # Only rewire if the base edge exists
                # FIXED: Skip rewiring if j not in adj[i] (edge-case for small n).
                if j not in adj[i]:
                    continue
                adj[i].discard(j)
                adj[j].discard(i)
                if len(adj[i]) >= n - 1:
                    continue
                # Bounded attempts to find a new neighbor
                for _ in range(10):
                    u = rng.randrange(n)
                    if u != i and u not in adj[i]:
                        adj[i].add(u)
                        adj[u].add(i)
                        break
                else:
                    # fallback: restore original edge if no candidate found
                    adj[i].add(j)
                    adj[j].add(i)
    return adj


class Person:
    """
    Represents an individual agent with social, economic, and behavioral attributes.

    Attributes:
        pid (int): Unique person identifier.
        age (int): Age in years.
        income (float): Annual income proxy.
        ses_quintile (int): Socioeconomic status quintile [1..5].
        risk (float): Perceived disease risk [0,1].
        trust (float): Trust in institutions/authority [0,1].
        compliance_trait (float): Compliance propensity trait [0,1].
        attitude (float): Attitudinal leaning toward adoption [-1,1].
        threshold (float): Social threshold for adoption [0,1].
        baseline (float): Baseline adoption propensity [0,1].
        fatigue (float): Fatigue from sustained adoption [0,1].
        habit (float): Habit strength [0,1].
        adopting (bool): Whether currently adopting the behavior.
        mask_inventory (int): On-hand inventory of masks.
        media_subscribed (bool): High exposure to media outlet.
        fines_paid (float): Cumulative fines paid.
        disease_state (str): One of 'S','E','I','R'.
        days_in_state (int): Days elapsed in current disease state.
    """
    pass

    def __init__(
        self,
        pid: int,
        age: int,
        income: float,
        risk: float,
        trust: float,
        compliance_trait: float,
        attitude: float,
        threshold: float,
        baseline: float,
        media_subscribed: bool,
    ):
        """
        Initialize a Person.

        Args:
            pid: Person id.
            age: Age.
            income: Income proxy.
            risk: Risk perception in [0,1].
            trust: Trust in [0,1].
            compliance_trait: Compliance trait [0,1].
            attitude: Attitude [-1,1].
            threshold: Social threshold [0,1].
            baseline: Baseline adoption propensity [0,1].
            media_subscribed: If true, high media exposure.
        """
        pass
        self.pid = pid
        self.age = age
        self.income = income
        self.ses_quintile = 3  # will be assigned after population creation
        self.risk = clamp(risk, 0.0, 1.0)
        self.trust = clamp(trust, 0.0, 1.0)
        self.compliance_trait = clamp(compliance_trait, 0.0, 1.0)
        self.attitude = clamp(attitude, -1.0, 1.0)
        self.threshold = clamp(threshold, 0.0, 1.0)
        self.baseline = clamp(baseline, 0.0, 1.0)
        self.fatigue = 0.0
        self.habit = 0.0
        self.adopting = False
        self.mask_inventory = 0
        self.media_subscribed = media_subscribed
        self.fines_paid = 0.0
        # Disease module fields
        self.disease_state = 'S'
        self.days_in_state = 0


class Location:
    """
    A location that agents may visit; may enforce policy via fines or entry refusal.

    Attributes:
        name (str): Location identifier.
        enforcement_level (float): Probability of enforcement when mandate active.
        mandate_sensitive (bool): Whether enforcement ties to mandate.
        entry_refusal_if_unmasked (bool): If True, may refuse entry to unmasked visitors.
        observed_norms (float): Share of masked among recent visitors.
    """
    pass

    def __init__(self, name: str, enforcement_level: float = 0.0, mandate_sensitive: bool = True, entry_refusal_if_unmasked: bool = False):
        """
        Initialize a Location.

        Args:
            name: Name of the location.
            enforcement_level: Probability of enforcement when mandate active.
            mandate_sensitive: True if enforcement applies during mandates.
            entry_refusal_if_unmasked: If True, location may refuse entry to unmasked visitors.
        """
        pass
        # FIXED: Added entry_refusal_if_unmasked attribute to support entry refusal enforcement behavior.
        self.name = name
        self.enforcement_level = clamp(enforcement_level, 0.0, 1.0)
        self.mandate_sensitive = mandate_sensitive
        self.entry_refusal_if_unmasked = entry_refusal_if_unmasked
        self.observed_norms = 0.0


class Government:
    """
    Government entity controlling policy interventions such as mandates and fines.

    Attributes:
        mandate_start_day (int): Day mandate starts (inclusive).
        mandate_end_day (int): Day mandate ends (inclusive).
        enforcement_prob (float): Base enforcement probability.
        fine_amount (float): Fine amount per violation.
        subsidy_amount (float): Monetary incentive; small effect in adoption utility.
        subsidy_effect_weight (float): Weight translating subsidy to utility.
    """
    pass

    def __init__(
        self,
        mandate_start_day: int,
        mandate_end_day: int,
        enforcement_prob: float,
        fine_amount: float,
        subsidy_amount: float = 0.0,
        subsidy_effect_weight: float = 0.0,
    ):
        """
        Initialize Government.

        Args:
            mandate_start_day: Policy start day.
            mandate_end_day: Policy end day.
            enforcement_prob: Base enforcement probability.
            fine_amount: Fine per violation.
            subsidy_amount: Subsidy nominal amount.
            subsidy_effect_weight: Weight mapping subsidy to adoption utility.
        """
        pass
        self.mandate_start_day = max(0, mandate_start_day)
        self.mandate_end_day = max(self.mandate_start_day, mandate_end_day)
        self.enforcement_prob = clamp(enforcement_prob, 0.0, 1.0)
        self.fine_amount = max(0.0, fine_amount)
        self.subsidy_amount = max(0.0, subsidy_amount)
        self.subsidy_effect_weight = max(0.0, subsidy_effect_weight)

    def mandate_active(self, day: int) -> bool:
        """
        Check if mandate is active.

        Args:
            day: Current day.

        Returns:
            bool: True if active.
        """
        pass
        active = self.mandate_start_day <= day <= self.mandate_end_day
        return active

    def policy_intensity(self, day: int) -> float:
        """
        Continuous policy intensity to feed adoption utility.

        Args:
            day: Current day.

        Returns:
            float: Intensity in [0,1].
        """
        pass
        intensity = 1.0 if self.mandate_active(day) else 0.0
        return intensity

    def policy_effect(self) -> float:
        """
        Compute scalar policy effect from subsidy settings for adoption utility.

        Returns:
            float: Weighted policy effect.
        """
        pass
        effect = self.subsidy_amount * self.subsidy_effect_weight
        return effect


class Media:
    """
    Media outlet broadcasting a daily intensity signal affecting adoption.

    Attributes:
        baseline_intensity (float): Base intensity.
        shock_day (int): Day of media shock.
        shock_magnitude (float): Additive magnitude at shock day.
    """
    pass

    def __init__(self, baseline_intensity: float, shock_day: int, shock_magnitude: float):
        """
        Initialize Media.

        Args:
            baseline_intensity: Baseline signal intensity.
            shock_day: Day of a one-time shock.
            shock_magnitude: Added intensity at shock day.
        """
        pass
        self.baseline_intensity = clamp(baseline_intensity, 0.0, 2.0)
        self.shock_day = max(0, shock_day)
        self.shock_magnitude = clamp(shock_magnitude, 0.0, 2.0)

    def signal(self, day: int, subscribed: bool) -> float:
        """
        Compute media signal for a person on a given day.

        Args:
            day: Current day.
            subscribed: Whether the person is highly exposed/subscribed.

        Returns:
            float: Media signal intensity.
        """
        pass
        base = self.baseline_intensity
        if day == self.shock_day:
            base += self.shock_magnitude
        # Subscribed users get full strength; others receive a diminished fraction
        intensity = base if subscribed else 0.5 * base
        return intensity


class Retailer:
    """
    Retailer carrying mask inventory with restocking and basic pricing.

    Attributes:
        name (str): Retailer identifier.
        inventory (int): Current inventory.
        restock_rate (int): Daily restock quantity.
        price (float): Unit price.
        ration_limit (int): Max units sold to a person per day.
        outage_days (int): Count of days with zero inventory.
    """
    pass

    def __init__(self, name: str, initial_inventory: int, restock_rate: int, price: float, ration_limit: int = 5):
        """
        Initialize Retailer.

        Args:
            name: Retailer name.
            initial_inventory: Starting inventory.
            restock_rate: Units to restock each day.
            price: Unit price.
            ration_limit: Per-person daily purchase limit.
        """
        pass
        self.name = name
        self.inventory = max(0, int(initial_inventory))
        self.restock_rate = max(0, int(restock_rate))
        self.price = max(0.01, float(price))
        self.ration_limit = max(1, int(ration_limit))
        self.outage_days = 0

    def restock(self) -> None:
        """
        Restock inventory by restock_rate.
        """
        pass
        self.inventory += self.restock_rate

    def can_sell(self) -> bool:
        """
        Check availability.

        Returns:
            bool: True if inventory > 0.
        """
        pass
        available = self.inventory > 0
        return available

    def sell(self, desired_qty: int) -> int:
        """
        Sell up to desired quantity, limited by ration_limit and inventory.

        Args:
            desired_qty: Desired units.

        Returns:
            int: Units actually sold.
        """
        pass
        qty = max(0, min(desired_qty, self.ration_limit, self.inventory))
        self.inventory -= qty
        return qty


def adoption_probability(
    person: Person,
    peer_share: float,
    policy_intensity: float,
    media_signal: float,
    policy_effect_scalar: float,
) -> float:
    """
    Compute adoption probability for a person.

    Args:
        person: Person instance.
        peer_share: Fraction of neighbors adopting.
        policy_intensity: Government mandate intensity [0,1].
        media_signal: Media signal intensity [0,inf).
        policy_effect_scalar: Weighted policy monetary effect.

    Returns:
        float: Adoption probability in [0,1].
    """
    pass
    # FIXED: Reintroduced logistic adoption combining peer/social, policy, media, and personal factors.
    w_social = 0.8
    w_policy = 1.0
    w_media = 0.6
    w_personal = 0.5

    # Base utility linear combination; constants calibrated for qualitative S-shape.
    linear = (
        w_social * (peer_share - person.threshold)
        + w_policy * (person.trust * policy_intensity + policy_effect_scalar)
        + w_media * media_signal
        + w_personal * (person.baseline + person.risk - 0.5 + 0.4 * person.compliance_trait + 0.4 * person.attitude)
        + 0.6 * person.habit
        - 0.8 * person.fatigue
        - 2.2
    )
    prob = clamp(sigmoid(3.5 * linear), 0.0, 1.0)
    return prob


class SocialAdoptionSimulation:
    """
    Main simulation class coordinating agents, environment, policy, and metrics.

    This class implements:
    - Media broadcast
    - Market restocking and rationed sales
    - Peer influence via a small-world network
    - Adoption decision and backsliding
    - Visits/enforcement with fines influencing future behavior and entry refusal
    - Optional SEIR-like disease dynamics with counterfactual tracking
    - Metrics aggregation and saving/visualization utilities
    """
    pass

    def __init__(
        self,
        population_size: int,
        time_horizon_days: int,
        random_seed: int = 42,
        include_disease_module: bool = False,
        media_params: Optional[Dict[str, Any]] = None,
        policy_params: Optional[Dict[str, Any]] = None,
        network_mean_degree: int = 8,
        network_rewiring_prob: float = 0.05,
    ):
        """
        Initialize the simulation.

        Args:
            population_size: Number of agents.
            time_horizon_days: Simulation length in days.
            random_seed: RNG seed.
            include_disease_module: Whether to run epidemiological module.
            media_params: Parameters for Media.
            policy_params: Parameters for Government.
            network_mean_degree: Mean degree for the small-world network.
            network_rewiring_prob: Rewiring probability for the small-world network.
        """
        pass
        # FIXED: Restored a runnable simulation with core entities, loop, and metrics.
        self.population_size = int(population_size)
        self.days = int(time_horizon_days)
        self.rng = random.Random(int(random_seed))
        self.include_disease_module = include_disease_module

        # Entities
        self.people: List[Person] = []
        self.network: Dict[int, set] = {}

        # Government policy with defaults including a short mandate
        policy_params = policy_params or {
            "mandate_start_day": max(1, self.days // 3),
            "mandate_end_day": max(1, self.days // 2),
            "enforcement_prob": 0.15,
            "fine_amount": 50.0,
            "subsidy_amount": 10.0,
            "subsidy_effect_weight": 0.01,
        }
        self.gov = Government(**policy_params)

        # Media with a small shock on day ~1/4
        media_params = media_params or {
            "baseline_intensity": 0.25,
            "shock_day": max(1, self.days // 4),
            "shock_magnitude": 0.5,
        }
        self.media = Media(**media_params)

        # Retailers pool
        self.retailers: List[Retailer] = [
            Retailer("Pharmacy-A", initial_inventory=200, restock_rate=60, price=1.0, ration_limit=5),
            Retailer("Pharmacy-B", initial_inventory=200, restock_rate=60, price=1.1, ration_limit=5),
            Retailer("Corner-Store", initial_inventory=150, restock_rate=40, price=1.2, ration_limit=3),
        ]

        # Public locations; some enforce more strongly
        # FIXED: Enabled entry refusal option on mandate-sensitive locations to model realistic enforcement behavior.
        self.locations: List[Location] = [
            Location("Transit", enforcement_level=0.2, mandate_sensitive=True, entry_refusal_if_unmasked=True),
            Location("Workplace", enforcement_level=0.3, mandate_sensitive=True, entry_refusal_if_unmasked=True),
            Location("Park", enforcement_level=0.05, mandate_sensitive=False, entry_refusal_if_unmasked=False),
        ]

        # Network parameters
        self.network_mean_degree = int(network_mean_degree)
        self.network_rewiring_prob = float(network_rewiring_prob)

        # Metrics tracked
        self.metrics: Dict[str, Any] = {}
        self.adoption_rate_over_time: List[float] = []
        self.Rt_over_time: List[float] = []
        self.new_infections_over_time: List[int] = []
        self.counterfactual_infections_over_time: List[int] = []
        self.total_fines_count: int = 0
        # FIXED: Added enforcement_actions_count to include fines + entry refusals.
        self.enforcement_actions_count: int = 0
        self.total_fines_value: float = 0.0
        self.inventory_outage_retailer_days: int = 0
        self.peer_share_cache: List[float] = []
        # FIXED: Norms influence cache from previous day
        self.prev_day_norms: float = 0.0
        # FIXED: Added mask purchase tracking metrics.
        self.masks_purchased_cumulative: int = 0
        self.masks_purchased_daily: List[int] = []
        # FIXED: Added daily compliance tracking (masked_visitors, total_visitors) for mandate-sensitive locations.
        self.daily_compliance: List[Tuple[int, int]] = []
        # FIXED: Adoption streak tracking for average compliance duration metric.
        self.current_streak: List[int] = []
        self.completed_streaks: List[int] = []

    def initialize_population(self) -> None:
        """
        Create agents with heterogeneous attributes and assign SES quintiles.

        Returns:
            None
        """
        pass
        ages = [max(18, int(self.rng.gauss(40, 15))) for _ in range(self.population_size)]
        # FIXED: Calibrated incomes to realistic annual range to make prices meaningful.
        incomes = [self.rng.uniform(15000.0, 150000.0) for _ in range(self.population_size)]
        risks = [clamp(self.rng.random(), 0.0, 1.0) for _ in range(self.population_size)]
        trusts = [clamp(self.rng.random(), 0.0, 1.0) for _ in range(self.population_size)]
        compliances = [clamp(self.rng.random(), 0.0, 1.0) for _ in range(self.population_size)]
        attitudes = [clamp(self.rng.gauss(0.0, 0.5), -1.0, 1.0) for _ in range(self.population_size)]
        thresholds = [clamp(self.rng.gauss(0.5, 0.15), 0.0, 1.0) for _ in range(self.population_size)]
        baselines = [clamp(self.rng.gauss(0.3, 0.15), 0.0, 1.0) for _ in range(self.population_size)]
        subscriptions = [self.rng.random() < 0.7 for _ in range(self.population_size)]

        for i in range(self.population_size):
            p = Person(
                pid=i,
                age=ages[i],
                income=incomes[i],
                risk=risks[i],
                trust=trusts[i],
                compliance_trait=compliances[i],
                attitude=attitudes[i],
                threshold=thresholds[i],
                baseline=baselines[i],
                media_subscribed=subscriptions[i],
            )
            self.people.append(p)

        # Assign SES quintiles by rank
        sorted_incomes = sorted([(p.income, idx) for idx, p in enumerate(self.people)])
        for rank, (_, idx) in enumerate(sorted_incomes):
            q = int(5 * rank / max(1, self.population_size))
            q = min(4, q)  # 0-based 0..4
            self.people[idx].ses_quintile = q + 1

        # Initialize streaks
        # FIXED: Initialize per-agent streak state to track compliance duration.
        self.current_streak = [0] * self.population_size
        self.completed_streaks = []

        # Initial adoption and inventory
        init_frac = 0.12
        init_ids = set(self.rng.sample(range(self.population_size), max(1, int(self.population_size * init_frac))))
        for i in init_ids:
            self.people[i].adopting = True
            self.people[i].habit = 0.35 + 0.3 * self.rng.random()
            self.people[i].mask_inventory = self.rng.randint(1, 5)
            # FIXED: Seed initial compliance streaks for adopters.
            self.current_streak[i] = 1

        # Initial infections if disease enabled: seed small
        if self.include_disease_module:
            infectious_seed = max(1, self.population_size // 50)
            seed_ids = set(self.rng.sample(range(self.population_size), infectious_seed))
            for i in seed_ids:
                self.people[i].disease_state = 'I'
                self.people[i].days_in_state = 0

        # Network
        # FIXED: Exposed network degree and rewiring probability to constructor parameters.
        self.network = ring_small_world(self.population_size, k=self.network_mean_degree, p=self.network_rewiring_prob, rng=self.rng)

    def compute_peer_share(self) -> List[float]:
        """
        Compute peer adoption share for each node in the network.

        Returns:
            list: Peer shares in [0,1] for each person.
        """
        pass
        shares = [0.0] * self.population_size
        for i, p in enumerate(self.people):
            neighbors = list(self.network.get(i, []))
            if not neighbors:
                shares[i] = 0.0
            else:
                shares[i] = safe_div(sum(1.0 if self.people[j].adopting else 0.0 for j in neighbors), len(neighbors), 0.0)
        self.peer_share_cache = shares
        return shares

    def daily_media_signal(self, day: int) -> List[float]:
        """
        Compute daily media signal per person.

        Args:
            day: Current day.

        Returns:
            list: Media signals.
        """
        pass
        signals = [self.media.signal(day, p.media_subscribed) for p in self.people]
        return signals

    def retailer_restock(self) -> None:
        """
        Restock all retailers.

        Returns:
            None
        """
        pass
        # FIXED: Count outages only at end-of-day. Removed outage counting here to avoid double-counting.
        for r in self.retailers:
            r.restock()

    def adoption_step(self, day: int, media_signals: List[float], peer_shares: List[float]) -> None:
        """
        Execute adoption decisions and update habit/fatigue.

        Args:
            day: Current day index.
            media_signals: Per-person media signals.
            peer_shares: Per-person peer adoption share.

        Returns:
            None
        """
        pass
        # FIXED: Reintroduced adoption decisions integrating media, policy, peer, and personal factors.
        # FIXED: Integrated observed norms from previous day into adoption probability.
        policy_intensity = self.gov.policy_intensity(day)
        policy_effect_scalar = self.gov.policy_effect()
        norms_effect = 0.3 * (self.prev_day_norms - 0.5)  # simple norms term centered at 0.5

        for i, p in enumerate(self.people):
            # Must have access to adopt (inventory on-hand)
            access_ok = p.mask_inventory > 0
            base_prob = adoption_probability(
                person=p,
                peer_share=peer_shares[i],
                policy_intensity=policy_intensity,
                media_signal=media_signals[i],
                policy_effect_scalar=policy_effect_scalar,
            )
            prob = clamp(base_prob + norms_effect, 0.0, 1.0)
            if not p.adopting:
                if access_ok and self.rng.random() < prob:
                    p.adopting = True
                    # FIXED: Start a new compliance streak upon adoption.
                    self.current_streak[i] = 1
                else:
                    # remains non-adopting; if streak somehow positive, close it
                    if self.current_streak[i] > 0:
                        self.completed_streaks.append(self.current_streak[i])
                        self.current_streak[i] = 0
            else:
                # Allow backslide due to fatigue with small probability
                drop = clamp(sigmoid(4.0 * (-prob + 0.25 + 0.6 * p.fatigue)), 0.0, 1.0)
                if self.rng.random() < drop:
                    # FIXED: Close streak on drop/disadoption.
                    if self.current_streak[i] > 0:
                        self.completed_streaks.append(self.current_streak[i])
                    p.adopting = False
                    self.current_streak[i] = 0
                else:
                    # Increase streak when continuing to adopt
                    self.current_streak[i] += 1
            # Update habit and fatigue
            if p.adopting:
                p.habit = clamp(p.habit + 0.08 * (1.0 - p.habit), 0.0, 1.0)
                p.fatigue = clamp(p.fatigue + 0.015, 0.0, 1.0)
            else:
                p.habit = clamp(p.habit - 0.03, 0.0, 1.0)
                p.fatigue = clamp(p.fatigue - 0.015, 0.0, 1.0)

    def purchase_step(self) -> None:
        """
        People attempt to purchase masks if adopting and low on stock.

        Returns:
            None
        """
        pass
        # FIXED: Implemented rationing-based purchasing; do not count outages here (counted end-of-day).
        # FIXED: Price-sensitive purchasing calibrated to realistic incomes.
        # FIXED: Added tracking for masks purchased (daily and cumulative).
        daily_sales = 0
        for p in self.people:
            # Demand heuristic: keep up to 7 units if adopting, else up to 2 units
            target_stock = 7 if p.adopting else 2
            need = max(0, target_stock - p.mask_inventory)
            if need <= 0:
                continue

            remaining = need
            for r in self.retailers:
                if remaining <= 0:
                    break
                if not r.can_sell():
                    continue
                # Affordability and success probability as a function of price/income and SES
                daily_income = max(1e-6, p.income / 365.0)
                afford_ratio = r.price / daily_income  # higher => harder to afford
                # Base success increases with SES; decreases with affordability ratio
                base_success = 0.65 + 0.06 * (p.ses_quintile - 3)
                # FIXED: Increase sensitivity now that income is realistic.
                price_penalty = 0.25 * math.tanh(afford_ratio - 0.3)
                success_prob = clamp(base_success - price_penalty, 0.05, 0.98)

                if self.rng.random() > success_prob:
                    continue

                buy_qty = min(remaining, r.ration_limit)
                sold = r.sell(buy_qty)
                if sold > 0:
                    p.mask_inventory += sold
                    remaining -= sold
                    daily_sales += sold
        # Track mask purchases
        self.masks_purchased_cumulative += daily_sales
        self.masks_purchased_daily.append(daily_sales)
        # Do not count outages here; counted once at end-of-day in step()

    def _compute_daily_visits(self) -> Dict[str, List[int]]:
        """
        Compute and return a snapshot of visitors per location for the day.

        Returns:
            Dict[str, List[int]]: Mapping from location name to list of visitor person IDs.
        """
        pass
        visitors_by_location: Dict[str, List[int]] = {loc.name: [] for loc in self.locations}
        for p in self.people:
            if self.rng.random() < 0.6:
                loc = self.rng.choice(self.locations)
                visitors_by_location[loc.name].append(p.pid)
        return visitors_by_location

    def visits_and_enforcement(self, day: int, visitors_by_location: Dict[str, List[int]]) -> None:
        """
        Simulate daily visits to locations and apply enforcement for non-compliance.

        Args:
            day: Current day index.
            visitors_by_location: Precomputed mapping of location to visitor IDs.

        Returns:
            None
        """
        pass
        # FIXED: Use cached visitation snapshot; compute norms and store average for next-day influence.
        mandate_active = self.gov.mandate_active(day)

        # Compute observed norms and daily compliance at mandate-sensitive locations
        norms_sum, norms_count = 0.0, 0
        masked_total, visitors_total = 0, 0
        for loc in self.locations:
            visitors = visitors_by_location.get(loc.name, [])
            if visitors:
                frac_masked = safe_div(sum(1.0 if self.people[i].adopting else 0.0 for i in visitors), len(visitors), 0.0)
                loc.observed_norms = frac_masked
                norms_sum += frac_masked
                norms_count += 1
                if mandate_active and loc.mandate_sensitive:
                    visitors_total += len(visitors)
                    masked_total += sum(1 for i in visitors if self.people[i].adopting)
            else:
                loc.observed_norms = 0.0
        if norms_count:
            self.prev_day_norms = norms_sum / norms_count
        else:
            # No visits observed: decay towards neutral 0.5
            self.prev_day_norms = 0.5 * self.prev_day_norms + 0.5 * 0.5

        # FIXED: Track daily compliance numerators/denominators for policy evaluation.
        if mandate_active:
            self.daily_compliance.append((masked_total, visitors_total))
        else:
            self.daily_compliance.append((0, 0))

        # Enforcement: fines and entry refusal
        for loc in self.locations:
            visitors = visitors_by_location.get(loc.name, [])
            if not visitors:
                continue
            refused_ids: List[int] = []
            for pid in visitors:
                p = self.people[pid]
                # Mask consumption: occasional usage leads to depletion
                if p.adopting and p.mask_inventory > 0 and self.rng.random() < 0.2:
                    p.mask_inventory -= 1
                # Enforcement applies when mandate active and location mandates
                apply_enforcement = mandate_active and loc.mandate_sensitive
                enforcement_prob = clamp(self.gov.enforcement_prob * loc.enforcement_level, 0.0, 1.0)
                if apply_enforcement and not p.adopting:
                    if self.rng.random() < enforcement_prob:
                        # FIXED: Enforcement actions now include potential entry refusal.
                        self.enforcement_actions_count += 1
                        if getattr(loc, "entry_refusal_if_unmasked", False) and self.rng.random() < 0.5:
                            # Entry refusal instead of fine
                            refused_ids.append(pid)
                        else:
                            # Fine
                            self.total_fines_count += 1
                            self.total_fines_value += self.gov.fine_amount
                            p.fines_paid += self.gov.fine_amount
                            # Enforcement nudge: slight increases in risk and compliance
                            p.risk = clamp(p.risk + 0.02, 0.0, 1.0)
                            p.compliance_trait = clamp(p.compliance_trait + 0.02, 0.0, 1.0)
            # If refused, remove from location visitors to reflect no entry to location-driven processes (e.g., disease)
            if refused_ids:
                visitors_by_location[loc.name] = [pid for pid in visitors if pid not in set(refused_ids)]

    def disease_step(self, day: int, visitors_by_location: Dict[str, List[int]]) -> None:
        """
        Execute SEIR-lite dynamics with simple location-based mixing.

        Args:
            day: Current day index.
            visitors_by_location: Precomputed mapping of location to visitor IDs.

        Returns:
            None
        """
        pass
        if not self.include_disease_module:
            return

        # Epidemiological parameters
        beta = 0.06  # baseline daily infection risk per contact unit
        incubation_days = 3
        infectious_days = 7
        mask_effectiveness = 0.4  # reduction factor due to masks

        new_infections = 0
        counterfactual_new_infections = 0
        infectious_count = sum(1 for p in self.people if p.disease_state == 'I')

        for loc in self.locations:
            visitors = visitors_by_location.get(loc.name, [])
            if not visitors:
                continue
            # Aggregate stats
            infectious_visitors = [pid for pid in visitors if self.people[pid].disease_state == 'I']
            susceptible_visitors = [pid for pid in visitors if self.people[pid].disease_state == 'S']
            if not infectious_visitors or not susceptible_visitors:
                continue

            frac_masked_inf = safe_div(sum(1.0 if self.people[pid].adopting else 0.0 for pid in infectious_visitors),
                                       len(infectious_visitors), 0.0)
            frac_masked_sus = safe_div(sum(1.0 if self.people[pid].adopting else 0.0 for pid in susceptible_visitors),
                                       len(susceptible_visitors), 0.0)
            # Effective infectious pressure reduced by masks on either side
            effective_pressure = len(infectious_visitors) * (
                (1.0 - mask_effectiveness * frac_masked_inf) * (1.0 - mask_effectiveness * frac_masked_sus)
            )
            # Counterfactual ignores masks
            cf_pressure = len(infectious_visitors)

            # Each susceptible experiences risk proportional to pressure/visitors
            risk = beta * safe_div(effective_pressure, max(1, len(visitors)), 0.0)
            cf_risk = beta * safe_div(cf_pressure, max(1, len(visitors)), 0.0)

            for pid in susceptible_visitors:
                p = self.people[pid]
                if self.rng.random() < risk:
                    p.disease_state = 'E'
                    p.days_in_state = 0
                    new_infections += 1
                # Counterfactual
                if self.rng.random() < cf_risk:
                    counterfactual_new_infections += 1

        # Progress disease states
        for p in self.people:
            if p.disease_state == 'E':
                p.days_in_state += 1
                if p.days_in_state >= incubation_days:
                    p.disease_state = 'I'
                    p.days_in_state = 0
            elif p.disease_state == 'I':
                p.days_in_state += 1
                if p.days_in_state >= infectious_days:
                    p.disease_state = 'R'
                    p.days_in_state = 0

        self.new_infections_over_time.append(new_infections)
        self.counterfactual_infections_over_time.append(counterfactual_new_infections)
        Rt_same_day = safe_div(new_infections, infectious_count, 0.0)
        # Improved Rt using a simple generation-interval proxy (serial interval ~5 days)
        serial = 5
        if len(self.new_infections_over_time) > serial:
            denom = self.new_infections_over_time[-serial - 1]
            Rt_gen = safe_div(new_infections, denom, 0.0)
        else:
            Rt_gen = Rt_same_day
        self.Rt_over_time.append(Rt_gen)

    def step(self, day: int) -> None:
        """
        Perform a single simulation day in sequence:
        media -> market restock -> purchases -> peer exposure -> adoption -> visits/enforcement -> disease -> metrics update.

        Args:
            day: Current day index.

        Returns:
            None
        """
        pass
        # FIXED: Reordered day sequence to allow purchases before adoption to remove one-day lag.
        media_signals = self.daily_media_signal(day)
        self.retailer_restock()
        self.purchase_step()  # moved before adoption to allow same-day adoption if inventory acquired
        peer_shares = self.compute_peer_share()
        self.adoption_step(day, media_signals, peer_shares)
        # FIXED: Cache visits once and reuse across modules
        visits_by_loc = self._compute_daily_visits()
        self.visits_and_enforcement(day, visits_by_loc)
        self.disease_step(day, visits_by_loc)

        # Count retailer outages once at end-of-day and update both global and per-retailer metrics
        # FIXED: Count outages once per day at a single point (end-of-day).
        daily_outages = 0
        for r in self.retailers:
            if r.inventory <= 0:
                r.outage_days += 1
                daily_outages += 1
        self.inventory_outage_retailer_days += daily_outages

        # Adoption rate aggregation
        adopt_rate = safe_div(sum(1 for p in self.people if p.adopting), self.population_size, 0.0)
        self.adoption_rate_over_time.append(adopt_rate)

    def run(self) -> None:
        """
        Run the simulation across all days.

        Returns:
            None
        """
        pass
        self.initialize_population()
        for day in range(self.days):
            self.step(day)
        # After loop, compute final metrics
        self.compute_metrics()

    def compute_metrics(self) -> None:
        """
        Compute required metrics, including equity and enforcement metrics.

        Returns:
            None
        """
        pass
        # Aggregated required metrics and corrected outage counting logic.
        adoption_series = self.adoption_rate_over_time[:]
        peak_adopt = max(adoption_series) if adoption_series else 0.0
        peak_day = adoption_series.index(peak_adopt) if adoption_series else -1
        t50 = next((i for i, v in enumerate(adoption_series) if v >= 0.5), None)
        # FIXED: Added time_to_70_percent_adoption as required by specification.
        t70 = next((i for i, v in enumerate(adoption_series) if v >= 0.7), None)

        # Sustained adoption post mandate
        post_window = 7
        sustained_val = None
        if self.days > self.gov.mandate_end_day + post_window:
            window_vals = adoption_series[self.gov.mandate_end_day + 1: self.gov.mandate_end_day + 1 + post_window]
            if window_vals:
                sustained_val = sum(window_vals) / len(window_vals)

        # Compliance distribution by group: age groups and SES quintiles at final day
        def age_group(age: int) -> str:
            if age < 25:
                return "<25"
            elif age < 45:
                return "25-44"
            elif age < 65:
                return "45-64"
            return "65+"

        by_age: Dict[str, float] = {}
        by_ses: Dict[str, float] = {}
        for label in ["<25", "25-44", "45-64", "65+"]:
            group = [p for p in self.people if age_group(p.age) == label]
            by_age[label] = safe_div(sum(1 for p in group if p.adopting), len(group), 0.0)

        for q in range(1, 6):
            group = [p for p in self.people if p.ses_quintile == q]
            by_ses[str(q)] = safe_div(sum(1 for p in group if p.adopting), len(group), 0.0)

        # Inequity metric: top vs bottom quintile adoption and access (inventory)
        q1 = [p for p in self.people if p.ses_quintile == 1]
        q5 = [p for p in self.people if p.ses_quintile == 5]
        adop_q1 = safe_div(sum(1 for p in q1 if p.adopting), len(q1), 0.0)
        adop_q5 = safe_div(sum(1 for p in q5 if p.adopting), len(q5), 0.0)
        inv_q1 = safe_div(sum(p.mask_inventory for p in q1), len(q1), 0.0)
        inv_q5 = safe_div(sum(p.mask_inventory for p in q5), len(q5), 0.0)
        mask_access_inequity = {
            "adoption_gap_q5_minus_q1": adop_q5 - adop_q1,
            "inventory_gap_q5_minus_q1": inv_q5 - inv_q1,
            "q1_adoption": adop_q1,
            "q5_adoption": adop_q5,
            "q1_avg_inventory": inv_q1,
            "q5_avg_inventory": inv_q5,
        }

        # Fines and enforcement
        fines = {"count": self.total_fines_count, "total_value": self.total_fines_value}

        # Disease metrics
        infections_averted = None
        Rt_series = None
        incidence_rate = None
        if self.include_disease_module:
            total_inf = sum(self.new_infections_over_time)
            total_cf = sum(self.counterfactual_infections_over_time)
            infections_averted = max(0, total_cf - total_inf)
            Rt_series = self.Rt_over_time[:]
            # FIXED: Added incidence rate (per 100k) timeseries for disease impact reporting.
            incidence_rate = [(x / max(1, self.population_size)) * 100000.0 for x in self.new_infections_over_time]

        # Retailer outage breakdown
        retailer_outages = {r.name: r.outage_days for r in self.retailers}

        # FIXED: Average compliance duration computed from adoption streaks (completed + trailing).
        trailing = [self.current_streak[i] for i, p in enumerate(self.people) if p.adopting and self.current_streak[i] > 0]
        all_streaks = self.completed_streaks + trailing
        avg_streak = safe_div(sum(all_streaks), len(all_streaks), 0.0)

        # FIXED: Policy compliance rate measured at mandate-sensitive locations during mandate.
        masked_sum = sum(m for m, t in self.daily_compliance if t > 0)
        total_visitors_sum = sum(t for _, t in self.daily_compliance if t > 0)
        policy_compliance_rate = safe_div(masked_sum, total_visitors_sum, 0.0)
        policy_compliance_timeseries = [safe_div(m, t, 0.0) if t > 0 else None for (m, t) in self.daily_compliance]

        # Assemble
        self.metrics = {
            "adoption_rate_over_time": adoption_series,
            "peak_adoption": {"value": peak_adopt, "day": peak_day},
            "time_to_50_percent_adoption": t50,
            # FIXED: Added time_to_70_percent_adoption per feedback.
            "time_to_70_percent_adoption": t70,
            "sustained_adoption_post_mandate": sustained_val,
            # FIXED: Added average_compliance_duration (in days).
            "average_compliance_duration": avg_streak,
            # FIXED: Added policy compliance rate and timeseries.
            "policy_compliance_rate": policy_compliance_rate,
            "policy_compliance_timeseries": policy_compliance_timeseries,
            "compliance_distribution_by_group": {
                "age_groups": by_age,
                "ses_quintiles": by_ses,
            },
            "mask_access_inequity": mask_access_inequity,
            "inventory_outage_days": self.inventory_outage_retailer_days,
            "retailer_outages_by_name": retailer_outages,
            "fines_issued": fines,
            # FIXED: Added enforcement actions counter (fines + refusals).
            "enforcement_actions_count": self.enforcement_actions_count,
            # FIXED: Added masks_purchased metrics.
            "masks_purchased": {
                "cumulative": self.masks_purchased_cumulative,
                "daily": self.masks_purchased_daily,
            },
            "infections_averted": infections_averted,
            "Rt": Rt_series,
            "incidence_rate": incidence_rate,
        }

    def results(self) -> Dict[str, Any]:
        """
        Return structured results.

        Returns:
            dict: Metrics and timeseries results.
        """
        pass
        res = {
            "metrics": self.metrics,
            "timeseries": {
                "adoption_rate": self.adoption_rate_over_time,
                "Rt": self.Rt_over_time if self.include_disease_module else None,
                "new_infections": self.new_infections_over_time if self.include_disease_module else None,
                "counterfactual_new_infections": self.counterfactual_infections_over_time if self.include_disease_module else None,
            },
        }
        return res

    def save_json(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save results to a JSON file, ensuring directories exist.

        Args:
            path: Output file path.
            metadata: Optional metadata to include.

        Returns:
            None
        """
        pass
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {"results": self.results(), "metadata": metadata or {}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def save_results(self, filename: str) -> None:
        """
        Save a simple CSV of daily adoption rate and, if present, Rt.

        Args:
            filename: CSV path.

        Returns:
            None
        """
        pass
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        headers = ["day", "adoption_rate"]
        include_rt = self.include_disease_module and len(self.Rt_over_time) == self.days
        if include_rt:
            headers.append("Rt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for day in range(self.days):
                row = [str(day), f"{self.adoption_rate_over_time[day]:.6f}"]
                if include_rt:
                    row.append(f"{self.Rt_over_time[day]:.6f}")
                f.write(",".join(row) + "\n")

    def visualize(self) -> None:
        """
        Visualize adoption trajectory and optionally Rt if matplotlib is available.

        Returns:
            None
        """
        pass
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:
            print(f"Visualization skipped (matplotlib not available): {e}", file=sys.stderr)
            return

        try:
            plt.figure(figsize=(10, 4))
            plt.plot(self.adoption_rate_over_time, label="Adoption rate")
            if self.include_disease_module and self.Rt_over_time:
                plt.plot(self.Rt_over_time, label="Rt (gen-interval proxy)", alpha=0.7)
            plt.axvline(self.gov.mandate_start_day, color="gray", linestyle="--", alpha=0.5, label="Mandate start")
            plt.axvline(self.gov.mandate_end_day, color="gray", linestyle="-.", alpha=0.5, label="Mandate end")
            plt.legend()
            plt.title("Adoption and Rt over time")
            plt.xlabel("Day")
            plt.ylabel("Value")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization error (continuing): {e}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """
    Build and parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    pass
    # FIXED: Implemented argparse CLI with smoke-test mode and output path.
    # FIXED: Added --stdout-json flag and parse_known_args to tolerate unknown args in embedded contexts.
    parser = argparse.ArgumentParser(description="Agent-based social adoption simulation")
    parser.add_argument("--population-size", type=int, default=500, help="Number of agents")
    parser.add_argument("--time-horizon-days", type=int, default=60, help="Simulation length in days")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include-disease-module", action="store_true", help="Enable disease dynamics")
    parser.add_argument("--output", type=str, default="./artifacts/metrics.json", help="JSON output path")
    parser.add_argument("--smoke-test", action="store_true", help="Run a very small scenario for CI")
    # FIXED: Make stdout emit only JSON if requested; logs go to stderr.
    parser.add_argument("--stdout-json", action="store_true", help="Emit only JSON to stdout and send logs to stderr")
    # FIXED: Expose key parameters for scenario control.
    parser.add_argument("--network-mean-degree", type=int, default=8, help="Average degree of the small-world network")
    parser.add_argument("--network-rewiring-prob", type=float, default=0.05, help="Rewiring probability of the small-world network")
    parser.add_argument("--enforcement-prob", type=float, default=0.15, help="Base enforcement probability")
    parser.add_argument("--fine-amount", type=float, default=50.0, help="Fine amount for violations")
    args = parser.parse_known_args()[0]
    # Environment override for embedded contexts
    if os.environ.get("STDOUT_JSON", "").lower() in ("1", "true", "yes"):
        setattr(args, "stdout_json", True)
    return args


def main() -> None:
    """
    Entry point to initialize, run, visualize, and save the simulation.

    Returns:
        None
    """
    pass
    # FIXED: Restored functional main that runs the simulation and writes outputs as JSON.
    # FIXED: Added smoke-test support for CI and made output directory robust.
    # FIXED: Print only JSON to stdout when --stdout-json; route all other logs to stderr.
    args = parse_args()
    pop = 50 if args.smoke_test else args.population_size
    days = 3 if args.smoke_test else args.time_horizon_days
    include_disease = False if args.smoke_test else args.include_disease_module

    policy_params = {
        "mandate_start_day": max(1, days // 3),
        "mandate_end_day": max(1, days // 2),
        "enforcement_prob": clamp(args.enforcement_prob, 0.0, 1.0),
        "fine_amount": max(0.0, float(args.fine_amount)),
        "subsidy_amount": 10.0,
        "subsidy_effect_weight": 0.01,
    }

    sim = SocialAdoptionSimulation(
        population_size=pop,
        time_horizon_days=days,
        random_seed=args.random_seed,
        include_disease_module=include_disease,
        policy_params=policy_params,
        network_mean_degree=args.network_mean_degree,
        network_rewiring_prob=args.network_rewiring_prob,
    )
    sim.run()

    # Prepare metadata
    metadata = {
        "population": pop,
        "days": days,
        "seed": args.random_seed,
        "include_disease_module": include_disease,
        "notes": "ABM with peer influence, media, policy, market access, enforcement (fines + entry refusal), and optional SEIR-lite. Outage counting fixed; purchases before adoption; improved Rt; visitation caching; norms influence integrated; stdout JSON option; added compliance metrics, mask purchases, and enforcement actions.",
    }

    # Output handling
    payload = {"results": sim.results(), "metadata": metadata}
    if getattr(args, "stdout_json", False):
        # FIXED: Emit only JSON to stdout to satisfy harness parser expectations.
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        sim.save_json(args.output, metadata=metadata)
        print(f"Wrote results to {args.output}", file=sys.stderr)
        # Demonstrate CSV save per requirement
        csv_path = os.path.join(os.path.dirname(args.output), "results.csv")
        sim.save_results(csv_path)
        print(f"Wrote daily results CSV to {csv_path}", file=sys.stderr)
        # Optional visualization (safe to skip in headless CI)
        try:
            sim.visualize()
        except Exception as e:
            print(f"Visualization error (continuing): {e}", file=sys.stderr)


# Execute main for both direct execution and sandbox wrapper invocation
# NOTE: Intentionally retaining 'pass' statements as per sandbox requirement to guarantee syntactic correctness.
main()