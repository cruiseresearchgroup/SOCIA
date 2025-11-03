# FIXED: Applied feedback snippet from simulation_code_iter_89.py
def main():
    parser = argparse.ArgumentParser(description="Mask adoption simulation")
    parser.add_argument("--params", type=str, default=None, help="JSON string or path to JSON file with parameters")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    params: Dict[str, Any] = {}
    if args.params:
        try:
            # Try to load as JSON string
            params = json.loads(args.params)
        except json.JSONDecodeError:
            # Fallback: treat as file path
            with open(args.params, "r", encoding="utf-8") as f:
                params = json.load(f)

    sim = Simulation(params)
    results = sim.run()

    out_str = json.dumps(results)
    print(out_str)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_str)
        except Exception:
            pass

import os
import json
import math
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else None


def clamp01(x: float) -> float:
    """
    Clamp a numeric value into the [0.0, 1.0] interval.

    Args:
        x: Input value.

    Returns:
        The clamped value within [0.0, 1.0].
    """
    # FIXED: Removed redundant pass statements.
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid/logistic function.

    Args:
        x: Input value.

    Returns:
        Sigmoid of x.
    """
    # FIXED: Removed redundant pass statements.
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """
    Safe division utility to avoid ZeroDivisionError.

    Args:
        n: Numerator.
        d: Denominator.
        default: Default value to return when d is zero.

    Returns:
        n / d if d != 0, otherwise default.
    """
    # FIXED: Removed redundant pass statements.
    return default if d == 0 else n / d


def mean(values: List[float]) -> float:
    """
    Compute the arithmetic mean of a list of numbers.

    Args:
        values: List of numeric values.

    Returns:
        Mean value or 0.0 for empty list.
    """
    # FIXED: Removed redundant pass statements.
    return safe_div(sum(values), len(values), 0.0)


def generate_small_world(n: int, k: int, beta: float, rng: random.Random) -> List[List[int]]:
    """
    Generate a Watts–Strogatz-like small-world network adjacency list.

    Args:
        n: Number of nodes.
        k: Each node is joined with its k nearest neighbors in a ring topology.
        beta: The rewiring probability.
        rng: Random generator.

    Returns:
        Adjacency list representing the network.
    """
    # FIXED: Removed redundant pass statements.
    if n <= 1:
        return [[] for _ in range(n)]
    # Ensure k is even and <= n-1
    k = max(0, min(k, n - 1))
    if k % 2 == 1:
        k -= 1
    half = k // 2
    neighbors = [set() for _ in range(n)]
    # Ring lattice
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            neighbors[i].add(j)
            neighbors[j].add(i)
    # Rewire edges
    for i in range(n):
        for j in list(neighbors[i]):
            if j > i and rng.random() < beta:
                # Remove existing edge (i, j)
                neighbors[i].discard(j)
                neighbors[j].discard(i)
                # FIXED: Optimized small-world rewiring candidate selection to avoid O(n) scans.
                # Bounded random attempts with fallback to restore original edge if none found.
                new_j = None
                for _ in range(20):
                    cand = rng.randrange(n)
                    if cand != i and cand not in neighbors[i]:
                        new_j = cand
                        break
                if new_j is not None:
                    neighbors[i].add(new_j)
                    neighbors[new_j].add(i)
                else:
                    # Restore original edge if rewiring failed to preserve degree distribution
                    neighbors[i].add(j)
                    neighbors[j].add(i)
    return [sorted(s) for s in neighbors]


class Person:
    """
    A simple agent representing an individual in the simulation.

    Attributes:
        id: Unique identifier.
        socioeconomic_status: SES scaled [0,1].
        risk_perception: Perceived risk scaled [0,1].
        trust_in_authorities: Trust scaled [0,1].
        compliance_propensity: Personal propensity to comply [0,1].
        susceptibility_to_influence: Social influence sensitivity [0,1].
        fatigue_level: Mask-wearing fatigue [0,1].
        mask_adoption_state: 1 if currently wearing, else 0.
        mask_inventory: Count of masks available.
        belief_bias: Bias in processing information, [-1,1] expected; applied boundedly.
        network_neighbors: List of neighbor indices in the social network.
        cost_paid: Accumulated cost burden due to purchases/fines.
    """

    def __init__(
        self,
        pid: int,
        ses: float,
        risk: float,
        trust: float,
        comp: float,
        sus: float,
        rng: Optional[random.Random] = None,
    ):
        """
        Initialize a Person.

        Args:
            pid: Person ID.
            ses: Socioeconomic status [0,1].
            risk: Initial risk perception [0,1].
            trust: Initial trust in authorities [0,1].
            comp: Compliance propensity [0,1].
            sus: Susceptibility to social influence [0,1].
            rng: Optional random generator for initial state variability.
        """
        # FIXED: Removed redundant pass statements.
        self.id = pid
        self.socioeconomic_status = clamp01(ses)
        self.risk_perception = clamp01(risk)
        self.trust_in_authorities = clamp01(trust)
        self.compliance_propensity = clamp01(comp)
        self.susceptibility_to_influence = clamp01(sus)
        self.fatigue_level = clamp01((rng.random() * 0.1) if rng else 0.0)
        self.mask_adoption_state = 1 if (rng.random() < 0.05 if rng else random.random() < 0.05) else 0
        self.mask_inventory = 1
        self.belief_bias = 0.0
        self.network_neighbors: List[int] = []
        self.cost_paid: float = 0.0

    def update_from_messages(self, authority_delta: Tuple[float, float], media_delta: Tuple[float, float]):
        """
        Update internal states based on broadcast messages.

        Args:
            authority_delta: Tuple (delta_risk, delta_trust) from public health authority.
            media_delta: Tuple (delta_risk, delta_attitude) from media outlet.
        """
        # FIXED: Removed redundant pass statements.
        # Authority messaging directly affects risk perception and trust
        d_risk_a, d_trust_a = authority_delta
        self.risk_perception = clamp01(self.risk_perception + d_risk_a * (0.5 + 0.5 * self.trust_in_authorities))
        self.trust_in_authorities = clamp01(self.trust_in_authorities + d_trust_a)

        # Media messaging affects risk perception and compliance propensity through bias
        d_risk_m, d_att_m = media_delta
        # Belief bias attenuates or amplifies media effects; clamp to keep robust
        bias_factor = clamp01(0.5 + 0.5 * (1.0 - abs(self.belief_bias)))
        self.risk_perception = clamp01(self.risk_perception + d_risk_m * bias_factor)
        self.compliance_propensity = clamp01(self.compliance_propensity + d_att_m * bias_factor)

    def decide_adoption(
        self,
        peer_rate: float,
        policy_strength: float,
        enforcement_effect: float,
        price: float,
        weights: Dict[str, float],
        rng: random.Random,
    ) -> int:
        """
        Decide whether to wear a mask today.

        Args:
            peer_rate: Average mask wearing among social neighbors [0,1].
            policy_strength: Policy mandate strength [0,1].
            enforcement_effect: Effective enforcement pressure [0,1].
            price: Current mask price (unitless scale).
            weights: Dictionary of weighted terms.
            rng: Random generator.

        Returns:
            1 if wearing mask, else 0.
        """
        # FIXED: Removed redundant pass statements.
        social = weights.get("social", 0.6) * self.susceptibility_to_influence * clamp01(peer_rate)
        policy_term = weights.get("policy", 0.35) * (self.trust_in_authorities * clamp01(policy_strength) + clamp01(enforcement_effect))
        risk_term = weights.get("risk", 0.5) * clamp01(self.risk_perception)
        personal = weights.get("personal", 0.3) * clamp01(self.compliance_propensity)
        fatigue = weights.get("fatigue", 0.01) * clamp01(self.fatigue_level)
        # Normalize price impact to [0,1] scale around 0.5 baseline
        econ = weights.get("price", 0.2) * clamp01((price - 0.5) / 2.0)
        util = social + policy_term + risk_term + personal - fatigue - econ
        p = sigmoid(2.5 * (util - 0.5))
        return 1 if rng.random() < p else 0

    def purchase_if_needed(self, retailer: "Retailer", desired_qty: int = 1) -> int:
        """
        Attempt to purchase masks if inventory is insufficient.

        Args:
            retailer: Retailer to purchase from.
            desired_qty: Desired quantity to purchase.

        Returns:
            Quantity actually purchased.
        """
        # FIXED: Removed redundant pass statements.
        if self.mask_inventory >= desired_qty:
            return 0
        to_buy = desired_qty - self.mask_inventory
        bought, cost = retailer.sell(to_buy)
        self.mask_inventory += bought
        self.cost_paid += cost
        return bought

    def apply_enforcement(self, fine_amount: float) -> None:
        """
        Apply an enforcement fine to the individual.

        Args:
            fine_amount: Monetary penalty to add to cost burden.
        """
        # FIXED: Removed redundant pass statements.
        if fine_amount > 0:
            self.cost_paid += fine_amount

    def update_fatigue(self, wore_mask: bool, rate: float) -> None:
        """
        Update fatigue based on today's behavior, scaled by fatigue rate.

        Args:
            wore_mask: Whether the person wore a mask today.
            rate: Fatigue rate parameter controlling step magnitudes.
        """
        # FIXED: Align fatigue dynamics with configured fatigue_rate.
        step_up = 0.5 * rate
        step_down = -0.2 * rate
        delta = step_up if wore_mask else step_down
        self.fatigue_level = clamp01(self.fatigue_level + delta)


class Location:
    """
    A location that agents may visit, potentially under a mask mandate.

    Attributes:
        id: Unique identifier.
        mandate_active: Whether a mask mandate is active at this location.
        enforcement_level: Strength of enforcement [0,1].
        enforcement_probability: Probability that noncompliance triggers an incident [0,1].
        daily_masked_visits: Count of masked visits today.
        daily_total_visits: Total visits today.
        daily_noncompliance_incidents: Incidents recorded today.
    """

    def __init__(self, lid: int, mandate_active: bool, enforcement_level: float, enforcement_probability: float):
        """
        Initialize a Location instance.

        Args:
            lid: Location ID.
            mandate_active: Mask mandate status.
            enforcement_level: Enforcement strength [0,1].
            enforcement_probability: Probability of incident [0,1].
        """
        # FIXED: Removed redundant pass statements.
        self.id = lid
        self.mandate_active = mandate_active
        self.enforcement_level = clamp01(enforcement_level)
        self.enforcement_probability = clamp01(enforcement_probability)
        self.daily_masked_visits = 0
        self.daily_total_visits = 0
        self.daily_noncompliance_incidents = 0

    def reset_daily_counters(self) -> None:
        """
        Reset daily counters for a new simulation day.
        """
        # FIXED: Removed redundant pass statements.
        self.daily_masked_visits = 0
        self.daily_total_visits = 0
               # noqa: E127
        self.daily_noncompliance_incidents = 0

    def record_visit(self, masked: bool) -> None:
        """
        Record a visit to this location.

        Args:
            masked: Whether the visitor was masked.
        """
        # FIXED: Removed redundant pass statements.
        self.daily_total_visits += 1
        if masked:
            self.daily_masked_visits += 1

    def enforce(self, masked: bool, rng: random.Random) -> bool:
        """
        Apply enforcement logic for unmasked visits when a mandate is active.

        Args:
            masked: Whether the visitor was masked.
            rng: Random generator.

        Returns:
            True if a noncompliance incident occurred, else False.
        """
        # FIXED: Removed redundant pass statements.
        if self.mandate_active and not masked:
            if rng.random() < self.enforcement_probability * self.enforcement_level:
                self.daily_noncompliance_incidents += 1
                return True
        return False


class Retailer:
    """
    Retailer selling masks with simple stock and pricing dynamics.

    Attributes:
        mask_stock: Current stock of masks (nonnegative).
        price: Current unit price (>= 0).
        rationing_policy: Max units sold per customer per day.
        scarcity_sensitivity: Sensitivity of price to scarcity.
    """

    def __init__(self, stock: int, price: float, ration: int = 2, scarcity_sensitivity: float = 0.2):
        """
        Initialize a Retailer.

        Args:
            stock: Initial stock.
            price: Initial price.
            ration: Max units per customer per purchase.
            scarcity_sensitivity: Price adjustment factor based on scarcity.
        """
        # FIXED: Removed redundant pass statements.
        self.mask_stock = max(0, int(stock))
        self.price = max(0.0, float(price))
        self.rationing_policy = max(1, int(ration))
        self.scarcity_sensitivity = max(0.0, float(scarcity_sensitivity))

    def sell(self, q: int) -> Tuple[int, float]:
        """
        Sell up to q masks subject to rationing and stock constraints.

        Args:
            q: Requested quantity.

        Returns:
            Tuple of (sold_quantity, total_cost).
        """
        # FIXED: Removed redundant pass statements.
        q = max(0, min(self.rationing_policy, int(q)))
        sold = min(self.mask_stock, q)
        self.mask_stock -= sold
        total_cost = sold * self.price
        return sold, total_cost

    def restock(self, add: int) -> None:
        """
        Add stock units.

        Args:
            add: Units to add to stock.
        """
        # FIXED: Removed redundant pass statements.
        self.mask_stock += max(0, int(add))

    def adjust_price(self, baseline_stock: int) -> None:
        """
        Adjust price based on scarcity relative to a baseline stock.

        Args:
            baseline_stock: Baseline stock for computing scarcity.
        """
        # FIXED: Removed redundant pass statements.
        baseline_stock = max(1, int(baseline_stock))
        scarcity = 1.0 - clamp01(self.mask_stock / baseline_stock)
        # Limit price adjustments to a reasonable band to avoid runaway values
        self.price = max(0.1, min(10.0, self.price * (1.0 + self.scarcity_sensitivity * scarcity)))


class PublicHealthAuthority:
    """
    A simple broadcaster representing public health authority communications.

    Attributes:
        policy_mandate_strength: Global policy strength [0,1].
        trust_update_rate: Increment to public trust per day due to consistent messaging.
        risk_message_strength: Increment to risk perception per day.
    """

    def __init__(self, policy_mandate_strength: float, trust_update_rate: float, risk_message_strength: float):
        """
        Initialize the PublicHealthAuthority.

        Args:
            policy_mandate_strength: Mask mandate strength [0,1].
            trust_update_rate: Daily trust update magnitude.
            risk_message_strength: Daily risk perception update magnitude.
        """
        # FIXED: Removed redundant pass statements.
        self.policy_mandate_strength = clamp01(policy_mandate_strength)
        self.trust_update_rate = float(trust_update_rate)
        self.risk_message_strength = float(risk_message_strength)

    def broadcast(self) -> Tuple[float, float]:
        """
        Produce the daily message deltas.

        Returns:
            Tuple of (delta_risk, delta_trust).
        """
        # FIXED: Removed redundant pass statements.
        return self.risk_message_strength, self.trust_update_rate


class MediaOutlet:
    """
    A media outlet broadcasting messages that may increase or decrease risk perception and attitudes.

    Attributes:
        misinformation_intensity: Magnitude of misleading info; positive raises risk, negative lowers risk.
        attitude_influence: Influence on compliance attitude.
        variability: Day-to-day variability of the message.
        rng: Random generator.
    """

    def __init__(self, misinformation_intensity: float, attitude_influence: float, variability: float, rng: random.Random):
        """
        Initialize a MediaOutlet.

        Args:
            misinformation_intensity: Base effect on risk perception (can be negative).
            attitude_influence: Base effect on compliance attitude (can be negative).
            variability: Random noise scale per day.
            rng: Random generator.
        """
        # FIXED: Removed redundant pass statements.
        self.misinformation_intensity = float(misinformation_intensity)
        self.attitude_influence = float(attitude_influence)
        self.variability = max(0.0, float(variability))
        self.rng = rng

    def broadcast(self) -> Tuple[float, float]:
        """
        Produce the daily media message deltas.

        Returns:
            Tuple of (delta_risk, delta_attitude).
        """
        # FIXED: Removed redundant pass statements.
        noise_r = (self.rng.random() - 0.5) * 2.0 * self.variability
        noise_a = (self.rng.random() - 0.5) * 2.0 * self.variability
        return self.misinformation_intensity + noise_r, self.attitude_influence + noise_a


class Simulation:
    """
    Main simulation class coordinating agents, environment, and daily dynamics.

    Responsibilities:
        - Initialize agents, network, retailers, locations, and broadcasters.
        - Run daily steps including messaging, decisions, visits, enforcement, purchasing, and stock updates.
        - Track metrics and validations.
        - Provide result export and visualization capabilities.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Construct a Simulation with the provided parameters.

        Args:
            params: Configuration dictionary. Key parameters include:
                - population_size
                - time_horizon_days
                - average_degree
                - rewiring_prob
                - policy_mandate_strength
                - enforcement_probability
                - mask_price
                - retailer_restock_rate
                - retailer_baseline_stock_factor
                - social_influence_strength
                - risk_perception_sensitivity
                - fatigue_rate
                - random_seed
                - num_locations
                - mandate_location_fraction
                - enforcement_level
                - fine_amount
                - media_misinformation_intensity
                - media_attitude_influence
                - media_variability
                - rationing_policy
                - scarcity_sensitivity
        """
        # FIXED: Removed redundant pass statements.
        self.params = params.copy()
        self.rng = random.Random(int(params.get("random_seed", 42)))
        self.population_size = int(params.get("population_size", 300))
        self.days = int(params.get("time_horizon_days", 10))
        self.avg_degree = int(params.get("average_degree", 8))
        self.rewiring_prob = float(params.get("rewiring_prob", 0.05))

        # Policy and enforcement
        self.policy_mandate_strength = float(params.get("policy_mandate_strength", 0.7))
        self.enforcement_probability = float(params.get("enforcement_probability", 0.2))
        self.enforcement_level = float(params.get("enforcement_level", 0.6))
        self.fine_amount = float(params.get("fine_amount", 0.0))

        # Retail and price dynamics
        self.mask_price = float(params.get("mask_price", 1.0))
        self.restock_rate = float(params.get("retailer_restock_rate", 0.1))
        self.rationing_policy = int(params.get("rationing_policy", 2))
        self.scarcity_sensitivity = float(params.get("scarcity_sensitivity", 0.2))
        self.retailer_baseline_stock_factor = float(params.get("retailer_baseline_stock_factor", 1.0))

        # Messaging parameters
        self.risk_perception_sensitivity = float(params.get("risk_perception_sensitivity", 0.5))
        self.social_influence_strength = float(params.get("social_influence_strength", 0.6))
        self.fatigue_rate = float(params.get("fatigue_rate", 0.01))
        self.trust_update_rate = float(params.get("trust_update_rate", 0.01))
        self.risk_message_strength = float(params.get("risk_message_strength", 0.01))
        self.media_misinformation_intensity = float(params.get("media_misinformation_intensity", 0.0))
        self.media_attitude_influence = float(params.get("media_attitude_influence", 0.0))
        self.media_variability = float(params.get("media_variability", 0.02))

        # Locations
        self.num_locations = int(params.get("num_locations", max(1, self.population_size // 50)))
        self.mandate_location_fraction = float(params.get("mandate_location_fraction", 0.5))

        # Weights for decision
        self.weights = {
            "social": self.social_influence_strength,
            "policy": 0.35,
            "risk": self.risk_perception_sensitivity,
            "personal": 0.3,
            "fatigue": self.fatigue_rate,
            "price": 0.2,
        }

        # Entities
        self.people: List[Person] = []
        self.locations: List[Location] = []
        self.retailer: Optional[Retailer] = None
        self.network: List[List[int]] = []

        # Metrics
        self.adoption_rate_over_time: List[float] = []
        self.noncompliance_incidents_per_day: List[int] = []
        self.compliance_rate_in_mandated_areas: List[Optional[float]] = []
        self.cumulative_masks_purchased: int = 0
        self.cost_burden_per_capita: float = 0.0
        self.daily_cost_burden: List[float] = []

        # Broadcasters
        self.authority = PublicHealthAuthority(self.policy_mandate_strength, self.trust_update_rate, self.risk_message_strength)
        self.media = MediaOutlet(self.media_misinformation_intensity, self.media_attitude_influence, self.media_variability, self.rng)

        # Initialization
        self._initialize_entities()

    def _initialize_entities(self) -> None:
        """
        Initialize people, network, retailer, and locations.
        """
        # FIXED: Removed redundant pass statements.
        # People
        self.people = [
            Person(
                pid=i,
                ses=self.rng.random(),
                risk=self.rng.random(),
                trust=self.rng.random(),
                comp=self.rng.random(),
                sus=self.rng.random(),
                rng=self.rng,
            )
            for i in range(self.population_size)
        ]
        # Network
        self.network = generate_small_world(self.population_size, self.avg_degree, self.rewiring_prob, self.rng)
        for i, nbrs in enumerate(self.network):
            self.people[i].network_neighbors = nbrs

        # Retailer
        baseline_stock = int(max(1, self.population_size * self.retailer_baseline_stock_factor))
        self.retailer = Retailer(stock=baseline_stock, price=self.mask_price, ration=self.rationing_policy, scarcity_sensitivity=self.scarcity_sensitivity)

        # Locations
        self.num_locations = max(1, self.num_locations)
        num_mandated = int(round(self.num_locations * self.mandate_location_fraction))
        num_mandated = min(num_mandated, self.num_locations)
        mandated_ids = set(self.rng.sample(range(self.num_locations), k=num_mandated)) if num_mandated > 0 else set()
        self.locations = [
            Location(
                lid=j,
                mandate_active=(j in mandated_ids and self.policy_mandate_strength > 0.0),
                enforcement_level=self.enforcement_level,
                enforcement_probability=self.enforcement_probability,
            )
            for j in range(self.num_locations)
        ]

    def _choose_location(self) -> Location:
        """
        Choose a random location for a visit.

        Returns:
            A randomly selected Location.
        """
        # FIXED: Removed redundant pass statements.
        return self.rng.choice(self.locations)

    def _social_peer_rate(self, idx: int) -> float:
        """
        Compute the social peer mask-wearing rate for the given person.

        Args:
            idx: Person index.

        Returns:
            Average mask wearing among neighbors.
        """
        # FIXED: Removed redundant pass statements.
        nbrs = self.people[idx].network_neighbors
        if not nbrs:
            return 0.0
        return safe_div(sum(self.people[j].mask_adoption_state for j in nbrs), len(nbrs), 0.0)

    def step(self, day: int) -> None:
        """
        Execute one day of simulation.

        Args:
            day: Current day index (0-based).
        """
        # FIXED: Removed redundant pass statements.
        assert self.retailer is not None, "Retailer must be initialized."

        # Reset daily counters at locations
        for loc in self.locations:
            loc.reset_daily_counters()

        # Broadcast messages
        auth_delta = self.authority.broadcast()
        media_delta = self.media.broadcast()

        # Apply messaging to people
        for p in self.people:
            p.update_from_messages(auth_delta, media_delta)

        # Decisions, visiting, enforcement, purchases
        wearing_count = 0
        incidents_today = 0
        mandated_masked_visits = 0
        mandated_total_visits = 0
        daily_purchases = 0

        # Compute peer rates first to avoid intra-day ordering effects
        peer_rates = [self._social_peer_rate(i) for i in range(self.population_size)]

        for i, p in enumerate(self.people):
            # Evaluate decision
            location = self._choose_location()
            decision = p.decide_adoption(
                peer_rate=peer_rates[i],
                policy_strength=self.policy_mandate_strength if location.mandate_active else 0.0,
                enforcement_effect=location.enforcement_level * location.enforcement_probability,
                price=self.retailer.price,
                weights=self.weights,
                rng=self.rng,
            )
            # If decided to wear but lacking inventory, attempt purchase first
            if decision == 1 and p.mask_inventory <= 0:
                bought = p.purchase_if_needed(self.retailer, desired_qty=1)
                daily_purchases += bought

            wore_mask = False
            if decision == 1 and p.mask_inventory > 0:
                p.mask_inventory -= 1
                wore_mask = True

            # Record visit
            location.record_visit(masked=wore_mask)
            if location.mandate_active:
                mandated_total_visits += 1
                if wore_mask:
                    mandated_masked_visits += 1

            # Enforcement if not wearing under mandate
            if location.mandate_active and not wore_mask:
                incident = location.enforce(masked=wore_mask, rng=self.rng)
                if incident:
                    incidents_today += 1
                    p.apply_enforcement(self.fine_amount)

            # Update today's adoption state for social signaling next day
            p.mask_adoption_state = 1 if wore_mask else 0
            wearing_count += p.mask_adoption_state

            # Update fatigue
            p.update_fatigue(wore_mask=wore_mask, rate=self.fatigue_rate)  # FIXED: Pass fatigue_rate to align dynamics.

        # Restock retailer and adjust price based on scarcity
        restock_units = int(max(0, round(self.population_size * self.restock_rate)))
        self.retailer.restock(restock_units)
        self.retailer.adjust_price(baseline_stock=int(max(1, self.population_size * self.retailer_baseline_stock_factor)))

        # Metrics for the day
        self.adoption_rate_over_time.append(safe_div(wearing_count, self.population_size, 0.0))
        self.noncompliance_incidents_per_day.append(incidents_today)
        self.cumulative_masks_purchased += daily_purchases

        # FIXED: Compliance collection uses None for days with no mandated visits to avoid downward bias.
        compliance_today = safe_div(mandated_masked_visits, mandated_total_visits, 0.0) if mandated_total_visits > 0 else None
        self.compliance_rate_in_mandated_areas.append(compliance_today)

        # Cost burden
        total_cost_paid = sum(p.cost_paid for p in self.people)
        per_capita = safe_div(total_cost_paid, self.population_size, 0.0)
        self.daily_cost_burden.append(per_capita)

    def run(self) -> Dict[str, Any]:
        """
        Run the simulation over the configured time horizon.

        Returns:
            A dictionary containing metrics and validation results.
        """
        # FIXED: Removed redundant pass statements.
        for day in range(self.days):
            self.step(day)
        return self.evaluate()

    def evaluate(self) -> Dict[str, Any]:
        """
        Compute final metrics and validations.

        Returns:
            A results dictionary ready for JSON serialization.
        """
        # FIXED: Removed redundant pass statements.
        final_adoption_rate = self.adoption_rate_over_time[-1] if self.adoption_rate_over_time else 0.0
        time_to_50 = next((i for i, v in enumerate(self.adoption_rate_over_time) if v >= 0.5), None)
        peak_adoption = max(self.adoption_rate_over_time) if self.adoption_rate_over_time else 0.0

        # Inequality (SES quartiles) on last day using current mask_adoption_state
        inequality = None
        if self.people:
            ordered = sorted(self.people, key=lambda x: x.socioeconomic_status)
            q = max(1, len(ordered) // 4)
            low = ordered[:q]
            high = ordered[-q:]
            low_rate = safe_div(sum(p.mask_adoption_state for p in low), len(low), 0.0)
            high_rate = safe_div(sum(p.mask_adoption_state for p in high), len(high), 0.0)
            inequality = {"by_ses_diff": high_rate - low_rate}

        # Validations
        adoption_bounds = all(0.0 <= a <= 1.0 for a in self.adoption_rate_over_time)
        stock_nonnegative = self.retailer.mask_stock >= 0 if self.retailer else True
        convergence_check = False
        if len(self.adoption_rate_over_time) >= 7:
            convergence_check = abs(self.adoption_rate_over_time[-1] - self.adoption_rate_over_time[-7]) < 0.005

        # Compliance rate average across days (only counts when mandate is active)
        mandate_on = self.policy_mandate_strength > 0.0 and any(loc.mandate_active for loc in self.locations)
        if mandate_on:
            valid_days = [v for v in self.compliance_rate_in_mandated_areas if v is not None]
            compliance_rate_avg = mean(valid_days) if valid_days else None
        else:
            compliance_rate_avg = None

        # Cost burden per capita final
        total_cost_paid = sum(p.cost_paid for p in self.people)
        cost_burden_per_capita = safe_div(total_cost_paid, self.population_size, 0.0)

        results = {
            "adoption_rate_over_time": self.adoption_rate_over_time,
            "final_adoption_rate": final_adoption_rate,
            "time_to_50_percent": time_to_50,
            "peak_adoption": peak_adoption,
            "compliance_rate_in_mandated_areas": compliance_rate_avg,
            "inequality_in_adoption": inequality,
            "noncompliance_incidents_per_day": self.noncompliance_incidents_per_day,
            "cumulative_masks_purchased": self.cumulative_masks_purchased,
            "cost_burden_per_capita": cost_burden_per_capita,
            "validations": {
                "adoption_bounds": adoption_bounds,
                "stock_nonnegative": stock_nonnegative,
                "convergence_check": convergence_check,
            },
        }

        # Optional: support dynamic evaluation_metrics if provided in params
        dyn_metrics = {}
        eval_list = self.params.get("evaluation_metrics")
        if isinstance(eval_list, list):
            for m in eval_list:
                name = str(m)
                if name == "adoption_rate_over_time":
                    dyn_metrics[name] = self.adoption_rate_over_time
                elif name == "time_to_50_percent_adoption":
                    dyn_metrics[name] = time_to_50
                elif name == "peak_adoption":
                    dyn_metrics[name] = peak_adoption
                elif name == "compliance_by_location_type":
                    # Placeholder: only global compliance implemented
                    dyn_metrics[name] = {"global_mandated": compliance_rate_avg}
                elif name == "inequality_in_adoption":
                    dyn_metrics[name] = inequality
                elif name == "stockouts_count":
                    # Proxy: count days retailer stock is zero at end-of-day (approx)
                    dyn_metrics[name] = int(self.retailer.mask_stock == 0) if self.retailer else 0
                else:
                    # Unrecognized metric placeholder
                    dyn_metrics[name] = None
        if dyn_metrics:
            results["dynamic_metrics"] = dyn_metrics

        return results

    def visualize(self, show: bool = False, save_path: Optional[str] = None) -> None:
        """
        Visualize simulation results using matplotlib if available.

        Args:
            show: If True, display figures interactively (may fail in headless environments).
            save_path: Optional path to save a PNG figure.
        """
        # FIXED: Completed matplotlib plotting block (previously left with an open '[').
        try:
            import matplotlib.pyplot as plt  # type: ignore

            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            t = list(range(len(self.adoption_rate_over_time)))

            # Top plot: adoption rate
            ax[0].plot(t, self.adoption_rate_over_time, label="Adoption rate", color="tab:blue")
            ax[0].set_ylabel("Adoption rate")
            ax[0].set_ylim(0.0, 1.0)
            ax[0].legend(loc="best")

            # Bottom plot: noncompliance incidents
            ax[1].bar(t, self.noncompliance_incidents_per_day, label="Noncompliance incidents", color="tab:orange")
            ax[1].set_ylabel("Incidents")
            ax[1].set_xlabel("Day")
            ax[1].legend(loc="best")

            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150)
            if show:
                plt.show()
            plt.close(fig)
        except Exception:
            # Silently ignore visualization errors in headless or minimal environments
            pass


# Execute main for both direct execution and sandbox wrapper invocation
main()