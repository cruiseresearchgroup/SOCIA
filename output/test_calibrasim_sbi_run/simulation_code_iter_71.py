def main():
    pass

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Project-relative data path setup per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def sigmoid(x: float) -> float:
    """
    Compute the logistic sigmoid function.

    Args:
        x: Input value.

    Returns:
        The logistic sigmoid of x in [0, 1].
    """
    pass
    try:
        # Numerical stability clamp
        x = max(-50.0, min(50.0, x))
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def logistic_fit_r2(series: List[float]) -> Optional[float]:
    """
    Fit a simple two-parameter logistic curve by logit-linearization and compute R^2.

    The method:
    - Clips series to (eps, 1-eps).
    - Transforms y via logit: z = log(y / (1 - y)).
    - Fits z = a + b * t using closed-form OLS.
    - Reconstructs y_hat and computes R^2.

    Args:
        series: A list of adoption rates in [0, 1].

    Returns:
        R^2 of the logistic fit if fitting is stable; otherwise None.
    """
    pass
    try:
        if len(series) < 5:
            return None
        eps = 1e-6
        y = [min(1.0 - eps, max(eps, v)) for v in series]
        z = [math.log(v / (1.0 - v)) for v in y]
        n = len(z)
        t_vals = list(range(n))
        t_mean = sum(t_vals) / n
        z_mean = sum(z) / n
        s_tt = sum((t - t_mean) ** 2 for t in t_vals)
        if s_tt == 0:
            return None
        s_tz = sum((t - t_mean) * (zz - z_mean) for t, zz in zip(t_vals, z))
        b = s_tz / s_tt
        a = z_mean - b * t_mean
        z_hat = [a + b * t for t in t_vals]
        y_hat = [1.0 / (1.0 + math.exp(-zz)) for zz in z_hat]
        ss_tot = sum((yy - (sum(y) / n)) ** 2 for yy in y)
        ss_res = sum((yy - yh) ** 2 for yy, yh in zip(y, y_hat))
        if ss_tot <= 0:
            return None
        r2 = 1.0 - ss_res / ss_tot
        return max(0.0, min(1.0, r2))
    except Exception:
        return None


def build_small_world_neighbors(n: int, k: int, beta: float, rng: random.Random) -> List[List[int]]:
    """
    Build a Watts-Strogatz-like small-world network without external dependencies.

    Args:
        n: Number of nodes.
        k: Each node is connected to k nearest neighbors in ring topology (k must be even).
        beta: Rewiring probability.
        rng: Random generator.

    Returns:
        A list of neighbor lists per node.
    """
    pass
    if k <= 0:
        return [[] for _ in range(n)]
    if k % 2 != 0:
        k -= 1  # FIXED: enforce even k for ring lattice
    neighbors: List[set] = [set() for _ in range(n)]
    half = k // 2

    # Create ring lattice
    for i in range(n):
        for j in range(1, half + 1):
            a = i
            b = (i + j) % n
            neighbors[a].add(b)
            neighbors[b].add(a)

    # Rewire edges with probability beta
    for i in range(n):
        for j in list(neighbors[i]):
            if j <= i:
                continue  # handle each undirected edge once
            if rng.random() < beta:
                # Remove old edge
                neighbors[i].remove(j)
                neighbors[j].remove(i)
                # Find a new node to connect to
                potential = set(range(n)) - {i} - neighbors[i]
                if not potential:
                    # Restore old edge if no potential found
                    neighbors[i].add(j)
                    neighbors[j].add(i)
                    continue
                new_j = rng.choice(list(potential))
                neighbors[i].add(new_j)
                neighbors[new_j].add(i)
    return [sorted(list(s)) for s in neighbors]


@dataclass
class Person:
    """
    Person agent with attributes and behaviors relevant to mask adoption dynamics.

    Attributes:
        id: Unique identifier.
        age: Age in years.
        household_id: Household grouping identifier.
        current_location_id: The current location visited (if any).
        social_neighbors_ids: Neighbor indices in the social network.
        adoption_state: Whether the person is currently wearing a mask.
        adoption_threshold: Baseline threshold for adoption (not directly used in this simplified model).
        susceptibility_to_peer_influence: Weight on peer average influence [0, 1].
        trust_in_authorities: Trust level in health authorities [0, 1].
        misinformation_exposure: Susceptibility to misinformation [0, 1].
        risk_perception: Perceived health risk [0, 1].
        perceived_cost_of_masking: Composite cost (comfort, money, inconvenience) [0, 1].
        fatigue_level: Fatigue from prolonged masking [0, 1].
        habit_strength: Habit formation strength [0, 1].
        mask_inventory: Number of available masks (integer).
    """
    pass
    id: int = 0
    age: int = 30
    household_id: int = 0
    current_location_id: Optional[int] = None
    social_neighbors_ids: List[int] = field(default_factory=list)
    adoption_state: bool = False
    adoption_threshold: float = 0.5
    susceptibility_to_peer_influence: float = 0.5
    trust_in_authorities: float = 0.5
    misinformation_exposure: float = 0.2
    risk_perception: float = 0.2
    perceived_cost_of_masking: float = 0.3
    fatigue_level: float = 0.0
    habit_strength: float = 0.0
    mask_inventory: int = 0

    def observe_peers(self, neighbor_adoption_fraction: float) -> float:
        """
        Observe peers and compute peer influence on attitude.

        Args:
            neighbor_adoption_fraction: Fraction of neighbors currently adopting [0, 1].

        Returns:
            A peer influence delta in [-1, 1] mapped from adoption fraction.
        """
        pass
        # Map neighbor adoption fraction [0,1] to influence in [-1,1]
        peer_signal = 2.0 * neighbor_adoption_fraction - 1.0
        delta = self.susceptibility_to_peer_influence * peer_signal
        return max(-1.0, min(1.0, delta))

    def receive_information(self, authority_signal: float, media_signal: float) -> Tuple[float, float]:
        """
        Receive information from public health authorities and media, updating trust and risk.

        Args:
            authority_signal: Guidance effect from authority in [-1, 1].
            media_signal: Net media effect in [-1, 1].

        Returns:
            Updated (trust_in_authorities, risk_perception).
        """
        pass
        # Trust nudged by authority signal
        self.trust_in_authorities = max(0.0, min(1.0, self.trust_in_authorities + 0.1 * authority_signal))
        # Risk perception responds to both signals, modulated by misinformation exposure
        info_effect = 0.1 * authority_signal + 0.15 * media_signal * (1.0 - self.misinformation_exposure)
        self.risk_perception = max(0.0, min(1.0, self.risk_perception + info_effect))
        return self.trust_in_authorities, self.risk_perception

    def decide_adoption(self, base_attitude: float, price: float, rng: random.Random) -> bool:
        """
        Decide whether to adopt mask-wearing this day, based on attitude, trust, risk, costs, habit, and fatigue.

        Args:
            base_attitude: Current latent attitude in [-1, 1] before mapping to probability.
            price: Current market price for masks.
            rng: Random generator.

        Returns:
            Updated adoption state (True/False).
        """
        pass
        # Combine multiple factors into an intention score
        # Authority alignment increases intention; fatigue & perceived cost reduce it.
        cost_component = self.perceived_cost_of_masking + 0.1 * price
        habit_boost = 0.4 * self.habit_strength
        fatigue_penalty = 0.5 * self.fatigue_level
        risk_boost = 0.6 * self.risk_perception
        trust_boost = 0.3 * self.trust_in_authorities

        score = base_attitude + habit_boost + risk_boost + trust_boost - cost_component - fatigue_penalty
        p_adopt = sigmoid(score)
        will_adopt = rng.random() < p_adopt
        # If adopting, ensure mask availability (inventory will be handled externally)
        return will_adopt

    def update_fatigue_and_habit(self) -> None:
        """
        Update fatigue and habit strength based on current adoption state.

        Returns:
            None.
        """
        pass
        if self.adoption_state:
            # Wearing daily builds habit but also increases fatigue
            self.habit_strength = max(0.0, min(1.0, self.habit_strength + 0.02))
            self.fatigue_level = max(0.0, min(1.0, self.fatigue_level + 0.01))
            # Consume a mask unit if using disposable; if no inventory, adoption may drop next day
            if self.mask_inventory > 0:
                self.mask_inventory -= 1
        else:
            # Not wearing reduces fatigue over time and may decay habit slightly
            self.fatigue_level = max(0.0, self.fatigue_level - 0.02)
            self.habit_strength = max(0.0, self.habit_strength - 0.01)

    def acquire_masks(self, qty: int) -> None:
        """
        Increase mask inventory by a specified quantity.

        Args:
            qty: Number of masks acquired.

        Returns:
            None.
        """
        pass
        self.mask_inventory += max(0, int(qty))


@dataclass
class Location:
    """
    Location entity where compliance can be enforced.

    Attributes:
        id: Unique identifier.
        type: Type label (e.g., household, workplace).
        capacity: Nominal capacity.
        contact_rate: Contact rate weight.
        mask_policy: One of {'none', 'recommended', 'required'}.
        enforcement_strength: Probability of enforcing compliance when masks required [0, 1].
        open: Whether the location is open.
    """
    pass
    id: int = 0
    type: str = "retail"
    capacity: int = 100
    contact_rate: float = 1.0
    mask_policy: str = "none"
    enforcement_strength: float = 0.5
    open: bool = True

    def enforce_policy(self, wearing_mask: bool, rng: random.Random) -> Tuple[bool, bool]:
        """
        Enforce policy on entrance; may deny entry or allow a violation.

        Args:
            wearing_mask: Whether the entrant is wearing a mask.
            rng: Random generator.

        Returns:
            Tuple (allowed_entry, violation_occurred).
        """
        pass
        if not self.open:
            return False, False
        if self.mask_policy != "required":
            # No enforcement
            return True, False
        if wearing_mask:
            return True, False
        # Non-wearer attempting entry
        enforce = rng.random() < self.enforcement_strength
        if enforce:
            return False, False  # denied
        else:
            return True, True  # violation allowed

    def host_contacts(self, entrants: int) -> int:
        """
        Placeholder for contact events; returns number of contacts proportional to entrants and rate.

        Args:
            entrants: Number of people entering.

        Returns:
            Estimated number of contacts.
        """
        pass
        return int(self.contact_rate * max(0, entrants))


@dataclass
class PublicHealthAuthority:
    """
    Public health authority managing mandates and messaging.

    Attributes:
        id: Unique identifier.
        policy_schedule: Dict with keys 'mandate_start_day' and 'mandate_end_day' or None.
        enforcement_resources: Scalar affecting location enforcement levels [0, 1].
        messaging_strategy: Dict describing campaign intensity and start day.
        credibility: Credibility weight [0, 1].
    """
    pass
    id: int = 0
    policy_schedule: Dict[str, Optional[int]] = field(default_factory=dict)
    enforcement_resources: float = 0.5
    messaging_strategy: Dict[str, Any] = field(default_factory=dict)
    credibility: float = 0.7

    def mandate_active(self, day: int) -> bool:
        """
        Determine if the mandate is active on a given day.

        Args:
            day: Current simulation day.

        Returns:
            True if mandate is active, False otherwise.
        """
        pass
        start = self.policy_schedule.get("mandate_start_day")
        end = self.policy_schedule.get("mandate_end_day")
        if start is None:
            return False
        if end is None:
            return day >= start
        return start <= day <= end

    def broadcast_guidance(self, day: int) -> float:
        """
        Broadcast guidance signal.

        Args:
            day: Current simulation day.

        Returns:
            Guidance signal strength in [-1, 1], positive promoting mask use.
        """
        pass
        intensity = float(self.messaging_strategy.get("campaign_intensity", 0.3))
        start_day = self.messaging_strategy.get("campaign_start_day")
        active_signal = intensity if (start_day is not None and day >= start_day) else 0.0
        # Mandate provides an additional signal scaled by credibility
        policy_signal = 0.2 if self.mandate_active(day) else 0.0
        combined = (active_signal + policy_signal) * self.credibility
        return max(-1.0, min(1.0, combined))


@dataclass
class MediaChannel:
    """
    Media channel broadcasting information or misinformation.

    Attributes:
        id: Unique identifier.
        reach: Fraction of population reached daily [0, 1].
        bias: Orientation toward masks in [-1, 1], negative means anti-mask.
        message_frequency: Probability of broadcasting each day [0, 1].
        misinformation_probability: Probability that carried message is misinformation [0, 1].
    """
    pass
    id: int = 0
    reach: float = 0.6
    bias: float = 0.0
    message_frequency: float = 0.6
    misinformation_probability: float = 0.1

    def broadcast_messages(self, day: int, rng: random.Random) -> float:
        """
        Produce a media signal for the day.

        Args:
            day: Current simulation day (unused placeholder for potential scheduling).
            rng: Random generator.

        Returns:
            Media signal in [-1, 1].
        """
        pass
        if rng.random() > self.message_frequency:
            return 0.0
        misinfo = rng.random() < self.misinformation_probability
        base_signal = 0.25 * (1.0 if not misinfo else -1.0)
        signal = base_signal + 0.5 * self.bias
        return max(-1.0, min(1.0, signal))


@dataclass
class SupplyMarket:
    """
    Simple supply market for masks with production, distribution, price, and elasticity.

    Attributes:
        mask_supply_level: Current stock level (units).
        production_rate: Units produced per day.
        distribution_rate: Units leaving warehouse per day (baseline).
        price: Current price per unit.
        elasticity: Price elasticity factor controlling adjustment.
    """
    pass
    mask_supply_level: float = 1000.0
    production_rate: float = 200.0
    distribution_rate: float = 150.0
    price: float = 1.0
    elasticity: float = 0.3

    def update_daily(self, demand: float) -> None:
        """
        Update supply level and price based on demand pressure.

        Args:
            demand: Total units demanded today.

        Returns:
            None.
        """
        pass
        # FIXED: Implemented supply and price update to remove no-op
        self.mask_supply_level = max(0.0, self.mask_supply_level + self.production_rate - self.distribution_rate)
        sold = min(self.mask_supply_level, demand)
        unmet = max(0.0, demand - sold)
        # Price adjustment based on unmet demand ratio
        stress = 0.0 if demand <= 0 else unmet / demand
        target_factor = 1.0 + 0.2 * (2.0 * stress - 1.0)
        self.price = max(0.1, self.price * (1.0 + self.elasticity * (target_factor - 1.0)))
        self.mask_supply_level -= sold

    def allocate(self, demander_ids: List[int], rng: random.Random) -> List[int]:
        """
        Allocate available masks to a subset of demanders.

        Args:
            demander_ids: List of agent IDs requesting one unit each.
            rng: Random generator.

        Returns:
            List of agent IDs who received a mask.
        """
        pass
        available = int(max(0.0, self.mask_supply_level))
        if available <= 0 or len(demander_ids) == 0:
            return []
        if available >= len(demander_ids):
            allocated = list(demander_ids)
        else:
            allocated = rng.sample(demander_ids, available)
        self.mask_supply_level -= len(allocated)
        return allocated


class Simulation:
    """
    Main simulation class coordinating agents, locations, market, authority, and media.

    Provides methods to run the simulation loop, compute metrics/validations,
    visualize outcomes, and save results.

    Usage:
        sim = Simulation(params)
        results = sim.run()
        sim.save_results("results.csv")
        sim.visualize(show=False)
    """
    pass

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the simulation with given parameters.

        Args:
            params: Configuration parameters including population size, time horizon, and model controls.

        Returns:
            None.
        """
        pass
        # FIXED: Implement deterministic seeding and direct execution without external tools
        self.params = params.copy()
        self.seed = int(self.params.get("seed", 42))
        self.rng = random.Random(self.seed)

        # Simulation scales
        self.population_size = int(self.params.get("population_size", 1000))
        self.days = int(self.params.get("time_horizon_days", 180))
        self.network_type = str(self.params.get("network_type", "small_world"))
        self.average_degree = int(self.params.get("average_degree", 8))
        self.small_world_beta = float(self.params.get("rewiring_probability", 0.1))
        self.initial_adoption_rate = float(self.params.get("initial_adoption_rate", 0.05))

        # Influence weights
        self.peer_influence_strength = float(self.params.get("peer_influence_strength", 0.5))
        self.authority_influence_weight = float(self.params.get("authority_influence_weight", 0.4))
        self.risk_perception_weight = float(self.params.get("risk_perception_weight", 0.3))
        self.attitude_stubbornness = float(self.params.get("attitude_stubbornness", 0.6))

        # Campaign and misinformation
        self.campaign_intensity = float(self.params.get("campaign_intensity", 0.3))
        self.campaign_start_day = self.params.get("campaign_start_day", 14)
        self.misinformation_intensity = float(self.params.get("misinformation_intensity", 0.2))
        self.misinformation_start_day = self.params.get("misinformation_start_day", 10)

        # Policy
        self.mandate_start_day = self.params.get("mandate_start_day", None)
        self.mandate_end_day = self.params.get("mandate_end_day", None)
        self.enforcement_probability = float(self.params.get("enforcement_probability", 0.5))

        # Market
        self.mask_supply_level = float(self.params.get("mask_supply_level", 1000.0))
        self.supply_restock_rate_per_day = float(self.params.get("supply_restock_rate_per_day", 200.0))
        self.distribution_rate_per_day = float(self.params.get("distribution_rate_per_day", 150.0))
        self.mask_cost = float(self.params.get("mask_cost", 1.0))
        self.price_elasticity = float(self.params.get("price_elasticity", 0.3))

        # Locations
        self.location_types = list(self.params.get("location_types", [
            "household",
            "workplace",
            "school",
            "retail",
            "public_transport",
            "outdoors",
        ]))
        self.contact_rate_by_location_type = dict(self.params.get("contact_rate_by_location_type", {
            "household": 4.0,
            "workplace": 6.0,
            "school": 8.0,
            "retail": 3.0,
            "public_transport": 5.0,
            "outdoors": 1.0,
        }))

        # Containers
        self.people: List[Person] = []
        self.locations: List[Location] = []
        self.authority: PublicHealthAuthority
        self.media: MediaChannel
        self.market: SupplyMarket

        # Metrics logging
        self.history: Dict[str, List[Any]] = {
            "adoption_rate": [],
            "compliance_mandated": [],
            "violation_rate": [],
        }
        self.adoption_by_location_type: Dict[str, List[float]] = {lt: [] for lt in self.location_types}
        self._build_world()

    def _build_world(self) -> None:
        """
        Construct agents, social network, locations, and institutions.

        Returns:
            None.
        """
        pass
        # Build network
        neighbors = build_small_world_neighbors(self.population_size, self.average_degree, self.small_world_beta, self.rng)

        # Initialize people
        for i in range(self.population_size):
            adopted = self.rng.random() < self.initial_adoption_rate
            attitude = (0.5 if adopted else -0.5) + self.rng.uniform(-0.1, 0.1)
            p = Person(
                id=i,
                age=self.rng.randint(18, 80),
                household_id=i // 3,
                current_location_id=None,
                social_neighbors_ids=neighbors[i],
                adoption_state=adopted,
                adoption_threshold=max(0.0, min(1.0, self.rng.gauss(0.5, 0.15))),
                susceptibility_to_peer_influence=max(0.0, min(1.0, self.rng.uniform(0.3, 0.8))),
                trust_in_authorities=max(0.0, min(1.0, self.rng.uniform(0.3, 0.8))),
                misinformation_exposure=max(0.0, min(1.0, self.rng.uniform(0.0, 0.6))),
                risk_perception=max(0.0, min(1.0, self.rng.uniform(0.1, 0.5))),
                perceived_cost_of_masking=max(0.0, min(1.0, self.rng.uniform(0.1, 0.5))),
                fatigue_level=max(0.0, min(1.0, self.rng.uniform(0.0, 0.2))),
                habit_strength=max(0.0, min(1.0, 0.5 if adopted else 0.0)),
                mask_inventory=self.rng.randint(0, 3) if adopted else 0,
            )
            # Store an extra attribute for dynamic attitude
            setattr(p, "attitude", max(-1.0, min(1.0, attitude)))
            self.people.append(p)

        # Initialize locations
        loc_count = max(20, self.population_size // 20)
        self.locations = []
        type_weights = [self.contact_rate_by_location_type.get(lt, 1.0) for lt in self.location_types]
        weight_sum = sum(type_weights) if sum(type_weights) > 0 else 1.0
        normalized = [w / weight_sum for w in type_weights]
        for i in range(loc_count):
            # Randomly assign types weighted by contact rates
            r = self.rng.random()
            cum = 0.0
            idx = 0
            for j, w in enumerate(normalized):
                cum += w
                if r <= cum:
                    idx = j
                    break
            lt = self.location_types[idx]
            enforcement_strength = self.enforcement_probability if lt not in ["household", "outdoors"] else 0.2
            loc = Location(
                id=i,
                type=lt,
                capacity=self.rng.randint(50, 500),
                contact_rate=float(self.contact_rate_by_location_type.get(lt, 1.0)),
                mask_policy="none",  # will be updated when mandate active
                enforcement_strength=enforcement_strength,
                open=True,
            )
            self.locations.append(loc)

        # Initialize authority
        self.authority = PublicHealthAuthority(
            id=0,
            policy_schedule={"mandate_start_day": self.mandate_start_day, "mandate_end_day": self.mandate_end_day},
            enforcement_resources=self.enforcement_probability,
            messaging_strategy={"campaign_intensity": self.campaign_intensity, "campaign_start_day": self.campaign_start_day},
            credibility=0.7,
        )

        # Initialize media with slight pro-mask bias by default
        self.media = MediaChannel(
            id=0,
            reach=0.7,
            bias=0.1,
            message_frequency=0.8,
            misinformation_probability=self.misinformation_intensity,
        )

        # Initialize market
        self.market = SupplyMarket(
            mask_supply_level=self.mask_supply_level,
            production_rate=self.supply_restock_rate_per_day,
            distribution_rate=self.distribution_rate_per_day,
            price=self.mask_cost,
            elasticity=self.price_elasticity,
        )

    def _update_location_policies(self, day: int) -> None:
        """
        Update each location's mask_policy based on authority mandate.

        Args:
            day: Current simulation day.

        Returns:
            None.
        """
        pass
        mandate = self.authority.mandate_active(day)
        for loc in self.locations:
            if mandate and loc.type not in ["household", "outdoors"]:
                loc.mask_policy = "required"
            else:
                # If not mandated, recommend in higher-risk settings
                loc.mask_policy = "recommended" if loc.type in ["public_transport", "school"] else "none"

    def _compute_neighbor_adoption(self) -> List[float]:
        """
        Compute fraction of neighbors adopting for each person.

        Returns:
            List of fractions aligned with person indices.
        """
        pass
        adoption_array = [1.0 if p.adoption_state else 0.0 for p in self.people]
        fractions: List[float] = []
        for p in self.people:
            neigh = p.social_neighbors_ids
            if not neigh:
                fractions.append(0.0)
                continue
            s = sum(adoption_array[j] for j in neigh)
            fractions.append(s / len(neigh))
        return fractions

    def _simulate_visits_and_enforcement(self, wearing_mask: List[bool]) -> Tuple[float, float, Dict[str, float]]:
        """
        Simulate location visits and enforcement effects.

        Args:
            wearing_mask: A list of booleans indicating mask wearing for each person.

        Returns:
            Tuple of (compliance_in_mandated_locations, violation_rate, adoption_by_type for the day).
        """
        pass
        # Sample a target location for each person weighted by contact rates
        type_weights = [self.contact_rate_by_location_type.get(lt, 1.0) for lt in self.location_types]
        weight_sum = sum(type_weights) if sum(type_weights) > 0 else 1.0
        probs = [w / weight_sum for w in type_weights]

        def choose_type() -> str:
            r = self.rng.random()
            cum = 0.0
            for lt, p in zip(self.location_types, probs):
                cum += p
                if r <= cum:
                    return lt
            return self.location_types[-1]

        # Map type -> locations of that type
        loc_by_type: Dict[str, List[Location]] = {}
        for loc in self.locations:
            loc_by_type.setdefault(loc.type, []).append(loc)

        entries_mandated = 0
        compliant_entries = 0
        violations = 0

        type_entries: Dict[str, int] = {lt: 0 for lt in self.location_types}
        type_masked_entries: Dict[str, int] = {lt: 0 for lt in self.location_types}

        for i, p in enumerate(self.people):
            lt = choose_type()
            locs = loc_by_type.get(lt, [])
            if not locs:
                continue
            loc = self.rng.choice(locs)
            allowed, violation = loc.enforce_policy(wearing_mask[i], self.rng)

            if loc.mask_policy == "required":
                entries_mandated += 1
                if wearing_mask[i]:
                    compliant_entries += 1
                if violation:
                    violations += 1

            if allowed:
                type_entries[lt] += 1
                if wearing_mask[i]:
                    type_masked_entries[lt] += 1

        compliance_in_mandated = (compliant_entries / entries_mandated) if entries_mandated > 0 else 1.0
        violation_rate = (violations / entries_mandated) if entries_mandated > 0 else 0.0
        adoption_by_type_day: Dict[str, float] = {
            lt: (type_masked_entries[lt] / type_entries[lt] if type_entries[lt] > 0 else 0.0)
            for lt in self.location_types
        }
        return compliance_in_mandated, violation_rate, adoption_by_type_day


# Execute main for both direct execution and sandbox wrapper invocation
main()