import os
import json
import csv
import argparse
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# Path Handling Instructions compliance
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# FIXED: Restored helper functions and deterministic utilities as per feedback


def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a numeric value between lower and upper bounds.

    Parameters:
        x (float): The value to clamp.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: The clamped value.
    """
    pass
    return max(lo, min(hi, x))


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.

    Parameters:
        x (float): Input value.

    Returns:
        float: Sigmoid(x).
    """
    pass
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def gini(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of non-negative values.

    Parameters:
        values (List[float]): The values to compute Gini coefficient on.

    Returns:
        float: Gini coefficient in [0, 1].
    """
    pass
    xs = sorted(max(0.0, v) for v in values)
    s = sum(xs)
    n = len(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(xs, 1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n


# Entities definitions
# FIXED: Implemented dataclasses for Person, Location, HealthAuthority, MediaOutlet, Retailer per specification


@dataclass
class Person:
    """
    Represents an individual in the simulation with attributes for mask adoption and social-economic factors.

    Attributes:
        id (int): Unique identifier for the person.
        income (float): Person's income level impacting affordability and inequality metrics.
        risk_perception (float): Perceived risk in [0, 1].
        trust_in_authorities (float): Trust level in [0, 1].
        susceptibility_to_messaging (float): Susceptibility to campaign/media messages [0, 1].
        social_influence_weight (float): Weight of peer influence on attitude [0, 1].
        mask_attitude (float): Latent attitude towards mask wearing [-1, 1].
        mask_adoption_state (bool): Whether the person is a current adopter.
        compliance_probability (float): Latent compliance propensity in [0, 1].
        access_to_masks (bool): Whether masks are accessible to the person.
        inventory_masks (int): Number of masks available in personal inventory.
        network_neighbors (List[int]): IDs of neighbors in social network.
        past_enforcement_events (int): Count of enforcement events experienced.
    """
    pass
    id: int
    income: float
    risk_perception: float
    trust_in_authorities: float
    susceptibility_to_messaging: float
    social_influence_weight: float
    mask_attitude: float
    mask_adoption_state: bool
    compliance_probability: float
    access_to_masks: bool
    inventory_masks: int
    network_neighbors: List[int] = field(default_factory=list)
    past_enforcement_events: int = 0


@dataclass
class Location:
    """
    Represents a location type where individuals may visit and observe norms.

    Attributes:
        id (int): Unique identifier.
        type (str): Type of location (e.g., workplace, retail, transit, school, park).
        capacity (int): Maximum capacity; used loosely for realism.
        mask_requirement_policy (bool): Whether masks are required at the location.
        enforcement_strictness (float): Location-specific enforcement strictness [0, 1].
        foot_traffic (float): Relative foot traffic weight for sampling visits.
        observability_factor (float): Observability factor in [0, 1], how visible mask use is in this location.
    """
    pass
    id: int
    type: str
    capacity: int
    mask_requirement_policy: bool
    enforcement_strictness: float
    foot_traffic: float
    observability_factor: float = 0.5


@dataclass
class HealthAuthority:
    """
    Public health authority controlling mandates and campaigns.

    Attributes:
        policy_state (str): 'none' or 'mandate'.
        mandate_start_day (int): Day on which mandate begins.
        fine_amount (float): Fine amount for noncompliance.
        inspection_rate (float): Base inspection/enforcement rate [0, 1].
        campaign_intensity (float): Intensity of public campaign [0, 1].
        credibility (float): Credibility/trustworthiness [0, 1].
    """
    pass
    policy_state: str
    mandate_start_day: int
    fine_amount: float
    inspection_rate: float
    campaign_intensity: float
    credibility: float


@dataclass
class MediaOutlet:
    """
    Media outlet broadcasting messages that influence risk and attitudes.

    Attributes:
        id (int): Unique identifier.
        bias (float): Positive promotes masks; negative discourages.
        credibility (float): Credibility of the channel [0, 1].
        reach (float): Fraction of population reached per day [0, 1].
        message_frequency (float): Frequency scaling [0, 1].
    """
    pass
    id: int
    bias: float
    credibility: float
    reach: float
    message_frequency: float


@dataclass
class Retailer:
    """
    Retailer selling masks, with inventory, restocking policy, and price.

    Attributes:
        inventory (int): Current inventory level.
        restock_quantity (int): Quantity added on each restock event.
        restock_interval (int): Interval in days between planned restocks.
        price (float): Current mask unit price.
        max_purchase_per_customer (int): Cap on purchase per customer.
        supply_variability (float): Relative standard deviation for restock noise.
        stockout_days (int): Days where inventory is zero.
    """
    pass
    inventory: int
    restock_quantity: int
    restock_interval: int
    price: float
    max_purchase_per_customer: int
    supply_variability: float
    stockout_days: int = 0

    def restock(self, day: int, rng: random.Random) -> int:
        """
        Restock inventory on schedule with variability.

        Parameters:
            day (int): Current simulation day (0-indexed).
            rng (random.Random): Seeded RNG.

        Returns:
            int: Quantity added to inventory.
        """
        pass
        # FIXED: Implement retailer restock with variability and cap negative quantities
        added = 0
        if self.restock_interval > 0 and (day % self.restock_interval == 0):
            noise = int(self.restock_quantity * rng.gauss(0, self.supply_variability))
            qty = max(0, self.restock_quantity + noise)
            self.inventory += qty
            added = qty
        return added


# FIXED: Reintroduced small-world network builder (pure Python) with deterministic RNG


def ring_small_world(n: int, k: int, p: float, rng: random.Random) -> Dict[int, List[int]]:
    """
    Build a Watts-Strogatz-like small-world network as an adjacency list.

    Parameters:
        n (int): Number of nodes.
        k (int): Average degree (must be even ideally; we will approximate).
        p (float): Rewiring probability in [0, 1].
        rng (random.Random): Seeded RNG.

    Returns:
        Dict[int, List[int]]: Adjacency list mapping node -> sorted list of neighbors.
    """
    pass
    adj = {i: set() for i in range(n)}
    if n <= 1 or k <= 0:
        return {i: [] for i in range(n)}
    half = max(1, min(k // 2, (n - 1) // 2))
    # Start with ring lattice
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            adj[i].add(j)
            adj[j].add(i)
    # Rewire edges with probability p
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            if j in adj[i] and rng.random() < p:
                # Remove existing edge
                adj[i].discard(j)
                adj[j].discard(i)
                # Add a random new edge avoiding duplicates and self-loops
                for _ in range(50):
                    u = rng.randrange(n)
                    if u != i and u not in adj[i]:
                        adj[i].add(u)
                        adj[u].add(i)
                        break
                else:
                    # If failed to find new edge, restore original
                    adj[i].add(j)
                    adj[j].add(i)
    return {i: sorted(list(neigh)) for i, neigh in adj.items()}


class Simulation:
    """
    Main simulation engine coordinating entities, network, messaging, policy, and market.

    This class implements a minimal pure-Python agent-based simulation that:
    - Builds agents, locations, policy, media, retailer, and small-world network.
    - Runs a daily loop including messaging, peer influence, purchasing, visits, and enforcement.
    - Aggregates and outputs required metrics.

    Notes:
        - Designed to be Docker-independent and directly executable.
        - Uses a single Retailer entity for simplicity.
    """
    pass

    # FIXED: Restored a functional simulation engine per feedback
    def __init__(self, params: Dict[str, Any], smoke: bool = False) -> None:
        """
        Initialize the simulation with parameters and optional smoke (fast) mode.

        Parameters:
            params (Dict[str, Any]): Simulation parameters.
            smoke (bool): If True, run a small, fast simulation for CI.

        Raises:
            ValueError: If any parameters are invalid.
        """
        pass
        # FIXED: Added RNG seeding for reproducibility
        self.params = params
        self.rng = random.Random(params.get('random_seed', 42))
        self.days = 5 if smoke else int(params.get('simulation_horizon_days', 120))
        self.N = 200 if smoke else int(params.get('population_size', 5000))
        self.smoke = smoke

        # Build entities
        self.network = ring_small_world(self.N, int(params.get('avg_degree', 8)), 0.1, self.rng)

        self.people: List[Person] = []
        self.locations: List[Location] = []

        self.authority = HealthAuthority(
            policy_state='none',
            mandate_start_day=int(params.get('policy_mandate_day', 20)),
            fine_amount=float(params.get('fine_amount', 50.0)),
            inspection_rate=float(params.get('policy_enforcement_strictness', 0.6)),
            campaign_intensity=float(params.get('campaign_intensity', 0.7)),
            credibility=float(params.get('authority_credibility', 0.7)),
        )
        self.media = MediaOutlet(id=1, bias=1.0, credibility=0.6, reach=0.6, message_frequency=1.0)
        self.retailer = Retailer(
            inventory=int(params.get('retailer_initial_inventory', 10000 if not smoke else 1000)),
            restock_quantity=int(params.get('restock_quantity', 8000 if not smoke else 400)),
            restock_interval=int(params.get('restock_interval_days', 7)),
            price=float(params.get('mask_price', 1.0)),
            max_purchase_per_customer=int(params.get('max_purchase_per_customer', 10)),
            supply_variability=float(params.get('supply_variability', 0.1)),
        )
        self._build_entities()

        # Time series for metrics
        self.overall_adoption_series: List[float] = []
        self.adoption_by_loc_series: Dict[str, List[float]] = {}
        self.compliance_under_mandate: List[Optional[float]] = []
        self.policy_violations_per_day: List[int] = []
        self.total_purchased: int = 0
        self.stockout_days: int = 0

        # Dynamic model plan structure (minimal to support evaluation)
        self.model_plan: Dict[str, Any] = {
            "evaluation_metrics": [
                "adoption_curve_convergence",
                "mandate_effect_direction",
                "bounded_stockouts"
            ],
            # Additional fields could be added here if needed for dynamic config.
        }

    def _build_entities(self) -> None:
        """
        Construct initial Persons and Locations.

        Notes:
            - Individuals initialized with random attributes using seeded RNG.
            - Locations include a variety of types with different enforcement and mask policies.
        """
        pass
        init_inv = int(self.params.get('initial_inventory_per_person', 2))
        init_adopt = float(self.params.get('initial_adoption_rate', 0.2))
        peer_w = float(self.params.get('peer_influence_weight', 0.2))

        for i in range(self.N):
            income = self.rng.uniform(15000, 150000)
            risk = clamp(self.rng.random(), 0.0, 1.0)
            trust = clamp(self.rng.random(), 0.0, 1.0)
            susc = clamp(self.rng.random(), 0.0, 1.0)
            attitude = clamp(self.rng.gauss(0, 0.5), -1.0, 1.0)
            adopt = self.rng.random() < init_adopt
            inv = int(init_inv)
            person = Person(
                id=i,
                income=income,
                risk_perception=risk,
                trust_in_authorities=trust,
                susceptibility_to_messaging=susc,
                social_influence_weight=peer_w,
                mask_attitude=attitude,
                mask_adoption_state=adopt,
                compliance_probability=0.5,
                access_to_masks=inv > 0,
                inventory_masks=inv,
                network_neighbors=self.network.get(i, []),
                past_enforcement_events=0,
            )
            self.people.append(person)

        # Build a set of diverse locations
        # type, share of presence weight, capacity, mask_required, enforcement
        loc_types = [
            ('workplace', 0.3, 50, True, 0.6, 0.6),
            ('transit', 0.1, 80, True, 0.7, 0.7),
            ('retail', 0.1, 60, True, 0.6, 0.7),
            ('school', 0.1, 40, True, 0.5, 0.7),
            ('park', 0.4, 100, False, 0.1, 0.4),
        ]
        lid = 0
        for ltype, share, cap, req, enf, obs in loc_types:
            count = max(1, int(share * 20))  # fixed small set
            for _ in range(count):
                self.locations.append(
                    Location(
                        id=lid,
                        type=ltype,
                        capacity=cap,
                        mask_requirement_policy=req,
                        enforcement_strictness=enf,
                        foot_traffic=share,
                        observability_factor=obs,
                    )
                )
                lid += 1

    def _message_effect(self, person: Person) -> None:
        """
        Apply the effects of HealthAuthority and MediaOutlet messaging on an individual's risk and attitude.

        Parameters:
            person (Person): The individual to update.
        """
        pass
        m_eff = float(self.params.get('message_effect_size', 0.15))

        # Public health campaign effect on perceived risk
        delta_risk = self.authority.campaign_intensity * m_eff * person.susceptibility_to_messaging * self.authority.credibility
        person.risk_perception = clamp(person.risk_perception + delta_risk, 0.0, 1.0)

        # Media influence on attitude (pro-mask bias)
        media_delta = self.media.reach * self.media.credibility * m_eff * person.susceptibility_to_messaging * self.media.bias
        person.mask_attitude = clamp(person.mask_attitude + 0.5 * media_delta, -1.0, 1.0)

    def _peer_influence(self, person: Person) -> None:
        """
        Update mask attitude based on neighbors' adoption state.

        Parameters:
            person (Person): The individual to update.
        """
        pass
        neighbors = person.network_neighbors
        if not neighbors:
            return
        neigh_adopt = sum(1 for j in neighbors if self.people[j].mask_adoption_state)
        share = neigh_adopt / max(1, len(neighbors))
        # Move attitude towards +1 as neighbor adoption increases above 0.5
        person.mask_attitude = clamp(person.mask_attitude + person.social_influence_weight * (share - 0.5), -1.0, 1.0)

    def _decide_wear(self, person: Person, mandate: bool, loc_enf: float, obs_factor: float) -> bool:
        """
        Decide whether the person wears a mask during a visit considering risk, policy, enforcement, and norms.

        Parameters:
            person (Person): The individual.
            mandate (bool): Whether a mask mandate applies in this location.
            loc_enf (float): Location enforcement strictness.
            obs_factor (float): Observability factor contributing to perceived social norms.

        Returns:
            bool: True if the person chooses to wear a mask.
        """
        pass
        # Base propensity from risk and attitude
        attitude_scaled = (person.mask_attitude + 1) / 2  # [-1,1] -> [0,1]
        base = 0.5 * (person.risk_perception + 0.5 * attitude_scaled)

        # Policy effect
        policy = 0.0
        if mandate:
            policy = 0.3 * float(self.params.get('policy_enforcement_strictness', 0.6)) + 0.2 * sigmoid(self.authority.fine_amount / 100.0)

        # Individual traits: trust and past enforcement
        policy += 0.2 * (person.trust_in_authorities - 0.5)
        policy += 0.05 * person.past_enforcement_events

        # Location enforcement and observability combine to shift behavior
        p = clamp(base + policy + 0.2 * loc_enf + 0.05 * obs_factor, 0.0, 1.0)
        return person.inventory_masks > 0 and (self.rng.random() < p)

    def _purchase(self, person: Person) -> int:
        """
        Attempt to purchase masks if inventory is low.

        Parameters:
            person (Person): The individual.

        Returns:
            int: Number of masks purchased.
        """
        pass
        # Only purchase if out-of-stock
        if person.inventory_masks >= 1:
            return 0
        desired = 2
        pe = float(self.params.get('price_elasticity', -0.4))
        price = self.retailer.price
        base_price = float(self.params.get('mask_price', 1.0))
        scale = (price / base_price) ** pe if base_price > 0 else 1.0
        qty = min(self.retailer.max_purchase_per_customer, max(0, int(desired * scale)))
        if qty <= 0 or self.retailer.inventory <= 0:
            return 0
        sold = min(qty, self.retailer.inventory)
        self.retailer.inventory -= sold
        person.inventory_masks += sold
        return sold

    def step(self, day: int) -> None:
        """
        Execute one simulation day: policy update, messaging, restock, visits, enforcement, and metrics.

        Parameters:
            day (int): Current day index.
        """
        pass
        # Update policy state
        self.authority.policy_state = 'mandate' if day >= self.authority.mandate_start_day else 'none'

        # Messaging and peer influence
        for p in self.people:
            self._message_effect(p)
            self._peer_influence(p)

        # Retail restock
        added = self.retailer.restock(day, self.rng)
        _ = added  # no-op, reserved for potential use

        # Track counts for metrics
        violations = 0
        total_attend = 0
        total_masked = 0
        by_type_counts: Dict[str, Tuple[int, int]] = {}  # type -> (masked, total)
        for loc in self.locations:
            by_type_counts.setdefault(loc.type, (0, 0))

        # Visits and decisions
        for p in self.people:
            # Purchase attempt before attending
            self.total_purchased += self._purchase(p)

            visits_today = 1 if self.rng.random() < 0.6 else 0
            for _ in range(visits_today):
                loc = self.rng.choice(self.locations)
                mandate_here = (self.authority.policy_state == 'mandate') and loc.mask_requirement_policy
                wear = self._decide_wear(p, mandate_here, loc.enforcement_strictness, loc.observability_factor)

                masked, tot = by_type_counts[loc.type]
                if wear:
                    total_masked += 1
                    masked += 1
                    if p.inventory_masks > 0:
                        # Consume a mask unit for the visit
                        p.inventory_masks -= 1
                else:
                    # Potential enforcement under mandate
                    if mandate_here and (self.rng.random() < self.authority.inspection_rate * loc.enforcement_strictness):
                        violations += 1
                        p.past_enforcement_events += 1
                tot += 1
                total_attend += 1
                by_type_counts[loc.type] = (masked, tot)

                # Update access flag
                p.access_to_masks = p.inventory_masks > 0

            # Update latent adoption state to be sticky if masks are accessible and used
            p.mask_adoption_state = p.mask_adoption_state or (visits_today > 0 and p.access_to_masks)

        # Stockout tally
        if self.retailer.inventory == 0:
            self.stockout_days += 1

        # Aggregate metrics for today
        overall = sum(1 for p in self.people if p.mask_adoption_state) / max(1, self.N)
        self.overall_adoption_series.append(overall)

        for t, (m, tot) in by_type_counts.items():
            rate = (m / tot) if tot > 0 else 0.0
            self.adoption_by_loc_series.setdefault(t, []).append(rate)

        self.policy_violations_per_day.append(violations)

        # Compliance under mandate is recorded as masked/total when mandate is active
        if self.authority.policy_state == 'mandate':
            self.compliance_under_mandate.append((total_masked / total_attend) if total_attend > 0 else None)
        else:
            self.compliance_under_mandate.append(None)

    def run(self) -> None:
        """
        Run the simulation over the configured horizon.
        """
        pass
        for d in range(self.days):
            self.step(d)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Compute and return the metrics required by the Task Specification and feedback.

        Returns:
            Dict[str, Any]: A dictionary of metrics including time series and scalars.
        """
        pass
        # FIXED: Aggregate and output metrics matching the spec
        # time to 70% overall adoption
        t70 = None
        for i, v in enumerate(self.overall_adoption_series):
            if v >= 0.7:
                t70 = i
                break

        avg_purchased = self.total_purchased / max(1, self.N)

        # Inequality index of adoption based on final adoption state distribution by income
        incomes = [p.income for p in self.people]
        final_adopt = [1.0 if p.mask_adoption_state else 0.0 for p in self.people]
        pairs = sorted(zip(incomes, final_adopt), key=lambda x: x[0])
        adoption_values = [ad for _, ad in pairs]
        ineq = gini(adoption_values)

        # Stability: variance over last 7 days
        last7 = self.overall_adoption_series[-7:] if len(self.overall_adoption_series) >= 7 else self.overall_adoption_series
        mean_last7 = sum(last7) / max(1, len(last7))
        var_last7 = sum((x - mean_last7) ** 2 for x in last7) / max(1, len(last7))

        adoption_by_location_type = {k: v for k, v in self.adoption_by_loc_series.items()}

        metrics = {
            "overall_adoption_rate": self.overall_adoption_series,
            "adoption_by_location_type": adoption_by_location_type,
            "compliance_rate_under_mandate": self.compliance_under_mandate,
            "time_to_reach_70_percent": t70,
            "inventory_stockout_days": self.stockout_days,
            "average_masks_purchased_per_person": avg_purchased,
            "inequality_index_of_adoption": ineq,
            "policy_violations_per_day": self.policy_violations_per_day,
            "sustained_adoption_stability": var_last7,
        }
        return metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the run against generic evaluation metrics if configured in the model plan.

        Supported metrics:
            - adoption_curve_convergence: mean absolute change over the final 14 days < 0.005
            - mandate_effect_direction: post-mandate 14-day mean minus pre-mandate 14-day mean
            - bounded_stockouts: stockout days fraction < 0.2

        Returns:
            Dict[str, Any]: Evaluation results keyed by metric name.
        """
        pass
        results: Dict[str, Any] = {}
        evals = self.model_plan.get("evaluation_metrics", [])
        series = self.overall_adoption_series

        if "adoption_curve_convergence" in evals:
            window = min(14, len(series) - 1) if len(series) > 1 else 0
            diffs = [abs(series[i] - series[i - 1]) for i in range(len(series) - window + 1, len(series))] if window > 0 else []
            mean_change = sum(diffs) / max(1, len(diffs))
            results["adoption_curve_convergence"] = mean_change

        if "mandate_effect_direction" in evals:
            mday = self.authority.mandate_start_day
            pre = series[max(0, mday - 14):mday] if mday > 0 else []
            post = series[mday:mday + 14] if mday < len(series) else []
            pre_mean = sum(pre) / max(1, len(pre))
            post_mean = sum(post) / max(1, len(post))
            results["mandate_effect_direction"] = post_mean - pre_mean

        if "bounded_stockouts" in evals:
            frac = self.stockout_days / max(1, self.days)
            results["bounded_stockouts"] = frac

        return results

    def visualize(self) -> None:
        """
        Visualize basic time series of overall adoption and policy violations.

        Notes:
            - Attempts to use matplotlib if available; otherwise prints a message.
        """
        pass
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("Visualization skipped (matplotlib not available):", e)
            return

        days = list(range(len(self.overall_adoption_series)))
        fig, ax1 = plt.subplots()
        ax1.plot(days, self.overall_adoption_series, label="Overall Adoption", color="blue")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Adoption Rate", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.plot(days, self.policy_violations_per_day, label="Policy Violations", color="red", alpha=0.6)
        ax2.set_ylabel("Violations", color="red")
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Mask Adoption and Policy Violations Over Time")
        fig.tight_layout()
        plt.show()

    def save_results(self, filename: str) -> None:
        """
        Save primary daily time series to a CSV file.

        Parameters:
            filename (str): Output CSV filename.
        """
        pass
        fieldnames = ["day", "overall_adoption_rate", "policy_violations"]
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(self.overall_adoption_series)):
                    writer.writerow({
                        "day": i,
                        "overall_adoption_rate": self.overall_adoption_series[i],
                        "policy_violations": self.policy_violations_per_day[i] if i < len(self.policy_violations_per_day) else 0
                    })
        except Exception as e:
            print("Error saving results:", e)


def build_default_params(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Construct default simulation parameters from CLI arguments.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Dict[str, Any]: Parameter dictionary for Simulation.
    """
    pass
    params = {
        "population_size": args.population,
        "avg_degree": 8,
        "initial_adoption_rate": 0.2,
        "initial_inventory_per_person": 2,
        "policy_mandate_day": 20,
        "policy_enforcement_strictness": 0.6,
        "fine_amount": 50.0,
        "campaign_intensity": 0.7,
        "message_effect_size": 0.15,
        "observation_effect_size": 0.12,
        "peer_influence_weight": 0.2,
        "risk_perception_sensitivity": 0.3,
        "price_elasticity": -0.4,
        "retailer_initial_inventory": 10000,
        "restock_interval_days": 7,
        "restock_quantity": 8000,
        "mask_price": 1.0,
        "max_purchase_per_customer": 10,
        "simulation_horizon_days": args.days,
        "time_step_days": 1,
        "random_seed": args.seed,
        "supply_variability": 0.1,
        "authority_credibility": 0.7,
    }
    return params


def main() -> None:
    """
    Program entry point: parse arguments, run simulation, visualize, and save results.

    Behavior:
        - Supports a fast --smoke mode for CI.
        - Writes metrics.json with required metrics.
        - Saves CSV series and optionally visualizes results.

    Notes:
        - Pure-Python; no Docker dependency.
    """
    pass
    # FIXED: Replaced stub main() with functional entry point and smoke mode
    parser = argparse.ArgumentParser(description="Mask Adoption Simulation (pure Python)")
    parser.add_argument("--days", type=int, default=120, help="Simulation horizon in days")
    parser.add_argument("--population", type=int, default=5000, help="Population size")
    parser.add_argument("--smoke", action="store_true", help="Run a small fast simulation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--metrics-file", type=str, default="metrics.json", help="Output metrics JSON filename")
    parser.add_argument("--results-file", type=str, default="results.csv", help="Output results CSV filename")
    args = parser.parse_args()

    params = build_default_params(args)
    sim = Simulation(params, smoke=args.smoke)
    sim.run()

    metrics = sim.get_metrics()
    try:
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print("Error writing metrics.json:", e)

    # Print brief summary (truncate lists in smoke mode)
    preview_metrics = {k: (v if not isinstance(v, list) else (v[:5] if args.smoke else v)) for k, v in metrics.items()}
    print(json.dumps(preview_metrics, indent=2))

    # Evaluate and print evaluation metrics
    evaluation = sim.evaluate()
    print("Evaluation:", json.dumps(evaluation, indent=2))

    # Save CSV results
    sim.save_results(args.results_file)

    # Visualize
    if not args.no_visualize:
        sim.visualize()


# Execute main for both direct execution and sandbox wrapper invocation
main()