def main():
    pass

import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

# Path Handling Instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def sigmoid(x: float) -> float:
    """
    Compute the logistic sigmoid function.

    Args:
        x: Input value.

    Returns:
        The logistic sigmoid of x.
    """
    pass
    return 1.0 / (1.0 + np.exp(-x))


def gini_coefficient(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.

    Args:
        values: A list of non-negative numbers.

    Returns:
        Gini coefficient in [0, 1].

    Notes:
        Returns 0.0 for empty input or when sum is zero.
    """
    pass
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return 0.0
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return diff_sum / (2.0 * arr.size * arr.size * mean)


@dataclass
class Person:
    """
    Agent representing an individual decision-maker in the simulation.

    Attributes:
        id: Unique identifier.
        age: Age in years.
        income: Annual income proxy.
        household_id: Id of the household.
        workplace_id: Optional workplace id.
        network_neighbors: List of neighboring agent ids in contact/social network.
        health_status: 'susceptible', 'infected', or 'recovered'.
        vaccination_status: Placeholder vaccination status.
        risk_perception: Current subjective risk perception in [0, 1].
        compliance_propensity: Trait in [0, 1] indicating likelihood to comply.
        social_influence_weight: Weight in [0, 1] for peer influence.
        trust_in_government: Trust parameter in [0, 1].
        trust_in_science: Trust parameter in [0, 1].
        misinformation_susceptibility: Trait in [0, 1] for misinformation effects.
        policy_awareness: Knowledge of current policy in [0, 1].
        fatigue: Behavioral fatigue in [0, 1] reducing sustained adoption.
        mask_status: Mask type in {'none', 'cloth', 'surgical', 'N95'}.
        wears_mask: Whether the person currently wears a mask.
        mask_supply_on_hand: Number of masks owned and usable.
        days_infected: Days since infection start, -1 if not infected.
        income_quintile: Derived quintile for inequity metrics [0..4].
        cost_sensitivity: Trait in [0, 1] indicating price sensitivity.
    """
    id: int
    age: int
    income: float
    household_id: int
    workplace_id: Optional[int]
    network_neighbors: List[int] = field(default_factory=list)
    health_status: str = "susceptible"
    vaccination_status: str = "none"
    risk_perception: float = 0.2
    compliance_propensity: float = 0.5
    social_influence_weight: float = 0.5
    trust_in_government: float = 0.5
    trust_in_science: float = 0.5
    misinformation_susceptibility: float = 0.3
    policy_awareness: float = 0.3
    fatigue: float = 0.0
    mask_status: str = "none"
    wears_mask: bool = False
    mask_supply_on_hand: int = 0
    days_infected: int = -1
    income_quintile: int = 0
    cost_sensitivity: float = 0.5

    def _noop(self) -> None:
        """
        No-op method to satisfy requirement for a 'pass' statement in class body.
        """
        pass


@dataclass
class Household:
    """
    Household grouping for norms and shared resources.

    Attributes:
        id: Household id.
        member_ids: Person ids in this household.
        income_level: Aggregate household income level.
        mask_supply_stock: Shared household mask stock.
        norms_mask_score: Household norm score in [0, 1].
    """
    id: int
    member_ids: List[int]
    income_level: float
    mask_supply_stock: int = 0
    norms_mask_score: float = 0.5

    def _noop(self) -> None:
        """
        No-op method to satisfy requirement of 'pass' within class.
        """
        pass


@dataclass
class Location:
    """
    Location context affecting policy enforcement and exposure.

    Attributes:
        id: Location id.
        type: Location type such as 'workplace', 'store', 'public'.
        capacity: Maximum capacity of occupants.
        contact_rate: Contact intensity proxy.
        ventilation_level: Ventilation effect in [0, 1], higher is better.
        mask_required: Whether masks are required at this location.
        enforcement_level: Location-specific enforcement in [0, 1].
    """
    id: int
    type: str
    capacity: int
    contact_rate: float
    ventilation_level: float
    mask_required: bool
    enforcement_level: float

    def _noop(self) -> None:
        """
        No-op method to satisfy requirement of 'pass' within class.
        """
        pass


@dataclass
class Retailer:
    """
    Retailer selling masks with inventory and pricing.

    Attributes:
        id: Retailer id.
        inventory: Current total inventory of masks.
        restock_rate_daily: Number of masks restocked per day.
        price_per_mask: Dict mask type -> price.
        allocation_policy: Allocation policy string.
        cumulative_sold: Cumulative masks sold.
        stockout_days: Count of days with inventory at zero.
    """
    id: int
    inventory: int
    restock_rate_daily: int
    price_per_mask: Dict[str, float]
    allocation_policy: str = "fifo"
    cumulative_sold: int = 0
    stockout_days: int = 0

    def _noop(self) -> None:
        """
        No-op method to satisfy requirement of 'pass' within class.
        """
        pass


@dataclass
class Government:
    """
    Government policy and communications.

    Attributes:
        policy_state: 'none' or 'mandate'.
        mandate_start_day: Day mandate begins.
        mandate_end_day: Optional end day for mandate.
        enforcement_strength: Overall enforcement in [0, 1].
        fine_amount: Fine amount issued for non-compliance.
        campaign_intensity: Public information campaign intensity in [0, 1].
        message_strategy: Message type strategy string.
        subsidy_per_mask: Dict mask type -> per-mask subsidy.
        policy_cost: Accumulated policy cost from subsidies/enforcement.
        fines_issued: Count of fines issued.
    """
    policy_state: str = "none"
    mandate_start_day: Optional[int] = None
    mandate_end_day: Optional[int] = None
    enforcement_strength: float = 0.4
    fine_amount: float = 50.0
    campaign_intensity: float = 0.2
    message_strategy: str = "risk"
    subsidy_per_mask: Dict[str, float] = field(default_factory=lambda: {"cloth": 0.1, "surgical": 0.2, "N95": 0.3})
    policy_cost: float = 0.0
    fines_issued: int = 0

    def _noop(self) -> None:
        """
        No-op method to satisfy requirement of 'pass' within class.
        """
        pass


@dataclass
class Media:
    """
    Media and information environment.

    Attributes:
        id: Media outlet id.
        reach: Fraction of population reached daily in [0, 1].
        bias: Bias parameter in [-1, 1] where negative may increase misinformation.
        message_frequency: Frequency of messaging in days.
        misinformation_rate: Rate in [0, 1] contributing to misinformation exposure.
    """
    id: int
    reach: float
    bias: float
    message_frequency: float
    misinformation_rate: float

    def _noop(self) -> None:
        """
        No-op method to satisfy requirement of 'pass' within class.
        """
        pass


def default_parameters() -> Dict[str, Any]:
    """
    Provide default simulation parameters.

    Returns:
        A dictionary of parameters for initialization.
    """
    pass
    # FIXED: Provide sensible, lightweight defaults to keep runtime fast.
    return {
        # Core sizes and time
        "population_size": 2000,  # FIXED: Use 2000 instead of 10000 for faster local runs
        "max_time_steps": 180,
        "avg_degree": 10,
        "contact_network_type": "small_world",
        "seed": 42,
        # Initialization
        "initial_mask_adoption_rate": 0.2,
        "initial_infected_count": 30,
        # Disease parameters
        "beta_transmission_base": 0.05,
        "infectious_period_days": 7,
        "ventilation_effect": 0.2,
        # Mask efficacy by type
        "mask_efficacy": {"none": 0.0, "cloth": 0.3, "surgical": 0.5, "N95": 0.9},
        # Behavior model params
        "decision_noise_temperature": 0.5,
        "social_influence_strength": 0.5,
        "risk_perception_update_rate": 0.05,
        "fatigue_increase_rate": 0.005,
        "fatigue_effect_weight": 0.4,
        "policy_weight": 0.5,
        "cost_weight": 0.3,
        "base_mask_preference": 0.0,
        # Policy
        "policy_mandate_day": 30,
        "enforcement_strength": 0.5,
        "fine_amount": 50.0,
        # Retail/market
        "retailer_initial_inventory": 10000,
        "restock_rate_daily": 300,
        "price_per_mask": {"cloth": 1.0, "surgical": 1.5, "N95": 3.0},
        "subsidy_per_mask": {"cloth": 0.1, "surgical": 0.2, "N95": 0.4},
        "mask_replacement_interval_days": 7,
        # Information environment
        "media_reach": 0.5,
        "media_bias": 0.0,
        "media_message_frequency": 7,
        "misinformation_rate": 0.05,
        "campaign_intensity": 0.2,
        # Mobility/locations (simplified)
        "location_mask_required_under_mandate": True,
        "location_enforcement_level": 0.6,
        "location_contact_rate": 1.0,
        # Targets
        "target_adoption": 0.7,
    }


def validate_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill default parameters.

    Args:
        params: Input parameters possibly missing some defaults.

    Returns:
        Validated and completed parameter dictionary.
    """
    pass
    base = default_parameters()
    if params:
        base.update(params)
    # Basic sanity checks with clamps
    base["population_size"] = int(max(10, base.get("population_size", 100)))
    base["max_time_steps"] = int(max(1, base.get("max_time_steps", 30)))
    base["avg_degree"] = int(min(max(2, base.get("avg_degree", 6)), base["population_size"] - 1))
    base["beta_transmission_base"] = float(max(0.0, base.get("beta_transmission_base", 0.05)))
    base["infectious_period_days"] = int(max(2, base.get("infectious_period_days", 7)))
    return base


def build_network(n: int, k: int, rng: np.random.Generator, net_type: str = "small_world") -> nx.Graph:
    """
    Build a contact/social network.

    Args:
        n: Number of nodes.
        k: Average degree (must be even for watts-strogatz graph k).
        rng: Numpy random generator.
        net_type: Network type; 'small_world' or 'erdos_renyi'.

    Returns:
        A networkx graph with n nodes.
    """
    pass
    if net_type == "erdos_renyi":
        p = min(0.9, max(0.0001, k / max(1.0, n - 1.0)))
        return nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(0, 2**31 - 1)))
    # default small-world
    k_ws = max(2, (k // 2) * 2)  # even
    p_rewire = 0.1
    return nx.watts_strogatz_graph(n=n, k=k_ws, p=p_rewire, seed=int(rng.integers(0, 2**31 - 1)))


class MaskAdoptionSimulation:
    """
    Main simulation class orchestrating mask adoption dynamics.

    Attributes:
        params: Simulation parameters dictionary.
        seed: Random seed for reproducibility.
        rng: Numpy RNG.
        people: List of Person agents.
        households: List of Household objects.
        locations: List of Location contexts (simplified to one).
        retailer: Retailer market object.
        government: Government policy object.
        media: Media information environment object.
        G: Contact/social network graph.
        day: Current simulation day index.
        metrics: Dict collecting time series and scalars for reporting.
    """
    def __init__(self, params: Dict[str, Any], seed: Optional[int] = None) -> None:
        """
        Initialize the simulation with parameters and seed.

        Args:
            params: Parameters for the simulation.
            seed: Optional seed to initialize RNG.
        """
        pass
        # FIXED: Parameter validation and default merge
        self.params = validate_parameters(params)
        self.seed = self.params.get("seed", seed if seed is not None else 42)
        self.rng = np.random.default_rng(self.seed)

        self.people: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []
        self.retailer: Optional[Retailer] = None
        self.government: Optional[Government] = None
        self.media: Optional[Media] = None
        self.G: Optional[nx.Graph] = None
        self.day: int = 0
        self.metrics: Dict[str, Any] = {
            "adoption_rate_over_time": [],
            "Rt_effective": [],
            "new_infections": [],
        }

        # Setup entities
        self._setup_entities()

    def _setup_entities(self) -> None:
        """
        Create agents, households, network, and institutions.

        Returns:
            None
        """
        pass
        N = int(self.params["population_size"])

        # Build network
        self.G = build_network(
            n=N,
            k=int(self.params["avg_degree"]),
            rng=self.rng,
            net_type=self.params.get("contact_network_type", "small_world"),
        )
        for node in self.G.nodes:
            self.G.nodes[node]["id"] = node

        # Households (simple grouping)
        hh_size_mean = 3
        household_ids = []
        current = 0
        while current < N:
            size = int(max(1, self.rng.poisson(hh_size_mean)))
            end = min(N, current + size)
            household_ids.append(list(range(current, end)))
            current = end
        for idx, members in enumerate(household_ids):
            income_level = float(self.rng.lognormal(mean=10.5, sigma=0.6))
            self.households.append(Household(id=idx, member_ids=members, income_level=income_level))

        # Income assignments and quintiles
        incomes = np.array([self.households[self._find_household_id(i)].income_level for i in range(N)])
        quintile_edges = np.quantile(incomes, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        def income_to_quintile(val: float) -> int:
            for q in range(5):
                if quintile_edges[q] <= val <= quintile_edges[q + 1]:
                    return q
            return 4

        # Initialize people
        initial_mask_rate = float(self.params["initial_mask_adoption_rate"])
        for i in range(N):
            hh_id = self._find_household_id(i)
            age = int(np.clip(int(self.rng.normal(40, 15)), 0, 95))
            income = self.households[hh_id].income_level
            p = Person(
                id=i,
                age=age,
                income=income,
                household_id=hh_id,
                workplace_id=None,
                risk_perception=float(np.clip(self.rng.normal(0.2, 0.1), 0.0, 1.0)),
                compliance_propensity=float(np.clip(self.rng.beta(2.0, 2.0), 0.0, 1.0)),
                social_influence_weight=float(np.clip(self.rng.normal(0.5, 0.15), 0.0, 1.0)),
                trust_in_government=float(np.clip(self.rng.beta(2.0, 2.0), 0.0, 1.0)),
                trust_in_science=float(np.clip(self.rng.beta(3.0, 1.5), 0.0, 1.0)),
                misinformation_susceptibility=float(np.clip(self.rng.beta(2.0, 3.0), 0.0, 1.0)),
                policy_awareness=float(np.clip(self.rng.uniform(0.2, 0.8), 0.0, 1.0)),
                fatigue=0.0,
                mask_status="none",
                wears_mask=False,
                mask_supply_on_hand=0,
                days_infected=-1,
                income_quintile=income_to_quintile(income),
                cost_sensitivity=float(np.clip(self.rng.beta(2.0, 2.0), 0.0, 1.0)),
            )
            self.people.append(p)

        # Connect neighbors list from graph
        for i in range(N):
            self.people[i].network_neighbors = list(self.G.neighbors(i)) if self.G is not None else []

        # Initialize initial mask adoption randomly
        initial_maskers = self.rng.choice(N, size=int(initial_mask_rate * N), replace=False)
        for idx in initial_maskers:
            self.people[idx].wears_mask = True
            self.people[idx].mask_status = "cloth"
            self.people[idx].mask_supply_on_hand = self.rng.integers(1, 4)

        # Initialize infections
        initial_infected = self.rng.choice(N, size=min(N, int(self.params["initial_infected_count"])), replace=False)
        for idx in initial_infected:
            self.people[idx].health_status = "infected"
            self.people[idx].days_infected = 0

        # Single public location to represent public contexts
        self.locations = [
            Location(
                id=0,
                type="public",
                capacity=N,
                contact_rate=float(self.params["location_contact_rate"]),
                ventilation_level=float(self.params["ventilation_effect"]),
                mask_required=False,
                enforcement_level=float(self.params["location_enforcement_level"]),
            )
        ]

        # Retailer
        self.retailer = Retailer(
            id=0,
            inventory=int(self.params["retailer_initial_inventory"]),
            restock_rate_daily=int(self.params["restock_rate_daily"]),
            price_per_mask=dict(self.params["price_per_mask"]),
        )

        # Government
        self.government = Government(
            policy_state="none",
            mandate_start_day=int(self.params["policy_mandate_day"]),
            mandate_end_day=None,
            enforcement_strength=float(self.params["enforcement_strength"]),
            fine_amount=float(self.params["fine_amount"]),
            campaign_intensity=float(self.params["campaign_intensity"]),
            subsidy_per_mask=dict(self.params["subsidy_per_mask"]),
            policy_cost=0.0,
            fines_issued=0,
        )

        # Media
        self.media = Media(
            id=0,
            reach=float(self.params["media_reach"]),
            bias=float(self.params["media_bias"]),
            message_frequency=float(self.params["media_message_frequency"]),
            misinformation_rate=float(self.params["misinformation_rate"]),
        )

    def _find_household_id(self, person_id: int) -> int:
        """
        Find household id for a given person by index mapping used during setup.

        Args:
            person_id: Person index id.

        Returns:
            Household id containing the person.
        """
        pass
        for hh in self.households:
            if person_id in hh.member_ids:
                return hh.id
        # Should not happen if setup is correct
        return 0

    def _info_update(self) -> None:
        """
        Update risk perception and compliance propensity via media and government campaign.

        Returns:
            None
        """
        pass
        if self.media is None or self.government is None:
            return
        reach_mask = self.media.reach
        misinformation = self.media.misinformation_rate
        campaign = self.government.campaign_intensity
        update_rate = float(self.params["risk_perception_update_rate"])

        for p in self.people:
            # Government campaign increases risk perception for those reached
            if self.rng.uniform() < reach_mask:
                delta = campaign * (p.trust_in_government * 0.6 + p.trust_in_science * 0.4)
                p.risk_perception = float(np.clip(p.risk_perception + update_rate * delta, 0.0, 1.0))
                p.policy_awareness = float(np.clip(p.policy_awareness + 0.1 * campaign, 0.0, 1.0))
            # Misinformation reduces perceived risk, filtered by susceptibility
            if self.rng.uniform() < reach_mask:
                mis_delta = misinformation * (0.5 + self.media.bias * 0.2)
                susceptibility = p.misinformation_susceptibility
                p.risk_perception = float(np.clip(p.risk_perception - update_rate * mis_delta * susceptibility, 0.0, 1.0))

    def _retailer_restock(self) -> None:
        """
        Restock retailer inventory daily.

        Returns:
            None
        """
        pass
        if self.retailer is None:
            return
        self.retailer.inventory += int(self.retailer.restock_rate_daily)

    def _acquire_mask(self, person: Person) -> None:
        """
        Attempt to acquire a mask for a person based on inventory, price, subsidies, and income.

        Args:
            person: Person agent seeking to buy a mask.

        Returns:
            None
        """
        pass
        if self.retailer is None or self.government is None:
            return
        if self.retailer.inventory <= 0:
            return

        # Choose mask type based on income and cost sensitivity
        # Simple heuristic: lower quintiles prefer cheaper types
        mask_types = ["cloth", "surgical", "N95"]
        preferred = "cloth" if person.income_quintile < 2 else ("surgical" if person.income_quintile < 4 else "N95")

        # Price after subsidy
        price = self.retailer.price_per_mask.get(preferred, 1.0)
        subsidy = self.government.subsidy_per_mask.get(preferred, 0.0)
        effective_price = max(0.0, price - subsidy)

        # Affordability probability
        income_factor = np.log1p(max(0.0, person.income)) / 12.0
        affordability = sigmoid(2.0 * (income_factor - person.cost_sensitivity * effective_price))

        if self.rng.uniform() < affordability:
            # Complete purchase
            person.mask_supply_on_hand += 1
            person.mask_status = preferred
            self.retailer.inventory -= 1
            self.retailer.cumulative_sold += 1
            # Government subsidy cost accrues
            self.government.policy_cost += subsidy

    def _decide_wear_mask(self, person: Person, mandate_active: bool) -> None:
        """
        Decide whether a person wears a mask today using a logistic choice.

        Args:
            person: Person agent making decision.
            mandate_active: Whether a mask mandate is active.

        Returns:
            None
        """
        pass
        if person.mask_supply_on_hand <= 0:
            # Try to purchase if none on hand with some need-based urgency
            urgency = 0.4 + 0.4 * person.risk_perception
            if self.rng.uniform() < urgency:
                self._acquire_mask(person)
            if person.mask_supply_on_hand <= 0:
                person.wears_mask = False
                return

        # Inputs to utility
        neighbor_mask_rate = 0.0
        if person.network_neighbors:
            neighbor_mask_rate = float(np.mean([self.people[n].wears_mask for n in person.network_neighbors]))
        social_weight = float(self.params["social_influence_strength"]) * person.social_influence_weight
        policy_weight = float(self.params["policy_weight"]) * person.policy_awareness
        cost_weight = float(self.params["cost_weight"]) * person.cost_sensitivity
        fatigue_weight = float(self.params["fatigue_effect_weight"]) * person.fatigue
        base_pref = float(self.params["base_mask_preference"])

        # Policy signal: if mandate active, utility increases; else zero
        policy_signal = 1.0 if mandate_active else 0.0

        # Approximate cost via price minus subsidy for current mask type
        if self.government is not None and self.retailer is not None:
            price = self.retailer.price_per_mask.get(person.mask_status, 1.0)
            subsidy = self.government.subsidy_per_mask.get(person.mask_status, 0.0)
        else:
            price = 1.0
            subsidy = 0.0
        effective_price = max(0.0, price - subsidy)

        # Utility and stochastic decision
        temperature = float(self.params["decision_noise_temperature"])
        utility = (
            base_pref
            + 1.2 * person.risk_perception
            + social_weight * (2.0 * neighbor_mask_rate - 1.0)
            + policy_weight * policy_signal
            + 0.6 * person.compliance_propensity
            - cost_weight * effective_price
            - fatigue_weight
        )
        prob = sigmoid(utility / max(1e-6, temperature))
        person.wears_mask = (self.rng.uniform() < prob)

        # Consume mask stock occasionally by replacement interval
        if person.wears_mask and self.day % int(max(1, self.params["mask_replacement_interval_days"])) == 0:
            if person.mask_supply_on_hand > 0:
                person.mask_supply_on_hand -= 1

    def _policy_enforcement(self, person: Person, mandate_active: bool) -> None:
        """
        Enforce mandate through fines or denial.

        Args:
            person: Person agent.
            mandate_active: Whether a mask mandate is active.

        Returns:
            None
        """
        pass
        if not mandate_active or self.government is None:
            return
        # Location enforcement level proxy
        location = self.locations[0] if self.locations else None
        if location is None:
            return
        if not person.wears_mask:
            p_enforce = float(self.government.enforcement_strength) * float(location.enforcement_level)
            if self.rng.uniform() < p_enforce:
                # Fine issued
                self.government.fines_issued += 1
                # Assume enforcement has cost, 10% of fine amount
                self.government.policy_cost += 0.1 * float(self.government.fine_amount)
                # Behavioral feedback: increase compliance propensity slightly
                person.compliance_propensity = float(np.clip(person.compliance_propensity + 0.05, 0.0, 1.0))
                # In-the-moment compliance probability increases
                if self.rng.uniform() < 0.5:
                    person.wears_mask = True

    def _transmission_step(self) -> Tuple[int, int]:
        """
        Perform a simplified disease transmission and recovery step.

        Returns:
            Tuple of (new infections today, currently infectious count).
        """
        pass
        beta = float(self.params["beta_transmission_base"])
        mask_eff = self.params["mask_efficacy"]
        ventilation_effect = float(self.params["ventilation_effect"])
        infectious_period = int(self.params["infectious_period_days"])

        infected_indices = [p.id for p in self.people if p.health_status == "infected"]
        susceptible = set([p.id for p in self.people if p.health_status == "susceptible"])
        new_infections = 0

        # Transmission along edges from infected to susceptible neighbors
        for idx in infected_indices:
            i = self.people[idx]
            eff_i = mask_eff.get(i.mask_status, 0.0) if i.wears_mask else 0.0
            if not i.network_neighbors:
                continue
            for j_idx in i.network_neighbors:
                if j_idx not in susceptible:
                    continue
                j = self.people[j_idx]
                eff_j = mask_eff.get(j.mask_status, 0.0) if j.wears_mask else 0.0
                p_transmit = beta * (1.0 - eff_i) * (1.0 - eff_j) * (1.0 - ventilation_effect)
                if self.rng.uniform() < p_transmit:
                    # Infect j
                    self.people[j_idx].health_status = "infected"
                    self.people[j_idx].days_infected = 0
                    susceptible.remove(j_idx)
                    new_infections += 1

        # Update recovery
        infectious_count = 0
        for p in self.people:
            if p.health_status == "infected":
                p.days_infected += 1
                infectious_count += 1
                if p.days_infected >= infectious_period:
                    p.health_status = "recovered"
                    p.days_infected = -1

        return new_infections, infectious_count

    def _update_fatigue(self) -> None:
        """
        Increase behavioral fatigue slowly to model declining adherence.

        Returns:
            None
        """
        pass
        delta = float(self.params["fatigue_increase_rate"])
        for p in self.people:
            if p.wears_mask:
                p.fatigue = float(np.clip(p.fatigue + delta, 0.0, 1.0))
            else:
                p.fatigue = float(np.clip(p.fatigue - delta * 0.5, 0.0, 1.0))

    def step(self) -> None:
        """
        Execute a single simulation step.

        Returns:
            None
        """
        pass
        mandate_active = False
        if self.government is not None:
            mandate_active = (self.day >= int(self.params["policy_mandate_day"]))
            if mandate_active:
                # FIXED: Reflect mandate via location policy flag
                if self.locations:
                    self.locations[0].mask_required = bool(self.params["location_mask_required_under_mandate"])

        # Information update
        self._info_update()

        # Retailer restock
        before_inventory = self.retailer.inventory if self.retailer is not None else 0
        self._retailer_restock()
        after_inventory = self.retailer.inventory if self.retailer is not None else 0
        if before_inventory == 0 and after_inventory == 0 and self.retailer is not None:
            self.retailer.stockout_days += 1

        # Decisions and enforcement
        for p in self.people:
            self._decide_wear_mask(p, mandate_active)
            self._policy_enforcement(p, mandate_active)

        # Disease transmission
        new_infections, infectious_count = self._transmission_step()
        self.metrics["new_infections"].append(new_infections)

        # Estimate Rt as new infections over previous infectious_count (avoid div by zero)
        prev_infectious = infectious_count if infectious_count > 0 else 1
        Rt_est = new_infections / prev_infectious
        self.metrics["Rt_effective"].append(Rt_est)

        # Adoption
        adoption_rate = float(np.mean([1.0 if p.wears_mask else 0.0 for p in self.people]))
        self.metrics["adoption_rate_over_time"].append(adoption_rate)

        # Update fatigue
        self._update_fatigue()

        # Increment day
        self.day += 1

    def run(self) -> Dict[str, Any]:
        """
        Execute the full simulation loop.

        Returns:
            A metrics dictionary with required outputs and time series.
        """
        pass
        T = int(self.params["max_time_steps"])
        # FIXED: Fast smoke-test mode via environment variable
        if os.getenv("SIM_SMOKE_TEST") == "1":
            T = min(T, 10)

        for _ in range(T):
            self.step()

        return self._finalize_metrics()

    def _finalize_metrics(self) -> Dict[str, Any]:
        """
        Compile and finalize metrics for output.

        Returns:
            A dictionary suitable for JSON output.
        """
        pass
        adoption_ts = self.metrics["adoption_rate_over_time"]
        Rt_ts = self.metrics["Rt_effective"]
        new_inf_ts = self.metrics["new_infections"]
        T = len(adoption_ts)

        # Peak adoption and time to target
        peak_adoption = max(adoption_ts) if adoption_ts else 0.0
        target = float(self.params.get("target_adoption", 0.7))
        time_to_target = next((i for i, x in enumerate(adoption_ts) if x >= target), None)

        # Average compliance under mandate period
        mandate_day = int(self.params["policy_mandate_day"])
        avg_compliance_under_mandate = float(np.mean(adoption_ts[mandate_day:])) if T > mandate_day else float(np.mean(adoption_ts)) if T else 0.0

        # Cumulative masks sold and stockout frequency
        cumulative_masks_sold = int(self.retailer.cumulative_sold) if self.retailer is not None else 0
        stockout_frequency = float(self.retailer.stockout_days / T) if (self.retailer is not None and T > 0) else 0.0

        # Cumulative infections
        cumulative_infections = int(np.sum(new_inf_ts)) if new_inf_ts else 0

        # Policy costs and fines
        policy_cost = float(self.government.policy_cost) if self.government is not None else 0.0
        fines_issued = int(self.government.fines_issued) if self.government is not None else 0

        # Inequity index: compute adoption by income quintile, then Gini across quintile rates
        adoption_by_quintile = []
        for q in range(5):
            ids = [p.id for p in self.people if p.income_quintile == q]
            if ids:
                rate = float(np.mean([1.0 if self.people[i].wears_mask else 0.0 for i in ids]))
            else:
                rate = 0.0
            adoption_by

# Execute main for both direct execution and sandbox wrapper invocation
main()