def main():
    args = parse_args()

    # Load params
    params = default_params()
    if args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    params.update(loaded)
        except Exception as e:
            print(f"Error reading input JSON: {e}", file=sys.stderr)

    if args.population_size is not None:
        params["population_size"] = args.population_size

    days = args.days if args.days is not None else int(params.get("time_horizon_days", 180))
    seed = args.seed

    # Fast smoke run mode
    if args.fast_smoke_run:
        params["population_size"] = min(500, int(params.get("population_size", 10000)))
        days = min(10, days)

    sim = Simulation(params, seed=seed)
    results = sim.run(days)

    # Optional CSV
    if args.csv_out:
        sim.save_results(args.csv_out)

    # Validations
    if args.validate:
        which = [s.strip() for s in args.validate.split(",") if s.strip()]
        ok, messages = run_validations(sim, which)
        results["validations_ok"] = ok
        results["validation_messages"] = messages

    # Output JSON
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error writing output JSON: {e}", file=sys.stderr)
    else:
        print(json.dumps(results, indent=2))

    if args.fast_smoke_run:
        print("SMOKE_OK")

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Path handling constants (may be used when reading/writing data files)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp a number to the [lo, hi] range.

    Parameters
    ----------
    x : float
        Input value to clamp.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    float
        Clamped value.
    """
    pass
    return max(lo, min(hi, x))


def gini(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.

    Parameters
    ----------
    values : List[float]
        The list of values.

    Returns
    -------
    float
        Gini coefficient in [0,1], or 0 for empty or constant lists.
    """
    pass
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(values)
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_values, start=1):
        cumulative += v
        weighted_sum += i * v
    if cumulative == 0:
        return 0.0
    return (2 * weighted_sum) / (n * cumulative) - (n + 1) / n


def poisson_sample(lam: float, rng) -> int:
    """
    Sample a Poisson-distributed non-negative integer using Knuth's algorithm.

    Parameters
    ----------
    lam : float
        Poisson rate parameter lambda (>0).
    rng : random.Random
        RNG instance.

    Returns
    -------
    int
        Poisson random variate.
    """
    pass
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            break
    return max(0, k - 1)


@dataclass
class Household:
    """
    Household entity representing co-residing individuals and intra-household influence.

    Attributes
    ----------
    id : int
        Unique household identifier.
    member_ids : List[int]
        IDs of members belonging to this household.
    household_norm_mask_use : float
        Current normative mask use within the household [0,1].
    socioeconomic_status : float
        Placeholder indicator for SES of the household [0,1].
    intra_household_influence_strength : float
        Weight of intra-household influence [0,1].
    """
    id: int
    member_ids: List[int] = field(default_factory=list)
    household_norm_mask_use: float = 0.0
    socioeconomic_status: float = 0.5
    intra_household_influence_strength: float = 0.6

    def update_norm(self, adopted_prev: List[float]) -> None:
        """
        Update household normative mask use based on previous-day adoption of members.

        Parameters
        ----------
        adopted_prev : List[float]
            Binary adoption states for all agents from previous day.
        """
        pass
        if not self.member_ids:
            self.household_norm_mask_use = 0.0
            return
        vals = [adopted_prev[i] for i in self.member_ids]
        self.household_norm_mask_use = sum(vals) / max(1, len(vals))


@dataclass
class Person:
    """
    Person agent representing an individual in the simulation with attributes related to mask adoption behavior.

    Attributes
    ----------
    id : int
        Unique identifier.
    age : int
        Age of the person.
    income : float
        Income used for affordability decisions.
    household_id : int
        Household identifier.
    workplace_id : int
        Workplace identifier.
    network_neighbors : List[int]
        List of neighbor person IDs in the social network.
    trust_in_authority : float
        Trust in policy authority [0,1].
    susceptibility_to_peer_influence : float
        Sensitivity to peer influence [0,1].
    risk_perception : float
        Perceived risk in [0,1].
    perceived_mask_benefit : float
        Perceived mask benefit [0,1].
    perceived_mask_cost : float
        Perceived mask cost/inconvenience [0,1].
    mask_inventory : int
        Number of masks held.
    mask_adopted : bool
        Whether the person intends to wear a mask for the day (baseline).
    current_mask_use : bool
        Transient state: for compliance at locations on this day.
    habit_strength : float
        Habit of wearing masks [0,1].
    compliance_propensity : float
        Tendency to comply with rules in [0,1].
    education_level : int
        Proxy for subgroup analysis (0,1,2).
    exposure_to_misinformation : float
        Susceptibility to misinformation [0,1].
    days_worn : int
        Cumulative days the person wore a mask (adopted or complied).

    Methods
    -------
    reset_daily_state()
        Reset transient states for the new day.
    perceive_risk(...)
        Update risk perception using peer/policy/media and prevalence signals.
    update_attitude(...)
        Update perceived benefits/costs and habit strength.
    decide_adoption(...)
        Decide whether to adopt mask wearing for the day (consumes inventory if adopted).
    purchase_masks(...)
        Decide whether and how many masks to purchase based on affordability and mandate.
    comply_with_policy(...)
        Decide compliance upon entry; may consume inventory for temporary use.
    """
    id: int
    age: int
    income: float
    household_id: int
    workplace_id: int
    network_neighbors: List[int] = field(default_factory=list)
    trust_in_authority: float = 0.6
    susceptibility_to_peer_influence: float = 0.5
    risk_perception: float = 0.0
    perceived_mask_benefit: float = 0.4
    perceived_mask_cost: float = 0.2
    mask_inventory: int = 0
    mask_adopted: bool = False
    current_mask_use: bool = False
    habit_strength: float = 0.0
    compliance_propensity: float = 0.5
    education_level: int = 1
    exposure_to_misinformation: float = 0.2
    days_worn: int = 0

    def reset_daily_state(self) -> None:
        """
        Reset transient state for the day.
        """
        pass
        self.current_mask_use = False

    def perceive_risk(
        self,
        peer_share: float,
        policy_signal: float,
        media_signal: float,
        prevalence_signal: float,
        risk_perception_sensitivity_to_prevalence: float,
        external_prevalence_signal: float,
        w_peer: float,
        w_policy: float,
        w_media: float,
        household_share: float = 0.0,
        w_household: float = 0.6,
    ) -> None:
        """
        Update risk perception using weighted signals and prevalence sensitivity.

        Parameters
        ----------
        peer_share : float
            Observed share of peers adopting [0,1].
        policy_signal : float
            Strength of policy guidance [0,1].
        media_signal : float
            Aggregated media signal in [-1,1], where positive supports adoption.
        prevalence_signal : float
            Endogenous prevalence of mask use observed externally [0,1].
        risk_perception_sensitivity_to_prevalence : float
            Sensitivity to prevalence for updating risk [0,1].
        external_prevalence_signal : float
            Exogenous prevalence-like signal [0,1].
        w_peer : float
            Weight for peer signal.
        w_policy : float
            Weight for policy signal.
        w_media : float
            Weight for media signal.
        household_share : float
            Observed share of household members adopting [0,1].
        w_household : float
            Weight for household signal.
        """
        pass
        media_component = 0.5 * (media_signal + 1.0)
        # FIXED: Integrate prevalence sensitivity as per feedback
        prevalence_component = clamp(
            risk_perception_sensitivity_to_prevalence * (0.5 * prevalence_signal + 0.5 * external_prevalence_signal)
        )
        signal = (
            w_peer * peer_share
            + w_household * household_share
            + w_policy * policy_signal * self.trust_in_authority
            + w_media * media_component * (1.0 - self.exposure_to_misinformation)
            + prevalence_component
        )
        signal = clamp(signal, 0.0, 1.0)
        inertia = 0.7
        self.risk_perception = clamp(inertia * self.risk_perception + (1 - inertia) * signal)

    def update_attitude(
        self,
        habit_formation_rate: float,
        compliance_decay_rate: float,
        mask_effectiveness_perceived: float,
    ) -> None:
        """
        Update perceived benefits/costs and habit strength.

        Parameters
        ----------
        habit_formation_rate : float
            Daily increment to habit strength when wearing.
        compliance_decay_rate : float
            Daily decay of habit/compliance when not wearing.
        mask_effectiveness_perceived : float
            Perceived mask effectiveness multiplier [0,1].
        """
        pass
        # Benefits increase with risk and effectiveness; costs decrease slightly with habit
        self.perceived_mask_benefit = clamp(
            0.4 * self.perceived_mask_benefit + 0.6 * clamp(self.risk_perception * mask_effectiveness_perceived)
        )
        self.perceived_mask_cost = clamp(
            0.7 * self.perceived_mask_cost + 0.3 * (1.0 - self.habit_strength)
        )
        # Habit update happens in decide_adoption based on actual wear

    def decide_adoption(
        self,
        price: float,
        policy_active: bool,
        rng,
        enforcement_level: float = 0.0,
        penalty_amount: float = 0.0,
        habit_formation_rate: float = 0.02,
        compliance_decay_rate: float = 0.01,
    ) -> bool:
        """
        Decide whether to wear a mask today.

        Parameters
        ----------
        price : float
            Current mask price.
        policy_active : bool
            Whether a mask mandate is active.
        rng : random.Random
            RNG instance.
        enforcement_level : float
            Current enforcement intensity [0,1].
        penalty_amount : float
            Penalty amount; increases policy pressure.
        habit_formation_rate : float
            Increment for habit when wearing.
        compliance_decay_rate : float
            Decay for habit when not wearing.

        Returns
        -------
        bool
            True if adopting (intends to wear for the day), otherwise False.
        """
        pass
        # FIXED: Enhance adoption decision policy term with enforcement and penalties
        policy_pressure = clamp(enforcement_level * self.compliance_propensity * min(1.0, penalty_amount / 100.0))
        peer_term_proxy = 0.0  # Already included in risk; keep simple here
        benefit_term = self.perceived_mask_benefit
        cost_term = self.perceived_mask_cost + min(0.1, price / 20.0)
        habit_term = self.habit_strength
        policy_term = (0.5 if policy_active else 0.0) * policy_pressure
        linear_util = peer_term_proxy + benefit_term - cost_term + habit_term + policy_term
        p_wear = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, linear_util))))
        will_wear = rng.random() < p_wear
        # Consume one mask if adopting and inventory available
        if will_wear and self.mask_inventory > 0:
            self.mask_adopted = True
            self.mask_inventory -= 1
            # FIXED: Use habit parameters correctly
            self.habit_strength = clamp(self.habit_strength + habit_formation_rate, 0.0, 1.0)
            return True
        else:
            self.mask_adopted = False
            # FIXED: Use compliance_decay_rate directly
            self.habit_strength = clamp(self.habit_strength * (1.0 - compliance_decay_rate), 0.0, 1.0)
            return False

    def purchase_masks(
        self,
        rng,
        price: float,
        bundle: int,
        subsidy_rate: float,
        mandate_active: bool,
        procurement_access_fraction: float,
    ) -> int:
        """
        Decide how many masks to purchase given affordability, subsidy, and access.

        Parameters
        ----------
        rng : random.Random
            RNG.
        price : float
            Base unit price.
        bundle : int
            Default purchase bundle size.
        subsidy_rate : float
            Subsidy fraction [0,1].
        mandate_active : bool
            Whether a mandate is in force (increases purchase likelihood).
        procurement_access_fraction : float
            Probability of access to procurement channels.

        Returns
        -------
        int
            Desired quantity to buy (subject to supply availability).
        """
        pass
        if rng.random() > procurement_access_fraction:
            return 0
        effective_price = max(0.0, price * (1.0 - subsidy_rate))
        affordability = self.income / (self.income + 10.0 * effective_price + 1e-6)
        intent = 0.3 + 0.5 * affordability + (0.2 if mandate_active else 0.0)
        intent = clamp(intent)
        if self.mask_inventory > 0 and not mandate_active and rng.random() > 0.25:
            return 0
        if rng.random() < intent:
            need = 1 if self.mask_inventory == 0 else 0
            qty = max(need, bundle if affordability > 0.6 else 1)
            return qty
        return 0

    def comply_with_policy(self, enforcement_prob: float, signage_strength: float, rng) -> bool:
        """
        Determine if the person complies with a mask requirement at a location.

        Parameters
        ----------
        enforcement_prob : float
            Effective enforcement probability [0,1].
        signage_strength : float
            Salience of signage prompting compliance [0,1].
        rng : random.Random
            RNG instance.

        Returns
        -------
        bool
            True if the person attempts to comply.
        """
        pass
        base = clamp(0.5 * self.compliance_propensity + 0.3 * self.trust_in_authority + 0.2 * signage_strength)
        adjusted = clamp(base + 0.4 * enforcement_prob * self.trust_in_authority)
        return rng.random() < adjusted


@dataclass
class Location:
    """
    Location where interactions and policy enforcement may occur.

    Attributes
    ----------
    id : int
        Identifier for the location.
    type : str
        Type of location (e.g., 'retail', 'work', 'public').
    capacity : int
        Maximum people that can be present.
    policy_requires_mask : bool
        Whether masks are required.
    enforcement_strictness : float
        Baseline enforcement probability [0,1].
    signage_strength : float
        Effectiveness of signage prompting compliance [0,1].
    foot_traffic_rate : float
        Probability an individual visits per day [0,1].
    policy_eligible : bool
        Whether this location is covered by venue policy when mandate active.

    Methods
    -------
    enforce_mask_policy(person, agency_enforcement, rng)
        Simulate enforcement interaction for a visiting person; consumes inventory on compliance.
    """
    id: int
    type: str
    capacity: int
    policy_requires_mask: bool
    enforcement_strictness: float
    signage_strength: float
    foot_traffic_rate: float
    policy_eligible: bool = True

    def enforce_mask_policy(self, person: Person, agency_enforcement: float, rng) -> Tuple[bool, bool]:
        """
        Enforce mask policy with certain probability.

        Parameters
        ----------
        person : Person
            The visiting person.
        agency_enforcement : float
            Additional enforcement scaling from the policy authority [0,1].
        rng : random.Random
            RNG for stochastic checks.

        Returns
        -------
        Tuple[bool, bool]
            (incident_occurred, compliant_now)
        """
        pass
        if not self.policy_requires_mask:
            return (False, person.mask_adopted or person.current_mask_use)

        # Already wearing a mask (adopted) or already complied
        if person.mask_adopted or person.current_mask_use:
            return (False, True)

        # Decide to comply at entry
        will_comply = person.comply_with_policy(
            enforcement_prob=clamp(self.enforcement_strictness * agency_enforcement),
            signage_strength=self.signage_strength,
            rng=rng,
        )

        incident = False
        if will_comply:
            # FIXED: Consume inventory upon compliance for temporary use per feedback
            if person.mask_inventory > 0:
                person.mask_inventory -= 1
                person.current_mask_use = True
                return (False, True)
            else:
                # Cannot comply due to no inventory; treated as noncompliance
                will_comply = False

        if not will_comply:
            check_prob = clamp(self.enforcement_strictness * agency_enforcement)
            if rng.random() < check_prob:
                incident = True

        return (incident, False)


@dataclass
class PolicyAuthority:
    """
    Policy authority controlling mandates, enforcement, and communications.

    Attributes
    ----------
    id : int
        Identifier.
    mandate_enabled : bool
        Whether mandates are used.
    mandate_start_day : int
        Start day for mandates.
    mandate_end_day : int
        End day for mandates.
    penalty_amount : float
        Penalty amount for non-compliance (informational).
    incentive_amount : float
        Incentive amount for compliance (informational).
    enforcement_level : float
        Baseline enforcement level [0,1].
    communication_frequency : float
        Probability of issuing campaign communications per day [0,1].
    message_strategy : float
        Strength of pro-mask messaging [0,1].
    subsidy_rate : float
        Subsidy fraction for mask price [0,1].
    enforcement_capacity_per_day : int
        Maximum number of enforcement actions (incidents) that can be recorded per day.
    """
    id: int
    mandate_enabled: bool
    mandate_start_day: int
    mandate_end_day: int
    penalty_amount: float
    incentive_amount: float
    enforcement_level: float
    communication_frequency: float
    message_strategy: float
    subsidy_rate: float
    enforcement_capacity_per_day: int = 0  # FIXED: Added capacity attribute

    def issue_mandates(self, day: int) -> bool:
        """
        Determine if mandates are active on the given day.

        Parameters
        ----------
        day : int
            Day index.

        Returns
        -------
        bool
            True if mandate is active.
        """
        pass
        # FIXED: Implement mandate window per feedback
        if not self.mandate_enabled:
            return False
        return self.mandate_start_day <= day <= self.mandate_end_day

    def run_information_campaign(self, rng) -> float:
        """
        Run a campaign information broadcast.

        Parameters
        ----------
        rng : random.Random
            RNG.

        Returns
        -------
        float
            Policy guidance signal [0,1].
        """
        pass
        if rng.random() < clamp(self.communication_frequency):
            return clamp(self.message_strategy)
        return 0.0

    def adjust_enforcement(self, day: int) -> float:
        """
        Adjust enforcement level (e.g., stronger during mandates).

        Parameters
        ----------
        day : int
            Day index.

        Returns
        -------
        float
            Effective enforcement probability scaling [0,1].
        """
        pass
        # FIXED: Increase enforcement during mandate period
        return clamp(self.enforcement_level * (1.2 if self.issue_mandates(day) else 1.0))


@dataclass
class SupplyChain:
    """
    Supply chain for mask production, distribution, and pricing.

    Attributes
    ----------
    total_stock : int
        Current total stock.
    production_rate_per_day : int
        Units produced per day.
    distribution_delay_days : int
        Days of pipeline delay.
    price_per_mask : float
        Current price per mask.
    rationing_policy : str
        Rationing mode ("price" or "first_come").
    min_price : float
        Minimum price bound.
    max_price : float
        Maximum price bound.
    cumulative_produced : int
        Cumulative units shipped into stock from production pipeline.
    cumulative_distributed : int
        Cumulative units distributed to consumers.

    Methods
    -------
    produce_masks()
        Move production through pipeline into stock.
    distribute_masks(demand)
        Fulfill demand up to available stock.
    adjust_prices(stockout)
        Adjust prices depending on stockout status.
    """
    total_stock: int
    production_rate_per_day: int
    distribution_delay_days: int
    price_per_mask: float
    rationing_policy: str = "price"
    min_price: float = 0.5
    max_price: float = 50.0
    _pipeline: List[int] = field(default_factory=list)
    cumulative_produced: int = 0
    cumulative_distributed: int = 0

    def __post_init__(self) -> None:
        """
        Initialize the distribution pipeline.
        """
        pass
        self._pipeline = [0] * max(0, int(self.distribution_delay_days))

    def produce_masks(self) -> None:
        """
        Produce masks and progress them through the distribution pipeline.
        """
        pass
        self._pipeline.append(int(self.production_rate_per_day))
        shipped = self._pipeline.pop(0) if self._pipeline else int(self.production_rate_per_day)
        self.total_stock += shipped
        # FIXED: Track cumulative production for validation
        self.cumulative_produced += shipped

    def distribute_masks(self, demand: int) -> int:
        """
        Distribute masks to meet demand, subject to stock.

        Parameters
        ----------
        demand : int
            Requested quantity.

        Returns
        -------
        int
            Quantity actually distributed.
        """
        pass
        sold = min(int(demand), self.total_stock)
        self.total_stock -= sold
        # FIXED: Track cumulative distribution for validation
        self.cumulative_distributed += sold
        return sold

    def adjust_prices(self, stockout: bool) -> None:
        """
        Adjust price based on stockout status under rationing.

        Parameters
        ----------
        stockout : bool
            True if stockout occurred.
        """
        pass
        # FIXED: Implement price adjustment per feedback
        if self.rationing_policy == "price":
            if stockout:
                self.price_per_mask = clamp(self.price_per_mask * 1.1, self.min_price, self.max_price)
            else:
                self.price_per_mask = clamp(self.price_per_mask * 0.98, self.min_price, self.max_price)


@dataclass
class MediaChannel:
    """
    Media channel broadcasting messages that can support or undermine mask adoption.

    Attributes
    ----------
    id : int
        Identifier.
    reach : float
        Audience reach in [0,1].
    message_frequency : float
        Daily probability of broadcasting a message [0,1].
    bias : float
        Bias direction in [-1,1], where positive supports adoption.
    misinformation_probability : float
        Probability that a message is misinformation (flips sign).

    Methods
    -------
    broadcast_message(rng)
        Produce a message signal in [-1,1] with frequency and misinformation.
    """
    id: int
    reach: float
    message_frequency: float
    bias: float
    misinformation_probability: float

    def broadcast_message(self, rng) -> float:
        """
        Broadcast a message according to frequency and misinformation probability.

        Parameters
        ----------
        rng : random.Random
            RNG for stochastic decisions.

        Returns
        -------
        float
            Message signal in [-1,1], or 0 if no message broadcast today.
        """
        pass
        if rng.random() >= self.message_frequency:
            return 0.0
        sign = self.bias
        if rng.random() < self.misinformation_probability:
            sign = -sign
        # Scale by reach
        return clamp(sign, -1.0, 1.0) * clamp(self.reach, 0.0, 1.0)


class Simulation:
    """
    Main simulation orchestrator for mask adoption dynamics.

    Parameters
    ----------
    params : Dict[str, float]
        Configuration parameters for the run.
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    p : Dict[str, float]
        Parameters.
    rng : random.Random
        Deterministic RNG instance.
    people : List[Person]
        Agent population.
    households : List[Household]
        Household entities.
    locations : List[Location]
        List of non-work and public locations.
    workplaces : List[Location]
        List of workplace locations.
    supply_chain : SupplyChain
        Supply chain handling production, distribution, and pricing.
    policy : PolicyAuthority
        Policy authority.
    media : List[MediaChannel]
        Media channels.
    series : Dict[str, List[float]]
        Time series metrics: adoption_rate, price, etc.
    daily_counters : Dict[str, List[float]]
        Additional per-day counters (visits, incidents, compliance).
    cumulative_acquired : int
        Total masks acquired by people (purchases) for validation.
    initial_total_stock : int
        Initial total stock used for mass-balance validation.
    """
    def __init__(self, params: Dict[str, float], seed: int = 42):
        """
        Initialize the simulation with parameters and seed.

        Parameters
        ----------
        params : Dict[str, float]
            Parameters for the simulation.
        seed : int
            Random seed.
        """
        pass
        self.p = dict(params)
        # FIXED: Parameter alias mapping for spec conformance
        if "fine_amount" in self.p and "penalty_amount" not in self.p:
            self.p["penalty_amount"] = self.p["fine_amount"]
        if "mask_cost" in self.p and "mask_price" not in self.p:
            self.p["mask_price"] = self.p["mask_cost"]
        if "trust_in_authority_mean" in self.p and "trust_in_authorities_mean" not in self.p:
            self.p["trust_in_authorities_mean"] = self.p["trust_in_authority_mean"]
        if "household_size_distribution" in self.p and isinstance(self.p["household_size_distribution"], dict):
            if "lambda" in self.p["household_size_distribution"]:
                self.p["household_size_lambda"] = self.p["household_size_distribution"]["lambda"]
        if "location_policy_mask_required_fraction" in self.p and "venue_policy_coverage" not in self.p:
            self.p["venue_policy_coverage"] = self.p["location_policy_mask_required_fraction"]

        self.rng = __import__("random").Random(int(seed))

        # Entities
        self.people: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []
        self.workplaces: List[Location] = []

        # FIXED: Introduced SupplyChain per feedback and replaced Retailer
        initial_stock_guess = int(
            self.p.get("population_size", 10000)
            * self.p.get("mask_supply_per_capita", 5.0)
            * self.p.get("supplier_initial_inventory_ratio", 1.0)
        )
        self.initial_total_stock = int(self.p.get("initial_total_stock", initial_stock_guess))  # FIXED: Track initial
        self.supply_chain = SupplyChain(
            total_stock=self.initial_total_stock,
            production_rate_per_day=int(self.p.get("production_rate_per_day", 500)),
            distribution_delay_days=int(self.p.get("distribution_delay_days", 2)),
            price_per_mask=float(self.p.get("mask_price", 2.0)),
            rationing_policy=str(self.p.get("rationing_policy", "price")),
            min_price=float(self.p.get("min_mask_price", 0.5)),
            max_price=float(self.p.get("max_mask_price", 50.0)),
        )

        # FIXED: Replaced GovernmentAgency with PolicyAuthority aligned to spec/feedback
        self.policy = PolicyAuthority(
            id=1,
            mandate_enabled=bool(self.p.get("mandate_enabled", False)),
            mandate_start_day=int(self.p.get("mandate_start_day", 30)),
            mandate_end_day=int(self.p.get("mandate_end_day", 120)),
            penalty_amount=float(self.p.get("penalty_amount", 50.0)),
            incentive_amount=float(self.p.get("incentive_amount", 0.0)),
            enforcement_level=float(self.p.get("enforcement_level", 0.5)),
            communication_frequency=float(self.p.get("communication_frequency", 0.5)),
            message_strategy=float(self.p.get("message_strategy", 0.6)),
            subsidy_rate=float(self.p.get("subsidy_rate", 0.0)),
            enforcement_capacity_per_day=int(self.p.get("enforcement_capacity_per_day", 0)),  # FIXED: init capacity
        )

        self.media: List[MediaChannel] = [
            MediaChannel(
                id=1,
                reach=float(self.p.get("media_reach_main", 0.7)),
                message_frequency=float(self.p.get("message_frequency_per_week", 3)) / 7.0,
                bias=float(self.p.get("media_bias", 0.0)),
                misinformation_probability=float(self.p.get("misinformation_rate", 0.05)),
            )
        ]

        # Series and counters
        self.series: Dict[str, List[float]] = {
            "adoption_rate": [],
            "average_price": [],
            "supplier_inventory": [],  # FIXED: renamed; keep alias below
            "retailer_inventory": [],  # legacy alias for backward compatibility
            "enforcement_incidents_per_1000": [],
            "compliance_rate": [],
            "compliance_rate_under_mandate": [],
            "policy_enforcement_level": [],
            "daily_new_adopters": [],
            "adoption_work": [],
            "adoption_public": [],
            # FIXED: Track demand vs sold for shortage metrics
            "daily_demand": [],
            "daily_sold": [],
        }
        self.daily_counters: Dict[str, List[float]] = {
            "visits_public": [],
            "incidents_public": [],
            "compliant_public": [],
            "visits_work": [],
            "incidents_work": [],
            "compliant_work": [],
        }

        # Accumulators for validation and metrics
        self.cumulative_acquired: int = 0

        # Observed series for RMSE computation if provided
        self.observed_adoption_series: List[float] = []
        if isinstance(self.p.get("observed_adoption_series", []), list):
            self.observed_adoption_series = list(self.p.get("observed_adoption_series", []))

    def _small_world(self, N: int, k: int, beta: float) -> List[List[int]]:
        """
        Create a Watts–Strogatz small-world network adjacency list.

        Parameters
        ----------
        N : int
            Number of nodes.
        k : int
            Each node connected to k nearest neighbors (k even preferred).
        beta : float
            Rewiring probability.

        Returns
        -------
        List[List[int]]
            Adjacency list of neighbors for each node.
        """
        pass
        # Ring lattice
        k = max(2, k)
        if k % 2 == 1:
            k += 1
        half = k // 2
        adj = [set() for _ in range(N)]
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                adj[i].add(j)
                adj[j].add(i)
        # FIXED: Optimize rewiring to avoid O(N) candidate list per rewire
        for i in range(N):
            for d in range(1, half + 1):
                j = (i + d) % N
                if self.rng.random() < beta:
                    attempts = 0
                    k2 = j
                    while attempts < 10:
                        cand = self.rng.randrange(N)
                        if cand != i and cand not in adj[i]:
                            k2 = cand
                            break
                        attempts += 1
                    if k2 != j:
                        adj[i].discard(j)
                        adj[j].discard(i)
                        adj[i].add(k2)
                        adj[k2].add(i)
        return [list(nei) for nei in adj]

    def _build_households_poisson(self, N: int, lam: float) -> List[Household]:
        """
        Build households by sampling household sizes from a Poisson distribution until N agents are allocated.

        Parameters
        ----------
        N : int
            Population size.
        lam : float
            Poisson lambda for household size.

        Returns
        -------
        List[Household]
            Constructed households with member id ranges.
        """
        pass
        sizes: List[int] = []
        remaining = N
        while remaining > 0:
            sz = max(1, poisson_sample(lam, self.rng))
            if sz > remaining:
                sz = remaining
            sizes.append(sz)
            remaining -= sz
        households: List[Household] = []
        start = 0
        for hid, sz in enumerate(sizes):
            member_ids = list(range(start, start + sz))
            households.append(
                Household(
                    id=hid,
                    member_ids=member_ids,
                    household_norm_mask_use=0.0,
                    socioeconomic_status=clamp(self.rng.random()),
                    intra_household_influence_strength=float(self.p.get("household_influence_weight", 0.6)),
                )
            )
            start += sz
        return households

    def initialize(self) -> None:
        """
        Initialize population, households, network, locations, and supply chain.
        """
        pass
        N = int(self.p.get("population_size", 10000))
        init_rate = float(self.p.get("initial_adoption_rate", 0.1))
        avg_deg = int(self.p.get("avg_degree", 10))
        risk_init = float(self.p.get("risk_level", 0.2))

        # Households built via Poisson size distribution
        lam = float(self.p.get("household_size_lambda", 3.0))
        self.households = self._build_households_poisson(N, lam)

        # People
        self.people = [None] * N  # type: ignore
        # Assign workplace ids
        num_workplaces = max(1, int(self.p.get("num_workplaces", 50)))
        for i in range(N):
            # Log-normal income proxy
            mu = float(self.p.get("income_lognorm_mu", 3.0))
            sigma = float(self.p.get("income_lognorm_sigma", 0.5))
            income = math.exp(self.rng.normalvariate(mu, sigma))
            adopted = self.rng.random() < init_rate
            education_level = self.rng.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
            person = Person(
                id=i,
                age=self.rng.randint(18, 85),
                income=income,
                household_id=0,  # temporary, set later
                workplace_id=i % num_workplaces,
                trust_in_authority=clamp(
                    self.rng.normalvariate(self.p.get("trust_in_authorities_mean", 0.5),
                                           self.p.get("trust_in_authorities_std", 0.2))
                ),
                susceptibility_to_peer_influence=clamp(self.rng.random()),
                risk_perception=clamp(risk_init + self.rng.uniform(-0.05, 0.05)),
                perceived_mask_benefit=float(self.p.get("perceived_benefit_base", 0.4)),
                perceived_mask_cost=float(self.p.get("perceived_cost_base", 0.2)),
                mask_inventory=(1 if adopted else 0),
                mask_adopted=adopted,
                habit_strength=clamp(0.4 + 0.2 * self.rng.random()) if adopted else clamp(0.1 * self.rng.random()),
                compliance_propensity=clamp(self.rng.uniform(0.3, 0.9)),
                education_level=education_level,
                exposure_to_misinformation=clamp(self.rng.uniform(0.1, 0.6)),
            )
            self.people[i] = person

        # Assign household IDs to persons
        for hh in self.households:
            for pid in hh.member_ids:
                if 0 <= pid < N:
                    self.people[pid].household_id = hh.id

        # Social network
        neighbors = self._small_world(N, avg_deg, beta=float(self.p.get("social_network_rewiring_p", 0.05)))
        for i, p in enumerate(self.people):
            p.network_neighbors = neighbors[i]

        # Locations: create workplaces and public venues; apply venue_policy_coverage
        venue_policy_coverage = float(self.p.get("venue_policy_coverage", 0.6))
        self.workplaces = []
        for wid in range(num_workplaces):
            eligible = self.rng.random() < venue_policy_coverage
            self.workplaces.append(
                Location(
                    id=wid,
                    type="work",
                    capacity=int(self.p.get("workplace_capacity_mean", 200)),
                    policy_requires_mask=False,
                    enforcement_strictness=float(self.p.get("workplace_enforcement_strictness", 0.5)),
                    signage_strength=float(self.p.get("workplace_signage_effect", self.p.get("signage_effect", 0.05))),
                    foot_traffic_rate=float(self.p.get("workplace_attendance_rate", 0.5)),
                    policy_eligible=eligible,
                )
            )
        # Public locations
        public_venues_count = max(1, int(self.p.get("public_venues_count", 5)))
        self.locations = []
        for lid in range(public_venues_count):
            eligible = self.rng.random() < venue_policy_coverage
            self.locations.append(
                Location(
                    id=lid,
                    type="public",
                    capacity=int(self.p.get("location_capacity_mean", 2000)),
                    policy_requires_mask=False,
                    enforcement_strictness=float(self.p.get("location_enforcement_strictness_mean", 0.5)),
                    signage_strength=float(self.p.get("signage_effect", 0.05)),
                    foot_traffic_rate=float(self.p.get("public_venue_visit_rate", 0.3)),
                    policy_eligible=eligible,
                )
            )

    def _aggregate_media_signal(self) -> float:
        """
        Aggregate media messages into a single signal in [-1,1].

        Returns
        -------
        float
            Aggregated media signal.
        """
        pass
        # FIXED: Simplified and optimized aggregation
        total = sum(ch.broadcast_message(self.rng) for ch in self.media)
        return clamp(total, -1.0, 1.0)

    def _peer_share(self, adopted_prev: List[float], neighbors: List[int], contact_rate_per_day: int) -> float:
        """
        Compute the share of peers adopting, based on sampled contacts.

        Parameters
        ----------
        adopted_prev : List[float]
            Binary list of previous adoption states.
        neighbors : List[int]
            Neighbor indices.
        contact_rate_per_day : int
            Number of contacts sampled per day.

        Returns
        -------
        float
            Peer adoption rate in [0,1].
        """
        pass
        if not neighbors:
            return 0.0
        # Sample up to contact_rate_per_day neighbors
        k = min(contact_rate_per_day, len(neighbors))
        if k <= 0:
            return 0.0
        # Randomly sample without replacement
        idxs = set()
        while len(idxs) < k:
            idxs.add(neighbors[self.rng.randrange(len(neighbors))])
        vals = [adopted_prev[j] for j in idxs]
        return sum(vals) / max(1, len(vals))

    def _adaptive_policy_adjustment(self, day: int) -> None:
        """
        Adaptively adjust policy based on recent adoption and compliance.

        Parameters
        ----------
        day : int
            Current day index.
        """
        pass
        if day < 7:
            return
        recent_adoption = statistics.mean(self.series["adoption_rate"][-7:]) if self.series["adoption_rate"] else 0.0
        recent_compliance = statistics.mean(self.series["compliance_rate"][-7:]) if self.series["compliance_rate"] else 0.0
        target_adoption = float(self.p.get("adoption_target_recent", 0.6))
        target_compliance = float(self.p.get("compliance_target_recent", 0.7))

        # FIXED: Add policy adjustment routine based on moving average thresholds
        if (recent_adoption < target_adoption or recent_compliance < target_compliance) and self.policy.mandate_enabled:
            # Increase enforcement and extend mandates modestly
            self.policy.enforcement_level = clamp(self.policy.enforcement_level + 0.05)
            if day > self.policy.mandate_end_day - 7:
                self.policy.mandate_end_day += 14
        elif recent_adoption > target_adoption + 0.1 and recent_compliance > target_compliance + 0.1:
            # Taper enforcement slightly
            self.policy.enforcement_level = clamp(self.policy.enforcement_level - 0.02)

    def step(self, day: int) -> None:
        """
        Execute one simulation day: update perceptions, attitudes, decisions, purchases, visits, enforcement, and metrics.

        Parameters
        ----------
        day : int
            Day index.
        """
        pass
        # Reset transient states
        for person in self.people:
            person.reset_daily_state()

        # Policy and media
        mandate_active = self.policy.issue_mandates(day)
        agency_enforcement = self.policy.adjust_enforcement(day)
        policy_signal = self.policy.run_information_campaign(self.rng)
        media_signal = self._aggregate_media_signal()

        # Apply mandate to locations with coverage
        for loc in self.locations:
            loc.policy_requires_mask = mandate_active and loc.policy_eligible
        for wloc in self.workplaces:
            wloc.policy_requires_mask = mandate_active and wloc.policy_eligible

        # Weights and sensitivities
        w_peer = float(self.p.get("social_influence_weight", 0.4))
        w_policy = float(self.p.get("policy_influence_weight", 0.3))
        w_media = float(self.p.get("media_influence_weight", 0.2))
        w_household = float(self.p.get("household_influence_weight", 0.6))
        habit_formation_rate = float(self.p.get("habit_formation_rate", 0.02))
        compliance_decay_rate = float(self.p.get("compliance_decay_rate", 0.01))
        mask_effectiveness_perceived = float(self.p.get("mask_effectiveness_perceived", 0.5))
        contact_rate_per_day = int(self.p.get("contact_rate_per_day", 10))
        risk_perc_sens_prev = float(self.p.get("risk_perception_sensitivity_to_prevalence", 0.6))
        external_prev_signal = float(self.p.get("external_prevalence_signal", 0.1))
        procurement_access_fraction = float(self.p.get("procurement_access_fraction", 0.9))
        penalty_amount = float(self.p.get("penalty_amount", 50.0))

        # Precompute peer states
        adopted_prev_flags = [1.0 if p.mask_adopted else 0.0 for p in self.people]
        endogenous_prevalence = sum(adopted_prev_flags) / max(1, len(self.people))

        # Update household norms based on previous day
        for hh in self.households:
            hh.update_norm(adopted_prev_flags)

        # Supply production
        self.supply_chain.produce_masks()

        # Person-level updates
        daily_demand = 0
        daily_sold = 0  # FIXED: Track sold for shortage days
        prev_any_use_flags = [1 if (p.mask_adopted or p.current_mask_use) else 0 for p in self.people]

        for i, person in enumerate(self.people):
            peer_share = self._peer_share(adopted_prev_flags, person.network_neighbors, contact_rate_per_day)
            # Household share from household norm
            household_share = 0.0
            if 0 <= person.household_id < len(self.households):
                household_share = self.households[person.household_id].household_norm_mask_use

            person.perceive_risk(
                peer_share=peer_share,
                policy_signal=policy_signal,
                media_signal=media_signal,
                prevalence_signal=endogenous_prevalence,
                risk_perception_sensitivity_to_prevalence=risk_perc_sens_prev,
                external_prevalence_signal=external_prev_signal,
                w_peer=w_peer,
                w_policy=w_policy,
                w_media=w_media,
                household_share=household_share,
                w_household=w_household,
            )
            person.update_attitude(
                habit_formation_rate=habit_formation_rate,
                compliance_decay_rate=compliance_decay_rate,
                mask_effectiveness_perceived=mask_effectiveness_perceived,
            )

            # Purchase if needed or under mandate pressure
            if person.mask_inventory <= 0 or mandate_active:
                desired = person.purchase_masks(
                    self.rng,
                    price=self.supply_chain.price_per_mask,
                    bundle=int(self.p.get("purchase_bundle", 5)),
                    subsidy_rate=self.policy.subsidy_rate,
                    mandate_active=mandate_active,
                    procurement_access_fraction=procurement_access_fraction,
                )
                if desired > 0:
                    bought = self.supply_chain.distribute_masks(desired)
                    person.mask_inventory += bought
                    # FIXED: Track cumulative acquired for validation
                    self.cumulative_acquired += bought
                    daily_demand += desired
                    daily_sold += bought  # FIXED: Sold quantity

            # Decide adoption for the day; consumes one mask if adopting
            person.decide_adoption(
                price=self.supply_chain.price_per_mask,
                policy_active=mandate_active,
                rng=self.rng,
                enforcement_level=agency_enforcement,
                penalty_amount=penalty_amount,
                habit_formation_rate=habit_formation_rate,
                compliance_decay_rate=compliance_decay_rate,
            )

        # Visits and enforcement at public locations
        total_public_visits = 0
        public_incidents = 0
        public_compliant_entries = 0

        # FIXED: Cap enforcement by daily capacity
        remaining_capacity = int(self.policy.enforcement_capacity_per_day or self.p.get("enforcement_capacity_per_day", 0))

        for person in self.people:
            for ploc in self.locations:
                if self.rng.random() < ploc.foot_traffic_rate:
                    total_public_visits += 1
                    incident, compliant_now = ploc.enforce_mask_policy(person, agency_enforcement, self.rng)
                    if incident and remaining_capacity > 0:
                        public_incidents += 1
                        remaining_capacity -= 1
                    if compliant_now or person.mask_adopted or person.current_mask_use:
                        public_compliant_entries += 1

        # Workplace visits and enforcement
        total_work_visits = 0
        work_incidents = 0
        work_compliant_entries = 0
        for person in self.people:
            if 0 <= person.workplace_id < len(self.workplaces):
                wloc = self.workplaces[person.workplace_id]
                if self.rng.random() < wloc.foot_traffic_rate:
                    total_work_visits += 1
                    incident, compliant_now = wloc.enforce_mask_policy(person, agency_enforcement, self.rng)
                    if incident and remaining_capacity > 0:
                        work_incidents += 1
                        remaining_capacity -= 1
                    if compliant_now or person.mask_adopted or person.current_mask_use:
                        work_compliant_entries += 1

        # After all enforcement interactions, compute daily new adopters (end-of-day vs start-of-day)
        end_any_use_flags = [1 if (p.mask_adopted or p.current_mask_use) else 0 for p in self.people]
        daily_new_adopters_count = sum(
            1 for i in range(len(self.people)) if prev_any_use_flags[i] == 0 and end_any_use_flags[i] == 1
        )  # FIXED: Moved to post-enforcement

        # Update days_worn for each person based on final wearing state for day
        for person in self.people:
            if person.mask_adopted or person.current_mask_use:
                person.days_worn += 1

        # Supply price adjustment
        stockout = self.supply_chain.total_stock <= 0
        self.supply_chain.adjust_prices(stockout=stockout)

        # Metrics
        adoption = sum(1 for p in self.people if (p.mask_adopted or p.current_mask_use)) / max(1, len(self.people))
        avg_price = self.supply_chain.price_per_mask
        inventory = self.supply_chain.total_stock
        public_compliance_rate = (public_compliant_entries / max(1, total_public_visits)) if total_public_visits > 0 else 0.0
        work_compliance_rate = (work_compliant_entries / max(1, total_work_visits)) if total_work_visits > 0 else 0.0
        total_visits_all = total_public_visits + total_work_visits
        total_incidents_all = public_incidents + work_incidents
        incidents_per_1000 = (total_incidents_all / max(1, total_visits_all)) * 1000.0 if total_visits_all > 0 else 0.0
        overall_compliance_rate = ((public_compliant_entries + work_compliant_entries) / max(1, total_visits_all)) if total_visits_all > 0 else 0.0

        self.series["adoption_rate"].append(clamp(adoption))
        self.series["average_price"].append(avg_price)
        # FIXED: Write to supplier_inventory and alias legacy
        self.series["supplier_inventory"].append(inventory)
        self.series["retailer_inventory"].append(inventory)
        self.series["enforcement_incidents_per_1000"].append(incidents_per_1000)
        self.series["compliance_rate"].append(clamp(overall_compliance_rate))
        if mandate_active:
            # Track mandate-day compliance
            self.series["compliance_rate_under_mandate"].append(clamp(overall_compliance_rate))
        self.series["policy_enforcement_level"].append(clamp(self.policy.enforcement_level))
        self.series["daily_new_adopters"].append(daily_new_adopters_count)  # FIXED: post-enforcement
        self.series["adoption_work"].append(clamp(work_compliance_rate))
        self.series["adoption_public"].append(clamp(public_compliance_rate))
        # FIXED: Record demand/sold for shortage metrics
        self.series["daily_demand"].append(daily_demand)
        self.series["daily_sold"].append(daily_sold)

        self.daily_counters["visits_public"].append(total_public_visits)
        self.daily_counters["incidents_public"].append(public_incidents)
        self.daily_counters["compliant_public"].append(public_compliant_entries)
        self.daily_counters["visits_work"].append(total_work_visits)
        self.daily_counters["incidents_work"].append(work_incidents)
        self.daily_counters["compliant_work"].append(work_compliant_entries)

        # FIXED: Adaptive policy adjustment after observing today's outcomes
        self._adaptive_policy_adjustment(day)

    def run(self, days: int) -> Dict[str, object]:
        """
        Run the simulation for a specified number of days.

        Parameters
        ----------
        days : int
            Number of days to simulate.

        Returns
        -------
        Dict[str, object]
            Results including time series and summary metrics.
        """
        pass
        self.initialize()
        for day in range(days):
            self.step(day)

        # Summary metrics
        target = float(self.p.get("target_adoption_rate", 0.8))
        time_to_threshold: Optional[int] = None
        for d, val in enumerate(self.series["adoption_rate"]):
            if val >= target:
                time_to_threshold = d
                break

        # FIXED: time_to_50_percent_adoption
        time_to_50 = None
        for d, val in enumerate(self.series["adoption_rate"]):
            if val >= 0.5:
                time_to_50 = d
                break

        # mask_stockouts: daily boolean/count
        mask_stockouts = sum(1 for inv in self.series.get("supplier_inventory", self.series.get("retailer_inventory", [])) if inv <= 0)

        # Enforcement incidents totals
        total_visits_all = int(sum(self.daily_counters["visits_public"]) + sum(self.daily_counters["visits_work"]))
        total_incidents_all = int(sum(self.daily_counters["incidents_public"]) + sum(self.daily_counters["incidents_work"]))

        # Compliance rate under mandate (average)
        avg_compliance_under_mandate = statistics.mean(self.series["compliance_rate_under_mandate"]) if self.series["compliance_rate_under_mandate"] else 0.0

        # Post-mandate persistence: average adoption in 14 days after last mandate day
        post_persistence = 0.0
        last_mandate_day = self.policy.mandate_end_day if self.policy.mandate_enabled else -1
        if last_mandate_day >= 0 and last_mandate_day < len(self.series["adoption_rate"]) - 1:
            window = self.series["adoption_rate"][last_mandate_day + 1: last_mandate_day + 15]
            if window:
                post_persistence = statistics.mean(window)

        # Inequality in adoption across income quintiles
        incomes = [p.income for p in self.people]
        sorted_idx = sorted(range(len(incomes)), key=lambda i: incomes[i])
        quintile_size = max(1, len(self.people) // 5)
        quintile_rates = []
        for q in range(5):
            start = q * quintile_size
            end = (q + 1) * quintile_size if q < 4 else len(self.people)
            idxs = sorted_idx[start:end]
            if not idxs:
                quintile_rates.append(0.0)
            else:
                wearers = sum(1 for i in idxs if (self.people[i].mask_adopted or self.people[i].current_mask_use))
                quintile_rates.append(wearers / max(1, len(idxs)))
        # FIXED: Use Gini for inequality per feedback
        adoption_inequality_index = gini(quintile_rates) if len(quintile_rates) > 1 else 0.0

        # Adoption variance by age groups
        def age_group(a: int) -> int:
            if a < 30:
                return 0
            if a < 45:
                return 1
            if a < 65:
                return 2
            return 3

        age_bins: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
        for i, p in enumerate(self.people):
            age_bins[age_group(p.age)].append(i)
        age_rates = []
        for g, idxs in age_bins.items():
            if not idxs:
                age_rates.append(0.0)
            else:
                wearers = sum(1 for i in idxs if (self.people[i].mask_adopted or self.people[i].current_mask_use))
                age_rates.append(wearers / max(1, len(idxs)))
        adoption_variance_by_age = statistics.pvariance(age_rates) if len(age_rates) > 1 else 0.0

        # Average mask days per person
        avg_mask_days_per_person = statistics.mean([p.days_worn for p in self.people]) if self.people else 0.0

        # Peak adoption rate and day
        peak_adoption_rate = 0.0
        peak_day = None
        if self.series["adoption_rate"]:
            peak_adoption_rate = max(self.series["adoption_rate"])
            peak_day = self.series["adoption_rate"].index(peak_adoption_rate)

        # FIXED: Counterfactual guard to avoid infinite recursion
        policy_effectiveness_index = 0.0
        if (not self.p.get("_is_counterfactual", False)) and self.series["adoption_rate"]:
            cf_params = dict(self.p)
            cf_params["_is_counterfactual"] = True  # FIXED: Guard flag
            cf_params["mandate_enabled"] = False
            cf_params["enforcement_level"] = 0.0
            cf_sim = Simulation(cf_params, seed=9991)
            cf_res = cf_sim.run(len(self.series["adoption_rate"]))
            if cf_res.get("adoption_rate"):
                policy_effectiveness_index = self.series["adoption_rate"][-1] - cf_res["adoption_rate"][-1]

        # FIXED: Supply shortage days based on demand vs sold
        supply_shortage_days = 0
        if self.series.get("daily_demand") and self.series.get("daily_sold"):
            supply_shortage_days = sum(1 for d, s in zip(self.series["daily_demand"], self.series["daily_sold"]) if s < d)

        results = {
            "adoption_rate": self.series["adoption_rate"],
            "average_price": self.series["average_price"],
            "supplier_inventory": self.series["supplier_inventory"],
            "retailer_inventory": self.series["retailer_inventory"],  # legacy alias
            "enforcement_incidents_per_1000": self.series["enforcement_incidents_per_1000"],
            "compliance_rate": self.series["compliance_rate"],
            "compliance_rate_under_mandate_series": self.series["compliance_rate_under_mandate"],
            "daily_new_adopters": self.series["daily_new_adopters"],
            "adoption_work": self.series["adoption_work"],
            "adoption_public": self.series["adoption_public"],
            # FIXED: Metric naming per feedback
            "time_to_target_adoption": time_to_threshold,
            "time_to_50_percent_adoption": time_to_50,
            "mask_stockouts": mask_stockouts,
            "enforcement_incidents_rate": (total_incidents_all / max(1, total_visits_all)) * 1000.0 if total_visits_all > 0 else 0.0,
            "sustained_adoption_rate": statistics.mean(self.series["adoption_rate"][-min(30, len(self.series["adoption_rate"])) :]) if self.series["adoption_rate"] else 0.0,
            # FIXED: Replace variance with Gini index
            "adoption_inequality_index": float(adoption_inequality_index),
            "adoption_variance_by_age": float(adoption_variance_by_age),
            "average_mask_days_per_person": float(avg_mask_days_per_person),
            "peak_adoption_rate": float(peak_adoption_rate),
            "peak_adoption_day": peak_day,
            "compliance_rate_under_mandate": float(avg_compliance_under_mandate),
            "post_mandate_persistence": float(post_persistence),
            "policy_effectiveness_index": float(policy_effectiveness_index),
            "noncompliance_events": int(total_incidents_all),
            "total_visits": total_visits_all,
            "total_incidents": total_incidents_all,
            # FIXED: Additional aligned metrics
            "final_adoption_rate": self.series["adoption_rate"][-1] if self.series["adoption_rate"] else 0.0,
            "supply_shortage_days": int(supply_shortage_days),
            "enforcement_actions_count": int(total_incidents_all),
        }

        # FIXED: RMSE to observed in main results if provided
        observed = self.observed_adoption_series
        if observed and len(observed) == len(self.series["adoption_rate"]):
            diffsq = [(a - b) ** 2 for a, b in zip(self.series["adoption_rate"], observed)]
            results["rmse_to_observed"] = math.sqrt(sum(diffsq) / max(1, len(diffsq)))

        return results

    def evaluate(self, evaluation_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate additional metrics based on selected names.

        Parameters
        ----------
        evaluation_metrics : Optional[List[str]]
            List of metric names to compute.

        Returns
        -------
        Dict[str, float]
            Mapping of metric name to value.
        """
        pass
        if not evaluation_metrics:
            return {}

        metrics_out: Dict[str, float] = {}
        for name in evaluation_metrics:
            if name == "adoption_rate_final":
                metrics_out[name] = self.series["adoption_rate"][-1] if self.series["adoption_rate"] else 0.0
            elif name == "price_final":
                metrics_out[name] = self.series["average_price"][-1] if self.series["average_price"] else 0.0
            elif name == "rmse_to_observed":
                observed = self.p.get("observed_adoption_series", [])
                if observed and len(observed) == len(self.series["adoption_rate"]):
                    diffsq = [(a - b) ** 2 for a, b in zip(self.series["adoption_rate"], observed)]
                    metrics_out[name] = math.sqrt(sum(diffsq) / max(1, len(diffsq)))
                else:
                    metrics_out[name] = float("nan")
            elif name == "time_to_50_percent_adoption":
                t50 = None
                for d, v in enumerate(self.series.get("adoption_rate", [])):
                    if v >= 0.5:
                        t50 = d
                        break
                metrics_out[name] = float(t50) if t50 is not None else float("nan")
            elif name == "final_adoption_rate":
                metrics_out[name] = self.series["adoption_rate"][-1] if self.series["adoption_rate"] else 0.0
            elif name == "adoption_inequality_index":
                # Recompute Gini over quintile adoption rates
                incomes = [p.income for p in self.people]
                sorted_idx = sorted(range(len(incomes)), key=lambda i: incomes[i])
                quintile_size = max(1, len(self.people) // 5)
                quintile_rates = []
                for q in range(5):
                    start = q * quintile_size
                    end = (q + 1) * quintile_size if q < 4 else len(self.people)
                    idxs = sorted_idx[start:end]
                    if not idxs:
                        quintile_rates.append(0.0)
                    else:
                        wearers = sum(1 for i in idxs if (self.people[i].mask_adopted or self.people[i].current_mask_use))
                        quintile_rates.append(wearers / max(1, len(idxs)))
                metrics_out[name] = gini(quintile_rates) if len(quintile_rates) > 1 else 0.0
            else:
                metrics_out[name] = float("nan")
        return metrics_out

    def visualize(self) -> None:
        """
        Display a simple textual visualization of adoption over time.
        """
        pass
        series = self.series.get("adoption_rate", [])
        if not series:
            print("No data to visualize.")
            return
        print("Adoption over time (ASCII sparkline):")
        blocks = "▁▂▃▄▅▆▇█"
        for _, v in enumerate(series):
            idx = int(clamp(v, 0.0, 1.0) * (len(blocks) - 1))
            sys.stdout.write(blocks[idx])
        sys.stdout.write("\n")
        print(f"Final adoption: {series[-1]:.3f}")

    def save_results(self, filename: str) -> None:
        """
        Save key time series to a CSV file.

        Parameters
        ----------
        filename : str
            Output CSV filename.
        """
        pass
        header = [
            "day",
            "adoption_rate",
            "average_price",
            "supplier_inventory",
            "enforcement_incidents_per_1000",
            "compliance_rate",
            "adoption_work",
            "adoption_public",
            "daily_new_adopters",
            "daily_demand",
            "daily_sold",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                days = len(self.series["adoption_rate"])
                for d in range(days):
                    writer.writerow(
                        [
                            d,
                            self.series["adoption_rate"][d],
                            self.series["average_price"][d],
                            self.series["supplier_inventory"][d],
                            self.series["enforcement_incidents_per_1000"][d],
                            self.series["compliance_rate"][d],
                            self.series["adoption_work"][d],
                            self.series["adoption_public"][d],
                            self.series["daily_new_adopters"][d],
                            self.series["daily_demand"][d],
                            self.series["daily_sold"][d],
                        ]
                    )
        except Exception as e:
            # FIXED: Wrap file writes in try/except to avoid crashes in restricted environments
            print(f"Error writing CSV: {e}", file=sys.stderr)

    # FIXED: Existing validation utilities updated with additional checks
    def validate_policy_monotonicity(self) -> Tuple[bool, str]:
        """
        Validation: Policy monotonicity.

        Compare scenarios with low vs high enforcement and ensure adoption with high policy >= adoption with low policy.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        base_params = dict(self.p)
        short_days = 40

        low_params = dict(base_params)
        low_params.update({"enforcement_level": 0.05, "mandate_enabled": True})
        low_sim = Simulation(low_params, seed=123)
        low_res = low_sim.run(short_days)
        low_final = low_res["adoption_rate"][-1]

        high_params = dict(base_params)
        high_params.update({"enforcement_level": 0.9, "mandate_enabled": True})
        high_sim = Simulation(high_params, seed=123)
        high_res = high_sim.run(short_days)
        high_final = high_res["adoption_rate"][-1]

        ok = high_final >= low_final - 1e-9
        msg = f"policy_monotonicity: high={high_final:.3f}, low={low_final:.3f}"
        return ok, msg

    def validate_no_influence_stability(self) -> Tuple[bool, str]:
        """
        Validation: Stability with no social/policy/media influence.

        Set all influence weights to zero and check adoption remains near initial rate.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        params2 = dict(self.p)
        params2.update(
            {
                "social_influence_weight": 0.0,
                "policy_influence_weight": 0.0,
                "media_influence_weight": 0.0,
                "habit_formation_rate": 0.0,
                "compliance_decay_rate": 0.0,
                "mandate_enabled": False,
            }
        )
        sim = Simulation(params2, seed=456)
        days = 60
        res = sim.run(days)
        init = float(params2.get("initial_adoption_rate", 0.1))
        avg_last14 = statistics.mean(res["adoption_rate"][-14:]) if len(res["adoption_rate"]) >= 14 else res["adoption_rate"][-1]
        ok = abs(avg_last14 - init) < 0.1
        msg = f"no_influence_stability: avg_last14={avg_last14:.3f}, init={init:.3f}"
        return ok, msg

    def validate_convergence_check(self) -> Tuple[bool, str]:
        """
        Validation: Convergence.

        Confirm that the change in adoption rate over the last 14 days is below epsilon.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        res = self.series.get("adoption_rate", [])
        if len(res) < 15:
            return False, "convergence_check: insufficient data (<15 days)"
        delta = abs(res[-1] - res[-15])
        ok = delta < float(self.p.get("convergence_epsilon", 0.001))
        msg = f"convergence_check: delta14={delta:.6f}"
        return ok, msg

    def validate_population_accounting(self) -> Tuple[bool, str]:
        """
        Validation: Population accounting.

        Check that adopters + non-adopters equals N and inventories are non-negative.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        N = len(self.people)
        adopters = sum(1 for p in self.people if (p.mask_adopted or p.current_mask_use))
        non_adopters = N - adopters
        inv_nonneg = all(p.mask_inventory >= 0 for p in self.people)
        ok = (adopters + non_adopters == N) and inv_nonneg and N > 0
        msg = f"population_accounting: N={N}, adopters={adopters}, non_adopters={non_adopters}, inventories_ok={inv_nonneg}"
        return ok, msg

    def validate_bounded_probabilities(self) -> Tuple[bool, str]:
        """
        Validation: Bounded probabilities.

        Ensure all configured probabilities and rates are within [0,1].

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        keys = [
            "social_influence_weight",
            "policy_influence_weight",
            "media_influence_weight",
            "mandate_enabled",
            "communication_frequency",
            "misinformation_rate",
            "external_prevalence_signal",
            "risk_perception_sensitivity_to_prevalence",
            "procurement_access_fraction",
        ]
        violated = []
        for k in keys:
            v = self.p.get(k, 0.5)
            if isinstance(v, (int, float)):
                if not (0.0 <= float(v) <= 1.0):
                    violated.append((k, v))
        ok = len(violated) == 0
        msg = "bounded_probabilities: ok" if ok else f"bounded_probabilities: violations={violated}"
        return ok, msg

    def validate_replicate_consistency(self, runs: int = 3, cv_threshold: float = 0.2) -> Tuple[bool, str]:
        """
        Validation: Replicate consistency.

        Run multiple replicates and ensure coefficient of variation (CV) of final adoption is below a threshold.

        Parameters
        ----------
        runs : int
            Number of replicates.
        cv_threshold : float
            Maximum acceptable coefficient of variation.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        finals = []
        for r in range(runs):
            # FIXED: Lightweight replicate consistency by downscaling N/days in validation runs
            p2 = dict(self.p)
            p2["population_size"] = min(1000, int(p2.get("population_size", 10000)))
            sim = Simulation(p2, seed=1000 + r)
            res = sim.run(min(90, int(p2.get("time_horizon_days", 180))))
            finals.append(res["adoption_rate"][-1])
        mean_val = statistics.mean(finals) if finals else 0.0
        stdev_val = statistics.pstdev(finals) if len(finals) > 1 else 0.0
        cv = (stdev_val / mean_val) if mean_val > 0 else float("inf")
        ok = cv <= cv_threshold
        msg = f"replicate_consistency: cv={cv:.3f}, mean={mean_val:.3f}, stdev={stdev_val:.3f}"
        return ok, msg

    def validate_supply_demand_balance_check(self) -> Tuple[bool, str]:
        """
        Validation: Supply-demand mass balance.

        Ensure cumulative produced >= cumulative distributed and distributed == acquired within tolerance.

        Returns
        -------
        Tuple[bool, str]
            (result, message)
        """
        pass
        # FIXED: Account for initial stock in produced for validation
        produced = self.supply_chain.cumulative_produced + max(0, int(self.initial_total_stock))
        distributed = self.supply_chain.cumulative_distributed
        acquired = self.cumulative_acquired
        ok = (produced >= distributed) and (distributed == acquired)
        msg = f"supply_demand_balance: produced={produced}, distributed={distributed}, acquired={acquired}, ok={ok}"
        return ok, msg


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the simulation CLI.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    pass
    parser = argparse.ArgumentParser(description="Mask Adoption Dynamics Simulation CLI")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input JSON with parameters.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file for results.")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV output for time series.")
    parser.add_argument("--population-size", type=int, default=None, help="Override population size.")
    parser.add_argument("--days", type=int, default=None, help="Number of days to simulate (overrides time_horizon_days).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fast-smoke-run", action="store_true", help="Run a fast smoke test (prints SMOKE_OK).")
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Comma-separated validations: policy_monotonicity,no_influence_stability,convergence_check,population_accounting,bounded_probabilities,replicate_consistency,supply_demand_balance",
    )
    return parser.parse_args()


def default_params() -> Dict[str, float]:
    """
    Construct default parameters for the simulation.

    Returns
    -------
    Dict[str, float]
        Default parameter dictionary.
    """
    pass
    return {
        # FIXED: Spec defaults aligned
        "population_size": 10000,
        "time_horizon_days": 180,
        "decision_interval_days": 1,
        "initial_adoption_rate": 0.1,
        # Network
        "avg_degree": 10,
        "social_network_rewiring_p": 0.05,
        # Influence weights
        "social_influence_weight": 0.4,
        "policy_influence_weight": 0.3,
        "media_influence_weight": 0.2,
        "contact_rate_per_day": 10,
        # Household
        "household_size_lambda": 3.0,
        "household_influence_weight": 0.6,
        # Risk and media
        "risk_level": 0.2,
        "risk_perception_sensitivity_to_prevalence": 0.6,
        "external_prevalence_signal": 0.1,
        "media_reach_main": 0.7,
        "message_frequency_per_week": 3.0,
        "media_bias": 0.0,
        "misinformation_rate": 0.05,
        # Policy authority
        "mandate_enabled": False,
        "mandate_start_day": 30,
        "mandate_end_day": 120,
        "enforcement_level": 0.5,
        "penalty_amount": 50.0,
        "incentive_amount": 0.0,
        "communication_frequency": 0.5,
        "message_strategy": 0.6,
        "subsidy_rate": 0.0,
        "venue_policy_coverage": 0.6,
        "enforcement_capacity_per_day": 0,
        # Locations
        "workplace_capacity_mean": 200,
        "workplace_enforcement_strictness": 0.5,
        "workplace_signage_effect": 0.05,
        "workplace_attendance_rate": 0.5,
        "public_venues_count": 5,
        "location_capacity_mean": 2000,
        "location_enforcement_strictness_mean": 0.5,
        "public_venue_visit_rate": 0.3,
        "signage_effect": 0.05,
        "num_workplaces": 50,
        # Supply chain
        "mask_supply_per_capita": 5.0,
        "supplier_initial_inventory_ratio": 1.0,
        "production_rate_per_day": 500,
        "distribution_delay_days": 2,
        "mask_price": 2.0,
        "min_mask_price": 0.5,
        "max_mask_price": 50.0,
        "rationing_policy": "price",
        "purchase_bundle": 5,
        "procurement_access_fraction": 0.9,
        # Decision and habit
        "perceived_benefit_base": 0.4,
        "perceived_cost_base": 0.2,
        "habit_formation_rate": 0.02,
        "compliance_decay_rate": 0.01,
        "mask_effectiveness_perceived": 0.5,
        # Targets for adaptive policy
        "target_adoption_rate": 0.8,
        "target_adoption_deadline_day": 90,
        "adoption_target_recent": 0.6,
        "compliance_target_recent": 0.7,
        # Validation
        "convergence_epsilon": 0.001,
    }


def run_validations(sim: Simulation, which: List[str]) -> Tuple[bool, List[str]]:
    """
    Run selected validation checks on the simulation model.

    Parameters
    ----------
    sim : Simulation
        Simulation instance (parameters used for baselines).
    which : List[str]
        List of validation names.

    Returns
    -------
    Tuple[bool, List[str]]
        Overall result and list of messages.
    """
    pass
    results: List[str] = []
    all_ok = True
    for name in which:
        if name == "policy_monotonicity":
            ok, msg = sim.validate_policy_monotonicity()
        elif name == "no_influence_stability":
            ok, msg = sim.validate_no_influence_stability()
        elif name == "convergence_check":
            ok, msg = sim.validate_convergence_check()
        elif name == "population_accounting":
            ok, msg = sim.validate_population_accounting()
        elif name == "bounded_probabilities":
            ok, msg = sim.validate_bounded_probabilities()
        elif name == "replicate_consistency":
            ok, msg = sim.validate_replicate_consistency()
        elif name == "supply_demand_balance":
            ok, msg = sim.validate_supply_demand_balance_check()
        else:
            ok, msg = (False, f"unknown_validation: {name}")
        all_ok = all_ok and ok
        results.append(msg)
    return all_ok, results


# Execute main for both direct execution and sandbox wrapper invocation
main()