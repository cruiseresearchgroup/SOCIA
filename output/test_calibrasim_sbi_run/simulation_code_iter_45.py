def main():
    pass

import json
import logging
import math
import os
import random
import sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# FIXED: Defer plotting imports into visualize() to avoid ImportError at import time
import networkx as nx
import numpy as np

# Path handling per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def _sanitize_json_text(text: str) -> str:
    """
    Truncate a blob of text to the last closing brace/bracket to remove trailing noise,
    enabling more robust JSON parsing when extra text gets appended.

    Returns:
        A substring that ends at the last '}' or ']'.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    last_obj = text.rfind("}")
    last_arr = text.rfind("]")
    cut = max(last_obj, last_arr)
    return text[: cut + 1] if cut != -1 else text


def clip01(x: float) -> float:
    """
    Clip a float to the [0, 1] interval.

    Args:
        x: Input value.

    Returns:
        Clipped value between 0 and 1.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    return max(0.0, min(1.0, x))


def logistic(x: float) -> float:
    """
    Compute logistic function (sigmoid): 1 / (1 + exp(-x)).

    Args:
        x: Input value.

    Returns:
        Sigmoid of x.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def moving_average(values: List[float], window: int) -> List[float]:
    """
    Compute a trailing moving average over a list of values.

    Args:
        values: Sequence of numeric values.
        window: Window size for averaging.

    Returns:
        List of smoothed values of the same length as input.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    if window <= 1 or not values:
        return list(values)
    window = max(1, int(window))
    buf = deque(maxlen=window)
    output: List[float] = []
    s = 0.0
    for v in values:
        if len(buf) == window:
            s -= buf[0]
        buf.append(v)
        s += v
        denom = len(buf)
        output.append(s / denom if denom > 0 else 0.0)
    return output


def sample_poisson(lam: float, rng: random.Random) -> int:
    """
    Sample from a Poisson distribution. Uses a numpy Generator seeded from the provided
    Python RNG for reproducibility; otherwise, falls back to the Knuth algorithm.

    Note:
        This function creates a new numpy Generator per call which can be slower.
        For high-frequency sampling, use a cached numpy Generator as implemented in Retailer.
        # FIXED: Performance overhead note added; Retailer caches RNG for Poisson sampling.

    Args:
        lam: Expected rate (lambda).
        rng: Python random.Random instance for reproducibility.

    Returns:
        Non-negative integer sample from Poisson(lam).
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    lam = max(0.0, lam)
    try:
        ss = np.random.SeedSequence(rng.getrandbits(128))
        rg = np.random.default_rng(ss)
        return int(rg.poisson(lam))
    except Exception:
        # Fallback: Knuth algorithm (Python-only)
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        return max(0, k - 1)


def gini(values: List[float]) -> float:
    """
    Compute the Gini coefficient of a list of non-negative values.

    Args:
        values: List of non-negative numbers.

    Returns:
        Gini coefficient between 0 and 1.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumvals = np.cumsum(sorted_vals)
    total = cumvals[-1] if len(cumvals) > 0 else 0.0
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n)


@dataclass
class Person:
    """
    Agent representing an individual with mask-specific daily decision dynamics.

    Attributes:
        id: Unique identifier of the person.
        age_group: Age category string.
        income_level: Broad income label for heterogeneity (unused if income numeric is present).
        home_location_id: Identifier of home location (optional).
        work_location_id: Identifier of work location (optional).
        trust_in_authority: Trust scalar in [0, 1].
        risk_perception: Perceived risk scalar in [0, 1].
        social_influence_susceptibility: Susceptibility to peers/social norms [0, 1].
        compliance_propensity: Tendency to comply with mandates [0, 1].
        mask_owned_count: Integer count of masks in possession.
        is_wearing_mask: Whether the person is wearing a mask today (dynamic).
        fatigue_level: Discomfort/fatigue level dampening wearing [0, 1].
        daily_contacts: Approximate daily social contacts (used to scale social norms if needed).
        info_true_level: Awareness of true information (legacy for campaign influence).
        info_misinfo_level: Exposure to misinformation (legacy for campaign influence).
        stubborn: If True, the person rarely complies regardless of context.
        household_id: Household identifier.
        income: Numeric income proxy for affordability and inequality metrics.
        habit_strength: Habit strength [0, 1] reinforcing wearing decisions.
        cumulative_spend: Total spending on masks and fines.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    id: int = 0
    age_group: str = "18_34"
    income_level: str = "medium"
    home_location_id: int = -1
    work_location_id: int = -1
    trust_in_authority: float = 0.5
    risk_perception: float = 0.4
    social_influence_susceptibility: float = 0.5
    compliance_propensity: float = 0.6
    mask_owned_count: int = 0
    is_wearing_mask: bool = False
    fatigue_level: float = 0.0
    daily_contacts: int = 10
    info_true_level: float = 0.0
    info_misinfo_level: float = 0.0
    stubborn: bool = False
    household_id: int = -1
    income: float = 1.0
    habit_strength: float = 0.0
    cumulative_spend: float = 0.0

    def decide_wear_mask(self, social_norm: float, mandate_active: bool, enforcement_level: float, household_norm: float, household_weight: float) -> None:
        """
        Decide whether to wear a mask today based on social norms, household norm, risk, trust, fatigue, and mandate.

        Args:
            social_norm: Share of mask wearing among observed peers/neighbors yesterday.
            mandate_active: Whether a mask mandate is in effect.
            enforcement_level: Effective enforcement probability multiplier [0, 1].
            household_norm: Share of mask wearing in the person's household yesterday.
            household_weight: Weight for household norm in decision process.

        Returns:
            None. Updates is_wearing_mask in-place.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        base = -2.0
        # FIXED: Use stubborn attribute to cap responsiveness and reduce social/mandate weights
        stubborn_penalty = 0.8 if self.stubborn else 0.0
        social_w = 1.5 if self.stubborn else 2.2
        mandate_w = 0.6 if self.stubborn else 1.5
        enforce_w = 0.8 if self.stubborn else 2.0
        # FIXED: Include household norm influence via household_weight
        combined_social = (1.0 - household_weight) * social_norm + household_weight * household_norm
        logit = (
            base
            + social_w * self.social_influence_susceptibility * combined_social
            + 1.0 * self.risk_perception
            + 0.6 * self.trust_in_authority
            + 1.2 * self.habit_strength
            - 2.0 * self.fatigue_level
            - stubborn_penalty
        )
        if mandate_active:
            logit += mandate_w * self.compliance_propensity + enforce_w * enforcement_level
        p = min(0.7, logistic(logit)) if self.stubborn else logistic(logit)
        self.is_wearing_mask = (self.mask_owned_count > 0) and (random.random() < p)

    def replace_mask(self, day: int, interval: int) -> None:
        """
        Consume or replace masks according to a replacement interval.

        Args:
            day: Current simulation day (0-indexed).
            interval: Replacement interval in days; if > 0, consume one mask every interval.

        Returns:
            None. Decrements mask_owned_count if applicable.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # FIXED: Implement replacement cycle so inventory depletes over time
        if interval > 0 and self.mask_owned_count > 0:
            if (day % interval == 0) and self.is_wearing_mask:
                self.mask_owned_count -= 1

    def purchase_mask(self, retailers: List["Retailer"], threshold: int, rng: random.Random) -> int:
        """
        Attempt to purchase masks if inventory is at/below threshold.

        Args:
            retailers: List of available retailers.
            threshold: Purchase reorder threshold.
            rng: Random generator.

        Returns:
            Number of masks purchased.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if self.mask_owned_count > threshold:
            return 0
        if not retailers:
            return 0
        # Choose retailer with inventory and lowest price
        candidates = [r for r in retailers if r.total_inventory() > 0]
        if not candidates:
            return 0
        r = min(candidates, key=lambda x: x.price_per_mask)
        # Budget rule-of-thumb: up to 1% of monthly income per purchase attempt, capped 20
        budget = min(0.01 * max(0.0, self.income), 20.0)
        if r.price_per_mask <= 0.0:
            price = 1e-6
        else:
            price = r.price_per_mask
        max_afford = int(budget // price)
        if max_afford <= 0:
            return 0
        # Target to acquire up to (threshold + 2) additional units
        target = max(1, threshold + 2 - self.mask_owned_count)
        qty_wanted = max(1, min(max_afford, target))
        bought, cost = r.sell_masks(qty_wanted)
        if bought > 0:
            self.mask_owned_count += bought
            self.cumulative_spend += cost
        return bought


@dataclass
class Retailer:
    """
    Retailer handling mask inventory, pricing, and restocking.

    Attributes:
        id: Retailer identifier.
        inventory: Current stock level (integer).
        price_per_mask: Current price per mask (float).
        base_price: Base price used for reference.
        restock_rate: Mean daily restock quantity (Poisson mean).
        backlog: Outstanding unmet demand (for price pressure).
        yesterday_demand: Tracked demand yesterday (units requested).
        yesterday_sales: Actual sales completed yesterday (units sold).
        price_adjustment_sensitivity: Multiplier for price changes due to demand pressure.
        rg: Cached numpy Generator for Poisson sampling.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    id: int = 0
    inventory: int = 0
    price_per_mask: float = 1.0
    base_price: float = 1.0
    restock_rate: float = 0.0
    backlog: int = 0
    yesterday_demand: int = 0
    yesterday_sales: int = 0
    price_adjustment_sensitivity: float = 0.3
    rg: np.random.Generator = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to ensure consistent state.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if self.rg is None:
            # FIXED: Cache numpy Generator seeded from python RNG for reproducible and faster Poisson sampling
            ss = np.random.SeedSequence(random.getrandbits(128))
            self.rg = np.random.default_rng(ss)

    def total_inventory(self) -> int:
        """
        Return current total inventory for this retailer.

        Returns:
            Integer inventory level.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        return max(0, int(self.inventory))

    def restock_daily(self) -> int:
        """
        Restock inventory according to a Poisson process with mean restock_rate.

        Returns:
            Number of units added to inventory today.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        arrivals = 0
        lam = max(0.0, float(self.restock_rate))
        if lam > 0.0:
            # FIXED: Use cached numpy Generator for performance
            arrivals = int(self.rg.poisson(lam))
        self.inventory += arrivals
        return arrivals

    def sell_masks(self, quantity_requested: int) -> Tuple[int, float]:
        """
        Sell up to quantity_requested masks, constrained by inventory.

        Args:
            quantity_requested: Desired units.

        Returns:
            Tuple of (units_sold, revenue_collected).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        q = max(0, int(quantity_requested))
        self.yesterday_demand += q
        if q <= 0:
            return 0, 0.0
        sell_q = min(q, max(0, int(self.inventory)))
        self.inventory -= sell_q
        self.yesterday_sales += sell_q
        revenue = float(sell_q) * max(0.0, float(self.price_per_mask))
        # Track backlog (unmet demand)
        self.backlog += max(0, q - sell_q)
        return sell_q, revenue

    def adjust_price(self) -> None:
        """
        Adjust price based on demand pressure relative to supply.

        Returns:
            None. Updates price_per_mask in-place.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        supply = max(1, self.yesterday_sales)
        pressure = float(self.yesterday_demand) / float(supply)
        # Price multiplier grows with pressure; bounded
        multiplier = 1.0 + self.price_adjustment_sensitivity * (pressure - 1.0)
        multiplier = float(np.clip(multiplier, 0.5, 2.0))
        # Update price anchored to base price
        self.price_per_mask = max(0.2, self.base_price * multiplier)
        # Reset daily trackers
        self.yesterday_demand = 0
        self.yesterday_sales = 0


@dataclass
class PolicyAuthority:
    """
    Public Health Authority managing mandates and enforcement.

    Attributes:
        mandate_day: Day index to start mandate (0-indexed).
        enforcement_level: Enforcement strength [0, 1].
        fine_amount: Fine amount per noncompliance event.
        communication_frequency: Frequency of messaging (unused placeholder).
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    mandate_day: int = 9999
    enforcement_level: float = 0.0
    fine_amount: float = 0.0
    communication_frequency: float = 1.0

    def step(self, t: int) -> Dict[str, Any]:
        """
        Emit policy signals for day t.

        Args:
            t: Day index.

        Returns:
            Dict with mandate_active and enforcement_level.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        mandate_active = bool(t >= int(self.mandate_day))
        return {
            "mandate_active": mandate_active,
            "enforcement_level": float(self.enforcement_level),
            "fine_amount": float(self.fine_amount),
        }


@dataclass
class Location:
    """
    Minimal location entity to enable policy enforcement context.

    Attributes:
        id: Location identifier.
        type: Location type string.
        capacity: Nominal capacity (unused in this simplified version).
        mask_requirement: Whether masks are required at this location.
        enforcement_level: Enforcement probability modifier at this location [0, 1].
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    id: int = 0
    type: str = "public"
    capacity: int = 0
    mask_requirement: bool = False
    enforcement_level: float = 0.0


@dataclass
class MediaOutlet:
    """
    Media outlet broadcasting messages influencing risk perception and trust.

    Attributes:
        message_bias: Signed bias toward pro-mask messaging (positive) or anti-mask (negative).
        misinformation_rate: Fraction of content that is misinformation [0, 1].
        reach: Probability an individual is exposed on a given day [0, 1].
        credibility: Weight of impact on beliefs [0, 1].
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    message_bias: float = 0.5
    misinformation_rate: float = 0.1
    reach: float = 0.5
    credibility: float = 0.7

    def broadcast(self, people: List[Person], media_effect_weight: float, rng: random.Random) -> None:
        """
        Broadcast to individuals and update their risk and trust.

        Args:
            people: List of Person agents.
            media_effect_weight: Global media influence strength [0, 1].
            rng: Random generator for exposure sampling.

        Returns:
            None. Updates people in-place.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # FIXED: Implement media influence per feedback
        for p in people:
            if rng.random() < self.reach:
                delta_risk = media_effect_weight * self.credibility * (1.0 - self.misinformation_rate) * self.message_bias
                delta_trust = 0.5 * media_effect_weight * self.credibility * (1.0 - 2.0 * self.misinformation_rate)
                p.risk_perception = clip01(p.risk_perception + delta_risk)
                p.trust_in_authority = clip01(p.trust_in_authority + delta_trust)


class DiseaseProcess:
    """
    Simple SEIR-like disease process with per-contact transmission modified by masks and location modifiers.

    Attributes:
        beta: Base probability of transmission per contact in neutral conditions.
        incubation_mean: Mean days in E (exposed) before becoming infectious.
        infectious_mean: Mean days in I (infectious) before recovering.
        mask_inward: Efficacy of masks for the susceptible person (inward).
        mask_outward: Efficacy of masks for the infectious person (outward).
        contact_modifiers: Dictionary of location_type -> multiplier on beta.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(
        self,
        N: int,
        base_transmission_rate: float,
        incubation_mean: float,
        infectious_mean: float,
        mask_inward_efficacy: float,
        mask_outward_efficacy: float,
        contact_modifiers: Dict[str, float],
        rng: random.Random,
        initial_infected_fraction: float = 0.01,
    ):
        """
        Initialize the disease process with population size and parameters.

        Args:
            N: Population size.
            base_transmission_rate: Base per-contact transmission probability.
            incubation_mean: Mean days incubating (E).
            infectious_mean: Mean days infectious (I).
            mask_inward_efficacy: Fractional reduction for susceptible mask.
            mask_outward_efficacy: Fractional reduction for infectious mask.
            contact_modifiers: Location-specific modifiers on transmission risk.
            rng: Random generator.
            initial_infected_fraction: Fraction initially infected (E).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.N = int(N)
        self.beta = float(base_transmission_rate)
        self.incubation_mean = float(incubation_mean)
        self.infectious_mean = float(infectious_mean)
        self.mask_inward = float(mask_inward_efficacy)
        self.mask_outward = float(mask_outward_efficacy)
        self.contact_modifiers = dict(contact_modifiers or {})
        self.rng = rng
        # Internal state arrays
        self.state: List[str] = ["S"] * self.N
        self.days_to_event: List[int] = [0] * self.N  # countdown to next transition for E/I
        # Series
        self.new_infections_daily: List[int] = []
        self.prevalence_over_time: List[float] = []
        self.cumulative_infections_over_time: List[int] = []
        self.Re_over_time: List[float] = []

        # Seed initial infections (as exposed)
        init_E = max(0, int(round(initial_infected_fraction * self.N)))
        init_idxs = self._rng_sample_without_replacement(range(self.N), init_E)
        for i in init_idxs:
            self.state[i] = "E"
            self.days_to_event[i] = max(1, int(round(self._sample_incubation())))

        # Track cumulative infections
        self.cumulative_infections = init_E

    def _rng_sample_without_replacement(self, population, k: int) -> List[int]:
        """
        Sample k distinct elements from population using the Python RNG.

        Args:
            population: Iterable of elements.
            k: Number of samples.

        Returns:
            List of sampled elements.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        pop = list(population)
        if k <= 0 or not pop:
            return []
        k = min(k, len(pop))
        return random.sample(pop, k)

    def _sample_incubation(self) -> float:
        """
        Sample incubation period in days.

        Returns:
            Positive float number of days.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # Use Poisson with mean incubation_mean
        lam = max(1e-3, self.incubation_mean)
        return max(1.0, float(np.random.poisson(lam)))

    def _sample_infectious(self) -> float:
        """
        Sample infectious period in days.

        Returns:
            Positive float number of days.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        lam = max(1e-3, self.infectious_mean)
        return max(1.0, float(np.random.poisson(lam)))

    def transmit_given_contact(self, src_masked: bool, dst_masked: bool, location_type: str) -> float:
        """
        Compute per-contact transmission probability given masks and location type.

        Args:
            src_masked: Whether the infectious (source) wears a mask.
            dst_masked: Whether the susceptible (destination) wears a mask.
            location_type: Location type string.

        Returns:
            Probability of transmission in [0, 1].
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        m_out = (1.0 - self.mask_outward) if src_masked else 1.0
        m_in = (1.0 - self.mask_inward) if dst_masked else 1.0
        loc = self.contact_modifiers.get(location_type, 1.0)
        return clip01(self.beta * m_out * m_in * loc)

    def progress(self) -> Tuple[int, int, int, int]:
        """
        Progress disease states by one day for all individuals.

        Returns:
            Tuple of counts (S, E, I, R) after progression.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        S = E = I = R = 0
        for i in range(self.N):
            st = self.state[i]
            if st == "E":
                self.days_to_event[i] -= 1
                if self.days_to_event[i] <= 0:
                    self.state[i] = "I"
                    self.days_to_event[i] = max(1, int(round(self._sample_infectious())))
            elif st == "I":
                self.days_to_event[i] -= 1
                if self.days_to_event[i] <= 0:
                    self.state[i] = "R"
                    self.days_to_event[i] = 0
        # Count states after update
        for st in self.state:
            if st == "S":
                S += 1
            elif st == "E":
                E += 1
            elif st == "I":
                I += 1
            else:
                R += 1
        return S, E, I, R

    def apply_infections(self, infections: List[int]) -> int:
        """
        Infect a set of susceptible individuals transitioning them to E.

        Args:
            infections: List of individual indices to infect.

        Returns:
            Number of newly infected.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        new_inf = 0
        for j in infections:
            if 0 <= j < self.N and self.state[j] == "S":
                self.state[j] = "E"
                self.days_to_event[j] = max(1, int(round(self._sample_incubation())))
                new_inf += 1
        self.cumulative_infections += new_inf
        return new_inf

    def record_daily_metrics(self, I_prev: int, new_infections: int) -> None:
        """
        Record daily prevalence, cumulative infections, and effective reproduction number.

        Args:
            I_prev: Number infectious on previous day.
            new_infections: Number of new infections today.

        Returns:
            None.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.new_infections_daily.append(int(new_infections))
        prevalence = sum(1 for s in self.state if s == "I") / float(self.N or 1)
        self.prevalence_over_time.append(float(prevalence))
        self.cumulative_infections_over_time.append(int(self.cumulative_infections))
        Re = float(new_infections) / float(max(1, I_prev))
        self.Re_over_time.append(Re)


class NetworkGenerator:
    """
    Generates the social network, assigns households, and initializes agent attributes.

    Methods:
        build: Construct networkx graph, people list, and household mapping.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Dict[str, Any], rng: random.Random):
        """
        Initialize the generator with configuration and RNG.

        Args:
            cfg: Configuration dictionary.
            rng: Random generator.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = cfg
        self.rng = rng

    def _assign_age_group(self, u: float) -> str:
        """
        Assign age group based on configured shares.

        Args:
            u: Uniform random number in [0, 1].

        Returns:
            Age group string.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        s0 = float(self.cfg.get("share_age_0_17", 0.2))
        s1 = float(self.cfg.get("share_age_18_34", 0.35))
        s2 = float(self.cfg.get("share_age_35_64", 0.3))
        s3 = float(self.cfg.get("share_age_65_plus", 0.15))
        total = max(1e-9, s0 + s1 + s2 + s3)
        s0, s1, s2, s3 = (s0 / total, s1 / total, s2 / total, s3 / total)
        if u < s0:
            return "0_17"
        elif u < s0 + s1:
            return "18_34"
        elif u < s0 + s1 + s2:
            return "35_64"
        else:
            return "65_plus"

    def build(self) -> Tuple[nx.Graph, List[Person], Dict[int, List[int]]]:
        """
        Build the social network and initialize agent attributes aligned to mask dynamics.

        Returns:
            Tuple of (graph, people, households).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # FIXED: Align configuration keys with spec (population_size, network_topology, average_degree)
        N = int(self.cfg.get("population_size", self.cfg.get("n_agents", 5000)))
        topology = str(self.cfg.get("network_topology", self.cfg.get("network_type", "small_world")))
        avg_degree = max(2, int(round(float(self.cfg.get("average_degree", self.cfg.get("avg_degree", 8.0))))))
        rewiring_prob = float(self.cfg.get("rewiring_prob_small_world", self.cfg.get("rewiring_prob", 0.1)))
        m_ba = int(self.cfg.get("m_ba", 3))
        rng_seed = int(self.cfg.get("random_seed", 42))

        # Build base graph
        if topology in ("scale_free", "barabasi_albert"):
            m = max(1, min(m_ba, N - 1))
            G = nx.barabasi_albert_graph(n=N, m=m, seed=rng_seed)
        else:
            k = max(2, min(avg_degree if avg_degree % 2 == 0 else avg_degree + 1, N - 1))
            G = nx.watts_strogatz_graph(n=N, k=k, p=rewiring_prob, seed=rng_seed)

        # Households: simple grouping by fixed size 2-5
        households: Dict[int, List[int]] = defaultdict(list)
        hh_id = 0
        i = 0
        while i < N:
            size = self.rng.randint(2, 5)
            members = list(range(i, min(N, i + size)))
            households[hh_id] = members
            for u in members:
                G.nodes[u]["household_id"] = hh_id
            hh_id += 1
            i += size

        # Add intra-household edges with probability
        hh_cluster_p = float(self.cfg.get("household_cluster_prob", 0.3))
        for hid, members in households.items():
            for a in members:
                for b in members:
                    if a < b and self.rng.random() < hh_cluster_p:
                        if not G.has_edge(a, b):
                            G.add_edge(a, b)

        # Initialize people with attributes
        trust_mean = float(self.cfg.get("trust_in_authorities_mean", 0.6))
        risk_mean = float(self.cfg.get("risk_perception_initial_mean", 0.4))
        comp_mean = float(self.cfg.get("compliance_propensity_mean", 0.6))
        init_adopt = float(self.cfg.get("initial_adoption_rate", 0.1))
        stubborn_fraction = float(self.cfg.get("stubborn_fraction", 0.05))
        sus_mean = float(self.cfg.get("social_influence_susceptibility_mean", 0.5))
        sus_sd = float(self.cfg.get("social_influence_susceptibility_sd", 0.15))

        people: List[Person] = []
        for u in range(N):
            age_group = self._assign_age_group(self.rng.random())
            trust = float(np.clip(np.random.beta(2.0, max(0.1, (2.0 / max(1e-3, trust_mean)) - 2.0)), 0.0, 1.0))
            risk = float(np.clip(np.random.beta(2.0, max(0.1, (2.0 / max(1e-3, risk_mean)) - 2.0)), 0.0, 1.0))
            compliance = float(np.clip(np.random.beta(2.0, max(0.1, (2.0 / max(1e-3, comp_mean)) - 2.0)), 0.0, 1.0))
            susc = float(np.clip(self.rng.gauss(sus_mean, sus_sd), 0.0, 1.0))
            stubborn = self.rng.random() < stubborn_fraction
            household_id = G.nodes[u].get("household_id", -1)
            income = float(np.random.lognormal(mean=math.log(max(1e-3, float(self.cfg.get("income_mean", 3000.0)))), sigma=float(self.cfg.get("income_dispersion", 0.5))))
            person = Person(
                id=u,
                age_group=age_group,
                trust_in_authority=trust,
                risk_perception=risk,
                social_influence_susceptibility=susc,
                compliance_propensity=compliance,
                mask_owned_count=0,
                is_wearing_mask=False,
                fatigue_level=0.0,
                daily_contacts=max(5, int(self.rng.gauss(10, 3))),
                info_true_level=0.0,
                info_misinfo_level=0.0,
                stubborn=stubborn,
                household_id=household_id,
                income=income,
                habit_strength=0.0,
                cumulative_spend=0.0,
            )
            people.append(person)

        # Assign tie strengths as edge weights
        ew_mean = float(self.cfg.get("edge_weight_mean", 1.0))
        ew_sd = float(self.cfg.get("edge_weight_sd", 0.3))
        for a, b in G.edges():
            w = float(np.clip(self.rng.gauss(ew_mean, ew_sd), 0.1, 3.0))
            G[a][b]["weight"] = w

        return G, people, households


class Simulation:
    """
    Main simulation class coordinating mask-specific daily behavior, retail supply, policy, media, locations, and disease.

    Methods:
        run: Execute the simulation loop.
        evaluate: Compute spec-aligned metrics and validation.
        visualize: Create simple plots of key series.
        save_results: Save results to a CSV file.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation with a configuration, merging with defaults,
        normalizing keys to match the specification, and seeding RNGs.

        Args:
            cfg: Optional configuration overrides.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = self._default_config()
        if cfg:
            self.cfg.update(cfg)
        # FIXED: Normalize config keys for backward compatibility (population_size, topology, etc.)
        self._normalize_config()

        seed = int(self.cfg.get("random_seed", self.cfg.get("seed", 42)))
        random.seed(seed)
        np.random.seed(seed)
        self.rng = random.Random(seed)

        # Build network and agents
        self.netgen = NetworkGenerator(self.cfg, self.rng)
        self.G, self.people, self.households = self.netgen.build()
        # FIXED: Cache neighbor lists once to improve performance
        self.neighbors: Dict[int, List[int]] = {i: list(self.G.neighbors(i)) for i in self.G.nodes}

        # Initialize retailers and policy
        self.retailers: List[Retailer] = self._init_retailers()
        self.policy = PolicyAuthority(
            mandate_day=int(self.cfg.get("mandate_day", self.cfg.get("mandate_start_day", 9999))),
            enforcement_level=float(self.cfg.get("mandate_enforcement_level", self.cfg.get("mandate_enforcement_prob", 0.0))),
            fine_amount=float(self.cfg.get("compliance_penalty", self.cfg.get("noncompliance_penalty", 0.0))),
            communication_frequency=float(self.cfg.get("communication_frequency", 1.0)),
        )

        # Derived flags
        self.with_supply: bool = bool(self.cfg.get("with_supply", True))
        self.with_disease: bool = bool(self.cfg.get("with_disease", True))

        # Initialize media outlets
        self.media_outlets: List[MediaOutlet] = self._init_media_outlets()

        # Initialize locations
        self.locations_by_type: Dict[str, List[Location]] = self._init_locations()
        self._assign_roles_and_locations()

        # Initialize personal inventories and initial wearing
        self._init_inventories_and_wearing()

        # Initialize disease process
        self.disease: Optional[DiseaseProcess] = self._init_disease()

        # Series storage
        self.series: Dict[str, List[float]] = defaultdict(list)
        self._init_series()

        # Track per-person cumulative cost for average computation
        self._per_person_costs: List[float] = [0.0 for _ in self.people]

    def _default_config(self) -> Dict[str, Any]:
        """
        Return a dictionary of default configuration values aligned to the mask adoption spec.

        Returns:
            Default configuration dict.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        return {
            # Core timeframe
            "simulation_days": 120,
            "random_seed": 42,
            # Population and network (align to spec)
            "population_size": 5000,
            "network_topology": "small_world",
            "average_degree": 10,
            "rewiring_prob_small_world": 0.05,
            # Behavior and social weights
            "initial_adoption_rate": 0.25,
            "trust_in_authorities_mean": 0.6,
            "risk_perception_initial_mean": 0.4,
            "compliance_propensity_mean": 0.6,
            "social_influence_susceptibility_mean": 0.5,
            "social_influence_susceptibility_sd": 0.15,
            "fatigue_rate": 0.01,
            "habit_formation_rate": 0.05,
            "habit_decay_rate": 0.01,
            "mask_discomfort_cost": 0.1,
            "household_norm_weight": 0.3,
            # Policy and enforcement
            "mandate_day": 30,
            "mandate_enforcement_level": 0.7,
            "compliance_penalty": 50.0,
            "communication_frequency": 1.0,
            # Supply & pricing
            "with_supply": True,
            "retailer_count": 30,
            "initial_inventory_per_capita": 2.0,
            "restock_rate_per_day": 0.2,
            "price_per_mask": 1.0,
            "price_adjustment_sensitivity": 0.5,
            "mask_replacement_interval_days": 7,
            "purchase_threshold": 1,
            # Media defaults
            "media_effect_weight": 0.2,
            "media_outlets": [
                {"message_bias": 0.6, "misinformation_rate": 0.1, "reach": 0.6, "credibility": 0.7}
            ],
            # Locations and contacts
            "location_counts": {"workplace": 200, "school": 50, "public_space": 120},
            "location_enforcement_distribution": {"workplace": 0.7, "school": 0.8, "public_space": 0.5},
            "contact_rate_by_location": {"home": 3.0, "workplace": 6.0, "school": 8.0, "public_space": 4.0},
            # Disease process parameters
            "with_disease": True,
            "initial_infected_fraction": 0.01,
            "base_transmission_rate": 0.04,
            "mask_inward_efficacy": 0.5,
            "mask_outward_efficacy": 0.5,
            "location_transmission_modifiers": {"home": 1.0, "workplace": 0.8, "school": 0.9, "public_space": 0.6},
            "incubation_mean_days": 4.0,
            "infectious_mean_days": 6.0,
            # Validation and targets
            "target_adoption_threshold": 0.7,
            "validation_window_days": 7,
            # Visualization
            "smoothing_window_days": 3,
            # Demographics (age splits)
            "share_age_0_17": 0.2,
            "share_age_18_34": 0.35,
            "share_age_35_64": 0.3,
            "share_age_65_plus": 0.15,
            # Edge weights
            "edge_weight_mean": 1.0,
            "edge_weight_sd": 0.3,
            # Disparity and stubbornness
            "stubborn_fraction": 0.05,
            # Household influence
            "household_cluster_prob": 0.3,
            # Dynamic evaluation metric handling (optional)
            "evaluation_metrics": [],
            # Neighbor correlation computation
            "compute_neighbor_corr": False,
            "neighbor_corr_frequency_days": 7,
        }

    def _normalize_config(self) -> None:
        """
        Normalize configuration keys to the spec's naming while maintaining backward compatibility.

        Returns:
            None. Modifies self.cfg in-place.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        cfg = self.cfg
        # Map if old keys provided
        if "n_agents" in cfg and "population_size" not in cfg:
            cfg["population_size"] = cfg["n_agents"]
        if "network_type" in cfg and "network_topology" not in cfg:
            nt = str(cfg["network_type"])
            cfg["network_topology"] = "small_world" if "watt" in nt else nt
        if "avg_degree" in cfg and "average_degree" not in cfg:
            cfg["average_degree"] = cfg["avg_degree"]
        if "mandate_start_day" in cfg and "mandate_day" not in cfg:
            cfg["mandate_day"] = cfg["mandate_start_day"]
        if "mandate_enforcement_prob" in cfg and "mandate_enforcement_level" not in cfg:
            cfg["mandate_enforcement_level"] = cfg["mandate_enforcement_prob"]
        if "noncompliance_penalty" in cfg and "compliance_penalty" not in cfg:
            cfg["compliance_penalty"] = cfg["noncompliance_penalty"]
        if "daily_replenishment_mean" in cfg and "restock_rate_per_day" not in cfg:
            cfg["restock_rate_per_day"] = cfg["daily_replenishment_mean"]

    def _init_media_outlets(self) -> List[MediaOutlet]:
        """
        Initialize media outlets from configuration.

        Returns:
            List of MediaOutlet instances.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        outlets_cfg = self.cfg.get("media_outlets", [])
        outlets: List[MediaOutlet] = []
        for m in outlets_cfg:
            try:
                outlets.append(
                    MediaOutlet(
                        message_bias=float(m.get("message_bias", 0.5)),
                        misinformation_rate=float(m.get("misinformation_rate", 0.1)),
                        reach=float(m.get("reach", 0.5)),
                        credibility=float(m.get("credibility", 0.7)),
                    )
                )
            except Exception:
                continue
        if not outlets:
            outlets.append(MediaOutlet())
        return outlets

    def _init_locations(self) -> Dict[str, List[Location]]:
        """
        Initialize locations by type based on configuration.

        Returns:
            Dictionary mapping location type to list of Location objects.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        locs: Dict[str, List[Location]] = {"home": [], "workplace": [], "school": [], "public_space": []}
        # Homes: one per household
        for hid in self.households.keys():
            locs["home"].append(Location(id=hid, type="home", capacity=0, mask_requirement=False, enforcement_level=0.0))
        counts_cfg = self.cfg.get("location_counts", {"workplace": 200, "school": 50, "public_space": 120})
        enforce_cfg = self.cfg.get("location_enforcement_distribution", {"workplace": 0.7, "school": 0.8, "public_space": 0.5})
        for ltype in ("workplace", "school", "public_space"):
            count = int(counts_cfg.get(ltype, 0))
            enf = float(enforce_cfg.get(ltype, 0.5))
            for i in range(count):
                locs[ltype].append(Location(id=i, type=ltype, capacity=0, mask_requirement=False, enforcement_level=enf))
        return locs

    def _assign_roles_and_locations(self) -> None:
        """
        Assign each person to a schedule role and to a primary workplace or school location.

        Returns:
            None. Updates person attributes in-place.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        workplaces = self.locations_by_type.get("workplace", [])
        schools = self.locations_by_type.get("school", [])
        # Assign work_location_id based on age
        for p in self.people:
            if p.age_group == "0_17":
                if schools:
                    p.work_location_id = random.choice(schools).id
                else:
                    p.work_location_id = -1
            elif p.age_group in ("18_34", "35_64"):
                if workplaces:
                    p.work_location_id = random.choice(workplaces).id
                else:
                    p.work_location_id = -1
            else:
                p.work_location_id = -1

    def _init_inventories_and_wearing(self) -> None:
        """
        Initialize personal mask inventories and initial wearing states.

        Returns:
            None. Updates people in-place.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        per_cap = float(self.cfg.get("initial_inventory_per_capita", 2.0))
        init_adopt = float(self.cfg.get("initial_adoption_rate", 0.1))
        # Distribute initial inventories roughly Poisson(per_cap)
        # FIXED: Seeded numpy RNG already set in __init__ for reproducibility
        for p in self.people:
            p.mask_owned_count = max(0, int(np.random.poisson(max(0.0, per_cap))))
        # Initialize initial wearing consistent with available masks
        for p in self.people:
            p.is_wearing_mask = (p.mask_owned_count > 0) and (self.rng.random() < init_adopt)
            p.habit_strength = 0.3 if p.is_wearing_mask else 0.0

    def _init_retailers(self) -> List[Retailer]:
        """
        Initialize a list of retailers with initial inventory and pricing.

        Returns:
            List of Retailer instances.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        count = max(1, int(self.cfg.get("retailer_count", 30)))
        pop = int(self.cfg.get("population_size", 5000))
        per_cap = float(self.cfg.get("initial_inventory_per_capita", 2.0))
        total_stock = max(0, int(round(per_cap * pop)))
        base_price = float(self.cfg.get("price_per_mask", 1.0))
        restock_rate = float(self.cfg.get("restock_rate_per_day", 0.2))
        price_sens = float(self.cfg.get("price_adjustment_sensitivity", 0.5))

        # Equal initial allocation
        per_retail = total_stock // count
        retailers: List[Retailer] = []
        for i in range(count):
            inv = per_retail + (1 if i < (total_stock % count) else 0)
            r = Retailer(
                id=i,
                inventory=inv,
                price_per_mask=base_price,
                base_price=base_price,
                restock_rate=restock_rate,
                backlog=0,
                price_adjustment_sensitivity=price_sens,
            )
            retailers.append(r)
        return retailers

    def _init_disease(self) -> Optional[DiseaseProcess]:
        """
        Initialize the DiseaseProcess if enabled.

        Returns:
            DiseaseProcess instance or None if disease disabled.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.with_disease:
            return None
        N = len(self.people)
        disease = DiseaseProcess(
            N=N,
            base_transmission_rate=float(self.cfg.get("base_transmission_rate", 0.04)),
            incubation_mean=float(self.cfg.get("incubation_mean_days", 4.0)),
            infectious_mean=float(self.cfg.get("infectious_mean_days", 6.0)),
            mask_inward_efficacy=float(self.cfg.get("mask_inward_efficacy", 0.5)),
            mask_outward_efficacy=float(self.cfg.get("mask_outward_efficacy", 0.5)),
            contact_modifiers=dict(self.cfg.get("location_transmission_modifiers", {"home": 1.0, "workplace": 0.8, "school": 0.9, "public_space": 0.6})),
            rng=self.rng,
            initial_infected_fraction=float(self.cfg.get("initial_infected_fraction", 0.01)),
        )
        return disease

    def _neighbor_mask_norm(self, person_id: int, prev_states: List[bool]) -> float:
        """
        Compute the share of neighbors wearing masks (based on previous day states).

        Args:
            person_id: Person index.
            prev_states: List of bool states for each person from prior day.

        Returns:
            Share of neighbors wearing masks in [0, 1].
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # FIXED: Use cached neighbors to avoid repeated graph traversal
        neighbors = self.neighbors.get(person_id, [])
        if not neighbors:
            return 0.0
        wearing = sum(1 for n in neighbors if prev_states[n])
        return wearing / float(len(neighbors))

    def _household_mask_norm(self, person_id: int, prev_states: List[bool]) -> float:
        """
        Compute the share of household members wearing masks yesterday.

        Args:
            person_id: Person index.
            prev_states: List of bool states for each person from prior day.

        Returns:
            Float share in [0, 1].
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        hid = self.people[person_id].household_id
        members = self.households.get(hid, [])
        if not members:
            return 0.0
        wearing = sum(1 for m in members if prev_states[m])
        return wearing / float(len(members))

    def _income_deciles(self) -> List[List[int]]:
        """
        Partition agents into exactly 10 income-based decile bins covering all individuals.

        Returns:
            List of lists of agent indices, one per decile.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        N = len(self.people)
        if N <= 0:
            return []
        idxs = sorted(range(N), key=lambda i: self.people[i].income)
        deciles: List[List[int]] = []
        for d in range(10):
            start = (d * N) // 10
            end = ((d + 1) * N) // 10
            deciles.append(idxs[start:end])
        return deciles

    def _mask_gini_by_income_deciles(self, states: List[bool]) -> float:
        """
        Compute Gini coefficient across income deciles using mask wearing rates per decile.

        Args:
            states: List of bool mask wearing states today.

        Returns:
            Gini coefficient in [0, 1].
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        deciles = self._income_deciles()
        if not deciles:
            return 0.0
        rates: List[float] = []
        for bin_idx in deciles:
            if not bin_idx:
                rates.append(0.0)
                continue
            r = sum(1 for i in bin_idx if states[i]) / float(len(bin_idx))
            rates.append(max(0.0, r))
        return gini(rates)

    def _any_stockout_today(self) -> int:
        """
        Determine if any retailer is stocked out today.

        Returns:
            1 if any retailer has zero inventory, else 0.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        return 1 if any(r.total_inventory() <= 0 for r in self.retailers) else 0

    def _retailer_stockout_share(self) -> float:
        """
        Compute share of retailers stocked out (inventory <= 0).

        Returns:
            Share in [0, 1].
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.retailers:
            return 0.0
        out = sum(1 for r in self.retailers if r.total_inventory() <= 0)
        return out / float(len(self.retailers))

    def _init_series(self) -> None:
        """
        Initialize time series containers and compute initial derived metrics for day 0 (pre-step).

        Returns:
            None
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        states0 = [p.is_wearing_mask for p in self.people]
        adoption0 = (sum(1 for s in states0 if s) / float(len(states0) or 1))
        self.series["adoption_rate_over_time"].append(adoption0)
        self.series["mandate_active"].append(1 if int(0) >= int(self.cfg.get("mandate_day", 9999)) else 0)
        self.series["masks_sold_over_time"].append(0)
        # FIXED: Always append retailer_stockout_share (0.0 when supply disabled) each day
        if self.with_supply:
            self.series["retailer_stockout_share"].append(self._retailer_stockout_share())
            self.series["any_stockout"].append(self._any_stockout_today())
        else:
            self.series["retailer_stockout_share"].append(0.0)
            self.series["any_stockout"].append(0)
        self.series["daily_gini"].append(self._mask_gini_by_income_deciles(states0))
        self.series["average_price_over_time"].append(self._average_retail_price())
        self.series["cumulative_cost_per_person"].append(0.0)
        # FIXED: Initialize new series for mandate enforcement metrics
        self.series["noncompliance_events"].append(0)
        self.series["policy_costs_collected_daily"].append(0.0)
        # FIXED: Initialize disease series if enabled
        if self.with_disease and self.disease is not None:
            self.series["new_infections_over_time"].append(0)
            self.series["prevalence_over_time"].append(0.0)
            self.series["cumulative_infections_over_time"].append(int(self.disease.cumulative_infections))
            self.series["effective_reproduction_number_Re"].append(0.0)
        else:
            self.series["new_infections_over_time"].append(0)
            self.series["prevalence_over_time"].append(0.0)
            self.series["cumulative_infections_over_time"].append(0)
            self.series["effective_reproduction_number_Re"].append(0.0)
        # FIXED: Initialize per-location adoption series
        for ltype in ("home", "workplace", "school", "public_space"):
            self.series[f"adoption_rate_by_location_type_{ltype}"].append(0.0)

    def _average_retail_price(self) -> float:
        """
        Compute the average retail price across retailers (unweighted).

        Returns:
            Mean price.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.retailers:
            return 0.0
        return float(np.mean([r.price_per_mask for r in self.retailers]))

    def _attend_locations_today(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Create today's attendance for each location type and location id, listing person ids.

        Returns:
            Nested dict: {location_type: {location_id: [person_ids]}}
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        attendance: Dict[str, Dict[int, List[int]]] = {"home": defaultdict(list), "workplace": defaultdict(list), "school": defaultdict(list), "public_space": defaultdict(list)}
        # Everyone attends home
        for hid, members in self.households.items():
            attendance["home"][hid].extend(members)
        # Workers and students
        for p in self.people:
            if p.age_group == "0_17":
                # student
                if self.locations_by_type.get("school"):
                    attendance["school"][max(0, p.work_location_id)].append(p.id)
            elif p.age_group in ("18_34", "35_64"):
                # worker
                if self.locations_by_type.get("workplace"):
                    attendance["workplace"][max(0, p.work_location_id)].append(p.id)
            # Everyone may visit a public space with some probability
            if self.locations_by_type.get("public_space"):
                if self.rng.random() < 0.6:
                    ps = random.choice(self.locations_by_type["public_space"])
                    attendance["public_space"][ps.id].append(p.id)
        return attendance

    def _host_contacts_and_transmissions(self, attendance: Dict[str, Dict[int, List[int]]], mandate_active: bool) -> Tuple[int, Dict[str, float]]:
        """
        For each location, generate contacts among attendees and apply disease transmission based on mask states.

        Args:
            attendance: Nested dict of attendees per location.
            mandate_active: Whether a mandate is in effect (for enforcement/deny-entry).

        Returns:
            Tuple of (new_infections_today, adoption_by_loc_type_today dict).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.with_disease or self.disease is None:
            # Compute adoption by location type even if no disease
            adoption_loc: Dict[str, float] = {}
            states = [p.is_wearing_mask for p in self.people]
            for ltype, locs in attendance.items():
                attendees = [pid for _, lst in locs.items() for pid in lst]
                if attendees:
                    rate = sum(1 for pid in attendees if states[pid]) / float(len(attendees))
                else:
                    rate = 0.0
                adoption_loc[ltype] = float(rate)
            return 0, adoption_loc

        contact_rates = self.cfg.get("contact_rate_by_location", {"home": 3.0, "workplace": 6.0, "school": 8.0, "public_space": 4.0})
        adoption_loc: Dict[str, float] = {}
        states_mask = [p.is_wearing_mask for p in self.people]
        new_infections_set: set = set()
        # For deny-entry logic, we simulate minimal effect: if location requires mask and person not wearing, with probability enforcement deny entry
        for ltype, locs in attendance.items():
            all_attendees = [pid for _, lst in locs.items() for pid in lst]
            # Compute adoption rate among allowed attendees
            allowed_attendees_by_loc: Dict[int, List[int]] = {}
            for loc_id, attendees in locs.items():
                allowed = []
                loc_enf = 0.0
                # get location object enforcement level
                if self.locations_by_type.get(ltype):
                    for loc in self.locations_by_type[ltype]:
                        if loc.id == loc_id:
                            loc_enf = float(loc.enforcement_level)
                            break
                for pid in attendees:
                    wearing = states_mask[pid]
                    if mandate_active and not wearing and self.rng.random() < (0.5 * loc_enf):
                        # denied entry
                        continue
                    allowed.append(pid)
                allowed_attendees_by_loc[loc_id] = allowed
            # Adoption among allowed attendees
            allowed_all = [pid for _, lst in allowed_attendees_by_loc.items() for pid in lst]
            if allowed_all:
                adoption_loc[ltype] = float(sum(1 for pid in allowed_all if states_mask[pid]) / float(len(allowed_all)))
            else:
                adoption_loc[ltype] = 0.0

            # Contacts and transmissions
            lam = float(contact_rates.get(ltype, 2.0))
            lam = max(0.0, lam)
            for loc_id, attendees in allowed_attendees_by_loc.items():
                if len(attendees) <= 1:
                    continue
                # For each attendee, sample a number of contacts
                for a in attendees:
                    c = int(np.random.poisson(lam))
                    # cap contacts for performance
                    c = int(min(c, 20))
                    for _ in range(c):
                        b = random.choice(attendees)
                        if b == a:
                            continue
                        ia = self.disease.state[a]
                        ib = self.disease.state[b]
                        # Transmission attempts both directions
                        if ia == "I" and ib == "S":
                            p_trans = self.disease.transmit_given_contact(src_masked=states_mask[a], dst_masked=states_mask[b], location_type=ltype)
                            if self.rng.random() < p_trans:
                                new_infections_set.add(b)
                        elif ib == "I" and ia == "S":
                            p_trans = self.disease.transmit_given_contact(src_masked=states_mask[b], dst_masked=states_mask[a], location_type=ltype)
                            if self.rng.random() < p_trans:
                                new_infections_set.add(a)

        new_infections_today = self.disease.apply_infections(list(new_infections_set))
        return new_infections_today, adoption_loc

    def run(self) -> Dict[str, List[float]]:
        """
        Execute the simulation over the configured number of days with mask-specific dynamics,
        including media influence, mandate enforcement with fines, location attendance and disease transmission.

        Returns:
            Dictionary of time series results.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        days = int(self.cfg.get("simulation_days", 120))
        replace_interval = int(self.cfg.get("mask_replacement_interval_days", 7))
        threshold = int(self.cfg.get("purchase_threshold", 1))
        fatigue_rate = float(self.cfg.get("fatigue_rate", 0.01))
        habit_form = float(self.cfg.get("habit_formation_rate", 0.05))
        habit_decay = float(self.cfg.get("habit_decay_rate", 0.01))
        media_effect_weight = float(self.cfg.get("media_effect_weight", 0.2))
        household_weight = float(self.cfg.get("household_norm_weight", 0.3))

        prev_states = [p.is_wearing_mask for p in self.people]
        I_prev = 0
        for t in range(1, days + 1):
            signals = self.policy.step(t)
            mandate_active = bool(signals.get("mandate_active", False))
            enforcement_level = float(signals.get("enforcement_level", 0.0))
            fine_amount = float(signals.get("fine_amount", 0.0))

            # Media broadcast
            for outlet in self.media_outlets:
                outlet.broadcast(self.people, media_effect_weight=media_effect_weight, rng=self.rng)

            # Retail: restock and adjust price
            masks_sold_today = 0
            if self.with_supply:
                for r in self.retailers:
                    r.restock_daily()
                    r.adjust_price()
            # Purchases and replacement
            for p in self.people:
                p.replace_mask(t, replace_interval)
                if self.with_supply:
                    bought = p.purchase_mask(self.retailers, threshold=threshold, rng=self.rng)
                    masks_sold_today += bought
                else:
                    # Supply disabled: assume free/unlimited supply at zero cost for purchase attempts
                    if p.mask_owned_count <= threshold:
                        p.mask_owned_count += 1

            # Daily wearing decision after purchases
            noncompliance_events = 0
            policy_costs_collected = 0.0
            for p in self.people:
                sn = self._neighbor_mask_norm(p.id, prev_states)
                hn = self._household_mask_norm(p.id, prev_states)
                p.decide_wear_mask(
                    social_norm=sn,
                    mandate_active=mandate_active,
                    enforcement_level=enforcement_level,
                    household_norm=hn,
                    household_weight=household_weight,
                )
                # FIXED: Explicit noncompliance detection and fines per feedback
                if mandate_active and not p.is_wearing_mask:
                    if random.random() < enforcement_level:
                        noncompliance_events += 1
                        fine = float(fine_amount)
                        p.cumulative_spend += fine
                        policy_costs_collected += fine
                # Update fatigue and habit
                if p.is_wearing_mask:
                    p.habit_strength = clip01(p.habit_strength + habit_form * (1.0 - p.habit_strength))
                    p.fatigue_level = clip01(p.fatigue_level + fatigue_rate * 0.5)
                else:
                    p.habit_strength = clip01(p.habit_strength * (1.0 - habit_decay))
                    p.fatigue_level = clip01(p.fatigue_level + fatigue_rate * 1.0)
            # store daily enforcement counts
            self.series["noncompliance_events"].append(noncompliance_events)
            self.series["policy_costs_collected_daily"].append(policy_costs_collected)

            # Attendance and disease transmission
            attendance = self._attend_locations_today()
            new_infections_today, adoption_by_loc_type_today = self._host_contacts_and_transmissions(attendance, mandate_active=mandate_active)
            # Progress disease states at end of day
            if self.with_disease and self.disease is not None:
                self.disease.record_daily_metrics(I_prev=I_prev, new_infections=new_infections_today)
                S, E, I, R = self.disease.progress()
                I_prev = I

            # Compute today's states and metrics
            states = [p.is_wearing_mask for p in self.people]
            adoption = (sum(1 for s in states if s) / float(len(states) or 1))
            self.series["adoption_rate_over_time"].append(adoption)
            self.series["mandate_active"].append(1 if mandate_active else 0)
            self.series["masks_sold_over_time"].append(masks_sold_today)
            # Stockouts
            if self.with_supply:
                self.series["retailer_stockout_share"].append(self._retailer_stockout_share())
                self.series["any_stockout"].append(self._any_stockout_today())
            else:
                # FIXED: Maintain alignment by appending zeros when supply disabled
                self.series["retailer_stockout_share"].append(0.0)
                self.series["any_stockout"].append(0)
            self.series["daily_gini"].append(self._mask_gini_by_income_deciles(states))
            self.series["average_price_over_time"].append(self._average_retail_price())

            # Disease series
            if self.with_disease and self.disease is not None:
                self.series["new_infections_over_time"].append(
                    int(self.disease.new_infections_daily[-1] if self.disease.new_infections_daily else new_infections_today)
                )
                self.series["prevalence_over_time"].append(
                    float(self.disease.prevalence_over_time[-1] if self.disease.prevalence_over_time else 0.0)
                )
                self.series["cumulative_infections_over_time"].append(
                    int(self.disease.cumulative_infections_over_time[-1] if self.disease.cumulative_infections_over_time else self.disease.cumulative_infections)
                )
                self.series["effective_reproduction_number_Re"].append(
                    float(self.disease.Re_over_time[-1] if self.disease.Re_over_time else 0.0)
                )
            else:
                self.series["new_infections_over_time"].append(0)
                self.series["prevalence_over_time"].append(0.0)
                self.series["cumulative_infections_over_time"].append(
                    int(self.series["cumulative_infections_over_time"][-1]) if self.series["cumulative_infections_over_time"] else 0
                )
                self.series["effective_reproduction_number_Re"].append(0.0)

            # Per-location adoption series
            for ltype in ("home", "workplace", "school", "public_space"):
                self.series[f"adoption_rate_by_location_type_{ltype}"].append(float(adoption_by_loc_type_today.get(ltype, 0.0)))

            # Average cumulative cost per person
            avg_cost = sum(p.cumulative_spend for p in self.people) / float(len(self.people) or 1)
            self.series["cumulative_cost_per_person"].append(float(avg_cost))

            # Prepare for next day
            prev_states = states

        return self.series

# Execute main for both direct execution and sandbox wrapper invocation
main()