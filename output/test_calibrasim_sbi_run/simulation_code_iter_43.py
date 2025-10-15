import json
import logging
import math
import os
import random
import sys
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# FIXED: Defer matplotlib import into visualize() to avoid ImportError at import time
# # FIXED: Applied feedback snippet from simulation.py
# defer import to visualize()
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

    Args:
        lam: Expected rate (lambda).
        rng: Python random.Random instance for reproducibility.

    Returns:
        Non-negative integer sample from Poisson(lam).
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    lam = max(0.0, lam)
    try:
        # FIXED: Use a numpy Generator seeded from provided rng to ensure reproducibility
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
    total = cumvals[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_vals) / (n * total)) - (n + 1) / n)


@dataclass
class Person:
    """
    Agent representing an individual in the social simulation, with attributes
    influencing information and behavior adoption dynamics.

    Attributes:
        id: Unique identifier of the person.
        age_group: Age category string.
        trust_in_institutions: Trust scalar in [0, 1].
        risk_perception: Perceived risk scalar in [0, 1].
        susceptibility_to_misinfo: Susceptibility scalar in [0, 1].
        adoption_state: 1 if adopted (e.g., vaccinated), else 0.
        stubborn: If True, the person never adopts.
        info_true_level: Salience level for true information.
        info_misinfo_level: Salience level for misinformation.
        fatigue: Fatigue level in [0, 1] due to info overload.
        household_id: Household identifier.
        income: Income proxy used for disparity metrics.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    id: int = 0
    age_group: str = "18_34"
    trust_in_institutions: float = 0.5
    risk_perception: float = 0.4
    susceptibility_to_misinfo: float = 0.4
    adoption_state: int = 0
    stubborn: bool = False
    info_true_level: float = 0.0
    info_misinfo_level: float = 0.0
    fatigue: float = 0.0
    household_id: int = -1
    income: float = 1.0


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
        # FIXED: Respect share_age_65_plus and renormalize all shares to sum to 1.0
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
        Build the social network and initialize agent attributes.

        Returns:
            Tuple of (graph, people, households).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        N = int(self.cfg.get("n_agents", 5000))
        network_type = self.cfg.get("network_type", "watts_strogatz")
        avg_degree = max(2, int(round(float(self.cfg.get("avg_degree", 8.0)))))
        rewiring_prob = float(self.cfg.get("rewiring_prob", 0.1))
        m_ba = int(self.cfg.get("m_ba", 3))
        rng_seed = int(self.cfg.get("random_seed", 42))

        # Build base graph
        if network_type == "barabasi_albert":
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
        mean_trust = float(self.cfg.get("mean_trust", 0.5))
        trust_sd = float(self.cfg.get("trust_sd", 0.2))
        mean_risk = float(self.cfg.get("mean_risk_perception", 0.4))
        risk_sd = float(self.cfg.get("risk_sd", 0.25))
        mean_misinfo = float(self.cfg.get("mean_susceptibility_misinfo", 0.4))
        misinfo_sd = float(self.cfg.get("misinfo_sd", 0.2))
        stubborn_fraction = float(self.cfg.get("stubborn_fraction", 0.1))
        init_adopt_frac = float(self.cfg.get("initial_adoption_fraction", 0.05))
        init_true_frac = float(self.cfg.get("initial_true_info_awareness_fraction", 0.1))
        init_misinfo_frac = float(self.cfg.get("initial_misinfo_belief_fraction", 0.1))

        people: List[Person] = []
        for u in range(N):
            age_group = self._assign_age_group(self.rng.random())
            trust = float(np.clip(self.rng.gauss(mean_trust, trust_sd), 0.0, 1.0))
            risk = float(np.clip(self.rng.gauss(mean_risk, risk_sd), 0.0, 1.0))
            susc = float(np.clip(self.rng.gauss(mean_misinfo, misinfo_sd), 0.0, 1.0))
            stubborn = self.rng.random() < stubborn_fraction
            adoption_state = 1 if (self.rng.random() < init_adopt_frac) and not stubborn else 0
            info_true_level = 1.0 if self.rng.random() < init_true_frac else 0.0
            info_misinfo_level = 1.0 if self.rng.random() < init_misinfo_frac else 0.0
            household_id = G.nodes[u].get("household_id", -1)
            # Income as lognormal proxy (seeded by np.random.seed in Simulation for reproducibility)
            income = float(np.random.lognormal(mean=math.log(30000), sigma=0.6))
            person = Person(
                id=u,
                age_group=age_group,
                trust_in_institutions=trust,
                risk_perception=risk,
                susceptibility_to_misinfo=susc,
                adoption_state=adoption_state,
                stubborn=stubborn,
                info_true_level=info_true_level,
                info_misinfo_level=info_misinfo_level,
                fatigue=0.0,
                household_id=household_id,
                income=income,
            )
            people.append(person)

        # Assign tie strengths as edge weights
        ew_mean = float(self.cfg.get("edge_weight_mean", 1.0))
        ew_sd = float(self.cfg.get("edge_weight_sd", 0.3))
        for a, b in G.edges():
            w = float(np.clip(self.rng.gauss(ew_mean, ew_sd), 0.1, 3.0))
            G[a][b]["weight"] = w

        return G, people, households


class PolicyCampaign:
    """
    Schedules and emits policy signals that affect information and adoption dynamics.

    Signals:
        trust_boost_t
        incentive_amount_t
        mandate_active_t
        misinfo_takedown_rate_t
        mandate_enforcement_prob
        noncompliance_penalty
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            cfg: Configuration dictionary.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = cfg
        self.mandate_status: bool = False

    def step(self, t: int) -> Dict[str, Any]:
        """
        Compute policy signals for day t.

        Args:
            t: Day index.

        Returns:
            Dictionary of policy signals.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        cs = int(self.cfg.get("campaign_start_day", 7))
        ce = int(self.cfg.get("campaign_end_day", 29))
        cint = float(self.cfg.get("campaign_intensity", 0.6))
        ctb = float(self.cfg.get("campaign_trust_boost", 0.15))
        incentive_amount = float(self.cfg.get("incentive_amount", 50.0))
        misinfo_takedown_rate = float(self.cfg.get("misinformation_takedown_rate", 0.1))
        mandate_start_day = int(self.cfg.get("mandate_start_day", 9999))
        enforcement_prob = float(self.cfg.get("mandate_enforcement_prob", 0.0))
        penalty = float(self.cfg.get("noncompliance_penalty", 0.0))

        trust_boost = ctb * cint if (cs <= t <= ce) else 0.0
        incentive = incentive_amount * (cint if (cs <= t <= ce) else 0.0)
        self.mandate_status = (t >= mandate_start_day)
        misinfo_takedown = misinfo_takedown_rate * (1.0 if t >= cs else 0.0)

        return {
            "trust_boost_t": trust_boost,
            "incentive_amount_t": incentive,
            "mandate_active_t": self.mandate_status,
            "misinfo_takedown_rate_t": misinfo_takedown,
            "mandate_enforcement_prob": enforcement_prob,
            "noncompliance_penalty": penalty,
        }


class InformationPropagation:
    """
    Propagates true information and misinformation over the social network using a weighted cascade.

    Methods:
        step: Apply decay, self-initiated exposures, and neighbor sharing effects for a time step.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Dict[str, Any], graph: nx.Graph, people: List[Person], rng: random.Random):
        """
        Initialize propagation module.

        Args:
            cfg: Configuration dictionary.
            graph: Social network graph.
            people: List of Person agents.
            rng: Random number generator.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = cfg
        self.G = graph
        self.people = people
        self.rng = rng
        self.last_true_exposures: int = 0
        self.last_misinfo_exposures: int = 0

    def step(self, policy_signals: Dict[str, Any]) -> Tuple[int, int]:
        """
        Perform one day of information propagation.

        Args:
            policy_signals: Current policy signals affecting propagation.

        Returns:
            Tuple of (true_exposures_today, misinfo_exposures_today).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        decay = float(self.cfg.get("info_decay_rate", 0.15))
        base_true = float(self.cfg.get("base_share_prob_true", 0.12))
        base_m = float(self.cfg.get("base_share_prob_misinfo", 0.18))
        fatigue_inc = float(self.cfg.get("fatigue_increment_per_exposure", 0.02))
        self_seek = float(self.cfg.get("self_initiated_exposure_rate", 0.01))
        tie_infl = float(self.cfg.get("tie_strength_influence", 0.5))
        trust_sens = float(self.cfg.get("trust_sensitivity", 0.6))
        rebuttal_eff = float(self.cfg.get("rebuttal_effectiveness", 0.3))
        fatigue_alpha = float(self.cfg.get("fatigue_sharing_alpha", 0.3))  # FIXED: Fatigue attenuation factor

        trust_boost = float(policy_signals.get("trust_boost_t", 0.0))
        misinfo_takedown = float(policy_signals.get("misinfo_takedown_rate_t", 0.0))

        true_increments = [0] * len(self.people)
        mis_increments = [0] * len(self.people)

        # Decay existing levels, and self-seeking exposures
        for p in self.people:
            p.info_true_level *= max(0.0, 1.0 - decay)
            p.info_misinfo_level *= max(0.0, 1.0 - decay)
            if self.rng.random() < self_seek:
                p.info_true_level += 1.0
                true_increments[p.id] += 1
            if self.rng.random() < self_seek:
                p.info_misinfo_level += 1.0
                mis_increments[p.id] += 1

        # Neighbor sharing along edges with fatigue attenuation
        for a, b, data in self.G.edges(data=True):
            w = float(data.get("weight", 1.0))
            # Sharing from a to b
            p_a = self.people[a]
            p_b = self.people[b]
            damp_a = clip01(1.0 - fatigue_alpha * p_a.fatigue)  # FIXED: Apply fatigue damping to sharing probability
            p_share_true_ab = clip01(
                base_true
                * (1.0 + tie_infl * (w - 1.0))
                * (1.0 + trust_sens * (p_a.trust_in_institutions + trust_boost - 0.5))
                * damp_a
            )
            if p_a.info_true_level > 0 and self.rng.random() < p_share_true_ab:
                p_b.info_true_level += 1.0
                true_increments[p_b.id] += 1

            p_share_m_ab = clip01(
                base_m
                * (1.0 - misinfo_takedown)
                * (1.0 + tie_infl * (w - 1.0))
                * (1.0 + trust_sens * (0.5 - p_a.trust_in_institutions))
                * damp_a
            )
            if p_a.info_misinfo_level > 0 and self.rng.random() < p_share_m_ab:
                p_b.info_misinfo_level += (1.0 - rebuttal_eff)  # mitigated by rebuttal
                mis_increments[p_b.id] += 1

            # Sharing from b to a
            damp_b = clip01(1.0 - fatigue_alpha * p_b.fatigue)
            p_share_true_ba = clip01(
                base_true
                * (1.0 + tie_infl * (w - 1.0))
                * (1.0 + trust_sens * (p_b.trust_in_institutions + trust_boost - 0.5))
                * damp_b
            )
            if p_b.info_true_level > 0 and self.rng.random() < p_share_true_ba:
                p_a.info_true_level += 1.0
                true_increments[p_a.id] += 1

            p_share_m_ba = clip01(
                base_m
                * (1.0 - misinfo_takedown)
                * (1.0 + tie_infl * (w - 1.0))
                * (1.0 + trust_sens * (0.5 - p_b.trust_in_institutions))
                * damp_b
            )
            if p_b.info_misinfo_level > 0 and self.rng.random() < p_share_m_ba:
                p_a.info_misinfo_level += (1.0 - rebuttal_eff)
                mis_increments[p_a.id] += 1

        # Update fatigue
        for p in self.people:
            exposures_today = true_increments[p.id] + mis_increments[p.id]
            p.fatigue = clip01(p.fatigue + fatigue_inc * exposures_today)

        t_true = int(sum(true_increments))
        t_mis = int(sum(mis_increments))
        self.last_true_exposures = t_true
        self.last_misinfo_exposures = t_mis
        return t_true, t_mis


class BehaviorAdoption:
    """
    Determines daily adoption decisions using a logistic function of social norms, perceived risk,
    trust, incentives, and mandates.

    Methods:
        step: Update adoption states for all non-stubborn agents.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Dict[str, Any], graph: nx.Graph, people: List[Person], rng: random.Random):
        """
        Initialize behavior adoption module.

        Args:
            cfg: Configuration dictionary.
            graph: Social network graph.
            people: List of agents.
            rng: Random generator.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = cfg
        self.G = graph
        self.people = people
        self.rng = rng
        self.new_adopters_today: int = 0
        self.enforced_events_today: int = 0

    def _compute_social_norm(self, person_id: int) -> float:
        """
        Compute the share of adopting neighbors, sampling up to max_daily_contacts_to_consider neighbors.

        Args:
            person_id: ID of the person.

        Returns:
            Social norm value in [0, 1].
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        neighbors = list(self.G.neighbors(person_id))
        if not neighbors:
            return 0.0
        k = min(len(neighbors), int(self.cfg.get("max_daily_contacts_to_consider", 20)))
        sampled = self.rng.sample(neighbors, k) if len(neighbors) > k else neighbors
        if not sampled:
            return 0.0
        adopters = sum(self.people[n].adoption_state for n in sampled)
        return adopters / float(len(sampled))

    def step(self, policy_signals: Dict[str, Any], supply_enabled: bool = False, inventory_available: int = 0) -> int:
        """
        Update adoption states given policy signals. Returns number of new adopters.

        Args:
            policy_signals: Policy signals dictionary.
            supply_enabled: If True, supply constraints apply to adoption.
            inventory_available: Number of doses available (if supply_enabled).

        Returns:
            Number of new adopters today.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        base_logit = float(self.cfg.get("base_adoption_logit", -2.2))
        w_social = float(self.cfg.get("weight_social_norm", 2.5))
        w_risk = float(self.cfg.get("weight_risk", 1.4))
        w_trust = float(self.cfg.get("weight_trust", 0.8))
        w_incent = float(self.cfg.get("weight_cost_incentive", 0.03))
        noise_sd = float(self.cfg.get("private_signal_noise", 0.5))
        habit_drop_prob = float(self.cfg.get("habit_drop_prob", 0.0))

        trust_boost = float(policy_signals.get("trust_boost_t", 0.0))
        incentive = float(policy_signals.get("incentive_amount_t", 0.0))
        mandate_active = bool(policy_signals.get("mandate_active_t", False))
        enforcement_prob = float(policy_signals.get("mandate_enforcement_prob", 0.0))
        penalty = float(policy_signals.get("noncompliance_penalty", 0.0))

        new_adopters = 0
        enforced_events = 0
        # Iterate non-adopters
        non_adopters_idx = [p.id for p in self.people if p.adoption_state == 0 and not p.stubborn]
        self.rng.shuffle(non_adopters_idx)

        for pid in non_adopters_idx:
            p = self.people[pid]
            norm_i = self._compute_social_norm(pid)
            belief_adjust = p.info_true_level - (1.0 - float(self.cfg.get("rebuttal_effectiveness", 0.3))) * p.info_misinfo_level * p.susceptibility_to_misinfo
            risk_i = clip01(p.risk_perception + 0.1 * belief_adjust)
            trust_eff = clip01(p.trust_in_institutions + trust_boost)
            eps = self.rng.gauss(0.0, noise_sd)
            logit = base_logit + w_social * norm_i + w_risk * risk_i + w_trust * trust_eff + w_incent * incentive + eps
            p_adopt = logistic(logit)

            # Mandate enforcement targeted to non-compliant individuals
            if mandate_active:
                if self.rng.random() < enforcement_prob:
                    # Enforced; strong compliance effect
                    p_adopt = max(p_adopt, 0.95)
                    enforced_events += 1
                else:
                    # Soft penalty nudges adoption probability
                    p_adopt = clip01(p_adopt + 0.002 * penalty)

            # FIXED: Condition inventory gating on supply_enabled
            if supply_enabled and inventory_available <= 0:
                # If stockout, sharply reduce chance to adopt
                p_adopt *= 0.1  # severe constraint under stockout

            will_adopt = self.rng.random() < p_adopt

            if will_adopt:
                if supply_enabled:
                    # Allocate dose only if available
                    if inventory_available > 0:
                        p.adoption_state = 1
                        new_adopters += 1
                        inventory_available -= 1
                else:
                    p.adoption_state = 1
                    new_adopters += 1

        # Habit drop (for behaviors where decay is relevant; keep 0 for vaccination)
        if habit_drop_prob > 0.0:
            for p in self.people:
                if p.adoption_state == 1 and self.rng.random() < habit_drop_prob:
                    p.adoption_state = 0

        self.new_adopters_today = new_adopters
        self.enforced_events_today = enforced_events
        return new_adopters


class SupplyChain:
    """
    Optional supply chain module to track inventory and stockouts for the behavior (e.g., vaccine doses).

    Methods:
        allocate_daily_supply: Add daily replenishment to inventory.
        consume: Reduce inventory by the number of adopters.
        stockout_share: Proportion of retailers (or analogous outlets) in stockout state.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Dict[str, Any], rng: random.Random):
        """
        Initialize supply parameters.

        Args:
            cfg: Configuration dictionary.
            rng: RNG instance.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = cfg
        self.rng = rng
        self.enabled: bool = bool(cfg.get("with_supply", False))
        # Model a simple centralized inventory for now; extendable to multiple retailers
        self.inventory: int = int(cfg.get("initial_inventory", 0))
        self.daily_replenishment_mean: float = float(cfg.get("daily_replenishment_mean", 0.0))
        self.any_stockout_today: int = 0

    def allocate_daily_supply(self) -> int:
        """
        Allocate today's supply arrival according to a Poisson around daily_replenishment_mean.

        Returns:
            New inventory level after replenishment.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.enabled:
            self.any_stockout_today = 0
            return self.inventory
        arrivals = sample_poisson(self.daily_replenishment_mean, self.rng)
        self.inventory += int(arrivals)
        self.any_stockout_today = 1 if self.inventory <= 0 else 0
        return self.inventory

    def consume(self, n: int) -> int:
        """
        Consume inventory by n units. Inventory cannot go below zero.

        Args:
            n: Units to consume.

        Returns:
            Actual units consumed (may be lower than n if stockout).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.enabled:
            return n
        consumed = min(self.inventory, max(0, int(n)))
        self.inventory -= consumed
        if self.inventory <= 0:
            self.any_stockout_today = 1
        return consumed

    def stockout_share(self) -> float:
        """
        Return the stockout share indicator. With centralized inventory, return 1 if stockout else 0.

        Returns:
            Float in [0, 1] indicating share of outlets stockout.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if not self.enabled:
            return 0.0
        return 1.0 if self.inventory <= 0 else 0.0


class AggregationAndMetrics:
    """
    Aggregates daily observables and applies optional smoothing and reporting adjustments.

    Methods:
        update: Accumulate daily observables.
        finalize: Apply smoothing and delays to produce observable series.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize aggregator parameters.

        Args:
            cfg: Configuration dict.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = cfg
        self.buffer_delay_days = int(cfg.get("reporting_delay_days", 0))
        self.underreporting_factor = float(cfg.get("underreporting_factor", 1.0))
        self.smoothing_window_days = int(cfg.get("smoothing_window_days", 3))
        self.new_adoptions_buffer: deque = deque()
        self.daily_new_raw: List[float] = []
        self.cumulative_adopters: List[int] = []
        self.true_info_exposures: List[int] = []
        self.misinfo_exposures: List[int] = []

    def update(self, new_adopters: int, cumulative_adopters: int, true_exposures: int, misinfo_exposures: int) -> None:
        """
        Update the aggregator state for a single day.

        Args:
            new_adopters: Number of new adopters today.
            cumulative_adopters: Total adopters so far.
            true_exposures: True info exposures today.
            misinfo_exposures: Misinfo exposures today.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        report_new = new_adopters
        if self.buffer_delay_days > 0:
            self.new_adoptions_buffer.append(report_new)
            if len(self.new_adoptions_buffer) > self.buffer_delay_days:
                report_new = self.new_adoptions_buffer.popleft()
            else:
                report_new = 0

        # Underreporting
        obs_new = round(report_new * self.underreporting_factor)
        obs_cum = int(cumulative_adopters * self.underreporting_factor)
        self.daily_new_raw.append(obs_new)
        self.cumulative_adopters.append(obs_cum)
        self.true_info_exposures.append(true_exposures)
        self.misinfo_exposures.append(misinfo_exposures)

    def finalize(self, n_agents: int) -> Dict[str, List[float]]:
        """
        Finalize observable series with smoothing.

        Args:
            n_agents: Total population size.

        Returns:
            Dict of observable series.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        smoothed_new = moving_average(self.daily_new_raw, self.smoothing_window_days)
        cum_share = [c / float(n_agents) if n_agents > 0 else 0.0 for c in self.cumulative_adopters]
        misinfo_idx = [m / float(n_agents) if n_agents > 0 else 0.0 for m in self.misinfo_exposures]
        true_idx = [t / float(n_agents) if n_agents > 0 else 0.0 for t in self.true_info_exposures]
        return {
            "observable.new_adoptions_daily": smoothed_new,
            "observable.cumulative_adoption_share": cum_share,
            "observable.misinfo_exposure_index": misinfo_idx,
            "observable.true_info_exposure_index": true_idx,
        }


class Simulation:
    """
    Main simulation class coordinating modules and time progression.

    Methods:
        run: Execute the simulation loop.
        evaluate: Compute metrics and summary statistics.
        visualize: Create simple plots of key series.
        save_results: Save results to a CSV file.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation with a configuration, merging with defaults.

        Args:
            cfg: Optional configuration overrides.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        self.cfg = self._default_config()
        if cfg:
            self.cfg.update(cfg)

        seed = int(self.cfg.get("random_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        self.rng = random.Random(seed)

        # Build network and agents
        self.netgen = NetworkGenerator(self.cfg, self.rng)
        self.G, self.people, self.households = self.netgen.build()

        # Policy and modules
        self.policy = PolicyCampaign(self.cfg)
        self.info = InformationPropagation(self.cfg, self.G, self.people, self.rng)
        self.adopt = BehaviorAdoption(self.cfg, self.G, self.people, self.rng)

        # Optional supply
        self.supply = SupplyChain(self.cfg, self.rng)
        self.with_supply = self.supply.enabled

        # Aggregation
        self.aggregator = AggregationAndMetrics(self.cfg)

        # Series storage
        self.series: Dict[str, List[float]] = defaultdict(list)
        self._init_series()

    def _default_config(self) -> Dict[str, Any]:
        """
        Return a dictionary of default configuration values from the model plan.

        Returns:
            Default configuration dict.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        return {
            "simulation_days": 60,
            "random_seed": 42,
            "time_step_days": 1,
            "initial_adoption_fraction": 0.05,
            "initial_misinfo_belief_fraction": 0.1,
            "initial_true_info_awareness_fraction": 0.1,
            "share_age_0_17": 0.2,
            "share_age_18_34": 0.35,
            "share_age_35_64": 0.3,
            "share_age_65_plus": 0.15,
            "mean_trust": 0.5,
            "trust_sd": 0.2,
            "mean_risk_perception": 0.4,
            "risk_sd": 0.25,
            "mean_susceptibility_misinfo": 0.4,
            "misinfo_sd": 0.2,
            "n_agents": 2000,
            "network_type": "watts_strogatz",
            "avg_degree": 8.0,
            "rewiring_prob": 0.1,
            "m_ba": 3,
            "homophily_age": 0.4,
            "assortativity_by_risk": 0.2,
            "household_cluster_prob": 0.3,
            "edge_weight_mean": 1.0,
            "edge_weight_sd": 0.3,
            "base_share_prob_true": 0.12,
            "base_share_prob_misinfo": 0.18,
            "info_decay_rate": 0.15,
            "fatigue_increment_per_exposure": 0.02,
            "rebuttal_effectiveness": 0.3,
            "self_initiated_exposure_rate": 0.01,
            "tie_strength_influence": 0.5,
            "trust_sensitivity": 0.6,
            "base_adoption_logit": -2.2,
            "weight_social_norm": 2.5,
            "weight_risk": 1.4,
            "weight_trust": 0.8,
            "weight_cost_incentive": 0.03,
            "private_signal_noise": 0.5,
            "stubborn_fraction": 0.1,
            "habit_drop_prob": 0.0,
            "max_daily_contacts_to_consider": 20,
            "campaign_start_day": 7,
            "campaign_end_day": 29,
            "campaign_intensity": 0.6,
            "campaign_trust_boost": 0.15,
            "incentive_amount": 50.0,
            "misinformation_takedown_rate": 0.1,
            "mandate_start_day": 9999,
            "mandate_enforcement_prob": 0.0,
            "noncompliance_penalty": 0.0,
            "smoothing_window_days": 3,
            "reporting_delay_days": 0,
            "underreporting_factor": 1.0,
            "observation_start_day": 0,
            "evaluation_start_day": 30,
            "evaluation_end_day": 39,
            "evaluation_metrics": [
                "RMSE_new_adoptions_daily",
                "MAPE_cumulative_adoption",
                "PeakError_new_adoptions",
                "TimeTo50_cumulative",
            ],
            # Optional supply parameters
            "with_supply": False,
            "initial_inventory": 0,
            "daily_replenishment_mean": 0.0,
            # FIXED: Config to control neighbor correlation computation cost
            "compute_neighbor_corr": True,
            "neighbor_corr_frequency_days": 7,
            # FIXED: Fatigue attenuation parameter for sharing probabilities
            "fatigue_sharing_alpha": 0.3,
        }

    def _init_series(self) -> None:
        """
        Initialize time series containers and compute initial derived metrics.

        Returns:
            None
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # Derived initial series at day 0 (pre-step)
        self.series["adoption_rate"].append(self._adoption_rate())
        self.series["mandate_active"].append(1 if self.policy.mandate_status else 0)
        self.series["true_info_exposures"].append(0)
        self.series["misinfo_exposures"].append(0)
        self.series["daily_new_adopters"].append(0)
        self.series["cumulative_adopters"].append(self._cumulative_adopters())
        self.series["observable.new_adoptions_daily"] = []
        self.series["observable.cumulative_adoption_share"] = []
        self.series["observable.misinfo_exposure_index"] = []
        self.series["observable.true_info_exposure_index"] = []
        self.series["retailer_stockout_share"].append(self.supply.stockout_share() if self.with_supply else 0.0)
        self.series["any_stockout"].append(1 if self.with_supply and self.supply.inventory <= 0 else 0)
        self.series["neighbor_corr"].append(self._neighbor_influence_corr() if self.cfg.get("compute_neighbor_corr", True) else 0.0)
        self.series["daily_gini"].append(self._adoption_gini_by_income_deciles())
        self.series["enforced_events"].append(0)

    def _adoption_rate(self) -> float:
        """
        Compute current adoption rate.

        Returns:
            Fraction of agents with adoption_state == 1.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        if len(self.people) == 0:
            return 0.0
        return sum(p.adoption_state for p in self.people) / float(len(self.people))

    def _cumulative_adopters(self) -> int:
        """
        Compute current cumulative adopters count.

        Returns:
            Number of agents with adoption_state == 1.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        return sum(p.adoption_state for p in self.people)

    def _neighbor_influence_corr(self) -> float:
        """
        Compute correlation between a person's adoption and neighbor-adoption share.

        Returns:
            Pearson correlation coefficient or 0.0 if undefined.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        xs = []
        ys = []
        for p in self.people:
            neighbors = list(self.G.neighbors(p.id))
            if not neighbors:
                continue
            adopters = sum(self.people[n].adoption_state for n in neighbors)
            share = adopters / float(len(neighbors))
            xs.append(p.adoption_state)
            ys.append(share)
        if len(xs) < 2:
            return 0.0
        x = np.array(xs)
        y = np.array(ys)
        try:
            corr = float(np.corrcoef(x, y)[0, 1])
            if math.isnan(corr):
                return 0.0
            return corr
        except Exception:
            return 0.0

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
        # FIXED: Compute exactly 10 contiguous bins covering all individuals
        for d in range(10):
            start = (d * N) // 10
            end = ((d + 1) * N) // 10
            deciles.append(idxs[start:end])
        return deciles

    def _adoption_gini_by_income_deciles(self) -> float:
        """
        Compute Gini coefficient across income deciles using adoption rates per decile.

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
            r = sum(self.people[i].adoption_state for i in bin_idx) / float(len(bin_idx))
            rates.append(max(0.0, r))
        return gini(rates)

    def run(self) -> Dict[str, List[float]]:
        """
        Execute the simulation over the configured number of days.

        Returns:
            Dictionary of time series results.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        days = int(self.cfg.get("simulation_days", 60))
        n_agents = len(self.people)
        compute_corr = bool(self.cfg.get("compute_neighbor_corr", True))
        corr_freq = max(1, int(self.cfg.get("neighbor_corr_frequency_days", 7)))

        for t in range(days):
            # Policy signals
            signals = self.policy.step(t)

            # Supply allocation (if enabled)
            if self.with_supply:
                self.supply.allocate_daily_supply()
                self.series["retailer_stockout_share"].append(self.supply.stockout_share())
                # Record daily any_stockout indicator for this day
                self.series["any_stockout"].append(1 if self.supply.inventory <= 0 else 0)

            # Information propagation
            true_exp, mis_exp = self.info.step(signals)

            # Behavior adoption
            inv_available = self.supply.inventory if self.with_supply else 0
            new_adopters = self.adopt.step(signals, supply_enabled=self.with_supply, inventory_available=inv_available)
            if self.with_supply:
                # Consume allocated doses equal to actual new adopters
                self.supply.consume(new_adopters)

            cum_adopters = self._cumulative_adopters()

            # Aggregation
            self.aggregator.update(new_adopters, cum_adopters, true_exp, mis_exp)

            # Update series for this step
            self.series["adoption_rate"].append(self._adoption_rate())
            self.series["mandate_active"].append(1 if self.policy.mandate_status else 0)
            self.series["true_info_exposures"].append(true_exp)
            self.series["misinfo_exposures"].append(mis_exp)
            self.series["daily_new_adopters"].append(new_adopters)
            self.series["cumulative_adopters"].append(cum_adopters)

            # FIXED: Make neighbor correlation optional and less frequent
            last_corr = self.series["neighbor_corr"][-1] if self.series["neighbor_corr"] else 0.0
            if compute_corr and (t % corr_freq == 0):
                self.series["neighbor_corr"].append(self._neighbor_influence_corr())
            else:
                self.series["neighbor_corr"].append(last_corr)

            self.series["daily_gini"].append(self._adoption_gini_by_income_deciles())
            self.series["enforced_events"].append(self.adopt.enforced_events_today)
            # FIXED: any_stockout already appended at supply allocation; do not duplicate here

        # Finalize observables
        observables = self.aggregator.finalize(n_agents)
        for k, v in observables.items():
            self.series[k] = v

        return self.series

    def evaluate(self) -> Dict[str, Any]:
        """
        Compute metrics configured in 'evaluation_metrics' from the series.

        Returns:
            Dictionary of metric results and additional summaries.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        metrics_out: Dict[str, Any] = {}
        eval_list: List[str] = list(self.cfg.get("evaluation_metrics", []))
        days = len(self.series.get("observable.cumulative_adoption_share", []))
        # Map observable series for convenience
        new_daily = list(self.series.get("observable.new_adoptions_daily", []))
        cum_share = list(self.series.get("observable.cumulative_adoption_share", []))

        # Helper computations
        def rmse(arr: List[float]) -> float:
            if not arr:
                return 0.0
            arr_np = np.array(arr, dtype=float)
            return float(math.sqrt(np.mean(np.square(arr_np))))

        def mape(arr: List[float], eps: float = 1e-6) -> float:
            if not arr:
                return 0.0
            arr_np = np.array(arr, dtype=float)
            denom = np.maximum(eps, np.abs(arr_np) + eps)
            return float(np.mean(np.abs(arr_np) / denom))

        def peak_error(arr: List[float], window: int = 7) -> float:
            if not arr:
                return 0.0
            peak_value = max(arr)
            # naive: error is 0 without target data; here we report simply the peak value for lack of external target
            return float(peak_value)

        def time_to_threshold(arr: List[float], thr: float) -> int:
            for i, v in enumerate(arr):
                if v >= thr:
                    return i
            return -1

        for name in eval_list:
            if name == "RMSE_new_adoptions_daily":
                metrics_out[name] = rmse(new_daily)
            elif name == "MAPE_cumulative_adoption":
                metrics_out[name] = mape(cum_share)
            elif name == "PeakError_new_adoptions":
                metrics_out[name] = peak_error(new_daily, window=7)
            elif name == "TimeTo50_cumulative":
                metrics_out[name] = time_to_threshold(cum_share, 0.5)
            else:
                metrics_out[name] = None

        # Additional summaries beyond configured metrics
        # time_to_50_percent_adoption using raw adoption_rate
        ar = self.series.get("adoption_rate", [])
        metrics_out["time_to_50_percent_adoption"] = -1
        for i, v in enumerate(ar):
            if v >= 0.5:
                metrics_out["time_to_50_percent_adoption"] = i
                break

        # sustained_adoption_duration: consecutive days adoption_rate >= 0.5 at the end
        thr = 0.5
        sustained = 0
        for v in reversed(ar):
            if v >= thr:
                sustained += 1
            else:
                break
        metrics_out["sustained_adoption_duration"] = sustained

        # policy_compliance_rate: approximate using enforced events vs. non-adopters when mandate active
        mandates = self.series.get("mandate_active", [])
        enforced = self.series.get("enforced_events", [])
        if mandates and enforced:
            total_enforced = int(sum(enforced))
            mandate_days = int(sum(mandates))
            metrics_out["policy_enforcement_events_total"] = total_enforced
            metrics_out["mandate_active_days"] = mandate_days
            metrics_out["policy_compliance_rate_proxy"] = float(total_enforced) / float(mandate_days + 1e-6)
        else:
            metrics_out["policy_enforcement_events_total"] = 0
            metrics_out["mandate_active_days"] = 0
            metrics_out["policy_compliance_rate_proxy"] = 0.0

        # supply metrics
        if self.with_supply:
            metrics_out["any_stockout_days"] = int(sum(self.series.get("any_stockout", [])))
            metrics_out["final_inventory"] = int(self.supply.inventory)
        else:
            metrics_out["any_stockout_days"] = 0
            metrics_out["final_inventory"] = 0

        # misinformation impact: difference in average adoption_rate before vs after campaign start
        cs = int(self.cfg.get("campaign_start_day", 7))
        pre = ar[:cs]
        post = ar[cs:] if cs < len(ar) else []
        metrics_out["misinformation_impact_proxy"] = (float(np.mean(post)) - float(np.mean(pre))) if pre and post else 0.0

        # disparity metric (income-decile gini) average over horizon
        gini_series = self.series.get("daily_gini", [])
        metrics_out["subgroup_disparity_index_mean"] = float(np.mean(gini_series)) if gini_series else 0.0

        return metrics_out

    def visualize(self, show: bool = False, save_path: Optional[str] = None) -> None:
        """
        Plot key series for quick inspection.

        Args:
            show: If True, display the plot window.
            save_path: If provided, save the plot to this path.
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        # FIXED: Defer matplotlib import and guard in headless environments
        try:
            import matplotlib
            # Use non-interactive backend for headless environments
            try:
                matplotlib.use("Agg")
            except Exception:
                pass
            import matplotlib.pyplot as plt
        except Exception as e:
            logging.warning(f"Plotting disabled: {e}")
            return

        plt.figure(figsize=(12, 8))
        # Plot adoption rate
        ar = self.series.get("adoption_rate", [])
        plt.subplot(2, 2, 1)
        plt.plot(ar, label="Adoption rate")
        plt.title("Adoption Rate")
        plt.ylim(0, 1)
        plt.legend()

        # Plot new adoptions (observable)
        new_obs = self.series.get("observable.new_adoptions_daily", [])
        plt.subplot(2, 2, 2)
        plt.plot(new_obs, label="New adoptions (obs)")
        plt.title("New Adoptions (Observable)")
        plt.legend()

        # Info exposures
        true_exp = self.series.get("true_info_exposures", [])
        mis_exp = self.series.get("misinfo_exposures", [])
        plt.subplot(2, 2, 3)
        plt.plot(true_exp, label="True info exposures")
        plt.plot(mis_exp, label="Misinfo exposures")
        plt.title("Information Exposures")
        plt.legend()

        # Disparity
        gini_series = self.series.get("daily_gini", [])
        plt.subplot(2, 2, 4)
        plt.plot(gini_series, label="Gini (income-decile adoption rates)")
        plt.title("Disparity Over Time")
        plt.legend()

        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path, dpi=150)
            except Exception as e:
                logging.warning(f"Could not save plot to {save_path}: {e}")
        if show:
            plt.show()
        else:
            plt.close()

    def save_results(self, filename: str) -> None:
        """
        Save time series results to a CSV file with columns as series keys.

        Args:
            filename: Output CSV filename (under DATA_DIR if relative).
        """
        pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
        import csv

        out_path = filename
        if not os.path.isabs(out_path):
            out_path = os.path.join(DATA_DIR, filename)

        # Normalize series lengths
        max_len = max(len(v) for v in self.series.values() if isinstance(v, list))
        keys = sorted([k for k in self.series.keys() if isinstance(self.series[k], list)])
        rows = []
        for i in range(max_len):
            row = {}
            for k in keys:
                arr = self.series.get(k, [])
                row[k] = arr[i] if i < len(arr) else ""
            rows.append(row)

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        except Exception:
            pass

        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        except Exception as e:
            logging.error(f"Failed to save results to {out_path}: {e}")


def main():
    """
    Entrypoint: parse JSON config from stdin or file path in argv, run the simulation, and print JSON.

    Behavior:
        - Reads JSON from stdin. If empty and argv[1] exists, read from file.
        - Sanitizes JSON to last closing brace/bracket on error.
        - Runs Simulation with merged config.
        - Prints JSON with 'series' and 'metrics'.
        - Demonstrates saving results to 'results.csv' in DATA_DIR and visualizes to 'plot.png'.

    Notes:
        - FIXED: Implemented robust main() with JSON parsing and error reporting.
        - FIXED: Removed stray non-Python text that previously caused SyntaxError.
        - FIXED: Route JSON parse errors to stderr and keep stdout clean for JSON output.
    """
    pass  # NOTE: pass retained per instruction to guarantee syntactic correctness
    logging.basicConfig(level=logging.INFO)
    raw = sys.stdin.read()
    raw = raw.strip()
    cfg: Dict[str, Any] = {}
    if not raw:
        # Try file from argv
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            try:
                with open(sys.argv[1], "r") as f:
                    raw = f.read().strip()
            except Exception as e:
                # FIXED: Route file read errors to stderr
                sys.stderr.write(f"Error reading config file: {e}\n")
                raw = ""

    if raw:
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError as e:
            try:
                cfg = json.loads(_sanitize_json_text(raw))
            except json.JSONDecodeError as e2:
                # FIXED: Route parse errors to stderr instead of stdout to avoid contaminating JSON output
                msg = f"Error parsing JSON response: {e2.msg}: line {getattr(e2, 'lineno', '?')} column {getattr(e2, 'colno', '?')} (char {getattr(e2, 'pos', '?')})\n"
                sys.stderr.write(msg)
                # Proceed with defaults but keep stdout clean JSON
                cfg = {}
    else:
        cfg = {}

    # Instantiate and run simulation
    sim = Simulation(cfg)
    series = sim.run()
    metrics = sim.evaluate()

    # Save results CSV and plot
    try:
        sim.save_results("results.csv")
    except Exception as e:
        logging.warning(f"Could not save results: {e}")
    try:
        plot_path = os.path.join(DATA_DIR, "plot.png")
        sim.visualize(show=False, save_path=plot_path)
    except Exception as e:
        logging.warning(f"Could not visualize results: {e}")

    # Print JSON output
    try:
        print(json.dumps({"series": series, "metrics": metrics}, indent=2))
    except Exception as e:
        # FIXED: Route serialization errors to stderr
        sys.stderr.write(f"Error serializing output to JSON: {e}\n")


# FIXED: Removed stray non-Python text from previous corrupted iteration.
# Execute main for both direct execution and sandbox wrapper invocation
main()