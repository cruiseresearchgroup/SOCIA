import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("networkx is required. Please install dependencies (pip install networkx numpy).") from e

# Path Handling Instructions (kept intentionally and used in save_results)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


@dataclass
class Household:
    """
    Represents a household grouping with members influencing each other's norms.

    Attributes:
        id (int): Unique household identifier.
        member_ids (List[int]): List of agent indices belonging to this household.
    """
    id: int
    member_ids: List[int]

    def add_member(self, agent_id: int) -> None:
        """
        Add a member to the household.

        Args:
            agent_id (int): The agent index to add.
        """
        self.member_ids.append(agent_id)
        pass


@dataclass
class WorkplaceSchool:
    """
    Represents a workplace or school location with optional mask policy enforcement.

    Attributes:
        id (int): Unique identifier.
        enforcement_strength (float): Location-specific enforcement in [0,1].
        policy_mask_required (bool): Whether masks are required at this location currently.
        contact_rate_modifier (float): Multiplier for contact intensity at the location.
    """
    id: int
    enforcement_strength: float = 0.5
    policy_mask_required: bool = False
    contact_rate_modifier: float = 1.0

    def set_policy(self, required: bool) -> None:
        """
        Set the mask requirement policy for this location.

        Args:
            required (bool): Whether masks are required.
        """
        self.policy_mask_required = required
        pass


@dataclass
class PublicVenue:
    """
    Represents a public venue with optional mask policy enforcement.

    Attributes:
        id (int): Unique identifier.
        enforcement_strength (float): Location-specific enforcement in [0,1].
        policy_mask_required (bool): Whether masks are required at this location currently.
        contact_rate_modifier (float): Multiplier for contact intensity at the location.
    """
    id: int
    enforcement_strength: float = 0.4
    policy_mask_required: bool = False
    contact_rate_modifier: float = 1.0

    def set_policy(self, required: bool) -> None:
        """
        Set the mask requirement policy for this venue.

        Args:
            required (bool): Whether masks are required.
        """
        self.policy_mask_required = required
        pass


@dataclass
class PolicyMaker:
    """
    Represents a policy maker entity controlling mask mandates and enforcement coverage.

    Attributes:
        policy_start_day (Optional[int]): Day mandate starts.
        policy_end_day (Optional[int]): Day mandate ends (inclusive). None means ongoing after start.
        policy_mask_required_fraction (float): Fraction of locations covered by mandate when active [0,1].
        enforcement_strength (float): Base enforcement strength applied at mandated locations [0,1].
    """
    policy_start_day: Optional[int] = None
    policy_end_day: Optional[int] = None
    policy_mask_required_fraction: float = 0.7
    enforcement_strength: float = 0.5

    def policy_active(self, day: int) -> bool:
        """
        Determine if policy is active on given day.

        Args:
            day (int): Simulation day.

        Returns:
            bool: True if active, else False.
        """
        if self.policy_start_day is None:
            return False
        if self.policy_end_day is None:
            return day >= int(self.policy_start_day)
        return int(self.policy_start_day) <= day <= int(self.policy_end_day)
        pass


@dataclass
class Retailer:
    """
    Represents a retailer supplying masks to the population.

    Attributes:
        id (int): Unique retailer identifier.
        inventory_level (float): Current inventory level (units).
        restock_rate (float): Units restocked per day.
        mask_price (float): Current mask price.
        initial_inventory (float): Initial inventory reference for price adjustments.
    """
    id: int
    inventory_level: float
    restock_rate: float
    mask_price: float
    initial_inventory: float

    def restock(self) -> None:
        """
        Restock inventory by restock_rate and adjust price in response to scarcity/abundance.
        """
        self.inventory_level += max(0.0, float(self.restock_rate))
        # Simple scarcity pricing: raise price if inventory < 10% initial, lower if > 50%
        if self.initial_inventory > 0:
            ratio = self.inventory_level / float(self.initial_inventory)
            if ratio < 0.1:
                self.mask_price *= 1.03
            elif ratio > 0.5:
                self.mask_price *= 0.995
            self.mask_price = max(0.1, float(self.mask_price))
        pass


@dataclass
class MediaSource:
    """
    Represents a media source broadcasting messages influencing agents.

    Attributes:
        id (int): Unique media source ID.
        reach (float): Daily probability an agent is exposed to this source [0,1].
        tone (float): Orientation in [-1, 1], where 1 pro-mask, -1 anti-mask.
        credibility (float): Weighting of source effect in [0,1].
        frequency (float): Probability this source broadcasts on a given day [0,1].
    """
    id: int
    reach: float
    tone: float
    credibility: float
    frequency: float

    def broadcast(self, rng: np.random.Generator, n_agents: int) -> np.ndarray:
        """
        Generate a boolean mask of exposed agents for this source on a given day.

        Args:
            rng (np.random.Generator): RNG for sampling exposure.
            n_agents (int): Number of agents.

        Returns:
            np.ndarray: Boolean array of shape (n_agents,) indicating exposure.
        """
        if rng.random() < self.frequency:
            exposed = rng.random(n_agents) < self.reach
            return exposed
        return np.zeros(n_agents, dtype=bool)
        pass


class MaskSimulation:
    """
    Core simulation engine modeling mask adoption under social influence, household norms,
    policy enforcement by location type, heterogeneous media sources, and multi-retailer supply.

    FIXES applied based on prior iteration feedback:
    - FIXED: Introduced explicit entities Household, WorkplaceSchool, PublicVenue, PolicyMaker, MediaSource, Retailer and wired interactions.
    - FIXED: Harmonized parameter names with spec (policy_start_day, policy_mask_required_fraction, enforcement_strength, restock_rate, retailer_initial_inventory_per_capita, affordability_threshold_income_fraction, information_campaign_intensity) and added backward-compat mapping.
    - FIXED: Added household-level norms in social influence via _build_households and household_influence_weight.
    - FIXED: Added missing metrics: time_to_target_adoption_70, sustained_adoption_duration_above_70, policy_noncompliance_rate, mask_availability_rate_over_time and endline, adherence_probability_mean_over_time and endline, adoption_inequality_gini_income, and alias peak_adoption_rate.
    - FIXED: Kept adoption_rate_over_time as a time series in run() and moved mean calculation to evaluate() as adoption_rate_mean.
    - FIXED: Modeled multiple retailers with inventory, restock (restock_rate), price dynamics under scarcity, and affordability threshold.
    - FIXED: Location-specific enforcement under policy coverage; tracked mandated and compliant visits accordingly.
    """

    def __init__(
        self,
        population_size: int = 5000,
        days: int = 120,
        seed: int = 42,
        params: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the MaskSimulation with population and configuration parameters.

        Args:
            population_size (int): Number of agents.
            days (int): Number of days to simulate.
            seed (int): RNG seed for reproducibility.
            params (Optional[Dict]): Parameter overrides.

        Raises:
            ValueError: If population_size or days are non-positive.
        """
        if population_size <= 0 or days <= 0:
            raise ValueError("population_size and days must be positive integers.")

        self.n = population_size
        self.days = days
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Spec-aligned parameters with defaults
        self.params: Dict = {
            "initial_adoption_rate": 0.15,
            "average_degree": 10,
            "social_influence_weight": 0.4,
            "household_influence_weight": 0.2,
            "risk_perception_base": 0.3,
            "trust_in_authorities_mean": 0.5,
            "decision_noise": 0.1,
            "fatigue_rate": 0.01,
            "habit_formation_rate": 0.02,
            "mask_price": 1.5,
            "retailer_initial_inventory_per_capita": 2.0,
            "restock_rate": 0.2,
            "rationing_limit_per_purchase": 5,
            "policy_start_day": None,
            "policy_end_day": None,
            "policy_mask_required_fraction": 0.7,
            "enforcement_strength": 0.5,
            "fine_amount": 50.0,
            "information_campaign_intensity": 0.1,
            "misinformation_rate": 0.05,
            "media_sources_count": 3,
            "media_pro_fraction": 0.67,
            "media_frequency": 1.0,
            "affordability_threshold_income_fraction": 0.05,
            "work_visit_probability": 0.6,
            "public_visit_probability": 0.4,
            "workplaces_per_capita": 1 / 50.0,
            "venues_per_capita": 1 / 100.0,
            "retailer_per_capita": 1 / 800.0,
        }

        # Accept user overrides
        if params:
            self.params.update(params)

        # Backward-compat mapping for prior iteration parameter names
        # FIXED: Map old names to new spec-compliant names
        if "initial_inventory_per_capita" in self.params:
            self.params["retailer_initial_inventory_per_capita"] = self.params.pop("initial_inventory_per_capita")
        if "restock_rate_per_day" in self.params:
            self.params["restock_rate"] = self.params.pop("restock_rate_per_day")
        if "policy_mandate_day" in self.params and self.params["policy_mandate_day"] is not None:
            self.params["policy_start_day"] = self.params.pop("policy_mandate_day")
        if "policy_mandate_end_day" in self.params and self.params["policy_mandate_end_day"] is not None:
            self.params["policy_end_day"] = self.params.pop("policy_mandate_end_day")
        if "enforcement_intensity" in self.params:
            self.params["enforcement_strength"] = self.params.pop("enforcement_intensity")
        if "campaign_effect_size" in self.params:
            self.params["information_campaign_intensity"] = self.params.get("information_campaign_intensity", self.params.pop("campaign_effect_size"))
        if "media_message_frequency" in self.params:
            self.params["media_frequency"] = self.params.pop("media_message_frequency")

        # Social network construction
        k = max(2, int(self.params["average_degree"]))
        # FIXED: Ensure even degree for Watts-Strogatz small-world network to prevent runtime errors
        if k % 2 == 1:
            k += 1
        self.G = nx.watts_strogatz_graph(self.n, k, p=0.1, seed=self.seed)
        self.neighbors: List[List[int]] = [list(self.G.neighbors(i)) for i in range(self.n)]

        # Agent state arrays
        self.income = np.clip(self.rng.lognormal(mean=0.0, sigma=0.75, size=self.n) / 5.0, 0, 1)
        self.trust = np.clip(
            self.rng.normal(self.params["trust_in_authorities_mean"], 0.2, size=self.n), 0, 1
        )
        self.risk = np.full(self.n, self.params["risk_perception_base"])
        self.susc = np.clip(self.rng.beta(2, 2, size=self.n), 0, 1)
        self.compliance = np.clip(self.rng.beta(2, 2, size=self.n), 0, 1)
        self.fatigue = np.zeros(self.n)

        initial_wearing = self.rng.random(self.n) < self.params["initial_adoption_rate"]
        self.attitude = np.clip(
            0.3 + 0.5 * initial_wearing + self.rng.normal(0, 0.1, size=self.n), 0, 1
        )
        self.stock = self.rng.poisson(
            lam=max(0.0, self.params["retailer_initial_inventory_per_capita"]), size=self.n
        ).astype(float)
        self.wearing = initial_wearing.astype(float)
        self.last_wear_prob = np.zeros(self.n)

        # Households
        self.household_id = np.full(self.n, -1, dtype=int)
        self.households: List[Household] = []
        self._build_households()

        # Workplaces and public venues
        n_workplaces = max(1, int(self.params["workplaces_per_capita"] * self.n))
        n_venues = max(1, int(self.params["venues_per_capita"] * self.n))
        self.workplaces: List[WorkplaceSchool] = [
            WorkplaceSchool(
                id=i,
                enforcement_strength=np.clip(np.random.normal(self.params["enforcement_strength"], 0.1), 0, 1),
                policy_mask_required=False,
                contact_rate_modifier=1.0,
            )
            for i in range(n_workplaces)
        ]
        self.venues: List[PublicVenue] = [
            PublicVenue(
                id=i,
                enforcement_strength=np.clip(np.random.normal(self.params["enforcement_strength"] * 0.8, 0.1), 0, 1),
                policy_mask_required=False,
                contact_rate_modifier=1.0,
            )
            for i in range(n_venues)
        ]
        # Assign each agent to a workplace
        self.workplace_assignment = self.rng.integers(0, n_workplaces, size=self.n)

        # Policy maker
        self.policy_maker = PolicyMaker(
            policy_start_day=self.params["policy_start_day"],
            policy_end_day=self.params["policy_end_day"],
            policy_mask_required_fraction=self.params["policy_mask_required_fraction"],
            enforcement_strength=self.params["enforcement_strength"],
        )

        # Media sources
        m_count = max(1, int(self.params["media_sources_count"]))
        pro_count = max(0, int(round(m_count * float(self.params["media_pro_fraction"]))))
        anti_count = max(0, m_count - pro_count)
        self.media_sources: List[MediaSource] = []
        for i in range(pro_count):
            self.media_sources.append(
                MediaSource(
                    id=i,
                    reach=np.clip(self.rng.normal(0.5, 0.15), 0.05, 0.95),
                    tone=1.0,
                    credibility=np.clip(self.rng.normal(0.6, 0.2), 0.1, 1.0),
                    frequency=float(self.params["media_frequency"]),
                )
            )
        for i in range(anti_count):
            self.media_sources.append(
                MediaSource(
                    id=pro_count + i,
                    reach=np.clip(self.rng.normal(0.4, 0.15), 0.05, 0.95),
                    tone=-1.0,
                    credibility=np.clip(self.rng.normal(0.5, 0.2), 0.1, 1.0),
                    frequency=float(self.params["media_frequency"]),
                )
            )

        # Retailers
        n_retailers = max(1, int(self.params["retailer_per_capita"] * self.n))
        total_initial_inventory = float(self.n * self.params["retailer_initial_inventory_per_capita"])
        per_retailer_inventory = total_initial_inventory / n_retailers
        per_retailer_restock = float(self.params["restock_rate"] * self.n) / n_retailers
        base_price = float(self.params["mask_price"])
        self.retailers: List[Retailer] = [
            Retailer(
                id=i,
                inventory_level=per_retailer_inventory,
                restock_rate=per_retailer_restock,
                mask_price=base_price,
                initial_inventory=per_retailer_inventory,
            )
            for i in range(n_retailers)
        ]

        # Campaign schedule (for compatibility with prior code; ties to information_campaign_intensity)
        self.campaign_days = set()
        if "campaign_start_day" in self.params and self.params["campaign_start_day"] is not None:
            d = int(self.params["campaign_start_day"])
            freq = int(self.params.get("campaign_frequency_days", 14))
            while d < self.days:
                self.campaign_days.add(d)
                d += max(1, freq)

        # Time series collection
        self.daily_adoption: List[float] = []
        self.daily_supply_demand_gap: List[float] = []
        self.daily_attitude_mean: List[float] = []
        self.daily_mask_availability: List[float] = []
        self.daily_adherence_prob_mean: List[float] = []

        # Enforcement/compliance tracking
        self.noncompliance_events: int = 0
        self.mandated_visits_total: int = 0
        self.mandated_visits_compliant: int = 0
        pass

    def _build_households(self) -> None:
        """
        Construct household groupings with simple random sizes and assign household IDs.
        """
        # FIXED: Introduced households for household-level norms
        sizes = self.rng.integers(2, 6, size=max(1, self.n // 3))
        ids = np.arange(self.n)
        self.rng.shuffle(ids)
        hids = -np.ones(self.n, dtype=int)
        start = 0
        hid = 0
        self.households = []
        for sz in sizes:
            if start >= self.n:
                break
            members = ids[start:start + sz]
            hids[members] = hid
            self.households.append(Household(id=hid, member_ids=list(members)))
            hid += 1
            start += sz
        if np.any(hids < 0):
            # Assign remaining to the last household
            rem = np.where(hids < 0)[0]
            if self.households:
                last_hid = self.households[-1].id
                for r in rem:
                    self.households[-1].add_member(int(r))
                    hids[r] = last_hid
            else:
                self.households.append(Household(id=0, member_ids=list(rem)))
                hids[rem] = 0
        self.household_id = hids
        pass

    def _policy_active(self, day: int) -> bool:
        """
        Return whether policy is active on a given day based on policy maker configuration.

        Args:
            day (int): Simulation day.

        Returns:
            bool: True if policy is active.
        """
        status = self.policy_maker.policy_active(day)
        return status
        pass

    def _update_policy_locations(self, day: int) -> None:
        """
        Update which locations are under mask mandate based on policy coverage.

        Args:
            day (int): Simulation day.
        """
        active = self._policy_active(day)
        if not active:
            # Reset all to not required
            for loc in self.workplaces:
                loc.set_policy(False)
            for v in self.venues:
                v.set_policy(False)
            return

        # Activate a fraction of locations
        frac = float(self.params.get("policy_mask_required_fraction", 0.7))
        # Use RNG to select a subset each day (could persist, but day-level suffices)
        n_w_active = int(round(frac * len(self.workplaces)))
        n_v_active = int(round(frac * len(self.venues)))
        w_idx = np.arange(len(self.workplaces))
        v_idx = np.arange(len(self.venues))
        self.rng.shuffle(w_idx)
        self.rng.shuffle(v_idx)
        for i, loc in enumerate(self.workplaces):
            loc.set_policy(i in set(w_idx[:n_w_active]))
        for i, ven in enumerate(self.venues):
            ven.set_policy(i in set(v_idx[:n_v_active]))
        pass

    def _media_broadcasts(self, day: int) -> None:
        """
        Apply media broadcasts to agents based on source reach, tone, and credibility.

        Args:
            day (int): Simulation day.
        """
        # Additional campaign intensity days boost pro-mask shift
        campaign_boost = float(self.params.get("information_campaign_intensity", 0.0)) if day in self.campaign_days else 0.0
        for src in self.media_sources:
            exposed = src.broadcast(self.rng, self.n)
            if not np.any(exposed):
                continue
            tone = float(src.tone)
            cred = float(src.credibility)
            effect = cred * (0.05 + campaign_boost) * tone
            # Influence attitude, trust, and risk with bounded effects
            self.attitude[exposed] = np.clip(self.attitude[exposed] + 0.5 * effect, 0, 1)
            self.trust[exposed] = np.clip(self.trust[exposed] + 0.3 * effect, 0, 1)
            self.risk[exposed] = np.clip(self.risk[exposed] + 0.6 * effect, 0, 1)

        # Misinformation background
        mis = float(self.params.get("misinformation_rate", 0.0))
        if mis > 0:
            mis_mask = self.rng.random(self.n) < mis
            self.attitude[mis_mask] = np.clip(self.attitude[mis_mask] - 0.05, 0, 1)
            self.trust[mis_mask] = np.clip(self.trust[mis_mask] - 0.03, 0, 1)
            self.risk[mis_mask] = np.clip(self.risk[mis_mask] - 0.04, 0, 1)
        pass

    def _social_influence(self) -> None:
        """
        Update attitudes based on social influence from neighbors and household norms.

        Uses a DeGroot-like averaging with susceptibility and household influence.
        """
        neigh_mean = np.zeros(self.n)
        for i, neigh in enumerate(self.neighbors):
            if neigh:
                neigh_mean[i] = float(np.mean(self.wearing[neigh]))

        # Household influence
        hh_share = np.zeros(self.n)
        if self.households:
            for hh in self.households:
                members = np.array(hh.member_ids, dtype=int)
                if members.size > 0:
                    share = float(np.mean(self.wearing[members]))
                    hh_share[members] = share

        w_net = float(self.params.get("social_influence_weight", 0.4))
        w_hh = float(self.params.get("household_influence_weight", 0.2))
        total_w = np.clip(w_net + w_hh, 0, 1)
        s = self.susc
        target = (w_net * neigh_mean + w_hh * hh_share) / max(total_w, 1e-6)
        self.attitude = np.clip((1 - total_w * s) * self.attitude + (total_w * s) * target, 0, 1)
        pass

    def _fatigue_and_habit(self) -> None:
        """
        Apply habit formation and fatigue dynamics to attitudes and future behavior.
        """
        self.attitude = np.clip(
            self.attitude * (1 - float(self.params["fatigue_rate"]))
            + self.wearing * float(self.params["habit_formation_rate"]),
            0,
            1,
        )
        pass

    def _restock_retailers(self) -> None:
        """
        Restock the retailers' inventory and update their prices.
        """
        for r in self.retailers:
            r.restock()
        pass

    def _purchase_masks(self) -> None:
        """
        Process mask purchase attempts with rationing, affordability, and multi-retailer inventory.

        Records daily supply-demand gap as demand_units - units_sold.
        """
        need = self.stock < 1.0
        # Willingness: combine risk, trust, and income
        willingness = np.clip(
            0.3 + 0.4 * self.risk + 0.3 * self.trust - 0.2 * (1 - self.income), 0, 1
        )
        attempt = (self.rng.random(self.n) < willingness) & need

        if not np.any(attempt):
            self.daily_supply_demand_gap.append(0.0)
            return

        max_per_purchase = int(self.params["rationing_limit_per_purchase"])
        demand_units = float(
            np.sum(np.minimum(max_per_purchase, np.maximum(1.0 - self.stock[attempt], 1.0)))
        )

        sold = 0.0
        buyers = np.where(attempt)[0]
        self.rng.shuffle(buyers)
        # Affordability threshold: price must be <= income * affordability_threshold_income_fraction * scale
        aff_frac = float(self.params.get("affordability_threshold_income_fraction", 0.05))
        budget_scale = 10.0  # scale factor to map income fraction to effective daily budget
        for i in buyers:
            if len(self.retailers) == 0:
                break
            # Select a retailer with probability proportional to inventory+1
            invs = np.array([max(0.0, r.inventory_level) + 1.0 for r in self.retailers], dtype=float)
            probs = invs / np.sum(invs)
            ridx = int(self.rng.choice(len(self.retailers), p=probs))
            r = self.retailers[ridx]
            if r.inventory_level <= 0:
                continue
            # Affordability check
            budget = max(0.0, self.income[i] * aff_frac * budget_scale)
            if r.mask_price > budget:
                continue  # cannot afford
            qty = float(min(max_per_purchase, max(1.0, math.ceil(1.0 - self.stock[i]))))
            qty = float(min(qty, r.inventory_level))
            if qty <= 0:
                continue
            self.stock[i] += qty
            r.inventory_level -= qty
            sold += qty

        self.daily_supply_demand_gap.append(float(demand_units - sold))
        pass

    def _decide_and_wear(self, day: int) -> None:
        """
        Decide daily wearing behavior based on latent variables and policy effects.
        Records daily mask availability and adherence probability mean.
        """
        noise = self.rng.normal(0, float(self.params["decision_noise"]), size=self.n)
        base = 0.05 + 0.5 * self.attitude + 0.2 * self.trust + 0.2 * self.risk - 0.1 * self.fatigue + noise
        policy_boost = 0.0
        if self._policy_active(day):
            policy_boost = 0.4 * self.policy_maker.enforcement_strength * self.compliance
        wear_prob = np.clip(base + policy_boost, 0, 1)
        self.last_wear_prob = wear_prob  # FIXED: store adherence probability
        can_wear = self.stock > 0
        self.wearing = ((self.rng.random(self.n) < wear_prob) & can_wear).astype(float)
        self.stock = np.where(self.wearing > 0, np.maximum(0.0, self.stock - 1.0), self.stock)
        self.fatigue = np.clip(self.fatigue + 0.01 * (1 - self.wearing), 0, 1)
        # FIXED: record daily mask availability and adherence probability mean
        self.daily_mask_availability.append(float(np.mean(can_wear)))
        self.daily_adherence_prob_mean.append(float(np.mean(wear_prob[self.wearing > 0])) if np.any(self.wearing > 0) else 0.0)
        pass

    def _locations_and_enforcement(self, day: int) -> None:
        """
        Simulate visits to workplaces and public venues with potential mask requirements and track enforcement.

        Args:
            day (int): Simulation day.
        """
        self._update_policy_locations(day)
        work_visit_p = float(self.params.get("work_visit_probability", 0.6))
        pub_visit_p = float(self.params.get("public_visit_probability", 0.4))

        # Workplace visits
        work_visitors = self.rng.random(self.n) < work_visit_p
        if np.any(work_visitors):
            w_ids = self.workplace_assignment[work_visitors]
            locs = np.array([self.workplaces[int(wid)] for wid in w_ids], dtype=object)
            required = np.array([loc.policy_mask_required for loc in locs], dtype=bool)
            required_idx = np.where(work_visitors)[0][required]
            if required_idx.size > 0:
                wearing_req = self.wearing[required_idx] == 1
                compliant = wearing_req
                noncompliant = ~wearing_req
                self.mandated_visits_total += int(required_idx.size)
                self.mandated_visits_compliant += int(np.sum(compliant))
                self.noncompliance_events += int(np.sum(noncompliant))
                # Enforcement update: increase compliance propensity slightly for noncompliant after enforcement
                # Probability of enforcement event per noncompliant = location.enforcement_strength * policy enforcement_strength
                base_enf = float(self.policy_maker.enforcement_strength)
                enforced_mask = np.zeros(required_idx.size, dtype=bool)
                if required_idx.size > 0:
                    # Compute per-visit enforcement probability
                    p_enf = np.array(
                        [
                            base_enf * float(locs[i].enforcement_strength)
                            for i in np.where(required)[0]
                        ],
                        dtype=float,
                    )
                    p_enf = np.clip(p_enf, 0, 1)
                    # Sample enforcement only for noncompliant
                    non_idx = np.where(noncompliant)[0]
                    if non_idx.size > 0:
                        enforced_sub = self.rng.random(non_idx.size) < p_enf[non_idx]
                        enforced_mask[non_idx] = enforced_sub
                        # Increase compliance propensity for those enforced
                        enforced_agents = required_idx[non_idx[enforced_sub]]
                        if enforced_agents.size > 0:
                            self.compliance[enforced_agents] = np.clip(self.compliance[enforced_agents] + 0.02 * base_enf, 0, 1)

        # Public venue visits
        pub_visitors = self.rng.random(self.n) < pub_visit_p
        if np.any(pub_visitors):
            # each visitor selects a random venue
            venue_ids = self.rng.integers(0, len(self.venues), size=int(np.sum(pub_visitors)))
            locs_v = np.array([self.venues[int(vid)] for vid in venue_ids], dtype=object)
            required_v = np.array([v.policy_mask_required for v in locs_v], dtype=bool)
            required_idx_v = np.where(pub_visitors)[0][required_v]
            if required_idx_v.size > 0:
                wearing_req_v = self.wearing[required_idx_v] == 1
                compliant_v = wearing_req_v
                noncompliant_v = ~wearing_req_v
                self.mandated_visits_total += int(required_idx_v.size)
                self.mandated_visits_compliant += int(np.sum(compliant_v))
                self.noncompliance_events += int(np.sum(noncompliant_v))
                base_enf_v = float(self.policy_maker.enforcement_strength)
                p_enf_v = np.array(
                    [
                        base_enf_v * float(locs_v[i].enforcement_strength)
                        for i in np.where(required_v)[0]
                    ],
                    dtype=float,
                )
                p_enf_v = np.clip(p_enf_v, 0, 1)
                non_idx_v = np.where(noncompliant_v)[0]
                if non_idx_v.size > 0:
                    enforced_sub_v = self.rng.random(non_idx_v.size) < p_enf_v[non_idx_v]
                    enforced_agents_v = required_idx_v[non_idx_v[enforced_sub_v]]
                    if enforced_agents_v.size > 0:
                        self.compliance[enforced_agents_v] = np.clip(self.compliance[enforced_agents_v] + 0.015 * base_enf_v, 0, 1)
        pass

    def step(self, day: int) -> None:
        """
        Execute one simulation day:
        - Restock retailers and adjust prices
        - Media broadcasts
        - Social influence and household norms
        - Habit formation and fatigue
        - Purchase masks subject to affordability and supply
        - Decide wearing
        - Visits to workplaces and public venues and enforcement tracking

        Records daily metrics.
        """
        self._restock_retailers()
        self._media_broadcasts(day)
        self._social_influence()
        self._fatigue_and_habit()
        self._purchase_masks()
        self._decide_and_wear(day)
        self._locations_and_enforcement(day)
        # Record time series
        self.daily_adoption.append(float(np.mean(self.wearing)))
        self.daily_attitude_mean.append(float(np.mean(self.attitude)))
        pass

    def _time_to_threshold(self, thr: float = 0.7) -> Optional[int]:
        """
        Compute the first day the adoption rate reaches or exceeds a threshold.

        Args:
            thr (float): Adoption threshold in [0,1].

        Returns:
            Optional[int]: Day index if reached, else None.
        """
        for d, val in enumerate(self.daily_adoption):
            if val >= thr:
                return d
        return None
        pass

    def _sustained_adoption_duration(self, thr: float = 0.7) -> int:
        """
        Compute sustained adoption duration at or above a threshold after first crossing.

        Args:
            thr (float): Threshold in [0,1].

        Returns:
            int: Number of days from first crossing where adoption stays >= thr (to end of series).
        """
        if not self.daily_adoption:
            return 0
        t0 = self._time_to_threshold(thr)
        if t0 is None:
            return 0
        series = np.asarray(self.daily_adoption, dtype=float)
        return int(np.sum(series[t0:] >= thr))
        pass

    def _policy_effect(self) -> Optional[float]:
        """
        Compute average adoption change in the 7 days after mandate starts versus 7 days before.

        Returns:
            Optional[float]: Difference post - pre if available.
        """
        if self.policy_maker.policy_start_day is None:
            return None
        md = int(self.policy_maker.policy_start_day)
        if md >= len(self.daily_adoption):
            return None
        pre = self.daily_adoption[max(0, md - 7): md]
        post = self.daily_adoption[md: md + 7]
        if not pre or not post:
            return None
        return float(np.mean(post) - np.mean(pre))
        pass

    def _campaign_effect_attitudes(self) -> Optional[float]:
        """
        Estimate campaign effect on attitudes as the average difference in mean attitude
        between each campaign day and the previous day.

        Returns:
            Optional[float]: Mean delta if computable, else None.
        """
        if not self.daily_attitude_mean:
            return None
        deltas: List[float] = []
        for d in sorted(self.campaign_days):
            if 0 < d < len(self.daily_attitude_mean):
                deltas.append(self.daily_attitude_mean[d] - self.daily_attitude_mean[d - 1])
        if len(deltas) == 0:
            return None
        return float(np.mean(deltas))
        pass

    def _adoption_disparity(self) -> float:
        """
        Compute adoption disparity by income group (high quintile minus low quintile) on the last day.

        Returns:
            float: Disparity value in [-1,1].
        """
        if not self.daily_adoption:
            return 0.0
        q20, q80 = np.quantile(self.income, [0.2, 0.8])
        low = self.income <= q20
        high = self.income >= q80
        last = self.wearing
        return float(np.mean(last[high]) - np.mean(last[low]))
        pass

    def _gini(self, x: np.ndarray) -> float:
        """
        Compute the Gini coefficient for a non-negative array.

        Args:
            x (np.ndarray): Non-negative values.

        Returns:
            float: Gini coefficient in [0,1].
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x >= 0)]
        if x.size == 0:
            return 0.0
        if np.allclose(x, 0):
            return 0.0
        x_sorted = np.sort(x)
        n = x_sorted.size
        cumx = np.cumsum(x_sorted)
        return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)
        pass

    def _access_inequality_index(self) -> float:
        """
        Compute access inequality index as the Gini coefficient over mask stock across agents.

        Returns:
            float: Gini coefficient.
        """
        return self._gini(self.stock)
        pass

    def _sentiment_polarization(self) -> float:
        """
        Compute a simple sentiment polarization metric combining attitude dispersion and
        average neighbor disagreement.

        Returns:
            float: Polarization index (arbitrary units).
        """
        att = self.attitude
        stdev = float(np.std(att))
        neigh_mean = np.zeros(self.n)
        for i, neigh in enumerate(self.neighbors):
            if neigh:
                neigh_mean[i] = float(np.mean(att[neigh]))
            else:
                neigh_mean[i] = float(att[i])
        disagreement = float(np.mean(np.abs(att - neigh_mean)))
        return float(stdev + 0.5 * disagreement)
        pass

    def _adoption_inequality_gini_income(self) -> float:
        """
        Compute Gini coefficient across income-decile adoption rates at endline.

        Returns:
            float: Gini of decile adoption means.
        """
        if not self.daily_adoption:
            return 0.0
        # Compute deciles
        deciles = [np.quantile(self.income, q) for q in np.linspace(0.1, 0.9, 9)]
        last = self.wearing
        rates = []
        prev_thr = -np.inf
        for thr in deciles:
            mask = (self.income > prev_thr) & (self.income <= thr)
            if np.any(mask):
                rates.append(float(np.mean(last[mask])))
            else:
                rates.append(0.0)
            prev_thr = thr
        # Top decile
        mask_top = self.income > prev_thr
        if np.any(mask_top):
            rates.append(float(np.mean(last[mask_top])))
        else:
            rates.append(0.0)
        rates_arr = np.array(rates, dtype=float)
        return self._gini(rates_arr)
        pass

    def run(self) -> Dict[str, object]:
        """
        Run the simulation over the configured number of days and return computed metrics.

        Returns:
            Dict[str, object]: Dictionary containing time series and summary metrics.
        """
        for day in range(self.days):
            self.step(day)

        # Derived metrics
        time_to_50 = self._time_to_threshold(0.5)
        time_to_70 = self._time_to_threshold(0.7)
        sustained_70 = self._sustained_adoption_duration(0.7)
        peak = float(np.max(self.daily_adoption) if self.daily_adoption else 0.0)
        endline = float(self.daily_adoption[-1]) if self.daily_adoption else 0.0
        compliance_rate = float(self.mandated_visits_compliant / self.mandated_visits_total) if self.mandated_visits_total > 0 else 0.0
        noncompliance_rate = 1.0 - compliance_rate  # FIXED: policy_noncompliance_rate per spec

        results = {
            "adoption_rate_over_time": [float(x) for x in self.daily_adoption],
            "time_to_50_percent_adoption": time_to_50,
            "time_to_target_adoption_70": time_to_70,  # FIXED
            "sustained_adoption_duration_above_70": sustained_70,  # FIXED
            "peak_adoption": peak,
            "peak_adoption_rate": peak,  # FIXED alias
            "endline_adoption": endline,
            "policy_compliance_rate": compliance_rate,
            "policy_noncompliance_rate": noncompliance_rate,  # FIXED
            "mask_availability_rate_over_time": [float(x) for x in self.daily_mask_availability],  # FIXED
            "mask_availability_rate_endline": float(self.daily_mask_availability[-1]) if self.daily_mask_availability else 0.0,  # FIXED
            "adherence_probability_mean_over_time": [float(x) for x in self.daily_adherence_prob_mean],  # FIXED
            "adherence_probability_mean_endline": float(self.daily_adherence_prob_mean[-1]) if self.daily_adherence_prob_mean else 0.0,  # FIXED
            "access_inequality_index": self._access_inequality_index(),
            "adoption_inequality_gini_income": self._adoption_inequality_gini_income(),  # FIXED
            "sentiment_polarization": self._sentiment_polarization(),
            "purchase_backlog": [float(x) for x in self.daily_supply_demand_gap],
            "adoption_disparity_by_group": self._adoption_disparity(),
            "policy_effect_on_adoption": self._policy_effect(),
            "campaign_effect_on_attitudes": self._campaign_effect_attitudes(),
            "noncompliance_events": int(self.noncompliance_events),
        }
        return results
        pass

    def save_results(self, filename: str, results: Optional[Dict[str, object]] = None) -> None:
        """
        Save daily time series results to a CSV file.

        Notes:
            - Column renamed to purchase_backlog to match spec.
            - Uses DATA_DIR (if provided) per path handling instructions.

        Args:
            filename (str): Output CSV filename.
            results (Optional[Dict[str, object]]): Results to save; if None, uses current daily series.
        """
        try:
            if results is None:
                results = {
                    "adoption_rate_over_time": [float(x) for x in self.daily_adoption],
                    "purchase_backlog": [float(x) for x in self.daily_supply_demand_gap],
                    "mask_availability_rate_over_time": [float(x) for x in self.daily_mask_availability],
                    "adherence_probability_mean_over_time": [float(x) for x in self.daily_adherence_prob_mean],
                }
            days = max(
                len(results.get("adoption_rate_over_time", [])),
                len(results.get("purchase_backlog", [])),
                len(results.get("mask_availability_rate_over_time", [])),
                len(results.get("adherence_probability_mean_over_time", [])),
            )
            lines = ["day,adoption_rate,purchase_backlog,mask_availability,adherence_probability_mean\n"]
            for d in range(days):
                ar = results["adoption_rate_over_time"][d] if d < len(results["adoption_rate_over_time"]) else ""
                pb = results["purchase_backlog"][d] if d < len(results["purchase_backlog"]) else ""
                ma = results["mask_availability_rate_over_time"][d] if d < len(results["mask_availability_rate_over_time"]) else ""
                ap = results["adherence_probability_mean_over_time"][d] if d < len(results["adherence_probability_mean_over_time"]) else ""
                lines.append(f"{d},{ar},{pb},{ma},{ap}\n")
            out_path = filename
            try:
                if not os.path.isabs(out_path) and DATA_DIR:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    out_path = os.path.join(DATA_DIR, filename)
            except Exception:
                out_path = filename
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error saving results to {filename}: {e}")
        pass

    def visualize(self) -> None:
        """
        Visualize the adoption rate, purchase backlog, and mask availability over time using matplotlib, if available.
        """
        try:
            import matplotlib.pyplot as plt  # Optional dependency

            fig, ax1 = plt.subplots(figsize=(11, 6))
            days_range = np.arange(len(self.daily_adoption))

            ax1.plot(days_range, self.daily_adoption, color="tab:blue", label="Adoption Rate")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Adoption Rate", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.set_ylim(0, 1)

            ax2 = ax1.twinx()
            ax2.plot(
                days_range,
                self.daily_supply_demand_gap[: len(days_range)],
                color="tab:red",
                label="Purchase Backlog",
            )
            ax2.set_ylabel("Purchase Backlog", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            # Mask availability
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.1))
            ax3.plot(days_range, self.daily_mask_availability[: len(days_range)], color="tab:green", label="Mask Availability")
            ax3.set_ylabel("Mask Availability", color="tab:green")
            ax3.tick_params(axis="y", labelcolor="tab:green")
            ax3.set_ylim(0, 1)

            if self.policy_maker.policy_start_day is not None:
                md = int(self.policy_maker.policy_start_day)
                if md < len(days_range):
                    ax1.axvline(md, color="gray", linestyle="--", alpha=0.7, label="Mandate Start")
            for d in sorted(self.campaign_days):
                if d < len(days_range):
                    ax1.axvline(d, color="green", linestyle=":", alpha=0.3)

            fig.tight_layout()
            plt.title("Mask Adoption, Purchase Backlog, and Availability Over Time")
            plt.show()
        except ImportError:
            print("matplotlib not installed; skipping visualization.")
        except Exception as e:
            print(f"Visualization error: {e}")
        pass

    def evaluate(self, evaluation_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the simulation results based on a list of metric names.

        Supported metrics:
            - adoption_rate_mean
            - peak_adoption_rate (alias: peak_adoption)
            - time_to_target_adoption_70
            - time_to_50_percent_adoption
            - policy_noncompliance_rate
            - mask_availability_rate_endline
            - adherence_probability_mean_endline

        Args:
            evaluation_metrics (Optional[List[str]]): Metrics to compute.

        Returns:
            Dict[str, float]: Mapping from metric names to values (NaN for unknown).
        """
        results = {}
        if evaluation_metrics is None:
            return results
        series = np.asarray(self.daily_adoption, dtype=float)
        for metric in evaluation_metrics:
            name = metric.lower()
            if name == "adoption_rate_mean":
                results[metric] = float(np.mean(series)) if series.size else 0.0
            elif name in ("peak_adoption_rate", "peak_adoption"):
                results[metric] = float(np.max(series)) if series.size else 0.0
            elif name in ("time_to_target_adoption_70", "time_to_50_percent_adoption"):
                thr = 0.7 if "70" in name else 0.5
                t = next((d for d, v in enumerate(series) if v >= thr), None)
                results[metric] = float(t) if t is not None else float("nan")
            elif name == "policy_noncompliance_rate":
                rate = 1.0 - (float(self.mandated_visits_compliant / self.mandated_visits_total) if self.mandated_visits_total > 0 else 0.0)
                results[metric] = float(rate)
            elif name == "mask_availability_rate_endline":
                results[metric] = float(self.daily_mask_availability[-1]) if self.daily_mask_availability else 0.0
            elif name == "adherence_probability_mean_endline":
                results[metric] = float(self.daily_adherence_prob_mean[-1]) if self.daily_adherence_prob_mean else 0.0
            else:
                results[metric] = float("nan")
        return results
        pass


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Mask Adoption Behavior Simulation")
    parser.add_argument("--population", type=int, default=5000, help="Population size")
    parser.add_argument("--days", type=int, default=120, help="Number of simulation days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Spec-aligned parameters
    parser.add_argument("--policy_start_day", type=int, default=None, help="Mandate start day")
    parser.add_argument("--policy_end_day", type=int, default=None, help="Mandate end day")
    parser.add_argument("--policy_mask_required_fraction", type=float, default=0.7, help="Fraction of locations requiring masks when policy active [0,1]")
    parser.add_argument("--enforcement_strength", type=float, default=0.5, help="Base enforcement strength [0,1]")

    parser.add_argument("--restock_rate", type=float, default=0.2, help="Retailer restock rate per capita per day")
    parser.add_argument("--retailer_initial_inventory_per_capita", type=float, default=2.0, help="Initial inventory per capita")
    parser.add_argument("--mask_price", type=float, default=1.5, help="Baseline mask price")
    parser.add_argument("--affordability_threshold_income_fraction", type=float, default=0.05, help="Income fraction threshold for affordability")

    parser.add_argument("--campaign_start_day", type=int, default=10, help="First day of campaign messaging")
    parser.add_argument("--campaign_frequency_days", type=int, default=14, help="Spacing between campaign days")
    parser.add_argument("--information_campaign_intensity", type=float, default=0.1, help="Effect size of campaign per day")
    parser.add_argument("--misinformation_rate", type=float, default=0.05, help="Daily fraction of agents affected by misinformation")

    # Backward-compatible flags (mapped internally)
    parser.add_argument("--mandate_day", type=int, default=None, help="(Deprecated) Mandate start day")
    parser.add_argument("--mandate_end_day", type=int, default=None, help="(Deprecated) Mandate end day")
    parser.add_argument("--enforcement", type=float, default=None, help="(Deprecated) Enforcement intensity [0,1]")

    parser.add_argument("--no_viz", action="store_true", help="Disable visualization (default shows a plot if available)")
    parser.add_argument("--save", type=str, default="results.csv", help="CSV filename to save daily results")
    return parser.parse_args()
    pass


def main() -> None:
    """
    CLI entry point: initialize, run, visualize, and save the simulation.

    This function demonstrates how to:
    - Configure and run the simulation
    - Print results as JSON
    - Save time series to a CSV file
    - Visualize results (if matplotlib is available)
    """
    args = parse_args()

    # Merge CLI args into params with backward-compatible mapping
    params = {
        "policy_start_day": args.policy_start_day if args.policy_start_day is not None else args.mandate_day,
        "policy_end_day": args.policy_end_day if args.policy_end_day is not None else args.mandate_end_day,
        "policy_mask_required_fraction": args.policy_mask_required_fraction,
        "enforcement_strength": args.enforcement_strength if args.enforcement_strength is not None else (args.enforcement if args.enforcement is not None else 0.5),
        "restock_rate": args.restock_rate,
        "retailer_initial_inventory_per_capita": args.retailer_initial_inventory_per_capita,
        "mask_price": args.mask_price,
        "affordability_threshold_income_fraction": args.affordability_threshold_income_fraction,
        "campaign_start_day": args.campaign_start_day,
        "campaign_frequency_days": args.campaign_frequency_days,
        "information_campaign_intensity": args.information_campaign_intensity,
        "misinformation_rate": args.misinformation_rate,
    }

    sim = MaskSimulation(
        population_size=args.population,
        days=args.days,
        seed=args.seed,
        params=params,
    )

    results = sim.run()
    print(json.dumps(results, default=float))

    if args.save:
        sim.save_results(args.save, results=results)

    if not args.no_viz:
        sim.visualize()
    pass


# Execute main for both direct execution and sandbox wrapper invocation
main()