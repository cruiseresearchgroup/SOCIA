import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
class Person:
    """
    Represents a person agent in the simulation with attributes relevant to mask adoption and behavior.

    Attributes:
        id (int): Unique identifier for the agent.
        age (Optional[int]): Age of the person. Placeholder for extended models.
        household_id (Optional[int]): Household identifier. Placeholder for extended models.
        socioeconomic_status (Optional[str]): Socioeconomic status categorization. Placeholder for extended models.
        health_risk_level (float): Health risk level in [0,1]. Placeholder for extended models.
        risk_perception (float): Perceived risk in [0,1].
        attitude_toward_masks (float): Attitude in [0,1].
        trust_in_authorities (float): Trust in [0,1].
        susceptibility_to_influence (float): Social susceptibility in [0,1].
        conformity_bias (float): Conformity bias in [0,1].
        political_identity (Optional[float]): Placeholder for extended models.
        access_to_masks (float): Access in [0,1]. Placeholder for extended models.
        has_mask (bool): Whether the person currently holds a mask unit.
        mask_use_state (int): 1 if wearing currently, else 0.
        compliance_propensity (float): Propensity to comply in [0,1].
        social_neighbors (List[int]): IDs of neighbors in the social network.
        workplace_id (Optional[int]): Placeholder for extended models.
        last_fined_day (int): Last day person was fined. Placeholder for extended models.
        last_enforcement_event_day (int): Last enforcement day. Placeholder for extended models.
        opinion_signal (float): Opinion signal placeholder.
    """

    id: int = 0
    age: Optional[int] = None
    household_id: Optional[int] = None
    socioeconomic_status: Optional[str] = None
    health_risk_level: float = 0.0
    risk_perception: float = 0.0
    attitude_toward_masks: float = 0.0
    trust_in_authorities: float = 0.0
    susceptibility_to_influence: float = 0.0
    conformity_bias: float = 0.5
    political_identity: Optional[float] = None
    access_to_masks: float = 0.0
    has_mask: bool = False
    mask_use_state: int = 0
    compliance_propensity: float = 0.0
    social_neighbors: List[int] = field(default_factory=list)
    workplace_id: Optional[int] = None
    last_fined_day: int = -1
    last_enforcement_event_day: int = -1
    opinion_signal: float = 0.0


@dataclass
class Location:
    """
    Represents a location where interactions can occur and mask policies may be enforced.

    Attributes:
        id (int): Unique location ID.
        type (str): Type of the location (e.g., 'home', 'work', 'public').
        capacity (int): Capacity of the location.
        enforces_mask_policy (bool): Whether masks are required and enforced at this location.
        transit_probability (float): Probability of transient visit; placeholder for extended models.
        contact_rate_modifier (float): Multiplier to contact intensity; placeholder for extended models.
    """

    id: int = 0
    type: str = "public"
    capacity: int = 10
    enforces_mask_policy: bool = False
    transit_probability: float = 0.0
    contact_rate_modifier: float = 1.0


@dataclass
class Government:
    """
    Represents government policy configuration affecting mandates and enforcement.

    Attributes:
        mandate_level (str): 'none', 'partial', or 'full'.
        enforcement_probability (float): Probability of enforcement in [0,1].
        fine_amount (float): Fine amount for noncompliance.
        policy_start_day (Optional[int]): Day policy activates.
        policy_end_day (Optional[int]): Day policy deactivates.
        public_guidance_strength (float): Strength of guidance via media.
    """

    mandate_level: str = "none"
    enforcement_probability: float = 0.3
    fine_amount: float = 50.0
    policy_start_day: Optional[int] = None
    policy_end_day: Optional[int] = None
    public_guidance_strength: float = 0.4


@dataclass
class Retailer:
    """
    Represents a retailer supplying masks to the population.

    Attributes:
        id (int): Unique retailer ID.
        inventory_level (float): Current inventory level.
        restock_rate_per_day (float): Number of masks restocked per day.
        mask_price (float): Price per mask; placeholder for extended models.
    """

    id: int = 0
    inventory_level: float = 0.0
    restock_rate_per_day: float = 0.0
    mask_price: float = 1.0


@dataclass
class Media:
    """
    Represents the media system producing messages affecting attitudes and risk perception.

    Attributes:
        message_orientation (str): 'pro_mask', 'neutral', or 'anti_mask'.
        message_intensity (float): Intensity of messaging in [0,1].
        misinformation_rate (float): Fraction of counteracting misinformation [0,1].
        reach_fraction (float): Probability a person receives the message each day [0,1].
    """

    message_orientation: str = "pro_mask"
    message_intensity: float = 0.5
    misinformation_rate: float = 0.1
    reach_fraction: float = 0.7


@dataclass
class Policy:
    """
    Represents a policy object encapsulating mandate configuration and schedule.

    Attributes:
        name (str): Name of the policy.
        mandate_scope (str): Scope description.
        mandate_level (str): 'none', 'partial', or 'full'.
        start_day (Optional[int]): Day policy activates.
        end_day (Optional[int]): Day policy ends.
        enforcement_probability (float): Probability of enforcement upon noncompliance [0,1].
        penalty (float): Penalty for noncompliance; placeholder for extended models.
    """

    name: str = "Mask Mandate"
    mandate_scope: str = "work+public"
    mandate_level: str = "none"
    start_day: Optional[int] = None
    end_day: Optional[int] = None
    enforcement_probability: float = 0.3
    penalty: float = 50.0


class MaskSimulation:
    """
    Core simulation engine modeling mask adoption under social influence, policy, media, and supply constraints.

    This implementation uses a compact agent-based approach with numpy arrays and a small-world network
    produced by networkx. It computes the required metrics and provides methods to run, visualize, and save results.

    Key FIXES applied based on prior iteration feedback:
    - FIXED: Reintroduced a functional simulation engine producing required metrics (MaskSimulation.run()).
    - FIXED: Implemented policy activation handling with start and end days, modulating wearing probabilities.
    - FIXED: Implemented adherence dynamics (habit formation and fatigue) influencing attitudes and behavior.
    - FIXED: Implemented supply-demand with retailer inventory, restock, and rationing, tracking purchase_backlog.
    - FIXED: Implemented media campaigns and misinformation effects; added campaign_effect_on_attitudes metric.
    - FIXED: Implemented adoption disparity by income groups.
    - FIXED: Implemented noncompliance events via location interactions and enforcement.
    - FIXED: Added CLI runner in main(), removed any Docker dependency from runtime path.
    - FIXED: Enforced even k in small-world graph to prevent runtime errors.
    - FIXED: Added required metrics per spec: time_to_50_percent_adoption, peak_adoption, endline_adoption,
             policy_compliance_rate, access_inequality_index, sentiment_polarization, purchase_backlog.
    - FIXED: Tracked mandated visits and compliant visits for policy_compliance_rate.
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

        # Default parameters for this compact engine; can be overridden by input.
        self.params: Dict = {
            "initial_adoption_rate": 0.15,
            "average_degree": 10,
            "social_influence_weight": 0.4,
            "risk_perception_base": 0.3,
            "trust_in_authorities_mean": 0.5,
            "decision_noise": 0.1,
            "fatigue_rate": 0.01,
            "habit_formation_rate": 0.02,
            "mask_price": 1.5,
            "initial_inventory_per_capita": 2.0,
            "restock_rate_per_day": 0.2,
            "rationing_limit_per_purchase": 5,
            "policy_initial_state": "recommendation",
            "policy_mandate_day": None,
            "policy_mandate_end_day": None,
            "enforcement_intensity": 0.5,
            "fine_amount": 50.0,
            "campaign_start_day": 10,
            "campaign_frequency_days": 14,
            "campaign_effect_size": 0.1,
            "misinformation_rate": 0.05,
            "media_message_frequency": 1.0,
        }
        if params:
            self.params.update(params)

        # Social network construction
        k = max(2, int(self.params["average_degree"]))
        # Ensure even degree for Watts-Strogatz small-world network to prevent runtime errors
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
            lam=max(0.0, self.params["initial_inventory_per_capita"]), size=self.n
        ).astype(float)
        self.wearing = initial_wearing.astype(float)

        # Retailer inventory pool (aggregate)
        self.retailer_inventory = float(self.n * self.params["initial_inventory_per_capita"])

        # Policy and campaign schedule
        self.mandate_day = self.params["policy_mandate_day"]
        self.mandate_end_day = self.params["policy_mandate_end_day"]
        self.enforcement = float(self.params["enforcement_intensity"])

        self.campaign_days = set()
        if self.params["campaign_start_day"] is not None:
            d = int(self.params["campaign_start_day"])
            while d < self.days:
                self.campaign_days.add(d)
                d += max(1, int(self.params["campaign_frequency_days"]))

        # Time series collection
        self.daily_adoption: List[float] = []
        self.daily_supply_demand_gap: List[float] = []
        self.daily_attitude_mean: List[float] = []
        # Enforcement/compliance tracking
        self.noncompliance_events: int = 0
        self.mandated_visits_total: int = 0
        self.mandated_visits_compliant: int = 0

    def _policy_active(self, day: int) -> bool:
        """
        Return whether policy is active on a given day based on configuration.
        """
        if self.mandate_day is None:
            return False
        if self.mandate_end_day is None:
            return day >= int(self.mandate_day)
        return int(self.mandate_day) <= day <= int(self.mandate_end_day)

    def _social_influence(self) -> None:
        """
        Update attitudes based on social influence from neighbors' observed mask wearing.
        Uses a DeGroot-like averaging with susceptibility scaling.
        """
        neigh_mean = np.zeros(self.n)
        for i, neigh in enumerate(self.neighbors):
            if neigh:
                neigh_mean[i] = float(np.mean(self.wearing[neigh]))
        w = float(self.params["social_influence_weight"])
        s = self.susc
        self.attitude = np.clip((1 - w * s) * self.attitude + (w * s) * neigh_mean, 0, 1)

    def _media_influence(self, day: int) -> None:
        """
        Apply media influence, including campaign boosts and misinformation offsets.
        """
        if self.params["media_message_frequency"] <= 0:
            return
        camp = float(self.params["campaign_effect_size"]) if day in self.campaign_days else 0.0
        mis = float(self.params["misinformation_rate"])
        mis_mask = self.rng.random(self.n) < mis

        delta_att = 0.5 * camp * (~mis_mask) - 0.5 * camp * (mis_mask)
        delta_trust = 0.3 * camp * (~mis_mask) - 0.2 * camp * (mis_mask)
        delta_risk = 1.0 * camp * (~mis_mask) - 0.5 * camp * (mis_mask)

        self.attitude = np.clip(self.attitude + delta_att, 0, 1)
        self.trust = np.clip(self.trust + delta_trust, 0, 1)
        self.risk = np.clip(self.risk + delta_risk, 0, 1)

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

    def _restock_retailer(self) -> None:
        """
        Restock the retailer inventory pool based on daily restock rate per capita.
        """
        self.retailer_inventory += float(self.params["restock_rate_per_day"] * self.n)

    def _purchase_masks(self) -> None:
        """
        Process mask purchase attempts with rationing and limited inventory.
        Records daily supply-demand gap as demand_units - units_sold.
        """
        need = self.stock < 1.0
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

        if self.retailer_inventory > 0:
            buyers = np.where(attempt)[0]
            self.rng.shuffle(buyers)
            for i in buyers:
                if self.retailer_inventory <= 0:
                    break
                qty = float(min(max_per_purchase, max(1.0, math.ceil(1.0 - self.stock[i]))))
                qty = float(min(qty, self.retailer_inventory))
                self.stock[i] += qty
                self.retailer_inventory -= qty
                sold += qty

        self.daily_supply_demand_gap.append(float(demand_units - sold))

    def _decide_and_wear(self, day: int) -> None:
        """
        Decide daily wearing behavior based on latent variables and policy effects.
        """
        noise = self.rng.normal(0, float(self.params["decision_noise"]), size=self.n)
        base = 0.05 + 0.5 * self.attitude + 0.2 * self.trust + 0.2 * self.risk - 0.1 * self.fatigue + noise
        policy_boost = 0.0
        if self._policy_active(day):
            policy_boost = 0.4 * self.enforcement * self.compliance
        wear_prob = np.clip(base + policy_boost, 0, 1)
        can_wear = self.stock > 0
        self.wearing = ((self.rng.random(self.n) < wear_prob) & can_wear).astype(float)
        self.stock = np.where(self.wearing > 0, np.maximum(0.0, self.stock - 1.0), self.stock)
        self.fatigue = np.clip(self.fatigue + 0.01 * (1 - self.wearing), 0, 1)

    def _locations_and_enforcement(self, day: int) -> None:
        """
        Simulate visits to locations with potential mask requirements and track enforcement.
        """
        required_fraction = 0.5 + 0.3 * self.enforcement if self._policy_active(day) else 0.1
        visitors = self.rng.random(self.n) < 0.6
        required = (self.rng.random(self.n) < required_fraction) & visitors
        noncompliant = required & (self.wearing == 0)
        compliant = required & (self.wearing == 1)

        self.mandated_visits_total += int(np.sum(required))
        self.mandated_visits_compliant += int(np.sum(compliant))

        events = int(np.sum(noncompliant))
        self.noncompliance_events += events

        if events > 0:
            idx = np.where(noncompliant)[0]
            self.compliance[idx] = np.clip(self.compliance[idx] + 0.02 * self.enforcement, 0, 1)

    def step(self, day: int) -> None:
        """
        Execute one simulation day: restock, media influence, social influence, habit/fatigue,
        purchase masks, decide wearing, and location enforcement. Records metrics.
        """
        self._restock_retailer()
        self._media_influence(day)
        self._social_influence()
        self._fatigue_and_habit()
        self._purchase_masks()
        self._decide_and_wear(day)
        self._locations_and_enforcement(day)
        self.daily_adoption.append(float(np.mean(self.wearing)))
        self.daily_attitude_mean.append(float(np.mean(self.attitude)))

    def _time_to_threshold(self, thr: float = 0.7) -> Optional[int]:
        """
        Compute the first day the adoption rate reaches or exceeds a threshold.
        """
        for d, val in enumerate(self.daily_adoption):
            if val >= thr:
                return d
        return None

    def _policy_effect(self) -> Optional[float]:
        """
        Compute average adoption change in the 7 days after mandate starts versus 7 days before.
        """
        if self.mandate_day is None:
            return None
        md = int(self.mandate_day)
        if md >= len(self.daily_adoption):
            return None
        pre = self.daily_adoption[max(0, md - 7) : md]
        post = self.daily_adoption[md : md + 7]
        if not pre or not post:
            return None
        return float(np.mean(post) - np.mean(pre))

    def _campaign_effect_attitudes(self) -> Optional[float]:
        """
        Estimate campaign effect on attitudes as the average difference in mean attitude
        between each campaign day and the previous day.
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

    def _adoption_disparity(self) -> float:
        """
        Compute adoption disparity by income group (high quintile minus low quintile) on the last day.
        """
        if not self.daily_adoption:
            return 0.0
        q20, q80 = np.quantile(self.income, [0.2, 0.8])
        low = self.income <= q20
        high = self.income >= q80
        last = self.wearing
        return float(np.mean(last[high]) - np.mean(last[low]))

    def _gini(self, x: np.ndarray) -> float:
        """
        Compute the Gini coefficient for a non-negative array.
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x >= 0)]
        if x.size == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = x_sorted.size
        cumx = np.cumsum(x_sorted)
        if cumx[-1] <= 0:
            return 0.0
        return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)

    def _access_inequality_index(self) -> float:
        """
        Compute access inequality index as the Gini coefficient over mask stock across agents.
        """
        return self._gini(self.stock)

    def _sentiment_polarization(self) -> float:
        """
        Compute a simple sentiment polarization metric combining attitude dispersion and
        average neighbor disagreement.
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

    def run(self) -> Dict[str, object]:
        """
        Run the simulation over the configured number of days and return computed metrics.
        """
        for day in range(self.days):
            self.step(day)
        results = {
            "adoption_rate_over_time": [float(x) for x in self.daily_adoption],
            "time_to_50_percent_adoption": self._time_to_threshold(0.5),
            "peak_adoption": float(np.max(self.daily_adoption) if self.daily_adoption else 0.0),
            "endline_adoption": float(self.daily_adoption[-1]) if self.daily_adoption else 0.0,
            "policy_compliance_rate": float(self.mandated_visits_compliant / self.mandated_visits_total) if self.mandated_visits_total > 0 else 0.0,
            "access_inequality_index": self._access_inequality_index(),
            "sentiment_polarization": self._sentiment_polarization(),
            "purchase_backlog": [float(x) for x in self.daily_supply_demand_gap],
            "adoption_disparity_by_group": self._adoption_disparity(),
            "policy_effect_on_adoption": self._policy_effect(),
            "campaign_effect_on_attitudes": self._campaign_effect_attitudes(),
            "noncompliance_events": int(self.noncompliance_events),
        }
        return results

    def save_results(self, filename: str, results: Optional[Dict[str, object]] = None) -> None:
        """
        Save daily time series results to a CSV file.

        Notes:
            - Column renamed to purchase_backlog to match spec.
            - Uses DATA_DIR (if provided) per path handling instructions.
        """
        try:
            if results is None:
                results = {
                    "adoption_rate_over_time": [float(x) for x in self.daily_adoption],
                    "purchase_backlog": [float(x) for x in self.daily_supply_demand_gap],
                }
            days = max(
                len(results.get("adoption_rate_over_time", [])),
                len(results.get("purchase_backlog", [])),
            )
            lines = ["day,adoption_rate,purchase_backlog\n"]
            for d in range(days):
                ar = results["adoption_rate_over_time"][d] if d < len(results["adoption_rate_over_time"]) else ""
                pb = results["purchase_backlog"][d] if d < len(results["purchase_backlog"]) else ""
                lines.append(f"{d},{ar},{pb}\n")
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

    def visualize(self) -> None:
        """
        Visualize the adoption rate and purchase backlog over time using matplotlib, if available.
        """
        try:
            import matplotlib.pyplot as plt  # Optional dependency

            fig, ax1 = plt.subplots(figsize=(10, 5))
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

            if self.mandate_day is not None:
                md = int(self.mandate_day)
                if md < len(days_range):
                    ax1.axvline(md, color="gray", linestyle="--", alpha=0.7, label="Mandate Start")
            for d in sorted(self.campaign_days):
                if d < len(days_range):
                    ax1.axvline(d, color="green", linestyle=":", alpha=0.4)

            fig.tight_layout()
            plt.title("Mask Adoption and Purchase Backlog Over Time")
            plt.show()
        except ImportError:
            print("matplotlib not installed; skipping visualization.")
        except Exception as e:
            print(f"Visualization error: {e}")

    def evaluate(self, evaluation_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate the simulation results based on a list of metric names.
        """
        results = {}
        if evaluation_metrics is None:
            return results
        series = np.asarray(self.daily_adoption, dtype=float)
        for metric in evaluation_metrics:
            name = metric.lower()
            if name == "adoption_rate_over_time":
                results[metric] = float(np.mean(series)) if series.size else 0.0
            elif name == "peak_adoption_rate" or name == "peak_adoption":
                results[metric] = float(np.max(series)) if series.size else 0.0
            elif name == "time_to_target_adoption_70" or name == "time_to_50_percent_adoption":
                thr = 0.7 if "70" in name else 0.5
                t = None
                for d, v in enumerate(series):
                    if v >= thr:
                        t = d
                        break
                results[metric] = float(t) if t is not None else float("nan")
            else:
                results[metric] = float("nan")
        return results


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the simulation.
    """
    parser = argparse.ArgumentParser(description="Mask Adoption Behavior Simulation")
    parser.add_argument("--population", type=int, default=5000, help="Population size")
    parser.add_argument("--days", type=int, default=120, help="Number of simulation days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mandate_day", type=int, default=None, help="Mandate start day")
    parser.add_argument("--mandate_end_day", type=int, default=None, help="Mandate end day")
    parser.add_argument("--enforcement", type=float, default=0.5, help="Enforcement intensity [0,1]")
    parser.add_argument(
        "--campaign_start_day", type=int, default=10, help="First day of campaign messaging"
    )
    parser.add_argument(
        "--campaign_frequency_days", type=int, default=14, help="Spacing between campaign days"
    )
    parser.add_argument(
        "--campaign_effect_size", type=float, default=0.1, help="Effect size of campaign per day"
    )
    parser.add_argument(
        "--misinformation_rate",
        type=float,
        default=0.05,
        help="Daily fraction of agents affected by misinformation",
    )
    parser.add_argument(
        "--no_viz", action="store_true", help="Disable visualization (default shows a plot if available)"
    )
    parser.add_argument(
        "--save", type=str, default="results.csv", help="CSV filename to save daily results"
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point: initialize, run, visualize, and save the simulation.
    """
    args = parse_args()

    sim = MaskSimulation(
        population_size=args.population,
        days=args.days,
        seed=args.seed,
        params={
            "policy_mandate_day": args.mandate_day,
            "policy_mandate_end_day": args.mandate_end_day,
            "enforcement_intensity": args.enforcement,
            "campaign_start_day": args.campaign_start_day,
            "campaign_frequency_days": args.campaign_frequency_days,
            "campaign_effect_size": args.campaign_effect_size,
            "misinformation_rate": args.misinformation_rate,
        },
    )

    results = sim.run()
    print(json.dumps(results, default=float))

    if args.save:
        sim.save_results(args.save, results=results)

    if not args.no_viz:
        sim.visualize()



# Execute main for both direct execution and sandbox wrapper invocation
main()