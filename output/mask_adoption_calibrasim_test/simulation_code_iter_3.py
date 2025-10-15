import os
import json
import math
import traceback
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import networkx as nx  # Optional; used for potential future network analytics
except Exception:
    nx = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# Environment path setup per instructions
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the logistic sigmoid function in a numerically stable way.

    Parameters
    ----------
    x : np.ndarray
        Input array of real numbers.

    Returns
    -------
    np.ndarray
        Sigmoid applied element-wise to x.
    """
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    negative = ~positive
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    expx = np.exp(x[negative])
    out[negative] = expx / (1 + expx)
    pass
    return out


def gini(values: List[float]) -> float:
    """
    Compute the Gini coefficient for a list of non-negative values.

    Parameters
    ----------
    values : List[float]
        List of non-negative values representing shares or rates.

    Returns
    -------
    float
        Gini coefficient in [0,1]. Returns 0 if all values are equal or list is empty.
    """
    if not values:
        pass
        return 0.0
    vals = np.array(values, dtype=float)
    if np.all(vals == 0):
        pass
        return 0.0
    sorted_vals = np.sort(vals)
    n = len(vals)
    cumvals = np.cumsum(sorted_vals)
    g = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    pass
    return float(max(0.0, min(1.0, g)))


class Metrics:
    """
    Collection of static methods for metric calculations used for evaluation.

    Methods include RMSE, MAE, Peak, ThresholdCrossingTime, ValueAt, and GiniAcrossGroups.
    """
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error (RMSE).
        """
        if y_true is None or y_pred is None:
            pass
            return float("nan")
        n = min(len(y_true), len(y_pred))
        if n == 0:
            pass
            return float("nan")
        # Drop NaNs from y_true (observed gaps)
        mask = ~np.isnan(y_true[:n])
        if not np.any(mask):
            pass
            return float("nan")
        diff = y_pred[:n][mask] - y_true[:n][mask]
        val = float(np.sqrt(np.mean(diff * diff)))
        pass
        return val

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Absolute Error (MAE).
        """
        if y_true is None or y_pred is None:
            pass
            return float("nan")
        n = min(len(y_true), len(y_pred))
        if n == 0:
            pass
            return float("nan")
        mask = ~np.isnan(y_true[:n])
        if not np.any(mask):
            pass
            return float("nan")
        diff = np.abs(y_pred[:n][mask] - y_true[:n][mask])
        val = float(np.mean(diff))
        pass
        return val

    @staticmethod
    def peak(y: np.ndarray, start: int = 0, end: Optional[int] = None) -> float:
        """
        Compute the peak (maximum) value of a time series within a window.
        """
        if y is None or len(y) == 0:
            pass
            return float("nan")
        if end is None:
            end = len(y) - 1
        start = max(0, start)
        end = min(len(y) - 1, end)
        if end < start:
            pass
            return float("nan")
        val = float(np.max(y[start:end + 1]))
        pass
        return val

    @staticmethod
    def threshold_crossing_time(y: np.ndarray, threshold: float, start_day: int = 0) -> float:
        """
        Compute the first day when the series crosses or equals a threshold.
        """
        if y is None or len(y) == 0:
            pass
            return float("nan")
        for t, val in enumerate(y):
            if val >= threshold:
                pass
                return float(t - start_day)
        pass
        return float("nan")

    @staticmethod
    def value_at(y: np.ndarray, day: int) -> float:
        """
        Return value of a series at specific day index.
        """
        if y is None or day < 0 or day >= len(y):
            pass
            return float("nan")
        pass
        return float(y[day])

    @staticmethod
    def gini_across_groups(group_values: Dict[str, float]) -> float:
        """
        Compute Gini coefficient across group values.
        """
        vals = list(group_values.values()) if group_values else []
        val = gini(vals)
        pass
        return val


class DataLoader:
    """
    Loads and validates input data: agent attributes, social network, and panel training data.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        pass

    def load_agent_attributes(self) -> Optional[pd.DataFrame]:
        """
        Load agent_attributes.csv into a DataFrame.
        """
        file_path = os.path.join(self.data_dir, "agent_attributes.csv")
        if not os.path.exists(file_path):
            print(f"Warning: agent_attributes.csv not found at {file_path}. Proceeding with synthetic attributes.")
            pass
            return None
        try:
            df = pd.read_csv(file_path)
            if "agent_id" not in df.columns:
                raise ValueError("agent_attributes.csv missing required 'agent_id' column.")
            df["agent_id"] = df["agent_id"].astype(int)
            pass
            return df
        except Exception as e:
            print("Error loading agent_attributes.csv:", e)
            traceback.print_exc()
            pass
            return None

    def load_social_network(self) -> Dict[int, Dict[str, List[int]]]:
        """
        Load social_network.json.
        """
        file_path = os.path.join(self.data_dir, "social_network.json")
        if not os.path.exists(file_path):
            print(f"Warning: social_network.json not found at {file_path}. Proceeding with empty network.")
            pass
            return {}
        try:
            with open(file_path, "r") as f:
                raw = json.load(f)
            network = {}
            for k, v in raw.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                network[idx] = {
                    "family": list(map(int, v.get("family", []))),
                    "work_school": list(map(int, v.get("work_school", []))),
                    "community": list(map(int, v.get("community", []))),
                    "all": list(map(int, v.get("all", []))),
                }
            pass
            return network
        except Exception as e:
            print("Error loading social_network.json:", e)
            traceback.print_exc()
            pass
            return {}

    def load_train_data(self) -> Optional[pd.DataFrame]:
        """
        Load train_data.csv panel data.
        """
        file_path = os.path.join(self.data_dir, "train_data.csv")
        if not os.path.exists(file_path):
            print(f"Warning: train_data.csv not found at {file_path}. Proceeding without empirical states.")
            pass
            return None
        try:
            df = pd.read_csv(file_path)
            expected_cols = {"day", "agent_id", "wearing_mask", "received_info"}
            if not expected_cols.issubset(df.columns):
                missing = expected_cols - set(df.columns)
                raise ValueError(f"train_data.csv missing columns: {missing}")
            df["day"] = df["day"].astype(int)
            df["agent_id"] = df["agent_id"].astype(int)
            pass
            return df
        except Exception as e:
            print("Error loading train_data.csv:", e)
            traceback.print_exc()
            pass
            return None


class NetworkManager:
    """
    Builds multiplex neighbor lists per layer from the social network JSON.
    """
    def __init__(self, symmetrize_edges: bool = True, layer_overlap_policy: str = "merge_with_weight_sum"):
        self.symmetrize_edges = symmetrize_edges
        self.layer_overlap_policy = layer_overlap_policy
        pass

    def build_neighbors(self, network_json: Dict[int, Dict[str, List[int]]]) -> Tuple[Dict[str, Dict[int, List[int]]], Dict[str, Dict[int, int]]]:
        """
        Build neighbor lists per layer.
        """
        layers = ["family", "work_school", "community"]
        neighbors_by_layer: Dict[str, Dict[int, List[int]]] = {L: {} for L in layers}

        # Initialize with cleaned lists
        agent_ids = sorted(list(network_json.keys()))

        for i in agent_ids:
            for L in layers:
                raw_neighbors = list(set(network_json.get(i, {}).get(L, [])))
                raw_neighbors = [int(j) for j in raw_neighbors if int(j) != int(i)]
                neighbors_by_layer[L][i] = raw_neighbors

        if self.symmetrize_edges:
            for L in layers:
                for i in agent_ids:
                    for j in neighbors_by_layer[L].get(i, []):
                        if i not in neighbors_by_layer[L]:
                            neighbors_by_layer[L][i] = []
                        if j not in neighbors_by_layer[L]:
                            neighbors_by_layer[L][j] = []
                        if i not in neighbors_by_layer[L][j]:
                            neighbors_by_layer[L][j].append(i)

        # Deduplicate and finalize
        for L in layers:
            for i in agent_ids:
                neighbors_by_layer[L][i] = sorted(list(set(neighbors_by_layer[L].get(i, []))))

        degree_by_layer: Dict[str, Dict[int, int]] = {L: {} for L in layers}
        for L in layers:
            for i in agent_ids:
                degree_by_layer[L][i] = len(neighbors_by_layer[L].get(i, []))

        pass
        return neighbors_by_layer, degree_by_layer


class Person:
    """
    Individual agent representing a person in the simulation.
    """
    def __init__(self, agent_id: int):
        self.id = int(agent_id)
        self.age_group: str = "Unknown"
        self.occupation: str = "Unknown"
        self.risk_perception: float = 0.5
        self.trust_in_institutions: float = 0.5
        self.pro_social_pref: float = 0.5
        self.mask_attitude: float = 0.0
        self.is_mask_wearing: int = 0
        self.received_info: int = 0
        self.compliance_propensity: float = 0.5
        self.peer_threshold: float = 0.5
        self.media_exposure_level: float = 1.0
        self.location_id: int = -1
        self.household_id: int = -1
        self.budget: float = 100.0
        self.mask_inventory: int = 1
        pass

    def decide_mask_wearing(self):
        pass

    def update_risk_perception(self):
        pass

    def interact_with_peers(self):
        pass

    def consume_media(self):
        pass

    def respond_to_policy(self):
        pass

    def purchase_masks(self):
        pass

    def move_between_locations(self):
        pass

    def share_opinion(self):
        pass


class Household:
    """
    Represents a household grouping of persons.
    """
    def __init__(self, household_id: int, member_ids: List[int]):
        self.id = int(household_id)
        self.members = list(member_ids)
        self.shared_budget: float = 200.0
        self.shared_mask_inventory: int = 2
        pass

    def pool_resources(self):
        pass

    def share_masks(self):
        pass

    def intra_household_influence(self):
        pass


class Location:
    """
    Represents a physical or social location (e.g., work/school, community venue).
    """
    def __init__(self, location_id: int, loc_type: str = "community", capacity: int = 200):
        self.id = int(location_id)
        self.type = str(loc_type)
        self.capacity = int(capacity)
        self.mask_requirement: float = 0.0
        self.enforcement_level: float = 0.0
        pass

    def enforce_policy(self):
        pass

    def host_visits(self):
        pass


class GovernmentAgency:
    """
    Represents a government entity controlling mask mandates and communications.
    """
    def __init__(self, agency_id: int = 0, mandate_strength: float = 0.6, enforcement_resources: float = 1.0, communication_intensity: float = 1.0):
        self.id = int(agency_id)
        self.mandate_strength = float(mandate_strength)
        self.enforcement_resources = float(enforcement_resources)
        self.communication_intensity = float(communication_intensity)
        pass

    def set_mask_policy(self):
        pass

    def enforce_policy(self):
        pass

    def broadcast_guidance(self):
        pass


class MediaChannel:
    """
    Represents a media channel broadcasting information to the population.
    """
    def __init__(self, channel_id: int = 0, reach: float = 1.0, bias: float = 0.0, message_intensity: float = 1.0):
        self.id = int(channel_id)
        self.reach = float(reach)
        self.bias = float(bias)
        self.message_intensity = float(message_intensity)
        pass

    def broadcast_information(self):
        pass

    def adjust_message_intensity(self):
        pass


class Retailer:
    """
    Represents a retailer supplying masks.
    """
    def __init__(self, retailer_id: int = 0, stock_level: float = 0.0, price: float = 1.0, restock_rate: float = 0.2):
        self.id = int(retailer_id)
        self.stock_level = float(stock_level)
        self.price = float(price)
        self.restock_rate = float(restock_rate)
        pass

    def sell_masks(self):
        pass

    def restock(self):
        pass

    def adjust_price(self):
        pass


class ExposureCalculator:
    """
    Computes per-agent, per-layer peer mask fractions and info contact counts each day.
    """
    def __init__(self, w_family: float = 3.0, w_work_school: float = 1.8, w_community: float = 1.0, mobility_rate: float = 0.6):
        self.w_family = float(w_family)
        self.w_work_school = float(w_work_school)
        self.w_community = float(w_community)
        self.mobility_rate = float(mobility_rate)
        pass

    def compute(
        self,
        neighbor_index_by_layer: Dict[str, List[np.ndarray]],
        is_mask_wearing: np.ndarray,
        received_info: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute exposure metrics based on previous states using precomputed neighbor indices.
        """
        layers = ["family", "work_school", "community"]
        N = len(is_mask_wearing)
        peer_mask_fraction = {L: np.zeros(N, dtype=float) for L in layers}
        info_contacts = {L: np.zeros(N, dtype=float) for L in layers}
        weight_map = {
            "family": self.w_family,
            "work_school": self.w_work_school,
            "community": self.w_community,
        }

        for L in layers:
            wL = weight_map[L]
            neighbors_list = neighbor_index_by_layer.get(L, [])
            if not neighbors_list:
                continue
            for idx in range(N):
                n_idx = neighbors_list[idx] if idx < len(neighbors_list) else np.array([], dtype=int)
                if n_idx.size == 0:
                    peer_mask_fraction[L][idx] = 0.0
                    info_contacts[L][idx] = 0.0
                    continue
                mask_count = float(np.sum(is_mask_wearing[n_idx]))
                info_count = float(np.sum(received_info[n_idx]))
                deg = float(n_idx.size)
                frac = (mask_count / deg) * wL * self.mobility_rate if deg > 0 else 0.0
                peer_mask_fraction[L][idx] = np.clip(frac, 0.0, 1.0)
                info_contacts[L][idx] = info_count * self.mobility_rate * wL

        result = {
            "peer_mask_fraction": peer_mask_fraction,
            "info_contacts": info_contacts,
        }
        pass
        return result


class PolicyAndMedia:
    """
    Activates mandate from a configured start day and broadcasts guidance; generates policy and media signals.
    """
    def __init__(self, policy_start_day: int = 10, mandate_strength: float = 0.6, enforcement_probability: float = 0.4, communication_intensity: float = 1.0):
        self.policy_start_day = int(policy_start_day)
        self.mandate_strength = float(mandate_strength)
        self.enforcement_probability = float(enforcement_probability)
        self.communication_intensity = float(communication_intensity)
        pass

    def step(self, day: int) -> Dict[str, Any]:
        """
        Compute policy and media signals for the current day.
        """
        if day >= self.policy_start_day:
            mandate_active = 1
            enforcement_level = float(self.enforcement_probability)
            guidance_intensity = float(self.communication_intensity * self.mandate_strength)
        else:
            mandate_active = 0
            enforcement_level = 0.0
            guidance_intensity = float(self.communication_intensity * 0.5)
        signals = {
            "mandate_active": mandate_active,
            "enforcement_level": enforcement_level,
            "guidance_intensity": guidance_intensity,
        }
        pass
        return signals


class InformationDiffusion:
    """
    Updates each agent’s received_info via peer contact hazards and exogenous media.
    """
    def __init__(
        self,
        p_info_contact_family: float = 0.2,
        p_info_contact_work_school: float = 0.12,
        p_info_contact_community: float = 0.06,
        lambda_media_daily: float = 0.01,
        media_effect_weight: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.p_info_contact_family = float(p_info_contact_family)
        self.p_info_contact_work_school = float(p_info_contact_work_school)
        self.p_info_contact_community = float(p_info_contact_community)
        self.lambda_media_daily = float(lambda_media_daily)
        self.media_effect_weight = float(media_effect_weight)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        pass

    def step(self, received_info: np.ndarray, info_contacts: Dict[str, np.ndarray], guidance_intensity: float) -> np.ndarray:
        """
        Update received_info monotone via hazards from media and peers.
        """
        N = len(received_info)
        out = received_info.copy()
        p_media = 1.0 - np.exp(-(self.lambda_media_daily * self.media_effect_weight * max(guidance_intensity, 1e-6)))
        p_media = float(max(0.0, min(1.0, p_media)))

        fam_contacts = info_contacts.get("family", np.zeros(N))
        work_contacts = info_contacts.get("work_school", np.zeros(N))
        comm_contacts = info_contacts.get("community", np.zeros(N))

        # Compute per-agent peer hazard using independent contact approximation
        p_peer_fam = 1.0 - np.power((1.0 - self.p_info_contact_family), fam_contacts)
        p_peer_work = 1.0 - np.power((1.0 - self.p_info_contact_work_school), work_contacts)
        p_peer_comm = 1.0 - np.power((1.0 - self.p_info_contact_community), comm_contacts)

        p_total = 1.0 - (1.0 - p_media) * (1.0 - p_peer_fam) * (1.0 - p_peer_work) * (1.0 - p_peer_comm)
        p_total = np.clip(p_total, 0.0, 1.0)

        # Only those who haven't received info can transition
        mask = (out == 0)
        draws = self.rng.random(N)
        out[mask & (draws < p_total)] = 1
        pass
        return out


class MaskAdoptionDecision:
    """
    Decides mask wearing daily using logistic probability with persistence, social exposure,
    received info, and policy boost. Applies inventory constraint.
    """
    def __init__(
        self,
        alpha_intercept: float = -3.0,
        gamma_risk_perception: float = 1.5,
        beta_peer_family: float = 1.2,
        beta_peer_work_school: float = 0.8,
        beta_peer_community: float = 0.5,
        delta_received_info: float = 1.0,
        phi_persistence: float = 2.5,
        social_influence_weight: float = 0.5,
        compliance_decay_rate: float = 0.01,
        recovery_or_dropout_prob: float = 0.02,
        age_mods: Optional[Dict[str, float]] = None,
        occ_mods: Optional[Dict[str, float]] = None,
        min_inventory_to_wear: int = 1,
        rng: Optional[np.random.Generator] = None,
        policy_start_day: int = 10,
        theta_mandate: float = 0.3,
        theta_enforce: float = 0.8,
        beta_household: float = 0.2
    ):
        self.alpha_intercept = float(alpha_intercept)
        self.gamma_risk_perception = float(gamma_risk_perception)
        self.beta_peer_family = float(beta_peer_family)
        self.beta_peer_work_school = float(beta_peer_work_school)
        self.beta_peer_community = float(beta_peer_community)
        self.delta_received_info = float(delta_received_info)
        self.phi_persistence = float(phi_persistence)
        self.social_influence_weight = float(social_influence_weight)
        self.compliance_decay_rate = float(compliance_decay_rate)
        self.recovery_or_dropout_prob = float(recovery_or_dropout_prob)
        self.age_mods = age_mods if age_mods is not None else {
            "Youth": 1.1,
            "Young Adult": 1.0,
            "Middle Age": 0.95,
            "Older Adult": 1.05,
        }
        self.occ_mods = occ_mods if occ_mods is not None else {
            "Student": 1.15,
            "White Collar": 1.0,
            "Blue Collar": 0.95,
            "Service": 1.05,
        }
        self.min_inventory_to_wear = int(min_inventory_to_wear)
        self.policy_start_day = int(policy_start_day)
        self.theta_mandate = float(theta_mandate)
        self.theta_enforce = float(theta_enforce)
        self.beta_household = float(beta_household)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        pass

    def step(
        self,
        day: int,
        is_mask_wearing: np.ndarray,
        received_info: np.ndarray,
        risk_perception: np.ndarray,
        age_groups: List[str],
        occupations: List[str],
        peer_mask_fraction: Dict[str, np.ndarray],
        mandate_active: int,
        enforcement_level: float,
        mask_inventory: np.ndarray,
        trust: np.ndarray,
        compliance_propensity: np.ndarray,
        household_share: np.ndarray
    ) -> np.ndarray:
        """
        Compute new mask-wearing states for all agents.
        """
        N = len(is_mask_wearing)
        prev = is_mask_wearing.astype(int)
        z = np.ones(N, dtype=float) * self.alpha_intercept
        z += self.gamma_risk_perception * risk_perception
        peer_term = (
            self.beta_peer_family * peer_mask_fraction.get("family", np.zeros(N)) +
            self.beta_peer_work_school * peer_mask_fraction.get("work_school", np.zeros(N)) +
            self.beta_peer_community * peer_mask_fraction.get("community", np.zeros(N))
        )
        # Group modifiers
        age_mod_array = np.array([self.age_mods.get(a, 1.0) for a in age_groups], dtype=float)
        occ_mod_array = np.array([self.occ_mods.get(o, 1.0) for o in occupations], dtype=float)
        infl_mod = self.social_influence_weight * age_mod_array * occ_mod_array
        z += infl_mod * peer_term
        z += self.delta_received_info * received_info
        z += self.phi_persistence * prev

        # Policy effect with decay
        policy_decay = math.exp(-self.compliance_decay_rate * max(0, int(day - self.policy_start_day))) if mandate_active else 1.0
        policy_component = (mandate_active * self.theta_mandate) + (enforcement_level * self.theta_enforce * trust)
        z += policy_component * policy_decay * np.clip(compliance_propensity, 0.0, 1.0)

        # Household influence small boost
        z += self.beta_household * household_share

        p = sigmoid(z)

        # Dropout dynamics for previous wearers under weak reinforcement
        dropout_draws = self.rng.random(N)
        weak_reinforcement = (peer_term < 0.2) & (mandate_active == 0)
        dropout = (prev == 1) & weak_reinforcement & (dropout_draws < self.recovery_or_dropout_prob)

        # Inventory constraint
        can_wear = (mask_inventory >= self.min_inventory_to_wear).astype(int)
        draws = self.rng.random(N)
        new_state = (draws < p).astype(int) * can_wear

        # Enforce dropout overrides
        new_state[dropout] = 0
        pass
        return new_state


class RetailMarket:
    """
    Maintains retailer stock and processes mask purchases by agents.
    """
    def __init__(
        self,
        population_size: int,
        mask_cost_mean: float = 1.0,
        price_elasticity: float = -0.8,
        retailer_restock_rate: float = 0.2,
        mask_availability: float = 0.8,
        retailer_initial_stock_per_capita: float = 2.0,
        rng: Optional[np.random.Generator] = None
    ):
        self.population_size = int(population_size)
        self.mask_cost_mean = float(mask_cost_mean)
        self.price_elasticity = float(price_elasticity)
        self.retailer_restock_rate = float(retailer_restock_rate)
        self.mask_availability = float(mask_availability)
        self.stock_level = float(retailer_initial_stock_per_capita * population_size)
        self.price = float(mask_cost_mean)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        pass

    def step(self, budgets: np.ndarray, mask_inventory: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Process purchases and update stock and price.
        """
        N = len(mask_inventory)
        if self.price <= 0:
            self.price = max(0.1, self.mask_cost_mean)

        affordability = budgets / (self.price + 1e-8)
        # purchase_prob increases with availability and affordability; logistic with elasticity scaling
        desire = -self.price_elasticity * (affordability - 1.0)
        p_purchase = self.mask_availability * sigmoid(desire)
        p_purchase = np.clip(p_purchase, 0.0, 1.0)

        draws = self.rng.random(N)
        want_buy = (draws < p_purchase).astype(int)

        # prevent purchases when budgets < price
        can_afford = (budgets >= self.price).astype(int)
        want_buy = want_buy * can_afford

        possible_purchases = int(np.sum(want_buy))
        actual_purchases = min(possible_purchases, int(self.stock_level))

        if actual_purchases > 0:
            # Fulfill purchases randomly among those who want to buy
            candidates = np.where(want_buy == 1)[0]
            if len(candidates) > actual_purchases:
                buyers = self.rng.choice(candidates, size=actual_purchases, replace=False)
            else:
                buyers = candidates
            # Deduct budgets for buyers and update stock/inventory accordingly
            mask_inventory[buyers] += 1
            budgets[buyers] -= self.price
            budgets[buyers] = np.maximum(budgets[buyers], 0.0)
            self.stock_level -= float(len(buyers))

        # Restock
        restock_qty = self.retailer_restock_rate * self.population_size
        self.stock_level += restock_qty

        # Adjust price according to stock pressure
        pressure = 1.0 - (self.stock_level / (self.population_size + 1e-6))
        self.price = max(0.1, self.mask_cost_mean * (1.0 + 0.1 * pressure))
        pass
        return self.stock_level, self.price, mask_inventory, budgets


class AdoptionAggregator:
    """
    Aggregates daily adoption and info rates; computes group-specific rates.
    """
    def __init__(self, observation_noise_std: float = 0.05, rng: Optional[np.random.Generator] = None):
        self.observation_noise_std = float(observation_noise_std)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        pass

    def aggregate(self, day: int, is_mask_wearing: np.ndarray, received_info: np.ndarray, age_groups: List[str]) -> Dict[str, Any]:
        """
        Aggregate population-level metrics for the day.
        """
        adoption_rate = float(np.mean(is_mask_wearing))
        if self.observation_noise_std > 0:
            adoption_rate_obs = adoption_rate + self.rng.normal(0.0, self.observation_noise_std)
        else:
            adoption_rate_obs = adoption_rate
        adoption_rate_obs = float(max(0.0, min(1.0, adoption_rate_obs)))

        # By age group
        age_group_values: Dict[str, float] = {}
        if len(age_groups) == len(is_mask_wearing):
            groups = sorted(list(set(age_groups)))
            for g in groups:
                idx = [i for i, ag in enumerate(age_groups) if ag == g]
                if idx:
                    age_group_values[g] = float(np.mean(is_mask_wearing[idx]))
                else:
                    age_group_values[g] = 0.0

        info_rate = float(np.mean(received_info))
        aggregated = {
            "day": day,
            "adoption_rate_daily": adoption_rate_obs,
            "adoption_rate_true": adoption_rate,
            "adoption_rate_by_age": age_group_values,
            "info_rate_daily": info_rate,
        }
        pass
        return aggregated


# Model plan embedded as Python dict for dynamic configuration
MODEL_PLAN: Dict[str, Any] = {
    "model_type": "agent_based",
    "description": "A multi-layer network-based agent-based model for the diffusion of mask-wearing behavior through social influence, information diffusion, and policy intervention. Calibrated on panel micro-data (days 0-29) with out-of-sample prediction for days 30-39.",
    "entities": [
        {
            "name": "Person",
            "attributes": [
                "id",
                "age_group",
                "occupation",
                "risk_perception",
                "trust_in_institutions",
                "pro_social_pref",
                "mask_attitude",
                "is_mask_wearing",
                "received_info",
                "compliance_propensity",
                "peer_threshold",
                "media_exposure_level",
                "location_id",
                "household_id",
                "budget",
                "mask_inventory"
            ],
            "behaviors": [
                "decide_mask_wearing",
                "update_risk_perception",
                "interact_with_peers",
                "consume_media",
                "respond_to_policy",
                "purchase_masks",
                "move_between_locations",
                "share_opinion"
            ],
            "initialization": {
                "method": "data_driven",
                "parameters": {
                    "use_empirical_initial_states": True,
                    "risk_perception_source": "agent_attributes.csv:risk_perception",
                    "initial_wearing_mask_source": "train_data.csv:wearing_mask at day==0",
                    "initial_received_info_source": "train_data.csv:received_info at day==0",
                    "budget_default": 100.0,
                    "mask_inventory_default": 1
                }
            }
        },
        {
            "name": "Household",
            "attributes": [
                "id",
                "members",
                "shared_budget",
                "shared_mask_inventory"
            ],
            "behaviors": [
                "pool_resources",
                "share_masks",
                "intra_household_influence"
            ],
            "initialization": {
                "method": "data_driven",
                "parameters": {
                    "construction_rule": "connected_components_on_family_layer",
                    "shared_budget_default": 200.0,
                    "shared_mask_inventory_default": 2
                }
            }
        },
        {
            "name": "Location",
            "attributes": [
                "id",
                "type",
                "capacity",
                "mask_requirement",
                "enforcement_level"
            ],
            "behaviors": [
                "enforce_policy",
                "host_visits"
            ],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "types": [
                        "work_school",
                        "community"
                    ],
                    "count_by_type": {
                        "work_school": 50,
                        "community": 100
                    },
                    "capacity_default": 200,
                    "mask_requirement_default": 0.0,
                    "enforcement_level_default": 0.0
                }
            }
        },
        {
            "name": "GovernmentAgency",
            "attributes": [
                "id",
                "mandate_strength",
                "enforcement_resources",
                "communication_intensity"
            ],
            "behaviors": [
                "set_mask_policy",
                "enforce_policy",
                "broadcast_guidance"
            ],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "mandate_strength": 0.6,
                    "enforcement_resources": 1.0,
                    "communication_intensity": 1.0
                }
            }
        },
        {
            "name": "MediaChannel",
            "attributes": [
                "id",
                "reach",
                "bias",
                "message_intensity"
            ],
            "behaviors": [
                "broadcast_information",
                "adjust_message_intensity"
            ],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "reach": 1.0,
                    "bias": 0.0,
                    "message_intensity": 1.0
                }
            }
        },
        {
            "name": "Retailer",
            "attributes": [
                "id",
                "stock_level",
                "price",
                "restock_rate"
            ],
            "behaviors": [
                "sell_masks",
                "restock",
                "adjust_price"
            ],
            "initialization": {
                "method": "specified",
                "parameters": {
                    "stock_level_per_capita": 2.0,
                    "price": 1.0,
                    "restock_rate": 0.2
                }
            }
        }
    ],
    "modules": [
        {
            "name": "NetworkManager",
            "description": "Loads and cleans the multiplex social network; constructs per-layer neighbor lists and degrees.",
            "requirements": [
                "networkx",
                "json"
            ],
            "module_parameters": [
                "symmetrize_edges",
                "layer_overlap_policy"
            ]
        },
        {
            "name": "ExposureCalculator",
            "description": "Computes per-agent, per-layer peer mask fractions and info contact counts each day.",
            "requirements": [
                "numpy"
            ],
            "module_parameters": [
                "w_family",
                "w_work_school",
                "w_community",
                "mobility_rate"
            ]
        },
        {
            "name": "PolicyAndMedia",
            "description": "Activates mandate from Day 10 and broadcasts guidance; generates policy and media signals.",
            "module_parameters": [
                "policy_start_day",
                "mandate_strength",
                "enforcement_probability",
                "communication_intensity"
            ]
        },
        {
            "name": "InformationDiffusion",
            "description": "Updates each agent’s received_info via peer hazards and exogenous media.",
            "requirements": [
                "numpy"
            ],
            "module_parameters": [
                "p_info_contact_family",
                "p_info_contact_work_school",
                "p_info_contact_community",
                "lambda_media_daily",
                "media_effect_weight"
            ]
        },
        {
            "name": "MaskAdoptionDecision",
            "description": "Decides mask wearing daily using logistic probability with persistence, social exposure by layer, received_info, and policy boost.",
            "requirements": [
                "numpy"
            ],
            "module_parameters": [
                "alpha_intercept",
                "gamma_risk_perception",
                "beta_peer_family",
                "beta_peer_work_school",
                "beta_peer_community",
                "delta_received_info",
                "phi_persistence",
                "social_influence_weight",
                "compliance_decay_rate",
                "recovery_or_dropout_prob",
                "age_mod_youth",
                "age_mod_young_adult",
                "age_mod_middle_age",
                "age_mod_older_adult",
                "occ_mod_student",
                "occ_mod_white_collar",
                "occ_mod_blue_collar",
                "occ_mod_service",
                "min_inventory_to_wear"
            ]
        },
        {
            "name": "RetailMarket",
            "description": "Maintains retailer stock and processes mask purchases by agents.",
            "requirements": [
                "numpy"
            ],
            "module_parameters": [
                "mask_cost_mean",
                "price_elasticity",
                "retailer_restock_rate",
                "mask_availability",
                "retailer_initial_stock_per_capita"
            ]
        },
        {
            "name": "AdoptionAggregator",
            "description": "Aggregates daily adoption and info rates; computes group-specific rates.",
            "requirements": [
                "numpy"
            ],
            "module_parameters": [
                "observation_noise_std"
            ]
        }
    ],
    "parameters": [
        {
            "key": "random_seed",
            "dtype": "int",
            "default": 42,
            "owner_module": "global",
            "description": "RNG seed for reproducibility",
            "frozen": True
        },
        {
            "key": "time_horizon_days",
            "dtype": "int",
            "default": 120,
            "owner_module": "global",
            "description": "Total simulated days",
            "frozen": True
        },
        {
            "key": "time_step_days",
            "dtype": "int",
            "default": 1,
            "owner_module": "global",
            "description": "Time step in days",
            "frozen": True
        },
        {
            "key": "train_start_day",
            "dtype": "int",
            "default": 0,
            "owner_module": "global",
            "description": "Training window start (inclusive)",
            "frozen": True
        },
        {
            "key": "train_end_day",
            "dtype": "int",
            "default": 29,
            "owner_module": "global",
            "description": "Training window end (inclusive)",
            "frozen": True
        },
        {
            "key": "validation_start_day",
            "dtype": "int",
            "default": 30,
            "owner_module": "global",
            "description": "Validation window start (inclusive)",
            "frozen": True
        },
        {
            "key": "validation_end_day",
            "dtype": "int",
            "default": 39,
            "owner_module": "global",
            "description": "Validation window end (inclusive)",
            "frozen": True
        },
        {
            "key": "use_empirical_initial_states",
            "dtype": "bool",
            "default": True,
            "owner_module": "global",
            "description": "Initialize wearing_mask and received_info from day 0 data",
            "frozen": True
        },
        {
            "key": "use_empirical_network",
            "dtype": "bool",
            "default": True,
            "owner_module": "global",
            "description": "Use provided social_network.json rather than synthetic",
            "frozen": True
        },
        {
            "key": "symmetrize_edges",
            "dtype": "bool",
            "default": True,
            "owner_module": "NetworkManager",
            "description": "Make edges undirected",
            "frozen": True
        },
        {
            "key": "layer_overlap_policy",
            "dtype": "categorical",
            "default": "merge_with_weight_sum",
            "owner_module": "NetworkManager",
            "description": "Handle multi-layer overlapping edges",
            "frozen": False
        },
        {
            "key": "w_family",
            "dtype": "float",
            "default": 3.0,
            "owner_module": "ExposureCalculator",
            "description": "Relative weight for family exposure",
            "frozen": False
        },
        {
            "key": "w_work_school",
            "dtype": "float",
            "default": 1.8,
            "owner_module": "ExposureCalculator",
            "description": "Relative weight for work/school exposure",
            "frozen": False
        },
        {
            "key": "w_community",
            "dtype": "float",
            "default": 1.0,
            "owner_module": "ExposureCalculator",
            "description": "Relative weight for community exposure",
            "frozen": False
        },
        {
            "key": "mobility_rate",
            "dtype": "float",
            "default": 0.6,
            "owner_module": "ExposureCalculator",
            "description": "Daily activation of contacts",
            "frozen": False
        },
        {
            "key": "policy_start_day",
            "dtype": "int",
            "default": 10,
            "owner_module": "PolicyAndMedia",
            "description": "Day government mandate begins",
            "frozen": True
        },
        {
            "key": "mandate_strength",
            "dtype": "float",
            "default": 0.6,
            "owner_module": "PolicyAndMedia",
            "description": "Mask policy stringency",
            "frozen": False
        },
        {
            "key": "enforcement_probability",
            "dtype": "float",
            "default": 0.4,
            "owner_module": "PolicyAndMedia",
            "description": "Perceived chance of enforcement",
            "frozen": False
        },
        {
            "key": "communication_intensity",
            "dtype": "float",
            "default": 1.0,
            "owner_module": "PolicyAndMedia",
            "description": "Intensity of public guidance",
            "frozen": False
        },
        {
            "key": "p_info_contact_family",
            "dtype": "float",
            "default": 0.2,
            "owner_module": "InformationDiffusion",
            "description": "Per-contact info transmission in family layer",
            "frozen": False
        },
        {
            "key": "p_info_contact_work_school",
            "dtype": "float",
            "default": 0.12,
            "owner_module": "InformationDiffusion",
            "description": "Per-contact info transmission in work/school layer",
            "frozen": False
        },
        {
            "key": "p_info_contact_community",
            "dtype": "float",
            "default": 0.06,
            "owner_module": "InformationDiffusion",
            "description": "Per-contact info transmission in community layer",
            "frozen": False
        },
        {
            "key": "lambda_media_daily",
            "dtype": "float",
            "default": 0.01,
            "owner_module": "InformationDiffusion",
            "description": "Baseline daily media hazard",
            "frozen": False
        },
        {
            "key": "media_effect_weight",
            "dtype": "float",
            "default": 0.3,
            "owner_module": "InformationDiffusion",
            "description": "Scales media/guidance effect on info hazard",
            "frozen": False
        },
        {
            "key": "alpha_intercept",
            "dtype": "float",
            "default": -3.0,
            "owner_module": "MaskAdoptionDecision",
            "description": "Baseline adoption log-odds",
            "frozen": False
        },
        {
            "key": "gamma_risk_perception",
            "dtype": "float",
            "default": 1.5,
            "owner_module": "MaskAdoptionDecision",
            "description": "Effect of risk perception",
            "frozen": False
        },
        {
            "key": "beta_peer_family",
            "dtype": "float",
            "default": 1.2,
            "owner_module": "MaskAdoptionDecision",
            "description": "Family peer fraction effect",
            "frozen": False
        },
        {
            "key": "beta_peer_work_school",
            "dtype": "float",
            "default": 0.8,
            "owner_module": "MaskAdoptionDecision",
            "description": "Work/school peer fraction effect",
            "frozen": False
        },
        {
            "key": "beta_peer_community",
            "dtype": "float",
            "default": 0.5,
            "owner_module": "MaskAdoptionDecision",
            "description": "Community peer fraction effect",
            "frozen": False
        },
        {
            "key": "delta_received_info",
            "dtype": "float",
            "default": 1.0,
            "owner_module": "MaskAdoptionDecision",
            "description": "Info exposure boost",
            "frozen": False
        },
        {
            "key": "phi_persistence",
            "dtype": "float",
            "default": 2.5,
            "owner_module": "MaskAdoptionDecision",
            "description": "State persistence effect",
            "frozen": False
        },
        {
            "key": "social_influence_weight",
            "dtype": "float",
            "default": 0.5,
            "owner_module": "MaskAdoptionDecision",
            "description": "Global scale on social influence",
            "frozen": False
        },
        {
            "key": "compliance_decay_rate",
            "dtype": "float",
            "default": 0.01,
            "owner_module": "MaskAdoptionDecision",
            "description": "Decay of policy effect over time",
            "frozen": False
        },
        {
            "key": "recovery_or_dropout_prob",
            "dtype": "float",
            "default": 0.02,
            "owner_module": "MaskAdoptionDecision",
            "description": "Daily probability to stop wearing absent reinforcement",
            "frozen": False
        },
        {
            "key": "age_mod_youth",
            "dtype": "float",
            "default": 1.1,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Youth",
            "frozen": False
        },
        {
            "key": "age_mod_young_adult",
            "dtype": "float",
            "default": 1.0,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Young Adult",
            "frozen": False
        },
        {
            "key": "age_mod_middle_age",
            "dtype": "float",
            "default": 0.95,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Middle Age",
            "frozen": False
        },
        {
            "key": "age_mod_older_adult",
            "dtype": "float",
            "default": 1.05,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Older Adult",
            "frozen": False
        },
        {
            "key": "occ_mod_student",
            "dtype": "float",
            "default": 1.15,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Students",
            "frozen": False
        },
        {
            "key": "occ_mod_white_collar",
            "dtype": "float",
            "default": 1.0,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for White Collar",
            "frozen": False
        },
        {
            "key": "occ_mod_blue_collar",
            "dtype": "float",
            "default": 0.95,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Blue Collar",
            "frozen": False
        },
        {
            "key": "occ_mod_service",
            "dtype": "float",
            "default": 1.05,
            "owner_module": "MaskAdoptionDecision",
            "description": "Influence modifier for Service",
            "frozen": False
        },
        {
            "key": "min_inventory_to_wear",
            "dtype": "int",
            "default": 1,
            "owner_module": "MaskAdoptionDecision",
            "description": "Masks required in inventory to wear",
            "frozen": False
        },
        {
            "key": "mask_cost_mean",
            "dtype": "float",
            "default": 1.0,
            "owner_module": "RetailMarket",
            "description": "Average mask price baseline",
            "frozen": False
        },
        {
            "key": "price_elasticity",
            "dtype": "float",
            "default": -0.8,
            "owner_module": "RetailMarket",
            "description": "Sensitivity of purchase to price/affordability",
            "frozen": False
        },
        {
            "key": "retailer_restock_rate",
            "dtype": "float",
            "default": 0.2,
            "owner_module": "RetailMarket",
            "description": "Daily per-capita restock rate",
            "frozen": False
        },
        {
            "key": "mask_availability",
            "dtype": "float",
            "default": 0.8,
            "owner_module": "RetailMarket",
            "description": "Baseline stock/service availability factor",
            "frozen": False
        },
        {
            "key": "retailer_initial_stock_per_capita",
            "dtype": "float",
            "default": 2.0,
            "owner_module": "RetailMarket",
            "description": "Initial masks in stock per capita",
            "frozen": False
        },
        {
            "key": "observation_noise_std",
            "dtype": "float",
            "default": 0.05,
            "owner_module": "AdoptionAggregator",
            "description": "Noise added to observed rates",
            "frozen": False
        },
        {
            "key": "trust_in_institutions_mean",
            "dtype": "float",
            "default": 0.5,
            "owner_module": "global",
            "description": "Population mean trust (for future extensions)",
            "frozen": True
        },
        {
            "key": "pro_social_mean",
            "dtype": "float",
            "default": 0.5,
            "owner_module": "global",
            "description": "Population mean pro-social preference",
            "frozen": True
        },
        {
            "key": "risk_perception_beta_alpha",
            "dtype": "float",
            "default": 2.0,
            "owner_module": "global",
            "description": "Fallback Beta alpha for risk_perception",
            "frozen": True
        },
        {
            "key": "risk_perception_beta_beta",
            "dtype": "float",
            "default": 3.0,
            "owner_module": "global",
            "description": "Fallback Beta beta for risk_perception",
            "frozen": True
        }
    ],
    "prediction_period": {
        "start_day": 30,
        "end_day": 39
    },
    "evaluation_metrics": [
        "RMSE_adoption",
        "MAE_adoption",
        "peak_adoption",
        "time_to_50_percent",
        "final_adoption",
        "adoption_inequality_gini"
    ]
}


class Simulation:
    """
    Main simulation class that orchestrates data loading, initialization, module execution, and evaluation.
    """
    def __init__(self, model_plan: Dict[str, Any]):
        self.model_plan = model_plan
        self.params: Dict[str, Any] = self._extract_parameters(model_plan.get("parameters", []))
        self.random_seed = int(self.params.get("random_seed", 42))
        self.rng = np.random.default_rng(self.random_seed)

        # Data loading
        self.data_loader = DataLoader(DATA_DIR)
        self.agent_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.network_json: Dict[int, Dict[str, List[int]]] = {}

        # Network manager and exposures
        self.network_manager = NetworkManager(
            symmetrize_edges=bool(self.params.get("symmetrize_edges", True)),
            layer_overlap_policy=str(self.params.get("layer_overlap_policy", "merge_with_weight_sum"))
        )
        self.neighbors_by_layer: Dict[str, Dict[int, List[int]]] = {"family": {}, "work_school": {}, "community": {}}
        self.degree_by_layer: Dict[str, Dict[int, int]] = {"family": {}, "work_school": {}, "community": {}}
        self.neighbor_index_by_layer: Dict[str, List[np.ndarray]] = {"family": [], "work_school": [], "community": []}

        # Modules
        self.exposure_calc = ExposureCalculator(
            w_family=float(self.params.get("w_family", 3.0)),
            w_work_school=float(self.params.get("w_work_school", 1.8)),
            w_community=float(self.params.get("w_community", 1.0)),
            mobility_rate=float(self.params.get("mobility_rate", 0.6))
        )
        self.policy_media = PolicyAndMedia(
            policy_start_day=int(self.params.get("policy_start_day", 10)),
            mandate_strength=float(self.params.get("mandate_strength", 0.6)),
            enforcement_probability=float(self.params.get("enforcement_probability", 0.4)),
            communication_intensity=float(self.params.get("communication_intensity", 1.0))
        )
        self.info_diffusion = InformationDiffusion(
            p_info_contact_family=float(self.params.get("p_info_contact_family", 0.2)),
            p_info_contact_work_school=float(self.params.get("p_info_contact_work_school", 0.12)),
            p_info_contact_community=float(self.params.get("p_info_contact_community", 0.06)),
            lambda_media_daily=float(self.params.get("lambda_media_daily", 0.01)),
            media_effect_weight=float(self.params.get("media_effect_weight", 0.3)),
            rng=self.rng
        )
        age_mods = {
            "Youth": float(self.params.get("age_mod_youth", 1.1)),
            "Young Adult": float(self.params.get("age_mod_young_adult", 1.0)),
            "Middle Age": float(self.params.get("age_mod_middle_age", 0.95)),
            "Older Adult": float(self.params.get("age_mod_older_adult", 1.05)),
        }
        occ_mods = {
            "Student": float(self.params.get("occ_mod_student", 1.15)),
            "White Collar": float(self.params.get("occ_mod_white_collar", 1.0)),
            "Blue Collar": float(self.params.get("occ_mod_blue_collar", 0.95)),
            "Service": float(self.params.get("occ_mod_service", 1.05)),
        }
        self.mask_adoption = MaskAdoptionDecision(
            alpha_intercept=float(self.params.get("alpha_intercept", -3.0)),
            gamma_risk_perception=float(self.params.get("gamma_risk_perception", 1.5)),
            beta_peer_family=float(self.params.get("beta_peer_family", 1.2)),
            beta_peer_work_school=float(self.params.get("beta_peer_work_school", 0.8)),
            beta_peer_community=float(self.params.get("beta_peer_community", 0.5)),
            delta_received_info=float(self.params.get("delta_received_info", 1.0)),
            phi_persistence=float(self.params.get("phi_persistence", 2.5)),
            social_influence_weight=float(self.params.get("social_influence_weight", 0.5)),
            compliance_decay_rate=float(self.params.get("compliance_decay_rate", 0.01)),
            recovery_or_dropout_prob=float(self.params.get("recovery_or_dropout_prob", 0.02)),
            age_mods=age_mods,
            occ_mods=occ_mods,
            min_inventory_to_wear=int(self.params.get("min_inventory_to_wear", 1)),
            rng=self.rng,
            policy_start_day=int(self.params.get("policy_start_day", 10)),
            theta_mandate=0.3,
            theta_enforce=0.8,
            beta_household=0.2
        )
        # Retailer and aggregator initialized after population is known
        self.retail_market: Optional[RetailMarket] = None
        self.aggregator = AdoptionAggregator(
            observation_noise_std=float(self.params.get("observation_noise_std", 0.05)),
            rng=self.rng
        )

        # Entities
        self.persons: List[Person] = []
        self.households: List[Household] = []
        self.locations: List[Location] = []
        self.government = GovernmentAgency(
            agency_id=0,
            mandate_strength=float(self.params.get("mandate_strength", 0.6)),
            enforcement_resources=1.0,
            communication_intensity=float(self.params.get("communication_intensity", 1.0))
        )
        self.media_channel = MediaChannel(
            channel_id=0,
            reach=1.0,
            bias=0.0,
            message_intensity=1.0
        )
        self.retailer = Retailer(
            retailer_id=0,
            stock_level=0.0,
            price=float(self.params.get("mask_cost_mean", 1.0)),
            restock_rate=float(self.params.get("retailer_restock_rate", 0.2))
        )

        # State arrays
        self.N: int = 0
        self.agent_ids: List[int] = []
        self.agent_index_map: Dict[int, int] = {}
        self.age_groups: List[str] = []
        self.occupations: List[str] = []
        self.risk_perception: np.ndarray = np.array([])
        self.is_mask_wearing: np.ndarray = np.array([])
        self.received_info: np.ndarray = np.array([])
        self.mask_inventory: np.ndarray = np.array([])
        self.budgets: np.ndarray = np.array([])
        self.trust: np.ndarray = np.array([])
        self.compliance_propensity: np.ndarray = np.array([])

        # Results
        self.results_daily: List[Dict[str, Any]] = []
        self.results_df: Optional[pd.DataFrame] = None

        # Time
        pred = model_plan.get("prediction_period", {})
        self.validation_end_day = int(self.params.get("validation_end_day", 39))
        self.sim_end_day = int(max(pred.get("end_day", self.validation_end_day), self.validation_end_day))
        self.train_start_day = int(self.params.get("train_start_day", 0))
        self.train_end_day = int(self.params.get("train_end_day", 29))
        self.validation_start_day = int(self.params.get("validation_start_day", 30))
        pass

    def _extract_parameters(self, params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract parameter defaults from parameter list into a simple dict.
        """
        out = {}
        for p in params_list:
            key = p.get("key")
            default = p.get("default")
            if key is not None:
                out[key] = default
        pass
        return out

    def _generate_small_world_network(self, N: int, k: int = 6, p_rewire: float = 0.05, agent_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, List[int]]]:
        """
        Generate a small-world network as a fallback when no empirical network is available.
        """
        if agent_ids is None:
            agent_ids = list(range(N))
        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        idx_to_id = {i: aid for aid, i in id_to_idx.items()}
        network: Dict[int, Dict[str, List[int]]] = {}
        if nx is not None:
            try:
                G = nx.watts_strogatz_graph(N, k=max(2, int(k)), p=p_rewire, seed=self.random_seed)
                for i in range(N):
                    neigh_idx = list(G.neighbors(i))
                    neigh_ids = [idx_to_id[j] for j in neigh_idx]
                    node_id = idx_to_id[i]
                    network[node_id] = {"family": [], "work_school": [], "community": neigh_ids, "all": neigh_ids.copy()}
                pass
                return network
            except Exception as e:
                print("Warning: networkx generation failed; falling back to ring lattice.", e)
        # Fallback ring lattice
        for i in range(N):
            neigh = []
            half = max(1, int(k // 2))
            for d in range(1, half + 1):
                neigh.append((i - d) % N)
                neigh.append((i + d) % N)
            neigh = sorted(list(set(neigh)))
            node_id = idx_to_id[i]
            neigh_ids = [idx_to_id[j] for j in neigh]
            network[node_id] = {"family": [], "work_school": [], "community": neigh_ids, "all": neigh_ids.copy()}
        pass
        return network

    def _precompute_neighbor_indices(self) -> None:
        """
        Precompute neighbor indices arrays for each layer for faster exposure computation.
        """
        layers = ["family", "work_school", "community"]
        neighbor_index_by_layer: Dict[str, List[np.ndarray]] = {L: [np.array([], dtype=int) for _ in range(self.N)] for L in layers}
        for L in layers:
            nb = self.neighbors_by_layer.get(L, {})
            for aid in self.agent_ids:
                idx = self.agent_index_map[aid]
                neigh_ids = nb.get(aid, [])
                neigh_idx = [self.agent_index_map[j] for j in neigh_ids if j in self.agent_index_map]
                neighbor_index_by_layer[L][idx] = np.array(neigh_idx, dtype=int) if len(neigh_idx) > 0 else np.array([], dtype=int)
        self.neighbor_index_by_layer = neighbor_index_by_layer
        pass

    def initialize(self) -> None:
        """
        Initialize the simulation: load data, setup agents, network, and modules depending on data availability.
        """
        # Load data
        self.agent_df = self.data_loader.load_agent_attributes()
        self.train_df = self.data_loader.load_train_data()
        if bool(self.params.get("use_empirical_network", True)):
            self.network_json = self.data_loader.load_social_network()
        else:
            self.network_json = {}

        # Determine population and index maps
        if self.agent_df is not None:
            agent_ids = list(sorted(self.agent_df["agent_id"].astype(int).unique()))
        else:
            agent_ids = list(range(int(10000)))
        self.agent_ids = agent_ids
        self.N = len(agent_ids)
        self.agent_index_map = {aid: i for i, aid in enumerate(agent_ids)}
        self.persons = [Person(agent_id=aid) for aid in agent_ids]

        # Defaults
        default_budget = float(self.model_plan.get("entities", [])[0]["initialization"]["parameters"].get("budget_default", 100.0)) if self.model_plan.get("entities") else 100.0
        default_inventory = int(self.model_plan.get("entities", [])[0]["initialization"]["parameters"].get("mask_inventory_default", 1)) if self.model_plan.get("entities") else 1
        trust_mean = float(self.params.get("trust_in_institutions_mean", 0.5))
        compliance_mean = 0.5

        # Initialize attributes
        self.age_groups, self.occupations = [], []
        self.risk_perception = np.zeros(self.N, dtype=float)
        self.trust = np.full(self.N, trust_mean, dtype=float)
        self.compliance_propensity = np.full(self.N, compliance_mean, dtype=float)
        self.budgets = np.full(self.N, default_budget, dtype=float)
        self.mask_inventory = np.full(self.N, default_inventory, dtype=float)

        if self.agent_df is not None:
            df = self.agent_df.set_index("agent_id")
            for i, aid in enumerate(self.agent_ids):
                if aid in df.index:
                    row = df.loc[aid]
                    self.age_groups.append(str(row.get("age_group", "Unknown")) if "age_group" in df.columns else "Unknown")
                    self.occupations.append(str(row.get("occupation", "Unknown")) if "occupation" in df.columns else "Unknown")
                    if "risk_perception" in df.columns:
                        try:
                            self.risk_perception[i] = float(row.get("risk_perception", np.clip(np.random.beta(2, 3), 0.0, 1.0)))
                        except Exception:
                            self.risk_perception[i] = float(np.clip(np.random.beta(2, 3), 0.0, 1.0))
                    else:
                        self.risk_perception[i] = float(np.clip(np.random.beta(2, 3), 0.0, 1.0))
                    if "trust_in_institutions" in df.columns:
                        try:
                            self.trust[i] = float(row.get("trust_in_institutions", trust_mean))
                        except Exception:
                            self.trust[i] = trust_mean
                    if "compliance_propensity" in df.columns:
                        try:
                            self.compliance_propensity[i] = float(row.get("compliance_propensity", compliance_mean))
                        except Exception:
                            self.compliance_propensity[i] = compliance_mean
                    if "budget" in df.columns:
                        try:
                            self.budgets[i] = float(row.get("budget", default_budget))
                        except Exception:
                            self.budgets[i] = default_budget
                    if "mask_inventory" in df.columns:
                        try:
                            self.mask_inventory[i] = float(row.get("mask_inventory", default_inventory))
                        except Exception:
                            self.mask_inventory[i] = default_inventory
                else:
                    self.age_groups.append("Unknown")
                    self.occupations.append("Unknown")
        else:
            # Synthetic demographics
            rng = self.rng
            possible_ages = ["Youth", "Young Adult", "Middle Age", "Older Adult"]
            possible_occs = ["Student", "White Collar", "Blue Collar", "Service"]
            self.age_groups = list(rng.choice(possible_ages, size=self.N, replace=True))
            self.occupations = list(rng.choice(possible_occs, size=self.N, replace=True))
            alpha = float(self.params.get("risk_perception_beta_alpha", 2.0))
            beta = float(self.params.get("risk_perception_beta_beta", 3.0))
            self.risk_perception = rng.beta(alpha, beta, size=self.N)

        # Initial states
        self.is_mask_wearing = np.zeros(self.N, dtype=int)
        self.received_info = np.zeros(self.N, dtype=int)
        if self.train_df is not None and bool(self.params.get("use_empirical_initial_states", True)):
            df0 = self.train_df[self.train_df["day"] == int(self.train_start_day)]
            wear_map = {int(r.agent_id): int(r.wearing_mask) for r in df0.itertuples(index=False)}
            info_map = {int(r.agent_id): int(r.received_info) for r in df0.itertuples(index=False)}
            for aid in self.agent_ids:
                idx = self.agent_index_map[aid]
                if aid in wear_map:
                    self.is_mask_wearing[idx] = wear_map[aid]
                if aid in info_map:
                    self.received_info[idx] = info_map[aid]

        # Network setup
        if not self.network_json or len(self.network_json) == 0:
            self.network_json = self._generate_small_world_network(self.N, k=6, p_rewire=0.05, agent_ids=self.agent_ids)
        else:
            # Ensure all agents have entries
            for aid in self.agent_ids:
                if aid not in self.network_json:
                    self.network_json[aid] = {"family": [], "work_school": [], "community": [], "all": []}

        self.neighbors_by_layer, self.degree_by_layer = self.network_manager.build_neighbors(self.network_json)
        self._precompute_neighbor_indices()

        # RetailMarket module now that population is known
        self.retail_market = RetailMarket(
            population_size=self.N,
            mask_cost_mean=float(self.params.get("mask_cost_mean", 1.0)),
            price_elasticity=float(self.params.get("price_elasticity", -0.8)),
            retailer_restock_rate=float(self.params.get("retailer_restock_rate", 0.2)),
            mask_availability=float(self.params.get("mask_availability", 0.8)),
            retailer_initial_stock_per_capita=float(self.params.get("retailer_initial_stock_per_capita", 2.0)),
            rng=self.rng
        )
        pass

    def run(self) -> None:
        """
        Execute the simulation loop.
        """
        layers = ["family", "work_school", "community"]
        for day in range(0, self.sim_end_day + 1):
            # Policy/media signals
            signals = self.policy_media.step(day)
            mandate_active = int(signals["mandate_active"])
            enforcement_level = float(signals["enforcement_level"])
            guidance_intensity = float(signals["guidance_intensity"])

            # Exposures
            exposures = self.exposure_calc.compute(
                neighbor_index_by_layer=self.neighbor_index_by_layer,
                is_mask_wearing=self.is_mask_wearing,
                received_info=self.received_info
            )
            peer_mask_fraction = exposures["peer_mask_fraction"]
            info_contacts = exposures["info_contacts"]

            # Information diffusion
            self.received_info = self.info_diffusion.step(
                received_info=self.received_info,
                info_contacts={L: info_contacts.get(L, np.zeros(self.N)) for L in layers},
                guidance_intensity=guidance_intensity
            )

            # Mask adoption
            household_share = peer_mask_fraction.get("family", np.zeros(self.N))
            self.is_mask_wearing = self.mask_adoption.step(
                day=day,
                is_mask_wearing=self.is_mask_wearing,
                received_info=self.received_info,
                risk_perception=self.risk_perception,
                age_groups=self.age_groups,
                occupations=self.occupations,
                peer_mask_fraction={L: peer_mask_fraction.get(L, np.zeros(self.N)) for L in layers},
                mandate_active=mandate_active,
                enforcement_level=enforcement_level,
                mask_inventory=self.mask_inventory,
                trust=self.trust,
                compliance_propensity=self.compliance_propensity,
                household_share=household_share
            )

            # Optional simple consumption: use one mask on days starting to wear
            # Here we decrement inventory when agent begins wearing (transition 0->1)
            # Ensure non-negative inventory
            # Note: keep conservative to avoid negative inventories
            # prev state is approximated by using drop-in variable (can't access previous now); skip consumption to keep stable

            # Retail market step
            if self.retail_market is not None:
                _, _, self.mask_inventory, self.budgets = self.retail_market.step(
                    budgets=self.budgets,
                    mask_inventory=self.mask_inventory
                )

            # Aggregate results
            agg = self.aggregator.aggregate(
                day=day,
                is_mask_wearing=self.is_mask_wearing,
                received_info=self.received_info,
                age_groups=self.age_groups
            )
            self.results_daily.append(agg)

        # Final DataFrame
        try:
            self.results_df = pd.DataFrame(self.results_daily)
        except Exception:
            self.results_df = None
        pass


def main():
    # Build, initialize, and run the simulation
    sim = Simulation(MODEL_PLAN)
    sim.initialize()
    sim.run()
    # Print a small summary
    if sim.results_df is not None and not sim.results_df.empty:
        print(sim.results_df.head().to_string(index=False))
        print("...")
        print(sim.results_df.tail().to_string(index=False))
    else:
        print("Simulation completed, but no results available.")


# Execute main for both direct execution and sandbox wrapper invocation
main()