import os
import json
import random
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd

# Optional dependencies handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False


# Path Handling with default fallback to data folder
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH) if PROJECT_ROOT and DATA_PATH else "data_fitting/mask_adoption_data/"


def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility across numpy, random.

    Parameters:
        seed (int): Random seed.

    Notes:
        This function ensures that numpy and Python's random use the same seed for reproducible runs.
    """
    np.random.seed(seed)
    random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid-transformed array.
    """
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error between two arrays.

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: RMSE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error between two arrays.

    Parameters:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: MAE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


class Person:
    """
    Represents an individual agent in the simulation.
    """

    def __init__(self, agent_id: int) -> None:
        self.id: int = agent_id
        self.age: Optional[int] = None
        self.age_group: Optional[str] = None
        self.occupation: Optional[str] = None
        self.household_id: Optional[int] = None
        self.workplace_id: Optional[int] = None
        self.adoption_state: int = 0
        self.compliance_probability: float = 0.0
        self.risk_perception: float = 0.0
        self.mask_attitude: float = 0.0
        self.susceptibility_to_influence: float = 1.0
        self.peer_influence_weight: float = 1.0
        self.policy_sensitivity: float = 1.0
        self.information_exposure: float = 0.0
        self.degree: Dict[str, int] = {"family": 0, "work": 0, "community": 0}

    def decide_adopt_mask(self) -> None:
        pass

    def interact_with_peers(self) -> None:
        pass

    def attend_location(self) -> None:
        pass

    def respond_to_policy(self) -> None:
        pass

    def update_beliefs(self) -> None:
        pass


class Location:
    """
    Represents a location where agents may interact.
    """

    def __init__(self, loc_id: int, loc_type: str) -> None:
        self.id: int = loc_id
        self.type: str = loc_type
        self.capacity: int = 0
        self.mask_requirement_level: float = 0.0
        self.enforcement_level: float = 0.0
        self.contact_rate_multiplier: float = 1.0
        self.opening_hours: Tuple[int, int] = (8, 20)

    def host_contacts(self) -> None:
        pass

    def enforce_policy(self) -> None:
        pass

    def report_compliance(self) -> None:
        pass


class DataLoader:
    """
    Handles loading and validation of simulation data from CSV/JSON files.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.agent_file = os.path.join(data_dir, "agent_attributes.csv")
        self.network_file = os.path.join(data_dir, "social_network.json")
        self.train_file = os.path.join(data_dir, "train_data.csv")

        self.agents_df: Optional[pd.DataFrame] = None
        self.social_network: Optional[Dict[str, Dict[str, List[int]]]] = None
        self.train_df: Optional[pd.DataFrame] = None

    def load_agents(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(self.agent_file)
            # Normalize columns
            expected_cols = {"agent_id", "age_group", "occupation", "risk_perception"}
            missing = expected_cols - set(df.columns)
            if missing:
                for col in missing:
                    if col == "agent_id":
                        continue
                    if col == "risk_perception":
                        df[col] = 0.5
                    else:
                        df[col] = "Unknown"
            # Ensure correct types
            df["agent_id"] = df["agent_id"].astype(int)
            df["risk_perception"] = df["risk_perception"].clip(0, 1).astype(float)

            # Optional heterogeneity defaults if missing
            for col, default_val in [
                ("compliance_probability", 0.5),
                ("policy_sensitivity", 1.0),
                ("susceptibility_to_influence", 1.0),
                ("peer_influence_weight", 1.0),
            ]:
                if col not in df.columns:
                    df[col] = default_val
            self.agents_df = df
            return df
        except Exception as e:
            print(f"Warning: Could not load agents file at {self.agent_file}: {e}")
            self.agents_df = None
            return None

    def load_social_network(self) -> Optional[Dict[str, Dict[str, List[int]]]]:
        try:
            with open(self.network_file, "r") as f:
                net = json.load(f)
            if not isinstance(net, dict):
                raise ValueError("Network JSON is not a dictionary at the top level.")
            self.social_network = net
            return net
        except Exception as e:
            print(f"Warning: Could not load social network at {self.network_file}: {e}")
            self.social_network = None
            return None

    def load_train_data(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(self.train_file)
            expected_cols = {"day", "agent_id", "wearing_mask", "received_info"}
            missing = expected_cols - set(df.columns)
            if missing:
                raise ValueError(f"train_data.csv missing columns: {missing}")

            df["day"] = df["day"].astype(int)
            df["agent_id"] = df["agent_id"].astype(int)
            df["wearing_mask"] = df["wearing_mask"].astype(int).clip(0, 1)
            df["received_info"] = df["received_info"].astype(int).clip(0, 1)

            self.train_df = df
            return df
        except Exception as e:
            print(f"Warning: Could not load training data at {self.train_file}: {e}")
            self.train_df = None
            return None

    def validate_and_align(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Dict[str, List[int]]]]]:
        """
        Align agents and network node sets by intersection; drop mismatched IDs and add isolated nodes if needed.
        """
        if self.agents_df is None and self.social_network is None:
            return None, None

        if self.agents_df is not None and self.social_network is None:
            return self.agents_df, None

        if self.agents_df is None and self.social_network is not None:
            # infer agents from network keys
            try:
                keys = [int(k) for k in self.social_network.keys()]
            except Exception:
                keys = []
            if keys:
                df = pd.DataFrame({"agent_id": keys})
                df["age_group"] = "Unknown"
                df["occupation"] = "Unknown"
                df["risk_perception"] = 0.5
                for col, default_val in [
                    ("compliance_probability", 0.5),
                    ("policy_sensitivity", 1.0),
                    ("susceptibility_to_influence", 1.0),
                    ("peer_influence_weight", 1.0),
                ]:
                    df[col] = default_val
                self.agents_df = df
            return self.agents_df, self.social_network

        # Both available: align
        agents_ids = set(self.agents_df["agent_id"].astype(int).tolist())
        try:
            net_ids = set(int(k) for k in self.social_network.keys())
        except Exception:
            net_ids = set()

        intersect_ids = agents_ids.intersection(net_ids)
        dropped_agents = len(agents_ids - intersect_ids)
        dropped_net_nodes = len(net_ids - intersect_ids)

        if dropped_agents > 0:
            print(f"Alignment: Dropping {dropped_agents} agents not in network.")
        if dropped_net_nodes > 0:
            print(f"Alignment: Dropping {dropped_net_nodes} network nodes not in agents list.")

        # Filter agents
        self.agents_df = self.agents_df[self.agents_df["agent_id"].isin(intersect_ids)].reset_index(drop=True)

        # Filter network
        aligned_net: Dict[str, Dict[str, List[int]]] = {}
        for k, v in self.social_network.items():
            try:
                a = int(k)
            except Exception:
                continue
            if a not in intersect_ids:
                continue
            # keep only neighbors that are in intersect
            layers = {}
            for layer_key in ["family", "work_school", "community"]:
                lst = v.get(layer_key, []) or []
                filtered = []
                for nb in lst:
                    try:
                        nb_int = int(nb)
                    except Exception:
                        continue
                    if nb_int in intersect_ids:
                        filtered.append(nb_int)
                layers[layer_key] = filtered
            aligned_net[str(a)] = layers

        self.social_network = aligned_net
        return self.agents_df, self.social_network


class NetworkPreprocessor:
    """
    Preprocesses the multiplex social network and constructs neighbor indices and degrees per layer.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def build_neighbor_index(self, network_json: Dict[str, Dict[str, List[int]]],
                             agent_ids: List[int]) -> Dict[int, Dict[str, List[int]]]:
        directed_mode = bool(self.params.get("directed_mode", False))
        symmetrize = bool(self.params.get("symmetrize_edges", True))
        remove_self = bool(self.params.get("remove_self_loops", True))

        agent_set = set(agent_ids)
        neighbor_index: Dict[int, Dict[str, List[int]]] = {}
        for aid in agent_ids:
            neighbor_index[aid] = {"family": [], "work": [], "community": []}

        def add_edge(a: int, b: int, layer: str) -> None:
            if a not in neighbor_index:
                return
            if b not in neighbor_index:
                return
            if remove_self and a == b:
                return
            neighbor_index[a][layer].append(b)

        for key, layers in network_json.items():
            try:
                a = int(key)
            except Exception:
                continue
            if a not in agent_set:
                continue
            fam_list = layers.get("family", []) or []
            work_list = layers.get("work_school", []) or []
            com_list = layers.get("community", []) or []
            for b in fam_list:
                try:
                    add_edge(a, int(b), "family")
                except (TypeError, ValueError):
                    print(f"Warning: Invalid neighbor id in family layer for agent {a}: {b}")
                    continue
            for b in work_list:
                try:
                    add_edge(a, int(b), "work")
                except (TypeError, ValueError):
                    print(f"Warning: Invalid neighbor id in work layer for agent {a}: {b}")
                    continue
            for b in com_list:
                try:
                    add_edge(a, int(b), "community")
                except (TypeError, ValueError):
                    print(f"Warning: Invalid neighbor id in community layer for agent {a}: {b}")
                    continue

        if symmetrize and not directed_mode:
            for a in list(neighbor_index.keys()):
                for layer in ["family", "work", "community"]:
                    for b in list(neighbor_index[a][layer]):
                        if b not in neighbor_index:
                            continue
                        if a not in neighbor_index[b][layer]:
                            neighbor_index[b][layer].append(a)

        # Remove duplicates and clean
        for a in neighbor_index:
            for layer in ["family", "work", "community"]:
                lst = neighbor_index[a][layer]
                seen = set()
                cleaned = []
                for x in lst:
                    if x not in seen:
                        seen.add(x)
                        cleaned.append(x)
                neighbor_index[a][layer] = cleaned

        return neighbor_index

    def compute_degrees(self, neighbor_index: Dict[int, Dict[str, List[int]]]) -> Dict[int, Dict[str, int]]:
        degrees: Dict[int, Dict[str, int]] = {}
        for aid, layers in neighbor_index.items():
            degrees[aid] = {
                "family": len(layers.get("family", [])),
                "work": len(layers.get("work", [])),
                "community": len(layers.get("community", [])),
            }
        return degrees


class ExposureCalculator:
    """
    Computes layer-weighted social exposure to mask-wearing for each agent, adjusted by homophily and smoothed over time.
    """

    def __init__(self, params: Dict[str, Any], agents_df: pd.DataFrame) -> None:
        self.params = params
        self.age_group_map = dict(zip(agents_df["agent_id"].astype(int), agents_df["age_group"].astype(str)))
        self.occupation_map = dict(zip(agents_df["agent_id"].astype(int), agents_df["occupation"].astype(str)))

    @staticmethod
    def _is_known_attr(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
        if isinstance(val, str) and val.strip().lower() in {"unknown", "unk", "na", "none", ""}:
            return False
        return True

    def compute_exposure(self,
                         neighbor_index: Dict[int, Dict[str, List[int]]],
                         adoption_prev: np.ndarray,
                         agent_ids: List[int],
                         exposure_prev: Optional[np.ndarray] = None,
                         daily_layer_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        # Layer weights: use daily if provided, else static params
        if daily_layer_weights is not None:
            w_family = float(daily_layer_weights.get("family", 0.0))
            w_work = float(daily_layer_weights.get("work", 0.0))
            w_comm = float(daily_layer_weights.get("community", 0.0))
            total_w = max(1e-9, w_family + w_work + w_comm)
            w_family /= total_w
            w_work /= total_w
            w_comm /= total_w
        else:
            w_family = float(self.params.get("layer_weight_family", 0.5))
            w_work = float(self.params.get("layer_weight_work", 0.3))
            w_comm = float(self.params.get("layer_weight_community", 0.2))
            total_w = max(1e-9, w_family + w_work + w_comm)
            w_family /= total_w
            w_work /= total_w
            w_comm /= total_w

        h_age = float(self.params.get("homophily_weight_same_age", 1.2))
        h_occ = float(self.params.get("homophily_weight_same_occ", 1.15))
        gamma = float(self.params.get("exposure_smoothing_gamma", 0.1))

        id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
        N = len(agent_ids)
        exposure = np.zeros(N, dtype=float)

        for idx, aid in enumerate(agent_ids):
            exp_val = 0.0
            for layer, layer_weight in zip(["family", "work", "community"], [w_family, w_work, w_comm]):
                neighbors = neighbor_index.get(aid, {}).get(layer, [])
                if not neighbors:
                    frac = 0.0
                else:
                    masked_sum = 0.0
                    w_sum = 0.0
                    ai_age = self.age_group_map.get(aid, None)
                    ai_occ = self.occupation_map.get(aid, None)
                    for nb in neighbors:
                        w = 1.0
                        nb_age = self.age_group_map.get(nb, None)
                        nb_occ = self.occupation_map.get(nb, None)
                        if self._is_known_attr(ai_age) and self._is_known_attr(nb_age) and (nb_age == ai_age):
                            w *= h_age
                        if self._is_known_attr(ai_occ) and self._is_known_attr(nb_occ) and (nb_occ == ai_occ):
                            w *= h_occ
                        w_sum += w
                        nb_idx = id_to_idx.get(nb, None)
                        if nb_idx is not None:
                            masked_sum += w * adoption_prev[nb_idx]
                    frac = masked_sum / max(1e-9, w_sum)
                exp_val += layer_weight * frac
            exposure[idx] = exp_val

        if exposure_prev is None:
            exposure_prev = np.zeros_like(exposure)
        exposure_smoothed = (1.0 - gamma) * exposure + gamma * exposure_prev
        exposure_smoothed = np.clip(exposure_smoothed, 0.0, 1.0)
        return exposure_smoothed


class PolicyAuthority:
    """
    Schedules and emits daily policy signals including mandate, enforcement, and communication.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def update(self, t: int) -> Dict[str, float]:
        start = int(self.params.get("policy_start_day", 10))
        end = int(self.params.get("policy_end_day", 60))
        ramp_rate = float(self.params.get("policy_ramp_rate", 0.1))
        mandate_base = float(self.params.get("mandate_level_base", 0.5))
        comm_base = float(self.params.get("communication_intensity_base", 0.3))
        enforce_global = float(self.params.get("enforcement_level_global", 0.3))
        subsidy_base = float(self.params.get("subsidy_incentive_base", 0.0))
        mandate_to_enforcement_multiplier = float(self.params.get("mandate_to_enforcement_multiplier", 0.5))

        if t < start or t > end:
            mandate = 0.0
            comm = 0.0
            enforce = 0.0
            subsidy = 0.0
        else:
            ramp = min(1.0, ramp_rate * max(0, t - start))
            mandate = mandate_base * ramp
            comm = comm_base * ramp
            enforce = min(1.0, enforce_global * (1.0 + mandate_to_enforcement_multiplier * mandate))
            subsidy = subsidy_base * ramp

        signals = {
            "mandate_level": float(mandate),
            "communication_intensity": float(comm),
            "enforcement_level": float(enforce),
            "subsidy_incentive": float(subsidy),
        }
        return signals


# Backward compatibility alias
PolicyScheduler = PolicyAuthority


class InfoDiffusion:
    """
    Generates information receipt per agent via neighbor contagion and exogenous broadcasts influenced by policy communication.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def update(self,
               neighbor_index: Dict[int, Dict[str, List[int]]],
               received_info_prev: np.ndarray,
               communication_intensity: float,
               agent_ids: List[int]) -> np.ndarray:
        exo_rate = float(self.params.get("exogenous_broadcast_rate", 0.01))
        comm_mult = float(self.params.get("policy_comm_to_info_multiplier", 0.8))
        mem_decay = float(self.params.get("info_memory_decay", 0.2))
        p_family = float(self.params.get("info_transmission_family", 0.4))
        p_work = float(self.params.get("info_transmission_work", 0.2))
        p_comm = float(self.params.get("info_transmission_community", 0.1))

        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        N = len(agent_ids)
        received_info_t = np.zeros(N, dtype=int)

        p_broadcast = exo_rate * (1.0 + comm_mult * communication_intensity)
        p_broadcast = float(np.clip(p_broadcast, 0.0, 1.0))

        for idx, aid in enumerate(agent_ids):
            retain_prev = (received_info_prev[idx] == 1)
            retain_flag = 1 if (retain_prev and (np.random.rand() < (1.0 - mem_decay))) else 0

            p_neighbor = 0.0
            for layer, pL in zip(["family", "work", "community"], [p_family, p_work, p_comm]):
                neighbors = neighbor_index.get(aid, {}).get(layer, [])
                if not neighbors:
                    p_layer = 0.0
                else:
                    informed_count = 0
                    for nb in neighbors:
                        nb_idx = id_to_idx.get(nb, None)
                        if nb_idx is not None and received_info_prev[nb_idx] == 1:
                            informed_count += 1
                    if informed_count <= 0:
                        p_layer = 0.0
                    else:
                        p_layer = 1.0 - (1.0 - pL) ** informed_count
                p_neighbor += (1.0 / 3.0) * p_layer

            p_neighbor = float(np.clip(p_neighbor, 0.0, 1.0))
            p_total = 1.0 - (1.0 - p_broadcast) * (1.0 - p_neighbor)
            new_info_flag = 1 if (np.random.rand() < p_total) else 0
            received_info_t[idx] = max(retain_flag, new_info_flag)

        return received_info_t


class AttendanceAndContacts:
    """
    Computes daily contact weights across layers, potentially influenced by policy openings.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def compute_weights(self) -> Dict[str, float]:
        base_family = float(self.params.get("household_fraction", 0.3))
        base_work = float(self.params.get("workplace_fraction", 0.5))
        base_comm = float(self.params.get("public_space_fraction", 0.2))

        base_family = max(0.0, base_family)
        base_work = max(0.0, base_work)
        base_comm = max(0.0, base_comm)
        s = max(1e-9, base_family + base_work + base_comm)
        base_family /= s
        base_work /= s
        base_comm /= s

        open_work = float(self.params.get("location_open_fraction_work_school", 1.0))
        open_comm = float(self.params.get("location_open_fraction_public", 1.0))
        open_family = 1.0

        w_family = base_family * open_family
        w_work = base_work * open_work
        w_comm = base_comm * open_comm
        s2 = max(1e-9, w_family + w_work + w_comm)
        w_family /= s2
        w_work /= s2
        w_comm /= s2

        weights = {"family": float(w_family), "work": float(w_work), "community": float(w_comm)}
        return weights


class LocationEnforcement:
    """
    Applies location-specific mask requirement and enforcement levels with policy enforcement signal.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def compute_enforcement(self, policy_enforcement: float) -> Dict[str, float]:
        req_family = float(self.params.get("location_mask_requirement_level_family", 0.1))
        req_work = float(self.params.get("location_mask_requirement_level_work", 0.6))
        req_comm = float(self.params.get("location_mask_requirement_level_community", 0.5))

        base_family = float(self.params.get("location_enforcement_level_family", 0.05))
        base_work = float(self.params.get("location_enforcement_level_work", 0.6))
        base_comm = float(self.params.get("location_enforcement_level_community", 0.4))

        eff_family = float(np.clip(base_family * policy_enforcement * req_family, 0.0, 1.0))
        eff_work = float(np.clip(base_work * policy_enforcement * req_work, 0.0, 1.0))
        eff_comm = float(np.clip(base_comm * policy_enforcement * req_comm, 0.0, 1.0))

        return {"family": eff_family, "work": eff_work, "community": eff_comm}

    def observed_enforcement_with_noise(self, effective_enforcement: Dict[str, float]) -> Dict[str, float]:
        sigma = float(self.params.get("location_report_noise_sigma", 0.0))
        if sigma <= 0.0:
            return dict(effective_enforcement)
        noisy = {}
        for k, v in effective_enforcement.items():
            noisy[k] = float(np.clip(v + np.random.normal(0.0, sigma), 0.0, 1.0))
        return noisy


class AdoptionDynamics:
    """
    Updates mask-wearing states using a discrete-time hazard/logit influenced by peer exposure, information, policy, and risk perception.
    """

    def __init__(self, params: Dict[str, Any], risk_perception: np.ndarray, agents_df: pd.DataFrame) -> None:
        self.params = params
        self.risk_perception = np.array(risk_perception, dtype=float)

        # Per-agent heterogeneity arrays
        self.agent_ids = agents_df["agent_id"].astype(int).tolist()
        self.policy_sensitivity = agents_df.get("policy_sensitivity", pd.Series(1.0, index=agents_df.index)).to_numpy(dtype=float)
        self.compliance_probability = agents_df.get("compliance_probability", pd.Series(0.5, index=agents_df.index)).to_numpy(dtype=float)
        self.susceptibility_to_influence = agents_df.get("susceptibility_to_influence", pd.Series(1.0, index=agents_df.index)).to_numpy(dtype=float)
        self.peer_influence_weight = agents_df.get("peer_influence_weight", pd.Series(1.0, index=agents_df.index)).to_numpy(dtype=float)

    def update(self,
               adoption_prev: np.ndarray,
               exposure_curr: np.ndarray,
               received_info_prev: np.ndarray,
               policy_signals: Dict[str, float],
               location_enforcement: Dict[str, float],
               contact_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        lambda_peer = float(self.params.get("lambda_peer", 1.2))
        beta_info = float(self.params.get("beta_info", 0.8))
        beta0 = float(self.params.get("beta0", -2.0))
        phi_persist = float(self.params.get("phi_persist", 0.9))
        delta_drop = float(self.params.get("delta_drop", 0.02))
        mask_disutility = float(self.params.get("mask_disutility", 0.2))
        adoption_threshold = float(self.params.get("adoption_threshold", 0.6))
        risk_decay = float(self.params.get("risk_perception_decay", 0.01))
        policy_effect_strength = float(self.params.get("policy_effect_strength", 0.4))
        peer_cap = float(self.params.get("peer_influence_cap", 0.95))

        N = adoption_prev.shape[0]
        adoption_curr = np.array(adoption_prev, copy=True)

        # Weighted enforcement across layers using attendance weights if provided
        if contact_weights is None:
            w_family = w_work = w_comm = 1.0 / 3.0
        else:
            w_family = float(contact_weights.get("family", 0.0))
            w_work = float(contact_weights.get("work", 0.0))
            w_comm = float(contact_weights.get("community", 0.0))
            s = max(1e-9, w_family + w_work + w_comm)
            w_family /= s
            w_work /= s
            w_comm /= s

        avg_enf = float(
            w_family * location_enforcement.get("family", 0.0)
            + w_work * location_enforcement.get("work", 0.0)
            + w_comm * location_enforcement.get("community", 0.0)
        )

        base_policy_term = policy_effect_strength * (float(policy_signals.get("mandate_level", 0.0)) + 0.5 * avg_enf)

        I = np.minimum(peer_cap, np.maximum(0.0, exposure_curr))
        info = np.array(received_info_prev, dtype=float)

        masked_idxs = np.where(adoption_prev == 1)[0]
        unmasked_idxs = np.where(adoption_prev == 0)[0]

        if masked_idxs.size > 0:
            phi = np.minimum(1.0, (phi_persist + 0.1 * I[masked_idxs]))
            # Increase persistence based on compliance_probability
            phi *= np.clip(0.8 + 0.4 * self.compliance_probability[masked_idxs], 0.0, 1.2)
            phi = np.clip(phi, 0.0, 1.0)
            stay = (np.random.rand(masked_idxs.size) < phi).astype(int)
            revert_needed = (stay == 0)
            if np.any(revert_needed):
                idxs = masked_idxs[revert_needed]
                delta = delta_drop * (1.0 - I[idxs]) * (1.0 - 0.5 * self.compliance_probability[idxs])
                delta = np.clip(delta, 0.0, 1.0)
                dropout = (np.random.rand(idxs.size) < delta).astype(int)
                adoption_curr[idxs] = (1 - dropout)
            adoption_curr[masked_idxs[stay == 1]] = 1

        if unmasked_idxs.size > 0:
            # Apply per-agent susceptibility and peer influence weight to exposure effect
            I_eff = I[unmasked_idxs] * self.susceptibility_to_influence[unmasked_idxs] * self.peer_influence_weight[unmasked_idxs]
            I_eff = np.clip(I_eff, 0.0, peer_cap)
            policy_term_i = base_policy_term * self.policy_sensitivity[unmasked_idxs]
            base = beta0 + lambda_peer * I_eff + beta_info * info[unmasked_idxs] + policy_term_i - mask_disutility + 0.8 * self.risk_perception[unmasked_idxs]
            z = base - adoption_threshold
            p_adopt = sigmoid(z)
            adopt = (np.random.rand(unmasked_idxs.size) < p_adopt).astype(int)
            adoption_curr[unmasked_idxs] = adopt

        self.risk_perception = np.maximum(0.0, self.risk_perception * (1.0 - risk_decay))
        return adoption_curr


class Aggregation:
    """
    Aggregates population-level observables for calibration and evaluation.
    """

    def __init__(self, observation_noise_sigma: float = 0.0) -> None:
        self.observation_noise_sigma = float(observation_noise_sigma)

    def aggregate(self, adoption_state: np.ndarray, received_info_state: np.ndarray) -> Dict[str, float]:
        adoption_rate = float(np.mean(adoption_state))
        info_rate = float(np.mean(received_info_state))
        if self.observation_noise_sigma > 0.0:
            adoption_rate = float(np.clip(adoption_rate + np.random.normal(0.0, self.observation_noise_sigma), 0.0, 1.0))
            info_rate = float(np.clip(info_rate + np.random.normal(0.0, self.observation_noise_sigma), 0.0, 1.0))
        return {"adoption_rate": adoption_rate, "received_info_rate": info_rate}


class Simulation:
    """
    Main simulation class to orchestrate an agent-based diffusion model of mask-wearing behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.model_plan: Dict[str, Any] = self._build_model_plan_placeholder()
        self.config = self._build_default_config()
        if config:
            self.config.update(config)

        seed = int(self.config.get("random_seed", 42))
        set_seed(seed)

        self.data_loader = DataLoader(DATA_DIR)
        agents_df = self.data_loader.load_agents()
        social_net = self.data_loader.load_social_network()
        train_df = self.data_loader.load_train_data()

        # Align data if both available
        if agents_df is not None or social_net is not None:
            aligned_agents, aligned_net = self.data_loader.validate_and_align()
            if aligned_agents is not None:
                agents_df = aligned_agents
            if aligned_net is not None:
                social_net = aligned_net

        # If no agents_df, synthesize basic population
        pop_size = int(self.config.get("population_size", 10000))
        if agents_df is None:
            agents_df = self._synthesize_agents(pop_size)
        else:
            pop_size = len(agents_df)

        self.agent_ids = list(agents_df["agent_id"].astype(int).tolist())
        self.agents_df = agents_df.reset_index(drop=True)

        # Network processing
        self.network_preprocessor = NetworkPreprocessor(self.config)
        if social_net is None:
            # Generate synthetic multiplex network
            neighbor_index = self._synthesize_network(self.agent_ids)
        else:
            neighbor_index = self.network_preprocessor.build_neighbor_index(social_net, self.agent_ids)
        degrees_by_layer = self.network_preprocessor.compute_degrees(neighbor_index)

        self.neighbor_index = neighbor_index
        self.degrees_by_layer = degrees_by_layer

        # Modules
        self.exposure_calculator = ExposureCalculator(self.config, self.agents_df)
        self.policy_scheduler = PolicyAuthority(self.config)
        self.info_diffusion = InfoDiffusion(self.config)
        self.attendance_module = AttendanceAndContacts(self.config)
        self.enforcement_module = LocationEnforcement(self.config)
        initial_risk = self.agents_df["risk_perception"].to_numpy(dtype=float)
        self.adoption_dynamics = AdoptionDynamics(self.config, initial_risk, self.agents_df)
        self.aggregator = Aggregation(observation_noise_sigma=float(self.config.get("observation_noise_sigma", 0.0)))

        # Time horizon
        self.T = int(self.config.get("simulation_days", 60))
        self.N = pop_size

        # States over time
        self.adoption_state = np.zeros((self.T, self.N), dtype=int)
        self.received_info = np.zeros((self.T, self.N), dtype=int)
        self.exposure_state = np.zeros((self.T, self.N), dtype=float)
        self.observables = {
            "day": [],
            "adoption_rate": [],
            "received_info_rate": [],
            "is_prediction_window": [],
        }

        # Observed aggregate from train data if available
        self.observed_adoption_rate: Optional[np.ndarray] = None
        if train_df is not None:
            try:
                grp = train_df.groupby("day")["wearing_mask"].mean()
                observed = np.full(self.T, np.nan, dtype=float)
                for d, val in grp.items():
                    if 0 <= int(d) < self.T:
                        observed[int(d)] = float(val)
                self.observed_adoption_rate = observed
            except Exception as e:
                print("Warning: Could not compute observed aggregate adoption:", e)
                self.observed_adoption_rate = None

        # Initialize day 0 states
        self._initialize_day0_states(train_df)

        pp = self.model_plan.get("prediction_period", {"start_day": 30, "end_day": 39})
        self.prediction_window = (pp.get("start_day", 30), pp.get("end_day", 39))

    def _build_model_plan_placeholder(self) -> Dict[str, Any]:
        plan = {
            "prediction_period": {
                "start_day": 30,
                "end_day": 39
            },
            "evaluation_metrics": [
                "RMSE",
                "MAE",
                "FinalAdoptionRate",
                "StabilityLastWindow"
            ],
            "entities": ["Person", "Location", "PolicyAuthority"],
            "interactions": ["peer_influence", "policy_compliance", "location_enforcement", "attendance_contacts"],
            "environment": {"type": "network_multiplex", "time_step": 1, "time_unit": "days"},
        }
        return plan

    def _build_default_config(self) -> Dict[str, Any]:
        cfg = {
            # Global
            "simulation_days": 60,
            "population_size": 10000,
            "random_seed": 42,
            "observation_noise_sigma": 0.0,
            "average_degree": 8.0,
            "network_type": "scale_free",
            "calibration_enabled": False,
            # Network Preprocessor
            "symmetrize_edges": True,
            "remove_self_loops": True,
            "directed_mode": False,
            # ExposureCalculator
            "layer_weight_family": 0.5,
            "layer_weight_work": 0.3,
            "layer_weight_community": 0.2,
            "homophily_weight_same_age": 1.2,
            "homophily_weight_same_occ": 1.15,
            "exposure_smoothing_gamma": 0.1,
            # PolicyAuthority
            "policy_start_day": 10,
            "policy_end_day": 60,
            "mandate_level_base": 0.5,
            "communication_intensity_base": 0.3,
            "enforcement_level_global": 0.3,
            "subsidy_incentive_base": 0.0,
            "policy_ramp_rate": 0.1,
            "mandate_to_enforcement_multiplier": 0.5,
            # InfoDiffusion
            "info_transmission_family": 0.4,
            "info_transmission_work": 0.2,
            "info_transmission_community": 0.1,
            "exogenous_broadcast_rate": 0.01,
            "policy_comm_to_info_multiplier": 0.8,
            "info_memory_decay": 0.2,
            # Attendance
            "contact_rate_daily": 12,
            "household_fraction": 0.3,
            "workplace_fraction": 0.5,
            "public_space_fraction": 0.2,
            "attendance_variance": 0.1,
            "location_open_fraction_public": 1.0,
            "location_open_fraction_work_school": 1.0,
            # Location Enforcement
            "location_mask_requirement_level_family": 0.1,
            "location_mask_requirement_level_work": 0.6,
            "location_mask_requirement_level_community": 0.5,
            "location_enforcement_level_family": 0.05,
            "location_enforcement_level_work": 0.6,
            "location_enforcement_level_community": 0.4,
            "location_report_noise_sigma": 0.0,
            # AdoptionDynamics
            "lambda_peer": 1.2,
            "beta_info": 0.8,
            "beta0": -2.0,
            "phi_persist": 0.9,
            "delta_drop": 0.02,
            "mask_disutility": 0.2,
            "adoption_threshold": 0.6,
            "risk_perception_decay": 0.01,
            "policy_effect_strength": 0.4,
            "peer_influence_cap": 0.95,
            # Initialization
            "initial_adoption_rate": 0.2,
        }
        return cfg

    def _synthesize_agents(self, n: int) -> pd.DataFrame:
        age_groups = ["Youth", "Young Adult", "Middle Age"]
        occupations = ["Student", "White Collar", "Blue Collar"]

        rng = np.random.default_rng(self.config.get("random_seed", 42))
        df = pd.DataFrame({
            "agent_id": np.arange(n, dtype=int),
            "age_group": rng.choice(age_groups, size=n, p=[0.3, 0.4, 0.3]),
            "occupation": rng.choice(occupations, size=n, p=[0.3, 0.4, 0.3]),
            "risk_perception": np.clip(rng.beta(2.0, 5.0, size=n), 0.0, 1.0),
            "compliance_probability": np.clip(rng.normal(0.5, 0.15, size=n), 0.0, 1.0),
            "policy_sensitivity": np.clip(rng.normal(1.0, 0.2, size=n), 0.0, 2.0),
            "susceptibility_to_influence": np.clip(rng.normal(1.0, 0.25, size=n), 0.2, 2.0),
            "peer_influence_weight": np.clip(rng.normal(1.0, 0.25, size=n), 0.2, 2.0),
        })
        return df

    def _synthesize_network(self, agent_ids: List[int]) -> Dict[int, Dict[str, List[int]]]:
        """
        Create a synthetic multiplex network when no social network is provided.
        Uses NetworkX if available; otherwise falls back to a simple random graph.
        """
        n = len(agent_ids)
        id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        idx_to_id = {i: aid for aid, i in id_to_idx.items()}

        network_type = str(self.config.get("network_type", "scale_free"))
        avg_deg = float(self.config.get("average_degree", 8.0))
        m = max(1, int(avg_deg // 2))

        # Initialize empty adjacency per layer
        neighbor_index: Dict[int, Dict[str, List[int]]] = {aid: {"family": [], "work": [], "community": []} for aid in agent_ids}

        if NETWORKX_AVAILABLE and n > 2 and m > 0:
            try:
                if network_type.lower() in ["scale_free", "barabasi", "ba"]:
                    G = nx.barabasi_albert_graph(n=n, m=min(m, n - 1), seed=self.config.get("random_seed", 42))
                elif network_type.lower() in ["erdos_renyi", "er", "gnp"]:
                    p = min(1.0, max(0.0, avg_deg / max(1, n - 1)))
                    G = nx.erdos_renyi_graph(n=n, p=p, seed=self.config.get("random_seed", 42))
                else:
                    p = min(1.0, max(0.0, avg_deg / max(1, n - 1)))
                    G = nx.erdos_renyi_graph(n=n, p=p, seed=self.config.get("random_seed", 42))
                # Distribute edges across layers by attendance fractions
                layer_probs = self.attendance_module.compute_weights() if hasattr(self, "attendance_module") else {
                    "family": 0.3, "work": 0.5, "community": 0.2
                }
                layers = list(layer_probs.keys())
                probs = np.array([layer_probs["family"], layer_probs["work"], layer_probs["community"]], dtype=float)
                probs = probs / max(1e-9, probs.sum())
                for u, v in G.edges():
                    a = idx_to_id[u]
                    b = idx_to_id[v]
                    layer = np.random.choice(layers, p=probs)
                    neighbor_index[a][layer].append(b)
                    neighbor_index[b][layer].append(a)
            except Exception as e:
                print(f"Warning: Synthetic network generation with networkx failed: {e}. Falling back to random.")
                # Fall back
                neighbor_index = self._fallback_random_network(agent_ids, avg_deg)
        else:
            neighbor_index = self._fallback_random_network(agent_ids, avg_deg)

        # Deduplicate
        for aid in neighbor_index:
            for L in ["family", "work", "community"]:
                neighbor_index[aid][L] = list(dict.fromkeys(neighbor_index[aid][L]))
        return neighbor_index

    def _fallback_random_network(self, agent_ids: List[int], avg_deg: float) -> Dict[int, Dict[str, List[int]]]:
        n = len(agent_ids)
        neighbor_index: Dict[int, Dict[str, List[int]]] = {aid: {"family": [], "work": [], "community": []} for aid in agent_ids}
        if n <= 1:
            return neighbor_index
        # approximate number of edges
        m_edges = int((avg_deg * n) / 2)
        layers = ["family", "work", "community"]
        layer_probs = self.attendance_module.compute_weights() if hasattr(self, "attendance_module") else {
            "family": 0.3, "work": 0.5, "community": 0.2
        }
        probs = [layer_probs["family"], layer_probs["work"], layer_probs["community"]]
        for _ in range(m_edges):
            a, b = np.random.choice(agent_ids, size=2, replace=False)
            layer = np.random.choice(layers, p=np.array(probs) / max(1e-9, sum(probs)))
            if b not in neighbor_index[a][layer]:
                neighbor_index[a][layer].append(b)
            if a not in neighbor_index[b][layer]:
                neighbor_index[b][layer].append(a)
        return neighbor_index

    def _initialize_day0_states(self, train_df: Optional[pd.DataFrame]) -> None:
        id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        N = self.N

        if train_df is not None and (train_df["day"] == 0).any():
            day0 = train_df.loc[train_df["day"] == 0].copy()
            adopt0 = np.zeros(N, dtype=int)
            info0 = np.zeros(N, dtype=int)
            for _, row in day0.iterrows():
                aid = int(row["agent_id"])
                i = id_to_idx.get(aid, None)
                if i is not None:
                    adopt0[i] = int(row["wearing_mask"])
                    info0[i] = int(row["received_info"])
            self.adoption_state[0, :] = adopt0
            self.received_info[0, :] = info0
        else:
            init_rate = float(self.config.get("initial_adoption_rate", 0.2))
            risk = self.agents_df["risk_perception"].to_numpy(dtype=float)
            logit_p = np.log(init_rate / max(1e-9, 1.0 - init_rate)) + 1.5 * (risk - 0.5)
            p = sigmoid(logit_p)
            self.adoption_state[0, :] = (np.random.rand(N) < p).astype(int)
            self.received_info[0, :] = np.zeros(N, dtype=int)

        # Initialize exposure for day 0 using adoption at day 0
        daily_weights = self.attendance_module.compute_weights()
        self.exposure_state[0, :] = self.exposure_calculator.compute_exposure(
            self.neighbor_index, self.adoption_state[0, :], self.agent_ids, exposure_prev=None, daily_layer_weights=daily_weights
        )

    def run(self) -> None:
        iterator = range(0, self.T)
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc="Simulating days")

        for t in iterator:
            policy_signals = self.policy_scheduler.update(t)
            contact_weights = self.attendance_module.compute_weights()

            effective_enforcement = self.enforcement_module.compute_enforcement(policy_signals.get("enforcement_level", 0.0))
            _ = self.enforcement_module.observed_enforcement_with_noise(effective_enforcement)

            if t > 0:
                self.exposure_state[t, :] = self.exposure_calculator.compute_exposure(
                    self.neighbor_index, self.adoption_state[t - 1, :], self.agent_ids, exposure_prev=self.exposure_state[t - 1, :], daily_layer_weights=contact_weights
                )
                self.received_info[t, :] = self.info_diffusion.update(
                    self.neighbor_index, self.received_info[t - 1, :], policy_signals.get("communication_intensity", 0.0), self.agent_ids
                )
                self.adoption_state[t, :] = self.adoption_dynamics.update(
                    self.adoption_state[t - 1, :],
                    self.exposure_state[t, :],
                    self.received_info[t - 1, :],
                    policy_signals,
                    effective_enforcement,
                    contact_weights=contact_weights
                )

            obs = self.aggregator.aggregate(self.adoption_state[t, :], self.received_info[t, :])
            self.observables["day"].append(t)
            self.observables["adoption_rate"].append(obs["adoption_rate"])
            self.observables["received_info_rate"].append(obs["received_info_rate"])
            s, e = self.prediction_window
            in_pred = int(s is not None and e is not None and s <= t <= e)
            self.observables["is_prediction_window"].append(in_pred)

    def evaluate(self) -> Dict[str, Union[float, int]]:
        metrics: Dict[str, Union[float, int]] = {}
        sim_adoption = np.array(self.observables["adoption_rate"], dtype=float)
        observed = self.observed_adoption_rate

        start_day = 0
        end_day = min(29, self.T - 1)

        for m in self.model_plan.get("evaluation_metrics", []):
            if m == "RMSE":
                if observed is not None:
                    y_true = observed[start_day:end_day + 1]
                    y_pred = sim_adoption[start_day:end_day + 1]
                    mask = ~np.isnan(y_true)
                    metrics["RMSE"] = rmse(y_true[mask], y_pred[mask]) if mask.any() else float("nan")
                else:
                    metrics["RMSE"] = float("nan")
            elif m == "MAE":
                if observed is not None:
                    y_true = observed[start_day:end_day + 1]
                    y_pred = sim_adoption[start_day:end_day + 1]
                    mask = ~np.isnan(y_true)
                    metrics["MAE"] = mae(y_true[mask], y_pred[mask]) if mask.any() else float("nan")
                else:
                    metrics["MAE"] = float("nan")
            elif m == "FinalAdoptionRate":
                if observed is not None and end_day < len(sim_adoption):
                    y_true = observed[end_day]
                    y_pred = sim_adoption[end_day]
                    metrics["FinalAdoptionRate"] = abs(y_pred - y_true) if not np.isnan(y_true) else float("nan")
                else:
                    metrics["FinalAdoptionRate"] = float("nan")
            elif m == "StabilityLastWindow":
                last_n = 7
                if len(sim_adoption) >= last_n + 1:
                    diffs = np.abs(np.diff(sim_adoption[-(last_n + 1):]))
                    metrics["StabilityLastWindow"] = float(np.mean(diffs))
                else:
                    metrics["StabilityLastWindow"] = float("nan")
            elif m == "PeakAdoptionDay":
                metrics["PeakAdoptionDay"] = int(np.nanargmax(sim_adoption)) if sim_adoption.size > 0 else -1
            elif m == "TimeToTargetAdoption":
                target = 0.7
                idxs = np.where(sim_adoption >= target)[0]
                metrics["TimeToTargetAdoption"] = int(idxs[0]) if idxs.size > 0 else -1
            else:
                metrics[m] = float("nan")

        return metrics

    def visualize(self) -> None:
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib is not available; skipping visualization.")
            return

        days = np.array(self.observables["day"])
        adoption = np.array(self.observables["adoption_rate"])
        info = np.array(self.observables["received_info_rate"])

        plt.figure(figsize=(10, 6))
        plt.plot(days, adoption, label="Simulated Adoption Rate", color="tab:blue")
        plt.plot(days, info, label="Simulated Received Info Rate", color="tab:orange", alpha=0.7)
        if self.observed_adoption_rate is not None:
            plt.plot(days, self.observed_adoption_rate[:len(days)], label="Observed Adoption Rate", color="tab:green", linestyle="--", alpha=0.8)
        if self.prediction_window:
            s, e = self.prediction_window
            if s is not None and e is not None:
                plt.axvspan(s, e, color="gray", alpha=0.15, label="Prediction Window")
        plt.xlabel("Day")
        plt.ylabel("Rate")
        plt.title("Mask Adoption and Information Rates Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str) -> None:
        try:
            df = pd.DataFrame(self.observables)
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results to {filename}: {e}")

    def save_prediction_results(self, filename: str) -> None:
        try:
            df = pd.DataFrame(self.observables)
            pred_df = df[df["is_prediction_window"] == 1].copy()
            pred_df.to_csv(filename, index=False)
            print(f"Prediction window results saved to {filename}")
        except Exception as e:
            print(f"Error saving prediction results to {filename}: {e}")

    @staticmethod
    def calibrate_grid(base_config: Dict[str, Any], param_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Simple grid search calibration over a handful of parameters minimizing RMSE over days 0-29.
        Returns an updated config with the best-found parameters.
        """
        if param_grid is None:
            # Define a small grid around defaults
            def around(val, scale=0.2, n=3, low=None, high=None):
                vals = np.linspace(val * (1 - scale), val * (1 + scale), n)
                if low is not None:
                    vals = np.maximum(vals, low)
                if high is not None:
                    vals = np.minimum(vals, high)
                return list(vals)
            param_grid = {
                "lambda_peer": around(base_config.get("lambda_peer", 1.2), scale=0.5, n=3, low=0.0, high=3.0),
                "beta_info": around(base_config.get("beta_info", 0.8), scale=0.5, n=3, low=0.0, high=2.0),
                "policy_effect_strength": around(base_config.get("policy_effect_strength", 0.4), scale=0.5, n=3, low=0.0, high=1.5),
                "mask_disutility": around(base_config.get("mask_disutility", 0.2), scale=0.5, n=3, low=0.0, high=1.0),
                "adoption_threshold": around(base_config.get("adoption_threshold", 0.6), scale=0.3, n=3, low=0.0, high=1.0),
                "beta0": around(base_config.get("beta0", -2.0), scale=0.2, n=3, low=-5.0, high=0.0),
            }

        # Build all combinations
        keys = list(param_grid.keys())
        grids = [param_grid[k] for k in keys]
        best_rmse = float("inf")
        best_cfg = dict(base_config)

        total_combos = 1
        for g in grids:
            total_combos *= len(g)
        combo_count = 0

        for values in np.array(np.meshgrid(*grids, indexing="ij")).T.reshape(-1, len(keys)):
            cfg_try = dict(base_config)
            for k, v in zip(keys, values):
                cfg_try[k] = float(v)
            # Ensure reproducibility
            cfg_try["random_seed"] = base_config.get("random_seed", 42)
            # Run simulation
            sim = Simulation(config=cfg_try)
            sim.run()
            metrics = sim.evaluate()
            current_rmse = float(metrics.get("RMSE", float("inf")))
            combo_count += 1
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_cfg = cfg_try
        print(f"Calibration complete. Best RMSE={best_rmse}")
        return best_cfg


def main() -> None:
    config = {
        "simulation_days": 60,
        "random_seed": 42,
        # Enable calibration if desired
        "calibration_enabled": False,
    }

    if config.get("calibration_enabled", False):
        print("Starting calibration...")
        best_config = Simulation.calibrate_grid(config)
        print("Using calibrated configuration.")
        sim = Simulation(config=best_config)
    else:
        sim = Simulation(config=config)

    sim.run()
    metrics = sim.evaluate()
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    sim.visualize()
    sim.save_results("results.csv")
    sim.save_prediction_results("prediction_window_results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()