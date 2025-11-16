import argparse
import json
import math
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd


# ==============================
# Utilities and Global Settings
# ==============================

def set_global_seed(seed: int) -> None:
    """
    Set global random seeds for reproducibility.

    Args:
        seed: Seed value to apply to Python's random and NumPy RNG.
    """
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    """
    Save a JSON-serializable object to a file.

    Args:
        obj: Python object that is JSON-serializable.
        path: Destination file path.
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    """
    Load a JSON object from a file.

    Args:
        path: Source file path.

    Returns:
        Deserialized Python object.
    """
    with open(path, "r") as f:
        return json.load(f)


def timestamp() -> str:
    """
    Generate a human-readable timestamp string for artifact directories.

    Returns:
        A timestamp string in the format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =====================
# Data Loading Helpers
# =====================

def resolve_data_dir() -> str:
    """
    Resolve data directory using environment variables PROJECT_ROOT and DATA_PATH.

    Returns:
        Absolute data directory path.
    """
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    data_path = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
    data_dir = os.path.join(project_root, data_path)
    return data_dir


def load_agent_attributes(path: str) -> pd.DataFrame:
    """
    Load agent attributes CSV.

    Args:
        path: CSV file path.

    Returns:
        DataFrame with agent attributes.

    Raises:
        RuntimeError: If file cannot be loaded.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load agent_attributes from {path}: {e}") from e


def load_social_network(path: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Load social network JSON.

    Args:
        path: JSON file path.

    Returns:
        Dictionary mapping agent ID to adjacency lists per layer.

    Raises:
        RuntimeError: If file cannot be loaded or parse fails.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load social_network from {path}: {e}") from e


def load_train_data(path: str) -> pd.DataFrame:
    """
    Load training data CSV (longitudinal panel).

    Args:
        path: CSV file path.

    Returns:
        DataFrame with time-indexed micro-data.

    Raises:
        RuntimeError: On load failure.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load train_data from {path}: {e}") from e


def align_ids(
    agents_df: pd.DataFrame,
    social: Dict[str, Dict[str, List[int]]],
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[int, int], pd.DataFrame, Dict[str, Dict[str, List[int]]], pd.DataFrame]:
    """
    Align agent IDs across data sources and filter to common population.

    Args:
        agents_df: Agent attributes DataFrame.
        social: Social network data mapping agent IDs to neighbor lists per layer.
        train_df: Train data DataFrame with columns agent_id, day, wearing_mask, received_info.

    Returns:
        Tuple of (common_ids array, id2idx mapping, filtered agents_df, filtered social network, filtered train_df).
    """
    agents_ids = set(agents_df["agent_id"].astype(int).tolist())
    social_ids = set(int(k) for k in social.keys())
    train_ids = set(train_df["agent_id"].astype(int).unique().tolist())
    common = sorted(list(agents_ids & social_ids & train_ids))
    if len(common) == 0:
        raise RuntimeError("No common agent IDs across agent_attributes.csv, social_network.json, and train_data.csv")
    id2idx = {aid: i for i, aid in enumerate(common)}
    agents_df_f = agents_df[agents_df["agent_id"].isin(common)].copy()
    train_df_f = train_df[train_df["agent_id"].isin(common)].copy()
    social_f: Dict[str, Dict[str, List[int]]] = {}
    for k_str, v in social.items():
        ik = int(k_str)
        if ik in id2idx:
            social_f[k_str] = {
                "family": [int(x) for x in v.get("family", []) if int(x) in id2idx],
                "work_school": [int(x) for x in v.get("work_school", []) if int(x) in id2idx],
                "community": [int(x) for x in v.get("community", []) if int(x) in id2idx],
            }
    return np.array(common, dtype=int), id2idx, agents_df_f, social_f, train_df_f


def build_multiplex_adjacency(
    social: Dict[str, Dict[str, List[int]]],
    id2idx: Dict[int, int],
    n: int,
) -> Dict[str, List[np.ndarray]]:
    """
    Build multiplex adjacency lists (family, work/school, community), symmetrized.

    Args:
        social: Filtered social network mapping.
        id2idx: Mapping from agent_id to row index.
        n: Population size.

    Returns:
        Dict mapping layer name to list of neighbor arrays per agent.
    """
    layers = ["family", "work_school", "community"]
    adj: Dict[str, List[set]] = {layer: [set() for _ in range(n)] for layer in layers}
    for k_str, v in social.items():
        i = id2idx[int(k_str)]
        for layer in layers:
            for nbr in v.get(layer, []):
                if nbr in id2idx:
                    j = id2idx[nbr]
                    if i != j:
                        adj[layer][i].add(j)
                        adj[layer][j].add(i)
    adj_arrays: Dict[str, List[np.ndarray]] = {}
    for layer in layers:
        arr_list: List[np.ndarray] = []
        for i in range(n):
            if len(adj[layer][i]) == 0:
                arr_list.append(np.array([], dtype=int))
            else:
                arr_list.append(np.fromiter(adj[layer][i], dtype=int))
        adj_arrays[layer] = arr_list
    return adj_arrays


def pivot_states(train_df: pd.DataFrame, id2idx: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Pivot long-format train_data into wide arrays.

    Args:
        train_df: Panel DataFrame.
        id2idx: Mapping of agent ids to indices.

    Returns:
        Tuple (wearing array T x N, received_info array T x N, days list).
    """
    days = sorted(train_df["day"].unique().tolist())
    n_days = len(days)
    n_agents = len(id2idx)
    wearing = np.zeros((n_days, n_agents), dtype=np.float64)
    received = np.zeros((n_days, n_agents), dtype=np.float64)
    df_sorted = train_df.sort_values(["day", "agent_id"])
    day_to_idx = {d: i for i, d in enumerate(days)}
    for _, row in df_sorted.iterrows():
        d = int(row["day"])
        a = int(row["agent_id"])
        i_day = day_to_idx[d]
        i_agent = id2idx[a]
        wearing[i_day, i_agent] = 1.0 if bool(row["wearing_mask"]) else 0.0
        received[i_day, i_agent] = 1.0 if bool(row["received_info"]) else 0.0
    return wearing, received, days


def encode_demographics(agents_df: pd.DataFrame, common_ids: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Encode age_group and occupation as one-hot excluding baseline category per feature.

    Args:
        agents_df: Filtered agent DataFrame covering common IDs.
        common_ids: Array of agent IDs in the desired order.

    Returns:
        Tuple (age_oh, age_cat_names, occ_oh, occ_cat_names).
    """
    df_sorted = agents_df.set_index("agent_id").loc[common_ids]
    age_groups = df_sorted["age_group"].astype(str).tolist()
    occs = df_sorted["occupation"].astype(str).tolist()
    age_cats_all = sorted(list(pd.unique(df_sorted["age_group"].astype(str))))
    occ_cats_all = sorted(list(pd.unique(df_sorted["occupation"].astype(str))))
    # Choose baseline to reduce multicollinearity
    age_baseline = "Middle Age" if "Middle Age" in age_cats_all else age_cats_all[0]
    occ_baseline = "White Collar" if "White Collar" in occ_cats_all else occ_cats_all[0]
    age_cats = [c for c in age_cats_all if c != age_baseline]
    occ_cats = [c for c in occ_cats_all if c != occ_baseline]
    n = len(common_ids)
    age_oh = np.zeros((n, len(age_cats)), dtype=np.float64)
    occ_oh = np.zeros((n, len(occ_cats)), dtype=np.float64)
    age_index_map = {c: idx for idx, c in enumerate(age_cats)}
    occ_index_map = {c: idx for idx, c in enumerate(occ_cats)}
    for i in range(n):
        ag = age_groups[i]
        oc = occs[i]
        if ag in age_index_map:
            age_oh[i, age_index_map[ag]] = 1.0
        if oc in occ_index_map:
            occ_oh[i, occ_index_map[oc]] = 1.0
    age_cat_names = age_cats
    occ_cat_names = occ_cats
    return age_oh, age_cat_names, occ_oh, occ_cat_names


def compute_mem_info(received_info: np.ndarray, rho: float) -> np.ndarray:
    """
    Compute decaying memory of information over time per agent.

    Args:
        received_info: Binary matrix T x N of info receipt.
        rho: Decay rate in [0,1]; higher retains more memory.

    Returns:
        Memory matrix T x N representing decayed info memory.
    """
    T, N = received_info.shape
    mem = np.zeros((T, N), dtype=np.float64)
    for t in range(1, T):
        mem[t, :] = rho * mem[t - 1, :] + (1.0 - rho) * received_info[t, :]
    return mem


def compute_layer_neighbor_share(states: np.ndarray, neighbors: List[np.ndarray]) -> np.ndarray:
    """
    Compute share of neighbors wearing mask per agent for a given layer.

    Args:
        states: Binary vector N of previous wearing states.
        neighbors: List of neighbor arrays per agent.

    Returns:
        Share vector N with averages per agent per layer.
    """
    n = states.shape[0]
    shares = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = neighbors[i]
        if neigh.size == 0:
            shares[i] = 0.0
        else:
            shares[i] = float(np.mean(states[neigh]))
    return shares


def derive_layer_weights_and_betas(theta_f: float, theta_w: float, theta_c: float) -> Tuple[float, float, float, float, float, float]:
    """
    Normalize layer coefficients to weights that sum to 1 and return absolute betas.

    Args:
        theta_f: Family peer coefficient.
        theta_w: Work/school peer coefficient.
        theta_c: Community peer coefficient.

    Returns:
        Tuple of (w_f, w_w, w_c, beta_f, beta_w, beta_c).
    """
    coefs = np.array([theta_f, theta_w, theta_c], dtype=np.float64)
    abs_coefs = np.abs(coefs)
    total = np.sum(abs_coefs)
    if total <= 1e-12:
        w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)
    else:
        w = abs_coefs / total
    beta_f, beta_w, beta_c = float(abs_coefs[0]), float(abs_coefs[1]), float(abs_coefs[2])
    return float(w[0]), float(w[1]), float(w[2]), beta_f, beta_w, beta_c


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.

    Args:
        z: Input vector/matrix.

    Returns:
        Sigmoid-transformed array.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# ==========================
# Parameter System and I/O
# ==========================

@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.

    This dataclass adheres to the Calibrasim parameter interface requirements.
    """
    decision_weights: Dict[str, Any]              # e.g., alpha, gamma, theta_f, ..., age_effects, occ_effects
    layer_weights: Dict[str, float]               # e.g., family, work_school, community
    info_params: Dict[str, float]                 # e.g., phi_family, ..., lambda_broadcast_base, lambda_factor...
    noise_params: Dict[str, float]                # e.g., tau
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Optional per-module params
    engine_type: str = "calibrasim"               # Engine compatibility identifier
    meta: Dict[str, Any] = field(default_factory=dict)  # Seed, calibrator_name, windows, notes

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dict representation of this FittedParams instance.
        """
        return asdict(self)


class ParamsAdapter(ABC):
    """
    Adapts FittedParams to simulation parameter system.

    Implementations should handle:
    - Frozen parameter checks using a parameter_definitions.json
    - Applying nested decision/info/layer weights to simulation
    - Persisting parameters_used.json
    """

    @abstractmethod
    def apply(self, simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json.

        Args:
            simulation: Simulation instance supporting set_params and params registry.
            params: Fitted parameters to apply.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def capture(self, simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams snapshot derived from simulation internal params.
        """
        pass

    @abstractmethod
    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen params and return warnings.

        Args:
            params: Parameters to validate.

        Returns:
            Dict mapping parameter names to warning strings for ignored updates.
        """
        pass


class DefaultParamsAdapter(ParamsAdapter):
    """
    Default ParamsAdapter that maps FittedParams to Simulation parameter system.

    This adapter loads parameter_definitions.json if available to respect frozen flags.
    """

    def __init__(self, defs_path: str, params_used_path: str):
        """
        Initialize adapter with definition and output paths.

        Args:
            defs_path: Path to parameter_definitions.json.
            params_used_path: Path to write parameters_used.json upon apply().
        """
        self.defs_path = defs_path
        self.params_used_path = params_used_path
        self.param_defs = self._load_or_default_defs()

    def _load_or_default_defs(self) -> Dict[str, Any]:
        """
        Load parameter definitions from JSON or create defaults.

        Returns:
            Dict of parameter definitions with frozen flags.
        """
        if os.path.exists(self.defs_path):
            try:
                with open(self.defs_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        # Default definitions as fallback (non-frozen)
        return {
            "decision": {"frozen": False, "children": {}},
            "layers": {"frozen": False, "children": {}},
            "info": {"frozen": False, "children": {}},
            "noise": {"frozen": False, "children": {}},
        }

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate frozen parameters and return warnings to ignore updates.

        Args:
            params: Fitted parameters.

        Returns:
            Dict mapping param path to warning string.
        """
        warnings: Dict[str, str] = {}

        def is_frozen(path: List[str]) -> bool:
            node = self.param_defs
            for p in path:
                if p in node:
                    node = node[p]
                elif "children" in node and p in node["children"]:
                    node = node["children"][p]
                else:
                    # Unknown path, treat as not frozen
                    return False
            return bool(node.get("frozen", False))

        # Check decision scalar keys (alpha, gamma, theta_*, beta_*)
        for key in ["alpha", "gamma", "theta_f", "theta_w", "theta_c", "beta_r", "beta_i"]:
            if key in params.decision_weights and is_frozen(["decision", key]):
                warnings[f"decision.{key}"] = "Frozen; ignoring update"

        # Check decision nested effects
        for group in ["age_effects", "occ_effects"]:
            if group in params.decision_weights:
                for k in params.decision_weights[group].keys():
                    if is_frozen(["decision", group, k]):
                        warnings[f"decision.{group}.{k}"] = "Frozen; ignoring update"

        # Check layer weights
        for k in ["family", "work_school", "community"]:
            if k in params.layer_weights and is_frozen(["layers", k]):
                warnings[f"layers.{k}"] = "Frozen; ignoring update"

        # Check info params
        for k in ["phi_family", "phi_work", "phi_community", "lambda_broadcast_base", "lambda_broadcast_factor_after_day10", "rho_info_decay"]:
            if k in params.info_params and is_frozen(["info", k]):
                warnings[f"info.{k}"] = "Frozen; ignoring update"

        # Check noise
        for k in ["tau"]:
            if k in params.noise_params and is_frozen(["noise", k]):
                warnings[f"noise.{k}"] = "Frozen; ignoring update"

        return warnings

    def apply(self, simulation, params: FittedParams) -> None:
        """
        Apply FittedParams to simulation, respecting frozen flags, and persist parameters_used.json.

        Args:
            simulation: Simulation engine instance to be updated.
            params: Fitted parameter bundle.

        Returns:
            None.
        """
        warnings = self.validate_frozen(params)
        # Decision weights
        decision = dict(params.decision_weights)
        for k in ["alpha", "gamma", "theta_f", "theta_w", "theta_c", "beta_r", "beta_i"]:
            if f"decision.{k}" in warnings:
                if k in decision:
                    del decision[k]
        for group in ["age_effects", "occ_effects"]:
            if group in decision:
                sub = dict(decision[group])
                for sk in list(sub.keys()):
                    if f"decision.{group}.{sk}" in warnings:
                        del sub[sk]
                decision[group] = sub

        # Layer weights
        layers = dict(params.layer_weights)
        for k in list(layers.keys()):
            if f"layers.{k}" in warnings:
                del layers[k]

        # Info params
        info = dict(params.info_params)
        for k in list(info.keys()):
            if f"info.{k}" in warnings:
                del info[k]

        # Noise params
        noise = dict(params.noise_params)
        for k in list(noise.keys()):
            if f"noise.{k}" in warnings:
                del noise[k]

        # Apply to simulation
        simulation.set_params(
            decision=decision,
            layers=layers,
            info=info,
            noise=noise
        )

        # Persist used parameters including frozen and applied overrides
        used = {
            "decision": simulation.params.get("decision", {}),
            "layers": simulation.params.get("layers", {}),
            "info": simulation.params.get("info", {}),
            "noise": simulation.params.get("noise", {}),
            "warnings": warnings,
            "meta": params.meta,
        }
        ensure_dir(os.path.dirname(self.params_used_path))
        save_json(used, self.params_used_path)

    def capture(self, simulation) -> FittedParams:
        """
        Capture a snapshot of simulation parameters.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams created from simulation internal param state.
        """
        dec = simulation.params.get("decision", {})
        layers = simulation.params.get("layers", {})
        info = simulation.params.get("info", {})
        noise = simulation.params.get("noise", {})
        return FittedParams(
            decision_weights=dec,
            layer_weights=layers,
            info_params=info,
            noise_params=noise,
            module_params={},
            meta={"captured_at": timestamp()}
        )


# ==========================
# Simulation Engine Classes
# ==========================

@dataclass
class SimulationConfig:
    """
    Configuration for simulation runs and calibration.

    This is populated from CLI parameters and/or parameter file.
    """
    seed: int = 42
    k_runs: int = 20
    l2_reg: float = 1.0
    max_iter: int = 400
    learning_rate: float = 0.1
    gov_intervention_day: int = 10
    gov_lam_factor_default: float = 1.5
    rho_info_decay_default: float = 0.5
    forecast_days: int = 10
    artifacts_dir: str = "artifacts"
    output_dir: str = "outputs"
    verbose: bool = True


class ModuleBase(ABC):
    """
    Base class for all simulation modules.

    Modules read from simulation state and buffers, write to buffers, but do not commit to state.
    """

    def __init__(self, name: str, simulation: "Simulation"):
        """
        Initialize a module with a name and simulation reference.

        Args:
            name: Unique module name.
            simulation: Simulation engine instance.
        """
        self.name = name
        self.sim = simulation

    @abstractmethod
    def forward(self, t: int) -> None:
        """
        Compute outputs at time t and write to buffers.

        Args:
            t: Time index (0-based for days array).

        Returns:
            None.
        """
        pass


class InfoPropagationModule(ModuleBase):
    """
    Module that models information propagation via peers and exogenous broadcast.

    Outputs:
        - buffer["received_info"][t, :]
        - buffer["mem_info"][t, :] updated with decay
    """

    def forward(self, t: int) -> None:
        """
        Execute info propagation step for day t based on previous wearing state.

        Args:
            t: Time index.

        Returns:
            None.
        """
        # Last wearing state
        prev_states = self.sim.state["wearing_prev"]
        neighbors = self.sim.neighbors
        phi_f = self.sim.params["info"]["phi_family"]
        phi_w = self.sim.params["info"]["phi_work"]
        phi_c = self.sim.params["info"]["phi_community"]
        lam_base = self.sim.params["info"]["lambda_broadcast_base"]
        lam_factor = self.sim.params["info"]["lambda_broadcast_factor_after_day10"]
        # FIXED: Use cfg.gov_intervention_day instead of hard-coded day 10
        # Apply intervention factor when day index matches configuration
        gov_day = self.sim.cfg.gov_intervention_day
        lam = float(lam_base * (lam_factor if (t >= gov_day) else 1.0))
        # Compute neighbor shares
        share_f = compute_layer_neighbor_share(prev_states, neighbors["family"])
        share_w = compute_layer_neighbor_share(prev_states, neighbors["work_school"])
        share_c = compute_layer_neighbor_share(prev_states, neighbors["community"])
        # Info probability
        u = phi_f * share_f + phi_w * share_w + phi_c * share_c + lam
        p_info = 1.0 - np.exp(-np.clip(u, 0.0, 50.0))
        rec = (np.random.rand(self.sim.N) < p_info).astype(np.float64)
        self.sim.buffers["received_info"][t, :] = rec
        # Memory decay update
        rho = self.sim.params["info"]["rho_info_decay"]
        if t == 0:
            prev_mem = np.zeros(self.sim.N, dtype=np.float64)
        else:
            prev_mem = self.sim.buffers["mem_info"][t - 1, :]
        mem_t = rho * prev_mem + (1.0 - rho) * rec
        self.sim.buffers["mem_info"][t, :] = mem_t


class DecisionModule(ModuleBase):
    """
    Module that computes adoption probabilities from inputs using logistic rule.

    Outputs:
        - buffer["prob_wear"][t, :]
    """

    def forward(self, t: int) -> None:
        """
        Compute adoption probability for day t.

        Args:
            t: Time index.

        Returns:
            None.
        """
        prev_states = self.sim.state["wearing_prev"]
        neighbors = self.sim.neighbors
        # Peer shares
        share_f = compute_layer_neighbor_share(prev_states, neighbors["family"])
        share_w = compute_layer_neighbor_share(prev_states, neighbors["work_school"])
        share_c = compute_layer_neighbor_share(prev_states, neighbors["community"])
        # Decision parameters
        dec = self.sim.params["decision"]
        alpha = dec.get("alpha", 0.0)
        gamma = dec.get("gamma", 1.0)
        theta_f = dec.get("theta_f", 1.0)
        theta_w = dec.get("theta_w", 1.0)
        theta_c = dec.get("theta_c", 1.0)
        beta_r = dec.get("beta_r", 0.0)
        beta_i = dec.get("beta_i", 0.0)
        age_effects = dec.get("age_effects", {})
        occ_effects = dec.get("occ_effects", {})

        # Demographic vectors aligned to one-hot
        # FIXED: Removed hard-coded demographic category names; use dynamic names from encode_demographics
        age_vec = np.array([age_effects.get(cat, 0.0) for cat in self.sim.age_cat_names], dtype=np.float64) if self.sim.age_oh.shape[1] > 0 else np.zeros(0)
        occ_vec = np.array([occ_effects.get(cat, 0.0) for cat in self.sim.occ_cat_names], dtype=np.float64) if self.sim.occ_oh.shape[1] > 0 else np.zeros(0)

        # Compose logits
        logits = (
            alpha
            + gamma * prev_states
            + theta_f * share_f
            + theta_w * share_w
            + theta_c * share_c
            + beta_r * self.sim.risk
            + beta_i * self.sim.buffers["mem_info"][t, :]
        )
        if self.sim.age_oh.shape[1] > 0:
            logits += self.sim.age_oh @ age_vec
        if self.sim.occ_oh.shape[1] > 0:
            logits += self.sim.occ_oh @ occ_vec
        tau = float(self.sim.params["noise"].get("tau", 1.0))
        if tau > 0:
            logits = logits / tau
        self.sim.buffers["prob_wear"][t, :] = sigmoid(logits)


class StateUpdateModule(ModuleBase):
    """
    Module that samples new wearing state from probability.

    Outputs:
        - buffer["wearing"][t, :]
    """

    def forward(self, t: int) -> None:
        """
        Sample wearing state for day t from prob_wear.

        Args:
            t: Time index.

        Returns:
            None.
        """
        p = self.sim.buffers["prob_wear"][t, :]
        new_state = (np.random.rand(self.sim.N) < p).astype(np.float64)
        self.sim.buffers["wearing"][t, :] = new_state


class Simulation:
    """
    Main Simulation engine that coordinates modules and maintains state.

    Provides methods to run, evaluate, save results, and export module I/O.
    """

    def __init__(
        self,
        wearing_obs: np.ndarray,
        received_obs: np.ndarray,
        neighbors: Dict[str, List[np.ndarray]],
        risk: np.ndarray,
        age_oh: np.ndarray,
        occ_oh: np.ndarray,
        age_cat_names: List[str],
        occ_cat_names: List[str],
        cfg: SimulationConfig,
    ):
        """
        Initialize Simulation instance with data and configuration.

        Args:
            wearing_obs: Observed wearing mask states T x N.
            received_obs: Observed received info T x N.
            neighbors: Multiplex adjacency lists per layer.
            risk: Risk perception vector N.
            age_oh: Age one-hot matrix N x A (baseline removed).
            occ_oh: Occupation one-hot matrix N x O (baseline removed).
            age_cat_names: Names for age_oh columns.
            occ_cat_names: Names for occ_oh columns.
            cfg: Simulation configuration.
        """
        self.cfg = cfg
        self.wearing_obs = wearing_obs
        self.received_obs = received_obs
        self.T = wearing_obs.shape[0]
        self.N = wearing_obs.shape[1]
        self.neighbors = neighbors
        self.risk = risk
        self.age_oh = age_oh
        self.occ_oh = occ_oh
        self.age_cat_names = age_cat_names
        self.occ_cat_names = occ_cat_names
        self.params: Dict[str, Any] = {
            "decision": {
                "alpha": 0.0, "gamma": 1.0, "theta_f": 1.0, "theta_w": 1.0, "theta_c": 1.0,
                "beta_r": 0.0, "beta_i": 0.0, "age_effects": {}, "occ_effects": {}
            },
            "layers": {"family": 1.0, "work_school": 1.0, "community": 1.0},
            "info": {
                "phi_family": 0.1, "phi_work": 0.1, "phi_community": 0.1,
                "lambda_broadcast_base": 0.05,
                "lambda_broadcast_factor_after_day10": self.cfg.gov_lam_factor_default,
                "rho_info_decay": self.cfg.rho_info_decay_default
            },
            "noise": {"tau": 1.0},
        }
        # State and buffers
        self.state: Dict[str, Any] = {"wearing_prev": np.zeros(self.N, dtype=np.float64)}
        self.buffers: Dict[str, Any] = {
            "received_info": np.zeros((self.T, self.N), dtype=np.float64),
            "mem_info": np.zeros((self.T, self.N), dtype=np.float64),
            "prob_wear": np.zeros((self.T, self.N), dtype=np.float64),
            "wearing": np.zeros((self.T, self.N), dtype=np.float64),
        }
        # Modules (dependency order)
        self.modules: List[ModuleBase] = [
            InfoPropagationModule("info", self),
            DecisionModule("decision", self),
            StateUpdateModule("state_update", self),
        ]
        # Default initial previous wearing state: day 0 observed (if available)
        init_day_index = 0
        self.state["wearing_prev"] = wearing_obs[init_day_index, :].copy()
        # For evaluation, we will set observed slices externally

    def set_params(self, decision: Dict[str, Any], layers: Dict[str, float], info: Dict[str, float], noise: Dict[str, float]) -> None:
        """
        Set simulation parameters by sections.

        Args:
            decision: Decision rule parameters including demographics.
            layers: Layer weights dictionary.
            info: Information propagation parameters.
            noise: Noise/temperature parameters.

        Returns:
            None.
        """
        # Merge updates
        self.params["decision"].update(decision or {})
        self.params["layers"].update(layers or {})
        self.params["info"].update(info or {})
        self.params["noise"].update(noise or {})

    def run(self, start_day_idx: int, end_day_idx: int) -> Dict[str, Any]:
        """
        Run the simulation from start_day_idx (inclusive) to end_day_idx (exclusive).
        Initializes previous state from day start_day_idx if >0, else from day 0 observed.

        Args:
            start_day_idx: Start day index in [0, T-1].
            end_day_idx: End day index in (start_day_idx, T].

        Returns:
            Dict with per-day arrays produced: wearing, received_info, prob_wear.
        """
        if not (0 <= start_day_idx < end_day_idx <= self.T):
            raise ValueError(f"Invalid run window [{start_day_idx}, {end_day_idx}) for T={self.T}")
        # Initialize previous state from observed day start_day_idx - 1 if possible
        if start_day_idx > 0:
            self.state["wearing_prev"] = self.wearing_obs[start_day_idx - 1, :].copy()
        else:
            self.state["wearing_prev"] = self.wearing_obs[0, :].copy()
        # Reset buffers for this window
        self.buffers["received_info"][start_day_idx:end_day_idx, :] = 0.0
        self.buffers["mem_info"][start_day_idx:end_day_idx, :] = 0.0
        self.buffers["prob_wear"][start_day_idx:end_day_idx, :] = 0.0
        self.buffers["wearing"][start_day_idx:end_day_idx, :] = 0.0
        # Run modules per day
        for t in range(start_day_idx, end_day_idx):
            for m in self.modules:
                m.forward(t)
            # Commit: update previous wearing
            self.state["wearing_prev"] = self.buffers["wearing"][t, :].copy()
        return {
            "wearing": self.buffers["wearing"][start_day_idx:end_day_idx, :],
            "received_info": self.buffers["received_info"][start_day_idx:end_day_idx, :],
            "prob_wear": self.buffers["prob_wear"][start_day_idx:end_day_idx, :],
        }

    def evaluate(self, start_day_idx: int, end_day_idx: int) -> Dict[str, Any]:
        """
        Evaluate metrics comparing simulation buffers to observed data on given window.

        Args:
            start_day_idx: Start day index.
            end_day_idx: End day index.

        Returns:
            Metrics dictionary including RMSE, MAE, Brier, TransitionFit and daily rate arrays.
        """
        sim_states = self.buffers["wearing"][start_day_idx:end_day_idx, :]
        sim_probs = self.buffers["prob_wear"][start_day_idx:end_day_idx, :]
        obs_states = self.wearing_obs[start_day_idx:end_day_idx, :]
        # Aggregate rates
        sim_rates = sim_states.mean(axis=1)
        obs_rates = obs_states.mean(axis=1)
        rmse = math.sqrt(float(np.mean((sim_rates - obs_rates) ** 2)))
        mae = float(np.mean(np.abs(sim_rates - obs_rates)))
        brier = float(np.mean((sim_probs - obs_states) ** 2))
        # Transition fit
        if end_day_idx - start_day_idx > 1:
            prev_obs = self.wearing_obs[start_day_idx:end_day_idx - 1, :].flatten()
            curr_obs = self.wearing_obs[start_day_idx + 1:end_day_idx, :].flatten()
            prev_sim = sim_states[:-1, :].flatten()
            curr_sim = sim_states[1:, :].flatten()
            obs_p01 = np.mean((prev_obs == 0.0) & (curr_obs == 1.0))
            obs_p11 = np.mean((prev_obs == 1.0) & (curr_obs == 1.0))
            obs_p10 = np.mean((prev_obs == 1.0) & (curr_obs == 0.0))
            obs_p00 = np.mean((prev_obs == 0.0) & (curr_obs == 0.0))
            sim_p01 = np.mean((prev_sim == 0.0) & (curr_sim == 1.0))
            sim_p11 = np.mean((prev_sim == 1.0) & (curr_sim == 1.0))
            sim_p10 = np.mean((prev_sim == 1.0) & (curr_sim == 0.0))
            sim_p00 = np.mean((prev_sim == 0.0) & (curr_sim == 0.0))
            trans_err = float(np.mean([
                abs(obs_p01 - sim_p01),
                abs(obs_p11 - sim_p11),
                abs(obs_p10 - sim_p10),
                abs(obs_p00 - sim_p00),
            ]))
        else:
            trans_err = 0.0
        return {
            "RMSE_aggregate": rmse,
            "MAE_aggregate": mae,
            "Brier": brier,
            "TransitionFit": trans_err,
            "observed_daily_rates": obs_rates.tolist(),
            "predicted_daily_rates": sim_rates.tolist(),
        }

    def save_results(self, path: str, metrics: Dict[str, Any]) -> None:
        """
        Save simulation results and metrics.

        Args:
            path: Directory path to save results.
            metrics: Metrics dict from evaluate().

        Returns:
            None.
        """
        ensure_dir(path)
        # Save metrics
        save_json(metrics, os.path.join(path, "metrics.json"))
        # Save arrays as CSVs
        df_rates = pd.DataFrame({
            "day_index": list(range(len(metrics["observed_daily_rates"]))),
            "observed_rate": metrics["observed_daily_rates"],
            "predicted_rate": metrics["predicted_daily_rates"],
        })
        df_rates.to_csv(os.path.join(path, "daily_rates.csv"), index=False)

    def save_module_io(self, module_name: str, path: str) -> None:
        """
        Save I/O arrays produced by a single module.

        Args:
            module_name: Name of module.
            path: Destination path directory.

        Returns:
            None.
        """
        ensure_dir(path)
        if module_name == "info":
            pd.DataFrame(self.buffers["received_info"]).to_csv(os.path.join(path, "received_info.csv"), index=False)
            pd.DataFrame(self.buffers["mem_info"]).to_csv(os.path.join(path, "mem_info.csv"), index=False)
        elif module_name == "decision":
            pd.DataFrame(self.buffers["prob_wear"]).to_csv(os.path.join(path, "prob_wear.csv"), index=False)
        elif module_name == "state_update":
            pd.DataFrame(self.buffers["wearing"]).to_csv(os.path.join(path, "wearing.csv"), index=False)

    def save_all_io(self, root_dir: str) -> None:
        """
        Save I/O arrays for all modules.

        Args:
            root_dir: Root directory for module I/O.

        Returns:
            None.
        """
        ensure_dir(root_dir)
        for m in self.modules:
            self.save_module_io(m.name, os.path.join(root_dir, m.name))

    def visualize(self, start_day_idx: int, end_day_idx: int, out_png: Optional[str] = None) -> None:
        """
        Quick visualization of observed vs predicted daily rates.

        Args:
            start_day_idx: Start day index used for evaluation.
            end_day_idx: End day index used for evaluation.
            out_png: Optional file path to save the plot.

        Returns:
            None.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping visualization")
            return
        sim_states = self.buffers["wearing"][start_day_idx:end_day_idx, :]
        obs_states = self.wearing_obs[start_day_idx:end_day_idx, :]
        sim_rates = sim_states.mean(axis=1)
        obs_rates = obs_states.mean(axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(range(start_day_idx, end_day_idx), obs_rates, label="Observed", marker="o")
        plt.plot(range(start_day_idx, end_day_idx), sim_rates, label="Predicted", marker="x")
        plt.xlabel("Day index")
        plt.ylabel("Mask-wearing rate")
        plt.title("Observed vs Predicted Daily Adoption Rates")
        plt.grid(True)
        plt.legend()
        if out_png:
            ensure_dir(os.path.dirname(out_png))
            plt.savefig(out_png, bbox_inches="tight")
        else:
            plt.show()


# ========================
# Evaluator and Features
# ========================

def build_feature_matrix(
    wearing: np.ndarray,
    mem_info: np.ndarray,
    share_f_by_day: np.ndarray,
    share_w_by_day: np.ndarray,
    share_c_by_day: np.ndarray,
    risk: np.ndarray,
    age_oh: np.ndarray,
    occ_oh: np.ndarray,
    day_start: int,
    day_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build logistic regression feature matrix from micro-transitions.

    Feature order:
      0: intercept
      1: previous wearing state
      2: share_family
      3: share_work
      4: share_community
      5: risk_perception
      6: mem_info
      [7..7+A-1]: age one-hot columns
      [7+A..7+A+O-1]: occ one-hot columns

    Args:
        wearing: Observed wearing states T x N.
        mem_info: Memory of info T x N.
        share_f_by_day: Daily family share T x N computed from wearing[t-1].
        share_w_by_day: Daily work/school share T x N computed from wearing[t-1].
        share_c_by_day: Daily community share T x N computed from wearing[t-1].
        risk: Risk perception vector N.
        age_oh: Age one-hot matrix N x A.
        occ_oh: Occupation one-hot matrix N x O.
        day_start: First day index to build labels (uses t-1 for features).
        day_end: One past last day index.

    Returns:
        X design matrix (samples x features), y binary labels vector (samples,).
    """
    T, N = wearing.shape
    assert 0 <= day_start < day_end <= T
    rows = []
    labels = []
    for t in range(day_start, day_end):
        wear_prev = wearing[t - 1, :]
        share_f = share_f_by_day[t - 1, :]
        share_w = share_w_by_day[t - 1, :]
        share_c = share_c_by_day[t - 1, :]
        mem_t = mem_info[t, :]
        intercept = np.ones(N, dtype=np.float64)
        base = np.stack([intercept, wear_prev, share_f, share_w, share_c, risk, mem_t], axis=1)
        if age_oh.shape[1] > 0:
            base = np.concatenate([base, age_oh], axis=1)
        if occ_oh.shape[1] > 0:
            base = np.concatenate([base, occ_oh], axis=1)
        rows.append(base)
        labels.append(wearing[t, :])
    X = np.vstack(rows)
    y = np.concatenate(labels, axis=0)
    return X, y


def fit_logistic_l2(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1.0,
    max_iter: int = 400,
    lr: float = 0.1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fit L2-regularized logistic regression with Adam optimizer.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels vector (n_samples,).
        l2_reg: L2 regularization coefficient (intercept not regularized).
        max_iter: Max optimization iterations.
        lr: Learning rate for Adam.
        verbose: Whether to print progress.

    Returns:
        Weight vector (n_features,).
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=np.float64)
    reg_mask = np.ones(n_features, dtype=np.float64)
    reg_mask[0] = 0.0
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for it in range(1, max_iter + 1):
        z = X @ w
        p = sigmoid(z)
        grad = X.T @ (p - y) / n_samples + l2_reg * reg_mask * w / n_samples
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** it)
        v_hat = v / (1 - beta2 ** it)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        if verbose and (it % 50 == 0 or it == max_iter):
            nll = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) / n_samples
            reg_term = 0.5 * l2_reg * np.sum((reg_mask * w) ** 2) / n_samples
            total = nll + reg_term
            print(f"[fit_logistic_l2] iter={it:4d} nll={nll:.6f} reg={reg_term:.6f} total={total:.6f}")
    return w


def calibrate_info_params_simple(
    received_info: np.ndarray,
    share_f_by_day: np.ndarray,
    share_w_by_day: np.ndarray,
    share_c_by_day: np.ndarray,
    gov_intervention_day: int,
    default_factor: float,
) -> Tuple[float, float, float, float, float]:
    """
    Calibrate info propagation parameters using method-of-moments heuristic.

    Args:
        received_info: Observed info receipt T x N.
        share_f_by_day: Family share T x N (observed-based).
        share_w_by_day: Work share T x N.
        share_c_by_day: Community share T x N.
        gov_intervention_day: Day index when intervention starts.
        default_factor: Default factor for lambda broadcast after intervention if cannot estimate reliably.

    Returns:
        Tuple (phi_f, phi_w, phi_c, lambda_broadcast_base, lambda_broadcast_factor_after).
    """
    T, N = received_info.shape
    p_obs = received_info.mean(axis=1)
    sf = share_f_by_day.mean(axis=1)
    sw = share_w_by_day.mean(axis=1)
    sc = share_c_by_day.mean(axis=1)

    phi_f = 0.3
    phi_w = 0.2
    phi_c = 0.1

    pre_mask = np.arange(T) < gov_intervention_day
    post_mask = ~pre_mask
    if not pre_mask.any():
        pre_mask[:] = True
    p0 = float(p_obs[pre_mask].mean())
    s0 = float((phi_f * sf[pre_mask] + phi_w * sw[pre_mask] + phi_c * sc[pre_mask]).mean())
    lam_base = max(0.0, -math.log(max(1e-9, 1.0 - p0)) - s0)

    if post_mask.any():
        p1 = float(p_obs[post_mask].mean())
        s1 = float((phi_f * sf[post_mask] + phi_w * sw[post_mask] + phi_c * sc[post_mask]).mean())
        lam1 = max(0.0, -math.log(max(1e-9, 1.0 - p1)) - s1)
        lam_factor = (lam1 / lam_base) if lam_base > 1e-9 else default_factor
        lam_factor = max(1.0, min(5.0, lam_factor))
    else:
        lam_factor = default_factor
    lam_base = float(max(0.0, min(0.5, lam_base)))
    return phi_f, phi_w, phi_c, lam_base, lam_factor


# ==================================
# Calibration Interface and Classes
# ==================================

class Calibrator(ABC):
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """

    @abstractmethod
    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: str | None = None,
            params_adapter: ParamsAdapter | None = None) -> FittedParams:
        """
        Return FittedParams, fitted strictly on the training window.

        Args:
            bundle: Tuple with training bundle data needed by the calibrator.
            simulator: Simulation class or instance if required (not used by some calibrators).
            evaluator: Callable evaluate_params(simulator, fitted_params, window) -> metrics dict.
            train_window: Tuple (start_idx, end_idx) for training on micro transitions.
            seed: Global seed for reproducibility.
            budget: Iterations or trials budget for calibrator.
            artifacts_dir: Directory to save trial artifacts.
            params_adapter: Adapter to map params to simulation.

        Returns:
            FittedParams instance representing best parameters per objective.
        """
        pass


def evaluate_params(simulator_class, params: FittedParams, window) -> Dict[str, Any]:
    """
    Apply `params`, run a forward simulation on `window`, and return a metrics dict
    containing at least: 'RMSE_aggregate', 'MAE_aggregate', 'Brier',
    'TransitionFit' (with P01, P11, P10, P00).

    For simulation engines: use Simulation.run(start,end) + Simulation.evaluate(),
    read metrics from artifacts/results/, build dict with required keys.
    If micro-transitions unavailable, degrade gracefully or use placeholder values.
    """
    # Unpack window: we pass everything required
    (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg, window_range) = window
    start_idx, end_idx = window_range
    # Create simulation
    sim = simulator_class(
        wearing_obs=wearing,
        received_obs=received,
        neighbors=neighbors,
        risk=risk,
        age_oh=age_oh,
        occ_oh=occ_oh,
        age_cat_names=age_cat_names,
        occ_cat_names=occ_cat_names,
        cfg=cfg
    )
    # Apply parameters via default adapter
    adapter = DefaultParamsAdapter(
        defs_path=os.path.join(cfg.artifacts_dir, "parameter_definitions.json"),
        params_used_path=os.path.join(cfg.artifacts_dir, "parameters_used.json"),
    )
    adapter.apply(sim, params)
    # Run and evaluate
    sim.run(start_idx, end_idx)
    metrics = sim.evaluate(start_idx, end_idx)
    return metrics


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head from micro-transitions on days_train (L2 regularized; intercept not regularized).
    """

    def __init__(self, l2_reg: float = 1.0, max_iter: int = 400, learning_rate: float = 0.1, verbose: bool = True):
        """
        Initialize logit-head calibrator.

        Args:
            l2_reg: L2 regularization strength.
            max_iter: Max optimization iterations.
            learning_rate: Optimizer learning rate.
            verbose: Whether to print optimization logs.
        """
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: str | None = None,
            params_adapter: ParamsAdapter | None = None) -> FittedParams:
        """
        Fit logistic head on micro transitions using observed features including received_info memory.

        Args:
            bundle: Tuple (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg).
            simulator: Simulation class (not used).
            evaluator: Evaluator callable.
            train_window: (start_idx, end_idx) to build transitions.
            seed: Random seed.
            budget: Not used here (single fit).
            artifacts_dir: Directory to save artifacts.
            params_adapter: Optional params adapter (not used here).

        Returns:
            FittedParams.
        """
        (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg) = bundle
        train_start, train_end = train_window
        set_global_seed(seed)

        T, N = wearing.shape
        # Precompute neighbor shares per day (based on observed wearing)
        share_f_by_day = np.zeros_like(wearing)
        share_w_by_day = np.zeros_like(wearing)
        share_c_by_day = np.zeros_like(wearing)
        for t in range(T):
            s_prev = wearing[t, :]
            share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
            share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
            share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])

        # Use actual received_info to build memory
        rho = cfg.rho_info_decay_default
        mem_info = compute_mem_info(received, rho=rho)

        # Build feature matrix in the training window
        X, y = build_feature_matrix(
            wearing=wearing,
            mem_info=mem_info,
            share_f_by_day=share_f_by_day,
            share_w_by_day=share_w_by_day,
            share_c_by_day=share_c_by_day,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            day_start=train_start,
            day_end=train_end
        )

        if self.verbose:
            print(f"LogitHeadCalibrator: fitting logistic head with X={X.shape}, y={y.shape}, L2={self.l2_reg}")
        # Optimize
        beta = fit_logistic_l2(
            X, y, l2_reg=self.l2_reg, max_iter=self.max_iter, lr=self.learning_rate, verbose=self.verbose
        )
        # FIXED: Corrected feature mapping and included received_info
        # Mapping: 0 intercept, 1 prev, 2 share_f, 3 share_w, 4 share_c, 5 risk, 6 mem_info
        alpha = float(beta[0])
        gamma = float(beta[1])
        theta_f = float(beta[2])
        theta_w = float(beta[3])
        theta_c = float(beta[4])
        beta_r = float(beta[5]) if len(beta) > 5 else 0.0
        beta_i = float(beta[6]) if len(beta) > 6 else 0.0
        base_idx = 7
        n_age = age_oh.shape[1]
        n_occ = occ_oh.shape[1]
        age_effects = {age_cat_names[i]: float(beta[base_idx + i]) for i in range(n_age)} if n_age > 0 else {}
        occ_effects = {occ_cat_names[i]: float(beta[base_idx + n_age + i]) for i in range(n_occ)} if n_occ > 0 else {}

        # Normalize layer weights and create layer_weights dict
        w_f, w_w, w_c, _, _, _ = derive_layer_weights_and_betas(theta_f, theta_w, theta_c)
        layer_weights = {"family": w_f, "work_school": w_w, "community": w_c}

        # Calibrate info params from observed received_info
        phi_f, phi_w, phi_c, lam_base, lam_factor = calibrate_info_params_simple(
            received_info=received,
            share_f_by_day=share_f_by_day,
            share_w_by_day=share_w_by_day,
            share_c_by_day=share_c_by_day,
            gov_intervention_day=cfg.gov_intervention_day,
            default_factor=cfg.gov_lam_factor_default,
        )

        decision_weights = {
            "alpha": alpha,
            "gamma": gamma,
            "theta_f": theta_f,
            "theta_w": theta_w,
            "theta_c": theta_c,
            "beta_r": beta_r,
            "beta_i": beta_i,
            "age_effects": age_effects,
            "occ_effects": occ_effects,
        }
        info_params = {
            "phi_family": phi_f,
            "phi_work": phi_w,
            "phi_community": phi_c,
            "lambda_broadcast_base": lam_base,
            "lambda_broadcast_factor_after_day10": lam_factor,
            "rho_info_decay": cfg.rho_info_decay_default,
        }
        noise_params = {"tau": 1.0}
        fitted = FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            meta={
                "calibrator_name": "logit_head",
                "train_window": train_window,
                "seed": seed,
                "l2_reg": self.l2_reg,
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
            }
        )

        # Evaluate on training window for reporting
        window = (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg, train_window)
        metrics = evaluator(Simulation, fitted, window)
        if artifacts_dir:
            ensure_dir(artifacts_dir)
            save_json(fitted.to_dict(), os.path.join(artifacts_dir, "fitted_params.json"))
            save_json(metrics, os.path.join(artifacts_dir, "training_metrics.json"))
        if self.verbose:
            print(f"LogitHeadCalibrator: Training RMSE={metrics['RMSE_aggregate']:.4f} MAE={metrics['MAE_aggregate']:.4f}")
        return fitted


class RandomSearchCalibrator(Calibrator):
    """
    Random search calibrator over decision and info parameters within ranges.
    """

    def __init__(self, n_trials: int = 100, verbose: bool = True):
        """
        Initialize random search.

        Args:
            n_trials: Number of random trials.
            verbose: Print logs during search.
        """
        self.n_trials = n_trials
        self.verbose = verbose

    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: str | None = None,
            params_adapter: ParamsAdapter | None = None) -> FittedParams:
        """
        Run random search calibration.

        Args:
            bundle: Tuple (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg).
            simulator: Simulation class (ignored).
            evaluator: Evaluator callable.
            train_window: Training window indices.
            seed: Random seed.
            budget: Max trials to run (overrides n_trials if provided).
            artifacts_dir: Artifact base directory.
            params_adapter: Adapter (ignored here).

        Returns:
            Best FittedParams found.
        """
        (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg) = bundle
        train_start, train_end = train_window
        T, N = wearing.shape

        set_global_seed(seed)
        trials = int(budget) if budget is not None else self.n_trials

        # Precompute neighbor shares for info calibration baseline
        share_f_by_day = np.zeros_like(wearing)
        share_w_by_day = np.zeros_like(wearing)
        share_c_by_day = np.zeros_like(wearing)
        for t in range(T):
            s_prev = wearing[t, :]
            share_f_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["family"])
            share_w_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["work_school"])
            share_c_by_day[t, :] = compute_layer_neighbor_share(s_prev, neighbors["community"])

        # Baseline info params via heuristic
        phi_f0, phi_w0, phi_c0, lam_base0, lam_factor0 = calibrate_info_params_simple(
            received_info=received,
            share_f_by_day=share_f_by_day,
            share_w_by_day=share_w_by_day,
            share_c_by_day=share_c_by_day,
            gov_intervention_day=cfg.gov_intervention_day,
            default_factor=cfg.gov_lam_factor_default
        )

        best_score = float("inf")
        best_params: Optional[FittedParams] = None
        calib_dir = os.path.join(cfg.artifacts_dir, "calibration_random", timestamp()) if artifacts_dir is None else artifacts_dir
        ensure_dir(calib_dir)

        for trial in range(trials):
            trial_seed = seed + trial
            set_global_seed(trial_seed)
            decision_weights = {
                "alpha": np.random.uniform(-2.0, 2.0),
                "gamma": np.random.uniform(0.5, 3.0),
                "theta_f": np.random.uniform(0.0, 2.5),
                "theta_w": np.random.uniform(0.0, 2.5),
                "theta_c": np.random.uniform(0.0, 2.5),
                "beta_r": np.random.uniform(-1.5, 1.5),
                "beta_i": np.random.uniform(0.0, 2.0),
                "age_effects": {},
                "occ_effects": {},
            }
            w_f, w_w, w_c, _, _, _ = derive_layer_weights_and_betas(
                decision_weights["theta_f"], decision_weights["theta_w"], decision_weights["theta_c"]
            )
            layer_weights = {"family": w_f, "work_school": w_w, "community": w_c}
            info_params = {
                "phi_family": np.clip(np.random.normal(phi_f0, 0.1), 0.01, 0.6),
                "phi_work": np.clip(np.random.normal(phi_w0, 0.1), 0.01, 0.6),
                "phi_community": np.clip(np.random.normal(phi_c0, 0.1), 0.01, 0.6),
                "lambda_broadcast_base": np.clip(np.random.normal(lam_base0, 0.02), 0.0, 0.5),
                "lambda_broadcast_factor_after_day10": np.clip(np.random.normal(lam_factor0, 0.3), 1.0, 5.0),
                "rho_info_decay": np.clip(np.random.uniform(0.2, 0.9), 0.01, 0.99),
            }
            noise_params = {"tau": np.random.uniform(0.5, 2.0)}
            candidate = FittedParams(
                decision_weights=decision_weights, layer_weights=layer_weights, info_params=info_params, noise_params=noise_params,
                meta={"calibrator_name": "random_search", "trial": trial, "seed": trial_seed, "train_window": train_window}
            )
            window = (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg, train_window)
            metrics = evaluator(Simulation, candidate, window)
            score = metrics["RMSE_aggregate"]
            trial_dir = os.path.join(calib_dir, f"trial_{trial:04d}")
            ensure_dir(trial_dir)
            save_json(candidate.to_dict(), os.path.join(trial_dir, "params.json"))
            save_json(metrics, os.path.join(trial_dir, "metrics.json"))
            if self.verbose and trial % max(1, trials // 10) == 0:
                print(f"RandomSearchCalibrator: trial {trial}/{trials} RMSE={score:.4f} best={best_score:.4f}")
            if score < best_score:
                best_score = score
                best_params = candidate

        if best_params is None:
            # Fallback default
            best_params = FittedParams(
                decision_weights={"alpha": 0.0, "gamma": 1.0, "theta_f": 1.0, "theta_w": 1.0, "theta_c": 1.0, "beta_r": 0.0, "beta_i": 0.0,
                                  "age_effects": {}, "occ_effects": {}},
                layer_weights={"family": 1 / 3, "work_school": 1 / 3, "community": 1 / 3},
                info_params={"phi_family": phi_f0, "phi_work": phi_w0, "phi_community": phi_c0, "lambda_broadcast_base": lam_base0,
                             "lambda_broadcast_factor_after_day10": lam_factor0, "rho_info_decay": cfg.rho_info_decay_default},
                noise_params={"tau": 1.0},
                meta={"calibrator_name": "random_search", "train_window": train_window, "seed": seed}
            )
        # Save best
        best_dir = os.path.join(calib_dir, "best")
        ensure_dir(best_dir)
        save_json(best_params.to_dict(), os.path.join(best_dir, "fitted_params.json"))
        report = {
            "budget": trials,
            "best_rmse": float(best_score),
            "meta": {"seed": seed, "train_window": train_window}
        }
        save_json(report, os.path.join(calib_dir, "calibration_report.json"))
        return best_params


class SNPECalibrator(Calibrator):
    """
    True SBI using neural networks for Bayesian parameter inference.

    If dependencies are missing, falls back to RandomSearchCalibrator.
    """

    def __init__(self, n_simulations: int = 1000, k_observables: int = 5, flow_type: str = "maf", verbose: bool = True):
        """
        Initialize SNPE calibrator.

        Args:
            n_simulations: Number of simulations to generate training data.
            k_observables: Dimension of per-day observables (1 or 5).
            flow_type: 'maf' or 'nsf'.
            verbose: Verbosity flag.
        """
        self.n_simulations = n_simulations
        self.k_observables = k_observables
        self.flow_type = flow_type
        self.verbose = verbose

    def _define_prior(self) -> Dict[str, Tuple[float, float]]:
        """
        Define uniform prior bounds for parameters.

        Returns:
            Dict mapping parameter names to (low, high) bounds.
        """
        return {
            "alpha": (-3.0, 3.0),
            "gamma": (0.5, 3.0),
            "theta_f": (0.0, 3.0),
            "theta_w": (0.0, 3.0),
            "theta_c": (0.0, 3.0),
            "beta_r": (-2.0, 2.0),
            "beta_i": (0.0, 3.0),
            "phi_family": (0.01, 0.6),
            "phi_work": (0.01, 0.6),
            "phi_community": (0.01, 0.6),
            "lambda_broadcast_base": (0.0, 0.5),
            "lambda_broadcast_factor_after_day10": (1.0, 5.0),
            "rho_info_decay": (0.01, 0.99),
            "tau": (0.5, 2.0),
            # FIXED: Standardize demographic naming to 'age_cat_i'/'occ_cat_i'
            "age_cat_0": (-2.0, 2.0),
            "age_cat_1": (-2.0, 2.0),
            "age_cat_2": (-2.0, 2.0),
            "occ_cat_0": (-2.0, 2.0),
            "occ_cat_1": (-2.0, 2.0),
            "occ_cat_2": (-2.0, 2.0),
        }

    def _sample_prior(self, n: int, seed: int) -> Tuple[np.ndarray, List[str]]:
        """
        Sample parameters from prior.

        Args:
            n: Number of samples.
            seed: RNG seed.

        Returns:
            Tuple of samples array and parameter names.
        """
        set_global_seed(seed)
        bounds = self._define_prior()
        names = list(bounds.keys())
        samples = np.zeros((n, len(names)), dtype=np.float64)
        for i, name in enumerate(names):
            lo, hi = bounds[name]
            samples[:, i] = np.random.uniform(lo, hi, size=n)
        return samples, names

    def _params_from_vector(self, vec: np.ndarray, names: List[str], age_cat_names: List[str], occ_cat_names: List[str]) -> FittedParams:
        """
        Convert sampled vector to FittedParams.

        Args:
            vec: Parameter vector.
            names: Names list.
            age_cat_names: Actual age category names.
            occ_cat_names: Actual occupation category names.

        Returns:
            FittedParams object.
        """
        d = {n: float(vec[i]) for i, n in enumerate(names)}
        decision = {
            "alpha": d.get("alpha", 0.0),
            "gamma": d.get("gamma", 1.0),
            "theta_f": d.get("theta_f", 1.0),
            "theta_w": d.get("theta_w", 1.0),
            "theta_c": d.get("theta_c", 1.0),
            "beta_r": d.get("beta_r", 0.0),
            "beta_i": d.get("beta_i", 0.0),
            "age_effects": {},
            "occ_effects": {},
        }
        # FIXED: Use 'age_cat_i' and 'occ_cat_i' consistent with encode_demographics outputs
        for i, cat in enumerate(age_cat_names):
            decision["age_effects"][cat] = d.get(f"age_cat_{i}", 0.0)
        for i, cat in enumerate(occ_cat_names):
            decision["occ_effects"][cat] = d.get(f"occ_cat_{i}", 0.0)

        w_f, w_w, w_c, _, _, _ = derive_layer_weights_and_betas(decision["theta_f"], decision["theta_w"], decision["theta_c"])
        layers = {"family": w_f, "work_school": w_w, "community": w_c}
        info = {
            "phi_family": d.get("phi_family", 0.1),
            "phi_work": d.get("phi_work", 0.1),
            "phi_community": d.get("phi_community", 0.1),
            "lambda_broadcast_base": d.get("lambda_broadcast_base", 0.05),
            "lambda_broadcast_factor_after_day10": d.get("lambda_broadcast_factor_after_day10", 1.5),
            "rho_info_decay": d.get("rho_info_decay", 0.5),
        }
        noise = {"tau": d.get("tau", 1.0)}
        return FittedParams(decision_weights=decision, layer_weights=layers, info_params=info, noise_params=noise, meta={"calibrator_name": "snpe"})

    def _trajectory(self, sim: Simulation, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Create observable trajectory matrix for SNPE.

        Args:
            sim: Simulation instance with buffers populated.
            start_idx: Start day index.
            end_idx: End day index.

        Returns:
            Trajectory matrix T x K.
        """
        sim_states = sim.buffers["wearing"][start_idx:end_idx, :]
        T = sim_states.shape[0]
        if self.k_observables == 1:
            traj = np.mean(sim_states, axis=1).reshape(-1, 1)
            return traj
        else:
            traj = np.zeros((T, 5), dtype=np.float64)
            traj[:, 0] = np.mean(sim_states, axis=1)
            traj[0, 1:] = 0.0
            for t in range(1, T):
                prev = sim_states[t - 1, :]
                curr = sim_states[t, :]
                p01 = np.mean((prev == 0.0) & (curr == 1.0))
                p11 = np.mean((prev == 1.0) & (curr == 1.0))
                p10 = np.mean((prev == 1.0) & (curr == 0.0))
                p00 = np.mean((prev == 0.0) & (curr == 0.0))
                traj[t, 1:] = [p01, p11, p10, p00]
            return traj

    def fit(self, bundle, simulator, evaluator, train_window: Tuple[int, int], seed: int,
            budget: int = 100, artifacts_dir: str | None = None,
            params_adapter: ParamsAdapter | None = None) -> FittedParams:
        """
        Fit parameters via SNPE if available; otherwise fall back to RandomSearch.

        Args:
            bundle: Tuple (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg).
            simulator: Simulation class.
            evaluator: Evaluator callable.
            train_window: Training window.
            seed: RNG seed.
            budget: Unused (SNPE uses n_simulations).
            artifacts_dir: Artifact directory for SNPE.
            params_adapter: Params adapter (not used here).

        Returns:
            FittedParams estimated via SNPE or fallback.
        """
        (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg) = bundle
        train_start, train_end = train_window
        # Try import sbi & torch
        try:
            import torch
            from sbi import utils as sbi_utils
            from sbi.inference import NPE
        except Exception as e:
            if self.verbose:
                print(f"SNPECalibrator: SBI dependencies not available ({e}); falling back to RandomSearch")
            return RandomSearchCalibrator(n_trials=min(self.n_simulations, 200), verbose=self.verbose).fit(
                bundle, simulator, evaluator, train_window, seed, budget=min(self.n_simulations, 200), artifacts_dir=artifacts_dir, params_adapter=params_adapter
            )

        # Prepare observed trajectory from training window
        obs_sim = simulator(
            wearing_obs=wearing,
            received_obs=received,
            neighbors=neighbors,
            risk=risk,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            cfg=cfg
        )
        # Use observed wearing directly to compute observed trajectory vector
        T_train = train_end - train_start
        obs_traj = np.zeros((T_train, self.k_observables), dtype=np.float64)
        if self.k_observables == 1:
            obs_traj[:, 0] = np.mean(wearing[train_start:train_end, :], axis=1)
        else:
            obs_traj[:, 0] = np.mean(wearing[train_start:train_end, :], axis=1)
            obs_traj[0, 1:] = 0.0
            for t in range(1, T_train):
                prev = wearing[train_start + t - 1, :]
                curr = wearing[train_start + t, :]
                p01 = np.mean((prev == 0.0) & (curr == 1.0))
                p11 = np.mean((prev == 1.0) & (curr == 1.0))
                p10 = np.mean((prev == 1.0) & (curr == 0.0))
                p00 = np.mean((prev == 0.0) & (curr == 0.0))
                obs_traj[t, 1:] = [p01, p11, p10, p00]
        obs_vec = obs_traj.flatten()

        # Build prior and generate training data by simulation
        bounds = self._define_prior()
        names = list(bounds.keys())
        prior_min = torch.tensor([bounds[n][0] for n in names], dtype=torch.float32)
        prior_max = torch.tensor([bounds[n][1] for n in names], dtype=torch.float32)
        prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max)

        if self.verbose:
            print(f"SNPECalibrator: Generating {self.n_simulations} simulations for training data")

        theta_list = []
        x_list = []

        set_global_seed(seed)
        for i in range(self.n_simulations):
            # Sample from prior
            vec = np.array([np.random.uniform(bounds[n][0], bounds[n][1]) for n in names], dtype=np.float64)
            # Convert to params and run a simulation
            fitted = self._params_from_vector(vec, names, age_cat_names, occ_cat_names)
            sim = simulator(
                wearing_obs=wearing,
                received_obs=received,
                neighbors=neighbors,
                risk=risk,
                age_oh=age_oh,
                occ_oh=occ_oh,
                age_cat_names=age_cat_names,
                occ_cat_names=occ_cat_names,
                cfg=cfg
            )
            adapter = DefaultParamsAdapter(
                defs_path=os.path.join(cfg.artifacts_dir, "parameter_definitions.json"),
                params_used_path=os.path.join(cfg.artifacts_dir, "parameters_used.json"),
            )
            adapter.apply(sim, fitted)
            sim.run(train_start, train_end)
            traj = self._trajectory(sim, train_start, train_end)
            theta_list.append(vec.astype(np.float32))
            x_list.append(traj.astype(np.float32).flatten())
            if self.verbose and (i + 1) % max(1, self.n_simulations // 10) == 0:
                print(f"  SNPE: simulated {i + 1}/{self.n_simulations}")

        theta_tensor = torch.tensor(np.vstack(theta_list), dtype=torch.float32)
        x_tensor = torch.tensor(np.vstack(x_list), dtype=torch.float32)

        # Train NPE
        density_estimator = NPE(prior=prior, density_estimator=self.flow_type)
        density_estimator = density_estimator.append_simulations(theta_tensor, x_tensor).train(max_num_epochs=50, training_batch_size=128)
        posterior = density_estimator.build_posterior(density_estimator)

        # Sample from posterior conditioned on observed vector
        x_obs = torch.tensor(obs_vec.reshape(1, -1), dtype=torch.float32)
        post_samples = posterior.sample((min(500, self.n_simulations),), x=x_obs)
        post_mean = post_samples.mean(dim=0).detach().cpu().numpy()

        fitted_mean = self._params_from_vector(post_mean, names, age_cat_names, occ_cat_names)
        fitted_mean.meta.update({
            "calibrator_name": "snpe",
            "n_simulations": self.n_simulations,
            "k_observables": self.k_observables,
            "flow_type": self.flow_type,
            "seed": seed,
            "train_window": train_window,
        })

        # Evaluate and save artifacts
        window = (wearing, received, neighbors, risk, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg, train_window)
        metrics = evaluator(Simulation, fitted_mean, window)
        snpe_dir = os.path.join(cfg.artifacts_dir, "calibration_snpe", timestamp()) if artifacts_dir is None else artifacts_dir
        ensure_dir(snpe_dir)
        save_json(fitted_mean.to_dict(), os.path.join(snpe_dir, "fitted_params.json"))
        save_json(metrics, os.path.join(snpe_dir, "training_metrics.json"))
        return fitted_mean


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: str | None = None, **kwargs):
    """
    Get calibrator by name with optional configuration loaded from JSON.

    Args:
        name: Calibrator key name.
        config_path: Optional JSON config path to load kwargs.

    Returns:
        Calibrator instance.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    loaded_kwargs = {}
    if config_path and os.path.exists(config_path):
        try:
            loaded_kwargs = load_json(config_path)
        except Exception:
            loaded_kwargs = {}
    loaded_kwargs.update(kwargs or {})
    return CALIBRATOR_REGISTRY[name](**loaded_kwargs)


# ====================
# Workflow Orchestration
# ====================

def temporal_holdout(days: List[int]) -> Tuple[int, int]:
    """
    Temporal holdout: split days into train and validation.

    Uses last min(10, 20% of T) days for validation.

    Args:
        days: Sorted list of unique day indices.

    Returns:
        Tuple (train_end_idx, val_start_idx), val_end_idx equals T.
    """
    T = len(days)
    if T < 3:
        raise RuntimeError("Not enough days to perform temporal split.")
    val_len = max(1, min(10, int(math.ceil(0.2 * T))))
    train_end = T - val_len
    val_start = train_end
    if val_start <= 0:
        raise RuntimeError("No validation days available after temporal split.")
    return train_end, val_start


def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Args namespace with fields used across workflow.
    """
    parser = argparse.ArgumentParser(description="Multiplex social simulation with pluggable calibration.")
    parser.add_argument("--param-file", type=str, default=None, help="Path to parameters.json")
    parser.add_argument("--set", action="append", default=[], help="Override parameters via key=value (repeatable)")
    parser.add_argument("--calibrator", type=str, default="logit_head", choices=list(CALIBRATOR_REGISTRY.keys()), help="Calibrator type")
    parser.add_argument("--budget", type=int, default=100, help="Calibration budget (iterations or trials)")
    parser.add_argument("--calib-window", type=str, default=None, help="Training window 'start:end' (indices), optional")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts directory")
    # Align with其他版本：支持双重蒙特卡洛与测试窗输出
    parser.add_argument("--double-mc", action="store_true", help="Enable Double Monte Carlo on test window")
    parser.add_argument("--mc-M", type=int, default=50, help="Number of parameter samples for Double Monte Carlo")
    parser.add_argument("--mc-K", type=int, default=20, help="Number of runs per parameter sample")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides of form key=value into a dictionary (flat keys).

    Args:
        config: Original config dict.
        overrides: List of 'key=value' strings.

    Returns:
        Updated config dict.
    """
    result = dict(config)
    for item in overrides:
        if "=" not in item:
            print(f"Warning: ignoring malformed override '{item}'")
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Try to parse val into number or bool
        parsed: Any = val
        if val.lower() in ["true", "false"]:
            parsed = val.lower() == "true"
        else:
            try:
                if "." in val:
                    parsed = float(val)
                else:
                    parsed = int(val)
            except Exception:
                parsed = val
        result[key] = parsed
    return result


def dict_to_cfg(params: Dict[str, Any]) -> SimulationConfig:
    """
    Convert a parameter dict to SimulationConfig.

    Args:
        params: Dictionary with keys corresponding to SimulationConfig fields.

    Returns:
        SimulationConfig instance.
    """
    cfg = SimulationConfig()
    for k in vars(cfg).keys():
        if k in params:
            setattr(cfg, k, params[k])
    return cfg


def main() -> None:
    """
    Orchestrate full workflow:
    - Parse CLI
    - Load data
    - Build network and features
    - Temporal holdout
    - Calibrate parameters
    - Evaluate on validation
    - Forecast next days
    - Save results and artifacts
    """
    args = parse_cli()
    # Load parameter file or build defaults
    params_file = args.param_file
    base_params: Dict[str, Any] = {}
    if params_file and os.path.exists(params_file):
        try:
            base_params = load_json(params_file)
        except Exception as e:
            print(f"Warning: failed to load param file {params_file}: {e}")
    # CLI overrides
    merged_params = apply_overrides(base_params, args.__dict__.get("set", []))
    # Build config
    cfg = dict_to_cfg(merged_params)
    if args.verbose:
        cfg.verbose = True
    set_global_seed(cfg.seed)
    # Prepare artifacts/output
    data_dir = resolve_data_dir()
    out_dir = os.path.join(data_dir, cfg.output_dir)
    ensure_dir(out_dir)
    artifacts_dir = args.artifacts_dir or os.path.join(data_dir, cfg.artifacts_dir, timestamp())
    ensure_dir(artifacts_dir)

    # Load data
    agents_path = os.path.join(data_dir, "agent_attributes.csv")
    social_path = os.path.join(data_dir, "social_network.json")
    train_path = os.path.join(data_dir, "train_data.csv")

    try:
        agents_df = load_agent_attributes(agents_path)
        social_raw = load_social_network(social_path)
        train_df = load_train_data(train_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Align IDs
    common_ids, id2idx, agents_df_f, social_f, train_df_f = align_ids(agents_df, social_raw, train_df)
    n_agents = len(common_ids)

    # Build network
    neighbors = build_multiplex_adjacency(social_f, id2idx, n_agents)

    # Pivot time series
    wearing, received, days = pivot_states(train_df_f, id2idx)
    T = wearing.shape[0]

    # Risk perception vector
    risk_perception = agents_df_f.set_index("agent_id").loc[common_ids]["risk_perception"].to_numpy(dtype=np.float64)

    # Demographics encoding
    age_oh, age_cat_names, occ_oh, occ_cat_names = encode_demographics(agents_df_f, common_ids)

    # Temporal holdout
    train_end_idx, val_start_idx = temporal_holdout(days)
    val_end_idx = T
    train_window = (1, train_end_idx)  # use transitions from day 0..train_end-1 to predict 1..train_end
    val_window = (val_start_idx, val_end_idx)

    # Prepare bundle for calibrator
    # FIXED: evaluator and calibrators now pass/use age_cat_names and occ_cat_names dynamically
    bundle = (wearing, received, neighbors, risk_perception, age_oh, occ_oh, age_cat_names, occ_cat_names, cfg)

    # Get calibrator
    calibrator = get_calibrator(args.calibrator)

    # Fit parameters
    print(f"Calibration using {args.calibrator}...")
    fitted_params = calibrator.fit(
        bundle=bundle,
        simulator=Simulation,
        evaluator=evaluate_params,
        train_window=train_window,
        seed=cfg.seed,
        budget=args.budget,
        artifacts_dir=os.path.join(artifacts_dir, f"calibration_{args.calibrator}"),
        params_adapter=None
    )

    # Evaluate on validation window with K-run averaging
    print("Evaluating on validation window...")
    k_runs = cfg.k_runs
    rmse_list = []
    mae_list = []
    brier_list = []
    trans_list = []
    daily_runs = []

    for r in range(k_runs):
        set_global_seed(cfg.seed + r)
        sim = Simulation(
            wearing_obs=wearing,
            received_obs=received,
            neighbors=neighbors,
            risk=risk_perception,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            cfg=cfg
        )
        adapter = DefaultParamsAdapter(
            defs_path=os.path.join(artifacts_dir, "parameter_definitions.json"),
            params_used_path=os.path.join(artifacts_dir, f"parameters_used_run{r:03d}.json"),
        )
        adapter.apply(sim, fitted_params)
        sim.run(val_window[0], val_window[1])
        metrics = sim.evaluate(val_window[0], val_window[1])
        rmse_list.append(metrics["RMSE_aggregate"])
        mae_list.append(metrics["MAE_aggregate"])
        brier_list.append(metrics["Brier"])
        trans_list.append(metrics["TransitionFit"])
        daily_runs.append(metrics["predicted_daily_rates"])

    def mean_ci(x: List[float]) -> Tuple[float, float]:
        arr = np.array(x, dtype=np.float64)
        mean = float(arr.mean())
        ci = float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, ci

    rmse_mean, rmse_ci = mean_ci(rmse_list)
    mae_mean, mae_ci = mean_ci(mae_list)
    brier_mean, brier_ci = mean_ci(brier_list)
    trans_mean, trans_ci = mean_ci(trans_list)
    daily_runs_arr = np.array(daily_runs, dtype=np.float64)
    daily_mean = daily_runs_arr.mean(axis=0).tolist()
    daily_ci = (1.96 * daily_runs_arr.std(axis=0, ddof=1) / math.sqrt(k_runs)).tolist() if k_runs > 1 else [0.0] * len(daily_mean)
    obs_rates_val = wearing[val_window[0]:val_window[1], :].mean(axis=1).tolist()

    validation_metrics = {
        "RMSE_aggregate_mean": rmse_mean,
        "RMSE_aggregate_CI95": rmse_ci,
        "MAE_aggregate_mean": mae_mean,
        "MAE_aggregate_CI95": mae_ci,
        "Brier_mean": brier_mean,
        "Brier_CI95": brier_ci,
        "TransitionFit_mean": trans_mean,
        "TransitionFit_CI95": trans_ci,
        "observed_daily_rates": obs_rates_val,
        "predicted_daily_rates_mean": daily_mean,
        "predicted_daily_rates_CI95": daily_ci,
        "k_runs": k_runs,
        "val_window": [val_window[0], val_window[1]],
    }

    # Forecast next days starting from last observed day in validation
    print("Forecasting future days...")
    forecast_days = cfg.forecast_days
    forecast_runs = []
    last_idx = val_window[1] - 1
    for r in range(k_runs):
        set_global_seed(cfg.seed + 1000 + r)
        sim = Simulation(
            wearing_obs=wearing,
            received_obs=received,
            neighbors=neighbors,
            risk=risk_perception,
            age_oh=age_oh,
            occ_oh=occ_oh,
            age_cat_names=age_cat_names,
            occ_cat_names=occ_cat_names,
            cfg=cfg
        )
        # Initialize prev state to last day of validation
        sim.state["wearing_prev"] = wearing[last_idx, :].copy()
        adapter = DefaultParamsAdapter(
            defs_path=os.path.join(artifacts_dir, "parameter_definitions.json"),
            params_used_path=os.path.join(artifacts_dir, f"parameters_used_forecast{r:03d}.json"),
        )
        adapter.apply(sim, fitted_params)
        # Run from last_idx to last_idx + forecast_days
        start_idx = last_idx
        end_idx = min(sim.T, last_idx + forecast_days)
        result = sim.run(start_idx, end_idx)
        forecast_runs.append(result["wearing"].mean(axis=1).tolist())

    forecast_arr = np.array(forecast_runs, dtype=np.float64)
    fore_mean = forecast_arr.mean(axis=0).tolist()
    fore_ci = (1.96 * forecast_arr.std(axis=0, ddof=1) / math.sqrt(k_runs)).tolist() if k_runs > 1 else [0.0] * len(fore_mean)
    forecast = {
        "start_index": last_idx + 1,
        "days": list(range(last_idx + 1, last_idx + 1 + len(fore_mean))),
        "forecast_mean_rates": fore_mean,
        "forecast_CI95": fore_ci,
        "k_runs": k_runs,
    }

    # Save overall outputs
    print("Saving outputs...")
    save_json({"config": asdict(cfg), "params_file": params_file, "overrides": args.__dict__.get("set", [])},
              os.path.join(out_dir, "config.json"))
    save_json(fitted_params.to_dict(), os.path.join(out_dir, "calibrated_parameters.json"))
    save_json(validation_metrics, os.path.join(out_dir, "validation_metrics.json"))
    save_json(forecast, os.path.join(out_dir, "forecast.json"))

    # Save validation daily rates CSV
    df_val = pd.DataFrame({
        "day_index": list(range(val_window[0], val_window[1])),
        "observed_rate": validation_metrics["observed_daily_rates"],
        "predicted_rate_mean": validation_metrics["predicted_daily_rates_mean"],
        "predicted_rate_CI95": validation_metrics["predicted_daily_rates_CI95"],
    })
    df_val.to_csv(os.path.join(out_dir, "validation_daily_rates.csv"), index=False)

    # Save forecast CSV
    df_fore = pd.DataFrame({
        "day_index": forecast["days"],
        "forecast_mean_rate": forecast["forecast_mean_rates"],
        "forecast_CI95": forecast["forecast_CI95"],
    })
    df_fore.to_csv(os.path.join(out_dir, "forecast_daily_rates.csv"), index=False)

    # -------------------------------
    # Optional: Double Monte Carlo on test window (30-39) for alignment
    # -------------------------------
    def _perturb_params_gaussian(base: FittedParams, scale: float, rng: np.random.RandomState) -> FittedParams:
        def jn(x, s=scale):
            try:
                return float(x) + float(rng.normal(0.0, s))
            except Exception:
                return x
        # decision
        dw = dict(base.decision_weights) if isinstance(base.decision_weights, dict) else {}
        for k in ["alpha", "gamma", "theta_f", "theta_w", "theta_c", "beta_r", "beta_i"]:
            if k in dw:
                dw[k] = jn(dw[k])
        if "age_effects" in dw and isinstance(dw["age_effects"], dict):
            ae = dict(dw["age_effects"])
            for k in ae.keys():
                ae[k] = jn(ae[k])
            dw["age_effects"] = ae
        if "occ_effects" in dw and isinstance(dw["occ_effects"], dict):
            oe = dict(dw["occ_effects"])
            for k in oe.keys():
                oe[k] = jn(oe[k])
            dw["occ_effects"] = oe
        # info
        ip = dict(base.info_params) if isinstance(base.info_params, dict) else {}
        for k in ["phi_family", "phi_work", "phi_community", "lambda_broadcast_base", "lambda_broadcast_factor_after_day10", "rho_info_decay"]:
            if k in ip:
                ip[k] = max(0.0, jn(ip[k], scale * 0.5))
        # noise
        nz = dict(base.noise_params) if isinstance(base.noise_params, dict) else {}
        if "tau" in nz:
            nz["tau"] = max(0.1, jn(nz["tau"], scale * 0.2))
        # layers normalize
        lw = dict(base.layer_weights) if isinstance(base.layer_weights, dict) else {"family": 1.0, "work_school": 1.0, "community": 1.0}
        for k in ["family", "work_school", "community"]:
            if k in lw:
                lw[k] = max(0.0, jn(lw[k], scale * 0.05))
        s = sum(lw.values()) or 1.0
        for k in lw:
            lw[k] = float(lw[k]) / s
        return FittedParams(decision_weights=dw, layer_weights=lw, info_params=ip, noise_params=nz, module_params=base.module_params, engine_type=base.engine_type, meta=dict(base.meta))

    if args.double_mc:
        # Prefer test_data.csv if available; otherwise, attempt to use days 30..39 from training set
        test_path = os.path.join(data_dir, "test_data.csv")
        # Build a fresh simulation for either source
        if os.path.exists(test_path):
            try:
                test_df = pd.read_csv(test_path)
                # Align IDs with training selection
                test_df_f = test_df[test_df["agent_id"].isin(common_ids)].copy()
                wearing_test, received_test, days_test = pivot_states(test_df_f, id2idx)
                sim_test = Simulation(
                    wearing_obs=wearing_test,
                    received_obs=received_test,
                    neighbors=build_multiplex_adjacency(social_f, id2idx, n_agents),
                    risk=risk_perception,
                    age_oh=age_oh,
                    occ_oh=occ_oh,
                    age_cat_names=age_cat_names,
                    occ_cat_names=occ_cat_names,
                    cfg=cfg
                )
                t_start, t_end = 0, len(days_test)
                test_days = days_test
            except Exception as e:
                print(f"Warning: failed to load test_data.csv ({e}); falling back to training window 30–39 if available.")
                sim_test = Simulation(
                    wearing_obs=wearing,
                    received_obs=received,
                    neighbors=neighbors,
                    risk=risk_perception,
                    age_oh=age_oh,
                    occ_oh=occ_oh,
                    age_cat_names=age_cat_names,
                    occ_cat_names=occ_cat_names,
                    cfg=cfg
                )
                t_start, t_end = (30, min(40, T)) if T > 30 else (max(0, T - 10), T)
                test_days = days[t_start:t_end]
        else:
            sim_test = Simulation(
                wearing_obs=wearing,
                received_obs=received,
                neighbors=neighbors,
                risk=risk_perception,
                age_oh=age_oh,
                occ_oh=occ_oh,
                age_cat_names=age_cat_names,
                occ_cat_names=occ_cat_names,
                cfg=cfg
            )
            t_start, t_end = (30, min(40, T)) if T > 30 else (max(0, T - 10), T)
            test_days = days[t_start:t_end]

        M = max(1, int(args.mc_M))
        K = max(1, int(args.mc_K))
        rng = np.random.RandomState(cfg.seed)
        per_m_means = []
        for m in range(M):
            sampled = _perturb_params_gaussian(fitted_params, scale=0.1, rng=rng)
            # Apply once then average K runs
            run_rates = []
            for k in range(K):
                set_global_seed(cfg.seed + m * K + k)
                # Fresh sim each run to reset buffers/state
                sim_k = Simulation(
                    wearing_obs=sim_test.wearing_obs,
                    received_obs=sim_test.received_obs,
                    neighbors=sim_test.neighbors,
                    risk=sim_test.risk,
                    age_oh=sim_test.age_oh,
                    occ_oh=sim_test.occ_oh,
                    age_cat_names=sim_test.age_cat_names,
                    occ_cat_names=sim_test.occ_cat_names,
                    cfg=cfg
                )
                adapter_k = DefaultParamsAdapter(
                    defs_path=os.path.join(artifacts_dir, "parameter_definitions.json"),
                    params_used_path=os.path.join(artifacts_dir, f"parameters_used_dmc_m{m:03d}_k{k:03d}.json"),
                )
                adapter_k.apply(sim_k, sampled)
                sim_k.run(t_start, t_end)
                run_rates.append(sim_k.buffers["wearing"][t_start:t_end, :].mean(axis=1))
            per_m_means.append(np.mean(np.stack(run_rates, axis=0), axis=0))
        per_m = np.stack(per_m_means, axis=0)  # M x T_run
        mean_rates = per_m.mean(axis=0)
        std_rates = per_m.std(axis=0, ddof=1) if M > 1 else np.zeros_like(mean_rates)
        ci95 = 1.96 * std_rates / math.sqrt(M) if M > 1 else np.zeros_like(mean_rates)
        obs_rates = sim_test.wearing_obs[t_start:t_end, :].mean(axis=1)
        rmse_mean = float(math.sqrt(np.mean((mean_rates - obs_rates) ** 2)))
        mae_mean = float(np.mean(np.abs(mean_rates - obs_rates)))
        # Save files aligned命名
        results_dir = out_dir  # 与前两版保持在 outputs 目录
        try:
            import pandas as _pd
            _pd.DataFrame({
                "day": test_days,
                "observed_rate": obs_rates,
                "predicted_rate_mean": mean_rates,
                "predicted_rate_CI95": ci95,
            }).to_csv(os.path.join(results_dir, "daily_rates_double_mc.csv"), index=False)
        except Exception:
            pass
        save_json({
            "RMSE_aggregate_mean": rmse_mean,
            "MAE_aggregate_mean": mae_mean,
            "observed_daily_rates": obs_rates.tolist(),
            "predicted_daily_rates_mean": mean_rates.tolist(),
            "predicted_daily_rates_CI95": ci95.tolist(),
            "M": M,
            "K": K,
            "test_window": [int(test_days[0]) if len(test_days)>0 else t_start, int(test_days[-1]) if len(test_days)>0 else (t_end-1)]
        }, os.path.join(results_dir, "double_mc_test_metrics.json"))

    if cfg.verbose:
        print("Calibration and evaluation complete.")
        print(f"Artifacts saved to {artifacts_dir}")
        print(f"Outputs saved to {out_dir}")


# Execute main for both direct execution and sandbox wrapper invocation
main()