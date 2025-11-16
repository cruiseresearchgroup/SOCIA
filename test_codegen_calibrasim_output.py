import os
import json
import math
import random
import argparse
import time
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
import copy

import numpy as np
import pandas as pd

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# FIXED: No previous feedback; initial implementation


# =========================
# Environment Path Handling
# =========================

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/mask_adoption_data")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)


# ==================
# Utility and Logger
# ==================

def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating any necessary parent directories.

    Args:
        path: The target directory path.

    Returns:
        None
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    pass


def log(msg: str) -> None:
    """
    Lightweight logger for console output with timestamp.

    Args:
        msg: Message to print.

    Returns:
        None
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    pass


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically stable sigmoid function.

    Args:
        x: Input scalar or array.

    Returns:
        Sigmoid(x).
    """
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def safe_json_dump(path: str, data: Dict[str, Any]) -> None:
    """
    Safely dump JSON to file with directory creation.

    Args:
        path: Target JSON filename.
        data: Dictionary to serialize.

    Returns:
        None
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        ensure_dir(dirpath)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    pass


def parse_bool(x: Any) -> bool:
    """
    Parse a flexible boolean value.

    Args:
        x: Input value (string, bool, int).

    Returns:
        Boolean interpretation.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
    return False


def parse_number(x: str) -> Union[float, int, str]:
    """
    Parse string to int or float when possible.

    Args:
        x: Numeric string.

    Returns:
        Parsed number (int if appropriate else float), or original string if parsing fails.
    """
    try:
        if "." in x or "e" in x.lower():
            return float(x)
        return int(x)
    except Exception:
        try:
            return float(x)
        except Exception:
            return x
    pass


# ========================
# Data Loading and Parsing
# ========================

def load_agent_attributes(agent_file: str) -> pd.DataFrame:
    """
    Load agent attributes CSV. If file is missing, return an empty DataFrame.

    Args:
        agent_file: Path to agent_attributes.csv

    Returns:
        DataFrame with columns: agent_id, age_group, occupation, risk_perception
    """
    if not os.path.exists(agent_file):
        log(f"WARNING: Agent file not found at {agent_file}. Proceeding with synthetic population.")
        return pd.DataFrame()
    df = pd.read_csv(agent_file)
    if "agent_id" not in df.columns:
        raise ValueError("agent_attributes.csv must include 'agent_id' column")
    df["agent_id"] = df["agent_id"].astype(int)
    if "risk_perception" in df.columns:
        df["risk_perception"] = pd.to_numeric(df["risk_perception"], errors="coerce").fillna(0.5).clip(0, 1)
    else:
        df["risk_perception"] = 0.5
    if "age_group" not in df.columns:
        df["age_group"] = "Unknown"
    if "occupation" not in df.columns:
        df["occupation"] = "Unknown"
    return df


def load_social_network(network_file: str) -> Dict[int, Dict[str, List[int]]]:
    """
    Load multiplex social network from JSON. Deduplicate and symmetrize per layer.

    Args:
        network_file: Path to social_network.json

    Returns:
        Dictionary keyed by agent_id with fields 'family','work_school','community','all'
    """
    if not os.path.exists(network_file):
        log(f"WARNING: Network file not found at {network_file}. Proceeding with synthetic network.")
        return {}
    with open(network_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    net: Dict[int, Dict[str, List[int]]] = {}
    for k, v in raw.items():
        i = int(k)
        fam = list(sorted(set(int(x) for x in v.get("family", []))))
        work = list(sorted(set(int(x) for x in v.get("work_school", []))))
        comm = list(sorted(set(int(x) for x in v.get("community", []))))
        all_list = list(sorted(set(fam + work + comm)))
        net[i] = {
            "family": fam,
            "work_school": work,
            "community": comm,
            "all": all_list,
        }
    # Symmetrize
    for i, layers in net.items():
        for layer in ["family", "work_school", "community"]:
            for j in list(layers[layer]):
                if j not in net:
                    continue
                if i not in net[j][layer]:
                    net[j][layer].append(i)
                if j not in net[j]["all"]:
                    net[j]["all"].append(i)
            # Deduplicate
            layers[layer] = list(sorted(set(layers[layer])))
        layers["all"] = list(sorted(set(layers["family"] + layers["work_school"] + layers["community"])))
    return net


def load_train_data(train_file: str) -> pd.DataFrame:
    """
    Load time series train data CSV.

    Args:
        train_file: Path to train_data.csv

    Returns:
        DataFrame with columns: day, agent_id, wearing_mask (bool), received_info (bool)
    """
    if not os.path.exists(train_file):
        log(f"WARNING: Train data file not found at {train_file}. Proceeding without ground-truth time series.")
        return pd.DataFrame()
    df = pd.read_csv(train_file)
    if "day" not in df.columns or "agent_id" not in df.columns:
        raise ValueError("train_data.csv must include 'day' and 'agent_id' columns")
    df["day"] = df["day"].astype(int)
    df["agent_id"] = df["agent_id"].astype(int)
    for col in ["wearing_mask", "received_info"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_bool)
        else:
            df[col] = False
    return df


# ====================
# Parameter Management
# ====================

class ParameterRegistry:
    """
    Registry for simulation parameters with definitions, values, bounds, and module ownership.
    Provides set/get and override application with frozen checks.

    Attributes:
        definitions: Parameter definitions loaded from model plan or parameter_definitions.json
        values: Current parameter key->value mapping
        module_params_index: module_name -> list of parameter keys
        frozen_flags: key -> bool frozen
    """

    def __init__(self, plan: Dict[str, Any], definitions_path: str = "parameter_definitions.json") -> None:
        """
        Initialize the registry.

        Args:
            plan: Model plan dictionary providing parameter defaults and bounds.
            definitions_path: Path to save/load parameter definitions.

        Returns:
            None
        """
        self.plan = plan
        self.definitions_path = definitions_path
        self.definitions: Dict[str, Dict[str, Any]] = {}
        self.values: Dict[str, Any] = {}
        self.module_params_index: Dict[str, List[str]] = defaultdict(list)
        self.frozen_flags: Dict[str, bool] = {}
        self.load_or_build_definitions()
        pass

    def load_or_build_definitions(self) -> None:
        """
        Load parameter definitions from file if available; otherwise build from plan.

        Returns:
            None
        """
        if os.path.exists(self.definitions_path):
            with open(self.definitions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            defs_list = data.get("parameters", [])
        else:
            defs_list = self.plan.get("parameters", [])
            safe_json_dump(self.definitions_path, {"parameters": defs_list})
        for p in defs_list:
            key = p.get("key")
            if not key:
                continue
            self.definitions[key] = p
            owner = p.get("owner_module", "global")
            self.module_params_index[owner].append(key)
            self.frozen_flags[key] = parse_bool(p.get("frozen", False))
            # Initialize default
            self.values[key] = p.get("default")
        pass

    def load_param_values(self, param_file: str | None) -> None:
        """
        Load parameter values from external JSON file.

        Args:
            param_file: Path to parameters.json

        Returns:
            None
        """
        if not param_file or not os.path.exists(param_file):
            log(f"WARNING: Parameter file not found: {param_file}; using defaults from plan.")
            return
        with open(param_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k not in self.values:
                log(f"WARNING: Unknown parameter key in param file: {k}; ignoring.")
                continue
            self.values[k] = v
        pass

    def apply_overrides(self, overrides: List[str]) -> Dict[str, str]:
        """
        Apply CLI overrides as key=value pairs. Frozen params are ignored with warnings.

        Args:
            overrides: List of "key=value" strings.

        Returns:
            Dict mapping ignored frozen keys to messages.
        """
        ignored: Dict[str, str] = {}
        for item in overrides or []:
            if "=" not in item:
                log(f"WARNING: Invalid override format '{item}', expected key=value.")
                continue
            key, val = item.split("=", 1)
            key = key.strip()
            if key not in self.values:
                log(f"WARNING: Unknown parameter override key '{key}'; ignoring.")
                continue
            if self.frozen_flags.get(key, False):
                msg = f"Override for frozen parameter '{key}' ignored."
                log(f"WARNING: {msg}")
                ignored[key] = msg
                continue
            v_def = self.definitions.get(key, {})
            dtype = v_def.get("dtype", "float")
            parsed: Any = val
            if dtype == "int":
                try:
                    parsed = int(float(val))
                except Exception:
                    parsed = self.values[key]
            elif dtype == "float":
                try:
                    parsed = float(val)
                except Exception:
                    parsed = self.values[key]
            elif dtype == "bool":
                parsed = parse_bool(val)
            elif dtype == "categorical":
                parsed = val
            else:
                # fallback
                parsed = parse_number(val)
            # Bounds check
            bounds = v_def.get("bounds")
            if isinstance(parsed, (int, float)) and bounds and bounds.get("low") is not None and bounds.get("high") is not None:
                lo, hi = bounds["low"], bounds["high"]
                if parsed < lo or parsed > hi:
                    log(f"WARNING: Value {parsed} for '{key}' out of bounds [{lo}, {hi}]; clipping.")
                    parsed = max(lo, min(hi, parsed))
            self.values[key] = parsed
        return ignored

    def set_params(self, module: str, **kwargs) -> None:
        """
        Set parameters for a specific module, with frozen enforcement and bounds.

        Args:
            module: Owner module name.
            **kwargs: key-value pairs.

        Returns:
            None
        """
        for k, v in kwargs.items():
            if k not in self.values:
                log(f"WARNING: set_params unknown key '{k}' for module '{module}'; ignoring.")
                continue
            owner = self.definitions.get(k, {}).get("owner_module", "global")
            if owner != module:
                log(f"WARNING: Parameter '{k}' does not belong to module '{module}' (owner is '{owner}'); setting anyway.")
            if self.frozen_flags.get(k, False):
                log(f"WARNING: set_params ignored for frozen parameter '{k}'.")
                continue
            # Type align
            dtype = self.definitions.get(k, {}).get("dtype", "float")
            if dtype == "int":
                v = int(v)
            elif dtype == "float":
                v = float(v)
            elif dtype == "bool":
                v = parse_bool(v)
            self.values[k] = v
        pass

    def get_params(self, module: str | None = None) -> Dict[str, Any]:
        """
        Get parameters, optionally filtered by owner module.

        Args:
            module: Module name or None for all.

        Returns:
            Dictionary of parameter key->value.
        """
        if not module or module == "all":
            return dict(self.values)
        keys = self.module_params_index.get(module, [])
        return {k: self.values[k] for k in keys if k in self.values}

    def save_used(self, path: str = "parameters_used.json") -> None:
        """
        Save the final parameters actually used.

        Args:
            path: Output JSON filename.

        Returns:
            None
        """
        safe_json_dump(path, self.values)
        pass


# ===================================
# Calibration Interfaces and Classes
# ===================================

@dataclass
class FittedParams:
    """
    Container for all parameters needed by the simulator.

    Attributes:
        decision_weights: Dictionary of decision head weights (e.g., logistic regression coefficients).
        layer_weights: Influence weights per network layer.
        info_params: Information diffusion related parameters.
        noise_params: Noise-related parameters (e.g., temperature).
        module_params: Additional module-specific params.
        engine_type: String identifier for engine compatibility.
        meta: Metadata dictionary for calibration run details.
    """
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert FittedParams to dictionary.

        Returns:
            Dictionary representation.
        """
        return asdict(self)


class ParamsAdapter(ABC):
    """
    Adapts FittedParams to simulation parameter system.
    """

    @abstractmethod
    def apply(self, simulation, params: FittedParams) -> None:
        """
        Apply params via simulation parameter system: set_params() + parameters_used.json

        Args:
            simulation: Simulation object.
            params: FittedParams to apply.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def capture(self, simulation) -> FittedParams:
        """
        Capture current effective params from simulation into FittedParams.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams snapshot of current parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Check frozen params and return warnings.

        Args:
            params: FittedParams

        Returns:
            Mapping of frozen param keys to warning messages.
        """
        raise NotImplementedError


class Calibrator(ABC):
    """
    Pluggable calibrator interface with a stable evaluation callback signature.
    """

    @abstractmethod
    def fit(
        self,
        bundle: Dict[str, Any],
        simulator,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: str | None = None,
        params_adapter: ParamsAdapter | None = None,
    ) -> FittedParams:
        """
        Return FittedParams, fitted strictly on the training window.

        Args:
            bundle: Data and plan bundle.
            simulator: Simulation instance.
            evaluator: Callback evaluate_params.
            train_window: (start_day, end_day) inclusive.
            seed: RNG seed.
            budget: Iteration budget.
            artifacts_dir: Directory for artifacts.
            params_adapter: Adapter to apply/capture parameters.

        Returns:
            FittedParams.
        """
        raise NotImplementedError


class SimulationParamsAdapter(ParamsAdapter):
    """
    Concrete ParamsAdapter that maps FittedParams to registry keys and applies to Simulation.
    """

    def __init__(self, registry: ParameterRegistry, plan: Dict[str, Any], defs_path: str = "parameter_definitions.json") -> None:
        """
        Initialize with a ParameterRegistry and model plan.

        Args:
            registry: ParameterRegistry
            plan: Model plan dict
            defs_path: Path to parameter_definitions.json

        Returns:
            None
        """
        self.registry = registry
        self.plan = plan
        self.defs_path = defs_path
        pass

    def apply(self, simulation, params: FittedParams) -> None:
        """
        Apply fitted parameters into the simulation via registry.

        Args:
            simulation: Simulation instance
            params: FittedParams

        Returns:
            None
        """
        # Map decision head weights to adoption coefficients
        dw = params.decision_weights or {}
        mapping_pairs = [
            ("adoption_logit_alpha", "b0", -2.0),
            ("adoption_beta_neighbors", "b_neighbors", 3.0),
            ("adoption_beta_neighbors_sq", "b_neighbors_sq", 1.5),
            ("adoption_gamma_info", "g_info", 1.2),
            ("adoption_gamma_risk", "g_risk", 0.8),
            ("adoption_gamma_risk_x_neighbors", "g_risk_x_neighbors", 0.5),
            ("adoption_gamma_layer_family", "g_family", 0.5),
            ("adoption_gamma_layer_work", "g_work", 0.3),
            ("adoption_gamma_layer_community", "g_community", 0.1),
        ]
        for reg_key, src_key, default_val in mapping_pairs:
            val = float(dw.get(src_key, simulation.registry.values.get(reg_key, default_val)))
            simulation.registry.set_params("SocialInfluenceAdoption", **{reg_key: val})

        # Layer weights
        lw = params.layer_weights or {}
        lw_pairs = [
            ("family", "layer_weight_family"),
            ("work_school", "layer_weight_work"),
            ("community", "layer_weight_community"),
        ]
        for layer_key, reg_key in lw_pairs:
            if layer_key in lw:
                simulation.registry.set_params("SocialNetworkEngine", **{reg_key: float(lw[layer_key])})

        # Info parameters
        ip = params.info_params or {}
        info_pairs = [
            ("info_hazard_base", "InfoDiffusion"),
            ("info_peer_effect_per_adopting_neighbor", "InfoDiffusion"),
            ("info_external_rate", "InfoDiffusion"),
            ("messaging_intensity", "PolicyAndMessaging"),
            ("message_credibility", "PolicyAndMessaging"),
        ]
        for src_key, owner in info_pairs:
            if src_key in ip:
                simulation.registry.set_params(owner, **{src_key: float(ip[src_key])})

        # Module-specific params
        for module_name, subdict in (params.module_params or {}).items():
            for k, v in (subdict or {}).items():
                simulation.registry.set_params(module_name, **{k: v})

        # Save parameters used
        simulation.registry.save_used("parameters_used.json")
        pass

    def capture(self, simulation) -> FittedParams:
        """
        Read current parameters from simulation to build a FittedParams snapshot.

        Args:
            simulation: Simulation instance.

        Returns:
            FittedParams.
        """
        vals = simulation.registry.get_params("all")
        decision_weights = {
            "b0": vals.get("adoption_logit_alpha", -2.0),
            "b_neighbors": vals.get("adoption_beta_neighbors", 3.0),
            "b_neighbors_sq": vals.get("adoption_beta_neighbors_sq", 1.5),
            "g_info": vals.get("adoption_gamma_info", 1.2),
            "g_risk": vals.get("adoption_gamma_risk", 0.8),
            "g_risk_x_neighbors": vals.get("adoption_gamma_risk_x_neighbors", 0.5),
            "g_family": vals.get("adoption_gamma_layer_family", 0.5),
            "g_work": vals.get("adoption_gamma_layer_work", 0.3),
            "g_community": vals.get("adoption_gamma_layer_community", 0.1),
        }
        layer_weights = {
            "family": vals.get("layer_weight_family", 0.6),
            "work_school": vals.get("layer_weight_work", 0.3),
            "community": vals.get("layer_weight_community", 0.1),
        }
        info_params = {
            "info_hazard_base": vals.get("info_hazard_base", 0.05),
            "info_peer_effect_per_adopting_neighbor": vals.get("info_peer_effect_per_adopting_neighbor", 0.02),
            "info_external_rate": vals.get("info_external_rate", 0.01),
            "messaging_intensity": vals.get("messaging_intensity", 0.3),
            "message_credibility": vals.get("message_credibility", 0.7),
        }
        noise_params = {}
        fp = FittedParams(
            decision_weights=decision_weights,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params=noise_params,
            module_params={},
            engine_type="calibrasim",
            meta={"captured_at": time.time()},
        )
        return fp

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate against frozen parameters and return warnings.

        Args:
            params: FittedParams to check.

        Returns:
            Mapping of frozen param names to reasons.
        """
        warnings: Dict[str, str] = {}
        frozen = self.registry.frozen_flags
        # Check layer weights
        for lk, rk in [("family", "layer_weight_family"), ("work_school", "layer_weight_work"), ("community", "layer_weight_community")]:
            if frozen.get(rk, False) and lk in params.layer_weights:
                warnings[rk] = "Frozen; requested layer weight ignored."
        # Decision weights mapping keys
        for dk, rk in [
            ("b0", "adoption_logit_alpha"),
            ("b_neighbors", "adoption_beta_neighbors"),
            ("b_neighbors_sq", "adoption_beta_neighbors_sq"),
            ("g_info", "adoption_gamma_info"),
            ("g_risk", "adoption_gamma_risk"),
            ("g_risk_x_neighbors", "adoption_gamma_risk_x_neighbors"),
            ("g_family", "adoption_gamma_layer_family"),
            ("g_work", "adoption_gamma_layer_work"),
            ("g_community", "adoption_gamma_layer_community"),
        ]:
            if frozen.get(rk, False) and dk in (params.decision_weights or {}):
                warnings[rk] = "Frozen; requested decision weight ignored."
        return warnings


def evaluate_params(simulator, params: FittedParams, window) -> Dict[str, Any]:
    """
    Apply `params`, reset simulator to the start of `window`, run forward, and return metrics.

    Args:
        simulator: Simulation instance
        params: FittedParams to apply
        window: (start_day, end_day) inclusive

    Returns:
        Dictionary of evaluation metrics.
    """
    adapter = SimulationParamsAdapter(simulator.registry, simulator.plan)
    adapter.apply(simulator, params)
    start, end = int(window[0]), int(window[1])
    simulator.reset_for_window(start)
    simulator.run(start_day=start, end_day=end)
    eval_res = simulator.evaluate(window=(start, end))
    # Build required keys
    rmse = float(eval_res.get("RMSE", np.nan))
    mae = float(eval_res.get("MAE", np.nan))
    brier = float(eval_res.get("Brier", rmse**2 if not np.isnan(rmse) else np.nan))
    transfit = eval_res.get("TransitionFit", {"P01": np.nan, "P11": np.nan, "P10": np.nan, "P00": np.nan})
    out = {
        "RMSE_aggregate": rmse,
        "MAE_aggregate": mae,
        "Brier": brier,
        "TransitionFit": transfit,
        "TimeTo50Error": eval_res.get("TimeTo50Error", float("nan")),
        "Rb_MAE": eval_res.get("Rb_MAE", float("nan")),
        "Churn_MAE": eval_res.get("Churn_MAE", float("nan")),
    }
    return out


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head using micro-transitions from the training window with L2 regularization.
    Degrades gracefully if micro-transitions are unavailable by using defaults.
    """

    def __init__(self, l2: float = 1.0, max_iter: int = 200) -> None:
        """
        Initialize the logistic head calibrator.

        Args:
            l2: L2 regularization strength.
            max_iter: Maximum gradient iterations.

        Returns:
            None
        """
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        pass

    def _build_micro_dataset(self, bundle: Dict[str, Any], train_window: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build micro-level dataset X, y from train_data within train_window.

        Features include: prev_adopt, informed_t-1, risk_perception, global_adoption_t-1.
        If network data available, include approximate neighbor fraction.

        Args:
            bundle: Data and plan bundle.
            train_window: (start_day, end_day)

        Returns:
            X: Feature matrix [N, d]
            y: Labels [N,]
        """
        train_df: pd.DataFrame = bundle.get("train_data", pd.DataFrame()).copy()
        if train_df.empty:
            return np.zeros((0, 6)), np.zeros((0,))
        start, end = train_window
        df = train_df[(train_df["day"] >= start) & (train_df["day"] <= end)].copy()
        if df.empty:
            return np.zeros((0, 6)), np.zeros((0,))
        # compute global adoption per day for lagging
        day_adopt = df.groupby("day")["wearing_mask"].mean().to_dict()
        df["adopt_lag"] = df.groupby("agent_id")["wearing_mask"].shift(1).fillna(False)
        df["info_lag"] = df.groupby("agent_id")["received_info"].shift(1).fillna(False)
        df["global_adopt_lag"] = df["day"].apply(lambda d: day_adopt.get(d - 1, day_adopt.get(d, 0.0)))
        # Merge risk perception from agent attributes
        attrs: pd.DataFrame = bundle.get("agent_attributes", pd.DataFrame()).copy()
        if "risk_perception" in attrs.columns:
            df = df.merge(attrs[["agent_id", "risk_perception"]], on="agent_id", how="left")
        else:
            df["risk_perception"] = 0.5
        df = df.dropna(subset=["adopt_lag", "info_lag", "risk_perception", "global_adopt_lag", "wearing_mask"])
        # Build features
        X = np.vstack([
            np.ones(len(df)),  # intercept
            df["global_adopt_lag"].astype(float).values,
            (df["global_adopt_lag"].astype(float).values ** 2),
            df["info_lag"].astype(float).values,
            df["risk_perception"].astype(float).values,
            (df["risk_perception"].astype(float).values * df["global_adopt_lag"].astype(float).values),
        ]).T
        y = df["wearing_mask"].astype(float).values
        return X, y

    def _fit_logit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit logistic regression with L2 regularization using simple gradient descent.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Coefficients vector
        """
        if X.size == 0 or y.size == 0:
            return np.zeros(X.shape[1] if X.ndim == 2 else 1)
        n, d = X.shape
        w = np.zeros(d)
        lr = 0.1
        for it in range(self.max_iter):
            z = X @ w
            p = sigmoid(z)
            grad = X.T @ (p - y) / n + self.l2 * np.r_[0.0, w[1:]] / n  # don't regularize intercept
            w -= lr * grad
            if np.linalg.norm(grad) < 1e-6:
                break
        return w

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: str | None = None,
        params_adapter: ParamsAdapter | None = None,
    ) -> FittedParams:
        """
        Fit FittedParams by regressing micro transitions and build a decision head.

        Args:
            bundle: Data and plan bundle.
            simulator: Simulation instance.
            evaluator: Evaluation callback.
            train_window: Training day window (inclusive).
            seed: RNG seed.
            budget: Unused here; included to conform interface.
            artifacts_dir: Directory for saving artifacts.
            params_adapter: Params adapter to apply/capture.

        Returns:
            FittedParams.
        """
        random.seed(seed)
        np.random.seed(seed)
        X, y = self._build_micro_dataset(bundle, train_window)
        if X.size == 0:
            log("WARNING: No micro-transitions available; using default decision weights.")
            dw = params_adapter.capture(simulator).decision_weights if params_adapter else {}
            fp = FittedParams(
                decision_weights=dw,
                layer_weights=params_adapter.capture(simulator).layer_weights if params_adapter else {},
                info_params=params_adapter.capture(simulator).info_params if params_adapter else {},
                noise_params={},
                module_params={},
                engine_type="calibrasim",
                meta={"calibrator": "logit_head", "used_defaults": True},
            )
            return fp
        w = self._fit_logit(X, y)
        # Map weights to keys
        # X cols: [1, global_adopt, global_adopt^2, info_lag, risk, risk*global_adopt]
        dw = {
            "b0": float(w[0]),
            "b_neighbors": float(w[1]),
            "b_neighbors_sq": float(w[2]),
            "g_info": float(w[3]),
            "g_risk": float(w[4]),
            "g_risk_x_neighbors": float(w[5]),
            "g_family": 0.5,
            "g_work": 0.3,
            "g_community": 0.1,
        }
        # Fixed layer weights from current simulator as baseline
        current_fp = params_adapter.capture(simulator) if params_adapter else None
        layer_weights = current_fp.layer_weights if current_fp else {"family": 0.6, "work_school": 0.3, "community": 0.1}
        info_params = current_fp.info_params if current_fp else {
            "info_hazard_base": 0.05, "info_peer_effect_per_adopting_neighbor": 0.02, "info_external_rate": 0.01,
            "messaging_intensity": 0.3, "message_credibility": 0.7
        }
        fp = FittedParams(
            decision_weights=dw,
            layer_weights=layer_weights,
            info_params=info_params,
            noise_params={},
            module_params={},
            engine_type="calibrasim",
            meta={"calibrator": "logit_head", "seed": seed, "train_window": train_window},
        )
        # Save artifacts
        if artifacts_dir:
            safe_json_dump(os.path.join(artifacts_dir, "logit_head", "fitted_params.json"), fp.to_dict())
        return fp


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search over selected simulator parameters within bounds.
    """

    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Initialize with an optional search space dict.

        Args:
            search_space: Mapping param_key -> (low, high)

        Returns:
            None
        """
        self.search_space = search_space or {}
        pass

    def _default_search_space(self, simulator) -> Dict[str, Tuple[float, float]]:
        """
        Provide default bounds from registry definitions for selected keys.

        Args:
            simulator: Simulation instance.

        Returns:
            Search space mapping.
        """
        keys = [
            "adoption_logit_alpha",
            "adoption_beta_neighbors",
            "adoption_beta_neighbors_sq",
            "adoption_gamma_info",
            "adoption_gamma_risk",
            "adoption_gamma_risk_x_neighbors",
            "adoption_gamma_layer_family",
            "adoption_gamma_layer_work",
            "adoption_gamma_layer_community",
            "layer_weight_family",
            "layer_weight_work",
            "layer_weight_community",
            "info_hazard_base",
            "info_peer_effect_per_adopting_neighbor",
            "info_external_rate",
            "messaging_intensity",
            "message_credibility",
        ]
        sp: Dict[str, Tuple[float, float]] = {}
        for k in keys:
            d = simulator.registry.definitions.get(k, {})
            b = d.get("bounds")
            if b and "low" in b and "high" in b:
                sp[k] = (float(b["low"]), float(b["high"]))
            else:
                # fallback ranges
                sp[k] = (-3.0, 3.0) if "adoption_" in k else (0.0, 1.0)
        return sp

    def _sample_params(self, sp: Dict[str, Tuple[float, float]], rng: random.Random) -> FittedParams:
        """
        Sample a FittedParams from the search space.

        Args:
            sp: Search space
            rng: RNG

        Returns:
            FittedParams
        """
        # Decision weights
        dw = {
            "b0": rng.uniform(-4, 0),
            "b_neighbors": rng.uniform(0.0, 6.0),
            "b_neighbors_sq": rng.uniform(-1.0, 4.0),
            "g_info": rng.uniform(-1.0, 3.0),
            "g_risk": rng.uniform(-1.0, 2.0),
            "g_risk_x_neighbors": rng.uniform(-1.0, 2.0),
            "g_family": rng.uniform(-1.0, 2.0),
            "g_work": rng.uniform(-1.0, 2.0),
            "g_community": rng.uniform(-1.0, 2.0),
        }
        lw = {
            "family": rng.uniform(0.0, 1.0),
            "work_school": rng.uniform(0.0, 1.0),
            "community": rng.uniform(0.0, 1.0),
        }
        # Normalize layer weights to sum to 1 (avoid zero-sum edge case)
        s = max(1e-6, lw["family"] + lw["work_school"] + lw["community"])
        for k in lw:
            lw[k] /= s
        ip = {
            "info_hazard_base": rng.uniform(0.0, 0.2),
            "info_peer_effect_per_adopting_neighbor": rng.uniform(0.0, 0.1),
            "info_external_rate": rng.uniform(0.0, 0.1),
            "messaging_intensity": rng.uniform(0.0, 1.0),
            "message_credibility": rng.uniform(0.0, 1.0),
        }
        fp = FittedParams(
            decision_weights=dw, layer_weights=lw, info_params=ip, noise_params={},
            module_params={}, engine_type="calibrasim",
            meta={"sampler": "random_search"},
        )
        return fp

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: str | None = None,
        params_adapter: ParamsAdapter | None = None,
    ) -> FittedParams:
        """
        Execute random search calibration, selecting the best params by RMSE_aggregate on the training window.

        Args:
            bundle: Data and plan bundle.
            simulator: Simulation instance.
            evaluator: Callback to evaluate params.
            train_window: (start_day, end_day) inclusive
            seed: RNG seed.
            budget: Number of trials.
            artifacts_dir: Directory for artifacts.
            params_adapter: ParamsAdapter.

        Returns:
            Best FittedParams.
        """
        rng = random.Random(seed)
        sp = self.search_space or self._default_search_space(simulator)
        best_score = float("inf")
        best_fp: Optional[FittedParams] = None
        trials: List[Dict[str, Any]] = []
        for i in range(int(budget)):
            fp = self._sample_params(sp, rng)
            # Apply and evaluate
            metrics = evaluator(simulator, fp, train_window)
            score = metrics.get("RMSE_aggregate", float("inf"))
            trials.append({"trial": i, "params": fp.to_dict(), "metrics": metrics})
            if artifacts_dir:
                trial_dir = os.path.join(artifacts_dir, f"trial_{i}")
                safe_json_dump(os.path.join(trial_dir, "params_applied.json"), fp.to_dict())
                safe_json_dump(os.path.join(trial_dir, "metrics.json"), metrics)
            if score < best_score:
                best_score, best_fp = score, fp
        # Save best
        if best_fp and artifacts_dir:
            best_dir = os.path.join(artifacts_dir, "best")
            safe_json_dump(os.path.join(best_dir, "fitted_params.json"), best_fp.to_dict())
            report = {"budget": budget, "best_score": best_score, "trials": trials}
            safe_json_dump(os.path.join(artifacts_dir, "calibration_report.json"), report)
        if not best_fp:
            # fallback to current
            best_fp = params_adapter.capture(simulator) if params_adapter else FittedParams({}, {}, {}, {}, {})
        return best_fp


class SNPECalibrator(Calibrator):
    """
    SNPECalibrator is a placeholder that currently falls back to random search and local perturbations.
    """

    def __init__(self, num_simulations: int = 100, posterior_samples: int = 10) -> None:
        """
        Initialize SNPE calibrator.

        Args:
            num_simulations: Number of simulations for SBI.
            posterior_samples: Samples from posterior to select best.

        Returns:
            None
        """
        self.num_simulations = int(num_simulations)
        self.posterior_samples = int(posterior_samples)
        pass

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator,
        evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: str | None = None,
        params_adapter: ParamsAdapter | None = None,
    ) -> FittedParams:
        """
        Fit using fallback search and perturbations.

        Args:
            bundle: Data and plan bundle.
            simulator: Simulation instance.
            evaluator: Evaluate callback.
            train_window: (start_day, end_day) inclusive.
            seed: RNG seed.
            budget: Budget; if provided used as num_simulations.
            artifacts_dir: Optional artifacts path.
            params_adapter: Params adapter.

        Returns:
            FittedParams.
        """
        try:
            import torch  # noqa
            from sbi.inference import SNPE as SNPEngine  # noqa
            rng = random.Random(seed)
            rs = RandomSearchCalibrator()
            best_fp = rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)
            # Small perturbation candidates
            candidates: List[FittedParams] = []
            for _ in range(max(1, self.posterior_samples)):
                base = params_adapter.capture(simulator) if params_adapter else best_fp
                # small perturbation
                dw = dict(base.decision_weights)
                for k in dw:
                    dw[k] = float(dw[k] + rng.uniform(-0.1, 0.1))
                lw = dict(base.layer_weights)
                for k in lw:
                    lw[k] = float(max(0.0, lw[k] + rng.uniform(-0.05, 0.05)))
                # renormalize lw
                s = sum(lw.values())
                if s > 0:
                    for k in lw:
                        lw[k] /= s
                ip = dict(base.info_params)
                for k in ip:
                    ip[k] = float(max(0.0, ip[k] + rng.uniform(-0.05, 0.05)))
                candidates.append(FittedParams(dw, lw, ip, base.noise_params, base.module_params, base.engine_type, base.meta))
            best_score = float("inf")
            best = None
            for fp in candidates:
                m = evaluator(simulator, fp, train_window)
                s = m.get("RMSE_aggregate", float("inf"))
                if s < best_score:
                    best_score, best = s, fp
            if best is None:
                best = best_fp
            if artifacts_dir:
                safe_json_dump(os.path.join(artifacts_dir, "best", "fitted_params_snpe.json"), best.to_dict())
            return best
        except Exception as e:
            log(f"WARNING: SNPE dependencies unavailable or error '{e}'. Falling back to RandomSearch.")
            rs = RandomSearchCalibrator()
            return rs.fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: str | None):
    """
    Factory to instantiate a calibrator by name with optional config.

    Args:
        name: Calibrator name.
        config_path: Path to JSON/YAML config.

    Returns:
        Calibrator instance.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    kwargs = {}
    if config_path and os.path.exists(config_path):
        try:
            if config_path.endswith(".json"):
                with open(config_path, "r", encoding="utf-8") as f:
                    kwargs = json.load(f)
            else:
                # YAML optional
                import yaml  # type: ignore
                with open(config_path, "r", encoding="utf-8") as f:
                    kwargs = yaml.safe_load(f)
        except Exception:
            kwargs = {}
    return CALIBRATOR_REGISTRY[name](**kwargs)


# ===========
# Base Module
# ===========

class BaseModule(ABC):
    """
    Abstract base class for modules with a forward pass writing outputs to buffers.
    Entities are represented via vectorized arrays in Simulation.state for performance.
    """

    def __init__(self, name: str, dependencies: List[str], tick_rate_days: int = 1, sbi_calibration: bool = False) -> None:
        """
        Initialize module.

        Args:
            name: Module name.
            dependencies: List of module names to run before this one.
            tick_rate_days: Execution frequency in days.
            sbi_calibration: Whether module supports calibration artifact outputs.

        Returns:
            None
        """
        self.name = name
        self.dependencies = dependencies or []
        self.tick_rate_days = max(1, int(tick_rate_days))
        self.sbi_calibration = bool(sbi_calibration)
        self.last_buffer_snapshot: Dict[str, Any] = {}
        pass

    @abstractmethod
    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Execute module logic and write outputs to buffers.

        Args:
            state: Simulation state dictionary.
            buffers: Buffers dictionary to write signals.
            params: Module-specific parameters.
            t: Current time step (day).

        Returns:
            None
        """
        raise NotImplementedError

    def aggregate_and_save(self, state: Dict[str, Any], buffers: Dict[str, Any], observables: List[Dict[str, Any]], results_dir: str, gt_df: pd.DataFrame | None = None) -> None:
        """
        Aggregate module outputs for SBI calibration and save to JSON.

        Args:
            state: Simulation state dict.
            buffers: Buffers dict.
            observables: Global observables list from plan.
            results_dir: Base directory to save results.
            gt_df: Ground truth DataFrame for same observables if available.

        Returns:
            None
        """
        # Default implementation saves placeholder or module-relevant signals.
        out: Dict[str, Any] = {}
        gt: Dict[str, Any] = {}
        module_dir = os.path.join(results_dir, "observables")
        ensure_dir(module_dir)
        # Example: if this module provided 'assortativity_by_adoption' in buffers
        if "assortativity_by_adoption" in state.get("daily", {}):
            out["observable.assortativity_by_adoption"] = state["daily"]["assortativity_by_adoption"]
        # GT: compute from train data if available
        if gt_df is not None and not gt_df.empty and "wearing_mask" in gt_df.columns and "day" in gt_df.columns and nx is not None:
            if state.get("graph") is not None:
                g = state["graph"]
                daily_series = []
                for day in sorted(gt_df["day"].unique()):
                    df_day = gt_df[gt_df["day"] == day]
                    # Using id_to_idx if available
                    adopt_map = {int(state["id_to_idx"].get(int(a), -1)): int(parse_bool(v)) for a, v in zip(df_day["agent_id"], df_day["wearing_mask"])}
                    for n in g.nodes():
                        g.nodes[n]["adopt"] = adopt_map.get(int(n), 0)
                    try:
                        r = nx.attribute_assortativity_coefficient(g, "adopt")
                        assort = float(r)
                    except Exception:
                        assort = float("nan")
                    daily_series.append(assort)
                gt["observable.assortativity_by_adoption"] = daily_series
        safe_json_dump(os.path.join(module_dir, f"{self.name}.json"), out)
        safe_json_dump(os.path.join(module_dir, f"{self.name}_gt.json"), gt)
        pass


# Helper to read params with defaults
def get_param(state: Dict[str, Any], key: str, default: Any) -> Any:
    return state.get("params", {}).get(key, default)


# ====================
# Module Implementions
# ====================

class SocialNetworkEngine(BaseModule):
    """
    Maintains multiplex network, computes contact events and exposures, and performs homophily-based rewiring.

    Emits:
        - signal.frac_neighbors_by_layer
        - signal.frac_neighbors_overall
        - signal.exposures_by_layer
    """

    def __init__(self) -> None:
        """
        Initialize the SocialNetworkEngine module.
        """
        super().__init__(name="SocialNetworkEngine", dependencies=[], tick_rate_days=1, sbi_calibration=True)
        pass

    def _compute_overlap_multiplier(self, i: int, j: int, state: Dict[str, Any]) -> float:
        """
        Compute multiplex overlap multiplier for dyad (i,j).

        Args:
            i: Ego
            j: Alter
            state: Simulation state

        Returns:
            Multiplier (> 1 if multiple layers connect).
        """
        layers = state["neighbors_layers"]
        count = 0
        if j in layers["family"].get(i, []):
            count += 1
        if j in layers["work_school"].get(i, []):
            count += 1
        if j in layers["community"].get(i, []):
            count += 1
        if count >= 2:
            return float(get_param(state, "multiplex_overlap_multiplier", 1.3))
        return 1.0

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Implement the pseudocode for exposure computation and dynamic rewiring.

        Args:
            state: Simulation state.
            buffers: Signals buffer to write results.
            params: Module-specific parameters.
            t: Current day.

        Returns:
            None
        """
        N = state["N"]
        adopt = state["adoption_state"]
        # Parameters with defaults
        pF = float(get_param(state, "layer_contact_prob_family", 0.9))
        pW = float(get_param(state, "layer_contact_prob_work", 0.6))
        pC = float(get_param(state, "layer_contact_prob_community", 0.25))
        wF = float(get_param(state, "layer_weight_family", 0.6))
        wW = float(get_param(state, "layer_weight_work", 0.3))
        wC = float(get_param(state, "layer_weight_community", 0.1))
        # Access neighbors
        layers = state["neighbors_layers"]
        rng = state["rng"]
        # Compute per-layer exposures and fractions
        frac_F = np.zeros(N, dtype=float)
        frac_W = np.zeros(N, dtype=float)
        frac_C = np.zeros(N, dtype=float)
        exp_F = np.zeros(N, dtype=float)
        exp_W = np.zeros(N, dtype=float)
        exp_C = np.zeros(N, dtype=float)
        for i in range(N):
            for layer_key, p, frac_arr, exp_arr in [
                ("family", pF, frac_F, exp_F),
                ("work_school", pW, frac_W, exp_W),
                ("community", pC, frac_C, exp_C),
            ]:
                neighs = layers[layer_key].get(i, [])
                if not neighs:
                    frac_arr[i] = 0.0
                    continue
                weight_sum = 0.0
                adopting_weight_sum = 0.0
                for j in neighs:
                    if rng.random() < p:
                        mult = self._compute_overlap_multiplier(i, j, state)
                        weight_sum += mult
                        if adopt[j]:
                            adopting_weight_sum += mult
                exp_arr[i] = adopting_weight_sum
                denom = max(1e-9, weight_sum)
                frac_val = float(adopting_weight_sum / denom)
                frac_arr[i] = float(max(0.0, min(1.0, frac_val)))
        # Combine layers weighted
        denom_w = max(1e-9, (wF + wW + wC))
        frac_overall = (wF * frac_F + wW * frac_W + wC * frac_C) / denom_w
        # Exposure window smoothing
        W = int(get_param(state, "exposure_window", 3))
        hist = state["history"]["frac_overall"]
        hist.append(frac_overall.copy())
        if len(hist) > W:
            hist.popleft()
        smoothed = np.mean(hist, axis=0) if len(hist) > 0 else frac_overall
        # Dynamic homophily-based rewiring (Simple heuristic on community layer)
        if rng.random() < float(get_param(state, "dynamic_rewire_prob", 0.02)):
            # Pick random ego
            i = rng.randrange(N)
            # Choose a random neighbor j overall
            all_neighs = state["neighbors_all"].get(i, [])
            if all_neighs:
                j = rng.choice(all_neighs)
                if adopt[i] != adopt[j] and float(get_param(state, "homophily_strength", 0.2)) > 0:
                    # Propose k not connected to i
                    attempts = 0
                    while attempts < 5:
                        k = rng.randrange(N)
                        if k == i or k in all_neighs:
                            attempts += 1
                            continue
                        # Homophily score
                        score = 1.0
                        if adopt[k] == adopt[i]:
                            score += float(get_param(state, "homophily_strength", 0.2))
                        if state["age_group"][k] == state["age_group"][i]:
                            score += 0.5 * float(get_param(state, "homophily_strength", 0.2))
                        # Accept if score passes random threshold
                        if rng.random() < sigmoid(score - 1.0):
                            # Rewire in community layer preserving degree
                            if j in layers["community"].get(i, []):
                                layers["community"][i] = [x for x in layers["community"][i] if x != j]
                                if i in layers["community"].get(j, []):
                                    layers["community"][j] = [x for x in layers["community"][j] if x != i]
                            if k not in layers["community"][i]:
                                layers["community"][i].append(k)
                            if i not in layers["community"][k]:
                                layers["community"][k].append(i)
                            # Deduplicate lists
                            layers["community"][i] = list(sorted(set(layers["community"][i])))
                            layers["community"][k] = list(sorted(set(layers["community"][k])))
                            # Update all neighbors union
                            state["neighbors_all"][i] = sorted(list(set(
                                layers["family"].get(i, []) + layers["work_school"].get(i, []) + layers["community"].get(i, [])
                            )))
                            state["neighbors_all"][j] = sorted(list(set(
                                layers["family"].get(j, []) + layers["work_school"].get(j, []) + layers["community"].get(j, [])
                            )))
                            state["neighbors_all"][k] = sorted(list(set(
                                layers["family"].get(k, []) + layers["work_school"].get(k, []) + layers["community"].get(k, [])
                            )))
                            # Update degree_all for affected nodes
                            state["degree_all"][i] = float(len(state["neighbors_all"][i]))
                            state["degree_all"][j] = float(len(state["neighbors_all"][j]))
                            state["degree_all"][k] = float(len(state["neighbors_all"][k]))
                            break
                        attempts += 1
        # Save signals to buffers
        buffers["signals"]["frac_neighbors_by_layer"] = {
            "family": frac_F,
            "work_school": frac_W,
            "community": frac_C,
        }
        buffers["signals"]["frac_neighbors_overall"] = smoothed
        buffers["signals"]["exposures_by_layer"] = {"family": exp_F, "work_school": exp_W, "community": exp_C}
        # Update graph for assortativity (if available)
        if nx is not None and state.get("graph") is not None:
            g = state["graph"]
            for n in g.nodes():
                g.nodes[n]["adopt"] = int(adopt[n])
            try:
                assort = nx.attribute_assortativity_coefficient(g, "adopt")
            except Exception:
                assort = float("nan")
            state["daily"]["assortativity_by_adoption"].append(float(assort))
        self.last_buffer_snapshot = {"signals": {k: (v if not isinstance(v, dict) else {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}) for k, v in buffers["signals"].items()}}
        pass


class PolicyAndMessaging(BaseModule):
    """
    Activates mandates and broadcasts public health messages via media channels.

    Emits:
        - signal.policy_active
        - signal.policy_odds_multiplier
        - signal.messaging_intensity_effective
    """

    def __init__(self) -> None:
        """
        Initialize the PolicyAndMessaging module.
        """
        super().__init__(name="PolicyAndMessaging", dependencies=[], tick_rate_days=1, sbi_calibration=True)
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Implement policy and messaging signals.

        Args:
            state: Simulation state
            buffers: Buffers to write signals
            params: Module-specific params
            t: Current day

        Returns:
            None
        """
        mandate_start = int(get_param(state, "mandate_start_day", 10))
        policy_active = bool(t >= mandate_start)
        policy_odds_multiplier = float(get_param(state, "policy_odds_multiplier", 1.5) if policy_active else 1.0)
        # Effective messaging intensity
        messaging_intensity = float(get_param(state, "messaging_intensity", 0.3))
        media_reach = float(get_param(state, "media_reach", 0.8))
        message_credibility = float(get_param(state, "message_credibility", 0.7))
        message_bias = float(get_param(state, "message_bias", 0.0))
        message_frequency = float(get_param(state, "message_frequency", 1.0))
        messaging_intensity_effective = messaging_intensity * media_reach * message_credibility * (1.0 + 0.1 * message_bias) * message_frequency
        buffers["signals"]["policy_active"] = policy_active
        buffers["signals"]["policy_odds_multiplier"] = policy_odds_multiplier
        buffers["signals"]["messaging_intensity_effective"] = messaging_intensity_effective
        self.last_buffer_snapshot = {"signals": buffers["signals"].copy()}
        pass


class InfoDiffusion(BaseModule):
    """
    Updates each agent's informed state based on base hazard, peer adoption, and messaging signals.

    Emits:
        - signal.informed_flags (boolean array indicating newly informed)
    """

    def __init__(self) -> None:
        """
        Initialize InfoDiffusion module.
        """
        super().__init__(name="InfoDiffusion", dependencies=["PolicyAndMessaging", "SocialNetworkEngine"], tick_rate_days=1, sbi_calibration=True)
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Implement the info diffusion hazard update.

        Args:
            state: Simulation state
            buffers: Buffers to write signals
            params: Module params
            t: Current day

        Returns:
            None
        """
        N = state["N"]
        informed = state["informed"]
        frac_overall = buffers["signals"].get("frac_neighbors_overall", np.zeros(N))
        msg_term = float(get_param(state, "info_external_rate", 0.01)) * float(buffers["signals"].get("messaging_intensity_effective", 0.0))
        base = float(get_param(state, "info_hazard_base", 0.05))
        peer_effect = float(get_param(state, "info_peer_effect_per_adopting_neighbor", 0.02))
        # Approximate expected contacts adopting
        deg_all = state["degree_all"]
        pF = float(get_param(state, "layer_contact_prob_family", 0.9))
        pW = float(get_param(state, "layer_contact_prob_work", 0.6))
        pC = float(get_param(state, "layer_contact_prob_community", 0.25))
        wF = float(get_param(state, "layer_weight_family", 0.6))
        wW = float(get_param(state, "layer_weight_work", 0.3))
        wC = float(get_param(state, "layer_weight_community", 0.1))
        p_contact_avg = (wF * pF + wW * pW + wC * pC) / max(1e-9, (wF + wW + wC))
        expected_contacts_adopting = deg_all * p_contact_avg * frac_overall
        peer_term = 1.0 - np.power((1.0 - peer_effect), np.clip(expected_contacts_adopting, 0, None))
        p_info = 1.0 - (1.0 - base) * (1.0 - peer_term) * (1.0 - msg_term)
        p_info = np.clip(p_info, 0.0, 1.0)
        rng = state["rng"]
        new_informed = np.zeros(N, dtype=bool)
        for i in range(N):
            if not informed[i]:
                if rng.random() < p_info[i]:
                    new_informed[i] = True
        # Observation noise: flip with small probability if configured (frozen by design)
        if float(get_param(state, "observation_noise", 0.0)) > 0.0:
            noise_p = float(get_param(state, "observation_noise", 0.0))
            for i in range(N):
                if rng.random() < noise_p:
                    new_informed[i] = not new_informed[i]
        buffers["signals"]["informed_flags"] = new_informed
        self.last_buffer_snapshot = {"signals": buffers["signals"].copy()}
        pass


class SocialInfluenceAdoption(BaseModule):
    """
    Computes adoption hazards and executes adoption with optional delay.

    Emits:
        - signal.adoption_flags (dict with keys 'adopt_now_ids', 'scheduled': {day:[ids]})
    """

    def __init__(self) -> None:
        """
        Initialize SocialInfluenceAdoption module.
        """
        super().__init__(name="SocialInfluenceAdoption", dependencies=["InfoDiffusion", "PolicyAndMessaging", "SocialNetworkEngine"], tick_rate_days=1, sbi_calibration=True)
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Implement adoption decision rules using logistic hazard or threshold model.

        Args:
            state: Simulation state
            buffers: Buffers to write adoption signals
            params: Module params
            t: Current day

        Returns:
            None
        """
        N = state["N"]
        adopt_state = state["adoption_state"]
        informed = state["informed"]
        frac_overall = buffers["signals"].get("frac_neighbors_overall", np.zeros(N))
        by_layer = buffers["signals"].get("frac_neighbors_by_layer", {"family": np.zeros(N), "work_school": np.zeros(N), "community": np.zeros(N)})
        frac_F = by_layer.get("family", np.zeros(N))
        frac_W = by_layer.get("work_school", np.zeros(N))
        frac_C = by_layer.get("community", np.zeros(N))
        policy_active = bool(buffers["signals"].get("policy_active", False))
        policy_odds_multiplier = float(buffers["signals"].get("policy_odds_multiplier", 1.0))
        beta0 = float(get_param(state, "adoption_logit_alpha", -2.0))
        b1 = float(get_param(state, "adoption_beta_neighbors", 3.0))
        b2 = float(get_param(state, "adoption_beta_neighbors_sq", 1.5))
        g_info = float(get_param(state, "adoption_gamma_info", 1.2))
        g_risk = float(get_param(state, "adoption_gamma_risk", 0.8))
        g_rxn = float(get_param(state, "adoption_gamma_risk_x_neighbors", 0.5))
        gF = float(get_param(state, "adoption_gamma_layer_family", 0.5))
        gW = float(get_param(state, "adoption_gamma_layer_work", 0.3))
        gC = float(get_param(state, "adoption_gamma_layer_community", 0.1))
        benefit = float(get_param(state, "benefit_perceived", 0.3))
        cost = float(get_param(state, "compliance_cost", 0.2))
        rng = state["rng"]
        stubborn_set = state["stubborn_set"]
        adoption_function = get_param(state, "adoption_function", "logistic")
        mu_log = float(get_param(state, "adoption_delay_mu_log", 0.0))
        sigma_log = float(get_param(state, "adoption_delay_sigma_log", 0.75))
        threshold_lambda = float(get_param(state, "adoption_threshold_lambda", 2.0))
        # Initialize scheduled dict
        scheduled: Dict[int, List[int]] = defaultdict(list)
        adopt_now_ids: List[int] = []
        risk = state["risk_perception"]
        # Prepare exposures cumulation if threshold model
        if adoption_function == "threshold":
            if "exposures_cum" not in state:
                state["exposures_cum"] = np.zeros(N, dtype=float)
            if "threshold_K" not in state:
                state["threshold_K"] = state["np_rng"].poisson(lam=max(0.1, threshold_lambda), size=N)
            # Add today's exposures
            exp_l = buffers["signals"].get("exposures_by_layer", {})
            total_exp = exp_l.get("family", np.zeros(N)) + exp_l.get("work_school", np.zeros(N)) + exp_l.get("community", np.zeros(N))
            state["exposures_cum"] += total_exp
        # Decision
        for i in range(N):
            if adopt_state[i]:
                continue
            is_stubborn = (i in stubborn_set)
            if adoption_function == "logistic":
                x = float(frac_overall[i])
                x2 = x * x
                info = 1.0 if informed[i] else 0.0
                logit = beta0 + b1 * x + b2 * x2 + g_info * info + g_risk * float(risk[i]) + g_rxn * float(risk[i]) * x
                logit += gF * float(frac_F[i]) + gW * float(frac_W[i]) + gC * float(frac_C[i])
                if policy_active and policy_odds_multiplier > 0:
                    logit += math.log(max(1e-9, policy_odds_multiplier))
                logit += (benefit - cost)
                # Stubborn: adopt only if policy active
                if is_stubborn and not policy_active:
                    continue
                p = float(sigmoid(logit))
                if rng.random() < p:
                    # Delay adoption
                    delay = max(0, int(round(state["np_rng"].lognormal(mean=mu_log, sigma=sigma_log))))
                    if delay == 0:
                        adopt_now_ids.append(i)
                    else:
                        scheduled[t + delay].append(i)
            else:
                # Threshold model
                if is_stubborn and not policy_active:
                    continue
                if "exposures_cum" in state and "threshold_K" in state and state["exposures_cum"][i] >= state["threshold_K"][i]:
                    delay = max(0, int(round(state["np_rng"].lognormal(mean=mu_log, sigma=sigma_log))))
                    if delay == 0:
                        adopt_now_ids.append(i)
                    else:
                        scheduled[t + delay].append(i)
        buffers["signals"]["adoption_flags"] = {"adopt_now_ids": adopt_now_ids, "scheduled": scheduled}
        self.last_buffer_snapshot = {"signals": {"adopt_now_ids": adopt_now_ids, "scheduled": {int(k): list(v) for k, v in scheduled.items()}}}
        pass


class DropoutAndFatigue(BaseModule):
    """
    Models churn among adopters driven by low local adoption, low risk, and fatigue; enforces a minimum duration.

    Emits:
        - signal.dropout_flags (list of agent ids to drop)
    """

    def __init__(self) -> None:
        """
        Initialize DropoutAndFatigue module.
        """
        super().__init__(name="DropoutAndFatigue", dependencies=["SocialNetworkEngine", "PolicyAndMessaging"], tick_rate_days=1, sbi_calibration=True)
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute drop hazard and emit dropout flags.

        Args:
            state: Simulation state
            buffers: Buffers to write signals
            params: Module parameters
            t: Current day

        Returns:
            None
        """
        N = state["N"]
        adopt = state["adoption_state"]
        tsince = state["time_since_adoption"]
        nfrac = buffers["signals"].get("frac_neighbors_overall", np.zeros(N))
        risk = state["risk_perception"]
        rng = state["rng"]
        base_rate = float(get_param(state, "dropout_base_rate", 0.01))
        fatigue_rate = float(get_param(state, "fatigue_rate", 0.005))
        intercept = float(get_param(state, "drop_logit_intercept", -4.0))
        bN = float(get_param(state, "drop_beta_one_minus_neighbor_frac", 2.0))
        bR = float(get_param(state, "drop_beta_one_minus_risk", 1.5))
        min_duration = int(get_param(state, "dropout_min_duration_days", 2))
        cap = float(get_param(state, "dropout_probability_cap", 0.5))
        policy_active = bool(buffers["signals"].get("policy_active", False))
        enforcement = float(get_param(state, "mandate_enforcement_strength", 0.6))
        drop_ids: List[int] = []
        for i in range(N):
            if adopt[i] and int(tsince[i]) >= min_duration:
                logit_drop = intercept + bN * (1.0 - float(nfrac[i])) + bR * (1.0 - float(risk[i])) + fatigue_rate * float(tsince[i])
                if policy_active:
                    logit_drop -= math.log(1.0 + max(0.0, enforcement))
                p_drop = min(cap, float(sigmoid(logit_drop)) + base_rate)
                if rng.random() < p_drop:
                    drop_ids.append(i)
        buffers["signals"]["dropout_flags"] = drop_ids
        self.last_buffer_snapshot = {"signals": buffers["signals"].copy()}
        pass


class AdoptionAggregator(BaseModule):
    """
    Aggregates daily observables and computes evaluation metrics inputs.

    Emits:
        - observable.adoption_rate_daily
        - observable.adoption_rate_by_group
        - observable.final_adoption_rate
        - observable.time_to_50_percent_adoption
        - observable.Rb_series
        - observable.churn_rate_daily
        - observable.mean_exposures_before_adoption
        - observable.assortativity_by_adoption
        - observable.inequality_of_adoption
        - observable.policy_effect_size
        - observable.info_rate_daily
        - observable.peak_adoption_rate
    """

    def __init__(self) -> None:
        """
        Initialize AdoptionAggregator module.
        """
        super().__init__(name="AdoptionAggregator", dependencies=["SocialNetworkEngine", "InfoDiffusion", "SocialInfluenceAdoption", "DropoutAndFatigue"], tick_rate_days=1, sbi_calibration=False)
        pass

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        """
        Compute and record daily observables.

        Args:
            state: Simulation state
            buffers: Buffers dict
            params: Module params
            t: Current day

        Returns:
            None
        """
        N = state["N"]
        # Adoption rate
        ar = float(np.mean(state["adoption_state"])) if N > 0 else 0.0
        state["observables"]["adoption_rate_daily"].append(ar)
        # Adoption rate by group
        group_field = state["params"].get("inequality_group_field", "age_group")
        if group_field not in ["age_group", "occupation"]:
            group_field = "age_group"
        if group_field == "age_group":
            groups = state["age_group"]
        else:
            groups = state["occupation"]
        adop = state["adoption_state"]
        df_tmp = pd.DataFrame({"group": groups, "adopt": adop.astype(int)})
        by_group = df_tmp.groupby("group")["adopt"].mean().to_dict()
        state["observables"]["adoption_rate_by_group"].append({str(k): float(v) for k, v in by_group.items()})
        # Final adoption rate
        state["observables"]["final_adoption_rate"] = ar
        # Time to 50 percent adoption
        if state["observables"]["time_to_50_percent_adoption"] is None:
            if ar >= 0.5:
                state["observables"]["time_to_50_percent_adoption"] = t
        # Info rate
        ir = float(np.mean(state["informed"])) if N > 0 else 0.0
        state["observables"]["info_rate_daily"].append(ir)
        # Rb
        window = int(get_param(state, "Rb_window", 3))
        new_adopters_today = len(state["events"]["adopted_today"])
        state["observables"]["new_adopters_history"].append(new_adopters_today)
        if len(state["observables"]["adoption_rate_daily"]) >= 2:
            adopters_prev = max(1, int(state["observables"]["adoption_rate_daily"][-2] * N))
        else:
            adopters_prev = max(1, int(ar * N))
        new_in_window = sum(state["observables"]["new_adopters_history"][-window:])
        rb_val = float(new_in_window) / float(adopters_prev)
        state["observables"]["Rb_series"].append(rb_val)
        # Churn
        if len(state["observables"]["adoption_rate_daily"]) >= 2:
            adopters_prev_for_churn = max(1, int(state["observables"]["adoption_rate_daily"][-2] * N))
        else:
            adopters_prev_for_churn = max(1, int(ar * N))
        drops = len(state["events"]["dropped_today"])
        churn = float(drops) / float(adopters_prev_for_churn)
        state["observables"]["churn_rate_daily"].append(churn)
        # Peak adoption rate
        state["observables"]["peak_adoption_rate"] = max(state["observables"]["peak_adoption_rate"], ar)
        # Assortativity appended in SocialNetworkEngine daily list
        # Inequality by group (difference max - min)
        if by_group:
            vals = list(by_group.values())
            state["observables"]["inequality_of_adoption"].append(float(max(vals) - min(vals)))
        else:
            state["observables"]["inequality_of_adoption"].append(0.0)
        # Mean exposures before adoption
        exp_records = state["metrics_aux"].get("exposures_before_adoption", [])
        if exp_records:
            state["observables"]["mean_exposures_before_adoption"] = float(np.mean(exp_records))
        else:
            state["observables"]["mean_exposures_before_adoption"] = 0.0
        # Policy effect size approximate: slope post vs pre around mandate
        mandate_day = int(get_param(state, "mandate_start_day", 10))
        if len(state["observables"]["adoption_rate_daily"]) > mandate_day + 1:
            pre = state["observables"]["adoption_rate_daily"][:mandate_day]
            post = state["observables"]["adoption_rate_daily"][mandate_day:]
            if len(pre) >= 2 and len(post) >= 2:
                slope_pre = pre[-1] - pre[0]
                slope_post = post[-1] - post[0]
                state["observables"]["policy_effect_size"] = float(slope_post - slope_pre)
        self.last_buffer_snapshot = {"observables": copy.deepcopy(state["observables"])}
        pass


# ==============
# Entities (POJO)
# ==============

class Person:
    """
    Represents an individual agent with attributes and dynamic state in the simulation.

    This class serves primarily as documentation; behaviors are implemented via vectorized arrays in Simulation.state.
    Attributes:
        id: Unique agent identifier
        age_group: Age group label
        occupation: Occupation label
        risk_perception: Float [0,1]
    """

    def __init__(self, id: int, age_group: str, occupation: str, risk_perception: float) -> None:
        """
        Initialize a Person.

        Args:
            id: Agent ID
            age_group: Age group string
            occupation: Occupation string
            risk_perception: Float [0,1]

        Returns:
            None
        """
        self.id = id
        self.age_group = age_group
        self.occupation = occupation
        self.risk_perception = float(risk_perception)
        pass


class SocialNetwork:
    """
    Multiplex social network representation.

    Attributes:
        adjacency: Dict[int, Dict[str, List[int]]]
    """

    def __init__(self, adjacency: Dict[int, Dict[str, List[int]]]) -> None:
        """
        Initialize SocialNetwork.

        Args:
            adjacency: Nested adjacency mapping per layer.

        Returns:
            None
        """
        self.adjacency = adjacency or {}
        pass


class PublicHealthAuthority:
    """
    Public health authority for policy control.
    Behaviors are embedded in PolicyAndMessaging module.
    """

    def __init__(self) -> None:
        """
        Initialize with default configuration, parameters will be fetched from registry.
        """
        pass


class MediaChannel:
    """
    Media channel broadcasting messages.
    Behaviors are embedded in PolicyAndMessaging and InfoDiffusion modules.
    """

    def __init__(self) -> None:
        """
        Initialize with defaults; parameters handled through PolicyAndMessaging.
        """
        pass


class SimulationEnvironment:
    """
    Simulation environment and runtime control.

    Attributes:
        current_day: Current simulation day
        max_steps: Max steps to run
        rng_seed: Random seed
        time_step_length_days: Step length in days
        termination_condition: Optional termination function
    """

    def __init__(self, rng_seed: int = 42, max_steps: int = 40, time_step_length_days: int = 1) -> None:
        """
        Initialize the environment.

        Args:
            rng_seed: Random seed
            max_steps: Max steps (days)
            time_step_length_days: Step size

        Returns:
            None
        """
        self.current_day = 0
        self.max_steps = max_steps
        self.rng_seed = rng_seed
        self.time_step_length_days = time_step_length_days
        self.termination_condition = None
        pass


# ===========================
# Simulation and Orchestrator
# ===========================

class Simulation:
    """
    Main simulation class managing state, modules, scheduler, and IO.

    Methods:
        run(start_day, end_day): Execute simulation loop.
        save_results(path): Save simulation outputs.
        save_module_io(module, path): Save specific module I/O.
        save_all_io(root_dir): Save I/O for all modules.
        evaluate(): Compute evaluation metrics against ground-truth.
        visualize(): Generate plots for main observables.
    """

    def __init__(self, plan: Dict[str, Any], registry: ParameterRegistry, agent_df: pd.DataFrame, network: Dict[int, Dict[str, List[int]]], train_df: pd.DataFrame, artifacts_dir: str = "artifacts") -> None:
        """
        Construct a Simulation instance from plan and data.

        Args:
            plan: Model plan dict
            registry: ParameterRegistry
            agent_df: Agent attributes DataFrame
            network: Multiplex network adjacency
            train_df: Training time series DataFrame
            artifacts_dir: Base directory for artifacts

        Returns:
            None
        """
        self.plan = plan
        self.registry = registry
        self.agent_df = agent_df.copy()
        self.network = network.copy()
        self.train_df = train_df.copy()
        self.artifacts_dir = artifacts_dir
        ensure_dir(self.artifacts_dir)
        self.env = SimulationEnvironment(
            rng_seed=int(self.registry.values.get("rng_seed", 42)),
            max_steps=int(self.registry.values.get("simulation_steps", 40)),
            time_step_length_days=int(self.registry.values.get("time_step_length_days", 1)),
        )
        self.modules: Dict[str, BaseModule] = {}
        self.module_order: List[str] = []
        self.state: Dict[str, Any] = {}
        self.buffers: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.init_state()
        self.init_modules()
        self.build_scheduler()
        pass

    def _map_ids(self, df: pd.DataFrame) -> Tuple[Dict[int, int], List[int]]:
        agent_ids = df["agent_id"].astype(int).tolist()
        id_to_idx = {int(a): i for i, a in enumerate(agent_ids)}
        idx_to_id = agent_ids
        return id_to_idx, idx_to_id

    def init_state(self) -> None:
        """
        Initialize simulation state from data and parameters.

        Returns:
            None
        """
        # Build synthetic if data missing
        N = int(self.registry.values.get("num_agents", 1000))
        df = self.agent_df
        if df.empty:
            # Create synthetic
            ids = list(range(N))
            df = pd.DataFrame({
                "agent_id": ids,
                "age_group": np.random.choice(["Youth", "Adult", "Senior"], size=N, p=[0.3, 0.5, 0.2]),
                "occupation": np.random.choice(["Student", "Blue Collar", "White Collar", "Unemployed"], size=N),
                "risk_perception": np.clip(np.random.beta(2, 5, size=N), 0.0, 1.0),
            })
        else:
            N = len(df)
        # Map IDs to indices
        df = df.sort_values("agent_id").reset_index(drop=True)
        id_to_idx, idx_to_id = self._map_ids(df)
        # Network
        net = self.network
        if not net:
            # Generate small-world network for 'all' layer; distribute neighbors into layers heuristically
            avg_k = int(self.registry.values.get("avg_degree", 8.0))
            rew = float(self.registry.values.get("small_world_rewiring_prob", 0.1))
            if nx is not None:
                g_tmp = nx.watts_strogatz_graph(n=N, k=max(2, avg_k), p=rew, seed=int(self.registry.values.get("rng_seed", 42)))
                adjacency_all = {i: list(g_tmp.neighbors(i)) for i in g_tmp.nodes()}
            else:
                adjacency_all = {i: [j for j in range(N) if j != i and (j - i) % max(2, avg_k//2) == 0] for i in range(N)}
            # Split into layers
            net = {}
            for i in range(N):
                neighs = adjacency_all.get(i, [])
                random.shuffle(neighs)
                # family small fraction
                nF = min(2, len(neighs))
                nW = min(3, max(0, len(neighs) - nF))
                fam = neighs[:nF]
                work = neighs[nF:nF + nW]
                comm = neighs[nF + nW:]
                net[idx_to_id[i]] = {"family": [idx_to_id[x] for x in fam], "work_school": [idx_to_id[x] for x in work], "community": [idx_to_id[x] for x in comm], "all": sorted(list(set([idx_to_id[x] for x in neighs])))}
        # Remap network to index space
        neighbors_layers = {"family": defaultdict(list), "work_school": defaultdict(list), "community": defaultdict(list)}
        neighbors_all = defaultdict(list)
        for id_key, layers in net.items():
            if int(id_key) not in id_to_idx:
                continue
            i = id_to_idx[int(id_key)]
            for layer in ["family", "work_school", "community"]:
                mapped = [id_to_idx[j] for j in layers.get(layer, []) if int(j) in id_to_idx]
                neighbors_layers[layer][i] = list(sorted(set(mapped)))
            all_list = [id_to_idx[j] for j in layers.get("all", []) if int(j) in id_to_idx]
            neighbors_all[i] = list(sorted(set(all_list)))
        # Convert attributes to arrays
        agent_df = df
        agent_df["risk_perception"] = agent_df["risk_perception"].astype(float).clip(0, 1)
        age_group = agent_df["age_group"].astype(str).tolist()
        occupation = agent_df["occupation"].astype(str).tolist()
        risk = agent_df["risk_perception"].astype(float).values
        # Initialize adoption and informed from earliest or from defaults
        def build_state_from_day(day_val: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
            adopt_arr = np.zeros(N, dtype=bool)
            info_arr = np.zeros(N, dtype=bool)
            if not self.train_df.empty:
                if day_val is None:
                    min_day = int(self.train_df["day"].min())
                    sel_day = min_day
                else:
                    sel_day = day_val
                df0 = self.train_df[self.train_df["day"] == sel_day]
                init_map_adopt = {id_to_idx[int(a)]: parse_bool(v) for a, v in zip(df0["agent_id"], df0["wearing_mask"]) if int(a) in id_to_idx}
                init_map_info = {id_to_idx[int(a)]: parse_bool(v) for a, v in zip(df0["agent_id"], df0["received_info"]) if int(a) in id_to_idx}
                for idx in range(N):
                    adopt_arr[idx] = init_map_adopt.get(idx, False)
                    info_arr[idx] = init_map_info.get(idx, False)
            else:
                init_adopt_rate = float(self.registry.values.get("initial_adoption_rate", 0.05))
                init_info_rate = float(self.registry.values.get("initial_informed_rate", 0.2))
                rng_loc = random.Random(int(self.registry.values.get("rng_seed", 42)))
                adopt_arr = np.array([rng_loc.random() < init_adopt_rate for _ in range(N)], dtype=bool)
                info_arr = np.array([rng_loc.random() < init_info_rate for _ in range(N)], dtype=bool)
            return adopt_arr, info_arr

        adopt_init, info_init = build_state_from_day(None)
        # Degrees
        degree_all = np.array([len(neighbors_all[i]) for i in range(N)], dtype=float)
        # Graph for assortativity
        g = None
        if nx is not None:
            g = nx.Graph()
            g.add_nodes_from(range(N))
            for i in range(N):
                for j in neighbors_all[i]:
                    if i < j:
                        g.add_edge(i, j)
        # Stubborn set sampling using risk-based probability
        stubborn_fraction = float(self.registry.values.get("stubborn_fraction", 0.1))
        rng = random.Random(int(self.registry.values.get("rng_seed", 42)))
        # Probability inversely proportional to risk perception
        prob = (1.0 - risk) + 1e-9
        prob = prob / max(1e-9, prob.sum())
        count_stubborn = int(round(stubborn_fraction * N))
        count_stubborn = max(0, min(N, count_stubborn))
        if count_stubborn > 0:
            stubborn_indices = list(np.random.default_rng(int(self.registry.values.get("rng_seed", 42))).choice(np.arange(N), size=count_stubborn, replace=False, p=prob))
        else:
            stubborn_indices = []
        stubborn_set = set(int(x) for x in stubborn_indices)
        # Initialize state dict
        np_rng = np.random.default_rng(int(self.registry.values.get("rng_seed", 42)))
        self.state = {
            "N": N,
            "age_group": age_group,
            "occupation": occupation,
            "risk_perception": risk,
            "adoption_state": adopt_init.copy(),
            "informed": info_init.copy(),
            "time_since_adoption": np.zeros(N, dtype=int),
            "neighbors_layers": neighbors_layers,
            "neighbors_all": neighbors_all,
            "degree_all": degree_all,
            "history": {"frac_overall": deque()},
            "daily": {"assortativity_by_adoption": []},
            "observables": {
                "adoption_rate_daily": [],
                "adoption_rate_by_group": [],
                "info_rate_daily": [],
                "Rb_series": [],
                "churn_rate_daily": [],
                "inequality_of_adoption": [],
                "policy_effect_size": 0.0,
                "mean_exposures_before_adoption": 0.0,
                "peak_adoption_rate": 0.0,
                "final_adoption_rate": 0.0,
                "time_to_50_percent_adoption": None,
                "new_adopters_history": [],
                "convergence_time": None,
            },
            "events": {"adopted_today": [], "dropped_today": []},
            "metrics_aux": {"first_adoption_day": {}, "exposures_before_adoption": []},
            "graph": g,
            "pending_adoptions": defaultdict(list),
            "params": self.registry.get_params("all"),
            "rng": random.Random(int(self.registry.values.get("rng_seed", 42))),
            "np_rng": np_rng,
            "id_to_idx": id_to_idx,
            "idx_to_id": idx_to_id,
        }
        # Initialize time_since_adoption
        for i in range(N):
            if self.state["adoption_state"][i]:
                self.state["time_since_adoption"][i] = 1
        # Buffers
        self.buffers = {"signals": {}, "observables": {}, "events": {}}
        # Results
        self.results = {"observables": self.state["observables"], "daily": self.state["daily"]}
        pass

    def init_modules(self) -> None:
        """
        Instantiate modules.

        Returns:
            None
        """
        self.modules = {
            "SocialNetworkEngine": SocialNetworkEngine(),
            "PolicyAndMessaging": PolicyAndMessaging(),
            "InfoDiffusion": InfoDiffusion(),
            "SocialInfluenceAdoption": SocialInfluenceAdoption(),
            "DropoutAndFatigue": DropoutAndFatigue(),
            "AdoptionAggregator": AdoptionAggregator(),
        }
        pass

    def build_scheduler(self) -> None:
        """
        Build a DAG scheduler order based on module dependencies.

        Returns:
            None
        """
        # Simple topological sort
        deps = {name: set(mod.dependencies) for name, mod in self.modules.items()}
        order: List[str] = []
        resolved: set[str] = set()
        while len(order) < len(self.modules):
            progressed = False
            for name, reqs in deps.items():
                if name in resolved:
                    continue
                if all(r in resolved or r not in self.modules for r in reqs):
                    order.append(name)
                    resolved.add(name)
                    progressed = True
            if not progressed:
                # cycle or unresolved
                remaining = [n for n in self.modules if n not in resolved]
                order.extend(remaining)
                break
        self.module_order = order
        pass

    def reset_observables(self) -> None:
        self.state["observables"] = {
            "adoption_rate_daily": [],
            "adoption_rate_by_group": [],
            "info_rate_daily": [],
            "Rb_series": [],
            "churn_rate_daily": [],
            "inequality_of_adoption": [],
            "policy_effect_size": 0.0,
            "mean_exposures_before_adoption": 0.0,
            "peak_adoption_rate": 0.0,
            "final_adoption_rate": 0.0,
            "time_to_50_percent_adoption": None,
            "new_adopters_history": [],
            "convergence_time": None,
        }
        self.state["daily"] = {"assortativity_by_adoption": []}
        self.results = {"observables": self.state["observables"], "daily": self.state["daily"]}

    def _initialize_states_from_data_day(self, baseline_day: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        N = self.state["N"]
        adopt_arr = np.zeros(N, dtype=bool)
        info_arr = np.zeros(N, dtype=bool)
        id_to_idx = self.state["id_to_idx"]
        if not self.train_df.empty and baseline_day is not None:
            df0 = self.train_df[self.train_df["day"] == baseline_day]
            init_map_adopt = {id_to_idx[int(a)]: parse_bool(v) for a, v in zip(df0["agent_id"], df0["wearing_mask"]) if int(a) in id_to_idx}
            init_map_info = {id_to_idx[int(a)]: parse_bool(v) for a, v in zip(df0["agent_id"], df0["received_info"]) if int(a) in id_to_idx}
            for idx in range(N):
                adopt_arr[idx] = init_map_adopt.get(idx, False)
                info_arr[idx] = init_map_info.get(idx, False)
        else:
            # fall back to existing initial state (already set)
            adopt_arr = self.state["adoption_state"].copy()
            info_arr = self.state["informed"].copy()
        return adopt_arr, info_arr

    def reset_for_window(self, start_day: int) -> None:
        """
        Reset simulator dynamic state for a new evaluation window, initializing baseline from train_df at day start_day-1 if available.

        Args:
            start_day: Start day of the window.

        Returns:
            None
        """
        baseline_day = None
        if not self.train_df.empty:
            days = sorted(self.train_df["day"].unique().tolist())
            # Use day immediately before start_day if available; otherwise use closest <= start_day
            candidates = [d for d in days if d <= start_day]
            baseline_day = max(candidates) if candidates else days[0]
        adopt_init, info_init = self._initialize_states_from_data_day(baseline_day)
        self.state["adoption_state"] = adopt_init.copy()
        self.state["informed"] = info_init.copy()
        self.state["time_since_adoption"] = np.zeros(self.state["N"], dtype=int)
        for i in range(self.state["N"]):
            if self.state["adoption_state"][i]:
                self.state["time_since_adoption"][i] = 1
        self.state["history"]["frac_overall"] = deque()
        self.state["daily"] = {"assortativity_by_adoption": []}
        self.state["events"] = {"adopted_today": [], "dropped_today": []}
        self.state["metrics_aux"] = {"first_adoption_day": {}, "exposures_before_adoption": []}
        self.state["pending_adoptions"] = defaultdict(list)
        # Reset RNGs for reproducibility
        seed = int(self.registry.values.get("rng_seed", 42))
        self.state["rng"] = random.Random(seed)
        self.state["np_rng"] = np.random.default_rng(seed)
        self.reset_observables()

    def step(self, t: int) -> None:
        """
        Execute a single simulation step: run eligible modules, commit buffers to state, and record observables.

        Args:
            t: Current day index

        Returns:
            None
        """
        # Reset buffers for this step
        self.buffers = {"signals": {}, "observables": {}, "events": {}}
        # Update params snapshot in state from registry
        self.state["params"] = self.registry.get_params("all")
        # Process pending adoptions scheduled for today
        pending_today = self.state["pending_adoptions"].pop(t, [])
        self.state["events"]["adopted_today"] = []
        self.state["events"]["dropped_today"] = []
        # Run modules in order, excluding AdoptionAggregator (run post-commit)
        for name in self.module_order:
            if name == "AdoptionAggregator":
                continue
            mod = self.modules[name]
            if self.env.time_step_length_days > 0 and (t % mod.tick_rate_days) != 0:
                continue
            mod.forward(self.state, self.buffers, self.registry.get_params(name), t)
        # Commit phase
        N = self.state["N"]
        # Update informed
        if "informed_flags" in self.buffers["signals"]:
            new_inf = self.buffers["signals"]["informed_flags"]
            self.state["informed"] = np.logical_or(self.state["informed"], new_inf)
        # Immediate adoptions
        if "adoption_flags" in self.buffers["signals"]:
            flags = self.buffers["signals"]["adoption_flags"]
            adopt_now = flags.get("adopt_now_ids", [])
            for i in adopt_now + pending_today:
                if not self.state["adoption_state"][i]:
                    self.state["adoption_state"][i] = True
                    self.state["time_since_adoption"][i] = 0
                    self.state["events"]["adopted_today"].append(i)
                    # Record exposures before first adoption
                    if i not in self.state["metrics_aux"]["first_adoption_day"]:
                        self.state["metrics_aux"]["first_adoption_day"][i] = t
                        exp_l = self.buffers["signals"].get("exposures_by_layer", {})
                        total_exp = float(exp_l.get("family", np.zeros(N))[i] + exp_l.get("work_school", np.zeros(N))[i] + exp_l.get("community", np.zeros(N))[i])
                        self.state["metrics_aux"]["exposures_before_adoption"].append(total_exp)
            # Schedule future adoptions
            for day, lst in flags.get("scheduled", {}).items():
                for i in lst:
                    self.state["pending_adoptions"][int(day)].append(i)
        # Dropouts
        if "dropout_flags" in self.buffers["signals"]:
            for i in self.buffers["signals"]["dropout_flags"]:
                if self.state["adoption_state"][i]:
                    self.state["adoption_state"][i] = False
                    self.state["time_since_adoption"][i] = 0
                    self.state["events"]["dropped_today"].append(i)
        # Increment time since adoption
        for i in range(N):
            if self.state["adoption_state"][i]:
                self.state["time_since_adoption"][i] += 1
        # Run aggregator after commit exactly once
        agg = self.modules.get("AdoptionAggregator")
        if agg:
            agg.forward(self.state, self.buffers, self.registry.get_params("AdoptionAggregator"), t)
        pass

    def run(self, start_day: int, end_day: int) -> None:
        """
        Run simulation over an inclusive window [start_day, end_day].

        Args:
            start_day: Starting day
            end_day: Ending day inclusive

        Returns:
            None
        """
        self.env.current_day = start_day
        convergence_delta = float(self.registry.values.get("convergence_delta_threshold", 0.001))
        convergence_lookback = int(self.registry.values.get("convergence_lookback", 10))
        for t in range(start_day, end_day + 1):
            self.step(t)
            # Convergence check
            obs = self.state["observables"]["adoption_rate_daily"]
            if len(obs) >= convergence_lookback:
                recent = obs[-convergence_lookback:]
                if max(recent) - min(recent) <= convergence_delta:
                    if self.state["observables"]["convergence_time"] is None:
                        self.state["observables"]["convergence_time"] = t
                    break
        pass

    def reset_results(self) -> None:
        """
        Reset observables and daily logs for a fresh run.

        Returns:
            None
        """
        self.reset_observables()
        pass

    def save_results(self, filename: str = "results/simulation_outputs.json") -> None:
        """
        Save simulation outputs to JSON.

        Args:
            filename: Output path.

        Returns:
            None
        """
        safe_json_dump(filename, self.results)
        pass

    def save_module_io(self, module_name: str, path: str) -> None:
        """
        Save last I/O snapshot for a specific module.

        Args:
            module_name: Name of module.
            path: Output path.

        Returns:
            None
        """
        mod = self.modules.get(module_name)
        if not mod:
            log(f"WARNING: Module '{module_name}' not found; cannot save IO.")
            return
        safe_json_dump(path, mod.last_buffer_snapshot)
        pass

    def save_all_io(self, root_dir: str = "artifacts/io") -> None:
        """
        Save IO snapshots for all modules.

        Args:
            root_dir: Base directory to write module IO.

        Returns:
            None
        """
        for name, mod in self.modules.items():
            path = os.path.join(root_dir, f"{name}.json")
            self.save_module_io(name, path)
        pass

    def _aggregate_ground_truth(self, window: Tuple[int, int]) -> Dict[str, Any]:
        """
        Aggregate ground-truth observables from train_df for a specific window.

        Args:
            window: (start, end) days inclusive

        Returns:
            Dictionary of observables series
        """
        res: Dict[str, Any] = {}
        if self.train_df.empty:
            return res
        start, end = window
        df = self.train_df[(self.train_df["day"] >= start) & (self.train_df["day"] <= end)].copy()
        if df.empty:
            return res
        adopt_rate_daily = df.groupby("day")["wearing_mask"].mean().sort_index().tolist()
        info_rate_daily = df.groupby("day")["received_info"].mean().sort_index().tolist()
        res["adoption_rate_daily"] = adopt_rate_daily
        res["info_rate_daily"] = info_rate_daily
        # Churn approximated by transitions aggregated
        df_sorted = df.sort_values(["agent_id", "day"])
        df_sorted["adopt_lag"] = df_sorted.groupby("agent_id")["wearing_mask"].shift(1).fillna(False)
        df_sorted["drop_event"] = (df_sorted["adopt_lag"] & (~df_sorted["wearing_mask"]))
        # compute churn rate as drops / adopters_prev
        churn_series = []
        for d, df_day in df.groupby("day"):
            day_rows = df_sorted[df_sorted["day"] == d]
            drops = day_rows["drop_event"].sum()
            # adopters previous day
            prev_day_rows = df_sorted[df_sorted["day"] == d - 1]
            adopters_prev = prev_day_rows["wearing_mask"].sum()
            if adopters_prev <= 0:
                churn_val = 0.0
            else:
                churn_val = float(drops) / float(adopters_prev)
            churn_series.append(churn_val)
        res["churn_rate_daily"] = churn_series
        # Rb_series ground truth
        Rb_window = int(self.registry.values.get("Rb_window", 3))
        gt_days = sorted(df["day"].unique())
        # compute new adopters per day
        df_sorted["new_adopt_event"] = ((~df_sorted["adopt_lag"]) & (df_sorted["wearing_mask"]))
        new_adopts_by_day = df_sorted.groupby("day")["new_adopt_event"].sum().to_dict()
        adopters_prev_by_day = df_sorted.groupby("day")["wearing_mask"].sum().to_dict()
        rb_series = []
        for d in gt_days:
            window_sum = 0
            for k in range(Rb_window):
                window_sum += int(new_adopts_by_day.get(d - k, 0))
            adopters_prev = int(adopters_prev_by_day.get(d - 1, 0))
            rb_val = float(window_sum) / float(max(1, adopters_prev))
            rb_series.append(rb_val)
        res["Rb_series"] = rb_series
        # time_to_50
        time_to_50 = None
        for idx, val in enumerate(adopt_rate_daily):
            if val >= 0.5:
                time_to_50 = gt_days[idx]
                break
        res["time_to_50_percent_adoption"] = time_to_50
        return res

    def evaluate(self, window: Tuple[int, int] | None = None) -> Dict[str, Any]:
        """
        Compute evaluation metrics comparing simulated observables to ground-truth over a window.

        Args:
            window: (start, end) days inclusive. If None, uses entire span available.

        Returns:
            Metrics dictionary with RMSE, MAE, Brier, and additional metrics.
        """
        # Compute RMSE/MAE on adoption_rate_daily
        if window:
            start, end = window
        else:
            start, end = (0, len(self.state["observables"]["adoption_rate_daily"]) - 1)
        gt = self._aggregate_ground_truth((start, end))
        sim_series = self.state["observables"]["adoption_rate_daily"]
        # Align series lengths
        if gt and "adoption_rate_daily" in gt:
            y_true = np.array(gt["adoption_rate_daily"], dtype=float)
            y_pred = np.array(sim_series[:len(y_true)], dtype=float)
            if len(y_true) > 0 and len(y_pred) > 0:
                m = min(len(y_true), len(y_pred))
                y_true = y_true[:m]
                y_pred = y_pred[:m]
                rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                mae = float(np.mean(np.abs(y_pred - y_true)))
                brier = float(np.mean((y_pred - y_true) ** 2))
            else:
                rmse = mae = brier = float("nan")
        else:
            rmse = mae = brier = float("nan")
        # Additional metrics
        # TimeTo50Error
        sim_time_to_50 = None
        for idx, val in enumerate(self.state["observables"]["adoption_rate_daily"]):
            if val >= 0.5:
                sim_time_to_50 = start + idx
                break
        gt_time_to_50 = gt.get("time_to_50_percent_adoption", None) if gt else None
        if sim_time_to_50 is not None and gt_time_to_50 is not None:
            time_to_50_err = abs(int(sim_time_to_50) - int(gt_time_to_50))
        else:
            time_to_50_err = float("nan")
        # Rb_MAE
        sim_rb = self.state["observables"]["Rb_series"]
        gt_rb = gt.get("Rb_series", []) if gt else []
        if sim_rb and gt_rb:
            m = min(len(sim_rb), len(gt_rb))
            rb_mae = float(np.mean(np.abs(np.array(sim_rb[:m]) - np.array(gt_rb[:m]))))
        else:
            rb_mae = float("nan")
        # Churn_MAE
        sim_churn = self.state["observables"]["churn_rate_daily"]
        gt_churn = gt.get("churn_rate_daily", []) if gt else []
        if sim_churn and gt_churn:
            m = min(len(sim_churn), len(gt_churn))
            churn_mae = float(np.mean(np.abs(np.array(sim_churn[:m]) - np.array(gt_churn[:m]))))
        else:
            churn_mae = float("nan")
        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "Brier": brier,
            "TimeTo50Error": time_to_50_err,
            "Rb_MAE": rb_mae,
            "Churn_MAE": churn_mae,
            "TransitionFit": {"P01": float("nan"), "P11": float("nan"), "P10": float("nan"), "P00": float("nan")},
        }
        # Save
        safe_json_dump(os.path.join(self.artifacts_dir, "results", "metrics.json"), metrics)
        return metrics

    def visualize(self, show: bool = False, save_path: str = "artifacts/figs/overview.png") -> None:
        """
        Visualize key observables over time.

        Args:
            show: Whether to show the plot interactively.
            save_path: Path to save the figure.

        Returns:
            None
        """
        if plt is None:
            log("Matplotlib not available; skipping visualization.")
            return
        obs = self.state["observables"]
        days = list(range(len(obs["adoption_rate_daily"])))
        plt.figure(figsize=(10, 6))
        plt.plot(days, obs["adoption_rate_daily"], label="Adoption rate")
        plt.plot(days, obs["info_rate_daily"], label="Info rate")
        plt.xlabel("Day")
        plt.ylabel("Rate")
        plt.legend()
        ensure_dir(os.path.dirname(save_path))
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close()
        pass


# ===============
# Plan Validation
# ===============

def validate_plan(plan: Dict[str, Any]) -> None:
    """
    Validate the model plan for structural consistency.

    Checks include:
     - At least 3 modules
     - Inputs/outputs consistency (soft check)
     - Parameters with valid owners and bounds
     - Observables with target_data_field
     - Metrics only reference observables (soft check)
     - run_config steps vs prediction period (soft check)

    Args:
        plan: Model plan dict.

    Returns:
        None; raises ValueError on failure.
    """
    modules = plan.get("modules", [])
    if len(modules) < 3:
        raise ValueError("Plan validation error: At least 3 modules are required.")
    # Parameters checks
    params = plan.get("parameters", [])
    valid_owners = set(["global"] + [m.get("name") for m in modules])
    for p in params:
        if p.get("owner_module", "global") not in valid_owners:
            raise ValueError(f"Parameter owner_module invalid: {p.get('owner_module')}")
        b = p.get("bounds")
        if b is not None:
            if "low" not in b or "high" not in b:
                raise ValueError(f"Parameter bounds invalid for key: {p.get('key')}")
    # Observables
    for ob in plan.get("observables", []):
        if "target_data_field" not in ob or not ob["target_data_field"]:
            raise ValueError(f"Observable missing target_data_field: {ob.get('id')}")
    pass


# ==============
# CLI and Orchestration
# ==============

def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse Namespace with fields: plan_file, param_file, set, calibrator, budget, calib_window, artifacts_dir
    """
    parser = argparse.ArgumentParser(description="Mask Adoption Social Simulation")
    parser.add_argument("--plan-file", type=str, default=os.path.join(PROJECT_ROOT, "model_plan.json"), help="Path to model plan JSON")
    parser.add_argument("--param-file", type=str, default=os.path.join(PROJECT_ROOT, "parameters.json"), help="Path to parameters JSON")
    parser.add_argument("--set", dest="overrides", action="append", help="Parameter override key=value", default=[])
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()))
    parser.add_argument("--budget", type=int, default=20, help="Calibration budget")
    parser.add_argument("--calib-window", type=str, default=None, help="Training window 'start:end'")
    parser.add_argument("--artifacts-dir", type=str, default=os.path.join(PROJECT_ROOT, "artifacts"))
    parser.add_argument("--calib-config", type=str, default=None, help="Optional calibrator config path")
    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main execution: load plan and data, validate, build simulation, calibrate, run, evaluate, and save outputs.

    Returns:
        None
    """
    args = parse_cli()
    ensure_dir(args.artifacts_dir)
    # Load plan
    if not os.path.exists(args.plan_file):
        log(f"WARNING: Plan file not found at {args.plan_file}. Attempting to proceed with minimal default plan.")
        plan = {
            "modules": [
                {"name": "SocialNetworkEngine"},
                {"name": "PolicyAndMessaging"},
                {"name": "InfoDiffusion"},
                {"name": "SocialInfluenceAdoption"},
                {"name": "DropoutAndFatigue"},
                {"name": "AdoptionAggregator"},
            ],
            "parameters": [],
            "observables": [],
            "prediction_period": {"start_day": 30, "end_day": 39},
        }
    else:
        with open(args.plan_file, "r", encoding="utf-8") as f:
            plan = json.load(f)
    try:
        validate_plan(plan)
    except Exception as e:
        log(f"Plan validation warning: {e}. Proceeding cautiously.")

    # Load data
    agent_file = os.path.join(DATA_DIR, "agent_attributes.csv")
    network_file = os.path.join(DATA_DIR, "social_network.json")
    train_file = os.path.join(DATA_DIR, "train_data.csv")
    agent_df = load_agent_attributes(agent_file)
    network = load_social_network(network_file)
    train_df = load_train_data(train_file)

    # Parameter registry
    registry = ParameterRegistry(plan)
    registry.load_param_values(args.param_file)
    ignored = registry.apply_overrides(args.overrides)
    if ignored:
        safe_json_dump(os.path.join(args.artifacts_dir, "ignored_overrides.json"), ignored)
    registry.save_used(os.path.join(args.artifacts_dir, "parameters_used.json"))

    # Build simulation
    sim = Simulation(plan, registry, agent_df, network, train_df, artifacts_dir=args.artifacts_dir)

    # Temporal split: by unique days in train data
    if not train_df.empty:
        days = sorted(train_df["day"].unique().tolist())
        if not days:
            raise RuntimeError("No days available in train data.")
        split_idx = int(len(days) * 0.8)
        if split_idx <= 0 or split_idx >= len(days):
            raise RuntimeError("No validation days available after temporal split.")
        train_window = (int(days[0]), int(days[split_idx - 1]))
        val_window = (int(days[split_idx]), int(days[-1]))
    else:
        # Fallback: arbitrary windows
        train_window = (0, min(19, int(registry.values.get("simulation_steps", 40)) - 1))
        val_window = (train_window[1] + 1, min(train_window[1] + 10, int(registry.values.get("simulation_steps", 40)) - 1))

    # Calibration
    calibrator = get_calibrator(args.calibrator, args.calib_config)
    adapter = SimulationParamsAdapter(registry, plan)
    fitted = calibrator.fit(
        bundle={"plan": plan, "agent_attributes": agent_df, "social_network": network, "train_data": train_df},
        simulator=sim,
        evaluator=evaluate_params,
        train_window=train_window,
        seed=int(registry.values.get("rng_seed", 42)),
        budget=int(args.budget),
        artifacts_dir=os.path.join(args.artifacts_dir, "calibration"),
        params_adapter=adapter,
    )
    # Apply best params
    adapter.apply(sim, fitted)
    safe_json_dump(os.path.join(args.artifacts_dir, "results", "fitted_params_final.json"), fitted.to_dict())

    # Run on validation window and compute metrics
    sim.reset_for_window(val_window[0])
    sim.run(start_day=val_window[0], end_day=val_window[1])
    metrics = sim.evaluate(window=val_window)
    log(f"Validation metrics: {metrics}")

    # Save outputs and module IO
    sim.save_results(os.path.join(args.artifacts_dir, "results", "simulation_outputs.json"))
    sim.save_all_io(os.path.join(args.artifacts_dir, "io"))
    sim.visualize(save_path=os.path.join(args.artifacts_dir, "figs", "overview.png"))

    # SBI calibration observables per module (if supported), with GT
    for name, mod in sim.modules.items():
        if mod.sbi_calibration:
            mod.aggregate_and_save(sim.state, sim.buffers, plan.get("observables", []), os.path.join(args.artifacts_dir, "results"), gt_df=train_df)

    log("Simulation completed.")
    pass


# Execute main for both direct execution and sandbox wrapper invocation

# Execute main for both direct execution and sandbox wrapper invocation
main()