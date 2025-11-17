#!/usr/bin/env python3
import argparse
import ast
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional torch import; SBI is lazily imported inside SBICalibrator._fit_sbi()
try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    HAS_TORCH = False

# Optional POT (Python Optimal Transport) for Wasserstein with OT
try:
    import ot  # type: ignore
    HAS_POT = True
except Exception:
    ot = None  # type: ignore
    HAS_POT = False

# Path Handling Instructions (REQUIRED FORMAT)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
# Provide sane defaults if environment variables are not set
if PROJECT_ROOT is None:
    os.environ["PROJECT_ROOT"] = os.getcwd()
    PROJECT_ROOT = os.environ["PROJECT_ROOT"]
# Default to task-specified data folder
if DATA_PATH is None:
    os.environ["DATA_PATH"] = "data_fitting/supply_data/"
    DATA_PATH = os.environ["DATA_PATH"]
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Filenames (may be overridden by metadata["data_files"])
train_file = os.path.join(DATA_DIR, "train_data.csv")
val_file = os.path.join(DATA_DIR, "val_data.csv")
test_file = os.path.join(DATA_DIR, "test_data.csv")
metadata_file = os.path.join(DATA_DIR, "metadata.json")

# -----------------------
# Global Utilities
# -----------------------


def set_global_seed(seed: int) -> None:
    """
    Set seeds for numpy, random, and torch for deterministic behavior.

    Parameters
    ----------
    seed : int
        Global random seed.
    """
    if seed is None:
        seed = 42
    np.random.seed(seed)
    random.seed(seed)
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)  # type: ignore


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists; create it if it does not exist.

    Parameters
    ----------
    path : str
        Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_json_dumps(obj: Any) -> str:
    """
    Safe JSON dump with sorted keys and indentation.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(obj, sort_keys=True, indent=2, default=str)


def parse_pipeline_items(value: Any) -> List[Dict[str, int]]:
    """
    Parse the `pipeline_items` field from CSV, which may be a JSON string,
    list of dicts or list of tuples/lists, or NaN. Returns a standardized list
    of dicts with keys: {'qty': int, 'remaining': int}.

    Parameters
    ----------
    value : Any
        Raw pipeline_items field.

    Returns
    -------
    List[Dict[str, int]]
        Parsed list of pipeline items.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"none", "nan", "null"}:
            return []
        try:
            items = json.loads(value)
        except Exception:
            try:
                items = ast.literal_eval(value)
            except Exception:
                return []
    else:
        return []

    result = []
    for it in items:
        if isinstance(it, dict):
            q = it.get("qty", it.get("quantity", it.get("q", 0)))
            r = it.get("remaining", it.get("remaining_lead", it.get("lead", 0)))
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            q, r = it[0], it[1]
        else:
            continue
        try:
            q = int(round(float(q)))
            r = int(round(float(r)))
        except Exception:
            continue
        if q < 0:
            q = 0
        result.append({"qty": q, "remaining": r})
    return result


def wasserstein_distance_nd(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute multi-dimensional Wasserstein distance using POT (aligned with GSIM).
    
    This is the same method used in generative-simulations/libs/SUPPLY/env.py
    
    Parameters
    ----------
    X : np.ndarray
        Sample set 1, shape (N, d) where d is the dimension.
    Y : np.ndarray
        Sample set 2, shape (M, d).
    
    Returns
    -------
    float
        Wasserstein distance.
    """
    if not HAS_POT:
        # Fallback: use 1D Wasserstein on flattened data
        X_flat = X.reshape(X.shape[0], -1).mean(axis=1) if X.ndim > 1 else X.ravel()
        Y_flat = Y.reshape(Y.shape[0], -1).mean(axis=1) if Y.ndim > 1 else Y.ravel()
        return wasserstein_1d(X_flat, Y_flat)
    
    N, M = X.shape[0], Y.shape[0]
    if N == 0 or M == 0:
        return float("nan")
    X, Y = X.reshape(N, -1), Y.reshape(M, -1)
    cost_matrix = ot.dist(X, Y, metric="euclidean")  # type: ignore
    a, b = np.ones(N) / N, np.ones(M) / M
    transport_plan = ot.emd(a, b, cost_matrix)  # type: ignore
    return float(np.sum(cost_matrix * transport_plan))


def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the 1D first Wasserstein (Earth Mover's) distance between two samples.

    Parameters
    ----------
    x : np.ndarray
        Sample 1.
    y : np.ndarray
        Sample 2.

    Returns
    -------
    float
        Wasserstein distance.
    """
    x = np.asarray(x).astype(float).ravel()
    y = np.asarray(y).astype(float).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    # remove NaNs
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    n = x_sorted.size
    m = y_sorted.size
    u = np.linspace(0, 1, max(n, m), endpoint=True)
    try:
        x_quant = np.quantile(x_sorted, u, method="linear")
        y_quant = np.quantile(y_sorted, u, method="linear")
    except TypeError:
        x_quant = np.quantile(x_sorted, u, interpolation="linear")
        y_quant = np.quantile(y_sorted, u, interpolation="linear")
    return float(np.mean(np.abs(x_quant - y_quant)))


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared distances between two 1D arrays.

    Parameters
    ----------
    x : np.ndarray
        Array x of shape (n,).
    y : np.ndarray
        Array y of shape (m,).

    Returns
    -------
    np.ndarray
        Matrix of shape (n, m) with squared distances.
    """
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    return (x - y) ** 2


def mmd_gaussian_1d(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None, max_samples: int = 1000) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two 1D samples using a Gaussian kernel.
    Uses subsampling to cap O(n^2) memory/time.

    Parameters
    ----------
    x : np.ndarray
        Sample 1, shape (n,).
    y : np.ndarray
        Sample 2, shape (m,).
    sigma : Optional[float]
        Gaussian kernel bandwidth. If None, use median heuristic on a small subsample.
    max_samples : int
        Maximum number of samples per set used for pairwise computations.

    Returns
    -------
    float
        Unbiased MMD estimate (nonnegative).
    """
    x = np.asarray(x).astype(float).ravel()
    y = np.asarray(y).astype(float).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    if x.size > max_samples:
        idx = rng.choice(x.size, max_samples, replace=False)
        x = x[idx]
    if y.size > max_samples:
        idx = rng.choice(y.size, max_samples, replace=False)
        y = y[idx]
    if sigma is None:
        xs = x if x.size <= 200 else rng.choice(x, 200, replace=False)
        ys = y if y.size <= 200 else rng.choice(y, 200, replace=False)
        xy = np.concatenate([xs, ys])
        pairwise = _pairwise_squared_distances(xy, xy)
        tri = pairwise[np.triu_indices_from(pairwise, k=1)]
        med = np.median(tri) if tri.size > 0 else 1.0
        sigma = math.sqrt(0.5 * med) if med > 0 else 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = np.exp(-gamma * _pairwise_squared_distances(x, x))
    k_yy = np.exp(-gamma * _pairwise_squared_distances(y, y))
    k_xy = np.exp(-gamma * _pairwise_squared_distances(x, y))

    n = x.size
    m = y.size
    if n < 2 or m < 2:
        return float("nan")
    mmd2 = (np.sum(k_xx) - np.sum(np.diag(k_xx))) / (n * (n - 1) + 1e-8)
    mmd2 += (np.sum(k_yy) - np.sum(np.diag(k_yy))) / (m * (m - 1) + 1e-8)
    mmd2 -= 2.0 * np.sum(k_xy) / (n * m + 1e-8)
    return float(max(mmd2, 0.0))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Root Mean Squared Error between two arrays.

    Parameters
    ----------
    x : np.ndarray
        First array.
    y : np.ndarray
        Second array.

    Returns
    -------
    float
        RMSE.
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size != y.size or x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(x: np.ndarray, y: np.ndarray) -> float:
    """
    Mean Absolute Error between two arrays.

    Parameters
    ----------
    x : np.ndarray
        First array.
    y : np.ndarray
        Second array.

    Returns
    -------
    float
        MAE.
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size != y.size or x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x - y)))


def stable_int_from_str(s: str) -> int:
    """
    Produce a stable 32-bit integer from a string using MD5 (deterministic across runs).

    Parameters
    ----------
    s : str

    Returns
    -------
    int
        Stable nonnegative integer in [0, 2**31 - 1]
    """
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


# -----------------------
# Data Structures
# -----------------------


@dataclass
class PipelineItem:
    """
    A single shipment item in the pipeline with a remaining lead time.

    Attributes
    ----------
    qty : int
        Quantity in the shipment.
    remaining : int
        Remaining lead time (in periods) before arrival.
    """

    qty: int
    remaining: int


@dataclass
class TrajectoryData:
    """
    Container for a single trajectory's observed data.

    Attributes
    ----------
    trajectory_id : str
        Identifier of the trajectory.
    actions : np.ndarray
        Action (orders placed) per time step; shape (T,).
    inventory : np.ndarray
        Observed on-hand inventory per time step; shape (T,).
    backlog : np.ndarray
        Observed backlog per time step; shape (T,).
    pipeline_items_by_step : List[List[PipelineItem]]
        Parsed pipeline state per time step as list of PipelineItem.
    init_inventory : int
        Initial on-hand inventory from t=0 row.
    init_backlog : int
        Initial backlog from t=0 row.
    init_pipeline : List[PipelineItem]
        Initial pipeline items from t=0 row.
    episode_length : int
        Number of steps in the trajectory.
    """

    trajectory_id: str
    actions: np.ndarray
    inventory: np.ndarray
    backlog: np.ndarray
    pipeline_items_by_step: List[List[PipelineItem]]
    init_inventory: int
    init_backlog: int
    init_pipeline: List[PipelineItem]
    episode_length: int

    def validate(self) -> None:
        """
        Validate the internal consistency of the trajectory data.

        Raises
        ------
        ValueError
            If arrays are inconsistent in length or invalid values detected.
        """
        T = len(self.actions)
        if T == 0:
            raise ValueError(f"Trajectory {self.trajectory_id} has zero length.")
        if not (len(self.inventory) == T and len(self.backlog) == T):
            raise ValueError(
                f"Trajectory {self.trajectory_id} array lengths mismatch: "
                f"actions={len(self.actions)}, inventory={len(self.inventory)}, backlog={len(self.backlog)}"
            )
        if len(self.pipeline_items_by_step) != T:
            raise ValueError(
                f"Trajectory {self.trajectory_id} pipeline length mismatch: "
                f"{len(self.pipeline_items_by_step)} vs steps {T}"
            )
        if self.episode_length != T:
            logging.debug(
                "Trajectory %s: adjusting episode_length from %d to %d",
                self.trajectory_id,
                self.episode_length,
                T,
            )
            self.episode_length = T
        if self.init_inventory < 0 or self.init_backlog < 0:
            raise ValueError(
                f"Trajectory {self.trajectory_id} invalid negative initial states."
            )


@dataclass
class SimulationTrajResult:
    """
    Simulation results for a single trajectory.

    Attributes
    ----------
    trajectory_id : str
        ID of the trajectory.
    inventory : np.ndarray
        Simulated on-hand inventory per time step.
    backlog : np.ndarray
        Simulated backlog per time step.
    pipeline_len : np.ndarray
        Number of pipeline shipments per time step.
    pipeline_occupancy : List[Dict[int, int]]
        Per step dict from remaining lead time -> total quantity.
    demand_draws : np.ndarray
        Demand realized per time step.
    arrivals : np.ndarray
        Arrivals (total quantity) per time step.
    """

    trajectory_id: str
    inventory: np.ndarray
    backlog: np.ndarray
    pipeline_len: np.ndarray
    pipeline_occupancy: List[Dict[int, int]]
    demand_draws: np.ndarray
    arrivals: np.ndarray


@dataclass
class SimulationResults:
    """
    Collection of simulation results across multiple trajectories.

    Attributes
    ----------
    results_by_traj : Dict[str, SimulationTrajResult]
        Mapping from trajectory ID to simulation results.
    """

    results_by_traj: Dict[str, SimulationTrajResult] = field(default_factory=dict)

    def concatenate(self, order: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """
        Concatenate arrays across trajectories for analysis/evaluation.

        Parameters
        ----------
        order : Optional[Sequence[str]]
            Optional ordering of trajectory_ids to ensure alignment with observed sequences.

        Returns
        -------
        Dict[str, np.ndarray]
            Concatenated arrays for keys: inventory, backlog, pipeline_len, demand_draws, arrivals.
        """
        inv = []
        b = []
        pl = []
        dd = []
        arr = []
        if order is None:
            keys = sorted(self.results_by_traj.keys(), key=lambda x: str(x))
        else:
            keys = [k for k in order if k in self.results_by_traj]
        for tid in keys:
            res = self.results_by_traj.get(tid)
            if res is None:
                continue
            inv.append(res.inventory)
            b.append(res.backlog)
            pl.append(res.pipeline_len)
            dd.append(res.demand_draws)
            arr.append(res.arrivals)
        return dict(
            inventory=np.concatenate(inv) if inv else np.array([]),
            backlog=np.concatenate(b) if b else np.array([]),
            pipeline_len=np.concatenate(pl) if pl else np.array([]),
            demand_draws=np.concatenate(dd) if dd else np.array([]),
            arrivals=np.concatenate(arr) if arr else np.array([]),
        )


# -----------------------
# Demand Models
# -----------------------


class DemandModel:
    """
    Base class for demand models.

    Subclasses must implement reset(seed: int) and sample(t: int) -> int.
    """

    def __init__(
        self,
        seasonal_amplitude: float = 0.0,
        seasonal_period: int = 12,
        demand_noise_scale: float = 0.0,
    ) -> None:
        """
        Initialize base demand model.

        Parameters
        ----------
        seasonal_amplitude : float
            Amplitude of seasonal component (sinusoidal).
        seasonal_period : int
            Period of seasonality (in time steps).
        demand_noise_scale : float
            Additive Gaussian noise scale applied to realized demand before rounding and clamping.
        """
        self.seasonal_amplitude = max(0.0, float(seasonal_amplitude))
        self.seasonal_period = max(2, int(round(seasonal_period)))
        self.demand_noise_scale = max(0.0, float(demand_noise_scale))
        self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the internal RNG state.

        Parameters
        ----------
        seed : Optional[int]
            Seed for NumPy RNG.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def seasonality(self, t: int) -> float:
        """
        Compute deterministic seasonal component at time t.

        Parameters
        ----------
        t : int
            Time step.

        Returns
        -------
        float
            Seasonal adjustment.
        """
        if self.seasonal_amplitude <= 0.0:
            return 0.0
        return self.seasonal_amplitude * math.sin(2.0 * math.pi * (t % self.seasonal_period) / self.seasonal_period)

    def noise(self) -> float:
        """
        Draw additive demand noise.

        Returns
        -------
        float
            Noise value.
        """
        if self.demand_noise_scale <= 0.0:
            return 0.0
        return float(self.rng.normal(0.0, self.demand_noise_scale))

    def sample(self, t: int) -> int:
        """
        Sample demand at time t. Must be implemented by subclasses.

        Parameters
        ----------
        t : int
            Time step.

        Returns
        -------
        int
            Demand quantity (non-negative integer).
        """
        raise NotImplementedError


class PoissonDemandModel(DemandModel):
    """
    Poisson demand with optional AR(1) noise on rate and seasonality.
    """

    def __init__(
        self,
        lam: float,
        rate_ar1_phi: float = 0.0,
        rate_noise_sigma: float = 0.0,
        seasonal_amplitude: float = 0.0,
        seasonal_period: int = 12,
        demand_noise_scale: float = 0.0,
    ) -> None:
        """
        Initialize Poisson demand model.

        Parameters
        ----------
        lam : float
            Base rate (lambda).
        rate_ar1_phi : float
            AR(1) coefficient for latent rate noise (stability |phi|<1 enforced).
        rate_noise_sigma : float
            Std dev of AR(1) innovations added to rate.
        seasonal_amplitude : float
            Seasonality amplitude.
        seasonal_period : int
            Seasonality period.
        demand_noise_scale : float
            Additive Gaussian noise to realized demand before rounding.
        """
        super().__init__(seasonal_amplitude, seasonal_period, demand_noise_scale)
        self.lam = max(0.0, float(lam))
        self.rate_ar1_phi = float(np.clip(rate_ar1_phi, -0.99, 0.99))
        self.rate_noise_sigma = max(0.0, float(rate_noise_sigma))
        self._z = 0.0

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._z = 0.0

    def sample(self, t: int) -> int:
        seasonal = self.seasonality(t)
        eps = self.rng.normal(0.0, self.rate_noise_sigma) if self.rate_noise_sigma > 0 else 0.0
        self._z = self.rate_ar1_phi * self._z + eps
        rate = max(1e-6, self.lam + seasonal + self._z)
        demand = self.rng.poisson(rate)
        demand = demand + self.noise()
        return int(max(0, round(demand)))


class NegativeBinomialDemandModel(DemandModel):
    """
    Negative Binomial demand model with mean parameterization via (r, p).

    Notes
    -----
    The NumPy RNG uses nbinom(n, p) with number of failures n, success prob p,
    returning number of successes until n failures occur. We map to mean demand
    by controlling (r, p) appropriately.
    """

    def __init__(
        self,
        r: float,
        p: float,
        seasonal_amplitude: float = 0.0,
        seasonal_period: int = 12,
        demand_noise_scale: float = 0.0,
    ) -> None:
        """
        Initialize Negative Binomial demand.

        Parameters
        ----------
        r : float
            Number of failures (shape), > 0.
        p : float
            Success probability in (0,1).
        seasonal_amplitude : float
            Seasonality amplitude.
        seasonal_period : int
            Seasonality period.
        demand_noise_scale : float
            Additive Gaussian noise to realized demand before rounding.
        """
        super().__init__(seasonal_amplitude, seasonal_period, demand_noise_scale)
        self.r = max(1e-3, float(r))
        self.p = float(np.clip(p, 1e-3, 1.0 - 1e-3))

    def sample(self, t: int) -> int:
        seasonal = max(0.0, self.seasonality(t))
        mean = (self.r * (1.0 - self.p) / self.p) + seasonal
        p_eff = float(self.r / max(self.r + mean, 1e-6))
        p_eff = float(np.clip(p_eff, 1e-3, 1 - 1e-3))
        demand = self.rng.negative_binomial(self.r, p_eff)
        demand = demand + self.noise()
        return int(max(0, round(demand)))


class AR1DemandModel(DemandModel):
    """
    AR(1)-Normal latent demand, truncated at 0 and rounded to integer.

    demand_t = max(0, round(mu + z_t + seasonal + noise)), z_t = phi * z_{t-1} + eps_t
    """

    def __init__(
        self,
        mu: float,
        phi: float,
        sigma: float,
        seasonal_amplitude: float = 0.0,
        seasonal_period: int = 12,
        demand_noise_scale: float = 0.0,
    ) -> None:
        """
        Initialize AR(1) demand model.

        Parameters
        ----------
        mu : float
            Unconditional mean component.
        phi : float
            AR(1) coefficient (|phi| < 1 enforced).
        sigma : float
            Innovation std dev (>0).
        seasonal_amplitude : float
            Seasonality amplitude.
        seasonal_period : int
            Seasonality period.
        demand_noise_scale : float
            Additive Gaussian noise.
        """
        super().__init__(seasonal_amplitude, seasonal_period, demand_noise_scale)
        self.mu = max(0.0, float(mu))
        self.phi = float(np.clip(phi, -0.99, 0.99))
        self.sigma = max(1e-6, float(sigma))
        self._z = 0.0

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self._z = 0.0

    def sample(self, t: int) -> int:
        eps = self.rng.normal(0.0, self.sigma)
        self._z = self.phi * self._z + eps
        seasonal = self.seasonality(t)
        latent = self.mu + self._z + seasonal + self.noise()
        return int(max(0, round(latent)))


# -----------------------
# Inventory Node
# -----------------------


class InventoryNode:
    """
    Inventory node modeling on-hand inventory, backlog, and pipeline with lead times.

    The event order per time step:
      1) Receive arrivals based on remaining lead time and arrival convention.
      2) Fulfill backlog first, then current demand from inventory.
      3) Update backlog with unmet demand.
      4) Add today's order to pipeline at lead time L.
      5) Decrement remaining lead times by 1 for all pipeline shipments.

    Arrival convention (<= threshold semantics):
      - deliver_at_remaining_0: shipments with remaining <= 0 arrive at step start.
      - deliver_at_remaining_1: shipments with remaining <= 1 arrive at step start.
    """

    def __init__(
        self,
        lead_time: int,
        arrival_convention: str = "deliver_at_remaining_1",
        pipeline_loss_prob: float = 0.0,
        pipeline_initial_bias: float = 0.0,
        initial_state_noise_sigma: float = 0.0,
        event_order: str = "arrivals_first",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Initialize the inventory node.

        Parameters
        ----------
        lead_time : int
            Lead time L for new orders.
        arrival_convention : str
            "deliver_at_remaining_0" or "deliver_at_remaining_1".
        pipeline_loss_prob : float
            Probability that a unit in a shipment is lost before arrival. Applied as binomial thinning.
        pipeline_initial_bias : float
            Bias added to initial pipeline quantities for robustness in calibration.
        initial_state_noise_sigma : float
            Additive Gaussian noise applied to initial inventory and backlog during reset.
        event_order : str
            "arrivals_first" or "demand_first": toggles event sequencing ambiguity across datasets.
        rng : Optional[np.random.Generator]
            NumPy random generator for deterministic behavior.
        """
        self.lead_time = max(1, int(round(lead_time)))
        if arrival_convention not in {"deliver_at_remaining_0", "deliver_at_remaining_1"}:
            raise ValueError(
                "arrival_convention must be 'deliver_at_remaining_0' or 'deliver_at_remaining_1'"
            )
        self.arrival_convention = arrival_convention
        self.pipeline_loss_prob = float(np.clip(pipeline_loss_prob, 0.0, 1.0))
        self.pipeline_initial_bias = float(pipeline_initial_bias)
        self.initial_state_noise_sigma = max(0.0, float(initial_state_noise_sigma))
        if event_order not in {"arrivals_first", "demand_first"}:
            raise ValueError("event_order must be 'arrivals_first' or 'demand_first'")
        self.event_order = event_order
        self.rng = rng if rng is not None else np.random.default_rng()

        # Internal state
        self.inventory = 0
        self.backlog = 0
        self.pipeline: List[PipelineItem] = []
        self.t = 0

    def reset(
        self,
        init_inventory: int,
        init_backlog: int,
        init_pipeline: List[PipelineItem],
    ) -> None:
        """
        Reset the node state to provided initial values with optional noise and bias.

        Parameters
        ----------
        init_inventory : int
            Initial on-hand inventory.
        init_backlog : int
            Initial backlog.
        init_pipeline : List[PipelineItem]
            Initial pipeline items with remaining lead time.

        Notes
        -----
        - Applies Gaussian noise to initial inventory/backlog if initial_state_noise_sigma > 0.
        - Applies pipeline_initial_bias to pipeline quantities, preserving non-negativity.
        """
        inv = max(0.0, float(init_inventory))
        back = max(0.0, float(init_backlog))
        if self.initial_state_noise_sigma > 0:
            inv += float(self.rng.normal(0.0, self.initial_state_noise_sigma))
            back += float(self.rng.normal(0.0, self.initial_state_noise_sigma))
        self.inventory = int(max(0, round(inv)))
        self.backlog = int(max(0, round(back)))

        self.pipeline = []
        for item in init_pipeline:
            qty = int(max(0, round(item.qty + self.pipeline_initial_bias)))
            rem = int(max(0, round(item.remaining)))
            if qty > 0:
                self.pipeline.append(PipelineItem(qty=qty, remaining=rem))
        self.t = 0

    def _arrive_shipments(self) -> int:
        """
        Process arrivals based on arrival convention (<= threshold semantics).

        Returns
        -------
        int
            Total quantity arrived this step.
        """
        thresh = 0 if self.arrival_convention == "deliver_at_remaining_0" else 1
        arrivals = []
        remaining_pipeline = []
        for item in self.pipeline:
            if item.remaining <= thresh:
                arrivals.append(item)
            else:
                remaining_pipeline.append(item)
        self.pipeline = remaining_pipeline

        total_arrived = 0
        for it in arrivals:
            if self.pipeline_loss_prob > 0.0:
                keep = self.rng.binomial(it.qty, 1.0 - self.pipeline_loss_prob)
            else:
                keep = it.qty
            if keep > 0:
                total_arrived += keep
        self.inventory += total_arrived
        return total_arrived

    @staticmethod
    def _fulfill(inventory: int, backlog: int, demand: int) -> Tuple[int, int]:
        """
        Fulfill backlog first, then current demand from inventory.

        Parameters
        ----------
        inventory : int
            Available inventory at step start after arrivals.
        backlog : int
            Existing backlog at step start.
        demand : int
            New demand arriving this step.

        Returns
        -------
        Tuple[int, int]
            Updated (inventory, backlog) after fulfillment of backlog and demand.
        """
        # Serve backlog first
        if backlog > 0 and inventory > 0:
            serve_backlog = min(inventory, backlog)
            backlog -= serve_backlog
            inventory -= serve_backlog
        # Serve current demand
        if demand > 0 and inventory > 0:
            serve_demand = min(inventory, demand)
            demand -= serve_demand
            inventory -= serve_demand
        backlog += demand
        return int(max(0, inventory)), int(max(0, backlog))

    def _decrement_pipeline(self) -> None:
        """
        Decrement remaining lead times by 1 for all pipeline items, clamping at 0.
        """
        for it in self.pipeline:
            it.remaining = int(max(0, it.remaining - 1))

    def _add_order_to_pipeline(self, order_qty: int) -> None:
        """
        Append today's order to the pipeline at position lead_time.

        Parameters
        ----------
        order_qty : int
            Quantity ordered today.
        """
        q = int(max(0, round(order_qty)))
        if q <= 0:
            return
        self.pipeline.append(PipelineItem(qty=q, remaining=int(self.lead_time)))

    def step(self, action: int, demand: int) -> Tuple[int, int, int]:
        """
        Execute a single simulation step.

        Parameters
        ----------
        action : int
            Order quantity placed today.
        demand : int
            Customer demand realized today.

        Returns
        -------
        Tuple[int, int, int]
            (inventory, backlog, arrivals) after the step.
        """
        arrivals = 0
        if self.event_order == "arrivals_first":
            arrivals = self._arrive_shipments()
            self.inventory, self.backlog = self._fulfill(self.inventory, self.backlog, int(max(0, demand)))
        else:
            # demand first, then arrivals, then fulfill backlog with arrivals
            self.inventory, self.backlog = self._fulfill(self.inventory, self.backlog, int(max(0, demand)))
            arrivals = self._arrive_shipments()
            # After arrivals, try to clear backlog further if possible (post-arrival catch-up)
            if self.backlog > 0 and self.inventory > 0:
                serve = min(self.inventory, self.backlog)
                self.inventory -= serve
                self.backlog -= serve
        self._add_order_to_pipeline(int(max(0, action)))
        self._decrement_pipeline()
        self.t += 1
        return self.inventory, self.backlog, arrivals

    def pipeline_occupancy_by_remaining(self) -> Dict[int, int]:
        """
        Aggregate pipeline quantities by remaining lead times.

        Returns
        -------
        Dict[int, int]
            Mapping from remaining lead time to total quantity.
        """
        occ: Dict[int, int] = {}
        for it in self.pipeline:
            occ[it.remaining] = occ.get(it.remaining, 0) + it.qty
        return occ


# -----------------------
# Data Loading and Synthetic Generation
# -----------------------


def generate_synthetic_dataset(
    num_train_traj: int = 30,
    num_val_traj: int = 10,
    num_test_traj: int = 10,
    episode_length: int = 61,
    seed: int = 123,
    constant_action: bool = True,
) -> None:
    """
    Generate a synthetic dataset and metadata to ensure end-to-end execution.

    The synthetic data matches the expected schema:
      - Columns: trajectory_id, time_step, inventory, backlog, action, pipeline_items (JSON).
      - Pipeline items are lists of lists [qty, remaining].
      - The underlying simulator parameters are hidden but used to generate trajectories.

    Parameters
    ----------
    num_train_traj : int
        Number of training trajectories.
    num_val_traj : int
        Number of validation trajectories.
    num_test_traj : int
        Number of test trajectories.
    episode_length : int
        Number of time steps per trajectory.
    seed : int
        Random seed.
    constant_action : bool
        If True, fix action to 4 (aligns with schema). Otherwise, use a heuristic policy.
    """
    ensure_dir(DATA_DIR)
    rng = np.random.default_rng(seed)

    # Ground truth parameters for synthetic generation
    gt_params = dict(
        lead_time_L=3,
        arrival_convention="deliver_at_remaining_1",
        event_order="arrivals_first",
        demand_family="ar1",
        ar1_mu=8.0,
        ar1_phi=0.5,
        ar1_sigma=2.0,
        seasonal_amplitude=2.0,
        seasonal_period=12,
        demand_noise_scale=0.5,
        pipeline_loss_prob=0.02,
        pipeline_initial_bias=0.0,
        initial_state_noise_sigma=0.0,
    )

    def make_dm() -> DemandModel:
        if gt_params["demand_family"] == "ar1":
            dm = AR1DemandModel(
                mu=gt_params["ar1_mu"],
                phi=gt_params["ar1_phi"],
                sigma=gt_params["ar1_sigma"],
                seasonal_amplitude=gt_params["seasonal_amplitude"],
                seasonal_period=gt_params["seasonal_period"],
                demand_noise_scale=gt_params["demand_noise_scale"],
            )
        else:
            dm = PoissonDemandModel(
                lam=8.0,
                rate_ar1_phi=0.3,
                rate_noise_sigma=1.0,
                seasonal_amplitude=gt_params["seasonal_amplitude"],
                seasonal_period=gt_params["seasonal_period"],
                demand_noise_scale=gt_params["demand_noise_scale"],
            )
        dm.reset(seed=seed)
        return dm

    def simulate_one(trajectory_id: int) -> pd.DataFrame:
        dm = make_dm()
        node = InventoryNode(
            lead_time=gt_params["lead_time_L"],
            arrival_convention=gt_params["arrival_convention"],
            pipeline_loss_prob=gt_params["pipeline_loss_prob"],
            pipeline_initial_bias=gt_params["pipeline_initial_bias"],
            initial_state_noise_sigma=gt_params["initial_state_noise_sigma"],
            event_order=gt_params["event_order"],
            rng=rng,
        )
        init_inventory = int(rng.integers(15, 30))
        init_backlog = int(rng.integers(0, 5))
        init_pipeline = []
        num_init_shipments = int(rng.integers(0, 3))
        for _ in range(num_init_shipments):
            rem = int(rng.integers(1, gt_params["lead_time_L"] + 1))
            qty = int(rng.integers(2, 8))
            init_pipeline.append(PipelineItem(qty=qty, remaining=rem))
        node.reset(init_inventory, init_backlog, init_pipeline)

        rows = []
        last_demand = 8
        order_cap = 12
        alpha = 0.6
        forecast = 8.0
        base_stock_S = 30
        safety_stock = 12

        for t in range(episode_length):
            if constant_action:
                action = 4
            else:
                forecast = alpha * last_demand + (1 - alpha) * forecast
                on_order = sum(it.qty for it in node.pipeline)
                desired = base_stock_S - (node.inventory + on_order) + safety_stock
                action = int(np.clip(desired, 0, order_cap))
                if rng.random() < 0.1:
                    action = int(max(0, min(order_cap, action + int(rng.integers(-2, 3)))))

            demand = dm.sample(t)
            last_demand = demand
            inv_before = node.inventory
            backlog_before = node.backlog
            pipeline_items_snapshot = [[it.qty, it.remaining] for it in node.pipeline]
            node.step(action, demand)

            rows.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "time_step": t,
                    "inventory": inv_before,
                    "backlog": backlog_before,
                    "action": action,
                    "pipeline_items": json.dumps(pipeline_items_snapshot),
                    "pipeline_len": len(pipeline_items_snapshot),
                    "t": t,
                }
            )
        return pd.DataFrame(rows)

    splits = [
        ("train", num_train_traj),
        ("val", num_val_traj),
        ("test", num_test_traj),
    ]
    offset = 0
    for split_name, n in splits:
        all_rows = []
        for i in range(n):
            df = simulate_one(trajectory_id=offset + i)
            all_rows.append(df)
        offset += n
        out = pd.concat(all_rows, ignore_index=True)
        out.to_csv(
            os.path.join(DATA_DIR, f"{split_name}_data.csv"), index=False
        )

    metadata = {
        "description": "SupplyChain (Synthetic)",
        "n_trajectories": {
            "train": num_train_traj,
            "val": num_val_traj,
            "test": num_test_traj,
        },
        "trajectory_length": episode_length,
        "seed": seed,
        "state_variables": ["inventory", "backlog", "pipeline_len", "t"],
        "action": 4,
        "data_files": {
            "train": "train_data.csv",
            "val": "val_data.csv",
            "test": "test_data.csv"
        },
        "notes": "Synthetic dataset generated by simulator.",
        "random_seed": seed,
        "episode_length": episode_length
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def generate_ood_dataset(
    num_ood_traj: int = 10,
    episode_length: int = 61,
    seed: int = 321,
    lead_time_L: int = 5,
    demand_family: str = "poisson",
) -> None:
    """
    Generate an OOD dataset with altered parameters (e.g., different lead time) and save to ood_test_data.csv.
    Updates metadata.json to include the 'ood' entry in data_files if possible.
    """
    ensure_dir(DATA_DIR)
    rng = np.random.default_rng(seed)

    # OOD ground truth parameters
    if demand_family == "poisson":
        dm = PoissonDemandModel(lam=5.0, rate_ar1_phi=0.2, rate_noise_sigma=0.5, seasonal_amplitude=0.0, seasonal_period=12, demand_noise_scale=0.3)
    elif demand_family == "negbin":
        dm = NegativeBinomialDemandModel(r=8.0, p=0.6, seasonal_amplitude=0.0, seasonal_period=12, demand_noise_scale=0.3)
    else:
        dm = AR1DemandModel(mu=6.0, phi=0.3, sigma=1.5, seasonal_amplitude=1.0, seasonal_period=12, demand_noise_scale=0.3)
    dm.reset(seed=seed)

    def simulate_one(trajectory_id: int) -> pd.DataFrame:
        node = InventoryNode(
            lead_time=lead_time_L,
            arrival_convention="deliver_at_remaining_1",
            pipeline_loss_prob=0.0,
            pipeline_initial_bias=0.0,
            initial_state_noise_sigma=0.0,
            event_order="arrivals_first",
            rng=rng,
        )
        init_inventory = int(rng.integers(15, 30))
        init_backlog = int(rng.integers(0, 5))
        init_pipeline = []
        num_init_shipments = int(rng.integers(0, 3))
        for _ in range(num_init_shipments):
            rem = int(rng.integers(1, lead_time_L + 1))
            qty = int(rng.integers(2, 8))
            init_pipeline.append(PipelineItem(qty=qty, remaining=rem))
        node.reset(init_inventory, init_backlog, init_pipeline)

        rows = []
        for t in range(episode_length):
            action = 4
            demand = dm.sample(t)
            inv_before = node.inventory
            backlog_before = node.backlog
            pipeline_items_snapshot = [[it.qty, it.remaining] for it in node.pipeline]
            node.step(action, demand)
            rows.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "time_step": t,
                    "inventory": inv_before,
                    "backlog": backlog_before,
                    "action": action,
                    "pipeline_items": json.dumps(pipeline_items_snapshot),
                    "pipeline_len": len(pipeline_items_snapshot),
                    "t": t,
                }
            )
        return pd.DataFrame(rows)

    all_rows = []
    for i in range(num_ood_traj):
        df = simulate_one(trajectory_id=i)
        all_rows.append(df)
    out = pd.concat(all_rows, ignore_index=True)
    ood_path = os.path.join(DATA_DIR, "ood_test_data.csv")
    out.to_csv(ood_path, index=False)

    # Update metadata to reference OOD file if present
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                meta = json.load(f)
        else:
            meta = {
                "description": "SupplyChain",
                "data_files": {}
            }
        dfiles = meta.get("data_files", {})
        dfiles["ood"] = "ood_test_data.csv"
        meta["data_files"] = dfiles
        with open(metadata_file, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logging.warning("Failed to update metadata.json with OOD file: %s", str(e))


def load_metadata() -> Dict[str, Any]:
    """
    Load metadata from metadata.json, generating synthetic data if file is missing.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If metadata cannot be found or created.
    """
    ensure_dir(DATA_DIR)
    if not os.path.exists(metadata_file):
        logging.warning("metadata.json not found. Generating synthetic dataset...")
        generate_synthetic_dataset()
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"metadata.json not found at {metadata_file}")
    with open(metadata_file, "r") as f:
        return json.load(f)


def _resolve_data_paths_from_metadata(metadata: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Resolve data file paths using metadata['data_files'] if present.

    Returns
    -------
    Tuple[str, str, str]
        (train_path, val_path, test_path)
    """
    dfiles = metadata.get("data_files", {})
    train_name = dfiles.get("train", "train_data.csv")
    val_name = dfiles.get("val", "val_data.csv")
    test_name = dfiles.get("test", "test_data.csv")
    return (
        os.path.join(DATA_DIR, train_name),
        os.path.join(DATA_DIR, val_name),
        os.path.join(DATA_DIR, test_name),
    )


def load_dataframes(metadata: Optional[Dict[str, Any]] = None, force_synthetic: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test DataFrames from CSV files.
    If files are missing, synthetic data is generated.

    Parameters
    ----------
    metadata : Optional[Dict[str, Any]]
        Metadata dict to resolve paths.
    force_synthetic : bool
        If True, generate synthetic dataset even if files exist.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    ensure_dir(DATA_DIR)
    if metadata is None:
        metadata = load_metadata()
    train_p, val_p, test_p = _resolve_data_paths_from_metadata(metadata)
    if force_synthetic or not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        logging.warning(
            "One or more data files missing or synthetic forced. Generating synthetic dataset at %s",
            DATA_DIR,
        )
        generate_synthetic_dataset()
        metadata = load_metadata()
        train_p, val_p, test_p = _resolve_data_paths_from_metadata(metadata)
    try:
        train_df = pd.read_csv(train_p)
    except Exception as e:
        raise RuntimeError(f"Failed to read train CSV at {train_p}: {e}")
    try:
        val_df = pd.read_csv(val_p)
    except Exception as e:
        raise RuntimeError(f"Failed to read val CSV at {val_p}: {e}")
    try:
        test_df = pd.read_csv(test_p)
    except Exception as e:
        raise RuntimeError(f"Failed to read test CSV at {test_p}: {e}")
    required_cols = {"trajectory_id", "time_step", "inventory", "backlog", "action", "pipeline_items"}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name}_data.csv is missing required columns: {missing}")
        # Validate pipeline_len if present
        if "pipeline_len" in df.columns:
            parsed_len = df["pipeline_items"].apply(lambda x: len(parse_pipeline_items(x))).astype(int).to_numpy()
            given_len = df["pipeline_len"].astype(int).to_numpy()
            if not np.array_equal(parsed_len, given_len):
                logging.warning("%s_data.csv: pipeline_len does not match parsed pipeline_items; proceeding with parsed values.", name)
    return train_df, val_df, test_df


def build_trajectories(df: pd.DataFrame, episode_length: Optional[int] = None) -> List[TrajectoryData]:
    """
    Build a list of TrajectoryData from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns trajectory_id, time_step, inventory, backlog, action, pipeline_items.
    episode_length : Optional[int]
        If provided, enforce episode length per trajectory (truncate/pad if necessary).

    Returns
    -------
    List[TrajectoryData]
        List of trajectory objects.
    """
    trajectories: List[TrajectoryData] = []
    has_pipeline_len = "pipeline_len" in df.columns
    for tid, g in df.groupby("trajectory_id"):
        g = g.sort_values("time_step")
        actions = g["action"].astype(float).round().astype(int).to_numpy()
        inventory = g["inventory"].astype(float).round().astype(int).to_numpy()
        backlog = g["backlog"].astype(float).round().astype(int).to_numpy()
        pipeline_items_by_step: List[List[PipelineItem]] = []
        for v in g["pipeline_items"].tolist():
            items = parse_pipeline_items(v)
            pipeline_items_by_step.append([PipelineItem(qty=it["qty"], remaining=it["remaining"]) for it in items])
        if has_pipeline_len:
            observed_len = g["pipeline_len"].astype(int).to_numpy()
            parsed_len = np.array([len(it) for it in pipeline_items_by_step], dtype=int)
            if not np.array_equal(observed_len, parsed_len):
                logging.debug("pipeline_len mismatch detected in trajectory %s; using parsed pipeline length.", str(tid))

        T = len(actions)
        if episode_length is not None and episode_length > 0:
            if T > episode_length:
                actions = actions[:episode_length]
                inventory = inventory[:episode_length]
                backlog = backlog[:episode_length]
                pipeline_items_by_step = pipeline_items_by_step[:episode_length]
                T = episode_length
            elif T < episode_length:
                pad = episode_length - T
                # pad inventory/backlog/pipeline with last observed values to avoid zero bias
                last_inv = inventory[-1] if T > 0 else 0
                last_back = backlog[-1] if T > 0 else 0
                last_pipe = pipeline_items_by_step[-1] if T > 0 else []
                actions = np.concatenate([actions, np.zeros(pad, dtype=int)])
                inventory = np.concatenate([inventory, np.full(pad, last_inv, dtype=int)])
                backlog = np.concatenate([backlog, np.full(pad, last_back, dtype=int)])
                pipeline_items_by_step.extend([last_pipe for _ in range(pad)])
                T = episode_length

        init_inventory = int(inventory[0]) if T > 0 else 0
        init_backlog = int(backlog[0]) if T > 0 else 0
        init_pipeline = pipeline_items_by_step[0] if T > 0 else []

        traj = TrajectoryData(
            trajectory_id=str(tid),
            actions=actions,
            inventory=inventory,
            backlog=backlog,
            pipeline_items_by_step=pipeline_items_by_step,
            init_inventory=init_inventory,
            init_backlog=init_backlog,
            init_pipeline=init_pipeline,
            episode_length=T,
        )
        traj.validate()
        trajectories.append(traj)
    return trajectories


# -----------------------
# Holdout
# -----------------------


@dataclass
class HoldoutInfo:
    """
    Holds indices defining the temporal holdout within trajectories.

    Attributes
    ----------
    train_end : int
        End index (exclusive) for training window, equal across trajectories.
    val_start : int
        Start index (inclusive) for validation window.
    total_steps : int
        Total steps per trajectory (assumed equal).
    """

    train_end: int
    val_start: int
    total_steps: int


def holdout_split(trajectories: List[TrajectoryData], train_end: Optional[int] = None) -> HoldoutInfo:
    """
    Determine temporal holdout indices across trajectories.

    Parameters
    ----------
    trajectories : List[TrajectoryData]
        Trajectories to base the holdout on.
    train_end : Optional[int]
        If provided, use this index as training end (exclusive). Otherwise, default to ~80% of length.

    Returns
    -------
    HoldoutInfo
        Dataclass with indices for train and validation windows.
    """
    if not trajectories:
        raise ValueError("No trajectories provided for holdout split.")
    T = trajectories[0].episode_length
    for traj in trajectories:
        if traj.episode_length != T:
            raise ValueError("All trajectories must have the same episode length for temporal holdout.")
    if train_end is None:
        train_end = int(round(0.8 * T))
    train_end = max(1, min(T - 1, train_end))
    val_start = train_end
    return HoldoutInfo(train_end=train_end, val_start=val_start, total_steps=T)


# -----------------------
# Simulator
# -----------------------


@dataclass
class SimulatorConfig:
    """
    Configuration for the Simulator.

    Attributes
    ----------
    lead_time_L : int
        Lead time parameter.
    arrival_convention : str
        Arrival convention: "deliver_at_remaining_0" or "deliver_at_remaining_1".
    demand_family : str
        Demand family: "poisson", "negbin", or "ar1".
    # Demand params:
    poisson_lambda : float
        Poisson base rate (if demand_family == "poisson").
    rate_ar1_phi : float
        AR(1) coefficient for Poisson rate noise (poisson only).
    rate_noise_sigma : float
        Std dev for rate noise (poisson only).
    negbin_r : float
        Negative Binomial r (shape).
    negbin_p : float
        Negative Binomial p (prob).
    ar1_mu : float
        AR(1) mean (ar1 only).
    ar1_phi : float
        AR(1) coefficient (ar1 only).
    ar1_sigma : float
        AR(1) innovation std dev (ar1 only).
    seasonal_amplitude : float
        Demand seasonality amplitude.
    seasonal_period : int
        Demand seasonality period.
    demand_noise_scale : float
        Additive Gaussian noise on realized demand.
    pipeline_loss_prob : float
        Probability of losing units in pipeline before arrival.
    pipeline_initial_bias : float
        Bias added to initial pipeline quantities.
    initial_state_noise_sigma : float
        Noise added to initial inventory/backlog at reset.
    inventory_obs_noise_sigma : float
        Observational noise scale for inventory (used in calibration summaries).
    state_recording : str
        "pre" or "post" for recording states relative to step processing.
    event_order : str
        "arrivals_first" or "demand_first"
    """

    lead_time_L: int
    arrival_convention: str
    demand_family: str
    # Demand parameters
    poisson_lambda: float = 8.0
    rate_ar1_phi: float = 0.0
    rate_noise_sigma: float = 0.0
    negbin_r: float = 10.0
    negbin_p: float = 0.5
    ar1_mu: float = 8.0
    ar1_phi: float = 0.0
    ar1_sigma: float = 1.0
    seasonal_amplitude: float = 0.0
    seasonal_period: int = 12
    demand_noise_scale: float = 0.0
    # Pipeline and state
    pipeline_loss_prob: float = 0.0
    pipeline_initial_bias: float = 0.0
    initial_state_noise_sigma: float = 0.0
    inventory_obs_noise_sigma: float = 0.0
    state_recording: str = "pre"
    event_order: str = "arrivals_first"

    def make_demand_model(self) -> DemandModel:
        """
        Instantiate the appropriate demand model from this config.

        Returns
        -------
        DemandModel
            Initialized demand model.
        """
        if self.demand_family == "poisson":
            dm = PoissonDemandModel(
                lam=float(self.poisson_lambda),
                rate_ar1_phi=float(self.rate_ar1_phi),
                rate_noise_sigma=float(self.rate_noise_sigma),
                seasonal_amplitude=float(self.seasonal_amplitude),
                seasonal_period=int(self.seasonal_period),
                demand_noise_scale=float(self.demand_noise_scale),
            )
        elif self.demand_family == "negbin":
            dm = NegativeBinomialDemandModel(
                r=float(self.negbin_r),
                p=float(self.negbin_p),
                seasonal_amplitude=float(self.seasonal_amplitude),
                seasonal_period=int(self.seasonal_period),
                demand_noise_scale=float(self.demand_noise_scale),
            )
        elif self.demand_family == "ar1":
            dm = AR1DemandModel(
                mu=float(self.ar1_mu),
                phi=float(self.ar1_phi),
                sigma=float(self.ar1_sigma),
                seasonal_amplitude=float(self.seasonal_amplitude),
                seasonal_period=int(self.seasonal_period),
                demand_noise_scale=float(self.demand_noise_scale),
            )
        else:
            raise ValueError(f"Unsupported demand_family: {self.demand_family}")
        return dm


class Simulator:
    """
    Orchestrates rollouts for a set of trajectories using given parameters.

    Supports action playback (from data) or base-stock policy (for counterfactuals),
    but during calibration and validation we use action playback.
    """

    def __init__(self, config: SimulatorConfig, seed: int = 42) -> None:
        """
        Initialize the simulator.

        Parameters
        ----------
        config : SimulatorConfig
            Configuration including demand model and pipeline parameters.
        seed : int
            Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _init_node(self, rng: Optional[np.random.Generator] = None) -> InventoryNode:
        """
        Create a new InventoryNode per trajectory.

        Returns
        -------
        InventoryNode
            A fresh InventoryNode instance configured with current parameters.
        """
        return InventoryNode(
            lead_time=self.config.lead_time_L,
            arrival_convention=self.config.arrival_convention,
            pipeline_loss_prob=self.config.pipeline_loss_prob,
            pipeline_initial_bias=self.config.pipeline_initial_bias,
            initial_state_noise_sigma=self.config.initial_state_noise_sigma,
            event_order=self.config.event_order,
            rng=rng if rng is not None else self.rng,
        )

    def simulate_trajectory(
        self,
        traj: TrajectoryData,
        start: int,
        end: int,
        actions_playback: bool = True,
        policy_params: Optional[Dict[str, float]] = None,
    ) -> SimulationTrajResult:
        """
        Simulate a single trajectory over [start, end).

        Parameters
        ----------
        traj : TrajectoryData
            Source trajectory for initial state and actions.
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).
        actions_playback : bool
            If True, use observed actions; otherwise, base-stock policy.
        policy_params : Optional[Dict[str, float]]
            Parameters for base-stock policy if actions_playback is False.

        Returns
        -------
        SimulationTrajResult
            Simulation results for the specified window.
        """
        if end > traj.episode_length or start < 0 or end <= start:
            raise ValueError(
                f"Invalid window [{start}, {end}) for trajectory length {traj.episode_length}"
            )
        # Per-trajectory RNG for independence and reproducibility
        tr_seed = (self.seed + stable_int_from_str(str(traj.trajectory_id))) % (2**31 - 1)
        trng = np.random.default_rng(tr_seed)
        node = self._init_node(rng=trng)
        node.reset(traj.init_inventory, traj.init_backlog, traj.init_pipeline)
        dm = self.config.make_demand_model()
        dm.reset(seed=tr_seed)

        for t in range(0, start):
            action = int(traj.actions[t]) if actions_playback else 0
            demand = dm.sample(t)
            node.step(action, demand)

        T = end - start
        inv = np.zeros(T, dtype=int)
        back = np.zeros(T, dtype=int)
        pl_len = np.zeros(T, dtype=int)
        occ: List[Dict[int, int]] = []
        demands = np.zeros(T, dtype=int)
        arrivals = np.zeros(T, dtype=int)

        forecast = 0.0
        alpha = 0.5
        S = 50.0
        k = 0.5
        A_max = 50.0
        if not actions_playback and policy_params is not None:
            alpha = float(np.clip(policy_params.get("alpha", 0.5), 0.0, 1.0))
            S = float(np.clip(policy_params.get("S", 50.0), 0.0, 500.0))
            k = float(np.clip(policy_params.get("k", 0.5), 0.0, 10.0))
            A_max = float(np.clip(policy_params.get("A_max", 50.0), 1.0, 1000.0))

        last_demand = 0.0
        for idx, t in enumerate(range(start, end)):
            if self.config.state_recording == "pre":
                inventory_before = node.inventory
                backlog_before = node.backlog
                inv[idx] = inventory_before
                back[idx] = backlog_before
                pl_len[idx] = len(node.pipeline)
                occ.append(node.pipeline_occupancy_by_remaining())

            if actions_playback:
                action = int(traj.actions[t])
            else:
                forecast = alpha * last_demand + (1 - alpha) * forecast
                on_order = sum(item.qty for item in node.pipeline)
                desired = S - (node.inventory + on_order) + k * node.backlog + forecast
                action = int(np.clip(desired, 0, A_max))

            demand = dm.sample(t)
            last_demand = demand

            _, _, arr = node.step(action, demand)
            arrivals[idx] = arr
            demands[idx] = demand

            if self.config.state_recording == "post":
                inv[idx] = node.inventory
                back[idx] = node.backlog
                pl_len[idx] = len(node.pipeline)
                occ.append(node.pipeline_occupancy_by_remaining())

        if self.config.inventory_obs_noise_sigma > 0.0:
            noise_i = trng.normal(0.0, self.config.inventory_obs_noise_sigma, size=inv.shape)
            noise_b = trng.normal(0.0, self.config.inventory_obs_noise_sigma, size=back.shape)
            inv = np.maximum(0, np.round(inv + noise_i).astype(int))
            back = np.maximum(0, np.round(back + noise_b).astype(int))

        return SimulationTrajResult(
            trajectory_id=traj.trajectory_id,
            inventory=inv,
            backlog=back,
            pipeline_len=pl_len,
            pipeline_occupancy=occ,
            demand_draws=demands,
            arrivals=arrivals,
        )

    def rollout(
        self,
        trajectories: List[TrajectoryData],
        start: int,
        end: int,
        actions_playback: bool = True,
        policy_params: Optional[Dict[str, float]] = None,
    ) -> SimulationResults:
        """
        Roll out simulation for multiple trajectories.

        Parameters
        ----------
        trajectories : List[TrajectoryData]
            List of trajectories to simulate.
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).
        actions_playback : bool
            Whether to use action playback.
        policy_params : Optional[Dict[str, float]]
            Policy parameters if not using playback.

        Returns
        -------
        SimulationResults
            Aggregated simulation results.
        """
        results = SimulationResults()
        for traj in trajectories:
            res = self.simulate_trajectory(
                traj,
                start=start,
                end=end,
                actions_playback=actions_playback,
                policy_params=policy_params,
            )
            results.results_by_traj[traj.trajectory_id] = res
        return results


# -----------------------
# Summaries for SBI
# -----------------------


def compute_summary_from_results(sim: SimulationResults) -> np.ndarray:
    """
    Compute a summary statistics vector from simulation results.

    The summary includes:
      - Mean and std of inventory, backlog, pipeline length
      - Lag-1 autocorrelation of inventory and backlog
      - Fraction of stockouts (inventory == 0)
      - Quantiles (25%, 50%, 75%) of inventory and backlog

    Parameters
    ----------
    sim : SimulationResults
        Simulation results over trajectories.

    Returns
    -------
    np.ndarray
        Summary vector.
    """
    agg = sim.concatenate()
    inv = agg["inventory"]
    back = agg["backlog"]
    pl_len = agg["pipeline_len"]

    def acf1(x: np.ndarray) -> float:
        x = x.astype(float)
        if x.size < 2:
            return float("nan")
        x0 = x[:-1]
        x1 = x[1:]
        x0c = x0 - x0.mean()
        x1c = x1 - x1.mean()
        denom = (np.sqrt(np.mean(x0c**2)) * np.sqrt(np.mean(x1c**2)) + 1e-8)
        return float(np.mean(x0c * x1c) / denom)

    def quantiles(x: np.ndarray) -> Tuple[float, float, float]:
        if x.size == 0:
            return float("nan"), float("nan"), float("nan")
        q25 = float(np.quantile(x, 0.25))
        q50 = float(np.quantile(x, 0.50))
        q75 = float(np.quantile(x, 0.75))
        return q25, q50, q75

    summary = [
        float(np.mean(inv)) if inv.size else float("nan"),
        float(np.std(inv)) if inv.size else float("nan"),
        float(np.mean(back)) if back.size else float("nan"),
        float(np.std(back)) if back.size else float("nan"),
        float(np.mean(pl_len)) if pl_len.size else float("nan"),
        float(np.std(pl_len)) if pl_len.size else float("nan"),
        acf1(inv) if inv.size else float("nan"),
        acf1(back) if back.size else float("nan"),
        float(np.mean(inv == 0)) if inv.size else float("nan"),
    ]
    inv_q = quantiles(inv)
    back_q = quantiles(back)
    summary.extend(list(inv_q))
    summary.extend(list(back_q))
    summary = np.array(summary, dtype=float)
    return summary


def compute_observed_summary(
    trajectories: List[TrajectoryData], start: int, end: int
) -> np.ndarray:
    """
    Compute summary statistics from observed trajectories in a given window.

    Parameters
    ----------
    trajectories : List[TrajectoryData]
        Observed data trajectories.
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).

    Returns
    -------
    np.ndarray
        Summary vector.
    """
    sim_like = SimulationResults()
    for traj in trajectories:
        inv = traj.inventory[start:end]
        back = traj.backlog[start:end]
        pl_len = np.array([len(items) for items in traj.pipeline_items_by_step[start:end]], dtype=int)
        # demand/arrivals are latent in observed data; keep zeros (ignored in summary)
        dem = np.zeros_like(inv)
        arr = np.zeros_like(inv)
        res = SimulationTrajResult(
            trajectory_id=traj.trajectory_id,
            inventory=inv.copy(),
            backlog=back.copy(),
            pipeline_len=pl_len,
            pipeline_occupancy=[],
            demand_draws=dem,
            arrivals=arr,
        )
        sim_like.results_by_traj[traj.trajectory_id] = res
    return compute_summary_from_results(sim_like)


# -----------------------
# Calibrator using SBI (NPE)
# -----------------------


@dataclass
class ParameterSpec:
    """
    Specification of a calibratable parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    low : float
        Lower bound for uniform prior.
    high : float
        Upper bound for uniform prior.
    """

    name: str
    low: float
    high: float


class SBICalibrator:
    """
    Calibrates simulator parameters using Simulation-Based Inference (SBI) with Neural Posterior Estimation (NPE).

    Fallback to random search if SBI is unavailable or fails.
    """

    def __init__(
        self,
        trajectories: List[TrajectoryData],
        holdout: HoldoutInfo,
        demand_family: str = "ar1",
        num_simulations: int = 300,
        num_posterior_samples: int = 1000,
        sampling_timeout: int = 30,
        device: str = "cpu",
        seed: int = 42,
        npe_max_epochs: int = 200,
        npe_stop_patience: int = 10,
    ) -> None:
        """
        Initialize SBICalibrator.

        Parameters
        ----------
        trajectories : List[TrajectoryData]
            Training trajectories (full; temporal window specified by holdout).
        holdout : HoldoutInfo
            Temporal split info (train window used for calibration).
        demand_family : str
            Demand family: "poisson", "negbin", or "ar1".
        num_simulations : int
            Number of simulations to generate for SBI.
        num_posterior_samples : int
            Number of posterior samples to draw for parameter estimation.
        sampling_timeout : int
            Timeout in seconds for posterior sampling to avoid stalling.
        device : str
            Torch device; "cpu" is recommended here.
        seed : int
            Global random seed.
        npe_max_epochs : int
            Maximum number of epochs for NPE training.
        npe_stop_patience : int
            Early stopping patience (epochs without improvement).
        """
        if demand_family not in {"poisson", "negbin", "ar1"}:
            raise ValueError("demand_family must be 'poisson', 'negbin', or 'ar1'.")
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")
        if num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be positive.")
        if sampling_timeout <= 0:
            raise ValueError("sampling_timeout must be positive.")

        self.trajectories = trajectories
        self.holdout = holdout
        self.demand_family = demand_family
        self.num_simulations = int(num_simulations)
        self.num_posterior_samples = int(num_posterior_samples)
        self.sampling_timeout = int(sampling_timeout)
        self.device = device
        self.seed = seed
        self.npe_max_epochs = int(max(1, npe_max_epochs))
        self.npe_stop_patience = int(max(1, npe_stop_patience))

        self.prior: Optional[Any] = None
        self.param_names: List[str] = []
        self.param_specs: List[ParameterSpec] = []
        self.observed_summary: Optional[np.ndarray] = None

        # cache observed concatenated arrays for fallback loss
        self._obs_concat_cache: Optional[Dict[str, np.ndarray]] = None

    def _build_param_space(self) -> None:
        """
        Define the parameter space with bounds (at least 10 parameters).
        The mapping to SimulatorConfig is handled in _vector_to_config.

        Included parameters (all active irrespective of demand family to meet >=10 requirement):
          - arrival_convention_flag [0.0, 1.0] -> thresholded to convention
          - event_order_flag [0.0, 1.0] -> arrivals_first vs demand_first
          - state_recording_flag [0.0, 1.0] -> pre vs post
          - seasonal_amplitude [0, 10]
          - seasonal_period [2, 30]
          - demand_noise_scale [0, 5]
          - pipeline_loss_prob [0, 0.2]
          - pipeline_initial_bias [-5, 5]
          - initial_state_noise_sigma [0, 5]
          - inventory_obs_noise_sigma [0, 2]
          Note: lead_time_L is fixed to 2 (not optimized) to match training data.
          - and demand family specific:
            * poisson: poisson_lambda [0, 20], rate_ar1_phi [-0.95, 0.95], rate_noise_sigma [0, 5]
            * negbin: negbin_r [0.1, 50], negbin_p [0.05, 0.95]
            * ar1: ar1_mu [0, 20], ar1_phi [-0.95, 0.95], ar1_sigma [0.01, 10]
        """
        ps: List[ParameterSpec] = [
            ParameterSpec("arrival_convention_flag", 0.0, 1.0),
            ParameterSpec("event_order_flag", 0.0, 1.0),
            ParameterSpec("state_recording_flag", 0.0, 1.0),
            ParameterSpec("seasonal_amplitude", 0.0, 10.0),
            ParameterSpec("seasonal_period", 2.0, 30.0),
            ParameterSpec("demand_noise_scale", 0.0, 5.0),
            ParameterSpec("pipeline_loss_prob", 0.0, 0.2),
            ParameterSpec("pipeline_initial_bias", -5.0, 5.0),
            ParameterSpec("initial_state_noise_sigma", 0.0, 5.0),
            ParameterSpec("inventory_obs_noise_sigma", 0.0, 2.0),
        ]
        if self.demand_family == "poisson":
            ps.extend(
                [
                    ParameterSpec("poisson_lambda", 0.0, 20.0),
                    ParameterSpec("rate_ar1_phi", -0.95, 0.95),
                    ParameterSpec("rate_noise_sigma", 0.0, 5.0),
                ]
            )
        elif self.demand_family == "negbin":
            ps.extend(
                [
                    ParameterSpec("negbin_r", 0.1, 50.0),
                    ParameterSpec("negbin_p", 0.05, 0.95),
                ]
            )
        elif self.demand_family == "ar1":
            ps.extend(
                [
                    ParameterSpec("ar1_mu", 0.0, 20.0),
                    ParameterSpec("ar1_phi", -0.95, 0.95),
                    ParameterSpec("ar1_sigma", 0.01, 10.0),
                ]
            )
        if len(ps) < 10:
            raise RuntimeError("Parameter space must include at least 10 parameters.")
        self.param_specs = ps
        self.param_names = [p.name for p in ps]

    def _vector_to_config(self, theta: np.ndarray) -> SimulatorConfig:
        """
        Map a parameter vector to SimulatorConfig.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector.

        Returns
        -------
        SimulatorConfig
            Config with correct parameter mapping.
        """
        p = {name: float(val) for name, val in zip(self.param_names, theta)}
        # lead_time_L is fixed to 2 (not optimized) to match training data
        lead_time_L = 2
        arrival_convention = "deliver_at_remaining_1" if p["arrival_convention_flag"] >= 0.5 else "deliver_at_remaining_0"
        seasonal_period = int(max(2, round(p["seasonal_period"])))
        event_order = "arrivals_first" if p.get("event_order_flag", 1.0) >= 0.5 else "demand_first"
        state_recording = "pre" if p.get("state_recording_flag", 1.0) >= 0.5 else "post"

        cfg = SimulatorConfig(
            lead_time_L=lead_time_L,
            arrival_convention=arrival_convention,
            demand_family=self.demand_family,
            seasonal_amplitude=float(p["seasonal_amplitude"]),
            seasonal_period=seasonal_period,
            demand_noise_scale=float(p["demand_noise_scale"]),
            pipeline_loss_prob=float(np.clip(p["pipeline_loss_prob"], 0.0, 1.0)),
            pipeline_initial_bias=float(p["pipeline_initial_bias"]),
            initial_state_noise_sigma=float(p["initial_state_noise_sigma"]),
            inventory_obs_noise_sigma=float(p["inventory_obs_noise_sigma"]),
            state_recording=state_recording,
            event_order=event_order,
        )
        if self.demand_family == "poisson":
            cfg.poisson_lambda = float(p["poisson_lambda"])
            cfg.rate_ar1_phi = float(p["rate_ar1_phi"])
            cfg.rate_noise_sigma = float(p["rate_noise_sigma"])
        elif self.demand_family == "negbin":
            cfg.negbin_r = float(p["negbin_r"])
            cfg.negbin_p = float(p["negbin_p"])
        elif self.demand_family == "ar1":
            cfg.ar1_mu = float(p["ar1_mu"])
            cfg.ar1_phi = float(p["ar1_phi"])
            cfg.ar1_sigma = float(p["ar1_sigma"])
        return cfg

    def _simulate_for_theta(self, theta: np.ndarray) -> np.ndarray:
        """
        Perform simulation for a single theta on the training window and return summary vector.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Summary statistics vector for the simulation.
        """
        cfg = self._vector_to_config(theta)
        sim = Simulator(cfg, seed=self.seed)
        res = sim.rollout(
            trajectories=self.trajectories,
            start=0,
            end=self.holdout.train_end,
            actions_playback=True,
        )
        return compute_summary_from_results(res)

    def _simulate_results_for_theta(self, theta: np.ndarray) -> SimulationResults:
        """
        Simulate and return full results for train window for computing distributional losses.

        Parameters
        ----------
        theta : np.ndarray

        Returns
        -------
        SimulationResults
        """
        cfg = self._vector_to_config(theta)
        sim = Simulator(cfg, seed=self.seed)
        res = sim.rollout(
            trajectories=self.trajectories,
            start=0,
            end=self.holdout.train_end,
            actions_playback=True,
        )
        return res

    def _get_observed_concat(self) -> Dict[str, np.ndarray]:
        """
        Concatenate observed arrays for train window for loss computation.

        Returns
        -------
        Dict[str, np.ndarray]
        """
        if self._obs_concat_cache is not None:
            return self._obs_concat_cache
        inv_obs = []
        back_obs = []
        pl_len_obs = []
        for traj in self.trajectories:
            inv_obs.append(traj.inventory[: self.holdout.train_end])
            back_obs.append(traj.backlog[: self.holdout.train_end])
            pl_len_obs.append(np.array([len(x) for x in traj.pipeline_items_by_step[: self.holdout.train_end]], dtype=int))
        out = dict(
            inventory=np.concatenate(inv_obs) if inv_obs else np.array([]),
            backlog=np.concatenate(back_obs) if back_obs else np.array([]),
            pipeline_len=np.concatenate(pl_len_obs) if pl_len_obs else np.array([]),
        )
        self._obs_concat_cache = out
        return out

    def _fit_sbi(self) -> Dict[str, Any]:
        """
        Run the full SBI-NPE pipeline to estimate posterior over parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys: 'optimized_params', 'posterior_samples', 'posterior_mean', 'posterior_std'.
        """
        # Lazy imports to avoid hard dependency at module import time
        try:
            import torch as _torch  # type: ignore
            from sbi import utils as sbi_utils  # type: ignore
            from sbi.inference import NPE, simulate_for_sbi  # type: ignore
        except Exception as e:
            raise ImportError(f"SBI dependencies not available: {e}")

        self._build_param_space()

        # Observed summary from data
        x_obs = compute_observed_summary(
            self.trajectories, start=0, end=self.holdout.train_end
        )
        # impute any NaNs in observed summary with finite fallback statistics
        if not np.all(np.isfinite(x_obs)):
            nan_mask = ~np.isfinite(x_obs)
            x_obs_imp = x_obs.copy()
            x_obs_imp[nan_mask] = 0.0
            x_obs = x_obs_imp
        self.observed_summary = x_obs.copy()
        x_obs_t = _torch.tensor(x_obs.astype(np.float32))  # type: ignore

        # Build prior using sbi public API
        low = _torch.tensor([p.low for p in self.param_specs], dtype=_torch.float32)  # type: ignore
        high = _torch.tensor([p.high for p in self.param_specs], dtype=_torch.float32)  # type: ignore
        self.prior = sbi_utils.BoxUniform(low=low, high=high)  # type: ignore

        def sim_fn(theta_batch: "torch.Tensor") -> "torch.Tensor":  # type: ignore
            """
            Simulator function that handles both single and batch inputs.
            
            Parameters
            ----------
            theta_batch : torch.Tensor
                Parameter tensor of shape (batch_size, n_params) or (n_params,)
            
            Returns
            -------
            torch.Tensor
                Summary statistics tensor of shape (batch_size, summary_dim) or (summary_dim,)
            """
            theta_np = theta_batch.detach().cpu().numpy()
            # Handle both single and batch inputs
            if theta_np.ndim == 1:
                # Single theta vector
                summary = self._simulate_for_theta(theta_np)
                summary = np.asarray(summary, dtype=np.float32)
                return _torch.tensor(summary)  # type: ignore
            else:
                # Batch of theta vectors
                summaries = []
                for theta in theta_np:
                    summary = self._simulate_for_theta(theta)
                    summaries.append(summary)
                summaries_array = np.asarray(summaries, dtype=np.float32)
                return _torch.tensor(summaries_array)  # type: ignore

        logging.info("Starting SBI simulations: num_simulations=%d", self.num_simulations)
        nworkers_local = int(os.environ.get("SBI_NUM_WORKERS", "1"))
        theta_samples, x = simulate_for_sbi(  # type: ignore
            simulator=sim_fn,
            proposal=self.prior,
            num_simulations=self.num_simulations,
            show_progress_bar=True,
            num_workers=max(1, nworkers_local),
        )

        x_np = x.detach().cpu().numpy()
        mask = np.all(np.isfinite(x_np), axis=1)
        theta_samples = theta_samples[mask]
        x = x[mask]
        logging.info("Valid simulations after NaN removal: %d", x.shape[0])

        inference = NPE(prior=self.prior, show_progress_bars=True, device=self.device)  # type: ignore
        density_estimator = inference.append_simulations(theta_samples, x).train(
            max_num_epochs=self.npe_max_epochs,
            stop_after_epochs=self.npe_stop_patience,
            validation_fraction=0.1,
        )
        posterior = inference.build_posterior(density_estimator)

        # Portable timeout using ThreadPoolExecutor
        def sample_posterior():
            return posterior.sample((self.num_posterior_samples,), x=x_obs_t)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(sample_posterior)
                posterior_samples = fut.result(timeout=self.sampling_timeout)
        except concurrent.futures.TimeoutError as te:
            raise TimeoutError(f"Posterior sampling timed out after {self.sampling_timeout} seconds.") from te

        posterior_np = posterior_samples.detach().cpu().numpy()
        posterior_mean = posterior_np.mean(axis=0)
        posterior_std = posterior_np.std(axis=0)

        optimized_params = {name: float(val) for name, val in zip(self.param_names, posterior_mean)}
        return dict(
            optimized_params=optimized_params,
            posterior_samples=posterior_np,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
        )

    def _random_search_fallback(self, num_trials: int = 200) -> Dict[str, Any]:
        """
        Gradient-free fallback: random search over the parameter bounds to minimize a composite loss
        emphasizing Wasserstein distance between simulated and observed distributions.

        Parameters
        ----------
        num_trials : int
            Number of random samples to try.

        Returns
        -------
        Dict[str, Any]
            Dictionary with key 'optimized_params' among others.
        """
        logging.warning("Falling back to random search optimizer with %d trials.", num_trials)
        self._build_param_space()
        x_obs = compute_observed_summary(
            self.trajectories, start=0, end=self.holdout.train_end
        )
        self.observed_summary = x_obs.copy()

        low = np.array([p.low for p in self.param_specs], dtype=float)
        high = np.array([p.high for p in self.param_specs], dtype=float)
        rng = np.random.default_rng(self.seed)

        obs_concat = self._get_observed_concat()
        w_wass = 1.0
        w_mmd = 0.2
        w_mse = 0.1
        patience = 50
        best_loss = float("inf")
        best_theta = None
        no_improve = 0
        t0 = time.time()
        trials = max(num_trials, 2 * self.num_simulations)
        for i in range(trials):
            u = rng.random(low.size)
            theta = low + (high - low) * u
            # Simulate and compute losses
            res = self._simulate_results_for_theta(theta)
            sim_concat = res.concatenate()
            inv_sim = sim_concat["inventory"]
            back_sim = sim_concat["backlog"]

            inv_obs = obs_concat["inventory"]
            back_obs = obs_concat["backlog"]

            wass = 0.5 * (wasserstein_1d(inv_sim, inv_obs) + wasserstein_1d(back_sim, back_obs))
            mmd = 0.5 * (mmd_gaussian_1d(inv_sim, inv_obs) + mmd_gaussian_1d(back_sim, back_obs))
            # MSE on means/stds (summary MSE)
            sim_sum = compute_summary_from_results(res)
            loss_mse = float(np.nanmean((sim_sum - x_obs) ** 2))

            loss = w_wass * wass + w_mmd * mmd + w_mse * loss_mse

            if loss < best_loss:
                best_loss = loss
                best_theta = theta.copy()
                no_improve = 0
            else:
                no_improve += 1

            if (i + 1) % 20 == 0:
                logging.info("Random search progress: %d/%d, best_loss=%.4f", i + 1, trials, best_loss)
            if no_improve >= patience and i > 100:
                logging.info("Early stopping random search at %d due to no improvement for %d trials.", i + 1, patience)
                break

        elapsed = time.time() - t0
        logging.info("Random search completed in %.2f sec. Best loss=%.4f", elapsed, best_loss)
        if best_theta is None:
            # fallback to mid-point if nothing improved
            best_theta = (low + high) / 2.0
        optimized_params = {name: float(val) for name, val in zip(self.param_names, best_theta)}
        return dict(optimized_params=optimized_params, posterior_samples=None)

    def fit(self) -> Dict[str, Any]:
        """
        Run calibration and return optimized parameters dictionary.

        Returns
        -------
        Dict[str, Any]
            'optimized_params' and optionally posterior info.
        """
        try:
            return self._fit_sbi()
        except Exception as e:
            logging.error("SBI fitting failed or unavailable: %s\n%s", str(e), traceback.format_exc())
            return self._random_search_fallback(num_trials=max(200, self.num_simulations * 2))


# -----------------------
# Evaluator
# -----------------------


class Evaluator:
    """
    Evaluates simulation results against observed trajectories using several metrics.

    Metrics:
      - Wasserstein distance (inventory and backlog distributions)
      - MMD (Gaussian kernel) on inventory distributions
      - MSE and RMSE on trajectories (inventory and backlog)
      - Pipeline metrics: MAE on pipeline length, L1 on occupancy vectors
      - POT-based per-time-step Wasserstein/MMD with 200 samples (if POT available)
      - MSE per dimension (inventory, backlog, pipeline_len, t)
    """

    def __init__(self, lead_time_L: int, use_pot: bool = False, mmd_sigma: float = 1.0, sample_size: int = 200) -> None:
        """
        Initialize evaluator with lead time for pipeline occupancy vector length.

        Parameters
        ----------
        lead_time_L : int
            Lead time to interpret pipeline positions 1..L.
        use_pot : bool
            Whether to use POT-based Wasserstein for per-time-step evaluation if available.
        mmd_sigma : float
            Sigma for MMD Gaussian kernel in time-step evaluation.
        sample_size : int
            Number of samples per time step per distribution for POT/MMD evaluation.
        """
        self.lead_time_L = int(max(1, lead_time_L))
        self.use_pot = bool(use_pot and HAS_POT)
        self.mmd_sigma = float(mmd_sigma)
        self.sample_size = int(max(10, sample_size))

    def _occupancy_vector(self, occ_dict: Dict[int, int]) -> np.ndarray:
        """
        Build pipeline occupancy vector of length self.lead_time_L from a dict of remaining->qty.

        Parameters
        ----------
        occ_dict : Dict[int, int]
            Mapping from remaining lead time to total quantity.

        Returns
        -------
        np.ndarray
            Vector of length L with quantities at positions 1..L.
        """
        vec = np.zeros(self.lead_time_L, dtype=float)
        for rem, qty in occ_dict.items():
            if 1 <= rem <= self.lead_time_L:
                vec[rem - 1] += float(qty)
        return vec

    def _per_t_sampling(self, values: np.ndarray, size: int) -> np.ndarray:
        """
        Sample with replacement from values to fixed size.

        Parameters
        ----------
        values : np.ndarray
            1D array of values.
        size : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape (size,). If values is empty or all NaN, returns NaN array.
        """
        values = np.asarray(values).ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.full(size, np.nan, dtype=float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, max(1, values.size), size=size)
        return values[idx].astype(float)

    def _wass_and_mmd_time_series(
        self,
        sim_results: SimulationResults,
        observed: List[TrajectoryData],
        start: int,
        end: int,
    ) -> Dict[str, float]:
        """
        Compute per-time-step Wasserstein (using POT if available) and MMD by sampling.

        Returns
        -------
        Dict[str, float]
            Aggregated metrics across time steps and dims.
        """
        T = end - start
        if T <= 0:
            return {"Wass_ts_inventory": float("nan"), "Wass_ts_backlog": float("nan"), "MMD_ts_inventory": float("nan"), "MMD_ts_backlog": float("nan")}
        wass_inv_list = []
        wass_back_list = []
        mmd_inv_list = []
        mmd_back_list = []
        # Assemble sim arrays aligned to observed per t
        for t_idx in range(T):
            inv_sim_t = []
            back_sim_t = []
            inv_obs_t = []
            back_obs_t = []
            for traj in observed:
                sim_traj = sim_results.results_by_traj.get(traj.trajectory_id)
                if sim_traj is None:
                    continue
                inv_sim_t.append(sim_traj.inventory[t_idx])
                back_sim_t.append(sim_traj.backlog[t_idx])
                inv_obs_t.append(traj.inventory[start + t_idx])
                back_obs_t.append(traj.backlog[start + t_idx])

            inv_sim_samp = self._per_t_sampling(np.array(inv_sim_t), self.sample_size)
            inv_obs_samp = self._per_t_sampling(np.array(inv_obs_t), self.sample_size)
            back_sim_samp = self._per_t_sampling(np.array(back_sim_t), self.sample_size)
            back_obs_samp = self._per_t_sampling(np.array(back_obs_t), self.sample_size)

            # drop NaNs
            inv_sim_samp = inv_sim_samp[np.isfinite(inv_sim_samp)]
            inv_obs_samp = inv_obs_samp[np.isfinite(inv_obs_samp)]
            back_sim_samp = back_sim_samp[np.isfinite(back_sim_samp)]
            back_obs_samp = back_obs_samp[np.isfinite(back_obs_samp)]

            if inv_sim_samp.size == 0 or inv_obs_samp.size == 0 or back_sim_samp.size == 0 or back_obs_samp.size == 0:
                continue

            # Wasserstein
            if self.use_pot and HAS_POT:
                a_inv = np.ones(inv_sim_samp.size) / inv_sim_samp.size
                b_inv = np.ones(inv_obs_samp.size) / inv_obs_samp.size
                a_back = np.ones(back_sim_samp.size) / back_sim_samp.size
                b_back = np.ones(back_obs_samp.size) / back_obs_samp.size
                M_inv = ot.dist(inv_sim_samp.reshape(-1, 1), inv_obs_samp.reshape(-1, 1), metric="euclidean")  # type: ignore
                M_back = ot.dist(back_sim_samp.reshape(-1, 1), back_obs_samp.reshape(-1, 1), metric="euclidean")  # type: ignore
                wass_inv = float(ot.emd2(a_inv, b_inv, M_inv))  # type: ignore
                wass_back = float(ot.emd2(a_back, b_back, M_back))  # type: ignore
            else:
                wass_inv = wasserstein_1d(inv_sim_samp, inv_obs_samp)
                wass_back = wasserstein_1d(back_sim_samp, back_obs_samp)

            # MMD
            mmd_inv = mmd_gaussian_1d(inv_sim_samp, inv_obs_samp, sigma=self.mmd_sigma, max_samples=self.sample_size)
            mmd_back = mmd_gaussian_1d(back_sim_samp, back_obs_samp, sigma=self.mmd_sigma, max_samples=self.sample_size)

            wass_inv_list.append(wass_inv)
            wass_back_list.append(wass_back)
            mmd_inv_list.append(mmd_inv)
            mmd_back_list.append(mmd_back)

        return {
            "Wass_ts_inventory": float(np.nanmean(wass_inv_list)) if len(wass_inv_list) > 0 else float("nan"),
            "Wass_ts_backlog": float(np.nanmean(wass_back_list)) if len(wass_back_list) > 0 else float("nan"),
            "MMD_ts_inventory": float(np.nanmean(mmd_inv_list)) if len(mmd_inv_list) > 0 else float("nan"),
            "MMD_ts_backlog": float(np.nanmean(mmd_back_list)) if len(mmd_back_list) > 0 else float("nan"),
        }

    def compute_metrics(
        self,
        sim_results: SimulationResults,
        observed: List[TrajectoryData],
        start: int,
        end: int,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Parameters
        ----------
        sim_results : SimulationResults
            Simulated results over trajectories.
        observed : List[TrajectoryData]
            Observed trajectories for comparison.
        start : int
            Start index.
        end : int
            End index (exclusive).

        Returns
        -------
        Dict[str, Any]
            Dictionary with metrics.
        """
        order = [traj.trajectory_id for traj in observed]
        concat_sim = sim_results.concatenate(order=order)
        inv_sim = concat_sim["inventory"]
        back_sim = concat_sim["backlog"]
        pl_len_sim = concat_sim["pipeline_len"]

        inv_obs_list = []
        back_obs_list = []
        pl_len_obs_list = []
        occ_l1 = []
        t_list_obs = []
        t_list_sim = []

        T = end - start
        for traj in observed:
            inv_obs_list.append(traj.inventory[start:end])
            back_obs_list.append(traj.backlog[start:end])
            pl_len_obs_list.append(np.array([len(x) for x in traj.pipeline_items_by_step[start:end]], dtype=int))
            t_list_obs.append(np.arange(start, end, dtype=int))
            sim_traj = sim_results.results_by_traj.get(traj.trajectory_id)
            if sim_traj is None:
                continue
            t_list_sim.append(np.arange(0, T, dtype=int))
            for t in range(end - start):
                sim_vec = self._occupancy_vector(sim_traj.pipeline_occupancy[t])
                obs_occ_items = traj.pipeline_items_by_step[start + t]
                obs_occ_dict: Dict[int, int] = {}
                for it in obs_occ_items:
                    rem = int(max(0, it.remaining))
                    qty = int(max(0, it.qty))
                    if rem > 0:
                        obs_occ_dict[rem] = obs_occ_dict.get(rem, 0) + qty
                obs_vec = self._occupancy_vector(obs_occ_dict)
                occ_l1.append(float(np.sum(np.abs(sim_vec - obs_vec))))

        inv_obs = np.concatenate(inv_obs_list) if inv_obs_list else np.array([])
        back_obs = np.concatenate(back_obs_list) if back_obs_list else np.array([])
        pl_len_obs = np.concatenate(pl_len_obs_list) if pl_len_obs_list else np.array([])
        t_obs = np.concatenate(t_list_obs) if t_list_obs else np.array([])
        t_sim = np.concatenate(t_list_sim) if t_list_sim else np.array([])

        metrics: Dict[str, Any] = {}
        
        # GSIM-aligned wass metric: multi-dimensional Wasserstein distance
        # This matches the 'wass' metric in generative-simulations/libs/SUPPLY/env.py
        # Combines inventory and backlog into a 2D state vector
        # This is the ONLY Wasserstein metric that aligns with GSIM
        if inv_sim.size > 0 and back_sim.size > 0 and inv_obs.size > 0 and back_obs.size > 0:
            min_size = min(inv_sim.size, back_sim.size, inv_obs.size, back_obs.size)
            if min_size > 0:
                # Align sizes by truncating to minimum
                inv_sim_aligned = inv_sim[:min_size]
                back_sim_aligned = back_sim[:min_size]
                inv_obs_aligned = inv_obs[:min_size]
                back_obs_aligned = back_obs[:min_size]
                # Combine into 2D state vectors (inventory, backlog)
                sim_states = np.column_stack([inv_sim_aligned, back_sim_aligned])
                obs_states = np.column_stack([inv_obs_aligned, back_obs_aligned])
                metrics["wass"] = wasserstein_distance_nd(sim_states, obs_states)
            else:
                metrics["wass"] = float("nan")
        else:
            metrics["wass"] = float("nan")
        
        metrics["mmd"] = 0.5 * (mmd_gaussian_1d(inv_sim, inv_obs) + mmd_gaussian_1d(back_sim, back_obs))

        def mse_concat(a: np.ndarray, b: np.ndarray) -> float:
            a = np.asarray(a).astype(float)
            b = np.asarray(b).astype(float)
            if a.size == 0 or b.size == 0 or a.size != b.size:
                return float("nan")
            return float(np.mean((a - b) ** 2))

        metrics["MSE_inventory"] = mse_concat(inv_sim, inv_obs)
        metrics["MSE_backlog"] = mse_concat(back_sim, back_obs)
        metrics["RMSE_inventory"] = rmse(inv_sim, inv_obs)
        metrics["RMSE_backlog"] = rmse(back_sim, back_obs)
        metrics["MAE_pipeline_len"] = mae(pl_len_sim, pl_len_obs)
        metrics["PipelineComp_L1"] = float(np.mean(occ_l1)) if len(occ_l1) > 0 else float("nan")

        # Per-dimension MSE (inventory, backlog, pipeline_len, t)
        mse_per_dim = {
            "inventory": metrics["MSE_inventory"],
            "backlog": metrics["MSE_backlog"],
            "pipeline_len": mse_concat(pl_len_sim, pl_len_obs),
            "t": mse_concat(t_sim, t_obs),
        }
        metrics["MSE_per_dimension"] = mse_per_dim

        # Per-time-step metrics (MMD only, wass is computed above as single metric)
        ts_metrics = self._wass_and_mmd_time_series(sim_results, observed, start, end)
        # Only include MMD time-series metrics, not Wasserstein (we only use 'wass' above)
        if "MMD_ts_inventory" in ts_metrics:
            metrics["MMD_ts_inventory"] = ts_metrics["MMD_ts_inventory"]
        if "MMD_ts_backlog" in ts_metrics:
            metrics["MMD_ts_backlog"] = ts_metrics["MMD_ts_backlog"]
        return metrics


# -----------------------
# CLI and Orchestration
# -----------------------


def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Single-Stage Beer Game Simulator with SBI Calibration")
    parser.add_argument("--demand_family", type=str, default="ar1", choices=["poisson", "negbin", "ar1"], help="Demand model family.")
    parser.add_argument("--num_simulations", type=int, default=300, help="Number of simulations for SBI training.")
    parser.add_argument("--num_posterior_samples", type=int, default=1000, help="Number of posterior samples.")
    parser.add_argument("--sampling_timeout", type=int, default=30, help="Timeout for posterior sampling in seconds.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g., 'cpu'.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    parser.add_argument("--train_end", type=int, default=None, help="Train window end index (exclusive). Defaults to 80%% of length.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides metadata).")
    parser.add_argument("--ood_eval", action="store_true", help="Enable OOD evaluation if OOD dataset is available.")
    parser.add_argument("--force_synthetic", action="store_true", help="Force generation and use of synthetic data.")
    parser.add_argument("--use_pot", action="store_true", help="Use POT for per-time-step Wasserstein if installed.")
    parser.add_argument("--mmd_sigma", type=float, default=1.0, help="Sigma for MMD kernel in per-time-step evaluation.")
    parser.add_argument("--sample_size", type=int, default=200, help="Sample size per time step for OT/MMD evaluation.")
    parser.add_argument("--data_path", type=str, default=None, help="Optional override for DATA_PATH.")
    parser.add_argument("--npe_max_epochs", type=int, default=200, help="Max epochs for NPE training.")
    parser.add_argument("--npe_stop_patience", type=int, default=10, help="Early stopping patience (epochs) for NPE.")
    args = parser.parse_args()
    return args


def load_data() -> Tuple[Dict[str, Any], List[TrajectoryData], List[TrajectoryData], List[TrajectoryData]]:
    """
    Load metadata and train/val/test trajectories from disk. Generate synthetic data if necessary.

    Returns
    -------
    Tuple[Dict[str, Any], List[TrajectoryData], List[TrajectoryData], List[TrajectoryData]]
        (metadata, train_trajectories, val_trajectories, test_trajectories)
    """
    metadata = load_metadata()
    ep_len = int(metadata.get("episode_length", metadata.get("trajectory_length", 61)))
    train_df, val_df, test_df = load_dataframes(metadata=metadata, force_synthetic=False)
    train_traj = build_trajectories(train_df, episode_length=ep_len)
    val_traj = build_trajectories(val_df, episode_length=ep_len)
    test_traj = build_trajectories(test_df, episode_length=ep_len)
    return metadata, train_traj, val_traj, test_traj


def build_trajectories_wrapper(metadata: Dict[str, Any], train_traj: List[TrajectoryData], val_traj: List[TrajectoryData], test_traj: List[TrajectoryData]) -> Tuple[List[TrajectoryData], List[TrajectoryData], List[TrajectoryData]]:
    """
    Wrapper maintained for consistency with orchestrator. Currently forwards inputs.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary.
    train_traj : List[TrajectoryData]
        Training trajectories.
    val_traj : List[TrajectoryData]
        Validation trajectories.
    test_traj : List[TrajectoryData]
        Test trajectories.

    Returns
    -------
    Tuple[List[TrajectoryData], List[TrajectoryData], List[TrajectoryData]]
        The same trajectories.
    """
    return train_traj, val_traj, test_traj


def _try_load_ood(metadata: Dict[str, Any]) -> Optional[List[TrajectoryData]]:
    """
    Attempt to load an OOD dataset using common filenames or metadata['data_files']['ood'].

    Returns
    -------
    Optional[List[TrajectoryData]]
        Parsed OOD trajectories if available, else None.
    """
    candidates = []
    dfiles = metadata.get("data_files", {})
    if isinstance(dfiles, dict) and "ood" in dfiles:
        candidates.append(os.path.join(DATA_DIR, dfiles["ood"]))
    # common names
    candidates.extend([
        os.path.join(DATA_DIR, "ood_test_data.csv"),
        os.path.join(DATA_DIR, "test_ood.csv"),
        os.path.join(DATA_DIR, "test_ood_data.csv"),
    ])
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                ep_len = int(metadata.get("episode_length", metadata.get("trajectory_length", 61)))
                return build_trajectories(df, episode_length=ep_len)
            except Exception as e:
                logging.warning("Failed to load OOD dataset from %s: %s", path, str(e))
    return None


def save_results(
    optimized_params: Dict[str, Any],
    metrics_val: Dict[str, Any],
    metrics_test: Dict[str, Any],
    metrics_ood: Optional[Dict[str, Any]],
    posterior_info: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    metadata: Dict[str, Any],
) -> None:
    """
    Save results to JSON files with metadata.

    Parameters
    ----------
    optimized_params : Dict[str, Any]
        Calibrated parameters.
    metrics_val : Dict[str, Any]
        Metrics on validation set.
    metrics_test : Dict[str, Any]
        Metrics on test set.
    metrics_ood : Optional[Dict[str, Any]]
        Metrics for OOD evaluation, if computed.
    posterior_info : Optional[Dict[str, Any]]
        Posterior summary (mean, std) if available.
    args : argparse.Namespace
        CLI arguments for reproducibility.
    metadata : Dict[str, Any]
        Dataset metadata.
    """
    ensure_dir(DATA_DIR)
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "optimized_params": optimized_params,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "metrics_ood": metrics_ood,
        "posterior_summary": {
            "mean": posterior_info.get("posterior_mean").tolist() if posterior_info and posterior_info.get("posterior_mean") is not None else None,
            "std": posterior_info.get("posterior_std").tolist() if posterior_info and posterior_info.get("posterior_std") is not None else None,
            "param_names": posterior_info.get("param_names") if posterior_info else None,
        },
        "args": vars(args),
        "metadata": metadata,
    }
    out_path = os.path.join(DATA_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info("Saved results to %s", out_path)


def main() -> None:
    """
    Orchestrator main function executing the full pipeline:
      parse_cli() → load_data() → build_trajectories() → holdout_split()
      → calibrator.fit() → simulator.rollout() → evaluator.compute_metrics() → save_results()
    """
    args = parse_cli()
    if args.data_path:
        # Override data path if provided
        global DATA_PATH, DATA_DIR, train_file, val_file, test_file, metadata_file
        DATA_PATH = args.data_path
        DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
        train_file = os.path.join(DATA_DIR, "train_data.csv")
        val_file = os.path.join(DATA_DIR, "val_data.csv")
        test_file = os.path.join(DATA_DIR, "test_data.csv")
        metadata_file = os.path.join(DATA_DIR, "metadata.json")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    metadata = load_metadata()
    train_df, val_df, test_df = load_dataframes(metadata=metadata, force_synthetic=args.force_synthetic)
    ep_len = int(metadata.get("episode_length", metadata.get("trajectory_length", 61)))
    train_traj = build_trajectories(train_df, episode_length=ep_len)
    val_traj = build_trajectories(val_df, episode_length=ep_len)
    test_traj = build_trajectories(test_df, episode_length=ep_len)
    train_traj, val_traj, test_traj = build_trajectories_wrapper(metadata, train_traj, val_traj, test_traj)

    seed = args.seed if args.seed is not None else int(metadata.get("random_seed", metadata.get("seed", 42)))
    set_global_seed(seed)
    logging.info("Using global seed: %d", seed)

    holdout = holdout_split(train_traj, train_end=args.train_end)
    logging.info("Temporal holdout: train 0..%d, val %d..%d", holdout.train_end - 1, holdout.val_start, holdout.total_steps - 1)

    calibrator = SBICalibrator(
        trajectories=train_traj,
        holdout=holdout,
        demand_family=args.demand_family,
        num_simulations=args.num_simulations,
        num_posterior_samples=args.num_posterior_samples,
        sampling_timeout=args.sampling_timeout,
        device=args.device,
        seed=seed,
        npe_max_epochs=args.npe_max_epochs,
        npe_stop_patience=args.npe_stop_patience,
    )
    calib_info = calibrator.fit()
    optimized_params = calib_info["optimized_params"]
    posterior_info = {
        "posterior_mean": calib_info.get("posterior_mean"),
        "posterior_std": calib_info.get("posterior_std"),
        "param_names": calibrator.param_names,
    }

    def dict_to_config(params: Dict[str, Any], demand_family: str) -> SimulatorConfig:
        """
        Map an optimized parameter dictionary to a concrete SimulatorConfig.
        """
        arrival_convention = "deliver_at_remaining_1" if float(params.get("arrival_convention_flag", 1.0)) >= 0.5 else "deliver_at_remaining_0"
        event_order = "arrivals_first" if float(params.get("event_order_flag", 1.0)) >= 0.5 else "demand_first"
        state_recording = "pre" if float(params.get("state_recording_flag", 1.0)) >= 0.5 else "post"
        # lead_time_L is fixed to 2 (not optimized) to match training data
        lead_time_L = int(params.get("lead_time_L", 2)) if "lead_time_L" in params else 2
        cfg = SimulatorConfig(
            lead_time_L=lead_time_L,
            arrival_convention=arrival_convention,
            demand_family=demand_family,
            seasonal_amplitude=float(params["seasonal_amplitude"]),
            seasonal_period=int(max(2, round(float(params["seasonal_period"])))),
            demand_noise_scale=float(params["demand_noise_scale"]),
            pipeline_loss_prob=float(np.clip(float(params["pipeline_loss_prob"]), 0.0, 1.0)),
            pipeline_initial_bias=float(params["pipeline_initial_bias"]),
            initial_state_noise_sigma=float(params["initial_state_noise_sigma"]),
            inventory_obs_noise_sigma=float(params["inventory_obs_noise_sigma"]),
            state_recording=state_recording,
            event_order=event_order,
        )
        if demand_family == "poisson":
            cfg.poisson_lambda = float(params["poisson_lambda"])
            cfg.rate_ar1_phi = float(params["rate_ar1_phi"])
            cfg.rate_noise_sigma = float(params["rate_noise_sigma"])
        elif demand_family == "negbin":
            cfg.negbin_r = float(params["negbin_r"])
            cfg.negbin_p = float(params["negbin_p"])
        elif demand_family == "ar1":
            cfg.ar1_mu = float(params["ar1_mu"])
            cfg.ar1_phi = float(params["ar1_phi"])
            cfg.ar1_sigma = float(params["ar1_sigma"])
        return cfg

    cfg_opt = dict_to_config(optimized_params, args.demand_family)
    simulator = Simulator(cfg_opt, seed=seed)

    evaluator_val = Evaluator(lead_time_L=cfg_opt.lead_time_L, use_pot=args.use_pot, mmd_sigma=args.mmd_sigma, sample_size=args.sample_size)
    evaluator_test = Evaluator(lead_time_L=cfg_opt.lead_time_L, use_pot=args.use_pot, mmd_sigma=args.mmd_sigma, sample_size=args.sample_size)

    if val_traj:
        sim_results_val = simulator.rollout(
            trajectories=val_traj,
            start=0,
            end=val_traj[0].episode_length,
            actions_playback=True,
        )
        metrics_val = evaluator_val.compute_metrics(
            sim_results=sim_results_val,
            observed=val_traj,
            start=0,
            end=val_traj[0].episode_length,
        )
    else:
        metrics_val = {}

    if test_traj:
        sim_results_test = simulator.rollout(
            trajectories=test_traj,
            start=0,
            end=test_traj[0].episode_length,
            actions_playback=True,
        )
        metrics_test = evaluator_test.compute_metrics(
            sim_results=sim_results_test,
            observed=test_traj,
            start=0,
            end=test_traj[0].episode_length,
        )
    else:
        metrics_test = {}

    logging.info("Validation metrics: %s", safe_json_dumps(metrics_val))
    logging.info("Test metrics: %s", safe_json_dumps(metrics_test))

    metrics_ood = None
    if args.ood_eval:
        ood_traj = _try_load_ood(metadata)
        if not ood_traj:
            logging.info("No OOD dataset found. Generating OOD dataset on-the-fly...")
            generate_ood_dataset(
                num_ood_traj=int(metadata.get("n_trajectories", {}).get("test", 10)) if isinstance(metadata.get("n_trajectories"), dict) else 10,
                episode_length=ep_len,
                seed=seed + 123,
                lead_time_L=max(2, cfg_opt.lead_time_L + 2),
                demand_family="poisson" if args.demand_family != "poisson" else "ar1",
            )
            metadata = load_metadata()
            ood_traj = _try_load_ood(metadata)
        if ood_traj:
            sim_results_ood = simulator.rollout(
                trajectories=ood_traj,
                start=0,
                end=ood_traj[0].episode_length,
                actions_playback=True,
            )
            evaluator_ood = Evaluator(lead_time_L=cfg_opt.lead_time_L, use_pot=args.use_pot, mmd_sigma=args.mmd_sigma, sample_size=args.sample_size)
            metrics_ood = evaluator_ood.compute_metrics(
                sim_results=sim_results_ood,
                observed=ood_traj,
                start=0,
                end=ood_traj[0].episode_length,
            )
            logging.info("OOD metrics: %s", safe_json_dumps(metrics_ood))
        else:
            logging.warning("OOD evaluation requested but OOD dataset still not found. Skipping OOD metrics.")

    save_results(
        optimized_params=optimized_params,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        metrics_ood=metrics_ood,
        posterior_info=posterior_info,
        args=args,
        metadata=metadata,
    )


# Execute main for both direct execution and sandbox wrapper invocation
main()