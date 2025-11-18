import argparse
import json
import math
import os
import random
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional libraries
TORCH_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

SBI_AVAILABLE = False
try:
    if TORCH_AVAILABLE:
        from sbi import utils as sbi_utils
        from sbi.inference import SNPE as NPE
        from sbi.inference import simulate_for_sbi as sbi_simulate_for_sbi

        SBI_AVAILABLE = True
except Exception:
    SBI_AVAILABLE = False

POT_AVAILABLE = False
try:
    import ot

    POT_AVAILABLE = True
except Exception:
    POT_AVAILABLE = False

# Global deterministic seed
GLOBAL_SEED = 1337
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(GLOBAL_SEED)

# Path handling
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")


# ------------- Utility, dataclasses, and parsing -------------

@dataclass
class Item:
    """
    Represents a pipeline item with a quantity and remaining lead time.

    Attributes:
        qty: Quantity of the shipment.
        remaining: Remaining lead time before delivery (integer periods).
    """
    qty: int
    remaining: int


@dataclass
class TrajectoryData:
    """
    Trajectory data for a single simulation run.

    Attributes:
        trajectory_id: Identifier for the trajectory.
        t: 1D array of time indices.
        actions: 1D array of actions per time step.
        inventory_obs: Observed inventory per time step.
        backlog_obs: Observed backlog per time step.
        pipeline_items_obs: List of lists of Item objects per time step.
        pipeline_len_obs_counts: Observed count of pipeline items per time step.
        init_inventory: Initial inventory.
        init_backlog: Initial backlog.
        init_pipeline: Initial pipeline as a list of Item.
        max_remaining_lead: Max remaining lead time observed in this trajectory.
        episode_length: Total number of steps in the episode.
    """
    trajectory_id: str
    t: np.ndarray
    actions: np.ndarray
    inventory_obs: np.ndarray
    backlog_obs: np.ndarray
    pipeline_items_obs: List[List[Item]]
    pipeline_len_obs_counts: np.ndarray
    init_inventory: int
    init_backlog: int
    init_pipeline: List[Item]
    max_remaining_lead: int
    episode_length: int


@dataclass
class SimulationResults:
    """
    Container for results of a rollout vs. observations.

    Attributes:
        inventory_sim: Simulated inventory per trajectory_id.
        backlog_sim: Simulated backlog per trajectory_id.
        pipeline_len_sim: Simulated pipeline length per trajectory_id.
        pipeline_occ_sim: Simulated pipeline occupancy vectors per time step per trajectory.
        inventory_obs: Observed inventory per trajectory_id.
        backlog_obs: Observed backlog per trajectory_id.
        pipeline_len_obs: Observed pipeline length per trajectory_id.
        pipeline_occ_obs: Observed pipeline occupancy vectors per time step per trajectory.
    """
    inventory_sim: Dict[str, np.ndarray]
    backlog_sim: Dict[str, np.ndarray]
    pipeline_len_sim: Dict[str, np.ndarray]
    pipeline_occ_sim: Dict[str, List[np.ndarray]]
    inventory_obs: Dict[str, np.ndarray]
    backlog_obs: Dict[str, np.ndarray]
    pipeline_len_obs: Dict[str, np.ndarray]
    pipeline_occ_obs: Dict[str, List[np.ndarray]]


def validate_env_paths(cli_data_dir: Optional[str] = None) -> str:
    """
    Validate environment variables and return the data directory path; CLI has precedence.

    Adds a fallback to 'data_fitting/supply_data/' if neither env nor CLI are provided.
    """
    if cli_data_dir is not None:
        data_dir = cli_data_dir
    else:
        if PROJECT_ROOT is None or DATA_PATH is None:
            fallback = os.path.join("data_fitting", "supply_data")
            if os.path.isdir(fallback):
                data_dir = fallback
            else:
                raise EnvironmentError(
                    "Data directory not specified. Set PROJECT_ROOT and DATA_PATH env vars, pass --data-dir, or ensure 'data_fitting/supply_data/' exists."
                )
        else:
            data_dir = os.path.join(PROJECT_ROOT, DATA_PATH)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")
    return data_dir


def infer_time_column(df: pd.DataFrame) -> str:
    """
    Infer the time column name from a DataFrame.
    """
    candidates = ["t", "time", "time_step", "step", "period_index"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer time column; columns: {list(df.columns)}")


def safe_int(x: Any) -> int:
    """
    Safely cast a value to int.
    """
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Cannot cast to int: {x}") from e


_PARSE_PIPELINE_WARN_COUNT = 0
_PARSE_PIPELINE_WARN_LIMIT = 5


def parse_pipeline_items(raw: Any) -> List[Item]:
    """
    Parse a variety of string/list formats into a list of Item(qty, remaining).

    Supported:
      - JSON list of dicts: [{"qty": 4, "remaining_lead": 2}, ...]
      - JSON list of pairs: [[qty, remaining], ...]
      - JSON list of numbers representing quantities per remaining slot
      - Semicolon separated "qty@remaining"
    """
    global _PARSE_PIPELINE_WARN_COUNT
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        data = raw
    else:
        s = str(raw).strip()
        if s == "" or s == "[]":
            return []
        try:
            data = json.loads(s)
        except Exception:
            items = []
            try:
                parts = [p for p in s.split(";") if "@" in p]
                for p in parts:
                    q_s, r_s = p.split("@")
                    qty = safe_int(q_s)
                    rem = safe_int(r_s)
                    if qty > 0 and rem >= 0:
                        items.append(Item(qty=qty, remaining=rem))
                return items
            except Exception:
                if _PARSE_PIPELINE_WARN_COUNT < _PARSE_PIPELINE_WARN_LIMIT:
                    warnings.warn(f"Failed to parse pipeline_items; treating as empty. Sample: {s[:64]}...")
                    _PARSE_PIPELINE_WARN_COUNT += 1
                return []

    items_out: List[Item] = []
    if isinstance(data, list):
        if len(data) == 0:
            return []
        first = data[0]
        if isinstance(first, dict):
            for d in data:
                qty = safe_int(d.get("qty", d.get("quantity", 0)))
                rem = d.get("remaining_lead", d.get("remaining", d.get("lead", 0)))
                rem = safe_int(rem)
                if qty > 0 and rem >= 0:
                    items_out.append(Item(qty=qty, remaining=rem))
        elif isinstance(first, (list, tuple)) and len(first) == 2:
            for pair in data:
                try:
                    qty = safe_int(pair[0])
                    rem = safe_int(pair[1])
                    if qty > 0 and rem >= 0:
                        items_out.append(Item(qty=qty, remaining=rem))
                except Exception:
                    continue
        else:
            for idx, qty in enumerate(data, start=1):
                try:
                    q = safe_int(qty)
                except Exception:
                    q = 0
                if q > 0:
                    items_out.append(Item(qty=q, remaining=idx))
    return items_out


def load_data(
    data_dir: str,
    train_file: str,
    val_file: str,
    test_file: Optional[str],
    metadata_file: str,
    test_ood_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load datasets and metadata.
    """
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    meta_path = os.path.join(data_dir, metadata_file)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Missing val file: {val_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(os.path.join(data_dir, test_file)) if test_file and os.path.isfile(os.path.join(data_dir, test_file)) else None
    test_ood_df = pd.read_csv(os.path.join(data_dir, test_ood_file)) if test_ood_file and os.path.isfile(os.path.join(data_dir, test_ood_file)) else None
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return train_df, val_df, test_df, test_ood_df, metadata


def build_trajectories(df: pd.DataFrame, metadata: Dict[str, Any]) -> List[TrajectoryData]:
    """
    Build a list of TrajectoryData from a DataFrame.
    """
    if df is None or len(df) == 0:
        return []
    required = ["trajectory_id", "action", "inventory", "backlog"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")
    time_col = infer_time_column(df)
    episode_length = int(metadata.get("episode_length", metadata.get("trajectory_length", 61)))
    trajectories: List[TrajectoryData] = []
    mismatch_count = 0
    mismatch_examples = []
    contig_warn = 0
    contig_examples = []
    for tid, g in df.groupby("trajectory_id"):
        g_sorted = g.sort_values(by=time_col)
        t = g_sorted[time_col].to_numpy().astype(int)
        actions = g_sorted["action"].to_numpy().astype(int)
        inv = g_sorted["inventory"].to_numpy().astype(float)
        bklg = g_sorted["backlog"].to_numpy().astype(float)
        pipeline_col = "pipeline_items" if "pipeline_items" in g_sorted.columns else None
        pipeline_items_seq: List[List[Item]] = []
        max_rem = 0
        pipeline_len_counts = []
        if pipeline_col is not None:
            raw_list = g_sorted[pipeline_col].tolist()
            for raw in raw_list:
                items = parse_pipeline_items(raw)
                for it in items:
                    max_rem = max(max_rem, int(it.remaining))
                pipeline_items_seq.append(items)
                pipeline_len_counts.append(len(items))
        else:
            pipeline_items_seq = [[] for _ in range(len(g_sorted))]
            pipeline_len_counts = [0 for _ in range(len(g_sorted))]
        if "pipeline_len" in g_sorted.columns:
            plcol = g_sorted["pipeline_len"].to_numpy().astype(int)
            for i, cnt in enumerate(pipeline_len_counts):
                if plcol[i] != cnt:
                    mismatch_count += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append((str(tid), int(t[i]), plcol[i], cnt))
        t0 = int(t.min())
        t1 = int(t.max())
        expected = np.arange(t0, t1 + 1)
        if not np.array_equal(t, expected) or t0 != 0:
            contig_warn += 1
            if len(contig_examples) < 5:
                contig_examples.append((str(tid), t0, t1, len(t)))

        init_inventory = int(round(inv[0]))
        init_backlog = int(round(bklg[0]))
        init_pipeline = pipeline_items_seq[0] if len(pipeline_items_seq) > 0 else []

        trajectories.append(
            TrajectoryData(
            trajectory_id=str(tid),
            t=t,
            actions=actions,
            inventory_obs=inv,
            backlog_obs=bklg,
            pipeline_items_obs=pipeline_items_seq,
            pipeline_len_obs_counts=np.array(pipeline_len_counts, dtype=int),
            init_inventory=init_inventory,
            init_backlog=init_backlog,
            init_pipeline=[Item(qty=int(it.qty), remaining=int(it.remaining)) for it in init_pipeline],
            max_remaining_lead=int(max_rem),
            episode_length=episode_length,
        )
        )
    if mismatch_count > 0:
        warnings.warn(
            f"pipeline_len mismatches parsed counts in {mismatch_count} rows. Examples: {mismatch_examples}"
        )
    if contig_warn > 0:
        warnings.warn(f"Found {contig_warn} trajectories with non-contiguous steps or t_min!=0. Examples: {contig_examples}")
    return trajectories


# ------------- Demand models -------------

class DemandModel(ABC):
    """
    Abstract base class for demand models. Implement sample(t) to draw demand.

    Attributes:
        rng: Numpy random generator.
    """
    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize with a RNG."""
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state before a new trajectory."""
        pass

    @abstractmethod
    def sample(self, t: int) -> int:
        """
        Sample demand at time step t.

        Args:
            t: Time step.

        Returns:
            Nonnegative integer demand.
        """
        pass


class PoissonDemandModel(DemandModel):
    """
    Poisson demand with optional sinusoidal seasonality.

    Args:
        base_lambda: Base Poisson rate.
        seasonal_amplitude: Amplitude of sinusoid.
        seasonal_period: Period of seasonality.
        rng: RNG.
    """
    def __init__(self, base_lambda: float, seasonal_amplitude: float = 0.0, seasonal_period: int = 7,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(rng=rng)
        if base_lambda < 0:
            raise ValueError("base_lambda must be >=0")
        if seasonal_amplitude < 0:
            raise ValueError("seasonal_amplitude must be >=0")
        if seasonal_period < 2:
            raise ValueError("seasonal_period must be >=2")
        self.base_lambda = float(base_lambda)
        self.seasonal_amplitude = float(seasonal_amplitude)
        self.seasonal_period = int(seasonal_period)

    def reset(self) -> None:
        """No internal state; nothing to reset."""
        return

    def _mean_t(self, t: int) -> float:
        """Compute time-varying mean if seasonality enabled."""
        if self.seasonal_amplitude <= 1e-8:
            return max(0.0, self.base_lambda)
        val = self.base_lambda + self.seasonal_amplitude * math.sin(2.0 * math.pi * (t % self.seasonal_period) / float(self.seasonal_period))
        return max(0.0, val)

    def sample(self, t: int) -> int:
        lam = self._mean_t(t)
        d = self.rng.poisson(lam=max(0.0, lam))
        return int(d)


class NegativeBinomialDemandModel(DemandModel):
    """
    Negative Binomial via Gamma-Poisson mixture allowing real-valued r.

    Args:
        mu: Mean demand.
        r: Shape/dispersion (>0).
        rng: RNG.
    """
    def __init__(self, mu: float, r: float, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(rng=rng)
        if mu < 0 or r <= 0:
            raise ValueError("Invalid NegBin params.")
        self.mu = float(mu)
        self.r = float(r)

    def reset(self) -> None:
        """No internal state; nothing to reset."""
        return

    def sample(self, t: int) -> int:
        scale = self.mu / max(self.r, 1e-12)
        lam = self.rng.gamma(shape=self.r, scale=scale)
        return int(self.rng.poisson(lam=max(0.0, lam)))


class AR1DemandModel(DemandModel):
    """
    AR(1) latent Gaussian demand rounded and truncated to nonnegative.

    Args:
        mu: Unconditional mean.
        phi: AR coefficient (-1<phi<1).
        sigma: Innovation std.
        rng: RNG.
    """
    def __init__(self, mu: float, phi: float, sigma: float, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(rng=rng)
        if mu < 0 or not (-0.95 <= phi <= 0.95) or sigma < 0:
            raise ValueError("Invalid AR1 params.")
        self.mu = float(mu)
        self.phi = float(phi)
        self.sigma = float(sigma)
        self._x_prev = self.mu

    def reset(self) -> None:
        """Reset latent to mean."""
        self._x_prev = self.mu

    def sample(self, t: int) -> int:
        eps = self.rng.normal(0.0, 1.0)
        x_t = self.mu + self.phi * (self._x_prev - self.mu) + self.sigma * eps
        self._x_prev = float(x_t)
        return int(max(0, round(x_t)))


def create_demand_model_from_params(params: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> DemandModel:
    """
    Factory for demand model from parameter dict.
    """
    fam = params.get("demand_family", "Poisson")
    if fam == "Poisson":
        return PoissonDemandModel(
            base_lambda=float(params.get("poisson_lambda", 5.0)),
            seasonal_amplitude=float(params.get("seasonal_amplitude", 0.0)),
            seasonal_period=int(params.get("seasonal_period", 7)),
            rng=rng,
        )
    if fam == "NegBin":
        return NegativeBinomialDemandModel(
            mu=float(params.get("negbin_mu", 5.0)),
            r=float(params.get("negbin_r", 5.0)),
            rng=rng,
        )
    if fam == "AR1":
        return AR1DemandModel(
            mu=float(params.get("ar1_mu", 5.0)),
            phi=float(params.get("ar1_phi", 0.0)),
            sigma=float(params.get("ar1_sigma", 1.0)),
            rng=rng,
        )
        raise ValueError(f"Unsupported demand_family: {fam}")


# ------------- Inventory node -------------

class InventoryNode:
    """
    Single retailer node managing inventory, backlog, and a pipeline of items.

    Update order:
        1) Deliver arrivals from pipeline based on arrival_convention.
        2) Serve backlog.
        3) Serve current demand.
        4) Append new order to pipeline.
        5) Compute pipeline occupancy.

    Arrival conventions:
        - deliver_at_remaining_0: decrement to 0 then deliver.
        - deliver_at_remaining_1: deliver items with remaining==1 before decrement; decrement others with floor at 1.
    """
    def __init__(
        self,
        init_inventory: int,
        init_backlog: int,
        init_pipeline: List[Item],
        lead_time: int,
        arrival_convention: str = "deliver_at_remaining_0",
    ) -> None:
        """Initialize the node."""
        if init_inventory < 0 or init_backlog < 0:
            raise ValueError("Initial inventory/backlog must be nonnegative.")
        if lead_time < 0:
            raise ValueError("lead_time must be >=0")
        if arrival_convention not in {"deliver_at_remaining_0", "deliver_at_remaining_1"}:
            raise ValueError("Invalid arrival_convention")
        self.inventory = int(init_inventory)
        self.backlog = int(init_backlog)
        self.pipeline: List[Item] = [Item(qty=int(max(0, it.qty)), remaining=int(max(0, it.remaining))) for it in init_pipeline if int(it.qty) > 0]
        self.lead_time = int(lead_time)
        self.arrival_convention = arrival_convention
        self.t = 0

    def _deliveries_and_decrement(self) -> int:
        """
        Process arrivals and decrement remaining lead times.

        Returns:
            Delivered quantity this step.
        """
        delivered = 0
        if self.arrival_convention == "deliver_at_remaining_1":
            # Deliver only items with remaining==1; decrement others with floor at 1
            remaining_items: List[Item] = []
            for it in self.pipeline:
                if it.remaining == 1:
                    delivered += it.qty
                else:
                    remaining_items.append(it)
            new_pipeline: List[Item] = []
            for it in remaining_items:
                new_rem = max(1, it.remaining - 1)
                new_pipeline.append(Item(qty=it.qty, remaining=new_rem))
            self.pipeline = new_pipeline
        else:
            decremented: List[Item] = []
            for it in self.pipeline:
                new_rem = max(0, it.remaining - 1)
                decremented.append(Item(qty=it.qty, remaining=new_rem))
            keep: List[Item] = []
            for it in decremented:
                if it.remaining == 0:
                    delivered += it.qty
                else:
                    keep.append(it)
            self.pipeline = keep
        return delivered

    def _append_order(self, qty: int) -> None:
        """
        Append order to the pipeline at position lead_time.

        Args:
            qty: Quantity ordered this step.
        """
        q = int(max(0, qty))
        if self.lead_time == 0:
            self.inventory += q
            return
        if q > 0:
            self.pipeline.append(Item(qty=q, remaining=self.lead_time))

    def step(self, action: int, demand: int) -> Dict[str, Any]:
        """
        Advance one period.

        Args:
            action: Order quantity to append to pipeline.
            demand: Demand to serve.

        Returns:
            Snapshot dict with keys: t, inventory, backlog, pipeline_len, pipeline_occ.
        """
        delivered = self._deliveries_and_decrement()
        self.inventory += int(delivered)

        # Serve backlog first
        serve_b = min(self.inventory, self.backlog)
        self.inventory -= serve_b
        self.backlog -= serve_b

        # Serve current demand
        d = max(0, int(demand))
        served_d = min(self.inventory, d)
        self.inventory -= served_d
        unmet = d - served_d
        self.backlog += int(unmet)

        # Append order
        self._append_order(action)

        # Occ vector (quantities per remaining slot)
        pipeline_len_count = len(self.pipeline)
        occ_vec = np.zeros(max(1, self.lead_time), dtype=float)
        for it in self.pipeline:
            rem = int(it.remaining)
            if rem <= 0:
                idx = 0
            elif 1 <= rem <= self.lead_time:
                idx = rem - 1
            else:
                idx = self.lead_time - 1
            occ_vec[idx] += it.qty

        snapshot = {
            "t": self.t,
            "inventory": int(self.inventory),
            "backlog": int(self.backlog),
            "pipeline_len": int(pipeline_len_count),
            "pipeline_occ": occ_vec.copy(),
        }
        self.t += 1
        return snapshot


# ------------- Metrics and utilities -------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Root mean squared error between arrays.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean absolute error between arrays.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean squared error between arrays.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.mean((a - b) ** 2))


def mmd_gaussian(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0, max_samples: Optional[int] = None, unbiased: bool = True) -> float:
    """
    Gaussian kernel MMD between samples X and Y.

    Uses an unbiased estimator (diagonal excluded) and optional subsampling to reduce O(n^2) cost.

    Args:
        X: Samples (n_x, d) or (n_x,).
        Y: Samples (n_y, d) or (n_y,).
        sigma: Kernel bandwidth.
        max_samples: If provided, subsample up to this many samples from each set (with replacement if needed).
        unbiased: If True, use unbiased estimator (exclude diagonals).

    Returns:
        MMD^2 value (nonnegative).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if X.size == 0 or Y.size == 0:
        return 0.0

    rng = np.random.default_rng(GLOBAL_SEED + 98765)
    def _subsample(M: np.ndarray) -> np.ndarray:
        if max_samples is None:
            return M
        n = M.shape[0]
        m = int(max_samples)
        if n >= m:
            idx = rng.choice(n, size=m, replace=False)
        else:
            idx = rng.choice(n, size=m, replace=True)
        return M[idx, :]

    X = _subsample(X)
    Y = _subsample(Y)

    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        AA = np.sum(A * A, axis=1, keepdims=True)
        BB = np.sum(B * B, axis=1, keepdims=True).T
        D2 = AA + BB - 2.0 * A @ B.T
        return np.exp(-gamma * np.maximum(D2, 0.0))

    Kxx = rbf(X, X)
    Kyy = rbf(Y, Y)
    Kxy = rbf(X, Y)

    if unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        m = X.shape[0]
        n = Y.shape[0]
        term_xx = np.sum(Kxx) / (m * (m - 1) + 1e-12)
        term_yy = np.sum(Kyy) / (n * (n - 1) + 1e-12)
        term_xy = np.sum(Kxy) / (m * n + 1e-12)
        mmd2 = term_xx + term_yy - 2.0 * term_xy
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return float(max(0.0, mmd2))


def _wasserstein_1d_fallback(x: np.ndarray, y: np.ndarray, n_samples: int = 200) -> float:
    """
    Compute 1D empirical Wasserstein distance using sorted samples.

    If sample sizes differ, resample with replacement to n_samples.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size >= n_samples:
        xi = np.random.choice(x.size, size=n_samples, replace=False)
        x = x[xi]
    else:
        xi = np.random.choice(x.size, size=n_samples, replace=True)
        x = x[xi]
    if y.size >= n_samples:
        yi = np.random.choice(y.size, size=n_samples, replace=False)
        y = y[yi]
    else:
        yi = np.random.choice(y.size, size=n_samples, replace=True)
        y = y[yi]
    xs = np.sort(x)
    ys = np.sort(y)
    return float(np.mean(np.abs(xs - ys)))


# ------------- Simulator -------------

class BeerGameSimulator:
    """
    Simulator orchestrator for single-stage Beer Game system.

    Provides:
        - set_params: apply parameter dict.
        - rollout: run on provided trajectories with action playback.
        - get_params: current params dict.

    Notes:
        Per-trajectory independent RNGs are used for demand sampling.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize with a parameter dict."""
        self.params = dict(params)

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set new parameters for the simulator.

        Args:
            params: Dict of parameters.
        """
        self.params.update(params)

    def get_params(self) -> Dict[str, Any]:
        """
        Get a copy of current parameters.

        Returns:
            Parameter dict.
        """
        return dict(self.params)

    def _arrival_convention_from_flag(self, flag: int) -> str:
        """
        Map arrival_flag to arrival convention string.

        Args:
            flag: 0 or 1.

        Returns:
            Convention string.
        """
        return "deliver_at_remaining_1" if int(flag) == 1 else "deliver_at_remaining_0"

    def rollout(self, trajectories: List[TrajectoryData]) -> SimulationResults:
        """
        Rollout simulation on trajectories using current parameters.

        Args:
            trajectories: List of TrajectoryData.

        Returns:
            SimulationResults for comparison with observations.
        """
        if len(trajectories) == 0:
            return SimulationResults({}, {}, {}, {}, {}, {}, {}, {})
        L = int(self.params.get("lead_time_L", 2))
        arrival_flag = int(self.params.get("arrival_flag", 0))
        arrival_convention = self._arrival_convention_from_flag(arrival_flag)
        fam = self.params.get("demand_family", "Poisson")

        results_inv_s: Dict[str, np.ndarray] = {}
        results_b_s: Dict[str, np.ndarray] = {}
        results_plen_s: Dict[str, np.ndarray] = {}
        results_occ_s: Dict[str, List[np.ndarray]] = {}

        results_inv_o: Dict[str, np.ndarray] = {}
        results_b_o: Dict[str, np.ndarray] = {}
        results_plen_o: Dict[str, np.ndarray] = {}
        results_occ_o: Dict[str, List[np.ndarray]] = {}

        seed_val = GLOBAL_SEED + L * 17 + (1 if arrival_convention.endswith("_1") else 0) * 29 + 123

        for tr in trajectories:
            # Per-trajectory independent RNG robust to non-numeric IDs
            try:
                tid_int = int(tr.trajectory_id)
            except Exception:
                tid_int = int(abs(hash(str(tr.trajectory_id))) % 1_000_000_000)
            tr_seed = seed_val + tid_int * 1009
            rng_tr = np.random.default_rng(tr_seed)
            params_for_model = dict(self.params)
            params_for_model["demand_family"] = fam
            demand_model = create_demand_model_from_params(params_for_model, rng=rng_tr)
            demand_model.reset()

            node = InventoryNode(
                init_inventory=int(tr.init_inventory),
                init_backlog=int(tr.init_backlog),
                init_pipeline=[Item(int(it.qty), int(it.remaining)) for it in tr.init_pipeline],
                lead_time=L,
                arrival_convention=arrival_convention,
            )
            inv_rec = []
            b_rec = []
            plen_rec = []
            occ_s_rec: List[np.ndarray] = []

            horizon = len(tr.t)
            actions = tr.actions
            for tt in range(horizon):
                d = demand_model.sample(t=tt)
                snap = node.step(action=int(actions[tt]), demand=int(d))
                inv_rec.append(snap["inventory"])
                b_rec.append(snap["backlog"])
                plen_rec.append(snap["pipeline_len"])
                occ_s_rec.append(snap["pipeline_occ"])

            inv_obs_arr = tr.inventory_obs.astype(float)
            b_obs_arr = tr.backlog_obs.astype(float)
            plen_obs_arr = tr.pipeline_len_obs_counts.astype(float)

            occ_o_rec: List[np.ndarray] = []
            for items in tr.pipeline_items_obs:
                occ_v = np.zeros(max(1, L), dtype=float)
                for it in items:
                    rem = int(it.remaining)
                    if rem <= 0:
                        idx = 0
                    elif 1 <= rem <= L:
                        idx = rem - 1
                    else:
                        idx = L - 1
                    occ_v[idx] += int(it.qty)
                occ_o_rec.append(occ_v)

            results_inv_s[tr.trajectory_id] = np.array(inv_rec, dtype=float)
            results_b_s[tr.trajectory_id] = np.array(b_rec, dtype=float)
            results_plen_s[tr.trajectory_id] = np.array(plen_rec, dtype=float)
            results_occ_s[tr.trajectory_id] = occ_s_rec

            results_inv_o[tr.trajectory_id] = inv_obs_arr
            results_b_o[tr.trajectory_id] = b_obs_arr
            results_plen_o[tr.trajectory_id] = plen_obs_arr
            results_occ_o[tr.trajectory_id] = occ_o_rec

        return SimulationResults(
            inventory_sim=results_inv_s,
            backlog_sim=results_b_s,
            pipeline_len_sim=results_plen_s,
            pipeline_occ_sim=results_occ_s,
            inventory_obs=results_inv_o,
            backlog_obs=results_b_o,
            pipeline_len_obs=results_plen_o,
            pipeline_occ_obs=results_occ_o,
        )


# ------------- Evaluator -------------

class Evaluator:
    """
    Evaluator computing metrics between simulated and observed trajectories.

    Implements:
        - compute_metrics: per-trajectory RMSE/MAE and distributional metrics, plus MSE and MSE per dimension.
        - joint 3D Wasserstein (inventory, backlog, pipeline_len) per time step with configurable OT method.
    """
    def __init__(self, ot_method: str = "sinkhorn", ot_epsilon: float = 0.05, ot_max_iter: int = 2000, n_samples: int = 200) -> None:
        """
        Initialize Evaluator configuration.

        Args:
            ot_method: 'sinkhorn', 'emd', or 'fallback' (1D avg).
            ot_epsilon: Regularization epsilon for sinkhorn.
            ot_max_iter: Max iterations for sinkhorn.
            n_samples: Subsample size per time step for Wasserstein/MMD.
        """
        self.ot_method = ot_method
        self.ot_epsilon = float(ot_epsilon)
        self.ot_max_iter = int(ot_max_iter)
        self.n_samples = int(n_samples)

    def compute_joint_wasserstein_per_t(self, results: SimulationResults, n_samples: Optional[int] = None) -> float:
        """
        Compute Wasserstein distance between joint state samples [inv, backlog] aligned with GSIM env.py.
        
        This method matches the 'wass' metric in generative-simulations/libs/SUPPLY/env.py:
        - Uses all data (no sampling)
        - Combines inventory and backlog into 2D state vectors
        - Uses ot.emd() for true Wasserstein distance (not regularized Sinkhorn)
        - Computes distance on all time steps concatenated together
        """
        inv_sim = results.inventory_sim
        inv_obs = results.inventory_obs
        b_sim = results.backlog_sim
        b_obs = results.backlog_obs

        # Collect all inventory and backlog values across all trajectories and time steps
        # Use intersection of trajectory IDs to ensure proper alignment
        inv_sim_all = []
        inv_obs_all = []
        b_sim_all = []
        b_obs_all = []
        
        # Get intersection of trajectory IDs from both sim and obs
        tids_sim = set(inv_sim.keys()) & set(b_sim.keys())
        tids_obs = set(inv_obs.keys()) & set(b_obs.keys())
        tids = list(tids_sim & tids_obs)  # Only process trajectories present in both
        
        if len(tids) == 0:
            return float("nan")
        
        for tid in tids:
            # Collect all time steps for this trajectory
            # Ensure we have matching lengths
            inv_sim_traj = inv_sim[tid]
            b_sim_traj = b_sim[tid]
            inv_obs_traj = inv_obs[tid]
            b_obs_traj = b_obs[tid]
            
            # Align lengths within trajectory
            min_len = min(len(inv_sim_traj), len(b_sim_traj), len(inv_obs_traj), len(b_obs_traj))
            if min_len == 0:
                continue
            
            inv_sim_all.extend(inv_sim_traj[:min_len].tolist())
            b_sim_all.extend(b_sim_traj[:min_len].tolist())
            inv_obs_all.extend(inv_obs_traj[:min_len].tolist())
            b_obs_all.extend(b_obs_traj[:min_len].tolist())
        
        # Convert to numpy arrays
        inv_sim_arr = np.asarray(inv_sim_all, dtype=float)
        inv_obs_arr = np.asarray(inv_obs_all, dtype=float)
        b_sim_arr = np.asarray(b_sim_all, dtype=float)
        b_obs_arr = np.asarray(b_obs_all, dtype=float)
        
        # Align sizes (truncate to minimum)
        min_size = min(len(inv_sim_arr), len(b_sim_arr), len(inv_obs_arr), len(b_obs_arr))
        if min_size == 0:
            return float("nan")
        
        inv_sim_aligned = inv_sim_arr[:min_size]
        b_sim_aligned = b_sim_arr[:min_size]
        inv_obs_aligned = inv_obs_arr[:min_size]
        b_obs_aligned = b_obs_arr[:min_size]
        
        # Combine into 2D state vectors (inventory, backlog) - aligned with GSIM
        sim_states = np.column_stack([inv_sim_aligned, b_sim_aligned])
        obs_states = np.column_stack([inv_obs_aligned, b_obs_aligned])
        
        # Compute Wasserstein distance using ot.emd() (true Wasserstein, not regularized)
        # This matches GSIM's wasserstein_distance_nd() function
        if POT_AVAILABLE:
            try:
                N, M = sim_states.shape[0], obs_states.shape[0]
                if N == 0 or M == 0:
                    return float("nan")
                
                # Compute cost matrix (Euclidean L2 distance)
                cost_matrix = ot.dist(sim_states, obs_states, metric="euclidean")
                a = np.ones(N, dtype=float) / float(N)
                b = np.ones(M, dtype=float) / float(M)
                
                # Use ot.emd() for true Wasserstein distance (aligned with GSIM)
                transport_plan = ot.emd(a, b, cost_matrix)
                wass = float(np.sum(cost_matrix * transport_plan))
                return wass if np.isfinite(wass) else float("nan")
            except Exception as e:
                # Fallback to 1D Wasserstein if EMD fails
                warnings.warn(f"EMD computation failed: {e}, falling back to 1D Wasserstein")
                w_inv = _wasserstein_1d_fallback(inv_sim_aligned, inv_obs_aligned, n_samples=min(N, M))
                w_b = _wasserstein_1d_fallback(b_sim_aligned, b_obs_aligned, n_samples=min(N, M))
                return float(np.mean([w_inv, w_b]))
        else:
            # Fallback if POT not available
            w_inv = _wasserstein_1d_fallback(inv_sim_aligned, inv_obs_aligned, n_samples=min_size)
            w_b = _wasserstein_1d_fallback(b_sim_aligned, b_obs_aligned, n_samples=min_size)
            return float(np.mean([w_inv, w_b]))

    def wasserstein_between_results(self, res_a: SimulationResults, res_b: SimulationResults, n_samples: Optional[int] = None) -> float:
        """
        Compute mean per-time-step Wasserstein distance (W1) between two simulated result sets.

        Uses joint vectors [inventory_sim, backlog_sim, pipeline_len_sim] from both.
        """
        tids = list(set(res_a.inventory_sim.keys()).intersection(set(res_b.inventory_sim.keys())))
        if len(tids) == 0:
            return 0.0
        max_T = 0
        for tid in tids:
            max_T = max(max_T, min(res_a.inventory_sim[tid].shape[0], res_b.inventory_sim[tid].shape[0]))
        vals = []
        n_samp = self.n_samples if n_samples is None else int(n_samples)
        for t in range(max_T):
            Xa = []
            Xb = []
            for tid in tids:
                if (
                    t < res_a.inventory_sim[tid].shape[0]
                    and t < res_a.backlog_sim[tid].shape[0]
                    and t < res_a.pipeline_len_sim[tid].shape[0]
                    and t < res_b.inventory_sim[tid].shape[0]
                    and t < res_b.backlog_sim[tid].shape[0]
                    and t < res_b.pipeline_len_sim[tid].shape[0]
                ):
                    Xa.append([res_a.inventory_sim[tid][t], res_a.backlog_sim[tid][t], res_a.pipeline_len_sim[tid][t]])
                    Xb.append([res_b.inventory_sim[tid][t], res_b.backlog_sim[tid][t], res_b.pipeline_len_sim[tid][t]])
            if len(Xa) == 0 or len(Xb) == 0:
                continue
            Xa = np.asarray(Xa, dtype=float)
            Xb = np.asarray(Xb, dtype=float)
            if POT_AVAILABLE and self.ot_method in {"sinkhorn", "emd"}:
                def sample_rows(M: np.ndarray) -> np.ndarray:
                    if M.shape[0] >= n_samp:
                        idx = np.random.choice(M.shape[0], size=n_samp, replace=False)
                    else:
                        idx = np.random.choice(M.shape[0], size=n_samp, replace=True)
                    return M[idx, :]
                Xa_s = sample_rows(Xa)
                Xb_s = sample_rows(Xb)
                C = np.linalg.norm(Xa_s[:, None, :] - Xb_s[None, :, :], axis=2)
                a = np.ones((Xa_s.shape[0],), dtype=float) / float(Xa_s.shape[0])
                b = np.ones((Xb_s.shape[0],), dtype=float) / float(Xb_s.shape[0])
                if self.ot_method == "sinkhorn":
                    try:
                        # Suppress numerical warnings for sinkhorn (they're handled by fallback)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning, message=".*numerical errors.*")
                            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide.*")
                            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
                            sink_cost = ot.sinkhorn2(a, b, C, reg=self.ot_epsilon, numItermax=self.ot_max_iter)
                        if np.isfinite(sink_cost):
                            vals.append(float(sink_cost))
                    except Exception:
                        try:
                            emd_cost = ot.emd2(a, b, C)
                            if np.isfinite(emd_cost):
                                vals.append(float(emd_cost))
                        except Exception:
                            w_inv = _wasserstein_1d_fallback(Xa[:, 0], Xb[:, 0], n_samples=n_samp)
                            w_b = _wasserstein_1d_fallback(Xa[:, 1], Xb[:, 1], n_samples=n_samp)
                            w_pl = _wasserstein_1d_fallback(Xa[:, 2], Xb[:, 2], n_samples=n_samp)
                            vals.append(float(np.mean([w_inv, w_b, w_pl])))
                else:
                    try:
                        emd_cost = ot.emd2(a, b, C)
                        if np.isfinite(emd_cost):
                            vals.append(float(emd_cost))
                    except Exception:
                        w_inv = _wasserstein_1d_fallback(Xa[:, 0], Xb[:, 0], n_samples=n_samp)
                        w_b = _wasserstein_1d_fallback(Xa[:, 1], Xb[:, 1], n_samples=n_samp)
                        w_pl = _wasserstein_1d_fallback(Xa[:, 2], Xb[:, 2], n_samples=n_samp)
                        vals.append(float(np.mean([w_inv, w_b, w_pl])))
            else:
                w_inv = _wasserstein_1d_fallback(Xa[:, 0], Xb[:, 0], n_samples=n_samp)
                w_b = _wasserstein_1d_fallback(Xa[:, 1], Xb[:, 1], n_samples=n_samp)
                w_pl = _wasserstein_1d_fallback(Xa[:, 2], Xb[:, 2], n_samples=n_samp)
                vals.append(float(np.mean([w_inv, w_b, w_pl])))
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    def compute_metrics(self, results: SimulationResults, n_samples_wass_mmd: int = 200, mmd_sigma: float = 1.0) -> Dict[str, Any]:
        """
        Compute comprehensive metrics, including RMSE/MAE, MSE (overall and per dimension),
        pipeline composition L1, and distributional Wasserstein and MMD.
        """
        per_traj: Dict[str, Dict[str, float]] = {}
        pipeline_l1s = []

        # For MSE aggregates
        se_inv_all = []
        se_bk_all = []
        se_pl_all = []
        se_t_all = []

        for tid in results.inventory_sim.keys():
            inv_s = results.inventory_sim[tid]
            inv_o = results.inventory_obs[tid]
            b_s = results.backlog_sim[tid]
            b_o = results.backlog_obs[tid]
            pl_s = results.pipeline_len_sim[tid]
            pl_o = results.pipeline_len_obs[tid]
            T = min(len(inv_s), len(inv_o), len(b_s), len(b_o), len(pl_s), len(pl_o))
            if T <= 0:
                continue
            inv_s, inv_o = inv_s[:T], inv_o[:T]
            b_s, b_o = b_s[:T], b_o[:T]
            pl_s, pl_o = pl_s[:T], pl_o[:T]

            rmse_i = rmse(inv_s, inv_o)
            rmse_b = rmse(b_s, b_o)
            mae_pl = mae(pl_s, pl_o)

            occ_s_list = results.pipeline_occ_sim[tid][:T]
            occ_o_list = results.pipeline_occ_obs[tid][:T]
            l1_vals = []
            for os_s, os_o in zip(occ_s_list, occ_o_list):
                L = min(len(os_s), len(os_o))
                if L == 0:
                    l1_vals.append(0.0)
                else:
                    l1_vals.append(float(np.sum(np.abs(os_s[:L] - os_o[:L]))))
            l1_mean = float(np.mean(l1_vals)) if len(l1_vals) > 0 else 0.0
            pipeline_l1s.append(l1_mean)

            # MSE accumulators
            se_inv_all.append((inv_s - inv_o) ** 2)
            se_bk_all.append((b_s - b_o) ** 2)
            se_pl_all.append((pl_s - pl_o) ** 2)
            # time index mse (simulated time vs observed time); both 0..T-1
            t_idx = np.arange(T, dtype=float)
            se_t_all.append((t_idx - t_idx) ** 2)

            per_traj[tid] = {
                "RMSE_inventory": rmse_i,
                "RMSE_backlog": rmse_b,
                "MAE_pipeline_len": mae_pl,
                "PipelineComp_L1": l1_mean,
            }

        # Distributional metrics
        wass_joint = self.compute_joint_wasserstein_per_t(results, n_samples=n_samples_wass_mmd)

        # Build joint samples for MMD across all tids/time
        Xs_list: List[List[float]] = []
        Xo_list: List[List[float]] = []
        common_tids = set(results.inventory_sim.keys()).intersection(set(results.inventory_obs.keys()))
        for tid in common_tids:
            inv_s = results.inventory_sim[tid]
            inv_o = results.inventory_obs[tid]
            b_s = results.backlog_sim[tid]
            b_o = results.backlog_obs[tid]
            pl_s = results.pipeline_len_sim[tid]
            pl_o = results.pipeline_len_obs[tid]
            T = min(len(inv_s), len(inv_o), len(b_s), len(b_o), len(pl_s), len(pl_o))
            for i in range(T):
                Xs_list.append([float(inv_s[i]), float(b_s[i]), float(pl_s[i])])
                Xo_list.append([float(inv_o[i]), float(b_o[i]), float(pl_o[i])])
        if len(Xs_list) > 0 and len(Xo_list) > 0:
            Xs = np.asarray(Xs_list, dtype=float)
            Xo = np.asarray(Xo_list, dtype=float)
            mmd2_val = mmd_gaussian(Xs, Xo, sigma=mmd_sigma, max_samples=n_samples_wass_mmd, unbiased=True)
            mmd_val = float(np.sqrt(max(0.0, mmd2_val)))
        else:
            mmd2_val = 0.0
            mmd_val = 0.0

        # Aggregates
        agg = {
                "RMSE_inventory_mean": float(np.mean([d["RMSE_inventory"] for d in per_traj.values()])) if per_traj else 0.0,
                "RMSE_inventory_std": float(np.std([d["RMSE_inventory"] for d in per_traj.values()])) if per_traj else 0.0,
                "RMSE_backlog_mean": float(np.mean([d["RMSE_backlog"] for d in per_traj.values()])) if per_traj else 0.0,
                "RMSE_backlog_std": float(np.std([d["RMSE_backlog"] for d in per_traj.values()])) if per_traj else 0.0,
                "MAE_pipeline_len_mean": float(np.mean([d["MAE_pipeline_len"] for d in per_traj.values()])) if per_traj else 0.0,
                "MAE_pipeline_len_std": float(np.std([d["MAE_pipeline_len"] for d in per_traj.values()])) if per_traj else 0.0,
            "PipelineComp_L1_mean": float(np.mean(pipeline_l1s)) if pipeline_l1s else 0.0,
            "PipelineComp_L1_std": float(np.std(pipeline_l1s)) if pipeline_l1s else 0.0,
        }

        n_traj = max(1, len(per_traj))

        def ci95(std: float) -> float:
            return 1.96 * std / math.sqrt(n_traj)

        agg["RMSE_inventory_CI95"] = ci95(agg["RMSE_inventory_std"])
        agg["RMSE_backlog_CI95"] = ci95(agg["RMSE_backlog_std"])
        agg["MAE_pipeline_len_CI95"] = ci95(agg["MAE_pipeline_len_std"])
        agg["PipelineComp_L1_CI95"] = ci95(agg["PipelineComp_L1_std"])

        # MSE computations
        if len(se_inv_all) > 0:
            mse_inv = float(np.mean(np.concatenate(se_inv_all)))
            mse_bk = float(np.mean(np.concatenate(se_bk_all)))
            mse_pl = float(np.mean(np.concatenate(se_pl_all)))
            mse_t = float(np.mean(np.concatenate(se_t_all)))
        else:
            mse_inv = mse_bk = mse_pl = mse_t = 0.0

        mse_per_dim = {
            "inventory": mse_inv,
            "backlog": mse_bk,
            "pipeline_len": mse_pl,
            "t": mse_t,
        }
        # Exclude 't' from joint MSE to avoid dilution
        mse_joint_state = float(np.mean([mse_inv, mse_bk, mse_pl]))

        out = {
            "per_trajectory": per_traj,
            "aggregate": agg,
            "distributional": {
                "Wasserstein_per_t_joint": float(wass_joint),
                "MMD_joint": float(mmd_val),
                "MMD2_joint": float(mmd2_val),
            },
            "MSE_per_dim": mse_per_dim,
            "MSE_joint_state": mse_joint_state,
            "MSE_joint": mse_joint_state,
        }
        return out


# ------------- Calibrasim parameter container and adapter -------------

@dataclass
class FittedParams:
    """
    Container for parameters needed by the simulator and modular engine.

    Attributes:
        decision_weights: Decision head weights (not used in playback; kept for compatibility).
        layer_weights: Layer weights (not used; kept for compatibility).
        info_params: Information dynamics parameters (not used; kept for compatibility).
        noise_params: Noise settings (e.g., temperature; not used; kept for compatibility).
        module_params: Module-specific parameters.
        engine_type: Compatibility tag (string).
        meta: Arbitrary metadata dict (seed, notes).
    """
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of params."""
        return asdict(self)


class ParamsAdapter(ABC):
    """
    Adapter to apply/capture FittedParams to/from the simulator parameter system.
    """
    @abstractmethod
    def apply(self, simulation: "SupplySimulation", params: FittedParams) -> None:
        """Apply params via simulation.set_params() and write parameters_used.json."""
        pass

    @abstractmethod
    def capture(self, simulation: "SupplySimulation") -> FittedParams:
        """Capture current effective params from the simulation into a FittedParams object."""
        pass

    @abstractmethod
    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """Check frozen parameters and return warnings mapping param->message."""
        pass


class SupplyParamsAdapter(ParamsAdapter):
    """
    Concrete adapter mapping FittedParams.module_params to BeerGameSimulator parameters.

    It also considers parameter_definitions.json for 'frozen' constraints.
    """
    def __init__(self, param_definitions_path: Optional[str] = None, persist: bool = True) -> None:
        """Initialize with optional path to parameter_definitions.json and persistence flag."""
        self.param_definitions_path = param_definitions_path
        self.definitions = self._load_definitions(param_definitions_path)
        self.persist = bool(persist)

    def _load_definitions(self, path: Optional[str]) -> Dict[str, Any]:
        """
        Load parameter definitions to check 'frozen' and bounds.
        """
        default_defs = {
            "parameters": [
                {"name": "lead_time_L", "dtype": "int", "bounds": [1, 8], "frozen": False, "default": 2},
                {"name": "arrival_flag", "dtype": "int", "bounds": [0, 1], "frozen": False, "default": 0},
                {"name": "demand_family", "dtype": "str", "frozen": False, "default": "Poisson"},
                {"name": "poisson_lambda", "dtype": "float", "bounds": [0.0, 20.0], "frozen": False, "default": 5.0},
                {"name": "negbin_mu", "dtype": "float", "bounds": [0.0, 20.0], "frozen": False, "default": 5.0},
                {"name": "negbin_r", "dtype": "float", "bounds": [0.1, 50.0], "frozen": False, "default": 5.0},
                {"name": "ar1_mu", "dtype": "float", "bounds": [0.0, 20.0], "frozen": False, "default": 5.0},
                {"name": "ar1_phi", "dtype": "float", "bounds": [-0.95, 0.95], "frozen": False, "default": 0.0},
                {"name": "ar1_sigma", "dtype": "float", "bounds": [0.0, 10.0], "frozen": False, "default": 1.0},
                {"name": "seasonal_amplitude", "dtype": "float", "bounds": [0.0, 10.0], "frozen": False, "default": 0.0},
                {"name": "seasonal_period", "dtype": "int", "bounds": [2, 30], "frozen": False, "default": 7},
            ]
        }
        if path is None or not os.path.isfile(path):
            warnings.warn("parameter_definitions.json not found; using default parameter definitions.")
            return default_defs
        try:
            with open(path, "r") as f:
                defs = json.load(f)
            return defs
        except Exception:
            warnings.warn("Failed to load parameter_definitions.json; using default definitions.")
            return default_defs

    def _is_frozen(self, name: str) -> bool:
        """Check if a parameter is frozen per definitions."""
        try:
            for p in self.definitions.get("parameters", []):
                if p.get("name") == name:
                    return bool(p.get("frozen", False))
        except Exception:
            return False
        return False

    def _bounds(self, name: str) -> Optional[Tuple[float, float]]:
        """Return bounds if available else None."""
        try:
            for p in self.definitions.get("parameters", []):
                if p.get("name") == name:
                    b = p.get("bounds", None)
                    if b is None:
                        return None
                    return float(b[0]), float(b[1])
        except Exception:
            return None
        return None

    def _coerce(self, name: str, value: Any) -> Any:
        """Coerce value to dtype defined in parameter_definitions if available."""
        dtype_map = {}
        for p in self.definitions.get("parameters", []):
            dtype_map[p.get("name")] = p.get("dtype")
        dt = dtype_map.get(name, None)
        if dt is None:
            return value
        try:
            if dt == "int":
                return int(round(float(value)))
            if dt == "float":
                return float(value)
            if dt == "str":
                return str(value)
            if dt == "bool":
                vv = str(value).lower().strip()
                return vv in {"1", "true", "t", "yes", "y"}
        except Exception:
            return value
        return value

    def _clip_bounds(self, name: str, value: Any) -> Any:
        """Clip numeric value to bounds if provided."""
        b = self._bounds(name)
        if b is None:
            return value
        lo, hi = b
        try:
            val = float(value)
            val = min(hi, max(lo, val))
            if isinstance(value, int):
                return int(round(val))
            return val
        except Exception:
            return value

    def apply(self, simulation: "SupplySimulation", params: FittedParams) -> None:
        """
        Apply parameters to the simulation.

        Maps module_params["supply"]["lead_time_L"], ["arrival_flag"], and
        module_params["demand"](... family-specific ...) to BeerGameSimulator parameters.
        Frozen parameters are ignored with a warning.

        Writes parameters_used.json with final applied values (optional).
        """
        module_params = params.module_params or {}
        supply = module_params.get("supply", {})
        demand = module_params.get("demand", {})
        applied: Dict[str, Any] = simulation.simulator.get_params()

        # Supply
        for k in ["lead_time_L", "arrival_flag"]:
            if k in supply:
                if self._is_frozen(k):
                    warnings.warn(f"Ignoring override for frozen parameter '{k}'.")
                else:
                    v = self._coerce(k, supply[k])
                    v = self._clip_bounds(k, v)
                    applied[k] = v

        # Demand
        fam = demand.get("demand_family", applied.get("demand_family", "Poisson"))
        if self._is_frozen("demand_family"):
            pass
        else:
            applied["demand_family"] = str(fam)
        if fam == "Poisson":
            for k in ["poisson_lambda", "seasonal_amplitude", "seasonal_period"]:
                if k in demand:
                    if self._is_frozen(k):
                        warnings.warn(f"Ignoring override for frozen parameter '{k}'.")
                    else:
                        v = self._coerce(k, demand[k])
                        v = self._clip_bounds(k, v)
                        applied[k] = v
        elif fam == "NegBin":
            for k in ["negbin_mu", "negbin_r"]:
                if k in demand:
                    if self._is_frozen(k):
                        warnings.warn(f"Ignoring override for frozen parameter '{k}'.")
                    else:
                        v = self._coerce(k, demand[k])
                        v = self._clip_bounds(k, v)
                        applied[k] = v
        elif fam == "AR1":
            for k in ["ar1_mu", "ar1_phi", "ar1_sigma"]:
                if k in demand:
                    if self._is_frozen(k):
                        warnings.warn(f"Ignoring override for frozen parameter '{k}'.")
                    else:
                        v = self._coerce(k, demand[k])
                        v = self._clip_bounds(k, v)
                        applied[k] = v
        else:
            warnings.warn(f"Unknown demand_family '{fam}' in adapter; keeping previous.")

        simulation.simulator.set_params(applied)
        # Persist
        if self.persist:
            try:
                os.makedirs(simulation.artifacts_dir, exist_ok=True)
                with open(os.path.join(simulation.artifacts_dir, "parameters_used.json"), "w") as f:
                    json.dump(applied, f, indent=2)
            except Exception as exc:
                warnings.warn(f"Failed to save parameters_used.json: {exc}")

    def capture(self, simulation: "SupplySimulation") -> FittedParams:
        """
        Capture current simulator parameters into a FittedParams container.
        """
        p = simulation.simulator.get_params()
        supply = {
            "lead_time_L": int(p.get("lead_time_L", 2)),
            "arrival_flag": int(p.get("arrival_flag", 0)),
        }
        demand_fam = p.get("demand_family", "Poisson")
        demand: Dict[str, Any] = {"demand_family": demand_fam}
        if demand_fam == "Poisson":
            demand["poisson_lambda"] = float(p.get("poisson_lambda", 5.0))
            demand["seasonal_amplitude"] = float(p.get("seasonal_amplitude", 0.0))
            demand["seasonal_period"] = int(p.get("seasonal_period", 7))
        elif demand_fam == "NegBin":
            demand["negbin_mu"] = float(p.get("negbin_mu", 5.0))
            demand["negbin_r"] = float(p.get("negbin_r", 5.0))
        elif demand_fam == "AR1":
            demand["ar1_mu"] = float(p.get("ar1_mu", 5.0))
            demand["ar1_phi"] = float(p.get("ar1_phi", 0.0))
            demand["ar1_sigma"] = float(p.get("ar1_sigma", 1.0))
        return FittedParams(
            decision_weights={},
            layer_weights={},
            info_params={},
            noise_params={},
            module_params={"supply": supply, "demand": demand},
            engine_type="calibrasim",
            meta={"captured_at": time.time()},
        )

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        """
        Validate a FittedParams object against frozen constraints.

        Returns:
            Dict mapping param names to warning strings for frozen violations.
        """
        warnings_map: Dict[str, str] = {}
        mp = params.module_params or {}
        for section, d in mp.items():
            for k, _v in d.items():
                if self._is_frozen(k):
                    warnings_map[k] = f"Parameter '{k}' is frozen; override will be ignored."
        return warnings_map


# ------------- Calibrators -------------

class Calibrator(ABC):
    """
    Pluggable calibrator interface with a stable evaluation callback signature.

    Implementations:
        - fit: returns FittedParams optimized on the training window.
    """
    @abstractmethod
    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: BeerGameSimulator,
        evaluator: Evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Fit parameters using the training bundle.
        """
        pass


def evaluate_params(
    simulator: BeerGameSimulator,
    params: FittedParams,
    window: Tuple[int, int],
    bundle: Dict[str, Any],
    eval_split: str = "val",
    evaluator: Optional[Evaluator] = None,
) -> Dict[str, Any]:
    """
    Apply params, run forward simulation on selected split within window, and return metrics dict.

    Args:
        simulator: BeerGameSimulator.
        params: FittedParams to apply.
        window: (start, end) within each trajectory.
        bundle: Data bundle containing trajectories.
        eval_split: One of {'train','val','test','test_ood'}.
        evaluator: Optional Evaluator instance.

    Returns:
        Metrics dict without placeholder fields.
    """
    # Apply params via adapter without persisting in inner loop
    sim = BeerGameSimulator(simulator.get_params())
    adapter = SupplyParamsAdapter(persist=False)
    adapter.apply(SupplySimulation(sim, bundle, artifacts_dir=bundle.get("artifacts_dir", ".")), params)

    # Choose trajectories
    if eval_split == "train":
        eval_trajs: List[TrajectoryData] = bundle["train_trajectories"]
    elif eval_split == "val":
        eval_trajs = bundle["val_trajectories"]
    elif eval_split == "test":
        eval_trajs = bundle.get("test_trajectories", [])
    else:
        eval_trajs = bundle.get("test_ood_trajectories", [])

    results = sim.rollout(eval_trajs)

    # Slice results to the provided window with per-trajectory clamping
    start, end = int(window[0]), int(window[1])

    def slice_array(arr: np.ndarray, s: int, e: int) -> np.ndarray:
        T = arr.shape[0]
        s0 = max(0, min(s, T - 1)) if T > 0 else 0
        e0 = max(0, min(e, T - 1)) if T > 0 else -1
        if T == 0 or e0 < s0:
            return arr[:0]
        return arr[s0:e0 + 1]

    def slice_list(lst: List[Any], s: int, e: int) -> List[Any]:
        T = len(lst)
        s0 = max(0, min(s, T - 1)) if T > 0 else 0
        e0 = max(0, min(e, T - 1)) if T > 0 else -1
        if T == 0 or e0 < s0:
            return []
        return lst[s0:e0 + 1]

    def slice_res(res: SimulationResults, s: int, e: int) -> SimulationResults:
        inv_s, b_s, pl_s, occ_s = {}, {}, {}, {}
        inv_o, b_o, pl_o, occ_o = {}, {}, {}, {}
        for tid in res.inventory_sim.keys():
            inv_s[tid] = slice_array(res.inventory_sim[tid], s, e)
            b_s[tid] = slice_array(res.backlog_sim[tid], s, e)
            pl_s[tid] = slice_array(res.pipeline_len_sim[tid], s, e)
            occ_s[tid] = slice_list(res.pipeline_occ_sim[tid], s, e)
            inv_o[tid] = slice_array(res.inventory_obs[tid], s, e)
            b_o[tid] = slice_array(res.backlog_obs[tid], s, e)
            pl_o[tid] = slice_array(res.pipeline_len_obs[tid], s, e)
            occ_o[tid] = slice_list(res.pipeline_occ_obs[tid], s, e)
        return SimulationResults(inv_s, b_s, pl_s, occ_s, inv_o, b_o, pl_o, occ_o)

    results_window = slice_res(results, start, end)
    ev = evaluator or Evaluator()
    metrics = ev.compute_metrics(results_window, n_samples_wass_mmd=ev.n_samples, mmd_sigma=1.0)
    agg = metrics.get("aggregate", {})
    rmse_inv = agg.get("RMSE_inventory_mean", 0.0)
    rmse_b = agg.get("RMSE_backlog_mean", 0.0)
    mae_pl = agg.get("MAE_pipeline_len_mean", 0.0)
    out = {
        "RMSE_aggregate": float(np.mean([rmse_inv, rmse_b])),
        "MAE_aggregate": float(mae_pl),
        "distributional": metrics.get("distributional", {}),
        "aggregate": agg,
        "per_trajectory": metrics.get("per_trajectory", {}),
        "MSE_per_dim": metrics.get("MSE_per_dim", {}),
        "MSE_joint": metrics.get("MSE_joint", 0.0),
    }
    return out


class LogitHeadCalibrator(Calibrator):
    """
    Fits a logistic decision head from micro-transitions if available; otherwise degrades gracefully.

    In this supply setting with action playback, micro-transitions are unavailable; this calibrator
    returns the captured current parameters with a meta note.
    """
    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: BeerGameSimulator,
        evaluator: Evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """Return current captured params with a note indicating degraded behavior."""
        adapter = params_adapter or SupplyParamsAdapter()
        sim_wrapper = SupplySimulation(simulator, bundle, artifacts_dir=artifacts_dir or ".")
        fitted = adapter.capture(sim_wrapper)
        fitted.meta.update({
            "calibrator": "logit_head",
            "note": "Degraded: micro-transitions unavailable; returning captured params.",
            "seed": seed,
        })
        return fitted


class RandomSearchCalibrator(Calibrator):
    """
    Black-box random search over selected simulator parameters.

    Parameters searched:
      - lead_time_L via small grid search across adapter bounds
      - arrival_flag optionally explored (default locked to 0 for determinism)
      - demand parameters based on chosen family (Poisson/NegBin/AR1)

    If demand_family='auto', searches families evenly and chooses best.
    """
    def __init__(
        self,
        demand_family: str = "auto",
        w_wass: float = 1.0,
        w_mmd: float = 0.0,
        w_rmse: float = 0.5,
        explore_arrival: bool = False,
        persist_trials: bool = False,
        persist_every: int = 10,
    ) -> None:
        """Initialize with defaults and objective weights."""
        self.demand_family = demand_family
        self.w_wass = float(w_wass)
        self.w_mmd = float(w_mmd)
        self.w_rmse = float(w_rmse)
        self.explore_arrival = bool(explore_arrival)
        self.persist_trials = bool(persist_trials)
        self.persist_every = int(max(1, persist_every))

    def _sample_params(
        self,
        rng: np.random.RandomState,
        base: FittedParams,
        bounds: Dict[str, Tuple[float, float]],
        chosen_family: str,
        lead_time: int,
        arrival_flag: int,
    ) -> FittedParams:
        """
        Propose a new FittedParams sample by mutating supply and demand module_params.
        """
        fp = json.loads(json.dumps(base.to_dict()))
        module_params = fp.get("module_params", {})
        supply = module_params.get("supply", {})
        demand = module_params.get("demand", {})
        supply["lead_time_L"] = int(lead_time)
        supply["arrival_flag"] = int(arrival_flag)
        demand["demand_family"] = chosen_family
        if chosen_family == "Poisson":
            lo, hi = bounds.get("poisson_lambda", (0.0, 20.0))
            demand["poisson_lambda"] = float(rng.uniform(lo, hi))
        elif chosen_family == "NegBin":
            lo_mu, hi_mu = bounds.get("negbin_mu", (0.0, 20.0))
            lo_r, hi_r = bounds.get("negbin_r", (0.1, 50.0))
            demand["negbin_mu"] = float(rng.uniform(lo_mu, hi_mu))
            demand["negbin_r"] = float(rng.uniform(lo_r, hi_r))
        else:
            lo_mu, hi_mu = bounds.get("ar1_mu", (0.0, 20.0))
            lo_phi, hi_phi = bounds.get("ar1_phi", (-0.95, 0.95))
            lo_s, hi_s = bounds.get("ar1_sigma", (0.0, 10.0))
            demand["ar1_mu"] = float(rng.uniform(lo_mu, hi_mu))
            demand["ar1_phi"] = float(rng.uniform(lo_phi, hi_phi))
            demand["ar1_sigma"] = float(rng.uniform(lo_s, hi_s))
        module_params["supply"] = supply
        module_params["demand"] = demand
        fp["module_params"] = module_params
        return FittedParams(
            decision_weights={},
            layer_weights={},
            info_params={},
            noise_params={},
            module_params=module_params,
            engine_type="calibrasim",
            meta={"proposed": True},
        )

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: BeerGameSimulator,
        evaluator: Evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Perform random search/grid over lead time and demand; return best FittedParams by composite objective.

        Optimizes on the training trajectories (within train_window). Validation/test are reserved for evaluation.
        """
        rng = np.random.RandomState(seed + 4321)
        adapter = params_adapter or SupplyParamsAdapter()
        sim_wrapper = SupplySimulation(simulator, bundle, artifacts_dir=artifacts_dir or ".")
        base_params = adapter.capture(sim_wrapper)

        # Bounds from adapter definitions
        bounds: Dict[str, Tuple[float, float]] = {}
        for name in ["lead_time_L", "poisson_lambda", "negbin_mu", "negbin_r", "ar1_mu", "ar1_phi", "ar1_sigma"]:
            b = adapter._bounds(name)  # type: ignore
            if b is not None:
                bounds[name] = b
        L_lo, L_hi = bounds.get("lead_time_L", (1.0, 8.0))
        grid_L = list(range(int(L_lo), int(L_hi) + 1))

        families_to_try = [self.demand_family] if self.demand_family.lower() != "auto" else ["Poisson", "NegBin"]
        family_budgets = [budget] if len(families_to_try) == 1 else [max(1, budget // len(families_to_try))] * len(families_to_try)

        best_score = float("inf")
        best_fp = base_params
        trials_meta = []

        os.makedirs(artifacts_dir or ".", exist_ok=True)

        trial_idx = 0
        for fam, fam_budget in zip(families_to_try, family_budgets):
            # Distribute budget across L grid
            per_L_budget = max(1, fam_budget // max(1, len(grid_L)))
            for L in grid_L:
                for i in range(per_L_budget):
                    trial_dir = os.path.join(artifacts_dir or ".", f"trial_{trial_idx}")
                    if self.persist_trials and (i % self.persist_every == 0):
                        os.makedirs(trial_dir, exist_ok=True)

                    arr_flag = 0 if not self.explore_arrival else int(rng.choice([0, 1]))
                    candidate = self._sample_params(rng, base_params, bounds, fam, L, arr_flag)
                    sim_tmp = BeerGameSimulator(simulator.get_params())
                    sim_bndl = {
                        "train_trajectories": bundle["train_trajectories"],
                        "val_trajectories": bundle["val_trajectories"],
                        "test_trajectories": bundle.get("test_trajectories", []),
                        "test_ood_trajectories": bundle.get("test_ood_trajectories", []),
                        "artifacts_dir": trial_dir,
                    }
                    metrics = evaluate_params(sim_tmp, candidate, train_window, sim_bndl, eval_split="train", evaluator=evaluator)
                    dist = metrics.get("distributional", {})
                    wass = float(dist.get("Wasserstein_per_t_joint", 0.0))
                    mmd = float(dist.get("MMD_joint", 0.0))
                    rmse_agg = float(metrics.get("RMSE_aggregate", 0.0))
                    # Composite objective (lower is better)
                    score = self.w_rmse * rmse_agg
                    if np.isfinite(wass):
                        score += self.w_wass * wass
                    score += self.w_mmd * mmd

                    trials_meta.append({"trial": trial_idx, "score": score, "params": candidate.to_dict(), "metrics": metrics})
                    try:
                        if self.persist_trials and (i % self.persist_every == 0):
                            with open(os.path.join(trial_dir, "params_applied.json"), "w") as f:
                                json.dump(candidate.to_dict(), f, indent=2)
                            with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
                                json.dump(metrics, f, indent=2)
                    except Exception:
                        pass
                    if score < best_score:
                        best_score = score
                        best_fp = candidate
                    trial_idx += 1

        report = {
            "calibrator": "random_search",
            "budget": budget,
            "best_score": best_score,
            "best_params": best_fp.to_dict(),
            "trials": trials_meta[:50],  # truncate report to limit size
        }
        try:
            with open(os.path.join(artifacts_dir or ".", "calibration_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            best_dir = os.path.join(artifacts_dir or ".", "best")
            os.makedirs(best_dir, exist_ok=True)
            with open(os.path.join(best_dir, "fitted_params.json"), "w") as f:
                json.dump(best_fp.to_dict(), f, indent=2)
        except Exception:
            pass

        best_fp.meta.update({"calibrator": "random_search", "seed": seed})
        return best_fp


class SNPECalibrator(Calibrator):
    """
    True Simulation-Based Inference using SNPE to infer posterior over parameters.

    If torch/sbi is unavailable, falls back to RandomSearchCalibrator gracefully.
    """
    def __init__(self, num_simulations: int = 500, num_posterior_samples: int = 1000, sampling_timeout: int = 60) -> None:
        """Initialize SNPE calibrator configuration."""
        self.num_simulations = int(num_simulations)
        self.num_posterior_samples = int(num_posterior_samples)
        self.sampling_timeout = int(sampling_timeout)

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: BeerGameSimulator,
        evaluator: Evaluator,
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        """
        Perform SNPE; infer posterior; pick mean sample as point estimate; return FittedParams.

        Fallback to random search if dependencies are missing or errors occur.
        """
        if not (TORCH_AVAILABLE and SBI_AVAILABLE):
            warnings.warn("SBI/torch not available; falling back to RandomSearchCalibrator.")
            return RandomSearchCalibrator().fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)

        adapter = params_adapter or SupplyParamsAdapter()
        sim_wrapper = SupplySimulation(simulator, bundle, artifacts_dir=artifacts_dir or ".")
        base_params = adapter.capture(sim_wrapper)

        # Observed summary features computed directly from observed train data (no simulation)
        def features_from_observed(trajs: List[TrajectoryData], window: Tuple[int, int]) -> np.ndarray:
            start, end = int(window[0]), int(window[1])
            inv_vals, b_vals, pl_vals = [], [], []
            dinv_vals, db_vals = [], []
            for tr in trajs:
                inv = tr.inventory_obs
                bk = tr.backlog_obs
                pl = tr.pipeline_len_obs_counts
                T = min(len(inv), len(bk), len(pl))
                if T == 0:
                    continue
                s = max(0, min(start, T - 1))
                e = max(0, min(end, T - 1))
                if e < s:
                    s, e = 0, T - 1
                inv_w = inv[s:e+1]
                bk_w = bk[s:e+1]
                pl_w = pl[s:e+1]
                inv_vals.append(inv_w)
                b_vals.append(bk_w)
                pl_vals.append(pl_w)
                if len(inv_w) > 1:
                    dinv_vals.append(np.diff(inv_w))
                if len(bk_w) > 1:
                    db_vals.append(np.diff(bk_w))
            def _agg(arrs: List[np.ndarray]) -> Tuple[float, float]:
                if len(arrs) == 0:
                    return 0.0, 0.0
                x = np.concatenate([np.asarray(a).ravel() for a in arrs], axis=0)
                return float(np.mean(x)), float(np.std(x))
            inv_mean, inv_std = _agg(inv_vals)
            bk_mean, bk_std = _agg(b_vals)
            pl_mean, pl_std = _agg(pl_vals)
            dinv_mean, dinv_std = _agg(dinv_vals)
            db_mean, db_std = _agg(db_vals)
            feats = np.array([inv_mean, inv_std, bk_mean, bk_std, pl_mean, pl_std, dinv_mean, dinv_std, db_mean, db_std], dtype=np.float32)
            return feats

        x_obs = features_from_observed(bundle["train_trajectories"], train_window)
        x_obs_t = torch.tensor(x_obs, dtype=torch.float32)

        # Prior bounds for parameters (lead_time_L, arrival_flag, demand params) from adapter definitions
        b_L = adapter._bounds("lead_time_L")  # type: ignore
        L_low, L_high = (1.0, 8.0) if b_L is None else (float(b_L[0]), float(b_L[1]))
        demand_family = base_params.module_params.get("demand", {}).get("demand_family", "Poisson")
        low_list = [L_low, 0.0]
        high_list = [L_high, 1.0]
        param_names = ["lead_time_L", "arrival_flag"]
        if demand_family == "Poisson":
            low_list += [0.0]
            high_list += [20.0]
            param_names += ["poisson_lambda"]
        elif demand_family == "NegBin":
            low_list += [0.0, 0.1]
            high_list += [20.0, 50.0]
            param_names += ["negbin_mu", "negbin_r"]
        elif demand_family == "AR1":
            low_list += [0.0, -0.95, 0.0]
            high_list += [20.0, 0.95, 10.0]
            param_names += ["ar1_mu", "ar1_phi", "ar1_sigma"]
        else:
            demand_family = "Poisson"
            low_list += [0.0]
            high_list += [20.0]
            param_names += ["poisson_lambda"]

        low_t = torch.tensor(low_list, dtype=torch.float32)
        high_t = torch.tensor(high_list, dtype=torch.float32)
        prior = sbi_utils.BoxUniform(low=low_t, high=high_t)

        # Prepare reusable simulator and adapter (no file I/O during SBI)
        sim_tmp = BeerGameSimulator(simulator.get_params())
        adapter_tmp = SupplyParamsAdapter(persist=False)

        def sim_wrapper_fn(theta_t: torch.Tensor) -> torch.Tensor:
            # Handle batch input: sbi_simulate_for_sbi may pass batches
            theta_np = theta_t.detach().cpu().numpy().astype(float)
            # Flatten to 1D if needed (handle both single sample and batch)
            if theta_np.ndim > 1:
                # If batch, process each sample separately and stack results
                results = []
                for theta_single in theta_np:
                    theta = theta_single.flatten()
                    # Map theta to FittedParams
                    supply = {
                        "lead_time_L": int(round(float(theta[0]))),
                        "arrival_flag": int(round(float(theta[1]))),
                    }
                    demand: Dict[str, Any] = {"demand_family": demand_family}
                    idx = 2
                    if demand_family == "Poisson":
                        demand["poisson_lambda"] = float(theta[idx])
                    elif demand_family == "NegBin":
                        demand["negbin_mu"] = float(theta[idx])
                        demand["negbin_r"] = float(theta[idx + 1])
                    elif demand_family == "AR1":
                        demand["ar1_mu"] = float(theta[idx])
                        demand["ar1_phi"] = float(theta[idx + 1])
                        demand["ar1_sigma"] = float(theta[idx + 2])

                    fp = FittedParams(decision_weights={}, layer_weights={}, info_params={}, noise_params={}, module_params={"supply": supply, "demand": demand})
                    # Simulate train trajectories
                    try:
                        adapter_tmp.apply(SupplySimulation(sim_tmp, bundle, artifacts_dir=artifacts_dir or "."), fp)
                        res = sim_tmp.rollout(bundle["train_trajectories"])

                        # Build features from simulated sequences within window
                        start, end = int(train_window[0]), int(train_window[1])
                        inv_vals, b_vals, pl_vals = [], [], []
                        dinv_vals, db_vals = [], []
                        for tid in res.inventory_sim.keys():
                            inv = res.inventory_sim[tid]
                            bk = res.backlog_sim[tid]
                            pl = res.pipeline_len_sim[tid]
                            T = min(len(inv), len(bk), len(pl))
                            if T == 0:
                                continue
                            s = max(0, min(start, T - 1))
                            e = max(0, min(end, T - 1))
                            if e < s:
                                s, e = 0, T - 1
                            inv_w = inv[s:e+1]
                            bk_w = bk[s:e+1]
                            pl_w = pl[s:e+1]
                            inv_vals.append(inv_w)
                            b_vals.append(bk_w)
                            pl_vals.append(pl_w)
                            if len(inv_w) > 1:
                                dinv_vals.append(np.diff(inv_w))
                            if len(bk_w) > 1:
                                db_vals.append(np.diff(bk_w))
                        def _agg(arrs: List[np.ndarray]) -> Tuple[float, float]:
                            if len(arrs) == 0:
                                return 0.0, 0.0
                            x = np.concatenate([np.asarray(a).ravel() for a in arrs], axis=0)
                            return float(np.mean(x)), float(np.std(x))
                        inv_mean, inv_std = _agg(inv_vals)
                        bk_mean, bk_std = _agg(b_vals)
                        pl_mean, pl_std = _agg(pl_vals)
                        dinv_mean, dinv_std = _agg(dinv_vals)
                        db_mean, db_std = _agg(db_vals)
                        feats = np.array([inv_mean, inv_std, bk_mean, bk_std, pl_mean, pl_std, dinv_mean, dinv_std, db_mean, db_std], dtype=np.float32)
                        results.append(torch.tensor(feats, dtype=torch.float32))
                    except Exception:
                        # Return NaNs to be filtered
                        results.append(torch.full((10,), float("nan"), dtype=torch.float32))
                return torch.stack(results)
            
            # Single sample case
            theta = theta_np.flatten()
            # Map theta to FittedParams
            supply = {
                "lead_time_L": int(round(float(theta[0]))),
                "arrival_flag": int(round(float(theta[1]))),
            }
            demand: Dict[str, Any] = {"demand_family": demand_family}
            idx = 2
            if demand_family == "Poisson":
                demand["poisson_lambda"] = float(theta[idx])
            elif demand_family == "NegBin":
                demand["negbin_mu"] = float(theta[idx])
                demand["negbin_r"] = float(theta[idx + 1])
            elif demand_family == "AR1":
                demand["ar1_mu"] = float(theta[idx])
                demand["ar1_phi"] = float(theta[idx + 1])
                demand["ar1_sigma"] = float(theta[idx + 2])

            fp = FittedParams(decision_weights={}, layer_weights={}, info_params={}, noise_params={}, module_params={"supply": supply, "demand": demand})
            # Simulate train trajectories
            try:
                adapter_tmp.apply(SupplySimulation(sim_tmp, bundle, artifacts_dir=artifacts_dir or "."), fp)
                res = sim_tmp.rollout(bundle["train_trajectories"])

                # Build features from simulated sequences within window
                start, end = int(train_window[0]), int(train_window[1])
                inv_vals, b_vals, pl_vals = [], [], []
                dinv_vals, db_vals = [], []
                for tid in res.inventory_sim.keys():
                    inv = res.inventory_sim[tid]
                    bk = res.backlog_sim[tid]
                    pl = res.pipeline_len_sim[tid]
                    T = min(len(inv), len(bk), len(pl))
                    if T == 0:
                        continue
                    s = max(0, min(start, T - 1))
                    e = max(0, min(end, T - 1))
                    if e < s:
                        s, e = 0, T - 1
                    inv_w = inv[s:e+1]
                    bk_w = bk[s:e+1]
                    pl_w = pl[s:e+1]
                    inv_vals.append(inv_w)
                    b_vals.append(bk_w)
                    pl_vals.append(pl_w)
                    if len(inv_w) > 1:
                        dinv_vals.append(np.diff(inv_w))
                    if len(bk_w) > 1:
                        db_vals.append(np.diff(bk_w))
                def _agg(arrs: List[np.ndarray]) -> Tuple[float, float]:
                    if len(arrs) == 0:
                        return 0.0, 0.0
                    x = np.concatenate([np.asarray(a).ravel() for a in arrs], axis=0)
                    return float(np.mean(x)), float(np.std(x))
                inv_mean, inv_std = _agg(inv_vals)
                bk_mean, bk_std = _agg(b_vals)
                pl_mean, pl_std = _agg(pl_vals)
                dinv_mean, dinv_std = _agg(dinv_vals)
                db_mean, db_std = _agg(db_vals)
                feats = np.array([inv_mean, inv_std, bk_mean, bk_std, pl_mean, pl_std, dinv_mean, dinv_std, db_mean, db_std], dtype=np.float32)
                return torch.tensor(feats, dtype=torch.float32)
            except Exception:
                # Return NaNs to be filtered
                return torch.full((10,), float("nan"), dtype=torch.float32)

        # Collect simulations
        try:
            thetas, xs = sbi_simulate_for_sbi(sim_wrapper_fn, prior, num_simulations=self.num_simulations)
            nan_mask = torch.any(~torch.isfinite(xs), dim=1)
            thetas = thetas[~nan_mask]
            xs = xs[~nan_mask]

            inference = NPE(prior=prior)
            density_estimator = inference.append_simulations(thetas, xs).train()
            posterior = inference.build_posterior(density_estimator)

            # Observed feature vector x_o computed directly from data
            x_o = x_obs_t

            # Sample posterior
            start_time = time.time()
            samples_list = []
            chunk = min(1024, self.num_posterior_samples)
            while len(samples_list) < self.num_posterior_samples and (time.time() - start_time) < self.sampling_timeout:
                need = min(chunk, self.num_posterior_samples - len(samples_list))
                s = posterior.sample((need,), x=x_o)
                samples_list.append(s.detach().cpu())
            if len(samples_list) == 0:
                samples = (low_t + (high_t - low_t) / 2.0).unsqueeze(0)
            else:
                samples = torch.cat(samples_list, dim=0)
            theta_opt = torch.mean(samples, dim=0).detach().cpu().numpy().astype(float)

            # Map back to FittedParams
            supply = {
                "lead_time_L": int(round(float(theta_opt[0]))),
                "arrival_flag": int(round(float(theta_opt[1]))),
            }
            demand: Dict[str, Any] = {"demand_family": demand_family}
            if demand_family == "Poisson":
                demand["poisson_lambda"] = float(theta_opt[2])
            elif demand_family == "NegBin":
                demand["negbin_mu"] = float(theta_opt[2])
                demand["negbin_r"] = float(theta_opt[3])
            elif demand_family == "AR1":
                demand["ar1_mu"] = float(theta_opt[2])
                demand["ar1_phi"] = float(theta_opt[3])
                demand["ar1_sigma"] = float(theta_opt[4])

            fitted = FittedParams(
                decision_weights={},
                layer_weights={},
                info_params={},
                noise_params={},
                module_params={"supply": supply, "demand": demand},
                engine_type="calibrasim",
                meta={"calibrator": "snpe", "seed": seed},
            )
            best_dir = os.path.join(artifacts_dir or ".", "best")
            os.makedirs(best_dir, exist_ok=True)
            with open(os.path.join(best_dir, "fitted_params.json"), "w") as f:
                json.dump(fitted.to_dict(), f, indent=2)
            return fitted
        except Exception as exc:
            traceback.print_exc()
            warnings.warn(f"SBI calibrator failed ({exc}); falling back to RandomSearch.")
            return RandomSearchCalibrator().fit(bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter)


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str]) -> Calibrator:
    """
    Get a calibrator by name; optional config JSON may set its kwargs.
    """
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Unknown calibrator: {name}")
    kwargs: Dict[str, Any] = {}
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                kwargs.update(cfg)
        except Exception:
            warnings.warn("Failed to load calibrator config; using defaults.")
    return CALIBRATOR_REGISTRY[name](**kwargs)


# ------------- Simulation wrapper -------------

class SupplySimulation:
    """
    High-level simulation wrapper holding the simulator, datasets, and artifacts path.

    Provides:
        - run(start, end, split): runs rollout and window slicing for given split.
        - evaluate(): uses Evaluator on current params
        - save_results(): writes metrics to artifacts
    """
    def __init__(self, simulator: BeerGameSimulator, bundle: Dict[str, Any], artifacts_dir: str = ".", evaluator: Optional[Evaluator] = None, enable_module_io: bool = False) -> None:
        """Initialize with simulator, data bundle, and artifacts path."""
        self.simulator = simulator
        self.bundle = bundle
        self.artifacts_dir = artifacts_dir
        self.evaluator = evaluator or Evaluator()
        self.enable_module_io = bool(enable_module_io)

    def run(self, start: int, end: int, split: str = "val") -> SimulationResults:
        """
        Run simulator rollout on selected split, then slice to [start, end].

        Args:
            start: start index inclusive.
            end: end index inclusive.
            split: 'val', 'test', or 'test_ood'.

        Returns:
            SimulationResults sliced to window.
        """
        if split == "val":
            results = self.simulator.rollout(self.bundle["val_trajectories"])
        elif split == "test":
            results = self.simulator.rollout(self.bundle.get("test_trajectories", []))
        else:
            results = self.simulator.rollout(self.bundle.get("test_ood_trajectories", []))

        def slice_array(arr: np.ndarray, s: int, e: int) -> np.ndarray:
            T = arr.shape[0]
            s0 = max(0, min(s, T - 1)) if T > 0 else 0
            e0 = max(0, min(e, T - 1)) if T > 0 else -1
            if T == 0 or e0 < s0:
                return arr[:0]
            return arr[s0:e0 + 1]

        def slice_list(lst: List[Any], s: int, e: int) -> List[Any]:
            T = len(lst)
            s0 = max(0, min(s, T - 1)) if T > 0 else 0
            e0 = max(0, min(e, T - 1)) if T > 0 else -1
            if T == 0 or e0 < s0:
                return []
            return lst[s0:e0 + 1]

        inv_s, b_s, pl_s, occ_s = {}, {}, {}, {}
        inv_o, b_o, pl_o, occ_o = {}, {}, {}, {}
        for tid in results.inventory_sim.keys():
            inv_s[tid] = slice_array(results.inventory_sim[tid], start, end)
            b_s[tid] = slice_array(results.backlog_sim[tid], start, end)
            pl_s[tid] = slice_array(results.pipeline_len_sim[tid], start, end)
            occ_s[tid] = slice_list(results.pipeline_occ_sim[tid], start, end)
            inv_o[tid] = slice_array(results.inventory_obs[tid], start, end)
            b_o[tid] = slice_array(results.backlog_obs[tid], start, end)
            pl_o[tid] = slice_array(results.pipeline_len_obs[tid], start, end)
            occ_o[tid] = slice_list(results.pipeline_occ_obs[tid], start, end)
        return SimulationResults(inv_s, b_s, pl_s, occ_s, inv_o, b_o, pl_o, occ_o)

    def evaluate(self, split: str = "val") -> Dict[str, Any]:
        """
        Evaluate current simulator parameters on the chosen split.

        Args:
            split: 'val', 'test', or 'test_ood' depending on bundle content.

        Returns:
            Metrics dictionary.
        """
        if split == "val":
            results = self.simulator.rollout(self.bundle["val_trajectories"])
        elif split == "test":
            results = self.simulator.rollout(self.bundle.get("test_trajectories", []))
        else:
            results = self.simulator.rollout(self.bundle.get("test_ood_trajectories", []))
        metrics = self.evaluator.compute_metrics(results, n_samples_wass_mmd=self.evaluator.n_samples, mmd_sigma=1.0)
        return metrics

    def _sanity_check_metrics(self, metrics: Dict[str, Any]) -> None:
        try:
            agg = metrics.get("aggregate", {})
            n_traj = len(metrics.get("per_trajectory", {}))
            if n_traj > 0 and (agg.get("RMSE_inventory_mean", 0.0) == 0.0 or agg.get("RMSE_backlog_mean", 0.0) == 0.0):
                warnings.warn("Sanity check: RMSE mean is 0.0 while trajectories are non-empty. Verify evaluation.")
        except Exception:
            pass

    def save_results(self, metrics: Dict[str, Any], split: str = "val") -> None:
        """
        Save metrics to artifacts directory.

        Args:
            metrics: Metrics dict.
            split: Split tag for filename.
        """
        self._sanity_check_metrics(metrics)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        try:
            with open(os.path.join(self.artifacts_dir, f"metrics_{split}.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as exc:
            warnings.warn(f"Failed to save metrics: {exc}")

    _warned_module_io = False

    def save_module_io(self, module_name: str, path: str) -> None:
        """
        Feature-gated placeholder to save per-module IO.
        """
        if not self.enable_module_io:
            return
        if not SupplySimulation._warned_module_io:
            warnings.warn("Module IO capture is not implemented in this build.")
            SupplySimulation._warned_module_io = True
        return

    def save_all_io(self, root_dir: str) -> None:
        """
        Feature-gated placeholder to save IO for all modules.
        """
        if not self.enable_module_io:
            return
        if not SupplySimulation._warned_module_io:
            warnings.warn("Module IO capture is not implemented in this build.")
            SupplySimulation._warned_module_io = True
        return

    def visualize(self) -> None:
        """
        Optional visualization using matplotlib, if available.

        Plots mean simulated vs observed inventory/backlog over trajectories.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("Matplotlib not available; skipping visualize().")
            return
        # Compute time-aligned means on validation set robustly
        results = self.simulator.rollout(self.bundle["val_trajectories"])
        if len(results.inventory_sim) == 0 or len(results.inventory_obs) == 0:
            warnings.warn("No validation trajectories available for visualization.")
            return
        try:
            # Align by time index; compute per-time means ignoring missing
            maxT_sim = max((arr.shape[0] for arr in results.inventory_sim.values()), default=0)
            maxT_obs = max((arr.shape[0] for arr in results.inventory_obs.values()), default=0)
            T = max(maxT_sim, maxT_obs)
            inv_sim_means = []
            inv_obs_means = []
            bk_sim_means = []
            bk_obs_means = []
            for t in range(T):
                inv_s_vals = [arr[t] for arr in results.inventory_sim.values() if t < arr.shape[0]]
                inv_o_vals = [arr[t] for arr in results.inventory_obs.values() if t < arr.shape[0]]
                bk_s_vals = [arr[t] for arr in results.backlog_sim.values() if t < arr.shape[0]]
                bk_o_vals = [arr[t] for arr in results.backlog_obs.values() if t < arr.shape[0]]
                inv_sim_means.append(np.mean(inv_s_vals) if len(inv_s_vals) > 0 else np.nan)
                inv_obs_means.append(np.mean(inv_o_vals) if len(inv_o_vals) > 0 else np.nan)
                bk_sim_means.append(np.mean(bk_s_vals) if len(bk_s_vals) > 0 else np.nan)
                bk_obs_means.append(np.mean(bk_o_vals) if len(bk_o_vals) > 0 else np.nan)
            plt.figure(figsize=(10, 5))
            plt.plot(inv_sim_means, label="Inventory Sim")
            plt.plot(inv_obs_means, label="Inventory Obs", linestyle="--")
            plt.plot(bk_sim_means, label="Backlog Sim")
            plt.plot(bk_obs_means, label="Backlog Obs", linestyle="--")
            plt.legend()
            plt.title("Validation Means")
            plt.xlabel("t")
            plt.ylabel("Units")
            plt.tight_layout()
            fig_path = os.path.join(self.artifacts_dir, "validation_means.png")
            try:
                plt.savefig(fig_path)
                print(f"Saved figure: {fig_path}")
            except Exception:
                pass
            plt.close()
        except Exception:
            warnings.warn("Failed to compute means for visualization; skipping plot.")
            return


# ------------- Holdout split -------------

def holdout_split(trajectories: List[TrajectoryData], train_end_inclusive: int = 48) -> Tuple[List[TrajectoryData], Dict[str, Tuple[int, int]]]:
    """
    Build in-trajectory train window mapping.

    Args:
        trajectories: TrajectoryData list.
        train_end_inclusive: Last index in training window.

    Returns:
        Tuple (trajectories, ranges_map) where ranges_map[tid]=(train_end, t_max).
    """
    ranges = {}
    for tr in trajectories:
        if tr.t.size == 0:
            continue
        tmax = int(np.max(tr.t))
        te = int(min(max(0, train_end_inclusive), tmax))
        ranges[tr.trajectory_id] = (te, tmax)
    return trajectories, ranges


# ------------- CLI and parameter handling -------------

def parse_cli() -> argparse.Namespace:
    """
    Parse command line arguments for the simulator and calibration pipeline.
    """
    parser = argparse.ArgumentParser(description="Beer Game Simulator with Pluggable Calibrators")
    # Data and paths
    parser.add_argument("--data-dir", type=str, default=None, help="Base data directory (overrides env).")
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--val-file", type=str, default="val_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--test-ood-file", type=str, default=None)
    parser.add_argument("--metadata-file", type=str, default="metadata.json")
    parser.add_argument("--artifacts-dir", type=str, default="results", help="Directory to save outputs.")

    # Calibration
    parser.add_argument("--calibrator", type=str, default="random_search", choices=list(CALIBRATOR_REGISTRY.keys()))
    parser.add_argument("--calibrator-config", type=str, default=None, help="Optional JSON config path for calibrator.")
    parser.add_argument("--budget", type=int, default=100, help="Calibration budget (iterations or sims).")
    parser.add_argument("--calib-window", type=str, default="0:48", help="Train window start:end inclusive within trajectory.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed.")

    # Parameters file and overrides
    parser.add_argument("--param-file", type=str, default="parameters.json", help="Initial parameters JSON file.")
    parser.add_argument("--param-defs", type=str, default="parameter_definitions.json", help="Parameter definitions JSON with 'frozen' info.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override parameters: key=value (repeatable).")

    # Demand model family selection default
    parser.add_argument("--demand-family", type=str, default="Poisson", choices=["Poisson", "NegBin", "AR1", "auto"], help="Default demand family to use if missing in parameters.json")

    # OT and metric computation config
    parser.add_argument("--ot-method", type=str, default="sinkhorn", choices=["sinkhorn", "emd", "fallback"], help="OT method for Wasserstein.")
    parser.add_argument("--ot-epsilon", type=float, default=0.01, help="Sinkhorn regularization epsilon.")
    parser.add_argument("--ot-max-iter", type=int, default=2000, help="Sinkhorn max iterations.")
    parser.add_argument("--ot-samples", type=int, default=200, help="Samples per time step for OT/MMD.")

    # Random search persistence options
    parser.add_argument("--persist-trials", action="store_true", help="Persist per-trial artifacts during random search.")
    parser.add_argument("--persist-every", type=int, default=10, help="Persist every N trials if enabled.")
    args = parser.parse_args()
    return args


def load_parameters(param_file: str, default_family: str = "Poisson") -> Dict[str, Any]:
    """
    Load initial parameters from JSON file.
    """
    if not os.path.isfile(param_file):
        warnings.warn(f"Parameter file not found: {param_file}. Using defaults.")
    return {
            "lead_time_L": 2,
            "arrival_flag": 0,
            "demand_family": default_family,
            "poisson_lambda": 5.0,
        }
    try:
        with open(param_file, "r") as f:
            p = json.load(f)
    except Exception:
        warnings.warn("Failed to load param file. Using defaults.")
        p = {}
    p.setdefault("lead_time_L", 2)
    p.setdefault("arrival_flag", 0)
    p.setdefault("demand_family", p.get("demand_family", default_family))
    if p["demand_family"] == "Poisson":
        p.setdefault("poisson_lambda", 5.0)
    elif p["demand_family"] == "NegBin":
        p.setdefault("negbin_mu", 5.0)
        p.setdefault("negbin_r", 5.0)
    elif p["demand_family"] == "AR1":
        p.setdefault("ar1_mu", 5.0)
        p.setdefault("ar1_phi", 0.0)
        p.setdefault("ar1_sigma", 1.0)
    return p


def apply_overrides(params: Dict[str, Any], overrides: List[str], defs_path: Optional[str]) -> Dict[str, Any]:
    """
    Apply CLI --set key=value overrides respecting 'frozen' flags in parameter_definitions.json.
    Validate override keys against known definitions.
    """
    adapter = SupplyParamsAdapter(param_definitions_path=defs_path, persist=False)
    known_names = set(p.get("name") for p in adapter.definitions.get("parameters", []))
    for ov in overrides or []:
        if "=" not in ov:
            warnings.warn(f"Ignoring malformed override '{ov}'. Expected format key=value.")
            continue
        key, val = ov.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key not in known_names:
            warnings.warn(f"Unknown parameter override '{key}'; ignoring.")
            continue
        if adapter._is_frozen(key):  # type: ignore
            warnings.warn(f"Ignoring override for frozen parameter '{key}'.")
            continue
        # Coerce type if known
        coerced = adapter._coerce(key, val)  # type: ignore
        coerced = adapter._clip_bounds(key, coerced)  # type: ignore
        params[key] = coerced
    return params


def save_parameters_used(artifacts_dir: str, params: Dict[str, Any]) -> None:
    """
    Save parameters_used.json to artifacts directory.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    try:
        with open(os.path.join(artifacts_dir, "parameters_used.json"), "w") as f:
            json.dump(params, f, indent=2)
    except Exception as exc:
        warnings.warn(f"Failed to save parameters_used.json: {exc}")


# ------------- Main orchestration -------------

def main() -> None:
    """
    Orchestrate loading data, calibration, simulation, evaluation, and saving artifacts.

    Steps:
        - Parse CLI and validate data directory.
        - Load datasets and metadata.
        - Build trajectories.
        - Load parameters and apply overrides.
        - Initialize simulator.
        - Select and run calibrator.fit on training window.
        - Apply fitted params and run evaluation on val/test(+OOD).
        - Save metrics, parameters_used.json, and optional visualization.
    """
    args = parse_cli()
    data_dir = validate_env_paths(args.data_dir)
    train_df, val_df, test_df, test_ood_df, metadata = load_data(
        data_dir=data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        metadata_file=args.metadata_file,
        test_ood_file=args.test_ood_file,
    )
    train_trajectories = build_trajectories(train_df, metadata)
    val_trajectories = build_trajectories(val_df, metadata)
    test_trajectories = build_trajectories(test_df, metadata) if test_df is not None else []
    test_ood_trajectories = build_trajectories(test_ood_df, metadata) if test_ood_df is not None else []

    # Window
    try:
        start_s, end_s = args.calib_window.split(":")
        calib_window = (int(start_s), int(end_s))
    except Exception:
        calib_window = (0, 48)

    # Load parameters and overrides
    base_params = load_parameters(args.param_file, default_family=args.demand_family)
    base_params = apply_overrides(base_params, args.overrides, args.param_defs)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    save_parameters_used(args.artifacts_dir, base_params)

    # Simulator initialized with base params
    simulator = BeerGameSimulator(params=base_params)
    evaluator = Evaluator(ot_method=args.ot_method, ot_epsilon=args.ot_epsilon, ot_max_iter=args.ot_max_iter, n_samples=args.ot_samples)

    # Bundle
    bundle = {
        "train_trajectories": train_trajectories,
        "val_trajectories": val_trajectories,
        "test_trajectories": test_trajectories,
        "test_ood_trajectories": test_ood_trajectories,
        "metadata": metadata,
        "artifacts_dir": args.artifacts_dir,
    }

    # Calibrator
    calibrator = get_calibrator(args.calibrator, args.calibrator_config)
    # If calibrator supports persistence control, set via config; not changed here.

    # Fit
    params_adapter = SupplyParamsAdapter(param_definitions_path=args.param_defs)
    fitted = calibrator.fit(
        bundle=bundle,
        simulator=simulator,
        evaluator=evaluator,
        train_window=calib_window,
        seed=int(args.seed),
        budget=int(args.budget),
        artifacts_dir=args.artifacts_dir,
        params_adapter=params_adapter,
    )

    # Apply fitted parameters
    sim_wrapper = SupplySimulation(simulator, bundle, artifacts_dir=args.artifacts_dir, evaluator=evaluator)
    params_adapter.apply(sim_wrapper, fitted)
    fitted_path = os.path.join(args.artifacts_dir, "fitted_params.json")
    try:
        with open(fitted_path, "w") as f:
            json.dump(fitted.to_dict(), f, indent=2)
    except Exception:
        pass

    # Evaluate on validation set
    metrics_val = sim_wrapper.evaluate(split="val")
    sim_wrapper.save_results(metrics_val, split="val")

    # Evaluate on test set (if present)
    w_in_val = None
    metrics_test = None
    if len(test_trajectories) > 0:
        metrics_test = sim_wrapper.evaluate(split="test")
        res_test = simulator.rollout(test_trajectories)
        w_in = evaluator.compute_joint_wasserstein_per_t(res_test, n_samples=evaluator.n_samples)
        w_in_val = float(w_in)
        metrics_test.setdefault("distributional", {})
        metrics_test["distributional"]["W_in"] = float(w_in_val)
        # We'll compute W_total later once W_out is available (if any)
        sim_wrapper.save_results(metrics_test, split="test")

    # Evaluate on OOD test if present, else fallback programmatic OOD
    w_out_val = None
    metrics_test_ood = None
    if len(test_ood_trajectories) > 0:
        # For OOD evaluation, we need to use lead_time=5 (OOD data was generated with lead_time=5)
        # Create a temporary simulator with lead_time=5 but keep other optimized parameters
        ood_params_dict = fitted.to_dict()
        ood_params_dict["module_params"]["supply"]["lead_time_L"] = 5  # Force lead_time=5 for OOD evaluation
        ood_simulator = BeerGameSimulator(ood_params_dict["module_params"])
        res_ood = ood_simulator.rollout(test_ood_trajectories)
        w_out = evaluator.compute_joint_wasserstein_per_t(res_ood, n_samples=evaluator.n_samples)
        w_out_val = float(w_out)
        # Also compute metrics using the OOD simulator
        metrics_test_ood = sim_wrapper.evaluate(split="test_ood")
        metrics_test_ood.setdefault("distributional", {})
        metrics_test_ood["distributional"]["W_out"] = float(w_out_val)
        sim_wrapper.save_results(metrics_test_ood, split="test_ood")
    else:
        # Programmatic OOD generation if not provided: compare current simulator vs. ground-truth with L=5
        if len(test_trajectories) > 0:
            gt_params = {
                "lead_time_L": 5,
                "arrival_flag": 0,
                "demand_family": "Poisson",
                "poisson_lambda": 5.0,
            }
            gt_sim = BeerGameSimulator(gt_params)
            res_sim_cur = simulator.rollout(test_trajectories)
            res_sim_gt = gt_sim.rollout(test_trajectories)
            w_out = evaluator.wasserstein_between_results(res_sim_cur, res_sim_gt, n_samples=evaluator.n_samples)
            w_out_val = float(w_out)

    # Compute W_total and save consolidated JSON
    if w_in_val is not None or w_out_val is not None:
        if w_in_val is not None and w_out_val is not None:
            w_total = float(np.mean([w_in_val, w_out_val]))
        else:
            w_total = float(w_in_val if w_in_val is not None else w_out_val)
        try:
            with open(os.path.join(args.artifacts_dir, "wasserstein_in_out.json"), "w") as f:
                json.dump({"W_in": float(w_in_val) if w_in_val is not None else None,
                           "W_out": float(w_out_val) if w_out_val is not None else None,
                           "W_total": float(w_total)}, f, indent=2)
        except Exception:
            pass
        # Also attach W_total into metric dicts for convenience
        if metrics_test is not None:
            metrics_test.setdefault("distributional", {})
            metrics_test["distributional"]["W_total"] = float(w_total)
            sim_wrapper.save_results(metrics_test, split="test")
        if metrics_test_ood is not None:
            metrics_test_ood.setdefault("distributional", {})
            metrics_test_ood["distributional"]["W_total"] = float(w_total)
            sim_wrapper.save_results(metrics_test_ood, split="test_ood")

    # Visualization
    sim_wrapper.visualize()

    print("Done. Artifacts saved to:", args.artifacts_dir)


# Execute main for both direct execution and sandbox wrapper invocation
main()