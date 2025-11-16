#!/usr/bin/env python3
"""
Single-stage Beer Game supply chain simulator with SBI (NPE) calibration or gradient-free fallback.

This program ingests trajectory data, builds an inventory simulator,
calibrates key parameters using Simulation-Based Inference (Neural Posterior Estimation) when available,
or falls back to gradient-free optimization (Nelder-Mead or random search) minimizing per-time-step Wasserstein,
then validates via forward simulation on a holdout set and computes error metrics.

Usage:
    python simulate.py --num_simulations 10000 --num_samples_posterior 10000 --sampling_timeout 300
"""
import argparse
import json
import math
import os
import random
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd

# Optional dependency: torch
TORCH_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional dependencies
SBI_AVAILABLE = False
NPE = None
simulate_for_sbi = None
if TORCH_AVAILABLE:
    try:
        from sbi import utils as sbi_utils
        # Try importing NPE and simulate_for_sbi directly (like env.py does)
        try:
            from sbi.inference import NPE, simulate_for_sbi
            SBI_AVAILABLE = True
        except ImportError as e:
            # If direct import fails, try alternative imports
            try:
                from sbi.inference import SNPE as NPE
                from sbi.inference import simulate_for_sbi
                SBI_AVAILABLE = True
            except ImportError:
                # Last resort: try to import from sbi.inference.base
                try:
                    from sbi.inference.base import NeuralInference as NPE
                    from sbi import utils
                    simulate_for_sbi = utils.simulate_for_sbi
                    SBI_AVAILABLE = True
                except ImportError:
                    SBI_AVAILABLE = False
                    warnings.warn(f"SBI import failed: {e}. SBI features will be disabled.")
    except Exception as e:
        SBI_AVAILABLE = False
        warnings.warn(f"SBI utils import failed: {e}. SBI features will be disabled.")

try:
    import ot  # POT - Python Optimal Transport
    POT_AVAILABLE = True
except Exception:
    POT_AVAILABLE = False

try:
    import scipy.optimize as scipy_optimize  # for Nelder-Mead fallback
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from SALib.sample import morris as salib_morris_sample
    from SALib.sample import saltelli as salib_saltelli_sample
    from SALib.analyze import morris as salib_morris_analyze
    from SALib.analyze import sobol as salib_sobol_analyze
    SALIB_AVAILABLE = True
except Exception:
    SALIB_AVAILABLE = False

# Path handling (required)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")

# Global deterministic seed
GLOBAL_SEED = 1337
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(GLOBAL_SEED)

# Throttled warning counter for parse_pipeline_items
_PARSE_PIPELINE_WARN_COUNT = 0
_PARSE_PIPELINE_WARN_LIMIT = 5





# ----------------------------
# Utility and data structures
# ----------------------------

@dataclass
class Item:
    """Represents a pipeline item with quantity and remaining lead time."""
    qty: int
    remaining: int


@dataclass
class TrajectoryData:
    """Holds all observed data for a single trajectory."""
    trajectory_id: str
    t: np.ndarray  # time indices
    actions: np.ndarray  # exogenous actions a_t
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
    """Stores simulation vs observed for a set of trajectories."""
    # Per-trajectory dicts mapping trajectory_id -> arrays
    inventory_sim: Dict[str, np.ndarray]
    backlog_sim: Dict[str, np.ndarray]
    pipeline_len_sim: Dict[str, np.ndarray]  # counts of items
    pipeline_occ_sim: Dict[str, List[np.ndarray]]  # per step occupancy vector length L (quantities)
    inventory_obs: Dict[str, np.ndarray]
    backlog_obs: Dict[str, np.ndarray]
    pipeline_len_obs: Dict[str, np.ndarray]  # counts of items
    pipeline_occ_obs: Dict[str, List[np.ndarray]]


def validate_env_paths(cli_data_dir: Optional[str] = None) -> str:
    """
    Validate environment variables and return the data directory path.
    If cli_data_dir is provided, it takes precedence.
    """
    if cli_data_dir is not None:
        data_dir = cli_data_dir
    else:
        if PROJECT_ROOT is None or DATA_PATH is None:
            raise EnvironmentError(
                "Data directory not specified. Set environment variables PROJECT_ROOT and DATA_PATH, "
                "or pass --data_dir CLI argument."
            )
        data_dir = os.path.join(PROJECT_ROOT, DATA_PATH)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {data_dir}. Ensure data files are available or pass a valid --data_dir."
        )
    return data_dir


def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Beer Game Simulator with SBI calibration and gradient-free fallback."
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Absolute path to the data directory. Overrides PROJECT_ROOT/DATA_PATH.")
    parser.add_argument("--num_simulations", type=int, default=1000,
                        help="Number of simulations for SBI training (default: 1000).")
    parser.add_argument("--num_samples_posterior", type=int, default=10000,
                        help="Number of posterior samples to draw when conditioning on observed features (default: 10000).")
    parser.add_argument("--sampling_timeout", type=int, default=60,
                        help="Timeout in seconds for posterior sampling (default: 60).")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to save results under DATA_DIR (default: 'results').")
    parser.add_argument("--demand_family", type=str, default="Poisson",
                        choices=["Poisson", "NegBin", "AR1"],
                        help="Demand family for calibration.")
    parser.add_argument("--train_file", type=str, default="train_data.csv",
                        help="Training data filename.")
    parser.add_argument("--val_file", type=str, default="val_data.csv",
                        help="Validation data filename.")
    parser.add_argument("--test_file", type=str, default="test_data.csv",
                        help="Test data filename (In-Distribution).")
    parser.add_argument("--test_ood_file", type=str, default=None,
                        help="OOD test data filename (Out-of-Distribution, e.g., test_ood_data.csv). If provided, uses real OOD data instead of parameter override.")
    parser.add_argument("--metadata_file", type=str, default="metadata.json",
                        help="Metadata filename.")
    parser.add_argument("--ood_lead_time", type=int, default=None,
                        help="If set and --test_ood_file not provided, evaluate OOD metrics on test set by overriding lead_time with this value.")
    parser.add_argument("--ood_demand_family", type=str, default="none",
                        choices=["none", "Poisson", "NegBin", "AR1"],
                        help="If set, evaluate OOD metrics on test set by overriding demand family. Use 'none' to disable.")
    parser.add_argument("--ood_demand_params", type=str, default=None,
                        help="JSON string of demand params to override for OOD evaluation (e.g., '{\"poisson_lambda\":8.0}').")
    parser.add_argument("--opt_method", type=str, default="auto",
                        choices=["auto", "sbi", "nelder-mead", "random-search"],
                        help="Calibration method. 'auto' uses SBI if available else fallback.")
    parser.add_argument("--max_evals", type=int, default=300,
                        help="Max evaluations for gradient-free optimizer (random-search or total function calls within Nelder-Mead).")
    parser.add_argument("--n_samples_wass_mmd", type=int, default=200,
                        help="Number of samples per time step for Wasserstein/MMD metrics (default: 200).")
    parser.add_argument("--mmd_sigma", type=float, default=1.0,
                        help="Sigma parameter for Gaussian kernel in MMD (default: 1.0).")
    # Double Monte Carlo options (to match mask adoption task configuration)
    parser.add_argument("--double-mc", action="store_true", 
                        help="Enable Double Monte Carlo on test window (posterior sampling x repeated sims)")
    parser.add_argument("--mc-M", type=int, default=50,
                        help="Number of parameter samples for Double Monte Carlo (default: 50)")
    parser.add_argument("--mc-K", type=int, default=20,
                        help="Number of runs per parameter sample for Double Monte Carlo (default: 20)")
    parser.add_argument("--use-alpha", action="store_true",
                        help="Enable embedded alpha noise for selected continuous parameters.")
    parser.add_argument("--alpha-params", type=str, default=None,
                        help="Comma-separated parameter names for alpha embedding (default depends on demand family).")
    parser.add_argument("--alpha-max-ratio", type=float, default=0.5,
                        help="Upper bound on alpha scaling factor r (ratio of distance to bounds).")
    parser.add_argument("--alpha-mode", type=str, default="additive",
                        choices=["additive", "multiplicative"],
                        help="Alpha embedding form: additive (λ + α·ξ) or multiplicative (λ·(1+α·ξ)).")
    parser.add_argument("--run-gsa", action="store_true",
                        help="Run global sensitivity analysis after calibration.")
    parser.add_argument("--gsa-method", type=str, default="both",
                        choices=["morris", "sobol", "both"],
                        help="GSA method to run (Method of Morris, Sobol, or both).")
    parser.add_argument("--gsa-num-samples", type=int, default=256,
                        help="Base sample size N for GSA (per SALib definition).")
    parser.add_argument("--gsa-num-levels", type=int, default=6,
                        help="Number of grid levels for Morris screening.")
    parser.add_argument("--gsa-second-order", action="store_true",
                        help="Compute second-order Sobol indices (requires more simulations).")
    parser.add_argument("--gsa-seed", type=int, default=None,
                        help="Random seed for GSA sampling (defaults to metadata/random seed).")
    parser.add_argument("--gsa-results-prefix", type=str, default="gsa",
                        help="Filename prefix when saving GSA outputs.")
    args = parser.parse_args()
    if args.num_simulations <= 0:
        raise ValueError("num_simulations must be positive.")
    if args.num_samples_posterior <= 0:
        raise ValueError("num_samples_posterior must be positive.")
    if args.sampling_timeout <= 0:
        raise ValueError("sampling_timeout must be positive.")
    if args.ood_lead_time is not None and args.ood_lead_time < 0:
        raise ValueError("ood_lead_time must be nonnegative if provided.")
    if args.n_samples_wass_mmd <= 0:
        raise ValueError("n_samples_wass_mmd must be positive.")
    if args.mmd_sigma <= 0:
        raise ValueError("mmd_sigma must be positive.")
    if args.max_evals <= 0:
        raise ValueError("max_evals must be positive.")
    if args.alpha_max_ratio < 0:
        raise ValueError("alpha_max_ratio must be nonnegative.")
    if args.gsa_num_samples <= 0:
        raise ValueError("gsa_num_samples must be positive.")
    if args.gsa_num_levels < 2:
        raise ValueError("gsa_num_levels must be at least 2 for Morris screening.")

    # Map 'none' string to None for OOD demand family
    if args.ood_demand_family == "none":
        args.ood_demand_family = None

    if args.alpha_params:
        args.alpha_params = [s.strip() for s in args.alpha_params.split(",") if s.strip()]
    else:
        args.alpha_params = None
    args.gsa_method = args.gsa_method.lower()
    args.alpha_mode = args.alpha_mode.lower()

    return args


# ----------------------------
# Data ingestion and parsing
# ----------------------------

def safe_int(x: Any) -> int:
    """
    Safely cast a value to an integer, raising errors for invalid inputs.
    """
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Cannot cast to int: {x}") from e


def parse_pipeline_items(raw: Any) -> List[Item]:
    """
    Parse pipeline_items from a CSV field into a list of Items.

    Supports:
    - JSON string of list of dicts: [{"qty": 4, "remaining_lead": 2}, ...]
    - JSON string of list of pairs: [[qty, remaining], ...]
    - JSON string of list of numbers representing quantities per remaining lead index (1..L)
    - Semicolon separated "qty@remaining" pairs, e.g., "4@2;3@1"
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
                pairs = s.split(";")
                for p in pairs:
                    if "@" not in p:
                        continue
                    qty_str, rem_str = p.split("@")
                    qty = safe_int(qty_str)
                    rem = safe_int(rem_str)
                    if qty > 0 and rem >= 0:
                        items.append(Item(qty=qty, remaining=rem))
                return items
            except Exception:
                if _PARSE_PIPELINE_WARN_COUNT < _PARSE_PIPELINE_WARN_LIMIT:
                    warnings.warn(
                        f"Failed to parse pipeline_items value; treating as empty. Sample: {s[:80]}..."
                    )
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


def infer_time_column(df: pd.DataFrame) -> str:
    """
    Infer the time column name from a DataFrame.
    """
    candidates = ["t", "time", "time_step", "step", "period_index"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer time column from columns: {list(df.columns)}. "
        "Expected one of: t, time, time_step, step, period_index."
    )


def load_data(
    data_dir: str,
    train_file: str,
    val_file: str,
    metadata_file: str,
    test_file: Optional[str] = None,
    test_ood_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load training/validation/test data and metadata.
    
    Returns:
        train_df, val_df, test_df (ID), test_df_ood (OOD), metadata
    """
    train_path = os.path.join(data_dir, train_file)
    val_path = os.path.join(data_dir, val_file)
    meta_path = os.path.join(data_dir, metadata_file)

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = None
    if test_file is not None:
        test_path = os.path.join(data_dir, test_file)
        if os.path.isfile(test_path):
            test_df = pd.read_csv(test_path)
        else:
            warnings.warn(f"Test data not found at {test_path}. Skipping test evaluation.")
    
    test_df_ood = None
    if test_ood_file is not None:
        test_ood_path = os.path.join(data_dir, test_ood_file)
        if os.path.isfile(test_ood_path):
            test_df_ood = pd.read_csv(test_ood_path)
        else:
            warnings.warn(f"OOD test data not found at {test_ood_path}. Skipping OOD evaluation.")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    return train_df, val_df, test_df, test_df_ood, metadata


def build_trajectories(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> List[TrajectoryData]:
    """
    Construct trajectory objects from a DataFrame.
    """
    if df is None or len(df) == 0:
        return []
    required = ["trajectory_id", "action", "inventory", "backlog"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Input data missing required column '{col}'. Found columns: {list(df.columns)}"
            )
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
        pipeline_len_col_present = "pipeline_len" in g_sorted.columns
        pipeline_len_counts = []

        # contiguity and start check
        t0 = int(t.min())
        t1 = int(t.max())
        expected = np.arange(t0, t1 + 1)
        if not np.array_equal(t, expected) or t0 != 0:
            contig_warn += 1
            if len(contig_examples) < 5:
                contig_examples.append((str(tid), int(t0), int(t1), len(t)))
        # Parse pipeline items per step
        pipeline_col = "pipeline_items" if "pipeline_items" in g_sorted.columns else None
        pipeline_items_seq: List[List[Item]] = []
        max_rem = 0
        if pipeline_col is not None:
            raw_list = g_sorted[pipeline_col].tolist()
            for raw in raw_list:
                items = parse_pipeline_items(raw)
                for it in items:
                    max_rem = max(max_rem, int(it.remaining))
                pipeline_items_seq.append(items)
                pipeline_len_counts.append(len(items))
        else:
            # If not present, infer empty pipeline
            pipeline_items_seq = [[] for _ in range(len(g_sorted))]
            pipeline_len_counts = [0 for _ in range(len(g_sorted))]

        # Optional validation: pipeline_len column should match count of items
        if pipeline_len_col_present:
            pl_col_vals = g_sorted["pipeline_len"].to_numpy().astype(int)
            for i, cnt in enumerate(pipeline_len_counts):
                if pl_col_vals[i] != cnt:
                    mismatch_count += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append((str(tid), int(t[i]), pl_col_vals[i], cnt))

        # Initial state from first row
        init_inventory = int(round(inv[0]))
        init_backlog = int(round(bklg[0]))
        init_pipeline = pipeline_items_seq[0] if len(pipeline_items_seq) > 0 else []

        traj = TrajectoryData(
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
        trajectories.append(traj)

    if mismatch_count > 0:
        warnings.warn(
            f"pipeline_len column mismatches parsed pipeline_items count in {mismatch_count} rows. "
            f"First examples (trajectory_id, t, pipeline_len, parsed_count): {mismatch_examples}"
        )
    if contig_warn > 0:
        warnings.warn(
            f"Found {contig_warn} trajectories with non-contiguous time indices or not starting at 0. "
            f"First examples (trajectory_id, t_min, t_max, n_rows): {contig_examples}"
        )

    return trajectories


# ----------------------------
# Demand models
# ----------------------------

class DemandModel:
    """Abstract base class for demand models."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def reset(self) -> None:
        """Reset internal state prior to a new trajectory."""
        return

    def sample(self, t: int) -> int:
        """
        Sample demand at period t.
        """
        raise NotImplementedError


class PoissonDemandModel(DemandModel):
    """
    Poisson demand model with optional sinusoidal seasonality on the mean.
    """

    def __init__(self, base_lambda: float,
                 seasonal_amplitude: float = 0.0,
                 seasonal_period: int = 7,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(rng=rng)
        if base_lambda < 0.0:
            raise ValueError("base_lambda must be >= 0.")
        if seasonal_amplitude < 0.0:
            raise ValueError("seasonal_amplitude must be >= 0.")
        if seasonal_period < 2:
            raise ValueError("seasonal_period must be >= 2.")
        self.base_lambda = float(base_lambda)
        self.seasonal_amplitude = float(seasonal_amplitude)
        self.seasonal_period = int(seasonal_period)

    def reset(self) -> None:
        return

    def _mean_t(self, t: int) -> float:
        if self.seasonal_amplitude <= 1e-8:
            return max(0.0, self.base_lambda)
        val = self.base_lambda + self.seasonal_amplitude * math.sin(
            2.0 * math.pi * (t % self.seasonal_period) / float(self.seasonal_period)
        )
        return max(0.0, val)

    def sample(self, t: int) -> int:
        lam = self._mean_t(t)
        d = self.rng.poisson(lam=lam)
        return int(d)


class NegativeBinomialDemandModel(DemandModel):
    """
    Negative Binomial demand model using Gamma-Poisson mixture to support real-valued r:

    Given mu >= 0 and r > 0:
    lambda ~ Gamma(shape=r, scale=mu/r), demand ~ Poisson(lambda)
    """

    def __init__(self, mu: float, r: float, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(rng=rng)
        if mu < 0.0:
            raise ValueError("mu must be >= 0.")
        if r <= 0.0:
            raise ValueError("r must be > 0.")
        self.mu = float(mu)
        self.r = float(r)

    def reset(self) -> None:
        return

    def sample(self, t: int) -> int:
        scale = self.mu / max(self.r, 1e-12)
        lam = self.rng.gamma(shape=self.r, scale=scale)
        d = self.rng.poisson(lam=lam)
        return int(d)


class AR1DemandModel(DemandModel):
    """
    AR(1) Gaussian demand on latent variable x_t with rounding and truncation to nonnegative integers:

    x_t = mu + phi * (x_{t-1} - mu) + sigma * eps_t, eps_t ~ N(0,1)
    demand_t = max(0, round(x_t))
    """

    def __init__(self, mu: float, phi: float, sigma: float, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__(rng=rng)
        if mu < 0.0:
            raise ValueError("mu must be >= 0.")
        if not (-0.99 <= phi <= 0.99):
            raise ValueError("phi out of supported bounds [-0.99, 0.99].")
        if sigma < 0.0:
            raise ValueError("sigma must be >= 0.")
        self.mu = float(mu)
        self.phi = float(phi)
        self.sigma = float(sigma)
        self._x_prev: float = self.mu

    def reset(self) -> None:
        self._x_prev = self.mu

    def sample(self, t: int) -> int:
        eps = self.rng.normal(loc=0.0, scale=1.0)
        x_t = self.mu + self.phi * (self._x_prev - self.mu) + self.sigma * eps
        self._x_prev = float(x_t)
        d = int(max(0, round(x_t)))
        return d


def create_demand_model_from_params(params: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> DemandModel:
    """
    Factory to create demand model instance from params dict.

    Supports:
        - Poisson: keys 'poisson_lambda', optional 'seasonal_amplitude', 'seasonal_period'
        - NegBin: keys 'negbin_mu' and 'negbin_r'
        - AR1: keys 'ar1_mu', 'ar1_phi', 'ar1_sigma'
    """
    fam = params.get("demand_family", "Poisson")
    if fam == "Poisson":
        return PoissonDemandModel(
            base_lambda=float(params.get("poisson_lambda", 5.0)),
            seasonal_amplitude=float(params.get("seasonal_amplitude", 0.0)),
            seasonal_period=int(params.get("seasonal_period", 7)),
            rng=rng,
        )
    elif fam == "NegBin":
        return NegativeBinomialDemandModel(
            mu=float(params.get("negbin_mu", 5.0)),
            r=float(params.get("negbin_r", 5.0)),
            rng=rng,
        )
    elif fam == "AR1":
        return AR1DemandModel(
            mu=float(params.get("ar1_mu", 5.0)),
            phi=float(params.get("ar1_phi", 0.0)),
            sigma=float(params.get("ar1_sigma", 1.0)),
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported demand_family: {fam}")


# ----------------------------
# Inventory node and simulation
# ----------------------------

class InventoryNode:
    """
    Single retailer node in a one-stage Beer Game-like system.

    State variables:
        - inventory (int), backlog (int), pipeline (list of Items), time step t.
    """

    def __init__(
        self,
        init_inventory: int,
        init_backlog: int,
        init_pipeline: List[Item],
        lead_time: int,
        arrival_convention: str = "deliver_at_remaining_0",
    ) -> None:
        """
        Create a new inventory node.
        """
        if init_inventory < 0 or init_backlog < 0:
            raise ValueError("Initial inventory and backlog must be nonnegative.")
        if lead_time < 0:
            raise ValueError("lead_time must be >= 0.")
        if arrival_convention not in {"deliver_at_remaining_0", "deliver_at_remaining_1"}:
            raise ValueError("Invalid arrival_convention.")
        self.inventory = int(init_inventory)
        self.backlog = int(init_backlog)
        self.pipeline: List[Item] = [
            Item(qty=int(max(0, p.qty)), remaining=int(max(0, p.remaining)))
            for p in init_pipeline if int(p.qty) > 0
        ]
        self.lead_time = int(lead_time)
        self.arrival_convention = arrival_convention
        self.t = 0

    def _deliveries_and_decrement(self) -> int:
        """
        Process pipeline deliveries and decrement remaining leads given convention.

        Returns:
            int: Total quantity delivered this step.
        """
        delivered = 0
        if self.arrival_convention == "deliver_at_remaining_1":
            remaining_items: List[Item] = []
            for it in self.pipeline:
                if it.remaining <= 1:
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
        Append an order to the pipeline at position lead_time L.
        """
        q = int(max(0, qty))
        if self.lead_time == 0:
            self.inventory += q
            return
        if q > 0:
            self.pipeline.append(Item(qty=q, remaining=int(self.lead_time)))

    def step(self, action: int, demand: int) -> Dict[str, Any]:
        """
        Advance one period: process deliveries, demand, backlog, and pipeline.
        """
        delivered = self._deliveries_and_decrement()
        self.inventory += int(delivered)

        # 2) Fulfill backlog first
        served_backlog = min(self.inventory, self.backlog)
        self.inventory -= served_backlog
        self.backlog -= served_backlog

        # 3) Fulfill current demand
        served_demand = min(self.inventory, max(0, demand))
        self.inventory -= served_demand
        unmet = max(0, demand) - served_demand
        self.backlog += int(unmet)

        # 4) Append order to pipeline
        self._append_order(action)

        # 5) Prepare outputs
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


# ----------------------------
# Distance utilities (efficient 1D Wasserstein and MMD)
# ----------------------------

def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute exact 1D Wasserstein (Earth Mover's) distance between empirical samples.

    Uses sort-based O(n log n) algorithm. For unequal sizes, integrates absolute CDF difference.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    nx = x.size
    ny = y.size
    if nx == 0 or ny == 0:
        return 0.0
    xs = np.sort(x)
    ys = np.sort(y)
    if nx == ny:
        return float(np.mean(np.abs(xs - ys)))
    vals = np.sort(np.concatenate([xs, ys]))
    cdf_x = np.searchsorted(xs, vals, side="right") / float(nx)
    cdf_y = np.searchsorted(ys, vals, side="right") / float(ny)
    dx = np.diff(vals)
    diffs = np.abs(cdf_x[:-1] - cdf_y[:-1])
    w = float(np.sum(diffs * dx))
    return w


def mmd_gaussian_1d(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute MMD with Gaussian kernel between two 1D samples (squared MMD).
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if x.size == 0 or y.size == 0:
        return 0.0
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
    nx, ny = x.shape[0], y.shape[0]
    if max(nx, ny) <= 200:
        def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            d2 = (a - b.T) ** 2
            return np.exp(-gamma * d2)
        k_xx = rbf(x, x)
        k_yy = rbf(y, y)
        k_xy = rbf(x, y)
        mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        return float(max(0.0, mmd2))
    m = min(nx, ny)
    idx_x1 = np.random.choice(nx, size=m, replace=(nx < m))
    idx_x2 = np.random.choice(nx, size=m, replace=(nx < m))
    idx_y1 = np.random.choice(ny, size=m, replace=(ny < m))
    idx_y2 = np.random.choice(ny, size=m, replace=(ny < m))
    x1, x2 = x[idx_x1], x[idx_x2]
    y1, y2 = y[idx_y1], y[idx_y2]
    k_xx = np.exp(-gamma * (x1 - x2) ** 2).mean()
    k_yy = np.exp(-gamma * (y1 - y2) ** 2).mean()
    k_xy = np.exp(-gamma * (x1 - y1) ** 2).mean()
    mmd2 = float(k_xx + k_yy - 2.0 * k_xy)
    return max(0.0, mmd2)


def approx_wasserstein_1d(x: np.ndarray, y: np.ndarray, n_quantiles: int = 100) -> float:
    """
    Approximate 1D Wasserstein distance by comparing quantile functions.
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0
    qs = np.linspace(0.0, 1.0, num=n_quantiles, endpoint=True)
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    return float(np.mean(np.abs(qx - qy)))


def compute_mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) with RBF kernel between two 1D samples.
    Quadratic estimator (used for proxies only).
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0

    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        dists = (a - b.T) ** 2
        return np.exp(-gamma * dists)

    xx = rbf(x, x)
    yy = rbf(y, y)
    xy = rbf(x, y)
    mmd2 = xx.mean() + yy.mean() - 2.0 * xy.mean()
    return float(max(0.0, mmd2))


# ----------------------------
# Summary features and metrics utils
# ----------------------------

def compute_summary_features(
    inventory_series: List[np.ndarray],
    backlog_series: List[np.ndarray],
    pipeline_len_series: List[np.ndarray],
) -> np.ndarray:
    """
    Compute summary statistics features from sequences across trajectories.
    """
    inv_all = np.concatenate(inventory_series) if len(inventory_series) > 0 else np.array([0.0])
    bklg_all = np.concatenate(backlog_series) if len(backlog_series) > 0 else np.array([0.0])
    plen_all = np.concatenate(pipeline_len_series) if len(pipeline_len_series) > 0 else np.array([0.0])

    def volatility(arrs: List[np.ndarray]) -> float:
        diffs = []
        for a in arrs:
            if len(a) >= 2:
                diffs.append(np.abs(np.diff(a)))
        if len(diffs) == 0:
            return 0.0
        return float(np.mean(np.concatenate(diffs)))

    def acf1(arrs: List[np.ndarray]) -> float:
        vals = []
        for a in arrs:
            if len(a) >= 2:
                x = a.astype(float)
                sd = x.std()
                if sd < 1e-8:
                    vals.append(0.0)
                    continue
                x = (x - x.mean()) / (sd + 1e-8)
                c = np.corrcoef(x[:-1], x[1:])[0, 1]
                if not np.isfinite(c):
                    c = 0.0
                vals.append(float(c))
        if len(vals) == 0:
            return 0.0
        vals = np.array(vals, dtype=float)
        vals[~np.isfinite(vals)] = 0.0
        return float(np.mean(vals))

    inv_mu = float(np.mean(inv_all))
    inv_sd = float(np.std(inv_all))
    b_mu = float(np.mean(bklg_all))
    b_sd = float(np.std(bklg_all))
    p_mu = float(np.mean(plen_all))
    p_sd = float(np.std(plen_all))
    inv_vol = volatility(inventory_series)
    b_vol = volatility(backlog_series)
    last_inv = float(np.mean([a[-1] for a in inventory_series if len(a) > 0])) if len(inventory_series) > 0 else 0.0
    last_b = float(np.mean([a[-1] for a in backlog_series if len(a) > 0])) if len(backlog_series) > 0 else 0.0
    inv_acf1 = acf1(inventory_series)
    b_acf1 = acf1(backlog_series)

    feat = np.array(
        [
            inv_mu, inv_sd,
            b_mu, b_sd,
            p_mu, p_sd,
            inv_vol, b_vol,
            last_inv, last_b,
            inv_acf1, b_acf1,
        ],
        dtype=float,
    )
    feat[~np.isfinite(feat)] = 0.0
    return feat


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute root mean squared error.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean absolute error.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


# ----------------------------
# Holdout split
# ----------------------------

def holdout_split(
    trajectories: List[TrajectoryData],
    train_end_inclusive: int = 48,
) -> Tuple[List[TrajectoryData], Dict[str, Tuple[int, int]]]:
    """
    Create temporal holdout range within each trajectory.
    """
    ranges = {}
    for tr in trajectories:
        t_all = tr.t
        if len(t_all) == 0:
            raise ValueError(f"Empty trajectory: {tr.trajectory_id}")
        t_max = int(np.max(t_all))
        if train_end_inclusive > t_max:
            te = int(min(train_end_inclusive, t_max))
        else:
            te = int(train_end_inclusive)
        ranges[tr.trajectory_id] = (te, t_max)
    return trajectories, ranges


# ----------------------------
# SBI Calibrator with NPE (optional)
# ----------------------------

def simulate_for_sbi_safe(simulation_wrapper: Callable[[Any], Any],
                          prior: Any, num_simulations: int, chunk_size: int = 512) -> Tuple[Any, Any]:
    """
    Version-agnostic simulator runner that samples from prior and calls simulation_wrapper.
    Returns torch.Tensors when SBI (and torch) are available.
    """
    # 严格限制参数采样次数为 num_simulations
    theta_all = prior.sample((num_simulations,))
    if theta_all.shape[0] != num_simulations:
        raise ValueError(f"Expected {num_simulations} parameter samples, got {theta_all.shape[0]}")
    
    xs = []
    # Chunked loop for reduced memory overhead
    actual_count = 0
    for i in range(0, num_simulations, chunk_size):
        end = min(num_simulations, i + chunk_size)
        for j in range(i, end):
            if actual_count >= num_simulations:
                break  # 严格限制不超过 num_simulations
            x_i = simulation_wrapper(theta_all[j])
            xs.append(x_i)
            actual_count += 1
    
    if len(xs) != num_simulations:
        raise ValueError(f"Expected {num_simulations} simulations, but got {len(xs)}")
    
    x = torch.stack(xs, dim=0)
    return theta_all, x


class SBICalibrator:
    """
    Calibration using SBI (NPE) with gradient-free fallback.
    """

    def __init__(
        self,
        train_trajectories: List[TrajectoryData],
        holdout_ranges: Dict[str, Tuple[int, int]],
        demand_family: str = "Poisson",
        num_simulations: int = 1000,
        num_samples_posterior: int = 10000,
        sampling_timeout: int = 60,
        seed: int = GLOBAL_SEED,
        opt_method: str = "auto",
        max_evals: int = 300,
        n_samples_wass_mmd: int = 200,
        mmd_sigma: float = 1.0,
        loss_weights: Tuple[float, float, float] = (1.0, 0.1, 0.01),
        use_alpha: bool = False,
        alpha_params: Optional[List[str]] = None,
        alpha_max_ratio: float = 0.5,
        alpha_mode: str = "additive",
    ) -> None:
        if demand_family not in {"Poisson", "NegBin", "AR1"}:
            raise ValueError("Unsupported demand family.")
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")
        if num_samples_posterior <= 0:
            raise ValueError("num_samples_posterior must be positive.")
        if sampling_timeout <= 0:
            raise ValueError("sampling_timeout must be positive.")
        if max_evals <= 0:
            raise ValueError("max_evals must be positive.")

        self.train_trajectories = train_trajectories
        self.holdout_ranges = holdout_ranges
        self.demand_family = demand_family
        self.num_simulations = int(num_simulations)
        self.num_samples_posterior = int(num_samples_posterior)
        self.sampling_timeout = int(sampling_timeout)
        self.seed = int(seed)
        self.opt_method = opt_method
        self.max_evals = int(max_evals)
        self.n_samples_wass_mmd = int(n_samples_wass_mmd)
        self.mmd_sigma = float(mmd_sigma)
        self.loss_weights = tuple(float(w) for w in loss_weights)
        if alpha_max_ratio < 0:
            raise ValueError("alpha_max_ratio must be nonnegative.")
        if alpha_mode not in {"additive", "multiplicative"}:
            raise ValueError("alpha_mode must be 'additive' or 'multiplicative'.")

        self.use_alpha = bool(use_alpha)
        self.alpha_mode = alpha_mode
        self.alpha_max_ratio = float(alpha_max_ratio)
        self.alpha_params_requested = list(alpha_params) if alpha_params else []

        # Base parameter space (without alpha augmentation)
        (
            self.base_param_names,
            self.base_low,
            self.base_high,
        ) = self._get_base_param_space(demand_family)
        self.param_bounds = {
            name: (float(self.base_low[idx]), float(self.base_high[idx]))
            for idx, name in enumerate(self.base_param_names)
        }
        self.alpha_params = self._resolve_alpha_params(self.alpha_params_requested)
        self.param_names, self.low, self.high = self._build_param_space_with_alpha()
        self.num_base_params = len(self.base_param_names)
        self.r_param_names = [name for name in self.param_names if name.startswith("r::")]
        self.alpha_default_r = 0.0 if not self.r_param_names else self.alpha_max_ratio / 2.0
        self.alpha_rng = np.random.default_rng(self.seed + 12345)

        # Observed features as numpy for deferred torch conversion
        self.x_observed_np = self._compute_observed_features()

        # Placeholders to store results
        self.optimized_params: Optional[np.ndarray] = None
        self.posterior_samples: Optional[np.ndarray] = None

    def _get_base_param_space(self, family: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        if family == "Poisson":
            names = ["lead_time_L", "arrival_flag", "poisson_lambda"]
            low = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([8.0, 1.0, 20.0], dtype=np.float32)
            return names, low, high
        if family == "NegBin":
            names = ["lead_time_L", "arrival_flag", "negbin_mu", "negbin_r"]
            low = np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32)
            high = np.array([8.0, 1.0, 20.0, 50.0], dtype=np.float32)
            return names, low, high
        if family == "AR1":
            names = ["lead_time_L", "arrival_flag", "ar1_mu", "ar1_phi", "ar1_sigma"]
            low = np.array([1.0, 0.0, 0.0, -0.95, 0.0], dtype=np.float32)
            high = np.array([8.0, 1.0, 20.0, 0.95, 10.0], dtype=np.float32)
            return names, low, high
        raise ValueError(f"Unsupported family: {family}")

    def _default_alpha_param_candidates(self) -> List[str]:
        if self.demand_family == "Poisson":
            return ["poisson_lambda"]
        if self.demand_family == "NegBin":
            return ["negbin_mu", "negbin_r"]
        if self.demand_family == "AR1":
            return ["ar1_mu", "ar1_phi", "ar1_sigma"]
        return []

    def _resolve_alpha_params(self, requested: List[str]) -> List[str]:
        if not self.use_alpha:
            return []
        candidates = requested or self._default_alpha_param_candidates()
        valid: List[str] = []
        for name in candidates:
            if name not in self.base_param_names:
                continue
            if name in {"lead_time_L", "arrival_flag"}:
                continue
            if name not in valid:
                valid.append(name)
        if not valid:
            self.use_alpha = False
            return []
        return valid

    def _build_param_space_with_alpha(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        names = list(self.base_param_names)
        low = self.base_low.astype(np.float32).copy()
        high = self.base_high.astype(np.float32).copy()
        if self.use_alpha and self.alpha_params:
            for pname in self.alpha_params:
                names.append(f"r::{pname}")
                low = np.concatenate([low, np.array([0.0], dtype=np.float32)])
                high = np.concatenate([high, np.array([self.alpha_max_ratio], dtype=np.float32)])
        else:
            self.use_alpha = False
        return names, low, high

    def _pad_theta_with_alpha(self, theta_base: np.ndarray, r_override: Optional[np.ndarray] = None) -> np.ndarray:
        theta_base = np.asarray(theta_base, dtype=float)
        if not self.r_param_names:
            return theta_base
        if r_override is None:
            r_override = np.full(len(self.r_param_names), self.alpha_default_r, dtype=float)
        r_override = np.asarray(r_override, dtype=float)
        return np.concatenate([theta_base, r_override])

    def evaluate_loss(self, theta_vec: np.ndarray) -> float:
        theta_vec = np.asarray(theta_vec, dtype=float)
        theta_clipped = np.minimum(self.high, np.maximum(self.low, theta_vec))
        params = self._theta_to_params(theta_clipped)
        loss, _ = self._compute_loss_distributional(params)
        return float(loss)

    def evaluate_loss_from_base(self, theta_base: np.ndarray, apply_alpha_noise: bool = False) -> float:
        theta_base = np.asarray(theta_base, dtype=float)
        theta_base = np.minimum(self.base_high, np.maximum(self.base_low, theta_base))
        if self.r_param_names:
            if apply_alpha_noise:
                theta_full = self._pad_theta_with_alpha(theta_base)
            else:
                theta_full = self._pad_theta_with_alpha(theta_base, np.zeros(len(self.r_param_names), dtype=float))
        else:
            theta_full = theta_base
        return self.evaluate_loss(theta_full)

    def _compute_observed_features(self) -> np.ndarray:
        """
        Compute observed summary features from training window across trajectories.
        """
        inv_series = []
        b_series = []
        pl_series = []
        for tr in self.train_trajectories:
            te, _ = self.holdout_ranges[tr.trajectory_id]
            mask = tr.t <= te
            inv_series.append(tr.inventory_obs[mask])
            b_series.append(tr.backlog_obs[mask])
            pl_series.append(tr.pipeline_len_obs_counts[mask].astype(float))
        feats = compute_summary_features(inv_series, b_series, pl_series)
        return feats

    def _theta_to_params(self, theta_vec: np.ndarray) -> Dict[str, Any]:
        """
        Convert raw theta vector (possibly augmented with r parameters) to simulator parameter dictionary.
        """
        theta_vec = np.asarray(theta_vec, dtype=float)
        if theta_vec.shape[0] != len(self.param_names):
            raise ValueError(f"Expected theta of length {len(self.param_names)}, got {theta_vec.shape[0]}")

        base_theta = theta_vec[: self.num_base_params]
        base_vals = {
            name: float(base_theta[idx]) for idx, name in enumerate(self.base_param_names)
        }
        L = int(np.clip(np.round(base_vals.get("lead_time_L", 2.0)), 1, 8))
        base_vals["lead_time_L"] = float(L)

        arrival_flag = float(np.clip(np.round(base_vals.get("arrival_flag", 1.0)), 0.0, 1.0))
        base_vals["arrival_flag"] = arrival_flag
        arrival_convention = "deliver_at_remaining_1" if arrival_flag >= 0.5 else "deliver_at_remaining_0"

        params: Dict[str, Any] = {
            "lead_time_L": L,
            "arrival_convention": arrival_convention,
            "demand_family": self.demand_family,
        }

        if self.demand_family == "Poisson":
            lam = float(np.clip(base_vals.get("poisson_lambda", 5.0), *self.param_bounds["poisson_lambda"]))
            base_vals["poisson_lambda"] = lam
            params.update({
                "poisson_lambda": lam,
                "seasonal_amplitude": 0.0,
                "seasonal_period": 7,
            })
        elif self.demand_family == "NegBin":
            mu = float(np.clip(base_vals.get("negbin_mu", 5.0), *self.param_bounds["negbin_mu"]))
            r = float(np.clip(base_vals.get("negbin_r", 5.0), *self.param_bounds["negbin_r"]))
            base_vals["negbin_mu"] = mu
            base_vals["negbin_r"] = r
            params.update({
                "negbin_mu": mu,
                "negbin_r": r,
            })
        elif self.demand_family == "AR1":
            mu = float(np.clip(base_vals.get("ar1_mu", 5.0), *self.param_bounds["ar1_mu"]))
            phi = float(np.clip(base_vals.get("ar1_phi", 0.0), *self.param_bounds["ar1_phi"]))
            sigma = float(np.clip(base_vals.get("ar1_sigma", 1.0), *self.param_bounds["ar1_sigma"]))
            base_vals["ar1_mu"] = mu
            base_vals["ar1_phi"] = phi
            base_vals["ar1_sigma"] = sigma
            params.update({
                "ar1_mu": mu,
                "ar1_phi": phi,
                "ar1_sigma": sigma,
            })
        else:
            raise ValueError(f"Unsupported family: {self.demand_family}")

        r_values: Dict[str, float] = {}
        if self.r_param_names:
            r_theta = theta_vec[self.num_base_params :]
            for name, raw_val in zip(self.r_param_names, r_theta):
                key = name.split("r::", 1)[1]
                r_values[key] = float(np.clip(raw_val, 0.0, self.alpha_max_ratio))

        self._apply_alpha_embedding(params, base_vals, r_values)
        return params

    def _apply_alpha_embedding(
        self,
        params: Dict[str, Any],
        base_vals: Dict[str, float],
        r_values: Dict[str, float],
    ) -> None:
        """
        Apply embedded model error α for selected parameters in-place.
        """
        if not (self.use_alpha and self.alpha_params):
            return
        for pname in self.alpha_params:
            if pname not in self.param_bounds:
                continue
            if pname not in base_vals:
                continue
            lower, upper = self.param_bounds[pname]
            value = float(base_vals[pname])
            if self.alpha_mode == "additive":
                margin = min(value - lower, upper - value)
                if margin <= 0:
                    continue
                scale = float(r_values.get(pname, self.alpha_default_r))
                scale = max(0.0, min(self.alpha_max_ratio, scale))
                alpha_abs = scale * margin
                if alpha_abs <= 0:
                    continue
                xi = float(np.clip(self.alpha_rng.normal(0.0, 1.0), -1.0, 1.0))
                value_eff = value + alpha_abs * xi
            else:  # multiplicative
                scale = float(r_values.get(pname, self.alpha_default_r))
                scale = max(0.0, min(self.alpha_max_ratio, scale))
                if scale == 0.0:
                    continue
                xi = float(np.clip(self.alpha_rng.normal(0.0, 1.0), -1.0, 1.0))
                value_eff = value * (1.0 + scale * xi)
            value_eff = float(np.clip(value_eff, lower, upper))
            base_vals[pname] = value_eff
            if pname in params:
                params[pname] = value_eff

    def _simulate_training_window(self, params: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Simulate all training trajectories over the training window using given parameters.
        """
        seed_val = self.seed + int(params["lead_time_L"]) * 17 + (1 if params["arrival_convention"].endswith("_1") else 0) * 29
        rng = np.random.default_rng(seed_val)

        inv_sim: Dict[str, np.ndarray] = {}
        b_sim: Dict[str, np.ndarray] = {}
        pl_sim: Dict[str, np.ndarray] = {}

        for tr in self.train_trajectories:
            te, _ = self.holdout_ranges[tr.trajectory_id]
            mask = tr.t <= te
            horizon = int(np.sum(mask))
            if horizon <= 0:
                continue
            node = InventoryNode(
                init_inventory=int(tr.init_inventory),
                init_backlog=int(tr.init_backlog),
                init_pipeline=[Item(int(it.qty), int(it.remaining)) for it in tr.init_pipeline],
                lead_time=int(params["lead_time_L"]),
                arrival_convention=str(params["arrival_convention"]),
            )
            demand_model = create_demand_model_from_params(params, rng=rng)
            demand_model.reset()

            actions = tr.actions[:horizon]
            inv_rec, b_rec, pl_rec = [], [], []
            for tt in range(horizon):
                d = demand_model.sample(t=tt)
                snapshot = node.step(action=int(actions[tt]), demand=int(d))
                inv_rec.append(snapshot["inventory"])
                b_rec.append(snapshot["backlog"])
                pl_rec.append(snapshot["pipeline_len"])

            inv_sim[tr.trajectory_id] = np.array(inv_rec, dtype=float)
            b_sim[tr.trajectory_id] = np.array(b_rec, dtype=float)
            pl_sim[tr.trajectory_id] = np.array(pl_rec, dtype=float)
        return inv_sim, b_sim, pl_sim

    def _compute_loss_distributional(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compute calibration loss dominated by per-time-step Wasserstein distance between simulated and observed
        distributions pooled across trajectories, with additional MMD and MSE terms.
        """
        inv_sim, b_sim, pl_sim = self._simulate_training_window(params)

        inv_obs: Dict[str, np.ndarray] = {}
        b_obs: Dict[str, np.ndarray] = {}
        pl_obs: Dict[str, np.ndarray] = {}
        max_T = 0
        for tr in self.train_trajectories:
            te, _ = self.holdout_ranges[tr.trajectory_id]
            mask = tr.t <= te
            inv_obs[tr.trajectory_id] = tr.inventory_obs[mask].astype(float)
            b_obs[tr.trajectory_id] = tr.backlog_obs[mask].astype(float)
            pl_obs[tr.trajectory_id] = tr.pipeline_len_obs_counts[mask].astype(float)
            max_T = max(max_T, inv_obs[tr.trajectory_id].shape[0])

        wass_vals = []
        mmd_vals = []

        n_samples = self.n_samples_wass_mmd
        sigma = self.mmd_sigma

        for t in range(max_T):
            inv_o_t = []
            inv_s_t = []
            b_o_t = []
            b_s_t = []
            pl_o_t = []
            pl_s_t = []
            for tid in inv_obs.keys():
                if t < inv_obs[tid].shape[0]:
                    inv_o_t.append(inv_obs[tid][t])
                if tid in inv_sim and t < inv_sim[tid].shape[0]:
                    inv_s_t.append(inv_sim[tid][t])
                if t < b_obs[tid].shape[0]:
                    b_o_t.append(b_obs[tid][t])
                if tid in b_sim and t < b_sim[tid].shape[0]:
                    b_s_t.append(b_sim[tid][t])
                if t < pl_obs[tid].shape[0]:
                    pl_o_t.append(pl_obs[tid][t])
                if tid in pl_sim and t < pl_sim[tid].shape[0]:
                    pl_s_t.append(pl_sim[tid][t])

            def sample_arr(arr: List[float]) -> np.ndarray:
                if len(arr) == 0:
                    return np.zeros((0,), dtype=float)
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.shape[0] >= n_samples:
                    idx = np.random.choice(arr_np.shape[0], size=n_samples, replace=False)
                else:
                    idx = np.random.choice(arr_np.shape[0], size=n_samples, replace=True)
                return arr_np[idx]

            xi = sample_arr(inv_s_t)
            xo = sample_arr(inv_o_t)
            xb = sample_arr(b_s_t)
            yb = sample_arr(b_o_t)
            xp = sample_arr(pl_s_t)
            yp = sample_arr(pl_o_t)

            ws = []
            if xi.size > 0 and xo.size > 0:
                ws.append(wasserstein_1d(xi, xo))
            if xb.size > 0 and yb.size > 0:
                ws.append(wasserstein_1d(xb, yb))
            if xp.size > 0 and yp.size > 0:
                ws.append(wasserstein_1d(xp, yp))
            if len(ws) > 0:
                wass_vals.append(float(np.mean(ws)))
            mm = []
            if xi.size > 0 and xo.size > 0:
                mm.append(mmd_gaussian_1d(xi, xo, sigma=sigma))
            if xb.size > 0 and yb.size > 0:
                mm.append(mmd_gaussian_1d(xb, yb, sigma=sigma))
            if xp.size > 0 and yp.size > 0:
                mm.append(mmd_gaussian_1d(xp, yp, sigma=sigma))
            if len(mm) > 0:
                mmd_vals.append(float(np.mean(mm)))

        wass_mean = float(np.mean(wass_vals)) if len(wass_vals) > 0 else 0.0
        mmd_mean = float(np.mean(mmd_vals)) if len(mmd_vals) > 0 else 0.0

        mse_vals = []
        for tid in inv_obs.keys():
            if tid in inv_sim:
                inv_o, inv_s = inv_obs[tid], inv_sim[tid]
                T = min(inv_o.shape[0], inv_s.shape[0])
                mse_vals.append(np.mean((inv_o[:T] - inv_s[:T]) ** 2))
            if tid in b_sim:
                b_o, b_s = b_obs[tid], b_sim[tid]
                T = min(b_o.shape[0], b_s.shape[0])
                mse_vals.append(np.mean((b_o[:T] - b_s[:T]) ** 2))
            if tid in pl_sim:
                p_o, p_s = pl_obs[tid], pl_sim[tid]
                T = min(p_o.shape[0], p_s.shape[0])
                mse_vals.append(np.mean((p_o[:T] - p_s[:T]) ** 2))
        mse_mean = float(np.mean(mse_vals)) if len(mse_vals) > 0 else 0.0

        w1, w2, w3 = self.loss_weights
        loss = float(w1 * wass_mean + w2 * mmd_mean + w3 * mse_mean)
        diagnostics = {"wass": wass_mean, "mmd": mmd_mean, "mse": mse_mean}
        return loss, diagnostics

    def _simulation_wrapper(self, theta: Any) -> Any:
        try:
            theta_np = theta.detach().cpu().numpy().astype(float)
            params = self._theta_to_params(theta_np)
            feats = self._simulate_with_params_for_sbi(params)
            if np.any(~np.isfinite(feats)):
                return torch.tensor(np.full_like(feats, np.nan), dtype=torch.float32)
            return torch.tensor(feats.astype(np.float32), dtype=torch.float32)
        except Exception:
            return torch.tensor(np.full((len(self.x_observed_np),), np.nan, dtype=np.float32))

    def _simulate_with_params_for_sbi(self, params: Dict[str, Any]) -> np.ndarray:
        seed_val = self.seed + int(params["lead_time_L"]) * 17 + (1 if params["arrival_convention"].endswith("_1") else 0) * 29
        rng = np.random.default_rng(seed_val)

        inv_series = []
        b_series = []
        pl_series = []

        for tr in self.train_trajectories:
            te, _ = self.holdout_ranges[tr.trajectory_id]
            horizon = int(np.sum(tr.t <= te))
            if horizon <= 0:
                continue
            node = InventoryNode(
                init_inventory=int(tr.init_inventory),
                init_backlog=int(tr.init_backlog),
                init_pipeline=[Item(int(it.qty), int(it.remaining)) for it in tr.init_pipeline],
                lead_time=int(params["lead_time_L"]),
                arrival_convention=str(params["arrival_convention"]),
            )
            demand_model = create_demand_model_from_params(params, rng=rng)
            demand_model.reset()

            inv_rec = []
            b_rec = []
            pl_rec = []

            actions = tr.actions[:horizon]
            for tt in range(horizon):
                d = demand_model.sample(t=tt)
                snapshot = node.step(action=int(actions[tt]), demand=int(d))
                inv_rec.append(snapshot["inventory"])
                b_rec.append(snapshot["backlog"])
                pl_rec.append(snapshot["pipeline_len"])

            inv_series.append(np.array(inv_rec, dtype=float))
            b_series.append(np.array(b_rec, dtype=float))
            pl_series.append(np.array(pl_rec, dtype=float))

        return compute_summary_features(inv_series, b_series, pl_series)

    def fit(self) -> Dict[str, Any]:
        use_sbi = (self.opt_method in ("auto", "sbi")) and SBI_AVAILABLE and TORCH_AVAILABLE
        if use_sbi:
            try:
                return self._fit_sbi()
            except Exception:
                traceback.print_exc()
                warnings.warn("SBI calibration failed. Falling back to gradient-free optimization.")
        return self._fit_gradient_free()

    def _fit_sbi(self) -> Dict[str, Any]:
        low_t = torch.tensor(self.low, dtype=torch.float32) if isinstance(self.low, np.ndarray) else self.low
        high_t = torch.tensor(self.high, dtype=torch.float32) if isinstance(self.high, np.ndarray) else self.high

        prior = sbi_utils.BoxUniform(low=low_t, high=high_t)

        print(f"Running SBI training with exactly {self.num_simulations} parameter samples...")
        theta, x = simulate_for_sbi_safe(
            self._simulation_wrapper,
            prior,
            num_simulations=self.num_simulations,
        )
        print(f"Completed {self.num_simulations} parameter samples for SBI training.")

        nan_mask = torch.any(~torch.isfinite(x), dim=1)
        theta = theta[~nan_mask]
        x = x[~nan_mask]
        if theta.shape[0] == 0:
            raise RuntimeError("All simulations produced NaNs. Check simulation wrapper and data.")

        inference = NPE(prior=prior)
        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator)

        x_observed_t = torch.tensor(self.x_observed_np, dtype=torch.float32)
        total_samples = self.num_samples_posterior
        samples_list = []
        start_time = time.time()
        chunk = min(2048, total_samples)
        while len(samples_list) < total_samples:
            if (time.time() - start_time) > self.sampling_timeout:
                warnings.warn("Posterior sampling timeout reached; using collected samples so far.")
                break
            need = min(chunk, total_samples - len(samples_list))
            try:
                s = posterior.sample((need,), x=x_observed_t)
                samples_list.append(s.detach().cpu())
            except Exception:
                try:
                    s = posterior.sample((need,))
                    samples_list.append(s.detach().cpu())
                except Exception:
                    traceback.print_exc()
                    break
        if len(samples_list) == 0:
            samples = (low_t + (high_t - low_t) / 2.0).unsqueeze(0)
        else:
            samples = torch.cat(samples_list, dim=0)

        samples_np = samples.detach().cpu().numpy()
        self.posterior_samples = samples_np
        optimized = samples.mean(dim=0).detach().cpu().numpy()
        self.optimized_params = optimized
        return self._theta_to_params(optimized)

    def _fit_gradient_free(self) -> Dict[str, Any]:
        rng = np.random.RandomState(self.seed + 777)

        low_np = self.low if isinstance(self.low, np.ndarray) else self.low.numpy()
        high_np = self.high if isinstance(self.high, np.ndarray) else self.high.numpy()

        def sample_theta() -> np.ndarray:
            u = rng.uniform(0.0, 1.0, size=low_np.shape[0])
            theta = low_np + u * (high_np - low_np)
            theta[0] = float(np.round(theta[0]))
            theta[1] = float(np.round(theta[1]))
            return theta

        starts: List[np.ndarray] = []
        L_grid = [2.0, 3.0, 4.0]
        A_grid = [0.0, 1.0]
        base_theta = (low_np + high_np) / 2.0
        for L in L_grid:
            for af in A_grid:
                th = base_theta.copy()
                th[0] = L
                th[1] = af
                if self.demand_family == "Poisson":
                    th[2] = min(20.0, max(0.0, 5.0))
                elif self.demand_family == "NegBin":
                    th[2] = min(20.0, max(0.0, 5.0))
                    th[3] = 5.0
                elif self.demand_family == "AR1":
                    th[2] = 5.0
                    th[3] = 0.0
                    th[4] = 1.0
                starts.append(th)
        for _ in range(max(0, 10 - len(starts))):
            starts.append(sample_theta())

        eval_count = 0
        best_loss = float("inf")
        best_theta = starts[0]

        def obj(theta_vec: np.ndarray) -> float:
            nonlocal eval_count, best_loss, best_theta
            eval_count += 1
            theta_c = np.minimum(high_np, np.maximum(low_np, theta_vec))
            params = self._theta_to_params(theta_c)
            loss, _ = self._compute_loss_distributional(params)
            if loss < best_loss:
                best_loss = loss
                best_theta = theta_c.copy()
            return float(loss)

        if SCIPY_AVAILABLE and (self.opt_method in ("auto", "nelder-mead")):
            options = {"maxfev": self.max_evals, "xatol": 1e-2, "fatol": 1e-3, "maxiter": self.max_evals}
            for start_theta in starts:
                try:
                    _ = scipy_optimize.minimize(
                        obj, x0=start_theta, method="Nelder-Mead", options=options
                    )
                    if eval_count >= self.max_evals:
                        break
                except Exception:
                    traceback.print_exc()
                    continue
        else:
            total = self.max_evals
            for st in starts:
                try:
                    _ = obj(st)
                except Exception:
                    continue
                if eval_count >= total:
                    break
            while eval_count < total:
                th = sample_theta()
                try:
                    _ = obj(th)
                except Exception:
                    continue

        self.optimized_params = best_theta
        return self._theta_to_params(best_theta)


# ----------------------------
# Simulator orchestrator
# ----------------------------

class BeerGameSimulator:
    """
    Orchestrates rollouts for a given parameter set on supplied trajectories.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def rollout(
        self,
        trajectories: List[TrajectoryData],
    ) -> SimulationResults:
        """
        Roll out the simulator deterministically on provided trajectories using playback actions.
        """
        seed_val = GLOBAL_SEED + int(self.params["lead_time_L"]) * 11 + (
            1 if self.params["arrival_convention"].endswith("_1") else 0
        ) * 37 + 123
        rng = np.random.default_rng(seed_val)
        if TORCH_AVAILABLE:
            torch.manual_seed(seed_val)

        inv_sim: Dict[str, np.ndarray] = {}
        b_sim: Dict[str, np.ndarray] = {}
        plen_sim: Dict[str, np.ndarray] = {}
        occ_sim: Dict[str, List[np.ndarray]] = {}

        inv_obs: Dict[str, np.ndarray] = {}
        b_obs: Dict[str, np.ndarray] = {}
        plen_obs: Dict[str, np.ndarray] = {}
        occ_obs: Dict[str, List[np.ndarray]] = {}

        for tr in trajectories:
            node = InventoryNode(
                init_inventory=int(tr.init_inventory),
                init_backlog=int(tr.init_backlog),
                init_pipeline=[Item(int(it.qty), int(it.remaining)) for it in tr.init_pipeline],
                lead_time=int(self.params["lead_time_L"]),
                arrival_convention=str(self.params["arrival_convention"]),
            )
            demand_model = create_demand_model_from_params(self.params, rng=rng)
            demand_model.reset()
            horizon = len(tr.t)
            if len(tr.actions) < horizon:
                raise ValueError(
                    f"Trajectory {tr.trajectory_id} has fewer actions ({len(tr.actions)}) than time steps ({horizon})."
                )

            inv_rec = []
            b_rec = []
            plen_rec = []
            occ_s_rec: List[np.ndarray] = []

            for tt in range(horizon):
                d = demand_model.sample(t=tt)
                snap = node.step(action=int(tr.actions[tt]), demand=int(d))
                inv_rec.append(snap["inventory"])
                b_rec.append(snap["backlog"])
                plen_rec.append(snap["pipeline_len"])
                occ_s_rec.append(snap["pipeline_occ"])

            inv_obs_arr = tr.inventory_obs.astype(float)
            b_obs_arr = tr.backlog_obs.astype(float)
            plen_obs_arr = tr.pipeline_len_obs_counts.astype(float)

            L = int(self.params["lead_time_L"])
            occ_o_rec: List[np.ndarray] = []
            for items in tr.pipeline_items_obs:
                occ = np.zeros(max(1, L), dtype=float)
                for it in items:
                    rem = int(it.remaining)
                    if rem <= 0:
                        idx = 0
                    elif 1 <= rem <= L:
                        idx = rem - 1
                    else:
                        idx = L - 1
                    occ[idx] += int(it.qty)
                occ_o_rec.append(occ)

            inv_sim[tr.trajectory_id] = np.array(inv_rec, dtype=float)
            b_sim[tr.trajectory_id] = np.array(b_rec, dtype=float)
            plen_sim[tr.trajectory_id] = np.array(plen_rec, dtype=float)
            occ_sim[tr.trajectory_id] = occ_s_rec

            inv_obs[tr.trajectory_id] = inv_obs_arr
            b_obs[tr.trajectory_id] = b_obs_arr
            plen_obs[tr.trajectory_id] = plen_obs_arr
            occ_obs[tr.trajectory_id] = occ_o_rec

        return SimulationResults(
            inventory_sim=inv_sim,
            backlog_sim=b_sim,
            pipeline_len_sim=plen_sim,
            pipeline_occ_sim=occ_sim,
            inventory_obs=inv_obs,
            backlog_obs=b_obs,
            pipeline_len_obs=plen_obs,
            pipeline_occ_obs=occ_obs,
        )


# ----------------------------
# Evaluator
# ----------------------------

class Evaluator:
    """
    Computes evaluation metrics comparing simulated trajectories to observations.
    """

    @staticmethod
    def _compute_wass_mmd_per_t(
        results: SimulationResults,
        n_samples: int = 200,
        sigma: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute per-time-step Wasserstein and MMD across pooled trajectories for
        inventory, backlog, and pipeline_len; average across dims and time.
        """
        inv_sim = results.inventory_sim
        inv_obs = results.inventory_obs
        b_sim = results.backlog_sim
        b_obs = results.backlog_obs
        pl_sim = results.pipeline_len_sim
        pl_obs = results.pipeline_len_obs

        max_T = 0
        for tid in inv_obs.keys():
            max_T = max(max_T, inv_obs[tid].shape[0])

        wass_vals = []
        mmd_vals = []

        for t in range(max_T):
            inv_o_t = []
            inv_s_t = []
            b_o_t = []
            b_s_t = []
            pl_o_t = []
            pl_s_t = []
            for tid in inv_obs.keys():
                if t < inv_obs[tid].shape[0]:
                    inv_o_t.append(inv_obs[tid][t])
                if tid in inv_sim and t < inv_sim[tid].shape[0]:
                    inv_s_t.append(inv_sim[tid][t])
                if t < b_obs[tid].shape[0]:
                    b_o_t.append(b_obs[tid][t])
                if tid in b_sim and t < b_sim[tid].shape[0]:
                    b_s_t.append(b_sim[tid][t])
                if t < pl_obs[tid].shape[0]:
                    pl_o_t.append(pl_obs[tid][t])
                if tid in pl_sim and t < pl_sim[tid].shape[0]:
                    pl_s_t.append(pl_sim[tid][t])

            def sample_arr(arr: List[float]) -> np.ndarray:
                if len(arr) == 0:
                    return np.zeros((0,), dtype=float)
                arr_np = np.asarray(arr, dtype=float)
                if arr_np.shape[0] >= n_samples:
                    idx = np.random.choice(arr_np.shape[0], size=n_samples, replace=False)
                else:
                    idx = np.random.choice(arr_np.shape[0], size=n_samples, replace=True)
                return arr_np[idx]

            xi = sample_arr(inv_s_t)
            xo = sample_arr(inv_o_t)
            xb = sample_arr(b_s_t)
            yb = sample_arr(b_o_t)
            xp = sample_arr(pl_s_t)
            yp = sample_arr(pl_o_t)

            ws = []
            if xi.size > 0 and xo.size > 0:
                ws.append(wasserstein_1d(xi, xo))
            if xb.size > 0 and yb.size > 0:
                ws.append(wasserstein_1d(xb, yb))
            if xp.size > 0 and yp.size > 0:
                ws.append(wasserstein_1d(xp, yp))
            if len(ws) > 0:
                wass_vals.append(float(np.mean(ws)))
            mm = []
            if xi.size > 0 and xo.size > 0:
                mm.append(mmd_gaussian_1d(xi, xo, sigma=sigma))
            if xb.size > 0 and yb.size > 0:
                mm.append(mmd_gaussian_1d(xb, yb, sigma=sigma))
            if xp.size > 0 and yp.size > 0:
                mm.append(mmd_gaussian_1d(xp, yp, sigma=sigma))
            if len(mm) > 0:
                mmd_vals.append(float(np.mean(mm)))

        return {
            "Wasserstein_per_t": float(np.mean(wass_vals)) if len(wass_vals) > 0 else 0.0,
            "MMD_per_t": float(np.mean(mmd_vals)) if len(mmd_vals) > 0 else 0.0,
        }

    @staticmethod
    def compute_metrics(results: SimulationResults, n_samples_wass_mmd: int = 200, mmd_sigma: float = 1.0) -> Dict[str, Any]:
        """
        Compute aggregated metrics across trajectories.
        """
        per_traj = {}
        all_inv_sim = []
        all_inv_obs = []
        all_b_sim = []
        all_b_obs = []
        all_pl_sim = []
        all_pl_obs = []
        l1_per_traj = []

        for tid in results.inventory_sim.keys():
            inv_s = results.inventory_sim[tid]
            inv_o = results.inventory_obs[tid]
            b_s = results.backlog_sim[tid]
            b_o = results.backlog_obs[tid]
            pl_s = results.pipeline_len_sim[tid]
            pl_o = results.pipeline_len_obs[tid]
            T = min(len(inv_s), len(inv_o), len(b_s), len(b_o), len(pl_s), len(pl_o))

            inv_s = inv_s[:T]
            inv_o = inv_o[:T]
            b_s = b_s[:T]
            b_o = b_o[:T]
            pl_s = pl_s[:T]
            pl_o = pl_o[:T]

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
            l1_per_traj.append(l1_mean)

            per_traj[tid] = {
                "RMSE_inventory": rmse_i,
                "RMSE_backlog": rmse_b,
                "MAE_pipeline_len": mae_pl,
                "PipelineComp_L1": l1_mean,
            }

            all_inv_sim.append(inv_s)
            all_inv_obs.append(inv_o)
            all_b_sim.append(b_s)
            all_b_obs.append(b_o)
            all_pl_sim.append(pl_s)
            all_pl_obs.append(pl_o)

        inv_sim_flat = np.concatenate(all_inv_sim) if len(all_inv_sim) > 0 else np.array([], dtype=float)
        inv_obs_flat = np.concatenate(all_inv_obs) if len(all_inv_obs) > 0 else np.array([], dtype=float)
        b_sim_flat = np.concatenate(all_b_sim) if len(all_b_sim) > 0 else np.array([], dtype=float)
        b_obs_flat = np.concatenate(all_b_obs) if len(all_b_obs) > 0 else np.array([], dtype=float)

        def subsample(arr: np.ndarray, n: int = 2000) -> np.ndarray:
            if arr.shape[0] <= n:
                return arr
            idx = np.linspace(0, arr.shape[0] - 1, num=n, dtype=int)
            return arr[idx]

        inv_s_sub = subsample(inv_sim_flat)
        inv_o_sub = subsample(inv_obs_flat)
        b_s_sub = subsample(b_sim_flat)
        b_o_sub = subsample(b_obs_flat)

        wass_mmd = Evaluator._compute_wass_mmd_per_t(results, n_samples=n_samples_wass_mmd, sigma=mmd_sigma)

        mse_inv = float(np.mean((np.concatenate(all_inv_sim) - np.concatenate(all_inv_obs)) ** 2)) if all_inv_sim else 0.0
        mse_b = float(np.mean((np.concatenate(all_b_sim) - np.concatenate(all_b_obs)) ** 2)) if all_b_sim else 0.0
        mse_pl = float(np.mean((np.concatenate(all_pl_sim) - np.concatenate(all_pl_obs)) ** 2)) if all_pl_sim else 0.0
        mse_t = 0.0

        metrics = {
            "per_trajectory": per_traj,
            "aggregate": {
                "RMSE_inventory_mean": float(np.mean([d["RMSE_inventory"] for d in per_traj.values()])) if per_traj else 0.0,
                "RMSE_inventory_std": float(np.std([d["RMSE_inventory"] for d in per_traj.values()])) if per_traj else 0.0,
                "RMSE_backlog_mean": float(np.mean([d["RMSE_backlog"] for d in per_traj.values()])) if per_traj else 0.0,
                "RMSE_backlog_std": float(np.std([d["RMSE_backlog"] for d in per_traj.values()])) if per_traj else 0.0,
                "MAE_pipeline_len_mean": float(np.mean([d["MAE_pipeline_len"] for d in per_traj.values()])) if per_traj else 0.0,
                "MAE_pipeline_len_std": float(np.std([d["MAE_pipeline_len"] for d in per_traj.values()])) if per_traj else 0.0,
                "PipelineComp_L1_mean": float(np.mean(l1_per_traj)) if l1_per_traj else 0.0,
                "PipelineComp_L1_std": float(np.std(l1_per_traj)) if l1_per_traj else 0.0,
            },
            "distributional": {
                "Wasserstein_per_t": wass_mmd["Wasserstein_per_t"],
                "MMD_per_t": wass_mmd["MMD_per_t"],
                "MSE": {
                    "inventory": mse_inv,
                    "backlog": mse_b,
                    "pipeline_len": mse_pl,
                    "t": mse_t,
                },
            },
            "sbi_distances_proxy": {
                "ApproxWasserstein_inventory": approx_wasserstein_1d(inv_s_sub, inv_o_sub),
                "MMD_inventory": compute_mmd_rbf(inv_s_sub, inv_o_sub, gamma=1.0),
                "MSE_inventory": float(np.mean((inv_sim_flat - inv_obs_flat) ** 2)) if inv_sim_flat.size > 0 else 0.0,
                "ApproxWasserstein_backlog": approx_wasserstein_1d(b_s_sub, b_o_sub),
                "MMD_backlog": compute_mmd_rbf(b_s_sub, b_o_sub, gamma=1.0),
                "MSE_backlog": float(np.mean((b_sim_flat - b_obs_flat) ** 2)) if b_sim_flat.size > 0 else 0.0,
            }
        }
        n_traj = max(1, len(per_traj))

        def ci95(std: float) -> float:
            return 1.96 * std / math.sqrt(n_traj)

        agg = metrics["aggregate"]
        agg["RMSE_inventory_CI95"] = ci95(agg["RMSE_inventory_std"])
        agg["RMSE_backlog_CI95"] = ci95(agg["RMSE_backlog_std"])
        agg["MAE_pipeline_len_CI95"] = ci95(agg["MAE_pipeline_len_std"])
        agg["PipelineComp_L1_CI95"] = ci95(agg["PipelineComp_L1_std"])

        return metrics


def run_double_monte_carlo(
    calibrator: "SBICalibrator",
    trajectories: List[TrajectoryData],
    M: int,
    K: int,
    seed: int,
    n_samples_wass_mmd: int = 200,
    mmd_sigma: float = 1.0,
) -> Dict[str, Any]:
    """
    Double Monte Carlo: sample M parameter sets from posterior; for each, run K simulations.
    Aggregate mean and 95% CI across M samples (inner run average for each sample).
    """
    if calibrator.posterior_samples is None or calibrator.posterior_samples.shape[0] == 0:
        raise ValueError("Posterior samples not available. SBI calibration must be used with posterior sampling.")
    
    rng = np.random.RandomState(seed)
    n_posterior = calibrator.posterior_samples.shape[0]
    
    # Sample M parameter sets from posterior (with replacement if M > n_posterior)
    if M <= n_posterior:
        selected_indices = rng.choice(n_posterior, size=M, replace=False)
    else:
        selected_indices = rng.choice(n_posterior, size=M, replace=True)
    
    selected_samples = calibrator.posterior_samples[selected_indices]
    
    # Storage for results across M parameter samples
    per_m_results = []
    per_m_metrics = []
    
    for m in range(M):
        # Convert theta vector to parameters
        theta_vec = selected_samples[m]
        params = calibrator._theta_to_params(theta_vec)
        
        # Run K simulations with different random seeds
        per_k_results = []
        per_k_metrics = []
        
        for k in range(K):
            # Create simulator with sampled parameters
            simulator = BeerGameSimulator(params=params)
            
            # Set random seed for this run
            run_seed = seed + m * K + k
            random.seed(run_seed)
            np.random.seed(run_seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(run_seed)
            
            # Run simulation
            sim_results = simulator.rollout(trajectories)
            per_k_results.append(sim_results)
            
            # Compute metrics for this run
            evaluator = Evaluator()
            metrics = evaluator.compute_metrics(sim_results, n_samples_wass_mmd=n_samples_wass_mmd, mmd_sigma=mmd_sigma)
            per_k_metrics.append(metrics)
        
        # Average results across K runs for this parameter sample
        # For simplicity, we'll average the metrics directly
        avg_metrics = {}
        if per_k_metrics:
            for key in per_k_metrics[0].keys():
                if isinstance(per_k_metrics[0][key], dict):
                    avg_metrics[key] = {}
                    for subkey in per_k_metrics[0][key].keys():
                        values = [m[key][subkey] for m in per_k_metrics if key in m and subkey in m[key]]
                        if values:
                            avg_metrics[key][subkey] = float(np.mean(values))
                else:
                    values = [m[key] for m in per_k_metrics if key in m]
                    if values:
                        avg_metrics[key] = float(np.mean(values))
        
        per_m_results.append(per_k_results[0])  # Store first result as representative
        per_m_metrics.append(avg_metrics)
    
    # Aggregate results across M parameter samples
    final_metrics = {}
    if per_m_metrics:
        # Aggregate metrics across M samples
        for key in per_m_metrics[0].keys():
            if isinstance(per_m_metrics[0][key], dict):
                final_metrics[key] = {}
                for subkey in per_m_metrics[0][key].keys():
                    values = [m[key][subkey] for m in per_m_metrics if key in m and subkey in m[key]]
                    if values:
                        mean_val = float(np.mean(values))
                        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        ci95 = 1.96 * std_val / math.sqrt(len(values)) if len(values) > 1 else 0.0
                        final_metrics[key][subkey] = mean_val
                        final_metrics[key][f"{subkey}_std"] = std_val
                        final_metrics[key][f"{subkey}_CI95"] = ci95
            else:
                values = [m[key] for m in per_m_metrics if key in m]
                if values:
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ci95 = 1.96 * std_val / math.sqrt(len(values)) if len(values) > 1 else 0.0
                    final_metrics[key] = mean_val
                    final_metrics[f"{key}_std"] = std_val
                    final_metrics[f"{key}_CI95"] = ci95
    
    # Use first result as representative for simulation results
    representative_results = per_m_results[0] if per_m_results else None
    
    return {
        "metrics": final_metrics,
        "sim_results": representative_results,
        "M": M,
        "K": K,
        "n_samples_used": len(selected_samples),
    }




class GlobalSensitivityAnalyzer:
    """
    Wrapper around SALib to compute global sensitivity indices using the trained calibrator.
    """

    def __init__(
        self,
        calibrator: "SBICalibrator",
        method: str = "both",
        n_samples: int = 256,
        num_levels: int = 6,
        calc_second_order: bool = False,
        seed: Optional[int] = None,
        apply_alpha_noise: bool = False,
    ) -> None:
        if not SALIB_AVAILABLE:
            raise ImportError("SALib is not installed; cannot run global sensitivity analysis.")
        self.calibrator = calibrator
        self.method = method.lower()
        self.n_samples = int(n_samples)
        self.num_levels = int(num_levels)
        self.calc_second_order = bool(calc_second_order)
        self.seed = seed
        self.apply_alpha_noise = bool(apply_alpha_noise)
        self.problem = {
            "num_vars": len(self.calibrator.base_param_names),
            "names": self.calibrator.base_param_names,
            "bounds": [
                [
                    float(self.calibrator.base_low[idx]),
                    float(self.calibrator.base_high[idx]),
                ]
                for idx in range(len(self.calibrator.base_param_names))
            ],
        }

    def run(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        target_methods = []
        if self.method in {"both", "morris"}:
            target_methods.append("morris")
        if self.method in {"both", "sobol"}:
            target_methods.append("sobol")
        for method in target_methods:
            if method == "morris":
                results["morris"] = self._run_morris()
            elif method == "sobol":
                results["sobol"] = self._run_sobol()
        return results

    def _run_morris(self) -> Dict[str, Any]:
        sample_kwargs = {
            "problem": self.problem,
            "N": self.n_samples,
            "num_levels": self.num_levels,
        }
        samples = self._call_with_seed(salib_morris_sample.sample, **sample_kwargs)
        Y = self._evaluate_samples(samples)
        analysis = salib_morris_analyze.analyze(
            self.problem,
            samples,
            Y,
            conf_level=0.95,
            num_levels=self.num_levels,
            print_to_console=False,
        )
        return {
            "names": self.calibrator.base_param_names,
            "mu": self._serialize_vector(analysis["mu"]),
            "mu_star": self._serialize_vector(analysis["mu_star"]),
            "sigma": self._serialize_vector(analysis["sigma"]),
            "mu_star_conf": self._serialize_vector(analysis.get("mu_star_conf", np.zeros_like(analysis["mu_star"]))),
        }

    def _run_sobol(self) -> Dict[str, Any]:
        sample_kwargs = {
            "problem": self.problem,
            "N": self.n_samples,
            "calc_second_order": self.calc_second_order,
        }
        samples = self._call_with_seed(salib_saltelli_sample.sample, **sample_kwargs)
        Y = self._evaluate_samples(samples)
        analysis = salib_sobol_analyze.analyze(
            self.problem,
            Y,
            calc_second_order=self.calc_second_order,
            print_to_console=False,
        )
        result: Dict[str, Any] = {
            "names": self.calibrator.base_param_names,
            "S1": self._serialize_vector(analysis["S1"]),
            "S1_conf": self._serialize_vector(analysis["S1_conf"]),
            "ST": self._serialize_vector(analysis["ST"]),
            "ST_conf": self._serialize_vector(analysis["ST_conf"]),
        }
        if self.calc_second_order and "S2" in analysis:
            result["S2"] = self._serialize_matrix(analysis["S2"])
            result["S2_conf"] = self._serialize_matrix(analysis.get("S2_conf", np.zeros_like(analysis["S2"])))
        return result

    def _evaluate_samples(self, samples: np.ndarray) -> np.ndarray:
        outputs = np.zeros(samples.shape[0], dtype=float)
        for idx, theta in enumerate(samples):
            outputs[idx] = self.calibrator.evaluate_loss_from_base(theta, apply_alpha_noise=self.apply_alpha_noise)
        if not np.all(np.isfinite(outputs)):
            raise RuntimeError("GSA evaluation produced non-finite values. Check simulator stability.")
        return outputs

    def _call_with_seed(self, func, **kwargs):
        if self.seed is not None:
            try:
                return func(seed=self.seed, **kwargs)
            except TypeError:
                np.random.seed(self.seed)
        return func(**kwargs)

    def _serialize_vector(self, vec: np.ndarray) -> List[Optional[float]]:
        result: List[Optional[float]] = []
        for val in np.asarray(vec).tolist():
            if val is None:
                result.append(None)
            else:
                try:
                    if math.isnan(val):
                        result.append(None)
                    else:
                        result.append(float(val))
                except (TypeError, ValueError):
                    result.append(None)
        return result

    def _serialize_matrix(self, mat: np.ndarray) -> List[List[Optional[float]]]:
        mat_np = np.asarray(mat)
        rows: List[List[Optional[float]]] = []
        for row in mat_np:
            rows.append(self._serialize_vector(row))
        return rows

def save_gsa_outputs(results_dir: str, prefix: str, gsa_results: Dict[str, Any]) -> None:
    os.makedirs(results_dir, exist_ok=True)
    for method, content in gsa_results.items():
        fname = f"{prefix}_{method}.json"
        with open(os.path.join(results_dir, fname), "w") as f:
            json.dump(content, f, indent=2)


# ----------------------------
# Results saving
# ----------------------------

def save_results(
    data_dir: str,
    results_dirname: Optional[str],
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    sim_results: SimulationResults,
    posterior_samples: Optional[np.ndarray] = None,
    split_tag: str = "val",
) -> str:
    """
    Save optimized parameters, metrics, and simulated traces to disk.
    """
    results_dir = os.path.join(data_dir, results_dirname or "results")
    os.makedirs(results_dir, exist_ok=True)

    if split_tag == "val":
        with open(os.path.join(results_dir, "optimized_params.json"), "w") as f:
            json.dump(params, f, indent=2)

        if posterior_samples is not None:
            np.save(os.path.join(results_dir, "posterior_samples.npy"), posterior_samples)

        config = {
            "arrival_convention": params.get("arrival_convention"),
            "demand_family": params.get("demand_family", "Poisson"),
            "lead_time_L": params.get("lead_time_L"),
            "poisson_lambda": params.get("poisson_lambda", None),
            "negbin_mu": params.get("negbin_mu", None),
            "negbin_r": params.get("negbin_r", None),
            "ar1_mu": params.get("ar1_mu", None),
            "ar1_phi": params.get("ar1_phi", None),
            "ar1_sigma": params.get("ar1_sigma", None),
            "seasonal_amplitude": params.get("seasonal_amplitude", 0.0),
            "seasonal_period": params.get("seasonal_period", 7),
        }
        with open(os.path.join(results_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    metrics_fname = "metrics.json" if split_tag == "val" else f"metrics_{split_tag}.json"
    with open(os.path.join(results_dir, metrics_fname), "w") as f:
        json.dump(metrics, f, indent=2)

    rows = []
    for tid in sim_results.inventory_sim.keys():
        inv_s = sim_results.inventory_sim[tid]
        inv_o = sim_results.inventory_obs[tid]
        b_s = sim_results.backlog_sim[tid]
        b_o = sim_results.backlog_obs[tid]
        pl_s = sim_results.pipeline_len_sim[tid]
        pl_o = sim_results.pipeline_len_obs[tid]
        T = min(len(inv_s), len(inv_o), len(b_s), len(b_o), len(pl_s), len(pl_o))
        for t in range(T):
            rows.append(
                {
                    "trajectory_id": tid,
                    "t": t,
                    "inventory_sim": float(inv_s[t]),
                    "inventory_obs": float(inv_o[t]),
                    "backlog_sim": float(b_s[t]),
                    "backlog_obs": float(b_o[t]),
                    "pipeline_len_sim": float(pl_s[t]),
                    "pipeline_len_obs": float(pl_o[t]),
                }
            )
    traces_path = os.path.join(results_dir, f"simulated_traces_{split_tag}.csv")
    pd.DataFrame(rows).to_csv(traces_path, index=False)
    return results_dir


# ----------------------------
# Orchestrator main flow
# ----------------------------

def main() -> None:
    """
    End-to-end workflow:
        parse_cli() -> load_data() -> build_trajectories() ->
        holdout_split() -> calibrator.fit() -> simulator.rollout() ->
        evaluator.compute_metrics() -> save_results()
    """
    args = parse_cli()
    data_dir = validate_env_paths(args.data_dir)
    
    # Try to load OOD test file (check metadata first, then CLI arg)
    test_ood_file = args.test_ood_file
    if test_ood_file is None:
        # Try to auto-detect from metadata
        metadata_path = os.path.join(data_dir, args.metadata_file)
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_temp = json.load(f)
                if "data_files" in metadata_temp and "test_ood" in metadata_temp["data_files"]:
                    test_ood_file = metadata_temp["data_files"]["test_ood"]
                    print(f"Auto-detected OOD test file from metadata: {test_ood_file}")
    
    train_df, val_df, test_df, test_df_ood, metadata = load_data(
        data_dir=data_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        metadata_file=args.metadata_file,
        test_file=args.test_file,
        test_ood_file=test_ood_file,
    )
    train_trajectories = build_trajectories(train_df, metadata)
    val_trajectories = build_trajectories(val_df, metadata)
    test_trajectories = build_trajectories(test_df, metadata) if test_df is not None else []
    test_trajectories_ood = build_trajectories(test_df_ood, metadata) if test_df_ood is not None else []

    train_trajs, holdout_ranges = holdout_split(train_trajectories, train_end_inclusive=48)

    calibrator = SBICalibrator(
        train_trajectories=train_trajs,
        holdout_ranges=holdout_ranges,
        demand_family=args.demand_family,
        num_simulations=args.num_simulations,
        num_samples_posterior=args.num_samples_posterior,
        sampling_timeout=args.sampling_timeout,
        seed=int(metadata.get("random_seed", metadata.get("seed", GLOBAL_SEED))),
        opt_method=args.opt_method,
        max_evals=args.max_evals,
        n_samples_wass_mmd=args.n_samples_wass_mmd,
        mmd_sigma=args.mmd_sigma,
        use_alpha=args.use_alpha,
        alpha_params=args.alpha_params,
        alpha_max_ratio=args.alpha_max_ratio,
        alpha_mode=args.alpha_mode,
    )
    optimized_params = calibrator.fit()

    alpha_report = None
    if calibrator.use_alpha and calibrator.optimized_params is not None and calibrator.r_param_names:
        import numpy as _np

        base_indices = {name: calibrator.base_param_names.index(name) for name in calibrator.alpha_params if name in calibrator.base_param_names}
        r_indices = {name: calibrator.param_names.index(f"r::{name}") for name in calibrator.alpha_params if f"r::{name}" in calibrator.param_names}
        bounds = {name: calibrator.param_bounds.get(name) for name in calibrator.alpha_params}

        def _collect_alpha_stats(arr: _np.ndarray, with_stats: bool) -> dict:
            result = {}
            for name in calibrator.alpha_params:
                if name not in base_indices or name not in r_indices or bounds.get(name) is None:
                    continue
                base_vals = arr[..., base_indices[name]]
                r_vals = arr[..., r_indices[name]]
                lower, upper = bounds[name]
                margins = _np.minimum(base_vals - lower, upper - base_vals)
                alpha_vals = r_vals * margins
                if with_stats:
                    result[name] = {
                        "base_mean": float(_np.mean(base_vals)),
                        "base_std": float(_np.std(base_vals)),
                        "r_mean": float(_np.mean(r_vals)),
                        "r_std": float(_np.std(r_vals)),
                        "alpha_mean": float(_np.mean(alpha_vals)),
                        "alpha_std": float(_np.std(alpha_vals)),
                    }
                else:
                    result[name] = {
                        "base_value": float(base_vals),
                        "r": float(r_vals),
                        "margin": float(margins),
                        "alpha": float(alpha_vals),
                    }
            return result

        alpha_report = {}
        theta_opt = _np.asarray(calibrator.optimized_params, dtype=float)
        if theta_opt.ndim == 1:
            alpha_report["optimized"] = _collect_alpha_stats(theta_opt, with_stats=False)
        if calibrator.posterior_samples is not None and calibrator.posterior_samples.size > 0:
            alpha_report["posterior"] = _collect_alpha_stats(calibrator.posterior_samples, with_stats=True)

    results_dir_base = os.path.join(data_dir, args.results_dir or "results")

    gsa_results: Optional[Dict[str, Any]] = None
    if args.run_gsa:
        if not SALIB_AVAILABLE:
            warnings.warn("SALib is not installed; skipping requested global sensitivity analysis.")
        else:
            gsa_seed = args.gsa_seed
            if gsa_seed is None:
                base_seed = int(metadata.get("random_seed", metadata.get("seed", GLOBAL_SEED)))
                gsa_seed = base_seed + 5000
            try:
                gsa_analyzer = GlobalSensitivityAnalyzer(
                    calibrator=calibrator,
                    method=args.gsa_method,
                    n_samples=args.gsa_num_samples,
                    num_levels=args.gsa_num_levels,
                    calc_second_order=args.gsa_second_order,
                    seed=gsa_seed,
                    apply_alpha_noise=args.use_alpha,
                )
                gsa_results = gsa_analyzer.run()
                gsa_dir = os.path.join(data_dir, args.results_dir or "results")
                save_gsa_outputs(gsa_dir, args.gsa_results_prefix, gsa_results)
                print(f"GSA results saved to {gsa_dir} (prefix='{args.gsa_results_prefix}').")
            except Exception as exc:
                traceback.print_exc()
                warnings.warn(f"GSA failed: {exc}")

    # Use Double Monte Carlo if enabled and SBI posterior samples are available
    use_double_mc = args.double_mc and calibrator.posterior_samples is not None and calibrator.posterior_samples.shape[0] > 0
    
    if use_double_mc:
        print(f"Running Double Monte Carlo: M={args.mc_M} parameter samples, K={args.mc_K} runs per sample")
        
        # Double Monte Carlo for validation set
        dmc_seed = int(metadata.get("random_seed", metadata.get("seed", GLOBAL_SEED))) + 1000
        dmc_results_val = run_double_monte_carlo(
            calibrator=calibrator,
            trajectories=val_trajectories,
            M=args.mc_M,
            K=args.mc_K,
            seed=dmc_seed,
            n_samples_wass_mmd=args.n_samples_wass_mmd,
            mmd_sigma=args.mmd_sigma,
        )
        metrics_val = dmc_results_val["metrics"]
        sim_results_val = dmc_results_val["sim_results"]
        
        results_dir_val = save_results(
            data_dir=data_dir,
            results_dirname=args.results_dir or "results",
            params=optimized_params,
            metrics=metrics_val,
            sim_results=sim_results_val,
            posterior_samples=calibrator.posterior_samples,
            split_tag="val",
        )
        print(f"Saved validation results (Double Monte Carlo) to: {results_dir_val}")
        print(f"  Used {dmc_results_val['n_samples_used']} posterior samples")
        
        # Double Monte Carlo for test set
        if len(test_trajectories) > 0:
            dmc_results_test = run_double_monte_carlo(
                calibrator=calibrator,
                trajectories=test_trajectories,
                M=args.mc_M,
                K=args.mc_K,
                seed=dmc_seed + 2000,
                n_samples_wass_mmd=args.n_samples_wass_mmd,
                mmd_sigma=args.mmd_sigma,
            )
            metrics_test = dmc_results_test["metrics"]
            sim_results_test = dmc_results_test["sim_results"]
            
            results_dir_test = save_results(
                data_dir=data_dir,
                results_dirname=args.results_dir or "results",
                params=optimized_params,
                metrics=metrics_test,
                sim_results=sim_results_test,
                posterior_samples=None,
                split_tag="test",
            )
            print(f"Saved test results (Double Monte Carlo) to: {results_dir_test}")
    else:
        # Standard single simulation (original behavior)
        if args.double_mc:
            print("Warning: Double Monte Carlo requested but posterior samples not available. Using standard single simulation.")
        
        simulator = BeerGameSimulator(params=optimized_params)
        sim_results_val = simulator.rollout(val_trajectories)

        evaluator = Evaluator()
        metrics_val = evaluator.compute_metrics(sim_results_val, n_samples_wass_mmd=args.n_samples_wass_mmd, mmd_sigma=args.mmd_sigma)

        results_dir_val = save_results(
            data_dir=data_dir,
            results_dirname=args.results_dir or "results",
            params=optimized_params,
            metrics=metrics_val,
            sim_results=sim_results_val,
            posterior_samples=calibrator.posterior_samples,
            split_tag="val",
        )
        print(f"Saved validation results to: {results_dir_val}")

        if len(test_trajectories) > 0:
            sim_results_test = simulator.rollout(test_trajectories)
            metrics_test = evaluator.compute_metrics(sim_results_test, n_samples_wass_mmd=args.n_samples_wass_mmd, mmd_sigma=args.mmd_sigma)
            results_dir_test = save_results(
                data_dir=data_dir,
                results_dirname=args.results_dir or "results",
                params=optimized_params,
                metrics=metrics_test,
                sim_results=sim_results_test,
                posterior_samples=None,
                split_tag="test",
            )
            print(f"Saved test results to: {results_dir_test}")

        # OOD评估：优先使用真实的OOD测试数据，否则使用参数覆盖方式
        ood_lead_time_to_use = None
        if len(test_trajectories_ood) > 0:
            # 使用真实的OOD测试数据（lead_time=5生成的真实数据）
            print("Using real OOD test data for W_out evaluation...")
            sim_results_test_ood = simulator.rollout(test_trajectories_ood)
            metrics_test_ood = evaluator.compute_metrics(sim_results_test_ood, n_samples_wass_mmd=args.n_samples_wass_mmd, mmd_sigma=args.mmd_sigma)
            results_dir_test_ood = save_results(
                data_dir=data_dir,
                results_dirname=args.results_dir or "results",
                params=optimized_params,
                metrics=metrics_test_ood,
                sim_results=sim_results_test_ood,
                posterior_samples=None,
                split_tag="test_ood",
            )
            print(f"Saved OOD test results (using real OOD test data, lead_time=5) to: {results_dir_test_ood}")
            ood_lead_time_to_use = 5  # 记录使用的lead_time
        elif len(test_trajectories) > 0:
            # 回退到参数覆盖方式（改变模拟器参数，但测试数据还是ID的）
            ood_lead_time_to_use = args.ood_lead_time if args.ood_lead_time is not None else 5
            print(f"Warning: No OOD test data found. Using parameter override (lead_time={ood_lead_time_to_use}) with ID test data.")
            params_ood = dict(optimized_params)
            params_ood["lead_time_L"] = int(ood_lead_time_to_use)
            simulator_ood = BeerGameSimulator(params=params_ood)
            sim_results_test_ood = simulator_ood.rollout(test_trajectories)
            metrics_test_ood = evaluator.compute_metrics(sim_results_test_ood, n_samples_wass_mmd=args.n_samples_wass_mmd, mmd_sigma=args.mmd_sigma)
            results_dir_test_ood = save_results(
                data_dir=data_dir,
                results_dirname=args.results_dir or "results",
                params=params_ood,
                metrics=metrics_test_ood,
                sim_results=sim_results_test_ood,
                posterior_samples=None,
                split_tag="test_ood",
            )
            print(f"Saved OOD test results (parameter override, lead_time={ood_lead_time_to_use}) to: {results_dir_test_ood}")

        # 汇总Wasserstein距离：W_in, W_out, W_total
        # 只有在有测试数据（ID或OOD）时才生成汇总
        if len(test_trajectories) > 0 or len(test_trajectories_ood) > 0:
            results_dir = os.path.join(data_dir, args.results_dir or "results")
            try:
                w_in = None
                w_out = None
                
                # 读取In-Distribution的Wasserstein距离（如果有ID测试数据）
                if len(test_trajectories) > 0:
                    with open(os.path.join(results_dir, "metrics_test.json"), "r") as f:
                        metrics_test = json.load(f)
                    w_in = metrics_test.get("distributional", {}).get("Wasserstein_per_t", 0.0)
                
                # 读取Out-of-Distribution的Wasserstein距离（如果有OOD测试数据）
                if ood_lead_time_to_use is not None:
                    with open(os.path.join(results_dir, "metrics_test_ood.json"), "r") as f:
                        metrics_test_ood = json.load(f)
                    w_out = metrics_test_ood.get("distributional", {}).get("Wasserstein_per_t", 0.0)
                
                # 只有当两者都存在时才计算平均值
                if w_in is not None and w_out is not None:
                    w_total = (w_in + w_out) / 2.0
                    
                    # 保存汇总结果（对齐env.py的输出格式）
                    wasserstein_summary = {
                        "W_in": float(w_in),
                        "W_out": float(w_out),
                        "W_total": float(w_total),
                        "method": "real_ood_data" if len(test_trajectories_ood) > 0 else "parameter_override",
                        "description": {
                            "W_in": "In-distribution Wasserstein distance (test set with lead_time=2, same as training)",
                            "W_out": f"Out-of-distribution Wasserstein distance (test set with lead_time={ood_lead_time_to_use})",
                            "W_total": "Average of W_in and W_out"
                        }
                    }
                    summary_path = os.path.join(results_dir, "wasserstein_summary.json")
                    with open(summary_path, "w") as f:
                        json.dump(wasserstein_summary, f, indent=2)
                    
                    print("\n" + "=" * 80)
                    print("📊 Wasserstein距离汇总 (对齐env.py格式):")
                    print("=" * 80)
                    print(f"   W_in (In-Distribution, lead_time=2):  {w_in:.6f}")
                    print(f"   W_out (Out-of-Distribution, lead_time={ood_lead_time_to_use}): {w_out:.6f}")
                    print(f"   W_total (Average):       {w_total:.6f}")
                    print("=" * 80)
                    print(f"✅ 汇总结果已保存到: {summary_path}")
                    print(f"   注意: 这些值可以与 env.py 的 Test Loss['wass'] 对比")
                elif w_in is not None:
                    print(f"\n⚠️  只有ID测试数据，W_in = {w_in:.6f} (需要OOD数据才能计算W_out)")
                elif w_out is not None:
                    print(f"\n⚠️  只有OOD测试数据，W_out = {w_out:.6f} (需要ID数据才能计算W_in)")
            except Exception as e:
                warnings.warn(f"Failed to generate Wasserstein summary: {e}")

        if args.ood_demand_family is not None:
            params_ood_d = dict(optimized_params)
            params_ood_d["demand_family"] = args.ood_demand_family
            if args.ood_demand_params is not None:
                try:
                    override = json.loads(args.ood_demand_params)
                    for k, v in override.items():
                        params_ood_d[k] = v
                except Exception:
                    warnings.warn("Failed to parse --ood_demand_params JSON. Using defaults for family.")
            fam = params_ood_d["demand_family"]
            if fam == "Poisson":
                params_ood_d.setdefault("poisson_lambda", optimized_params.get("poisson_lambda", 5.0))
            elif fam == "NegBin":
                params_ood_d.setdefault("negbin_mu", 5.0)
                params_ood_d.setdefault("negbin_r", 5.0)
            elif fam == "AR1":
                params_ood_d.setdefault("ar1_mu", 5.0)
                params_ood_d.setdefault("ar1_phi", 0.0)
                params_ood_d.setdefault("ar1_sigma", 1.0)
            simulator_ood_d = BeerGameSimulator(params=params_ood_d)
            sim_results_test_ood_d = simulator_ood_d.rollout(test_trajectories)
            metrics_test_ood_d = evaluator.compute_metrics(sim_results_test_ood_d, n_samples_wass_mmd=args.n_samples_wass_mmd, mmd_sigma=args.mmd_sigma)
            results_dir_test_ood_d = save_results(
                data_dir=data_dir,
                results_dirname=args.results_dir or "results",
                params=params_ood_d,
                metrics=metrics_test_ood_d,
                sim_results=sim_results_test_ood_d,
                posterior_samples=None,
                split_tag="test_ood_demand",
            )
            print(f"Saved OOD test results (demand_family={args.ood_demand_family}) to: {results_dir_test_ood_d}")

    if use_double_mc:
        if alpha_report:
            with open(os.path.join(results_dir_base, "alpha_report.json"), "w") as f:
                json.dump(alpha_report, f, indent=2)
        return
    else:
        if alpha_report:
            with open(os.path.join(results_dir_base, "alpha_report.json"), "w") as f:
                json.dump(alpha_report, f, indent=2)


# Execute main for both direct execution and sandbox wrapper invocation
main()