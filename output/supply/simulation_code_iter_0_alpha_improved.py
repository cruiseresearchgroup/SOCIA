import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional dependencies
TORCH_AVAILABLE = False
SBI_AVAILABLE = False
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sbi import utils as sbi_utils
    from sbi.inference import SNPE as NPE
    from sbi.inference import simulate_for_sbi

    SBI_AVAILABLE = True
except Exception:
    SBI_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Environment path handling
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
DATA_PATH = os.environ.get("DATA_PATH")
if PROJECT_ROOT and DATA_PATH:
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
else:
    DATA_DIR = os.getcwd()

# Global RNG
GLOBAL_SEED = 1337
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(GLOBAL_SEED)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def approx_wasserstein_1d(x: np.ndarray, y: np.ndarray, n_quantiles: int = 100) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0
    qs = np.linspace(0.0, 1.0, num=n_quantiles)
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    return float(np.mean(np.abs(qx - qy)))


def compute_mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if x.size == 0 or y.size == 0:
        return 0.0

    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = (a - b.T) ** 2
        return np.exp(-gamma * d)

    xx = rbf(x, x)
    yy = rbf(y, y)
    xy = rbf(x, y)
    mmd2 = xx.mean() + yy.mean() - 2.0 * xy.mean()
    return float(max(0.0, mmd2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


@dataclass
class FittedParams:
    decision_weights: Dict[str, float]
    layer_weights: Dict[str, float]
    info_params: Dict[str, float]
    noise_params: Dict[str, float]
    module_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    engine_type: str = "calibrasim"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ParamsAdapter(ABC):
    @abstractmethod
    def apply(self, simulation: "Simulation", params: FittedParams) -> None:
        raise NotImplementedError

    @abstractmethod
    def capture(self, simulation: "Simulation") -> FittedParams:
        raise NotImplementedError

    @abstractmethod
    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        raise NotImplementedError


class SimpleParamsAdapter(ParamsAdapter):
    def __init__(self, param_definitions_path: Optional[str] = None) -> None:
        self.param_defs: Dict[str, Dict[str, Any]] = {}
        if param_definitions_path and os.path.isfile(param_definitions_path):
            try:
                with open(param_definitions_path, "r") as f:
                    self.param_defs = json.load(f)
            except Exception:
                warnings.warn("Failed to load parameter_definitions.json; proceeding without frozen checks.")

    def apply(self, simulation: "Simulation", params: FittedParams) -> None:
        applied = {}
        mapped = dict(params.module_params)
        mapped["decision"] = params.decision_weights
        mapped["layers"] = params.layer_weights
        mapped["info"] = params.info_params
        mapped["noise"] = params.noise_params

        frozen_warnings = self.validate_frozen(params)
        for module, kv in mapped.items():
            for key, val in kv.items():
                full_key = f"{module}.{key}"
                is_frozen = False
                if full_key in self.param_defs:
                    is_frozen = bool(self.param_defs[full_key].get("frozen", False))
                if is_frozen:
                    warnings.warn(f"Ignoring override for frozen parameter: {full_key}")
                    continue
                simulation.set_params(module, **{key: val})
                applied[full_key] = val

        used_path = os.path.join(simulation.artifacts_dir, "parameters_used.json")
        ensure_dir(simulation.artifacts_dir)
        with open(used_path, "w") as f:
            json.dump(
                {
                    "applied": applied,
                    "frozen_warnings": frozen_warnings,
                    "engine_type": params.engine_type,
                    "meta": params.meta,
                },
                f,
                indent=2,
            )

    def capture(self, simulation: "Simulation") -> FittedParams:
        snap = simulation.param_registry_snapshot()
        fp = FittedParams(
            decision_weights=snap.get("decision", {}),
            layer_weights=snap.get("layers", {}),
            info_params=snap.get("info", {}),
            noise_params=snap.get("noise", {}),
            module_params={k: v for k, v in snap.items() if k not in ("decision", "layers", "info", "noise")},
            meta={"captured_at": time.time()},
        )
        return fp

    def validate_frozen(self, params: FittedParams) -> Dict[str, str]:
        warnings_map: Dict[str, str] = {}
        mapped = dict(params.module_params)
        mapped["decision"] = params.decision_weights
        mapped["layers"] = params.layer_weights
        mapped["info"] = params.info_params
        mapped["noise"] = params.noise_params
        for module, kv in mapped.items():
            for key, _ in kv.items():
                full_key = f"{module}.{key}"
                if full_key in self.param_defs and bool(self.param_defs[full_key].get("frozen", False)):
                    warnings_map[full_key] = "Attempt to override frozen parameter ignored."
        return warnings_map


class Calibrator(ABC):
    @abstractmethod
    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator: "Evaluator",
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        raise NotImplementedError


def evaluate_params(simulator: "Simulation", params: FittedParams, window: Tuple[int, int]) -> Dict[str, Any]:
    adapter = SimpleParamsAdapter(
        param_definitions_path=os.path.join(simulator.artifacts_dir, "parameter_definitions.json")
    )
    adapter.apply(simulator, params)
    start, end = window
    simulator.run(start_day=start, end_day=end, seed=simulator.seed, use_data_actions=False, record_io=False)
    metrics = simulator.evaluate(filename=None)
    result = {
        "RMSE_inventory": float(metrics.get("aggregate", {}).get("RMSE_inventory", 0.0)),
        "RMSE_backlog": float(metrics.get("aggregate", {}).get("RMSE_backlog", 0.0)),
        "MAE_inventory": float(metrics.get("aggregate", {}).get("MAE_inventory", 0.0)),
        "MAE_backlog": float(metrics.get("aggregate", {}).get("MAE_backlog", 0.0)),
        "ApproxWasserstein_inventory": float(metrics.get("aggregate", {}).get("ApproxWasserstein_inventory", 0.0)),
        "ApproxWasserstein_backlog": float(metrics.get("aggregate", {}).get("ApproxWasserstein_backlog", 0.0)),
        "MMD_inventory": float(metrics.get("aggregate", {}).get("MMD_inventory", 0.0)),
        "MMD_backlog": float(metrics.get("aggregate", {}).get("MMD_backlog", 0.0)),
        "RMSE_aggregate": float(metrics.get("aggregate", {}).get("RMSE_inventory", 0.0)),  # backward-compat
        "full_metrics": metrics,
    }
    return result


class LogitHeadCalibrator(Calibrator):
    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = float(l2)

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator: "Evaluator",
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        set_all_seeds(seed)
        if artifacts_dir is None:
            artifacts_dir = os.path.join(simulator.artifacts_dir, "calibration", "logit_head")
        ensure_dir(artifacts_dir)

        adapter = params_adapter or SimpleParamsAdapter()
        fitted = adapter.capture(simulator)
        fitted.meta.update(
            {
                "calibrator": "logit_head",
                "note": "No micro-transition data; returned current parameters.",
                "seed": seed,
            }
        )
        best_dir = os.path.join(artifacts_dir, "best")
        ensure_dir(best_dir)
        with open(os.path.join(best_dir, "fitted_params.json"), "w") as f:
            json.dump(fitted.to_dict(), f, indent=2)
        return fitted


class RandomSearchCalibrator(Calibrator):
    def __init__(self, search_space: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.search_space = search_space or {
            "policy.s_base": (5.0, 30.0),
            "policy.beta": (0.01, 0.9),
            "supply.lead_time_L": (1.0, 8.0),
            "supply.lead_time_jitter_std": (0.0, 3.0),
            "demand.poisson_lambda": (1.0, 20.0),
        }

    def _sample_params(self, adapter: ParamsAdapter, simulator: "Simulation") -> FittedParams:
        module_params: Dict[str, Dict[str, float]] = {}
        for full_key, (lo, hi) in self.search_space.items():
            mod, key = full_key.split(".", 1)
            val = float(np.random.uniform(lo, hi))
            if mod not in module_params:
                module_params[mod] = {}
            if "lead_time_L" in key:
                val = float(int(round(val)))
            module_params[mod][key] = val
        fp = FittedParams(
            decision_weights={},
            layer_weights={},
            info_params={},
            noise_params={},
            module_params=module_params,
            meta={"sampled": True},
        )
        return fp

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator: "Evaluator",
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        set_all_seeds(seed)
        if artifacts_dir is None:
            artifacts_dir = os.path.join(simulator.artifacts_dir, "calibration", "random_search")
        ensure_dir(artifacts_dir)
        adapter = params_adapter or SimpleParamsAdapter()
        best_score = float("inf")
        best_params: Optional[FittedParams] = None
        trials = []

        for i in range(max(1, budget)):
            fp = self._sample_params(adapter, simulator)
            trial_dir = os.path.join(artifacts_dir, f"trial_{i}")
            ensure_dir(trial_dir)
            with open(os.path.join(trial_dir, "params_applied.json"), "w") as f:
                json.dump(fp.to_dict(), f, indent=2)
            try:
                res = evaluate_params(simulator, fp, train_window)
            except Exception as exc:
                traceback.print_exc()
                res = {"RMSE_inventory": float("inf"), "error": str(exc), "traceback": traceback.format_exc()}
            with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
                json.dump(res, f, indent=2)
            score = float(res.get("RMSE_inventory", float("inf")))
            trials.append({"trial": i, "score": score})
            if score < best_score:
                best_score = score
                best_params = fp

        report = {"best_score": best_score, "trials": trials, "budget": budget}
        with open(os.path.join(artifacts_dir, "calibration_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        if best_params is None:
            best_params = adapter.capture(simulator)
        best_dir = os.path.join(artifacts_dir, "best")
        ensure_dir(best_dir)
        with open(os.path.join(best_dir, "fitted_params.json"), "w") as f:
            json.dump(best_params.to_dict(), f, indent=2)
        return best_params


class SNPECalibrator(Calibrator):
    def __init__(self, num_simulations: int = 200, num_posterior_samples: int = 512, timeout: int = 60) -> None:
        self.num_simulations = int(num_simulations)
        self.num_posterior_samples = int(num_posterior_samples)
        self.timeout = int(timeout)

    def fit(
        self,
        bundle: Dict[str, Any],
        simulator: "Simulation",
        evaluator: "Evaluator",
        train_window: Tuple[int, int],
        seed: int,
        budget: int = 100,
        artifacts_dir: Optional[str] = None,
        params_adapter: Optional[ParamsAdapter] = None,
    ) -> FittedParams:
        if not (TORCH_AVAILABLE and SBI_AVAILABLE):
            warnings.warn("SBI/torch unavailable; falling back to RandomSearchCalibrator.")
            return RandomSearchCalibrator().fit(
                bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter
            )

        set_all_seeds(seed)
        if artifacts_dir is None:
            artifacts_dir = os.path.join(simulator.artifacts_dir, "calibration", "snpe")
        ensure_dir(artifacts_dir)

        space = {
            "supply.lead_time_L": (1.0, 8.0),
            "supply.lead_time_jitter_std": (0.0, 3.0),
            "demand.poisson_lambda": (1.0, 20.0),
            "demand.negbin_mu": (1.0, 20.0),
            "demand.negbin_r": (0.5, 20.0),
            "demand.ar1_mu": (1.0, 20.0),
            "demand.ar1_phi": (-0.9, 0.9),
            "demand.ar1_sigma": (0.1, 10.0),
            "policy.beta": (0.01, 0.9),
            "policy.s_base": (5.0, 30.0),
            "demand_family_code": (0.0, 2.0),  # 0: Poisson, 1: NegBin, 2: AR1
        }
        keys = list(space.keys())
        low = torch.tensor([space[k][0] for k in keys], dtype=torch.float32)
        high = torch.tensor([space[k][1] for k in keys], dtype=torch.float32)
        prior = sbi_utils.BoxUniform(low=low, high=high)

        start, end = train_window

        def _safe_var(arr: np.ndarray) -> float:
            arr = np.asarray(arr, dtype=float)
            if arr.size <= 1:
                return 0.0
            return float(np.var(arr))

        def _lag_corr(a: np.ndarray, b: np.ndarray, lag: int = 1) -> float:
            a = np.asarray(a, dtype=float).ravel()
            b = np.asarray(b, dtype=float).ravel()
            if len(a) <= lag or len(b) <= lag:
                return 0.0
            a_l = a[lag:]
            b_l = b[:-lag]
            if a_l.size == 0 or b_l.size == 0:
                return 0.0
            if np.std(a_l) == 0 or np.std(b_l) == 0:
                return 0.0
            return float(np.corrcoef(a_l, b_l)[0, 1])

        def _summarize(obs_results: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
            inv = np.concatenate([v for v in obs_results.get("inventory", {}).values()]) if obs_results.get("inventory") else np.array([0.0])
            bklg = np.concatenate([v for v in obs_results.get("backlog", {}).values()]) if obs_results.get("backlog") else np.array([0.0])
            ship = np.concatenate([v for v in obs_results.get("shipments", {}).values()]) if obs_results.get("shipments") else np.array([0.0])
            orders_retailer = obs_results.get("orders", {}).get("retailer", np.array([0.0]))
            demand_retailer = obs_results.get("demand", {}).get("retailer", np.array([0.0]))
            # fallbacks
            if demand_retailer.size == 0 or np.all(demand_retailer == 0):
                demand_retailer = obs_results.get("shipments", {}).get("retailer", np.array([0.0]))
            feats = [
                float(np.mean(inv)),
                float(np.std(inv)),
                float(np.mean(bklg)),
                float(np.std(bklg)),
                float(np.mean(ship)),
                float(np.std(ship)),
            ]
            var_o = _safe_var(orders_retailer)
            var_d = _safe_var(demand_retailer)
            bullwhip = float(var_o / var_d) if var_d > 0 else 0.0
            feats.append(bullwhip)
            feats.append(_lag_corr(orders_retailer, demand_retailer, lag=1))
            feats.append(float(np.corrcoef(inv[: min(len(inv), len(demand_retailer))], demand_retailer[: min(len(inv), len(demand_retailer))])[0, 1]) if len(inv) > 1 and len(demand_retailer) > 1 and np.std(inv) > 0 and np.std(demand_retailer) > 0 else 0.0)
            return np.array(feats, dtype=np.float32)

        def sim_wrapper(theta: torch.Tensor) -> torch.Tensor:
            theta_np = theta.detach().cpu().numpy().astype(float)
            # Map theta to sim params
            local_params: Dict[str, Dict[str, float]] = {}
            for idx, full_key in enumerate(keys):
                val = float(theta_np[idx])
                if full_key == "supply.lead_time_L":
                    val = float(int(round(val)))
                if "." in full_key:
                    mod, k = full_key.split(".", 1)
                    local_params.setdefault(mod, {})[k] = val
                else:
                    local_params.setdefault("global", {})[full_key] = val
            # Handle demand family switch
            fam_code = int(round(local_params.get("global", {}).get("demand_family_code", 0.0)))
            fam_code = max(0, min(2, fam_code))
            fams = {0: "Poisson", 1: "NegBin", 2: "AR1"}
            demand_family = fams[fam_code]
            simulator.set_params("global", demand_family=demand_family)
            # Apply other params
            for mod, kv in local_params.items():
                for k, v in kv.items():
                    if k == "demand_family_code":
                        continue
                    simulator.set_params(mod, **{k: v})

            try:
                simulator.run(start_day=start, end_day=end, seed=simulator.seed, use_data_actions=False, record_io=False)
                sim_obs = simulator.get_observables(window=(start, end))
                feats = _summarize(sim_obs)
            except Exception:
                feats = np.full((10,), np.nan, dtype=np.float32)
            return torch.tensor(feats, dtype=torch.float32)

        theta, x = simulate_for_sbi(sim_wrapper, prior, num_simulations=self.num_simulations)
        nan_mask = torch.any(~torch.isfinite(x), dim=1)
        theta = theta[~nan_mask]
        x = x[~nan_mask]
        if theta.shape[0] == 0:
            warnings.warn("[SNPE] All simulations invalid; falling back to RandomSearch.")
            return RandomSearchCalibrator().fit(
                bundle, simulator, evaluator, train_window, seed, budget, artifacts_dir, params_adapter
            )

        inf = NPE(prior=prior)
        density_estimator = inf.append_simulations(theta, x).train()
        posterior = inf.build_posterior(density_estimator)

        # Observed summary from observed dataset in simulator
        s, e = start, end
        observed_slice: Dict[str, Dict[str, np.ndarray]] = {"inventory": {}, "backlog": {}, "shipments": {}, "orders": {}, "demand": {}}
        for m in ["inventory", "backlog", "shipments", "orders", "demand"]:
            observed_slice[m] = {}
            src = simulator.observed.get(m, {})
            for echelon, arr in src.items():
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    observed_slice[m][echelon] = arr[s : e + 1]
        x_obs = torch.tensor(_summarize(observed_slice), dtype=torch.float32)

        samples_list = []
        start_time = time.time()
        while len(samples_list) < self.num_posterior_samples:
            if (time.time() - start_time) > self.timeout:
                warnings.warn("Posterior sampling timeout; using collected samples.")
                break
            try:
                s = posterior.sample((min(256, self.num_posterior_samples - len(samples_list)),), x=x_obs)
                samples_list.append(s.detach().cpu())
            except Exception:
                break

        if len(samples_list) == 0:
            samples = (low + (high - low) / 2.0).unsqueeze(0)
        else:
            samples = torch.cat(samples_list, dim=0)
        theta_hat = samples.mean(dim=0).detach().cpu().numpy().astype(float)

        # persist posterior samples
        samples_path = os.path.join(artifacts_dir, "posterior_samples.npy")
        np.save(samples_path, samples.detach().cpu().numpy())

        fp = FittedParams(
            decision_weights={},
            layer_weights={},
            info_params={},
            noise_params={},
            module_params={},
            meta={"calibrator": "snpe", "posterior_samples_path": samples_path},
        )
        for idx, full_key in enumerate(keys):
            mod, k = (full_key.split(".", 1) if "." in full_key else ("global", full_key))
            if mod not in fp.module_params:
                fp.module_params[mod] = {}
            val = float(theta_hat[idx])
            if k == "lead_time_L":
                val = float(int(round(val)))
            if k == "demand_family_code":
                fam_code = int(round(val))
                fam_code = max(0, min(2, fam_code))
                fams = {0: "Poisson", 1: "NegBin", 2: "AR1"}
                fp.module_params.setdefault("global", {})
                fp.module_params["global"]["demand_family"] = fams[fam_code]
            else:
                fp.module_params[mod][k] = val

        best_dir = os.path.join(artifacts_dir, "best")
        ensure_dir(best_dir)
        with open(os.path.join(best_dir, "fitted_params.json"), "w") as f:
            json.dump(fp.to_dict(), f, indent=2)
        return fp


CALIBRATOR_REGISTRY = {
    "logit_head": LogitHeadCalibrator,
    "random_search": RandomSearchCalibrator,
    "snpe": SNPECalibrator,
}


def get_calibrator(name: str, config_path: Optional[str] = None) -> Calibrator:
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


def infer_time_column(df: pd.DataFrame) -> str:
    for c in ["t", "time", "time_step", "step", "period_index", "week"]:
        if c in df.columns:
            return c
    raise ValueError("Time column not found; expected one of: t, time, time_step, step, period_index, week.")


def load_dataset(data_dir: str, fname: str) -> pd.DataFrame:
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if "action" not in df.columns and "orders" in df.columns:
        df = df.rename(columns={"orders": "action"})
    if "week" in df.columns and not any(c in df.columns for c in ["t", "time", "time_step", "step", "period_index"]):
        df = df.rename(columns={"week": "t"})
    if "trajectory_id" not in df.columns:
        df["trajectory_id"] = 0
    if "echelon" in df.columns:
        df["trajectory_id"] = df["trajectory_id"].astype(str) + ":" + df["echelon"].astype(str)
    required = ["trajectory_id", "inventory", "backlog"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


@dataclass
class Item:
    qty: int
    remaining: int


class Module(ABC):
    @abstractmethod
    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        raise NotImplementedError


class DemandModule(Module):
    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        rng: np.random.Generator = state["_rng"]
        fam = params.get("demand_family", "Poisson")
        if fam == "Poisson":
            lam = float(params.get("demand.poisson_lambda", params.get("poisson_lambda", 5.0)))
            demand = int(rng.poisson(lam=max(0.0, lam)))
        elif fam == "NegBin":
            mu = float(params.get("demand.negbin_mu", 5.0))
            r = float(params.get("demand.negbin_r", 5.0))
            scale = mu / max(r, 1e-9)
            lam = rng.gamma(shape=max(1e-6, r), scale=scale)
            demand = int(rng.poisson(lam=max(0.0, lam)))
        elif fam == "AR1":
            mu = float(params.get("demand.ar1_mu", 5.0))
            phi = float(params.get("demand.ar1_phi", 0.0))
            sigma = float(params.get("demand.ar1_sigma", 1.0))
            x_prev = float(state.setdefault("_ar1_x_prev", mu))
            eps = float(rng.normal(0.0, 1.0))
            x_t = mu + phi * (x_prev - mu) + sigma * eps
            state["_ar1_x_prev"] = x_t
            demand = int(max(0, round(x_t)))
        else:
            demand = 0
        buffers["demand_t"] = demand


class InventoryFlowModule(Module):
    def __init__(self, arrival_convention: str = "deliver_at_remaining_0") -> None:
        self.arrival_convention = arrival_convention

    def _deliveries_and_decrement(self, pipeline: List[Item]) -> Tuple[int, List[Item]]:
        delivered = 0
        if self.arrival_convention == "deliver_at_remaining_1":
            remaining_items: List[Item] = []
            for it in pipeline:
                if it.remaining <= 1:
                    delivered += it.qty
                else:
                    remaining_items.append(it)
            new_pipeline: List[Item] = []
            for it in remaining_items:
                new_rem = max(1, it.remaining - 1)
                new_pipeline.append(Item(qty=it.qty, remaining=new_rem))
            return delivered, new_pipeline
        dec: List[Item] = []
        for it in pipeline:
            new_rem = max(0, it.remaining - 1)
            dec.append(Item(qty=it.qty, remaining=new_rem))
        keep: List[Item] = []
        for it in dec:
            if it.remaining == 0:
                delivered += it.qty
            else:
                keep.append(it)
        return delivered, keep

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        echelons = state["echelons"]
        inv = state["inventory"]
        backlog = state["backlog"]
        pipelines: Dict[str, List[Item]] = state["pipeline"]
        shipments_out: Dict[str, int] = {}
        shipments_in: Dict[str, int] = {}

        for e in echelons:
            delivered, new_pipe = self._deliveries_and_decrement(pipelines[e])
            shipments_in[e] = int(delivered)
            pipelines[e] = new_pipe

        for e in echelons:
            inv[e] = int(inv[e] + shipments_in[e])

        demand_t = int(buffers.get("demand_t", 0))
        for e_idx, e in enumerate(echelons):
            if e_idx == 0:
                downstream_demand = demand_t
            else:
                downstream = echelons[e_idx - 1]
                downstream_demand = int(buffers.get("orders_placed", {}).get(downstream, 0))

            served_b = min(inv[e], backlog[e])
            inv[e] -= served_b
            backlog[e] -= served_b

            served_d = min(inv[e], max(0, downstream_demand))
            inv[e] -= served_d
            unmet = max(0, downstream_demand) - served_d
            backlog[e] += int(unmet)

            shipments_out[e] = int(served_d)

        buffers["shipments_out"] = shipments_out
        buffers["shipments_in"] = shipments_in
        buffers["next_inventory"] = dict(inv)
        buffers["next_backlog"] = dict(backlog)
        buffers["next_pipeline"] = dict(pipelines)


class OrderPolicyModule(Module):
    def __init__(self, use_data_actions: bool = False) -> None:
        self.use_data_actions = bool(use_data_actions)

    @staticmethod
    def _apply_alpha_noise(val: float, alpha: float, rng: np.random.Generator) -> Tuple[float, float]:
        if alpha <= 0.0:
            return val, 0.0
        noise = float(rng.normal(0.0, alpha))
        noisy = max(0.0, val * (1.0 + noise))
        return noisy, noise

    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        echelons = state["echelons"]
        orders: Dict[str, int] = {}
        alpha_enabled = bool(params.get("noise.alpha_enabled", False))
        alpha_inv = float(params.get("noise.alpha_inv", 0.0))
        alpha_bklg = float(params.get("noise.alpha_bklg", 0.0))
        rng: np.random.Generator = state["_rng"]
        alpha_logs: Dict[str, Dict[str, float]] = {}

        if self.use_data_actions and "data_actions" in state:
            for e in echelons[:-1]:
                series = state["data_actions"].get(e, [])
                val = int(series[t]) if t < len(series) else 0
                orders[e] = max(0, val)
        else:
            beta = float(params.get("policy.beta", 0.5))
            s_base = float(params.get("policy.s_base", 15.0))
            for e in echelons[:-1]:
                inv = float(state["inventory"][e])
                bklg = float(state["backlog"][e])

                inv_used = inv
                bklg_used = bklg
                inv_noise = 0.0
                bklg_noise = 0.0
                if alpha_enabled:
                    inv_used, inv_noise = self._apply_alpha_noise(inv, alpha_inv, rng)
                    bklg_used, bklg_noise = self._apply_alpha_noise(bklg, alpha_bklg, rng)
                alpha_logs[e] = {"inv_noise": inv_noise, "bklg_noise": bklg_noise}

                target = s_base + bklg_used
                last_order = state["orders_last"].get(e, 0)
                raw = target - inv_used
                smoothed = (1.0 - beta) * last_order + beta * raw
                val = int(max(0, round(smoothed)))
                orders[e] = val

        if alpha_logs:
            buffers["alpha_noise"] = alpha_logs
        buffers["orders_placed"] = orders


class PipelineAppendModule(Module):
    def forward(self, state: Dict[str, Any], buffers: Dict[str, Any], params: Dict[str, Any], t: int) -> None:
        rng: np.random.Generator = state["_rng"]
        echelons = state["echelons"]
        lead_time = int(params.get("supply.lead_time_L", params.get("lead_time_L", 2)))
        jitter = float(params.get("supply.lead_time_jitter_std", params.get("lead_time_jitter_std", 0.0)))
        pipelines: Dict[str, List[Item]] = state["pipeline"]
        orders = buffers.get("orders_placed", {})

        def jittered(L: int) -> int:
            if jitter <= 0.0:
                return int(max(0, L))
            draw = int(max(0, round(rng.normal(L, jitter))))
            return int(draw)

        for e_idx, e in enumerate(echelons[:-1]):
            upstream = echelons[e_idx + 1]
            q = int(max(0, orders.get(e, 0)))
            if lead_time == 0:
                state["inventory"][upstream] += q
            else:
                rem = jittered(lead_time)
                if q > 0:
                    pipelines[upstream].append(Item(qty=q, remaining=rem))
        buffers["next_pipeline"] = dict(pipelines)


class Simulation:
    def __init__(self, artifacts_dir: str, seed: int = GLOBAL_SEED) -> None:
        self.artifacts_dir = artifacts_dir
        self.seed = int(seed)
        self.modules: List[Module] = []
        self.params: Dict[str, float] = {
            "demand_family": "Poisson",
            "demand.poisson_lambda": 8.0,
            "policy.beta": 0.5,
            "policy.s_base": 15.0,
            "supply.lead_time_L": 2.0,
            "supply.lead_time_jitter_std": 0.0,
            "arrival_convention": "deliver_at_remaining_0",
            "noise.alpha_enabled": False,
            "noise.alpha_inv": 0.0,
            "noise.alpha_bklg": 0.0,
        }
        self.state: Dict[str, Any] = {
            "echelons": ["retailer", "wholesaler", "distributor", "factory"],
            "inventory": {"retailer": 10, "wholesaler": 20, "distributor": 30, "factory": 40},
            "backlog": {"retailer": 0, "wholesaler": 0, "distributor": 0, "factory": 0},
            "pipeline": {"retailer": [], "wholesaler": [], "distributor": [], "factory": []},
            "orders_last": {"retailer": 0, "wholesaler": 0, "distributor": 0, "factory": 0},
            "_rng": np.random.default_rng(self.seed),
        }
        self.time_horizon = 60
        self.buffers: Dict[str, Any] = {}
        self.observed: Dict[str, Dict[str, np.ndarray]] = {
            "inventory": {},
            "backlog": {},
            "shipments": {},
            "orders": {},
            "demand": {},
        }
        self.module_io: Dict[str, List[Dict[str, Any]]] = {}
        self.results: Dict[str, Any] = {}
        self._build_modules()

    def _build_modules(self) -> None:
        self.modules = [
            DemandModule(),
            InventoryFlowModule(arrival_convention=str(self.params.get("arrival_convention", "deliver_at_remaining_0"))),
            OrderPolicyModule(use_data_actions=False),
            PipelineAppendModule(),
        ]

    def set_params(self, module: str, **kwargs: float) -> None:
        rebuild_required = False
        for k, v in kwargs.items():
            full_key = f"{module}.{k}" if "." not in module and "." not in k else (k if "." in k else f"{module}.{k}")
            self.params[full_key] = v
            if k == "arrival_convention":
                self.params["arrival_convention"] = v
                rebuild_required = True
        if "demand_family" in kwargs:
            self.params["demand_family"] = kwargs["demand_family"]
            rebuild_required = True
        if rebuild_required:
            self._build_modules()

    def param_registry_snapshot(self) -> Dict[str, Dict[str, float]]:
        snap: Dict[str, Dict[str, float]] = {}
        for k, v in self.params.items():
            if "." in k:
                mod, key = k.split(".", 1)
                if mod not in snap:
                    snap[mod] = {}
                snap[mod][key] = v
            else:
                snap.setdefault("global", {})[k] = v
        return snap

    def load_observed(self, df: pd.DataFrame) -> None:
        t_col = infer_time_column(df)
        echelons = list(self.state["echelons"])
        metrics = ["inventory", "backlog", "shipments", "action", "demand"]
        obs_map: Dict[str, Dict[str, List[Tuple[int, float]]]] = {m: {e: [] for e in echelons} for m in metrics}
        if "echelon" not in df.columns:
            warnings.warn("No 'echelon' column found; assuming all rows pertain to 'retailer'.")
            df = df.copy()
            df["echelon"] = "retailer"
        if "shipments" not in df.columns:
            df = df.copy()
            df["shipments"] = np.nan
        if "demand" not in df.columns:
            df = df.copy()
            df["demand"] = np.nan

        for _, row in df.iterrows():
            e = str(row["echelon"])
            if e not in echelons:
                continue
            t = int(row[t_col])
            if "inventory" in df.columns:
                obs_map["inventory"][e].append((t, float(row["inventory"])))
            if "backlog" in df.columns:
                obs_map["backlog"][e].append((t, float(row["backlog"])))
            if "shipments" in df.columns and not (isinstance(row["shipments"], float) and np.isnan(row["shipments"])):
                obs_map["shipments"][e].append((t, float(row["shipments"])))
            if "action" in df.columns:
                obs_map["action"][e].append((t, float(row["action"])))
            if "demand" in df.columns and not (isinstance(row["demand"], float) and np.isnan(row["demand"])):
                obs_map["demand"][e].append((t, float(row["demand"])))

        for m in ["inventory", "backlog", "shipments", "demand"]:
            self.observed[m] = {}
            for e in echelons:
                pairs = sorted(obs_map[m][e], key=lambda x: x[0])
                if pairs:
                    ts = [p[0] for p in pairs]
                    vs = [p[1] for p in pairs]
                    self.time_horizon = max(self.time_horizon, max(ts) + 1)
                    arr = np.zeros(self.time_horizon, dtype=float)
                    for t, val in pairs:
                        if 0 <= t < self.time_horizon:
                            arr[t] = val
                    self.observed[m][e] = arr
                else:
                    self.observed[m][e] = np.zeros(self.time_horizon, dtype=float)

        data_actions: Dict[str, List[int]] = {}
        for e in echelons:
            pairs = sorted(obs_map["action"][e], key=lambda x: x[0])
            arr = np.zeros(self.time_horizon, dtype=float)
            for t, val in pairs:
                if 0 <= t < self.time_horizon:
                    arr[t] = val
            data_actions[e] = [int(x) for x in arr.tolist()]
        self.state["data_actions"] = data_actions

    def run(
        self,
        start_day: int,
        end_day: int,
        seed: Optional[int] = None,
        use_data_actions: bool = False,
        record_io: bool = True,
    ) -> None:
        if seed is None:
            seed = self.seed
        self.state["_rng"] = np.random.default_rng(seed)
        self.state["inventory"] = {e: int(self.state["inventory"][e]) for e in self.state["echelons"]}
        self.state["backlog"] = {e: int(self.state["backlog"][e]) for e in self.state["echelons"]}
        self.state["pipeline"] = {e: list(self.state["pipeline"][e]) for e in self.state["echelons"]}
        self.state["orders_last"] = {e: 0 for e in self.state["echelons"]}
        for m in self.modules:
            if isinstance(m, OrderPolicyModule):
                m.use_data_actions = use_data_actions

        inv_rec: Dict[str, List[float]] = {e: [] for e in self.state["echelons"]}
        bklg_rec: Dict[str, List[float]] = {e: [] for e in self.state["echelons"]}
        ship_rec: Dict[str, List[float]] = {e: [] for e in self.state["echelons"]}
        orders_rec: Dict[str, List[float]] = {e: [] for e in self.state["echelons"]}
        demand_rec: Dict[str, List[float]] = {e: [] for e in self.state["echelons"]}
        alpha_noise_abs_series: List[float] = []

        self.module_io = {}

        for t in range(start_day, end_day + 1):
            self.buffers = {}
            for m in self.modules:
                m_name = m.__class__.__name__
                try:
                    m.forward(self.state, self.buffers, self.params, t)
                except Exception as exc:
                    traceback.print_exc()
                    raise RuntimeError(f"Module {m_name} failed at t={t}: {exc}")
                if record_io:
                    io_snapshot = {"t": t, "buffers": dict(self.buffers)}
                    self.module_io.setdefault(m_name, []).append(io_snapshot)

            if "next_inventory" in self.buffers:
                self.state["inventory"] = dict(self.buffers["next_inventory"])
            if "next_backlog" in self.buffers:
                self.state["backlog"] = dict(self.buffers["next_backlog"])
            if "next_pipeline" in self.buffers:
                self.state["pipeline"] = dict(self.buffers["next_pipeline"])
            if "orders_placed" in self.buffers:
                self.state["orders_last"] = dict(self.buffers["orders_placed"])

            for e in self.state["echelons"]:
                inv_rec[e].append(float(self.state["inventory"][e]))
                bklg_rec[e].append(float(self.state["backlog"][e]))
                shp = 0.0
                if "shipments_in" in self.buffers:
                    shp = float(self.buffers["shipments_in"].get(e, 0))
                ship_rec[e].append(shp)
                orders_rec[e].append(float(self.state["orders_last"].get(e, 0)))
            # demand only at retailer recorded
            demand_rec["retailer"].append(float(self.buffers.get("demand_t", 0)))
            for e in [x for x in self.state["echelons"] if x != "retailer"]:
                demand_rec[e].append(0.0)
            # alpha noise stats
            if "alpha_noise" in self.buffers:
                noises = self.buffers["alpha_noise"]
                vals = []
                for e, d in noises.items():
                    vals += [abs(float(d.get("inv_noise", 0.0))), abs(float(d.get("bklg_noise", 0.0)))]
                if vals:
                    alpha_noise_abs_series.append(float(np.mean(vals)))

        self.results["inventory"] = {e: np.array(v, dtype=float) for e, v in inv_rec.items()}
        self.results["backlog"] = {e: np.array(v, dtype=float) for e, v in bklg_rec.items()}
        self.results["shipments"] = {e: np.array(v, dtype=float) for e, v in ship_rec.items()}
        self.results["orders"] = {e: np.array(v, dtype=float) for e, v in orders_rec.items()}
        self.results["demand"] = {e: np.array(v, dtype=float) for e, v in demand_rec.items()}
        self.results["alpha_noise_abs_mean"] = float(np.mean(alpha_noise_abs_series)) if alpha_noise_abs_series else 0.0
        self.results["alpha_noise_abs_series"] = alpha_noise_abs_series

    def get_observables(self, window: Optional[Tuple[int, int]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        if not self.results:
            return {"inventory": {}, "backlog": {}, "shipments": {}, "orders": {}, "demand": {}}
        if window is None:
            return self.results
        s, e = window
        obs: Dict[str, Dict[str, np.ndarray]] = {}
        for m in ["inventory", "backlog", "shipments", "orders", "demand"]:
            obs[m] = {}
            for echelon, arr in self.results.get(m, {}).items():
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    obs[m][echelon] = arr[s : e + 1]
                else:
                    obs[m][echelon] = arr
        return obs

    def save_results(self, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        serializable = {
            m: {e: v.tolist() for e, v in self.results.get(m, {}).items()}
            for m in ["inventory", "backlog", "shipments", "orders", "demand"]
        }
        serializable["alpha_noise_abs_mean"] = self.results.get("alpha_noise_abs_mean", 0.0)
        serializable["alpha_noise_abs_series"] = self.results.get("alpha_noise_abs_series", [])
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def save_module_io(self, module: str, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(self.module_io.get(module, []), f, indent=2)

    def save_all_io(self, root_dir: str) -> None:
        ensure_dir(root_dir)
        for m in self.module_io.keys():
            self.save_module_io(m, os.path.join(root_dir, f"{m}.json"))

    def _build_alpha_report_dict(self) -> Dict[str, Any]:
        alpha_enabled = bool(self.params.get("noise.alpha_enabled", False))
        alpha_inv = float(self.params.get("noise.alpha_inv", 0.0))
        alpha_bklg = float(self.params.get("noise.alpha_bklg", 0.0))
        lt = int(float(self.params.get("supply.lead_time_L", 0.0)))
        lt_jitter = float(self.params.get("supply.lead_time_jitter_std", 0.0))
        report = {
            "alpha": {
                "enabled": alpha_enabled,
                "applies_to": ["policy.perception.inventory", "policy.perception.backlog"] if alpha_enabled else [],
                "excluded_parameters": [
                    "supply.lead_time_L",
                    "supply.lead_time_jitter_std",
                    "policy.beta",
                    "policy.s_base",
                    "demand_family",
                ],
                "scales": {
                    "inventory": alpha_inv,
                    "backlog": alpha_bklg,
                },
                "statistics": {
                    "abs_mean": float(self.results.get("alpha_noise_abs_mean", 0.0)),
                    "abs_series": list(self.results.get("alpha_noise_abs_series", [])),
                },
                "description": "Alpha noise is applied multiplicatively to perceived inventory and backlog only, not to structural/discrete parameters.",
            },
            "supply": {
                "lead_time_L": lt,
                "lead_time_jitter_std": lt_jitter,
            },
            "timestamp": time.time(),
        }
        return report

    def write_alpha_report(self, path: Optional[str] = None) -> Dict[str, Any]:
        report = self._build_alpha_report_dict()
        if path is None:
            path = os.path.join(self.artifacts_dir, "alpha_report.json")
        ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    def visualize(self, window: Optional[Tuple[int, int]] = None) -> None:
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available; skipping visualization.")
            return
        obs = self.get_observables(window)
        echelons = self.state["echelons"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for e in echelons:
            axes[0].plot(obs["inventory"].get(e, []), label=e)
            axes[1].plot(obs["backlog"].get(e, []), label=e)
            axes[2].plot(obs["shipments"].get(e, []), label=e)
        axes[0].set_ylabel("Inventory")
        axes[1].set_ylabel("Backlog")
        axes[2].set_ylabel("Shipments")
        axes[2].set_xlabel("Time")
        for ax in axes:
            ax.legend()
        plt.tight_layout()
        plt.show()

    def evaluate(self, filename: Optional[str] = None) -> Dict[str, Any]:
        evaluator = Evaluator()
        metrics = evaluator.compute_metrics(self.results, self.observed)
        if filename is None:
            filename = os.path.join(self.artifacts_dir, "results", "metrics.json")
        ensure_dir(os.path.dirname(filename))
        with open(filename, "w") as f:
            json.dump(metrics, f, indent=2)
        # Write alpha report artifact alongside metrics
        try:
            self.write_alpha_report(os.path.join(self.artifacts_dir, "alpha_report.json"))
        except Exception:
            warnings.warn("Failed to write alpha_report.json")
        return metrics


class Evaluator:
    def _flatten(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        if not d:
            return np.array([], dtype=float)
        arrs = [v for v in d.values() if isinstance(v, np.ndarray) and v.size > 0]
        return np.concatenate(arrs, axis=0) if len(arrs) > 0 else np.array([], dtype=float)

    def compute_metrics(
        self,
        sim: Dict[str, Dict[str, np.ndarray]],
        obs: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        per_echelon: Dict[str, Dict[str, float]] = {}
        echelons = sorted(set(list(sim.get("inventory", {}).keys()) + list(obs.get("inventory", {}).keys())))

        for e in echelons:
            inv_s = sim.get("inventory", {}).get(e, np.array([], dtype=float))
            inv_o = obs.get("inventory", {}).get(e, np.array([], dtype=float))
            b_s = sim.get("backlog", {}).get(e, np.array([], dtype=float))
            b_o = obs.get("backlog", {}).get(e, np.array([], dtype=float))
            sh_s = sim.get("shipments", {}).get(e, np.array([], dtype=float))
            sh_o = obs.get("shipments", {}).get(e, np.array([], dtype=float))

            T = min(inv_s.size, inv_o.size) if inv_o.size > 0 else inv_s.size
            Ti = T
            if T > 0:
                inv_s = inv_s[:T]
                inv_o = inv_o[:T]
            T = min(b_s.size, b_o.size) if b_o.size > 0 else b_s.size
            Tb = T
            if T > 0:
                b_s = b_s[:T]
                b_o = b_o[:T]
            T = min(sh_s.size, sh_o.size) if sh_o.size > 0 else sh_s.size
            Ts = T
            if T > 0:
                sh_s = sh_s[:T]
                sh_o = sh_o[:T]
            per_echelon[e] = {
                "RMSE_inventory": rmse(inv_s, inv_o) if Ti > 0 else 0.0,
                "RMSE_backlog": rmse(b_s, b_o) if Tb > 0 else 0.0,
                "RMSE_shipments": rmse(sh_s, sh_o) if Ts > 0 else 0.0,
                "MAE_inventory": mae(inv_s, inv_o) if Ti > 0 else 0.0,
                "MAE_backlog": mae(b_s, b_o) if Tb > 0 else 0.0,
                "MAE_shipments": mae(sh_s, sh_o) if Ts > 0 else 0.0,
            }

        inv_s_flat = self._flatten(sim.get("inventory", {}))
        inv_o_flat = self._flatten(obs.get("inventory", {}))
        b_s_flat = self._flatten(sim.get("backlog", {}))
        b_o_flat = self._flatten(obs.get("backlog", {}))
        sh_s_flat = self._flatten(sim.get("shipments", {}))
        sh_o_flat = self._flatten(obs.get("shipments", {}))

        agg = {
            "RMSE_inventory": float(np.mean([v["RMSE_inventory"] for v in per_echelon.values()])) if per_echelon else 0.0,
            "RMSE_backlog": float(np.mean([v["RMSE_backlog"] for v in per_echelon.values()])) if per_echelon else 0.0,
            "RMSE_shipments": float(np.mean([v["RMSE_shipments"] for v in per_echelon.values()])) if per_echelon else 0.0,
            "MAE_inventory": float(np.mean([v["MAE_inventory"] for v in per_echelon.values()])) if per_echelon else 0.0,
            "MAE_backlog": float(np.mean([v["MAE_backlog"] for v in per_echelon.values()])) if per_echelon else 0.0,
            "MAE_shipments": float(np.mean([v["MAE_shipments"] for v in per_echelon.values()])) if per_echelon else 0.0,
            "ApproxWasserstein_inventory": approx_wasserstein_1d(inv_s_flat, inv_o_flat),
            "MMD_inventory": compute_mmd_rbf(inv_s_flat, inv_o_flat, gamma=1.0),
            "ApproxWasserstein_backlog": approx_wasserstein_1d(b_s_flat, b_o_flat),
            "MMD_backlog": compute_mmd_rbf(b_s_flat, b_o_flat, gamma=1.0),
            "ApproxWasserstein_shipments": approx_wasserstein_1d(sh_s_flat, sh_o_flat),
            "MMD_shipments": compute_mmd_rbf(sh_s_flat, sh_o_flat, gamma=1.0),
        }
        # Backward compatibility aliases
        agg["RMSE_inventory_mean"] = agg["RMSE_inventory"]
        agg["RMSE_backlog_mean"] = agg["RMSE_backlog"]
        agg["MAE_inventory_mean"] = agg["MAE_inventory"]
        agg["MAE_backlog_mean"] = agg["MAE_backlog"]

        metrics = {
            "per_echelon": per_echelon,
            "aggregate": agg,
        }
        return metrics


def _avg_nested(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    out: Dict[str, Any] = {}
    keys = dicts[0].keys()
    for k in keys:
        vals = [d[k] for d in dicts if k in d]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            out[k] = _avg_nested(vals)
        elif isinstance(vals[0], (int, float, np.floating)):
            arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(arr)) if arr.size > 0 else 0.0
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            ci = 1.96 * std / math.sqrt(max(1, arr.size)) if arr.size > 1 else 0.0
            out[k] = mean
            out[f"{k}_std"] = std
            out[f"{k}_CI95"] = ci
    return out


def run_double_monte_carlo(
    simulator: Simulation,
    evaluator: "Evaluator",
    M: int,
    K: int,
    seed: int,
    window: Tuple[int, int],
) -> Dict[str, Any]:
    set_all_seeds(seed)
    base_params = simulator.param_registry_snapshot()

    def sample_params_from_base(base: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        params = {}
        lt = float(base.get("supply", {}).get("lead_time_L", simulator.params.get("supply.lead_time_L", 2.0)))
        lt = max(1.0, min(8.0, float(int(round(lt + np.random.randint(-1, 2))))))
        params["supply.lead_time_L"] = lt
        jitter = float(base.get("supply", {}).get("lead_time_jitter_std", simulator.params.get("supply.lead_time_jitter_std", 0.0)))
        jitter = max(0.0, min(3.0, jitter + np.random.normal(0.0, 0.2)))
        params["supply.lead_time_jitter_std"] = jitter
        lam = float(base.get("demand", {}).get("poisson_lambda", simulator.params.get("demand.poisson_lambda", 8.0)))
        lam = max(1.0, min(20.0, lam + np.random.normal(0.0, 0.5)))
        params["demand.poisson_lambda"] = lam
        beta = float(base.get("policy", {}).get("beta", simulator.params.get("policy.beta", 0.5)))
        beta = max(0.01, min(0.9, beta + np.random.normal(0.0, 0.05)))
        params["policy.beta"] = beta
        s_base = float(base.get("policy", {}).get("s_base", simulator.params.get("policy.s_base", 15.0)))
        s_base = max(5.0, min(30.0, s_base + np.random.normal(0.0, 1.0)))
        params["policy.s_base"] = s_base
        return params

    per_m_metrics: List[Dict[str, Any]] = []
    for m in range(max(1, M)):
        # Reset to base at start of each outer iteration
        for mod, kv in base_params.items():
            for key, val in kv.items():
                simulator.set_params(mod, **{key: val})
        sampled = sample_params_from_base(base_params)
        for k, v in sampled.items():
            mod, key = k.split(".", 1)
            simulator.set_params(mod, **{key: v})
        per_k_metrics: List[Dict[str, Any]] = []
        for k in range(max(1, K)):
            run_seed = seed + m * K + k
            simulator.run(start_day=window[0], end_day=window[1], seed=run_seed, use_data_actions=False, record_io=False)
            metrics = simulator.evaluate(filename=None)
            per_k_metrics.append(metrics)
        avg_metrics = _avg_nested(per_k_metrics)
        per_m_metrics.append(avg_metrics)

    final_metrics = _avg_nested(per_m_metrics)
    # Restore base params
    for mod, kv in base_params.items():
        for key, val in kv.items():
            simulator.set_params(mod, **{key: val})
    return {"metrics": final_metrics, "M": M, "K": K}


def run_gsa(
    simulator: Simulation,
    evaluator: "Evaluator",
    window: Tuple[int, int],
    search_space: Optional[Dict[str, Tuple[float, float]]] = None,
    num_samples: int = 50,
    seed: int = GLOBAL_SEED + 999,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    base_snapshot = simulator.param_registry_snapshot()
    if search_space is None:
        search_space = {
            "supply.lead_time_L": (1.0, 8.0),
            "supply.lead_time_jitter_std": (0.0, 3.0),
            "demand.poisson_lambda": (1.0, 20.0),
            "policy.beta": (0.01, 0.9),
            "policy.s_base": (5.0, 30.0),
        }

    def apply_set(full_key: str, val: float) -> None:
        mod, key = full_key.split(".", 1)
        simulator.set_params(mod, **{key: val})

    def score_current() -> float:
        sim_metrics = simulator.evaluate(filename=None)
        return float(sim_metrics.get("aggregate", {}).get("RMSE_inventory", 0.0))

    sensitivities: Dict[str, List[float]] = {k: [] for k in search_space.keys()}
    for i in range(num_samples):
        # Reset to base
        for mod, kv in base_snapshot.items():
            for key, val in kv.items():
                simulator.set_params(mod, **{key: val})
        # Sample a random base point in space
        base_point: Dict[str, float] = {}
        for full_key, (lo, hi) in search_space.items():
            if "lead_time_L" in full_key:
                base_point[full_key] = float(int(round(np.random.uniform(lo, hi))))
            else:
                base_point[full_key] = float(np.random.uniform(lo, hi))
            apply_set(full_key, base_point[full_key])

        # Evaluate base
        simulator.run(start_day=window[0], end_day=window[1], seed=seed + i, use_data_actions=False, record_io=False)
        y_base = score_current()

        # One-at-a-time perturbations
        for full_key, (lo, hi) in search_space.items():
            delta = 0.1 * (hi - lo) if hi > lo else 0.1
            perturbed_val = base_point[full_key] + delta
            if "lead_time_L" in full_key:
                perturbed_val = float(int(round(perturbed_val)))
                perturbed_val = max(lo, min(hi, perturbed_val))
            else:
                perturbed_val = max(lo, min(hi, perturbed_val))
            apply_set(full_key, perturbed_val)
            simulator.run(start_day=window[0], end_day=window[1], seed=seed + i + 427, use_data_actions=False, record_io=False)
            y_pert = score_current()
            effect = abs(y_pert - y_base) / (abs(delta) + 1e-8)
            sensitivities[full_key].append(float(effect))
            # Restore base value for next parameter
            apply_set(full_key, base_point[full_key])

    sens_summary = {k: {"mean_effect": float(np.mean(v)) if v else 0.0, "std_effect": float(np.std(v)) if len(v) > 1 else 0.0} for k, v in sensitivities.items()}
    return sens_summary


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-echelon supply chain simulator with pluggable calibrator.")
    p.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data directory path.")
    p.add_argument("--train_file", type=str, default="train_data.csv", help="Training data filename.")
    p.add_argument("--val_file", type=str, default="val_data.csv", help="Validation data filename.")
    p.add_argument("--test_file", type=str, default="test_data.csv", help="Test data filename.")
    p.add_argument("--param_file", type=str, default="parameters.json", help="Parameters JSON path.")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="Override parameter key=value; repeatable.")
    p.add_argument("--calibrator", type=str, default="random_search", choices=["logit_head", "random_search", "snpe"], help="Calibrator choice.")
    p.add_argument("--calib-config", type=str, default=None, help="Optional calibrator config JSON.")
    p.add_argument("--budget", type=int, default=50, help="Calibration budget (iterations).")
    p.add_argument("--calib-window", type=str, default="0:48", help="Calibration window start:end (inclusive).")
    p.add_argument("--results_dir", type=str, default="results", help="Artifacts/results directory under data_dir.")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global random seed.")
    p.add_argument("--double-mc", action="store_true", help="Enable Double Monte Carlo evaluation.")
    p.add_argument("--mc-M", type=int, default=20, help="DMC outer samples.")
    p.add_argument("--mc-K", type=int, default=10, help="DMC inner runs.")
    p.add_argument("--use-data-actions", action="store_true", help="Use data playback actions instead of policy.")
    p.add_argument("--ood-lead-time", type=int, default=None, help="If set, run OOD eval by overriding lead time on test.")
    p.add_argument("--alpha-noise", action="store_true", help="Enable alpha embedding noise in policy.")
    p.add_argument("--alpha-inv", type=float, default=0.0, help="Alpha noise scale for inventory perception.")
    p.add_argument("--alpha-bklg", type=float, default=0.0, help="Alpha noise scale for backlog perception.")
    p.add_argument("--gsa", action="store_true", help="Run Global Sensitivity Analysis after evaluation.")
    p.add_argument("--gsa-samples", type=int, default=50, help="Number of base samples for GSA.")
    args = p.parse_args()
    return args


def load_parameters(param_path: str) -> Dict[str, Any]:
    if not os.path.isfile(param_path):
        warnings.warn(f"Parameter file not found: {param_path}. Using defaults.")
        return {}
    with open(param_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        warnings.warn("Parameter file invalid; expected dict.")
        return {}
    return data


def apply_overrides(params: Dict[str, Any], overrides: List[str], frozen_defs: Optional[Dict[str, Any]], log_path: str) -> Dict[str, Any]:
    applied: Dict[str, Any] = {}
    frozen_defs = frozen_defs or {}
    for ov in overrides:
        if "=" not in ov:
            warnings.warn(f"Invalid override '{ov}'; expected key=value.")
            continue
        key, val = ov.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key in frozen_defs and bool(frozen_defs[key].get("frozen", False)):
            warnings.warn(f"Ignoring override for frozen parameter: {key}")
            continue
        try:
            parsed_val = json.loads(val)
        except Exception:
            try:
                parsed_val = float(val) if any(c in val for c in ".eE") else int(val)
            except Exception:
                parsed_val = val
        params[key] = parsed_val
        applied[key] = parsed_val

    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "w") as f:
        json.dump({"applied_overrides": applied}, f, indent=2)
    return params


def gate_ood_evaluation(args: argparse.Namespace) -> bool:
    return args.ood_lead_time is not None


def initialize_from_data(sim: Simulation, train_df: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    try:
        echelons = metadata.get("echelons", sim.state["echelons"])
        sim.state["echelons"] = echelons
        # Reset structures for specified echelons
        sim.state["inventory"] = {e: 0 for e in echelons}
        sim.state["backlog"] = {e: 0 for e in echelons}
        sim.state["pipeline"] = {e: [] for e in echelons}
        sim.state["orders_last"] = {e: 0 for e in echelons}
        # Set initial inventory/backlog from first week
        t_col = infer_time_column(train_df)
        first_t = int(train_df[t_col].min())
        for e in echelons:
            sub = train_df[(train_df.get("echelon") == e) & (train_df[t_col] == first_t)]
            if not sub.empty:
                inv0 = float(sub["inventory"].iloc[0]) if "inventory" in sub.columns else 0.0
                b0 = float(sub["backlog"].iloc[0]) if "backlog" in sub.columns else 0.0
                sim.state["inventory"][e] = int(round(inv0))
                sim.state["backlog"][e] = int(round(b0))
        # Baseline policy and demand from metadata if present
        pol = metadata.get("baseline_policy", {})
        if isinstance(pol, dict):
            if "beta" in pol:
                sim.set_params("policy", beta=float(pol["beta"]))
            if "s_base" in pol:
                sim.set_params("policy", s_base=float(pol["s_base"]))
        dem = metadata.get("demand", {})
        if isinstance(dem, dict):
            if "family" in dem:
                sim.set_params("global", demand_family=str(dem["family"]))
            if "poisson_lambda" in dem:
                sim.set_params("demand", **{"poisson_lambda": float(dem["poisson_lambda"])})
            if "negbin_mu" in dem:
                sim.set_params("demand", **{"negbin_mu": float(dem["negbin_mu"])})
            if "negbin_r" in dem:
                sim.set_params("demand", **{"negbin_r": float(dem["negbin_r"])})
        lt_cfg = metadata.get("lead_time", {})
        if isinstance(lt_cfg, dict):
            if "L" in lt_cfg:
                sim.set_params("supply", lead_time_L=float(int(lt_cfg["L"])))
            if "jitter" in lt_cfg:
                sim.set_params("supply", lead_time_jitter_std=float(lt_cfg["jitter"]))
    except Exception as exc:
        warnings.warn(f"Failed to initialize from metadata/train data: {exc}")


def main() -> None:
    args = parse_cli()
    set_all_seeds(args.seed)

    artifacts_dir = os.path.join(args.data_dir, args.results_dir)
    ensure_dir(artifacts_dir)

    param_defs_path = os.path.join(artifacts_dir, "parameter_definitions.json")
    if not os.path.isfile(param_defs_path):
        with open(param_defs_path, "w") as f:
            json.dump({}, f)
    with open(param_defs_path, "r") as f:
        param_defs = json.load(f)

    params = load_parameters(os.path.join(args.data_dir, args.param_file))
    params = apply_overrides(params, args.overrides, param_defs, os.path.join(artifacts_dir, "overrides_applied.json"))

    sim = Simulation(artifacts_dir=artifacts_dir, seed=args.seed)
    # Apply alpha noise overrides
    if args.alpha_noise:
        sim.set_params("noise", **{"alpha_enabled": True})
        sim.set_params("noise", **{"alpha_inv": float(args.alpha_inv)})
        sim.set_params("noise", **{"alpha_bklg": float(args.alpha_bklg)})

    for k, v in params.items():
        if "." in k:
            mod, key = k.split(".", 1)
            sim.set_params(mod, **{key: v})
        else:
            sim.params[k] = v

    # Load datasets
    try:
        train_df = load_dataset(args.data_dir, args.train_file)
        val_df = load_dataset(args.data_dir, args.val_file)
    except Exception as exc:
        traceback.print_exc()
        print(f"Failed to load train/val: {exc}")
        return

    test_df = None
    if args.test_file and os.path.isfile(os.path.join(args.data_dir, args.test_file)):
        test_df = load_dataset(args.data_dir, args.test_file)

    # Load metadata and calibrated_config if present
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    metadata = {}
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            warnings.warn("Failed to load metadata.json")

    calibrated_cfg_path = os.path.join(args.data_dir, "calibrated_config.json")
    if os.path.isfile(calibrated_cfg_path):
        try:
            with open(calibrated_cfg_path, "r") as f:
                cal_cfg = json.load(f)
            if isinstance(cal_cfg, dict):
                for k, v in cal_cfg.items():
                    if "." in k:
                        mod, key = k.split(".", 1)
                        sim.set_params(mod, **{key: v})
                    else:
                        sim.params[k] = v
        except Exception:
            warnings.warn("Failed to load calibrated_config.json")

    # Initialize from data and metadata
    initialize_from_data(sim, train_df, metadata)

    # Build observed series for validation/test; default to val set for evaluation
    sim.load_observed(val_df)

    try:
        ws = args.calib_window.split(":")
        train_window = (int(ws[0]), int(ws[1]))
    except Exception:
        train_window = (0, 48)

    calibrator = get_calibrator(args.calibrator, args.calib_config)
    evaluator = Evaluator()
    fitted = calibrator.fit(
        bundle={},
        simulator=sim,
        evaluator=evaluator,
        train_window=train_window,
        seed=args.seed,
        budget=args.budget,
        artifacts_dir=os.path.join(artifacts_dir, "calibration"),
        params_adapter=SimpleParamsAdapter(param_defs_path),
    )

    adapter = SimpleParamsAdapter(param_defs_path)
    adapter.apply(sim, fitted)

    sim.run(start_day=train_window[0], end_day=train_window[1], seed=args.seed, use_data_actions=args.use_data_actions, record_io=True)
    metrics_val = sim.evaluate(filename=os.path.join(artifacts_dir, "results", "metrics_val.json"))
    with open(os.path.join(artifacts_dir, "results", "metrics_val.json"), "w") as f:
        json.dump(metrics_val, f, indent=2)
    sim.save_results(os.path.join(artifacts_dir, "results", "simulated_traces_val.json"))
    sim.save_all_io(os.path.join(artifacts_dir, "io", "val"))

    if test_df is not None:
        sim.load_observed(test_df)
        sim.run(start_day=0, end_day=sim.time_horizon - 1, seed=args.seed, use_data_actions=args.use_data_actions, record_io=True)
        metrics_test = sim.evaluate(filename=os.path.join(artifacts_dir, "results", "metrics_test.json"))
        with open(os.path.join(artifacts_dir, "results", "metrics_test.json"), "w") as f:
            json.dump(metrics_test, f, indent=2)
        sim.save_results(os.path.join(artifacts_dir, "results", "simulated_traces_test.json"))
        sim.save_all_io(os.path.join(artifacts_dir, "io", "test"))

        if gate_ood_evaluation(args):
            base_lt = sim.params.get("supply.lead_time_L", 2.0)
            sim.set_params("supply", lead_time_L=float(int(args.ood_lead_time)))
            sim.run(start_day=0, end_day=sim.time_horizon - 1, seed=args.seed + 42, use_data_actions=args.use_data_actions, record_io=True)
            metrics_test_ood = sim.evaluate(filename=os.path.join(artifacts_dir, "results", "metrics_test_ood.json"))
            with open(os.path.join(artifacts_dir, "results", "metrics_test_ood.json"), "w") as f:
                json.dump(metrics_test_ood, f, indent=2)
            sim.save_results(os.path.join(artifacts_dir, "results", "simulated_traces_test_ood.json"))
            sim.set_params("supply", lead_time_L=float(base_lt))

    if args.double_mc:
        dmc = run_double_monte_carlo(
            simulator=sim, evaluator=evaluator, M=args.mc_M, K=args.mc_K, seed=args.seed + 1000, window=train_window
        )
        with open(os.path.join(artifacts_dir, "results", "metrics_val_dmc.json"), "w") as f:
            json.dump(dmc["metrics"], f, indent=2)

    if args.gsa:
        gsa_res = run_gsa(simulator=sim, evaluator=evaluator, window=train_window, num_samples=args.gsa_samples)
        with open(os.path.join(artifacts_dir, "results", "gsa_results.json"), "w") as f:
            json.dump(gsa_res, f, indent=2)

    sim.visualize(window=train_window)

    print("Simulation completed. Artifacts stored at:", artifacts_dir)



# Execute main for both direct execution and sandbox wrapper invocation
main()