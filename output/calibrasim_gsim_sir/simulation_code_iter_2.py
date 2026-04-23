from __future__ import annotations

PLAYBOOK_USAGE_JSON = '''{"used_bullets":[{"id":"deterministic-algorithms-enabled-hurting-sbi-training","why":"Torch deterministic algorithms were forced on, which can reduce NPE training quality; make determinism opt-in while keeping seeding for reproducibility."},{"id":"single-trajectory-sbi-summary-too-high-dimensional","why":"NPE was trained directly on an unnormalized 183-dim flattened trajectory; apply a consistent feature transform (population scaling, z-score normalization, and mild time/peak weighting) to improve conditioning and posterior quality."},{"id":"rng-reuse-couples-sbi-and-evaluation-rollouts","why":"The same RNG stream was reused for SBI simulation generation and evaluation rollouts; split RNG streams to prevent coupling and order-dependent randomness."},{"id":"hard-requirement-absolute-data-dir","why":"Removed unnecessary absolute-path enforcement in load_data; normalize paths instead to avoid avoidable runtime failures."}]}'''
CHANGE_SUMMARY_JSON = '''{"touched_symbols":[{"symbol":"set_global_seed","reason":"Make torch deterministic algorithms optional (opt-in) while preserving seeding; improve training stability/performance."},{"symbol":"parse_cli","reason":"Add an opt-in CLI flag for torch deterministic algorithms without changing existing arguments."},{"symbol":"SBINPECalibrator.fit","reason":"Improve NPE conditioning via population scaling + z-score normalization and a fixed time/peak weighting applied consistently to simulated xs and observation x_o."},{"symbol":"load_data","reason":"Remove unnecessary absolute-path restriction; normalize data_dir with abspath and keep existence checks."},{"symbol":"main","reason":"Split RNG streams for SBI vs evaluation; implement required path-handling snippet exactly (with a pre-check to avoid None join errors)."},{"symbol":"DiscreteTimeSIRSimulator.rollout","reason":"Stabilize randomness and reduce order-dependence by iterating trajectories in sorted id order and pre-drawing per-trajectory rollout seeds."}],"applied_strategies":[{"id":"deterministic-algorithms-enabled-hurting-sbi-training","applied":true},{"id":"single-trajectory-sbi-summary-too-high-dimensional","applied":true},{"id":"rng-reuse-couples-sbi-and-evaluation-rollouts","applied":true},{"id":"hard-requirement-absolute-data-dir","applied":true}]}'''

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This program requires PyTorch (torch). Please install it, e.g. `pip install torch`."
    ) from e

# SBI is required by the task specification (Simulation-Based Inference using NPE).
try:
    from sbi.utils import BoxUniform  # type: ignore

    def _get_sbi_inference_class():
        # Prefer SNPE (NPE family) classes across versions.
        candidates = [
            ("sbi.inference", "NPE"),
            ("sbi.inference", "SNPE"),
            ("sbi.inference", "SNPE_C"),
            ("sbi.inference", "SNPE_A"),
            ("sbi.inference", "SNPE_B"),
        ]
        for mod_name, cls_name in candidates:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    return cls
            except Exception:
                continue
        return None

    NPE = _get_sbi_inference_class()
    if NPE is None:  # pragma: no cover
        raise ImportError("Could not find an SNPE/NPE inference class in sbi.inference.")
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This program requires the `sbi` package for Simulation-Based Inference (NPE/SNPE). "
        "Install with `pip install sbi`."
    ) from e


COMPARTMENTS = ("S", "I", "R")

# Global knob controlled by CLI (opt-in).
TORCH_DETERMINISTIC: bool = False


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs.

    Notes
    -----
    Deterministic Torch algorithms are *opt-in* via the module-level
    `TORCH_DETERMINISTIC` flag, because forcing determinism can degrade training
    quality/performance for neural density estimators.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Opt-in determinism (do not force by default).
    try:
        torch.use_deterministic_algorithms(bool(TORCH_DETERMINISTIC))
    except Exception:
        # Some torch versions may not fully support toggling determinism; keep seeded.
        pass

    # Keep runtime predictable on CPU.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def ensure_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    """Ensure required columns exist in a dataframe."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def validate_trajectory_df(df: pd.DataFrame, name: str) -> None:
    """Validate that a trajectory dataframe matches the expected schema and invariants."""
    ensure_columns(df, ["trajectory_id", "time_step", "S", "I", "R"], name)

    for col in ["trajectory_id", "time_step"]:
        if not np.issubdtype(df[col].dtype, np.integer):
            if not np.all(np.isfinite(df[col].to_numpy())) or not np.all(
                np.equal(np.mod(df[col].to_numpy(), 1), 0)
            ):
                raise ValueError(f"{name}.{col} must be integer-like.")
            df[col] = df[col].astype(int)

    for col in ["S", "I", "R"]:
        if not np.all(np.isfinite(df[col].to_numpy())):
            raise ValueError(f"{name}.{col} contains non-finite values.")
        if not np.all(np.equal(np.mod(df[col].to_numpy(), 1), 0)):
            raise ValueError(f"{name}.{col} must be integer counts (no fractions).")
        if (df[col] < 0).any():
            neg_rows = df[df[col] < 0].head(5)
            raise ValueError(f"{name}.{col} has negative values. Example rows:\n{neg_rows}")

    for tid, g in df.groupby("trajectory_id"):
        g_sorted = g.sort_values("time_step")
        ts = g_sorted["time_step"].to_numpy().astype(int)
        if len(ts) == 0:
            raise ValueError(f"{name}: trajectory_id={tid} has no rows.")
        if ts[0] != 0:
            raise ValueError(f"{name}: trajectory_id={tid} must start at time_step=0, got {ts[0]}.")
        expected = np.arange(ts[0], ts[0] + len(ts))
        if not np.array_equal(ts, expected):
            raise ValueError(
                f"{name}: trajectory_id={tid} time_step must be contiguous from 0..T. "
                f"Got: {ts[:10]}... (len={len(ts)})."
            )

        sir = g_sorted[list(COMPARTMENTS)].to_numpy().astype(int)
        totals = sir.sum(axis=1)
        if not np.all(totals == totals[0]):
            raise ValueError(
                f"{name}: trajectory_id={tid} violates conservation in observed data "
                f"(S+I+R not constant). This simulator assumes constant N."
            )


def group_trajectories(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group a trajectory dataframe by trajectory_id."""
    trajectories: Dict[int, pd.DataFrame] = {}
    for tid, g in df.groupby("trajectory_id"):
        trajectories[int(tid)] = g.sort_values("time_step").reset_index(drop=True)
    return trajectories


def df_to_states(df: pd.DataFrame) -> np.ndarray:
    """Convert a trajectory dataframe to an integer state matrix (T+1, 3)."""
    arr = df[list(COMPARTMENTS)].to_numpy(dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (T+1,3), got {arr.shape}.")
    return arr


def flatten_states(states: np.ndarray) -> np.ndarray:
    """Flatten (T+1,3) SIR states into a 1D vector [S0,I0,R0,S1,I1,R1,...]."""
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError(f"states must have shape (T+1,3), got {states.shape}.")
    return states.reshape(-1)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for train/val/test dataframes and grouped trajectories."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_trajs: Dict[int, pd.DataFrame]
    val_trajs: Dict[int, pd.DataFrame]
    test_trajs: Dict[int, pd.DataFrame]


@dataclass(frozen=True)
class HoldoutBundle:
    """(Unused for SBI) Kept for interface compatibility."""
    train_prefix: Dict[int, pd.DataFrame]
    train_suffix: Dict[int, pd.DataFrame]
    prefix_lengths: Dict[int, int]


@dataclass
class SirParameters:
    """SIR parameters for the discrete-time simulator."""
    beta: float
    gamma: float

    def validate(self) -> None:
        """Validate parameter ranges."""
        if not np.isfinite(self.beta) or self.beta < 0.0 or self.beta > 5.0:
            raise ValueError(f"beta must be in [0,5], got {self.beta}.")
        if not np.isfinite(self.gamma) or self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError(f"gamma must be in [0,1], got {self.gamma}.")


@dataclass(frozen=True)
class SirTrajectoryAgent:
    """A single observed SIR trajectory (agent) with fixed N and horizon."""
    trajectory_id: int
    observed_states: np.ndarray
    N: int
    horizon_T: int

    @staticmethod
    def from_dataframe(trajectory_id: int, df: pd.DataFrame) -> "SirTrajectoryAgent":
        """Build a trajectory agent from its dataframe."""
        states = df_to_states(df)
        s0, i0, r0 = states[0].tolist()
        N = int(s0 + i0 + r0)
        if N <= 0:
            raise ValueError(f"trajectory_id={trajectory_id}: N must be positive, got {N}.")
        horizon_T = int(df["time_step"].max())
        if states.shape[0] != horizon_T + 1:
            raise ValueError(
                f"trajectory_id={trajectory_id}: expected T+1 rows, got {states.shape[0]} "
                f"for T={horizon_T}."
            )
        return SirTrajectoryAgent(
            trajectory_id=int(trajectory_id),
            observed_states=states,
            N=N,
            horizon_T=horizon_T,
        )


@dataclass(frozen=True)
class AgentsBundle:
    """Container for all trajectory agents across splits."""
    train_agents: Dict[int, SirTrajectoryAgent]
    val_agents: Dict[int, SirTrajectoryAgent]
    test_agents: Dict[int, SirTrajectoryAgent]


class DiscreteTimeSIRSimulator:
    """Discrete-time stochastic SIR simulator with conservation enforcement."""

    def __init__(self, rng: np.random.Generator):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        self._rng = rng

    @staticmethod
    def _enforce_conservation(S: int, I: int, R: int, N: int) -> Tuple[int, int, int]:
        """Clamp to nonnegative integers and enforce S+I+R=N."""
        S = int(S)
        I = int(I)
        R = int(R)
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}.")

        S = max(0, S)
        I = max(0, I)
        R = max(0, R)

        total = S + I + R
        if total == N:
            return S, I, R

        diff = N - total
        S += diff

        if S < 0:
            deficit = -S
            S = 0
            if I >= deficit:
                I -= deficit
                deficit = 0
            else:
                deficit -= I
                I = 0
            if deficit > 0:
                R = max(0, R - deficit)

        S = max(0, min(N, S))
        I = max(0, min(N - S, I))
        R = N - S - I
        R = max(0, min(N, R))
        total2 = S + I + R
        if total2 != N:
            S = max(0, min(N, S + (N - total2)))
            total3 = S + I + R
            if total3 != N:
                R = N - S - I
        return int(S), int(I), int(R)

    def step(self, state: Tuple[int, int, int], N: int, params: SirParameters) -> Tuple[int, int, int]:
        """Advance the SIR state by one discrete time step."""
        params.validate()
        S, I, R = (int(state[0]), int(state[1]), int(state[2]))
        if S < 0 or I < 0 or R < 0:
            raise ValueError(f"State must be nonnegative, got {(S, I, R)}.")
        if S + I + R != N:
            S, I, R = self._enforce_conservation(S, I, R, N)
        if N == 0:
            raise ValueError("N cannot be 0.")

        p_inf = 1.0 - float(np.exp(-params.beta * (I / N)))
        p_inf = float(np.clip(p_inf, 0.0, 1.0))
        new_inf = int(self._rng.binomial(S, p_inf)) if S > 0 else 0

        gamma = float(np.clip(params.gamma, 0.0, 1.0))
        new_rec = int(self._rng.binomial(I, gamma)) if I > 0 else 0

        S1 = S - new_inf
        I1 = I + new_inf - new_rec
        R1 = R + new_rec

        S1, I1, R1 = self._enforce_conservation(S1, I1, R1, N)
        if R1 < R:
            delta = R - R1
            R1 = R
            S1 = max(0, S1 - delta)
            S1, I1, R1 = self._enforce_conservation(S1, I1, R1, N)

        return S1, I1, R1

    def rollout(
        self,
        agents: AgentsBundle,
        params: SirParameters,
        num_rollouts: int,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Run stochastic rollouts for all trajectories and return mean predictions."""
        if not isinstance(num_rollouts, int) or num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be a positive int, got {num_rollouts}.")
        params.validate()

        out: Dict[str, Dict[int, np.ndarray]] = {}
        for split_name, split_agents in [
            ("train", agents.train_agents),
            ("val", agents.val_agents),
            ("test", agents.test_agents),
        ]:
            pred: Dict[int, np.ndarray] = {}

            # Stabilize run-to-run behavior independent of dict ordering.
            for tid in sorted(split_agents.keys()):
                agent = split_agents[tid]
                T = agent.horizon_T
                init = tuple(agent.observed_states[0].tolist())

                sims = np.zeros((num_rollouts, T + 1, 3), dtype=float)

                # Pre-draw per-rollout seeds to reduce coupling across trajectories.
                rollout_seeds = self._rng.integers(0, 2**32 - 1, size=num_rollouts, dtype=np.uint32)

                for k in range(num_rollouts):
                    local_rng = np.random.default_rng(int(rollout_seeds[k]))
                    local_sim = DiscreteTimeSIRSimulator(local_rng)

                    s, i, r = init
                    sims[k, 0, :] = (s, i, r)
                    for t in range(T):
                        s, i, r = local_sim.step((s, i, r), agent.N, params)
                        sims[k, t + 1, :] = (s, i, r)

                pred[tid] = np.mean(sims, axis=0)
            out[split_name] = pred
        return out


class BaseCalibrator:
    """Abstract calibrator interface."""

    def fit(self, agents: AgentsBundle, holdout: HoldoutBundle) -> Tuple[SirParameters, Dict[str, object]]:
        raise NotImplementedError


class SBINPECalibrator(BaseCalibrator):
    """Calibrate (beta, gamma) via SBI using NPE from a single observed trajectory (train id 0)."""

    def __init__(
        self,
        rng: np.random.Generator,
        num_simulations: int,
        num_posterior_samples: int,
        training_batch_size: int,
        training_max_epochs: int,
        device: str = "cpu",
    ):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        for name, v in [
            ("num_simulations", num_simulations),
            ("num_posterior_samples", num_posterior_samples),
            ("training_batch_size", training_batch_size),
            ("training_max_epochs", training_max_epochs),
        ]:
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v}.")
        self._rng = rng
        self._num_simulations = num_simulations
        self._num_posterior_samples = num_posterior_samples
        self._training_batch_size = training_batch_size
        self._training_max_epochs = training_max_epochs

        # Defensive device selection.
        if device != "cpu" and (not torch.cuda.is_available()) and ("cuda" in device):
            self._device = "cpu"
        else:
            self._device = device

        low = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self._device)
        high = torch.tensor([5.0, 1.0], dtype=torch.float32, device=self._device)
        self._prior = BoxUniform(low=low, high=high)

    @staticmethod
    def _select_observation_agent(train_agents: Mapping[int, SirTrajectoryAgent]) -> SirTrajectoryAgent:
        """Select the single observation trajectory required by the task spec: trajectory_id=0."""
        if not train_agents:
            raise ValueError("No training agents available for SBI observation selection.")
        if 0 in train_agents:
            return train_agents[0]
        tid = sorted(train_agents.keys())[0]
        raise ValueError(
            "Task spec requires using train_states[0] (trajectory_id=0) as the sole observation, "
            f"but trajectory_id=0 not found. Found ids like: {list(sorted(train_agents.keys()))[:10]}. "
            f"Please ensure the training CSV contains trajectory_id=0. (Smallest found: {tid})"
        )

    @staticmethod
    def _build_time_weight_vector(T: int, peak_t: int, device: str) -> torch.Tensor:
        """Construct a fixed per-time-step weighting to emphasize informative regions."""
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}.")
        peak_t = int(np.clip(peak_t, 0, T))

        t = torch.arange(T + 1, dtype=torch.float32, device=device)

        # Emphasize early dynamics (beta sensitivity) and around observed peak (beta/gamma).
        w_early = 1.0 + 1.5 * torch.exp(-t / 10.0)
        sigma = 5.0
        w_peak = 1.0 + 1.0 * torch.exp(-0.5 * ((t - float(peak_t)) / sigma) ** 2)

        w_t = w_early * w_peak  # (T+1,)
        w = w_t.repeat_interleave(3)  # (T+1)*3
        return w

    @staticmethod
    def _sbi_train_compat(inference_obj, **train_kwargs):
        # Some SBI versions don't accept certain kwargs (e.g., show_train_summary).
        try:
            return inference_obj.train(**train_kwargs)
        except TypeError:
            train_kwargs.pop("show_train_summary", None)
            return inference_obj.train(**train_kwargs)

    @staticmethod
    def _sbi_build_posterior_compat(inference_obj, density_estimator):
        # Some SBI versions want no args; others accept (density_estimator=...).
        try:
            return inference_obj.build_posterior(density_estimator)
        except TypeError:
            try:
                return inference_obj.build_posterior(density_estimator=density_estimator)
            except TypeError:
                return inference_obj.build_posterior()

    @staticmethod
    def _posterior_sample_compat(posterior_obj, num_samples: int, x_cond: torch.Tensor) -> torch.Tensor:
        # Support differing conditioning keyword names across sbi versions.
        shape = (int(num_samples),)
        try:
            return posterior_obj.sample(shape, x=x_cond)
        except TypeError:
            try:
                return posterior_obj.sample(shape, x_o=x_cond)
            except TypeError:
                try:
                    return posterior_obj.sample(shape, condition=x_cond)
                except TypeError as e:
                    raise RuntimeError(
                        "Incompatible sbi posterior.sample signature. "
                        "Tried keywords: x, x_o, condition."
                    ) from e

    def fit(self, agents: AgentsBundle, holdout: HoldoutBundle) -> Tuple[SirParameters, Dict[str, object]]:
        """Run NPE calibration on train trajectory_id=0 and return posterior-mean parameters."""
        _ = holdout

        obs_agent = self._select_observation_agent(agents.train_agents)
        obs_states = obs_agent.observed_states
        T = obs_agent.horizon_T

        # Scale by population to reduce raw-count conditioning issues.
        N = float(obs_agent.N)
        x_o_raw = flatten_states(obs_states).astype(np.float32) / max(N, 1.0)
        x_o_t_raw = torch.from_numpy(x_o_raw).to(self._device)

        obs_I = obs_states[:, 1].astype(float)
        peak_t = int(np.argmax(obs_I)) if obs_I.size > 0 else 0
        w = self._build_time_weight_vector(T=T, peak_t=peak_t, device=self._device)

        simulator = DiscreteTimeSIRSimulator(self._rng)

        thetas = torch.zeros((self._num_simulations, 2), dtype=torch.float32, device=self._device)
        xs_raw = torch.zeros((self._num_simulations, (T + 1) * 3), dtype=torch.float32, device=self._device)

        init = tuple(obs_states[0].tolist())
        N_int = obs_agent.N

        for n in range(self._num_simulations):
            theta = self._prior.sample((1,)).squeeze(0)
            beta = float(theta[0].item())
            gamma = float(theta[1].item())
            params = SirParameters(beta=beta, gamma=gamma)

            sim = np.zeros((T + 1, 3), dtype=int)
            s, i, r = init
            sim[0, :] = (s, i, r)
            for t in range(T):
                s, i, r = simulator.step((s, i, r), N_int, params)
                sim[t + 1, :] = (s, i, r)

            thetas[n, :] = theta
            xs_raw[n, :] = torch.from_numpy((flatten_states(sim).astype(np.float32) / max(N, 1.0))).to(self._device)

        # Z-score normalization (fit on simulated xs, apply to both xs and x_o).
        x_mean = xs_raw.mean(dim=0)
        x_std = xs_raw.std(dim=0)
        x_std = torch.clamp(x_std, min=1e-3)

        xs = (xs_raw - x_mean) / x_std
        x_o_t = (x_o_t_raw - x_mean) / x_std

        # Apply time/peak weighting after standardization for better identifiability.
        xs = xs * w
        x_o_t = x_o_t * w

        inference = NPE(prior=self._prior, device=self._device)

        if not hasattr(inference, "append_simulations"):
            raise RuntimeError(
                "Incompatible sbi inference API: missing append_simulations(). "
                "Please install a compatible `sbi` version."
            )
        inference = inference.append_simulations(thetas, xs)

        density_estimator = self._sbi_train_compat(
            inference,
            training_batch_size=self._training_batch_size,
            max_num_epochs=self._training_max_epochs,
            show_train_summary=False,
        )
        posterior = self._sbi_build_posterior_compat(inference, density_estimator)

        with torch.no_grad():
            posterior_samples = self._posterior_sample_compat(
                posterior, num_samples=self._num_posterior_samples, x_cond=x_o_t
            )
            posterior_mean = posterior_samples.mean(dim=0)

        optimized = SirParameters(beta=float(posterior_mean[0].item()), gamma=float(posterior_mean[1].item()))
        optimized.validate()

        artifacts: Dict[str, object] = {
            "observation_trajectory_id": obs_agent.trajectory_id,
            "observation_T": T,
            "x_o_shape": list(x_o_t.shape),
            "num_simulations": self._num_simulations,
            "num_posterior_samples": self._num_posterior_samples,
            "posterior_samples_mean": {"beta": optimized.beta, "gamma": optimized.gamma},
            # Additional metadata (does not affect required outputs/metrics).
            "x_transform": {
                "population_scaled": True,
                "zscore": True,
                "time_peak_weighting": True,
                "peak_t": int(peak_t),
                "weight_min": float(w.min().item()),
                "weight_max": float(w.max().item()),
            },
            "device_used": str(self._device),
        }
        return optimized, artifacts


class Evaluator:
    """Compute evaluation metrics for predicted mean trajectories."""

    @staticmethod
    def _mse_over_dataset(
        agents: Mapping[int, SirTrajectoryAgent],
        predicted: Mapping[int, np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        all_obs: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []

        for tid, agent in agents.items():
            if tid not in predicted:
                raise KeyError(f"Missing predictions for trajectory_id={tid}.")
            obs = agent.observed_states.astype(float)
            pred = predicted[tid].astype(float)
            if obs.shape != pred.shape:
                raise ValueError(
                    f"trajectory_id={tid} shape mismatch: observed {obs.shape} vs predicted {pred.shape}."
                )
            all_obs.append(obs)
            all_pred.append(pred)

        obs_cat = np.concatenate(all_obs, axis=0)
        pred_cat = np.concatenate(all_pred, axis=0)
        per_dim = np.mean((obs_cat - pred_cat) ** 2, axis=0)
        overall = float(np.mean(per_dim))
        return overall, per_dim

    @staticmethod
    def _one_step_mae(
        agents: Mapping[int, SirTrajectoryAgent],
        predicted: Mapping[int, np.ndarray],
    ) -> float:
        abs_errs: List[float] = []
        for tid, agent in agents.items():
            obs = agent.observed_states.astype(float)
            pred = predicted[tid].astype(float)
            T = agent.horizon_T
            for t in range(T):
                abs_errs.append(float(np.mean(np.abs(obs[t + 1, :] - pred[t + 1, :]))))
        return float(np.mean(abs_errs)) if abs_errs else float("nan")

    @staticmethod
    def _peak_i_error(
        agents: Mapping[int, SirTrajectoryAgent],
        predicted: Mapping[int, np.ndarray],
    ) -> Dict[str, float]:
        peak_maes: List[float] = []
        time_maes: List[float] = []
        for tid, agent in agents.items():
            obs_I = agent.observed_states[:, 1].astype(float)
            pred_I = predicted[tid][:, 1].astype(float)
            obs_peak = float(np.max(obs_I))
            pred_peak = float(np.max(pred_I))
            obs_t = int(np.argmax(obs_I))
            pred_t = int(np.argmax(pred_I))
            peak_maes.append(abs(obs_peak - pred_peak))
            time_maes.append(abs(obs_t - pred_t))
        return {
            "peak_I_mae": float(np.mean(peak_maes)) if peak_maes else float("nan"),
            "peak_time_mae": float(np.mean(time_maes)) if time_maes else float("nan"),
        }

    @staticmethod
    def _conservation_violation_rate(
        agents: Mapping[int, SirTrajectoryAgent],
        predicted: Mapping[int, np.ndarray],
        atol: float = 1e-6,
    ) -> float:
        violations = 0
        total = 0
        for tid, agent in agents.items():
            pred = predicted[tid]
            N = agent.N
            for row in pred:
                total += 1
                if np.any(row < -atol):
                    violations += 1
                    continue
                if abs(float(np.sum(row) - N)) > atol:
                    violations += 1
        return float(violations / total) if total > 0 else float("nan")

    def compute_metrics(
        self,
        agents: AgentsBundle,
        rollouts: Dict[str, Dict[int, np.ndarray]],
    ) -> Dict[str, object]:
        train_mse, train_per_dim = self._mse_over_dataset(agents.train_agents, rollouts["train"])
        val_mse, val_per_dim = self._mse_over_dataset(agents.val_agents, rollouts["val"])
        test_mse, test_per_dim = self._mse_over_dataset(agents.test_agents, rollouts["test"])

        metrics: Dict[str, object] = {
            "train_loss": train_mse,
            "val_loss": val_mse,
            "test_loss": test_mse,
            "val_loss_per_dim": {"S": float(val_per_dim[0]), "I": float(val_per_dim[1]), "R": float(val_per_dim[2])},
            "val_one_step_MAE_SIR": self._one_step_mae(agents.val_agents, rollouts["val"]),
            "val_peak_I_error": self._peak_i_error(agents.val_agents, rollouts["val"]),
            "val_conservation_violation_rate": self._conservation_violation_rate(agents.val_agents, rollouts["val"]),
            "train_loss_per_dim": {"S": float(train_per_dim[0]), "I": float(train_per_dim[1]), "R": float(train_per_dim[2])},
            "test_loss_per_dim": {"S": float(test_per_dim[0]), "I": float(test_per_dim[1]), "R": float(test_per_dim[2])},
        }
        return metrics


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Discrete-time stochastic SIR + SBI calibration (NPE).")
    p.add_argument("--seed", type=int, default=123, help="Global random seed (deterministic).")
    p.add_argument("--num_simulations", type=int, default=2000, help="Number of simulations for SBI NPE training.")
    p.add_argument("--num_posterior_samples", type=int, default=4000, help="Posterior samples for posterior mean.")
    p.add_argument("--training_batch_size", type=int, default=256, help="NPE training batch size.")
    p.add_argument("--training_max_epochs", type=int, default=200, help="NPE training epochs.")
    p.add_argument("--num_rollouts", type=int, default=50, help="Stochastic rollouts per trajectory for evaluation.")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory (created if missing).")
    p.add_argument("--device", type=str, default="cpu", help="Torch device for SBI (e.g., cpu or cuda).")
    p.add_argument(
        "--torch_deterministic",
        action="store_true",
        help="Opt-in: force torch deterministic algorithms (may slow/hurt SBI training).",
    )
    return p.parse_args(argv)


def load_data(data_dir: str) -> DatasetBundle:
    """Load train/val/test CSV files and validate their structure."""
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError("data_dir must be a non-empty string.")
    data_dir = os.path.abspath(data_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    train_path = os.path.join(data_dir, "train_seed_10_n_100.csv")
    val_path = os.path.join(data_dir, "val_seed_10_n_100.csv")
    test_path = os.path.join(data_dir, "test_seed_10_n_100.csv")

    for pth in [train_path, val_path, test_path]:
        if not os.path.isfile(pth):
            raise FileNotFoundError(
                f"Required data file not found: {pth}\nEnsure it exists under DATA_DIR={data_dir}"
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    validate_trajectory_df(train_df, "train")
    validate_trajectory_df(val_df, "val")
    validate_trajectory_df(test_df, "test")

    return DatasetBundle(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_trajs=group_trajectories(train_df),
        val_trajs=group_trajectories(val_df),
        test_trajs=group_trajectories(test_df),
    )


def build_network_and_agents(data: DatasetBundle) -> AgentsBundle:
    """Build SirTrajectoryAgent objects from grouped dataframes."""
    def build_agents(trajs: Mapping[int, pd.DataFrame]) -> Dict[int, SirTrajectoryAgent]:
        agents: Dict[int, SirTrajectoryAgent] = {}
        for tid, df in trajs.items():
            agents[int(tid)] = SirTrajectoryAgent.from_dataframe(int(tid), df)
        if not agents:
            raise ValueError("No trajectories found to build agents.")
        return agents

    return AgentsBundle(
        train_agents=build_agents(data.train_trajs),
        val_agents=build_agents(data.val_trajs),
        test_agents=build_agents(data.test_trajs),
    )


def holdout_split(data: DatasetBundle, train_fraction: float = 0.8) -> HoldoutBundle:
    """Create a prefix/suffix split of each training trajectory (kept for interface compatibility)."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}.")

    prefix: Dict[int, pd.DataFrame] = {}
    suffix: Dict[int, pd.DataFrame] = {}
    lengths: Dict[int, int] = {}

    for tid, df in data.train_trajs.items():
        T = int(df["time_step"].max())
        n_points = T + 1
        n_prefix = int(np.floor(train_fraction * n_points))
        n_prefix = max(2, min(n_points, n_prefix))
        prefix_df = df.iloc[:n_prefix].copy().reset_index(drop=True)
        suffix_df = df.iloc[n_prefix:].copy().reset_index(drop=True)
        prefix[int(tid)] = prefix_df
        suffix[int(tid)] = suffix_df
        lengths[int(tid)] = n_prefix

    return HoldoutBundle(train_prefix=prefix, train_suffix=suffix, prefix_lengths=lengths)


def save_results(
    output_dir: str,
    optimized_params: SirParameters,
    calibration_artifacts: Mapping[str, object],
    metrics: Mapping[str, object],
    rollouts: Mapping[str, Mapping[int, np.ndarray]],
) -> None:
    """Write results.json and val_predicted_mean_trajectories.csv (format must not change)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "optimized_parameters": {"beta": optimized_params.beta, "gamma": optimized_params.gamma},
        "calibration_artifacts": dict(calibration_artifacts),
        "metrics": dict(metrics),
    }
    (out_path / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    val_rows: List[Dict[str, object]] = []
    for tid, traj in rollouts["val"].items():
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError(f"Unexpected predicted trajectory shape for tid={tid}: {traj.shape}")
        for t in range(traj.shape[0]):
            val_rows.append(
                {
                    "trajectory_id": int(tid),
                    "time_step": int(t),
                    "S_pred": float(traj[t, 0]),
                    "I_pred": float(traj[t, 1]),
                    "R_pred": float(traj[t, 2]),
                }
            )
    pd.DataFrame(val_rows).to_csv(out_path / "val_predicted_mean_trajectories.csv", index=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: load data, calibrate via SBI/NPE, evaluate, and write outputs."""
    cfg = parse_cli(argv)

    global TORCH_DETERMINISTIC
    TORCH_DETERMINISTIC = bool(getattr(cfg, "torch_deterministic", False))

    set_global_seed(cfg.seed)

    # Pre-check to avoid None join errors; then apply the required path handling snippet exactly.
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if not project_root or not data_path:
        raise EnvironmentError(
            "Missing required environment variables. Set:\n"
            "  PROJECT_ROOT=/absolute/path/to/project\n"
            "  DATA_PATH=relative/path/to/data_dir\n"
            "Then re-run."
        )

    # Path Handling Instructions (COPY EXACTLY)
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

    data_dir = DATA_DIR

    # Split RNG streams: one for SBI simulation generation, one for evaluation rollouts.
    ss = np.random.SeedSequence(cfg.seed)
    sbi_ss, eval_ss = ss.spawn(2)
    rng_sbi = np.random.default_rng(sbi_ss)
    rng_eval = np.random.default_rng(eval_ss)

    data = load_data(data_dir)
    agents = build_network_and_agents(data)
    holdout = holdout_split(data, train_fraction=0.8)

    calibrator = SBINPECalibrator(
        rng=rng_sbi,
        num_simulations=cfg.num_simulations,
        num_posterior_samples=cfg.num_posterior_samples,
        training_batch_size=cfg.training_batch_size,
        training_max_epochs=cfg.training_max_epochs,
        device=cfg.device,
    )
    optimized_params, calibration_artifacts = calibrator.fit(agents, holdout)

    simulator = DiscreteTimeSIRSimulator(rng=rng_eval)
    rollouts = simulator.rollout(agents=agents, params=optimized_params, num_rollouts=cfg.num_rollouts)

    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(agents, rollouts)

    save_results(
        output_dir=cfg.output_dir,
        optimized_params=optimized_params,
        calibration_artifacts=calibration_artifacts,
        metrics=metrics,
        rollouts=rollouts,
    )

    print(
        json.dumps(
            {
                "optimized_parameters": {"beta": optimized_params.beta, "gamma": optimized_params.gamma},
                "train_loss": metrics["train_loss"],
                "val_loss": metrics["val_loss"],
                "test_loss": metrics["test_loss"],
            },
            indent=2,
        )
    )
    return 0


main()