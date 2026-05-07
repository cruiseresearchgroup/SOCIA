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

try:
    try:
        from sbi.inference import NPE as _NPE  # type: ignore
    except Exception:
        from sbi.inference import SNPE as _NPE  # type: ignore

    from sbi.utils import BoxUniform  # type: ignore

    NPE = _NPE
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This program requires the `sbi` package for Simulation-Based Inference (NPE/SNPE). "
        "Install with `pip install sbi`."
    ) from e


PLAYBOOK_USAGE_JSON = '''{
  "used_bullets": [
    {
      "id": "correlated-rollouts-per-theta-reuse-same-rng",
      "why": "Averaging multiple rollouts per theta only reduces noise if rollouts are independent; the previous implementation reused one RNG stream, limiting variance reduction and hurting posterior quality."
    },
    {
      "id": "single-trajectory-sbi-limits-generalization",
      "why": "Because calibration is constrained to a single observed trajectory, adding stable low-dimensional summary features helps identifiability and reduces overfitting to trajectory noise, improving generalization to val/test."
    },
    {
      "id": "evaluation-rollouts-not-independent-across-trajectories",
      "why": "Evaluation mean trajectories were computed from rollouts that all shared a single RNG stream; spawning per-rollout RNGs improves effective sample size and stabilizes mean predictions."
    }
  ]
}'''
CHANGE_SUMMARY_JSON = '''{
  "touched_symbols": [
    {
      "symbol": "SBINPECalibrator.fit",
      "reason": "Use independent RNG streams per (theta, rollout) when averaging rollouts; concatenate deterministic summary features to the flattened trajectory before z-scoring to improve identifiability and reduce single-trajectory overfitting."
    },
    {
      "symbol": "DiscreteTimeSIRSimulator.rollout",
      "reason": "Spawn per-(trajectory, rollout) RNG streams to ensure rollouts are independent and mean-of-K trajectories have higher effective sample size."
    },
    {
      "symbol": "main",
      "reason": "Make path setup match the required integration snippet exactly (PROJECT_ROOT/DATA_PATH/DATA_DIR)."
    }
  ],
  "applied_strategies": [
    {
      "id": "correlated-rollouts-per-theta-reuse-same-rng",
      "applied": true
    },
    {
      "id": "single-trajectory-sbi-limits-generalization",
      "applied": true
    },
    {
      "id": "evaluation-rollouts-not-independent-across-trajectories",
      "applied": true
    }
  ]
}'''

COMPARTMENTS = ("S", "I", "R")


def set_global_seed(seed: int) -> None:
    """Set Python/numpy/torch seeds for reproducibility."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def ensure_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    """Ensure that a DataFrame contains required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def validate_trajectory_df(df: pd.DataFrame, name: str) -> None:
    """Validate a trajectory CSV DataFrame matches the expected schema and invariants."""
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
    """Group a DataFrame into per-trajectory DataFrames sorted by time_step."""
    trajectories: Dict[int, pd.DataFrame] = {}
    for tid, g in df.groupby("trajectory_id"):
        trajectories[int(tid)] = g.sort_values("time_step").reset_index(drop=True)
    return trajectories


def df_to_states(df: pd.DataFrame) -> np.ndarray:
    """Convert a per-trajectory DataFrame into an (T+1,3) integer state array."""
    arr = df[list(COMPARTMENTS)].to_numpy(dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (T+1,3), got {arr.shape}.")
    return arr


def flatten_states(states: np.ndarray) -> np.ndarray:
    """Flatten an (T+1,3) state array to shape ((T+1)*3,)."""
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError(f"states must have shape (T+1,3), got {states.shape}.")
    return states.reshape(-1)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for raw train/val/test data and grouped trajectories."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_trajs: Dict[int, pd.DataFrame]
    val_trajs: Dict[int, pd.DataFrame]
    test_trajs: Dict[int, pd.DataFrame]


@dataclass(frozen=True)
class HoldoutBundle:
    """Unused time-based holdout split kept for interface compatibility."""
    train_prefix: Dict[int, pd.DataFrame]
    train_suffix: Dict[int, pd.DataFrame]
    prefix_lengths: Dict[int, int]


@dataclass
class SirParameters:
    """SIR parameters to infer: beta (transmission), gamma (recovery)."""
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
    """Single observed trajectory wrapper used for evaluation and SBI observation."""
    trajectory_id: int
    observed_states: np.ndarray
    N: int
    horizon_T: int

    @staticmethod
    def from_dataframe(trajectory_id: int, df: pd.DataFrame) -> "SirTrajectoryAgent":
        """Construct an agent from a per-trajectory DataFrame."""
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
    """Train/val/test bundles of trajectory agents."""
    train_agents: Dict[int, SirTrajectoryAgent]
    val_agents: Dict[int, SirTrajectoryAgent]
    test_agents: Dict[int, SirTrajectoryAgent]


class DiscreteTimeSIRSimulator:
    """Discrete-time stochastic SIR simulator with integer transitions and conservation."""

    def __init__(self, rng: np.random.Generator):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        self._rng = rng

    @staticmethod
    def _enforce_conservation(S: int, I: int, R: int, N: int) -> Tuple[int, int, int]:
        """Clamp and adjust (S,I,R) to be nonnegative and sum exactly to N."""
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
        """Advance SIR state by one time step with stochastic infection/recovery draws."""
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
        """Simulate mean predicted trajectories for all agents using mean-of-K rollouts.

        Rollouts are made statistically independent by spawning a fresh RNG stream
        per (trajectory_id, rollout_index). This stabilizes mean predictions.
        """
        if not isinstance(num_rollouts, int) or num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be a positive int, got {num_rollouts}.")
        params.validate()

        def _fresh_rng() -> np.random.Generator:
            seed = int(self._rng.integers(0, 2**32 - 1, dtype=np.uint32))
            return np.random.default_rng(seed)

        out: Dict[str, Dict[int, np.ndarray]] = {}
        for split_name, split_agents in [
            ("train", agents.train_agents),
            ("val", agents.val_agents),
            ("test", agents.test_agents),
        ]:
            pred: Dict[int, np.ndarray] = {}
            for tid, agent in split_agents.items():
                T = agent.horizon_T
                init = tuple(agent.observed_states[0].tolist())
                sims = np.zeros((num_rollouts, T + 1, 3), dtype=float)
                for k in range(num_rollouts):
                    sub_sim = DiscreteTimeSIRSimulator(_fresh_rng())
                    s, i, r = init
                    sims[k, 0, :] = (s, i, r)
                    for t in range(T):
                        s, i, r = sub_sim.step((s, i, r), agent.N, params)
                        sims[k, t + 1, :] = (s, i, r)
                pred[tid] = np.mean(sims, axis=0)
            out[split_name] = pred
        return out


class BaseCalibrator:
    """Abstract base class for SIR parameter calibrators."""
    def fit(self, agents: AgentsBundle, holdout: HoldoutBundle) -> Tuple[SirParameters, Dict[str, object]]:
        raise NotImplementedError


class SBINPECalibrator(BaseCalibrator):
    """Simulation-Based Inference (NPE/SNPE) calibrator using a single observation trajectory."""

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
        self._device = device

        low = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
        high = torch.tensor([5.0, 1.0], dtype=torch.float32, device=device)
        self._prior = BoxUniform(low=low, high=high)

    @staticmethod
    def _select_observation_agent(train_agents: Mapping[int, SirTrajectoryAgent]) -> SirTrajectoryAgent:
        """Select the required SBI observation agent (trajectory_id == 0)."""
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
    def _summary_features(traj_frac: np.ndarray, T: int) -> np.ndarray:
        """Compute stable summary features from a normalized (fractional) trajectory."""
        if traj_frac.ndim != 2 or traj_frac.shape[1] != 3:
            raise ValueError(f"traj_frac must have shape (T+1,3), got {traj_frac.shape}.")
        if traj_frac.shape[0] != T + 1:
            raise ValueError(f"traj_frac length mismatch: got {traj_frac.shape[0]}, expected {T+1}.")
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}.")

        S = traj_frac[:, 0]
        I = traj_frac[:, 1]
        R = traj_frac[:, 2]

        peak_I = float(np.max(I))
        peak_t = int(np.argmax(I))
        peak_t_frac = float(peak_t / float(T))

        final_R = float(R[-1])
        final_S = float(S[-1])

        I0 = float(I[0])
        if T >= 3:
            early_growth = float((I[3] - I[0]) / 3.0)
        else:
            early_growth = float(I[min(1, T)] - I[0])

        mean_I = float(np.mean(I))

        return np.asarray([peak_I, peak_t_frac, final_R, final_S, I0, early_growth, mean_I], dtype=np.float32)

    def fit(self, agents: AgentsBundle, holdout: HoldoutBundle) -> Tuple[SirParameters, Dict[str, object]]:
        """Fit (beta,gamma) by NPE using only train trajectory_id==0 as observation."""
        _ = holdout

        obs_agent = self._select_observation_agent(agents.train_agents)
        obs_states = obs_agent.observed_states
        T = obs_agent.horizon_T
        N = obs_agent.N
        if N <= 0:
            raise ValueError(f"Observation N must be positive, got {N}.")

        obs_frac = (obs_states.astype(np.float32) / float(N)).astype(np.float32)
        x_o = np.concatenate([flatten_states(obs_frac), self._summary_features(obs_frac, T)]).astype(np.float32)
        x_o_t = torch.from_numpy(x_o).to(self._device)

        thetas = torch.zeros((self._num_simulations, 2), dtype=torch.float32, device=self._device)
        xs = torch.zeros((self._num_simulations, x_o.shape[0]), dtype=torch.float32, device=self._device)

        init = tuple(obs_states[0].tolist())
        k_cal_rollouts = 7

        def _fresh_rng() -> np.random.Generator:
            seed = int(self._rng.integers(0, 2**32 - 1, dtype=np.uint32))
            return np.random.default_rng(seed)

        for n in range(self._num_simulations):
            theta = self._prior.sample((1,)).squeeze(0)
            beta = float(theta[0].item())
            gamma = float(theta[1].item())
            params = SirParameters(beta=beta, gamma=gamma)

            acc = np.zeros((T + 1, 3), dtype=np.float64)
            for _k in range(k_cal_rollouts):
                sim_engine = DiscreteTimeSIRSimulator(_fresh_rng())
                sim = np.zeros((T + 1, 3), dtype=np.float64)
                s, i, r = init
                sim[0, :] = (s, i, r)
                for t in range(T):
                    s, i, r = sim_engine.step((s, i, r), N, params)
                    sim[t + 1, :] = (s, i, r)
                acc += sim
            mean_sim = acc / float(k_cal_rollouts)

            sim_frac = (mean_sim / float(N)).astype(np.float32)
            x_sim = np.concatenate([flatten_states(sim_frac), self._summary_features(sim_frac, T)]).astype(np.float32)

            thetas[n, :] = theta
            xs[n, :] = torch.from_numpy(x_sim).to(self._device)

        with torch.no_grad():
            x_mu = xs.mean(dim=0)
            x_std = xs.std(dim=0, unbiased=False)
            x_std = torch.clamp(x_std, min=1e-6)
            xs_z = (xs - x_mu) / x_std
            x_o_z = (x_o_t - x_mu) / x_std

        inference = NPE(prior=self._prior, device=self._device)
        inference = inference.append_simulations(thetas, xs_z)

        density_estimator = inference.train(
            training_batch_size=self._training_batch_size,
            max_num_epochs=self._training_max_epochs,
            show_train_summary=False,
        )
        posterior = inference.build_posterior(density_estimator)

        with torch.no_grad():
            posterior_samples = posterior.sample((self._num_posterior_samples,), x=x_o_z)
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
            "sbi_x_scaling": {"type": "normalize_by_N_then_zscore", "N": int(N), "z_eps": 1e-6},
            "k_cal_rollouts_per_theta": int(k_cal_rollouts),
            "x_batch_mean_mean": float(x_mu.mean().item()),
            "x_batch_std_mean": float(x_std.mean().item()),
        }
        return optimized, artifacts


class Evaluator:
    """Computes train/val/test metrics from predicted mean trajectories."""

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
            "train_loss_per_dim": {
                "S": float(train_per_dim[0]),
                "I": float(train_per_dim[1]),
                "R": float(train_per_dim[2]),
            },
            "test_loss_per_dim": {"S": float(test_per_dim[0]), "I": float(test_per_dim[1]), "R": float(test_per_dim[2])},
        }
        return metrics


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Discrete-time stochastic SIR + SBI calibration (NPE).")
    p.add_argument("--seed", type=int, default=123, help="Global random seed (deterministic).")
    p.add_argument("--num_simulations", type=int, default=2000, help="Number of simulations for SBI NPE training.")
    p.add_argument("--num_posterior_samples", type=int, default=4000, help="Posterior samples for posterior mean.")
    p.add_argument("--training_batch_size", type=int, default=256, help="NPE training batch size.")
    p.add_argument("--training_max_epochs", type=int, default=200, help="NPE training epochs.")
    p.add_argument("--num_rollouts", type=int, default=50, help="Stochastic rollouts per trajectory for evaluation.")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory (created if missing).")
    p.add_argument("--device", type=str, default="cpu", help="Torch device for SBI (e.g., cpu or cuda).")
    return p.parse_args(argv)


def load_data(data_dir: str) -> DatasetBundle:
    """Load train/val/test CSVs from an absolute DATA_DIR and validate them."""
    if not os.path.isabs(data_dir):
        raise ValueError(f"DATA_DIR must be absolute; got {data_dir!r}. Check PROJECT_ROOT/DATA_PATH.")
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
    """Build SirTrajectoryAgent objects for each trajectory in each split."""
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
    """Create a temporal prefix/suffix split (unused; retained for compatibility)."""
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
    """Write results.json and val_predicted_mean_trajectories.csv (exact schema preserved)."""
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
    """Entry point: load data, calibrate using SBI, evaluate, and write outputs."""
    cfg = parse_cli(argv)
    set_global_seed(cfg.seed)

    # Path Handling Instructions (COPY EXACTLY)
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if not PROJECT_ROOT or not DATA_PATH:
        raise EnvironmentError(
            "Environment variables PROJECT_ROOT and DATA_PATH must be set to locate the data directory. "
            f"Got PROJECT_ROOT={PROJECT_ROOT!r}, DATA_PATH={DATA_PATH!r}."
        )
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
    DATA_DIR = os.path.abspath(DATA_DIR)

    rng = np.random.default_rng(cfg.seed)

    data = load_data(DATA_DIR)
    agents = build_network_and_agents(data)
    holdout = holdout_split(data, train_fraction=0.8)

    calibrator = SBINPECalibrator(
        rng=rng,
        num_simulations=cfg.num_simulations,
        num_posterior_samples=cfg.num_posterior_samples,
        training_batch_size=cfg.training_batch_size,
        training_max_epochs=cfg.training_max_epochs,
        device=cfg.device,
    )
    optimized_params, calibration_artifacts = calibrator.fit(agents, holdout)

    simulator = DiscreteTimeSIRSimulator(rng=rng)
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