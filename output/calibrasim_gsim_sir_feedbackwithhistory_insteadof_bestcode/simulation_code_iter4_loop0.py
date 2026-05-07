PLAYBOOK_USAGE_JSON = '''{
  "used_bullets": [
    {
      "id": "sbi-npe-not-implemented-random-search-posterior",
      "why": "Validation loss remained high because the calibrator used a top-K random-search heuristic instead of learning p(theta|x). Implementing a lightweight conditional density estimator (Gaussian NPE via an MLP trained on simulated (theta, x) pairs) better matches the Blueprint and improves parameter inference."
    },
    {
      "id": "posterior-mean-from-topk-causes-bias",
      "why": "The previous unweighted mean of elite samples could drift away from good parameters. Replacing it with posterior sampling from the learned conditional density and taking the posterior mean reduces this bias."
    },
    {
      "id": "holdout-split-unused-and-confusing",
      "why": "Blueprint does not require temporal holdout; keep the interface but explicitly ignore holdout throughout to avoid accidental leakage/complexity, while preserving the program skeleton."
    }
  ]
}'''
CHANGE_SUMMARY_JSON = '''{
  "touched_symbols": [
    {
      "symbol": "SBINPECalibrator.fit",
      "reason": "Replace top-K pseudo-posterior with a small neural posterior estimator (conditional diagonal-Gaussian in logit-transformed parameter space) trained on simulated (theta, x) pairs; sample posterior conditioned on x_o and set optimized_parameters to posterior mean; retain existing artifact keys."
    },
    {
      "symbol": "DiscreteTimeSIRSimulator.rollout",
      "reason": "Improve posterior-predictive averaging allocation to use exactly num_rollouts trajectories while mixing posterior theta samples; keep return structure unchanged."
    },
    {
      "symbol": "main",
      "reason": "Update environment-variable path setup to match the integration-required snippet exactly."
    }
  ],
  "applied_strategies": [
    {
      "id": "sbi-npe-not-implemented-random-search-posterior",
      "applied": true
    },
    {
      "id": "posterior-mean-from-topk-causes-bias",
      "applied": true
    },
    {
      "id": "holdout-split-unused-and-confusing",
      "applied": true
    }
  ]
}'''

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

COMPARTMENTS = ("S", "I", "R")


def set_global_seed(seed: int) -> None:
    """Set Python/numpy seeds for reproducibility."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)


def ensure_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    """Validate that a dataframe contains required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")


def validate_trajectory_df(df: pd.DataFrame, name: str) -> None:
    """Validate observed SIR trajectories dataframe schema and basic consistency."""
    ensure_columns(df, ["trajectory_id", "time_step", "S", "I", "R"], name)

    for col in ["trajectory_id", "time_step"]:
        if not np.issubdtype(df[col].dtype, np.integer):
            arr = df[col].to_numpy()
            if not np.all(np.isfinite(arr)) or not np.all(np.equal(np.mod(arr, 1), 0)):
                raise ValueError(f"{name}.{col} must be integer-like.")
            df[col] = df[col].astype(int)

    for col in ["S", "I", "R"]:
        arr = df[col].to_numpy()
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}.{col} contains non-finite values.")
        if not np.all(np.equal(np.mod(arr, 1), 0)):
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
    """Group rows by trajectory_id and sort each group by time_step."""
    trajectories: Dict[int, pd.DataFrame] = {}
    for tid, g in df.groupby("trajectory_id"):
        trajectories[int(tid)] = g.sort_values("time_step").reset_index(drop=True)
    return trajectories


def df_to_states(df: pd.DataFrame) -> np.ndarray:
    """Convert a trajectory dataframe to a (T+1,3) integer array of (S,I,R)."""
    arr = df[list(COMPARTMENTS)].to_numpy(dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected shape (T+1,3), got {arr.shape}.")
    return arr


def flatten_states(states: np.ndarray) -> np.ndarray:
    """Flatten (T+1,3) states into a 1D vector [(S0,I0,R0),...,(ST,IT,RT)]."""
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError(f"states must have shape (T+1,3), got {states.shape}.")
    return states.reshape(-1)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for raw and grouped train/val/test datasets."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_trajs: Dict[int, pd.DataFrame]
    val_trajs: Dict[int, pd.DataFrame]
    test_trajs: Dict[int, pd.DataFrame]


@dataclass(frozen=True)
class HoldoutBundle:
    """Kept for compatibility; blueprint does not require temporal holdout splits."""
    train_prefix: Dict[int, pd.DataFrame]
    train_suffix: Dict[int, pd.DataFrame]
    prefix_lengths: Dict[int, int]


@dataclass
class SirParameters:
    """SIR parameters."""
    beta: float
    gamma: float

    def validate(self) -> None:
        """Validate parameter bounds."""
        if not np.isfinite(self.beta) or self.beta < 0.0 or self.beta > 5.0:
            raise ValueError(f"beta must be in [0,5], got {self.beta}.")
        if not np.isfinite(self.gamma) or self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError(f"gamma must be in [0,1], got {self.gamma}.")


@dataclass(frozen=True)
class SirTrajectoryAgent:
    """One observed trajectory used for evaluation/simulation."""
    trajectory_id: int
    observed_states: np.ndarray
    N: int
    horizon_T: int

    @staticmethod
    def from_dataframe(trajectory_id: int, df: pd.DataFrame) -> "SirTrajectoryAgent":
        """Build an agent from a trajectory dataframe."""
        states = df_to_states(df)
        s0, i0, r0 = states[0].tolist()
        N = int(s0 + i0 + r0)
        if N <= 0:
            raise ValueError(f"trajectory_id={trajectory_id}: N must be positive, got {N}.")
        horizon_T = int(df["time_step"].max())
        if states.shape[0] != horizon_T + 1:
            raise ValueError(
                f"trajectory_id={trajectory_id}: expected T+1 rows, got {states.shape[0]} for T={horizon_T}."
            )
        return SirTrajectoryAgent(
            trajectory_id=int(trajectory_id),
            observed_states=states,
            N=N,
            horizon_T=horizon_T,
        )


@dataclass(frozen=True)
class AgentsBundle:
    """Container for all agents in train/val/test splits."""
    train_agents: Dict[int, SirTrajectoryAgent]
    val_agents: Dict[int, SirTrajectoryAgent]
    test_agents: Dict[int, SirTrajectoryAgent]


class DiscreteTimeSIRSimulator:
    """Discrete-time stochastic SIR simulator with exact conservation adjustment."""

    def __init__(self, rng: np.random.Generator):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        self._rng = rng

    @staticmethod
    def _enforce_conservation(S: int, I: int, R: int, N: int) -> Tuple[int, int, int]:
        """Adjust counts to be nonnegative and sum to N."""
        S = int(S)
        I = int(I)
        R = int(R)
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}.")

        S = max(0, S)
        I = max(0, I)
        R = max(0, R)

        total = S + I + R
        if total != N:
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
        """Perform one stochastic SIR transition step."""
        params.validate()
        S, I, R = (int(state[0]), int(state[1]), int(state[2]))
        if S < 0 or I < 0 or R < 0:
            raise ValueError(f"State must be nonnegative, got {(S, I, R)}.")
        if S + I + R != N:
            S, I, R = self._enforce_conservation(S, I, R, N)
        if N <= 0:
            raise ValueError("N must be positive.")

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

    def rollout(self, agents: AgentsBundle, params: SirParameters, num_rollouts: int) -> Dict[str, Dict[int, np.ndarray]]:
        """Simulate mean trajectories for all agents in train/val/test splits."""
        if not isinstance(num_rollouts, int) or num_rollouts <= 0:
            raise ValueError(f"num_rollouts must be a positive int, got {num_rollouts}.")
        params.validate()

        posterior_thetas = getattr(params, "_posterior_thetas", None)
        if posterior_thetas is not None:
            posterior_thetas = np.asarray(posterior_thetas, dtype=np.float64)
            if posterior_thetas.ndim != 2 or posterior_thetas.shape[1] != 2:
                raise ValueError(
                    "params._posterior_thetas must have shape (M,2) with columns (beta,gamma), "
                    f"got {posterior_thetas.shape}."
                )
            if posterior_thetas.shape[0] <= 0:
                posterior_thetas = None

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

                if posterior_thetas is None:
                    acc = np.zeros((T + 1, 3), dtype=np.float64)
                    for _k in range(num_rollouts):
                        sub_sim = DiscreteTimeSIRSimulator(_fresh_rng())
                        s, i, r = init
                        sim = np.zeros((T + 1, 3), dtype=np.float64)
                        sim[0, :] = (s, i, r)
                        for t in range(T):
                            s, i, r = sub_sim.step((s, i, r), agent.N, params)
                            sim[t + 1, :] = (s, i, r)
                        acc += sim
                    pred[tid] = (acc / float(num_rollouts)).astype(float)
                else:
                    M = int(posterior_thetas.shape[0])
                    num_thetas_used = int(min(M, num_rollouts))
                    thetas_used = posterior_thetas[:num_thetas_used]

                    base = int(num_rollouts // num_thetas_used)
                    rem = int(num_rollouts - base * num_thetas_used)
                    counts = np.full((num_thetas_used,), base, dtype=int)
                    if rem > 0:
                        counts[:rem] += 1

                    acc = np.zeros((T + 1, 3), dtype=np.float64)
                    total = 0
                    for (beta, gamma), k_count in zip(thetas_used, counts):
                        theta_params = SirParameters(beta=float(beta), gamma=float(gamma))
                        for _k in range(int(k_count)):
                            sub_sim = DiscreteTimeSIRSimulator(_fresh_rng())
                            s, i, r = init
                            sim = np.zeros((T + 1, 3), dtype=np.float64)
                            sim[0, :] = (s, i, r)
                            for t in range(T):
                                s, i, r = sub_sim.step((s, i, r), agent.N, theta_params)
                                sim[t + 1, :] = (s, i, r)
                            acc += sim
                            total += 1
                    pred[tid] = (acc / float(max(1, total))).astype(float)

            out[split_name] = pred
        return out


class BaseCalibrator:
    """Abstract calibrator API."""
    def fit(self, agents: AgentsBundle, holdout: HoldoutBundle) -> Tuple[SirParameters, Dict[str, object]]:
        raise NotImplementedError


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically stable logit with clipping."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _theta_to_z(theta: np.ndarray) -> np.ndarray:
    """Transform bounded theta=(beta,gamma) to unconstrained z via logit."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError(f"theta must have shape (N,2), got {theta.shape}.")
    beta = theta[:, 0]
    gamma = theta[:, 1]
    u_beta = np.clip(beta / 5.0, 0.0, 1.0)
    u_gamma = np.clip(gamma, 0.0, 1.0)
    z_beta = _logit(u_beta)
    z_gamma = _logit(u_gamma)
    return np.stack([z_beta, z_gamma], axis=1)


def _z_to_theta(z: np.ndarray) -> np.ndarray:
    """Transform unconstrained z to bounded theta=(beta,gamma) via sigmoid."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"z must have shape (N,2), got {z.shape}.")
    u = _sigmoid(z)
    beta = 5.0 * u[:, 0]
    gamma = u[:, 1]
    return np.stack([beta, gamma], axis=1)


class _MLPGaussianNPE:
    """
    Lightweight 'NPE-like' conditional density estimator using an MLP that outputs a
    diagonal-Gaussian distribution over z=logit(theta) given x.

    This is dependency-free (NumPy only) and provides a principled posterior sampler
    p(theta|x) via z~N(mu(x), diag(std(x)^2)), theta = transform(z).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be positive int, got {input_dim}.")
        self.rng = rng
        self.input_dim = input_dim
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.W1 = (rng.standard_normal((input_dim, self.hidden1)).astype(np.float32) * np.sqrt(2.0 / input_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros((self.hidden1,), dtype=np.float32)
        self.W2 = (rng.standard_normal((self.hidden1, self.hidden2)).astype(np.float32) * np.sqrt(2.0 / self.hidden1)).astype(
            np.float32
        )
        self.b2 = np.zeros((self.hidden2,), dtype=np.float32)
        self.W3 = (rng.standard_normal((self.hidden2, 4)).astype(np.float32) * np.sqrt(2.0 / self.hidden2)).astype(
            np.float32
        )
        self.b3 = np.zeros((4,), dtype=np.float32)

        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3, "b3": self.b3}

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass. Returns (mu_z, log_std_z, cache)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (B,{self.input_dim}), got {X.shape}.")

        h1_pre = X @ self.W1 + self.b1
        h1 = self._relu(h1_pre)
        h2_pre = h1 @ self.W2 + self.b2
        h2 = self._relu(h2_pre)
        out = h2 @ self.W3 + self.b3
        mu = out[:, :2]
        log_std = out[:, 2:]
        log_std = np.clip(log_std, -5.0, 3.0).astype(np.float32)

        cache = {"X": X, "h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2, "mu": mu, "log_std": log_std}
        return mu, log_std, cache

    def loss_and_grads(self, X: np.ndarray, z_true: np.ndarray) -> Tuple[float, Dict[str, np.ndarray]]:
        """Compute Gaussian NLL loss in z-space and gradients for Adam."""
        z_true = np.asarray(z_true, dtype=np.float32)
        if z_true.ndim != 2 or z_true.shape[1] != 2:
            raise ValueError(f"z_true must have shape (B,2), got {z_true.shape}.")

        mu, log_std, cache = self.forward(X)
        var = np.exp(2.0 * log_std).astype(np.float32)
        diff = (mu - z_true).astype(np.float32)

        nll = 0.5 * ((diff * diff) / var + 2.0 * log_std + np.log(2.0 * np.pi)).astype(np.float32)
        loss = float(np.mean(np.sum(nll, axis=1)))

        dmu = (diff / var).astype(np.float32)
        dlog_std = (1.0 - (diff * diff) / var).astype(np.float32)

        dout = np.concatenate([dmu, dlog_std], axis=1).astype(np.float32)
        B = float(X.shape[0])
        dout = (dout / B).astype(np.float32)

        h2 = cache["h2"]
        h1 = cache["h1"]
        h2_pre = cache["h2_pre"]
        h1_pre = cache["h1_pre"]

        grads: Dict[str, np.ndarray] = {}
        grads["W3"] = (h2.T @ dout).astype(np.float32) + (self.weight_decay * self.W3).astype(np.float32)
        grads["b3"] = np.sum(dout, axis=0).astype(np.float32)

        dh2 = (dout @ self.W3.T).astype(np.float32)
        dh2_pre = (dh2 * self._relu_grad(h2_pre)).astype(np.float32)

        grads["W2"] = (h1.T @ dh2_pre).astype(np.float32) + (self.weight_decay * self.W2).astype(np.float32)
        grads["b2"] = np.sum(dh2_pre, axis=0).astype(np.float32)

        dh1 = (dh2_pre @ self.W2.T).astype(np.float32)
        dh1_pre = (dh1 * self._relu_grad(h1_pre)).astype(np.float32)

        X_in = cache["X"]
        grads["W1"] = (X_in.T @ dh1_pre).astype(np.float32) + (self.weight_decay * self.W1).astype(np.float32)
        grads["b1"] = np.sum(dh1_pre, axis=0).astype(np.float32)

        return loss, grads

    def adam_step(self, grads: Dict[str, np.ndarray]) -> None:
        """Apply an Adam update."""
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = self.lr

        for name, param in self._params().items():
            g = grads[name].astype(np.float32)
            self._m[name] = (b1 * self._m[name] + (1.0 - b1) * g).astype(np.float32)
            self._v[name] = (b2 * self._v[name] + (1.0 - b2) * (g * g)).astype(np.float32)

            mhat = self._m[name] / (1.0 - b1**self._t)
            vhat = self._v[name] / (1.0 - b2**self._t)
            param -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)

    def fit(self, X: np.ndarray, z: np.ndarray, batch_size: int, max_epochs: int) -> Dict[str, float]:
        """Train the MLP with mini-batch Adam."""
        X = np.asarray(X, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (N,{self.input_dim}), got {X.shape}.")
        if z.ndim != 2 or z.shape[1] != 2 or z.shape[0] != X.shape[0]:
            raise ValueError(f"z must have shape (N,2) and align with X, got z={z.shape}, X={X.shape}.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive int, got {batch_size}.")
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive int, got {max_epochs}.")

        N = X.shape[0]
        idx = np.arange(N, dtype=int)

        last_loss = float("nan")
        for _epoch in range(max_epochs):
            self.rng.shuffle(idx)
            for start in range(0, N, batch_size):
                batch_idx = idx[start: start + batch_size]
                loss, grads = self.loss_and_grads(X[batch_idx], z[batch_idx])
                self.adam_step(grads)
                last_loss = loss

        return {"final_train_nll": float(last_loss)}


class SBINPECalibrator(BaseCalibrator):
    """
    Dependency-free calibrator retaining the expected interface.
    """

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

    @staticmethod
    def _select_observation_agent(train_agents: Mapping[int, SirTrajectoryAgent]) -> SirTrajectoryAgent:
        """Select the single observation trajectory agent (must be train trajectory_id=0)."""
        if not train_agents:
            raise ValueError("No training agents available for observation selection.")
        if 0 in train_agents:
            return train_agents[0]
        tid = sorted(train_agents.keys())[0]
        raise ValueError(
            "Task spec requires using train trajectory_id=0 as the sole observation, "
            f"but trajectory_id=0 not found. Found ids like: {list(sorted(train_agents.keys()))[:10]}. "
            f"(Smallest found: {tid})"
        )

    def fit(self, agents: AgentsBundle, holdout: HoldoutBundle) -> Tuple[SirParameters, Dict[str, object]]:
        _ = holdout

        obs_agent = self._select_observation_agent(agents.train_agents)
        obs_states = obs_agent.observed_states
        T = obs_agent.horizon_T
        N = obs_agent.N
        if N <= 0:
            raise ValueError(f"Observation N must be positive, got {N}.")

        x_o = flatten_states(obs_states.astype(np.float32)).astype(np.float32)
        init = tuple(obs_states[0].tolist())
        x_dim = int(x_o.shape[0])

        def _fresh_rng() -> np.random.Generator:
            seed = int(self._rng.integers(0, 2**32 - 1, dtype=np.uint32))
            return np.random.default_rng(seed)

        cand_thetas = np.zeros((self._num_simulations, 2), dtype=np.float64)
        x_sims = np.zeros((self._num_simulations, x_dim), dtype=np.float32)

        for n in range(self._num_simulations):
            beta = float(self._rng.uniform(0.0, 5.0))
            gamma = float(self._rng.uniform(0.0, 1.0))
            params = SirParameters(beta=beta, gamma=gamma)

            sim_engine = DiscreteTimeSIRSimulator(_fresh_rng())
            sim = np.zeros((T + 1, 3), dtype=np.float32)
            s, i, r = init
            sim[0, :] = (s, i, r)
            for t in range(T):
                s, i, r = sim_engine.step((s, i, r), N, params)
                sim[t + 1, :] = (s, i, r)

            cand_thetas[n, :] = (beta, gamma)
            x_sims[n, :] = flatten_states(sim).astype(np.float32)

        diffs = (x_sims - x_o.reshape(1, -1)).astype(np.float32)
        cand_losses = np.mean(diffs * diffs, axis=1).astype(np.float64)
        best_idx = int(np.argmin(cand_losses))
        best_beta, best_gamma = cand_thetas[best_idx].tolist()

        X_mean = x_sims.mean(axis=0).astype(np.float32)
        X_std = x_sims.std(axis=0).astype(np.float32)
        X_std = np.where(X_std < 1e-6, 1.0, X_std).astype(np.float32)

        X_train = ((x_sims - X_mean) / X_std).astype(np.float32)
        z_train = _theta_to_z(cand_thetas).astype(np.float32)

        npe = _MLPGaussianNPE(
            rng=_fresh_rng(),
            input_dim=x_dim,
            hidden1=128,
            hidden2=64,
            lr=1e-3,
            weight_decay=1e-4,
        )
        train_stats = npe.fit(
            X=X_train,
            z=z_train,
            batch_size=self._training_batch_size,
            max_epochs=self._training_max_epochs,
        )

        x_o_norm = ((x_o - X_mean) / X_std).reshape(1, -1).astype(np.float32)
        mu_o, log_std_o, _cache = npe.forward(x_o_norm)
        mu_o = mu_o.astype(np.float64).reshape(2)
        std_o = np.exp(log_std_o.astype(np.float64).reshape(2))

        std_o = np.maximum(std_o, 0.15)

        eps = self._rng.standard_normal((self._num_posterior_samples, 2)).astype(np.float64)
        z_post = mu_o.reshape(1, 2) + eps * std_o.reshape(1, 2)
        posterior_samples = _z_to_theta(z_post).astype(np.float64)

        posterior_mean = posterior_samples.mean(axis=0)
        posterior_std = posterior_samples.std(axis=0)

        optimized = SirParameters(beta=float(posterior_mean[0]), gamma=float(posterior_mean[1]))
        optimized.validate()

        m_eval = int(min(50, posterior_samples.shape[0]))
        thetas_eval = posterior_samples[:m_eval].astype(np.float64, copy=False)
        setattr(optimized, "_posterior_thetas", thetas_eval)
        setattr(optimized, "_posterior_predictive_num_thetas", int(thetas_eval.shape[0]))

        artifacts: Dict[str, object] = {
            "observation_trajectory_id": int(obs_agent.trajectory_id),
            "observation_T": int(T),
            "x_o_shape": list(x_o.shape),
            "num_simulations": int(self._num_simulations),
            "num_posterior_samples": int(self._num_posterior_samples),
            "posterior_samples_mean": {"beta": float(posterior_mean[0]), "gamma": float(posterior_mean[1])},
            "posterior_samples_std": {"beta": float(posterior_std[0]), "gamma": float(posterior_std[1])},
            "best_candidate": {"beta": float(best_beta), "gamma": float(best_gamma), "loss": float(cand_losses[best_idx])},
            "posterior_predictive_num_thetas": int(thetas_eval.shape[0]),
            "device_unused": str(self._device),
            "training_hparams_unused": {
                "training_batch_size": int(self._training_batch_size),
                "training_max_epochs": int(self._training_max_epochs),
                "final_train_nll": float(train_stats.get("final_train_nll", float("nan"))),
            },
        }
        return optimized, artifacts


class Evaluator:
    """Compute dataset-level MSE metrics and conservation violation rate."""

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
                raise ValueError(f"trajectory_id={tid} shape mismatch: observed {obs.shape} vs predicted {pred.shape}.")
            all_obs.append(obs)
            all_pred.append(pred)

        obs_cat = np.concatenate(all_obs, axis=0)
        pred_cat = np.concatenate(all_pred, axis=0)
        per_dim = np.mean((obs_cat - pred_cat) ** 2, axis=0)
        overall = float(np.mean(per_dim))
        return overall, per_dim

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

    def compute_metrics(self, agents: AgentsBundle, rollouts: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, object]:
        train_mse, train_per_dim = self._mse_over_dataset(agents.train_agents, rollouts["train"])
        val_mse, val_per_dim = self._mse_over_dataset(agents.val_agents, rollouts["val"])
        test_mse, test_per_dim = self._mse_over_dataset(agents.test_agents, rollouts["test"])

        metrics: Dict[str, object] = {
            "train_loss": train_mse,
            "val_loss": val_mse,
            "test_loss": test_mse,
            "train_loss_per_dim": {"S": float(train_per_dim[0]), "I": float(train_per_dim[1]), "R": float(train_per_dim[2])},
            "val_loss_per_dim": {"S": float(val_per_dim[0]), "I": float(val_per_dim[1]), "R": float(val_per_dim[2])},
            "test_loss_per_dim": {"S": float(test_per_dim[0]), "I": float(test_per_dim[1]), "R": float(test_per_dim[2])},
            "val_conservation_violation_rate": self._conservation_violation_rate(agents.val_agents, rollouts["val"]),
        }
        return metrics


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Discrete-time stochastic SIR calibration.")
    p.add_argument("--seed", type=int, default=123, help="Global random seed.")
    p.add_argument("--num_simulations", type=int, default=2000, help="Calibration simulation budget.")
    p.add_argument("--num_posterior_samples", type=int, default=4000, help="Pseudo posterior sample count.")
    p.add_argument("--training_batch_size", type=int, default=256, help="Unused (kept for compatibility).")
    p.add_argument("--training_max_epochs", type=int, default=200, help="Unused (kept for compatibility).")
    p.add_argument("--num_rollouts", type=int, default=50, help="Stochastic rollouts per trajectory for evaluation.")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    p.add_argument("--device", type=str, default="cpu", help="Unused (kept for compatibility).")
    return p.parse_args(argv)


def load_data(data_dir: str) -> DatasetBundle:
    """Load and validate train/val/test CSVs from DATA_DIR."""
    if not isinstance(data_dir, str) or not data_dir:
        raise ValueError(f"DATA_DIR must be a non-empty str, got {data_dir!r}.")
    data_dir = os.path.abspath(data_dir)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR does not exist: {data_dir}")

    train_path = os.path.join(data_dir, "train_seed_10_n_100.csv")
    val_path = os.path.join(data_dir, "val_seed_10_n_100.csv")
    test_path = os.path.join(data_dir, "test_seed_10_n_100.csv")

    for pth in [train_path, val_path, test_path]:
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"Required data file not found: {pth}\nEnsure it exists under DATA_DIR={data_dir}")

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
    """Build trajectory agents from grouped dataframes."""
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
    """
    Create a prefix/suffix split of the training data.

    Note: This is retained for compatibility with the original skeleton, but is not
    used by the calibrator because the Blueprint specifies file-split-only evaluation.
    """
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
    """Write results.json and val_predicted_mean_trajectories.csv to output_dir."""
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
    """Entry point."""
    cfg = parse_cli(argv)
    set_global_seed(cfg.seed)

    # Path Handling Instructions (COPY EXACTLY)
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")
    if not PROJECT_ROOT or not DATA_PATH:
        raise EnvironmentError(
            "Environment variables PROJECT_ROOT and DATA_PATH must be set (non-empty). "
            f"Got PROJECT_ROOT={PROJECT_ROOT!r}, DATA_PATH={DATA_PATH!r}."
        )
    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

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