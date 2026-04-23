PLAYBOOK_USAGE_JSON = '''{"used_bullets":[{"id":"backlog-dominates-loss-latent-demand-unconditioned-on-actions","why":"Backlog MSE dominates validation error; introduce minimal state-conditioned demand mean using existing state variables and existing parameters to reduce systematic backlog drift under single-trajectory SBI."},{"id":"pipeline-update-sorts-by-leadtime-breaks-order-sensitive-length-dynamics","why":"Sorting pipeline entries each step can alter queue semantics; preserve insertion order to better match data-derived pipeline countdown behavior and reduce pipeline_len errors."},{"id":"dummy-pipeline-zero-quantities-can-freeze-pipeline-length-and-distort-countdown","why":"If pipeline tuples are missing and only pipeline_len is available, dummy entries should not introduce arbitrary countdown/removal dynamics; make the fallback more neutral and only use it when pipeline values are actually missing."},{"id":"sbi-calibration-uses-single-trajectory-and-overfits-update-convention-via-argmax","why":"Hard argmax selection of update convention can hurt generalization when logits are uncertain; sample convention per-trajectory from learned softmax probabilities during evaluation (deterministic per-trajectory via seeded RNG)."}]}'''

CHANGE_SUMMARY_JSON = '''{"touched_symbols":[{"symbol":"DataLoader._dummy_pipeline_from_len","reason":"Make dummy pipeline neutral (no arbitrary countdown churn) by using large lead times for zero-quantity placeholders."},{"symbol":"DataLoader._parse_states","reason":"If `pipeline` exists but a row has missing/NaN pipeline while `pipeline_len` exists, fall back to dummy pipeline for that row only to preserve observed pipeline_len without fabricating arrivals."},{"symbol":"BeerGameModel._sample_demand","reason":"Keep a compatible wrapper but route demand sampling through a new state-conditioned mean model to reduce backlog mismatch while using only existing parameters."},{"symbol":"BeerGameModel.step","reason":"Sample demand conditioned on the current within-step inventory/backlog/pipeline_len; remove pipeline sorting to preserve insertion order semantics."},{"symbol":"BeerGameSimulator.__init__","reason":"Add an internal flag to control stochastic vs argmax update-convention selection without changing external interfaces."},{"symbol":"BeerGameSimulator.rollout_trajectory","reason":"During evaluation, choose update convention per-trajectory by sampling from softmax(logits) with deterministic seeding; keep deterministic argmax during SBI simulation generation to reduce training noise."},{"symbol":"NPECalibrator.fit","reason":"Disable stochastic update-convention selection during SBI simulation pair generation for a lower-variance learning signal."}],"applied_strategies":[{"id":"backlog-dominates-loss-latent-demand-unconditioned-on-actions","applied":true},{"id":"pipeline-update-sorts-by-leadtime-breaks-order-sensitive-length-dynamics","applied":true},{"id":"dummy-pipeline-zero-quantities-can-freeze-pipeline-length-and-distort-countdown","applied":true},{"id":"sbi-calibration-uses-single-trajectory-and-overfits-update-convention-via-argmax","applied":true}]}'''

import abc
import argparse
import ast
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This program requires PyTorch (`torch`). Please install it in the runtime environment."
    ) from exc


GLOBAL_SEED = 12345


def set_global_determinism(seed: int) -> None:
    """Set global RNG determinism controls for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(False)
        except TypeError:
            pass


def softplus(x: float) -> float:
    """Numerically-stable softplus for scalar inputs."""
    if x > 30:
        return x
    return float(np.log1p(np.exp(x)))


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over 1D logits."""
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits)
    ex = np.exp(logits - m)
    s = ex.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid softmax normalization encountered. logits={logits}")
    return ex / s


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid for scalar inputs."""
    x = float(x)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class State:
    inventory: int
    pipeline: Tuple[Tuple[int, int], ...]
    backlog: int
    t: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory": int(self.inventory),
            "pipeline": [(int(q), int(tau)) for (q, tau) in self.pipeline],
            "backlog": int(self.backlog),
            "t": int(self.t),
        }


@dataclass(frozen=True)
class Action:
    order: int

    def to_dict(self) -> Dict[str, Any]:
        return {"order": int(self.order)}


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: int
    states: List[State]
    actions: List[Action]


@dataclass(frozen=True)
class DatasetSplit:
    trajectories: List[Trajectory]


@dataclass(frozen=True)
class DataBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @staticmethod
    def _parse_actions(df: pd.DataFrame, tid: int, num_states: int) -> List[Action]:
        """Parse actions from CSV.

        If action/order column is missing, return default actions (zeros).
        Actions correspond to transitions, so length is num_states - 1.
        """
        if "action" in df.columns:
            a = df[df["trajectory_id"] == tid].sort_values("time_step")["action"].to_numpy()
            return [Action(order=int(x)) for x in a]
        if "order" in df.columns:
            a = df[df["trajectory_id"] == tid].sort_values("time_step")["order"].to_numpy()
            return [Action(order=int(x)) for x in a]
        num_actions = max(0, num_states - 1)
        return [Action(order=0) for _ in range(num_actions)]

    @staticmethod
    def _parse_pipeline(value: Any) -> Tuple[Tuple[int, int], ...]:
        """Parse pipeline value which may be a string representation of a list."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return tuple()

        if isinstance(value, (list, tuple)):
            parsed = value
        else:
            s = str(value).strip()
            if s == "" or s.lower() in {"none", "nan"}:
                return tuple()
            try:
                parsed = ast.literal_eval(s)
            except Exception as exc:
                raise ValueError(f"Failed to parse pipeline value: {value!r}") from exc

        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Pipeline must parse to a list/tuple. Got: {type(parsed)}")

        out: List[Tuple[int, int]] = []
        for i, item in enumerate(parsed):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(f"Pipeline entry must be a 2-tuple at index {i}. Got: {item!r}")
            q, tau = item
            try:
                q_i = int(q)
                tau_i = int(tau)
            except Exception as exc:
                raise ValueError(f"Pipeline entry values must be ints at index {i}. Got: {item!r}") from exc
            if q_i < 0 or tau_i < 0:
                raise ValueError(f"Pipeline entry must be nonnegative at index {i}. Got: {item!r}")
            out.append((q_i, tau_i))
        return tuple(out)

    @staticmethod
    def _is_missing_pipeline_value(value: Any) -> bool:
        """Return True if a pipeline cell is missing/empty (not a real '[]')."""
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        s = str(value).strip().lower()
        return s in {"", "none", "nan"}

    @staticmethod
    def _dummy_pipeline_from_len(pipeline_len: int) -> Tuple[Tuple[int, int], ...]:
        """Create a neutral pipeline representation from length only.

        Important: never fabricate positive quantities from a length aggregate.

        Also avoid introducing arbitrary *countdown/removal* dynamics: if we don't know the
        individual remaining lead times, we should not cause synthetic entries to churn
        within the 60-step horizon. We therefore assign a very large remaining lead time.
        """
        pl = int(pipeline_len)
        if pl <= 0:
            return tuple()
        # Large tau ensures these placeholder entries do not arrive/remove within horizon.
        tau = 10_000
        return tuple((0, tau) for _ in range(pl))

    @staticmethod
    def _parse_states(df: pd.DataFrame, tid: int) -> List[State]:
        sub = df[df["trajectory_id"] == tid].sort_values("time_step")
        inv = sub["inventory"].to_numpy()
        back = sub["backlog"].to_numpy()

        has_pipeline = "pipeline" in sub.columns
        has_pipeline_len = "pipeline_len" in sub.columns
        if not has_pipeline and not has_pipeline_len:
            raise ValueError("CSV must contain `pipeline` or `pipeline_len` column.")

        pipeline_data = sub["pipeline"].to_numpy() if has_pipeline else None
        pl_data = sub["pipeline_len"].to_numpy() if has_pipeline_len else None

        if "t" in sub.columns:
            tt = sub["t"].to_numpy()
        else:
            tt = sub["time_step"].to_numpy()

        states: List[State] = []
        for i in range(len(sub)):
            if has_pipeline:
                raw_val = pipeline_data[i]
                pipeline = DataLoader._parse_pipeline(raw_val)

                # If pipeline is missing/NaN but pipeline_len exists, fall back per-row.
                if has_pipeline_len and DataLoader._is_missing_pipeline_value(raw_val):
                    pipeline = DataLoader._dummy_pipeline_from_len(int(pl_data[i]))
            else:
                pipeline = DataLoader._dummy_pipeline_from_len(int(pl_data[i]))

            states.append(
                State(
                    inventory=int(inv[i]),
                    backlog=int(back[i]),
                    pipeline=pipeline,
                    t=int(tt[i]),
                )
            )
        return states

    def _load_split(self, filename: str) -> DatasetSplit:
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path)
        if "trajectory_id" not in df.columns:
            raise ValueError(f"{filename} must contain `trajectory_id` column.")
        tids = sorted(df["trajectory_id"].unique().tolist())
        trajs: List[Trajectory] = []
        for tid in tids:
            states = self._parse_states(df, tid)
            actions = self._parse_actions(df, tid, num_states=len(states))
            trajs.append(Trajectory(trajectory_id=int(tid), states=states, actions=actions))
        return DatasetSplit(trajectories=trajs)

    def load(self) -> DataBundle:
        train = self._load_split("train_seed_10_n_100.csv")
        val = self._load_split("val_seed_10_n_100.csv")
        test = self._load_split("test_seed_10_n_100.csv")
        return DataBundle(train=train, val=val, test=test)


class BeerGameModel:
    PARAMETER_NAMES = [
        "demand_rate_raw",
        "demand_noise_scale",
        "order_logit",
        "leadtime_logit_1",
        "leadtime_logit_2",
        "leadtime_logit_3",
        "leadtime_logit_4",
        "leadtime_logit_5",
        "update_logit_0",
        "update_logit_1",
        "update_logit_2",
        "backlog_priority_logit",
    ]

    def __init__(self):
        self.theta = np.zeros((len(self.PARAMETER_NAMES),), dtype=np.float64)
        self._fixed_update_convention: Optional[int] = None

    @staticmethod
    def get_parameters_uniform_prior_min_max() -> Tuple[np.ndarray, np.ndarray]:
        prior_min = np.array([-2.0, 0.0, -2.0] + [-2.0] * 5 + [-2.0] * 3 + [-2.0], dtype=np.float64)
        prior_max = np.array([2.0, 5.0, 2.0] + [2.0] * 5 + [2.0] * 3 + [2.0], dtype=np.float64)
        return prior_min, prior_max

    def set_parameters_from_vector(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != self.theta.shape:
            raise ValueError(f"Theta must have shape {self.theta.shape}, got {theta.shape}.")
        self.theta = theta.copy()

    def get_parameters_named(self) -> Dict[str, float]:
        return {name: float(self.theta[i]) for i, name in enumerate(self.PARAMETER_NAMES)}

    def _sample_demand(self, t: int, rng: np.random.Generator) -> int:
        """Compatibility wrapper: sample demand using only time.

        The main model uses state-conditioning via `_sample_demand_conditioned`. When called
        without state context, we fall back to neutral conditioning (zeros).
        """
        return self._sample_demand_conditioned(
            t=t,
            inventory=0,
            backlog=0,
            pipeline_len=0,
            rng=rng,
        )

    def _sample_demand_conditioned(
        self,
        t: int,
        inventory: int,
        backlog: int,
        pipeline_len: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample exogenous customer demand as a nonnegative count (state-conditioned mean).

        Motivation
        ----------
        With only one observed trajectory for SBI, a purely time-driven latent demand process
        can be poorly identified and yields systematic backlog drift on unseen trajectories.
        We therefore introduce *weak*, bounded state-conditioning of the demand mean using only
        existing state variables and existing parameters.

        Model
        -----
        demand_t ~ Poisson(lambda_t),  lambda_t ~ Gamma(k, mean_t/k)   (Gamma-Poisson mixture)

        mean_t is based on:
          - a base mean controlled by demand_rate_raw
          - mild seasonality (very small)
          - bounded uplift when backlog is high (controlled by demand_noise_scale)
        """
        if t < 0:
            raise ValueError("t must be nonnegative.")
        if inventory < 0 or backlog < 0 or pipeline_len < 0:
            raise ValueError("inventory/backlog/pipeline_len must be nonnegative.")

        demand_rate_raw = float(self.theta[0])
        demand_noise_scale = float(self.theta[1])

        # Base mean: use exp mapping for a wider plausible range than softplus while remaining stable.
        rate_clipped = float(np.clip(demand_rate_raw, -6.0, 6.0))
        base_mean = math.exp(rate_clipped) * 10.0  # ~ [0.25, 4034] if unclipped; clipped keeps safe.
        base_mean = float(np.clip(base_mean, 1e-6, 200.0))

        # Very mild seasonality; keep small to avoid harming generalization.
        amp = 0.03 * math.tanh(max(0.0, demand_noise_scale) / 5.0)  # in [0, 0.03)
        season = 1.0 + amp * math.sin(2.0 * math.pi * (float(t) / 30.0))

        # Weak state-conditioning: backlog uplift (bounded, nonnegative).
        # Uses demand_noise_scale as a gain knob, so calibration can choose to use (or ignore) it.
        gain = 0.35 * math.tanh(max(0.0, demand_noise_scale) / 3.0)  # in [0, ~0.35)
        backlog_uplift = 1.0 + gain * math.tanh(float(backlog) / 25.0)  # in [1, 1+gain)

        # Small dependence on pipeline_len (also bounded) to help explain bursts correlated with ordering.
        pl_gain = 0.08 * math.tanh(max(0.0, demand_noise_scale) / 5.0)
        pl_centered = float(pipeline_len) - 3.0
        pipeline_factor = 1.0 + pl_gain * math.tanh(pl_centered / 4.0)

        mean_t = base_mean * season * backlog_uplift * pipeline_factor
        mean_t = float(np.clip(mean_t, 1e-6, 250.0))

        # Overdispersion via Gamma-Poisson.
        # Larger demand_noise_scale => smaller k => more dispersion, but keep bounded.
        k = 18.0 / (1.0 + 2.5 * max(0.0, demand_noise_scale))
        k = float(np.clip(k, 0.6, 50.0))

        lam = float(rng.gamma(shape=k, scale=mean_t / k))
        d = int(rng.poisson(lam=max(1e-6, lam)))
        return int(max(0, d))

    def _sample_lead_time(self, rng: np.random.Generator) -> int:
        """Sample lead time in {1,2,3,4,5} from a categorical distribution parameterized by logits."""
        logits = np.asarray(self.theta[3:8], dtype=np.float64)
        probs = stable_softmax(logits)
        lt = int(rng.choice(np.arange(1, 6, dtype=int), p=probs))
        if lt < 1 or lt > 5:
            raise RuntimeError(f"Sampled invalid lead time: {lt}")
        return lt

    def _select_update_convention(self, rng: np.random.Generator) -> int:
        """Select update convention in {0,1,2} (deterministic argmax)."""
        _ = rng  # deterministic selection
        logits = np.asarray(self.theta[8:11], dtype=np.float64)
        return int(np.argmax(logits))

    def _select_update_convention_deterministic(self) -> int:
        """Deterministically select update convention via argmax."""
        logits = np.asarray(self.theta[8:11], dtype=np.float64)
        return int(np.argmax(logits))

    def _select_update_convention_stochastic(self, rng: np.random.Generator) -> int:
        """Sample update convention in {0,1,2} from the softmax probabilities."""
        logits = np.asarray(self.theta[8:11], dtype=np.float64)
        probs = stable_softmax(logits)
        return int(rng.choice(np.arange(3, dtype=int), p=probs))

    def _fulfill(self, inventory: int, backlog: int, demand: int, rng: np.random.Generator) -> Tuple[int, int]:
        """Fulfillment and backlog update using conserved shipped total.

        Blueprint-consistent accounting:
        - total_needed = backlog + demand
        - shipped_total = min(inventory, total_needed)
        - inventory decreases by shipped_total
        - backlog becomes total_needed - shipped_total

        Note: backlog_priority_logit does not affect (inventory, backlog) under this
        aggregated state representation, but the parameter is retained for compatibility.
        """
        if inventory < 0 or backlog < 0 or demand < 0:
            raise ValueError("inventory/backlog/demand must be nonnegative.")
        _ = rng  # kept for signature compatibility

        inv = int(inventory)
        total_needed = int(backlog) + int(demand)
        shipped_total = min(inv, total_needed)

        inv2 = inv - shipped_total
        new_backlog = total_needed - shipped_total

        if inv2 < 0 or new_backlog < 0:
            raise RuntimeError("Invariant violated: negative inventory/backlog after fulfillment.")
        return int(inv2), int(new_backlog)

    def step(self, state: State, action: Action, rng: np.random.Generator) -> State:
        """Advance the environment by one step given a state and an order action."""
        if state.inventory < 0 or state.backlog < 0:
            raise ValueError("State inventory/backlog must be nonnegative.")
        if any((q < 0 or tau < 0) for (q, tau) in state.pipeline):
            raise ValueError("State pipeline entries must be nonnegative.")

        if self._fixed_update_convention is None:
            convention = self._select_update_convention(rng)
        else:
            convention = int(self._fixed_update_convention)

        inventory = int(state.inventory)
        backlog = int(state.backlog)

        pipeline: List[Tuple[int, int]] = [(int(q), int(tau)) for (q, tau) in state.pipeline]
        new_pipeline: List[Tuple[int, int]] = []

        def do_arrivals() -> None:
            """Progress pipeline countdown and deliver arrivals into inventory."""
            nonlocal inventory, new_pipeline
            arrived = 0
            progressed: List[Tuple[int, int]] = []
            current = pipeline if not new_pipeline else new_pipeline
            for q, tau in current:
                if tau <= 1:
                    arrived += q
                else:
                    progressed.append((q, tau - 1))
            inventory += int(arrived)
            new_pipeline = progressed

        def do_demand() -> None:
            """Sample demand and update inventory/backlog via fulfillment."""
            nonlocal inventory, backlog
            pipeline_len = len(pipeline if not new_pipeline else new_pipeline)
            demand = self._sample_demand_conditioned(
                t=int(state.t),
                inventory=int(inventory),
                backlog=int(backlog),
                pipeline_len=int(pipeline_len),
                rng=rng,
            )
            inventory, backlog = self._fulfill(inventory, backlog, demand, rng)

        def do_order() -> None:
            """Append the provided action as a new pipeline entry (ACTION-FIRST)."""
            nonlocal new_pipeline
            lt = self._sample_lead_time(rng)

            if action is None:
                order_logit = float(self.theta[2])
                order_amt = int(np.clip(np.round(softplus(order_logit) * 5.0), 0, 500))
            else:
                order_amt = int(np.clip(int(action.order), 0, 500))

            if order_amt > 0:
                new_pipeline.append((order_amt, lt))

        # Three distinct update conventions aligned to blueprint:
        # 0: arrivals_then_demand_then_order
        # 1: demand_then_arrivals_then_order
        # 2: arrivals_then_order_then_demand
        if convention == 0:
            do_arrivals()
            do_demand()
            do_order()
        elif convention == 1:
            do_demand()
            do_arrivals()
            do_order()
        else:
            do_arrivals()
            do_order()
            do_demand()

        # Preserve insertion order; do NOT sort pipeline entries.
        new_pipeline_tuple = tuple((int(q), int(tau)) for (q, tau) in new_pipeline)
        return State(
            inventory=int(inventory),
            backlog=int(backlog),
            pipeline=new_pipeline_tuple,
            t=int(state.t + 1),
        )


class BeerGameSimulator:
    def __init__(self, model: BeerGameModel, global_seed: int):
        self.model = model
        self.global_seed = int(global_seed)
        # Internal control: use stochastic convention selection during evaluation rollouts.
        self.stochastic_update_convention: bool = True

    def rollout_trajectory(
        self,
        init_state: State,
        actions: Sequence[Action],
        rng: np.random.Generator,
    ) -> List[State]:
        """Roll out a single trajectory from init_state for len(actions) steps."""
        original_convention = self.model._fixed_update_convention
        try:
            # Choose update convention once per trajectory and reuse across steps.
            if self.stochastic_update_convention:
                self.model._fixed_update_convention = self.model._select_update_convention_stochastic(rng)
            else:
                self.model._fixed_update_convention = self.model._select_update_convention_deterministic()

            T = len(actions)
            states = [init_state]
            s = init_state
            for _t in range(T):
                s = self.model.step(s, actions[_t], rng)
                states.append(s)
            return states
        finally:
            self.model._fixed_update_convention = original_convention

    def rollout(
        self, splits: Dict[str, DatasetSplit], stochastic_rollouts_per_traj: int = 1
    ) -> Dict[str, Dict[int, List[State]]]:
        """Roll out predictions for train/val/test splits (one rollout per trajectory)."""
        _ = stochastic_rollouts_per_traj  # contract keeps parameter; current implementation uses 1
        preds: Dict[str, Dict[int, List[State]]] = {"train": {}, "val": {}, "test": {}}
        for split_name, split in splits.items():
            for traj in split.trajectories:
                rng = np.random.default_rng(self.global_seed + traj.trajectory_id + 1000)
                sim_states = self.rollout_trajectory(init_state=traj.states[0], actions=traj.actions, rng=rng)
                preds[split_name][traj.trajectory_id] = sim_states
        return preds


def trajectories_to_numpy(states: Sequence[State]) -> np.ndarray:
    """Encode a trajectory into the official (T+1,4) numeric array."""
    arr = np.zeros((len(states), 4), dtype=np.float64)
    for i, s in enumerate(states):
        arr[i, 0] = float(s.inventory)
        arr[i, 1] = float(s.backlog)
        arr[i, 2] = float(len(s.pipeline))
        arr[i, 3] = float(s.t)
    return arr


class Evaluator:
    def compute_metrics(self, data: DataBundle, preds: Dict[str, Dict[int, List[State]]]) -> Dict[str, Any]:
        """Compute train/val/test losses and per-dim validation loss (must match baseline contract)."""

        def split_loss(split: DatasetSplit, pred_split: Dict[int, List[State]]) -> Tuple[float, np.ndarray]:
            losses = []
            per_dim = np.zeros((4,), dtype=np.float64)
            n = 0
            for traj in split.trajectories:
                true_np = trajectories_to_numpy(traj.states)
                pred_np = trajectories_to_numpy(pred_split[traj.trajectory_id])
                m = min(true_np.shape[0], pred_np.shape[0])
                diff = true_np[:m] - pred_np[:m]
                per_dim += np.mean(diff * diff, axis=0)
                losses.append(float(np.mean(diff * diff)))
                n += 1
            per_dim = per_dim / max(1, n)
            return float(np.mean(losses)), per_dim

        train_loss, _ = split_loss(data.train, preds["train"])
        val_loss, val_per_dim = split_loss(data.val, preds["val"])
        test_loss, _ = split_loss(data.test, preds["test"])

        return {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "val_loss_per_dim": {
                "inventory_mse": float(val_per_dim[0]),
                "backlog_mse": float(val_per_dim[1]),
                "pipeline_len_mse": float(val_per_dim[2]),
                "t_mse": float(val_per_dim[3]),
            },
        }


class Calibrator(abc.ABC):
    @abc.abstractmethod
    def fit(self, train_split: DatasetSplit, observation_trajectory_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError


class ConditionalGaussianNPE(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * theta_dim),
        )
        self.theta_dim = int(theta_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean = out[:, : self.theta_dim]
        log_std = out[:, self.theta_dim :]
        log_std = torch.clamp(log_std, min=-6.0, max=3.0)
        return mean, log_std

    @staticmethod
    def nll(theta: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        return 0.5 * torch.mean(((theta - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))

    def sample(self, x: torch.Tensor, num_samples: int, rng_seed: int) -> torch.Tensor:
        torch.manual_seed(int(rng_seed))
        mean, log_std = self.forward(x.repeat(num_samples, 1))
        eps = torch.randn_like(mean)
        return mean + torch.exp(log_std) * eps


class NPECalibrator(Calibrator):
    def __init__(
        self,
        model_factory,
        simulator_factory,
        num_simulations: int,
        num_posterior_samples: int,
        training_steps: int,
        batch_size: int,
        lr: float,
        seed: int,
        device: Optional[str] = None,
    ):
        self.model_factory = model_factory
        self.simulator_factory = simulator_factory
        self.num_simulations = int(num_simulations)
        self.num_posterior_samples = int(num_posterior_samples)
        self.training_steps = int(training_steps)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

    @staticmethod
    def _encode_for_calibration(states: Sequence[State]) -> np.ndarray:
        """Encode states for SBI calibration using the official flattened trajectory encoding."""
        base = trajectories_to_numpy(states).astype(np.float32)  # (T+1,4)
        flat = base.reshape(-1).astype(np.float32)
        return flat

    def fit(self, train_split: DatasetSplit, observation_trajectory_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        if observation_trajectory_id < 0 or observation_trajectory_id >= len(train_split.trajectories):
            raise ValueError("Invalid observation trajectory id.")

        obs_traj = train_split.trajectories[observation_trajectory_id]
        obs_T = len(obs_traj.actions)
        if len(obs_traj.states) != obs_T + 1:
            raise ValueError("Observation trajectory states length must be T+1.")
        if len(obs_traj.actions) != obs_T:
            raise ValueError("Observation trajectory actions length must be T.")

        obs_vec = self._encode_for_calibration(obs_traj.states)
        x_o = torch.tensor(obs_vec, dtype=torch.float32).reshape(1, -1).to(self.device)

        prior_min, prior_max = BeerGameModel.get_parameters_uniform_prior_min_max()
        theta_dim = int(prior_min.shape[0])
        prior_min_t = torch.tensor(prior_min, dtype=torch.float32, device=self.device)
        prior_max_t = torch.tensor(prior_max, dtype=torch.float32, device=self.device)
        prior_span_t = (prior_max_t - prior_min_t).clamp_min(1e-6)

        x_dim = x_o.shape[1]
        thetas_u = torch.zeros((self.num_simulations, theta_dim), dtype=torch.float32, device=self.device)
        xs = torch.zeros((self.num_simulations, x_dim), dtype=torch.float32, device=self.device)

        init_state = obs_traj.states[0]
        actions = obs_traj.actions

        for i in range(self.num_simulations):
            rng = np.random.default_rng(self.seed + 10_000 + i)
            u = torch.rand((theta_dim,), dtype=torch.float32, device=self.device)
            theta_i = prior_min_t + prior_span_t * u

            thetas_u[i] = u
            model = self.model_factory()
            model.set_parameters_from_vector(theta_i.detach().cpu().numpy())

            simulator = self.simulator_factory(model=model, seed=self.seed + i)
            # Reduce training noise: keep update convention selection deterministic during SBI pair generation.
            simulator.stochastic_update_convention = False

            sim_states = simulator.rollout_trajectory(init_state=init_state, actions=actions, rng=rng)
            sim_vec = self._encode_for_calibration(sim_states)
            x_i = torch.tensor(sim_vec, dtype=torch.float32).reshape(-1).to(self.device)
            xs[i] = x_i

        x_mean = xs.mean(dim=0, keepdim=True)
        x_std = xs.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        xs_n = (xs - x_mean) / x_std
        x_o_n = (x_o - x_mean) / x_std

        net = ConditionalGaussianNPE(x_dim=x_dim, theta_dim=theta_dim, hidden=256).to(self.device)
        opt = optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        torch.manual_seed(self.seed)
        n = xs.shape[0]
        for _step in range(self.training_steps):
            idx = torch.randint(low=0, high=n, size=(self.batch_size,), device=self.device)
            x_b = xs_n[idx]
            theta_b = thetas_u[idx]

            mean, log_std = net(x_b)
            loss = ConditionalGaussianNPE.nll(theta_b, mean, log_std)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()

        posterior_samples_u = net.sample(
            x=x_o_n,
            num_samples=self.num_posterior_samples,
            rng_seed=self.seed + 999,
        ).clamp(0.0, 1.0)

        posterior_mean_u = posterior_samples_u.mean(dim=0)
        posterior_std_u = posterior_samples_u.std(dim=0, unbiased=False)

        posterior_mean = (prior_min_t + prior_span_t * posterior_mean_u).detach().cpu().numpy()
        posterior_std = (prior_span_t * posterior_std_u).detach().cpu().numpy()

        artifacts = {
            "observation_trajectory_id": int(observation_trajectory_id),
            "observation_T": int(obs_T),
            "x_o_shape": [int(x_dim)],
            "num_simulations": int(self.num_simulations),
            "num_posterior_samples": int(self.num_posterior_samples),
            "posterior_samples_mean": {BeerGameModel.PARAMETER_NAMES[i]: float(posterior_mean[i]) for i in range(theta_dim)},
            "posterior_samples_std": {BeerGameModel.PARAMETER_NAMES[i]: float(posterior_std[i]) for i in range(theta_dim)},
        }

        return posterior_mean, artifacts


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Beer Game SUPPLY simulator + SBI(NPE) calibrator.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write outputs.")
    parser.add_argument("--num_simulations", type=int, default=1500)
    parser.add_argument("--posterior_samples", type=int, default=2000)
    parser.add_argument("--training_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args(argv)


def load_data() -> DataBundle:
    """
    Load train/val/test Beer Game trajectory data from CSV files.

    Notes
    -----
    Path handling MUST follow the integration-required snippet exactly.
    """
    # Path Handling Instructions (COPY EXACTLY)
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    DATA_PATH = os.environ.get("DATA_PATH")

    if not PROJECT_ROOT:
        raise EnvironmentError("Environment variable PROJECT_ROOT must be set.")
    if not DATA_PATH:
        raise EnvironmentError("Environment variable DATA_PATH must be set.")

    DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

    data_dir = os.path.abspath(DATA_DIR)
    loader = DataLoader(data_dir=data_dir)
    return loader.load()


def build_network_and_agents(
    args: argparse.Namespace,
) -> Tuple[BeerGameSimulator, NPECalibrator, Evaluator]:
    """Factory wiring for simulator, calibrator, and evaluator."""

    def model_factory() -> BeerGameModel:
        return BeerGameModel()

    def simulator_factory(model: BeerGameModel, seed: int) -> BeerGameSimulator:
        return BeerGameSimulator(model=model, global_seed=seed)

    calibrator = NPECalibrator(
        model_factory=model_factory,
        simulator_factory=simulator_factory,
        num_simulations=int(args.num_simulations),
        num_posterior_samples=int(args.posterior_samples),
        training_steps=int(args.training_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    simulator = BeerGameSimulator(model=model_factory(), global_seed=int(args.seed))
    evaluator = Evaluator()
    return simulator, calibrator, evaluator


def holdout_split(data: DataBundle) -> Dict[str, DatasetSplit]:
    """Return the fixed train/val/test splits."""
    return {"train": data.train, "val": data.val, "test": data.test}


def save_results(
    output_dir: str,
    optimized_parameters: Dict[str, float],
    calibration_artifacts: Dict[str, Any],
    metrics: Dict[str, Any],
    val_predictions: Dict[int, List[State]],
) -> None:
    """Save results.json and validation prediction CSV (contract-preserving)."""
    ensure_dir(output_dir)

    results = {
        "optimized_parameters": optimized_parameters,
        "calibration_artifacts": calibration_artifacts,
        "metrics": metrics,
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    rows: List[Dict[str, Any]] = []
    for tid in sorted(val_predictions.keys()):
        states = val_predictions[tid]
        for time_step, s in enumerate(states):
            rows.append(
                {
                    "trajectory_id": int(tid),
                    "time_step": int(time_step),
                    "inventory": int(s.inventory),
                    "backlog": int(s.backlog),
                    "pipeline_len": int(len(s.pipeline)),
                    "t": int(s.t),
                }
            )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "val_predicted_mean_trajectories.csv")
    df.to_csv(csv_path, index=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point: calibrate on train trajectory 0, then evaluate all splits."""
    args = parse_cli(argv)
    set_global_determinism(int(args.seed))

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    data = load_data()
    simulator, calibrator, evaluator = build_network_and_agents(args)
    splits = holdout_split(data)

    optimized_theta, calibration_artifacts = calibrator.fit(
        train_split=splits["train"],
        observation_trajectory_id=0,
    )

    simulator.model.set_parameters_from_vector(optimized_theta)
    optimized_parameters = simulator.model.get_parameters_named()

    preds = simulator.rollout(splits=splits, stochastic_rollouts_per_traj=1)
    metrics = evaluator.compute_metrics(data=data, preds=preds)

    save_results(
        output_dir=output_dir,
        optimized_parameters=optimized_parameters,
        calibration_artifacts=calibration_artifacts,
        metrics=metrics,
        val_predictions=preds["val"],
    )


main()