from __future__ import annotations

PLAYBOOK_USAGE_JSON = '''{"used_bullets":[{"id":"calibration-not-used-in-evaluation-argmax-update-convention","why":"Update convention was a hard argmax over logits, hurting identifiability; switch to categorical sampling from softmax using rng to reflect the calibratable discrete choice."},{"id":"latent-demand-model-mismatch-poisson-with-additive-gaussian-noise","why":"Demand used Poisson plus additive Gaussian noise on counts, causing mismatch and poor identifiability; replace with a proper count model with intrinsic dispersion (Poisson/NB via demand_noise_scale)."},{"id":"lead-time-and-pipeline-length-mismatch-ignores-empirical-countdown-structure","why":"Lead-time probabilities were unconstrained and weakly identified; add a mild bias toward shorter lead times while preserving the same logits/normalization and countdown convention."},{"id":"sbi-training-objective-mismatch-uses-single-trajectory-without-action-variation-coverage","why":"Single-trajectory SBI produced near-prior posteriors; improve NPE conditioning by using a richer calibration-only encoder (include first differences) and feature normalization, without changing official evaluation encoding."},{"id":"unused-backlog-priority-parameter","why":"backlog_priority_logit had no effect on trajectories; implement a service-order/cap mechanism so it influences inventory/backlog transitions and becomes identifiable."}]}'''

CHANGE_SUMMARY_JSON = '''{"touched_symbols":[{"symbol":"load_data","reason":"Must use integration-required PROJECT_ROOT/DATA_PATH/DATA_DIR path handling snippet exactly, with validation."},{"symbol":"BeerGameModel._sample_demand","reason":"Replace Poisson+Gaussian with a proper count process (Poisson/NB) using demand_noise_scale as overdispersion."},{"symbol":"BeerGameModel._select_update_convention","reason":"Use rng-driven categorical sampling from softmax over logits instead of hard argmax."},{"symbol":"BeerGameModel._sample_lead_time","reason":"Keep categorical over {1..5} but add a mild bias toward shorter lead times to stabilize/anchor inference."},{"symbol":"BeerGameModel._fulfill","reason":"Make backlog_priority_logit affect dynamics via probabilistic service order and capped backlog service when demand-first."},{"symbol":"BeerGameModel.step","reason":"Thread rng into update convention selection and fulfillment; preserve invariants and time progression."},{"symbol":"NPECalibrator.fit","reason":"Add calibration-only richer encoder (include deltas) and normalize x/theta to improve posterior learning while leaving official evaluation untouched."},{"symbol":"ConditionalGaussianNPE.forward","reason":"Slightly tighten log_std clamp to reduce pathological huge posterior std when training is uncertain."}],"applied_strategies":[{"id":"calibration-not-used-in-evaluation-argmax-update-convention","applied":true},{"id":"latent-demand-model-mismatch-poisson-with-additive-gaussian-noise","applied":true},{"id":"lead-time-and-pipeline-length-mismatch-ignores-empirical-countdown-structure","applied":true},{"id":"sbi-training-objective-mismatch-uses-single-trajectory-without-action-variation-coverage","applied":true},{"id":"unused-backlog-priority-parameter","applied":true}]}'''

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(False)
        except TypeError:
            pass


def softplus(x: float) -> float:
    if x > 30:
        return x
    return float(np.log1p(np.exp(x)))


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits)
    ex = np.exp(logits - m)
    s = ex.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"Invalid softmax normalization encountered. logits={logits}")
    return ex / s


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sigmoid(x: float) -> float:
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


@dataclass
class Trajectory:
    trajectory_id: int
    states: List[State]
    actions: List[Optional[int]]


@dataclass
class DatasetSplit:
    name: str
    trajectories: List[Trajectory]

    @property
    def trajectory_ids(self) -> List[int]:
        return [tr.trajectory_id for tr in self.trajectories]


@dataclass
class DataBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


class DataLoader:
    REQUIRED_COLUMNS = {"trajectory_id", "inventory", "backlog", "pipeline"}
    TIME_COLUMNS = ("time_step", "t")
    ACTION_COLUMNS = ("action", "order", "order_qty", "order_quantity")

    def __init__(self, data_dir: str):
        if not os.path.isabs(data_dir):
            raise ValueError(f"DATA_DIR must be an absolute path. Got: {data_dir}")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"DATA_DIR does not exist or is not a directory: {data_dir}")
        self.data_dir = data_dir

    def _parse_pipeline(self, value: Any, context: str) -> Tuple[Tuple[int, int], ...]:
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
                raise ValueError(f"Failed to parse pipeline at {context}. value={value!r}") from exc

        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Pipeline must parse to a list/tuple at {context}. Got: {type(parsed)}")

        out: List[Tuple[int, int]] = []
        for i, item in enumerate(parsed):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise ValueError(
                    f"Pipeline entry must be a 2-tuple at {context}, index {i}. Got: {item!r}"
                )
            q, tau = item
            try:
                q_i = int(q)
                tau_i = int(tau)
            except Exception as exc:
                raise ValueError(
                    f"Pipeline entry values must be ints at {context}, index {i}. Got: {item!r}"
                ) from exc
            if q_i < 0 or tau_i < 0:
                raise ValueError(
                    f"Pipeline entry must be nonnegative at {context}, index {i}. Got: {item!r}"
                )
            out.append((q_i, tau_i))
        return tuple(out)

    def _find_time_col(self, df: pd.DataFrame, filename: str) -> str:
        for col in self.TIME_COLUMNS:
            if col in df.columns:
                return col
        raise ValueError(
            f"{filename} must contain one of {self.TIME_COLUMNS} to index time. "
            f"Available columns: {list(df.columns)}"
        )

    def _find_action_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in self.ACTION_COLUMNS:
            if col in df.columns:
                return col
        return None

    def _load_csv(self, filename: str, split_name: str) -> DatasetSplit:
        path = os.path.join(self.data_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required data file: {path}")

        df = pd.read_csv(path)
        missing = sorted(self.REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")

        time_col = self._find_time_col(df, filename)
        action_col = self._find_action_col(df)

        for col in ["trajectory_id", time_col, "inventory", "backlog"]:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

        df = df.sort_values(["trajectory_id", time_col], kind="mergesort")

        trajectories: List[Trajectory] = []
        for tid, g in df.groupby("trajectory_id", sort=True):
            g = g.copy()
            times = g[time_col].to_numpy()
            if len(times) < 2:
                raise ValueError(f"trajectory_id={tid} in {filename} must have >=2 rows.")
            if not np.array_equal(times, np.arange(times.min(), times.min() + len(times))):
                raise ValueError(
                    f"trajectory_id={tid} in {filename} has non-consecutive {time_col} values. "
                    f"Got: {times[:10]}... (len={len(times)})"
                )

            states: List[State] = []
            actions: List[Optional[int]] = []

            for row in g.itertuples(index=False):
                row_dict = row._asdict()
                inventory = int(row_dict["inventory"])
                backlog = int(row_dict["backlog"])
                t_val = int(row_dict.get("t", row_dict[time_col]))
                pipeline = self._parse_pipeline(
                    row_dict["pipeline"],
                    context=f"{filename} tid={tid} {time_col}={row_dict[time_col]}",
                )

                if inventory < 0 or backlog < 0:
                    raise ValueError(
                        f"inventory/backlog must be nonnegative. Found at tid={tid}, "
                        f"{time_col}={row_dict[time_col]}: inventory={inventory}, backlog={backlog}"
                    )

                states.append(State(inventory=inventory, backlog=backlog, pipeline=pipeline, t=t_val))

                if action_col is not None:
                    a_int = int(row_dict[action_col])
                    if a_int < 0:
                        raise ValueError(
                            f"Action/order must be nonnegative. Found {a_int} at "
                            f"{filename} tid={tid} {time_col}={row_dict[time_col]}"
                        )
                    actions.append(a_int)
                else:
                    actions.append(None)

            actions_T = actions[:-1]
            trajectories.append(Trajectory(trajectory_id=int(tid), states=states, actions=actions_T))

        horizons = {len(tr.states) for tr in trajectories}
        if len(horizons) != 1:
            raise ValueError(f"Inconsistent trajectory lengths in split={split_name}: {sorted(horizons)}")
        return DatasetSplit(name=split_name, trajectories=trajectories)

    def load(self) -> DataBundle:
        train = self._load_csv("train_seed_10_n_100.csv", "train")
        val = self._load_csv("val_seed_10_n_100.csv", "val")
        test = self._load_csv("test_seed_10_n_100.csv", "test")
        return DataBundle(train=train, val=val, test=test)


class BeerGameModel:
    PARAMETER_NAMES: Tuple[str, ...] = (
        "demand_rate_raw",
        "demand_noise_scale",
        "shipment_quantity_noise_scale",
        "leadtime_logits[0]",
        "leadtime_logits[1]",
        "leadtime_logits[2]",
        "leadtime_logits[3]",
        "leadtime_logits[4]",
        "update_convention_logits[0]",
        "update_convention_logits[1]",
        "update_convention_logits[2]",
        "backlog_priority_logit",
    )

    def __init__(self) -> None:
        self.theta = np.zeros(len(self.PARAMETER_NAMES), dtype=np.float64)

    @classmethod
    def get_parameters_uniform_prior_min_max(cls) -> Tuple[np.ndarray, np.ndarray]:
        prior_min = np.array(
            [
                0.0,
                0.0,
                0.0,
                -3.0,
                -3.0,
                -3.0,
                -3.0,
                -3.0,
                -3.0,
                -3.0,
                -3.0,
                -3.0,
            ],
            dtype=np.float64,
        )
        prior_max = np.array(
            [
                5.0,
                10.0,
                5.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
            ],
            dtype=np.float64,
        )
        if prior_min.shape != prior_max.shape:
            raise RuntimeError("Prior bounds have inconsistent shapes.")
        return prior_min, prior_max

    def set_parameters_from_vector(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (len(self.PARAMETER_NAMES),):
            raise ValueError(
                f"theta must have shape ({len(self.PARAMETER_NAMES)},) but got {theta.shape}"
            )
        self.theta = theta.copy()

    def get_parameters_named(self) -> Dict[str, float]:
        return {name: float(self.theta[i]) for i, name in enumerate(self.PARAMETER_NAMES)}

    def _sample_demand(self, t: int, rng: np.random.Generator) -> int:
        demand_rate_raw = float(self.theta[0])
        demand_noise_scale = float(self.theta[1])

        mu = softplus(demand_rate_raw) * 10.0
        mu = float(max(1e-6, mu))

        if demand_noise_scale <= 1e-8:
            return int(rng.poisson(lam=mu))

        # Gamma-Poisson mixture (Negative Binomial) that supports non-integer dispersion.
        k = 1.0 / max(1e-6, float(demand_noise_scale))  # shape (can be non-integer)
        k = float(max(1e-6, k))
        scale = mu / k  # mean of gamma = k*scale = mu
        lam = float(rng.gamma(shape=k, scale=scale))
        d = int(rng.poisson(lam=max(0.0, lam)))
        return int(max(0, d))

    def _sample_lead_time(self, rng: np.random.Generator) -> int:
        logits = np.asarray(self.theta[3:8], dtype=np.float64)
        lead_times = np.arange(1, 6, dtype=np.float64)
        bias = -0.35 * (lead_times - 1.0)
        probs = stable_softmax(logits + bias)

        lt = int(rng.choice(np.arange(1, 6, dtype=int), p=probs))
        if lt < 1 or lt > 5:
            raise RuntimeError(f"Sampled invalid lead time: {lt}")
        return lt

    def _select_update_convention(self, rng: np.random.Generator) -> int:
        logits = np.asarray(self.theta[8:11], dtype=np.float64)
        temperature = 1.75
        probs = stable_softmax(logits * temperature)
        return int(rng.choice(np.arange(3, dtype=int), p=probs))

    def _fallback_action(self, state: State, rng: np.random.Generator) -> int:
        base_stock_level_S = 50
        inv = state.inventory
        backlog = state.backlog
        pipeline_qty = sum(q for (q, _tau) in state.pipeline)
        net = inv + pipeline_qty - backlog
        order = max(0, base_stock_level_S - net)
        return int(order)

    def _apply_arrivals(self, pipeline: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
        new_pipe: List[Tuple[int, int]] = []
        arrivals = 0
        for q, tau in pipeline:
            tau2 = tau - 1
            if tau2 <= 0:
                arrivals += q
            else:
                new_pipe.append((q, tau2))
        if arrivals < 0:
            raise RuntimeError("Arrivals became negative unexpectedly.")
        return new_pipe, arrivals

    def _place_order(
        self,
        pipeline: List[Tuple[int, int]],
        action: Optional[int],
        state: State,
        rng: np.random.Generator,
    ) -> List[Tuple[int, int]]:
        shipment_noise_scale = float(self.theta[2])
        if action is None:
            action_int = self._fallback_action(state, rng)
        else:
            action_int = int(action)

        if action_int < 0:
            raise ValueError(f"Action must be nonnegative. Got {action_int}")

        qty = float(action_int)
        if shipment_noise_scale > 0:
            qty = qty + rng.normal(loc=0.0, scale=shipment_noise_scale)
        qty_int = int(np.clip(np.round(qty), 0, None))

        lt = self._sample_lead_time(rng)
        pipeline.append((qty_int, lt))
        return pipeline

    def _fulfill(
        self, inventory: int, backlog: int, demand: int, rng: np.random.Generator
    ) -> Tuple[int, int]:
        if inventory < 0 or backlog < 0 or demand < 0:
            raise ValueError("inventory/backlog/demand must be nonnegative.")

        f = sigmoid(float(self.theta[11]))

        inv = int(inventory)
        B = int(backlog)
        D = int(demand)

        if rng.random() < f:
            ship_b = min(inv, B)
            inv -= ship_b
            B_rem = B - ship_b

            ship_d = min(inv, D)
            inv -= ship_d
            D_unmet = D - ship_d

            new_backlog = B_rem + D_unmet
            return int(inv), int(new_backlog)

        ship_d = min(inv, D)
        inv -= ship_d
        D_unmet = D - ship_d

        cap = int(math.floor(f * inv + 1e-9))
        ship_b = min(B, cap)
        inv -= ship_b
        B_rem = B - ship_b

        new_backlog = B_rem + D_unmet
        if inv < 0 or new_backlog < 0:
            raise RuntimeError("Invariant violated: negative inventory/backlog after fulfillment.")
        return int(inv), int(new_backlog)

    def step(self, state: State, action: Optional[int], rng: np.random.Generator) -> State:
        if state.inventory < 0 or state.backlog < 0:
            raise ValueError("State inventory/backlog must be nonnegative.")
        if any(q < 0 or tau < 0 for (q, tau) in state.pipeline):
            raise ValueError("State pipeline entries must be nonnegative.")

        convention = self._select_update_convention(rng)

        inventory = int(state.inventory)
        backlog = int(state.backlog)
        pipeline = [(int(q), int(tau)) for (q, tau) in state.pipeline]

        def do_arrivals() -> None:
            nonlocal inventory, pipeline
            pipeline, arrivals = self._apply_arrivals(pipeline)
            inventory += arrivals

        def do_demand_and_fulfill() -> None:
            nonlocal inventory, backlog
            demand = self._sample_demand(state.t, rng)
            inventory, backlog = self._fulfill(inventory, backlog, demand, rng)

        def do_order() -> None:
            nonlocal pipeline
            pipeline = self._place_order(pipeline, action, state, rng)

        if convention == 0:
            do_arrivals()
            do_demand_and_fulfill()
            do_order()
        elif convention == 1:
            do_demand_and_fulfill()
            do_arrivals()
            do_order()
        elif convention == 2:
            do_arrivals()
            do_order()
            do_demand_and_fulfill()
        else:
            raise RuntimeError(f"Invalid update convention id: {convention}")

        if inventory < 0 or backlog < 0:
            raise RuntimeError("Invariant violated: negative inventory/backlog after step.")
        for q, tau in pipeline:
            if q < 0 or tau < 0:
                raise RuntimeError("Invariant violated: negative pipeline entry after step.")

        return State(inventory=inventory, backlog=backlog, pipeline=tuple(pipeline), t=int(state.t + 1))


def trajectories_to_numpy(states: Sequence[State]) -> np.ndarray:
    if len(states) == 0:
        raise ValueError("states must be non-empty.")
    out = np.zeros((len(states), 4), dtype=np.float32)
    for i, s in enumerate(states):
        out[i, 0] = float(s.inventory)
        out[i, 1] = float(s.backlog)
        out[i, 2] = float(len(s.pipeline))
        out[i, 3] = float(s.t)
    return out


class BeerGameSimulator:
    _SPLIT_SEED_OFFSET: Dict[str, int] = {"train": 1, "val": 2, "test": 3}

    def __init__(self, model: BeerGameModel, global_seed: int = GLOBAL_SEED):
        self.model = model
        self.global_seed = int(global_seed)

    def rollout_trajectory(
        self, init_state: State, actions: Sequence[Optional[int]], rng: np.random.Generator
    ) -> List[State]:
        if init_state is None:
            raise ValueError("init_state must be provided.")
        T = len(actions)
        states = [init_state]
        s = init_state
        for _t in range(T):
            s = self.model.step(s, actions[_t], rng)
            states.append(s)
        return states

    def rollout(
        self,
        splits: Mapping[str, DatasetSplit],
        stochastic_rollouts_per_traj: int = 1,
    ) -> Dict[str, Dict[int, List[State]]]:
        if stochastic_rollouts_per_traj != 1:
            raise ValueError(
                "This implementation supports stochastic_rollouts_per_traj=1 for the output contract. "
                "Extend as needed if K>1 is required."
            )

        preds: Dict[str, Dict[int, List[State]]] = {}
        for split_name, split in splits.items():
            split_preds: Dict[int, List[State]] = {}
            split_offset = self._SPLIT_SEED_OFFSET.get(str(split_name), 0)
            for tr in split.trajectories:
                seed = self.global_seed + 1000 * int(split_offset) + int(tr.trajectory_id)
                rng = np.random.default_rng(seed)
                init = tr.states[0]
                actions = tr.actions
                if len(tr.states) != len(actions) + 1:
                    raise ValueError(
                        f"Trajectory horizon mismatch for {split_name} tid={tr.trajectory_id}: "
                        f"len(states)={len(tr.states)} len(actions)={len(actions)}"
                    )
                sim_states = self.rollout_trajectory(init, actions, rng)
                if len(sim_states) != len(tr.states):
                    raise RuntimeError("Simulator produced wrong number of states.")
                split_preds[int(tr.trajectory_id)] = sim_states
            preds[str(split_name)] = split_preds
        return preds


class Evaluator:
    FEATURE_NAMES = ("inventory", "backlog", "pipeline_len", "t")

    @staticmethod
    def _mse(true_arr: np.ndarray, pred_arr: np.ndarray) -> float:
        if true_arr.shape != pred_arr.shape:
            raise ValueError(f"MSE shape mismatch: true={true_arr.shape} pred={pred_arr.shape}")
        diff = true_arr.astype(np.float64) - pred_arr.astype(np.float64)
        return float(np.mean(diff * diff))

    @staticmethod
    def compute_metrics(
        data: DataBundle,
        preds: Dict[str, Dict[int, List[State]]],
    ) -> Dict[str, Any]:
        split_map = {"train": data.train, "val": data.val, "test": data.test}

        mse_by_split: Dict[str, float] = {}
        per_dim_val: Optional[np.ndarray] = None

        for split_name, split in split_map.items():
            if split_name not in preds:
                raise ValueError(f"Missing predictions for split '{split_name}'.")
            all_true: List[np.ndarray] = []
            all_pred: List[np.ndarray] = []

            for tr in split.trajectories:
                tid = int(tr.trajectory_id)
                if tid not in preds[split_name]:
                    raise ValueError(f"Missing prediction for {split_name} trajectory_id={tid}")
                true_np = trajectories_to_numpy(tr.states)
                pred_np = trajectories_to_numpy(preds[split_name][tid])
                all_true.append(true_np)
                all_pred.append(pred_np)

            true_stack = np.concatenate(all_true, axis=0)
            pred_stack = np.concatenate(all_pred, axis=0)
            mse = Evaluator._mse(true_stack, pred_stack)
            mse_by_split[split_name] = mse

            if split_name == "val":
                diff = true_stack.astype(np.float64) - pred_stack.astype(np.float64)
                per_dim_val = np.mean(diff * diff, axis=0)

        if per_dim_val is None or per_dim_val.shape != (4,):
            raise RuntimeError("Failed to compute validation per-dimension loss.")

        return {
            "train_loss": float(mse_by_split["train"]),
            "val_loss": float(mse_by_split["val"]),
            "test_loss": float(mse_by_split["test"]),
            "val_loss_per_dim": {
                "inventory": float(per_dim_val[0]),
                "backlog": float(per_dim_val[1]),
                "pipeline_len": float(per_dim_val[2]),
                "t": float(per_dim_val[3]),
            },
        }


class Calibrator(abc.ABC):
    @abc.abstractmethod
    def fit(
        self,
        train_split: DatasetSplit,
        observation_trajectory_id: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError


class ConditionalGaussianNPE(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, theta_dim)
        self.log_std_head = nn.Linear(hidden, theta_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        return mean, log_std

    @staticmethod
    def nll(theta: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * log_std)
        two_pi = 2.0 * float(math.pi)
        nll = 0.5 * (torch.log(two_pi * var) + (theta - mean) ** 2 / var)
        return nll.sum(dim=-1).mean()

    def sample(self, x: torch.Tensor, num_samples: int, rng_seed: int) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must be 2D.")
        torch.manual_seed(int(rng_seed))
        mean, log_std = self.forward(x[:1])
        std = torch.exp(log_std)
        eps = torch.randn((num_samples, mean.shape[-1]), dtype=mean.dtype, device=mean.device)
        return mean[0] + eps * std[0]


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
        seed: int = GLOBAL_SEED,
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if self.num_posterior_samples <= 0:
            raise ValueError("num_posterior_samples must be > 0")
        if self.training_steps <= 0:
            raise ValueError("training_steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    @staticmethod
    def _encode_for_calibration(states: Sequence[State]) -> np.ndarray:
        base = trajectories_to_numpy(states).astype(np.float32)
        deltas = np.diff(base[:, :3], axis=0).astype(np.float32)
        return np.concatenate([base.reshape(-1), deltas.reshape(-1)], axis=0).astype(np.float32)

    def fit(
        self,
        train_split: DatasetSplit,
        observation_trajectory_id: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_traj = None
        for tr in train_split.trajectories:
            if int(tr.trajectory_id) == int(observation_trajectory_id):
                obs_traj = tr
                break
        if obs_traj is None:
            raise ValueError(
                f"Observation trajectory_id={observation_trajectory_id} not found in train split."
            )

        T = len(obs_traj.states) - 1
        if T <= 0:
            raise ValueError("Observation trajectory must have length >= 2 states.")
        if len(obs_traj.actions) != T:
            raise ValueError("Observation trajectory actions length must be T.")

        obs_vec = self._encode_for_calibration(obs_traj.states)
        x_o = torch.tensor(obs_vec, dtype=torch.float32).reshape(1, -1).to(self.device)

        prior_min, prior_max = BeerGameModel.get_parameters_uniform_prior_min_max()
        theta_dim = len(BeerGameModel.PARAMETER_NAMES)
        if prior_min.shape != (theta_dim,) or prior_max.shape != (theta_dim,):
            raise RuntimeError("Prior bounds dimensionality mismatch with parameterization.")
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
            simulator = self.simulator_factory(model, self.seed)

            sim_states = simulator.rollout_trajectory(init_state=init_state, actions=actions, rng=rng)
            sim_vec = self._encode_for_calibration(sim_states)
            x_i = torch.tensor(sim_vec, dtype=torch.float32).reshape(-1).to(self.device)
            xs[i] = x_i

        x_mean = xs.mean(dim=0, keepdim=True)
        x_std = xs.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        xs_n = (xs - x_mean) / x_std
        x_o_n = (x_o - x_mean) / x_std

        net = ConditionalGaussianNPE(x_dim=x_dim, theta_dim=theta_dim, hidden=256).to(self.device)
        opt = optim.Adam(net.parameters(), lr=self.lr)

        torch.manual_seed(self.seed)
        n = self.num_simulations
        for _step in range(self.training_steps):
            idx = torch.randint(low=0, high=n, size=(self.batch_size,), device=self.device)
            x_b = xs_n[idx]
            theta_b = thetas_u[idx]

            mean, log_std = net(x_b)
            loss = ConditionalGaussianNPE.nll(theta_b, mean, log_std)

            opt.zero_grad()
            loss.backward()
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
            "observation_T": int(T),
            "num_simulations": int(self.num_simulations),
            "num_posterior_samples": int(self.num_posterior_samples),
            "posterior_samples_mean": self._named_params(posterior_mean),
            "posterior_samples_std": self._named_params(posterior_std),
            "device": str(self.device),
            "training_steps": int(self.training_steps),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "seed": int(self.seed),
        }
        return posterior_mean, artifacts

    @staticmethod
    def _named_params(theta: np.ndarray) -> Dict[str, float]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (len(BeerGameModel.PARAMETER_NAMES),):
            raise ValueError("Theta has wrong shape for naming.")
        return {name: float(theta[i]) for i, name in enumerate(BeerGameModel.PARAMETER_NAMES)}


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    return {"train": data.train, "val": data.val, "test": data.test}


def save_results(
    output_dir: str,
    optimized_parameters: Dict[str, float],
    calibration_artifacts: Dict[str, Any],
    metrics: Dict[str, Any],
    val_predictions: Dict[int, List[State]],
) -> None:
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