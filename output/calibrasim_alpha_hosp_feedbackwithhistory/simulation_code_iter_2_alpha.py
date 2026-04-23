from __future__ import annotations

"""
Hospital patient-flow multi-agent simulator with SBI-style calibration (Gaussian NPE surrogate)
and explicit structural-error latents α (embedded model error).

This program:
- Loads train/val/test trajectories from CSV files (absolute paths via PROJECT_ROOT + DATA_PATH).
- Parses per-state patient rosters from JSON strings.
- Implements a daily patient-flow simulator (progression, discharge, arrivals, bed assignment, LOS sampling).
- Calibrates simulator using joint (λ, α) SBI:
  - λ: interpretable parameters (arrival rates, p_icu, los_mean).
  - α: structural-error magnitudes (one per λ dimension).
  - Effective params: Λ_k = λ_k + α_k * ξ_k, ξ_k ~ N(0,1), with ξ sampled once per trajectory.
  - Samples (θ, x) from prior and simulator S_α(⋅), trains NPE for q(λ, α | y).
- Rolls out on train/val/test; evaluation uses λ only. Saves λ and α separately with named labels.
"""

import argparse
import dataclasses
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This program requires PyTorch. Please ensure torch is installed and importable."
    ) from e


# -----------------------------
# Global determinism utilities
# -----------------------------

def set_global_seed(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch to ensure deterministic behavior.

    Args:
        seed: Global random seed.

    Raises:
        ValueError: If seed is not a nonnegative integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a nonnegative int, got: {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA seeding must be guarded for CPU-only builds / no-CUDA environments
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings (may reduce performance); guard backend availability
    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not support this; best effort.
        pass


# -----------------------------
# Data models
# -----------------------------

@dataclass
class Patient:
    """
    Patient agent for in-hospital simulation.
    """
    patient_id: int
    disease_id: int
    bed_type: str
    los_remaining: int
    is_alive: bool
    day_in_hospital: int

    def tick(self) -> None:
        self.day_in_hospital += 1
        self.los_remaining = max(int(self.los_remaining) - 1, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease_id": int(self.disease_id),
            "bed_type": str(self.bed_type),
            "los_remaining": int(self.los_remaining),
            "is_alive": bool(self.is_alive),
            "day_in_hospital": int(self.day_in_hospital),
            "patient_id": int(self.patient_id),
        }


@dataclass
class HospitalSystemConfig:
    icu_capacity: int = 10_000
    standard_capacity: int = 10_000
    capacity_policy_toggle: bool = False


@dataclass
class ModelParameters:
    arrival_lambda: np.ndarray  # shape (3,)
    p_icu_given_disease: np.ndarray  # shape (3,)
    los_mean: np.ndarray  # shape (3,2) [ICU, Standard]
    max_los: int = 60

    def validate(self) -> None:
        if self.arrival_lambda.shape != (3,):
            raise ValueError(f"arrival_lambda must have shape (3,), got {self.arrival_lambda.shape}")
        if self.p_icu_given_disease.shape != (3,):
            raise ValueError(
                f"p_icu_given_disease must have shape (3,), got {self.p_icu_given_disease.shape}"
            )
        if self.los_mean.shape != (3, 2):
            raise ValueError(f"los_mean must have shape (3,2), got {self.los_mean.shape}")

        if np.any(self.arrival_lambda < 0.0) or np.any(self.arrival_lambda > 50.0):
            raise ValueError(f"arrival_lambda out of bounds [0,50]: {self.arrival_lambda}")
        if np.any(self.p_icu_given_disease < 0.0) or np.any(self.p_icu_given_disease > 1.0):
            raise ValueError(f"p_icu_given_disease out of bounds [0,1]: {self.p_icu_given_disease}")
        if np.any(self.los_mean < 1.0) or np.any(self.los_mean > 30.0):
            raise ValueError(f"los_mean out of bounds [1,30]: {self.los_mean}")
        if not isinstance(self.max_los, int) or self.max_los < 1:
            raise ValueError(f"max_los must be a positive int, got {self.max_los!r}")


# -----------------------------
# Parameter names and structural-error layout (λ + α)
# -----------------------------

LAMBDA_DIM = 12

# Names for λ and α, same ordering (used for output and "α[k] → 对应 param 的结构误差")
PARAM_NAMES = [
    "arrival_lambda_d0",
    "arrival_lambda_d1",
    "arrival_lambda_d2",
    "p_icu_given_disease_d0",
    "p_icu_given_disease_d1",
    "p_icu_given_disease_d2",
    "los_mean_d0_ICU",
    "los_mean_d0_Standard",
    "los_mean_d1_ICU",
    "los_mean_d1_Standard",
    "los_mean_d2_ICU",
    "los_mean_d2_Standard",
]


def lambda_alpha_xi_to_Lambda(
    lam: np.ndarray,
    alpha: np.ndarray,
    xi: np.ndarray,
    lam_min: np.ndarray,
    lam_max: np.ndarray,
) -> np.ndarray:
    """
    Effective parameters: Λ_k = λ_k + α_k * ξ_k, ξ_k ~ N(0,1).
    Clip Λ to [lam_min, lam_max]. One ξ per trajectory (structural error, not process noise).
    """
    lam = np.asarray(lam, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    if lam.shape != (LAMBDA_DIM,) or alpha.shape != (LAMBDA_DIM,) or xi.shape != (LAMBDA_DIM,):
        raise ValueError(
            f"lam, alpha, xi must have shape ({LAMBDA_DIM},), got {lam.shape}, {alpha.shape}, {xi.shape}"
        )
    Lambda = lam + alpha * xi
    return np.clip(Lambda, lam_min, lam_max)


# -----------------------------
# Data ingestion & preprocessing
# -----------------------------

REQUIRED_COLUMNS = [
    "trajectory_id",
    "time_step",
    "day",
    "icu_occupancy",
    "standard_occupancy",
    "patients",
]


def _require_env_paths() -> str:
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.environ.get("DATA_PATH")
    if not project_root:
        raise EnvironmentError("Missing required environment variable PROJECT_ROOT.")
    if not data_path:
        raise EnvironmentError("Missing required environment variable DATA_PATH.")
    data_dir = os.path.join(project_root, data_path)
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise EnvironmentError(f"DATA_DIR does not exist or is not a directory: {data_dir}")
    return data_dir


def _parse_patients_json(patients_str: str) -> List[Dict[str, Any]]:
    if patients_str is None or (isinstance(patients_str, float) and np.isnan(patients_str)):
        return []

    if not isinstance(patients_str, str):
        raise ValueError(f"patients must be a JSON string, got type={type(patients_str)}")

    try:
        obj = json.loads(patients_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse patients JSON: {patients_str[:200]!r}") from e

    if not isinstance(obj, list):
        raise ValueError(f"patients JSON must decode to a list, got: {type(obj)}")

    for i, p in enumerate(obj):
        if not isinstance(p, dict):
            raise ValueError(f"patients[{i}] must be a dict, got: {type(p)}")
        for k in ["disease_id", "bed_type", "los_remaining", "is_alive", "day_in_hospital"]:
            if k not in p:
                raise ValueError(f"patients[{i}] missing required key '{k}'. Keys: {sorted(p.keys())}")
    return obj


def _df_to_states(df: pd.DataFrame) -> List[List[Dict[str, Any]]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.copy()
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df["time_step"] = df["time_step"].astype(int)
    df["day"] = df["day"].astype(int)
    df["icu_occupancy"] = df["icu_occupancy"].astype(int)
    df["standard_occupancy"] = df["standard_occupancy"].astype(int)

    df.sort_values(["trajectory_id", "time_step"], inplace=True)

    trajectories: List[List[Dict[str, Any]]] = []
    for traj_id, g in df.groupby("trajectory_id", sort=True):
        g = g.sort_values("time_step")
        time_steps = g["time_step"].to_numpy()
        if len(time_steps) == 0:
            continue
        if time_steps[0] != 0:
            raise ValueError(f"Trajectory {traj_id} must start at time_step=0, got {time_steps[0]}")
        if not np.all(time_steps == np.arange(time_steps[0], time_steps[0] + len(time_steps))):
            raise ValueError(f"Trajectory {traj_id} has non-consecutive time_step values: {time_steps}")

        traj_states: List[Dict[str, Any]] = []
        for _, row in g.iterrows():
            patients = _parse_patients_json(row["patients"])
            st = {
                "trajectory_id": int(row["trajectory_id"]),
                "time_step": int(row["time_step"]),
                "day": int(row["day"]),
                "icu_occupancy": int(row["icu_occupancy"]),
                "standard_occupancy": int(row["standard_occupancy"]),
                "patients": patients,
            }
            traj_states.append(st)
        trajectories.append(traj_states)

    if not trajectories:
        raise ValueError("No trajectories found after processing the CSV.")
    return trajectories


def load_data(data_dir: str) -> Dict[str, List[List[Dict[str, Any]]]]:
    paths = {
        "train": os.path.join(data_dir, "train_seed_10_n_100.csv"),
        "val": os.path.join(data_dir, "val_seed_10_n_100.csv"),
        "test": os.path.join(data_dir, "test_seed_10_n_100.csv"),
    }
    for split, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required {split} file: {p}")

    out: Dict[str, List[List[Dict[str, Any]]]] = {}
    for split, p in paths.items():
        df = pd.read_csv(p)
        out[split] = _df_to_states(df)
    return out


def build_network_and_agents(
    train_states: List[List[Dict[str, Any]]],
    config: HospitalSystemConfig,
) -> Dict[str, Any]:
    disease_set = set()
    bed_types = set()
    for traj in train_states:
        for st in traj:
            for p in st["patients"]:
                disease_set.add(int(p["disease_id"]))
                bed_types.add(str(p["bed_type"]))

    if not disease_set:
        disease_set = {0, 1, 2}
    if not bed_types:
        bed_types = {"ICU", "Standard"}

    for bt in bed_types:
        if bt not in {"ICU", "Standard"}:
            raise ValueError(f"Unexpected bed_type in data: {bt!r}")
    for d in disease_set:
        if d not in {0, 1, 2}:
            raise ValueError(f"Unexpected disease_id in data: {d!r}")

    return {
        "disease_set": sorted(disease_set),
        "bed_types": sorted(bed_types),
        "config": config,
    }


def holdout_split(
    states: Dict[str, List[List[Dict[str, Any]]]]
) -> Dict[str, List[List[Dict[str, Any]]]]:
    for k in ["train", "val", "test"]:
        if k not in states:
            raise ValueError(f"states missing required key: {k}")
        if not isinstance(states[k], list) or not states[k]:
            raise ValueError(f"states[{k}] must be a non-empty list of trajectories.")
    return states


# -----------------------------
# Trajectory summarization (7-D)
# -----------------------------

FEATURE_NAMES = [
    "day",
    "icu_occupancy",
    "standard_occupancy",
    "disease0_alive",
    "disease1_alive",
    "disease2_alive",
    "total_alive",
]


def trajectories_to_numpy(trajectory: List[Dict[str, Any]]) -> np.ndarray:
    if not trajectory:
        raise ValueError("trajectory must be non-empty.")
    arr = np.zeros((len(trajectory), 7), dtype=np.float64)
    for t, st in enumerate(trajectory):
        for k in ["day", "icu_occupancy", "standard_occupancy", "patients"]:
            if k not in st:
                raise ValueError(f"State at index {t} missing key '{k}'. Keys: {sorted(st.keys())}")

        d_counts = {0: 0, 1: 0, 2: 0}
        total = 0
        for p in st["patients"]:
            if bool(p.get("is_alive", True)):
                did = int(p["disease_id"])
                if did in d_counts:
                    d_counts[did] += 1
                total += 1

        arr[t, 0] = float(int(st["day"]))
        arr[t, 1] = float(int(st["icu_occupancy"]))
        arr[t, 2] = float(int(st["standard_occupancy"]))
        arr[t, 3] = float(d_counts[0])
        arr[t, 4] = float(d_counts[1])
        arr[t, 5] = float(d_counts[2])
        arr[t, 6] = float(total)
    return arr


# -----------------------------
# Simulator / Model
# -----------------------------

class HospitalFlowModel:
    def __init__(
        self,
        system_config: HospitalSystemConfig,
        rng: np.random.Generator,
        parameters: Optional[ModelParameters] = None,
    ) -> None:
        self.system_config = system_config
        self.rng = rng
        if parameters is None:
            parameters = ModelParameters(
                arrival_lambda=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                p_icu_given_disease=np.array([0.2, 0.2, 0.2], dtype=np.float64),
                los_mean=np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float64),
                max_los=60,
            )
        parameters.validate()
        self.parameters = parameters
        self._next_patient_id = 1

    @staticmethod
    def get_lambda_prior_min_max() -> Tuple[np.ndarray, np.ndarray]:
        """Bounds for λ (12-D). Used for clipping Λ = λ + α ξ."""
        lam_min = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        lam_max = np.array(
            [50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            dtype=np.float64,
        )
        if lam_min.shape != (LAMBDA_DIM,) or lam_max.shape != (LAMBDA_DIM,):
            raise ValueError("Lambda prior bounds must have shape (12,).")
        if np.any(lam_max <= lam_min):
            raise ValueError("Invalid lambda prior bounds: some max <= min.")
        return lam_min, lam_max

    @staticmethod
    def get_parameters_uniform_prior_min_max() -> Tuple[np.ndarray, np.ndarray]:
        """Joint prior for (λ, α): 24-D. λ same as get_lambda_prior_min_max; α ∈ [-0.5, 0.5] per dim."""
        lam_min, lam_max = HospitalFlowModel.get_lambda_prior_min_max()
        alpha_min = np.full((LAMBDA_DIM,), -0.5, dtype=np.float64)
        alpha_max = np.full((LAMBDA_DIM,), 0.5, dtype=np.float64)
        theta_min = np.concatenate([lam_min, alpha_min])
        theta_max = np.concatenate([lam_max, alpha_max])
        if theta_min.shape != (24,) or theta_max.shape != (24,):
            raise ValueError("Joint prior (λ,α) must have shape (24,).")
        if np.any(theta_max <= theta_min):
            raise ValueError("Invalid prior bounds: some max <= min.")
        return theta_min, theta_max

    @staticmethod
    def theta_to_parameters(theta: np.ndarray) -> ModelParameters:
        """Expect 12-D (Λ or λ). Used for set_parameters(Λ) in rollout."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (12,):
            raise ValueError(f"theta must have shape (12,), got {theta.shape}")
        arrival_lambda = theta[0:3]
        p_icu = theta[3:6]
        los_mean_flat = theta[6:12]
        los_mean = np.vstack([los_mean_flat[0:2], los_mean_flat[2:4], los_mean_flat[4:6]])
        params = ModelParameters(
            arrival_lambda=arrival_lambda.copy(),
            p_icu_given_disease=p_icu.copy(),
            los_mean=los_mean.copy(),
            max_los=60,
        )
        params.validate()
        return params

    @staticmethod
    def parameters_to_theta(params: ModelParameters) -> np.ndarray:
        params.validate()
        out = np.zeros((12,), dtype=np.float64)
        out[0:3] = params.arrival_lambda
        out[3:6] = params.p_icu_given_disease
        out[6:12] = params.los_mean.reshape(-1)
        return out

    @staticmethod
    def theta_joint_to_lambda_alpha(theta_24: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split 24-D joint (λ, α) into lambda (12) and alpha (12)."""
        theta_24 = np.asarray(theta_24, dtype=np.float64)
        if theta_24.shape != (24,):
            raise ValueError(f"theta_joint must have shape (24,), got {theta_24.shape}")
        return theta_24[:LAMBDA_DIM].copy(), theta_24[LAMBDA_DIM:].copy()

    def set_parameters(self, theta: np.ndarray) -> None:
        """theta: 12-D (Λ or λ)."""
        self.parameters = self.theta_to_parameters(theta)

    def _sample_bed_type(self, disease_id: int, icu_occ: int = 0) -> str:
        p_icu = float(self.parameters.p_icu_given_disease[int(disease_id)])
        cap = float(max(1, self.system_config.icu_capacity))
        if disease_id == 0:
            p_icu_eff = p_icu * (1.0 - min(0.35, icu_occ / (cap * 0.2)))
            p_icu_eff = max(0.05, min(0.9, p_icu_eff))
        elif disease_id == 1:
            p_icu_eff = p_icu * (1.0 - min(0.15, icu_occ / (cap * 0.2)))
            p_icu_eff = max(0.1, min(0.9, p_icu_eff))
        else:
            p_icu_eff = p_icu
        u = float(self.rng.random())
        return "ICU" if u < p_icu_eff else "Standard"

    def _sample_los(self, disease_id: int, bed_type: str) -> int:
        bed_idx = 0 if bed_type == "ICU" else 1 if bed_type == "Standard" else None
        if bed_idx is None:
            raise ValueError(f"Invalid bed_type: {bed_type!r}")

        m = float(self.parameters.los_mean[int(disease_id), bed_idx])
        m = max(1.0, min(30.0, m))
        if (disease_id in (0, 2) and bed_type == "ICU") or (disease_id == 1 and bed_type == "Standard"):
            k = int(self.rng.poisson(m - 1.0) + 1)
            if disease_id == 1 and bed_type == "Standard" and self.rng.random() < 0.08:
                k = max(1, k - 2)
        else:
            p = 1.0 / m
            k = int(self.rng.geometric(p))
        k = min(max(k, 1), self.parameters.max_los)
        if disease_id == 1 and bed_type == "Standard":
            k = min(k, max(1, int(m * 2.5)))
        return int(k)

    def _enforce_capacity(self, bed_type: str, icu_occ: int, std_occ: int) -> bool:
        if not self.system_config.capacity_policy_toggle:
            return True
        if bed_type == "ICU":
            return icu_occ < self.system_config.icu_capacity
        return std_occ < self.system_config.standard_capacity

    @staticmethod
    def _state_from_roster(
        trajectory_id: int,
        time_step: int,
        day: int,
        roster: List[Patient],
        rejected_or_diverted_admissions: int,
    ) -> Dict[str, Any]:
        icu_occ = sum(1 for p in roster if p.bed_type == "ICU")
        std_occ = sum(1 for p in roster if p.bed_type == "Standard")
        roster_sorted = sorted(roster, key=lambda p: p.patient_id)
        return {
            "trajectory_id": int(trajectory_id),
            "time_step": int(time_step),
            "day": int(day),
            "icu_occupancy": int(icu_occ),
            "standard_occupancy": int(std_occ),
            "rejected_or_diverted_admissions": int(rejected_or_diverted_admissions),
            "patients": [p.to_dict() for p in roster_sorted],
        }

    def _initialize_roster_from_state(self, init_state: Dict[str, Any]) -> List[Patient]:
        roster: List[Patient] = []
        patients_list = init_state.get("patients", [])
        if not isinstance(patients_list, list):
            raise ValueError("init_state['patients'] must be a list.")

        for p in patients_list:
            pid = int(p.get("patient_id", self._next_patient_id))
            self._next_patient_id = max(self._next_patient_id, pid + 1)
            roster.append(
                Patient(
                    patient_id=pid,
                    disease_id=int(p["disease_id"]),
                    bed_type=str(p["bed_type"]),
                    los_remaining=int(p["los_remaining"]),
                    is_alive=bool(p.get("is_alive", True)),
                    day_in_hospital=int(p["day_in_hospital"]),
                )
            )
        return roster

    def rollout(self, init_state: Dict[str, Any], horizon: int) -> List[Dict[str, Any]]:
        if not isinstance(horizon, int) or horizon < 0:
            raise ValueError(f"horizon must be a nonnegative int, got {horizon!r}")
        for k in ["trajectory_id", "time_step", "day", "patients"]:
            if k not in init_state:
                raise ValueError(f"init_state missing key '{k}'. Keys: {sorted(init_state.keys())}")
        if int(init_state["time_step"]) != 0:
            raise ValueError("init_state must be at time_step==0 for rollout.")

        trajectory_id = int(init_state["trajectory_id"])
        day0 = int(init_state["day"])

        roster = self._initialize_roster_from_state(init_state)
        if roster:
            self._next_patient_id = max(self._next_patient_id, max(p.patient_id for p in roster) + 1)

        simulated: List[Dict[str, Any]] = []
        st0 = self._state_from_roster(
            trajectory_id=trajectory_id,
            time_step=0,
            day=day0,
            roster=roster,
            rejected_or_diverted_admissions=0,
        )
        simulated.append(st0)

        for t in range(horizon):
            for p in roster:
                p.tick()

            roster = [p for p in roster if p.los_remaining > 0 and p.is_alive]

            icu_occ = sum(1 for p in roster if p.bed_type == "ICU")
            std_occ = sum(1 for p in roster if p.bed_type == "Standard")

            rejected_today = 0
            total_occ = icu_occ + std_occ
            for disease_id in [0, 1, 2]:
                lam = float(self.parameters.arrival_lambda[disease_id])
                lam = max(0.0, min(50.0, lam))
                if disease_id == 1:
                    if total_occ > 40:
                        lam = lam * (1.0 - min(0.25, (total_occ - 40) / 300.0))
                    lam = max(0.0, lam)
                    n_arrivals = int(self.rng.poisson(lam))
                elif disease_id == 2:
                    if total_occ > 50:
                        lam = lam * (1.0 - min(0.4, (total_occ - 50) / 200.0))
                    lam = max(0.0, lam)
                    r = 2.0
                    p_nb = r / (r + lam)
                    n_arrivals = int(self.rng.negative_binomial(r, p_nb))
                else:
                    n_arrivals = int(self.rng.poisson(lam))
                for _ in range(n_arrivals):
                    bed_type = self._sample_bed_type(disease_id, icu_occ)
                    accepted = self._enforce_capacity(bed_type, icu_occ, std_occ)
                    if not accepted:
                        rejected_today += 1
                        continue

                    los = self._sample_los(disease_id, bed_type)
                    new_patient = Patient(
                        patient_id=self._next_patient_id,
                        disease_id=int(disease_id),
                        bed_type=bed_type,
                        los_remaining=int(los),
                        is_alive=True,
                        day_in_hospital=1,
                    )
                    self._next_patient_id += 1
                    roster.append(new_patient)
                    if bed_type == "ICU":
                        icu_occ += 1
                    else:
                        std_occ += 1

            st = self._state_from_roster(
                trajectory_id=trajectory_id,
                time_step=t + 1,
                day=day0 + (t + 1),
                roster=roster,
                rejected_or_diverted_admissions=rejected_today,
            )
            simulated.append(st)

        return simulated


# -----------------------------
# Calibrator
# -----------------------------

class GaussianPosteriorNet(nn.Module):
    def __init__(self, x_dim: int, theta_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = x_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, theta_dim)
        self.log_std_head = nn.Linear(prev, theta_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5.0, 3.0)
        return mean, log_std


@dataclass
class CalibrationArtifacts:
    theta_samples: np.ndarray
    x_samples: np.ndarray
    loss_history: List[float]
    x_mean: np.ndarray
    x_std: np.ndarray


class BaseCalibrator:
    def fit(
        self,
        model: HospitalFlowModel,
        train_states: List[List[Dict[str, Any]]],
        *,
        num_simulations: int,
        device: str,
    ) -> Tuple[np.ndarray, CalibrationArtifacts]:
        raise NotImplementedError


class NPEGaussianCalibrator(BaseCalibrator):
    def __init__(
        self,
        seed: int,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if lr <= 0:
            raise ValueError("lr must be positive.")
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

    @staticmethod
    def _sample_prior(
        rng: np.random.Generator,
        n: int,
        theta_min: np.ndarray,
        theta_max: np.ndarray,
    ) -> np.ndarray:
        u = rng.random((n, theta_min.shape[0]))
        return theta_min + u * (theta_max - theta_min)

    @staticmethod
    def _make_observation_xo(train_states: List[List[Dict[str, Any]]]) -> np.ndarray:
        if not train_states:
            raise ValueError("train_states is empty.")
        observed_traj = train_states[0]
        obs_np = trajectories_to_numpy(observed_traj)
        return obs_np.reshape(-1).astype(np.float32)

    def fit(
        self,
        model: HospitalFlowModel,
        train_states: List[List[Dict[str, Any]]],
        *,
        num_simulations: int,
        device: str,
    ) -> Tuple[np.ndarray, CalibrationArtifacts]:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")
        if not train_states or not train_states[0]:
            raise ValueError("train_states must contain at least one non-empty trajectory.")

        if device == "cuda" and (not hasattr(torch, "cuda") or not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but not available. Use --device cpu.")

        torch_device = torch.device(device)

        init_state = train_states[0][0]
        horizon = len(train_states[0]) - 1
        if horizon < 0:
            raise ValueError("Observed trajectory horizon must be >= 0.")

        xo = self._make_observation_xo(train_states)
        x_dim = int(xo.shape[0])

        theta_min, theta_max = model.get_parameters_uniform_prior_min_max()
        theta_dim = int(theta_min.shape[0])
        lam_min, lam_max = model.get_lambda_prior_min_max()

        theta_samples = self._sample_prior(model.rng, num_simulations, theta_min, theta_max).astype(
            np.float32
        )
        x_samples = np.zeros((num_simulations, x_dim), dtype=np.float32)

        for i in range(num_simulations):
            theta_i = theta_samples[i].astype(np.float64)
            lam_i, alpha_i = model.theta_joint_to_lambda_alpha(theta_i)
            xi = model.rng.standard_normal(size=LAMBDA_DIM).astype(np.float64)
            Lambda_i = lambda_alpha_xi_to_Lambda(lam_i, alpha_i, xi, lam_min, lam_max)
            model.set_parameters(Lambda_i)
            sim_traj = model.rollout(init_state=init_state, horizon=horizon)
            xi_out = trajectories_to_numpy(sim_traj).reshape(-1).astype(np.float32)
            if xi_out.shape[0] != x_dim:
                raise RuntimeError(
                    f"Simulator produced x_dim={xi_out.shape[0]} but expected {x_dim}. "
                    "Check horizon alignment."
                )
            x_samples[i] = xi_out

        x_mean = x_samples.mean(axis=0)
        x_std = x_samples.std(axis=0)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)
        x_samples_std = (x_samples - x_mean) / x_std
        xo_std = (xo - x_mean) / x_std

        X = torch.from_numpy(x_samples_std).to(torch_device)
        Y = torch.from_numpy(theta_samples).to(torch_device)

        net = GaussianPosteriorNet(
            x_dim=x_dim,
            theta_dim=theta_dim,
            hidden_sizes=self.hidden_sizes,
        ).to(torch_device)

        opt = optim.Adam(net.parameters(), lr=self.lr)

        def nll_diag_gaussian(mean: torch.Tensor, log_std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            var = torch.exp(2.0 * log_std)
            nll = 0.5 * (math.log(2.0 * math.pi) + 2.0 * log_std + ((y - mean) ** 2) / var)
            return nll.sum(dim=1).mean()

        loss_history: List[float] = []
        n = num_simulations
        indices = np.arange(n)

        for _epoch in range(self.epochs):
            model.rng.shuffle(indices)
            epoch_losses: List[float] = []

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]
                xb = X[batch_idx]
                yb = Y[batch_idx]

                mean, log_std = net(xb)
                loss = nll_diag_gaussian(mean, log_std, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                epoch_losses.append(float(loss.detach().cpu().item()))

            loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))

        with torch.no_grad():
            xo_t = torch.from_numpy(xo_std.reshape(1, -1)).to(torch_device)
            mean, _log_std = net(xo_t)
            optimized_theta = mean.squeeze(0).detach().cpu().numpy().astype(np.float64)

        theta_min_np = np.asarray(theta_min, dtype=np.float64)
        theta_max_np = np.asarray(theta_max, dtype=np.float64)
        optimized_theta = np.clip(optimized_theta, theta_min_np, theta_max_np)

        artifacts = CalibrationArtifacts(
            theta_samples=theta_samples.astype(np.float64),
            x_samples=x_samples.astype(np.float64),
            loss_history=loss_history,
            x_mean=x_mean.astype(np.float64),
            x_std=x_std.astype(np.float64),
        )
        return optimized_theta, artifacts


# -----------------------------
# Evaluation
# -----------------------------

@dataclass
class EvaluationResults:
    train_loss: float
    val_loss: float
    test_loss: float
    mse_per_dimension_val: Dict[str, float]
    state_consistency_rate: float


class Evaluator:
    @staticmethod
    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            raise ValueError(f"MSE inputs must have same shape, got {a.shape} vs {b.shape}")
        return float(np.mean((a - b) ** 2))

    @staticmethod
    def _state_consistency_rate(sim_trajectories: List[List[Dict[str, Any]]]) -> float:
        total = 0
        ok = 0
        for traj in sim_trajectories:
            for st in traj:
                total += 1
                roster = st.get("patients", [])
                icu_count = sum(1 for p in roster if p.get("bed_type") == "ICU")
                std_count = sum(1 for p in roster if p.get("bed_type") == "Standard")
                if int(st.get("icu_occupancy", -1)) == icu_count and int(
                    st.get("standard_occupancy", -1)
                ) == std_count:
                    ok += 1
        return float(ok / total) if total > 0 else 0.0

    def compute_metrics(
        self,
        model: HospitalFlowModel,
        splits: Dict[str, List[List[Dict[str, Any]]]],
        *,
        optimized_theta: np.ndarray,
    ) -> Tuple[EvaluationResults, Dict[str, Any]]:
        model.set_parameters(optimized_theta)

        payload: Dict[str, Any] = {"feature_names": FEATURE_NAMES, "splits": {}}
        sim_all_for_consistency: List[List[Dict[str, Any]]] = []

        def eval_split(name: str) -> float:
            true_trajs = splits[name]
            pred_trajs: List[List[Dict[str, Any]]] = []
            true_np_list: List[np.ndarray] = []
            pred_np_list: List[np.ndarray] = []

            for traj in true_trajs:
                init_state = traj[0]
                horizon = len(traj) - 1
                pred = model.rollout(init_state=init_state, horizon=horizon)

                true_np = trajectories_to_numpy(traj)
                pred_np = trajectories_to_numpy(pred)

                true_np_list.append(true_np)
                pred_np_list.append(pred_np)
                pred_trajs.append(pred)

            sim_all_for_consistency.extend(pred_trajs)

            true_np_all = np.stack(true_np_list, axis=0)
            pred_np_all = np.stack(pred_np_list, axis=0)

            # Store trajectory IDs for CSV output
            trajectory_ids = [traj[0].get("trajectory_id", i) for i, traj in enumerate(true_trajs)]

            payload["splits"][name] = {
                "true_np": true_np_all,
                "pred_np": pred_np_all,
                "trajectory_ids": trajectory_ids,
                "true_trajectories": true_trajs,  # Keep for reference
                "pred_trajectories": pred_trajs,  # Keep for reference
            }
            return self._mse(pred_np_all, true_np_all)

        train_loss = eval_split("train")
        val_loss = eval_split("val")
        test_loss = eval_split("test")

        val_true = payload["splits"]["val"]["true_np"]
        val_pred = payload["splits"]["val"]["pred_np"]
        mse_per_dim: Dict[str, float] = {}
        # Map feature names to patch-required names (disease0_alive -> d0_alive, etc.)
        feature_name_map = {
            "day": "day",
            "icu_occupancy": "icu_occupancy",
            "standard_occupancy": "standard_occupancy",
            "disease0_alive": "d0_alive",
            "disease1_alive": "d1_alive",
            "disease2_alive": "d2_alive",
            "total_alive": "total_alive",
        }
        for j, fname in enumerate(FEATURE_NAMES):
            mapped_name = feature_name_map.get(fname, fname)
            mse_per_dim[mapped_name] = self._mse(val_pred[..., j], val_true[..., j])

        consistency_rate = self._state_consistency_rate(sim_all_for_consistency)

        results = EvaluationResults(
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            test_loss=float(test_loss),
            mse_per_dimension_val=mse_per_dim,
            state_consistency_rate=float(consistency_rate),
        )
        return results, payload


# -----------------------------
# Saving results
# -----------------------------

def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return _jsonify(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def lambda_to_named_parameters(lam: np.ndarray) -> Dict[str, float]:
    """
    Convert 12-D λ vector to named parameter dictionary (for backward-compat / metrics).
    """
    lam = np.asarray(lam, dtype=np.float64)
    if lam.shape != (LAMBDA_DIM,):
        raise ValueError(f"lambda must have shape ({LAMBDA_DIM},), got {lam.shape}")
    return {name: float(lam[k]) for k, name in enumerate(PARAM_NAMES)}


def theta_to_named_parameters(theta: np.ndarray) -> Dict[str, float]:
    """
    Convert 12-D theta (Λ or λ) to named parameter dictionary as required by patch.
    """
    return lambda_to_named_parameters(theta)


def save_results(
    output_dir: str,
    *,
    optimized_lambda: np.ndarray,
    optimized_alpha: np.ndarray,
    optimized_parameters: ModelParameters,
    artifacts: CalibrationArtifacts,
    metrics: EvaluationResults,
    eval_payload: Dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    optimized_lambda = np.asarray(optimized_lambda, dtype=np.float64)
    optimized_alpha = np.asarray(optimized_alpha, dtype=np.float64)
    if optimized_lambda.shape != (LAMBDA_DIM,) or optimized_alpha.shape != (LAMBDA_DIM,):
        raise ValueError(
            f"optimized_lambda and optimized_alpha must have shape ({LAMBDA_DIM},), "
            f"got {optimized_lambda.shape}, {optimized_alpha.shape}"
        )

    lambda_named = lambda_to_named_parameters(optimized_lambda)
    alpha_named = {f"alpha_{name}": float(optimized_alpha[k]) for k, name in enumerate(PARAM_NAMES)}
    optimized_params_dict = {**lambda_named, **alpha_named}

    # Prepare calibration_artifacts for results.json
    calibration_artifacts_dict = {
        "loss_history": artifacts.loss_history,
        "x_mean_summary": {
            "mean_of_mean": float(np.mean(artifacts.x_mean)),
            "mean_of_std": float(np.mean(artifacts.x_std)),
            "x_dim": int(artifacts.x_mean.shape[0]),
        },
        "num_simulations": int(artifacts.theta_samples.shape[0]),
    }

    # Prepare metrics for results.json (rename mse_per_dimension_val to val_loss_per_dim)
    metrics_dict = {
        "train_loss": float(metrics.train_loss),
        "val_loss": float(metrics.val_loss),
        "test_loss": float(metrics.test_loss),
        "val_loss_per_dim": metrics.mse_per_dimension_val,  # Already has correct keys (d0_alive, etc.)
        "state_consistency_rate": float(metrics.state_consistency_rate),
    }

    structural_error_lines = [
        f"α[{k}] = {optimized_alpha[k]:.6f} → 对应 {name} 的结构误差"
        for k, name in enumerate(PARAM_NAMES)
    ]
    structural_error_alpha_list = [
        {"index": k, "alpha": float(optimized_alpha[k]), "param_name": name}
        for k, name in enumerate(PARAM_NAMES)
    ]

    results_json_payload = {
        "optimized_parameters": optimized_params_dict,
        "optimized_lambda": lambda_named,
        "optimized_alpha": alpha_named,
        "structural_error_alpha_labels": structural_error_alpha_list,
        "calibration_artifacts": calibration_artifacts_dict,
        "metrics": metrics_dict,
    }

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json_payload, f, indent=2, sort_keys=True)

    with open(os.path.join(output_dir, "structural_error_alpha.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(structural_error_lines))

    with open(os.path.join(output_dir, "optimized_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimized_lambda": optimized_lambda.tolist(),
                "optimized_alpha": optimized_alpha.tolist(),
                "optimized_lambda_named": lambda_named,
                "optimized_alpha_named": alpha_named,
                "structural_error_alpha_labels": structural_error_alpha_list,
                "parameters": _jsonify(optimized_parameters),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    with open(os.path.join(output_dir, "calibration_artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_artifacts_dict, f, indent=2, sort_keys=True)

    np.savez_compressed(
        os.path.join(output_dir, "calibration_samples.npz"),
        theta_samples=artifacts.theta_samples,
        x_samples=artifacts.x_samples,
        x_mean=artifacts.x_mean,
        x_std=artifacts.x_std,
    )

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify(metrics), f, indent=2, sort_keys=True)

    eval_npz_path = os.path.join(output_dir, "eval_payload.npz")
    npz_dict: Dict[str, Any] = {"feature_names": np.array(FEATURE_NAMES, dtype=object)}
    for split_name, d in eval_payload.get("splits", {}).items():
        if isinstance(d, dict) and "true_np" in d and "pred_np" in d:
            npz_dict[f"{split_name}_true_np"] = d["true_np"]
            npz_dict[f"{split_name}_pred_np"] = d["pred_np"]
    np.savez_compressed(eval_npz_path, **npz_dict)

    # Write val_predicted_mean_trajectories.csv as required by patch
    if "val" in eval_payload.get("splits", {}):
        val_split_data = eval_payload["splits"]["val"]
        val_pred_np = val_split_data["pred_np"]  # shape: (num_trajectories, T+1, 7)
        trajectory_ids = val_split_data.get("trajectory_ids", list(range(val_pred_np.shape[0])))
        
        csv_rows = []
        num_trajs = val_pred_np.shape[0]
        
        # Feature names for CSV columns (using patch-required names)
        csv_feature_names = ["day", "icu_occupancy", "standard_occupancy", "d0_alive", "d1_alive", "d2_alive", "total_alive"]
        
        for traj_idx in range(num_trajs):
            traj_pred = val_pred_np[traj_idx]  # shape: (T+1, 7)
            num_timesteps = traj_pred.shape[0]
            traj_id = trajectory_ids[traj_idx] if traj_idx < len(trajectory_ids) else traj_idx
            
            for t in range(num_timesteps):
                row = {
                    "trajectory_id": int(traj_id),
                    "time_step": int(t),
                }
                # Add the 7-D features
                for feat_idx, feat_name in enumerate(csv_feature_names):
                    row[feat_name] = float(traj_pred[t, feat_idx])
                csv_rows.append(row)
        
        # Write CSV to output_dir
        csv_path = os.path.join(output_dir, "val_predicted_mean_trajectories.csv")
        df_csv = pd.DataFrame(csv_rows)
        # Ensure columns are in the required order
        column_order = ["trajectory_id", "time_step", "day", "icu_occupancy", "standard_occupancy", 
                       "d0_alive", "d1_alive", "d2_alive", "total_alive"]
        df_csv = df_csv[column_order]
        df_csv.to_csv(csv_path, index=False)


# -----------------------------
# CLI / Orchestration
# -----------------------------

def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    epilog = """
Run requires env: PROJECT_ROOT, DATA_PATH (e.g. DATA_PATH=data_fitting/calibrasim_hosp).
Example:
  PROJECT_ROOT=/path/to/SOCIA DATA_PATH=data_fitting/calibrasim_hosp \\
    python simulation_code_iter_alpha.py --output_dir ./out --device cuda
"""
    p = argparse.ArgumentParser(
        description="Hospital SBI-style simulator and calibrator (λ + α structural error).",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results.")
    p.add_argument("--seed", type=int, default=123, help="Global random seed.")
    p.add_argument(
        "--num_simulations",
        type=int,
        default=5000,
        help="Simulations for NPE training (default 5000; recommend 5000–10000 for joint λ,α).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Training epochs for posterior net (default 250; recommend 200–300).",
    )
    p.add_argument("--batch_size", type=int, default=64, help="Minibatch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device.",
    )
    p.add_argument(
        "--capacity_policy_toggle",
        action="store_true",
        help="Enable capacity truncation (default: off).",
    )
    p.add_argument("--icu_capacity", type=int, default=10_000, help="ICU capacity if enabled.")
    p.add_argument("--standard_capacity", type=int, default=10_000, help="Standard capacity if enabled.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_cli(argv)

    set_global_seed(args.seed)
    data_dir = _require_env_paths()

    rng = np.random.default_rng(args.seed)

    states = load_data(data_dir)

    system_config = HospitalSystemConfig(
        icu_capacity=int(args.icu_capacity),
        standard_capacity=int(args.standard_capacity),
        capacity_policy_toggle=bool(args.capacity_policy_toggle),
    )

    _metadata = build_network_and_agents(states["train"], system_config)
    splits = holdout_split(states)

    model = HospitalFlowModel(system_config=system_config, rng=rng, parameters=None)

    calibrator: BaseCalibrator = NPEGaussianCalibrator(
        seed=args.seed,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        hidden_sizes=(256, 256),
    )

    optimized_theta, artifacts = calibrator.fit(
        model,
        splits["train"],
        num_simulations=int(args.num_simulations),
        device=str(args.device),
    )
    optimized_lambda, optimized_alpha = HospitalFlowModel.theta_joint_to_lambda_alpha(
        optimized_theta
    )
    optimized_parameters = HospitalFlowModel.theta_to_parameters(optimized_lambda)

    init_state = splits["train"][0][0]
    horizon = len(splits["train"][0]) - 1
    model.set_parameters(optimized_lambda)
    _ = model.rollout(init_state=init_state, horizon=horizon)

    evaluator = Evaluator()
    metrics, eval_payload = evaluator.compute_metrics(
        model,
        splits,
        optimized_theta=optimized_lambda,
    )

    save_results(
        args.output_dir,
        optimized_lambda=optimized_lambda,
        optimized_alpha=optimized_alpha,
        optimized_parameters=optimized_parameters,
        artifacts=artifacts,
        metrics=metrics,
        eval_payload=eval_payload,
    )

    print("Optimized lambda:", optimized_lambda)
    print("Optimized alpha:", optimized_alpha)
    for k, name in enumerate(PARAM_NAMES):
        print(f"  α[{k}] = {optimized_alpha[k]:.6f} → 对应 {name} 的结构误差")
    print("Train loss:", metrics.train_loss)
    print("Val loss:", metrics.val_loss)
    print("Test loss:", metrics.test_loss)
    print("Val MSE per dimension:", metrics.mse_per_dimension_val)
    print("State consistency rate:", metrics.state_consistency_rate)
    print("Saved results to:", os.path.abspath(args.output_dir))


main()