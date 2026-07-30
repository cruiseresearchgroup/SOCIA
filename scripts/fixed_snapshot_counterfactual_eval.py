#!/usr/bin/env python3
"""Fixed-snapshot counterfactual evaluation.

This runner intentionally has no imports from the SOCIA orchestration pipeline
and never calls calibration, fitting, diagnosis, code-generation, or selection
functions. Simulator source is loaded through an AST gate that removes all
top-level execution entry points before definitions are evaluated.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import math
import random
import sys
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("intervention", "social", "persistence", "risk")
SETTINGS = {
    "intervention": (
        "baseline", "no_campaign", "strength_low", "strength_high",
        "timing_early", "timing_late",
    ),
    "social": ("baseline", "work_0", "work_2", "community_0", "community_2", "layer_swap"),
    "persistence": ("baseline", "balanced_toggle", "state_permutation"),
    "risk": ("baseline", "raise_low_risk", "lower_high_risk"),
}
FORBIDDEN_NAMES = {
    "fit", "calibrate", "optimize", "diagnose", "repair", "generate_code",
    "BayesianOptimization", "TuRBOCalibrator", "SBICalibrator",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


class SnapshotGate(ast.NodeTransformer):
    """Remove execution entry points and expose campaign day as an input hook."""

    def __init__(self) -> None:
        self.removed_entrypoints = 0
        self.campaign_threshold_rewrites = 0

    @staticmethod
    def _is_main_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "main"
        )

    @staticmethod
    def _is_name_main_test(node: ast.AST) -> bool:
        text = ast.dump(node, include_attributes=False)
        return "__name__" in text and "__main__" in text

    def visit_Module(self, node: ast.Module) -> ast.Module:
        body: List[ast.stmt] = []
        for stmt in node.body:
            if self._is_main_call(stmt):
                self.removed_entrypoints += 1
                continue
            if isinstance(stmt, ast.If) and self._is_name_main_test(stmt.test):
                self.removed_entrypoints += 1
                continue
            body.append(stmt)
        node.body = [self.visit(stmt) for stmt in body]
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        names = {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}
        day_names = {"global_day", "current_day", "day", "day_value"}
        if names & day_names:
            for index, comparator in enumerate(node.comparators):
                if isinstance(comparator, ast.Constant) and comparator.value == 10:
                    node.comparators[index] = ast.Name(id="__CF_CAMPAIGN_DAY__", ctx=ast.Load())
                    self.campaign_threshold_rewrites += 1
        return node


def safe_load_snapshot(code_path: Path, campaign_day: int) -> Tuple[types.ModuleType, Dict[str, Any]]:
    source = code_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(code_path))
    gate = SnapshotGate()
    tree = gate.visit(tree)
    ast.fix_missing_locations(tree)
    module_name = "fixed_snapshot_" + hashlib.sha256(str(code_path).encode()).hexdigest()[:12]
    module = types.ModuleType(module_name)
    module.__file__ = str(code_path)
    module.__dict__["__name__"] = module_name
    module.__dict__["__CF_CAMPAIGN_DAY__"] = int(campaign_day)
    sys.modules[module_name] = module
    compiled = compile(tree, str(code_path), "exec")
    exec(compiled, module.__dict__)
    if not callable(getattr(module, "simulate_window", None)):
        raise RuntimeError(f"{code_path}: missing callable simulate_window")
    audit = {
        "source_sha256": sha256(code_path),
        "removed_entrypoints": gate.removed_entrypoints,
        "campaign_threshold_rewrites": gate.campaign_threshold_rewrites,
        "simulate_window_signature": str(inspect.signature(module.simulate_window)),
    }
    return module, audit


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    forbidden = set(manifest.get("prohibitions", []))
    if not forbidden:
        raise RuntimeError("Manifest must explicitly record fixed-snapshot prohibitions.")
    return manifest


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_data(data_folder: Path) -> Dict[str, Any]:
    agents = pd.read_csv(data_folder / "agent_attributes.csv")
    train = pd.read_csv(data_folder / "train_data.csv")
    with (data_folder / "social_network.json").open("r", encoding="utf-8") as handle:
        social = json.load(handle)
    common = sorted(
        set(agents.agent_id.astype(int))
        & set(train.agent_id.astype(int))
        & {int(key) for key in social}
    )
    if not common:
        raise RuntimeError("No common agents across mask-adoption inputs.")
    id2idx = {agent_id: index for index, agent_id in enumerate(common)}
    days = sorted(int(day) for day in train.day.unique())
    day2idx = {day: index for index, day in enumerate(days)}
    wearing = np.zeros((len(days), len(common)), dtype=float)
    received = np.zeros_like(wearing)
    for row in train.itertuples(index=False):
        if int(row.agent_id) not in id2idx:
            continue
        t, i = day2idx[int(row.day)], id2idx[int(row.agent_id)]
        wearing[t, i] = float(bool(row.wearing_mask))
        received[t, i] = float(bool(row.received_info))
    neighbors: Dict[str, List[set]] = {
        layer: [set() for _ in common]
        for layer in ("family", "work_school", "community")
    }
    for key, layers in social.items():
        agent_id = int(key)
        if agent_id not in id2idx:
            continue
        i = id2idx[agent_id]
        for layer in neighbors:
            for neighbor_id in layers.get(layer, []):
                neighbor_id = int(neighbor_id)
                if neighbor_id in id2idx and neighbor_id != agent_id:
                    j = id2idx[neighbor_id]
                    neighbors[layer][i].add(j)
                    neighbors[layer][j].add(i)
    adjacency = {
        layer: [np.asarray(sorted(values), dtype=int) for values in per_agent]
        for layer, per_agent in neighbors.items()
    }
    aligned_agents = agents.set_index("agent_id").loc[common]
    risk = aligned_agents.risk_perception.to_numpy(dtype=float)
    return {
        "agent_ids": np.asarray(common, dtype=int),
        "agents": aligned_agents,
        "days": days,
        "wearing": wearing,
        "received": received,
        "neighbors": adjacency,
        "risk": risk,
    }


def encode_demographics(module: types.ModuleType, data: Dict[str, Any]) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    agents = data["agents"].reset_index()
    ids = data["agent_ids"]
    try:
        result = module.encode_demographics(agents, ids)
        if len(result) == 4:
            return result
    except Exception:
        pass
    frame = data["agents"]
    age_values = frame.age_group.astype(str)
    occ_values = frame.occupation.astype(str)
    age_base = "Middle Age" if "Middle Age" in set(age_values) else sorted(set(age_values))[0]
    occ_base = "White Collar" if "White Collar" in set(occ_values) else sorted(set(occ_values))[0]
    age_names = [value for value in sorted(set(age_values)) if value != age_base]
    occ_names = [value for value in sorted(set(occ_values)) if value != occ_base]
    age = np.column_stack([(age_values == value).to_numpy(float) for value in age_names])
    occ = np.column_stack([(occ_values == value).to_numpy(float) for value in occ_names])
    return age, age_names, occ, occ_names


def params_from_json(module: types.ModuleType, parameter_path: Path) -> Tuple[Any, Dict[str, Any]]:
    with parameter_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "parameters" in payload and "decision_weights" not in payload:
        payload = payload["parameters"]
    fitted_type = getattr(module, "FittedParams", None)
    if fitted_type is not None:
        fitted = fitted_type(**payload)
        if hasattr(fitted, "to_parameters"):
            return fitted.to_parameters(), payload
    decision = payload["decision_weights"]
    layers = payload["layer_weights"]
    info = payload["info_params"]
    noise = payload["noise_params"]
    flat = {
        "alpha": decision["alpha"],
        "gamma": decision["gamma"],
        "theta_f": decision["theta_f"],
        "theta_w": decision["theta_w"],
        "theta_c": decision["theta_c"],
        "beta_r": decision["beta_r"],
        "beta_i": decision["beta_i"],
        "age_effects": decision.get("age_effects", {}),
        "occ_effects": decision.get("occ_effects", {}),
        "tau": noise.get("tau", 1.0),
        "w_family": layers.get("family", 1.0),
        "w_work": layers.get("work_school", 1.0),
        "w_community": layers.get("community", 1.0),
        "phi_family": info.get("phi_family", 0.1),
        "phi_work": info.get("phi_work", 0.1),
        "phi_community": info.get("phi_community", 0.1),
        "lambda_broadcast_base": info.get("lambda_broadcast_base", 0.05),
        "lambda_broadcast_factor_after_day10": info.get("lambda_broadcast_factor_after_day10", 1.5),
        "rho_info_decay": info.get("rho_info_decay", 0.5),
    }
    signature = inspect.signature(module.Parameters)
    return module.Parameters(**{key: value for key, value in flat.items() if key in signature.parameters}), payload


def memory_prefix(module: types.ModuleType, received: np.ndarray, rho: float) -> np.ndarray:
    if callable(getattr(module, "compute_mem_info", None)):
        return np.asarray(module.compute_mem_info(received, rho), dtype=float)
    memory = np.zeros_like(received, dtype=float)
    for t in range(1, received.shape[0]):
        memory[t] = rho * memory[t - 1] + (1.0 - rho) * received[t]
    return memory


def copy_params(params: Any) -> Any:
    return copy.deepcopy(params)


def set_campaign_strength(params: Any, strength: float) -> None:
    original = float(params.lambda_broadcast_factor_after_day10)
    params.lambda_broadcast_factor_after_day10 = 1.0 + strength * (original - 1.0)


def make_perturbations(data: Dict[str, Any], val_start: int, seed: int = 20260729) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    initial = data["wearing"][val_start - 1].copy()
    zeros = np.flatnonzero(initial == 0)
    ones = np.flatnonzero(initial == 1)
    count = max(1, min(len(zeros), len(ones), int(round(0.2 * len(initial)))))
    up = np.sort(rng.choice(zeros, count, replace=False))
    down = np.sort(rng.choice(ones, count, replace=False))
    balanced = initial.copy()
    balanced[up] = 1.0
    balanced[down] = 0.0
    permuted = initial[rng.permutation(len(initial))]
    risk = data["risk"]
    q25, q75 = np.quantile(risk, [0.25, 0.75])
    low = np.flatnonzero(risk <= q25)
    high = np.flatnonzero(risk >= q75)
    return {
        "initial": initial,
        "balanced": balanced,
        "permuted": permuted,
        "up_indices": up,
        "down_indices": down,
        "q25": float(q25),
        "q75": float(q75),
        "low_indices": low,
        "high_indices": high,
    }


def validation_bounds(days: Sequence[int]) -> Tuple[int, int]:
    split = int(math.floor(0.8 * len(days)))
    return split, len(days)


def invoke_simulate(
    module: types.ModuleType,
    params: Any,
    data: Dict[str, Any],
    initial_states: np.ndarray,
    risk: np.ndarray,
    neighbors: Dict[str, List[np.ndarray]],
    age_oh: np.ndarray,
    age_names: List[str],
    occ_oh: np.ndarray,
    occ_names: List[str],
    start_index: int,
    end_index: int,
    day_values: Sequence[int],
    initial_mem: np.ndarray,
    seed: int,
    campaign_day: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    random.seed(seed)
    module.__dict__["__CF_CAMPAIGN_DAY__"] = int(campaign_day)
    signature = inspect.signature(module.simulate_window)
    supplied: Dict[str, Any] = {
        "start_states": initial_states,
        "neighbors": neighbors,
        "risk": risk,
        "age_oh": age_oh,
        "occ_oh": occ_oh,
        "age_cat_names": age_names,
        "occ_cat_names": occ_names,
        "params": params,
        "start_day_index": start_index,
        "end_day_index": end_index,
        "initial_mem": initial_mem,
        "start_mem": initial_mem,
        "start_info_state": np.zeros_like(initial_states),
        "seed": seed,
        "rng": np.random.default_rng(seed),
        "intervention_day": campaign_day,
        "gov_intervention_day": campaign_day,
        "day_values": list(day_values),
        "days": list(day_values),
        "return_diagnostics": False,
        "return_components": False,
    }
    kwargs = {name: supplied[name] for name in signature.parameters if name in supplied}
    missing = [
        name for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty and name not in kwargs
    ]
    if missing:
        raise RuntimeError(f"Unsupported simulate_window required arguments: {missing}")
    result = module.simulate_window(**kwargs)
    if not isinstance(result, tuple) or len(result) < 3:
        raise RuntimeError("simulate_window must return at least states, info, probabilities")
    states, info, probs = (np.asarray(result[index], dtype=float) for index in range(3))
    if states.shape != probs.shape or states.ndim != 2:
        raise RuntimeError(f"Invalid trajectory shapes: states={states.shape}, probs={probs.shape}")
    return states, info, probs


def objective_per_run(states: np.ndarray, probs: np.ndarray, observed: np.ndarray, initial: np.ndarray) -> float:
    rates = states.mean(axis=1)
    obs_rates = observed.mean(axis=1)
    rmse = math.sqrt(float(np.mean((rates - obs_rates) ** 2)))
    mae = float(np.mean(np.abs(rates - obs_rates)))
    brier = float(np.mean((probs - observed) ** 2))

    def joints(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        return np.asarray([
            np.mean((previous == 0) & (current == 1)),
            np.mean((previous == 1) & (current == 1)),
            np.mean((previous == 1) & (current == 0)),
            np.mean((previous == 0) & (current == 0)),
        ])

    prev_sim = np.vstack([initial[None, :], states[:-1]])
    prev_obs = np.vstack([initial[None, :], observed[:-1]])
    transition = float(np.mean(np.abs(joints(prev_sim, states) - joints(prev_obs, observed))))
    return 0.4 * rmse + 0.2 * mae + 0.2 * brier + 0.2 * transition


def setting_inputs(
    family: str,
    setting: str,
    params: Any,
    data: Dict[str, Any],
    perturb: Dict[str, Any],
    val_start: int,
    val_end: int,
) -> Dict[str, Any]:
    params = copy_params(params)
    neighbors = copy.deepcopy(data["neighbors"])
    risk = data["risk"].copy()
    initial = perturb["initial"].copy()
    campaign_day = 10
    if family == "intervention":
        start_index, end_index = 0, 19
        day_values = list(range(20))
        if setting == "no_campaign":
            set_campaign_strength(params, 0.0)
        elif setting == "strength_low":
            set_campaign_strength(params, 0.5)
        elif setting == "strength_high":
            set_campaign_strength(params, 2.0)
        elif setting == "timing_early":
            campaign_day = 7
        elif setting == "timing_late":
            campaign_day = 13
    else:
        start_index, end_index = val_start - 1, val_end - 1
        day_values = data["days"]
    if family == "social":
        if setting == "work_0":
            params.theta_w *= 0.0
        elif setting == "work_2":
            params.theta_w *= 2.0
        elif setting == "community_0":
            params.theta_c *= 0.0
        elif setting == "community_2":
            params.theta_c *= 2.0
        elif setting == "layer_swap":
            neighbors["work_school"], neighbors["community"] = (
                neighbors["community"], neighbors["work_school"]
            )
    elif family == "persistence":
        if setting == "balanced_toggle":
            initial = perturb["balanced"].copy()
        elif setting == "state_permutation":
            initial = perturb["permuted"].copy()
    elif family == "risk":
        if setting == "raise_low_risk":
            risk[perturb["low_indices"]] = perturb["q75"]
        elif setting == "lower_high_risk":
            risk[perturb["high_indices"]] = perturb["q25"]
    if hasattr(params, "gov_intervention_day"):
        params.gov_intervention_day = int(campaign_day)
    return {
        "params": params,
        "neighbors": neighbors,
        "risk": risk,
        "initial": initial,
        "campaign_day": campaign_day,
        "start_index": start_index,
        "end_index": end_index,
        "day_values": day_values,
    }


def run_setting(
    module: types.ModuleType,
    base_params: Any,
    data: Dict[str, Any],
    family: str,
    setting: str,
    seeds: Sequence[int],
    val_start: int,
    val_end: int,
    perturb: Dict[str, Any],
) -> Dict[str, Any]:
    inputs = setting_inputs(family, setting, base_params, data, perturb, val_start, val_end)
    rho = float(inputs["params"].rho_info_decay)
    prefix = memory_prefix(module, data["received"][:val_start], rho)
    initial_mem = prefix[-1].copy()
    age_oh, age_names, occ_oh, occ_names = encode_demographics(module, data)
    state_runs, info_runs, prob_runs = [], [], []
    objectives = []
    observed = data["wearing"][val_start:val_end]
    for seed in seeds:
        states, info, probs = invoke_simulate(
            module=module,
            params=inputs["params"],
            data=data,
            initial_states=inputs["initial"],
            risk=inputs["risk"],
            neighbors=inputs["neighbors"],
            age_oh=age_oh,
            age_names=age_names,
            occ_oh=occ_oh,
            occ_names=occ_names,
            start_index=inputs["start_index"],
            end_index=inputs["end_index"],
            day_values=inputs["day_values"],
            initial_mem=initial_mem,
            seed=int(seed),
            campaign_day=inputs["campaign_day"],
        )
        state_runs.append(states)
        info_runs.append(info)
        prob_runs.append(probs)
        if family != "intervention" and states.shape == observed.shape:
            objectives.append(objective_per_run(states, probs, observed, inputs["initial"]))
    return {
        "states": np.stack(state_runs),
        "info": np.stack(info_runs),
        "probs": np.stack(prob_runs),
        "daily_rates": np.stack([run.mean(axis=1) for run in state_runs]),
        "objectives": np.asarray(objectives, dtype=float),
        "initial": inputs["initial"],
        "risk": inputs["risk"],
        "campaign_day": inputs["campaign_day"],
    }


def save_setting(path: Path, result: Dict[str, Any], seeds: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seeds=np.asarray(seeds, dtype=int),
        states=result["states"],
        info=result["info"],
        probs=result["probs"],
        daily_rates=result["daily_rates"],
        objectives=result["objectives"],
        initial=result["initial"],
        risk=result["risk"],
        campaign_day=np.asarray([result["campaign_day"]], dtype=int),
    )


def load_setting(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def response_time(campaign_rates: np.ndarray, no_campaign_rates: np.ndarray) -> np.ndarray:
    """Center of absolute campaign response; NaN when a pathway has no effect."""
    response = np.abs(campaign_rates - no_campaign_rates)
    mass = response.sum(axis=1)
    clock = np.arange(response.shape[1], dtype=float)
    timing = np.full(response.shape[0], np.nan, dtype=float)
    active = mass > 1e-12
    timing[active] = (response[active] * clock).sum(axis=1) / mass[active]
    return timing


def uplift(daily_rates: np.ndarray, campaign_day: int) -> np.ndarray:
    before = daily_rates[:, campaign_day - 3:campaign_day].mean(axis=1)
    after = daily_rates[:, campaign_day + 1:campaign_day + 4].mean(axis=1)
    return after - before


def compute_deltas(base_dir: Path, method: str, perturb: Dict[str, Any]) -> Dict[str, np.ndarray]:
    def get(family: str, setting: str) -> Dict[str, np.ndarray]:
        return load_setting(base_dir / "trajectories" / method / family / f"{setting}.npz")

    intervention = {setting: get("intervention", setting) for setting in SETTINGS["intervention"]}
    social = {setting: get("social", setting) for setting in SETTINGS["social"]}
    persistence = {setting: get("persistence", setting) for setting in SETTINGS["persistence"]}
    risk = {setting: get("risk", setting) for setting in SETTINGS["risk"]}

    deltas: Dict[str, np.ndarray] = {}
    deltas["I1"] = uplift(intervention["strength_high"]["daily_rates"], 10) - uplift(
        intervention["strength_low"]["daily_rates"], 10
    )
    no_campaign = intervention["no_campaign"]["daily_rates"]
    time_base = response_time(intervention["baseline"]["daily_rates"], no_campaign)
    time_early = response_time(intervention["timing_early"]["daily_rates"], no_campaign)
    time_late = response_time(intervention["timing_late"]["daily_rates"], no_campaign)
    deltas["I2"] = time_base - time_early
    deltas["I3"] = time_late - time_base
    deltas["S1"] = social["work_2"]["daily_rates"].mean(axis=1) - social["work_0"]["daily_rates"].mean(axis=1)
    deltas["S2"] = social["community_2"]["daily_rates"].mean(axis=1) - social["community_0"]["daily_rates"].mean(axis=1)
    deltas["S3"] = social["layer_swap"]["objectives"] - social["baseline"]["objectives"]

    base_probs = persistence["baseline"]["probs"][:, 0, :]
    balanced_probs = persistence["balanced_toggle"]["probs"][:, 0, :]
    up, down = perturb["up_indices"], perturb["down_indices"]
    deltas["P1"] = 0.5 * (
        (balanced_probs[:, up] - base_probs[:, up]).mean(axis=1)
        + (base_probs[:, down] - balanced_probs[:, down]).mean(axis=1)
    )
    original = perturb["initial"]
    permuted = perturb["permuted"]
    perm_probs = persistence["state_permutation"]["probs"][:, 0, :]
    changed = np.flatnonzero(original != permuted)
    direction = permuted[changed] - original[changed]
    deltas["P2"] = ((perm_probs[:, changed] - base_probs[:, changed]) * direction).mean(axis=1)

    low, high = perturb["low_indices"], perturb["high_indices"]
    deltas["R1"] = (
        risk["raise_low_risk"]["states"][:, :, low].mean(axis=(1, 2))
        - risk["baseline"]["states"][:, :, low].mean(axis=(1, 2))
    )
    deltas["R2"] = (
        risk["baseline"]["states"][:, :, high].mean(axis=(1, 2))
        - risk["lower_high_risk"]["states"][:, :, high].mean(axis=(1, 2))
    )
    return deltas


def response_curves(base_dir: Path, method: str, perturb: Dict[str, Any]) -> Dict[str, np.ndarray]:
    def get(family: str, setting: str) -> Dict[str, np.ndarray]:
        return load_setting(base_dir / "trajectories" / method / family / f"{setting}.npz")

    i = {setting: get("intervention", setting) for setting in SETTINGS["intervention"]}
    s = {setting: get("social", setting) for setting in SETTINGS["social"]}
    p = {setting: get("persistence", setting) for setting in SETTINGS["persistence"]}
    r = {setting: get("risk", setting) for setting in SETTINGS["risk"]}
    no_campaign = i["no_campaign"]["daily_rates"]
    low, high = perturb["low_indices"], perturb["high_indices"]
    up, down = perturb["up_indices"], perturb["down_indices"]
    base_probs = p["baseline"]["probs"]
    balanced_probs = p["balanced_toggle"]["probs"]
    perm_probs = p["state_permutation"]["probs"]
    original, permuted = perturb["initial"], perturb["permuted"]
    changed = np.flatnonzero(original != permuted)
    direction = permuted[changed] - original[changed]
    return {
        "I1": i["strength_high"]["daily_rates"] - i["strength_low"]["daily_rates"],
        "I2": i["timing_early"]["daily_rates"] - no_campaign,
        "I3": i["timing_late"]["daily_rates"] - no_campaign,
        "S1": s["work_2"]["daily_rates"] - s["work_0"]["daily_rates"],
        "S2": s["community_2"]["daily_rates"] - s["community_0"]["daily_rates"],
        "S3": s["layer_swap"]["daily_rates"] - s["baseline"]["daily_rates"],
        "P1": 0.5 * (
            (balanced_probs[:, :, up] - base_probs[:, :, up]).mean(axis=2)
            + (base_probs[:, :, down] - balanced_probs[:, :, down]).mean(axis=2)
        ),
        "P2": (
            (perm_probs[:, :, changed] - base_probs[:, :, changed])
            * direction[None, None, :]
        ).mean(axis=2),
        "R1": (
            r["raise_low_risk"]["states"][:, :, low]
            - r["baseline"]["states"][:, :, low]
        ).mean(axis=2),
        "R2": (
            r["baseline"]["states"][:, :, high]
            - r["lower_high_risk"]["states"][:, :, high]
        ).mean(axis=2),
    }


def summarize(base_dir: Path, manifest: Dict[str, Any], methods: Sequence[str], perturb: Dict[str, Any]) -> None:
    all_deltas = {method: compute_deltas(base_dir, method, perturb) for method in methods}
    all_curves = {method: response_curves(base_dir, method, perturb) for method in methods}
    clean = all_deltas.get("clean")
    clean_curves = all_curves.get("clean")
    rows = []
    family_map = {
        "I": "intervention", "S": "social", "P": "persistence", "R": "risk"
    }
    for method, probes in all_deltas.items():
        for probe, values in probes.items():
            finite = values[np.isfinite(values)]
            mean = float(np.mean(finite)) if len(finite) else 0.0
            ci = (
                1.96 * float(np.std(finite, ddof=1)) / math.sqrt(len(finite))
                if len(finite) > 1 else 0.0
            )
            row = {
                "method": method,
                "label": manifest["methods"][method]["label"],
                "family": family_map[probe[0]],
                "probe": probe,
                "delta_mean": mean,
                "delta_ci95": ci,
                "positive_fraction": float(np.mean(finite > 0)) if len(finite) else 0.0,
                "blueprint_positive_pass": bool(mean > 0 and len(finite)),
                "blueprint_positive_strong_pass": bool(mean - ci > 0 and len(finite)),
                "finite_rollouts": int(len(finite)),
            }
            if clean is not None and method != "clean":
                clean_finite = clean[probe][np.isfinite(clean[probe])]
                clean_mean = float(np.mean(clean_finite)) if len(clean_finite) else 0.0
                row["reference_sign_agreement"] = bool(
                    len(finite) and len(clean_finite) and mean * clean_mean > 0
                )
                row["effect_error_vs_clean"] = abs(mean - clean_mean)
                paired = values - clean[probe]
                paired = paired[np.isfinite(paired)]
                row["paired_effect_rmse_vs_clean"] = (
                    math.sqrt(float(np.mean(paired ** 2))) if len(paired) else None
                )
                curve_diff = all_curves[method][probe] - clean_curves[probe]
                row["response_curve_rmse_vs_clean"] = math.sqrt(
                    float(np.mean(curve_diff ** 2))
                )
            rows.append(row)
    dump_json(base_dir / "probe_results.json", rows)
    with (base_dir / "probe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    method_summary = {}
    for method in methods:
        if method == "clean":
            continue
        selected = [row for row in rows if row["method"] == method]
        method_summary[method] = {
            "blueprint_positive_passes": sum(row["blueprint_positive_pass"] for row in selected),
            "blueprint_positive_strong_passes": sum(
                row["blueprint_positive_strong_pass"] for row in selected
            ),
            "reference_sign_agreements": sum(row["reference_sign_agreement"] for row in selected),
            "total_probes": len(selected),
            "mean_effect_error_vs_clean": float(np.mean([row["effect_error_vs_clean"] for row in selected])),
            "mean_response_curve_rmse_vs_clean": float(np.mean([
                row["response_curve_rmse_vs_clean"] for row in selected
            ])),
            "families": {
                family: {
                    "blueprint_positive_passes": sum(
                        row["blueprint_positive_pass"]
                        for row in selected if row["family"] == family
                    ),
                    "reference_sign_agreements": sum(
                        row["reference_sign_agreement"]
                        for row in selected if row["family"] == family
                    ),
                    "total": sum(row["family"] == family for row in selected),
                }
                for family in FAMILIES
            },
        }
    dump_json(base_dir / "summary.json", method_summary)


def parse_list(value: Optional[str], defaults: Iterable[str]) -> List[str]:
    if not value:
        return list(defaults)
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="experiments/fixed_snapshot_counterfactual/manifest.json")
    parser.add_argument("--output", default="output/fixed_snapshot_counterfactual")
    parser.add_argument("--methods", help="Comma-separated subset")
    parser.add_argument("--families", help="Comma-separated subset")
    parser.add_argument("--settings", help="Comma-separated subset")
    parser.add_argument("--seeds", help="Comma-separated integer seeds")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = resolve(args.manifest)
    output_dir = resolve(args.output)
    manifest = load_manifest(manifest_path)
    methods = parse_list(args.methods, manifest["methods"].keys())
    families = parse_list(args.families, FAMILIES)
    seeds = [int(value) for value in parse_list(args.seeds, [str(seed) for seed in manifest["seeds"]])]
    settings_filter = set(parse_list(args.settings, []))
    data = load_data(resolve(manifest["data_folder"]))
    val_start, val_end = validation_bounds(data["days"])
    perturb = make_perturbations(data, val_start)
    dump_json(output_dir / "perturbation_manifest.json", {
        "agent_ids": data["agent_ids"].tolist(),
        "validation_start_index": val_start,
        "validation_end_index": val_end,
        "up_indices": perturb["up_indices"].tolist(),
        "down_indices": perturb["down_indices"].tolist(),
        "low_indices": perturb["low_indices"].tolist(),
        "high_indices": perturb["high_indices"].tolist(),
        "risk_q25": perturb["q25"],
        "risk_q75": perturb["q75"],
        "population_mean_original": float(perturb["initial"].mean()),
        "population_mean_balanced": float(perturb["balanced"].mean()),
        "population_mean_permuted": float(perturb["permuted"].mean()),
    })

    frozen_entries = []
    completed = 0
    total = sum(
        sum(1 for setting in SETTINGS[family] if not settings_filter or setting in settings_filter)
        for _method in methods for family in families
    )
    if not args.summarize_only:
        for method in methods:
            for family in families:
                entry = manifest["methods"][method]["families"][family]
                code_path, parameter_path = resolve(entry["code"]), resolve(entry["parameters"])
                if not code_path.exists() or not parameter_path.exists():
                    raise FileNotFoundError(f"Missing frozen artifact: {code_path} or {parameter_path}")
                module, load_audit = safe_load_snapshot(code_path, campaign_day=10)
                params, _ = params_from_json(module, parameter_path)
                frozen_entries.append({
                    "method": method,
                    "family": family,
                    "code": str(code_path),
                    "parameters": str(parameter_path),
                    "code_sha256": sha256(code_path),
                    "parameters_sha256": sha256(parameter_path),
                    "selected_iteration": entry["selected_iteration"],
                    "selection_source": entry["selection_source"],
                    "loader_audit": load_audit,
                })
                for setting in SETTINGS[family]:
                    if settings_filter and setting not in settings_filter:
                        continue
                    result = run_setting(
                        module, params, data, family, setting, seeds,
                        val_start, val_end, perturb
                    )
                    target = output_dir / "trajectories" / method / family / f"{setting}.npz"
                    save_setting(target, result, seeds)
                    completed += 1
                    dump_json(output_dir / "progress.json", {
                        "status": "running",
                        "completed_settings": completed,
                        "total_settings": total,
                        "last": {"method": method, "family": family, "setting": setting},
                    })
                    print(f"[{completed}/{total}] {method}/{family}/{setting}", flush=True)
        dump_json(output_dir / "frozen_snapshot_manifest.json", {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path),
            "prohibitions": manifest["prohibitions"],
            "entries": frozen_entries,
        })
    expected = [
        output_dir / "trajectories" / method / family / f"{setting}.npz"
        for method in methods for family in families for setting in SETTINGS[family]
    ]
    if all(path.exists() for path in expected) and set(FAMILIES).issubset(families):
        summarize(output_dir, manifest, methods, perturb)
        dump_json(output_dir / "progress.json", {
            "status": "complete",
            "completed_settings": len(expected),
            "total_settings": len(expected),
        })
    else:
        dump_json(output_dir / "progress.json", {
            "status": "partial",
            "completed_settings": completed,
            "total_settings": total,
            "summary_deferred": True,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
