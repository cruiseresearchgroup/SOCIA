#!/usr/bin/env python3
"""Post-hoc exploratory fixed-snapshot layer-weight operability probes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import fixed_snapshot_counterfactual_eval as core


ROOT = Path(__file__).resolve().parents[1]
LAYERS = {
    "family": "w_family",
    "work": "w_work",
    "community": "w_community",
}


def run_condition(
    module: Any,
    params: Any,
    data: Dict[str, Any],
    seeds: Sequence[int],
    val_start: int,
    val_end: int,
    attribute: str,
    multiplier: float,
) -> Dict[str, np.ndarray]:
    changed = copy.deepcopy(params)
    frozen_value = float(getattr(changed, attribute))
    setattr(changed, attribute, multiplier * frozen_value)
    age_oh, age_names, occ_oh, occ_names = core.encode_demographics(module, data)
    prefix = core.memory_prefix(
        module,
        data["received"][:val_start],
        float(changed.rho_info_decay),
    )
    initial_mem = prefix[-1].copy()
    initial = data["wearing"][val_start - 1].copy()
    states, info, probs = [], [], []
    for seed in seeds:
        state, received, probability = core.invoke_simulate(
            module=module,
            params=changed,
            data=data,
            initial_states=initial,
            risk=data["risk"].copy(),
            neighbors=copy.deepcopy(data["neighbors"]),
            age_oh=age_oh,
            age_names=age_names,
            occ_oh=occ_oh,
            occ_names=occ_names,
            start_index=val_start - 1,
            end_index=val_end - 1,
            day_values=data["days"],
            initial_mem=initial_mem,
            seed=int(seed),
            campaign_day=10,
        )
        states.append(state)
        info.append(received)
        probs.append(probability)
    states_array = np.stack(states)
    return {
        "states": states_array,
        "info": np.stack(info),
        "probs": np.stack(probs),
        "daily_rates": states_array.mean(axis=2),
        "frozen_weight": np.asarray([frozen_value]),
        "applied_weight": np.asarray([multiplier * frozen_value]),
    }


def save_condition(path: Path, result: Dict[str, np.ndarray], seeds: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, seeds=np.asarray(seeds), **result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/fixed_snapshot_counterfactual/manifest.json",
    )
    parser.add_argument(
        "--probe-manifest",
        default="experiments/fixed_snapshot_counterfactual/layer_weight_exploratory_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="output/fixed_snapshot_layer_weight_exploratory",
    )
    args = parser.parse_args()
    manifest_path = core.resolve(args.manifest)
    probe_manifest_path = core.resolve(args.probe_manifest)
    output = core.resolve(args.output)
    manifest = core.load_manifest(manifest_path)
    with probe_manifest_path.open("r", encoding="utf-8") as handle:
        probe_manifest = json.load(handle)
    seeds = [int(seed) for seed in probe_manifest["seeds"]]
    data = core.load_data(core.resolve(manifest["data_folder"]))
    val_start, val_end = core.validation_bounds(data["days"])
    rows: List[Dict[str, Any]] = []
    frozen_entries = []

    for method, method_entry in manifest["methods"].items():
        entry = method_entry["families"]["social"]
        code_path = core.resolve(entry["code"])
        parameter_path = core.resolve(entry["parameters"])
        module, audit = core.safe_load_snapshot(code_path, campaign_day=10)
        params, _ = core.params_from_json(module, parameter_path)
        frozen_entries.append({
            "method": method,
            "code": str(code_path),
            "parameters": str(parameter_path),
            "code_sha256": core.sha256(code_path),
            "parameters_sha256": core.sha256(parameter_path),
            "loader_audit": audit,
        })
        for layer, attribute in LAYERS.items():
            results = {}
            for label, multiplier in (("low", 0.0), ("high", 2.0)):
                result = run_condition(
                    module, params, data, seeds, val_start, val_end,
                    attribute, multiplier,
                )
                save_condition(
                    output / "trajectories" / method / layer / f"{label}.npz",
                    result,
                    seeds,
                )
                results[label] = result
            delta = (
                results["high"]["daily_rates"].mean(axis=1)
                - results["low"]["daily_rates"].mean(axis=1)
            )
            mean = float(delta.mean())
            ci = 1.96 * float(delta.std(ddof=1)) / math.sqrt(len(delta))
            rows.append({
                "method": method,
                "label": method_entry["label"],
                "probe": f"LW_{layer}",
                "delta_mean": mean,
                "delta_ci95": ci,
                "positive_fraction": float(np.mean(delta > 0)),
                "directional_pass": bool(mean > 0),
                "strong_pass": bool(mean - ci > 0),
                "frozen_weight": float(results["low"]["frozen_weight"][0]),
            })
            print(
                f"{method}/LW_{layer}: delta={mean:.6f}, ci95={ci:.6f}",
                flush=True,
            )

    core.dump_json(output / "results.json", rows)
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in manifest["methods"]:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "directional_passes": sum(row["directional_pass"] for row in selected),
            "strong_passes": sum(row["strong_pass"] for row in selected),
            "total": len(selected),
        }
    core.dump_json(output / "summary.json", summary)
    core.dump_json(output / "frozen_snapshot_manifest.json", {
        "parent_manifest": str(manifest_path),
        "parent_manifest_sha256": core.sha256(manifest_path),
        "probe_manifest": str(probe_manifest_path),
        "probe_manifest_sha256": core.sha256(probe_manifest_path),
        "status": probe_manifest["status"],
        "constraints": probe_manifest["constraints"],
        "entries": frozen_entries,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
