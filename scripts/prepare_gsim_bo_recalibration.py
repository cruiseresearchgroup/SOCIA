#!/usr/bin/env python3
"""Prepare BO+TuRBO recalibration copies of the existing G-SIM/ES snapshots."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "experiment_A_gsim_BO_recalibration"

SOURCE_EXPERIMENTS = {
    "D1": PROJECT_ROOT / "output/test_mask_patch√/experiment_A_gsim_D1_es",
    "D2": PROJECT_ROOT / "output/test_mask_patch√/experiment_A_gsim_D2_es_blind",
    "D3": PROJECT_ROOT / "output/test_mask_patch√/experiment_A_gsim_D3_es_blind",
    "D4": PROJECT_ROOT
    / "output/test_mask_patch√/experiment_A_gsim_D4_es_blind_rerun_iter2_api",
    "D5": PROJECT_ROOT / "output/test_mask_patch√/experiment_A_gsim_D5_es_blind",
    "D6": PROJECT_ROOT / "output/test_mask_patch√/experiment_A_gsim_D6_es_blind",
    "D7": PROJECT_ROOT
    / "output/test_mask_patch√/experiment_A_gsim_D7_es_blind_controlled",
}

BO_TEMPLATE = """\
    calibrator = get_calibrator(
        "bo",
        n_trials=300,
        acquisition_function="EI",
        kernel_type="RBF",
        random_state=42,
        metric_type="composite",
        metric_weights={
            "rmse": 0.4,
            "mae": 0.2,
            "brier": 0.2,
            "transition": 0.2,
        },
        normalize_metrics=True,
        fast_mode_iterations=50,
        use_turbo=True,
        turbo_config={
            "trust_region_size": 0.8,
            "success_tolerance": 3,
            "failure_tolerance": 10,
            "expansion_factor": 2.0,
            "contraction_factor": 0.5,
            "min_trust_region": 1e-8,
            "max_trust_region": 1.0,
        },
    )
"""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_insertion_line(source: str) -> int:
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    for node in main.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "cfg"
                and target.attr == "output_folder"
            ):
                return node.lineno - 1
    raise RuntimeError("Could not find cfg.output_folder assignment in main()")


def prepare_snapshot(defect: str, iteration: int, source_path: Path) -> dict:
    source = source_path.read_text(encoding="utf-8")
    insertion_line = find_insertion_line(source)
    lines = source.splitlines(keepends=True)
    artifact_folder = (
        f"outputs_TuRBOCalibrator_gsim_bo_recal_{defect}_iter_{iteration}"
    )
    control_block = BO_TEMPLATE
    transformed = (
        "".join(lines[:insertion_line])
        + control_block
        + lines[insertion_line]
        + f'    cfg.output_folder = "{artifact_folder}"\n'
        + "".join(lines[insertion_line + 1 :])
    )

    ast.parse(transformed)
    if 'get_calibrator(\n        "bo",' not in transformed:
        raise RuntimeError("BO override was not inserted")
    if "use_turbo=True" not in transformed:
        raise RuntimeError("TuRBO is not enabled")
    if "n_trials=300" not in transformed:
        raise RuntimeError("BO trial budget is not 300")
    if "--output_dir" not in transformed:
        raise RuntimeError("Snapshot lacks the required --output_dir bridge")

    destination_dir = OUTPUT_ROOT / defect / f"iter_{iteration}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / "simulation_code_bo.py"
    destination_path.write_text(transformed, encoding="utf-8")

    return {
        "defect": defect,
        "iteration": iteration,
        "source_path": str(source_path),
        "source_sha256": sha256(source),
        "bo_code_path": str(destination_path),
        "bo_code_sha256": sha256(transformed),
        "internal_artifact_folder": artifact_folder,
        "result_dir": str(destination_dir / "results"),
        "status": "prepared",
    }


def main() -> None:
    manifest = []
    for defect, source_root in SOURCE_EXPERIMENTS.items():
        for iteration in range(3):
            source_path = source_root / f"simulation_code_iter_{iteration}.py"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            manifest.append(prepare_snapshot(defect, iteration, source_path))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest)} BO recalibration snapshots")
    print(manifest_path)


if __name__ == "__main__":
    main()
