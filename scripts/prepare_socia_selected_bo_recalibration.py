#!/usr/bin/env python3
"""Prepare fixed BO+TuRBO copies of selected SOCIA logit-head snapshots."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT / "output" / "test_mask_patch_socia_test_error_injection"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "output" / "experiment_socia_selected_logit_bo_recalibration"
)

SELECTED_SNAPSHOTS = {
    "D2": SOURCE_ROOT / "injected error D2" / "simulation_code_iter_2.py",
    "D5": SOURCE_ROOT / "injected error D5" / "simulation_code_iter_2.py",
    "D6": SOURCE_ROOT / "injected error D6" / "simulation_code_iter_1.py",
    "D7": SOURCE_ROOT / "injected error D7" / "simulation_code_iter_1.py",
}

BO_OVERRIDE = """\
# Fixed recalibration-only override; simulator logic above is unchanged.
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


def find_override_location(source: str) -> tuple[int, str]:
    tree = ast.parse(source)
    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if not main_functions:
        raise RuntimeError("Could not find main()")
    main = main_functions[-1]
    calibrator_assignments = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "calibrator"
            for target in node.targets
        ):
            calibrator_assignments.append(node)
    if not calibrator_assignments:
        raise RuntimeError("Could not find calibrator assignment in main()")
    assignment = max(calibrator_assignments, key=lambda node: node.lineno)
    source_line = source.splitlines()[assignment.lineno - 1]
    indentation = source_line[: len(source_line) - len(source_line.lstrip())]
    return assignment.end_lineno, indentation


def prepare_snapshot(defect: str, source_path: Path) -> dict:
    source = source_path.read_text(encoding="utf-8")
    insertion_line, indentation = find_override_location(source)
    lines = source.splitlines(keepends=True)
    override = "".join(
        indentation + line if line.strip() else line
        for line in BO_OVERRIDE.splitlines(keepends=True)
    )
    transformed = (
        "".join(lines[:insertion_line])
        + override
        + "".join(lines[insertion_line:])
    )

    ast.parse(transformed)
    if "calibrator = get_calibrator(" not in transformed:
        raise RuntimeError("BO override was not inserted")
    if "n_trials=300" not in transformed or "use_turbo=True" not in transformed:
        raise RuntimeError("Required BO+TuRBO configuration is missing")
    if "--output_dir" not in transformed:
        raise RuntimeError("Snapshot lacks the required --output_dir bridge")

    destination_dir = OUTPUT_ROOT / defect
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / "simulation_code_bo.py"
    destination_path.write_text(transformed, encoding="utf-8")

    return {
        "defect": defect,
        "source_path": str(source_path),
        "source_sha256": sha256(source),
        "bo_code_path": str(destination_path),
        "bo_code_sha256": sha256(transformed),
        "result_dir": str(destination_dir / "results"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = [
        prepare_snapshot(defect, source_path)
        for defect, source_path in SELECTED_SNAPSHOTS.items()
    ]
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Prepared {len(manifest)} selected SOCIA snapshots")
    print(manifest_path)


if __name__ == "__main__":
    main()
