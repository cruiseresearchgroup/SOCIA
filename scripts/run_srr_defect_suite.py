#!/usr/bin/env python3
"""Run the isolated SRR D1--D7 recovery suite serially with artifact gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_NAME = "simulation_code_using_calibration_template_SBI_BO_TuRBO_EVO_error_injection.py"
TASK = "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks."
ALL_DEFECTS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
FROZEN_SRR_FILES = (
    ROOT / "templates" / "code_generation_srr_prompt.txt",
    ROOT / "templates" / "code_generation_srr_patch_prompt.txt",
    ROOT / "templates" / "feedback_generation_srr_prompt.txt",
    ROOT / "agents" / "code_generation_srr" / "agent.py",
    ROOT / "agents" / "feedback_generation_srr" / "agent.py",
)


def save_progress(path: Path, payload: dict[str, Any]) -> None:
    """Write machine-readable suite progress after each state transition."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_iteration(run_dir: Path, iteration: int) -> dict[str, Any]:
    """Require successful execution, metrics, and active BO+TuRBO metadata."""
    result_path = run_dir / f"simulation_results_iter_{iteration}.json"
    if not result_path.exists():
        raise RuntimeError(f"missing {result_path.name}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = result.get("simulation_metrics")
    if result.get("execution_status") != "success" or not isinstance(metrics, dict) or metrics.get("val_loss") is None:
        raise RuntimeError(f"iteration {iteration} has no valid top-level metric schema")
    output = run_dir / f"output_iter_{iteration}" / "results.json"
    if not output.exists():
        raise RuntimeError(f"iteration {iteration} missing results.json artifact")
    payload = json.loads(output.read_text(encoding="utf-8"))
    calibrator = payload.get("calibrator", {})
    bo = calibrator.get("calibrator_name") == "bayesian_optimization"
    turbo = bool(calibrator.get("bo_config", {}).get("use_turbo"))
    if not (bo and turbo):
        raise RuntimeError(f"iteration {iteration} did not preserve BO+TuRBO")
    return metrics


def code_sha256(path: Path) -> str:
    """Return a stable digest used to prove that an iteration changed code."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_srr_manifest() -> dict[str, str]:
    """Record the exact generic SRR implementation used for every defect."""
    return {
        str(path.relative_to(ROOT)): code_sha256(path)
        for path in FROZEN_SRR_FILES
    }


def main() -> None:
    """Prepare the suite then execute defects sequentially under fixed settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--defects",
        nargs="+",
        choices=ALL_DEFECTS,
        default=list(ALL_DEFECTS),
    )
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    subprocess.run(
        [sys.executable, "scripts/prepare_srr_defect_suite.py", "--destination", str(output)],
        cwd=ROOT,
        check=True,
    )
    progress_path = output / "suite_progress.json"
    progress: dict[str, Any] = {
        "mode": "srr",
        "status": "running",
        "frozen_srr_sha256": frozen_srr_manifest(),
        "defects": {},
    }
    save_progress(progress_path, progress)
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(ROOT)
    env["DATA_PATH"] = "data_fitting/mask_adoption_data/"
    for defect in args.defects:
        run_dir = output / f"injected_error_{defect}"
        progress["current_defect"] = defect
        progress["defects"][defect] = {"status": "running"}
        save_progress(progress_path, progress)
        command = [
            sys.executable, "main.py", "--task", TASK,
            "--task-file", "examples/mask_adoption_task.json", "--output", str(run_dir),
            "--selfloop", "3", "--mode=SRR", "--auto",
            "--iterations", str(args.iterations),
            "--persisted-data-analysis-file", str(run_dir / "task_spec_iter_0.json"),
            "--persisted-code-file", str(run_dir / PROGRAM_NAME),
        ]
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        (run_dir / "suite_command_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / "suite_command_stderr.log").write_text(completed.stderr, encoding="utf-8")
        try:
            metrics = {
                str(i): verify_iteration(run_dir, i)
                for i in range(args.iterations)
            }
            if completed.returncode != 0:
                raise RuntimeError(f"runner exited {completed.returncode}")
            code_paths = [
                run_dir / f"simulation_code_iter_{i}.py"
                for i in range(args.iterations)
            ]
            if not all(path.exists() for path in code_paths):
                raise RuntimeError("one or more iteration code artifacts are missing")
            code_hashes = [code_sha256(path) for path in code_paths]
            if args.iterations > 1 and len(set(code_hashes)) == 1:
                raise RuntimeError(
                    "SRR produced no executable code change across iterations"
                )
        except Exception as error:
            progress["defects"][defect] = {"status": "failed", "error": str(error)}
            progress["status"] = "blocked"
            save_progress(progress_path, progress)
            raise
        progress["defects"][defect] = {
            "status": "complete",
            "metrics": metrics,
            "code_sha256": {
                str(i): digest for i, digest in enumerate(code_hashes)
            },
        }
        save_progress(progress_path, progress)
    progress["status"] = "complete"
    progress.pop("current_defect", None)
    save_progress(progress_path, progress)


if __name__ == "__main__":
    main()
