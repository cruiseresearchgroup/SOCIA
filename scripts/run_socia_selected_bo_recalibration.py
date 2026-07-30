#!/usr/bin/env python3
"""Run selected SOCIA snapshots with the fixed BO+TuRBO calibrator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    PROJECT_ROOT / "output" / "experiment_socia_selected_logit_bo_recalibration"
)
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"


def load_valid_result(result_dir: Path) -> dict | None:
    path = result_dir / "results.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    meta = payload.get("parameters", {}).get("meta", {})
    bo_config = meta.get("bo_config", {})
    if payload.get("metrics", {}).get("val_loss") is None:
        return None
    if meta.get("calibrator_name") != "bayesian_optimization":
        return None
    if meta.get("n_trials") != 300 or bo_config.get("use_turbo") is not True:
        return None
    if payload.get("artifact_schema") != "socia_ace_validation_v1":
        return None
    return payload


def run_one(item: dict) -> dict:
    defect = item["defect"]
    code_path = Path(item["bo_code_path"])
    result_dir = Path(item["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir.parent / "run.log"

    existing = load_valid_result(result_dir)
    if existing is not None:
        return {
            **item,
            "status": "skipped_valid",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "metrics": existing["metrics"],
        }

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "DATA_PATH": "data_fitting/mask_adoption_data/",
            "MPLCONFIGDIR": f"/private/tmp/socia_selected_bo_{defect}",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        str(code_path),
        "--output_dir",
        str(result_dir),
    ]

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    result = load_valid_result(result_dir)
    return {
        **item,
        "status": (
            "completed"
            if process.returncode == 0 and result is not None
            else "failed"
        ),
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "metrics": result.get("metrics", {}) if result else {},
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, item): item for item in manifest}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['defect']}: {result['status']} "
                f"({result['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    results.sort(key=lambda item: item["defect"])
    summary_path = OUTPUT_ROOT / "run_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    failures = [item for item in results if item["status"] == "failed"]
    print(f"Finished {len(results)} tasks; failures={len(failures)}")
    print(summary_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
