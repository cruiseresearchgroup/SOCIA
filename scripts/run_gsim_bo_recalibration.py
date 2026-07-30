#!/usr/bin/env python3
"""Run the prepared G-SIM BO recalibration snapshots without any agent/API calls."""

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
OUTPUT_ROOT = PROJECT_ROOT / "output" / "experiment_A_gsim_BO_recalibration"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"


def load_result(result_dir: Path) -> dict | None:
    result_path = result_dir / "results.json"
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = payload.get("metrics", {})
    calibrator = payload.get("calibrator", {})
    bo_config = calibrator.get("bo_config", {})
    n_trials = calibrator.get("n_trials", bo_config.get("n_trials"))
    use_turbo = calibrator.get("use_turbo", bo_config.get("use_turbo"))
    if metrics.get("val_loss") is None:
        return None
    if n_trials != 300:
        return None
    if use_turbo is not True:
        return None
    return payload


def run_one(item: dict) -> dict:
    defect = item["defect"]
    iteration = item["iteration"]
    code_path = Path(item["bo_code_path"])
    result_dir = Path(item["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir.parent / "run.log"

    existing = load_result(result_dir)
    if existing is not None:
        return {
            **item,
            "status": "skipped_valid",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "metrics": existing.get("metrics", {}),
            "log_path": str(log_path),
        }

    env = os.environ.copy()
    env.update(
        {
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "DATA_PATH": "data_fitting/mask_adoption_data/",
            "MPLCONFIGDIR": f"/private/tmp/socia_mpl_bo_recal_{defect}_i{iteration}",
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
    result = load_result(result_dir)
    status = "completed" if process.returncode == 0 and result is not None else "failed"
    return {
        **item,
        "status": status,
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
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **item,
                    "status": "failed",
                    "returncode": None,
                    "elapsed_seconds": None,
                    "metrics": {},
                    "error": repr(exc),
                }
            results.append(result)
            print(
                f"{result['defect']} iter_{result['iteration']}: "
                f"{result['status']} "
                f"({result.get('elapsed_seconds') or 0:.1f}s)",
                flush=True,
            )

    results.sort(key=lambda item: (item["defect"], item["iteration"]))
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
