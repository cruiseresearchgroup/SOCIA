#!/usr/bin/env python3
"""Create one reproducible SRR starting program for each mask-simulator defect."""

from __future__ import annotations

import json
import shutil
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "test_mask_patch_socia_test_error_injection" / (
    "simulation_code_using_calibration_template_SBI_BO_TuRBO_EVO_error_injection.py"
)
TASK_SPEC = ROOT / "output" / "test_mask_patch_socia_test_error_injection" / "injected error D1" / "task_spec_iter_0.json"
DEFAULT_DESTINATION = ROOT / "output" / "test_mask_patch_srr_error_injection"
PROGRAM_NAME = "simulation_code_using_calibration_template_SBI_BO_TuRBO_EVO_error_injection.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Apply exactly one intentional mutation, failing if the clean anchor moved."""
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def clean_program() -> str:
    """Restore the current source's only active injected high-risk sign inversion."""
    text = SOURCE.read_text(encoding="utf-8")
    return replace_once(
        text,
        "    risk_effect = params.beta_r * risk\n    risk_effect[risk > 0.4] *= -1.0\n",
        "    risk_effect = params.beta_r * risk\n",
        "clean D7 restoration",
    )


def inject(defect: str, text: str) -> str:
    """Return a program with exactly one structural mutation."""
    if defect == "D1":
        return replace_once(
            text,
            "        if global_day > 10:\n            lam *= params.lambda_broadcast_factor_after_day10\n",
            "",
            defect,
        )
    if defect == "D2":
        return replace_once(text, "        + params.beta_i * mem_info\n", "", defect)
    if defect == "D3":
        return replace_once(
            text,
            "        + params.theta_w * share_w\n        + params.theta_c * share_c\n",
            "        + 0.0 * share_w\n        + 0.0 * share_c\n",
            defect,
        )
    if defect == "D4":
        return replace_once(
            text,
            "        + params.theta_w * share_w\n        + params.theta_c * share_c\n",
            "        + params.theta_w * share_c\n        + params.theta_c * share_w\n",
            defect,
        )
    if defect == "D5":
        return replace_once(text, "        + params.gamma * prev_states\n", "        + 0.0 * prev_states\n", defect)
    if defect == "D6":
        return replace_once(
            text,
            "        + params.gamma * prev_states\n",
            "        + params.gamma * np.mean(prev_states)\n",
            defect,
        )
    if defect == "D7":
        return replace_once(
            text,
            "    risk_effect = params.beta_r * risk\n",
            "    risk_effect = params.beta_r * risk\n    risk_effect[risk > 0.4] *= -1.0\n",
            defect,
        )
    raise ValueError(f"Unsupported defect: {defect}")


def main(destination: Path) -> None:
    """Create isolated D1--D7 starting directories and an auditable manifest."""
    destination = destination.resolve()
    clean = clean_program()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "baseline_source": str(SOURCE.relative_to(ROOT)),
        "task_spec_source": str(TASK_SPEC.relative_to(ROOT)),
        "clean_restore": "removed the active D7 high-risk sign inversion",
        "defects": {},
    }
    for defect in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        run_dir = destination / f"injected_error_{defect}"
        run_dir.mkdir(exist_ok=True)
        program = inject(defect, clean)
        (run_dir / PROGRAM_NAME).write_text(program, encoding="utf-8")
        shutil.copy2(TASK_SPEC, run_dir / "task_spec_iter_0.json")
        manifest["defects"][defect] = {
            "start_code": str((run_dir / PROGRAM_NAME).relative_to(ROOT)),
            "task_spec": str((run_dir / "task_spec_iter_0.json").relative_to(ROOT)),
        }
    (destination / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    main(parser.parse_args().destination)
