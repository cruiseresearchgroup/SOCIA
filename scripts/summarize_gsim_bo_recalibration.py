#!/usr/bin/env python3
"""Summarize paired ES and BO metrics for the 21 G-SIM snapshots."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "experiment_A_gsim_BO_recalibration"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

METRICS = (
    "val_loss",
    "RMSE_aggregate_mean",
    "MAE_aggregate_mean",
    "Brier_mean",
    "TransitionFit_mean",
)


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    missing = [name for name in METRICS if metrics.get(name) is None]
    if missing:
        raise RuntimeError(f"{path}: missing metrics {missing}")
    return {name: float(metrics[name]) for name in METRICS}


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paired_rows = []
    for item in manifest:
        source_code = Path(item["source_path"])
        source_root = source_code.parent
        iteration = item["iteration"]
        es_result_path = source_root / f"output_iter_{iteration}" / "results.json"
        bo_result_path = Path(item["result_dir"]) / "results.json"
        es_metrics = read_metrics(es_result_path)
        bo_metrics = read_metrics(bo_result_path)
        es_payload = json.loads(es_result_path.read_text(encoding="utf-8"))
        bo_payload = json.loads(bo_result_path.read_text(encoding="utf-8"))
        behavioral_parameter_sections = (
            "decision_weights",
            "layer_weights",
            "info_params",
            "noise_params",
        )
        es_parameters = es_payload.get("parameters", {})
        bo_parameters = bo_payload.get("parameters", {})
        behavioral_parameters_changed = any(
            es_parameters.get(section) != bo_parameters.get(section)
            for section in behavioral_parameter_sections
        )

        row = {
            "defect": item["defect"],
            "iteration": iteration,
            "source_code": str(source_code),
            "source_sha256": item["source_sha256"],
            "bo_code": item["bo_code_path"],
            "bo_code_sha256": item["bo_code_sha256"],
            "es_result": str(es_result_path),
            "bo_result": str(bo_result_path),
            "bo_behavioral_parameters_changed": behavioral_parameters_changed,
        }
        for metric in METRICS:
            row[f"es_{metric}"] = es_metrics[metric]
            row[f"bo_{metric}"] = bo_metrics[metric]
            row[f"delta_bo_minus_es_{metric}"] = (
                bo_metrics[metric] - es_metrics[metric]
            )
        paired_rows.append(row)

    paired_rows.sort(key=lambda row: (row["defect"], row["iteration"]))
    comparison_json = OUTPUT_ROOT / "comparison.json"
    comparison_csv = OUTPUT_ROOT / "comparison.csv"
    comparison_json.write_text(
        json.dumps(paired_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    defect_rows = []
    for defect in sorted({row["defect"] for row in paired_rows}):
        rows = [row for row in paired_rows if row["defect"] == defect]
        best_es = min(rows, key=lambda row: row["es_val_loss"])
        best_bo = min(rows, key=lambda row: row["bo_val_loss"])
        defect_rows.append(
            {
                "defect": defect,
                "best_es_iteration": best_es["iteration"],
                "best_es_val_loss": best_es["es_val_loss"],
                "best_bo_iteration": best_bo["iteration"],
                "best_bo_val_loss": best_bo["bo_val_loss"],
                "delta_best_bo_minus_best_es": (
                    best_bo["bo_val_loss"] - best_es["es_val_loss"]
                ),
                "bo_better_on_same_snapshot_count": sum(
                    row["bo_val_loss"] < row["es_val_loss"] for row in rows
                ),
            }
        )

    defect_csv = OUTPUT_ROOT / "summary_by_defect.csv"
    with defect_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(defect_rows[0]))
        writer.writeheader()
        writer.writerows(defect_rows)

    deltas = [row["delta_bo_minus_es_val_loss"] for row in paired_rows]
    bo_wins = sum(row["bo_val_loss"] < row["es_val_loss"] for row in paired_rows)
    unchanged = [
        f"{row['defect']} iter {row['iteration']}"
        for row in paired_rows
        if not row["bo_behavioral_parameters_changed"]
    ]
    report_lines = [
        "# G-SIM snapshot recalibration: ES vs BO+TuRBO",
        "",
        "- Snapshots: 21 (D1–D7 × iterations 0–2)",
        "- BO configuration: 300 trials, EI, RBF, composite objective, TuRBO enabled",
        f"- BO lower val_loss on the same snapshot: {bo_wins}/21",
        f"- Mean BO−ES val_loss: {statistics.mean(deltas):.6f}",
        f"- Median BO−ES val_loss: {statistics.median(deltas):.6f}",
        (
            "- Snapshots where BO ran but behavioral parameters were unchanged: "
            + (", ".join(unchanged) if unchanged else "none")
        ),
        "",
        "| Defect | Iter | ES val_loss | BO val_loss | BO−ES | BO parameters applied |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in paired_rows:
        report_lines.append(
            f"| {row['defect']} | {row['iteration']} "
            f"| {row['es_val_loss']:.6f} | {row['bo_val_loss']:.6f} "
            f"| {row['delta_bo_minus_es_val_loss']:+.6f} "
            f"| {'yes' if row['bo_behavioral_parameters_changed'] else 'no'} |"
        )
    report_path = OUTPUT_ROOT / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(comparison_csv)
    print(comparison_json)
    print(defect_csv)
    print(report_path)
    for row in defect_rows:
        print(
            f"{row['defect']}: ES best iter={row['best_es_iteration']} "
            f"loss={row['best_es_val_loss']:.6f}; "
            f"BO best iter={row['best_bo_iteration']} "
            f"loss={row['best_bo_val_loss']:.6f}; "
            f"same-snapshot BO wins={row['bo_better_on_same_snapshot_count']}/3"
        )


if __name__ == "__main__":
    main()
