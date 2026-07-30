#!/usr/bin/env python3
"""Assemble adjacent-round SRR N2A strategy-evaluation prompts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "output" / "test_data_analysis_srr_llmob_n2a_6iter"
PROMPT_DIR = RUN_DIR / "strategy_evaluation_prompts"

RUBRIC = """You are evaluating a diagnosis / repair-strategy artifact for an iterative simulator-repair task. The simulator is an agent-based or micro-to-macro simulator. A repair strategy is generated after observing that the current simulator still has residual errors after calibration. Your task is to evaluate whether the proposed diagnosis and repair strategy are well supported by the available evidence. You will be given some or all of the following information:
Relevant code context:
key simulator functions,
parameter definitions,
update rules,
calibration/evaluation code,
and any code region referenced by the strategy.
Previous-round metrics:
aggregate and/or mechanism-specific metrics before the current repair,
for example RMSE, MAE, Brier score, transition fit, subgroup errors, intervention response errors, OOD metrics, or other task-specific indicators.
Current-round metrics:
the same or comparable metrics after the current repair and recalibration.
Previous-round strategy artifact:
the proposed diagnosis,
claimed faulty mechanism,
proposed code location,
proposed repair,
expected metric or behavioral effect,
and any rationale produced by the system.
Optional reference information:
known injected defect,
reference mechanism,
expected correct code region,
or expected counterfactual/intervention behavior. If this information is not provided, do NOT assume there is a ground-truth causal mechanism. In that case, evaluate whether the diagnosis is supported by the observed residuals, code context, proposed repair, and post-repair metric changes. Important evaluation principle: You are NOT asked to decide whether the simulator recovers the true real-world causal mechanism unless a known injected defect or reference mechanism is explicitly provided. When no ground truth is provided, rate the artifact as an evidence-supported diagnosis, not as a definitive causal discovery result. Rate the strategy on four dimensions using a 0–2 scale. Dimension 1: Fault localization Question: Does the diagnosis identify a mechanism or simulator component that is plausibly responsible for the observed residual pattern? Score: 0 = The diagnosis identifies an unrelated, unsupported, or purely generic mechanism. It is not meaningfully connected to the observed metric changes or code context. 1 = The diagnosis identifies a partially relevant mechanism, but it is incomplete, too broad, weakly supported, or does not clearly explain the key residual pattern. 2 = The diagnosis identifies the known injected/reference mechanism if provided; otherwise, it identifies a specific mechanism that is strongly supported by the residual pattern, code context, and metric changes. Dimension 2: Code localization Question: Does the strategy point to the correct or most relevant implementation region for the diagnosed mechanism? Score: 0 = The proposed code location is missing, wrong, or unrelated to the diagnosed mechanism. 1 = The proposed location is broadly relevant, such as the correct module or function, but it is imprecise or includes substantial irrelevant code. 2 = The proposed location is precise and actionable, such as the correct function, update block, parameter path, or line-level code region corresponding to the diagnosed mechanism. Dimension 3: Mechanism linkage Question: Does the proposed repair directly implement the diagnosed mechanism-level fix, rather than merely changing parameters or making an unrelated metric-driven adjustment? Score: 0 = The proposed repair is unrelated to the diagnosis, changes an unrelated part of the simulator, or appears to be metric chasing without mechanism correspondence. 1 = The proposed repair is plausibly related to the diagnosis but indirect, incomplete, underspecified, or likely to address only part of the mechanism. 2 = The proposed repair directly implements the diagnosed mechanism-level correction. If a known injected defect or reference mechanism is provided, the repair is exact or behaviorally equivalent. If no ground truth is provided, the repair is tightly linked to the diagnosis and code context. Dimension 4: Post-edit effectiveness Question: Do the current-round metrics indicate that the repair improved the targeted behavior without causing major regressions? Score: 0 = The repair does not improve the relevant target metrics, causes major regressions, or improves only unrelated metrics while the diagnosed residual remains unresolved. 1 = The repair produces partial improvement, mixed trade-offs, or improves aggregate metrics while leaving mechanism-specific indicators unclear or weak. 2 = The repair improves the targeted residual or mechanism-specific metric and does not introduce major regressions. If held-out, subgroup, transition, OOD, or intervention metrics are provided, they should also be consistent with the expected repair effect. Additional instructions:
Prefer mechanism-specific evidence over aggregate metrics alone.
Do not give a high fault-localization score merely because the strategy sounds plausible.
Do not give a high code-localization score unless the code region is specific and actionable.
Do not give a high mechanism-linkage score if the proposed repair only changes a tunable parameter while the diagnosis claims a structural mechanism error.
Do not give a high post-edit-effectiveness score if only the overall RMSE improves but the diagnosed mechanism-specific residual remains unchanged or unreported.
Penalize generic advice such as “improve calibration,” “adjust parameters,” or “add more features” unless it is tied to a concrete mechanism and code path.
If the current metrics improve but the proposed diagnosis is unsupported, score post-edit effectiveness higher only if justified, but keep fault localization and mechanism linkage low.
If no known injected defect is provided, explicitly evaluate evidence support rather than causal truth.
If the evidence is insufficient to judge a dimension, assign 1 only when the artifact is partially supported; assign 0 when the artifact is unsupported or unverifiable. Return JSON only. Do not include markdown, prose, or extra commentary outside the JSON. Required JSON format:
{ "fault_localization": 0, "code_localization": 0, "mechanism_linkage": 0, "post_edit_effectiveness": 0, "brief_rationale": { "fault_localization": "One concise sentence explaining the score.", "code_localization": "One concise sentence explaining the score.", "mechanism_linkage": "One concise sentence explaining the score.", "post_edit_effectiveness": "One concise sentence explaining the score." }, "overall_assessment": "One concise sentence summarizing whether the strategy is evidence-supported, partially supported, or unsupported." }
"""


def first_json_value(raw: str):
    """Return the first JSON value, excluding accidental trailing duplicate output."""
    stripped = raw.lstrip()
    value, _ = json.JSONDecoder().raw_decode(stripped)
    return value


def simulation_metrics(iteration: int):
    payload = json.loads(
        (RUN_DIR / f"simulation_results_iter_{iteration}.json").read_text()
    )
    return payload["simulation_metrics"]


def strategy_artifact(iteration: int):
    payload = json.loads((RUN_DIR / f"feedback_iter_{iteration}.json").read_text())
    summary = payload["summary"]
    try:
        return first_json_value(summary)
    except (json.JSONDecodeError, TypeError):
        return summary


def assemble(previous_iteration: int) -> str:
    current_iteration = previous_iteration + 1
    previous_metrics = json.dumps(
        {"simulation_metrics": simulation_metrics(previous_iteration)},
        ensure_ascii=False,
        indent=2,
    )
    current_metrics = json.dumps(
        {"simulation_metrics": simulation_metrics(current_iteration)},
        ensure_ascii=False,
        indent=2,
    )
    code = (
        RUN_DIR / f"simulation_code_iter_{current_iteration}.py"
    ).read_text().rstrip()
    artifact = strategy_artifact(previous_iteration)
    if isinstance(artifact, str):
        artifact_text = artifact
    else:
        artifact_text = json.dumps(artifact, ensure_ascii=False, indent=2)

    return (
        RUBRIC.rstrip()
        + "\n\n#################################\n"
        + "Previous-round metrics:\n"
        + "#################################\n\n"
        + previous_metrics
        + "\n\n#################################\n"
        + "Current-round metrics:\n"
        + "#################################\n\n"
        + current_metrics
        + "\n\n#################################\n"
        + "Relevant code context:\n"
        + "#################################\n\n"
        + code
        + "\n\n#################################\n"
        + "Previous-round strategy artifact:\n"
        + "#################################\n\n"
        + artifact_text.rstrip()
        + "\n"
    )


def main() -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for previous_iteration in range(5):
        current_iteration = previous_iteration + 1
        filename = (
            f"strategy_evaluation_prompt_iter_{previous_iteration}_to_"
            f"{current_iteration}.txt"
        )
        path = PROMPT_DIR / filename
        prompt = assemble(previous_iteration)
        path.write_text(prompt)
        manifest.append(
            {
                "transition": f"{previous_iteration}->{current_iteration}",
                "file": filename,
                "characters": len(prompt),
            }
        )
    (PROMPT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
