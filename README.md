<p align="center">
  <img src="docs/images/socia_logo_large.png" alt="SOCIA Logo" width="200px" />
</p>

# SOCIA: Simulation Orchestration for Computational Intelligence with Agents

SOCIA constructs and refines executable social simulators through a structured blueprint, fixed-structure calibration, and evidence-based diagnosis and repair. This repository includes the ACE workflow and the focused experiments used to evaluate controlled mechanism recovery and predicted counterfactual consistency.

## Usage

Run all commands from the repository root. Install the environment first, configure the API key in `keys.py`, and set the project and data paths:

```bash
conda activate SOCIA
pip install -r requirements.txt
export PROJECT_ROOT="$(pwd)"
export DATA_PATH="data_fitting/mask_adoption_data/"
```

### SOCIA ACE workflow

This is the main command for an ACE-mode mask-adoption run. It writes the task specification, generated simulator snapshots, calibration artifacts, diagnosis/repair artifacts, playbook, and log below the selected output directory.

```bash
python main.py \
  --task "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks." \
  --task-file examples/mask_adoption_task.json \
  --output output/together_llama3.3-70b_ace_code_mask \
  --selfloop 3 \
  --mode ace \
  --auto \
  --iterations 3
```

Use `python main.py --help` to view the complete command-line interface. The LLM key is read from `keys.py`; create it with `python setup_api_key.py` if needed.

### A+B: controlled injected-defect recovery and matched recalibration

The A+B study creates the seven isolated mask-adoption defects (D1–D7), runs the frozen SRR comparison, and performs fixed 300-trial BO+TuRBO recalibration over the saved G-SIM and selected SOCIA snapshots. The latter two stages are snapshot-based and do not make agent/API calls.

```bash
# 1. Generate the D1--D7 starting programs and run the SRR recovery suite.
python scripts/run_srr_defect_suite.py \
  --output output/test_mask_patch_srr_error_injection_frozen \
  --iterations 3

# 2. Reproduce matched G-SIM BO+TuRBO recalibration (all saved snapshots).
python scripts/prepare_gsim_bo_recalibration.py
python scripts/run_gsim_bo_recalibration.py --workers 4

# 3. Reproduce fixed BO+TuRBO recalibration of selected SOCIA snapshots.
python scripts/prepare_socia_selected_bo_recalibration.py
python scripts/run_socia_selected_bo_recalibration.py --workers 4
```

The exact source snapshots and their hashes are recorded in the generated `manifest.json` files. The comparison summary is in `output/experiment_A_gsim_BO_recalibration/SOCIA_vs_GSIM.md`.

### C: exploratory layer-weight operability probes

Experiment C evaluates whether the three blueprint-level social-layer weights are behaviorally operative in frozen snapshots. It is explicitly exploratory and is not part of the confirmatory counterfactual score.

```bash
python scripts/fixed_snapshot_layer_weight_probes.py \
  --manifest experiments/fixed_snapshot_counterfactual/manifest.json \
  --probe-manifest experiments/fixed_snapshot_counterfactual/layer_weight_exploratory_manifest.json \
  --output output/fixed_snapshot_layer_weight_exploratory
```

### Predicted Counterfactual Consistency Probes

This evaluation compares fixed simulator snapshots under pre-specified intervention, social, persistence, and risk probes. The runner prevents calibration, fitting, diagnosis, repair, selection, and parameter updates based on probe outcomes.

```bash
python scripts/fixed_snapshot_counterfactual_eval.py \
  --manifest experiments/fixed_snapshot_counterfactual/manifest.json \
  --output output/fixed_snapshot_counterfactual_confirmatory
```

To regenerate only the report from existing probe artifacts, append `--summarize-only`.

## Project structure

```text
main.py                         Main CLI entry point for SOCIA workflows
agents/                         Mode-specific implementations of workflow agents
core/                           Blueprint and ACE playbook persistence
orchestration/                  Dependency-injection container and workflow support
templates/                      Prompts used by generation and diagnosis agents
examples/                       Task specifications
data_fitting/                   Task data and calibration inputs
experiments/                    Frozen manifests for counterfactual experiments
scripts/                        Reproducible A+B, C, and counterfactual runners
output/                         Run artifacts, frozen snapshots, summaries, and logs
requirements.txt                Python dependencies
config.yaml                     Active runtime configuration
```

## Outputs and logs

Each main workflow run writes its artifacts to the directory supplied through `--output`. Key files include `task_spec_iter_*.json`, `simulation_code_iter_*.py`, `simulation_results_iter_*.json`, `verification_results_iter_*.json` (when applicable), `output_iter_*/`, and `socia.log`. ACE playbook snapshots are stored under `playbook_storage/`.

The experiment runners also write manifests, checksums, per-run logs, machine-readable summaries, and result tables to their requested output folders. Existing valid fixed-snapshot calibration artifacts are reused only when their schema and fixed-budget checks pass.

## Agents used in the ACE workflow

- **Task Understanding / Chain-of-Structure agent**: converts the task and data constraints into the simulator blueprint and mechanism requirements.
- **Data Analysis agent**: derives data semantics, empirical targets, and calibration/evaluation interfaces used by the simulator.
- **Code Generation agent**: turns the blueprint and current repair strategy into executable simulator code while preserving the required interface.
- **Simulation Execution agent**: executes generated simulator code and collects the calibration and behavioural results.
- **Evidence-Based Diagnosis (Feedback Generation) agent**: maps residual patterns to a localized mechanism hypothesis and an actionable code-level repair strategy.
- **Iteration Control agent**: accepts or rejects candidates, preserves the best-so-far simulator, and coordinates the next calibration/repair iteration.

The ACE playbook records reusable diagnosis and repair evidence across iterations. Auxiliary legacy or baseline-specific agents remain in the source tree for the experiments but are not part of the ACE workflow described above.

## Disclaimer

SOCIA generates simulator code using backbone LLMs. Generated simulators are research artifacts and are supplied without guarantees of accuracy, safety, or fitness for consequential financial, medical, legal, or policy decisions.
