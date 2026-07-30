# SOCIA

SOCIA constructs and refines executable social simulators with a structured simulator blueprint, calibration, and evidence-based repair. This repository contains the ACE workflow used for the main experiments, plus scripts and frozen artifacts for the controlled supplementary experiments.

## Usage

### Installation

Create the project environment and install the pinned dependencies:

```bash
conda activate SOCIA
pip install -r requirements.txt
```

Set an OpenAI key in `keys.py` with `python setup_api_key.py`, or provide it through the environment if your local configuration supports that. Run all commands from the repository root.

### Main SOCIA ACE workflow

The primary command-line entry point is `main.py`. The following command runs the ACE workflow on Mask Adoption, producing all per-iteration code, metrics, feedback, and playbook artifacts in the chosen output directory:

```bash
conda activate SOCIA && \
export PROJECT_ROOT="$(pwd)" && \
export DATA_PATH="data_fitting/mask_adoption_data/" && \
python main.py \
  --task "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks." \
  --task-file "examples/mask_adoption_task.json" \
  --output "output/ace_mask_adoption" \
  --selfloop 3 \
  --mode ace \
  --auto \
  --iterations 3
```

`--auto` enables non-interactive execution. Omit it to supply feedback interactively. `--iterations` caps outer repair iterations; `--selfloop` caps code-generation self-check attempts.

`--mode random` runs the same G-SIM/SOCIA workflow while appending an
authoritative prompt policy that permits only single-seed uniform random search
for calibration. Generated code and runtime artifacts are audited for
`calibrator_name: "random_search"`; other active calibrators fail the run.

### A+B: known-defect recovery experiments

The A+B controlled experiments start from the seven pre-specified Mask Adoption defects (D1–D7). The following runner prepares each frozen injected starting point, runs the matched structured-reflection (SRR) baseline with BO+TuRBO preserved, and writes an auditable artifact gate and progress record:

```bash
conda activate SOCIA && \
export PROJECT_ROOT="$(pwd)" && \
export DATA_PATH="data_fitting/mask_adoption_data/" && \
python scripts/run_srr_defect_suite.py \
  --output output/ab_srr_defect_suite \
  --defects D1 D2 D3 D4 D5 D6 D7 \
  --iterations 3
```

For a SOCIA ACE recovery run on one prepared defect, first prepare the suite, then supply the generated task specification and corrupted simulator snapshot to `main.py`:

```bash
python scripts/prepare_srr_defect_suite.py --destination output/ab_socia_defect_suite

python main.py \
  --task "Develop a multi-agent simulation system that models the spread of mask-wearing behavior through social networks." \
  --task-file examples/mask_adoption_task.json \
  --output output/ab_socia_defect_suite/injected_error_D1 \
  --mode ace --auto --selfloop 3 --iterations 3 \
  --persisted-data-analysis-file output/ab_socia_defect_suite/injected_error_D1/task_spec_iter_0.json \
  --persisted-code-file output/ab_socia_defect_suite/injected_error_D1/simulation_code_using_calibration_template_SBI_BO_TuRBO_EVO_error_injection.py
```

The checked-in `output/experiment_A_gsim_BO_recalibration/` and `output/experiment_socia_selected_logit_bo_recalibration/` manifests record the fixed snapshots and BO+TuRBO recalibration artifacts used in the reported matched comparison.

To regenerate the matched recalibration artifacts from the saved snapshots, run:

```bash
python scripts/prepare_gsim_bo_recalibration.py
python scripts/run_gsim_bo_recalibration.py --workers 4
python scripts/prepare_socia_selected_bo_recalibration.py
python scripts/run_socia_selected_bo_recalibration.py --workers 4
```

### Predicted Counterfactual Consistency Probes

This experiment evaluates frozen simulator snapshots under pre-registered intervention, social, persistence, and risk probes. The evaluator does not invoke SOCIA orchestration, calibration, LLM diagnosis, code generation, repair, or selection.

```bash
conda activate SOCIA && \
python scripts/fixed_snapshot_counterfactual_eval.py \
  --manifest experiments/fixed_snapshot_counterfactual/manifest.json \
  --output output/fixed_snapshot_counterfactual_confirmatory
```

To reproduce only the summary from an existing output directory, add `--summarize-only`. The exploratory layer-weight operability probes are intentionally separate from the confirmatory score:

```bash
python scripts/fixed_snapshot_layer_weight_probes.py \
  --manifest experiments/fixed_snapshot_counterfactual/manifest.json \
  --probe-manifest experiments/fixed_snapshot_counterfactual/layer_weight_exploratory_manifest.json \
  --output output/fixed_snapshot_layer_weight_exploratory
```

## Workflow agents

The paper's ACE workflow uses the following roles:

- **Task understanding / blueprinting** translates the task and available data into a structured simulator specification.
- **Data analysis** derives data semantics, empirical targets, and diagnostic evidence.
- **Code generation** produces the initial simulator and applies bounded structural repairs.
- **Simulation execution** runs the generated simulator and collects structured artifacts.
- **Feedback generation (Evidence-to-Text)** maps residuals and code context to explicit mechanism-level diagnoses and repair strategies.
- **Iteration control** accepts only improving candidates and manages the outer calibration/repair loop.
- **Playbook manager** persists and retrieves prior diagnosis-and-repair knowledge across ACE iterations.

Calibration is performed by the simulator code using the configured fixed-structure calibration routine (for example BO+TuRBO or SBI); it is not a separate LLM agent.

## Outputs and logs

For a main run with `--output output/ace_mask_adoption`, the key artifacts are:

```text
output/ace_mask_adoption/
├── socia.log
├── task_spec_iter_0.json
├── simulation_code_iter_<n>.py
├── simulation_results_iter_<n>.json
├── feedback_iter_<n>.json
├── verification_results_iter_<n>.json
└── output_iter_<n>/
    ├── results.json
    └── calibrated_parameters.json
```

The precise artifacts vary when a generated simulator changes its internal calibration backend. ACE playbook snapshots are stored under `playbook_storage/`.

## Project structure

```text
SOCIA/
├── main.py                         # primary ACE/experimental workflow entry point
├── agents/                         # workflow-agent implementations and mode variants
├── core/                           # blueprint, simulation, and playbook state
├── orchestration/                  # dependency-injection container and workflow support
├── templates/                      # prompts used by generation and diagnosis roles
├── examples/                       # task specifications
├── data_fitting/                   # benchmark data and calibration inputs
├── scripts/                        # A+B and fixed-snapshot experiment runners
├── experiments/                    # immutable probe manifests
├── output/                         # experiment artifacts and reported snapshots
├── playbook_storage/               # ACE playbook state and snapshots
└── requirements.txt
```

## Reproduction notes

`README.md` commands are intended to be run from the repository root. The data paths and frozen artifact paths in the supplementary commands are repository-relative. The fixed-snapshot manifests record the exact code and parameter snapshots, selected iterations, seeds, and prohibitions used by the counterfactual evaluation.
