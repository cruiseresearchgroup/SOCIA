# CalibraSim Experiment Records

| 版本 | 评估窗口 | 方法 | 入口代码 | 关键依赖 | 主要输出/Result |
| --- | --- | --- | --- | --- | --- |
| 原始（simulation_alpha.py） | 第30–39天（测试窗） | 双重蒙特卡洛（M=50, K=20） | `output/test_mask_patch/simulation_alpha.py` (default mode) | `data_fitting/mask_adoption_data/*.csv/json`、`test_run/io/*` | `test_run/results/{validation_metrics.json,daily_rates.csv,observed_vs_predicted.png}` |
| 改进（simulation_alpha_improved.py） | 第30–39天（测试窗） | 双重蒙特卡洛（M=50, K=20） | `output/test_mask_patch/simulation_alpha_improved.py` | 同上 + `output/test_mask_patch/alpha_feedback_inputs.json` | `test_run/results/double_mc_test_metrics.json`、`test_run/results/daily_rates_double_mc.csv`、`alpha_results.json` |
| 改进（simulation_Noalpha_improved.py） | 第30–39天（测试窗） | 双重蒙特卡洛（M=50, K=20） | `output/test_mask_patch/noalpha_code/simulation_Noalpha_improved.py` | `noalpha_code/run_out/*.json`、同一数据集 | `noalpha_logs/`、`test_run/results/*` (No-alpha) |
| 统一测试 BoCalibrator_TuRBO | 第30–39天（测试窗） | BoCalibrator_TuRBO（k_runs=20） | `output/test_mask_patch/run_all_calibrators_test_unified.py`（选择 Bo） | `simulation_code_using_calibration_template_BO_SBI.py`、`data_fitting/mask_adoption_data/outputs_BoCalibrator/*` | `outputs_BoCalibrator/*.json`、`test_run/results/test_metrics.json` |
| 统一测试 SBI | 第30–39天（测试窗） | SBI（M=50, K=20） | `run_all_calibrators_test_unified.py`（选择 SBI） | `simulation_code_using_calibration_template_SBI.py`、`outputs_SBICalibrator_K1/*`、`sbi-logs/` | `test_run/results/test_metrics.json`（SBI）、`sbi-logs/NPE_C/*` |
| G-Sim Baseline | 第0–11天（训练窗） | G-Sim（M=50, K=20, bank=1000） | `output/test_mask_patch/simulation_alpha.py --mode gsim` | `simulation_alpha.py` G-Sim分支、`outputs_SBICalibrator_K1/prior_info.json` | `data_fitting/mask_adoption_data/gsim_baseline/{simulation_bank.npz,metrics.json,posterior_curves.npz}` |

> 以上信息同步自 `output/test_mask_patch/experiments_log.csv`，便于快速定位各实验的代码、依赖与结果文件。
