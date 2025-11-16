# 7种Calibrator性能比较 - MSE和RMSE

## 性能指标表格

| 排名 | Calibrator | RMSE (均值 ± 95% CI) | MSE | 
|------|-----------|---------------------|-----|
| 🥇 1 | **BoCalibrator_TuRBO** | 0.073653 ± 0.006821 | 0.005425 |
| 🥈 2 | **SBI** | 0.109406 ± 0.010586 | 0.011970 |
| 🥉 3 | **LogitHead** | 0.210654 ± 0.013690 | 0.044375 |
| 4 | BoCalibrator_Vanilla | 0.244030 ± 0.007551 | 0.059551 |
| 5 | RandomSearch | 0.265170 ± 0.019527 | 0.070315 |
| 6 | BoCalibrator_TuRBO_LLM_Guide | 0.323205 ± 0.006747 | 0.104461 |
| 7 | EvoCalibrator_GA | 0.372266 ± 0.004093 | 0.138582 |

## 详细说明

- **测试窗口**: test_data.csv (days 30-39, 10天)
- **k_runs**: 20 (所有方法统一)
- **数据集**: data_fitting/mask_adoption_data
- **MSE计算**: MSE = RMSE²

## 性能差距分析

相对于最佳方法 (BoCalibrator_TuRBO):

| Calibrator | RMSE差距 | MSE差距 |
|-----------|---------|--------|
| SBI | +48.5% | +120.6% |
| LogitHead | +186.0% | +718.0% |
| BoCalibrator_Vanilla | +231.3% | +997.7% |
| RandomSearch | +260.0% | +1196.3% |
| BoCalibrator_TuRBO_LLM_Guide | +338.8% | +1826.5% |
| EvoCalibrator_GA | +405.4% | +2454.7% |

