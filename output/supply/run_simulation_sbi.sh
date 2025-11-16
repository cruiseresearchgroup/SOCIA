#!/bin/bash
# 运行使用 SBI 方法生成的 SUPPLY 模拟代码
# 使用方法: source odd/bin/activate && bash output/supply/run_simulation_sbi.sh

cd /Users/miaoji.norman/Desktop/SOCIA

# 激活 odd 环境
source odd/bin/activate

# 设置环境变量
export DATA_PATH="data_fitting/supply_data"
export PROJECT_ROOT="/Users/miaoji.norman/Desktop/SOCIA"

echo "=========================================="
echo "运行 SUPPLY 模拟代码 (使用 SBI)"
echo "=========================================="
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATH"
echo ""

# 验证生成的代码是否存在
if [ ! -f "$PROJECT_ROOT/output/supply/simulation_code_iter_0.py" ]; then
    echo "错误: 找不到生成的代码文件 output/supply/simulation_code_iter_0.py"
    echo "请先运行: bash output/supply/run_ood_supply_sbi.sh"
    exit 1
fi

# 验证数据文件是否存在
if [ ! -f "$PROJECT_ROOT/$DATA_PATH/train_data.csv" ]; then
    echo "错误: 找不到数据文件 $PROJECT_ROOT/$DATA_PATH/train_data.csv"
    exit 1
fi

echo "代码文件和数据文件验证通过"
echo ""

# 运行模拟代码（使用 SBI 方法，启用双重蒙特卡洛）
# --opt_method sbi: 使用 SBI 方法
# --double-mc: 启用双重蒙特卡洛（与口罩任务配置一致）
# --mc-M 50: 参数样本数量（与口罩任务一致）
# --mc-K 20: 每个参数样本的运行次数（与口罩任务一致）
# --num_simulations: SBI 训练的模拟数量（默认：1000）
# --num_samples_posterior: 后验采样数量（默认：10000）
# --sampling_timeout: 采样超时时间（默认：60秒）
python output/supply/simulation_code_iter_0.py \
  --opt_method sbi \
  --double-mc \
  --mc-M 50 \
  --mc-K 20 \
  --num_simulations 1000 \
  --num_samples_posterior 10000 \
  --sampling_timeout 120 \
  --n_samples_wass_mmd 200 \
  --mmd_sigma 1.0 \
  --demand_family Poisson

echo ""
echo "=========================================="
echo "模拟运行完成！"
echo "=========================================="
echo ""
echo "结果文件保存在: data_fitting/supply_data/"
echo ""
echo "查看结果："
echo "  - calibrated_config.json: 校准后的参数配置"
echo "  - validation_report.json: 验证报告（包含Wasserstein、MMD、MSE等指标）"
echo "  - per_trajectory_metrics.csv: 每个轨迹的指标"
echo ""
echo "查看指标："
echo "  python -c \"import json; data=json.load(open('data_fitting/supply_data/validation_report.json')); print(json.dumps(data, indent=2))\""

