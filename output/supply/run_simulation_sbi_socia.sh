#!/bin/bash
# 运行使用 SBI 方法生成的 SUPPLY 模拟代码（使用 socia 环境）
# 使用方法: source socia/bin/activate && bash output/supply/run_simulation_sbi_socia.sh

cd /Users/miaoji.norman/Desktop/SOCIA

# 激活 socia 环境
source socia/bin/activate

# 设置环境变量
export DATA_PATH="data_fitting/supply_data"
export PROJECT_ROOT="/Users/miaoji.norman/Desktop/SOCIA"

echo "=========================================="
echo "运行 SUPPLY 模拟代码 (使用 SBI，socia 环境)"
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

# 检查 torch 和 sbi 是否可用
echo "检查依赖库..."
python -c "import torch; print('✓ torch:', torch.__version__)" 2>&1 || echo "✗ torch: 未安装"
python -c "import sbi; print('✓ sbi:', sbi.__version__)" 2>&1 || echo "✗ sbi: 未安装"
echo ""

# 运行模拟代码（使用 SBI 方法，启用双重蒙特卡洛）
# --opt_method sbi: 使用 SBI 方法
# --double-mc: 启用双重蒙特卡洛（与口罩任务配置一致）
# --mc-M 50: 参数样本数量（与口罩任务一致）
# --mc-K 20: 每个参数样本的运行次数（与口罩任务一致）
# --num_simulations: SBI 训练的模拟数量（默认：1000）
# --num_samples_posterior: 后验采样数量（默认：10000）
# --sampling_timeout: 采样超时时间（默认：60秒）
echo "开始运行 SUPPLY 模拟（SBI + 双重蒙特卡洛）..."
echo ""

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
echo "结果文件保存在: data_fitting/supply_data/results/"
echo ""
echo "查看结果："
echo "  - optimized_params.json: 校准后的参数配置"
echo "  - metrics.json: 验证集指标（包含Wasserstein、MMD、MSE等）"
echo "  - metrics_test.json: 测试集指标"
echo "  - posterior_samples.npy: 后验采样（如果使用SBI）"
echo ""
echo "查看指标："
echo "  python -c \"import json; data=json.load(open('data_fitting/supply_data/results/metrics.json')); print(json.dumps(data['aggregate'], indent=2))\""



