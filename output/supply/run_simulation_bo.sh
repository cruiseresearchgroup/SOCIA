#!/bin/bash
# 运行使用 Bayesian Optimization (BO) 方法生成的 SUPPLY 模拟代码
# 使用方法: source socia/bin/activate && bash output/supply/run_simulation_bo.sh

cd /Users/miaoji.norman/Desktop/SOCIA

# 激活 socia 环境
source socia/bin/activate

# 设置环境变量
export DATA_PATH="data_fitting/supply_data"
export PROJECT_ROOT="/Users/miaoji.norman/Desktop/SOCIA"

echo "=========================================="
echo "运行 SUPPLY 模拟代码 (使用 Bayesian Optimization)"
echo "=========================================="
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATH"
echo ""

# 验证生成的代码是否存在
if [ ! -f "$PROJECT_ROOT/output/supply/simulation_code_iter_0_bo.py" ]; then
    echo "错误: 找不到生成的代码文件 output/supply/simulation_code_iter_0_bo.py"
    exit 1
fi

# 验证数据文件是否存在
if [ ! -f "$PROJECT_ROOT/$DATA_PATH/train_data.csv" ]; then
    echo "错误: 找不到数据文件 $PROJECT_ROOT/$DATA_PATH/train_data.csv"
    exit 1
fi

echo "代码文件和数据文件验证通过"
echo ""

# 检查 scikit-optimize 是否可用
echo "检查依赖库..."
python -c "import skopt; print('✓ scikit-optimize:', skopt.__version__)" 2>&1 || echo "✗ scikit-optimize: 未安装（将使用梯度下降回退）"
echo ""

# 运行模拟代码（使用 Bayesian Optimization）
# --n_trials 1000: 评估次数（匹配SBI的num_simulations=1000）
# --n_initial_points 100: 初始随机点数量（10%的n_trials）
# --acquisition_function EI: 采集函数（期望改进）
# --demand_family Poisson: 需求分布类型
# --n_samples_wass_mmd 200: Wasserstein/MMD指标采样数量
# --mmd_sigma 1.0: MMD高斯核参数
echo "开始运行 SUPPLY 模拟（Bayesian Optimization）..."
echo "注意：BO优化需要较长时间（1000次评估），每次评估都需要运行完整模拟"
echo "进度会每10次评估显示一次"
echo ""

python output/supply/simulation_code_iter_0_bo.py \
  --n_trials 1000 \
  --n_initial_points 100 \
  --acquisition_function EI \
  --demand_family Poisson \
  --n_samples_wass_mmd 200 \
  --mmd_sigma 1.0

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
echo ""
echo "查看指标："
echo "  python -c \"import json; data=json.load(open('data_fitting/supply_data/results/metrics.json')); print(json.dumps(data['aggregate'], indent=2))\""

