#!/bin/bash
# 运行 SUPPLY 模拟代码并查看指标结果
# 使用方法: source odd/bin/activate && bash output/supply/run_simulation.sh

cd /Users/miaoji.norman/Desktop/SOCIA

# 激活 odd 环境
source odd/bin/activate

# 设置数据路径环境变量（相对于项目根目录）
export DATA_PATH="data_fitting/supply_data"
export PROJECT_ROOT="/Users/miaoji.norman/Desktop/SOCIA"

# 运行模拟代码
# 默认参数：
#   - max_candidates=60 (参数校准候选数量，可以减小以加快速度)
#   - policy_mode=action_replay (使用数据中的action)
#   - sample_size=200 (分布指标采样数量)
#   - wass_slices=128 (Wasserstein距离切片数量)
#   - bootstrap_B=200 (Bootstrap重采样数量)

echo "开始运行 SUPPLY 模拟代码..."
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATH"
echo "完整数据目录: $PROJECT_ROOT/$DATA_PATH"
echo ""

# 验证数据文件是否存在
if [ ! -f "$PROJECT_ROOT/$DATA_PATH/train_data.csv" ]; then
    echo "错误: 找不到数据文件 $PROJECT_ROOT/$DATA_PATH/train_data.csv"
    exit 1
fi

echo "数据文件验证通过"
echo ""

# 快速测试运行（减少候选数量以加快速度）
python output/supply/simulation_code_iter_0.py \
  --max_candidates 30 \
  --policy_mode action_replay \
  --sample_size 200 \
  --wass_slices 128 \
  --bootstrap_B 200 \
  --mmd_sigma 1.0

echo ""
echo "运行完成！结果已保存到: data_fitting/supply_data/"
echo ""
echo "查看结果文件："
echo "  - calibrated_config.json: 校准后的参数配置"
echo "  - validation_report.json: 验证报告（包含Wasserstein、MMD、MSE等指标）"
echo "  - per_trajectory_metrics.csv: 每个轨迹的指标"
echo "  - reconstructed_demand_train.csv: 训练集重建需求"
echo "  - reconstructed_demand_val.csv: 验证集重建需求"

