#!/bin/bash
# 运行 SUPPLY 任务的 OOD pipeline，使用 SBI 方法
# 使用方法: source odd/bin/activate && bash output/supply/run_ood_supply_sbi.sh

cd /Users/miaoji.norman/Desktop/SOCIA

# 激活 odd 环境
source odd/bin/activate

# 设置环境变量
export DATA_PATH="data_fitting/supply_data"
export PROJECT_ROOT="/Users/miaoji.norman/Desktop/SOCIA"

echo "=========================================="
echo "运行 SUPPLY 任务的 OOD Pipeline (使用 SBI)"
echo "=========================================="
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "数据路径: $DATA_PATH"
echo "输出目录: output/supply"
echo ""

# 验证数据文件是否存在
if [ ! -f "$PROJECT_ROOT/$DATA_PATH/train_data.csv" ]; then
    echo "错误: 找不到数据文件 $PROJECT_ROOT/$DATA_PATH/train_data.csv"
    exit 1
fi

echo "数据文件验证通过"
echo ""

# 运行 OOD pipeline
python test_data_analysis.py \
  --task "Develop a supply chain simulation system that models a single-stage Beer Game supply chain environment. The simulator should automatically generate and calibrate parameters using simulation-based inference (SBI) or gradient-free optimization methods to optimize parameters on observed training state-action trajectories." \
  --task-file examples/supply_task.json \
  --output output/supply \
  --mode odd \
  --selfloop 3

echo ""
echo "=========================================="
echo "OOD Pipeline 运行完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  - output/supply/task_spec_iter_0.json: 任务规范"
echo "  - output/supply/generated_code_iter_0.json: 生成的代码（JSON格式）"
echo "  - output/supply/simulation_code_iter_0.py: 生成的模拟代码（Python格式）"
echo "  - output/supply/test_data_analysis_odd.log: 运行日志"
echo ""
echo "下一步：运行生成的模拟代码查看结果"
echo "  bash output/supply/run_simulation_sbi.sh"



