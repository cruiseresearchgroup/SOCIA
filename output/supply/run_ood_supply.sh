#!/bin/bash
# 运行 SUPPLY 任务的 OOD 管线
# 使用方法: source odd/bin/activate && bash output/supply/run_ood_supply.sh

cd /Users/miaoji.norman/Desktop/SOCIA

# 激活 odd 环境
source odd/bin/activate

# 运行 OOD 管线
python test_data_analysis.py \
  --task "Develop a supply chain simulation system that models a single-stage Beer Game supply chain environment with parameter calibration using simulation-based inference (SBI)" \
  --task-file examples/supply_task.json \
  --output output/supply \
  --mode odd \
  --selfloop 3



