#!/bin/bash
# 统一运行所有校准器的测试，使用test_data.csv作为测试窗口

cd "$(dirname "$0")"

echo "=========================================="
echo "运行所有校准器的统一测试"
echo "=========================================="
echo "测试窗口: test_data.csv (days 30-39)"
echo "k_runs: 20 (统一配置)"
echo "=========================================="
echo ""

# 运行Python脚本
python3 run_all_calibrators_test_unified.py

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "运行以下命令来比较结果："
echo "  python3 compare_specific_calibrators.py"

