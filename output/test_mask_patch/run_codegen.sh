#!/bin/bash
# 运行代码生成的便捷脚本

set -e

echo "======================================================================"
echo "🚀 运行代码生成"
echo "======================================================================"
echo ""

# 1. 激活环境
echo "【1】激活虚拟环境..."
cd /Users/miaoji/Desktop/SOCIA-1
source sbi_env_arm64/bin/activate

# 2. 检查依赖
echo ""
echo "【2】检查依赖..."
if ! python3 -c "import openai" 2>/dev/null; then
    echo "  ⚠️  openai包未安装，正在安装..."
    pip install openai
else
    echo "  ✅ openai包已安装"
fi

# 3. 检查API Key
echo ""
echo "【3】检查API Key..."
if python3 -c "from keys import OPENAI_API_KEY; print('✅ API Key已配置')" 2>/dev/null; then
    echo "  ✅ API Key已配置"
else
    echo "  ⚠️  API Key未配置，请检查keys.py文件"
    exit 1
fi

# 4. 运行代码生成
echo ""
echo "【4】运行代码生成..."
python3 output/test_mask_patch/generate_code_from_feedback.py 2>&1 | tee output/test_mask_patch/codegen_run_$(date +%Y%m%d_%H%M%S).log

# 5. 检查结果
echo ""
echo "【5】检查结果..."
if [ -f "output/test_mask_patch/simulation_alpha_improved.py" ]; then
    LINES=$(wc -l < output/test_mask_patch/simulation_alpha_improved.py)
    echo "  ✅ 代码已生成: $LINES 行"
    
    if [ $LINES -lt 1000 ]; then
        echo "  ⚠️  代码太短，可能生成失败"
    elif [ $LINES -gt 2000 ]; then
        echo "  ✅ 代码长度正常"
    fi
else
    echo "  ❌ 代码生成失败"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ 完成"
echo "======================================================================"
