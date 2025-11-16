#!/bin/bash
# 使用 CalibraSim Alpha prompt 生成代码的便捷脚本

set -e

PROJECT_ROOT="/Users/miaoji.norman/Desktop/SOCIA"
ENV_PATH="${PROJECT_ROOT}/socia/bin/activate"
PYTHON_BIN="python3"
GEN_SCRIPT="${PROJECT_ROOT}/output/test_mask_patch/generate_code_from_feedback.py"
LOG_DIR="${PROJECT_ROOT}/output/supply"
PROMPT_DESC="CalibraSim Alpha (Supply)"

mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "🚀 运行 CalibraSim Alpha 代码生成流水线"
echo "======================================================================"
echo ""

# 1. 激活环境
echo "【1】激活虚拟环境..."
if [ ! -f "$ENV_PATH" ]; then
    echo "  ❌ 找不到虚拟环境: $ENV_PATH"
    exit 1
fi
source "$ENV_PATH"
echo "  ✅ 已激活虚拟环境"

# 2. 检查依赖
echo ""
echo "【2】检查依赖..."
if ! $PYTHON_BIN -c "import openai" 2>/dev/null; then
    echo "  ⚠️  openai 包未安装，正在安装..."
    pip install openai
else
    echo "  ✅ openai 包已安装"
fi

# 3. 检查 API Key
echo ""
echo "【3】检查 API Key..."
if $PYTHON_BIN -c "from keys import OPENAI_API_KEY; print('✅ API Key 已配置')" 2>/dev/null; then
    echo "  ✅ API Key 已配置"
else
    echo "  ⚠️  API Key 未配置，请检查 keys.py 或环境变量"
    exit 1
fi

# 4. 运行代码生成
echo ""
echo "【4】运行代码生成 (${PROMPT_DESC})..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH="${LOG_DIR}/codegen_calibrasim_alpha_supply_${TIMESTAMP}.log"
$PYTHON_BIN "$GEN_SCRIPT" --scenario supply 2>&1 | tee "$LOG_PATH"

# 5. 检查结果
echo ""
echo "【5】检查生成结果..."
OUTPUT_CODE="${PROJECT_ROOT}/output/supply/simulation_code_iter_0_alpha_improved.py"
if [ -f "$OUTPUT_CODE" ]; then
    LINES=$(wc -l < "$OUTPUT_CODE")
    echo "  ✅ 代码已生成: $LINES 行"
    if [ $LINES -lt 1500 ]; then
        echo "  ⚠️  代码行数偏少，请确认是否生成完整"
    else
        echo "  ✅ 代码长度看起来正常"
    fi
else
    echo "  ❌ 未生成预期的 simulation_code_iter_0_alpha_improved.py"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ CalibraSim Alpha (Supply) 代码生成完成"
echo "日志: $LOG_PATH"
echo "======================================================================"
