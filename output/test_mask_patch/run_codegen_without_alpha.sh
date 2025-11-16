#!/bin/bash
# 运行代码生成的便捷脚本

set -e

echo "======================================================================"
echo "🚀 运行代码生成"
echo "======================================================================"
echo ""

# 1. 激活环境
echo "【1】激活虚拟环境..."
cd /Users/miaoji.norman/Desktop/SOCIA
source socia/bin/activate

# 2. 检查依赖
echo ""
echo "【2】检查依赖..."
if ! python -c "import openai" 2>/dev/null; then
    echo "  ⚠️  openai包未安装，正在安装..."
    pip install openai
else
    echo "  ✅ openai包已安装"
fi

# 3. 检查API Key
echo ""
echo "【3】检查API Key..."
if python -c "from keys import OPENAI_API_KEY; print('✅ API Key已配置')" 2>/dev/null; then
    echo "  ✅ API Key已配置"
else
    echo "  ⚠️  API Key未配置，请检查keys.py文件"
    exit 1
fi

# 4. 运行代码生成
echo ""
echo "【4】运行代码生成（直接重写模式）..."
python - <<'PY' 2>&1 | tee output/test_mask_patch/codegen_run_directrewrite_$(date +%Y%m%d_%H%M%S).log
import os, sys, json, time, threading
project_root = "/Users/miaoji.norman/Desktop/SOCIA"
sys.path.insert(0, project_root)
try:
    from keys import OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("✅ 已从keys.py加载API key")
except Exception:
    print("⚠️ 未能从keys.py加载API key，将依赖环境变量 OPENAI_API_KEY")

from agents.feedback_generation.agent import FeedbackGenerationAgent
from agents.code_generation_calibrasim.agent import CodeGenerationCalibrasimAgent

# 路径
code_path = os.path.join(project_root, "output/test_mask_patch/simulation_alpha.py")
task_spec_path = os.path.join(project_root, "output/test_mask_patch/task_spec_iter_0.json")
base_out = os.path.join(project_root, "output/test_mask_patch")
code_out_dir = os.path.join(base_out, "noalpha_code")
logs_out_dir = os.path.join(base_out, "noalpha_logs")
os.makedirs(code_out_dir, exist_ok=True)
os.makedirs(logs_out_dir, exist_ok=True)

print("="*70)
print("🚀 直接重写：生成反馈 → 生成新代码")
print("="*70)

print("\n📋 步骤1: 加载数据...")
with open(code_path, "r") as f:
    current_code = f.read()
if os.path.exists(task_spec_path):
    with open(task_spec_path, "r") as f:
        task_spec = json.load(f)
else:
    task_spec = {"goal": "Refactor and improve the simulation code."}
print("  ✅ 已加载current_code与task_spec")

evaluation_results = {}

print("\n📋 步骤2: 生成反馈...")
feedback_config = {
    "prompt_template": "templates/feedback_generation_prompt_calibrasim_Noalpha.txt",
    "output_format": "json",
}
feedback_agent = FeedbackGenerationAgent(config=feedback_config, output_path=code_out_dir)

try:
    feedback = feedback_agent.process(
        task_spec=task_spec,
        evaluation_results=evaluation_results,
        current_code=current_code,
        previous_code=None,
        iteration=0,
    )
except Exception as e:
    print(f"❌ 生成反馈失败: {e}")
    feedback = {"summary": "Fallback feedback", "critical_issues": [], "code_improvements": []}

with open(os.path.join(code_out_dir, "feedback_for_codegen_directrewrite.json"), "w") as f:
    json.dump(feedback, f, indent=2, ensure_ascii=False)
print("  ✅ 反馈已生成: output/test_mask_patch/noalpha_code/feedback_for_codegen_directrewrite.json")

print("\n📋 步骤3: 使用反馈生成新代码（Calibrasim 代码生成）...")
codegen_config = {
    "prompt_template": "templates/Calibrasim_code_generation_prompt.txt",
    "output_format": "python",
}
agent = CodeGenerationCalibrasimAgent(config=codegen_config)

start = time.time()
llm_text = agent._call_llm(
    agent._build_prompt(
        task_spec=task_spec,
        model_plan=None,
        data_analysis=None,
        feedback=feedback,
        data_path="data_fitting/mask_adoption_data",
        previous_code={"code": current_code},
        mode="full",
    ),
    reasoning={"effort": "high"},
)

code = agent._strip_markdown_fences(agent._extract_code(llm_text))

new_code_path = os.path.join(code_out_dir, "simulation_Noalpha_improved.py")
with open(new_code_path, "w") as f:
    f.write(code)
print(f"  ✅ 新代码已保存: {new_code_path}")
print(f"  ⏱️ 耗时: {int(time.time()-start)}s, 行数: {len(code.splitlines())}")

print("\n✅ 直接重写流程完成")
PY

# 5. 检查结果
echo ""
echo "【5】检查结果..."
if [ -f "output/test_mask_patch/noalpha_code/simulation_Noalpha_improved.py" ]; then
    LINES=$(wc -l < output/test_mask_patch/noalpha_code/simulation_Noalpha_improved.py)
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
