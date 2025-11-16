#!/usr/bin/env python3
"""
使用反馈生成新代码的完整流程
1. 使用prompt调用LLM生成反馈
2. 使用反馈生成新代码
"""

import argparse
import json
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 设置API key
try:
    from keys import OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print("✅ 已从keys.py加载API key")
except ImportError:
    print("⚠️  无法从keys.py导入API key，将使用环境变量")

from agents.feedback_generation.agent import FeedbackGenerationAgent
from agents.code_generation_calibrasim.agent import CodeGenerationCalibrasimAgent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate code from feedback for CalibraSim tasks.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="mask",
        choices=["mask", "supply"],
        help="Select which project scenario to run (default: mask).",
    )
    return parser.parse_args()


def _scenario_paths(scenario: str) -> dict:
    base = os.path.join(project_root, "output")
    if scenario == "mask":
        scenario_root = os.path.join(base, "test_mask_patch")
        return {
            "scenario_root": scenario_root,
            "alpha_results_path": os.path.join(scenario_root, "alpha_feedback_inputs.json"),
            "code_path": os.path.join(scenario_root, "simulation_alpha.py"),
            "task_spec_path": os.path.join(scenario_root, "task_spec_iter_0.json"),
            "model_plan_path": os.path.join(scenario_root, "model_plan_iter_0.json"),
            "data_analysis_path": os.path.join(scenario_root, "data_analysis_iter_0.json"),
            "new_code_path": os.path.join(scenario_root, "simulation_alpha_improved.py"),
            "result_path": os.path.join(scenario_root, "codegen_result.json"),
            "feedback_output_dir": scenario_root,
            "data_path": "data_fitting/mask_adoption_data",
            "llm_response_path": os.path.join(scenario_root, "llm_raw_response.txt"),
            "prompt_desc": "CalibraSim Alpha (Mask Adoption)",
        }

    if scenario == "supply":
        scenario_root = os.path.join(base, "supply")
        return {
            "scenario_root": scenario_root,
            "alpha_results_path": os.path.join(scenario_root, "alpha_feedback_inputs_supply.json"),
            "code_path": os.path.join(scenario_root, "simulation_code_iter_0_alpha.py"),
            "task_spec_path": os.path.join(scenario_root, "task_spec_supply.json"),
            "model_plan_path": os.path.join(scenario_root, "model_plan_supply.json"),
            "data_analysis_path": os.path.join(scenario_root, "data_analysis_supply.json"),
            "new_code_path": os.path.join(scenario_root, "simulation_code_iter_0_alpha_improved.py"),
            "result_path": os.path.join(scenario_root, "codegen_result_supply.json"),
            "feedback_output_dir": scenario_root,
            "data_path": "data_fitting/supply_data",
            "llm_response_path": os.path.join(scenario_root, "llm_raw_response_supply.txt"),
            "prompt_desc": "CalibraSim Alpha (Supply)",
        }

    raise ValueError(f"Unsupported scenario '{scenario}'")


def main():
    args = parse_args()
    paths = _scenario_paths(args.scenario)
    
    print("=" * 70)
    print("🚀 完整流程：使用prompt生成反馈，然后生成新代码")
    print(f"🗂️  场景: {args.scenario} -> {paths['prompt_desc']}")
    print("=" * 70)
    
    # 1. 读取数据
    alpha_results_path = paths["alpha_results_path"]
    code_path = paths["code_path"]
    task_spec_path = paths["task_spec_path"]
    
    print("\n📋 步骤1: 加载数据...")
    with open(alpha_results_path, 'r') as f:
        alpha_inputs = json.load(f)
    with open(code_path, 'r') as f:
        current_code = f.read()
    with open(task_spec_path, 'r') as f:
        task_spec = json.load(f)
    
    # 2. 准备evaluation_results（包含Alpha参数）
    evaluation_results = alpha_inputs['evaluation_results']
    evaluation_results['alpha_embedding'] = {
        'selected_params': alpha_inputs.get('selected_params', []),
        'alpha_stats': alpha_inputs.get('alpha_stats', {}),
        'alpha_norm_stats': alpha_inputs.get('alpha_norm_stats', {}),
        'r_stats': alpha_inputs.get('r_stats', {})
    }
    
    # 3. 生成反馈
    print("\n📋 步骤2: 生成反馈...")
    feedback_config = {
        "prompt_template": "templates/feedback_generation_prompt_calibrasim_alpha.txt",
        "output_format": "json"
    }
    
    feedback_agent = FeedbackGenerationAgent(
        config=feedback_config,
        output_path=paths["feedback_output_dir"]
    )
    
    try:
        feedback = feedback_agent.process(
            task_spec=task_spec,
            evaluation_results=evaluation_results,
            current_code=current_code,
            previous_code=None,
            iteration=0
        )
        
        # 保存反馈
        feedback_path = os.path.join(paths["feedback_output_dir"], "feedback_for_codegen.json")
        with open(feedback_path, 'w') as f:
            json.dump(feedback, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 反馈已生成: {feedback_path}")
        print(f"   Summary: {feedback.get('summary', 'N/A')[:100]}...")
        print(f"   Critical issues: {len(feedback.get('critical_issues', []))}")
        print(f"   Code improvements: {len(feedback.get('code_improvements', []))}")
        
    except Exception as e:
        print(f"❌ 生成反馈失败: {e}")
        print("   使用placeholder反馈继续...")
        feedback = {
            "summary": "Placeholder feedback - LLM call failed",
            "critical_issues": [],
            "code_improvements": [],
            "model_improvements": [],
            "data_alignment_suggestions": []
        }
    
    # 4. 使用反馈生成新代码
    print("\n📋 步骤3: 使用反馈生成新代码...")
    print("  这可能需要几分钟时间，请耐心等待...")
    import time
    import sys
    from datetime import datetime
    
    # 读取model_plan（如果有）
    model_plan_path = paths["model_plan_path"]
    model_plan = None
    if os.path.exists(model_plan_path):
        with open(model_plan_path, 'r') as f:
            model_plan = json.load(f)
        print("  ✅ 已加载model_plan")
    
    # 读取data_analysis（如果有）
    data_analysis_path = paths["data_analysis_path"]
    data_analysis = None
    if os.path.exists(data_analysis_path):
        with open(data_analysis_path, 'r') as f:
            data_analysis = json.load(f)
        print("  ✅ 已加载data_analysis")
    
    codegen_config = {
        "prompt_template": "templates/Calibrasim_code_generation_prompt.txt",
        "output_format": "python"
    }
    
    print("  🔧 初始化代码生成Agent...")
    codegen_agent = CodeGenerationCalibrasimAgent(config=codegen_config)
    print("  ✅ Agent初始化完成")
    
    try:
        # 准备previous_code
        previous_code = {"code": current_code} if current_code else None
        print(f"  📝 准备previous_code: {len(current_code):,} 字符, {len(current_code.splitlines())} 行")
        
        # 显示进度
        print("\n  ⏳ 开始生成代码...")
        print("     - Prompt长度: ~44,000 tokens")
        print("     - 预期输出: ~25,000 tokens (2397行)")
        print("     - 这可能需要 3-10 分钟，请耐心等待...")
        print("     - 开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        sys.stdout.flush()
        
        start_time = time.time()
        
        # 生成新代码（添加进度显示）
        def progress_callback(step, message):
            elapsed = time.time() - start_time
            print(f"  📊 [{int(elapsed)}s] {step}: {message}")
            sys.stdout.flush()
        
        # 由于process方法不直接支持回调，我们需要在调用前后显示进度
        print("  🔄 步骤 1/3: 构建prompt...")
        sys.stdout.flush()
        
        # 手动构建prompt以显示进度
        prompt = codegen_agent._build_prompt(
            task_spec=task_spec,
            model_plan=model_plan,
            data_analysis=data_analysis,
            feedback=feedback,
            data_path=paths["data_path"],
            previous_code=previous_code,
            mode="full"
        )
        print(f"  ✅ Prompt已构建: {len(prompt):,} 字符 (~{len(prompt)//4:,} tokens)")
        sys.stdout.flush()
        
        print("  🔄 步骤 2/3: 调用LLM生成代码...")
        print("     (这可能需要几分钟，LLM正在处理大量代码...)")
        print("     ⏳ 正在等待LLM响应...", end="", flush=True)
        sys.stdout.flush()
        
        # 使用线程显示进度动画
        import threading
        stop_animation = threading.Event()
        
        def show_progress():
            """显示进度动画"""
            dots = 0
            while not stop_animation.is_set():
                print(".", end="", flush=True)
                dots += 1
                if dots >= 3:
                    print("\b\b\b   \b\b\b", end="", flush=True)
                    dots = 0
                time.sleep(1)
        
        # 启动进度动画线程
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        try:
            # 调用LLM（这里会花费大部分时间）
            llm_start_time = time.time()
            llm_response = codegen_agent._call_llm(prompt, reasoning={"effort": "high"})
            llm_elapsed = time.time() - llm_start_time
        finally:
            # 停止进度动画
            stop_animation.set()
            progress_thread.join(timeout=1)
            print()  # 换行
        
        # 检查响应
        if llm_response.startswith("Error:"):
            error_msg = llm_response
            # 检查是否是Connection error或其他可恢复的错误
            if "Connection error" in error_msg or "model" in error_msg.lower():
                print(f"  ⚠️  LLM调用失败: {error_msg}")
                print(f"  💡 这通常意味着GPT-5不可用，已自动fallback到gpt-4o")
                # 不抛出异常，让fallback继续
                # 但如果没有fallback，仍然需要抛出
                if "fallback" not in error_msg.lower():
                    raise Exception(f"LLM调用失败且无fallback: {error_msg}")
            else:
                print(f"  ❌ LLM调用失败: {error_msg}")
                raise Exception(f"LLM调用失败: {error_msg}")
        
        # 保存LLM原始响应用于分析
        llm_response_path = paths["llm_response_path"]
        with open(llm_response_path, 'w') as f:
            f.write(llm_response)
        print(f"  💾 LLM原始响应已保存: {llm_response_path}")
        
        print(f"  ✅ LLM响应已收到: {len(llm_response):,} 字符 (~{len(llm_response)//4:,} tokens)")
        print(f"     LLM调用耗时: {int(llm_elapsed)} 秒 ({llm_elapsed/60:.1f} 分钟)")
        
        # 检查响应长度
        expected_tokens = len(current_code) // 4  # 原始代码的token数
        actual_tokens = len(llm_response) // 4
        if actual_tokens < expected_tokens * 0.1:
            print(f"  ⚠️  警告: LLM响应太短!")
            print(f"     预期: ~{expected_tokens:,} tokens (2397行)")
            print(f"     实际: ~{actual_tokens:,} tokens")
            print(f"     比例: {actual_tokens/expected_tokens*100:.1f}%")
            print(f"  💡 LLM可能没有按照要求输出完整代码")
            print(f"  💡 建议: 检查prompt或考虑禁用self-checking循环")
        
        sys.stdout.flush()
        
        print("  🔄 步骤 3/4: 提取和处理代码...")
        sys.stdout.flush()
        
        # 提取代码
        code = codegen_agent._extract_code(llm_response)
        code = codegen_agent._strip_markdown_fences(code)
        
        print(f"  ✅ 代码已提取: {len(code):,} 字符, {len(code.splitlines())} 行")
        sys.stdout.flush()
        
        # 运行self-checking循环（只在代码足够长时运行）
        code_lines = len(code.splitlines())
        original_lines = len(current_code.splitlines())
        
        # 如果生成的代码远少于原始代码，跳过self-checking（避免基于不完整代码改进）
        if code_lines < original_lines * 0.5:
            print(f"\n  ⚠️  代码太短 ({code_lines} 行 vs {original_lines} 行)，跳过Self-checking循环")
            print(f"     原因: 生成的代码只有原始代码的 {code_lines/original_lines*100:.1f}%")
            print(f"     建议: LLM可能没有按照要求输出完整代码")
            print(f"     操作: 直接使用LLM的第一次响应，不进行self-checking改进")
            code_summary = codegen_agent._generate_code_summary(code)
            result = {
                "code": code,
                "code_summary": code_summary,
                "metadata": {
                    "model_type": "full",
                    "entities": [],
                    "behaviors": [],
                    "mode": "full"
                }
            }
        elif code_lines > 100:
            print(f"\n  🔄 步骤 4/4: 运行Self-checking循环 (selfloop=3)...")
            print(f"     当前代码: {code_lines} 行 (原始: {original_lines} 行)")
            print("     (这可能需要额外的时间，正在进行代码质量检查...)")
            sys.stdout.flush()
            
            # 显示self-checking进度
            sc_start_time = time.time()
            
            # 修复参数：_run_self_checking_loop需要model_plan参数
            final_code = codegen_agent._run_self_checking_loop(
                code=code,
                task_spec=task_spec,
                model_plan=model_plan or {},
                feedback=feedback,
                historical_fix_log=None,
                max_attempts=3
            )
            
            sc_elapsed = time.time() - sc_start_time
            
            # 生成代码摘要
            code_summary = codegen_agent._generate_code_summary(final_code)
            
            result = {
                "code": final_code,
                "code_summary": code_summary,
                "metadata": {
                    "model_type": "full",
                    "entities": [],
                    "behaviors": [],
                    "mode": "full"
                }
            }
            print(f"  ✅ Self-checking循环完成 (耗时: {int(sc_elapsed)} 秒)")
            print(f"     最终代码: {len(final_code.splitlines())} 行")
        else:
            print(f"\n  ⚠️  代码太短 ({code_lines} 行)，跳过Self-checking循环")
            code_summary = codegen_agent._generate_code_summary(code)
            result = {
                "code": code,
                "code_summary": code_summary,
                "metadata": {
                    "model_type": "full",
                    "entities": [],
                    "behaviors": [],
                    "mode": "full"
                }
            }
        
        total_elapsed = time.time() - start_time
        print(f"\n  ✅ 代码生成完成!")
        print(f"     总耗时: {int(total_elapsed)} 秒 ({total_elapsed/60:.1f} 分钟)")
        print(f"     最终代码: {len(result.get('code', ''))} 字符, {len(result.get('code', '').splitlines())} 行")
        sys.stdout.flush()
        
        # 保存新代码
        new_code_path = paths["new_code_path"]
        with open(new_code_path, 'w') as f:
            f.write(result.get('code', ''))
        
        print(f"\n✅ 新代码已保存: {new_code_path}")
        print(f"   代码长度: {len(result.get('code', ''))} 字符, {len(result.get('code', '').splitlines())} 行")
        print(f"   代码摘要: {result.get('code_summary', 'N/A')[:100]}...")
        
        # 保存完整结果
        result_path = paths["result_path"]
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"   完整结果: {result_path}")
        
    except Exception as e:
        print(f"\n❌ 生成代码失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 流程完成")
    print("=" * 70)

if __name__ == "__main__":
    main()

