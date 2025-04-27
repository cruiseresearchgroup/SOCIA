import os
import json
import logging
from typing import List, Dict, Any, Optional

from ..agents.base_agent import BaseAgent, Action
from ..utils.llm_utils import LLMClient

logger = logging.getLogger("SOCIA-CAMEL")

class CodeGenerationAgent(BaseAgent):
    """
    代码生成智能体，负责将模型规划转换为可执行的Python代码
    """
    def __init__(self, config_path: str = "../config.yaml"):
        # 定义可执行的动作
        actions = [
            Action(name="编写代码", description="根据模型规划编写Python代码"),
            Action(name="优化代码", description="优化生成的代码以提高效率和可读性"),
            Action(name="注释代码", description="为代码添加注释以提高可读性"),
            Action(name="思考", description="思考给定的问题"),
            Action(name="分析", description="分析代码的实现难点和解决方案")
        ]
        
        # 初始化基类
        super().__init__(
            name="代码生成智能体",
            role="代码生成专家",
            goal="将模型规划转换为可执行的Python代码",
            actions=actions,
            config_path=config_path
        )
        
        # 创建LLM客户端
        self.llm_client = LLMClient(config_path)
    
    def _generate_response(self, message: str) -> str:
        """
        生成响应
        
        Args:
            message: 用户消息
            
        Returns:
            智能体响应
        """
        # 构建提示词
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message}
        ]
        
        # 添加历史对话，最多取最近5条
        history = self.memory.get_conversations(5)
        for conv in history:
            messages.append({"role": conv["role"], "content": conv["content"]})
        
        # 使用LLM生成响应
        response = self.llm_client.generate_response(messages)
        
        return response
    
    def _action_编写代码(self, model_plan: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据模型规划编写Python代码
        
        Args:
            model_plan: 模型规划
            task_analysis: 任务分析结果
            
        Returns:
            生成的代码
        """
        # 构建提示词
        prompt = f"""你是一位专业的Python开发者，现在需要根据以下模型规划和任务分析，编写一个完整的社会模拟代码。

模型规划:
{json.dumps(model_plan, ensure_ascii=False, indent=2)}

任务分析:
{json.dumps(task_analysis, ensure_ascii=False, indent=2)}

请编写一个完整、可运行的Python程序，实现上述模拟需求。代码应该：
1. 使用标准的Python库和常用的科学计算库（如numpy、pandas、matplotlib等）
2. 结构清晰，采用面向对象的方式组织代码
3. 包含必要的注释和文档字符串
4. 实现模拟的核心逻辑和可视化功能
5. 可以直接运行并生成结果

请按照以下结构返回完整的代码：

```python
# 在这里编写完整的Python代码
```

只返回代码，不需要额外的解释。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 提取代码
        code = self._extract_code(response)
        
        # 记录推理过程
        reasoning = f"我根据模型规划和任务分析，编写了实现模拟功能的Python代码。"
        reasoning += f"\n代码使用了面向对象的方式组织，包含必要的注释和文档字符串。"
        reasoning += f"\n代码实现了模拟的核心逻辑和可视化功能，可以直接运行并生成结果。"
        
        self.memory.add_reasoning("编写代码", reasoning, code[:500] + "..." if len(code) > 500 else code)
        
        return {
            "success": True,
            "code": code
        }
    
    def _action_优化代码(self, code: str, optimization_targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        优化生成的代码以提高效率和可读性
        
        Args:
            code: 待优化的代码
            optimization_targets: 优化目标，例如["性能", "内存使用", "代码可读性"]
            
        Returns:
            优化后的代码
        """
        if optimization_targets is None:
            optimization_targets = ["性能", "内存使用", "代码可读性"]
        
        # 构建提示词
        prompt = f"""你是一位Python性能优化专家，请对以下代码进行优化，重点关注以下方面：
{', '.join(optimization_targets)}

原代码:
```python
{code}
```

请提供优化后的完整代码，并确保功能与原代码相同。请按照以下结构返回：

```python
# 在这里编写优化后的Python代码
```

只返回优化后的代码，不需要额外的解释。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 提取代码
        optimized_code = self._extract_code(response)
        
        # 记录推理过程
        reasoning = f"我对原代码进行了优化，重点关注了以下方面：{', '.join(optimization_targets)}。"
        reasoning += f"\n优化过程中保持了代码的原有功能，同时提高了代码的效率和可读性。"
        
        self.memory.add_reasoning("优化代码", reasoning, optimized_code[:500] + "..." if len(optimized_code) > 500 else optimized_code)
        
        return {
            "success": True,
            "code": optimized_code
        }
    
    def _action_注释代码(self, code: str) -> Dict[str, Any]:
        """
        为代码添加注释以提高可读性
        
        Args:
            code: 待注释的代码
            
        Returns:
            添加注释后的代码
        """
        # 构建提示词
        prompt = f"""你是一位Python文档专家，请为以下代码添加详细的注释和文档字符串，使其更易于理解和维护。

原代码:
```python
{code}
```

请确保添加以下类型的注释：
1. 文件顶部的总体描述
2. 每个类的文档字符串，说明其功能和用途
3. 每个方法的文档字符串，包括参数和返回值的说明
4. 复杂逻辑的行内注释
5. 关键算法的解释

请提供添加注释后的完整代码。请按照以下结构返回：

```python
# 在这里编写添加注释后的Python代码
```

只返回添加注释后的代码，不需要额外的解释。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 提取代码
        commented_code = self._extract_code(response)
        
        # 记录推理过程
        reasoning = f"我为代码添加了详细的注释和文档字符串，使其更易于理解和维护。"
        reasoning += f"\n添加的注释包括文件总体描述、类和方法的文档字符串、复杂逻辑的行内注释以及关键算法的解释。"
        
        self.memory.add_reasoning("注释代码", reasoning, commented_code[:500] + "..." if len(commented_code) > 500 else commented_code)
        
        return {
            "success": True,
            "code": commented_code
        }
    
    def _action_思考(self, question: str) -> Dict[str, Any]:
        """
        思考给定的问题
        
        Args:
            question: 思考的问题
            
        Returns:
            思考结果
        """
        # 构建提示词
        prompt = f"""请仔细思考以下与代码生成相关的问题，并给出你的分析：

问题: "{question}"

思考过程中，请考虑：
1. 这个问题的核心是什么？
2. 有哪些相关的编程概念和模式？
3. 可能的解决方案有哪些？
4. 这些解决方案的优缺点是什么？

请详细说明你的思考过程。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 记录推理过程
        self.memory.add_reasoning("思考", f"我思考了问题：{question}", response)
        
        return {
            "success": True,
            "result": response
        }
    
    def _action_分析(self, model_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析代码的实现难点和解决方案
        
        Args:
            model_plan: 模型规划
            
        Returns:
            分析结果
        """
        # 构建提示词
        prompt = f"""请分析以下模型规划的代码实现难点和可能的解决方案：

模型规划:
{json.dumps(model_plan, ensure_ascii=False, indent=2)}

请考虑以下方面：
1. 核心算法的复杂性和实现难点
2. 数据结构的选择和设计
3. 性能优化的关键点
4. 可能的技术障碍和解决方案
5. 代码架构和模块划分

请以JSON格式返回分析结果：
{{
    "核心算法": [
        {{
            "算法名称": "算法名称",
            "复杂度": "算法复杂度",
            "实现难点": "实现难点描述",
            "解决方案": "解决方案描述"
        }},
        ...
    ],
    "数据结构": [
        {{
            "名称": "数据结构名称",
            "用途": "用途描述",
            "设计考虑": "设计考虑点"
        }},
        ...
    ],
    "性能优化点": [
        "优化点1",
        "优化点2",
        ...
    ],
    "技术障碍": [
        {{
            "障碍描述": "障碍描述",
            "解决方案": "解决方案描述"
        }},
        ...
    ],
    "代码架构": {{
        "模块划分": [
            "模块1",
            "模块2",
            ...
        ],
        "模块关系": "模块间关系描述"
    }}
}}

只返回JSON对象，不要有其他文字。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 解析JSON
        try:
            result = json.loads(response)
            
            # 记录推理过程
            reasoning = f"我分析了模型规划的代码实现难点和可能的解决方案。"
            reasoning += f"\n核心算法方面，识别出{len(result.get('核心算法', []))}个关键算法。"
            reasoning += f"\n数据结构方面，推荐了{len(result.get('数据结构', []))}种关键数据结构。"
            reasoning += f"\n识别出{len(result.get('性能优化点', []))}个性能优化点和{len(result.get('技术障碍', []))}个可能的技术障碍。"
            reasoning += f"\n建议的代码架构包含{len(result.get('代码架构', {}).get('模块划分', []))}个模块。"
            
            self.memory.add_reasoning("分析", reasoning, str(result))
            
            return {
                "success": True,
                "result": result
            }
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON失败: {e}")
            return {
                "success": False,
                "error": f"解析JSON失败: {e}",
                "raw_response": response
            }
    
    def _extract_code(self, text: str) -> str:
        """
        从文本中提取Python代码
        
        Args:
            text: 包含代码的文本
            
        Returns:
            提取的代码
        """
        if "```python" in text:
            # 提取markdown格式的代码块
            parts = text.split("```python")
            if len(parts) > 1:
                code_block = parts[1].split("```")[0]
                return code_block.strip()
        elif "```" in text:
            # 提取无语言标记的代码块
            parts = text.split("```")
            if len(parts) > 1:
                code_block = parts[1]
                return code_block.strip()
        
        # 如果没有明确的代码块标记，返回整个文本
        return text.strip()
    
    def generate_simulation_code(self, model_plan: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成完整的模拟代码，包括编写、优化和注释
        
        Args:
            model_plan: 模型规划
            task_analysis: 任务分析结果
            
        Returns:
            生成的最终代码
        """
        # 1. 分析代码实现难点和解决方案
        analysis_result = self.execute_action("分析", model_plan=model_plan)
        if not analysis_result.get("success", False):
            return analysis_result
        
        # 2. 编写初始代码
        code_result = self.execute_action("编写代码", model_plan=model_plan, task_analysis=task_analysis)
        if not code_result.get("success", False):
            return code_result
        
        initial_code = code_result["code"]
        
        # 3. 优化代码
        optimization_targets = ["性能", "内存使用", "代码可读性"]
        optimize_result = self.execute_action("优化代码", code=initial_code, optimization_targets=optimization_targets)
        if not optimize_result.get("success", False):
            return optimize_result
        
        optimized_code = optimize_result["code"]
        
        # 4. 添加注释
        comment_result = self.execute_action("注释代码", code=optimized_code)
        if not comment_result.get("success", False):
            return comment_result
        
        final_code = comment_result["code"]
        
        # 5. 返回最终代码
        return {
            "success": True,
            "code": final_code,
            "analysis": analysis_result["result"] if analysis_result.get("success", False) else None
        } 