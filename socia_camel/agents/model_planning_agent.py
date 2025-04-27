import os
import json
import logging
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent, Action
from ..utils.llm_utils import LLMClient

logger = logging.getLogger("SOCIA-CAMEL")

class ModelPlanningAgent(BaseAgent):
    """
    模型规划智能体，负责设计适合任务的模拟模型和架构
    """
    def __init__(self, config_path: str = "../config.yaml"):
        # 定义可执行的动作
        actions = [
            Action(name="选择模型", description="根据任务需求选择合适的模拟模型"),
            Action(name="设计架构", description="设计模拟系统的架构和组件"),
            Action(name="确定参数", description="确定模型的关键参数和初始值"),
            Action(name="思考", description="思考给定的问题"),
            Action(name="分析", description="分析不同模型的适用性和局限性")
        ]
        
        # 初始化基类
        super().__init__(
            name="模型规划智能体",
            role="模型规划专家",
            goal="设计适合任务的模拟模型和架构",
            actions=actions,
            config_path=config_path
        )
        
        # 创建LLM客户端
        self.llm_client = LLMClient(config_path)
        
        # 加载可用模型列表
        self.available_models = self._load_available_models()
    
    def _load_available_models(self) -> List[str]:
        """加载可用模型列表"""
        return self.config.get("agents", {}).get("model_planning", {}).get(
            "available_models", ["gravity", "agent_based", "sir", "network", "system_dynamics"]
        )
    
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
    
    def _action_选择模型(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据任务需求选择合适的模拟模型
        
        Args:
            task_analysis: 任务分析结果
            
        Returns:
            模型选择结果
        """
        # 构建提示词
        prompt = f"""作为社会模拟专家，请根据以下任务分析结果，选择最合适的模拟模型。

任务分析:
{json.dumps(task_analysis, ensure_ascii=False, indent=2)}

可用的模型类型有:
{', '.join(self.available_models)}

请针对此任务分析，提供以下内容：
1. 模型评估：列出每种可能适用的模型类型，包括其优缺点和适用场景
2. 推荐模型：选择一个最适合的模型
3. 选择理由：详细解释为什么这个模型最适合此任务
4. 备选方案：如果有的话，提供1-2个备选模型
5. 实现建议：有关如何实现该模型的建议

请以JSON格式返回，包含以下字段：
- 模型评估 (数组)：对各种可能模型的评估
- 推荐模型 (字符串)：最适合的模型名称
- 选择理由 (字符串)：选择该模型的理由
- 备选方案 (数组)：备选模型名称列表
- 实现建议 (字符串)：关于模型实现的建议
"""
        
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
            reasoning = f"我分析了任务需求，评估了{len(result.get('模型评估', []))}种可能的模型。"""
            reasoning += f"\n推荐使用{result.get('推荐模型', '未指定')}模型，选择理由是：{result.get('选择理由', '未给出')}。"""
            reasoning += f"\n备选方案包括：{', '.join(result.get('备选方案', ['无']))}。"""
            reasoning += f"\n实现建议：{result.get('实现建议', '无')}。"""
            
            self.memory.add_reasoning("选择模型", reasoning, str(result))
            
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
    
    def _action_设计架构(self, recommended_model: str, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        设计模拟系统的架构和组件
        
        Args:
            recommended_model: 推荐的模型名称
            task_analysis: 任务分析结果
            
        Returns:
            架构设计结果
        """
        # 构建提示词
        prompt = f"""请为以下模拟任务设计详细的系统架构和组件。

推荐模型: {recommended_model}

任务分析:
{json.dumps(task_analysis, ensure_ascii=False, indent=2)}

请设计一个完整的模拟系统架构，包括核心组件、数据流和交互关系。请以JSON格式返回设计结果：
{{
    "系统名称": "模拟系统名称",
    "整体架构": "整体架构描述",
    "核心组件": [
        {{
            "名称": "组件名称",
            "功能": "组件功能描述",
            "输入": ["输入1", "输入2", ...],
            "输出": ["输出1", "输出2", ...],
            "算法": "使用的算法或方法"
        }},
        ...
    ],
    "数据流": [
        {{
            "源组件": "源组件名称",
            "目标组件": "目标组件名称",
            "数据类型": "传递的数据类型",
            "说明": "数据流说明"
        }},
        ...
    ],
    "交互关系": "组件间的交互关系描述",
    "扩展性考虑": "系统扩展性的考虑",
    "性能考虑": "性能优化的考虑"
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
            reasoning = f"我为{recommended_model}模型设计了系统架构，命名为\"{result.get('系统名称', '未命名')}\"。"
            reasoning += f"\n架构设计基于以下考虑："
            reasoning += f"\n- 实体定义：{result.get('实体定义', '无')}"
            reasoning += f"\n- 交互机制：{result.get('交互机制', '无')}"
            reasoning += f"\n- 时间模型：{result.get('时间模型', '无')}"
            reasoning += f"\n- 环境设置：{result.get('环境设置', '无')}"
            
            logger.info(f"为{recommended_model}模型设计架构")
            
            self.memory.add_reasoning("设计架构", reasoning, str(result))
            
            return {
                "success": True,
                "result": result,
                "reasoning": reasoning
            }
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON失败: {e}")
            return {
                "success": False,
                "error": f"解析JSON失败: {e}",
                "raw_response": response
            }
    
    def _action_确定参数(self, model_architecture: Dict[str, Any], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        确定模型的关键参数和初始值
        
        Args:
            model_architecture: 模型架构设计
            task_analysis: 任务分析结果
            
        Returns:
            参数设置结果
        """
        # 构建提示词
        prompt = f"""请根据以下模型架构和任务分析，确定模拟模型的关键参数和初始值。

模型架构:
{json.dumps(model_architecture, ensure_ascii=False, indent=2)}

任务分析:
{json.dumps(task_analysis, ensure_ascii=False, indent=2)}

请确定模型的所有关键参数及其合理的初始值。请以JSON格式返回结果：
{{
    "模型参数": [
        {{
            "名称": "参数名称",
            "描述": "参数描述",
            "数据类型": "参数数据类型(int, float, bool, str, list, etc.)",
            "默认值": 默认值,
            "取值范围": "参数取值范围",
            "影响": "参数对模型的影响",
            "调优建议": "参数调优建议"
        }},
        ...
    ],
    "初始状态参数": [
        {{
            "名称": "参数名称",
            "描述": "参数描述",
            "数据类型": "参数数据类型",
            "默认值": 默认值,
            "说明": "参数说明"
        }},
        ...
    ],
    "环境参数": [
        {{
            "名称": "参数名称",
            "描述": "参数描述",
            "数据类型": "参数数据类型",
            "默认值": 默认值,
            "说明": "参数说明"
        }},
        ...
    ],
    "其他配置": {{
        "参数名称1": 值1,
        "参数名称2": 值2,
        ...
    }},
    "参数关系": [
        "参数间关系描述1",
        "参数间关系描述2",
        ...
    ],
    "参数敏感性": "模型对参数变化的敏感性分析"
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
            reasoning = f"我确定了模型所需的关键参数，包括{len(result.get('模型参数', []))}个核心模型参数、"
            reasoning += f"{len(result.get('初始状态参数', []))}个初始状态参数和{len(result.get('环境参数', []))}个环境参数。"
            reasoning += f"\n对每个参数，我确定了合适的默认值、取值范围和影响。"
            reasoning += f"\n我分析了{len(result.get('参数关系', []))}条参数间的关系，确保参数设置的合理性和一致性。"
            reasoning += f"\n同时，我提供了参数敏感性分析，说明哪些参数对模型结果影响较大。"
            
            self.memory.add_reasoning("确定参数", reasoning, str(result))
            
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
    
    def _action_思考(self, question: str) -> Dict[str, Any]:
        """
        思考给定的问题
        
        Args:
            question: 思考的问题
            
        Returns:
            思考结果
        """
        # 构建提示词
        prompt = f"""请仔细思考以下与模型规划相关的问题，并给出你的分析：

问题: "{question}"

思考过程中，请考虑：
1. 这个问题的核心是什么？
2. 有哪些相关的模型和模拟方法？
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
    
    def _action_分析(self, model_types: List[str], task_description: str) -> Dict[str, Any]:
        """
        分析不同模型的适用性和局限性
        
        Args:
            model_types: 模型类型列表
            task_description: 任务描述
            
        Returns:
            分析结果
        """
        if not model_types:
            model_types = self.available_models
        
        # 构建提示词
        prompt = f"""请分析以下模型类型对于给定任务的适用性和局限性：

模型类型:
{', '.join(model_types)}

任务描述:
{task_description}

请对每种模型类型进行详细分析，评估其在该任务中的适用性和局限性。请以JSON格式返回分析结果：
{{
    "模型分析": [
        {{
            "模型类型": "模型名称",
            "适用性": "适用性描述",
            "局限性": ["局限1", "局限2", ...],
            "适合的场景": ["场景1", "场景2", ...],
            "不适合的场景": ["场景1", "场景2", ...],
            "评分": 0-10的评分
        }},
        ...
    ],
    "最适合的模型": "最适合的模型名称",
    "选择理由": "选择该模型的理由",
    "混合模型可能性": "是否可以结合多种模型，如何结合",
    "总体建议": "模型选择的总体建议"
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
            reasoning = f"我分析了{len(result.get('模型分析', []))}种模型类型对于任务\"{task_description}\"的适用性和局限性。"
            reasoning += f"\n对每种模型，我评估了其适用场景、局限性，并给出了0-10的评分。"
            reasoning += f"\n综合分析，我认为{result.get('最适合的模型', '未指定')}最适合此任务，原因是：{result.get('选择理由', '未给出')}。"
            reasoning += f"\n关于混合模型的可能性：{result.get('混合模型可能性', '未评估')}。"
            reasoning += f"\n总体建议：{result.get('总体建议', '无')}。"
            
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
    
    def design_model(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        设计完整的模拟模型，包括选择模型、设计架构和确定参数
        
        Args:
            task_analysis: 任务分析结果
            
        Returns:
            完整的模型设计结果
        """
        # 1. 选择模型
        model_result = self.execute_action("选择模型", task_analysis=task_analysis)
        if not model_result.get("success", False):
            return model_result
        
        recommended_model = model_result["result"]["推荐模型"]
        
        # 2. 设计架构
        architecture_result = self.execute_action("设计架构", recommended_model=recommended_model, task_analysis=task_analysis)
        if not architecture_result.get("success", False):
            return architecture_result
        
        model_architecture = architecture_result["result"]
        
        # 3. 确定参数
        params_result = self.execute_action("确定参数", model_architecture=model_architecture, task_analysis=task_analysis)
        if not params_result.get("success", False):
            return params_result
        
        # 4. 组合所有结果
        final_result = {
            "model_selection": model_result["result"],
            "model_architecture": model_architecture,
            "model_parameters": params_result["result"]
        }
        
        return {
            "success": True,
            "result": final_result
        }
