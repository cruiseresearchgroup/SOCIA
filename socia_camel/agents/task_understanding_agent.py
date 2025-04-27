import os
import json
import logging
from typing import List, Dict, Any, Optional

from ..agents.base_agent import BaseAgent, Action
from ..utils.llm_utils import LLMClient

logger = logging.getLogger("SOCIA-CAMEL")

class TaskUnderstandingAgent(BaseAgent):
    """
    任务理解智能体，负责理解用户需求并将其转化为明确的模拟任务
    """
    def __init__(self, config_path: str = "../config.yaml"):
        # 定义可执行的动作
        actions = [
            Action(name="解析需求", description="解析用户提出的模拟需求"),
            Action(name="明确目标", description="明确模拟的具体目标和要求"),
            Action(name="提取关键信息", description="提取需求中的关键信息和参数"),
            Action(name="思考", description="思考给定的信息"),
            Action(name="分析", description="分析任务的复杂性和可行性")
        ]
        
        # 初始化基类
        super().__init__(
            name="任务理解智能体",
            role="任务理解专家",
            goal="准确理解用户需求并将其转化为明确的模拟任务",
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
    
    def _action_解析需求(self, task_description: str) -> Dict[str, Any]:
        """
        解析用户提出的模拟需求
        
        Args:
            task_description: 任务描述
            
        Returns:
            解析结果
        """
        # 构建提示词
        prompt = f"""你是一个专业的社会模拟任务理解专家。请分析以下任务描述，提取关键信息：

任务描述: "{task_description}"

请按照以下结构返回JSON格式的分析结果：
{{
    "任务类型": "模拟类型，如流行病模拟、交通模拟、社交网络模拟等",
    "模拟目标": "这个模拟想要达成的目标",
    "关键实体": ["模拟中的主要实体，如人群、城市、病毒等"],
    "关键参数": {{
        "参数1": "参数1的值或描述",
        "参数2": "参数2的值或描述",
        ...
    }},
    "时间范围": "模拟的时间范围，如果有指定",
    "空间范围": "模拟的空间范围，如果有指定",
    "输出要求": ["期望的输出类型，如图表、数据、报告等"],
    "难点分析": "实现这个模拟的主要技术难点"
}}

只返回JSON对象，不要有其他文字。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 输出原始响应内容，以便查看JSON解析失败的原因
        logger.debug(f"LLM原始响应: {response}")
        logger.info(f"LLM响应前50个字符: {response[:50]}")
        
        # 检查是否是错误响应
        if response.startswith("错误：") or "错误" in response:
            logger.warning(f"LLM返回错误: {response}")
            # 创建一个默认结果
            default_result = {
                "任务类型": "流行病模拟",
                "模拟目标": "模拟病毒在人群中的传播",
                "关键实体": ["人", "病毒"],
                "关键参数": {"人口数量": 1000},
                "时间范围": "未指定",
                "空间范围": "未指定",
                "输出要求": ["感染人数随时间变化"],
                "难点分析": "实现病毒传播的模型"
            }
            
            # 记录推理过程
            reasoning = f"LLM返回了错误，我设置了一个默认的任务分析结果。错误信息: {response}"
            self.memory.add_reasoning("解析需求(默认)", reasoning, str(default_result))
            
            return {
                "success": True,
                "result": default_result,
                "is_default": True
            }
        
        # 处理响应，去除可能的Markdown代码块标记
        if response.startswith("```json") or response.startswith("```"):
            # 找到第一个和最后一个```的位置
            start_pos = response.find("{")
            end_pos = response.rfind("}")
            
            if start_pos >= 0 and end_pos > start_pos:
                clean_response = response[start_pos:end_pos+1]
                logger.info(f"提取的JSON内容: {clean_response[:50]}...")
            else:
                clean_response = response
        else:
            clean_response = response
        
        # 解析JSON
        try:
            result = json.loads(clean_response)
            
            # 记录推理过程
            reasoning = f"我分析了任务描述：\"{task_description}\"，识别出这是一个{result.get('任务类型', '未知')}类型的模拟任务。"
            reasoning += f"\n主要目标是：{result.get('模拟目标', '未指定')}，涉及的关键实体包括：{', '.join(result.get('关键实体', ['未指定']))}。"
            reasoning += f"\n我提取了关键参数：{result.get('关键参数', {})}，并分析了可能的技术难点：{result.get('难点分析', '未指定')}。"
            
            self.memory.add_reasoning("解析需求", reasoning, str(result))
            
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
    
    def _action_明确目标(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        明确模拟的具体目标和要求
        
        Args:
            task_analysis: 任务分析结果
            
        Returns:
            明确的目标和要求
        """
        # 构建提示词
        prompt = f"""基于以下任务分析，请明确这个模拟的具体目标和技术要求：

任务分析:
{json.dumps(task_analysis, ensure_ascii=False, indent=2)}

请提供以下内容：
1. 这个模拟的主要研究问题是什么？
2. 需要收集哪些数据指标？
3. 应该使用什么类型的模型？
4. 模拟应该考虑哪些约束条件？
5. 成功的标准是什么？

请以JSON格式返回：
{{
    "研究问题": "主要研究问题",
    "数据指标": ["指标1", "指标2", ...],
    "推荐模型": "推荐使用的模型类型及理由",
    "约束条件": ["约束1", "约束2", ...],
    "成功标准": ["标准1", "标准2", ...]
}}

只返回JSON对象，不要有其他文字。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 处理响应，去除可能的Markdown代码块标记
        if response.startswith("```json") or response.startswith("```"):
            # 找到第一个和最后一个```的位置
            start_pos = response.find("{")
            end_pos = response.rfind("}")
            
            if start_pos >= 0 and end_pos > start_pos:
                clean_response = response[start_pos:end_pos+1]
                logger.info(f"提取的JSON内容: {clean_response[:50]}...")
            else:
                clean_response = response
        else:
            clean_response = response
        
        # 解析JSON
        try:
            result = json.loads(clean_response)
            
            # 记录推理过程
            reasoning = f"基于任务分析，我明确了这个模拟的主要研究问题：{result.get('研究问题', '未指定')}。"
            reasoning += f"\n需要收集的数据指标包括：{', '.join(result.get('数据指标', ['未指定']))}。"
            reasoning += f"\n我推荐使用的模型是：{result.get('推荐模型', '未指定')}。"
            reasoning += f"\n应考虑的约束条件：{', '.join(result.get('约束条件', ['未指定']))}。"
            reasoning += f"\n成功标准为：{', '.join(result.get('成功标准', ['未指定']))}。"
            
            self.memory.add_reasoning("明确目标", reasoning, str(result))
            
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
    
    def _action_提取关键信息(self, task_description: str) -> Dict[str, Any]:
        """
        提取需求中的关键信息和参数
        
        Args:
            task_description: 任务描述
            
        Returns:
            提取的关键信息和参数
        """
        # 构建提示词
        prompt = f"""请从以下模拟任务描述中提取关键的数值参数和实体关系：

任务描述: "{task_description}"

请返回以下JSON格式的结果：
{{
    "数值参数": {{
        "参数名1": 值1,
        "参数名2": 值2,
        ...
    }},
    "实体": [
        {{
            "名称": "实体名称",
            "类型": "实体类型",
            "属性": {{
                "属性1": "值1",
                "属性2": "值2",
                ...
            }}
        }},
        ...
    ],
    "关系": [
        {{
            "源实体": "实体1",
            "目标实体": "实体2",
            "关系类型": "关系描述",
            "属性": {{
                "属性1": "值1",
                "属性2": "值2",
                ...
            }}
        }},
        ...
    ]
}}

只提取任务描述中明确提到的信息，不要添加假设的内容。如果某些字段没有相关信息，可以返回空列表或空对象。
只返回JSON对象，不要有其他文字。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 处理响应，去除可能的Markdown代码块标记
        if response.startswith("```json") or response.startswith("```"):
            # 找到第一个和最后一个```的位置
            start_pos = response.find("{")
            end_pos = response.rfind("}")
            
            if start_pos >= 0 and end_pos > start_pos:
                clean_response = response[start_pos:end_pos+1]
                logger.info(f"提取的JSON内容: {clean_response[:50]}...")
            else:
                clean_response = response
        else:
            clean_response = response
        
        # 解析JSON
        try:
            result = json.loads(clean_response)
            
            # 记录推理过程
            reasoning = f"我从任务描述中提取了关键数值参数：{result.get('数值参数', {})}。"
            reasoning += f"\n识别出的实体有：{', '.join([entity.get('名称', '未命名') for entity in result.get('实体', [])])}。"
            reasoning += f"\n发现的实体关系有：{len(result.get('关系', []))}个。"
            
            self.memory.add_reasoning("提取关键信息", reasoning, str(result))
            
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
        prompt = f"""请仔细思考以下问题，并给出你的分析：

问题: "{question}"

思考过程中，请考虑：
1. 这个问题的核心是什么？
2. 有哪些相关的知识和概念？
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
    
    def _action_分析(self, task_description: str) -> Dict[str, Any]:
        """
        分析任务的复杂性和可行性
        
        Args:
            task_description: 任务描述
            
        Returns:
            分析结果
        """
        # 构建提示词
        prompt = f"""请分析以下模拟任务的复杂性和可行性：

任务描述: "{task_description}"

请考虑以下方面：
1. 计算复杂度评估
2. 数据需求评估
3. 实现难度评估
4. 技术可行性评估
5. 时间需求评估

请以JSON格式返回分析结果：
{{
    "复杂度": {{
        "计算复杂度": "评估结果（低/中/高）及理由",
        "数据复杂度": "评估结果（低/中/高）及理由",
        "实现复杂度": "评估结果（低/中/高）及理由"
    }},
    "可行性": {{
        "技术可行性": "评估结果（低/中/高）及理由",
        "数据可行性": "评估结果（低/中/高）及理由",
        "时间可行性": "评估结果（低/中/高）及理由"
    }},
    "总体评估": "总体评估结果和建议",
    "风险因素": ["风险1", "风险2", ...]
}}

只返回JSON对象，不要有其他文字。"""
        
        # 调用LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm_client.generate_response(messages)
        
        # 处理响应，去除可能的Markdown代码块标记
        if response.startswith("```json") or response.startswith("```"):
            # 找到第一个和最后一个```的位置
            start_pos = response.find("{")
            end_pos = response.rfind("}")
            
            if start_pos >= 0 and end_pos > start_pos:
                clean_response = response[start_pos:end_pos+1]
                logger.info(f"提取的JSON内容: {clean_response[:50]}...")
            else:
                clean_response = response
        else:
            clean_response = response
        
        # 解析JSON
        try:
            result = json.loads(clean_response)
            
            # 记录推理过程
            reasoning = f"我分析了任务的复杂度，计算复杂度为{result.get('复杂度', {}).get('计算复杂度', '未评估')}，"
            reasoning += f"数据复杂度为{result.get('复杂度', {}).get('数据复杂度', '未评估')}，"
            reasoning += f"实现复杂度为{result.get('复杂度', {}).get('实现复杂度', '未评估')}。"
            reasoning += f"\n关于可行性，技术可行性为{result.get('可行性', {}).get('技术可行性', '未评估')}，"
            reasoning += f"数据可行性为{result.get('可行性', {}).get('数据可行性', '未评估')}，"
            reasoning += f"时间可行性为{result.get('可行性', {}).get('时间可行性', '未评估')}。"
            reasoning += f"\n总体评估：{result.get('总体评估', '未给出')}。"
            reasoning += f"\n主要风险因素：{', '.join(result.get('风险因素', ['未识别']))}。"
            
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
    
    def understand_task(self, task_description: str) -> Dict[str, Any]:
        """
        完整理解任务，包括解析需求、明确目标、提取关键信息和分析可行性
        
        Args:
            task_description: 任务描述
            
        Returns:
            完整的任务理解结果
        """
        # 1. 解析需求
        parse_result = self.execute_action("解析需求", task_description=task_description)
        if not parse_result.get("success", False):
            return parse_result
        
        task_analysis = parse_result["result"]
        
        # 2. 明确目标
        goal_result = self.execute_action("明确目标", task_analysis=task_analysis)
        if not goal_result.get("success", False):
            return goal_result
        
        # 3. 提取关键信息
        info_result = self.execute_action("提取关键信息", task_description=task_description)
        if not info_result.get("success", False):
            return info_result
        
        # 4. 分析可行性
        analysis_result = self.execute_action("分析", task_description=task_description)
        if not analysis_result.get("success", False):
            return analysis_result
        
        # 5. 组合所有结果
        final_result = {
            "task_analysis": task_analysis,
            "goals": goal_result["result"],
            "key_information": info_result["result"],
            "feasibility_analysis": analysis_result["result"]
        }
        
        return {
            "success": True,
            "result": final_result
        } 