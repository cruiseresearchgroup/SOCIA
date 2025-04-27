import os
import json
import yaml
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger("SOCIA-CAMEL")

class Memory:
    """
    CAMEL架构的记忆模块，用于存储Agent的交互历史和重要信息
    """
    def __init__(self, max_history_length: int = 20, importance_threshold: float = 0.5):
        self.conversations: List[Dict[str, Any]] = []
        self.important_information: List[Dict[str, Any]] = []
        self.reasoning_history: List[Dict[str, Any]] = []
        self.max_history_length = max_history_length
        self.importance_threshold = importance_threshold
    
    def add_conversation(self, role: str, content: str, importance: float = 0.0) -> None:
        """
        添加一条对话记录
        
        Args:
            role: 发言者角色（'user', 'assistant', 'system'等）
            content: 对话内容
            importance: 重要性分数（0.0-1.0）
        """
        conversation = {
            "role": role,
            "content": content,
            "importance": importance,
            "timestamp": self._get_timestamp()
        }
        
        self.conversations.append(conversation)
        
        # 如果超过最大长度，移除最旧的记录
        if len(self.conversations) > self.max_history_length:
            self.conversations.pop(0)
        
        # 如果重要性超过阈值，加入重要信息列表
        if importance >= self.importance_threshold:
            self.important_information.append(conversation)
    
    def add_reasoning(self, action: str, reasoning: str, result: Optional[str] = None) -> None:
        """
        添加一条推理记录
        
        Args:
            action: 执行的动作
            reasoning: 推理过程
            result: 推理结果
        """
        reasoning_record = {
            "action": action,
            "reasoning": reasoning,
            "result": result,
            "timestamp": self._get_timestamp()
        }
        
        self.reasoning_history.append(reasoning_record)
    
    def get_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取最近的对话记录"""
        if limit:
            return self.conversations[-limit:]
        return self.conversations
    
    def get_important_information(self) -> List[Dict[str, Any]]:
        """获取重要信息列表"""
        return self.important_information
    
    def get_reasoning_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取推理历史"""
        if limit:
            return self.reasoning_history[-limit:]
        return self.reasoning_history
    
    def clear(self) -> None:
        """清空所有记忆"""
        self.conversations = []
        self.important_information = []
        self.reasoning_history = []
    
    def save_to_file(self, filepath: str) -> None:
        """保存记忆到文件"""
        memory_data = {
            "conversations": self.conversations,
            "important_information": self.important_information,
            "reasoning_history": self.reasoning_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """从文件加载记忆"""
        if not os.path.exists(filepath):
            logger.warning(f"记忆文件不存在: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            memory_data = json.load(f)
        
        self.conversations = memory_data.get("conversations", [])
        self.important_information = memory_data.get("important_information", [])
        self.reasoning_history = memory_data.get("reasoning_history", [])
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


class Action:
    """
    CAMEL架构的行动模块，定义Agent可以执行的操作
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class BaseAgent(ABC):
    """
    基于CAMEL架构的基础Agent类
    """
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        actions: List[Action],
        config_path: str = "config.yaml"
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.actions = actions
        self.memory = Memory()
        
        # 尝试从不同位置加载配置文件
        self.config = self._load_config(config_path)
        self.llm_config = self._get_llm_config()
        self.system_prompt = self._create_system_prompt()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            # 尝试从多个可能的位置加载配置文件
            potential_paths = [
                config_path,  # 直接使用传入的路径
                os.path.join(os.path.dirname(__file__), "..", config_path),  # 相对于当前文件的上级目录
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", config_path)  # 使用绝对路径
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return yaml.safe_load(f)
            
            # 如果所有路径都不存在，抛出异常
            raise FileNotFoundError(f"配置文件未找到：尝试了以下路径：{potential_paths}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        provider = self.config.get("llm", {}).get("provider", "openai")
        return self.config.get("llm_providers", {}).get(provider, {})
    
    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        base_prompt = self.config.get("llm", {}).get("system_prompt", "")
        
        return f"""{base_prompt}
        
你是一个名为 {self.name} 的智能体，你的角色是 {self.role}。
你的目标是: {self.goal}

在执行任务时，你应该遵循以下原则:
1. 始终清晰地思考问题，使用逐步推理
2. 当需要做决策时，考虑多种可能性并权衡利弊
3. 保持专业和专注，始终朝着目标前进

你可以执行的动作包括:
{self._format_actions()}
"""
    
    def _format_actions(self) -> str:
        """格式化动作列表"""
        return "\n".join([f"- {action}" for action in self.actions])
    
    def execute_action(self, action_name: str, **kwargs) -> Dict[str, Any]:
        """
        执行指定的动作
        
        Args:
            action_name: 动作名称
            **kwargs: 动作参数
            
        Returns:
            动作执行结果
        """
        # 检查动作是否存在
        action = next((a for a in self.actions if a.name == action_name), None)
        if not action:
            logger.error(f"未找到动作: {action_name}")
            return {"success": False, "error": f"未找到动作: {action_name}"}
        
        # 调用具体的动作方法
        method_name = f"_action_{action_name}"
        if not hasattr(self, method_name):
            logger.error(f"动作方法未实现: {method_name}")
            return {"success": False, "error": f"动作方法未实现: {method_name}"}
        
        # 执行动作前的思考
        reasoning = self._reasoning_before_action(action_name, **kwargs)
        
        # 记录推理过程
        self.memory.add_reasoning(action_name, reasoning)
        
        # 执行动作
        method = getattr(self, method_name)
        result = method(**kwargs)
        
        # 更新记忆
        self.memory.add_reasoning(action_name, reasoning, str(result))
        
        return result
    
    def _reasoning_before_action(self, action_name: str, **kwargs) -> str:
        """
        在执行动作前进行推理（Chain-of-Thought）
        
        Args:
            action_name: 动作名称
            **kwargs: 动作参数
            
        Returns:
            推理过程
        """
        # 这里可以调用LLM进行推理，但为简化起见，先提供一个基础实现
        reasoning = f"我需要执行 {action_name} 动作。\n"
        reasoning += f"考虑因素包括：\n"
        
        for key, value in kwargs.items():
            reasoning += f"- {key}: {value}\n"
        
        reasoning += f"\n基于上述因素，我决定执行{action_name}动作。"
        return reasoning
    
    def respond(self, message: str) -> str:
        """
        响应用户消息
        
        Args:
            message: 用户消息
            
        Returns:
            Agent响应
        """
        # 添加用户消息到记忆
        self.memory.add_conversation("user", message)
        
        # 生成响应（子类应该实现具体的响应生成逻辑）
        response = self._generate_response(message)
        
        # 添加响应到记忆
        self.memory.add_conversation("assistant", response)
        
        return response
    
    @abstractmethod
    def _generate_response(self, message: str) -> str:
        """
        生成响应，需要被子类实现
        
        Args:
            message: 用户消息
            
        Returns:
            Agent响应
        """
        pass
    
    def save_state(self, directory: str) -> None:
        """
        保存Agent状态
        
        Args:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)
        
        # 保存记忆
        memory_path = os.path.join(directory, f"{self.name}_memory.json")
        self.memory.save_to_file(memory_path)
        
        # 保存Agent配置
        config_path = os.path.join(directory, f"{self.name}_config.json")
        config_data = {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "actions": [{"name": a.name, "description": a.description} for a in self.actions]
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_state(self, directory: str) -> None:
        """
        加载Agent状态
        
        Args:
            directory: 加载目录
        """
        # 加载记忆
        memory_path = os.path.join(directory, f"{self.name}_memory.json")
        self.memory.load_from_file(memory_path)
        
        # 加载Agent配置
        config_path = os.path.join(directory, f"{self.name}_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.name = config_data.get("name", self.name)
            self.role = config_data.get("role", self.role)
            self.goal = config_data.get("goal", self.goal) 