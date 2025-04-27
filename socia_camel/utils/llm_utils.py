import os
import logging
import requests
import json
import yaml
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("SOCIA-CAMEL")

def load_api_key(key_name: str) -> Optional[str]:
    """
    尝试从keys.py文件或环境变量加载API密钥
    
    Args:
        key_name: API密钥名称
    
    Returns:
        API密钥值，如果未找到则返回None
    """
    # 首先尝试从keys.py文件加载
    try:
        # 获取脚本所在目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建可能的keys.py路径
        potential_paths = [
            os.path.join(os.getcwd(), "keys.py"),  # 当前工作目录
            os.path.join(current_dir, "keys.py"),  # utils目录
            os.path.join(current_dir, "..", "keys.py"),  # socia_camel目录
            os.path.join(current_dir, "..", "..", "keys.py"),  # 项目根目录
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"找到keys.py文件: {path}")
                import importlib.util
                spec = importlib.util.spec_from_file_location("keys", path)
                keys = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(keys)
                
                api_key = getattr(keys, key_name, None)
                if api_key:
                    logger.info(f"成功从{path}加载API密钥: {key_name}")
                    return api_key
    except Exception as e:
        logger.warning(f"从keys.py加载API密钥时出错: {e}")
    
    # 然后尝试从环境变量加载
    api_key = os.environ.get(key_name)
    if api_key:
        logger.info(f"成功从环境变量加载API密钥: {key_name}")
        return api_key
    
    logger.warning(f"无法找到API密钥: {key_name}")
    return None

class LLMClient:
    """
    大语言模型客户端，支持多种LLM提供商
    """
    def __init__(self, config_path: str = "../config.yaml"):
        self.config = self._load_config(config_path)
        self.provider = self.config.get("llm", {}).get("provider", "openai")
        self.provider_config = self.config.get("llm_providers", {}).get(self.provider, {})
        self.api_key = self._get_api_key()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _get_api_key(self) -> Optional[str]:
        """获取API密钥"""
        # 首先检查配置文件中是否有API密钥
        api_key = self.provider_config.get("api_key")
        if api_key:
            return api_key
        
        # 然后尝试从环境变量或keys.py加载
        key_name = f"{self.provider.upper()}_API_KEY"
        return load_api_key(key_name)
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        生成LLM响应
        
        Args:
            messages: 消息列表，每个消息包含'role'和'content'
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成令牌数
            
        Returns:
            LLM生成的响应
        """
        if self.provider == "openai":
            return self._openai_generate(messages, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._gemini_generate(messages, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._anthropic_generate(messages, temperature, max_tokens)
        elif self.provider == "llama":
            return self._llama_generate(messages, temperature, max_tokens)
        else:
            logger.error(f"不支持的LLM提供商: {self.provider}")
            return "错误：不支持的LLM提供商"
    
    def _openai_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """使用OpenAI API生成响应"""
        try:
            import openai
            
            # 设置API密钥
            openai.api_key = self.api_key
            
            # 设置参数
            if temperature is None:
                temperature = self.provider_config.get("temperature", 0.7)
            
            if max_tokens is None:
                max_tokens = self.provider_config.get("max_tokens", 4000)
            
            logger.debug(f"调用OpenAI API，模型: {self.provider_config.get('model', 'gpt-4o')}")
            
            # 调用API
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.provider_config.get("model", "gpt-4o"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.debug("OpenAI API调用成功")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return f"错误：OpenAI API调用失败 - {str(e)}"
    
    def _gemini_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """使用Google Gemini API生成响应"""
        try:
            import google.generativeai as genai
            
            # 设置API密钥
            genai.configure(api_key=self.api_key)
            
            # 设置参数
            if temperature is None:
                temperature = self.provider_config.get("temperature", 0.7)
            
            if max_tokens is None:
                max_tokens = self.provider_config.get("max_tokens", 8192)
            
            # 创建模型
            model = genai.GenerativeModel(
                model_name=self.provider_config.get("model", "models/gemini-1.5-pro"),
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            # 转换消息格式
            gemini_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
            
            # 调用API
            response = model.generate_content(gemini_messages)
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API调用失败: {e}")
            return f"错误：Gemini API调用失败 - {str(e)}"
    
    def _anthropic_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """使用Anthropic Claude API生成响应"""
        try:
            import anthropic
            
            # 设置API密钥
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # 设置参数
            if temperature is None:
                temperature = self.provider_config.get("temperature", 0.7)
            
            if max_tokens is None:
                max_tokens = self.provider_config.get("max_tokens", 4000)
            
            # 转换消息格式
            claude_messages = []
            for msg in messages:
                claude_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                })
            
            # 调用API
            response = client.messages.create(
                model=self.provider_config.get("model", "claude-3-opus-20240229"),
                messages=claude_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API调用失败: {e}")
            return f"错误：Anthropic API调用失败 - {str(e)}"
    
    def _llama_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """使用本地Llama模型生成响应"""
        try:
            # 这里假设已经安装了llama-cpp-python
            from llama_cpp import Llama
            
            # 设置参数
            if temperature is None:
                temperature = self.provider_config.get("temperature", 0.7)
            
            if max_tokens is None:
                max_tokens = self.provider_config.get("max_tokens", 2048)
            
            # 加载模型
            model_path = self.provider_config.get("model_path", "./models/llama-3-8b-instruct")
            llm = Llama(model_path=model_path)
            
            # 转换消息格式为提示词
            prompt = ""
            for msg in messages:
                role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                prompt += f"{role_prefix}{msg['content']}\n"
            
            prompt += "Assistant: "
            
            # 生成响应
            output = llm(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Llama模型调用失败: {e}")
            return f"错误：Llama模型调用失败 - {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        try:
            if self.provider == "openai":
                import openai
                openai.api_key = self.api_key
                
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                
                return response.data[0].embedding
            else:
                logger.warning(f"目前只支持OpenAI的嵌入功能，当前提供商是{self.provider}")
                # 返回空向量作为回退
                return [0.0] * 10
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            # 返回空向量作为回退
            return [0.0] * 10 