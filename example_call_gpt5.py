"""
示例：使用OpenAI Responses API调用gpt-5模型
"""
import os
from openai import OpenAI


def get_openai_api_key():
    """
    获取OpenAI API密钥
    优先级：环境变量 > keys.py文件
    """
    # 方法1: 从环境变量获取 (推荐)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        return api_key
    
    # # 方法2: 从keys.py文件获取 (如果存在)
    # try:
    #     import keys
    #     api_key = getattr(keys, "OPENAI_API_KEY", None)
    #     if api_key:
    #         return api_key
    # except ImportError:
    #     pass
    
    raise ValueError("OpenAI API key not found in environment or keys.py")


def call_gpt5_with_responses_api(prompt: str, model: str = "gpt-5", max_output_tokens: int = 4000):
    """
    使用OpenAI Responses API调用LLM
    
    Args:
        prompt: 输入的提示词
        model: 模型名称，默认 "gpt-5"
        max_output_tokens: 最大输出token数，默认4000
    
    Returns:
        str: LLM的响应文本
    """
    # 1. 获取API密钥
    api_key = get_openai_api_key()
    
    # 2. 初始化OpenAI客户端
    client = OpenAI(api_key=api_key)
    
    # 3. 构建Responses API的参数
    responses_kwargs = {
        "model": model,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
        ],
        # 注意：Responses API使用 max_output_tokens 而不是 max_tokens
        "max_output_tokens": max_output_tokens,
        # 注意：gpt-5等新模型可能不支持temperature参数
        # 如果需要temperature，请注释掉下一行
        # "temperature": 0.7,
    }
    
    # 4. 调用Responses API
    resp = client.responses.create(**responses_kwargs)
    
    # 5. 提取响应文本
    def extract_response(resp_obj):
        """从响应对象中提取文本"""
        # 优先使用SDK提供的输出方法
        if hasattr(resp_obj, "output_text") and isinstance(resp_obj.output_text, str):
            return resp_obj.output_text
        
        # 备用方法：从响应结构体中提取
        try:
            output = getattr(resp_obj, "output", None)
            if output and isinstance(output, list):
                content = output[0].get("content") if isinstance(output[0], dict) else None
                if content and isinstance(content, list) and len(content) > 0:
                    text = content[0].get("text")
                    if isinstance(text, str):
                        return text
        except Exception:
            pass
        
        # 最后备用：转换为字符串
        return str(resp_obj)
    
    return extract_response(resp)


def main():
    """主函数：演示完整调用流程"""
    
    # 示例提示词
    prompt = "请用中文解释什么是机器学习，并给出一个简单的例子。"
    
    print("=" * 60)
    print("GPT-5 Responses API 调用示例")
    print("=" * 60)
    print(f"提示词: {prompt}\n")
    
    try:
        # 调用LLM
        response = call_gpt5_with_responses_api(
            prompt=prompt,
            model="gpt-5",  # 或使用 "gpt-5-preview" 等其他gpt-5系列模型
            max_output_tokens=4000
        )
        
        print("响应:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
    except ValueError as e:
        print(f"错误: {e}")
        print("\n请确保已设置环境变量:")
        print('export OPENAI_API_KEY="sk-your_api_key_here"')
    except Exception as e:
        print(f"调用失败: {e}")


if __name__ == "__main__":
    main()
