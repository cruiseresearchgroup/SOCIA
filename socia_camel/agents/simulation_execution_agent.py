import os
import json
import logging
import tempfile
import subprocess
from typing import List, Dict, Any, Optional

from ..agents.base_agent import BaseAgent, Action
from ..utils.llm_utils import LLMClient

logger = logging.getLogger("SOCIA-CAMEL")

class SimulationExecutionAgent(BaseAgent):
    """
    模拟执行智能体，负责执行模拟代码并收集结果
    """
    def __init__(self, config_path: str = "../config.yaml"):
        # 定义可执行的动作
        actions = [
            Action(name="准备环境", description="准备模拟执行环境"),
            Action(name="执行模拟", description="执行模拟代码"),
            Action(name="收集结果", description="收集模拟结果"),
            Action(name="分析结果", description="分析模拟结果"),
            Action(name="思考", description="思考给定的问题")
        ]
        
        # 初始化基类
        super().__init__(
            name="模拟执行智能体",
            role="模拟执行专家",
            goal="执行模拟代码并收集结果",
            actions=actions,
            config_path=config_path
        )
        
        # 创建LLM客户端
        self.llm_client = LLMClient(config_path)
        
        # 临时文件和目录
        self.temp_dir = None
        self.code_file = None
        self.output_dir = None
    
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
    
    def _action_准备环境(self, code: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        准备模拟执行环境
        
        Args:
            code: 待执行的代码
            output_dir: 输出目录，如果不指定则创建临时目录
            
        Returns:
            环境准备结果
        """
        try:
            # 创建临时目录或使用指定的输出目录
            if output_dir:
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
            else:
                self.temp_dir = tempfile.TemporaryDirectory()
                self.output_dir = self.temp_dir.name
            
            # 创建代码文件
            self.code_file = os.path.join(self.output_dir, "simulation.py")
            with open(self.code_file, "w") as f:
                f.write(code)
            
            # 记录推理过程
            reasoning = f"我准备了模拟执行环境，将代码保存到文件：{self.code_file}。"
            reasoning += f"\n输出目录设置为：{self.output_dir}。"
            
            self.memory.add_reasoning("准备环境", reasoning)
            
            return {
                "success": True,
                "code_file": self.code_file,
                "output_dir": self.output_dir
            }
        except Exception as e:
            logger.error(f"准备环境失败: {e}")
            return {
                "success": False,
                "error": f"准备环境失败: {e}"
            }
    
    def _action_执行模拟(self, use_docker: bool = False) -> Dict[str, Any]:
        """
        执行模拟代码
        
        Args:
            use_docker: 是否使用Docker容器执行
            
        Returns:
            执行结果
        """
        if not self.code_file:
            return {
                "success": False,
                "error": "代码文件未准备，请先调用准备环境动作"
            }
        
        try:
            # 根据配置决定是否使用Docker执行
            if use_docker:
                return self._execute_in_docker()
            else:
                return self._execute_locally()
        except Exception as e:
            logger.error(f"执行模拟失败: {e}")
            return {
                "success": False,
                "error": f"执行模拟失败: {e}"
            }
    
    def _execute_locally(self) -> Dict[str, Any]:
        """本地执行模拟"""
        try:
            # 设置环境变量
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 获取相对路径
            code_file_name = os.path.basename(self.code_file)
            
            # 使用虚拟环境中的Python解释器
            python_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "socia_env", "bin", "python")
            
            # 执行Python脚本
            result = subprocess.run(
                [python_path, code_file_name],
                cwd=self.output_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # 记录详细的执行信息
            logger.info(f"执行命令: {python_path} {code_file_name}")
            logger.info(f"工作目录: {self.output_dir}")
            logger.info(f"环境变量: PYTHONPATH={env['PYTHONPATH']}")
            logger.info(f"返回码: {result.returncode}")
            if result.stdout:
                logger.info(f"标准输出: {result.stdout[:200]}...")  # 只记录前200个字符
            if result.stderr:
                logger.warning(f"标准错误: {result.stderr}")  # 改为warning级别
            
            # 记录推理过程
            reasoning = f"我在本地环境中执行了模拟代码。"
            reasoning += f"\n执行结果：{'成功' if result.returncode == 0 else '失败'}"
            if result.returncode != 0:
                reasoning += f"\n错误信息：{result.stderr}"
            
            self.memory.add_reasoning("执行模拟", reasoning)
            
            # 如果返回码为0，则认为执行成功
            success = result.returncode == 0
            
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            logger.error(f"本地执行失败: {e}")
            return {
                "success": False,
                "error": f"本地执行失败: {e}"
            }
    
    def _execute_in_docker(self) -> Dict[str, Any]:
        """在Docker容器中执行模拟"""
        try:
            # 检查Docker是否可用
            docker_check = subprocess.run(
                ["docker", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if docker_check.returncode != 0:
                logger.warning("Docker不可用，回退到本地执行")
                return self._execute_locally()
            
            # 创建Dockerfile
            dockerfile_path = os.path.join(self.output_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write("""
FROM python:3.9-slim

WORKDIR /app

# 安装常用科学计算库
RUN pip install --no-cache-dir numpy pandas matplotlib seaborn scipy networkx

# 复制模拟代码
COPY simulation.py /app/

# 创建输出目录
RUN mkdir -p /app/output

# 执行模拟
CMD ["python", "simulation.py"]
                """)
            
            # 构建Docker镜像
            build_result = subprocess.run(
                ["docker", "build", "-t", "socia-simulation", "."],
                cwd=self.output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if build_result.returncode != 0:
                logger.error(f"Docker镜像构建失败: {build_result.stderr}")
                return {
                    "success": False,
                    "error": f"Docker镜像构建失败: {build_result.stderr}"
                }
            
            # 运行Docker容器
            run_result = subprocess.run(
                ["docker", "run", "--rm", "-v", f"{self.output_dir}/output:/app/output", "socia-simulation"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # 记录推理过程
            reasoning = f"我在Docker容器中执行了模拟代码。"
            reasoning += f"\n执行结果：{'成功' if run_result.returncode == 0 else '失败'}"
            if run_result.returncode != 0:
                reasoning += f"\n错误信息：{run_result.stderr}"
            
            self.memory.add_reasoning("执行模拟", reasoning)
            
            return {
                "success": run_result.returncode == 0,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode
            }
        except Exception as e:
            logger.error(f"Docker执行失败: {e}")
            return {
                "success": False,
                "error": f"Docker执行失败: {e}"
            }
    
    def _action_收集结果(self) -> Dict[str, Any]:
        """
        收集模拟结果
        
        Returns:
            收集的结果
        """
        if not self.output_dir:
            return {
                "success": False,
                "error": "输出目录未设置，请先执行模拟"
            }
        
        try:
            # 查找输出文件
            output_files = []
            for root, _, files in os.walk(self.output_dir):
                for file in files:
                    if file.endswith((".png", ".jpg", ".csv", ".json", ".txt")) and file != "simulation.py":
                        file_path = os.path.join(root, file)
                        output_files.append({
                            "name": file,
                            "path": file_path,
                            "type": file.split(".")[-1]
                        })
            
            # 读取文本类型的结果文件
            text_results = {}
            for file_info in output_files:
                if file_info["type"] in ["txt", "csv", "json"]:
                    try:
                        with open(file_info["path"], "r") as f:
                            content = f.read()
                        text_results[file_info["name"]] = content
                    except Exception as e:
                        logger.warning(f"读取文件{file_info['path']}失败: {e}")
            
            # 记录推理过程
            reasoning = f"我收集了模拟结果，找到{len(output_files)}个输出文件。"
            reasoning += f"\n包括{len([f for f in output_files if f['type'] in ['png', 'jpg']])}个图像文件、"
            reasoning += f"{len([f for f in output_files if f['type'] == 'csv'])}个CSV文件、"
            reasoning += f"{len([f for f in output_files if f['type'] == 'json'])}个JSON文件和"
            reasoning += f"{len([f for f in output_files if f['type'] == 'txt'])}个文本文件。"
            
            self.memory.add_reasoning("收集结果", reasoning)
            
            return {
                "success": True,
                "output_files": output_files,
                "text_results": text_results
            }
        except Exception as e:
            logger.error(f"收集结果失败: {e}")
            return {
                "success": False,
                "error": f"收集结果失败: {e}"
            }
    
    def _action_分析结果(self, collection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析模拟结果
        
        Args:
            collection_result: 收集的结果
            
        Returns:
            分析结果
        """
        if not collection_result.get("success", False):
            return {
                "success": False,
                "error": "没有可用的结果进行分析"
            }
        
        # 构建提示词
        prompt = f"""作为模拟结果分析专家，请分析以下模拟执行结果。

输出文件列表:
{json.dumps([f['name'] for f in collection_result.get('output_files', [])], ensure_ascii=False, indent=2)}

文本结果:
{json.dumps(collection_result.get('text_results', {}), ensure_ascii=False, indent=2)}

请提供详细分析，包括：
1. 模拟是否成功执行
2. 关键结果指标
3. 结果的意义和解释
4. 是否有意外或异常现象
5. 建议的下一步行动

请以JSON格式返回分析结果：
{{
    "执行状态": "成功/失败",
    "关键指标": [
        {{
            "指标名称": "指标名称",
            "指标值": "指标值",
            "解释": "指标解释"
        }},
        ...
    ],
    "总体分析": "整体结果分析",
    "异常现象": ["异常1", "异常2", ...],
    "建议行动": ["建议1", "建议2", ...],
    "结论": "分析结论"
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
            # 添加日志记录原始LLM响应内容
            logger.info(f"LLM原始响应内容: {response}")
            
            # 处理可能的markdown代码块格式
            json_str = response
            if response.strip().startswith("```") and "```" in response:
                # 提取代码块中的内容
                lines = response.strip().split("\n")
                json_str = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
                # 如果第一行是```json，移除它
                if json_str.startswith("json"):
                    json_str = json_str[4:].strip()
                    
            logger.info(f"处理后的JSON字符串前100个字符: {json_str[:100]}")
            result = json.loads(json_str)
            
            # 记录推理过程
            reasoning = f"我分析了模拟结果，执行状态为{result.get('执行状态', '未知')}。"
            reasoning += f"\n识别出{len(result.get('关键指标', []))}个关键指标。"
            reasoning += f"\n发现{len(result.get('异常现象', []))}个异常现象。"
            reasoning += f"\n提出{len(result.get('建议行动', []))}条建议行动。"
            reasoning += f"\n总体结论：{result.get('结论', '未给出')}。"
            
            self.memory.add_reasoning("分析结果", reasoning, str(result))
            
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
        prompt = f"""请仔细思考以下与模拟执行相关的问题，并给出你的分析：

问题: "{question}"

思考过程中，请考虑：
1. 这个问题的核心是什么？
2. 有哪些相关的执行环境和工具？
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
    
    def execute_simulation(self, code: str, output_dir: Optional[str] = None, use_docker: bool = False) -> Dict[str, Any]:
        """
        执行完整的模拟流程，包括准备环境、执行模拟、收集结果和分析结果
        
        Args:
            code: 待执行的代码
            output_dir: 输出目录，如果不指定则创建临时目录
            use_docker: 是否使用Docker容器执行
            
        Returns:
            模拟执行结果
        """
        # 1. 准备环境
        env_result = self.execute_action("准备环境", code=code, output_dir=output_dir)
        if not env_result.get("success", False):
            return env_result
        
        # 2. 执行模拟
        exec_result = self.execute_action("执行模拟", use_docker=use_docker)
        if not exec_result.get("success", False):
            return exec_result
        
        # 3. 收集结果
        collection_result = self.execute_action("收集结果")
        if not collection_result.get("success", False):
            return collection_result
        
        # 4. 分析结果
        analysis_result = self.execute_action("分析结果", collection_result=collection_result)
        if not analysis_result.get("success", False):
            return analysis_result
        
        # 5. 组合所有结果
        final_result = {
            "execution": exec_result,
            "collection": collection_result,
            "analysis": analysis_result.get("result", {})
        }
        
        return {
            "success": True,
            "result": final_result
        }
    
    def cleanup(self):
        """清理临时文件和目录"""
        if self.temp_dir:
            self.temp_dir.cleanup()
            self.temp_dir = None
            self.code_file = None
            self.output_dir = None 