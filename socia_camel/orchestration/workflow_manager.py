import os
import json
import logging
import yaml
import sys
from typing import Dict, Any, Optional, List

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用绝对导入
from socia_camel.agents.task_understanding_agent import TaskUnderstandingAgent
from socia_camel.agents.model_planning_agent import ModelPlanningAgent
from socia_camel.agents.code_generation_agent import CodeGenerationAgent
from socia_camel.agents.simulation_execution_agent import SimulationExecutionAgent

logger = logging.getLogger("SOCIA-CAMEL")

class WorkflowManager:
    """
    工作流管理器，负责协调各个Agent的工作
    """
    def __init__(self, config_path: str = "../config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = "./output"
        self.agents = self._initialize_agents()
        self.workflow_history = []
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """初始化各个Agent"""
        agents = {}
        
        # 创建任务理解Agent
        agents["task_understanding"] = TaskUnderstandingAgent(config_path=self.config_path)
        
        # 创建模型规划Agent
        agents["model_planning"] = ModelPlanningAgent(config_path=self.config_path)
        
        # 创建代码生成Agent
        agents["code_generation"] = CodeGenerationAgent(config_path=self.config_path)
        
        # 创建模拟执行Agent
        agents["simulation_execution"] = SimulationExecutionAgent(config_path=self.config_path)
        
        return agents
    
    def set_output_dir(self, output_dir: str) -> None:
        """设置输出目录"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _save_result(self, stage: str, result: Dict[str, Any]) -> None:
        """保存阶段结果"""
        if not self.config.get("workflow", {}).get("save_intermediate_results", True):
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        result_file = os.path.join(self.output_dir, f"{stage}_result.json")
        
        try:
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"保存{stage}阶段结果到：{result_file}")
        except Exception as e:
            logger.error(f"保存{stage}阶段结果失败: {e}")
    
    def _record_workflow_step(self, stage: str, success: bool, message: str, result: Optional[Dict[str, Any]] = None) -> None:
        """记录工作流步骤"""
        step = {
            "stage": stage,
            "success": success,
            "message": message,
            "timestamp": self._get_timestamp(),
            "result": result
        }
        
        self.workflow_history.append(step)
        
        logger.info(f"工作流步骤：{stage} - {'成功' if success else '失败'}: {message}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def run_workflow(self, task_description: str) -> Dict[str, Any]:
        """
        运行完整的工作流程
        
        Args:
            task_description: 任务描述
            
        Returns:
            工作流执行结果
        """
        # 清空历史记录
        self.workflow_history = []
        
        try:
            # 1. 任务理解
            logger.info("开始任务理解阶段")
            task_agent = self.agents["task_understanding"]
            task_result = task_agent.understand_task(task_description)
            
            if not task_result.get("success", False):
                error_message = f"任务理解失败: {task_result.get('error', '未知错误')}"
                logger.error(error_message)
                self._record_workflow_step("task_understanding", False, error_message)
                return {"success": False, "error": error_message}
            
            task_analysis = task_result["result"]
            self._save_result("task_understanding", task_analysis)
            self._record_workflow_step("task_understanding", True, "任务理解成功", task_analysis)
            
            # 2. 模型规划
            logger.info("开始模型规划阶段")
            model_agent = self.agents["model_planning"]
            model_result = model_agent.design_model(task_analysis)
            
            if not model_result.get("success", False):
                error_message = f"模型规划失败: {model_result.get('error', '未知错误')}"
                logger.error(error_message)
                self._record_workflow_step("model_planning", False, error_message)
                return {"success": False, "error": error_message}
            
            model_plan = model_result["result"]
            self._save_result("model_planning", model_plan)
            self._record_workflow_step("model_planning", True, "模型规划成功", model_plan)
            
            # 3. 代码生成
            logger.info("开始代码生成阶段")
            code_agent = self.agents["code_generation"]
            code_result = code_agent.generate_simulation_code(model_plan, task_analysis)
            
            if not code_result.get("success", False):
                error_message = f"代码生成失败: {code_result.get('error', '未知错误')}"
                logger.error(error_message)
                self._record_workflow_step("code_generation", False, error_message)
                return {"success": False, "error": error_message}
            
            simulation_code = code_result["code"]
            code_analysis = code_result.get("analysis")
            
            # 保存生成的代码
            code_file = os.path.join(self.output_dir, "simulation.py")
            try:
                with open(code_file, 'w') as f:
                    f.write(simulation_code)
                logger.info(f"代码已保存到：{code_file}")
            except Exception as e:
                logger.error(f"保存代码失败: {e}")
            
            self._save_result("code_generation", {"analysis": code_analysis})
            self._record_workflow_step("code_generation", True, "代码生成成功", {"analysis": code_analysis})
            
            # 4. 模拟执行
            logger.info("开始模拟执行阶段")
            execution_agent = self.agents["simulation_execution"]
            
            # 检查是否使用Docker
            use_docker = self.config.get("agents", {}).get("simulation_execution", {}).get("sandbox", "local") == "docker"
            
            execution_result = execution_agent.execute_simulation(
                code=simulation_code,
                output_dir=os.path.join(self.output_dir, "simulation_output"),
                use_docker=use_docker
            )
            
            if not execution_result.get("success", False):
                error_message = f"模拟执行失败: {execution_result.get('error', '未知错误')}"
                logger.error(error_message)
                self._record_workflow_step("simulation_execution", False, error_message)
                return {"success": False, "error": error_message}
            
            simulation_result = execution_result["result"]
            self._save_result("simulation_execution", simulation_result)
            self._record_workflow_step("simulation_execution", True, "模拟执行成功", simulation_result)
            
            # 5. 清理临时资源
            execution_agent.cleanup()
            
            # 6. 返回最终结果
            final_result = {
                "success": True,
                "task_analysis": task_analysis,
                "model_plan": model_plan,
                "code_analysis": code_analysis,
                "simulation_result": simulation_result,
                "workflow_history": self.workflow_history
            }
            
            # 保存最终结果
            final_result_file = os.path.join(self.output_dir, "final_result.json")
            try:
                with open(final_result_file, 'w') as f:
                    json.dump(final_result, f, indent=2, ensure_ascii=False)
                logger.info(f"最终结果已保存到：{final_result_file}")
            except Exception as e:
                logger.error(f"保存最终结果失败: {e}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"工作流执行出错: {e}")
            self._record_workflow_step("workflow", False, f"工作流执行出错: {e}")
            return {"success": False, "error": str(e), "workflow_history": self.workflow_history}
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """获取工作流执行历史"""
        return self.workflow_history 