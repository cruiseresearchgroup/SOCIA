#!/usr/bin/env python3
# 流行病模拟模型示例

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import os
import json

class EpidemicSimulation:
    """
    简单的流行病模拟模型，基于SIR模型
    
    SIR模型将人口分为三类：
    - S: 易感者 (Susceptible)
    - I: 感染者 (Infected)
    - R: 康复者 (Recovered)
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟
        
        Args:
            config: 配置参数字典，包含以下键:
                - population_size: 总人口数
                - initial_infected: 初始感染人数
                - transmission_rate: 传播率
                - recovery_rate: 康复率
                - contact_radius: 接触半径
                - seed: 随机种子
        """
        # 从配置中读取参数
        self.population_size = config.get("population_size", 1000)
        self.initial_infected = config.get("initial_infected", 5)
        self.transmission_rate = config.get("transmission_rate", 0.3)
        self.recovery_rate = config.get("recovery_rate", 0.1)
        self.contact_radius = config.get("contact_radius", 0.02)
        
        # 设置随机种子
        seed = config.get("seed", None)
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化人口
        self.positions = np.random.rand(self.population_size, 2)  # 随机位置
        
        # 初始化状态
        self.status = np.zeros(self.population_size, dtype=int)  # 0: S, 1: I, 2: R
        
        # 初始感染者
        self.metrics_history = []
    
    def initialize(self):
        """初始化模拟状态"""
        # 随机选择初始感染者
        initial_infected_idx = np.random.choice(
            self.population_size, 
            min(self.initial_infected, self.population_size), 
            replace=False
        )
        self.status[initial_infected_idx] = 1  # 设置为感染状态
        
        # 记录初始状态
        self._record_metrics(0)
    
    def step(self):
        """执行一个时间步的模拟"""
        # 1. 感染过程：检查每个易感者
        susceptible_idx = np.where(self.status == 0)[0]
        infected_idx = np.where(self.status == 1)[0]
        
        new_infections = []
        
        for s_idx in susceptible_idx:
            # 计算该易感者与所有感染者的距离
            s_pos = self.positions[s_idx]
            
            for i_idx in infected_idx:
                i_pos = self.positions[i_idx]
                distance = np.sqrt(np.sum((s_pos - i_pos) ** 2))
                
                # 如果在接触半径内，有概率被感染
                if distance < self.contact_radius and np.random.random() < self.transmission_rate:
                    new_infections.append(s_idx)
                    break
        
        # 更新新感染者的状态
        self.status[new_infections] = 1
        
        # 2. 康复过程：检查每个感染者
        infected_idx = np.where(self.status == 1)[0]
        
        for i_idx in infected_idx:
            # 有概率康复
            if np.random.random() < self.recovery_rate:
                self.status[i_idx] = 2  # 设置为康复状态
        
        # 3. 更新位置：随机移动
        self.positions += (np.random.rand(self.population_size, 2) - 0.5) * 0.01
        
        # 确保位置在[0,1]范围内
        self.positions = np.clip(self.positions, 0, 1)
    
    def run(self, steps: int) -> Dict[str, Any]:
        """
        运行模拟
        
        Args:
            steps: 运行的时间步数
            
        Returns:
            模拟结果字典
        """
        for step in range(1, steps + 1):
            self.step()
            self._record_metrics(step)
            
            # 检查是否没有感染者了
            if np.sum(self.status == 1) == 0:
                break
        
        return {
            "metrics_history": self.metrics_history,
            "final_state": self.metrics_history[-1]
        }
    
    def _record_metrics(self, time_step: int):
        """记录当前状态的指标"""
        metrics = {
            "time_step": time_step,
            "susceptible_count": np.sum(self.status == 0),
            "infected_count": np.sum(self.status == 1),
            "recovered_count": np.sum(self.status == 2),
            "total_population": self.population_size
        }
        
        self.metrics_history.append(metrics)
    
    def plot_results(self, output_dir: str = "./"):
        """
        绘制结果并保存
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制时间序列
        self._plot_time_series(output_dir)
        
        # 2. 绘制最终状态空间分布
        self._plot_spatial_distribution(output_dir)
        
        # 3. 保存数据
        self._save_results(output_dir)
    
    def _plot_time_series(self, output_dir: str):
        """绘制SIR时间序列"""
        time_steps = [m["time_step"] for m in self.metrics_history]
        susceptible = [m["susceptible_count"] for m in self.metrics_history]
        infected = [m["infected_count"] for m in self.metrics_history]
        recovered = [m["recovered_count"] for m in self.metrics_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, susceptible, 'b-', label='易感者 (S)')
        plt.plot(time_steps, infected, 'r-', label='感染者 (I)')
        plt.plot(time_steps, recovered, 'g-', label='康复者 (R)')
        
        plt.title('SIR流行病模型模拟结果')
        plt.xlabel('时间步')
        plt.ylabel('人口数量')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(output_dir, "sir_time_series.png"))
        plt.close()
    
    def _plot_spatial_distribution(self, output_dir: str):
        """绘制空间分布"""
        plt.figure(figsize=(8, 8))
        
        # 易感者
        susceptible_idx = np.where(self.status == 0)[0]
        plt.scatter(
            self.positions[susceptible_idx, 0],
            self.positions[susceptible_idx, 1],
            c='blue', alpha=0.6, label='易感者 (S)'
        )
        
        # 感染者
        infected_idx = np.where(self.status == 1)[0]
        plt.scatter(
            self.positions[infected_idx, 0],
            self.positions[infected_idx, 1],
            c='red', alpha=0.6, label='感染者 (I)'
        )
        
        # 康复者
        recovered_idx = np.where(self.status == 2)[0]
        plt.scatter(
            self.positions[recovered_idx, 0],
            self.positions[recovered_idx, 1],
            c='green', alpha=0.6, label='康复者 (R)'
        )
        
        plt.title('人口空间分布')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(output_dir, "spatial_distribution.png"))
        plt.close()
    
    def _save_results(self, output_dir: str):
        """保存结果数据"""
        # 保存指标历史
        metrics_file = os.path.join(output_dir, "metrics_history.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # 保存最终状态
        final_status = {
            "susceptible": int(np.sum(self.status == 0)),
            "infected": int(np.sum(self.status == 1)),
            "recovered": int(np.sum(self.status == 2)),
            "total_population": int(self.population_size),
            "transmission_rate": float(self.transmission_rate),
            "recovery_rate": float(self.recovery_rate),
            "contact_radius": float(self.contact_radius),
            "simulation_steps": len(self.metrics_history) - 1
        }
        
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(final_status, f, indent=2)

def create_epidemic_simulation(config: Dict[str, Any]) -> EpidemicSimulation:
    """
    创建流行病模拟实例
    
    Args:
        config: 配置参数字典
        
    Returns:
        EpidemicSimulation实例
    """
    return EpidemicSimulation(config)

def run_simulation(config: Dict[str, Any], steps: int, output_dir: str = "./output") -> Dict[str, Any]:
    """
    运行流行病模拟并保存结果
    
    Args:
        config: 配置参数字典
        steps: 运行时间步数
        output_dir: 输出目录
        
    Returns:
        模拟结果
    """
    # 创建模拟实例
    simulation = create_epidemic_simulation(config)
    
    # 初始化
    simulation.initialize()
    
    # 运行
    results = simulation.run(steps)
    
    # 绘制并保存结果
    simulation.plot_results(output_dir)
    
    return results

if __name__ == "__main__":
    # 示例配置
    config = {
        "population_size": 1000,
        "initial_infected": 5,
        "transmission_rate": 0.3,
        "recovery_rate": 0.1,
        "contact_radius": 0.02,
        "seed": 42
    }
    
    # 运行模拟
    results = run_simulation(config, 100, "./output")
    
    # 打印最终状态
    final_state = results["final_state"]
    print(f"模拟完成，共运行 {final_state['time_step']} 个时间步")
    print(f"最终状态:")
    print(f"  易感者: {final_state['susceptible_count']}")
    print(f"  感染者: {final_state['infected_count']}")
    print(f"  康复者: {final_state['recovered_count']}") 