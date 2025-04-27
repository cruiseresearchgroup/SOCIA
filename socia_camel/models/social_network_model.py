#!/usr/bin/env python3
# 社交网络模拟模型示例

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import os
import json
import random

class SocialNetworkSimulation:
    """
    社交网络模拟模型，模拟信息传播和意见形成
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模拟
        
        Args:
            config: 配置参数字典，包含以下键:
                - num_agents: 智能体数量
                - network_type: 网络类型 ('random', 'scale_free', 'small_world')
                - connection_param: 连接参数（根据网络类型不同而不同）
                - initial_opinion_distribution: 初始意见分布 ('uniform', 'normal', 'polarized')
                - influence_factor: 影响因子
                - stubborness_range: 固执度范围 [min, max]
                - seed: 随机种子
        """
        # 从配置中读取参数
        self.num_agents = config.get("num_agents", 100)
        self.network_type = config.get("network_type", "small_world")
        self.connection_param = config.get("connection_param", 0.1)
        self.initial_opinion_distribution = config.get("initial_opinion_distribution", "uniform")
        self.influence_factor = config.get("influence_factor", 0.2)
        self.stubborness_range = config.get("stubborness_range", [0.1, 0.5])
        
        # 设置随机种子
        seed = config.get("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 初始化网络
        self.network = self._create_network()
        
        # 初始化智能体属性
        self.opinions = self._initialize_opinions()
        self.stubbornness = np.random.uniform(
            self.stubborness_range[0], 
            self.stubborness_range[1], 
            self.num_agents
        )
        
        # 初始化记录
        self.metrics_history = []
        self.opinion_history = []
    
    def _create_network(self) -> nx.Graph:
        """创建社交网络"""
        if self.network_type == "random":
            # Erdos-Renyi随机图
            p = self.connection_param  # 连接概率
            G = nx.erdos_renyi_graph(self.num_agents, p)
        
        elif self.network_type == "scale_free":
            # Barabasi-Albert无标度网络
            m = int(self.connection_param)  # 每个新节点连接到的现有节点数
            G = nx.barabasi_albert_graph(self.num_agents, m)
        
        elif self.network_type == "small_world":
            # Watts-Strogatz小世界网络
            k = max(2, int(self.num_agents * 0.1))  # 每个节点的邻居数
            p = self.connection_param  # 重连概率
            G = nx.watts_strogatz_graph(self.num_agents, k, p)
        
        else:
            # 默认使用完全图
            G = nx.complete_graph(self.num_agents)
        
        return G
    
    def _initialize_opinions(self) -> np.ndarray:
        """初始化智能体意见"""
        if self.initial_opinion_distribution == "uniform":
            # 均匀分布在[-1, 1]之间
            opinions = np.random.uniform(-1, 1, self.num_agents)
        
        elif self.initial_opinion_distribution == "normal":
            # 正态分布，均值0，标准差0.5
            opinions = np.random.normal(0, 0.5, self.num_agents)
            # 截断到[-1, 1]范围
            opinions = np.clip(opinions, -1, 1)
        
        elif self.initial_opinion_distribution == "polarized":
            # 两极分布，一半接近-1，一半接近1
            opinions = np.zeros(self.num_agents)
            half = self.num_agents // 2
            
            opinions[:half] = np.random.normal(-0.8, 0.2, half)
            opinions[half:] = np.random.normal(0.8, 0.2, self.num_agents - half)
            
            # 截断到[-1, 1]范围
            opinions = np.clip(opinions, -1, 1)
        
        else:
            # 默认使用均匀分布
            opinions = np.random.uniform(-1, 1, self.num_agents)
        
        return opinions
    
    def step(self):
        """执行一个时间步的模拟"""
        # 保存当前意见
        self.opinion_history.append(self.opinions.copy())
        
        # 创建新意见数组
        new_opinions = self.opinions.copy()
        
        # 对每个智能体
        for agent in range(self.num_agents):
            # 获取邻居
            neighbors = list(self.network.neighbors(agent))
            if not neighbors:
                continue
            
            # 计算邻居意见的平均值
            neighbor_avg_opinion = np.mean([self.opinions[n] for n in neighbors])
            
            # 更新意见
            # 新意见 = 固执度 * 当前意见 + (1-固执度) * 影响因子 * 邻居平均意见
            new_opinions[agent] = (
                self.stubbornness[agent] * self.opinions[agent] + 
                (1 - self.stubbornness[agent]) * self.influence_factor * neighbor_avg_opinion
            )
        
        # 更新意见
        self.opinions = new_opinions
        
        # 记录指标
        self._record_metrics()
    
    def _record_metrics(self):
        """记录当前状态的指标"""
        metrics = {
            "time_step": len(self.metrics_history),
            "mean_opinion": float(np.mean(self.opinions)),
            "std_opinion": float(np.std(self.opinions)),
            "polarization_index": float(self._calculate_polarization()),
            "opinion_groups": self._identify_opinion_groups()
        }
        
        self.metrics_history.append(metrics)
    
    def _calculate_polarization(self) -> float:
        """计算意见极化指数"""
        # 简单版本：计算意见的标准差与极值的比值
        std = np.std(self.opinions)
        range_val = max(self.opinions) - min(self.opinions)
        if range_val == 0:
            return 0
        return std / range_val
    
    def _identify_opinion_groups(self) -> Dict[str, Any]:
        """识别意见群体"""
        # 简单分组：负面、中性、正面
        negative = np.sum(self.opinions < -0.3)
        neutral = np.sum((self.opinions >= -0.3) & (self.opinions <= 0.3))
        positive = np.sum(self.opinions > 0.3)
        
        return {
            "negative": int(negative),
            "neutral": int(neutral),
            "positive": int(positive)
        }
    
    def run(self, steps: int) -> Dict[str, Any]:
        """
        运行模拟
        
        Args:
            steps: 运行的时间步数
            
        Returns:
            模拟结果字典
        """
        # 记录初始状态
        self._record_metrics()
        
        for _ in range(steps):
            self.step()
            
            # 检查收敛（意见变化小于阈值）
            if len(self.opinion_history) >= 2:
                change = np.mean(np.abs(self.opinion_history[-1] - self.opinion_history[-2]))
                if change < 0.001:  # 收敛阈值
                    break
        
        return {
            "metrics_history": self.metrics_history,
            "final_state": self.metrics_history[-1],
            "opinion_history": [list(map(float, o)) for o in self.opinion_history]
        }
    
    def plot_results(self, output_dir: str = "./"):
        """
        绘制结果并保存
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制意见演化
        self._plot_opinion_evolution(output_dir)
        
        # 2. 绘制网络结构
        self._plot_network_structure(output_dir)
        
        # 3. 绘制意见分布
        self._plot_opinion_distribution(output_dir)
        
        # 4. 保存数据
        self._save_results(output_dir)
    
    def _plot_opinion_evolution(self, output_dir: str):
        """绘制意见演化"""
        plt.figure(figsize=(10, 6))
        
        # 从指标历史中提取数据
        time_steps = list(range(len(self.metrics_history)))
        mean_opinions = [m["mean_opinion"] for m in self.metrics_history]
        std_opinions = [m["std_opinion"] for m in self.metrics_history]
        
        # 绘制平均意见
        plt.plot(time_steps, mean_opinions, 'b-', label='平均意见')
        
        # 绘制标准差范围
        plt.fill_between(
            time_steps,
            [m - s for m, s in zip(mean_opinions, std_opinions)],
            [m + s for m, s in zip(mean_opinions, std_opinions)],
            color='blue', alpha=0.2, label='意见标准差'
        )
        
        # 绘制极化指数
        polarization = [m["polarization_index"] for m in self.metrics_history]
        plt.plot(time_steps, polarization, 'r-', label='极化指数')
        
        plt.title('意见演化')
        plt.xlabel('时间步')
        plt.ylabel('意见值')
        plt.legend()
        plt.grid(True)
        plt.ylim(-1.1, 1.1)
        
        plt.savefig(os.path.join(output_dir, "opinion_evolution.png"))
        plt.close()
    
    def _plot_network_structure(self, output_dir: str):
        """绘制网络结构"""
        plt.figure(figsize=(10, 10))
        
        # 使用最终意见作为节点颜色
        node_colors = plt.cm.RdBu((self.opinions + 1) / 2)  # 映射到[0,1]范围
        
        # 使用Spring布局
        pos = nx.spring_layout(self.network, seed=42)
        
        # 绘制网络
        nx.draw(
            self.network, pos,
            node_color=node_colors,
            node_size=100,
            edge_color='gray',
            alpha=0.8,
            with_labels=False
        )
        
        plt.title('社交网络结构')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(-1, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('意见值')
        
        plt.savefig(os.path.join(output_dir, "network_structure.png"))
        plt.close()
    
    def _plot_opinion_distribution(self, output_dir: str):
        """绘制意见分布"""
        plt.figure(figsize=(8, 6))
        
        # 绘制最终意见的直方图
        plt.hist(self.opinions, bins=20, color='skyblue', edgecolor='black')
        
        plt.title('最终意见分布')
        plt.xlabel('意见值')
        plt.ylabel('智能体数量')
        plt.grid(True, alpha=0.3)
        plt.xlim(-1.1, 1.1)
        
        plt.savefig(os.path.join(output_dir, "opinion_distribution.png"))
        plt.close()
    
    def _save_results(self, output_dir: str):
        """保存结果数据"""
        # 保存指标历史
        metrics_file = os.path.join(output_dir, "metrics_history.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # 保存网络结构
        network_file = os.path.join(output_dir, "network_structure.json")
        with open(network_file, 'w') as f:
            # 将网络转换为可序列化格式
            network_data = {
                "nodes": list(self.network.nodes()),
                "edges": list(self.network.edges()),
                "type": self.network_type,
                "parameters": {
                    "num_agents": self.num_agents,
                    "connection_param": self.connection_param
                }
            }
            json.dump(network_data, f, indent=2)
        
        # 保存最终状态
        final_status = {
            "network_type": self.network_type,
            "num_agents": self.num_agents,
            "final_mean_opinion": float(np.mean(self.opinions)),
            "final_std_opinion": float(np.std(self.opinions)),
            "final_polarization": float(self._calculate_polarization()),
            "opinion_groups": self._identify_opinion_groups(),
            "simulation_steps": len(self.metrics_history) - 1
        }
        
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(final_status, f, indent=2)

def create_social_network_simulation(config: Dict[str, Any]) -> SocialNetworkSimulation:
    """
    创建社交网络模拟实例
    
    Args:
        config: 配置参数字典
        
    Returns:
        SocialNetworkSimulation实例
    """
    return SocialNetworkSimulation(config)

def run_simulation(config: Dict[str, Any], steps: int, output_dir: str = "./output") -> Dict[str, Any]:
    """
    运行社交网络模拟并保存结果
    
    Args:
        config: 配置参数字典
        steps: 运行时间步数
        output_dir: 输出目录
        
    Returns:
        模拟结果
    """
    # 创建模拟实例
    simulation = create_social_network_simulation(config)
    
    # 运行
    results = simulation.run(steps)
    
    # 绘制并保存结果
    simulation.plot_results(output_dir)
    
    return results

if __name__ == "__main__":
    # 示例配置
    config = {
        "num_agents": 100,
        "network_type": "small_world",  # 'random', 'scale_free', 'small_world'
        "connection_param": 0.1,
        "initial_opinion_distribution": "polarized",  # 'uniform', 'normal', 'polarized'
        "influence_factor": 0.2,
        "stubborness_range": [0.1, 0.5],
        "seed": 42
    }
    
    # 运行模拟
    results = run_simulation(config, 50, "./output/social_network")
    
    # 打印最终状态
    final_state = results["final_state"]
    print(f"模拟完成，共运行 {final_state['time_step']} 个时间步")
    print(f"最终状态:")
    print(f"  平均意见: {final_state['mean_opinion']:.4f}")
    print(f"  意见标准差: {final_state['std_opinion']:.4f}")
    print(f"  极化指数: {final_state['polarization_index']:.4f}")
    print(f"  意见群体: {final_state['opinion_groups']}") 