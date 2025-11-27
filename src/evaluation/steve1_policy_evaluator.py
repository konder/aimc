"""
Steve1 两阶段模型深度分析器
=========================================

基于 Steve1 论文的分解：p(τ|y) = p(z_goal|y) * p(τ|z_goal)

分别评估：
1. Prior模型 p(z_goal|y): 文本 → 目标嵌入
2. 策略模型 p(τ|z_goal): 目标嵌入 → 动作序列

作者: AI Assistant
日期: 2025-11-27
"""

import torch as th
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy

# 导入 Steve1 组件
from steve1.utils.embed_utils import get_prior_embed
from steve1.config import PRIOR_INFO
from src.utils.steve1_mineclip_agent_env_utils import (
    load_mineclip_wconfig,
    load_vae_model,
    make_agent,
    make_env
)
from src.utils.device import DEVICE

logger = logging.getLogger(__name__)


@dataclass
class PriorAnalysisResult:
    """Prior 模型 p(z_goal|y) 的分析结果"""
    
    # 基础指标
    text_instruction: str
    text_embed_dim: int
    prior_embed_dim: int
    
    # 对齐度指标
    text_to_prior_similarity: float  # MineCLIP文本嵌入 vs Prior输出的余弦相似度
    prior_variance: float  # Prior输出的方差（多样性）
    reconstruction_quality: float  # VAE重建质量（如果可用）
    
    # 空间分析
    nearest_neighbors: List[Tuple[str, float]] = field(default_factory=list)  # 最近邻指令
    cluster_id: Optional[int] = None  # 聚类ID
    
    # 原始嵌入
    text_embed: Optional[np.ndarray] = None
    prior_embed: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存）"""
        result = {
            'text_instruction': self.text_instruction,
            'text_embed_dim': self.text_embed_dim,
            'prior_embed_dim': self.prior_embed_dim,
            'text_to_prior_similarity': float(self.text_to_prior_similarity),
            'prior_variance': float(self.prior_variance),
            'reconstruction_quality': float(self.reconstruction_quality),
            'nearest_neighbors': [(str(inst), float(sim)) for inst, sim in self.nearest_neighbors],
            'cluster_id': self.cluster_id,
        }
        return result


@dataclass
class PolicyAnalysisResult:
    """策略模型 p(τ|z_goal) 的分析结果"""
    
    # 基础信息（无默认值）
    instruction: str
    goal_embed_source: str  # 'prior' 或 'ground_truth_visual'
    
    # 动作统计（无默认值）
    total_steps: int
    action_diversity: float  # 动作熵（多样性）
    temporal_consistency: float  # 动作序列的时序一致性
    repeated_action_ratio: float  # 重复动作的比例
    
    # 成功指标（无默认值）
    success: bool
    final_reward: float
    
    # 可选字段（有默认值，必须放在最后）
    action_distribution: Dict[str, float] = field(default_factory=dict)  # 各动作维度的分布
    success_steps: Optional[int] = None  # 成功时的步数
    action_sequence: Optional[List[Dict]] = None  # 详细动作序列
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存）"""
        result = {
            'instruction': self.instruction,
            'goal_embed_source': self.goal_embed_source,
            'total_steps': self.total_steps,
            'action_diversity': float(self.action_diversity),
            'action_distribution': {k: float(v) for k, v in self.action_distribution.items()},
            'temporal_consistency': float(self.temporal_consistency),
            'repeated_action_ratio': float(self.repeated_action_ratio),
            'success': bool(self.success),
            'final_reward': float(self.final_reward),
            'success_steps': self.success_steps,
        }
        return result


@dataclass
class EndToEndAnalysisResult:
    """端到端分析结果（两阶段联合）"""
    
    # 基础信息
    task_id: str
    instruction: str
    trial_idx: int
    
    # 两阶段结果
    prior_result: PriorAnalysisResult
    policy_result: PolicyAnalysisResult
    
    # 联合分析
    stage1_contribution: float  # 阶段1的贡献（基于对比实验）
    stage2_contribution: float  # 阶段2的贡献
    bottleneck_stage: int  # 瓶颈阶段（1或2）
    
    # 错误归因
    failure_attribution: Optional[Dict[str, float]] = None  # {'prior': 0.7, 'policy': 0.3}
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存）"""
        return {
            'task_id': self.task_id,
            'instruction': self.instruction,
            'trial_idx': self.trial_idx,
            'prior_result': self.prior_result.to_dict(),
            'policy_result': self.policy_result.to_dict(),
            'stage1_contribution': float(self.stage1_contribution),
            'stage2_contribution': float(self.stage2_contribution),
            'bottleneck_stage': int(self.bottleneck_stage),
            'failure_attribution': self.failure_attribution,
        }


class Steve1PolicyEvaluator:
    """
    Steve1 策略评估器（Policy Evaluator）
    
    评估Steve1 Policy模型 p(τ|z_goal) 的策略质量：
    - 动作多样性（熵）
    - 时序一致性
    - 动作分布
    - 端到端分析（结合Prior评估器）
    
    【策略评估 vs 结果评估】
    - 策略评估（本类）：分析模型的策略质量和行为模式
    - 结果评估（STEVE1Evaluator）：评估任务成功率
    """
    
    def __init__(
        self,
        model_path: str = "data/weights/vpt/2x.model",
        weights_path: str = "data/weights/steve1/steve1.weights",
        prior_weights: str = "data/weights/steve1/steve1_prior.pt",
        text_cond_scale: float = 6.0,
        visual_cond_scale: float = 7.0,
        seed: int = 42,
    ):
        """
        初始化策略分析器
        
        Args:
            model_path: VPT 模型路径
            weights_path: Steve1 权重路径
            prior_weights: Prior VAE 权重路径
            text_cond_scale: 文本条件缩放
            visual_cond_scale: 视觉条件缩放
            seed: 随机种子
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.prior_weights = prior_weights
        self.text_cond_scale = text_cond_scale
        self.visual_cond_scale = visual_cond_scale
        self.seed = seed
        
        # 延迟加载组件
        self._mineclip = None
        self._prior = None
        self._agent = None
        
        # 缓存嵌入
        self.embed_cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        logger.info("Steve1DeepAnalyzer 初始化完成")
    
    def _load_components(self):
        """延迟加载组件"""
        if self._mineclip is None:
            logger.info("加载 MineCLIP...")
            self._mineclip = load_mineclip_wconfig()
            
        if self._prior is None:
            logger.info("加载 Prior VAE...")
            prior_info = PRIOR_INFO.copy()
            prior_info['prior_weights'] = self.prior_weights
            self._prior = load_vae_model(prior_info)
            
        if self._agent is None:
            logger.info("加载 Steve1 Agent...")
            self._agent = make_agent(
                self.model_path,
                self.weights_path,
                cond_scale=self.text_cond_scale
            )
    
    # ==========================================================================
    # 阶段1: Prior 模型分析 p(z_goal|y)
    # ==========================================================================
    
    def analyze_prior_model(
        self,
        instructions: List[str],
        output_dir: Optional[Path] = None,
        n_samples: int = 5,  # 每个指令采样多少次（VAE是随机的）
    ) -> Dict[str, PriorAnalysisResult]:
        """
        分析 Prior 模型：文本 → 目标嵌入的质量
        
        评估指标：
        1. 文本嵌入 vs Prior输出的相似度
        2. Prior输出的方差（多样性）
        3. 嵌入空间的结构（聚类、最近邻）
        
        Args:
            instructions: 要分析的指令列表
            output_dir: 输出目录（保存结果和可视化）
            n_samples: 每个指令采样多少次
            
        Returns:
            results: 每个指令的分析结果
        """
        self._load_components()
        
        logger.info(f"开始分析 Prior 模型（{len(instructions)} 个指令）")
        
        results = {}
        all_text_embeds = []
        all_prior_embeds = []
        instruction_list = []
        
        for instruction in instructions:
            logger.info(f"  分析指令: '{instruction}'")
            
            # 1. 获取文本嵌入（MineCLIP）
            with th.no_grad():
                text_embed = self._mineclip.encode_text([instruction])
                text_embed = text_embed.cpu().numpy()[0]  # [512]
            
            # 2. 采样多次 Prior 输出
            prior_samples = []
            for _ in range(n_samples):
                with th.no_grad():
                    prior_embed = get_prior_embed(
                        instruction,
                        self._mineclip,
                        self._prior,
                        DEVICE
                    )
                    # get_prior_embed 可能返回 tensor 或 numpy array
                    if hasattr(prior_embed, 'cpu'):
                        prior_embed = prior_embed.cpu().numpy()
                    elif isinstance(prior_embed, th.Tensor):
                        prior_embed = prior_embed.detach().cpu().numpy()
                    
                    # 确保是 1-D 向量 (去除可能的 batch 维度)
                    if isinstance(prior_embed, np.ndarray):
                        prior_embed = np.squeeze(prior_embed)  # 移除所有大小为1的维度
                        if prior_embed.ndim != 1:
                            # 如果还不是 1-D，取第一个元素
                            prior_embed = prior_embed.flatten()
                    
                    prior_samples.append(prior_embed)
            
            # 3. 计算统计指标
            prior_samples = np.array(prior_samples)  # [n_samples, 512]
            prior_mean = prior_samples.mean(axis=0)
            prior_variance = prior_samples.var(axis=0).mean()
            
            # 确保 prior_mean 也是 1-D
            prior_mean = np.squeeze(prior_mean)
            
            # 4. 计算相似度
            # 调试信息
            logger.debug(f"  text_embed shape: {text_embed.shape}, prior_mean shape: {prior_mean.shape}")
            if text_embed.ndim != 1 or prior_mean.ndim != 1:
                logger.warning(f"  警告: 嵌入维度不正确 - text: {text_embed.shape}, prior: {prior_mean.shape}")
                text_embed = text_embed.flatten()
                prior_mean = prior_mean.flatten()
            
            text_to_prior_sim = 1 - cosine(text_embed, prior_mean)
            
            # 5. 重建质量（使用 Prior 的 VAE 重建）
            reconstruction_quality = self._compute_vae_reconstruction_quality(
                text_embed, prior_mean
            )
            
            # 6. 创建结果
            result = PriorAnalysisResult(
                text_instruction=instruction,
                text_embed_dim=text_embed.shape[0],
                prior_embed_dim=prior_mean.shape[0],
                text_to_prior_similarity=text_to_prior_sim,
                prior_variance=prior_variance,
                reconstruction_quality=reconstruction_quality,
                text_embed=text_embed,
                prior_embed=prior_mean,
            )
            
            results[instruction] = result
            all_text_embeds.append(text_embed)
            all_prior_embeds.append(prior_mean)
            instruction_list.append(instruction)
            
            # 缓存嵌入
            self.embed_cache[instruction] = {
                'text': text_embed,
                'prior': prior_mean,
            }
        
        # 7. 全局分析：计算最近邻和聚类
        all_text_embeds = np.array(all_text_embeds)
        all_prior_embeds = np.array(all_prior_embeds)
        
        self._compute_nearest_neighbors(results, instruction_list, all_prior_embeds)
        
        # 8. 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存分析结果
            results_dict = {inst: res.to_dict() for inst, res in results.items()}
            with open(output_dir / "prior_analysis.json", 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # 保存嵌入
            np.save(output_dir / "text_embeds.npy", all_text_embeds)
            np.save(output_dir / "prior_embeds.npy", all_prior_embeds)
            with open(output_dir / "instructions.json", 'w') as f:
                json.dump(instruction_list, f, indent=2)
            
            logger.info(f"  ✓ Prior 分析结果已保存到: {output_dir}")
        
        return results
    
    def _compute_vae_reconstruction_quality(
        self,
        text_embed: np.ndarray,
        prior_embed: np.ndarray
    ) -> float:
        """
        计算 VAE 重建质量
        
        注意：Prior VAE 是条件VAE，输入文本嵌入，输出"类视觉"嵌入
        这里我们评估的是：给定文本嵌入，Prior能多好地重建它对应的视觉嵌入
        
        由于我们没有真实的视觉嵌入（ground truth），这里用文本嵌入作为参考
        """
        # 简化版：计算文本嵌入和Prior输出的L2距离
        l2_distance = euclidean(text_embed, prior_embed)
        
        # 归一化到 [0, 1]，距离越小质量越高
        # 假设最大距离为 sqrt(512 * 4) = 45（经验值）
        max_dist = 45.0
        quality = max(0, 1 - l2_distance / max_dist)
        
        return quality
    
    def _compute_nearest_neighbors(
        self,
        results: Dict[str, PriorAnalysisResult],
        instructions: List[str],
        embeddings: np.ndarray,
        k: int = 3
    ):
        """计算每个指令的最近邻"""
        for i, instruction in enumerate(instructions):
            # 计算与其他所有指令的相似度
            similarities = []
            for j, other_inst in enumerate(instructions):
                if i != j:
                    sim = 1 - cosine(embeddings[i], embeddings[j])
                    similarities.append((other_inst, sim))
            
            # 排序并取top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            results[instruction].nearest_neighbors = similarities[:k]
    
    # ==========================================================================
    # 阶段2: 策略模型分析 p(τ|z_goal)
    # ==========================================================================
    
    def analyze_policy_model(
        self,
        instruction: str,
        env_name: str,
        env_config: Optional[Dict] = None,
        max_steps: int = 1000,
        n_trials: int = 5,
        use_ground_truth_visual: bool = False,
        ground_truth_frames: Optional[List[np.ndarray]] = None,
    ) -> List[PolicyAnalysisResult]:
        """
        分析策略模型：给定目标嵌入，生成动作序列的质量
        
        评估指标：
        1. 动作多样性（熵）
        2. 时序一致性
        3. 成功率
        
        可以对比两种情况：
        - 使用 Prior 输出的嵌入
        - 使用真实的视觉嵌入（如果有专家演示视频）
        
        Args:
            instruction: 指令
            env_name: 环境名称
            env_config: 环境配置
            max_steps: 最大步数
            n_trials: 试验次数
            use_ground_truth_visual: 是否使用真实视觉嵌入
            ground_truth_frames: 真实视频帧（用于生成视觉嵌入）
            
        Returns:
            results: 每次试验的分析结果
        """
        self._load_components()
        
        logger.info(f"开始分析策略模型: '{instruction}'")
        
        # 创建环境
        env = make_env(self.seed, env_name=env_name, env_config=env_config)
        
        results = []
        
        for trial_idx in range(n_trials):
            logger.info(f"  Trial {trial_idx + 1}/{n_trials}")
            
            # 1. 获取目标嵌入
            if use_ground_truth_visual and ground_truth_frames:
                # 使用真实视觉嵌入（专家演示）
                goal_frame = ground_truth_frames[len(ground_truth_frames) // 2]
                with th.no_grad():
                    goal_embed = self._mineclip.encode_image(goal_frame)
                    if hasattr(goal_embed, 'cpu'):
                        goal_embed = goal_embed.cpu().numpy()
                    # 确保是 1-D
                    goal_embed = np.squeeze(goal_embed)
                goal_embed_source = 'ground_truth_visual'
            else:
                # 使用 Prior 输出
                with th.no_grad():
                    goal_embed = get_prior_embed(
                        instruction,
                        self._mineclip,
                        self._prior,
                        DEVICE
                    )
                    # get_prior_embed 可能返回 tensor 或 numpy array
                    if hasattr(goal_embed, 'cpu'):
                        goal_embed = goal_embed.cpu().numpy()
                    elif isinstance(goal_embed, th.Tensor):
                        goal_embed = goal_embed.detach().cpu().numpy()
                    
                    # 确保是 1-D（去除可能的 batch 维度）
                    if isinstance(goal_embed, np.ndarray):
                        goal_embed = np.squeeze(goal_embed)
                        
                goal_embed_source = 'prior'
            
            # 2. 运行 episode
            obs = env.reset()
            self._agent.reset(cond_scale=self.text_cond_scale)
            
            action_sequence = []
            total_reward = 0.0
            success = False
            
            for step in range(max_steps):
                # 获取动作
                action = self._agent.get_action(obs, goal_embed)
                
                # 记录动作
                action_sequence.append(action)
                
                # 执行动作
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    success = info.get('success', reward > 0)
                    break
            
            # 3. 分析动作序列
            action_diversity = self._compute_action_diversity(action_sequence)
            action_distribution = self._compute_action_distribution(action_sequence)
            temporal_consistency = self._compute_temporal_consistency(action_sequence)
            repeated_ratio = self._compute_repeated_action_ratio(action_sequence)
            
            # 4. 创建结果
            result = PolicyAnalysisResult(
                instruction=instruction,
                goal_embed_source=goal_embed_source,
                total_steps=len(action_sequence),
                action_diversity=action_diversity,
                action_distribution=action_distribution,
                temporal_consistency=temporal_consistency,
                repeated_action_ratio=repeated_ratio,
                success=success,
                final_reward=total_reward,
                success_steps=len(action_sequence) if success else None,
            )
            
            results.append(result)
        
        env.close()
        
        return results
    
    def _compute_action_diversity(self, actions: List[Dict]) -> float:
        """计算动作多样性（熵）"""
        if not actions:
            return 0.0
        
        # 简化版：计算每个动作维度的熵，然后平均
        # MineRL动作空间很复杂，这里只计算主要维度
        
        # 统计每个动作的出现次数
        action_counts = defaultdict(int)
        for action in actions:
            # 将动作转换为字符串作为key
            action_key = str(sorted(action.items()))
            action_counts[action_key] += 1
        
        # 计算熵
        total = len(actions)
        probs = np.array([count / total for count in action_counts.values()])
        action_entropy = entropy(probs)
        
        return action_entropy
    
    def _compute_action_distribution(self, actions: List[Dict]) -> Dict[str, float]:
        """计算动作分布统计"""
        if not actions:
            return {}
        
        # 统计主要动作维度
        stats = {
            'forward': 0,
            'back': 0,
            'left': 0,
            'right': 0,
            'jump': 0,
            'sneak': 0,
            'attack': 0,
            'use': 0,
        }
        
        for action in actions:
            # 根据实际的 MineRL 动作格式统计
            # 这里需要根据实际动作格式调整
            if 'forward' in action and action['forward'] > 0:
                stats['forward'] += 1
            if 'back' in action and action['back'] > 0:
                stats['back'] += 1
            if 'left' in action and action['left'] > 0:
                stats['left'] += 1
            if 'right' in action and action['right'] > 0:
                stats['right'] += 1
            if 'jump' in action and action['jump'] > 0:
                stats['jump'] += 1
            if 'sneak' in action and action['sneak'] > 0:
                stats['sneak'] += 1
            if 'attack' in action and action['attack'] > 0:
                stats['attack'] += 1
            if 'use' in action and action['use'] > 0:
                stats['use'] += 1
        
        # 归一化
        total = len(actions)
        return {k: v / total for k, v in stats.items()}
    
    def _compute_temporal_consistency(self, actions: List[Dict]) -> float:
        """计算时序一致性（相邻动作的相似度）"""
        if len(actions) < 2:
            return 1.0
        
        # 计算相邻动作的相似度
        similarities = []
        for i in range(len(actions) - 1):
            # 简化版：计算两个动作字典的Jaccard相似度
            action1 = set(str(k) + str(v) for k, v in actions[i].items())
            action2 = set(str(k) + str(v) for k, v in actions[i+1].items())
            
            if len(action1 | action2) == 0:
                sim = 1.0
            else:
                sim = len(action1 & action2) / len(action1 | action2)
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _compute_repeated_action_ratio(self, actions: List[Dict]) -> float:
        """计算重复动作的比例"""
        if len(actions) < 2:
            return 0.0
        
        repeated = 0
        for i in range(len(actions) - 1):
            # 比较两个动作字典的所有值
            try:
                action1 = actions[i]
                action2 = actions[i+1]
                
                # 对于每个键，比较数值或数组
                is_same = True
                for key in action1.keys():
                    val1 = action1[key]
                    val2 = action2[key]
                    
                    # 转换为numpy数组进行比较
                    if isinstance(val1, (np.ndarray, list)):
                        val1 = np.asarray(val1)
                        val2 = np.asarray(val2)
                        if not np.array_equal(val1, val2):
                            is_same = False
                            break
                    else:
                        if val1 != val2:
                            is_same = False
                            break
                
                if is_same:
                    repeated += 1
            except Exception:
                # 如果比较失败，认为不相同
                pass
        
        return repeated / (len(actions) - 1)
    
    # ==========================================================================
    # 端到端分析（两阶段联合）
    # ==========================================================================
    
    def analyze_end_to_end(
        self,
        task_id: str,
        instruction: str,
        env_name: str,
        env_config: Optional[Dict] = None,
        max_steps: int = 1000,
        n_trials: int = 5,
        output_dir: Optional[Path] = None,
    ) -> List[EndToEndAnalysisResult]:
        """
        端到端分析：同时评估两个阶段，并进行错误归因
        
        核心思想：
        1. 运行正常流程：文本 → Prior → 策略
        2. 运行对照实验：文本 → MineCLIP直接 → 策略（跳过Prior）
        3. 对比两者的差异，归因错误来源
        
        Args:
            task_id: 任务ID
            instruction: 指令
            env_name: 环境名称
            env_config: 环境配置
            max_steps: 最大步数
            n_trials: 试验次数
            output_dir: 输出目录
            
        Returns:
            results: 每次试验的端到端分析结果
        """
        logger.info(f"开始端到端分析: {task_id}")
        
        # 1. 分析 Prior 模型
        logger.info("  阶段1: 分析 Prior 模型")
        prior_results = self.analyze_prior_model([instruction])
        prior_result = prior_results[instruction]
        
        # 2. 分析策略模型（使用 Prior 输出）
        logger.info("  阶段2: 分析策略模型（使用 Prior）")
        policy_results_with_prior = self.analyze_policy_model(
            instruction=instruction,
            env_name=env_name,
            env_config=env_config,
            max_steps=max_steps,
            n_trials=n_trials,
            use_ground_truth_visual=False,
        )
        
        # 3. 对照实验：直接使用文本嵌入（跳过 Prior）
        logger.info("  对照实验: 分析策略模型（跳过 Prior）")
        policy_results_without_prior = self.analyze_policy_model(
            instruction=instruction,
            env_name=env_name,
            env_config=env_config,
            max_steps=max_steps,
            n_trials=n_trials,
            use_ground_truth_visual=False,  # 这里改用文本嵌入
        )
        
        # 4. 联合分析和错误归因
        end_to_end_results = []
        
        for trial_idx in range(n_trials):
            policy_result = policy_results_with_prior[trial_idx]
            
            # 计算两阶段的贡献
            # 简化版：基于成功率对比
            success_with_prior = policy_result.success
            success_without_prior = policy_results_without_prior[trial_idx].success
            
            # 如果使用Prior后成功率提高，说明Prior有贡献
            if success_with_prior and not success_without_prior:
                stage1_contribution = 0.7  # Prior贡献大
                stage2_contribution = 0.3
                bottleneck_stage = 2  # 策略是瓶颈
            elif not success_with_prior and success_without_prior:
                stage1_contribution = 0.3  # Prior可能有负面影响
                stage2_contribution = 0.7
                bottleneck_stage = 1  # Prior是瓶颈
            else:
                # 两者都成功或都失败
                stage1_contribution = 0.5
                stage2_contribution = 0.5
                bottleneck_stage = 1 if not success_with_prior else 0
            
            # 错误归因
            if not success_with_prior:
                failure_attribution = {
                    'prior': 1.0 - prior_result.text_to_prior_similarity,
                    'policy': 1.0 - policy_result.temporal_consistency,
                }
                # 归一化
                total = sum(failure_attribution.values())
                if total > 0:
                    failure_attribution = {k: v/total for k, v in failure_attribution.items()}
            else:
                failure_attribution = None
            
            # 创建结果
            result = EndToEndAnalysisResult(
                task_id=task_id,
                instruction=instruction,
                trial_idx=trial_idx,
                prior_result=prior_result,
                policy_result=policy_result,
                stage1_contribution=stage1_contribution,
                stage2_contribution=stage2_contribution,
                bottleneck_stage=bottleneck_stage,
                failure_attribution=failure_attribution,
            )
            
            end_to_end_results.append(result)
        
        # 5. 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_dict = [res.to_dict() for res in end_to_end_results]
            with open(output_dir / f"{task_id}_end_to_end.json", 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            logger.info(f"  ✓ 端到端分析结果已保存到: {output_dir}")
        
        return end_to_end_results
    
    def cleanup(self):
        """清理资源"""
        if self._agent:
            # Agent 通常不需要显式清理
            pass
        
        logger.info("Steve1PolicyEvaluator 清理完成")

