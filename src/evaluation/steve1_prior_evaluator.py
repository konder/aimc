"""
Steve1 Prior 模型评估器（Prior Evaluator - 正确版本）
====================================================

基于正确的Prior评估思路重新设计：
1. 目标准确性：Prior输出 vs 真实成功画面（同空间比较）
2. 语义鲁棒性：同一任务不同表述的一致性
3. 一致性：同一表述多次采样的稳定性
4. 可区分性：不同任务的z_goal差异

【关键修正】
- ❌ 错误：直接比较文本嵌入和Prior输出（跨空间）
- ✅ 正确：Prior输出 vs 真实成功画面的视觉嵌入（同空间）

作者: AI Assistant
日期: 2025-11-27
"""

import torch as th
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 导入 Steve1 组件
from steve1.utils.embed_utils import get_prior_embed
from steve1.config import PRIOR_INFO
from src.utils.steve1_mineclip_agent_env_utils import (
    load_mineclip_wconfig,
    load_vae_model,
)
from src.utils.device import DEVICE

logger = logging.getLogger(__name__)


@dataclass
class PriorAnalysisResult:
    """Prior 模型分析结果（改进版本 - 支持 forward_reward_head）"""
    
    # 基础信息
    task_id: str
    instruction: str
    
    # 核心指标
    goal_accuracy: float  # 目标准确性均值（vs真实成功画面）
    consistency: float  # 一致性（多次采样）
    semantic_robustness: Optional[float] = None  # 语义鲁棒性（指令变体）
    
    # 新增：目标准确性标准差
    goal_accuracy_std: float = 0.0  # 目标准确性的标准差
    
    # 辅助信息
    n_success_visuals: int = 0  # 成功画面数量
    n_samples: int = 10  # 采样次数
    n_variants: int = 0  # 指令变体数量
    
    # 嵌入维度
    prior_embed_dim: int = 512
    visual_embed_dim: int = 512
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'instruction': self.instruction,
            'goal_accuracy': float(self.goal_accuracy),
            'goal_accuracy_std': float(self.goal_accuracy_std),
            'consistency': float(self.consistency),
            'semantic_robustness': float(self.semantic_robustness) if self.semantic_robustness else None,
            'n_success_visuals': self.n_success_visuals,
            'n_samples': self.n_samples,
            'n_variants': self.n_variants,
            'prior_embed_dim': self.prior_embed_dim,
            'visual_embed_dim': self.visual_embed_dim,
        }


@dataclass
class PriorEvaluationSummary:
    """Prior评估总结（多任务）"""
    
    # 任务级指标
    task_results: List[PriorAnalysisResult] = field(default_factory=list)
    
    # 跨任务指标
    discriminability: float = 0.0  # 可区分性
    avg_goal_accuracy: float = 0.0
    avg_consistency: float = 0.0
    avg_semantic_robustness: float = 0.0
    
    # 退化检测
    is_degraded: bool = False
    degradation_warning: str = ""
    
    # 可视化数据
    visualization_data: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """计算汇总指标"""
        if self.task_results:
            # 计算平均值
            self.avg_goal_accuracy = np.mean([r.goal_accuracy for r in self.task_results])
            self.avg_consistency = np.mean([r.consistency for r in self.task_results])
            
            robustness_scores = [r.semantic_robustness for r in self.task_results 
                                if r.semantic_robustness is not None]
            if robustness_scores:
                self.avg_semantic_robustness = np.mean(robustness_scores)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'task_results': [r.to_dict() for r in self.task_results],
            'discriminability': float(self.discriminability),
            'avg_goal_accuracy': float(self.avg_goal_accuracy),
            'avg_consistency': float(self.avg_consistency),
            'avg_semantic_robustness': float(self.avg_semantic_robustness),
            'is_degraded': self.is_degraded,
            'degradation_warning': self.degradation_warning,
            'visualization_data': self.visualization_data,
        }


class Steve1PriorEvaluator:
    """
    Steve1 Prior 评估器（正确版本）
    
    实现正确的Prior评估指标：
    1. 目标准确性：Prior vs 真实成功画面（同空间比较）
    2. 语义鲁棒性：同一任务不同表述的一致性
    3. 一致性：同一表述多次采样的稳定性
    4. 可区分性：不同任务的z_goal差异
    """
    
    def __init__(
        self,
        prior_weights: str = "data/weights/steve1/steve1_prior.pt",
        success_visuals_path: Optional[str] = None,
        instruction_variants_path: Optional[str] = None,
        seed: int = 42,
    ):
        """
        初始化 Prior 评估器
        
        Args:
            prior_weights: Prior VAE 权重路径
            success_visuals_path: 成功画面嵌入数据路径（.pkl）
            instruction_variants_path: 指令变体配置路径（.json）
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        th.manual_seed(seed)
        
        logger.info("=" * 80)
        logger.info("初始化 Steve1PriorEvaluator（正确版本）...")
        logger.info("=" * 80)
        
        # 加载 MineCLIP
        logger.info("加载 MineCLIP...")
        self._mineclip = load_mineclip_wconfig()
        logger.info(f"✓ MineCLIP 已加载")
        
        # 加载 Prior (VAE)
        logger.info(f"加载 Prior VAE: {prior_weights}")
        prior_info = PRIOR_INFO.copy()
        prior_info['prior_weights'] = prior_weights
        self._prior = load_vae_model(prior_info)
        logger.info(f"✓ Prior VAE 已加载")
        
        # 加载成功画面数据
        self.success_visuals = {}
        if success_visuals_path and Path(success_visuals_path).exists():
            logger.info(f"加载成功画面嵌入: {success_visuals_path}")
            with open(success_visuals_path, 'rb') as f:
                self.success_visuals = pickle.load(f)
            logger.info(f"✓ 已加载 {len(self.success_visuals)} 个任务的成功画面")
        else:
            logger.warning("⚠️ 未提供成功画面数据，将无法计算目标准确性")
        
        # 加载指令变体
        self.instruction_variants = {}
        if instruction_variants_path and Path(instruction_variants_path).exists():
            logger.info(f"加载指令变体: {instruction_variants_path}")
            with open(instruction_variants_path, 'r') as f:
                self.instruction_variants = json.load(f)
            logger.info(f"✓ 已加载 {len(self.instruction_variants)} 个任务的指令变体")
        else:
            logger.warning("⚠️ 未提供指令变体数据，将无法计算语义鲁棒性")
        
        logger.info("=" * 80)
        logger.info("✓ Steve1PriorEvaluator 初始化完成")
        logger.info("=" * 80)
    
    def _get_prior_embed(self, instruction: str) -> np.ndarray:
        """获取Prior嵌入（确保返回numpy数组）"""
        prior_embed = get_prior_embed(
            instruction,
            self._mineclip,
            self._prior,
            DEVICE
        )
        
        # 确保是 numpy array
        if hasattr(prior_embed, 'cpu'):
            prior_embed = prior_embed.cpu().numpy()
        elif isinstance(prior_embed, th.Tensor):
            prior_embed = prior_embed.detach().cpu().numpy()
        
        return np.squeeze(prior_embed)  # 确保是1-D
    
    def compute_goal_accuracy(
        self,
        task_id: str,
        instruction: str,
        use_reward_head: bool = True
    ) -> Tuple[float, float, int]:
        """
        计算目标准确性（改进版：使用 MineCLIP forward_reward_head）
        
        Args:
            task_id: 任务ID
            instruction: 指令文本
            use_reward_head: 是否使用 MineCLIP 的 forward_reward_head
            
        Returns:
            (goal_accuracy_mean, goal_accuracy_std, n_success_visuals)
        """
        # 检查是否有成功画面数据
        if task_id not in self.success_visuals:
            logger.warning(f"任务 {task_id} 没有成功画面数据")
            return 0.0, 0.0, 0
        
        # 获取Prior输出
        z_goal = self._get_prior_embed(instruction)
        
        # 获取真实成功画面嵌入
        success_visual_embeds = self.success_visuals[task_id]['success_visual_embeds']
        
        if use_reward_head:
            # 使用余弦相似度（0-1范围，易于理解）
            import torch.nn.functional as F
            # 转换为 torch tensor
            z_goal_tensor = th.from_numpy(z_goal).float().unsqueeze(0).to(DEVICE)  # [1, 512]
            z_visuals_tensor = th.from_numpy(
                np.array(success_visual_embeds)
            ).float().to(DEVICE)  # [N, 512]
            
            with th.no_grad():
                # L2归一化
                z_goal_norm = F.normalize(z_goal_tensor, p=2, dim=1)
                z_visuals_norm = F.normalize(z_visuals_tensor, p=2, dim=1)
                
                # 计算余弦相似度: [1, N]
                cosine_sim = th.matmul(z_goal_norm, z_visuals_norm.T)
                
                # 转换到0-1范围: (cosine + 1) / 2
                # 因为cosine范围是[-1, 1]
                similarities = ((cosine_sim + 1) / 2).squeeze(0).cpu().numpy()  # [N]
                
            goal_accuracy_mean = float(np.mean(similarities))
            goal_accuracy_std = float(np.std(similarities))
            
        else:
            # 方案 A: 先计算后平均（使用余弦相似度）
            similarities = []
            for z_visual in success_visual_embeds:
                z_visual = np.squeeze(z_visual)
                sim = 1 - cosine(z_goal, z_visual)
                similarities.append(sim)
            
            goal_accuracy_mean = float(np.mean(similarities))
            goal_accuracy_std = float(np.std(similarities))
        
        return goal_accuracy_mean, goal_accuracy_std, len(success_visual_embeds)
    
    def compute_consistency(
        self,
        instruction: str,
        n_samples: int = 10
    ) -> float:
        """
        计算一致性（同一表述多次采样）
        
        Args:
            instruction: 指令文本
            n_samples: 采样次数
            
        Returns:
            consistency (0-1)
        """
        # 多次采样Prior
        z_goals = [self._get_prior_embed(instruction) for _ in range(n_samples)]
        
        if len(z_goals) < 2:
            return 1.0
        
        # 计算两两相似度
        similarities = []
        for i in range(len(z_goals)):
            for j in range(i+1, len(z_goals)):
                sim = 1 - cosine(z_goals[i], z_goals[j])
                similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def compute_semantic_robustness(
        self,
        task_id: str
    ) -> Tuple[float, int]:
        """
        计算语义鲁棒性（同一任务不同表述）
        
        Args:
            task_id: 任务ID
            
        Returns:
            (semantic_robustness, n_variants)
        """
        # 检查是否有指令变体
        if task_id not in self.instruction_variants:
            logger.warning(f"任务 {task_id} 没有指令变体数据")
            return None, 0
        
        variants = self.instruction_variants[task_id]['variants']
        
        if len(variants) < 2:
            return None, 0
        
        # 为每个变体生成Prior嵌入
        z_goals = [self._get_prior_embed(v) for v in variants]
        
        # 计算类内相似度（两两相似度的平均）
        similarities = []
        for i in range(len(z_goals)):
            for j in range(i+1, len(z_goals)):
                sim = 1 - cosine(z_goals[i], z_goals[j])
                similarities.append(sim)
        
        semantic_robustness = float(np.mean(similarities))
        
        return semantic_robustness, len(variants)
    
    def analyze_prior_model(
        self,
        task_ids: List[str],
        n_samples: int = 10,
        output_dir: Optional[Path] = None
    ) -> PriorEvaluationSummary:
        """
        分析 Prior 模型在给定任务集上的性能
        
        Args:
            task_ids: 要分析的任务ID列表
            n_samples: 每个指令采样的 Prior 样本数（用于一致性评估）
            output_dir: 输出目录（保存结果）
            
        Returns:
            Prior评估总结
        """
        logger.info(f"开始分析 Prior 模型 ({len(task_ids)} 个任务)...")
        logger.info("=" * 80)
        
        results = []
        all_z_goals = []  # 用于计算可区分性
        
        for task_id in task_ids:
            logger.info(f"分析任务: {task_id}")
            
            # 获取指令
            if task_id in self.success_visuals:
                instruction = self.success_visuals[task_id]['instruction']
            elif task_id in self.instruction_variants:
                instruction = self.instruction_variants[task_id]['canonical']
            else:
                logger.warning(f"  跳过 {task_id}: 没有指令信息")
                continue
            
            # 1. 目标准确性（使用 forward_reward_head）
            goal_accuracy, goal_accuracy_std, n_success_visuals = self.compute_goal_accuracy(
                task_id, instruction, use_reward_head=True
            )
            logger.info(f"  目标准确性: {goal_accuracy:.4f} ± {goal_accuracy_std:.4f} (基于 {n_success_visuals} 个成功画面)")
            
            # 2. 一致性
            consistency = self.compute_consistency(instruction, n_samples)
            logger.info(f"  一致性: {consistency:.4f} ({n_samples} 次采样)")
            
            # 3. 语义鲁棒性
            semantic_robustness, n_variants = self.compute_semantic_robustness(task_id)
            if semantic_robustness is not None:
                logger.info(f"  语义鲁棒性: {semantic_robustness:.4f} ({n_variants} 个变体)")
            else:
                logger.info(f"  语义鲁棒性: N/A (没有变体数据)")
            
            # 保存z_goal用于可区分性计算
            z_goal = self._get_prior_embed(instruction)
            all_z_goals.append(z_goal)
            
            # 创建结果
            result = PriorAnalysisResult(
                task_id=task_id,
                instruction=instruction,
                goal_accuracy=goal_accuracy,
                goal_accuracy_std=goal_accuracy_std,
                consistency=consistency,
                semantic_robustness=semantic_robustness,
                n_success_visuals=n_success_visuals,
                n_samples=n_samples,
                n_variants=n_variants,
                prior_embed_dim=z_goal.shape[0],
                visual_embed_dim=512,  # MineCLIP固定维度
            )
            
            results.append(result)
            logger.info("")
        
        # 4. 计算可区分性（跨任务）
        discriminability = self._compute_discriminability(all_z_goals)
        logger.info(f"可区分性（跨任务）: {discriminability:.4f}")
        
        # 5. 退化检测
        variance = self._compute_variance(all_z_goals)
        is_degraded, warning = self._check_degradation(discriminability, variance)
        logger.info(f"退化检测: {warning}")
        
        # 6. 生成可视化数据
        logger.info("生成可视化数据...")
        visualization_data = self._generate_visualization_data(
            task_ids=task_ids,
            results=results,
            all_z_goals=all_z_goals
        )
        logger.info("✓ 可视化数据已生成")
        
        # 创建总结
        summary = PriorEvaluationSummary(
            task_results=results,
            discriminability=discriminability,
            is_degraded=is_degraded,
            degradation_warning=warning,
            visualization_data=visualization_data,
        )
        
        # 6. 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 JSON
            with open(output_dir / "prior_evaluation_summary.json", 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)
            
            logger.info(f"✓ Prior 评估结果已保存到: {output_dir}")
        
        logger.info("=" * 80)
        logger.info("✓ Prior 模型分析完成")
        logger.info("=" * 80)
        
        # 打印总结
        self._print_summary(summary)
        
        return summary
    
    def _compute_discriminability(self, z_goals: List[np.ndarray]) -> float:
        """计算可区分性（不同任务）"""
        if len(z_goals) < 2:
            return 1.0
        
        # 计算类间相似度
        inter_similarities = []
        for i in range(len(z_goals)):
            for j in range(i+1, len(z_goals)):
                sim = 1 - cosine(z_goals[i], z_goals[j])
                inter_similarities.append(sim)
        
        mean_inter_sim = np.mean(inter_similarities)
        discriminability = 1 - mean_inter_sim
        
        return float(discriminability)
    
    def _compute_variance(self, z_goals: List[np.ndarray]) -> float:
        """计算方差（多个任务）"""
        z_goals = np.array(z_goals)
        variance_per_dim = np.var(z_goals, axis=0)
        mean_variance = np.mean(variance_per_dim)
        return float(mean_variance)
    
    def _check_degradation(
        self,
        discriminability: float,
        variance: float
    ) -> Tuple[bool, str]:
        """检测Prior是否退化"""
        if discriminability < 0.3 and variance < 0.0001:
            return True, "⚠️ 警告：Prior可能退化（所有任务输出相似且方差极低）"
        elif discriminability < 0.3:
            return True, "⚠️ 警告：Prior可区分性低（不同任务输出过于相似）"
        elif variance < 0.0001:
            return True, "⚠️ 警告：Prior方差极低（可能退化为常数输出）"
        else:
            return False, "✓ Prior未退化"
    
    def _generate_visualization_data(
        self,
        task_ids: List[str],
        results: List[PriorAnalysisResult],
        all_z_goals: List[np.ndarray]
    ) -> Dict:
        """生成可视化数据"""
        
        vis_data = {}
        
        # 1. 收集Prior输出和真实画面嵌入
        prior_embeds = []
        visual_embeds = []
        labels = []
        
        for task_id, result in zip(task_ids, results):
            # Prior嵌入
            z_goal = self._get_prior_embed(result.instruction)
            prior_embeds.append(z_goal)
            
            # 真实画面嵌入（平均）
            if task_id in self.success_visuals:
                success_visual_embeds = self.success_visuals[task_id]['success_visual_embeds']
                z_visual_mean = np.mean(success_visual_embeds, axis=0)
                visual_embeds.append(z_visual_mean)
            else:
                # 如果没有真实画面，用零向量占位
                visual_embeds.append(np.zeros_like(z_goal))
            
            labels.append(task_id)
        
        prior_embeds = np.array(prior_embeds)
        visual_embeds = np.array(visual_embeds)
        
        # 2. t-SNE降维
        logger.info("  计算 t-SNE 降维...")
        combined_embeds = np.vstack([prior_embeds, visual_embeds])
        n_samples = combined_embeds.shape[0]
        perplexity = min(30, max(5, n_samples // 3))
        
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_coords = tsne.fit_transform(combined_embeds)
            
            vis_data['tsne'] = {
                'prior_coords': tsne_coords[:len(prior_embeds)].tolist(),
                'visual_coords': tsne_coords[len(prior_embeds):].tolist(),
                'labels': labels
            }
        except Exception as e:
            logger.warning(f"  t-SNE计算失败: {e}")
            vis_data['tsne'] = None
        
        # 3. PCA降维
        logger.info("  计算 PCA 降维...")
        try:
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(combined_embeds)
            
            vis_data['pca'] = {
                'prior_coords': pca_coords[:len(prior_embeds)].tolist(),
                'visual_coords': pca_coords[len(prior_embeds):].tolist(),
                'labels': labels,
                'explained_variance': pca.explained_variance_ratio_.tolist()
            }
        except Exception as e:
            logger.warning(f"  PCA计算失败: {e}")
            vis_data['pca'] = None
        
        # 4. 相似度矩阵
        logger.info("  计算相似度矩阵...")
        n_tasks = len(prior_embeds)
        similarity_matrix = np.zeros((n_tasks, n_tasks))
        
        for i in range(n_tasks):
            for j in range(n_tasks):
                sim = 1 - cosine(prior_embeds[i], prior_embeds[j])
                similarity_matrix[i, j] = sim
        
        vis_data['similarity_matrix'] = {
            'matrix': similarity_matrix.tolist(),
            'labels': labels
        }
        
        # 5. 方差分布
        logger.info("  计算方差分布...")
        variance_per_dim = np.var(prior_embeds, axis=0)
        
        vis_data['variance'] = {
            'variance_per_dim': variance_per_dim.tolist(),
            'mean_variance': float(np.mean(variance_per_dim)),
            'std_variance': float(np.std(variance_per_dim))
        }
        
        # 6. Top-K和Bottom-K任务
        logger.info("  计算 Top-K/Bottom-K 任务...")
        sorted_results = sorted(results, key=lambda r: r.goal_accuracy, reverse=True)
        
        top_k = 5
        bottom_k = 5
        
        vis_data['rankings'] = {
            'top_tasks': [
                {
                    'task_id': r.task_id,
                    'instruction': r.instruction,
                    'goal_accuracy': r.goal_accuracy,
                    'consistency': r.consistency,
                }
                for r in sorted_results[:top_k]
            ],
            'bottom_tasks': [
                {
                    'task_id': r.task_id,
                    'instruction': r.instruction,
                    'goal_accuracy': r.goal_accuracy,
                    'consistency': r.consistency,
                }
                for r in sorted_results[-bottom_k:]
            ]
        }
        
        # 7. 准确性vs一致性散点数据
        vis_data['scatter'] = {
            'task_ids': [r.task_id for r in results],
            'goal_accuracies': [r.goal_accuracy for r in results],
            'consistencies': [r.consistency for r in results],
        }
        
        return vis_data
    
    def _print_summary(self, summary: PriorEvaluationSummary):
        """打印评估总结"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("Prior 评估总结")
        logger.info("=" * 80)
        logger.info(f"任务数: {len(summary.task_results)}")
        logger.info(f"平均目标准确性: {summary.avg_goal_accuracy:.4f}")
        logger.info(f"平均一致性: {summary.avg_consistency:.4f}")
        if summary.avg_semantic_robustness > 0:
            logger.info(f"平均语义鲁棒性: {summary.avg_semantic_robustness:.4f}")
        logger.info(f"可区分性: {summary.discriminability:.4f}")
        logger.info("")
        logger.info("指标解读:")
        logger.info("  目标准确性 > 0.6: 优秀,  0.4-0.6: 良好,  < 0.4: 需改进")
        logger.info("  一致性 > 0.95: 优秀,  0.85-0.95: 良好,  < 0.85: 需改进")
        logger.info("  语义鲁棒性 > 0.9: 优秀,  0.7-0.9: 良好,  < 0.7: 需改进")
        logger.info("  可区分性 > 0.5: 优秀,  0.3-0.5: 良好,  < 0.3: 需改进")
        logger.info("=" * 80)
    
    def __del__(self):
        """清理资源"""
        logger.info("Steve1PriorEvaluator 清理完成")
