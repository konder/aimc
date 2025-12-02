"""
Prior评估框架 V2（配置驱动）
=========================================

【职责】
全面评估 Prior 模型的三个维度：
1. 内在质量 (Intrinsic Quality)
2. 输出质量 (Output Quality)  
3. 可控性 (Controllability)

【改进】
- 支持YAML配置文件
- 支持所有评估维度和指标
- 生成详细的HTML报告
- 支持CFG扫描分析

作者: AI Assistant
日期: 2025-12-01
版本: 2.0
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import logging
import pickle
import yaml
import json
import numpy as np
import torch as th
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.evaluation.steve1_prior_evaluator import Steve1PriorEvaluator
from src.utils.steve1_mineclip_agent_env_utils import load_mineclip_wconfig
from steve1.utils.embed_utils import get_prior_embed
from steve1.config import PRIOR_INFO
from src.utils.device import DEVICE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TaskEvaluationConfig:
    """单个任务的评估配置"""
    task_id: str
    instruction: str
    instruction_variants: List[str] = field(default_factory=list)
    success_visuals_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class DimensionResults:
    """单个维度的评估结果"""
    dimension_name: str
    enabled: bool
    metrics: Dict = field(default_factory=dict)
    visualizations: Dict = field(default_factory=dict)


@dataclass
class PriorEvaluationResults:
    """Prior评估完整结果"""
    # 基本信息
    config_file: str
    n_tasks: int
    task_ids: List[str]
    
    # 三个维度的结果
    intrinsic_quality: Optional[DimensionResults] = None
    output_quality: Optional[DimensionResults] = None
    controllability: Optional[DimensionResults] = None
    
    # 任务级详细结果
    task_results: Dict = field(default_factory=dict)
    
    # 总结
    summary: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'config_file': self.config_file,
            'n_tasks': self.n_tasks,
            'task_ids': self.task_ids,
            'intrinsic_quality': asdict(self.intrinsic_quality) if self.intrinsic_quality else None,
            'output_quality': asdict(self.output_quality) if self.output_quality else None,
            'controllability': asdict(self.controllability) if self.controllability else None,
            'task_results': self.task_results,
            'summary': self.summary,
        }


class PriorEvaluationFramework:
    """
    Prior评估框架（配置驱动）
    
    支持三个维度的完整评估：
    1. 内在质量：一致性、语义鲁棒性、输出多样性、区分度保持率
    2. 输出质量：目标对齐度、Prior增益、跨模态一致性
    3. 可控性：CFG敏感度分析
    """
    
    def __init__(self, config_path: str, task_ids: Optional[List[str]] = None, 
                 task_set: Optional[str] = None):
        """
        初始化框架
        
        Args:
            config_path: YAML配置文件路径
            task_ids: 指定要评估的任务ID列表（可选）
            task_set: 指定要评估的任务集名称（可选，如 'harvest,combat'）
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self._init_components()
        
        # 加载任务配置
        self.tasks = self._load_tasks(task_ids=task_ids, task_set=task_set)
        
        logger.info(f"✓ Prior评估框架初始化完成")
        logger.info(f"  配置文件: {config_path}")
        logger.info(f"  任务数量: {len(self.tasks)}")
        if task_ids:
            logger.info(f"  指定任务: {task_ids}")
        if task_set:
            logger.info(f"  指定任务集: {task_set}")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载YAML配置"""
        logger.info(f"加载配置文件: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _init_components(self):
        """初始化Prior和MineCLIP组件"""
        logger.info("初始化模型组件...")
        
        # 加载MineCLIP
        self.mineclip = load_mineclip_wconfig()
        logger.info("  ✓ MineCLIP已加载")
        
        # 加载Prior
        from src.utils.steve1_mineclip_agent_env_utils import load_vae_model
        prior_info = PRIOR_INFO.copy()
        prior_info['prior_weights'] = self.config['global']['prior_weights']
        self.prior = load_vae_model(prior_info)
        logger.info("  ✓ Prior已加载")
    
    def _load_tasks(self, task_ids: Optional[List[str]] = None, 
                    task_set: Optional[str] = None) -> List[TaskEvaluationConfig]:
        """
        从配置加载任务
        
        Args:
            task_ids: 指定要加载的任务ID列表
            task_set: 指定要加载的任务集名称（逗号分隔，如 'harvest,combat'）
            
        Returns:
            任务配置列表
        """
        # 确定要加载哪些任务
        selected_task_ids = None
        
        # 优先级1: 如果指定了具体的task_ids，使用它们
        if task_ids:
            selected_task_ids = set(task_ids)
            logger.info(f"使用指定的任务ID: {task_ids}")
        
        # 优先级2: 如果指定了task_set，从配置中解析
        elif task_set:
            selected_task_ids = self._get_task_ids_from_task_sets(task_set)
            logger.info(f"从任务集 '{task_set}' 加载任务: {sorted(selected_task_ids)}")
        
        # 优先级3: 否则加载配置中所有启用的任务集
        else:
            selected_task_ids = self._get_enabled_task_ids()
            if selected_task_ids:
                logger.info(f"加载所有启用的任务集任务: {len(selected_task_ids)} 个")
        
        # 从配置加载任务定义
        tasks = []
        all_tasks_config = self.config.get('tasks', {})
        
        for task_id, task_config in all_tasks_config.items():
            # 如果指定了任务过滤，只加载选中的任务
            if selected_task_ids is not None and task_id not in selected_task_ids:
                continue
            
            task = TaskEvaluationConfig(
                task_id=task_id,
                instruction=task_config['instruction'],
                instruction_variants=task_config.get('instruction_variants', []),
                success_visuals_path=task_config.get('success_visuals_path'),
                metadata=task_config.get('metadata', {})
            )
            tasks.append(task)
        
        if not tasks:
            logger.warning("未找到匹配的任务！")
        
        return tasks
    
    def _get_task_ids_from_task_sets(self, task_set_names: str) -> set:
        """
        从任务集名称获取任务ID集合
        
        Args:
            task_set_names: 任务集名称，逗号分隔（如 'harvest,combat'）
            
        Returns:
            任务ID集合
        """
        task_ids = set()
        task_sets_config = self.config.get('task_sets', {})
        
        # 解析逗号分隔的任务集名称
        requested_sets = [s.strip() for s in task_set_names.split(',')]
        
        for set_name in requested_sets:
            if set_name not in task_sets_config:
                logger.warning(f"任务集 '{set_name}' 不存在于配置中")
                continue
            
            set_config = task_sets_config[set_name]
            set_tasks = set_config.get('tasks', [])
            task_ids.update(set_tasks)
            logger.info(f"  任务集 '{set_name}': {len(set_tasks)} 个任务")
        
        return task_ids
    
    def _get_enabled_task_ids(self) -> Optional[set]:
        """
        获取所有启用的任务集中的任务ID
        
        Returns:
            任务ID集合，如果没有启用的任务集则返回None（加载所有任务）
        """
        task_sets_config = self.config.get('task_sets', {})
        
        if not task_sets_config:
            # 如果配置中没有task_sets，返回None表示加载所有任务
            logger.info("配置中没有定义task_sets，将加载所有任务")
            return None
        
        task_ids = set()
        enabled_sets = []
        
        for set_name, set_config in task_sets_config.items():
            if set_config.get('enabled', False):
                set_tasks = set_config.get('tasks', [])
                task_ids.update(set_tasks)
                enabled_sets.append(set_name)
        
        if enabled_sets:
            logger.info(f"启用的任务集: {', '.join(enabled_sets)}")
            return task_ids
        else:
            # 如果没有启用的任务集，返回None表示加载所有任务
            logger.info("没有启用的任务集，将加载所有任务")
            return None
    
    def _load_success_visuals(self, task: TaskEvaluationConfig) -> Optional[np.ndarray]:
        """
        加载任务的成功画面嵌入
        
        新结构：从多个trial目录加载pkl文件
        data/success_visuals/{task_id}/trial{num}/visual_embeds.pkl
        """
        if not task.success_visuals_path:
            return None
        
        path = Path(task.success_visuals_path)
        
        # 新结构：success_visuals_path指向任务目录
        if path.is_dir():
            logger.debug(f"从任务目录加载: {path}")
            embeds = []
            
            # 查找所有trial目录
            trial_dirs = sorted(path.glob("trial*"))
            
            if not trial_dirs:
                logger.warning(f"任务目录下未找到trial目录: {path}")
                return None
            
            for trial_dir in trial_dirs:
                pkl_file = trial_dir / "visual_embeds.pkl"
                if not pkl_file.exists():
                    logger.warning(f"trial目录缺少pkl文件: {pkl_file}")
                    continue
                
                try:
                    with open(pkl_file, 'rb') as f:
                        embed = pickle.load(f)
                    
                    # 确保是numpy数组
                    if not isinstance(embed, np.ndarray):
                        embed = np.array(embed)
                    
                    embeds.append(embed)
                    
                except Exception as e:
                    logger.warning(f"加载trial嵌入失败 {pkl_file}: {e}")
                    continue
            
            if not embeds:
                logger.warning(f"未能加载任何trial嵌入: {path}")
                return None
            
            logger.debug(f"加载了 {len(embeds)} 个trial嵌入")
            return np.array(embeds)
        
        # 旧结构兼容：直接是pkl文件
        elif path.is_file() and path.suffix == '.pkl':
            logger.debug(f"从pkl文件加载（旧结构）: {path}")
            try:
                with open(path, 'rb') as f:
                    visual_embeds = pickle.load(f)
                
                # 转换为numpy数组
                if isinstance(visual_embeds, list):
                    visual_embeds = np.array(visual_embeds)
                
                return visual_embeds
            
            except Exception as e:
                logger.error(f"加载成功画面失败: {e}")
                return None
        
        else:
            logger.warning(f"success_visuals_path既不是目录也不是pkl文件: {path}")
            return None
    
    def _get_prior_embed(self, instruction: str, cond_scale: float = 6.0) -> np.ndarray:
        """获取Prior嵌入"""
        # 注意：get_prior_embed 不接受 cond_scale 参数
        # CFG scale 需要在 agent 层面设置，这里只做简单的嵌入提取
        prior_embed = get_prior_embed(
            instruction,
            self.mineclip,
            self.prior,
            DEVICE
        )
        
        if hasattr(prior_embed, 'cpu'):
            prior_embed = prior_embed.cpu().numpy()
        
        return np.squeeze(prior_embed)
    
    # =========================================================================
    # 维度1: 内在质量 (Intrinsic Quality)
    # =========================================================================
    
    def evaluate_intrinsic_quality(self) -> DimensionResults:
        """评估内在质量"""
        logger.info("\n" + "=" * 80)
        logger.info("维度1: 内在质量评估")
        logger.info("=" * 80)
        
        dim_config = self.config['evaluation_dimensions']['intrinsic_quality']
        
        if not dim_config['enabled']:
            logger.info("维度已禁用，跳过")
            return DimensionResults(
                dimension_name="intrinsic_quality",
                enabled=False
            )
        
        results = DimensionResults(
            dimension_name="intrinsic_quality",
            enabled=True
        )
        
        # 指标1.1: 输出稳定性 (Consistency)
        if dim_config['consistency']['enabled']:
            logger.info("\n[指标1.1] 输出稳定性 (Consistency)")
            consistency_results = self._evaluate_consistency(
                n_samples=dim_config['consistency']['n_samples']
            )
            results.metrics['consistency'] = consistency_results
        
        # 指标1.2: 语义鲁棒性 (Semantic Robustness)
        if dim_config['semantic_robustness']['enabled']:
            logger.info("\n[指标1.2] 语义鲁棒性 (Semantic Robustness)")
            robustness_results = self._evaluate_semantic_robustness()
            results.metrics['semantic_robustness'] = robustness_results
        
        # 指标1.3: 输出多样性 (Output Diversity)
        if dim_config['output_diversity']['enabled']:
            logger.info("\n[指标1.3] 输出多样性 (Output Diversity)")
            diversity_results = self._evaluate_output_diversity()
            results.metrics['output_diversity'] = diversity_results
        
        # 指标1.4: 区分度保持率 (Discriminability Preservation)
        if dim_config['discriminability_preservation']['enabled']:
            logger.info("\n[指标1.4] 区分度保持率 (Discriminability Preservation)")
            disc_results = self._evaluate_discriminability_preservation()
            results.metrics['discriminability_preservation'] = disc_results
        
        return results
    
    def _evaluate_consistency(self, n_samples: int) -> Dict:
        """评估输出稳定性"""
        task_consistencies = {}
        
        for task in self.tasks:
            # 多次采样Prior
            z_goals = [self._get_prior_embed(task.instruction) 
                      for _ in range(n_samples)]
            
            # 计算两两相似度
            similarities = []
            for i in range(len(z_goals)):
                for j in range(i+1, len(z_goals)):
                    sim = 1 - cosine(z_goals[i], z_goals[j])
                    similarities.append(sim)
            
            consistency = float(np.mean(similarities))
            task_consistencies[task.task_id] = consistency
            
            logger.info(f"  {task.task_id}: {consistency:.4f}")
        
        avg_consistency = np.mean(list(task_consistencies.values()))
        
        return {
            'task_consistencies': task_consistencies,
            'avg_consistency': float(avg_consistency),
            'n_samples': n_samples,
            'interpretation': self._interpret_consistency(avg_consistency)
        }
    
    def _evaluate_semantic_robustness(self) -> Dict:
        """评估语义鲁棒性"""
        task_robustness = {}
        
        for task in self.tasks:
            if not task.instruction_variants:
                continue
            
            # 为每个变体生成Prior嵌入
            variants = [task.instruction] + task.instruction_variants
            z_goals = [self._get_prior_embed(v) for v in variants]
            
            # 计算类内相似度
            similarities = []
            for i in range(len(z_goals)):
                for j in range(i+1, len(z_goals)):
                    sim = 1 - cosine(z_goals[i], z_goals[j])
                    similarities.append(sim)
            
            robustness = float(np.mean(similarities))
            task_robustness[task.task_id] = {
                'robustness': robustness,
                'n_variants': len(variants)
            }
            
            logger.info(f"  {task.task_id}: {robustness:.4f} ({len(variants)} variants)")
        
        if task_robustness:
            avg_robustness = np.mean([r['robustness'] for r in task_robustness.values()])
        else:
            avg_robustness = 0.0
        
        return {
            'task_robustness': task_robustness,
            'avg_robustness': float(avg_robustness),
            'interpretation': self._interpret_robustness(avg_robustness)
        }
    
    def _evaluate_output_diversity(self) -> Dict:
        """评估输出多样性"""
        # 收集所有任务的Prior输出
        all_z_goals = []
        for task in self.tasks:
            z_goal = self._get_prior_embed(task.instruction)
            all_z_goals.append(z_goal)
        
        all_z_goals = np.array(all_z_goals)
        
        # 计算方差
        variance_per_dim = np.var(all_z_goals, axis=0)
        mean_variance = float(np.mean(variance_per_dim))
        std_variance = float(np.std(variance_per_dim))
        
        logger.info(f"  均值方差: {mean_variance:.6f}")
        logger.info(f"  标准差: {std_variance:.6f}")
        
        return {
            'mean_variance': mean_variance,
            'std_variance': std_variance,
            'variance_per_dim': variance_per_dim.tolist(),
            'interpretation': self._interpret_diversity(mean_variance)
        }
    
    def _evaluate_discriminability_preservation(self) -> Dict:
        """评估区分度保持率"""
        # 收集文本嵌入和Prior输出
        text_embeds = []
        prior_embeds = []
        
        for task in self.tasks:
            # 文本嵌入
            with th.no_grad():
                text_embed = self.mineclip.encode_text([task.instruction])[0]
                text_embeds.append(text_embed.cpu().numpy())
            
            # Prior嵌入
            prior_embed = self._get_prior_embed(task.instruction)
            prior_embeds.append(prior_embed)
        
        text_embeds = np.array(text_embeds)
        prior_embeds = np.array(prior_embeds)
        
        # 计算文本嵌入的区分度
        text_disc = self._compute_discriminability(text_embeds)
        
        # 计算Prior输出的区分度
        prior_disc = self._compute_discriminability(prior_embeds)
        
        # 保持率
        preservation_rate = prior_disc / (text_disc + 1e-6)
        
        logger.info(f"  文本区分度: {text_disc:.4f}")
        logger.info(f"  Prior区分度: {prior_disc:.4f}")
        logger.info(f"  保持率: {preservation_rate:.2f}x")
        
        return {
            'text_discriminability': float(text_disc),
            'prior_discriminability': float(prior_disc),
            'preservation_rate': float(preservation_rate),
            'interpretation': self._interpret_preservation(preservation_rate)
        }
    
    def _compute_discriminability(self, embeds: np.ndarray) -> float:
        """计算嵌入的区分度"""
        n = len(embeds)
        if n < 2:
            return 1.0
        
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                sim = 1 - cosine(embeds[i], embeds[j])
                similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        discriminability = 1 - mean_sim
        
        return float(discriminability)
    
    # =========================================================================
    # 维度2: 输出质量 (Output Quality)
    # =========================================================================
    
    def evaluate_output_quality(self) -> DimensionResults:
        """评估输出质量"""
        logger.info("\n" + "=" * 80)
        logger.info("维度2: 输出质量评估")
        logger.info("=" * 80)
        
        dim_config = self.config['evaluation_dimensions']['output_quality']
        
        if not dim_config['enabled']:
            logger.info("维度已禁用，跳过")
            return DimensionResults(
                dimension_name="output_quality",
                enabled=False
            )
        
        results = DimensionResults(
            dimension_name="output_quality",
            enabled=True
        )
        
        # 指标2.1: 目标对齐度 (Goal Alignment)
        if dim_config['goal_alignment']['enabled']:
            logger.info("\n[指标2.1] 目标对齐度 (Goal Alignment)")
            alignment_results = self._evaluate_goal_alignment(
                use_reward_head=dim_config['goal_alignment']['use_reward_head']
            )
            results.metrics['goal_alignment'] = alignment_results
        
        # 指标2.2: Prior增益 (Prior Gain)
        if dim_config['prior_gain']['enabled']:
            logger.info("\n[指标2.2] Prior增益 (Prior Gain)")
            gain_results = self._evaluate_prior_gain()
            results.metrics['prior_gain'] = gain_results
        
        # 指标2.3: 跨模态一致性 (Cross-Modal Consistency)
        if dim_config['cross_modal_consistency']['enabled']:
            logger.info("\n[指标2.3] 跨模态一致性 (Cross-Modal Consistency)")
            consistency_results = self._evaluate_cross_modal_consistency()
            results.metrics['cross_modal_consistency'] = consistency_results
        
        return results
    
    def _evaluate_goal_alignment(self, use_reward_head: bool = True) -> Dict:
        """评估目标对齐度"""
        task_alignments = {}
        
        for task in self.tasks:
            # 加载成功画面
            success_visuals = self._load_success_visuals(task)
            
            if success_visuals is None or len(success_visuals) == 0:
                logger.warning(f"  {task.task_id}: 无成功画面数据，跳过")
                continue
            
            # 获取Prior输出
            z_goal = self._get_prior_embed(task.instruction)
            
            if use_reward_head:
                # 使用余弦相似度（直接在0-1范围，易于理解）
                import torch.nn.functional as F
                z_goal_tensor = th.from_numpy(z_goal).float().unsqueeze(0).to(DEVICE)
                z_visuals_tensor = th.from_numpy(success_visuals).float().to(DEVICE)
                
                with th.no_grad():
                    # L2归一化
                    z_goal_norm = F.normalize(z_goal_tensor, p=2, dim=1)
                    z_visuals_norm = F.normalize(z_visuals_tensor, p=2, dim=1)
                    
                    # 计算余弦相似度: [1, N]
                    cosine_sim = th.matmul(z_goal_norm, z_visuals_norm.T)
                    
                    # 转换到0-1范围: (cosine_sim + 1) / 2
                    # 因为cosine范围是[-1, 1]
                    similarities = ((cosine_sim + 1) / 2).squeeze(0).cpu().numpy()
            else:
                # 使用余弦相似度
                similarities = []
                for z_visual in success_visuals:
                    sim = 1 - cosine(z_goal, z_visual)
                    similarities.append(sim)
                similarities = np.array(similarities)
            
            goal_alignment_mean = float(np.mean(similarities))
            goal_alignment_std = float(np.std(similarities))
            
            task_alignments[task.task_id] = {
                'mean': goal_alignment_mean,
                'std': goal_alignment_std,
                'n_visuals': len(success_visuals)
            }
            
            logger.info(f"  {task.task_id}: {goal_alignment_mean:.4f} ± {goal_alignment_std:.4f}")
        
        if task_alignments:
            avg_alignment = np.mean([a['mean'] for a in task_alignments.values()])
        else:
            avg_alignment = 0.0
        
        return {
            'task_alignments': task_alignments,
            'avg_alignment': float(avg_alignment),
            'use_reward_head': use_reward_head,
            'interpretation': self._interpret_alignment(avg_alignment)
        }
    
    def _evaluate_prior_gain(self) -> Dict:
        """评估Prior增益"""
        task_gains = {}
        
        for task in self.tasks:
            # 加载成功画面
            success_visuals = self._load_success_visuals(task)
            
            if success_visuals is None or len(success_visuals) == 0:
                continue
            
            # Prior路径
            z_prior = self._get_prior_embed(task.instruction)
            sims_prior = [1 - cosine(z_prior, z_v) for z_v in success_visuals]
            alignment_prior = np.mean(sims_prior)
            
            # 直接文本路径
            with th.no_grad():
                z_text = self.mineclip.encode_text([task.instruction])[0].cpu().numpy()
            sims_text = [1 - cosine(z_text, z_v) for z_v in success_visuals]
            alignment_text = np.mean(sims_text)
            
            # Prior增益
            gain = alignment_prior - alignment_text
            
            task_gains[task.task_id] = {
                'alignment_prior': float(alignment_prior),
                'alignment_text': float(alignment_text),
                'gain': float(gain)
            }
            
            logger.info(f"  {task.task_id}: Prior={alignment_prior:.4f}, Text={alignment_text:.4f}, Gain={gain:+.4f}")
        
        if task_gains:
            avg_gain = np.mean([g['gain'] for g in task_gains.values()])
        else:
            avg_gain = 0.0
        
        return {
            'task_gains': task_gains,
            'avg_gain': float(avg_gain),
            'interpretation': self._interpret_gain(avg_gain)
        }
    
    def _evaluate_cross_modal_consistency(self) -> Dict:
        """评估跨模态一致性"""
        # 收集Prior输出和真实视觉嵌入
        prior_embeds = []
        visual_embeds = []
        
        for task in self.tasks:
            # Prior嵌入
            z_prior = self._get_prior_embed(task.instruction)
            prior_embeds.append(z_prior)
            
            # 视觉嵌入
            success_visuals = self._load_success_visuals(task)
            if success_visuals is not None and len(success_visuals) > 0:
                visual_embeds.extend(success_visuals)
        
        if len(prior_embeds) == 0 or len(visual_embeds) == 0:
            logger.warning("  无足够数据进行跨模态一致性分析")
            return {
                'consistency_score': 0.0,
                'mean_wasserstein_distance_normalized': 0.0,
                'interpretation': '无数据'
            }
        
        prior_embeds = np.array(prior_embeds)
        visual_embeds = np.array(visual_embeds)
        
        # 计算Wasserstein距离（每个维度）
        distances = []
        for dim in range(512):
            dist = wasserstein_distance(
                visual_embeds[:, dim],
                prior_embeds[:, dim]
            )
            distances.append(dist)
        
        mean_distance = float(np.mean(distances))
        
        # 归一化到0-1区间
        # 使用sigmoid函数将距离映射到0-1
        # consistency_score越高表示越一致（距离越小）
        consistency_score = float(1 / (1 + mean_distance))
        
        # 同时提供归一化后的距离（0-1区间，0表示完全一致）
        normalized_distance = float(1 - consistency_score)
        
        logger.info(f"  平均Wasserstein距离: {mean_distance:.4f}")
        logger.info(f"  归一化距离: {normalized_distance:.4f} (0-1区间)")
        logger.info(f"  一致性得分: {consistency_score:.4f} (0-1区间)")
        
        return {
            'mean_wasserstein_distance': mean_distance,
            'mean_wasserstein_distance_normalized': normalized_distance,
            'consistency_score': consistency_score,
            'interpretation': self._interpret_cross_modal(consistency_score)
        }
    
    # =========================================================================
    # 维度3: 可控性 (Controllability)
    # =========================================================================
    
    def evaluate_controllability(self) -> DimensionResults:
        """评估可控性"""
        logger.info("\n" + "=" * 80)
        logger.info("维度3: 可控性评估")
        logger.info("=" * 80)
        
        dim_config = self.config['evaluation_dimensions']['controllability']
        
        if not dim_config['enabled']:
            logger.info("维度已禁用，跳过")
            return DimensionResults(
                dimension_name="controllability",
                enabled=False
            )
        
        results = DimensionResults(
            dimension_name="controllability",
            enabled=True
        )
        
        # 指标3.2: CFG敏感度 (CFG Sensitivity)
        if dim_config['cfg_sensitivity']['enabled']:
            logger.info("\n[指标3.2] CFG敏感度 (CFG Sensitivity)")
            cfg_results = self._evaluate_cfg_sensitivity(
                cfg_scales=dim_config['cfg_sensitivity']['cfg_scales']
            )
            results.metrics['cfg_sensitivity'] = cfg_results
        
        return results
    
    def _evaluate_cfg_sensitivity(self, cfg_scales: List[float]) -> Dict:
        """
        评估CFG敏感度
        
        注意：当前版本的 get_prior_embed 不支持 cond_scale 参数
        这个指标需要在 Policy 层面进行评估
        暂时禁用此功能
        """
        logger.warning("⚠️ CFG敏感度评估需要Policy层面支持，当前版本暂不支持")
        logger.warning("   建议：在端到端评估时测试不同CFG scale的影响")
        
        # 返回占位符结果
        return {
            'cfg_scales': cfg_scales,
            'task_cfg_analysis': {},
            'interpretation': 'CFG敏感度评估需要Policy支持，暂时跳过',
            'note': 'get_prior_embed 不支持 cond_scale 参数，此指标需要在完整的 agent 评估中测试'
        }
    
    # =========================================================================
    # 解释函数
    # =========================================================================
    
    def _interpret_consistency(self, value: float) -> str:
        """解释一致性得分"""
        thresholds = self.config['thresholds']['consistency']
        if value >= thresholds['excellent']:
            return f"优秀 (≥{thresholds['excellent']})"
        elif value >= thresholds['good']:
            return f"良好 (≥{thresholds['good']})"
        else:
            return f"需改进 (<{thresholds['poor']})"
    
    def _interpret_robustness(self, value: float) -> str:
        """解释鲁棒性得分"""
        thresholds = self.config['thresholds']['semantic_robustness']
        if value >= thresholds['excellent']:
            return f"优秀 (≥{thresholds['excellent']})"
        elif value >= thresholds['good']:
            return f"良好 (≥{thresholds['good']})"
        else:
            return f"需改进 (<{thresholds['poor']})"
    
    def _interpret_diversity(self, value: float) -> str:
        """解释多样性得分"""
        threshold = self.config['thresholds']['output_diversity']['excellent']
        if value >= threshold:
            return f"优秀 (≥{threshold})"
        else:
            return f"需改进 (<{threshold})"
    
    def _interpret_preservation(self, value: float) -> str:
        """解释保持率"""
        if value > 1.0:
            return f"优秀 (Prior放大了区分度 {value:.2f}x)"
        elif value >= 0.8:
            return f"良好 (保持了输入区分度)"
        else:
            return f"需改进 (降低了区分度)"
    
    def _interpret_alignment(self, value: float) -> str:
        """解释对齐度得分"""
        thresholds = self.config['thresholds']['goal_alignment']
        if value >= thresholds['excellent']:
            return f"优秀 (≥{thresholds['excellent']})"
        elif value >= thresholds['good']:
            return f"良好 (≥{thresholds['good']})"
        else:
            return f"需改进 (<{thresholds['poor']})"
    
    def _interpret_gain(self, value: float) -> str:
        """解释Prior增益"""
        threshold = self.config['thresholds']['prior_gain']['excellent']
        if value >= threshold:
            return f"优秀 (Prior显著优于直接文本)"
        elif value > 0:
            return f"良好 (Prior略优于直接文本)"
        else:
            return f"需改进 (Prior未带来改善)"
    
    def _interpret_cross_modal(self, value: float) -> str:
        """解释跨模态一致性"""
        if value >= 0.5:
            return "良好 (Prior输出在视觉空间)"
        else:
            return "需改进 (Prior输出偏离视觉空间)"
    
    # =========================================================================
    # 主评估流程
    # =========================================================================
    
    def run_evaluation(self) -> PriorEvaluationResults:
        """运行完整评估"""
        logger.info("\n" + "=" * 80)
        logger.info("开始 Prior 模型完整评估")
        logger.info("=" * 80)
        logger.info(f"配置文件: {self.config_path}")
        logger.info(f"评估任务: {len(self.tasks)} 个")
        logger.info("=" * 80)
        
        results = PriorEvaluationResults(
            config_file=self.config_path,
            n_tasks=len(self.tasks),
            task_ids=[t.task_id for t in self.tasks]
        )
        
        # 评估三个维度
        results.intrinsic_quality = self.evaluate_intrinsic_quality()
        results.output_quality = self.evaluate_output_quality()
        results.controllability = self.evaluate_controllability()
        
        # 生成总结
        results.summary = self._generate_summary(results)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Prior 模型评估完成")
        logger.info("=" * 80)
        
        return results
    
    def _generate_summary(self, results: PriorEvaluationResults) -> Dict:
        """生成评估总结"""
        summary = {
            'n_tasks': results.n_tasks,
            'dimensions_evaluated': []
        }
        
        if results.intrinsic_quality and results.intrinsic_quality.enabled:
            summary['dimensions_evaluated'].append('intrinsic_quality')
            summary['intrinsic_quality_summary'] = {
                'avg_consistency': results.intrinsic_quality.metrics.get('consistency', {}).get('avg_consistency'),
                'avg_semantic_robustness': results.intrinsic_quality.metrics.get('semantic_robustness', {}).get('avg_robustness'),
                'mean_variance': results.intrinsic_quality.metrics.get('output_diversity', {}).get('mean_variance'),
            }
        
        if results.output_quality and results.output_quality.enabled:
            summary['dimensions_evaluated'].append('output_quality')
            summary['output_quality_summary'] = {
                'avg_goal_alignment': results.output_quality.metrics.get('goal_alignment', {}).get('avg_alignment'),
                'avg_prior_gain': results.output_quality.metrics.get('prior_gain', {}).get('avg_gain'),
            }
        
        if results.controllability and results.controllability.enabled:
            summary['dimensions_evaluated'].append('controllability')
        
        return summary
    
    def generate_visualizations(self, results: PriorEvaluationResults, output_dir: Path):
        """生成可视化图表"""
        if not self.config['global']['report']['generate_plots']:
            return
        
        logger.info("\n生成可视化图表...")
        
        # 收集所有Prior嵌入
        prior_embeds = []
        task_labels = []
        
        for task in self.tasks:
            z_prior = self._get_prior_embed(task.instruction)
            prior_embeds.append(z_prior)
            task_labels.append(task.task_id)
        
        if len(prior_embeds) < 2:
            logger.warning("  任务数少于2个，跳过可视化")
            return
        
        prior_embeds = np.array(prior_embeds)
        
        # 创建图表 (2x2 布局)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Prior 嵌入可视化分析', fontsize=16, fontweight='bold')
        
        # 1. 相似度矩阵热图
        ax1 = axes[0, 0]
        self._plot_similarity_matrix(prior_embeds, task_labels, ax1)
        
        # 2. t-SNE可视化
        ax2 = axes[0, 1]
        self._plot_tsne(prior_embeds, task_labels, ax2)
        
        # 3. PCA可视化
        ax3 = axes[1, 0]
        self._plot_pca(prior_embeds, task_labels, ax3)
        
        # 4. 方差分布
        ax4 = axes[1, 1]
        self._plot_variance_distribution(prior_embeds, ax4)
        
        plt.tight_layout()
        
        # 保存
        viz_path = output_dir / "prior_evaluation_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 可视化已保存: {viz_path}")
    
    def _plot_similarity_matrix(self, embeds: np.ndarray, labels: List[str], ax):
        """绘制相似度矩阵"""
        n = len(embeds)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = 1 - cosine(embeds[i], embeds[j])
        
        im = ax.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_title('Prior 输出相似度矩阵')
        
        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相似度 (0-1)', rotation=270, labelpad=20)
        
        # 在格子中显示数值
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    def _plot_tsne(self, embeds: np.ndarray, labels: List[str], ax):
        """绘制t-SNE降维"""
        if len(embeds) < 3:
            ax.text(0.5, 0.5, '任务数少于3个\n无法进行t-SNE', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('t-SNE 可视化')
            return
        
        try:
            perplexity = min(30, max(2, len(embeds) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(embeds)
            
            ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)
            
            for i, label in enumerate(labels):
                ax.annotate(label, (coords[i, 0], coords[i, 1]), 
                          fontsize=8, ha='center')
            
            ax.set_xlabel('t-SNE 维度 1')
            ax.set_ylabel('t-SNE 维度 2')
            ax.set_title('t-SNE 可视化')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f't-SNE 失败:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('t-SNE 可视化')
    
    def _plot_pca(self, embeds: np.ndarray, labels: List[str], ax):
        """绘制PCA降维"""
        try:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeds)
            
            ax.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)
            
            for i, label in enumerate(labels):
                ax.annotate(label, (coords[i, 0], coords[i, 1]), 
                          fontsize=8, ha='center')
            
            var1, var2 = pca.explained_variance_ratio_
            ax.set_xlabel(f'PC1 ({var1*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var2*100:.1f}%)')
            ax.set_title('PCA 可视化')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'PCA 失败:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA 可视化')
    
    def _plot_variance_distribution(self, embeds: np.ndarray, ax):
        """绘制方差分布"""
        variance_per_dim = np.var(embeds, axis=0)
        
        ax.hist(variance_per_dim, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(variance_per_dim), color='red', 
                  linestyle='--', linewidth=2, label=f'均值: {np.mean(variance_per_dim):.6f}')
        ax.set_xlabel('方差 (0-1区间)')
        ax.set_ylabel('维度数量')
        ax.set_title('Prior 输出方差分布（跨512维）')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_results(self, results: PriorEvaluationResults, output_dir: str):
        """保存评估结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成可视化
        if self.config['global']['report']['generate_plots']:
            self.generate_visualizations(results, output_dir)
        
        # 保存JSON
        json_path = output_dir / "prior_evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ JSON结果已保存: {json_path}")
        
        # 生成HTML报告
        if self.config['global']['report']['generate_html']:
            from src.utils.prior_html_generator import PriorHTMLGenerator
            
            html_generator = PriorHTMLGenerator(output_dir)
            html_path = html_generator.generate_report(
                results.to_dict(),
                output_filename="prior_evaluation_report.html"
            )
            logger.info(f"✓ HTML报告已生成: {html_path}")
        
        return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Prior模型评估框架 V2（配置驱动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估所有启用的任务集
  python src/evaluation/prior_eval_framework.py --config config/eval_tasks_prior.yaml
  
  # 评估指定的任务集
  python src/evaluation/prior_eval_framework.py --config config/eval_tasks_prior.yaml --task-set harvest
  
  # 评估多个任务集
  python src/evaluation/prior_eval_framework.py --config config/eval_tasks_prior.yaml --task-set harvest,combat
  
  # 评估指定的任务ID
  python src/evaluation/prior_eval_framework.py --config config/eval_tasks_prior.yaml --task-ids harvest_1_log harvest_1_dirt
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML配置文件路径'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='输出目录（覆盖配置文件中的设置）'
    )
    
    parser.add_argument(
        '--task-set',
        type=str,
        help='指定要评估的任务集（逗号分隔，如 harvest,combat,techtree）'
    )
    
    parser.add_argument(
        '--task-ids',
        type=str,
        nargs='+',
        help='指定要评估的任务ID列表（如 harvest_1_log harvest_1_dirt）'
    )
    
    args = parser.parse_args()
    
    # 检查参数冲突
    if args.task_set and args.task_ids:
        logger.warning("同时指定了 --task-set 和 --task-ids，将优先使用 --task-ids")
    
    try:
        # 初始化框架
        framework = PriorEvaluationFramework(
            config_path=args.config,
            task_ids=args.task_ids,
            task_set=args.task_set
        )
        
        # 运行评估
        results = framework.run_evaluation()
        
        # 保存结果
        output_dir = args.output_dir or framework.config['global']['output_dir']
        framework.save_results(results, output_dir)
        
        # 打印总结
        print("\n" + "=" * 80)
        print("评估总结")
        print("=" * 80)
        print(f"任务数: {results.n_tasks}")
        print(f"评估维度: {', '.join(results.summary['dimensions_evaluated'])}")
        
        if 'intrinsic_quality_summary' in results.summary:
            iq = results.summary['intrinsic_quality_summary']
            print("\n[内在质量]")
            if iq.get('avg_consistency'):
                print(f"  平均一致性: {iq['avg_consistency']:.4f}")
            if iq.get('avg_semantic_robustness'):
                print(f"  平均语义鲁棒性: {iq['avg_semantic_robustness']:.4f}")
        
        if 'output_quality_summary' in results.summary:
            oq = results.summary['output_quality_summary']
            print("\n[输出质量]")
            if oq.get('avg_goal_alignment'):
                print(f"  平均目标对齐度: {oq['avg_goal_alignment']:.4f}")
            if oq.get('avg_prior_gain'):
                print(f"  平均Prior增益: {oq['avg_prior_gain']:+.4f}")
        
        print("=" * 80)
        print(f"\n✓ 结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"评估失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
