"""
Steve1 Prior 模型分析器（Prior Analyzer）
=========================================

专门评估 Prior 模型 p(z_goal|y)：文本 → 目标嵌入

【策略评估 vs 结果评估】
- 策略评估（本模块）：分析 Prior 模型的嵌入空间质量
  * 文本-Prior 对齐度
  * Prior 输出方差和多样性
  * VAE 重建质量
  
- 结果评估（STEVE1Evaluator）：评估任务成功率

作者: AI Assistant
日期: 2025-11-27
"""

import torch as th
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

from scipy.spatial.distance import cosine

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


class Steve1PriorEvaluator:
    """
    Steve1 Prior 评估器（Prior Evaluator）
    
    专门评估 Prior 模型 p(z_goal|y) 的性能：
    - 文本嵌入到Prior嵌入的转换质量
    - 文本-Prior相似度
    - 嵌入空间的结构和分布
    - VAE重建质量
    """
    
    def __init__(
        self,
        prior_weights: str = "data/weights/steve1/steve1_prior.pt",
        seed: int = 42,
    ):
        """
        初始化 Prior 分析器
        
        Args:
            prior_weights: Prior VAE 权重路径
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        th.manual_seed(seed)
        
        logger.info("=" * 80)
        logger.info("初始化 Steve1PriorEvaluator...")
        logger.info("=" * 80)
        
        # 加载 MineCLIP
        logger.info("加载 MineCLIP...")
        self._mineclip = load_mineclip_wconfig()
        logger.info(f"✓ MineCLIP 已加载")
        
        # 加载 Prior (VAE)
        logger.info(f"加载 Prior VAE: {prior_weights}")
        self._prior = load_vae_model(prior_weights, DEVICE)
        logger.info(f"✓ Prior VAE 已加载")
        
        logger.info("=" * 80)
        logger.info("✓ Steve1PriorAnalyzer 初始化完成")
        logger.info("=" * 80)
    
    def analyze_prior_model(
        self,
        instructions: List[str],
        n_samples: int = 10,
        output_dir: Optional[Path] = None
    ) -> List[PriorAnalysisResult]:
        """
        分析 Prior 模型在给定指令集上的性能
        
        Args:
            instructions: 要分析的指令列表
            n_samples: 每个指令采样的 Prior 样本数
            output_dir: 输出目录（保存结果）
            
        Returns:
            每个指令的分析结果列表
        """
        logger.info(f"开始分析 Prior 模型 ({len(instructions)} 个指令, 每个采样 {n_samples} 次)...")
        
        results = []
        all_text_embeds = []
        all_prior_embeds_mean = []
        
        for instruction in instructions:
            logger.info(f"分析指令: '{instruction}'")
            
            # 1. 获取文本嵌入
            with th.no_grad():
                text_cond = self._mineclip.encode_text(instruction).to(DEVICE)
            
            if isinstance(text_cond, th.Tensor):
                text_embed = text_cond.cpu().numpy()
            else:
                text_embed = text_cond
            
            text_embed = np.squeeze(text_embed)  # 确保是1-D
            text_embed_dim = text_embed.shape[0]
            all_text_embeds.append(text_embed)
            
            # 2. 采样 Prior 嵌入
            prior_samples = []
            for _ in range(n_samples):
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
                
                prior_embed = np.squeeze(prior_embed)  # 确保是1-D
                prior_samples.append(prior_embed)
            
            prior_samples = np.array(prior_samples)
            prior_embed_dim = prior_samples.shape[1]
            
            # 3. 计算统计指标
            prior_mean = prior_samples.mean(axis=0)
            prior_var = prior_samples.var(axis=0).mean()  # 平均方差
            
            # 文本-Prior 相似度（余弦相似度）
            text_to_prior_sim = 1 - cosine(text_embed, prior_mean)
            
            # VAE 重建质量（暂时用相似度近似）
            reconstruction_quality = text_to_prior_sim
            
            all_prior_embeds_mean.append(prior_mean)
            
            # 4. 创建结果
            result = PriorAnalysisResult(
                text_instruction=instruction,
                text_embed_dim=text_embed_dim,
                prior_embed_dim=prior_embed_dim,
                text_to_prior_similarity=text_to_prior_sim,
                prior_variance=prior_var,
                reconstruction_quality=reconstruction_quality,
                nearest_neighbors=[],
                cluster_id=None,
            )
            
            results.append(result)
        
        # 5. 计算最近邻
        all_prior_embeds_mean = np.array(all_prior_embeds_mean)
        for i, result in enumerate(results):
            # 计算与其他指令的距离
            distances = []
            for j, other_embed in enumerate(all_prior_embeds_mean):
                if i != j:
                    sim = 1 - cosine(all_prior_embeds_mean[i], other_embed)
                    distances.append((instructions[j], sim))
            
            # 排序并取前3
            distances.sort(key=lambda x: x[1], reverse=True)
            result.nearest_neighbors = distances[:3]
        
        # 6. 保存结果
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存 JSON
            results_dict = [r.to_dict() for r in results]
            with open(output_dir / "prior_analysis.json", 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            # 保存指令列表
            with open(output_dir / "instructions.json", 'w') as f:
                json.dump(instructions, f, indent=2)
            
            logger.info(f"✓ Prior 分析结果已保存到: {output_dir}")
        
        return results
    
    def __del__(self):
        """清理资源"""
        logger.info("Steve1PriorEvaluator 清理完成")

