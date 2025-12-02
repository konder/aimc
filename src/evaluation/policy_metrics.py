"""
策略评估指标（Policy Evaluation Metrics）
=========================================

统一定义和管理所有策略评估相关的指标：
- Prior模型指标: 文本-嵌入转换质量
- Policy模型指标: 动作执行质量
- 端到端指标: 瓶颈识别和贡献度

【设计原则】
1. 集中定义 - 所有指标定义在此统一管理
2. 标准化计算 - 统一的计算方法
3. 参考值体系 - 每个指标都有明确的参考值范围
4. 可扩展 - 易于添加新指标

作者: AI Assistant
日期: 2025-11-27
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import cosine


class MetricCategory(Enum):
    """指标类别"""
    PRIOR = "prior"  # Prior模型指标
    POLICY = "policy"  # Policy模型指标
    END_TO_END = "end_to_end"  # 端到端指标


@dataclass
class MetricDefinition:
    """指标定义元数据"""
    name: str  # 指标名称
    category: MetricCategory  # 指标类别
    description: str  # 指标描述
    unit: str = ""  # 单位
    
    # 参考值范围
    excellent_min: Optional[float] = None  # 优秀阈值（最小）
    excellent_max: Optional[float] = None  # 优秀阈值（最大）
    good_min: Optional[float] = None  # 良好阈值（最小）
    good_max: Optional[float] = None  # 良好阈值（最大）
    
    # 解释说明
    interpretation: str = ""  # 如何理解这个指标
    improvement_tips: str = ""  # 改进建议
    
    def evaluate_level(self, value: float) -> str:
        """
        评估指标水平
        
        Returns:
            "excellent", "good", "needs_improvement"
        """
        if self.excellent_min is not None and value >= self.excellent_min:
            if self.excellent_max is None or value <= self.excellent_max:
                return "excellent"
        
        if self.good_min is not None and value >= self.good_min:
            if self.good_max is None or value <= self.good_max:
                return "good"
        
        return "needs_improvement"


# =============================================================================
# 指标定义注册表
# =============================================================================

METRIC_DEFINITIONS = {
    # -------------------------------------------------------------------------
    # Prior 模型指标
    # -------------------------------------------------------------------------
    "text_to_prior_similarity": MetricDefinition(
        name="文本-Prior相似度",
        category=MetricCategory.PRIOR,
        description="MineCLIP文本嵌入与Prior输出的余弦相似度",
        unit="",
        excellent_min=0.5,
        good_min=0.3,
        good_max=1.0,
        interpretation="衡量Prior模型是否能准确地将文本转换为有效的目标嵌入。"
                      "越高表示Prior输出与文本语义越一致。",
        improvement_tips="<0.3时需要重新训练Prior模型，增加训练数据或调整text_cond_scale参数。"
    ),
    
    "prior_variance": MetricDefinition(
        name="Prior方差",
        category=MetricCategory.PRIOR,
        description="Prior输出的方差，衡量嵌入多样性",
        unit="",
        excellent_min=0.0001,
        excellent_max=0.001,
        good_min=0.00001,
        good_max=0.01,
        interpretation="适中的方差最佳。过高说明Prior输出不稳定，过低说明Prior缺乏表达能力。",
        improvement_tips="方差过高时增加Prior训练轮数，过低时增加VAE的latent维度。"
    ),
    
    "reconstruction_quality": MetricDefinition(
        name="重建质量",
        category=MetricCategory.PRIOR,
        description="VAE重建质量评分",
        unit="",
        excellent_min=0.5,
        good_min=0.3,
        good_max=1.0,
        interpretation="VAE能否准确重建输入。越高表示Prior能很好地保留文本信息。",
        improvement_tips="<0.3时需要增加VAE训练数据或调整VAE架构。"
    ),
    
    # -------------------------------------------------------------------------
    # Policy 模型指标
    # -------------------------------------------------------------------------
    "action_diversity": MetricDefinition(
        name="动作多样性",
        category=MetricCategory.POLICY,
        description="动作熵，衡量动作分布的多样性",
        unit="",
        excellent_min=1.5,
        excellent_max=2.5,
        good_min=1.0,
        good_max=3.0,
        interpretation="适中的多样性最佳。<1.0说明动作单调，>3.0说明动作混乱。",
        improvement_tips="动作单调时增加探索噪声，动作混乱时增加训练数据或调整Policy架构。"
    ),
    
    "temporal_consistency": MetricDefinition(
        name="时序一致性",
        category=MetricCategory.POLICY,
        description="相邻动作的一致性（平滑度）",
        unit="",
        excellent_min=0.85,
        good_min=0.7,
        good_max=1.0,
        interpretation="越高表示动作序列越平滑。<0.7说明动作抖动明显。",
        improvement_tips="<0.7时增加时序平滑或使用动作滤波器，检查goal_embed是否稳定。"
    ),
    
    "repeated_action_ratio": MetricDefinition(
        name="重复动作比例",
        category=MetricCategory.POLICY,
        description="连续重复相同动作的比例",
        unit="%",
        excellent_min=0.3,
        excellent_max=0.6,
        good_min=0.2,
        good_max=0.8,
        interpretation="适中的重复最佳。>80%可能卡住，<20%可能不稳定。",
        improvement_tips=">80%时检查是否陷入局部最优，<20%时增加动作平滑。"
    ),
    
    # -------------------------------------------------------------------------
    # 端到端指标
    # -------------------------------------------------------------------------
    "stage1_contribution": MetricDefinition(
        name="Prior贡献度",
        category=MetricCategory.END_TO_END,
        description="Prior模型对成功的贡献",
        unit="%",
        excellent_max=0.4,  # Prior贡献<40%说明不是瓶颈
        good_max=0.6,
        interpretation=">60%说明Prior很重要，是主要瓶颈，需优先优化Prior。",
        improvement_tips=">60%时优先优化Prior模型（提升相似度、降低方差）。"
    ),
    
    "stage2_contribution": MetricDefinition(
        name="Policy贡献度",
        category=MetricCategory.END_TO_END,
        description="Policy模型对成功的贡献",
        unit="%",
        excellent_max=0.4,  # Policy贡献<40%说明不是瓶颈
        good_max=0.6,
        interpretation=">60%说明Policy很重要，是主要瓶颈，需优先优化Policy。",
        improvement_tips=">60%时优先优化Policy模型（提升多样性、一致性）。"
    ),
}


# =============================================================================
# 指标计算类
# =============================================================================

class PolicyMetrics:
    """策略评估指标计算器"""
    
    # -------------------------------------------------------------------------
    # Prior 指标计算
    # -------------------------------------------------------------------------
    
    @staticmethod
    def compute_text_to_prior_similarity(
        text_embed: np.ndarray,
        prior_embed: np.ndarray
    ) -> float:
        """
        计算文本-Prior相似度
        
        Args:
            text_embed: 文本嵌入 [d]
            prior_embed: Prior嵌入 [d]
            
        Returns:
            余弦相似度 (0-1)
        """
        text_embed = np.squeeze(text_embed)
        prior_embed = np.squeeze(prior_embed)
        return float(1 - cosine(text_embed, prior_embed))
    
    @staticmethod
    def compute_prior_variance(prior_samples: np.ndarray) -> float:
        """
        计算Prior方差
        
        Args:
            prior_samples: Prior采样 [n_samples, d]
            
        Returns:
            平均方差
        """
        return float(np.var(prior_samples, axis=0).mean())
    
    @staticmethod
    def compute_reconstruction_quality(
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> float:
        """
        计算重建质量
        
        Args:
            original: 原始嵌入
            reconstructed: 重建嵌入
            
        Returns:
            相似度 (0-1)
        """
        original = np.squeeze(original)
        reconstructed = np.squeeze(reconstructed)
        return float(1 - cosine(original, reconstructed))
    
    # -------------------------------------------------------------------------
    # Policy 指标计算
    # -------------------------------------------------------------------------
    
    @staticmethod
    def compute_action_diversity(action_sequence: List[Dict]) -> float:
        """
        计算动作多样性（熵）
        
        Args:
            action_sequence: 动作序列，每个动作是一个dict
            
        Returns:
            动作熵
        """
        if not action_sequence:
            return 0.0
        
        # 提取所有动作维度
        all_actions = []
        for action in action_sequence:
            # 将动作转换为元组（可哈希）
            if isinstance(action, dict):
                action_tuple = tuple(sorted(action.items()))
            else:
                action_tuple = tuple(action) if hasattr(action, '__iter__') else (action,)
            all_actions.append(action_tuple)
        
        # 计算每个unique动作的频率
        unique_actions, counts = np.unique(all_actions, return_counts=True)
        probabilities = counts / len(all_actions)
        
        # 计算熵
        return float(scipy_entropy(probabilities, base=2))
    
    @staticmethod
    def compute_temporal_consistency(action_sequence: List[Dict]) -> float:
        """
        计算时序一致性
        
        Args:
            action_sequence: 动作序列
            
        Returns:
            一致性分数 (0-1)
        """
        if len(action_sequence) < 2:
            return 1.0
        
        # 计算相邻动作的相似度
        similarities = []
        for i in range(len(action_sequence) - 1):
            # 简化版：比较动作是否相同或相似
            action1 = action_sequence[i]
            action2 = action_sequence[i + 1]
            
            if isinstance(action1, dict) and isinstance(action2, dict):
                # 计算动作向量的余弦相似度
                keys = set(action1.keys()) | set(action2.keys())
                vec1 = np.array([action1.get(k, 0) for k in keys])
                vec2 = np.array([action2.get(k, 0) for k in keys])
                
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    @staticmethod
    def compute_repeated_action_ratio(action_sequence: List) -> float:
        """
        计算重复动作比例
        
        Args:
            action_sequence: 动作序列
            
        Returns:
            重复比例 (0-1)
        """
        if len(action_sequence) < 2:
            return 0.0
        
        repeated = 0
        for i in range(len(action_sequence) - 1):
            if isinstance(action_sequence[i], np.ndarray) and isinstance(action_sequence[i+1], np.ndarray):
                if np.array_equal(action_sequence[i], action_sequence[i+1]):
                    repeated += 1
            elif action_sequence[i] == action_sequence[i+1]:
                repeated += 1
        
        return repeated / (len(action_sequence) - 1)
    
    # -------------------------------------------------------------------------
    # 端到端指标计算
    # -------------------------------------------------------------------------
    
    @staticmethod
    def compute_stage_contributions(
        success_with_prior: bool,
        success_with_ground_truth: bool
    ) -> Tuple[float, float]:
        """
        计算两阶段的贡献度
        
        Args:
            success_with_prior: 使用Prior时是否成功
            success_with_ground_truth: 使用真实视觉嵌入时是否成功
            
        Returns:
            (stage1_contribution, stage2_contribution)
        """
        if success_with_ground_truth and not success_with_prior:
            # Policy能成功但Prior不行 → Prior是瓶颈
            return 0.7, 0.3
        elif not success_with_ground_truth:
            # 即使用真实视觉也失败 → Policy是瓶颈
            return 0.3, 0.7
        else:
            # 都成功 → 均衡
            return 0.5, 0.5
    
    @staticmethod
    def identify_bottleneck(
        stage1_contribution: float,
        stage2_contribution: float
    ) -> int:
        """
        识别瓶颈阶段
        
        Args:
            stage1_contribution: 阶段1贡献度
            stage2_contribution: 阶段2贡献度
            
        Returns:
            0=无瓶颈, 1=Prior瓶颈, 2=Policy瓶颈
        """
        if stage1_contribution > 0.6:
            return 1  # Prior瓶颈
        elif stage2_contribution > 0.6:
            return 2  # Policy瓶颈
        else:
            return 0  # 无明显瓶颈
    
    # -------------------------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_metric_definition(metric_name: str) -> Optional[MetricDefinition]:
        """获取指标定义"""
        return METRIC_DEFINITIONS.get(metric_name)
    
    @staticmethod
    def evaluate_metric_level(metric_name: str, value: float) -> str:
        """评估指标水平"""
        definition = METRIC_DEFINITIONS.get(metric_name)
        if definition:
            return definition.evaluate_level(value)
        return "unknown"
    
    @staticmethod
    def get_metrics_by_category(category: MetricCategory) -> Dict[str, MetricDefinition]:
        """获取某类别的所有指标"""
        return {
            name: defn
            for name, defn in METRIC_DEFINITIONS.items()
            if defn.category == category
        }
    
    @staticmethod
    def format_metric_value(metric_name: str, value: float) -> str:
        """格式化指标值"""
        definition = METRIC_DEFINITIONS.get(metric_name)
        if not definition:
            return f"{value:.4f}"
        
        if definition.unit == "%":
            return f"{value * 100:.1f}%"
        else:
            return f"{value:.4f}"


