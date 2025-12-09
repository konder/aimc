"""
评估指标计算
Evaluation Metrics - Compute success rates and other metrics
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class TrialResult:
    """单次试验结果"""
    task_id: str
    language: str           # 'en', 'zh_auto', 'zh_manual', 'zh_variant_N'
    instruction: str        # 使用的指令文本
    success: bool          # 是否成功
    steps: int             # 使用的步数
    time_seconds: float    # 使用的时间（秒）
    trajectory: List = field(default_factory=list)  # 轨迹数据（可选）
    final_inventory: Dict[str, int] = field(default_factory=dict)  # 最终库存
    actions: List[Dict] = field(default_factory=list)  # 动作序列（用于 Policy 评估）
    rewards: List[float] = field(default_factory=list)  # 奖励序列（用于 Policy 评估）
    total_reward: float = 0.0  # 总奖励
    
    # 目标接近度指标（Goal Progress）
    goal_distances: List[float] = field(default_factory=list)  # 采样帧到目标的距离序列
    goal_progress_rate: float = 0.0  # 进度率: (初始距离 - 最终距离) / 初始距离
    goal_monotonic_rate: float = 0.0  # 单调率: 距离递减的步数比例
    goal_initial_distance: float = 0.0  # 初始距离
    goal_final_distance: float = 0.0  # 最终距离
    goal_sample_indices: List[int] = field(default_factory=list)  # 采样帧索引


@dataclass
class TaskResult:
    """单个任务的评估结果"""
    task_id: str
    language: str
    instruction: str
    trials: List[TrialResult]
    
    # 统计指标
    success_rate: float = 0.0
    avg_steps: float = 0.0
    std_steps: float = 0.0
    avg_time: float = 0.0
    
    def __post_init__(self):
        """计算统计指标"""
        if self.trials:
            self.success_rate = sum(t.success for t in self.trials) / len(self.trials)
            
            successful_trials = [t for t in self.trials if t.success]
            if successful_trials:
                steps = [t.steps for t in successful_trials]
                times = [t.time_seconds for t in successful_trials]
                self.avg_steps = np.mean(steps)
                self.std_steps = np.std(steps)
                self.avg_time = np.mean(times)


class EvaluationMetrics:
    """评估指标计算器"""
    
    @staticmethod
    def compute_task_success_rate(results: List[TrialResult]) -> float:
        """
        计算任务成功率
        
        Args:
            results: 试验结果列表
            
        Returns:
            成功率 (0.0 到 1.0)
        """
        if not results:
            return 0.0
        return sum(r.success for r in results) / len(results)
    
    @staticmethod
    def compute_efficiency(results: List[TrialResult], 
                          expert_steps: int = None) -> float:
        """
        计算效率指标
        
        Args:
            results: 试验结果列表
            expert_steps: 专家步数（可选）
            
        Returns:
            效率分数
        """
        successful = [r for r in results if r.success]
        if not successful:
            return 0.0
        
        avg_steps = np.mean([r.steps for r in successful])
        
        if expert_steps:
            return min(1.0, expert_steps / avg_steps)
        else:
            return 1.0  # 没有专家数据时返回1.0
    
    @staticmethod
    def compute_language_equivalence_gap(
        en_results: TaskResult,
        zh_results: TaskResult
    ) -> float:
        """
        计算语言等价性gap
        
        Args:
            en_results: 英文结果
            zh_results: 中文结果
            
        Returns:
            gap值 (0.0 表示完全等价)
        """
        return abs(en_results.success_rate - zh_results.success_rate)
    
    @staticmethod
    def compute_semantic_variance(variant_results: List[TaskResult]) -> float:
        """
        计算语义变体鲁棒性（方差）
        
        Args:
            variant_results: 不同语义变体的结果列表
            
        Returns:
            方差 (越小越好)
        """
        if len(variant_results) < 2:
            return 0.0
        
        success_rates = [r.success_rate for r in variant_results]
        return float(np.std(success_rates))
    
    @staticmethod
    def compare_results(
        en_result: TaskResult,
        zh_auto_result: TaskResult,
        zh_manual_result: TaskResult = None,
        zh_variant_results: List[TaskResult] = None
    ) -> Dict[str, Any]:
        """
        全面对比不同语言/方法的结果
        
        Args:
            en_result: 英文结果 (baseline)
            zh_auto_result: 中文自动翻译结果
            zh_manual_result: 中文人工翻译结果（可选）
            zh_variant_results: 中文语义变体结果列表（可选）
            
        Returns:
            对比分析字典
        """
        comparison = {
            'task_id': en_result.task_id,
            'en_success_rate': en_result.success_rate,
            'en_avg_steps': en_result.avg_steps,
            'zh_auto_success_rate': zh_auto_result.success_rate,
            'zh_auto_avg_steps': zh_auto_result.avg_steps,
        }
        
        # 维度1: 自动翻译质量gap
        comparison['translation_quality_gap'] = EvaluationMetrics.compute_language_equivalence_gap(
            en_result, zh_auto_result
        )
        
        # 维度2: 语义等价性验证
        if zh_manual_result:
            comparison['zh_manual_success_rate'] = zh_manual_result.success_rate
            comparison['zh_manual_avg_steps'] = zh_manual_result.avg_steps
            comparison['semantic_equivalence_gap'] = EvaluationMetrics.compute_language_equivalence_gap(
                en_result, zh_manual_result
            )
        
        # 维度3: 语义变体鲁棒性
        if zh_variant_results:
            variant_rates = [r.success_rate for r in zh_variant_results]
            comparison['zh_variant_success_rates'] = variant_rates
            comparison['zh_variant_avg'] = np.mean(variant_rates)
            comparison['semantic_variance'] = EvaluationMetrics.compute_semantic_variance(
                zh_variant_results
            )
        
        return comparison
    
    @staticmethod
    def aggregate_results(task_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合多个任务的结果
        
        Args:
            task_comparisons: 任务对比结果列表
            
        Returns:
            聚合统计字典
        """
        if not task_comparisons:
            return {}
        
        aggregate = {
            'total_tasks': len(task_comparisons),
            'overall_en_success_rate': np.mean([c['en_success_rate'] for c in task_comparisons]),
            'overall_zh_auto_success_rate': np.mean([c['zh_auto_success_rate'] for c in task_comparisons]),
            'overall_translation_gap': np.mean([c['translation_quality_gap'] for c in task_comparisons]),
        }
        
        # 如果有人工翻译数据
        if 'semantic_equivalence_gap' in task_comparisons[0]:
            aggregate['overall_zh_manual_success_rate'] = np.mean([
                c['zh_manual_success_rate'] for c in task_comparisons
            ])
            aggregate['overall_semantic_equivalence_gap'] = np.mean([
                c['semantic_equivalence_gap'] for c in task_comparisons
            ])
        
        # 如果有语义变体数据
        if 'semantic_variance' in task_comparisons[0]:
            aggregate['overall_semantic_variance'] = np.mean([
                c['semantic_variance'] for c in task_comparisons
            ])
        
        return aggregate
    
    @staticmethod
    def format_comparison_table(task_comparisons: List[Dict[str, Any]]) -> str:
        """
        格式化对比表格（用于打印）
        
        Args:
            task_comparisons: 任务对比结果列表
            
        Returns:
            格式化的表格字符串
        """
        lines = []
        lines.append("\n" + "="*100)
        lines.append("评估结果对比表 (Evaluation Results Comparison)")
        lines.append("="*100)
        
        # 表头
        header = f"{'Task ID':<30} {'EN':>8} {'ZH(Auto)':>10} {'ZH(Man)':>10} {'Gap':>8} {'Var':>8}"
        lines.append(header)
        lines.append("-"*100)
        
        # 数据行
        for comp in task_comparisons:
            task_id = comp['task_id'][:28]  # 截断长ID
            en_rate = f"{comp['en_success_rate']*100:.1f}%"
            zh_auto_rate = f"{comp['zh_auto_success_rate']*100:.1f}%"
            
            # 人工翻译（可选）
            if 'zh_manual_success_rate' in comp:
                zh_manual_rate = f"{comp['zh_manual_success_rate']*100:.1f}%"
            else:
                zh_manual_rate = "N/A"
            
            gap = f"{comp['translation_quality_gap']*100:.1f}%"
            
            # 语义方差（可选）
            if 'semantic_variance' in comp:
                var = f"{comp['semantic_variance']*100:.1f}%"
            else:
                var = "N/A"
            
            line = f"{task_id:<30} {en_rate:>8} {zh_auto_rate:>10} {zh_manual_rate:>10} {gap:>8} {var:>8}"
            lines.append(line)
        
        lines.append("-"*100)
        
        # 汇总统计
        aggregate = EvaluationMetrics.aggregate_results(task_comparisons)
        if aggregate:
            lines.append(f"{'Overall Average':<30} "
                        f"{aggregate['overall_en_success_rate']*100:>7.1f}% "
                        f"{aggregate['overall_zh_auto_success_rate']*100:>9.1f}% "
                        f"{'N/A':>10} "
                        f"{aggregate['overall_translation_gap']*100:>7.1f}% "
                        f"{'N/A':>8}")
        
        lines.append("="*100)
        
        # 说明
        lines.append("\n说明:")
        lines.append("  EN:       英文baseline成功率")
        lines.append("  ZH(Auto): 中文自动翻译成功率")
        lines.append("  ZH(Man):  中文人工翻译成功率 (验证用)")
        lines.append("  Gap:      英文与中文自动翻译的差距 (越小越好)")
        lines.append("  Var:      中文语义变体方差 (越小越好)\n")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试代码
    
    # 创建模拟数据
    en_trials = [
        TrialResult("harvest_1_log", "en", "chop tree", True, 150, 30.0),
        TrialResult("harvest_1_log", "en", "chop tree", True, 180, 36.0),
        TrialResult("harvest_1_log", "en", "chop tree", False, 300, 60.0),
    ]
    
    zh_auto_trials = [
        TrialResult("harvest_1_log", "zh_auto", "砍树", True, 200, 40.0),
        TrialResult("harvest_1_log", "zh_auto", "砍树", False, 300, 60.0),
        TrialResult("harvest_1_log", "zh_auto", "砍树", False, 300, 60.0),
    ]
    
    en_result = TaskResult("harvest_1_log", "en", "chop tree", en_trials)
    zh_auto_result = TaskResult("harvest_1_log", "zh_auto", "砍树", zh_auto_trials)
    
    # 测试对比
    comparison = EvaluationMetrics.compare_results(en_result, zh_auto_result)
    print("对比结果:")
    for key, value in comparison.items():
        print(f"  {key}: {value}")
    
    # 测试格式化表格
    print(EvaluationMetrics.format_comparison_table([comparison]))

