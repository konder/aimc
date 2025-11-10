"""
评估报告生成器
Report Generator - Generate evaluation reports in various formats
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class ReportGenerator:
    """评估报告生成器"""
    
    def __init__(self, output_dir: str = "results/evaluation"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        comparisons: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            comparisons: 任务对比结果列表
            output_path: 输出路径（如果None则自动生成）
            
        Returns:
            报告文件路径
        """
        # 确定输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"evaluation_report_{timestamp}.json"
        else:
            output_path = self.output_dir / output_path
        
        # 构建完整报告
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(comparisons),
            },
            "tasks": comparisons,
            "summary": self._generate_summary(comparisons)
        }
        
        # 保存JSON报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 同时生成人类可读的文本报告
        txt_path = output_path.with_suffix('.txt')
        self._generate_text_report(report, txt_path)
        
        return str(output_path)
    
    def _generate_summary(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成摘要统计
        
        Args:
            comparisons: 任务对比结果列表
            
        Returns:
            摘要字典
        """
        if not comparisons:
            return {}
        
        import numpy as np
        
        summary = {
            "overall_en_success_rate": np.mean([c['en_success_rate'] for c in comparisons]),
            "overall_zh_auto_success_rate": np.mean([c['zh_auto_success_rate'] for c in comparisons]),
            "overall_translation_gap": np.mean([c['translation_quality_gap'] for c in comparisons]),
        }
        
        # 如果有人工翻译数据
        if 'semantic_equivalence_gap' in comparisons[0]:
            summary['overall_zh_manual_success_rate'] = np.mean([
                c['zh_manual_success_rate'] for c in comparisons
            ])
            summary['overall_semantic_equivalence_gap'] = np.mean([
                c['semantic_equivalence_gap'] for c in comparisons
            ])
        
        # 如果有语义变体数据
        if 'semantic_variance' in comparisons[0]:
            summary['overall_semantic_variance'] = np.mean([
                c['semantic_variance'] for c in comparisons
            ])
        
        # 统计成功/失败任务
        summary['tasks_above_80_percent'] = sum(
            1 for c in comparisons if c['en_success_rate'] >= 0.8
        )
        summary['tasks_above_60_percent'] = sum(
            1 for c in comparisons if c['en_success_rate'] >= 0.6
        )
        
        return summary
    
    def _generate_text_report(
        self,
        report: Dict[str, Any],
        output_path: Path
    ):
        """
        生成人类可读的文本报告
        
        Args:
            report: 完整报告字典
            output_path: 输出路径
        """
        lines = []
        
        # 标题
        lines.append("="*80)
        lines.append(" "*20 + "中文AIMC Agent 评估报告")
        lines.append(" "*15 + "Chinese AIMC Agent Evaluation Report")
        lines.append("="*80)
        lines.append("")
        
        # 元数据
        metadata = report['metadata']
        lines.append(f"生成时间: {metadata['timestamp']}")
        lines.append(f"任务总数: {metadata['total_tasks']}")
        lines.append("")
        
        # 摘要
        summary = report['summary']
        lines.append("="*80)
        lines.append("摘要统计 (Summary)")
        lines.append("="*80)
        lines.append(f"  整体英文成功率:         {summary['overall_en_success_rate']*100:6.2f}%")
        lines.append(f"  整体中文成功率(自动):   {summary['overall_zh_auto_success_rate']*100:6.2f}%")
        lines.append(f"  整体翻译质量Gap:        {summary['overall_translation_gap']*100:6.2f}%")
        
        if 'overall_semantic_equivalence_gap' in summary:
            lines.append(f"  整体语义等价Gap:        {summary['overall_semantic_equivalence_gap']*100:6.2f}%")
        
        if 'overall_semantic_variance' in summary:
            lines.append(f"  整体语义变体方差:       {summary['overall_semantic_variance']*100:6.2f}%")
        
        lines.append("")
        lines.append(f"  成功率 >= 80% 的任务数: {summary['tasks_above_80_percent']}")
        lines.append(f"  成功率 >= 60% 的任务数: {summary['tasks_above_60_percent']}")
        lines.append("")
        
        # 详细结果表格
        lines.append("="*80)
        lines.append("详细结果 (Detailed Results)")
        lines.append("="*80)
        lines.append("")
        
        # 表头
        lines.append(f"{'Task ID':<30} {'EN':>8} {'ZH(Auto)':>10} {'Gap':>8} {'Steps(EN)':>12}")
        lines.append("-"*80)
        
        # 数据行
        for comp in report['tasks']:
            task_id = comp['task_id'][:28]
            en_rate = f"{comp['en_success_rate']*100:.1f}%"
            zh_auto_rate = f"{comp['zh_auto_success_rate']*100:.1f}%"
            gap = f"{comp['translation_quality_gap']*100:.1f}%"
            en_steps = f"{comp.get('en_avg_steps', 0):.0f}"
            
            lines.append(f"{task_id:<30} {en_rate:>8} {zh_auto_rate:>10} {gap:>8} {en_steps:>12}")
        
        lines.append("="*80)
        lines.append("")
        
        # 建议
        lines.append("="*80)
        lines.append("评估建议 (Recommendations)")
        lines.append("="*80)
        
        gap = summary['overall_translation_gap']
        if gap < 0.05:
            lines.append("  ✅ 翻译质量优秀 (Gap < 5%)")
            lines.append("     建议: 当前翻译方案已经足够好，可以直接使用")
        elif gap < 0.10:
            lines.append("  ⚠️  翻译质量良好 (Gap < 10%)")
            lines.append("     建议: 继续优化术语词典，关注gap较大的任务")
        else:
            lines.append("  ❌ 翻译质量需要改进 (Gap >= 10%)")
            lines.append("     建议: 考虑进入阶段2，训练多语言MineCLIP适配层")
        
        lines.append("")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


if __name__ == "__main__":
    # 测试代码
    
    # 模拟数据
    test_comparisons = [
        {
            'task_id': 'harvest_1_log',
            'en_success_rate': 0.85,
            'zh_auto_success_rate': 0.75,
            'translation_quality_gap': 0.10,
            'en_avg_steps': 150,
            'zh_auto_avg_steps': 180,
        },
        {
            'task_id': 'harvest_1_dirt',
            'en_success_rate': 0.90,
            'zh_auto_success_rate': 0.85,
            'translation_quality_gap': 0.05,
            'en_avg_steps': 100,
            'zh_auto_avg_steps': 110,
        },
    ]
    
    # 创建生成器并生成报告
    generator = ReportGenerator()
    report_path = generator.generate(test_comparisons, "test_report.json")
    
    print(f"报告已生成: {report_path}")
    
    # 打印文本报告
    txt_path = Path(report_path).with_suffix('.txt')
    if txt_path.exists():
        print("\n文本报告内容:")
        print(txt_path.read_text(encoding='utf-8'))

