"""
策略评估框架（Policy Evaluation Framework）
=========================================

【职责】
本框架负责协调和管理策略评估流程：
1. 任务加载 - 从YAML配置加载评估任务
2. 评估调度 - 协调Prior/Policy评估器的运行
3. 结果汇总 - 收集和整合分析结果
4. 报告生成 - 生成HTML可视化报告

【策略评估 vs 结果评估】
- 策略评估（本框架）：分析模型的策略质量和行为模式
  * Prior评估器: 文本→嵌入的转换质量（相似度、方差）
  * Policy评估器: 嵌入→动作的执行质量（多样性、一致性）
  * 端到端分析: 瓶颈识别、贡献度分析
  
- 结果评估（eval_framework.py）：评估任务成功率
  * 任务完成情况
  * 成功率统计

【评估模式】
- prior: 仅评估Prior模型（快速）
- policy: 完整评估（Prior + Policy + 端到端）

使用方法:
    # 完整评估（默认）
    python src/evaluation/policy_eval_framework.py \
        --config config/eval_tasks_comprehensive.yaml \
        --task-set harvest_tasks
    
    # 仅Prior评估
    python src/evaluation/policy_eval_framework.py \
        --mode prior \
        --config config/eval_tasks_comprehensive.yaml

作者: AI Assistant  
日期: 2025-11-27
"""

# 修复 OpenMP 冲突警告
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List
import yaml
import json
from datetime import datetime
import logging

# 抑制 MineRL 的 WARNING 和 ERROR
logging.getLogger('minerl.env.malmo.instance').setLevel(logging.CRITICAL)

import warnings

# 忽略 OpenMP 警告
warnings.filterwarnings('ignore', message='.*OpenMP.*')

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.steve1_policy_evaluator import Steve1PolicyEvaluator
from src.evaluation.steve1_prior_evaluator import Steve1PriorEvaluator
from src.utils.steve1_policy_visualizer import Steve1PolicyVisualizer

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, verbose: bool = False):
    """设置日志"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # 文件日志
    log_file = output_dir / f"policy_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info(f"日志保存到: {log_file}")


def load_eval_config(config_path: Path) -> Dict:
    """加载评估配置"""
    logger.info(f"加载配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_tasks_from_config(config: Dict, task_set: str) -> List[Dict]:
    """从配置中提取任务"""
    logger.info(f"提取任务集: {task_set}")
    
    if task_set == 'all_tasks':
        # 提取所有任务
        all_task_ids = config.get('all_tasks', [])
        tasks = []
        
        # 从各个分类中查找任务
        for category in ['harvest_tasks', 'combat_tasks', 'techtree_tasks', 'special_tasks']:
            category_tasks = config.get(category, [])
            for task in category_tasks:
                if task['task_id'] in all_task_ids:
                    tasks.append(task)
    else:
        # 提取特定分类的任务
        tasks = config.get(task_set, [])
    
    logger.info(f"  ✓ 提取了 {len(tasks)} 个任务")
    
    return tasks


def run_prior_analysis(
    analyzer: Steve1PolicyEvaluator,
    visualizer: Steve1PolicyVisualizer,
    tasks: List[Dict],
    output_dir: Path,
):
    """运行 Prior 模型分析"""
    logger.info("=" * 80)
    logger.info("阶段1: Prior 模型分析 p(z_goal|y)")
    logger.info("=" * 80)
    
    # 提取所有指令
    instructions = []
    for task in tasks:
        instruction = task.get('en_instruction', '')
        if instruction:
            instructions.append(instruction)
    
    logger.info(f"分析 {len(instructions)} 个指令")
    
    # 运行分析
    prior_dir = output_dir / "prior_analysis"
    prior_dir.mkdir(parents=True, exist_ok=True)
    
    prior_results = analyzer.analyze_prior_model(
        instructions=instructions,
        output_dir=prior_dir,
        n_samples=5,
    )
    
    # 可视化
    logger.info("生成 Prior 模型可视化...")
    
    # 1. 嵌入空间可视化（t-SNE）
    text_embeds = []
    prior_embeds = []
    inst_list = []
    for inst, result in prior_results.items():
        text_embeds.append(result.text_embed)
        prior_embeds.append(result.prior_embed)
        inst_list.append(inst)
    
    import numpy as np
    text_embeds = np.array(text_embeds)
    prior_embeds = np.array(prior_embeds)
    
    # t-SNE
    visualizer.visualize_prior_embedding_space(
        text_embeds=text_embeds,
        prior_embeds=prior_embeds,
        instructions=inst_list,
        output_path=prior_dir / "embedding_space_tsne.png",
        method='tsne',
    )
    
    # PCA（更快，适合大数据集）
    visualizer.visualize_prior_embedding_space(
        text_embeds=text_embeds,
        prior_embeds=prior_embeds,
        instructions=inst_list,
        output_path=prior_dir / "embedding_space_pca.png",
        method='pca',
    )
    
    # 2. 相似度矩阵
    visualizer.visualize_prior_similarity_matrix(
        prior_embeds=prior_embeds,
        instructions=inst_list,
        output_path=prior_dir / "similarity_matrix.png",
    )
    
    # 3. 质量指标
    visualizer.visualize_prior_quality_metrics(
        prior_analysis_results=prior_results,
        output_path=prior_dir / "quality_metrics.png",
    )
    
    logger.info(f"✓ Prior 分析完成，结果保存到: {prior_dir}")
    
    return prior_results


def run_policy_analysis(
    analyzer: Steve1PolicyEvaluator,
    visualizer: Steve1PolicyVisualizer,
    task: Dict,
    n_trials: int,
    max_steps: int,
    output_dir: Path,
):
    """运行策略模型分析（单个任务）"""
    task_id = task['task_id']
    instruction = task.get('en_instruction', '')
    env_name = task.get('env_name', 'MineRLHarvestDefaultEnv-v0')
    env_config = task.get('env_config', {})
    
    logger.info(f"分析任务: {task_id}")
    logger.info(f"  指令: {instruction}")
    
    # 运行分析
    policy_results = analyzer.analyze_policy_model(
        instruction=instruction,
        env_name=env_name,
        env_config=env_config,
        max_steps=max_steps,
        n_trials=n_trials,
        use_ground_truth_visual=False,
    )
    
    # 保存结果
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = [r.to_dict() for r in policy_results]
    with open(task_dir / "policy_analysis.json", 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # 可视化
    visualizer.visualize_action_distribution(
        policy_results=policy_results,
        output_path=task_dir / "action_distribution.png",
    )
    
    visualizer.visualize_policy_metrics(
        policy_results=policy_results,
        output_path=task_dir / "policy_metrics.png",
    )
    
    return policy_results


def run_end_to_end_analysis(
    analyzer: Steve1PolicyEvaluator,
    visualizer: Steve1PolicyVisualizer,
    tasks: List[Dict],
    n_trials: int,
    max_steps: int,
    output_dir: Path,
    max_tasks: int = None,
):
    """运行端到端分析"""
    logger.info("=" * 80)
    logger.info("阶段3: 端到端分析（两阶段联合）")
    logger.info("=" * 80)
    
    # 限制任务数量（避免运行太久）
    if max_tasks:
        tasks = tasks[:max_tasks]
        logger.info(f"限制评估任务数量为: {max_tasks}")
    
    e2e_dir = output_dir / "end_to_end"
    e2e_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for i, task in enumerate(tasks):
        task_id = task['task_id']
        instruction = task.get('en_instruction', '')
        env_name = task.get('env_name', 'MineRLHarvestDefaultEnv-v0')
        env_config = task.get('env_config', {})
        
        logger.info(f"[{i+1}/{len(tasks)}] 端到端分析: {task_id}")
        
        # 运行分析
        try:
            e2e_results = analyzer.analyze_end_to_end(
                task_id=task_id,
                instruction=instruction,
                env_name=env_name,
                env_config=env_config,
                max_steps=max_steps,
                n_trials=n_trials,
                output_dir=e2e_dir / task_id,
            )
            
            all_results[task_id] = e2e_results
            
            # 可视化单个任务
            visualizer.visualize_end_to_end_analysis(
                end_to_end_results=e2e_results,
                output_path=e2e_dir / task_id / "end_to_end_analysis.png",
            )
            
            logger.info(f"  ✓ {task_id} 分析完成")
        
        except Exception as e:
            logger.error(f"  ✗ {task_id} 分析失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 多任务对比可视化
    if len(all_results) > 1:
        logger.info("生成多任务对比可视化...")
        visualizer.visualize_task_comparison(
            task_results=all_results,
            output_path=e2e_dir / "task_comparison.png",
        )
    
    logger.info(f"✓ 端到端分析完成，结果保存到: {e2e_dir}")
    
    return all_results


def generate_summary_report(
    prior_results: Dict,
    all_e2e_results: Dict,
    output_dir: Path,
):
    """生成总结报告"""
    logger.info("生成总结报告...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'prior_analysis': {},
        'policy_analysis': {},
        'end_to_end_analysis': {},
        'recommendations': [],
    }
    
    # Prior 分析总结
    if prior_results:
        text_to_prior_sims = [r.text_to_prior_similarity for r in prior_results.values()]
        report['prior_analysis'] = {
            'num_instructions': len(prior_results),
            'avg_text_to_prior_similarity': float(sum(text_to_prior_sims) / len(text_to_prior_sims)),
            'min_similarity': float(min(text_to_prior_sims)),
            'max_similarity': float(max(text_to_prior_sims)),
        }
    
    # 端到端分析总结
    if all_e2e_results:
        all_success_rates = []
        all_stage1_contributions = []
        all_stage2_contributions = []
        bottleneck_counts = {0: 0, 1: 0, 2: 0}
        
        for task_id, e2e_results in all_e2e_results.items():
            successes = [r.policy_result.success for r in e2e_results]
            success_rate = sum(successes) / len(successes)
            all_success_rates.append(success_rate)
            
            for r in e2e_results:
                all_stage1_contributions.append(r.stage1_contribution)
                all_stage2_contributions.append(r.stage2_contribution)
                bottleneck_counts[r.bottleneck_stage] = bottleneck_counts.get(r.bottleneck_stage, 0) + 1
        
        report['end_to_end_analysis'] = {
            'num_tasks': len(all_e2e_results),
            'avg_success_rate': float(sum(all_success_rates) / len(all_success_rates)),
            'avg_stage1_contribution': float(sum(all_stage1_contributions) / len(all_stage1_contributions)),
            'avg_stage2_contribution': float(sum(all_stage2_contributions) / len(all_stage2_contributions)),
            'bottleneck_distribution': {
                'no_bottleneck': bottleneck_counts[0],
                'prior_bottleneck': bottleneck_counts[1],
                'policy_bottleneck': bottleneck_counts[2],
            }
        }
    
    # 生成建议
    if report['prior_analysis']:
        avg_prior_sim = report['prior_analysis']['avg_text_to_prior_similarity']
        if avg_prior_sim < 0.7:
            report['recommendations'].append({
                'priority': 'high',
                'component': 'prior',
                'issue': f'Prior模型的文本-嵌入相似度较低 ({avg_prior_sim:.3f})',
                'suggestion': '考虑重新训练Prior模型，或增加训练数据'
            })
    
    if report['end_to_end_analysis']:
        bottleneck_dist = report['end_to_end_analysis']['bottleneck_distribution']
        if bottleneck_dist['prior_bottleneck'] > bottleneck_dist['policy_bottleneck']:
            report['recommendations'].append({
                'priority': 'high',
                'component': 'prior',
                'issue': 'Prior模型是主要瓶颈',
                'suggestion': '优先优化Prior模型：增加训练数据、调整架构、或使用更强的文本编码器'
            })
        elif bottleneck_dist['policy_bottleneck'] > bottleneck_dist['prior_bottleneck']:
            report['recommendations'].append({
                'priority': 'high',
                'component': 'policy',
                'issue': '策略模型是主要瓶颈',
                'suggestion': '优先优化策略模型：增加训练数据、调整超参数、或进行DAgger迭代'
            })
    
    # 保存报告
    report_path = output_dir / "summary_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 总结报告已保存: {report_path}")
    
    # 打印关键发现
    logger.info("=" * 80)
    logger.info("关键发现")
    logger.info("=" * 80)
    
    if report['prior_analysis']:
        logger.info(f"Prior 模型:")
        logger.info(f"  平均文本-Prior相似度: {report['prior_analysis']['avg_text_to_prior_similarity']:.3f}")
    
    if report['end_to_end_analysis']:
        logger.info(f"端到端性能:")
        logger.info(f"  平均成功率: {report['end_to_end_analysis']['avg_success_rate']*100:.1f}%")
        logger.info(f"  Prior 贡献: {report['end_to_end_analysis']['avg_stage1_contribution']*100:.1f}%")
        logger.info(f"  策略贡献: {report['end_to_end_analysis']['avg_stage2_contribution']*100:.1f}%")
    
    if report['recommendations']:
        logger.info("建议:")
        for rec in report['recommendations']:
            logger.info(f"  [{rec['priority'].upper()}] {rec['issue']}")
            logger.info(f"    → {rec['suggestion']}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Steve1 两阶段模型深度评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估 harvest 任务（快速测试）
  python src/evaluation/run_steve1_deep_evaluation.py \\
      --config config/eval_tasks_comprehensive.yaml \\
      --task-set harvest_tasks \\
      --max-tasks 3 \\
      --n-trials 3
  
  # 评估所有任务（完整评估）
  python src/evaluation/run_steve1_deep_evaluation.py \\
      --config config/eval_tasks_comprehensive.yaml \\
      --task-set all_tasks \\
      --n-trials 5
        """
    )
    
    # 基础参数
    parser.add_argument('--config', type=str, required=True,
                       help='评估配置文件路径')
    parser.add_argument('--task-set', type=str, default='harvest_tasks',
                       help='任务集名称（harvest_tasks, combat_tasks, techtree_tasks, all_tasks等）')
    parser.add_argument('--output-dir', type=str, default='results/policy_evaluation',
                       help='输出目录')
    
    # 评估参数
    parser.add_argument('--n-trials', type=int, default=5,
                       help='每个任务的试验次数')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='每个试验的最大步数')
    parser.add_argument('--max-tasks', type=int, default=None,
                       help='最大评估任务数（用于快速测试）')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, default='data/weights/vpt/2x.model',
                       help='VPT 模型路径')
    parser.add_argument('--weights-path', type=str, default='data/weights/steve1/steve1.weights',
                       help='Steve1 权重路径')
    parser.add_argument('--prior-weights', type=str, default='data/weights/steve1/steve1_prior.pt',
                       help='Prior VAE 权重路径')
    
    # 评估模式
    parser.add_argument('--mode', type=str, default='policy', 
                       choices=['prior', 'policy'],
                       help='评估模式：prior=仅Prior评估（快速），policy=完整评估（Prior+Policy+端到端）')
    
    # 分析选项（高级，与--mode互斥）
    parser.add_argument('--skip-prior', action='store_true',
                       help='跳过 Prior 分析（只分析策略模型）')
    parser.add_argument('--skip-policy', action='store_true',
                       help='跳过策略分析（只分析 Prior 模型）')
    parser.add_argument('--skip-e2e', action='store_true',
                       help='跳过端到端分析')
    
    # 其他选项
    parser.add_argument('--verbose', action='store_true',
                       help='详细日志')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 根据mode设置skip标志
    if args.mode == 'prior':
        args.skip_policy = True
        args.skip_e2e = True
    # mode='policy'时，不skip任何部分（完整评估）
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.verbose)
    
    logger.info("=" * 80)
    logger.info(f"策略评估框架 - {args.mode.upper()}模式")
    logger.info("=" * 80)
    logger.info(f"评估模式: {args.mode} ({'仅Prior' if args.mode == 'prior' else 'Prior + Policy + 端到端'})")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"任务集: {args.task_set}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"试验次数: {args.n_trials}")
    logger.info(f"最大步数: {args.max_steps}")
    
    # 加载配置
    config = load_eval_config(Path(args.config))
    tasks = extract_tasks_from_config(config, args.task_set)
    
    if not tasks:
        logger.error(f"未找到任务集: {args.task_set}")
        return 1
    
    # 初始化分析器和可视化器
    logger.info("初始化分析器和可视化器...")
    analyzer = Steve1PolicyEvaluator(
        model_path=args.model_path,
        weights_path=args.weights_path,
        prior_weights=args.prior_weights,
        seed=args.seed,
    )
    visualizer = Steve1PolicyVisualizer()
    
    # 运行分析
    prior_results = None
    all_e2e_results = None
    
    try:
        # 阶段1: Prior 分析
        if not args.skip_prior and not args.skip_e2e:
            prior_results = run_prior_analysis(
                analyzer=analyzer,
                visualizer=visualizer,
                tasks=tasks,
                output_dir=output_dir,
            )
        
        # 阶段2: 端到端分析（包含策略分析）
        if not args.skip_e2e:
            all_e2e_results = run_end_to_end_analysis(
                analyzer=analyzer,
                visualizer=visualizer,
                tasks=tasks,
                n_trials=args.n_trials,
                max_steps=args.max_steps,
                output_dir=output_dir,
                max_tasks=args.max_tasks,
            )
        
        # 生成总结报告
        generate_summary_report(
            prior_results=prior_results or {},
            all_e2e_results=all_e2e_results or {},
            output_dir=output_dir,
        )
        
        # 生成 HTML 报告
        logger.info("=" * 80)
        logger.info("生成 HTML 可视化报告...")
        logger.info("=" * 80)
        try:
            from src.utils.policy_html_generator import generate_policy_html_report
            html_path = generate_policy_html_report(output_dir)
            logger.info(f"✓ HTML 报告已生成: {html_path}")
            logger.info(f"  使用浏览器打开查看: open {html_path}")
        except Exception as html_error:
            logger.warning(f"⚠️ HTML 报告生成失败: {html_error}")
            logger.warning("  JSON 结果仍然可用，请查看 summary_report.json")
        
        logger.info("=" * 80)
        logger.info("✓ 深度评估完成！")
        logger.info(f"✓ 所有结果已保存到: {output_dir}")
        logger.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # 清理资源
        analyzer.cleanup()


if __name__ == '__main__':
    sys.exit(main())

