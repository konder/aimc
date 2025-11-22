#!/usr/bin/env python3
"""
从已完成的task-set目录生成分析报告
用于恢复丢失的或补充生成报告

Usage:
    python scripts/generate_taskset_report.py <task_set_dir>
    
Example:
    python scripts/generate_taskset_report.py results/evaluation/all_tasks_20251121_214545
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 直接导入，避免通过__init__.py触发gym导入
eval_dir = project_root / 'src' / 'evaluation'
if str(eval_dir) not in sys.path:
    sys.path.insert(0, str(eval_dir))

from metrics import TaskResult, TrialResult
from matrix_analyzer import MatrixAnalyzer
from html_report_generator import HTMLReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_result_from_json(result_json_path: Path) -> TaskResult:
    """从result.json文件加载TaskResult"""
    with open(result_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 重建TrialResult对象
    trials = []
    for trial_data in data['trials']:
        trial = TrialResult(
            task_id=data['task_id'],
            language=data['language'],
            instruction=data['instruction'],
            success=trial_data['success'],
            steps=trial_data['steps'],
            time_seconds=trial_data['time_seconds'],
            final_inventory=trial_data.get('final_inventory', {}),
            trajectory=[]  # 不加载trajectory
        )
        trials.append(trial)
    
    # 创建TaskResult对象
    task_result = TaskResult(
        task_id=data['task_id'],
        language=data['language'],
        instruction=data['instruction'],
        trials=trials
    )
    
    return task_result


def collect_task_results(task_set_dir: Path) -> List[TaskResult]:
    """收集task-set目录下所有任务的结果"""
    results = []
    
    # 遍历所有任务目录
    for task_dir in sorted(task_set_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        
        # 查找result.json文件
        result_json = task_dir / "result.json"
        if not result_json.exists():
            logger.warning(f"⚠️ 未找到result.json: {task_dir.name}")
            continue
        
        try:
            task_result = load_result_from_json(result_json)
            results.append(task_result)
            logger.info(f"✓ 加载任务结果: {task_result.task_id} "
                       f"(成功率: {task_result.success_rate*100:.1f}%, "
                       f"平均步数: {task_result.avg_steps:.1f})")
        except Exception as e:
            logger.error(f"❌ 加载失败 {task_dir.name}: {e}")
    
    return results


def generate_reports(task_set_dir: Path, results: List[TaskResult]):
    """生成所有报告"""
    
    # 1. 生成文本报告
    report_txt = task_set_dir / "task_set_report.txt"
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Task-Set 评估报告\n")
        f.write(f"目录: {task_set_dir.name}\n")
        f.write("="*80 + "\n\n")
        
        total_trials = sum(len(r.trials) for r in results)
        total_time = sum(sum(t.time_seconds for t in r.trials) for r in results)
        successful_results = [r for r in results if r.avg_steps > 0]  # 只统计有成功的任务
        
        f.write(f"总任务数: {len(results)}\n")
        f.write(f"总成功率: {sum(r.success_rate for r in results) / len(results) * 100:.1f}%\n")
        if successful_results:
            f.write(f"平均步数 (成功任务): {sum(r.avg_steps for r in successful_results) / len(successful_results):.1f}\n")
        f.write(f"总试验次数: {total_trials}\n")
        f.write(f"总评估时间: {total_time / 60:.1f} 分钟\n\n")
        
        f.write("="*80 + "\n")
        f.write("任务详情\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            success_count = sum(1 for t in result.trials if t.success)
            total_time_task = sum(t.time_seconds for t in result.trials)
            
            f.write(f"任务: {result.task_id}\n")
            f.write(f"  指令: {result.instruction}\n")
            f.write(f"  成功率: {result.success_rate*100:.1f}% ({success_count}/{len(result.trials)})\n")
            f.write(f"  平均步数: {result.avg_steps:.1f}\n")
            f.write(f"  平均时间: {result.avg_time:.1f}s\n")
            f.write(f"  总时间: {total_time_task:.1f}s\n\n")
    
    logger.info(f"✓ 文本报告已生成: {report_txt}")
    
    # 2. 生成矩阵分析报告
    try:
        analyzer = MatrixAnalyzer()
        
        # 准备分析输入（MatrixAnalyzer需要的格式）
        analysis_input = []
        for result in results:
            for trial in result.trials:
                analysis_input.append({
                    'task_id': result.task_id,
                    'language': result.language,
                    'instruction': result.instruction,
                    'success': trial.success,
                    'steps': trial.steps,
                    'time_seconds': trial.time_seconds,
                    'final_inventory': trial.final_inventory
                })
        
        # 执行矩阵分析
        matrix_analysis = analyzer.analyze_results(analysis_input)
        
        # 保存矩阵分析为JSON
        matrix_json = task_set_dir / "matrix_analysis.json"
        with open(matrix_json, 'w', encoding='utf-8') as f:
            json.dump(matrix_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 矩阵分析已生成: {matrix_json}")
    except Exception as e:
        logger.error(f"❌ 生成矩阵分析失败: {e}")
        import traceback
        traceback.print_exc()
        matrix_analysis = None
    
    # 3. 生成HTML报告
    try:
        if matrix_analysis is None:
            # 如果矩阵分析失败，创建简单的分析数据
            matrix_analysis = {
                "overall": {
                    "total_tasks": len(results),
                    "success_rate": sum(r.success_rate for r in results) / len(results),
                    "avg_steps": sum(r.avg_steps for r in results if r.avg_steps > 0) / len([r for r in results if r.avg_steps > 0]) if any(r.avg_steps > 0 for r in results) else 0
                },
                "dimensions": {},
                "tasks": [
                    {
                        "task_id": r.task_id,
                        "success_rate": r.success_rate,
                        "avg_steps": r.avg_steps
                    }
                    for r in results
                ]
            }
        
        html_generator = HTMLReportGenerator(str(task_set_dir))
        
        # 生成HTML报告
        html_path = html_generator.generate(
            analysis=matrix_analysis,
            config_file=task_set_dir.name,
            output_filename="task_set_report.html"
        )
        
        logger.info(f"✓ HTML报告已生成: {html_path}")
        logger.info(f"  在浏览器中打开: file://{html_path.absolute()}")
    except Exception as e:
        logger.error(f"❌ 生成HTML报告失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 生成JSON汇总
    summary_json = task_set_dir / "task_set_summary.json"
    
    total_trials = sum(len(r.trials) for r in results)
    total_time = sum(sum(t.time_seconds for t in r.trials) for r in results)
    successful_results = [r for r in results if r.avg_steps > 0]
    
    summary_data = {
        "task_set_name": task_set_dir.name,
        "total_tasks": len(results),
        "total_trials": total_trials,
        "overall_success_rate": sum(r.success_rate for r in results) / len(results),
        "avg_steps": sum(r.avg_steps for r in successful_results) / len(successful_results) if successful_results else 0,
        "total_time_minutes": total_time / 60,
        "tasks": [
            {
                "task_id": r.task_id,
                "success_rate": r.success_rate,
                "avg_steps": r.avg_steps,
                "avg_time": r.avg_time,
                "success_count": sum(1 for t in r.trials if t.success),
                "total_trials": len(r.trials)
            }
            for r in results
        ]
    }
    
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ JSON汇总已生成: {summary_json}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_taskset_report.py <task_set_dir>")
        print("Example: python scripts/generate_taskset_report.py results/evaluation/all_tasks_20251121_214545")
        sys.exit(1)
    
    task_set_dir = Path(sys.argv[1])
    
    if not task_set_dir.exists():
        logger.error(f"❌ 目录不存在: {task_set_dir}")
        sys.exit(1)
    
    if not task_set_dir.is_dir():
        logger.error(f"❌ 不是目录: {task_set_dir}")
        sys.exit(1)
    
    logger.info(f"{'='*80}")
    logger.info(f"生成Task-Set分析报告")
    logger.info(f"目录: {task_set_dir}")
    logger.info(f"{'='*80}\n")
    
    # 收集所有任务结果
    logger.info("1. 收集任务结果...")
    results = collect_task_results(task_set_dir)
    
    if not results:
        logger.error("❌ 未找到任何任务结果")
        sys.exit(1)
    
    logger.info(f"✓ 成功加载 {len(results)} 个任务的结果\n")
    
    # 生成报告
    logger.info("2. 生成分析报告...")
    generate_reports(task_set_dir, results)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 报告生成完成！")
    logger.info(f"{'='*80}")
    logger.info(f"\n生成的文件:")
    logger.info(f"  • task_set_report.txt - 文本报告")
    logger.info(f"  • matrix_analysis.txt - 矩阵分析")
    logger.info(f"  • task_set_report.html - HTML交互报告")
    logger.info(f"  • task_set_summary.json - JSON汇总")
    logger.info(f"\n在浏览器中打开HTML报告:")
    logger.info(f"  file://{task_set_dir / 'task_set_report.html'}")


if __name__ == "__main__":
    main()

