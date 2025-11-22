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
    
    # 3. 生成简单HTML报告（不依赖MatrixAnalyzer的复杂结构）
    try:
        # 直接生成一个简化的HTML报告
        html_path = task_set_dir / "task_set_report.html"
        
        # 准备任务数据
        task_data = []
        for result in results:
            success_count = sum(1 for t in result.trials if t.success)
            task_data.append({
                'task_id': result.task_id,
                'instruction': result.instruction,
                'success_rate': result.success_rate,
                'avg_steps': result.avg_steps,
                'success_count': success_count,
                'total_trials': len(result.trials)
            })
        
        # 排序：成功率从高到低
        task_data_sorted = sorted(task_data, key=lambda x: (-x['success_rate'], x['task_id']))
        
        # 生成HTML
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task-Set 评估报告 - {task_set_dir.name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        .stat-card h3 {{ font-size: 0.9em; opacity: 0.9; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2em; font-weight: 700; }}
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin-bottom: 30px;
        }}
        .chart-container h2 {{ margin-bottom: 20px; color: #667eea; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .success-high {{ color: #28a745; font-weight: 600; }}
        .success-medium {{ color: #ffc107; font-weight: 600; }}
        .success-low {{ color: #dc3545; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Steve1 评估报告</h1>
            <div class="subtitle">{task_set_dir.name}</div>
        </div>
        
        <div class="content">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>总任务数</h3>
                    <div class="value">{len(results)}</div>
                </div>
                <div class="stat-card">
                    <h3>总试验次数</h3>
                    <div class="value">{sum(len(r.trials) for r in results)}</div>
                </div>
                <div class="stat-card">
                    <h3>平均成功率</h3>
                    <div class="value">{sum(r.success_rate for r in results) / len(results) * 100:.1f}%</div>
                </div>
                <div class="stat-card">
                    <h3>总评估时间</h3>
                    <div class="value">{sum(sum(t.time_seconds for t in r.trials) for r in results) / 60:.0f}min</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>📊 各任务成功率</h2>
                <canvas id="successChart" height="400"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>🏃 各任务平均步数（仅成功任务）</h2>
                <canvas id="stepsChart" height="400"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>📋 任务详细列表</h2>
                <table>
                    <thead>
                        <tr>
                            <th>任务ID</th>
                            <th>指令</th>
                            <th>成功率</th>
                            <th>成功数/总数</th>
                            <th>平均步数</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        for task in task_data_sorted:
            success_class = 'success-high' if task['success_rate'] >= 0.7 else ('success-medium' if task['success_rate'] >= 0.3 else 'success-low')
            html_content += f"""
                        <tr>
                            <td>{task['task_id']}</td>
                            <td>{task['instruction']}</td>
                            <td class="{success_class}">{task['success_rate']*100:.1f}%</td>
                            <td>{task['success_count']}/{task['total_trials']}</td>
                            <td>{task['avg_steps']:.0f}</td>
                        </tr>"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // 成功率图表
        const successCtx = document.getElementById('successChart').getContext('2d');
        new Chart(successCtx, {
            type: 'bar',
            data: {
                labels: """ + json.dumps([t['task_id'] for t in task_data_sorted[:20]]) + """,
                datasets: [{
                    label: '成功率',
                    data: """ + json.dumps([t['success_rate'] for t in task_data_sorted[:20]]) + """,
                    backgroundColor: function(context) {
                        const value = context.parsed.y;
                        if (value >= 0.7) return 'rgba(40, 167, 69, 0.8)';
                        if (value >= 0.3) return 'rgba(255, 193, 7, 0.8)';
                        return 'rgba(220, 53, 69, 0.8)';
                    },
                    borderWidth: 0
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return '成功率: ' + (context.parsed.x * 100).toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });
        
        // 步数图表（只显示有成功的任务）
        const stepsData = """ + json.dumps([{'task_id': t['task_id'], 'steps': t['avg_steps']} for t in task_data_sorted if t['avg_steps'] > 0][:20]) + """;
        const stepsCtx = document.getElementById('stepsChart').getContext('2d');
        new Chart(stepsCtx, {
            type: 'bar',
            data: {
                labels: stepsData.map(d => d.task_id),
                datasets: [{
                    label: '平均步数',
                    data: stepsData.map(d => d.steps),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderWidth: 0
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return '平均步数: ' + context.parsed.x.toFixed(0);
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
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

