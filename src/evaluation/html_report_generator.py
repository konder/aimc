"""
HTML报告生成器
HTML Report Generator - Generate interactive HTML reports with charts

生成包含以下内容的HTML报告：
1. 综合得分和三维能力雷达图
2. 各维度详细分析和柱状图
3. 任务级别详细结果表格
4. 交互式图表（使用Chart.js）
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class HTMLReportGenerator:
    """HTML报告生成器"""
    
    HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steve1 评估报告 - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
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
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .score-section {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .score-value {{
            font-size: 4em;
            font-weight: 700;
            margin: 20px 0;
        }}
        
        .score-label {{
            font-size: 1.3em;
            opacity: 0.9;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }}
        
        .chart-container {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .chart-container h2 {{
            margin-bottom: 20px;
            color: #667eea;
            font-size: 1.5em;
        }}
        
        .dimension-section {{
            margin: 40px 0;
        }}
        
        .dimension-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px 12px 0 0;
            font-size: 1.5em;
            font-weight: 600;
        }}
        
        .dimension-body {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 0 0 12px 12px;
        }}
        
        .level-row {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }}
        
        .level-row:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }}
        
        .level-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .level-name {{
            font-size: 1.2em;
            font-weight: 600;
            color: #667eea;
        }}
        
        .level-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.9em;
            color: #666;
        }}
        
        .progress-bar {{
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }}
        
        .task-list {{
            margin-top: 15px;
        }}
        
        .task-item {{
            padding: 10px;
            background: #f8f9fa;
            border-left: 3px solid #667eea;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        
        .task-item .task-id {{
            font-weight: 600;
            color: #333;
        }}
        
        .task-item .task-rate {{
            float: right;
            color: #667eea;
            font-weight: 600;
        }}
        
        .success-rate-high {{ color: #28a745; }}
        .success-rate-medium {{ color: #ffc107; }}
        .success-rate-low {{ color: #dc3545; }}
        
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 30px 0;
            border-radius: 8px;
        }}
        
        .recommendations h2 {{
            color: #856404;
            margin-bottom: 15px;
        }}
        
        .recommendations ul {{
            list-style: none;
            padding: 0;
        }}
        
        .recommendations li {{
            padding: 10px 0;
            border-bottom: 1px solid #fff3cd;
        }}
        
        .recommendations li:last-child {{
            border-bottom: none;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }}
        
        .metadata {{
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .metadata-item {{
            padding: 10px;
            background: white;
            border-radius: 6px;
        }}
        
        .metadata-item .label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .metadata-item .value {{
            font-size: 1.1em;
            font-weight: 600;
            color: #333;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .content {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Steve1 模型评估报告</h1>
            <div class="subtitle">三维能力矩阵分析 | {timestamp}</div>
        </div>
        
        <div class="content">
            <!-- 元数据 -->
            <div class="metadata">
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <div class="label">评估时间</div>
                        <div class="value">{timestamp}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">总任务数</div>
                        <div class="value">{total_tasks}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="label">配置文件</div>
                        <div class="value">{config_file}</div>
                    </div>
                </div>
            </div>
            
            <!-- 综合得分 -->
            <div class="score-section">
                <div class="score-label">综合得分</div>
                <div class="score-value">{overall_score:.1%}</div>
                <div class="score-label">Harvest × 40% + Combat × 30% + TechTree × 30%</div>
            </div>
            
            <!-- 图表 -->
            <div class="charts-grid">
                <div class="chart-container">
                    <h2>📊 三维能力雷达图</h2>
                    <canvas id="radarChart"></canvas>
                </div>
                <div class="chart-container">
                    <h2>📈 维度得分对比</h2>
                    <canvas id="barChart"></canvas>
                </div>
            </div>
            
            <!-- 任务级别详细图表 -->
            <div class="task-charts-section">
                <h2 style="color: #667eea; margin: 40px 0 20px 0; font-size: 1.8em;">📋 任务详细分析</h2>
                
                <div class="charts-grid">
                    <div class="chart-container">
                        <h2>✅ 各任务成功率</h2>
                        <canvas id="taskSuccessChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <h2>👣 各任务平均步数</h2>
                        <canvas id="taskStepsChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- 各维度详情 -->
            {dimensions_html}
            
            <!-- 建议 -->
            {recommendations_html}
        </div>
        
        <div class="footer">
            <p>Generated by AIMC Evaluation Framework</p>
            <p>Powered by Steve1 & MineDojo</p>
        </div>
    </div>
    
    <script>
        // 雷达图
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: {radar_labels},
                datasets: [{{
                    label: 'Steve1能力评分',
                    data: {radar_data},
                    fill: true,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgb(102, 126, 234)',
                    pointBackgroundColor: 'rgb(102, 126, 234)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(102, 126, 234)',
                    borderWidth: 3
                }}]
            }},
            options: {{
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            stepSize: 0.2,
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // 柱状图
        const barCtx = document.getElementById('barChart').getContext('2d');
        new Chart(barCtx, {{
            type: 'bar',
            data: {{
                labels: {bar_labels},
                datasets: [{{
                    label: '得分',
                    data: {bar_data},
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(245, 87, 108, 0.8)',
                        'rgba(52, 211, 153, 0.8)'
                    ],
                    borderColor: [
                        'rgb(102, 126, 234)',
                        'rgb(245, 87, 108)',
                        'rgb(52, 211, 153)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
        
        // 任务成功率图表
        const taskSuccessCtx = document.getElementById('taskSuccessChart').getContext('2d');
        new Chart(taskSuccessCtx, {{
            type: 'bar',
            data: {{
                labels: {task_labels},
                datasets: [{{
                    label: '成功率',
                    data: {task_success_data},
                    backgroundColor: function(context) {{
                        const value = context.parsed.y;
                        if (value >= 0.7) return 'rgba(40, 167, 69, 0.8)';
                        if (value >= 0.4) return 'rgba(255, 193, 7, 0.8)';
                        return 'rgba(220, 53, 69, 0.8)';
                    }},
                    borderColor: function(context) {{
                        const value = context.parsed.y;
                        if (value >= 0.7) return 'rgb(40, 167, 69)';
                        if (value >= 0.4) return 'rgb(255, 193, 7)';
                        return 'rgb(220, 53, 69)';
                    }},
                    borderWidth: 2
                }}]
            }},
            options: {{
                indexAxis: 'y',
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return '成功率: ' + (context.parsed.x * 100).toFixed(1) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // 任务步数图表
        const taskStepsCtx = document.getElementById('taskStepsChart').getContext('2d');
        new Chart(taskStepsCtx, {{
            type: 'bar',
            data: {{
                labels: {task_labels},
                datasets: [{{
                    label: '平均步数',
                    data: {task_steps_data},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgb(102, 126, 234)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                indexAxis: 'y',
                scales: {{
                    x: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value) {{
                                return value.toFixed(0);
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return '平均步数: ' + context.parsed.x.toFixed(0);
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
    
    def __init__(self, output_dir: str = "results/evaluation"):
        """
        初始化HTML报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        analysis: Dict[str, Any],
        config_file: str = "eval_tasks.yaml",
        output_filename: Optional[str] = None
    ) -> Path:
        """
        生成HTML报告
        
        Args:
            analysis: 矩阵分析结果
            config_file: 配置文件名
            output_filename: 输出文件名（如果None则自动生成）
            
        Returns:
            生成的HTML文件路径
        """
        # 确定输出文件名
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluation_report_{timestamp}.html"
        
        output_path = self.output_dir / output_filename
        
        # 提取数据
        timestamp = analysis.get('timestamp', datetime.now().isoformat())
        overall_score = analysis.get('overall_score', 0.0)
        dimension_scores = analysis.get('dimension_scores', {})
        matrix_results = analysis.get('matrix_results', {})
        summary = analysis.get('summary', {})
        
        # 准备雷达图数据
        radar_labels = [
            f"Harvest Level {i}" for i in range(1, 5)
        ] + [
            f"Combat Level {i}" for i in range(1, 5)
        ] + [
            f"TechTree Level {i}" for i in range(1, 5)
        ]
        
        radar_data = []
        for dim in ['harvest', 'combat', 'techtree']:
            if dim in matrix_results:
                for level in range(1, 5):
                    level_data = matrix_results[dim]['levels'].get(str(level), {})
                    radar_data.append(level_data.get('avg_success_rate', 0.0))
            else:
                radar_data.extend([0.0] * 4)
        
        # 准备柱状图数据
        bar_labels = ['Harvest', 'Combat', 'TechTree']
        bar_data = [
            dimension_scores.get('harvest', 0.0),
            dimension_scores.get('combat', 0.0),
            dimension_scores.get('techtree', 0.0)
        ]
        
        # 准备任务级别图表数据
        task_labels = []
        task_success_data = []
        task_steps_data = []
        
        # 收集所有任务数据
        for dim_key in ['harvest', 'combat', 'techtree']:
            if dim_key in matrix_results:
                for level in range(1, 5):
                    level_data = matrix_results[dim_key]['levels'].get(str(level), {})
                    for task in level_data.get('tasks', []):
                        task_id = task.get('task_id', '')
                        success_rate = task.get('success_rate', 0.0)
                        avg_steps = task.get('avg_steps', 0.0)
                        
                        # 简化任务ID显示
                        task_label = task_id.replace('harvest_', '').replace('combat_', '').replace('techtree_', '')
                        task_label = task_label.replace('_', ' ').title()
                        if len(task_label) > 25:
                            task_label = task_label[:22] + '...'
                        
                        task_labels.append(task_label)
                        task_success_data.append(success_rate)
                        task_steps_data.append(avg_steps if avg_steps else 0)
        
        # 生成维度HTML
        dimensions_html = self._generate_dimensions_html(matrix_results)
        
        # 生成建议HTML
        recommendations_html = self._generate_recommendations_html(summary)
        
        # 填充模板
        html_content = self.HTML_TEMPLATE.format(
            timestamp=timestamp.replace('T', ' ').split('.')[0],
            overall_score=overall_score,
            total_tasks=summary.get('total_tasks', 0),
            config_file=config_file,
            dimensions_html=dimensions_html,
            recommendations_html=recommendations_html,
            radar_labels=json.dumps(radar_labels),
            radar_data=json.dumps(radar_data),
            bar_labels=json.dumps(bar_labels),
            bar_data=json.dumps(bar_data),
            task_labels=json.dumps(task_labels),
            task_success_data=json.dumps(task_success_data),
            task_steps_data=json.dumps(task_steps_data),
        )
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML报告已生成: {output_path}")
        return output_path
    
    def _generate_dimensions_html(self, matrix_results: Dict) -> str:
        """生成维度详情HTML"""
        html_parts = []
        
        dimension_names = {
            'harvest': 'Harvest (采集)',
            'combat': 'Combat (战斗)',
            'techtree': 'TechTree (科技树)'
        }
        
        for dim_key in ['harvest', 'combat', 'techtree']:
            if dim_key not in matrix_results:
                continue
            
            dim_data = matrix_results[dim_key]
            dim_name = dimension_names.get(dim_key, dim_key)
            
            html_parts.append(f'''
            <div class="dimension-section">
                <div class="dimension-header">{dim_name}</div>
                <div class="dimension-body">
            ''')
            
            for level in range(1, 5):
                level_key = str(level)
                if level_key not in dim_data['levels']:
                    continue
                
                level_data = dim_data['levels'][level_key]
                level_name = level_data['name']
                task_count = level_data['task_count']
                avg_success = level_data['avg_success_rate']
                avg_steps = level_data.get('avg_steps')
                tasks = level_data.get('tasks', [])
                
                # 确定成功率颜色类
                if avg_success >= 0.7:
                    rate_class = 'success-rate-high'
                elif avg_success >= 0.4:
                    rate_class = 'success-rate-medium'
                else:
                    rate_class = 'success-rate-low'
                
                html_parts.append(f'''
                <div class="level-row">
                    <div class="level-header">
                        <div class="level-name">Level {level} - {level_name}</div>
                        <div class="level-stats">
                            <span>{task_count}个任务</span>
                            <span class="{rate_class}">成功率: {avg_success:.1%}</span>
                            {f'<span>平均步数: {avg_steps:.0f}</span>' if avg_steps else ''}
                        </div>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {avg_success * 100}%"></div>
                    </div>
                ''')
                
                if tasks:
                    html_parts.append('<div class="task-list">')
                    for task in tasks:
                        task_id = task.get('task_id', 'Unknown')
                        task_rate = task.get('success_rate', 0.0)
                        
                        if task_rate >= 0.7:
                            task_rate_class = 'success-rate-high'
                        elif task_rate >= 0.4:
                            task_rate_class = 'success-rate-medium'
                        else:
                            task_rate_class = 'success-rate-low'
                        
                        html_parts.append(f'''
                        <div class="task-item">
                            <span class="task-id">{task_id}</span>
                            <span class="task-rate {task_rate_class}">{task_rate:.1%}</span>
                        </div>
                        ''')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
            
            html_parts.append('</div></div>')
        
        return ''.join(html_parts)
    
    def _generate_recommendations_html(self, summary: Dict) -> str:
        """生成建议HTML"""
        recommendations = summary.get('recommendations', [])
        
        if not recommendations:
            return ''
        
        html = '<div class="recommendations"><h2>💡 分析建议</h2><ul>'
        
        for rec in recommendations:
            html += f'<li>{rec}</li>'
        
        html += '</ul></div>'
        
        return html

