#!/usr/bin/env python
"""
生成 Steve1 策略评估的 HTML 报告（Policy Evaluation Report）

策略评估 vs 结果评估：
- 策略评估（本模块）：分析模型行为、策略质量、瓶颈识别
- 结果评估（eval_framework）：评估任务成功率、完成情况
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def image_to_base64(image_path: Path) -> str:
    """将图片转换为 base64 编码"""
    if not image_path.exists():
        return ""
    
    with open(image_path, 'rb') as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')


def load_json(json_path: Path) -> Dict:
    """加载 JSON 文件"""
    if not json_path.exists():
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_policy_html_report(results_dir: Path, output_path: Path = None):
    """
    生成 Steve1 策略评估 HTML 报告
    
    Args:
        results_dir: 结果目录 (results/policy_evaluation 或 results/deep_evaluation)
        output_path: 输出 HTML 文件路径（默认为 results_dir/policy_evaluation_report.html）
    """
    if output_path is None:
        output_path = results_dir / "policy_evaluation_report.html"
    
    # 加载数据
    summary = load_json(results_dir / "summary_report.json")
    
    # 获取图片
    prior_dir = results_dir / "prior_analysis"
    end_to_end_dir = results_dir / "end_to_end"
    
    # 转换图片为 base64
    images = {}
    
    # Prior 分析图片
    for img_name in ['embedding_space_tsne', 'embedding_space_pca', 
                     'similarity_matrix', 'quality_metrics']:
        img_path = prior_dir / f"{img_name}.png"
        if img_path.exists():
            images[img_name] = image_to_base64(img_path)
    
    # 端到端分析图片和详细数据
    e2e_images = {}
    e2e_details = {}
    if end_to_end_dir.exists():
        for task_dir in end_to_end_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                bottleneck_img = task_dir / "bottleneck_analysis.png"
                if bottleneck_img.exists():
                    e2e_images[task_name] = image_to_base64(bottleneck_img)
                
                # 加载端到端详细数据
                e2e_json = task_dir / f"{task_name}_end_to_end.json"
                if e2e_json.exists():
                    e2e_details[task_name] = load_json(e2e_json)
    
    # 生成 HTML
    html_content = generate_html_content(summary, images, e2e_images, e2e_details)
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML 报告已生成: {output_path}")
    return output_path


def generate_html_content(summary: Dict, images: Dict, e2e_images: Dict, e2e_details: Dict = None) -> str:
    """生成 HTML 内容"""
    
    if e2e_details is None:
        e2e_details = {}
    
    timestamp = summary.get('timestamp', datetime.now().isoformat())
    prior = summary.get('prior_analysis', {})
    e2e = summary.get('end_to_end_analysis', {})
    recommendations = summary.get('recommendations', [])
    
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steve1 深度评估报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 16px;
            opacity: 0.9;
        }}
        
        .header .timestamp {{
            margin-top: 15px;
            font-size: 14px;
            opacity: 0.8;
        }}
        
        /* Navigation */
        .nav {{
            background: #f8f9fa;
            padding: 15px 40px;
            border-bottom: 1px solid #e9ecef;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        .nav-links {{
            list-style: none;
            display: flex;
            gap: 30px;
            justify-content: center;
        }}
        
        .nav-links a {{
            color: #495057;
            text-decoration: none;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.3s;
        }}
        
        .nav-links a:hover {{
            background: #667eea;
            color: white;
        }}
        
        /* Content */
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 28px;
            color: #2d3748;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            font-weight: 600;
        }}
        
        /* Summary Cards */
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        .card-title {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .card-value {{
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .card-subtitle {{
            font-size: 13px;
            opacity: 0.8;
        }}
        
        .card.success {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        
        .card.warning {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        
        .card.info {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        /* Metrics Table */
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 30px;
        }}
        
        .metrics-table th {{
            background: #f8f9fa;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #e9ecef;
        }}
        
        .metrics-table td {{
            padding: 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .metrics-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge.high {{
            background: #fee;
            color: #c33;
        }}
        
        .badge.medium {{
            background: #ffeaa7;
            color: #d63031;
        }}
        
        .badge.low {{
            background: #dfe6e9;
            color: #636e72;
        }}
        
        /* Images */
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        
        .image-card {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}
        
        .image-card h3 {{
            font-size: 18px;
            color: #2d3748;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .image-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .image-card .description {{
            margin-top: 10px;
            font-size: 14px;
            color: #666;
            line-height: 1.6;
        }}
        
        /* Recommendations */
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        
        .recommendations h3 {{
            color: #856404;
            margin-bottom: 15px;
            font-size: 20px;
        }}
        
        .recommendation-item {{
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 3px solid #ffc107;
        }}
        
        .recommendation-item:last-child {{
            margin-bottom: 0;
        }}
        
        .recommendation-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        
        .recommendation-text {{
            color: #495057;
            line-height: 1.6;
        }}
        
        /* Footer */
        .footer {{
            background: #f8f9fa;
            padding: 30px 40px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            .cards {{
                grid-template-columns: 1fr;
            }}
            
            .image-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🎯 Steve1 深度评估报告</h1>
            <div class="subtitle">Prior 模型 & Policy 模型性能分析</div>
            <div class="timestamp">📅 生成时间: {timestamp}</div>
        </div>
        
        <!-- Navigation -->
        <nav class="nav">
            <ul class="nav-links">
                <li><a href="#overview">总览</a></li>
                <li><a href="#prior">Prior 分析</a></li>
                <li><a href="#end-to-end">端到端分析</a></li>
                <li><a href="#recommendations">建议</a></li>
            </ul>
        </nav>
        
        <!-- Content -->
        <div class="content">
            <!-- Overview Section -->
            <section id="overview" class="section">
                <h2 class="section-title">📊 评估总览</h2>
                
                <div class="cards">
                    <div class="card success">
                        <div class="card-title">成功率</div>
                        <div class="card-value">{e2e.get('avg_success_rate', 0) * 100:.1f}%</div>
                        <div class="card-subtitle">端到端任务完成率</div>
                    </div>
                    
                    <div class="card warning">
                        <div class="card-title">Prior 相似度</div>
                        <div class="card-value">{prior.get('avg_text_to_prior_similarity', 0):.3f}</div>
                        <div class="card-subtitle">文本-嵌入对齐程度</div>
                    </div>
                    
                    <div class="card info">
                        <div class="card-title">测试指令数</div>
                        <div class="card-value">{prior.get('num_instructions', 0)}</div>
                        <div class="card-subtitle">Prior 模型评估</div>
                    </div>
                    
                    <div class="card info">
                        <div class="card-title">评估任务数</div>
                        <div class="card-value">{e2e.get('num_tasks', 0)}</div>
                        <div class="card-subtitle">端到端评估</div>
                    </div>
                </div>
                
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                            <th>说明</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>文本-Prior 平均相似度</strong></td>
                            <td>{prior.get('avg_text_to_prior_similarity', 0):.4f}</td>
                            <td>MineCLIP 文本嵌入与 Prior 输出的余弦相似度</td>
                        </tr>
                        <tr>
                            <td><strong>相似度范围</strong></td>
                            <td>{prior.get('min_similarity', 0):.4f} - {prior.get('max_similarity', 0):.4f}</td>
                            <td>最小值到最大值的跨度</td>
                        </tr>
                        <tr>
                            <td><strong>平均成功率</strong></td>
                            <td>{e2e.get('avg_success_rate', 0) * 100:.1f}%</td>
                            <td>任务完成的百分比</td>
                        </tr>
                        <tr>
                            <td><strong>Prior 贡献</strong></td>
                            <td>{e2e.get('avg_stage1_contribution', 0) * 100:.1f}%</td>
                            <td>Prior 模型对成功的贡献度</td>
                        </tr>
                        <tr>
                            <td><strong>Policy 贡献</strong></td>
                            <td>{e2e.get('avg_stage2_contribution', 0) * 100:.1f}%</td>
                            <td>Policy 模型对成功的贡献度</td>
                        </tr>
                    </tbody>
                </table>
            </section>
            
            <!-- Prior Analysis Section -->
            <section id="prior" class="section">
                <h2 class="section-title">🎨 Prior 模型分析</h2>
                
                <p style="margin-bottom: 20px; color: #666; font-size: 15px;">
                    Prior 模型 p(z<sub>τ<sup>goal</sup></sub> | y) 负责将 MineCLIP 文本嵌入转换为"视觉风格"的目标嵌入。
                    以下可视化展示了 Prior 模型的嵌入空间特性和质量指标。
                </p>
                
                <div class="image-grid">
"""
    
    # Prior 分析图片
    if 'embedding_space_tsne' in images:
        html += f"""
                    <div class="image-card">
                        <h3>📈 t-SNE 嵌入空间</h3>
                        <img src="data:image/png;base64,{images['embedding_space_tsne']}" alt="t-SNE">
                        <div class="description">
                            使用 t-SNE 降维到 2D 空间，展示文本嵌入（蓝色）和 Prior 输出（绿色）的分布。
                            箭头表示文本 → Prior 的转换方向。理想情况下，箭头应该指向相似的语义区域。
                        </div>
                    </div>
"""
    
    if 'embedding_space_pca' in images:
        html += f"""
                    <div class="image-card">
                        <h3>📊 PCA 嵌入空间</h3>
                        <img src="data:image/png;base64,{images['embedding_space_pca']}" alt="PCA">
                        <div class="description">
                            使用 PCA 降维，保留最大方差方向。与 t-SNE 相比，PCA 更关注全局结构。
                            观察转换箭头的方向和长度，评估 Prior 模型的转换质量。
                        </div>
                    </div>
"""
    
    if 'similarity_matrix' in images:
        html += f"""
                    <div class="image-card">
                        <h3>🔥 相似度矩阵</h3>
                        <img src="data:image/png;base64,{images['similarity_matrix']}" alt="Similarity Matrix">
                        <div class="description">
                            不同指令之间的 Prior 输出相似度。对角线应该最亮（自己与自己），
                            相似任务应该有较高相似度（如 "dig dirt" vs "dig sand"）。
                        </div>
                    </div>
"""
    
    if 'quality_metrics' in images:
        html += f"""
                    <div class="image-card">
                        <h3>📉 质量指标</h3>
                        <img src="data:image/png;base64,{images['quality_metrics']}" alt="Quality Metrics">
                        <div class="description">
                            Prior 模型的关键质量指标：文本-Prior 相似度分布、Prior 输出方差等。
                            高质量的 Prior 应该有较高的相似度和适中的方差（既不过于相似，也不过于分散）。
                        </div>
                    </div>
"""
    
    html += """
                </div>
            </section>
            
            <!-- End-to-End Analysis Section -->
            <section id="end-to-end" class="section">
                <h2 class="section-title">🎯 端到端分析</h2>
                
                <p style="margin-bottom: 20px; color: #666; font-size: 15px;">
                    端到端分析评估完整的 Steve1 两阶段模型：Prior p(z | y) + Policy p(τ | z)。
                    通过对比使用 Prior 嵌入和真实视觉嵌入的表现，识别性能瓶颈。
                </p>
"""
    
    # 瓶颈分布统计
    bottleneck_dist = e2e.get('bottleneck_distribution', {})
    if bottleneck_dist:
        html += f"""
                <div class="cards" style="margin-bottom: 30px;">
                    <div class="card success">
                        <div class="card-title">无瓶颈</div>
                        <div class="card-value">{bottleneck_dist.get('no_bottleneck', 0)}</div>
                        <div class="card-subtitle">Prior 和 Policy 都表现良好</div>
                    </div>
                    
                    <div class="card warning">
                        <div class="card-title">Prior 瓶颈</div>
                        <div class="card-value">{bottleneck_dist.get('prior_bottleneck', 0)}</div>
                        <div class="card-subtitle">Prior 模型限制了性能</div>
                    </div>
                    
                    <div class="card warning">
                        <div class="card-title">Policy 瓶颈</div>
                        <div class="card-value">{bottleneck_dist.get('policy_bottleneck', 0)}</div>
                        <div class="card-subtitle">Policy 模型限制了性能</div>
                    </div>
                </div>
"""
    
    # 端到端详细指标表格
    if e2e_details:
        html += """
                <h3 style="margin: 30px 0 15px 0; font-size: 20px; color: #2d3748;">📋 端到端详细指标</h3>
                <p style="margin-bottom: 20px; color: #666; font-size: 14px;">
                    以下表格展示了每个任务的Prior模型和Policy模型的详细指标，帮助您深入理解两阶段模型的性能。
                </p>
"""
        for task_name, task_data_list in e2e_details.items():
            if not task_data_list:
                continue
            
            html += f"""
                <div style="margin-bottom: 40px;">
                    <h4 style="color: #667eea; margin-bottom: 15px;">🎯 任务: {task_name}</h4>
"""
            
            for trial_data in task_data_list:
                prior_result = trial_data.get('prior_result', {})
                policy_result = trial_data.get('policy_result', {})
                
                html += f"""
                    <table class="metrics-table" style="margin-bottom: 20px;">
                        <thead>
                            <tr>
                                <th colspan="3" style="background: #667eea; color: white;">🔹 Prior 模型指标 (文本→嵌入)</th>
                            </tr>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>定义与参考</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>文本-Prior 相似度</strong></td>
                                <td>{prior_result.get('text_to_prior_similarity', 0):.4f}</td>
                                <td>MineCLIP文本嵌入与Prior输出的余弦相似度。<br>
                                    <strong>参考:</strong> &gt;0.5优秀, 0.3-0.5良好, &lt;0.3需改进</td>
                            </tr>
                            <tr>
                                <td><strong>Prior 方差</strong></td>
                                <td>{prior_result.get('prior_variance', 0):.6f}</td>
                                <td>Prior输出的方差，衡量嵌入多样性。<br>
                                    <strong>参考:</strong> 适中最佳（0.0001-0.001），过高过低都不理想</td>
                            </tr>
                            <tr>
                                <td><strong>重建质量</strong></td>
                                <td>{prior_result.get('reconstruction_quality', 0):.4f}</td>
                                <td>VAE重建质量评分。<br>
                                    <strong>参考:</strong> 越高越好，表示Prior能很好地保留文本信息</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th colspan="3" style="background: #43e97b; color: white;">🔹 Policy 模型指标 (嵌入→动作)</th>
                            </tr>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>定义与参考</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>总步数</strong></td>
                                <td>{policy_result.get('total_steps', 0)}</td>
                                <td>完成任务所用的步数。<br>
                                    <strong>参考:</strong> 步数少说明效率高（前提是成功）</td>
                            </tr>
                            <tr>
                                <td><strong>动作多样性 (熵)</strong></td>
                                <td>{policy_result.get('action_diversity', 0):.4f}</td>
                                <td>动作熵，衡量动作分布的多样性。<br>
                                    <strong>参考:</strong> 1.5-2.5适中，&lt;1.0单调，&gt;3.0混乱</td>
                            </tr>
                            <tr>
                                <td><strong>时序一致性</strong></td>
                                <td>{policy_result.get('temporal_consistency', 0):.4f}</td>
                                <td>相邻动作的一致性（平滑度）。<br>
                                    <strong>参考:</strong> &gt;0.85优秀，0.7-0.85良好，&lt;0.7抖动</td>
                            </tr>
                            <tr>
                                <td><strong>重复动作比例</strong></td>
                                <td>{policy_result.get('repeated_action_ratio', 0):.2%}</td>
                                <td>连续重复相同动作的比例。<br>
                                    <strong>参考:</strong> 30-60%正常，&gt;80%可能卡住，&lt;20%不稳定</td>
                            </tr>
                            <tr>
                                <td><strong>任务成功</strong></td>
                                <td>{'✅ 是' if policy_result.get('success') else '❌ 否'}</td>
                                <td>是否成功完成任务</td>
                            </tr>
                            <tr>
                                <td><strong>最终奖励</strong></td>
                                <td>{policy_result.get('final_reward', 0):.1f}</td>
                                <td>任务结束时的奖励值</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th colspan="3" style="background: #fa709a; color: white;">🔹 联合分析指标</th>
                            </tr>
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                                <th>定义与参考</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Prior 贡献度</strong></td>
                                <td>{trial_data.get('stage1_contribution', 0):.1%}</td>
                                <td>Prior模型对成功的贡献。<br>
                                    <strong>参考:</strong> &gt;60%说明Prior很重要，需优先优化</td>
                            </tr>
                            <tr>
                                <td><strong>Policy 贡献度</strong></td>
                                <td>{trial_data.get('stage2_contribution', 0):.1%}</td>
                                <td>Policy模型对成功的贡献。<br>
                                    <strong>参考:</strong> &gt;60%说明Policy很重要，需优先优化</td>
                            </tr>
                            <tr>
                                <td><strong>瓶颈阶段</strong></td>
                                <td>{['无瓶颈', 'Prior瓶颈', 'Policy瓶颈'][trial_data.get('bottleneck_stage', 0)]}</td>
                                <td>性能限制主要来自哪个阶段。<br>
                                    <strong>参考:</strong> 针对瓶颈阶段优化能获得最大提升</td>
                            </tr>
                        </tbody>
                    </table>
"""
            
            html += """
                </div>
"""
    
    # 端到端分析图片
    if e2e_images:
        html += """
                <h3 style="margin: 30px 0 15px 0; font-size: 20px; color: #2d3748;">📊 可视化分析</h3>
                <div class="image-grid">
"""
        for task_name, img_base64 in e2e_images.items():
            html += f"""
                    <div class="image-card">
                        <h3>🎮 {task_name}</h3>
                        <img src="data:image/png;base64,{img_base64}" alt="{task_name}">
                        <div class="description">
                            <strong>图表说明：</strong><br>
                            • 左图：使用Prior嵌入 vs 真实视觉嵌入的成功率对比<br>
                            • 右图：瓶颈阶段分布饼图（无瓶颈/Prior瓶颈/Policy瓶颈）<br>
                            <strong>如何分析：</strong>如果Prior成功率显著低于真实视觉，说明Prior是瓶颈
                        </div>
                    </div>
"""
        html += """
                </div>
"""
    
    html += """
            </section>
            
            <!-- Recommendations Section -->
            <section id="recommendations" class="section">
                <h2 class="section-title">💡 改进建议</h2>
"""
    
    if recommendations:
        html += """
                <div class="recommendations">
                    <h3>⚠️ 需要关注的问题</h3>
"""
        for rec in recommendations:
            priority = rec.get('priority', 'low')
            component = rec.get('component', 'unknown')
            issue = rec.get('issue', '')
            suggestion = rec.get('suggestion', '')
            
            html += f"""
                    <div class="recommendation-item">
                        <div class="recommendation-header">
                            <span class="badge {priority}">{priority.upper()}</span>
                            <strong>{component.upper()} 组件</strong>
                        </div>
                        <div class="recommendation-text">
                            <strong>问题：</strong>{issue}<br>
                            <strong>建议：</strong>{suggestion}
                        </div>
                    </div>
"""
        html += """
                </div>
"""
    else:
        html += """
                <p style="color: #28a745; font-size: 16px; font-weight: 600;">
                    ✅ 太棒了！当前模型表现优秀，无需特别改进。
                </p>
"""
    
    html += """
            </section>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>
                📚 详细文档请参阅: 
                <a href="../docs/guides/DEEP_EVALUATION_METRICS_EXPLAINED.md">评估指标详解</a> | 
                <a href="../docs/guides/STEVE1_DEEP_EVALUATION_GUIDE.md">深度评估指南</a>
            </p>
            <p style="margin-top: 10px; font-size: 13px;">
                Steve1 Deep Evaluation System v1.0 | Powered by MineCLIP & VPT
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("results/policy_evaluation")
    
    generate_html_report(results_dir)

