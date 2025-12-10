"""
评估报告 HTML 生成器 (Evaluation Report Generator)
=====================================================

生成完整的 Prior 和 Policy 评估报告，包括：

第一部分：核心任务结果表格
- Prior 指标：目标嵌入基线、Prior 目标嵌入
- Policy 指标：接近率基线、Policy 接近率、成功率

第二部分：辅助指标表格
- Prior 辅助指标：变体鲁棒性、区分度、输出方差
- Policy 辅助指标：动作相似度、Camera相似度、动作熵、时序平滑度、动作覆盖率

第三部分：可视化分析
- Prior 可视化：t-SNE 嵌入对比、相似度矩阵、方差分布
- Policy 可视化：目标接近度、动作分布、混淆矩阵

作者: AI Assistant
日期: 2025-12-09
"""

import json
import base64
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from .html_styles import UNIFIED_CSS, get_badge_class, get_level_text


class EvaluationReportGenerator:
    """评估报告 HTML 生成器 (Prior + Policy)"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        results: Dict,
        output_filename: str = "prior_evaluation_report.html"
    ) -> Path:
        """生成 HTML 报告"""
        html_content = self._generate_html(results)
        
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html(self, results: Dict) -> str:
        """生成完整的 HTML 内容"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prior 和 Policy 评估报告</title>
    <style>
        {UNIFIED_CSS}
        {self._get_prior_specific_css()}
    </style>
</head>
<body>
        {self._generate_header(results)}
    
    <div class="container">
        {self._generate_overview_stats(results)}
        {self._generate_task_results_table(results)}
        {self._generate_auxiliary_metrics_table(results)}
        {self._generate_visualization_section(results)}
    </div>
    
    <div class="footer">
        STEVE-1 Prior Evaluation Framework | AIMC Project
    </div>
    
    <!-- 图片放大模态框 -->
    <div class="modal-overlay" id="imageModal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImage" src="" alt="放大图片">
    </div>
    
    <script>
        // 图片点击放大
        document.querySelectorAll('.viz-card img').forEach(img => {{
            img.onclick = function(e) {{
                e.stopPropagation();
                document.getElementById('modalImage').src = this.src;
                document.getElementById('imageModal').classList.add('active');
            }};
        }});
        
        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
        }}
        
        // ESC 键关闭
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') closeModal();
        }});
    </script>
</body>
</html>"""
    
    def _get_prior_specific_css(self) -> str:
        """Prior 特定的 CSS 样式 - 学术简洁风格"""
        return """
        /* 学术简洁风格 - 减少颜色使用 */
        body {
            font-family: 'Times New Roman', 'Noto Serif SC', serif;
            color: #333;
            line-height: 1.6;
        }
        
        /* 核心任务结果表格 */
        .task-results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }
        
        .task-results-table th {
            background: #f5f5f5;
            color: #333;
            font-weight: 700;
            padding: 12px 10px;
            text-align: center;
            border: 1px solid #ddd;
            font-size: 0.85em;
        }
        
        .task-results-table th:first-child {
            text-align: left;
            padding-left: 15px;
        }
        
        .task-results-table td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
        
        .task-results-table td:first-child {
            text-align: left;
            padding-left: 15px;
            font-weight: 600;
        }
        
        .task-results-table tr:nth-child(even) {
            background: #fafafa;
        }
        
        .task-results-table .avg-row {
            background: #f0f0f0;
            font-weight: 700;
        }
        
        .task-results-table .avg-row td {
            border-top: 2px solid #333;
        }
        
        /* 表头注释 */
        .th-note {
            font-weight: 400;
            font-size: 0.8em;
            color: #666;
            display: block;
            margin-top: 3px;
        }
        
        /* 指标单元格 */
        .sim-value {
            font-family: 'Consolas', monospace;
            font-weight: normal;
        }
        
        .sim-value.bold {
            font-weight: 600;
        }
        
        .sim-value.best {
            text-decoration: underline;
        }
        
        .sim-delta {
            font-size: 0.85em;
            margin-left: 5px;
        }
        
        .delta-positive { color: #2e7d32; }
        .delta-negative { color: #c62828; }
        .delta-neutral { color: #666; }
        
        /* 评估逻辑区块 */
        .eval-logic-block {
            background: #f8f8f8;
            border: 1px solid #ddd;
            padding: 15px 20px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
        
        .eval-logic-block .title {
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .eval-logic-block code {
            background: #fff;
            padding: 1px 5px;
            border: 1px solid #ddd;
            font-family: 'Consolas', monospace;
        }
        
        .logic-formulas {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 12px;
        }
        
        .formula-item {
            background: #fff;
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
        
        .formula-item .label {
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .formula-item .formula {
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
            color: #555;
        }
        
        .formula-item .note {
            font-size: 0.75em;
            color: #888;
            margin-top: 3px;
        }
        
        .viz-desc {
            margin-top: 12px;
            font-size: 0.85em;
            color: #666;
            line-height: 1.5;
        }
        
        /* 洞察卡片 - 简洁风格（未使用但保留兼容） */
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin: 15px 0;
        }
        
        .insight-card {
            background: #fafafa;
            border: 1px solid #ddd;
            padding: 15px;
            text-align: center;
        }
        
        .insight-card .icon {
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        
        .insight-card .title {
            font-weight: 700;
            color: #333;
            margin-bottom: 6px;
        }
        
        .insight-card .value {
            font-size: 1.5em;
            font-weight: 700;
            margin-bottom: 6px;
        }
        
        .insight-card .desc {
            font-size: 0.8em;
            color: #666;
        }
        
        /* Prior vs MineCLIP 对比 */
        .comparison-table {
            width: 100%;
            margin: 15px 0;
        }
        
        .comparison-table th, .comparison-table td {
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }
        
        .comparison-table th {
            background: #f8f9fc;
            font-weight: 600;
        }
        
        .gain-positive { color: #2e7d32; }
        .gain-negative { color: #c62828; }
        .gain-neutral { color: #666; }
        
        /* 评估逻辑说明 - 简洁风格（已移至表格上方） */
        .logic-explanation {
            background: #fafafa;
            border: 1px solid #ddd;
            padding: 15px 20px;
            margin-bottom: 15px;
        }
        
        .logic-flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-top: 10px;
        }
        
        .logic-box {
            background: white;
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: center;
        }
        
        .logic-box.sample {
            border: 1px solid #333;
        }
        
        .logic-box.eval {
            border: 1px solid #333;
        }
        
        .logic-box .box-title {
            font-size: 0.75em;
            color: #888;
            margin-bottom: 5px;
        }
        
        .logic-box .box-content {
            font-weight: 600;
            color: #333;
        }
        
        .logic-arrow {
            font-size: 1.5em;
            color: #888;
        }
        
        /* 列标签样式 */
        .col-label {
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 4px;
        }
        
        .col-label.baseline {
            background: #f5f5f5;
            color: #333;
        }
        
        .col-label.prior {
            background: #eee;
            color: #333;
            font-weight: 700;
        }
        
        .col-label.variant {
            background: #f5f5f5;
            color: #333;
        }
        
        .col-note {
            color: #888;
            font-style: italic;
        }
        
        /* 相似度单元格 */
        .sim-cell {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            padding: 5px;
        }
        
        .sim-value {
            font-size: 1.1em;
            font-weight: normal;
            color: #333;
        }
        
        .sim-value.bold {
            font-weight: 600;
        }
        
        .sim-value.best {
            font-weight: 700;
            text-decoration: underline;
        }
        
        .sim-bar {
            width: 100px;
            height: 8px;
            background: #e8eaf0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .sim-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .sim-fill.baseline-fill {
            background: #888;
        }
        
        .sim-fill.prior-fill {
            background: #444;
        }
        
        .sim-fill.variant-fill {
            background: #666;
        }
        
        .sim-label {
            font-size: 0.75em;
            color: #888;
        }
        
        .sim-delta {
            font-size: 0.85em;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 3px;
        }
        
        .delta-positive {
            color: #2e7d32;
        }
        
        .delta-negative {
            color: #c62828;
        }
        
        .delta-neutral {
            color: #666;
        }
        
        .sim-note {
            font-size: 0.8em;
            color: #888;
        }
        
        .sim-sep {
            color: #aaa;
            margin: 0 2px;
            font-size: 0.9em;
        }
        
        /* 分组表头样式 */
        .group-header-row th {
            padding: 8px 10px;
            font-weight: 600;
        }
        
        .group-header {
            text-align: center;
            border-bottom: 2px solid #333 !important;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        .prior-group {
            background: #f5f5f5;
            border-left: 1px solid #ddd;
            border-right: 1px solid #ddd;
        }
        
        .policy-group {
            background: #f5f5f5;
            border-left: 1px solid #ddd;
            border-right: 1px solid #ddd;
        }
        
        .task-cell {
            font-weight: 600;
            color: #333;
        }
        
        /* 结果解读（已移除，保留兼容） */
        .result-interpretation {
            display: none;
        }
        
        .auxiliary-metrics {
            display: none;
        }
        
        /* 概览卡片 - 简洁风格（未使用但保留兼容） */
        .insight-card.baseline-card {
            border-left: 3px solid #888;
        }
        
        .insight-card.prior-card {
            border-left: 3px solid #333;
        }
        
        .insight-card.variant-card {
            border-left: 3px solid #666;
        }
        
        .gain-badge {
            margin-top: 8px;
            padding: 3px 10px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .gain-badge.delta-positive {
            color: #2e7d32;
        }
        
        .gain-badge.delta-negative {
            color: #c62828;
        }
        
        .gain-badge.delta-neutral {
            color: #666;
        }
        
        /* 评估环境 - 学术简洁风格 */
        .env-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .env-card {
            background: #fafafa;
            border: 1px solid #ddd;
            padding: 12px 15px;
        }
        
        .env-title {
            font-weight: 700;
            color: #333;
            margin-bottom: 10px;
            font-size: 0.9em;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        
        .env-table {
            width: 100%;
            font-size: 0.85em;
        }
        
        .env-table td {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        
        .env-table td:first-child {
            color: #666;
            width: 30%;
        }
        
        .env-table code {
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
        }
        
        .env-size {
            text-align: right;
            color: #888;
            font-size: 0.9em;
        }
        
        /* 任务清单 - 简洁风格 */
        .task-list-section {
            background: #fafafa;
            border: 1px solid #ddd;
            padding: 12px 15px;
            margin-bottom: 15px;
        }
        
        .task-categories {
            font-size: 0.85em;
        }
        
        .task-category {
            margin-bottom: 6px;
            line-height: 1.6;
        }
        
        .category-label {
            font-weight: 700;
            margin-right: 8px;
        }
        
        .task-tags {
            display: inline;
        }
        
        .task-tag {
            display: inline;
            color: #555;
        }
        
        .task-tag:not(:last-child)::after {
            content: ", ";
        }
        
        /* 可视化 - 学术风格 */
        .viz-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        
        .viz-card {
            background: #fff;
            border: 1px solid #ddd;
            padding: 15px;
        }
        
        .viz-card.full-width {
            grid-column: 1 / -1;
        }
        
        .viz-card h3 {
            margin: 0 0 10px 0;
            font-size: 0.95em;
            font-weight: 700;
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }
        
        .viz-column-label {
            display: inline-block;
            background: #f5f5f5;
            border: 1px solid #ddd;
            padding: 2px 8px;
            font-size: 0.75em;
            margin-bottom: 10px;
        }
        
        .viz-card img {
            width: 100%;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
        
        .viz-explanation {
            background: #fafafa;
            border: 1px solid #ddd;
            padding: 10px;
            font-size: 0.85em;
            line-height: 1.5;
        }
        
        .viz-definition {
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px dashed #ccc;
        }
        
        .viz-definition strong,
        .viz-interpretation strong {
            font-weight: 700;
        }
        
        .viz-interpretation {
            color: #555;
        }
        
        /* 响应式 */
        @media (max-width: 768px) {
            .viz-grid { grid-template-columns: 1fr; }
            .viz-card.full-width { grid-column: span 1; }
            .env-grid { grid-template-columns: 1fr; }
            .logic-formulas { grid-template-columns: 1fr; }
        }
        
        /* 图片点击放大 - 模态框 */
        .viz-card img {
            cursor: pointer;
            transition: transform 0.2s;
        }
        .viz-card img:hover {
            transform: scale(1.02);
        }
        
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            cursor: pointer;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal-overlay img {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }
        .modal-close {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 30px;
            cursor: pointer;
        }
        
        /* Tier/Category 合并单元格样式 */
        .tier-cell {
            text-align: center;
            font-weight: 700;
            background: #f5f5f5;
            vertical-align: middle;
        }
        .category-cell {
            text-align: center;
            font-size: 0.85em;
            color: #555;
            vertical-align: middle;
        }
        """
    
    def _generate_header(self, results: Dict) -> str:
        """生成页眉 - 学术风格"""
        n_tasks = results.get('n_tasks', 0)
        has_goal_progress = bool(results.get('task_results', {}).get('goal_progress', {}).get('task_progress', {}))
        
        subtitle = "Prior: Instruction → Goal Embedding | Policy: Goal Embedding → Action Sequences" if has_goal_progress else "Instruction → Goal Video Embedding Transformation Quality Assessment"
        
        return f"""
        <div class="header" style="background: #fff; border-bottom: 2px solid #333; padding: 20px 0;">
            <h1 style="font-size: 1.8em; font-weight: 700; margin: 0 0 8px 0; color: #333;">Prior 和 Policy 评估报告</h1>
            <div style="font-size: 1em; color: #555; margin-bottom: 10px;">{subtitle}</div>
            <div style="font-size: 0.85em; color: #888;">
                {n_tasks} tasks | {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """
    
    def _generate_overview_stats(self, results: Dict) -> str:
        """生成评估环境信息 - 学术简洁风格"""
        import os
        
        config_file = results.get('config_file', 'N/A')
        task_ids = results.get('task_ids', [])
        task_info = results.get('task_info', {})
        n_tasks = results.get('n_tasks', len(task_ids))
        
        # 判断是否有 Policy 评估
        has_goal_progress = bool(results.get('task_results', {}).get('goal_progress', {}).get('task_progress', {}))
        
        # Prior 模型权重
        prior_weights = "data/weights/steve1/steve1_prior.pt"
        mineclip_weights = "data/weights/mineclip/attn.pth"
        
        # Policy 模型权重
        vpt_model = "data/weights/vpt/2x.model"
        policy_weights = "data/weights/steve1/steve1.weights"
        
        def get_file_size(path):
            try:
                size = os.path.getsize(path)
                if size > 1024 * 1024:
                    return f"{size / (1024 * 1024):.1f} MB"
                elif size > 1024:
                    return f"{size / 1024:.1f} KB"
                return f"{size} B"
            except:
                return "N/A"
        
        prior_size = get_file_size(prior_weights)
        mineclip_size = get_file_size(mineclip_weights)
        vpt_size = get_file_size(vpt_model)
        policy_size = get_file_size(policy_weights)
        
        # 评估参数
        intrinsic = results.get('intrinsic_quality', {}).get('metrics', {})
        consistency_samples = intrinsic.get('consistency', {}).get('n_samples', 10)
        
        # Goal Progress 评估参数
        goal_progress_data = results.get('task_results', {}).get('goal_progress', {})
        n_tasks_with_progress = goal_progress_data.get('n_tasks_with_data', 0)
        
        # 统计变体类别数量（从 task_info 中获取）
        total_variant_categories = sum(info.get('n_variant_categories', 0) for info in task_info.values())
        avg_variant_categories = total_variant_categories // n_tasks if n_tasks > 0 else 0
        total_variants = sum(info.get('n_variants', 0) for info in task_info.values())
        avg_variants = total_variants // n_tasks if n_tasks > 0 else 0
        
        # 按 Tier 和 Category 分类任务
        tier_tasks = {1: [], 2: [], 3: []}
        category_tasks = {'harvest': [], 'combat': [], 'techtree': [], 'other': []}
        
        for task_id in task_ids:
            info = task_info.get(task_id, {})
            tier = info.get('tier', 2)
            category = info.get('category', 'unknown')
            
            if tier in tier_tasks:
                tier_tasks[tier].append(task_id)
            
            if category in ['harvest', 'raw_resources', 'plants', 'animal_drops']:
                category_tasks['harvest'].append(task_id)
            elif category in ['combat']:
                category_tasks['combat'].append(task_id)
            elif category in ['techtree']:
                category_tasks['techtree'].append(task_id)
            else:
                category_tasks['other'].append(task_id)
        
        # Policy 模型配置卡片（仅当有 Goal Progress 数据时显示）
        policy_config_card = ""
        if has_goal_progress:
            policy_config_card = f"""
                <div class="env-card">
                    <div class="env-title">Policy 模型配置</div>
                    <table class="env-table">
                        <tr><td>VPT 基座</td><td><code>{vpt_model}</code></td><td class="env-size">{vpt_size}</td></tr>
                        <tr><td>Policy</td><td><code>{policy_weights}</code></td><td class="env-size">{policy_size}</td></tr>
                        <tr><td>CFG Scale</td><td colspan="2">6.0</td></tr>
                    </table>
                </div>
                
                <div class="env-card">
                    <div class="env-title">Policy 评估参数</div>
                    <table class="env-table">
                        <tr><td>样本目录</td><td colspan="2"><code>data/train_samples</code></td></tr>
                        <tr><td>评估任务数</td><td colspan="2">{n_tasks_with_progress}</td></tr>
                        <tr><td>采样间隔</td><td colspan="2">每 20 帧采样一次（用于目标接近度计算）</td></tr>
                        <tr><td>视频窗口</td><td colspan="2">16 帧滑动窗口（MineCLIP encode_video）</td></tr>
                        <tr><td>环境</td><td colspan="2">MineRL/MineDojo</td></tr>
                    </table>
                </div>
            """
        
        return f"""
        <div class="section">
            <h2>评估环境</h2>
            
            <div class="env-grid">
                <div class="env-card">
                    <div class="env-title">Prior 模型配置</div>
                    <table class="env-table">
                        <tr><td>Prior</td><td><code>{prior_weights}</code></td><td class="env-size">{prior_size}</td></tr>
                        <tr><td>MineCLIP</td><td><code>{mineclip_weights}</code></td><td class="env-size">{mineclip_size}</td></tr>
                        <tr><td>嵌入维度</td><td colspan="2">512</td></tr>
                    </table>
                </div>
                
                <div class="env-card">
                    <div class="env-title">Prior 评估参数</div>
                    <table class="env-table">
                        <tr><td>配置文件</td><td colspan="2"><code>{config_file}</code></td></tr>
                        <tr><td>任务数量</td><td colspan="2">{n_tasks}</td></tr>
                        <tr><td>一致性采样</td><td colspan="2">{consistency_samples} 次/任务（评估 Prior 输出稳定性）</td></tr>
                        <tr><td>变体类别</td><td colspan="2">~{avg_variant_categories} 类/任务 (~{avg_variants} 变体)</td></tr>
                    </table>
                </div>
                
                {policy_config_card}
            </div>
            
            <div class="task-list-section">
                <div class="env-title">任务清单（按 Tier 分组）</div>
                <div class="task-categories">
                    <div class="task-category">
                        <span class="category-label">Tier 1 - 基础 ({len(tier_tasks[1])}):</span>
                        <span class="task-tags">{''.join(f'<span class="task-tag">{t}</span>' for t in tier_tasks[1])}</span>
                    </div>
                    <div class="task-category">
                        <span class="category-label">Tier 2 - 中等 ({len(tier_tasks[2])}):</span>
                        <span class="task-tags">{''.join(f'<span class="task-tag">{t}</span>' for t in tier_tasks[2])}</span>
                    </div>
                    <div class="task-category">
                        <span class="category-label">Tier 3 - 困难 ({len(tier_tasks[3])}):</span>
                        <span class="task-tags">{''.join(f'<span class="task-tag">{t}</span>' for t in tier_tasks[3])}</span>
                    </div>
                </div>
            </div>
        </div>
        """
        
    def _generate_task_results_table(self, results: Dict) -> str:
        """
        生成核心任务结果表格 - 学术简洁风格
        
        评估逻辑：
        已知样本: 指令 X → 目标视频嵌入 Y (Ground Truth)
        
        五列指标：
        1. MineCLIP基线(消融): sim(MineCLIP.encode_text(X), Y)
        2. Prior输出(核心):    sim(Prior(X), Y)
        3. Prior变体(鲁棒性):  avg(sim(Prior(X'), Y)) 其中X'是X的变体
        4. 专家基线(新增):     专家演示帧嵌入到目标的距离变化
        5. 目标接近度(新增):   policy 执行过程中帧嵌入到目标的距离变化
        """
        output_quality = results.get('output_quality', {})
        intrinsic_quality = results.get('intrinsic_quality', {})
        task_info = results.get('task_info', {})
        
        # 获取目标接近度数据
        goal_progress_data = results.get('task_results', {}).get('goal_progress', {}).get('task_progress', {})
        has_goal_progress = bool(goal_progress_data)
        
        prior_gain_data = output_quality.get('metrics', {}).get('prior_gain', {}).get('task_gains', {})
        robustness_data = intrinsic_quality.get('metrics', {}).get('semantic_robustness', {}).get('task_robustness', {})
        
        # 获取执行次数信息
        consistency_data = intrinsic_quality.get('metrics', {}).get('consistency', {})
        n_samples = consistency_data.get('n_samples', 1)  # 一致性采样次数
        
        # 统计变体信息 - 从 task_info 获取更准确的信息
        total_variant_categories = 0
        total_variants = 0
        for info in task_info.values():
            total_variant_categories += info.get('n_variant_categories', 0)
            total_variants += info.get('n_variants', 0)
        n_tasks = len(task_info) if task_info else len(prior_gain_data)
        avg_variant_categories = total_variant_categories // n_tasks if n_tasks > 0 else 0
        avg_variants = total_variants // n_tasks if n_tasks > 0 else 0
        
        # 表头结构
        if has_goal_progress:
            # 有 Policy 指标时的布局
            html = f"""
        <div class="section">
            <h2>任务结果</h2>
            
            <!-- 评估逻辑说明 - Prior -->
            <div class="eval-logic-block">
                <div class="title">Prior 评估逻辑</div>
                <p>已知样本: <code>指令 X → 目标视频嵌入 Y</code>，评估模型输出 Ŷ 与 Y 的余弦相似度 sim(Ŷ, Y)</p>
                <div class="logic-formulas">
                    <div class="formula-item">
                        <div class="label">① 目标嵌入基线</div>
                        <div class="formula">Ŷ = MineCLIP.text(X)</div>
                        <div class="note">MineCLIP 直接编码</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">② Prior 目标嵌入</div>
                        <div class="formula">Ŷ = Prior(X)</div>
                        <div class="note">核心评估</div>
                    </div>
                </div>
            </div>
            
            <!-- 评估逻辑说明 - Policy -->
            <div class="eval-logic-block">
                <div class="title">Policy 评估逻辑</div>
                <p>已知样本: <code>目标嵌入 Y → 动作序列 τ = (a₁, a₂, ..., aₙ)</code>，评估模型输出 τ̂，比较模型接近率 P(τ̂, Y) 与样本接近率 P(τ, Y) 的关系</p>
                <div class="logic-formulas">
                    <div class="formula-item">
                        <div class="label">③ 接近率基线</div>
                        <div class="formula">P<sub>baseline</sub> = (d₀ - d<sub>N</sub>) / d₀</div>
                        <div class="note">专家帧序列 → 目标嵌入 Y 的距离变化率（基线参考）</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">④ Policy 接近率</div>
                        <div class="formula">P<sub>policy</sub> = (d₀ - d<sub>N</sub>) / d₀</div>
                        <div class="note">模型执行帧 → Y 的距离变化率，ΔP = P<sub>policy</sub> - P<sub>baseline</sub></div>
                    </div>
                    <div class="formula-item">
                        <div class="label">⑤ Policy 成功率</div>
                        <div class="formula">SR = n<sub>success</sub> / n<sub>trials</sub></div>
                        <div class="note">任务完成率，基于奖励信号判定</div>
                    </div>
                </div>
            </div>
            
            <table class="task-results-table">
            <thead>
                <!-- 第一行：分组表头 -->
                <tr class="group-header-row">
                    <th rowspan="2" style="width: 5%">层级</th>
                    <th rowspan="2" style="width: 8%; text-align: left;">类别</th>
                    <th rowspan="2" style="width: 14%; text-align: left;">任务 ID</th>
                    <th colspan="2" class="group-header prior-group">Prior 指标</th>
                    <th colspan="3" class="group-header policy-group">Policy 指标</th>
                </tr>
                <!-- 第二行：具体列头 -->
                <tr>
                    <th style="width: 14%">
                        ① 目标嵌入基线
                        <span class="th-note">Prior消融</span>
                    </th>
                    <th style="width: 14%">
                        <strong>② Prior 目标嵌入</strong>
                        <span class="th-note">Prior输出</span>
                    </th>
                    <th style="width: 15%">
                        ③ 接近率基线
                        <span class="th-note">进度 / 单调</span>
                    </th>
                    <th style="width: 15%">
                        <strong>④ Policy 接近率</strong>
                        <span class="th-note">进度 / 单调</span>
                    </th>
                    <th style="width: 12%">
                        <strong>⑤ Policy 成功率</strong>
                        <span class="th-note"></span>
                    </th>
                </tr>
            </thead>
            <tbody>
        """
        else:
            # 无 Policy 指标时的布局
            html = f"""
        <div class="section">
            <h2>任务结果</h2>
            
            <!-- 评估逻辑说明 -->
            <div class="eval-logic-block">
                <div class="title">Prior 评估逻辑</div>
                <p>已知样本: <code>指令 X → 目标视频嵌入 Y</code>，评估模型输出 Ŷ 与 Y 的余弦相似度 sim(Ŷ, Y)</p>
                <div class="logic-formulas">
                    <div class="formula-item">
                        <div class="label">① 目标嵌入基线</div>
                        <div class="formula">Ŷ = MineCLIP.text(X)</div>
                        <div class="note">MineCLIP 直接编码</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">② Prior 目标嵌入</div>
                        <div class="formula">Ŷ = Prior(X)</div>
                        <div class="note">核心评估</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">③ Prior 变体</div>
                        <div class="formula">Ŷ = avg(Prior(X'))</div>
                        <div class="note">~{avg_variant_categories}类变体</div>
                    </div>
                </div>
            </div>
            
            <table class="task-results-table">
            <thead>
                <tr>
                    <th style="width: 4%">Tier</th>
                    <th style="width: 8%">Category</th>
                    <th style="width: 16%">任务 ID</th>
                    <th style="width: 24%">
                        <strong>① 目标嵌入基线</strong>
                        <span class="th-note">MineCLIP 直接编码</span>
                    </th>
                    <th style="width: 24%">
                        <strong>② Prior 目标嵌入</strong>
                        <span class="th-note">核心评估</span>
                    </th>
                    <th style="width: 24%">
                        <strong>③ Prior 变体</strong>
                        <span class="th-note">~{avg_variant_categories}类变体均值</span>
                    </th>
                </tr>
            </thead>
            <tbody>
        """
        
        # 收集所有任务并按 tier, category 排序
        all_tasks = set(prior_gain_data.keys()) | set(robustness_data.keys())
        
        # 类别映射到简短显示名
        category_display = {
            'harvest': 'Harvest', 'raw_resources': 'Harvest', 'plants': 'Harvest', 'animal_drops': 'Harvest',
            'combat': 'Combat', 'techtree': 'Techtree', 'unknown': '-'
        }
        
        # 排序函数: tier 优先, 然后 category
        def sort_key(task_id):
            info = task_info.get(task_id, {})
            tier = info.get('tier', 2)
            category = info.get('category', 'unknown')
            category_order = {'harvest': 0, 'raw_resources': 0, 'plants': 0, 'animal_drops': 0,
                              'combat': 1, 'techtree': 2}
            cat_order = category_order.get(category, 3)
            return (tier, cat_order, task_id)
        
        sorted_tasks = sorted(all_tasks, key=sort_key)
        
        # 预计算每个 (tier, category) 组合的任务数量，用于 rowspan
        tier_category_counts = {}
        for task_id in sorted_tasks:
            info = task_info.get(task_id, {})
            tier = info.get('tier', 2)
            cat = info.get('category', 'unknown')
            cat_display = category_display.get(cat, cat)
            key = (tier, cat_display)
            tier_category_counts[key] = tier_category_counts.get(key, 0) + 1
        
        # 预计算每个 tier 的任务数量
        tier_counts = {}
        for task_id in sorted_tasks:
            info = task_info.get(task_id, {})
            tier = info.get('tier', 2)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        baseline_values = []
        prior_values = []
        variant_values = []
        
        current_tier = None
        current_category = None
        tier_first_row = {}  # {tier: True/False} 是否是该tier的第一行
        category_first_row = {}  # {(tier, category): True/False}
        
        for task_id in sorted_tasks:
            info = task_info.get(task_id, {})
            tier = info.get('tier', 2)
            cat = info.get('category', 'unknown')
            cat_display = category_display.get(cat, cat)
            
            gain_info = prior_gain_data.get(task_id, {})
            baseline_sim = gain_info.get('alignment_text', 0)
            prior_sim = gain_info.get('alignment_prior', 0)
            
            robust_info = robustness_data.get(task_id, {})
            variant_sim = robust_info.get('variant_alignment', prior_sim * robust_info.get('robustness', 1.0))
            
            baseline_values.append(baseline_sim)
            prior_values.append(prior_sim)
            variant_values.append(variant_sim)
            
            # 计算增益
            prior_gain = prior_sim - baseline_sim
            variant_gain = variant_sim - baseline_sim
            
            # Tier 单元格（合并）
            tier_cell = ""
            if tier != current_tier:
                current_tier = tier
                tier_label = {1: "T1", 2: "T2", 3: "T3"}.get(tier, f"T{tier}")
                tier_rowspan = tier_counts.get(tier, 1)
                tier_cell = f'<td class="tier-cell" rowspan="{tier_rowspan}">{tier_label}</td>'
                current_category = None  # 重置 category
            
            # Category 单元格（按 tier+category 合并）
            cat_cell = ""
            tc_key = (tier, cat_display)
            if tc_key not in category_first_row:
                category_first_row[tc_key] = True
                cat_rowspan = tier_category_counts.get(tc_key, 1)
                cat_cell = f'<td class="category-cell" rowspan="{cat_rowspan}">{cat_display}</td>'
            
            # Policy 指标列（接近率基线 + Policy接近率 + Policy成功率）
            expert_baseline_cell = ""
            model_progress_cell = ""
            success_rate_cell = ""
            if has_goal_progress:
                # 尝试匹配任务 ID（支持带/不带 _en 后缀）
                task_progress = goal_progress_data.get(task_id) or goal_progress_data.get(f"{task_id}_en")
                
                # 获取成功率（从 auxiliary_metrics 或 task_progress）
                auxiliary_metrics = results.get('auxiliary_metrics', {})
                task_auxiliary = auxiliary_metrics.get('task_auxiliary', {})
                task_aux = task_auxiliary.get(task_id, {})
                success_rate = task_aux.get('success_rate', 0)
                
                if task_progress:
                    # ③ 接近率基线（不需要颜色）
                    expert_progress = task_progress.get('expert_progress_rate', task_progress.get('progress_rate', 0))
                    expert_monotonic = task_progress.get('expert_monotonic_rate', task_progress.get('monotonic_rate', 0))
                    
                    expert_baseline_cell = f'''
                    <td>
                        <span class="sim-value">{expert_progress:+.1%}</span>
                        <span class="sim-sep">/</span>
                        <span class="sim-value">{expert_monotonic:.1%}</span>
                    </td>'''
                    
                    # ④ Policy 接近率
                    model_progress = task_progress.get('model_progress_rate', 0)
                    model_monotonic = task_progress.get('model_monotonic_rate', 0)
                    
                    model_progress_class = 'delta-positive' if model_progress > 0 else 'delta-negative' if model_progress < 0 else ''
                    
                    # 计算与基线的差异
                    progress_diff = model_progress - expert_progress
                    diff_class = 'delta-positive' if progress_diff > 0.01 else 'delta-negative' if progress_diff < -0.01 else 'delta-neutral'
                    
                    model_progress_cell = f'''
                    <td>
                        <span class="sim-value">{model_progress:+.1%}</span>
                        <span class="sim-sep">/</span>
                        <span class="sim-value">{model_monotonic:.1%}</span>
                        <br><span class="sim-delta {diff_class}">{progress_diff:+.1%}</span>
                    </td>'''
                    
                    # ⑤ Policy 成功率
                    success_rate_cell = f'''
                    <td>
                        <span class="sim-value">{success_rate:.1%}</span>
                    </td>'''
                else:
                    expert_baseline_cell = '<td><span class="sim-note">-</span></td>'
                    model_progress_cell = '<td><span class="sim-note">-</span></td>'
                    success_rate_cell = f'<td><span class="sim-value">{success_rate:.1%}</span></td>'
            
            # MineCLIP 基线显示：如果为 0 则显示 "-"
            baseline_display = f'<span class="sim-value">{baseline_sim:.4f}</span>' if baseline_sim > 0.001 else '<span class="sim-note">-</span>'
            
            html += f"""
                <tr>
                    {tier_cell}
                    {cat_cell}
                    <td style="text-align: left;">{task_id}</td>
                    <td>
                        {baseline_display}
                    </td>
                    <td>
                        <span class="sim-value">{prior_sim:.4f}</span>
                        <span class="sim-delta {self._get_delta_class(prior_gain)}">{prior_gain:+.4f}</span>
                    </td>
                    {expert_baseline_cell}
                    {model_progress_cell}
                    {success_rate_cell}
                </tr>
            """
        
        # 平均行
        avg_baseline = sum(baseline_values) / len(baseline_values) if baseline_values else 0
        avg_prior = sum(prior_values) / len(prior_values) if prior_values else 0
        avg_variant = sum(variant_values) / len(variant_values) if variant_values else 0
        avg_prior_gain = avg_prior - avg_baseline
        avg_variant_gain = avg_variant - avg_baseline
        
        # Policy 指标平均值
        avg_expert_cell = ""
        avg_model_cell = ""
        avg_success_cell = ""
        if has_goal_progress:
            goal_progress_summary = results.get('summary', {}).get('goal_progress_summary', {})
            avg_expert_progress = goal_progress_summary.get('avg_expert_progress_rate', goal_progress_summary.get('avg_progress_rate', 0))
            avg_expert_monotonic = goal_progress_summary.get('avg_expert_monotonic_rate', goal_progress_summary.get('avg_monotonic_rate', 0))
            avg_model_progress = goal_progress_summary.get('avg_model_progress_rate', 0)
            avg_model_monotonic = goal_progress_summary.get('avg_model_monotonic_rate', 0)
            
            # 获取平均成功率
            avg_success_rate = results.get('summary', {}).get('avg_success_rate', 0)
            
            # 接近率基线（不需要颜色）
            avg_expert_cell = f'''
                    <td>
                        <span class="sim-value">{avg_expert_progress:+.1%}</span>
                        <span class="sim-sep">/</span>
                        <span class="sim-value">{avg_expert_monotonic:.1%}</span>
                    </td>'''
            
            # Policy 接近率颜色
            avg_model_class = 'delta-positive' if avg_model_progress > 0 else 'delta-negative' if avg_model_progress < 0 else ''
            progress_diff = avg_model_progress - avg_expert_progress
            diff_class = 'delta-positive' if progress_diff > 0.01 else 'delta-negative' if progress_diff < -0.01 else 'delta-neutral'
            
            avg_model_cell = f'''
                    <td>
                        <span class="sim-value">{avg_model_progress:+.1%}</span>
                        <span class="sim-sep">/</span>
                        <span class="sim-value">{avg_model_monotonic:.1%}</span>
                        <br><span class="sim-delta {diff_class}">{progress_diff:+.1%}</span>
                    </td>'''
            
            avg_success_cell = f'''
                    <td>
                        <span class="sim-value">{avg_success_rate:.1%}</span>
                    </td>'''
        
        # Average 行的目标嵌入基线显示
        avg_baseline_display = f'<span class="sim-value">{avg_baseline:.4f}</span>' if avg_baseline > 0.001 else '<span class="sim-note">-</span>'
        
        html += f"""
                <tr class="avg-row">
                    <td colspan="3"><strong>Average</strong></td>
                    <td>{avg_baseline_display}</td>
                    <td>
                        <span class="sim-value">{avg_prior:.4f}</span>
                        <span class="sim-delta {self._get_delta_class(avg_prior_gain)}">{avg_prior_gain:+.4f}</span>
                    </td>
                    {avg_expert_cell}
                    {avg_model_cell}
                    {avg_success_cell}
                </tr>
            </tbody>
        </table>
        </div>
        """
        
        return html
    
    def _get_delta_class(self, delta: float) -> str:
        """获取增益的CSS类"""
        if delta > 0.01:
            return 'delta-positive'
        elif delta < -0.01:
            return 'delta-negative'
        else:
            return 'delta-neutral'
    
    def _generate_auxiliary_metrics_table(self, results: Dict) -> str:
        """
        生成辅助指标表格
        
        Prior 辅助指标：
        - Prior输出均值方差
        - Prior区分度
        
        Policy 辅助指标：
        - 动作相似度
        - Camera相似度
        - 动作熵
        - 时序平滑度
        - 动作覆盖率
        """
        auxiliary_metrics = results.get('auxiliary_metrics', {})
        task_auxiliary = auxiliary_metrics.get('task_auxiliary', {})
        task_info = results.get('task_info', {})
        summary = results.get('summary', {})
        
        if not task_auxiliary:
            return ""
        
        html = """
        <div class="section">
            <h2>辅助指标</h2>
            
            <div class="eval-logic-block" style="margin-bottom: 15px;">
                <div class="title">辅助指标说明</div>
                <div class="logic-formulas">
                    <div class="formula-item">
                        <div class="label">Prior 变体</div>
                        <div class="formula">V = avg(sim(Prior(x<sub>i</sub>), Y))</div>
                        <div class="note">不同表达方式的指令变体经 Prior 输出与目标嵌入的平均相似度</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">Prior 区分度</div>
                        <div class="formula">D = 1 - avg(sim(z<sub>i</sub>, z<sub>j</sub>))</div>
                        <div class="note">不同任务 Prior 输出之间的区分程度，越高越能区分任务</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">输出方差</div>
                        <div class="formula">σ = std(goal_accuracy<sub>1..n</sub>)</div>
                        <div class="note">Prior 多次采样输出的标准差，越小越稳定</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">动作相似度</div>
                        <div class="formula">S = |A<sub>m</sub> ∩ A<sub>e</sub>| / |A<sub>m</sub> ∪ A<sub>e</sub>|</div>
                        <div class="note">模型预测动作与专家动作的 Jaccard 相似度</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">Camera 相似度</div>
                        <div class="formula">C<sub>cam</sub> = cos(Δθ<sub>m</sub>, Δθ<sub>e</sub>)</div>
                        <div class="note">模型预测视角与专家视角的余弦相似度</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">动作熵</div>
                        <div class="formula">H = -Σ p(a) log p(a)</div>
                        <div class="note">预测动作的多样性，值越高越多样</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">时序平滑度</div>
                        <div class="formula">T = Σ I(a<sub>t</sub>=a<sub>t-1</sub>) / N</div>
                        <div class="note">连续相同动作的比例，值越高越平滑</div>
                    </div>
                    <div class="formula-item">
                        <div class="label">动作覆盖率</div>
                        <div class="formula">D = |{{a<sub>used</sub>}}| / |A<sub>total</sub>|</div>
                        <div class="note">使用的动作类型占总动作类型的比例</div>
                    </div>
                </div>
            </div>
            
            <table class="task-results-table">
            <thead>
                <!-- 第一行：分组表头 -->
                <tr class="group-header-row">
                    <th rowspan="2" style="width: 4%">层级</th>
                    <th rowspan="2" style="width: 12%; text-align: left;">任务 ID</th>
                    <th colspan="3" class="group-header prior-group">Prior 辅助指标</th>
                    <th colspan="5" class="group-header policy-group">Policy 辅助指标</th>
                </tr>
                <!-- 第二行：具体列头 -->
                <tr>
                    <th style="width: 10%">
                        Prior 变体
                        <span class="th-note">variant_alignment</span>
                    </th>
                    <th style="width: 10%">
                        Prior 区分度
                        <span class="th-note">discriminability</span>
                    </th>
                    <th style="width: 10%">
                        输出方差
                        <span class="th-note">goal_accuracy_std</span>
                    </th>
                    <th style="width: 10%">
                        动作相似度
                        <span class="th-note">Jaccard</span>
                    </th>
                    <th style="width: 10%">
                        Camera相似度
                        <span class="th-note">cos_sim</span>
                    </th>
                    <th style="width: 10%">
                        动作熵
                        <span class="th-note">entropy</span>
                    </th>
                    <th style="width: 10%">
                        时序平滑度
                        <span class="th-note">smoothness</span>
                    </th>
                    <th style="width: 10%">
                        动作覆盖率
                        <span class="th-note">coverage</span>
                    </th>
                </tr>
            </thead>
            <tbody>
        """
        
        # 按 tier 排序任务
        sorted_tasks = sorted(task_auxiliary.items(), 
                             key=lambda x: task_info.get(x[0], {}).get('tier', 1))
        
        current_tier = None
        tier_counts = {}
        for task_id, _ in sorted_tasks:
            tier = task_info.get(task_id, {}).get('tier', 1)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        for task_id, metrics in sorted_tasks:
            info = task_info.get(task_id, {})
            tier = info.get('tier', 1)
            
            # Tier 单元格（合并）
            tier_cell = ""
            if tier != current_tier:
                current_tier = tier
                tier_label = {1: "T1", 2: "T2", 3: "T3"}.get(tier, f"T{tier}")
                tier_rowspan = tier_counts.get(tier, 1)
                tier_cell = f'<td class="tier-cell" rowspan="{tier_rowspan}">{tier_label}</td>'
            
            # 提取指标
            prior_variant_alignment = metrics.get('prior_variant_alignment', 0)
            prior_discriminability = metrics.get('prior_discriminability', 0)
            prior_goal_accuracy_std = metrics.get('prior_goal_accuracy_std', 0)
            action_similarity = metrics.get('action_similarity', 0)
            camera_similarity = metrics.get('camera_similarity', 0)
            action_entropy = metrics.get('action_entropy', 0)
            temporal_smoothness = metrics.get('temporal_smoothness', 0)
            action_coverage = metrics.get('action_coverage', 0)
            
            html += f"""
                <tr>
                    {tier_cell}
                    <td class="task-cell" style="text-align: left;">{task_id}</td>
                    <td><span class="sim-value">{prior_variant_alignment:.4f}</span></td>
                    <td><span class="sim-value">{prior_discriminability:.4f}</span></td>
                    <td><span class="sim-value">{prior_goal_accuracy_std:.4f}</span></td>
                    <td><span class="sim-value">{action_similarity:.1%}</span></td>
                    <td><span class="sim-value">{camera_similarity:.1%}</span></td>
                    <td><span class="sim-value">{action_entropy:.2f}</span></td>
                    <td><span class="sim-value">{temporal_smoothness:.1%}</span></td>
                    <td><span class="sim-value">{action_coverage:.1%}</span></td>
                </tr>
            """
        
        # 平均行
        avg_prior_variant_alignment = auxiliary_metrics.get('avg_prior_variant_alignment', 0)
        avg_prior_discriminability = auxiliary_metrics.get('avg_prior_discriminability', 0)
        prior_mean_variance = auxiliary_metrics.get('prior_mean_variance', 0)
        avg_action_similarity = auxiliary_metrics.get('avg_action_similarity', 0)
        avg_camera_similarity = auxiliary_metrics.get('avg_camera_similarity', 0)
        avg_action_entropy = auxiliary_metrics.get('avg_action_entropy', 0)
        avg_temporal_smoothness = auxiliary_metrics.get('avg_temporal_smoothness', 0)
        avg_action_coverage = auxiliary_metrics.get('avg_action_coverage', 0)
        
        html += f"""
                <tr class="avg-row">
                    <td colspan="2"><strong>Average</strong></td>
                    <td><span class="sim-value">{avg_prior_variant_alignment:.4f}</span></td>
                    <td><span class="sim-value">{avg_prior_discriminability:.4f}</span></td>
                    <td><span class="sim-value">{prior_mean_variance:.4f}</span></td>
                    <td><span class="sim-value">{avg_action_similarity:.1%}</span></td>
                    <td><span class="sim-value">{avg_camera_similarity:.1%}</span></td>
                    <td><span class="sim-value">{avg_action_entropy:.2f}</span></td>
                    <td><span class="sim-value">{avg_temporal_smoothness:.1%}</span></td>
                    <td><span class="sim-value">{avg_action_coverage:.1%}</span></td>
                </tr>
            </tbody>
        </table>
        </div>
        """
        
        return html
    
    def _generate_visualization_section(self, results: Dict) -> str:
        """
        生成可视化部分（分 Prior 和 Policy 两部分）- 学术风格
        
        Prior 部分:
        图1: MineCLIP vs Prior 空间对比 → 对应第一列（消融对比）
        图2: 变体输出 vs 目标视频 → 对应第二列（鲁棒性）
        图3: Prior 输出 vs 目标视频 → 对应第三列（核心评估）
        辅助图1: Prior 输出相似度矩阵
        辅助图2: Prior 输出方差分布
        
        Policy 部分:
        图5: 目标接近度概览
        """
        has_goal_progress = bool(results.get('task_results', {}).get('goal_progress', {}).get('task_progress', {}))
        
        html = """
        <div class="section">
            <h2>Prior 可视化分析</h2>
        """
        
        # 定义可视化图片及其说明
        viz_configs = [
            {
                'filename': 'viz_1a_mineclip_vs_prior_tier.png',
                'title': '①-a MineCLIP vs Prior (按 Tier 着色)',
                'column': '消融对比 - Tier 视图',
                'definition': '''
                    <strong>定义</strong>：将 MineCLIP 直接编码文本的输出 (○) 和 Prior 转换后的输出 (▲) 
                    投影到同一 t-SNE 空间，按任务 Tier 着色（T1绿/T2橙/T3红）。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 虚线连接同一任务的 MineCLIP 和 Prior 输出
                    • 观察不同 Tier 任务的转换偏移是否一致
                    • 若 ▲ 系统性偏移 → Prior 学到了文本到视觉的转换
                ''',
                'full_width': False
            },
            {
                'filename': 'viz_1b_mineclip_vs_prior_category.png',
                'title': '①-b MineCLIP vs Prior (按 Category 着色)',
                'column': '消融对比 - Category 视图',
                'definition': '''
                    <strong>定义</strong>：将 MineCLIP 直接编码文本的输出 (○) 和 Prior 转换后的输出 (▲) 
                    投影到同一 t-SNE 空间，按任务类别着色（Harvest绿/Combat红/Techtree蓝）。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 同类别任务应该聚集
                    • 观察不同类别任务的转换方向是否一致
                    • 若类别间分离明显 → Prior 能区分任务类型
                ''',
                'full_width': False
            },
            {
                'filename': 'viz_2a_variants_by_category.png',
                'title': '②-a 变体输出 vs 目标视频 (按变体类别)',
                'column': '鲁棒性评估 - 类别视图',
                'definition': '''
                    <strong>定义</strong>：将同一任务的多个指令变体通过 Prior 的输出投影到 t-SNE 空间。
                    按 7 类变体着色：简单直接(绿)、位置描述(蓝)、目的说明(橙)、动作细节(紫)、
                    口语化(青)、复杂描述(红)、激励性(棕)。★ 为目标视频嵌入。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 同色变体应聚集 → 该类表达方式的语义一致性好
                    • 不同类若明显分离 → Prior 对表达风格敏感
                    • 所有 ▲ 应接近 ★ → 变体整体能对齐目标
                ''',
                'full_width': False
            },
            {
                'filename': 'viz_2b_variants_by_task.png',
                'title': '②-b 变体输出 vs 目标视频 (按任务)',
                'column': '鲁棒性评估 - 任务视图',
                'definition': '''
                    <strong>定义</strong>：将同一任务的多个指令变体通过 Prior 的输出投影到 t-SNE 空间。
                    按任务着色，✓ 表示该任务有结构化变体分类。★ 为目标视频嵌入。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 同任务变体应聚集 → 任务内指令鲁棒性好
                    • 各任务的 ▲ 应接近各自的 ★ → 目标对齐
                    • 任务间分离 → Prior 能区分不同任务的变体
                ''',
                'full_width': False
            },
            {
                'filename': 'viz_3a_prior_vs_visual_tier.png',
                'title': '③-a Prior vs 目标视频 (按 Tier 着色)',
                'column': '核心评估 - Tier 视图',
                'definition': '''
                    <strong>定义</strong>：将 Prior 对主指令的输出 (▲) 和对应任务的目标视频嵌入 (○)
                    投影到同一 t-SNE 空间，按 Tier 着色（T1绿/T2橙/T3红）。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 每个 ▲ 应接近同色的 ○ → Prior 输出接近目标
                    • 不同 Tier 的 ▲ 应该分离 → 难度梯度反映在嵌入中
                    • ▲ 和 ○ 距离越近 → 目标对齐度越高
                ''',
                'full_width': False
            },
            {
                'filename': 'viz_3b_prior_vs_visual_category.png',
                'title': '③-b Prior vs 目标视频 (按 Category 着色)',
                'column': '核心评估 - Category 视图',
                'definition': '''
                    <strong>定义</strong>：将 Prior 对主指令的输出 (▲) 和对应任务的目标视频嵌入 (○)
                    投影到同一 t-SNE 空间，按类别着色（Harvest绿/Combat红/Techtree蓝）。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 同类别的 ▲ 和 ○ 应该聚集
                    • 不同类别应该分离 → Prior 能区分任务类型
                    • 类别内 ▲-○ 距离反映对齐质量
                ''',
                'full_width': False
            },
            {
                'filename': 'task_similarity_matrix.png',
                'title': '辅助图1: Prior 输出相似度矩阵',
                'column': '辅助指标',
                'definition': '''
                    <strong>定义</strong>：计算 Prior 对所有任务输出的两两余弦相似度，形成 N×N 矩阵。
                    对角线为 1.0（自身相似度）。同 Tier/Category 任务按块分组。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 对角线外的值应该较低 (< 0.9) → 任务间有区分度
                    • 若非对角值过高 → 模型可能存在模式坍塌，对不同任务输出相似
                    • 同类任务（如都是采集）相似度略高是正常的
                ''',
                'full_width': False
            },
            {
                'filename': 'variance_distribution.png',
                'title': '辅助图2: Prior 输出方差分布',
                'column': '辅助指标',
                'definition': '''
                    <strong>定义</strong>：计算 Prior 输出在 512 个维度上的方差分布。
                    左图为直方图，右图为按方差排序的曲线。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 方差分布应该平滑 → 维度被均匀利用
                    • 若大量维度方差接近 0 → 维度利用率低
                    • 方差过低 (< 1e-4) 的维度过多 → 可能存在后验坍塌
                ''',
                'full_width': False
            },
        ]
        
        # Policy 可视化配置（单独处理）
        policy_viz_configs = [
            {
                'filename': 'viz_5_goal_progress_overview.png',
                'title': '目标接近度概览',
                'column': '目标接近度',
                'definition': '''
                    <strong>定义</strong>：评估专家演示过程中帧嵌入到目标的距离变化。
                    左图：各任务进度率（正数=接近目标）。
                    右图：进度率 vs 单调率散点图。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 进度率 > 0 → 最终比初始更接近目标
                    • 单调率高 → 每步都在稳定靠近目标
                    • 理想情况：右上角区域（高进度+高单调）
                ''',
                'full_width': True
            },
            {
                'filename': 'viz_6_action_distribution.png',
                'title': '动作分布对比',
                'column': '动作分析',
                'definition': '''
                    <strong>定义</strong>：对比专家演示和模型预测的动作类型分布。
                    蓝色=专家，橙色=模型。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 分布相近 → 模型学习到专家的动作偏好
                    • 差异大 → 模型行为模式与专家不同
                    • 关注高频动作的匹配程度
                ''',
                'full_width': False
            },
            {
                'filename': 'viz_7_confusion_matrix.png',
                'title': '动作混淆矩阵',
                'column': '动作分析',
                'definition': '''
                    <strong>定义</strong>：专家动作（行）vs 模型预测（列）的匹配关系。
                    按行归一化，显示专家各动作被模型预测为各类别的概率。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 对角线值高 → 模型正确预测该动作
                    • 非对角线值 → 模型的混淆模式
                    • 可发现模型系统性错误（如总是预测forward）
                ''',
                'full_width': False
            }
        ]
        
        # 添加汇总的逐帧相似度时间线图（所有任务）
        aggregated_timeline = self.output_dir / "viz_8_similarity_timeline_aggregated.png"
        if aggregated_timeline.exists():
            policy_viz_configs.append({
                'filename': 'viz_8_similarity_timeline_aggregated.png',
                'title': '逐帧相似度时间线（所有任务汇总）',
                'column': '时间线分析',
                'definition': '''
                    <strong>定义</strong>：显示所有任务所有 trial 的平均相似度趋势。
                    X轴为归一化进度（0-100%），上图：Action相似度；下图：Camera相似度。
                    虚线为移动平均，红线为整体平均。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 高峰值 → 该时段模型与专家行为一致
                    • 低谷 → 模型决策与专家不同
                    • 整体趋势反映 Policy 学习效果
                ''',
                'full_width': True
            })
        
        # 添加汇总的目标接近度对比图（所有任务）
        aggregated_comparison = self.output_dir / "viz_9_goal_comparison_aggregated.png"
        if aggregated_comparison.exists():
            policy_viz_configs.append({
                'filename': 'viz_9_goal_comparison_aggregated.png',
                'title': '目标接近度（所有任务汇总）',
                'column': '接近度对比',
                'definition': '''
                    <strong>定义</strong>：显示所有任务的平均目标距离变化。
                    蓝色为专家基线，绿色为模型输出。阴影区域为标准差。
                    X轴为归一化进度，Y轴为余弦距离（越低越好）。
                ''',
                'interpretation': '''
                    <strong>解读</strong>：
                    • 整体下降 → 正在接近目标
                    • 阴影区域大 → 任务间差异大
                    • 模型曲线低于专家 → Policy 超越专家基线
                ''',
                'full_width': True
            })
        
        # 渲染 Prior 可视化
        html += '<div class="viz-grid">'
        
        for viz in viz_configs:
            viz_path = self.output_dir / viz['filename']
            if viz_path.exists():
                img_base64 = self._image_to_base64(viz_path)
                if img_base64:
                    full_width_class = 'full-width' if viz.get('full_width') else ''
                    max_width_style = f"max-width: {viz['max_width']};" if viz.get('max_width') else ''
                    img_style = f"style='{max_width_style}'" if max_width_style else ''
                    html += f"""
                    <div class="viz-card {full_width_class}">
                        <h3>{viz['title']}</h3>
                        <div class="viz-column-label">{viz['column']}</div>
                        <img src="data:image/png;base64,{img_base64}" alt="{viz['title']}" {img_style}>
                        <div class="viz-explanation">
                            <div class="viz-definition">{viz['definition']}</div>
                            <div class="viz-interpretation">{viz['interpretation']}</div>
                        </div>
                    </div>
                    """
        
        html += '</div>'  # 关闭 viz-grid
        
        # 如果没有找到任何 Prior 可视化图片
        found_any = any((self.output_dir / viz['filename']).exists() for viz in viz_configs)
        if not found_any:
            html += """
            <div class="explanation" style="padding: 20px; background: #f9f9f9; border-radius: 8px; text-align: center;">
                <strong>暂无可视化图片</strong>
                <p style="color: #666; margin-top: 8px;">可视化功能需要运行完整评估流程（多任务集）生成 t-SNE 图。</p>
            </div>
            <p class="viz-desc" style="color: #888; font-size: 0.9em; margin-top: 10px;">
                • task_similarity_matrix.png - 相似度矩阵<br>
                • variance_distribution.png - 方差分布
            </p>
            """
        
        html += '</div>'  # 关闭 Prior 可视化 section
        
        # 渲染 Policy 可视化（仅当有 goal progress 数据时）
        if has_goal_progress:
            html += """
        <div class="section">
            <h2>Policy 可视化分析</h2>
            <div class="viz-grid">
            """
            
            found_policy_viz = False
            for viz in policy_viz_configs:
                viz_path = self.output_dir / viz['filename']
                if viz_path.exists():
                    found_policy_viz = True
                    img_base64 = self._image_to_base64(viz_path)
                    if img_base64:
                        full_width_class = 'full-width' if viz.get('full_width') else ''
                        html += f"""
                        <div class="viz-card {full_width_class}">
                            <h3>{viz['title']}</h3>
                            <div class="viz-column-label">{viz['column']}</div>
                            <img src="data:image/png;base64,{img_base64}" alt="{viz['title']}">
                            <div class="viz-explanation">
                                <div class="viz-definition">{viz['definition']}</div>
                                <div class="viz-interpretation">{viz['interpretation']}</div>
                            </div>
                        </div>
                        """
            
            html += '</div>'  # 关闭 viz-grid
            
            if not found_policy_viz:
                html += """
            <div class="explanation" style="padding: 20px; background: #f9f9f9; border-radius: 8px; text-align: center;">
                <strong>暂无可视化图片</strong>
                <p style="color: #666; margin-top: 8px;">可视化功能需要运行完整评估流程（多任务集）生成目标接近度图。</p>
            </div>
            <p class="viz-desc" style="color: #888; font-size: 0.9em; margin-top: 10px;">
                • goal_progress_overview.png - 目标接近度概览
            </p>
                """
            
            html += '</div>'  # 关闭 Policy 可视化 section
        
        return html
    
    def _generate_insights_section(self, results: Dict) -> str:
        """生成洞察分析部分"""
        summary = results.get('summary', {})
        output_quality = summary.get('output_quality_summary', {})
        intrinsic_quality = summary.get('intrinsic_quality_summary', {})
        
        avg_alignment = output_quality.get('avg_goal_alignment', 0)
        avg_robustness = intrinsic_quality.get('avg_semantic_robustness', 0)
        avg_gain = output_quality.get('avg_prior_gain', 0)
        preservation_rate = intrinsic_quality.get('preservation_rate', 1.0)
        mean_variance = intrinsic_quality.get('mean_variance', 0)
        
        # 分析并生成建议
        issues = []
        recommendations = []
        
        if avg_alignment < 0.5:
            issues.append("❌ 目标对齐度较低，Prior生成的嵌入与成功视频相似度不足")
            recommendations.append("考虑使用更多的text-visual配对数据重新训练Prior")
        elif avg_alignment < 0.7:
            issues.append("⚠️ 目标对齐度一般，仍有提升空间")
            recommendations.append("尝试调整Prior的训练策略或增加训练数据多样性")
        else:
            issues.append("✅ 目标对齐度良好")
        
        if avg_robustness < 0.8:
            issues.append("❌ 指令鲁棒性不足，对指令变体敏感")
            recommendations.append("在训练数据中加入更多指令变体以提升鲁棒性")
        elif avg_robustness < 0.9:
            issues.append("⚠️ 指令鲁棒性一般")
        else:
            issues.append("✅ 指令鲁棒性良好")
        
        if avg_gain < 0:
            issues.append("❌ Prior增益为负，不如直接使用MineCLIP")
            recommendations.append("检查Prior训练是否有问题，或考虑直接使用MineCLIP文本编码")
        elif avg_gain < 0.05:
            issues.append("⚠️ Prior增益较小，转换效果不明显")
            recommendations.append("尝试调整Prior架构或训练目标")
        else:
            issues.append("✅ Prior增益为正，转换有效")
        
        if mean_variance < 1e-4:
            issues.append("❌ 输出方差过低，可能存在模式坍塌")
            recommendations.append("检查VAE的KL散度权重是否过大，考虑降低beta值")
        
        if preservation_rate < 0.5:
            issues.append("⚠️ 区分度保持率低，Prior可能压缩了任务差异")
        
        html = """
        <div class="section">
            <h2>💡 分析洞察</h2>
        """
        
        # 问题列表
        html += """
            <div class="explanation">
                <strong>评估发现</strong>
                <ul style="margin-top: 10px; padding-left: 20px;">
        """
        for issue in issues:
            html += f"<li>{issue}</li>"
        html += """
                </ul>
        </div>
        """
        
        # 建议列表
        if recommendations:
            html += """
            <div class="explanation" style="background: #fff8e1; border-left-color: #ff9100;">
                <strong>优化建议</strong>
                <ul style="margin-top: 10px; padding-left: 20px;">
            """
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += """
                </ul>
            </div>
            """
        
        # Prior vs MineCLIP 详细对比
        prior_gain_data = results.get('output_quality', {}).get('metrics', {}).get('prior_gain', {}).get('task_gains', {})
        
        if prior_gain_data:
            html += """
            <h3 style="margin-top: 25px;">Prior vs MineCLIP 详细对比</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>任务</th>
                        <th>MineCLIP Text</th>
                        <th>Prior Output</th>
                        <th>增益</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for task_id, data in prior_gain_data.items():
                text_align = data.get('alignment_text', 0)
                prior_align = data.get('alignment_prior', 0)
                gain = data.get('gain', 0)
                gain_class = 'gain-positive' if gain > 0.01 else ('gain-negative' if gain < -0.01 else 'gain-neutral')
                html += f"""
                    <tr>
                        <td style="text-align: left; font-weight: 600;">{task_id}</td>
                        <td>{text_align:.4f}</td>
                        <td>{prior_align:.4f}</td>
                        <td class="{gain_class}">{gain:+.4f}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            
            <div class="viz-desc">
                此表对比了 MineCLIP 直接编码文本与 Prior 转换后嵌入对成功视频的对齐程度。
                正增益表示 Prior 转换有效提升了目标对齐度。
            </div>
            """
        
        html += '</div>'
        return html
    
    def _image_to_base64(self, image_path: Path) -> Optional[str]:
        """将图片转换为 base64 编码"""
        try:
            if not image_path.exists():
                return None
            with open(image_path, 'rb') as f:
                img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            print(f"无法读取图片 {image_path}: {e}")
            return None


# 向后兼容别名
PriorHTMLGenerator = EvaluationReportGenerator
