"""
统一的 HTML 报告样式
Unified HTML Report Styles

参考 CraftJarvis SkillDiscovery 网站风格
https://craftjarvis.github.io/SkillDiscovery/
"""

# 统一的 CSS 样式（学术风格）
UNIFIED_CSS = """
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans", "Helvetica Neue", Arial, sans-serif;
        line-height: 1.7;
        color: #333;
        background: #fafafa;
        font-size: 16px;
    }
    
    .container {
        max-width: 1100px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 60px 20px 40px;
        background: white;
        border-bottom: 1px solid #eee;
        margin-bottom: 40px;
    }
    
    .header h1 {
        font-size: 2.2em;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 15px;
        letter-spacing: -0.5px;
    }
    
    .header .subtitle {
        font-size: 1.1em;
        color: #666;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .header .meta {
        margin-top: 20px;
        font-size: 0.9em;
        color: #888;
    }
    
    .header .meta span {
        margin: 0 10px;
    }
    
    /* Section */
    .section {
        background: white;
        border-radius: 8px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .section h2 {
        font-size: 1.4em;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid #333;
        display: inline-block;
    }
    
    .section h3 {
        font-size: 1.1em;
        font-weight: 600;
        color: #444;
        margin: 25px 0 15px;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .stat-card {
        background: #f8f9fc;
        border: 1px solid #e8eaf0;
        border-radius: 8px;
        padding: 24px 20px;
        text-align: center;
    }
    
    .stat-card .label {
        font-size: 0.85em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .stat-card .value {
        font-size: 2em;
        font-weight: 700;
        color: #3d5afe;
    }
    
    .stat-card .value.success { color: #00c853; }
    .stat-card .value.warning { color: #ff9100; }
    .stat-card .value.error { color: #ff5252; }
    
    /* Table */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 0.95em;
    }
    
    .data-table th {
        background: #f8f9fc;
        color: #444;
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
        border-bottom: 2px solid #e8eaf0;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .data-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #f0f0f0;
        color: #555;
    }
    
    .data-table tr:hover {
        background: #fafbfd;
    }
    
    .data-table td:first-child {
        font-weight: 500;
        color: #333;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    
    .badge.excellent {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .badge.good {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    .badge.warning {
        background: #fff8e1;
        color: #f57c00;
    }
    
    .badge.poor {
        background: #ffebee;
        color: #c62828;
    }
    
    /* Metric Card */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    
    .metric-card {
        background: #f8f9fc;
        border: 1px solid #e8eaf0;
        border-radius: 8px;
        padding: 20px;
        border-left: 4px solid #3d5afe;
    }
    
    .metric-card h4 {
        font-size: 0.95em;
        font-weight: 600;
        color: #444;
        margin-bottom: 10px;
    }
    
    .metric-card .score {
        font-size: 1.8em;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .metric-card .score.excellent { color: #00c853; }
    .metric-card .score.good { color: #3d5afe; }
    .metric-card .score.warning { color: #ff9100; }
    .metric-card .score.poor { color: #ff5252; }
    
    .metric-card .desc {
        font-size: 0.85em;
        color: #666;
        line-height: 1.5;
    }
    
    /* Progress Bar */
    .progress-bar {
        height: 6px;
        background: #e8eaf0;
        border-radius: 3px;
        margin-top: 12px;
        overflow: hidden;
    }
    
    .progress-bar .fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    .progress-bar .fill.excellent { background: #00c853; }
    .progress-bar .fill.good { background: #3d5afe; }
    .progress-bar .fill.warning { background: #ff9100; }
    .progress-bar .fill.poor { background: #ff5252; }
    
    /* Sub-Metrics 子指标样式 */
    .sub-metrics {
        margin-top: 15px;
        border-top: 1px dashed #e0e0e0;
        padding-top: 10px;
    }
    
    .sub-metrics-toggle {
        color: #3d5afe;
        font-size: 0.85em;
        cursor: pointer;
        user-select: none;
        padding: 5px 0;
        transition: color 0.2s;
    }
    
    .sub-metrics-toggle:hover {
        color: #1e40af;
    }
    
    .sub-metrics-content {
        display: none;
        margin-top: 10px;
        padding: 10px;
        background: #fafbfc;
        border-radius: 6px;
    }
    
    .sub-metrics.expanded .sub-metrics-toggle {
        color: #1e40af;
    }
    
    .sub-metrics.expanded .sub-metrics-toggle::before {
        content: '▼ ';
    }
    
    .sub-metrics:not(.expanded) .sub-metrics-toggle::before {
        content: '▶ ';
    }
    
    .sub-metrics.expanded .sub-metrics-content {
        display: block;
    }
    
    .sub-metrics-table {
        width: 100%;
        font-size: 0.82em;
        border-collapse: collapse;
    }
    
    .sub-metrics-table th,
    .sub-metrics-table td {
        padding: 6px 10px;
        text-align: left;
        border-bottom: 1px solid #eee;
    }
    
    .sub-metrics-table th {
        font-weight: 600;
        color: #555;
        background: #f5f7fa;
    }
    
    .sub-metrics-table td:nth-child(2) {
        font-weight: 600;
        color: #3d5afe;
    }
    
    .sub-metrics-table td:nth-child(3) {
        color: #888;
        font-size: 0.9em;
    }
    
    .metric-note {
        margin-top: 10px;
        font-size: 0.8em;
        color: #888;
        font-style: italic;
        padding: 8px;
        background: #fff8e1;
        border-radius: 4px;
        border-left: 3px solid #ffb300;
    }
    
    .no-data {
        color: #999;
        font-size: 0.85em;
        font-style: italic;
        padding: 10px;
    }
    
    /* Policy 内在质量分析样式 */
    .intrinsic-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }
    
    .intrinsic-metric-card {
        background: #f8f9fc;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3d5afe;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .intrinsic-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .intrinsic-metric-card .metric-name {
        font-size: 0.9em;
        font-weight: 600;
        color: #444;
        margin-bottom: 8px;
    }
    
    .intrinsic-metric-card .metric-value {
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .intrinsic-metric-card .metric-value.excellent { color: #00c853; }
    .intrinsic-metric-card .metric-value.good { color: #3d5afe; }
    .intrinsic-metric-card .metric-value.warning { color: #ff9100; }
    .intrinsic-metric-card .metric-value.poor { color: #ff5252; }
    
    .intrinsic-metric-card .metric-bar {
        height: 6px;
        background: #e8eaf0;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    
    .intrinsic-metric-card .metric-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    .intrinsic-metric-card .metric-fill.excellent { background: #00c853; }
    .intrinsic-metric-card .metric-fill.good { background: #3d5afe; }
    .intrinsic-metric-card .metric-fill.warning { background: #ff9100; }
    .intrinsic-metric-card .metric-fill.poor { background: #ff5252; }
    
    .intrinsic-metric-card .metric-desc {
        font-size: 0.75em;
        color: #888;
        line-height: 1.4;
    }
    
    .section-desc {
        color: #666;
        font-size: 0.95em;
        margin-bottom: 20px;
    }
    
    /* Chart Container */
    .chart-container {
        max-width: 450px;
        margin: 20px auto;
        padding: 20px;
        background: #f8f9fc;
        border-radius: 8px;
    }
    
    /* Visualization */
    .visualization img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    /* Recommendations */
    .recommendation-list {
        list-style: none;
    }
    
    .recommendation-item {
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 6px;
        border-left: 4px solid #3d5afe;
        background: #f8f9fc;
        font-size: 0.95em;
        color: #555;
    }
    
    .recommendation-item.success {
        border-left-color: #00c853;
        background: #f1f8f4;
    }
    
    .recommendation-item.warning {
        border-left-color: #ff9100;
        background: #fffaf0;
    }
    
    .recommendation-item.error {
        border-left-color: #ff5252;
        background: #fef5f5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 20px;
        color: #888;
        font-size: 0.85em;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    
    /* Explanation Box */
    .explanation {
        background: #f0f4ff;
        border: 1px solid #d0dcff;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 15px 0;
        font-size: 0.9em;
        color: #444;
    }
    
    .explanation strong {
        color: #3d5afe;
    }
    
    /* Collapsible */
    .collapsible {
        cursor: pointer;
        user-select: none;
    }
    
    .collapsible:after {
        content: '▼';
        float: right;
        font-size: 0.7em;
        color: #888;
    }
    
    .collapsible.collapsed:after {
        content: '▶';
    }
    
    .collapsible-content {
        max-height: none;
        overflow: visible;
        transition: max-height 0.3s ease;
    }
    
    .collapsible-content.collapsed {
        max-height: 0;
        overflow: hidden;
    }
    
    /* Ablation Experiment Section */
    .ablation-section {
        background: linear-gradient(135deg, #f8f9fc 0%, #fff 100%);
    }
    
    .ablation-intro {
        text-align: center;
        margin-bottom: 25px;
        color: #555;
    }
    
    .ablation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 25px 0;
    }
    
    .ablation-card {
        background: #fff;
        border: 1px solid #e8eaf0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .ablation-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    .ablation-mode {
        font-size: 1.2em;
        font-weight: 700;
        color: #333;
        margin-bottom: 5px;
    }
    
    .ablation-desc {
        font-size: 0.85em;
        color: #888;
        margin-bottom: 15px;
    }
    
    .ablation-bar-container {
        height: 10px;
        background: #e8eaf0;
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    
    .ablation-bar {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    .ablation-bar.normal {
        background: linear-gradient(90deg, #3d5afe, #667eea);
    }
    
    .ablation-bar.fixed {
        background: linear-gradient(90deg, #00c853, #69f0ae);
    }
    
    .ablation-bar.random {
        background: linear-gradient(90deg, #ff9100, #ffab40);
    }
    
    .ablation-bar.zero {
        background: linear-gradient(90deg, #ff5252, #ff8a80);
    }
    
    .ablation-rate {
        font-size: 1em;
        font-weight: 600;
        color: #555;
    }
    
    .ablation-conclusion {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 25px;
        margin-top: 25px;
    }
    
    .ablation-conclusion h4 {
        font-size: 1.1em;
        color: #3d5afe;
        margin-bottom: 15px;
    }
    
    .ablation-conclusion ul {
        list-style: none;
        padding: 0;
    }
    
    .ablation-conclusion li {
        padding: 8px 0;
        border-bottom: 1px solid rgba(61, 90, 254, 0.1);
        color: #444;
    }
    
    .ablation-conclusion li:last-child {
        border-bottom: none;
    }
    
    .ablation-note {
        background: #fff8e1;
        border-left: 4px solid #ff9100;
        border-radius: 0 8px 8px 0;
        padding: 15px;
        margin-top: 20px;
        font-size: 0.9em;
        color: #795548;
    }
"""


def get_html_head(title: str, include_chartjs: bool = False) -> str:
    """
    获取统一的 HTML head 部分
    
    Args:
        title: 页面标题
        include_chartjs: 是否包含 Chart.js
        
    Returns:
        HTML head 字符串
    """
    chartjs_script = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>' if include_chartjs else ''
    
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {chartjs_script}
    <style>
        {UNIFIED_CSS}
    </style>
</head>
"""


def get_badge_class(value: float) -> str:
    """根据值获取徽章类名"""
    if value >= 0.8:
        return "excellent"
    elif value >= 0.5:
        return "good"
    elif value >= 0.3:
        return "warning"
    else:
        return "poor"


def get_level_text(level: str) -> str:
    """获取等级文本"""
    level_map = {
        "excellent": "优秀",
        "good": "良好",
        "warning": "一般",
        "poor": "需改进"
    }
    return level_map.get(level, "")

