"""
Prior评估HTML报告生成器
支持三个维度的完整评估结果展示，包含可视化图片和指标解读
"""

import json
import base64
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class PriorHTMLGenerator:
    """
    Prior评估HTML报告生成器 V2
    
    支持:
    - 内在质量维度
    - 输出质量维度
    - 可控性维度
    - 任务级详细结果
    - 可视化图表
    """
    
    def __init__(self, output_dir: str):
        """
        初始化生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        results: Dict,
        output_filename: str = "prior_evaluation_report.html"
    ) -> Path:
        """
        生成HTML报告
        
        Args:
            results: 评估结果字典
            output_filename: 输出文件名
            
        Returns:
            HTML文件路径
        """
        html_content = self._generate_html(results)
        
        output_path = self.output_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html(self, results: Dict) -> str:
        """生成完整的HTML内容"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prior 模型评估报告</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header(results)}
        {self._generate_summary(results)}
        {self._generate_metric_explanations()}
        {self._generate_visualization_section(results)}
        {self._generate_dimension_results(results)}
        {self._generate_task_details(results)}
        {self._generate_footer()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>"""
    
    def _get_css(self) -> str:
        """返回CSS样式"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .meta {
            opacity: 0.9;
            font-size: 0.95em;
        }
        
        .summary-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .summary-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .metric-card h3 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .metric-card .interpretation {
            font-size: 0.85em;
            color: #666;
        }
        
        .metric-card.excellent {
            border-left-color: #10b981;
        }
        
        .metric-card.good {
            border-left-color: #3b82f6;
        }
        
        .metric-card.poor {
            border-left-color: #ef4444;
        }
        
        .dimension-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .dimension-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            display: flex;
            align-items: center;
        }
        
        .dimension-section h2 .badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.5em;
            margin-left: 15px;
        }
        
        .dimension-section h3 {
            color: #764ba2;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .task-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .task-table th,
        .task-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .task-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }
        
        .task-table tr:hover {
            background: #f8f9fa;
        }
        
        .score {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .score.excellent {
            background: #d1fae5;
            color: #065f46;
        }
        
        .score.good {
            background: #dbeafe;
            color: #1e40af;
        }
        
        .score.poor {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        
        .collapsible:after {
            content: '\\25BC';
            float: right;
            margin-left: 10px;
            font-size: 0.8em;
        }
        
        .collapsible.collapsed:after {
            content: '\\25B6';
        }
        
        .collapsible-content {
            max-height: none;
            overflow: visible;
            transition: max-height 0.3s ease-out;
        }
        
        .collapsible-content.collapsed {
            max-height: 0;
            overflow: hidden;
        }
        
        .visualization-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .visualization-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .visualization-section img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        
        .metric-explanation {
            background: #f0f7ff;
            border-left: 4px solid #3b82f6;
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .metric-explanation h4 {
            color: #1e40af;
            margin-bottom: 8px;
            font-size: 1em;
        }
        
        .metric-explanation p {
            color: #1e3a8a;
            line-height: 1.6;
            margin: 5px 0;
        }
        
        .interpretation-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .interpretation-badge.excellent {
            background: #10b981;
            color: white;
        }
        
        .interpretation-badge.good {
            background: #3b82f6;
            color: white;
        }
        
        .interpretation-badge.warning {
            background: #f59e0b;
            color: white;
        }
        
        .interpretation-badge.poor {
            background: #ef4444;
            color: white;
        }
        """
    
    def _get_javascript(self) -> str:
        """返回JavaScript代码"""
        return """
        function toggleSection(id) {
            const content = document.getElementById(id);
            const header = content.previousElementSibling;
            
            content.classList.toggle('collapsed');
            header.classList.toggle('collapsed');
        }
        
        // 初始化：所有section默认展开
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Prior评估报告已加载');
        });
        """
    
    def _generate_header(self, results: Dict) -> str:
        """生成页眉"""
        n_tasks = results.get('n_tasks', 0)
        config_file = results.get('config_file', 'N/A')
        
        return f"""
        <div class="header">
            <h1>🎯 Prior 模型评估报告</h1>
            <div class="meta">
                <div>评估任务: {n_tasks} 个</div>
                <div>配置文件: {config_file}</div>
                <div>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </div>
        """
    
    def _image_to_base64(self, image_path: Path) -> Optional[str]:
        """将图片转换为base64编码"""
        try:
            if not image_path.exists():
                return None
            with open(image_path, 'rb') as f:
                img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            print(f"无法读取图片 {image_path}: {e}")
            return None
    
    def _generate_metric_explanations(self) -> str:
        """生成指标解释说明部分"""
        return """
        <div class="visualization-section">
            <h2>📖 指标说明与解读</h2>
            
            <div class="metric-explanation">
                <h4>🎯 维度1：内在质量 (Intrinsic Quality)</h4>
                <p><strong>1.1 输出稳定性 (Consistency)</strong>: 同一指令多次采样的相似度。越高越好（&gt;0.95优秀），表示Prior输出稳定可靠。</p>
                <p><strong>1.2 语义鲁棒性 (Semantic Robustness)</strong>: 同一任务不同表述的相似度。越高越好（&gt;0.90优秀），表示Prior对指令变化不敏感。</p>
                <p><strong>1.3 输出多样性 (Output Diversity)</strong>: 不同任务输出的方差。适中最好，太低表示所有任务输出过于相似，太高表示输出不稳定。</p>
                <p><strong>1.4 区分度保持率 (Discriminability Preservation)</strong>: Prior输出相对于文本输入的区分度变化。&gt;1.0表示Prior放大了任务差异（好），&lt;1.0表示Prior压缩了差异（可能有问题）。</p>
            </div>
            
            <div class="metric-explanation">
                <h4>🎯 维度2：输出质量 (Output Quality)</h4>
                <p><strong>2.1 目标对齐度 (Goal Alignment)</strong>: Prior输出与真实成功画面的MineCLIP相似度。越高越好（&gt;0.60优秀），表示Prior指向正确目标。</p>
                <p><strong>2.2 Prior增益 (Prior Gain)</strong>: Prior相对于直接使用文本嵌入的改进。正值表示Prior有提升，负值表示Prior反而降低了对齐度（需要调查）。</p>
                <p><strong>2.3 跨模态一致性 (Cross-Modal Consistency)</strong>: Prior输出是否真的在视觉空间。通过比较Prior输出和真实视觉嵌入的分布（Wasserstein距离）。越高越好，表示Prior输出接近真实视觉嵌入。</p>
            </div>
            
            <div class="metric-explanation">
                <h4>🎯 维度3：可控性 (Controllability)</h4>
                <p><strong>3.1 CFG敏感度</strong>: Classifier-Free Guidance在Policy层面的影响。Prior本身不支持CFG，这是Policy模型的参数。</p>
                <p><strong>注意</strong>: CFG是Policy级别的概念，Prior评估中此维度被禁用。</p>
            </div>
            
            <div class="metric-explanation">
                <h4>💡 如何解读结果</h4>
                <p><strong>优秀的Prior模型应该具备</strong>:</p>
                <p>✅ 高稳定性（Consistency &gt; 0.95）- 输出可靠</p>
                <p>✅ 高鲁棒性（Semantic Robustness &gt; 0.90）- 理解语义而非记忆文本</p>
                <p>✅ 高目标对齐度（Goal Alignment &gt; 0.60）- 指向正确目标</p>
                <p>✅ 正增益（Prior Gain &gt; 0）- 比直接文本更好</p>
                <p>✅ 适度多样性 - 不同任务有所区分</p>
                <p>✅ 保持或放大区分度 - 没有压缩任务差异</p>
            </div>
        </div>
        """
    
    def _generate_visualization_section(self, results: Dict) -> str:
        """生成可视化图片部分"""
        # 检查是否有可视化文件
        viz_path = self.output_dir / "prior_evaluation_visualization.png"
        
        if not viz_path.exists():
            return ""
        
        # 将图片转换为base64
        img_base64 = self._image_to_base64(viz_path)
        if not img_base64:
            return ""
        
        n_tasks = results.get('n_tasks', 0)
        
        html = f"""
        <div class="visualization-section">
            <h2>📊 Prior 嵌入空间可视化</h2>
            <p><strong>说明</strong>: 以下可视化展示了 Prior 输出在高维空间中的分布特征（基于 {n_tasks} 个任务）</p>
            
            <img src="data:image/png;base64,{img_base64}" alt="Prior Evaluation Visualization">
            
            <div class="metric-explanation">
                <h4>📈 如何解读可视化</h4>
                <p><strong>左上 - 相似度矩阵</strong>: 展示所有任务两两之间的Prior输出相似度。对角线应该是深色（自己与自己相似度为1），远离对角线应该有浅色区域（不同任务相似度低）。</p>
                <p><strong>右上 - t-SNE 降维</strong>: 将512维Prior输出降到2维。相似任务应该聚在一起形成簇，不同类型任务应该分离。</p>
                <p><strong>左下 - PCA 降维</strong>: 另一种降维方法，显示主成分方差解释率。如果前两个PC解释率很低，说明数据在高维空间分散。</p>
                <p><strong>右下 - 方差分布</strong>: 每个维度的方差分布。理想情况下应该有适度的方差（不是全0也不是极端值）。</p>
            </div>
        </div>
        """
        
        return html
    
    def _generate_summary(self, results: Dict) -> str:
        """生成总结部分"""
        summary = results.get('summary', {})
        
        html = """
        <div class="summary-section">
            <h2>📊 评估总结</h2>
            <div class="metric-grid">
        """
        
        # 内在质量总结
        if 'intrinsic_quality_summary' in summary:
            iq = summary['intrinsic_quality_summary']
            
            if iq.get('avg_consistency') is not None:
                value = iq['avg_consistency']
                grade = self._get_grade(value, 0.95, 0.85)
                html += self._metric_card(
                    "平均一致性",
                    f"{value:.4f}",
                    "同一指令多次采样的稳定性",
                    grade
                )
            
            if iq.get('avg_semantic_robustness') is not None:
                value = iq['avg_semantic_robustness']
                grade = self._get_grade(value, 0.90, 0.70)
                html += self._metric_card(
                    "平均语义鲁棒性",
                    f"{value:.4f}",
                    "不同表述的一致性",
                    grade
                )
            
            if iq.get('mean_variance') is not None:
                value = iq['mean_variance']
                html += self._metric_card(
                    "输出多样性",
                    f"{value:.6f}",
                    "不同任务输出的方差",
                    "good" if value > 0.0001 else "poor"
                )
        
        # 输出质量总结
        if 'output_quality_summary' in summary:
            oq = summary['output_quality_summary']
            
            if oq.get('avg_goal_alignment') is not None:
                value = oq['avg_goal_alignment']
                grade = self._get_grade(value, 0.60, 0.40)
                html += self._metric_card(
                    "平均目标对齐度",
                    f"{value:.4f}",
                    "Prior输出与成功画面的相似度",
                    grade
                )
            
            if oq.get('avg_prior_gain') is not None:
                value = oq['avg_prior_gain']
                sign = "+" if value >= 0 else ""
                grade = "excellent" if value > 0.05 else ("good" if value > 0 else "poor")
                html += self._metric_card(
                    "平均Prior增益",
                    f"{sign}{value:.4f}",
                    "相对于直接文本的改进",
                    grade
                )
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _metric_card(self, title: str, value: str, description: str, grade: str) -> str:
        """生成指标卡片"""
        return f"""
        <div class="metric-card {grade}">
            <h3>{title}</h3>
            <div class="value">{value}</div>
            <div class="interpretation">{description}</div>
        </div>
        """
    
    def _get_grade(self, value: float, excellent_threshold: float, good_threshold: float) -> str:
        """根据阈值判定等级"""
        if value >= excellent_threshold:
            return "excellent"
        elif value >= good_threshold:
            return "good"
        else:
            return "poor"
    
    def _generate_dimension_results(self, results: Dict) -> str:
        """生成各维度详细结果"""
        html = ""
        
        # 维度1: 内在质量
        if results.get('intrinsic_quality') and results['intrinsic_quality'].get('enabled'):
            html += self._generate_intrinsic_quality_section(results['intrinsic_quality'])
        
        # 维度2: 输出质量
        if results.get('output_quality') and results['output_quality'].get('enabled'):
            html += self._generate_output_quality_section(results['output_quality'])
        
        # 维度3: 可控性
        if results.get('controllability') and results['controllability'].get('enabled'):
            html += self._generate_controllability_section(results['controllability'])
        
        return html
    
    def _generate_intrinsic_quality_section(self, dimension: Dict) -> str:
        """生成内在质量部分"""
        metrics = dimension.get('metrics', {})
        
        html = f"""
        <div class="dimension-section">
            <h2 onclick="toggleSection('intrinsic-content')" class="collapsible">
                📐 维度1: 内在质量 (Intrinsic Quality)
                <span class="badge">4 个指标</span>
            </h2>
            <div id="intrinsic-content" class="collapsible-content">
        """
        
        # 一致性
        if 'consistency' in metrics:
            html += self._render_consistency(metrics['consistency'])
        
        # 语义鲁棒性
        if 'semantic_robustness' in metrics:
            html += self._render_semantic_robustness(metrics['semantic_robustness'])
        
        # 输出多样性
        if 'output_diversity' in metrics:
            html += self._render_output_diversity(metrics['output_diversity'])
        
        # 区分度保持率
        if 'discriminability_preservation' in metrics:
            html += self._render_discriminability(metrics['discriminability_preservation'])
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _render_consistency(self, data: Dict) -> str:
        """渲染一致性结果"""
        task_consistencies = data.get('task_consistencies', {})
        avg_consistency = data.get('avg_consistency', 0)
        interpretation = data.get('interpretation', '')
        
        # 判断等级
        if avg_consistency >= 0.95:
            badge = '<span class="interpretation-badge excellent">优秀</span>'
        elif avg_consistency >= 0.85:
            badge = '<span class="interpretation-badge good">良好</span>'
        else:
            badge = '<span class="interpretation-badge warning">需改进</span>'
        
        html = f"""
        <h3>📊 指标1.1: 输出稳定性 (Consistency) {badge}</h3>
        <p><strong>平均值:</strong> {avg_consistency:.4f} - {interpretation}</p>
        <p><strong>说明:</strong> 同一指令多次采样的一致性。高稳定性表示Prior输出可靠，不会因随机性产生大幅波动。</p>
        <p><strong>解读:</strong> 该指标通过对同一指令采样多次（默认10次），计算输出嵌入之间的余弦相似度。相似度越高，说明模型输出越稳定。</p>
        <table class="task-table">
            <thead>
                <tr>
                    <th>任务ID</th>
                    <th>一致性</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for task_id, value in task_consistencies.items():
            grade = self._get_grade(value, 0.95, 0.85)
            html += f"""
                <tr>
                    <td>{task_id}</td>
                    <td><span class="score {grade}">{value:.4f}</span></td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _render_semantic_robustness(self, data: Dict) -> str:
        """渲染语义鲁棒性结果"""
        task_robustness = data.get('task_robustness', {})
        avg_robustness = data.get('avg_robustness', 0)
        interpretation = data.get('interpretation', '')
        
        if not task_robustness:
            return """
            <h3>📊 指标1.2: 语义鲁棒性 (Semantic Robustness)</h3>
            <p><em>无可用数据</em></p>
            """
        
        # 判断等级
        if avg_robustness >= 0.90:
            badge = '<span class="interpretation-badge excellent">优秀</span>'
        elif avg_robustness >= 0.70:
            badge = '<span class="interpretation-badge good">良好</span>'
        else:
            badge = '<span class="interpretation-badge warning">需改进</span>'
        
        html = f"""
        <h3>📊 指标1.2: 语义鲁棒性 (Semantic Robustness) {badge}</h3>
        <p><strong>平均值:</strong> {avg_robustness:.4f} - {interpretation}</p>
        <p><strong>说明:</strong> 同一任务不同表述的一致性。高鲁棒性表示Prior理解任务的语义，而不是记忆特定文本。</p>
        <p><strong>解读:</strong> 该指标通过对比同一任务的不同指令变体（如"chop tree"和"cut down tree"），测试Prior对语义的理解能力。相似度高说明模型关注语义而非具体用词。</p>
        <table class="task-table">
            <thead>
                <tr>
                    <th>任务ID</th>
                    <th>鲁棒性</th>
                    <th>变体数</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for task_id, info in task_robustness.items():
            value = info['robustness']
            n_variants = info['n_variants']
            grade = self._get_grade(value, 0.90, 0.70)
            html += f"""
                <tr>
                    <td>{task_id}</td>
                    <td><span class="score {grade}">{value:.4f}</span></td>
                    <td>{n_variants}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _render_output_diversity(self, data: Dict) -> str:
        """渲染输出多样性结果"""
        mean_variance = data.get('mean_variance', 0)
        interpretation = data.get('interpretation', '')
        
        # 判断等级
        if mean_variance > 0.0001:
            badge = '<span class="interpretation-badge good">适中</span>'
        else:
            badge = '<span class="interpretation-badge warning">偏低</span>'
        
        return f"""
        <h3>📊 指标1.3: 输出多样性 (Output Diversity) {badge}</h3>
        <p><strong>均值方差:</strong> {mean_variance:.6f} - {interpretation}</p>
        <p><strong>说明:</strong> 不同任务Prior输出的方差，反映输出的多样性。</p>
        <p><strong>解读:</strong> 该指标计算所有任务Prior输出在512维空间中每个维度的方差均值。方差太低表示所有任务输出过于相似（可能退化），适度方差表示正常区分。注意：单任务评估此指标无意义。</p>
        """
    
    def _render_discriminability(self, data: Dict) -> str:
        """渲染区分度保持率结果"""
        text_disc = data.get('text_discriminability', 0)
        prior_disc = data.get('prior_discriminability', 0)
        preservation_rate = data.get('preservation_rate', 0)
        interpretation = data.get('interpretation', '')
        
        # 判断等级
        if preservation_rate >= 1.0:
            badge = '<span class="interpretation-badge excellent">保持/放大</span>'
        elif preservation_rate >= 0.5:
            badge = '<span class="interpretation-badge good">轻微压缩</span>'
        else:
            badge = '<span class="interpretation-badge warning">严重压缩</span>'
        
        return f"""
        <h3>📊 指标1.4: 区分度保持率 (Discriminability Preservation) {badge}</h3>
        <p><strong>文本区分度:</strong> {text_disc:.4f} （MineCLIP编码的文本之间的区分度）</p>
        <p><strong>Prior区分度:</strong> {prior_disc:.4f} （Prior输出之间的区分度）</p>
        <p><strong>保持率:</strong> {preservation_rate:.2f}x - {interpretation}</p>
        <p><strong>说明:</strong> Prior输出相对于输入文本的区分度变化。&gt;1.0表示Prior放大了任务差异，&lt;1.0表示压缩了差异。</p>
        <p><strong>解读:</strong> 区分度 = 1 - 平均任务间相似度。如果Prior区分度 &lt; 文本区分度，说明Prior把本来不同的任务变得相似了（潜在问题）。注意：单任务评估此指标无意义。</p>
        """
    
    def _generate_output_quality_section(self, dimension: Dict) -> str:
        """生成输出质量部分"""
        metrics = dimension.get('metrics', {})
        
        html = f"""
        <div class="dimension-section">
            <h2 onclick="toggleSection('output-quality-content')" class="collapsible">
                🎯 维度2: 输出质量 (Output Quality)
                <span class="badge">3 个指标</span>
            </h2>
            <div id="output-quality-content" class="collapsible-content">
        """
        
        # 目标对齐度
        if 'goal_alignment' in metrics:
            html += self._render_goal_alignment(metrics['goal_alignment'])
        
        # Prior增益
        if 'prior_gain' in metrics:
            html += self._render_prior_gain(metrics['prior_gain'])
        
        # 跨模态一致性
        if 'cross_modal_consistency' in metrics:
            html += self._render_cross_modal(metrics['cross_modal_consistency'])
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _render_goal_alignment(self, data: Dict) -> str:
        """渲染目标对齐度结果"""
        task_alignments = data.get('task_alignments', {})
        avg_alignment = data.get('avg_alignment', 0)
        interpretation = data.get('interpretation', '')
        use_reward_head = data.get('use_reward_head', False)
        
        method = "forward_reward_head (MineCLIP奖励函数)" if use_reward_head else "cosine similarity"
        
        # 判断等级
        if avg_alignment >= 0.60:
            badge = '<span class="interpretation-badge excellent">优秀</span>'
        elif avg_alignment >= 0.40:
            badge = '<span class="interpretation-badge good">良好</span>'
        else:
            badge = '<span class="interpretation-badge poor">需改进</span>'
        
        html = f"""
        <h3>🎯 指标2.1: 目标对齐度 (Goal Alignment) {badge}</h3>
        <p><strong>平均值:</strong> {avg_alignment:.4f} - {interpretation}</p>
        <p><strong>计算方法:</strong> {method}</p>
        <p><strong>说明:</strong> Prior输出与真实成功画面的相似度。高对齐度表示Prior指向正确的目标。</p>
        <p><strong>解读:</strong> 该指标使用MineCLIP的reward_head计算Prior输出嵌入与任务成功时的真实游戏画面嵌入的相似度。相似度高说明Prior确实在引导模型朝正确目标前进。注意：需要提供success_visuals_path。</p>
        <table class="task-table">
            <thead>
                <tr>
                    <th>任务ID</th>
                    <th>对齐度 (均值)</th>
                    <th>标准差</th>
                    <th>成功画面数</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for task_id, info in task_alignments.items():
            mean = info['mean']
            std = info.get('std', 0)
            n_visuals = info.get('n_visuals', 0)
            grade = self._get_grade(mean, 0.60, 0.40)
            html += f"""
                <tr>
                    <td>{task_id}</td>
                    <td><span class="score {grade}">{mean:.4f}</span></td>
                    <td>{std:.4f}</td>
                    <td>{n_visuals}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _render_prior_gain(self, data: Dict) -> str:
        """渲染Prior增益结果"""
        task_gains = data.get('task_gains', {})
        avg_gain = data.get('avg_gain', 0)
        interpretation = data.get('interpretation', '')
        
        sign = "+" if avg_gain >= 0 else ""
        
        # 判断等级
        if avg_gain > 0.05:
            badge = '<span class="interpretation-badge excellent">显著提升</span>'
        elif avg_gain > 0:
            badge = '<span class="interpretation-badge good">轻微提升</span>'
        else:
            badge = '<span class="interpretation-badge poor">负增益</span>'
        
        html = f"""
        <h3>📈 指标2.2: Prior增益 (Prior Gain) {badge}</h3>
        <p><strong>平均增益:</strong> {sign}{avg_gain:.4f} - {interpretation}</p>
        <p><strong>说明:</strong> Prior相对于直接使用文本嵌入的改进。正值表示Prior有价值，负值需要调查原因。</p>
        <p><strong>解读:</strong> 该指标对比两种方案的目标对齐度：(1) Prior(文本) → 视觉嵌入，(2) MineCLIP(文本) → 文本嵌入直接用作视觉嵌入。如果Prior增益为负，说明Prior反而降低了对齐度，可能需要重新训练。</p>
        <table class="task-table">
            <thead>
                <tr>
                    <th>任务ID</th>
                    <th>Prior对齐度</th>
                    <th>Text对齐度</th>
                    <th>增益</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for task_id, info in task_gains.items():
            alignment_prior = info['alignment_prior']
            alignment_text = info['alignment_text']
            gain = info['gain']
            grade = "excellent" if gain > 0.05 else ("good" if gain > 0 else "poor")
            sign = "+" if gain >= 0 else ""
            html += f"""
                <tr>
                    <td>{task_id}</td>
                    <td>{alignment_prior:.4f}</td>
                    <td>{alignment_text:.4f}</td>
                    <td><span class="score {grade}">{sign}{gain:.4f}</span></td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def _render_cross_modal(self, data: Dict) -> str:
        """渲染跨模态一致性结果"""
        consistency_score = data.get('consistency_score', 0)
        mean_distance = data.get('mean_wasserstein_distance', 0)
        interpretation = data.get('interpretation', '')
        
        # 判断等级（一致性得分越高越好，距离越小越好）
        if consistency_score >= 0.70:
            badge = '<span class="interpretation-badge excellent">高度一致</span>'
        elif consistency_score >= 0.50:
            badge = '<span class="interpretation-badge good">基本一致</span>'
        else:
            badge = '<span class="interpretation-badge warning">分布偏离</span>'
        
        return f"""
        <h3>🔀 指标2.3: 跨模态一致性 (Cross-Modal Consistency) {badge}</h3>
        <p><strong>一致性得分:</strong> {consistency_score:.4f} - {interpretation}</p>
        <p><strong>平均Wasserstein距离:</strong> {mean_distance:.4f} （越小越好）</p>
        <p><strong>说明:</strong> Prior输出是否真的在视觉空间。高一致性表示Prior输出接近真实视觉嵌入的分布。</p>
        <p><strong>解读:</strong> 该指标使用Wasserstein距离比较Prior输出的分布和真实视觉嵌入的分布。如果Prior只是生成任意512维向量而不是"视觉化"的嵌入，这个距离会很大。理想的Prior应该生成接近真实视觉空间的嵌入。</p>
        """
    
    def _generate_controllability_section(self, dimension: Dict) -> str:
        """生成可控性部分"""
        metrics = dimension.get('metrics', {})
        
        html = f"""
        <div class="dimension-section">
            <h2 onclick="toggleSection('controllability-content')" class="collapsible">
                🎮 维度3: 可控性 (Controllability)
                <span class="badge">CFG分析</span>
            </h2>
            <div id="controllability-content" class="collapsible-content">
        """
        
        # CFG敏感度
        if 'cfg_sensitivity' in metrics:
            html += self._render_cfg_sensitivity(metrics['cfg_sensitivity'])
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _render_cfg_sensitivity(self, data: Dict) -> str:
        """渲染CFG敏感度结果"""
        cfg_scales = data.get('cfg_scales', [])
        task_cfg_analysis = data.get('task_cfg_analysis', {})
        
        html = f"""
        <h3>🔧 指标3.2: CFG敏感度 (CFG Sensitivity)</h3>
        <p><strong>测试的CFG scales:</strong> {', '.join(map(str, cfg_scales))}</p>
        <p><strong>说明:</strong> 不同CFG scale对Prior输出的影响</p>
        """
        
        for task_id, analysis in task_cfg_analysis.items():
            baseline_diffs = analysis.get('baseline_diffs', {})
            
            if baseline_diffs:
                html += f"""
                <h4>任务: {task_id}</h4>
                <table class="task-table">
                    <thead>
                        <tr>
                            <th>对比</th>
                            <th>相似度</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for key, value in baseline_diffs.items():
                    html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value:.4f}</td>
                        </tr>
                    """
                
                html += """
                    </tbody>
                </table>
                """
        
        return html
    
    def _generate_task_details(self, results: Dict) -> str:
        """生成任务级详细结果"""
        task_results = results.get('task_results', {})
        
        if not task_results:
            return ""
        
        html = """
        <div class="dimension-section">
            <h2>📋 任务详细结果</h2>
        """
        
        for task_id, task_data in task_results.items():
            html += f"""
            <h3>任务: {task_id}</h3>
            <pre>{json.dumps(task_data, indent=2, ensure_ascii=False)}</pre>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_footer(self) -> str:
        """生成页脚"""
        return f"""
        <div class="footer">
            <p>Prior 模型评估报告 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>STEVE-1 Prior Evaluation Framework V2</p>
        </div>
        """


def generate_html_report(results_dict: Dict, output_dir: str) -> Path:
    """
    便捷函数：生成HTML报告
    
    Args:
        results_dict: 评估结果字典
        output_dir: 输出目录
        
    Returns:
        HTML文件路径
    """
    generator = PriorHTMLGenerator(output_dir)
    return generator.generate_report(results_dict)
