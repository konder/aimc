"""
Steve1 两阶段模型可视化工具
=========================================

可视化：
1. Prior模型的嵌入空间（t-SNE/UMAP）
2. 策略模型的动作分布
3. 端到端性能分析

作者: AI Assistant  
日期: 2025-11-27
"""

import numpy as np

# 在导入 matplotlib 之前设置后端
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端，避免 GUI 相关问题

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

logger = logging.getLogger(__name__)


class Steve1PolicyVisualizer:
    """
    Steve1 策略可视化器（Policy Visualizer）
    
    用于可视化策略评估结果（Prior 嵌入空间、Policy 行为模式等）
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        初始化可视化器
        
        Args:
            style: matplotlib 样式
        """
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        try:
            plt.style.use(style)
        except:
            logger.warning(f"样式 '{style}' 不可用，使用默认样式")
        
        # 设置配色方案
        self.colors = sns.color_palette("husl", 10)
        
        logger.info("Steve1Visualizer 初始化完成")
    
    # ==========================================================================
    # Prior 模型可视化
    # ==========================================================================
    
    def visualize_prior_embedding_space(
        self,
        text_embeds: np.ndarray,
        prior_embeds: np.ndarray,
        instructions: List[str],
        output_path: Path,
        method: str = 'tsne',
        perplexity: int = 30,
    ):
        """
        可视化 Prior 嵌入空间
        
        使用 t-SNE 或 UMAP 将 512 维嵌入降维到 2D 并可视化
        
        Args:
            text_embeds: MineCLIP 文本嵌入 [N, 512]
            prior_embeds: Prior 输出嵌入 [N, 512]
            instructions: 指令列表 [N]
            output_path: 输出路径
            method: 降维方法 ('tsne' 或 'umap' 或 'pca')
            perplexity: t-SNE 的 perplexity 参数
        """
        logger.info(f"可视化 Prior 嵌入空间（方法: {method}）")
        
        # 合并文本和Prior嵌入
        all_embeds = np.vstack([text_embeds, prior_embeds])
        n_samples = all_embeds.shape[0]
        
        # 1. 降维
        if method == 'tsne':
            # 自动调整 perplexity：必须小于样本数
            # 推荐值：5 到 50 之间，且 < n_samples
            auto_perplexity = min(perplexity, n_samples - 1, 50)
            auto_perplexity = max(auto_perplexity, 2)  # 至少为 2
            
            if auto_perplexity != perplexity:
                logger.info(f"  自动调整 perplexity: {perplexity} → {auto_perplexity} (样本数: {n_samples})")
            
            reducer = TSNE(n_components=2, perplexity=auto_perplexity, random_state=42)
        elif method == 'umap':
            # UMAP 也有类似的约束：n_neighbors < n_samples
            n_neighbors = min(15, n_samples - 1)
            n_neighbors = max(n_neighbors, 2)
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        elif method == 'pca':
            # PCA 没有此类限制
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"未知的降维方法: {method}")
        
        embeds_2d = reducer.fit_transform(all_embeds)
        
        # 分离并确保是numpy数组
        text_2d = np.asarray(embeds_2d[:len(text_embeds)])
        prior_2d = np.asarray(embeds_2d[len(text_embeds):])
        
        # 确保是float类型
        text_2d = text_2d.astype(np.float64)
        prior_2d = prior_2d.astype(np.float64)
        
        # 2. 绘图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 2.1 文本嵌入空间
        # 转换坐标为列表，避免 numpy 类型问题
        x1 = [float(x) for x in text_2d[:, 0]]
        y1 = [float(y) for y in text_2d[:, 1]]
        
        scatter1 = ax1.scatter(
            x1, y1,
            c=list(range(len(text_embeds))),
            cmap='tab20',
            s=100,
            alpha=0.7,
            edgecolors='k',  # 使用 'k' 而不是 'black'
            linewidths=1
        )
        ax1.set_title('MineCLIP Text Embeddings', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{method.upper()} Dimension 1')
        ax1.set_ylabel(f'{method.upper()} Dimension 2')
        
        # 添加标签（只标注部分，避免过于拥挤）
        for i, txt in enumerate(instructions):
            if i % max(1, len(instructions) // 10) == 0:  # 每10个标注1个
                ax1.annotate(
                    txt[:20] + '...' if len(txt) > 20 else txt,
                    (float(text_2d[i, 0]), float(text_2d[i, 1])),
                    fontsize=8,
                    alpha=0.7
                )
        
        # 2.2 Prior 嵌入空间
        x2 = [float(x) for x in prior_2d[:, 0]]
        y2 = [float(y) for y in prior_2d[:, 1]]
        
        scatter2 = ax2.scatter(
            x2, y2,
            c=list(range(len(prior_embeds))),
            cmap='tab20',
            s=100,
            alpha=0.7,
            edgecolors='k',
            linewidths=1
        )
        ax2.set_title('Prior Output Embeddings', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'{method.upper()} Dimension 1')
        ax2.set_ylabel(f'{method.upper()} Dimension 2')
        
        # 2.3 对比图：文本 vs Prior
        ax3.scatter(
            x1, y1,
            c='blue',
            s=100,
            alpha=0.5,
            label='Text Embed',
            edgecolors='k',
            linewidths=1
        )
        ax3.scatter(
            x2, y2,
            c='red',
            s=100,
            alpha=0.5,
            label='Prior Embed',
            edgecolors='k',
            linewidths=1,
            marker='^'
        )
        
        # 绘制箭头：文本 → Prior
        for i in range(len(text_2d)):
            # 确保箭头参数是标准float
            x_start = float(text_2d[i, 0])
            y_start = float(text_2d[i, 1])
            dx = float(prior_2d[i, 0] - text_2d[i, 0])
            dy = float(prior_2d[i, 1] - text_2d[i, 1])
            
            ax3.arrow(
                x_start, y_start, dx, dy,
                head_width=0.5,
                head_length=0.3,
                fc='gray',
                ec='gray',
                alpha=0.3,
                linewidth=0.5
            )
        
        ax3.set_title('Text → Prior Transformation', fontsize=14, fontweight='bold')
        ax3.set_xlabel(f'{method.upper()} Dimension 1')
        ax3.set_ylabel(f'{method.upper()} Dimension 2')
        ax3.legend()
        
        try:
            plt.tight_layout()
            # 先绘制到 canvas
            fig.canvas.draw()
            # 保存（使用更安全的参数）
            fig.savefig(str(output_path), dpi=150, format='png', bbox_inches='tight')
            logger.info(f"  ✓ 嵌入空间可视化已保存: {output_path}")
        except Exception as e:
            logger.error(f"  ✗ 保存可视化失败: {e}")
            logger.info(f"  跳过此可视化，继续执行...")
        finally:
            plt.close(fig)
            plt.close('all')
    
    def visualize_prior_similarity_matrix(
        self,
        prior_embeds: np.ndarray,
        instructions: List[str],
        output_path: Path,
    ):
        """
        可视化 Prior 嵌入的相似度矩阵
        
        Args:
            prior_embeds: Prior 输出嵌入 [N, 512]
            instructions: 指令列表 [N]
            output_path: 输出路径
        """
        logger.info("可视化 Prior 相似度矩阵")
        
        # 计算余弦相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(prior_embeds)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            sim_matrix,
            xticklabels=[inst[:30] for inst in instructions],
            yticklabels=[inst[:30] for inst in instructions],
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Prior Embedding Similarity Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ 相似度矩阵已保存: {output_path}")
    
    def visualize_prior_quality_metrics(
        self,
        prior_analysis_results: Dict,
        output_path: Path,
    ):
        """
        可视化 Prior 质量指标
        
        Args:
            prior_analysis_results: Prior 分析结果字典
            output_path: 输出路径
        """
        logger.info("可视化 Prior 质量指标")
        
        # 提取指标
        instructions = list(prior_analysis_results.keys())
        text_to_prior_sims = [r.text_to_prior_similarity for r in prior_analysis_results.values()]
        prior_variances = [r.prior_variance for r in prior_analysis_results.values()]
        reconstruction_qualities = [r.reconstruction_quality for r in prior_analysis_results.values()]
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 文本-Prior 相似度分布
        ax1 = axes[0, 0]
        ax1.hist(text_to_prior_sims, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(text_to_prior_sims), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(text_to_prior_sims):.3f}')
        ax1.set_xlabel('Text-Prior Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Text-Prior Similarity Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prior 方差分布
        ax2 = axes[0, 1]
        ax2.hist(prior_variances, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(prior_variances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(prior_variances):.3f}')
        ax2.set_xlabel('Prior Variance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prior Output Variance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 重建质量分布
        ax3 = axes[1, 0]
        ax3.hist(reconstruction_qualities, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(reconstruction_qualities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(reconstruction_qualities):.3f}')
        ax3.set_xlabel('Reconstruction Quality')
        ax3.set_ylabel('Frequency')
        ax3.set_title('VAE Reconstruction Quality Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Top/Bottom 指令
        ax4 = axes[1, 1]
        sorted_indices = np.argsort(text_to_prior_sims)
        top_k = 5
        bottom_indices = sorted_indices[:top_k]
        top_indices = sorted_indices[-top_k:]
        
        y_pos = np.arange(top_k * 2)
        values = [text_to_prior_sims[i] for i in bottom_indices] + [text_to_prior_sims[i] for i in top_indices]
        colors_bar = ['red'] * top_k + ['green'] * top_k
        labels = [instructions[i][:25] + '...' if len(instructions[i]) > 25 else instructions[i] for i in bottom_indices] + \
                 [instructions[i][:25] + '...' if len(instructions[i]) > 25 else instructions[i] for i in top_indices]
        
        ax4.barh(y_pos, values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.set_xlabel('Text-Prior Similarity')
        ax4.set_title(f'Top/Bottom {top_k} Instructions')
        ax4.axvline(np.mean(text_to_prior_sims), color='blue', linestyle='--', linewidth=1, alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ Prior 质量指标可视化已保存: {output_path}")
    
    # ==========================================================================
    # 策略模型可视化
    # ==========================================================================
    
    def visualize_action_distribution(
        self,
        policy_results: List,
        output_path: Path,
    ):
        """
        可视化策略模型的动作分布
        
        Args:
            policy_results: 策略分析结果列表
            output_path: 输出路径
        """
        logger.info("可视化动作分布")
        
        # 聚合所有试验的动作分布
        action_keys = set()
        for result in policy_results:
            action_keys.update(result.action_distribution.keys())
        action_keys = sorted(action_keys)
        
        # 计算平均分布
        avg_distribution = {}
        for key in action_keys:
            values = [r.action_distribution.get(key, 0) for r in policy_results]
            avg_distribution[key] = np.mean(values)
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 条形图
        ax1.bar(range(len(action_keys)), [avg_distribution[k] for k in action_keys], 
                color=self.colors[:len(action_keys)], alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(action_keys)))
        ax1.set_xticklabels(action_keys, rotation=45, ha='right')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Average Action Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 饼图
        ax2.pie(
            [avg_distribution[k] for k in action_keys],
            labels=action_keys,
            colors=self.colors[:len(action_keys)],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Action Distribution Breakdown', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ 动作分布可视化已保存: {output_path}")
    
    def visualize_policy_metrics(
        self,
        policy_results: List,
        output_path: Path,
    ):
        """
        可视化策略模型的性能指标
        
        Args:
            policy_results: 策略分析结果列表
            output_path: 输出路径
        """
        logger.info("可视化策略性能指标")
        
        # 提取指标
        diversities = [r.action_diversity for r in policy_results]
        consistencies = [r.temporal_consistency for r in policy_results]
        repeated_ratios = [r.repeated_action_ratio for r in policy_results]
        success_flags = [r.success for r in policy_results]
        steps = [r.total_steps for r in policy_results]
        
        # 绘图
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. 动作多样性
        ax1 = axes[0, 0]
        ax1.plot(range(len(diversities)), diversities, marker='o', linestyle='-', color='blue', alpha=0.7)
        ax1.axhline(np.mean(diversities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(diversities):.3f}')
        ax1.set_xlabel('Trial Index')
        ax1.set_ylabel('Action Diversity (Entropy)')
        ax1.set_title('Action Diversity Over Trials')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 时序一致性
        ax2 = axes[0, 1]
        ax2.plot(range(len(consistencies)), consistencies, marker='s', linestyle='-', color='green', alpha=0.7)
        ax2.axhline(np.mean(consistencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(consistencies):.3f}')
        ax2.set_xlabel('Trial Index')
        ax2.set_ylabel('Temporal Consistency')
        ax2.set_title('Temporal Consistency Over Trials')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 重复动作比例
        ax3 = axes[0, 2]
        ax3.plot(range(len(repeated_ratios)), repeated_ratios, marker='^', linestyle='-', color='orange', alpha=0.7)
        ax3.axhline(np.mean(repeated_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(repeated_ratios):.3f}')
        ax3.set_xlabel('Trial Index')
        ax3.set_ylabel('Repeated Action Ratio')
        ax3.set_title('Repeated Actions Over Trials')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 成功率
        ax4 = axes[1, 0]
        success_rate = np.mean(success_flags)
        ax4.bar(['Failed', 'Succeeded'], 
                [1 - success_rate, success_rate],
                color=['red', 'green'],
                alpha=0.7,
                edgecolor='black')
        ax4.set_ylabel('Ratio')
        ax4.set_title(f'Success Rate: {success_rate*100:.1f}%')
        ax4.set_ylim([0, 1])
        
        # 添加数值标签
        for i, (label, value) in enumerate([('Failed', 1-success_rate), ('Succeeded', success_rate)]):
            ax4.text(i, value + 0.02, f'{value*100:.1f}%', ha='center', fontweight='bold')
        
        # 5. 步数分布
        ax5 = axes[1, 1]
        ax5.hist(steps, bins=10, color='purple', alpha=0.7, edgecolor='black')
        ax5.axvline(np.mean(steps), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps):.1f}')
        ax5.set_xlabel('Total Steps')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Episode Length Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 成功 vs 失败的指标对比
        ax6 = axes[1, 2]
        success_diversities = [d for d, s in zip(diversities, success_flags) if s]
        fail_diversities = [d for d, s in zip(diversities, success_flags) if not s]
        
        data_to_plot = []
        labels_to_plot = []
        if success_diversities:
            data_to_plot.append(success_diversities)
            labels_to_plot.append('Success')
        if fail_diversities:
            data_to_plot.append(fail_diversities)
            labels_to_plot.append('Failure')
        
        if data_to_plot:
            bp = ax6.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['green', 'red'][:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax6.set_ylabel('Action Diversity')
            ax6.set_title('Diversity: Success vs Failure')
            ax6.grid(True, alpha=0.3, axis='y')
        else:
            ax6.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax6.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ 策略性能指标可视化已保存: {output_path}")
    
    # ==========================================================================
    # 端到端可视化
    # ==========================================================================
    
    def visualize_end_to_end_analysis(
        self,
        end_to_end_results: List,
        output_path: Path,
    ):
        """
        可视化端到端分析结果
        
        Args:
            end_to_end_results: 端到端分析结果列表
            output_path: 输出路径
        """
        logger.info("可视化端到端分析")
        
        # 提取指标
        stage1_contributions = [r.stage1_contribution for r in end_to_end_results]
        stage2_contributions = [r.stage2_contribution for r in end_to_end_results]
        bottleneck_stages = [r.bottleneck_stage for r in end_to_end_results]
        
        # 提取失败归因（只针对失败的试验）
        failure_attributions_prior = []
        failure_attributions_policy = []
        for r in end_to_end_results:
            if r.failure_attribution:
                failure_attributions_prior.append(r.failure_attribution.get('prior', 0))
                failure_attributions_policy.append(r.failure_attribution.get('policy', 0))
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 两阶段贡献对比
        ax1 = axes[0, 0]
        trial_indices = range(len(stage1_contributions))
        width = 0.35
        ax1.bar([i - width/2 for i in trial_indices], stage1_contributions, width, 
                label='Stage 1 (Prior)', color='blue', alpha=0.7, edgecolor='black')
        ax1.bar([i + width/2 for i in trial_indices], stage2_contributions, width,
                label='Stage 2 (Policy)', color='green', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Trial Index')
        ax1.set_ylabel('Contribution')
        ax1.set_title('Stage Contributions Over Trials')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 瓶颈阶段统计
        ax2 = axes[0, 1]
        bottleneck_counts = {0: 0, 1: 0, 2: 0}
        for stage in bottleneck_stages:
            bottleneck_counts[stage] = bottleneck_counts.get(stage, 0) + 1
        
        labels = ['No Bottleneck', 'Stage 1 (Prior)', 'Stage 2 (Policy)']
        colors_pie = ['lightgray', 'cornflowerblue', 'lightgreen']  # 使用浅色代替alpha效果
        values_pie = [bottleneck_counts[i] for i in [0, 1, 2]]
        
        # 注意: matplotlib 3.7.x 的 pie() 不支持 alpha 参数，使用浅色代替
        ax2.pie(values_pie, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Bottleneck Stage Distribution')
        
        # 3. 失败归因（如果有）
        ax3 = axes[1, 0]
        if failure_attributions_prior:
            avg_prior_fault = np.mean(failure_attributions_prior)
            avg_policy_fault = np.mean(failure_attributions_policy)
            
            ax3.bar(['Prior Model', 'Policy Model'], 
                   [avg_prior_fault, avg_policy_fault],
                   color=['blue', 'green'],
                   alpha=0.7,
                   edgecolor='black')
            ax3.set_ylabel('Average Failure Attribution')
            ax3.set_title('Failure Attribution Analysis')
            ax3.set_ylim([0, 1])
            
            # 添加数值标签
            for i, (label, value) in enumerate([('Prior', avg_prior_fault), ('Policy', avg_policy_fault)]):
                ax3.text(i, value + 0.02, f'{value*100:.1f}%', ha='center', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'All Trials Succeeded\n(No Failure Attribution)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Failure Attribution Analysis')
        
        # 4. 综合热力图：Prior质量 vs Policy性能
        ax4 = axes[1, 1]
        prior_similarities = [r.prior_result.text_to_prior_similarity for r in end_to_end_results]
        policy_successes = [1 if r.policy_result.success else 0 for r in end_to_end_results]
        
        # 创建2D直方图
        ax4.scatter(prior_similarities, policy_successes, 
                   c=range(len(prior_similarities)),
                   cmap='viridis',
                   s=100,
                   alpha=0.6,
                   edgecolors='black')
        ax4.set_xlabel('Prior Text-to-Embed Similarity')
        ax4.set_ylabel('Policy Success (0=Fail, 1=Success)')
        ax4.set_title('Prior Quality vs Policy Success')
        ax4.set_ylim([-0.1, 1.1])
        ax4.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(prior_similarities) > 1:
            z = np.polyfit(prior_similarities, policy_successes, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(prior_similarities), max(prior_similarities), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ 端到端分析可视化已保存: {output_path}")
    
    def visualize_task_comparison(
        self,
        task_results: Dict[str, List],
        output_path: Path,
    ):
        """
        对比多个任务的性能
        
        Args:
            task_results: 任务结果字典 {task_id: [end_to_end_results]}
            output_path: 输出路径
        """
        logger.info("可视化多任务对比")
        
        # 提取每个任务的统计指标
        task_ids = list(task_results.keys())
        success_rates = []
        avg_prior_sims = []
        avg_policy_diversities = []
        
        for task_id in task_ids:
            results = task_results[task_id]
            
            # 成功率
            successes = [r.policy_result.success for r in results]
            success_rates.append(np.mean(successes))
            
            # Prior 相似度
            prior_sims = [r.prior_result.text_to_prior_similarity for r in results]
            avg_prior_sims.append(np.mean(prior_sims))
            
            # Policy 多样性
            diversities = [r.policy_result.action_diversity for r in results]
            avg_policy_diversities.append(np.mean(diversities))
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 成功率对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(task_ids)), success_rates, color=self.colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(task_ids)))
        ax1.set_xticklabels([tid[:20] + '...' if len(tid) > 20 else tid for tid in task_ids], 
                            rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate Across Tasks')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 2. Prior 相似度对比
        ax2 = axes[0, 1]
        ax2.bar(range(len(task_ids)), avg_prior_sims, color=self.colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(task_ids)))
        ax2.set_xticklabels([tid[:20] + '...' if len(tid) > 20 else tid for tid in task_ids],
                            rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Average Text-Prior Similarity')
        ax2.set_title('Prior Quality Across Tasks')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Policy 多样性对比
        ax3 = axes[1, 0]
        ax3.bar(range(len(task_ids)), avg_policy_diversities, color=self.colors, alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(task_ids)))
        ax3.set_xticklabels([tid[:20] + '...' if len(tid) > 20 else tid for tid in task_ids],
                            rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Average Action Diversity')
        ax3.set_title('Policy Diversity Across Tasks')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 综合散点图：Prior 质量 vs 成功率
        ax4 = axes[1, 1]
        scatter = ax4.scatter(avg_prior_sims, success_rates,
                             c=range(len(task_ids)),
                             cmap='tab20',
                             s=200,
                             alpha=0.7,
                             edgecolors='black',
                             linewidths=2)
        
        # 添加任务标签
        for i, task_id in enumerate(task_ids):
            ax4.annotate(
                task_id[:15] + '...' if len(task_id) > 15 else task_id,
                (avg_prior_sims[i], success_rates[i]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        ax4.set_xlabel('Average Prior Similarity')
        ax4.set_ylabel('Success Rate')
        ax4.set_title('Prior Quality vs Success Rate')
        ax4.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(avg_prior_sims) > 1:
            z = np.polyfit(avg_prior_sims, success_rates, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(avg_prior_sims), max(avg_prior_sims), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✓ 多任务对比可视化已保存: {output_path}")

