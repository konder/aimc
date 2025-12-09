"""
统一的图表样式配置
Unified Chart Styles

参考 CraftJarvis SkillDiscovery 网站风格
https://craftjarvis.github.io/SkillDiscovery/
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 使用 Agg 后端，避免 GUI 问题
matplotlib.use('Agg')

# 主题色彩 (与 HTML 样式一致)
COLORS = {
    'primary': '#3d5afe',      # 主色 - 蓝色
    'secondary': '#667eea',    # 次要色 - 紫蓝
    'success': '#00c853',      # 成功 - 绿色
    'warning': '#ff9100',      # 警告 - 橙色
    'error': '#ff5252',        # 错误 - 红色
    'text': '#333333',         # 文本 - 深灰
    'text_light': '#666666',   # 浅文本
    'background': '#fafafa',   # 背景 - 浅灰
    'card': '#ffffff',         # 卡片背景
    'border': '#e8eaf0',       # 边框
    'grid': '#f0f0f0',         # 网格线
}

# 色彩方案
PALETTE = [
    '#3d5afe',  # 蓝
    '#00c853',  # 绿
    '#ff9100',  # 橙
    '#ff5252',  # 红
    '#9c27b0',  # 紫
    '#00bcd4',  # 青
    '#ffeb3b',  # 黄
    '#795548',  # 棕
    '#607d8b',  # 蓝灰
    '#e91e63',  # 粉
]


def setup_chart_style():
    """
    设置统一的图表样式
    
    调用此函数以应用统一风格到所有 matplotlib 图表
    """
    # 字体配置 (中文支持)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图表尺寸
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.facecolor'] = COLORS['background']
    
    # 字体大小
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    # 颜色
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['xtick.color'] = COLORS['text_light']
    plt.rcParams['ytick.color'] = COLORS['text_light']
    
    # 边框和网格
    plt.rcParams['axes.edgecolor'] = COLORS['border']
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['grid.color'] = COLORS['grid']
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.8
    
    # 图例
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = COLORS['border']
    plt.rcParams['legend.facecolor'] = COLORS['card']
    
    # 保存设置
    plt.rcParams['savefig.facecolor'] = COLORS['background']
    plt.rcParams['savefig.edgecolor'] = 'none'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.2


def create_figure(nrows: int = 1, ncols: int = 1, figsize: tuple = None, 
                  title: str = None) -> tuple:
    """
    创建统一风格的图表
    
    Args:
        nrows: 行数
        ncols: 列数
        figsize: 图表尺寸 (宽, 高)
        title: 总标题
        
    Returns:
        (fig, axes) 元组
    """
    setup_chart_style()
    
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.set_facecolor(COLORS['background'])
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='600', color=COLORS['text'], y=1.02)
    
    return fig, axes


def style_axis(ax, title: str = None, xlabel: str = None, ylabel: str = None,
               grid: bool = True, legend: bool = False):
    """
    为单个轴应用统一风格
    
    Args:
        ax: matplotlib 轴对象
        title: 标题
        xlabel: X轴标签
        ylabel: Y轴标签
        grid: 是否显示网格
        legend: 是否显示图例
    """
    # 背景
    ax.set_facecolor(COLORS['card'])
    
    # 边框
    for spine in ax.spines.values():
        spine.set_color(COLORS['border'])
        spine.set_linewidth(1)
    
    # 标题和标签
    if title:
        ax.set_title(title, fontsize=13, fontweight='600', color=COLORS['text'], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=COLORS['text_light'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=COLORS['text_light'])
    
    # 刻度
    ax.tick_params(colors=COLORS['text_light'], labelsize=10)
    
    # 网格
    if grid:
        ax.grid(True, linestyle='-', alpha=0.4, color=COLORS['grid'])
        ax.set_axisbelow(True)
    
    # 图例
    if legend:
        ax.legend(frameon=True, facecolor=COLORS['card'], 
                  edgecolor=COLORS['border'], framealpha=0.95)


def plot_heatmap(ax, data: np.ndarray, labels: list = None, 
                 title: str = None, cmap: str = 'RdYlBu_r',
                 vmin: float = 0, vmax: float = 1, 
                 show_values: bool = True, fmt: str = '.2f'):
    """
    绘制统一风格的热力图
    
    Args:
        ax: matplotlib 轴对象
        data: 2D 数据数组
        labels: 行/列标签
        title: 标题
        cmap: 颜色映射
        vmin, vmax: 颜色范围
        show_values: 是否显示数值
        fmt: 数值格式
    """
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # 设置标签
    if labels:
        n = len(labels)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
    
    # 添加数值
    if show_values and data.size <= 100:  # 避免过多数值
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                color = 'white' if value > (vmax + vmin) / 2 else COLORS['text']
                ax.text(j, i, f'{value:{fmt}}', ha='center', va='center', 
                       color=color, fontsize=8)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_color(COLORS['border'])
    
    style_axis(ax, title=title, grid=False)


def plot_scatter(ax, x: np.ndarray, y: np.ndarray, labels: list = None,
                 colors: list = None, title: str = None, 
                 xlabel: str = None, ylabel: str = None,
                 show_labels: bool = True, alpha: float = 0.8):
    """
    绘制统一风格的散点图
    
    Args:
        ax: matplotlib 轴对象
        x, y: 坐标数据
        labels: 点标签
        colors: 点颜色
        title: 标题
        xlabel, ylabel: 轴标签
        show_labels: 是否显示文本标签
        alpha: 透明度
    """
    if colors is None:
        colors = [COLORS['primary']] * len(x)
    
    scatter = ax.scatter(x, y, c=colors, s=80, alpha=alpha, edgecolors='white', linewidths=1)
    
    # 添加文本标签
    if show_labels and labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), fontsize=8, 
                       xytext=(5, 5), textcoords='offset points',
                       color=COLORS['text_light'])
    
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=True)


def plot_bar(ax, x: list, y: list, colors: list = None, 
             title: str = None, xlabel: str = None, ylabel: str = None,
             horizontal: bool = False):
    """
    绘制统一风格的条形图
    
    Args:
        ax: matplotlib 轴对象
        x: 类别标签
        y: 数值
        colors: 条形颜色
        title: 标题
        xlabel, ylabel: 轴标签
        horizontal: 是否水平
    """
    if colors is None:
        colors = COLORS['primary']
    
    if horizontal:
        bars = ax.barh(x, y, color=colors, edgecolor='white', linewidth=0.5)
    else:
        bars = ax.bar(x, y, color=colors, edgecolor='white', linewidth=0.5)
    
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=True)


def plot_histogram(ax, data: np.ndarray, bins: int = 30, 
                   title: str = None, xlabel: str = None, ylabel: str = 'Frequency'):
    """
    绘制统一风格的直方图
    
    Args:
        ax: matplotlib 轴对象
        data: 数据数组
        bins: 分箱数
        title: 标题
        xlabel, ylabel: 轴标签
    """
    ax.hist(data, bins=bins, color=COLORS['primary'], alpha=0.7, 
            edgecolor='white', linewidth=0.5)
    
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=True)


def get_color_for_value(value: float, thresholds: tuple = (0.3, 0.5, 0.8)) -> str:
    """
    根据数值获取颜色
    
    Args:
        value: 0-1 之间的数值
        thresholds: (poor, warning, good) 阈值
        
    Returns:
        颜色代码
    """
    poor, warning, good = thresholds
    
    if value >= good:
        return COLORS['success']
    elif value >= warning:
        return COLORS['primary']
    elif value >= poor:
        return COLORS['warning']
    else:
        return COLORS['error']


def save_figure(fig, path: str, dpi: int = 150):
    """
    保存图表
    
    Args:
        fig: matplotlib 图表对象
        path: 保存路径
        dpi: 分辨率
    """
    fig.savefig(path, dpi=dpi, facecolor=COLORS['background'], 
                edgecolor='none', bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)


# 初始化时设置样式
setup_chart_style()

