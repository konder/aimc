"""
评估框架 - 任务管理与调度
Evaluation Framework - Task Management and Scheduling

职责:
- 管理 STEVE1Evaluator 实例
- 从 YAML 加载任务配置
- 单/批量任务调度
- 三阶段评估: Prior指标 → 动作相似度 → Policy执行
- 结果收集与聚合
- 生成综合HTML报告
"""

# ⚠️ 警告过滤必须在所有导入之前
import warnings
import os

# 屏蔽常见的第三方库警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*has_cuda.*deprecated.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*invalid value encountered in cast.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*RequestsDependencyWarning.*')
warnings.filterwarnings('ignore', message='.*Unable to find acceptable character detection.*')

# 屏蔽 Intel MKL 警告
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['KMP_WARNINGS'] = '0'

import sys
import logging
import json
import yaml
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy.spatial.distance import cosine

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 屏蔽 MineDojo 日志（必须在导入 minedojo 相关模块之前）
logging.getLogger('minedojo.tasks').setLevel(logging.WARNING)
logging.getLogger('minedojo').setLevel(logging.WARNING)

# 导入自定义环境（触发环境注册）
import src.envs

from src.evaluation.policy_evaluator import STEVE1Evaluator
from src.evaluation.metrics import TaskResult, TrialResult
from src.evaluation.task_loader import TaskLoader
from src.evaluation.checkpoint import CheckpointManager, CheckpointConfig
from src.evaluation.prior_evaluator import Steve1PriorEvaluator
from src.utils.evaluation_report_generator import PriorHTMLGenerator
from src.utils.device import DEVICE

# 延迟导入 (避免循环依赖)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置"""
    # STEVE-1 模型配置
    model_path: str = "data/weights/vpt/2x.model"
    weights_path: str = "data/weights/steve1/steve1.weights"
    prior_weights: str = "data/weights/steve1/steve1_prior.pt"
    text_cond_scale: float = 6.0
    visual_cond_scale: float = 7.0
    seed: int = 42
    enable_render: bool = False
    enable_report: bool = False
    video_size: Optional[Tuple[int, int]] = None  # 视频尺寸 (width, height)，None 表示不录制
    
    # 评估配置
    n_trials: Optional[int] = None  # None 表示使用配置文件中的值
    max_steps: Optional[int] = None  # None 表示使用配置文件中的值
    
    # 路径配置
    task_config_path: str = "config/eval_tasks.yaml"
    results_dir: str = "data/evaluation"
    output_dir: Optional[str] = None  # 自定义输出目录（优先级高于 results_dir）
    checkpoint_dir: Optional[str] = None  # 检查点目录（None表示使用输出目录下的checkpoints/）
    
    # 检查点配置
    enable_checkpoint: bool = True  # 启用检查点
    checkpoint_save_interval: int = 5  # 每N个trial保存一次检查点
    checkpoint_auto_resume: bool = True  # 自动恢复
    checkpoint_cleanup_on_complete: bool = True  # 完成后清理检查点
    
    # 环境重建策略配置
    rebuild_interval: int = 15  # 每N个trial完全重建环境（0表示每次重建，-1表示从不重建）
    
    # 三阶段评估开关
    enable_prior_eval: bool = True  # 启用 Prior 指标评估
    enable_action_similarity: bool = True  # 启用动作相似度评估
    enable_policy_eval: bool = True  # 启用 Policy 执行评估
    
    # Prior 评估配置
    prior_n_samples: int = 10  # Prior 一致性采样次数
    train_samples_dir: str = "data/train_samples"  # 训练样本目录


@dataclass
class TaskEvaluationResult:
    """单个任务的完整评估结果（包含三阶段）"""
    task_id: str
    instruction: str
    language: str
    category: str
    
    # Policy 执行结果
    policy_result: Optional[TaskResult] = None
    
    # Prior 指标
    prior_metrics: Dict = field(default_factory=dict)
    
    # 动作相似度指标
    action_similarity_metrics: Dict = field(default_factory=dict)
    
    # 目标接近度指标（从 Policy 执行中获取）
    goal_progress_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            'task_id': self.task_id,
            'instruction': self.instruction,
            'language': self.language,
            'category': self.category,
            'prior_metrics': self.prior_metrics,
            'action_similarity_metrics': self.action_similarity_metrics,
            'goal_progress_metrics': self.goal_progress_metrics,
        }
        
        if self.policy_result:
            result['policy_result'] = {
                'success_rate': self.policy_result.success_rate,
                'avg_steps': self.policy_result.avg_steps,
                'avg_time': self.policy_result.avg_time,
                'n_trials': len(self.policy_result.trials),
            }
        
        return result


class EvaluationFramework:
    """
    评估框架 - 任务管理与调度器
    
    架构:
        EvaluationFramework (Manager/Scheduler)
            ↓ 管理
        STEVE1Evaluator (Worker/Executor)
            ↓ 执行
        Environment + Agent
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        evaluator: Optional[STEVE1Evaluator] = None
    ):
        """
        初始化评估框架
        
        Args:
            config: 评估配置（如果None则使用默认配置）
            evaluator: STEVE1Evaluator 实例（如果None则自动创建）
        """
        # 配置日志过滤器，过滤掉不必要的警告
        self._setup_log_filters()

        logger.info(f"{'='*30}")
        logger.info(f"调度器加载...")
        logger.info(f"{'='*30}")
        
        self.config = config or EvaluationConfig()
        
        # 加载任务配置
        self.task_loader = TaskLoader(self.config.task_config_path)
        logger.info(f"加载任务配置: {self.config.task_config_path}")
        logger.info(f"  发现 {len(self.task_loader.tasks)} 个任务")
        
        # 确定输出目录和检查点目录
        self.base_output_dir = Path(self.config.output_dir) if self.config.output_dir else Path(self.config.results_dir)
        
        # 检查点管理器（延迟初始化，在evaluate_task_set中根据任务输出目录创建）
        self.checkpoint_manager = None
        self.checkpoint_config = CheckpointConfig(
            enabled=self.config.enable_checkpoint,
            save_interval=self.config.checkpoint_save_interval,
            auto_resume=self.config.checkpoint_auto_resume,
            cleanup_on_complete=self.config.checkpoint_cleanup_on_complete
        )
        if self.config.enable_checkpoint:
            logger.info(f"检查点功能已启用")
            logger.info(f"  保存间隔: 每{self.checkpoint_config.save_interval}个trial")
            logger.info(f"  自动恢复: {'是' if self.checkpoint_config.auto_resume else '否'}")
        
        # 环境重建策略
        if self.config.rebuild_interval == 0:
            logger.info(f"环境重建策略: 每次trial都重建（最稳定，最慢）")
        elif self.config.rebuild_interval > 0:
            logger.info(f"环境重建策略: 每{self.config.rebuild_interval}次trial重建（推荐）")
        else:
            logger.info(f"环境重建策略: 从不重建，只轻量清理（最快，可能不稳定）")

        # 保留 evaluator 参数用于向后兼容
        self.evaluator = evaluator  # 通常为 None
        if self.evaluator:
            logger.info("使用提供的评估器实例")
        
        # 共享的任务评估器（延迟初始化，模型只加载一次）
        self._shared_evaluator: Optional[STEVE1Evaluator] = None
        
        # 结果存储
        self.results: List[TaskResult] = []
        
        # Task-set 目录（用于批量评估时组织结果）
        self.current_task_set_dir: Optional[Path] = None
        
        # Prior 评估器（延迟初始化）
        self._prior_evaluator: Optional[Steve1PriorEvaluator] = None
        
        #logger.info("评估框架初始化完成")
    
    def _setup_log_filters(self):
        """配置日志系统：格式化、过滤器等"""
        import warnings
        from src.utils.logging_config import setup_evaluation_logging
        
        # 配置统一的日志格式和过滤器（缩短模块名、过滤不需要的日志）
        setup_evaluation_logging()
        
        # 1. 过滤 PyTorch 的 UserWarning（如 autocast 警告）
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')
        warnings.filterwarnings('ignore', message='.*CUDA is not available.*')
        warnings.filterwarnings('ignore', message='.*Implicit dimension choice for softmax.*')
        
        # 2. 完全静默 MineRL/Malmo 日志（包括 ERROR）
        minerl_loggers = [
            'minerl.env.malmo.instance',
            'minerl.env._multiagent',
            'minerl.env.malmo',
            'process_watcher',
        ]
        for logger_name in minerl_loggers:
            minerl_logger = logging.getLogger(logger_name)
            minerl_logger.setLevel(logging.CRITICAL + 1)  # 完全静默
            minerl_logger.propagate = False  # 不传播到父 logger
        
        # 3. 过滤 STEVE-1 的 UserWarning
        warnings.filterwarnings('ignore', category=UserWarning, module='steve1')
        
        # 4. 过滤 NumPy 和 MineRL 的 RuntimeWarning
        warnings.filterwarnings('ignore', message='.*invalid value encountered in cast.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*minerl.utils.process_watcher.*found in sys.modules.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='runpy')
        
        logger.debug("日志系统已配置：缩短模块名、过滤不需要的日志")
    
    def _get_prior_evaluator(self) -> Steve1PriorEvaluator:
        """获取 Prior 评估器（延迟初始化）"""
        if self._prior_evaluator is None:
            #logger.info("初始化 Prior 评估器...")
            
            # 从目录结构加载训练样本嵌入
            train_samples_dict = self._load_train_samples(
                self.config.train_samples_dir
            )
            
            # 从任务配置中提取指令变体
            instruction_variants_dict = self._load_instruction_variants_from_config()
            
            # 创建 Prior 评估器（不传路径，手动设置数据）
            self._prior_evaluator = Steve1PriorEvaluator(
                prior_weights=self.config.prior_weights,
                success_visuals_path=None,  # 不传路径
                seed=self.config.seed
            )
            
            # 手动设置已加载的训练样本数据
            self._prior_evaluator.success_visuals = train_samples_dict
            # if success_visuals_dict:
            #     logger.info(f"✓ 已加载 {len(success_visuals_dict)} 个任务的成功画面")
            
            # 手动设置指令变体数据
            self._prior_evaluator.instruction_variants = instruction_variants_dict
            # if instruction_variants_dict:
            #     logger.info(f"✓ 已加载 {len(instruction_variants_dict)} 个任务的指令变体")
            #     # 打印加载的任务ID用于调试
                #logger.debug(f"    变体任务: {list(instruction_variants_dict.keys())[:5]}...")
            
        return self._prior_evaluator
    
    def _get_shared_evaluator(self) -> STEVE1Evaluator:
        """
        获取共享的任务评估器（延迟初始化，模型只加载一次）
        
        模型（VPT、STEVE-1、MineCLIP、Prior）只在首次调用时加载，
        后续调用复用同一实例，只需更新环境配置。
        """
        if self._shared_evaluator is None:
            logger.info("初始化共享评估器（模型只加载一次）...")
            self._shared_evaluator = STEVE1Evaluator(
                model_path=self.config.model_path,
                weights_path=self.config.weights_path,
                prior_weights=self.config.prior_weights,
                text_cond_scale=self.config.text_cond_scale,
                visual_cond_scale=self.config.visual_cond_scale,
                seed=self.config.seed,
                env_name=None,  # 环境稍后按任务配置
                env_config=None,
                rebuild_interval=self.config.rebuild_interval,
                checkpoint_manager=self.checkpoint_manager,
                checkpoint_config=self.checkpoint_config
            )
            # 预加载模型（不加载环境）
            self._shared_evaluator._load_models()
            logger.info("✓ 共享评估器初始化完成（模型已加载）")
        
        return self._shared_evaluator
    
    def _load_instruction_variants_from_config(self) -> Dict:
        """
        从任务配置中提取指令变体数据
        
        将 YAML 中的嵌套格式转换为 Steve1PriorEvaluator 期望的格式:
        {task_id: {'variants': [list of variant strings]}}
        
        Returns:
            Dict[task_id, {'variants': List[str]}]
        """
        instruction_variants = {}
        
        # 遍历所有任务
        for task_id in self.task_loader.tasks.keys():
            task_config = self.task_loader.get_task(task_id)
            if not task_config:
                continue
            
            variants_config = task_config.get('instruction_variants', {})
            if not variants_config:
                logger.debug(f"  任务 {task_id} 没有 instruction_variants 配置")
                continue
            
            # 从嵌套结构中提取所有变体字符串
            all_variants = []
            
            # instruction_variants 是一个嵌套结构:
            # simple_direct:
            #   name: "..."
            #   variants:
            #     - "variant 1"
            #     - "variant 2"
            for category_id, category_data in variants_config.items():
                if isinstance(category_data, dict):
                    category_variants = category_data.get('variants', [])
                    if isinstance(category_variants, list):
                        all_variants.extend(category_variants)
                elif isinstance(category_data, list):
                    # 兼容简单列表格式
                    all_variants.extend(category_data)
            
            if all_variants:
                instruction_variants[task_id] = {
                    'variants': all_variants
                }
                logger.debug(f"  加载 {task_id} 的 {len(all_variants)} 个指令变体")
        
        return instruction_variants
    
    def _load_train_samples(self, base_dir: str) -> Dict:
        """
        从 train_samples 目录加载训练样本的视觉嵌入
        
        目录结构: {base_dir}/{task_id}/trial{N}/visual_embeds.pkl
        
        Args:
            base_dir: 训练样本目录（默认 data/train_samples）
            
        Returns:
            Dict[task_id, {'success_visual_embeds': List[np.ndarray]}]
        """
        train_samples = {}
        
        samples_dir = Path(base_dir)
        if not samples_dir.exists():
            logger.warning(f"训练样本目录不存在: {base_dir}")
            return {}
        
        for task_dir in samples_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_id = task_dir.name
            embeds = []
            
            # 查找所有 trial 目录中的 visual_embeds.pkl
            trial_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('trial')])
            
            for trial_dir in trial_dirs:
                embed_file = trial_dir / 'visual_embeds.pkl'
                if embed_file.exists():
                    try:
                        with open(embed_file, 'rb') as f:
                            embed = pickle.load(f)
                        if hasattr(embed, 'numpy'):
                            embed = embed.numpy()
                        embeds.append(np.squeeze(embed))
                    except Exception as e:
                        logger.debug(f"    加载失败 {embed_file}: {e}")
            
            if embeds:
                train_samples[task_id] = {
                    'success_visual_embeds': embeds
                }
                logger.debug(f"  加载 {task_id}: {len(embeds)} 个嵌入")
        
        return train_samples
    
    def _compute_prior_metrics(
        self,
        task_id: str,
        instruction: str,
        task_config: Dict
    ) -> Dict:
        """
        计算 Prior 相关指标
        
        Flow 1: Prior 模型评估
        - 目标准确性（Prior输出 vs 成功画面）
        - 一致性（多次采样稳定性）
        - 语义鲁棒性（指令变体）
        
        Args:
            task_id: 任务ID
            instruction: 指令文本
            task_config: 任务配置
            
        Returns:
            Dict containing prior metrics
        """
        if not self.config.enable_prior_eval:
            return {'enabled': False}
        
        try:
            prior_evaluator = self._get_prior_evaluator()
            
            # 计算目标准确性
            goal_accuracy, goal_accuracy_std, n_visuals = prior_evaluator.compute_goal_accuracy(
                task_id=task_id,
                instruction=instruction
            )
            
            # 计算 MineCLIP 基线（直接文本编码 vs 成功画面）
            mineclip_baseline = self._compute_mineclip_baseline(
                prior_evaluator, task_id, instruction
            )
            
            # 计算一致性
            consistency = prior_evaluator.compute_consistency(
                instruction=instruction,
                n_samples=self.config.prior_n_samples
            )
            
            # 计算语义鲁棒性（如果 prior_evaluator 中有该任务的变体）
            semantic_robustness = None
            n_variants = 0
            # 检查 prior_evaluator.instruction_variants 中是否有该任务
            if task_id in prior_evaluator.instruction_variants:
                robustness_result = prior_evaluator.compute_semantic_robustness(task_id)
                if robustness_result[0] is not None:
                    semantic_robustness, n_variants = robustness_result
            
            return {
                'enabled': True,
                'goal_accuracy': goal_accuracy,
                'goal_accuracy_std': goal_accuracy_std,
                'mineclip_baseline': mineclip_baseline,
                'consistency': consistency,
                'semantic_robustness': semantic_robustness,
                'n_success_visuals': n_visuals,
                'n_variants': n_variants
            }
            
        except Exception as e:
            logger.warning(f"Prior 评估失败: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def _compute_mineclip_baseline(
        self,
        prior_evaluator,
        task_id: str,
        instruction: str
    ) -> float:
        """
        计算 MineCLIP 基线（直接文本编码 vs 成功画面）
        
        这是一个消融对比：MineCLIP 直接编码文本的结果
        与 Prior 输出进行对比，评估 Prior 带来的增益
        
        Args:
            prior_evaluator: Prior 评估器（包含 mineclip 和 success_visuals）
            task_id: 任务ID
            instruction: 指令文本
            
        Returns:
            MineCLIP 基线相似度（0-1）
        """
        import torch as th
        from scipy.spatial.distance import cosine
        
        # 检查是否有成功画面
        if task_id not in prior_evaluator.success_visuals:
            return 0.0
        
        try:
            # 获取成功画面嵌入
            success_visual_embeds = prior_evaluator.success_visuals[task_id]['success_visual_embeds']
            
            # 使用 MineCLIP 直接编码文本
            with th.no_grad():
                z_text = prior_evaluator._mineclip.encode_text([instruction])[0].cpu().numpy()
            
            # 计算与每个成功画面的相似度
            similarities = []
            for z_visual in success_visual_embeds:
                z_visual = np.squeeze(z_visual)
                sim = 1 - cosine(z_text, z_visual)
                similarities.append(sim)
            
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.warning(f"MineCLIP 基线计算失败: {e}")
            return 0.0
    
    def _load_trial_samples(self, task_samples_dir: Path) -> Dict:
        """
        加载任务的训练样本数据
        
        目录结构: train_samples/{task_id}/trial{N}/
            - frames/*.png
            - actions.json
            - visual_embeds.pkl
        """
        import cv2
        
        trials_data = {}
        
        # 查找所有 trial 目录
        trial_dirs = sorted([d for d in task_samples_dir.iterdir() if d.is_dir() and d.name.startswith('trial')])
        
        for trial_dir in trial_dirs:
            trial_id = trial_dir.name
            frames = []
            
            # 标准结构: trial{N}/frames/*.png
            frames_dir = trial_dir / 'frames'
            if frames_dir.exists():
                frame_files = sorted(frames_dir.glob('*.png'))
                for frame_file in frame_files:
                    frame = cv2.imread(str(frame_file))
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
            
            # 加载动作（如果存在）
            actions = []
            actions_file = trial_dir / 'actions.json'
            if actions_file.exists():
                with open(actions_file, 'r') as f:
                    actions = json.load(f)
            
            # 加载目标嵌入
            goal_embed = None
            embed_file = trial_dir / 'visual_embeds.pkl'
            if embed_file.exists():
                with open(embed_file, 'rb') as f:
                    goal_embed = pickle.load(f)
            
            # 只有有帧数据时才添加
            if frames:
                trials_data[trial_id] = {
                    'frames': frames,
                    'actions': actions,
                    'goal_embed': goal_embed
                }
        
        return trials_data
    
    def evaluate_single_task(
        self,
        task_id: str,
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None,
        parent_dir: Optional[Path] = None,  # 父目录（用于 task-set）
        task_index: Optional[int] = None,  # 任务索引（用于进度显示）
        total_tasks: Optional[int] = None,  # 总任务数（用于进度显示）
    ) -> Tuple[TaskResult, Optional[Path]]:
        """
        评估单个任务
        
        Args:
            task_id: 任务ID
            n_trials: 试验次数（如果None则使用配置中的值）
            max_steps: 最大步数（如果None则使用配置中的值）
            parent_dir: 父目录（如果提供，任务目录将创建在这个目录下）
        
        Returns:
            Tuple[TaskResult, Optional[Path]]: 任务结果 + 输出目录路径
        """
        # 从配置加载任务
        task_config = self.task_loader.get_task(task_id)
        if not task_config:
            raise ValueError(f"任务不存在: {task_id}")
        
        # 确定参数（优先级：命令行参数 > 任务配置 > 默认值）
        # 
        # n_trials 优先级:
        # 1. 函数参数（通常来自命令行 --n-trials，如果用户显式指定了）
        # 2. 全局配置（来自 self.config.n_trials，如果不为 None）
        # 3. 任务配置中的 n_trials
        # 4. 默认值 3
        if n_trials is not None:
            pass  # 使用函数参数（命令行显式指定）
        elif self.config.n_trials is not None:
            n_trials = self.config.n_trials  # 使用全局配置（命令行参数）
        else:
            n_trials = task_config.get('n_trials', 3)  # 使用任务配置或默认值
        
        # max_steps 优先级:
        # 1. 函数参数（通常来自命令行 --max-steps，如果用户显式指定了）
        # 2. 全局配置（来自 self.config.max_steps，如果不为 None）
        # 3. 任务配置中的 max_steps
        # 4. 默认值 2000
        if max_steps is not None:
            pass  # 使用函数参数（命令行显式指定）
        elif self.config.max_steps is not None:
            max_steps = self.config.max_steps  # 使用全局配置（命令行参数）
        else:
            max_steps = task_config.get('max_steps', 2000)  # 使用任务配置或默认值
        
        # 确定指令和语言
        instruction = None
        language = "en"
        
        if 'en_instruction' in task_config:
            instruction = task_config['en_instruction']
            language = "en"
        elif 'zh_instruction' in task_config:
            instruction = task_config['zh_instruction']
            language = "zh"
        
        # 从任务配置读取环境配置（包括奖励配置）
        env_config = task_config.get('env_config', {}).copy()  # 复制一份，避免修改原配置
        env_name = task_config.get('env_name', 'MineRLHarvestEnv-v0')
        
        # 将 max_steps 添加到 env_config 中（作为 max_episode_steps）
        env_config['max_episode_steps'] = max_steps
        
        # 从全局配置读取 image_size（如果任务配置中没有指定）
        if 'image_size' not in env_config:
            global_config = self.task_loader.config.get('evaluation', {})
            global_image_size = global_config.get('image_size')
            if global_image_size:
                # 转换为 tuple 格式 (height, width)
                if isinstance(global_image_size, list) and len(global_image_size) == 2:
                    env_config['image_size'] = tuple(global_image_size)
                    #logger.info(f"使用全局 image_size: {env_config['image_size']}")
                else:
                    env_config['image_size'] = global_image_size
                    #logger.info(f"使用全局 image_size: {env_config['image_size']}")
        
        # 获取动作序列文件路径（如果配置了）
        replay_actions_file = task_config.get('replay_actions_file', None)
        if replay_actions_file:
            logger.info(f"  检测到动作序列文件: {replay_actions_file}")
        
        # 获取共享的评估器（模型只加载一次）
        task_evaluator = self._get_shared_evaluator()
        
        # 更新评估器的任务相关配置（环境配置、渲染设置等）
        task_evaluator.enable_render = self.config.enable_render
        task_evaluator.video_size = self.config.video_size
        task_evaluator.enable_report = self.config.enable_report
        task_evaluator.replay_actions_file = replay_actions_file
        task_evaluator.checkpoint_manager = self.checkpoint_manager
        task_evaluator.checkpoint_config = self.checkpoint_config
        
        logger.info(f"{'='*30}")
        logger.info(f"执行任务: {task_id}")
        logger.info(f"{'='*30}")
        logger.info(f"描述: {task_config.get('description', 'N/A')}")
        logger.info(f"类别: {task_config.get('category', 'N/A')}")
        logger.info(f"难度: {task_config.get('difficulty', 'N/A')}")
        logger.info(f"指令: {instruction}")
        logger.info(f"语言: {language}")
        logger.info(f"试验次数: {n_trials}")
        logger.info(f"最大步数: {max_steps}")
        if env_config.get('specified_biome'):
            logger.info(f"指定Biome: {env_config.get('specified_biome')}")
        if replay_actions_file:
            logger.info(f"回放模式: {replay_actions_file}")
        
        # 创建任务输出目录（总是创建，不管是否保存视频）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{task_id}_{language}_{timestamp}"
        
        # 如果提供了父目录，在父目录下创建任务目录
        if parent_dir:
            output_dir = parent_dir / dir_name
        else:
            # 单独任务评估：使用 base_output_dir
            output_dir = self.base_output_dir / dir_name
            
            # 为单独任务创建检查点管理器（如果还没有）
            if not self.checkpoint_manager and self.checkpoint_config.enabled:
                checkpoint_dir = output_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.checkpoint_manager = CheckpointManager(checkpoint_dir)
                logger.info(f"检查点目录: {checkpoint_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        #logger.info(f"  结果目录: {output_dir}")
        
        # ========== 三阶段评估 ==========
        prior_metrics = {}
        action_similarity_metrics = {}
        goal_progress_metrics = {}
        
        try:
            # === Flow 1: Prior 评估 (离线) ===
            if self.config.enable_prior_eval and instruction:
                logger.info(f"{'-'*10}")
                logger.info(f"Prior 评估")
                logger.info(f"{'-'*10}")
                prior_metrics = self._compute_prior_metrics(task_id, instruction, task_config)
                if prior_metrics.get('enabled'):
                    logger.info(f"目标准确性: {prior_metrics.get('goal_accuracy', 0):.4f}")
                    logger.info(f"一致性: {prior_metrics.get('consistency', 0):.4f}")
                else:
                    logger.info(f"跳过 Prior 评估: {prior_metrics.get('error', 'disabled')}")
            
            # === Flow 2: 动作相似度评估 (委托给 STEVE1Evaluator) ===
            if self.config.enable_action_similarity:
                logger.info(f"{'-'*10}")
                logger.info(f"动作相似度评估")
                logger.info(f"{'-'*10}")
                # 获取目标嵌入
                goal_embed = self._get_goal_embed_for_task(task_id, task_config)
                if goal_embed is not None:
                    # 使用 task_evaluator 计算（复用已加载的模型）
                    samples_dir = Path(task_config.get('train_samples_dir', self.config.train_samples_dir))
                    action_similarity_metrics = task_evaluator.evaluate_expert_data(
                        task_id=task_id,
                        samples_dir=samples_dir,
                        goal_embed=goal_embed
                    )
                    if action_similarity_metrics.get('enabled'):
                        logger.info(f"动作相似度: {action_similarity_metrics.get('action_similarity', 0):.4f}")
                        logger.info(f"专家进度率: {action_similarity_metrics.get('expert_progress_rate', 0):+.2%}")
                    else:
                        logger.info(f"跳过动作相似度: {action_similarity_metrics.get('error', 'disabled')}")
                else:
                    logger.info(f"跳过动作相似度: 无目标嵌入")
                    action_similarity_metrics = {'enabled': False, 'error': 'No goal embedding'}
            
            # === Flow 3: Policy 执行评估 (在线) ===
            policy_result = None
            if self.config.enable_policy_eval:
                logger.info(f"{'-'*10}")
                logger.info(f"Policy 评估")
                logger.info(f"{'-'*10}")
                policy_result = task_evaluator.evaluate_task(
                    task_id=task_id,
                    language=language,
                    n_trials=n_trials,
                    max_steps=max_steps,
                    instruction=instruction,
                    output_dir=output_dir,
                    task_index=task_index,
                    total_tasks=total_tasks,
                    env_name=env_name,      # 传递环境名称
                    env_config=env_config,  # 传递环境配置
                )
                
                # 从 Policy 执行结果中提取目标接近度指标
                if policy_result and policy_result.trials:
                    # 聚合所有 trial 的目标接近度
                    all_progress_rates = []
                    all_monotonic_rates = []
                    all_initial_distances = []
                    all_final_distances = []
                    
                    for trial in policy_result.trials:
                        if hasattr(trial, 'goal_progress_rate'):
                            all_progress_rates.append(trial.goal_progress_rate)
                            all_monotonic_rates.append(trial.goal_monotonic_rate)
                            all_initial_distances.append(trial.goal_initial_distance)
                            all_final_distances.append(trial.goal_final_distance)
                    
                    if all_progress_rates:
                        goal_progress_metrics = {
                            'enabled': True,
                            'model_progress_rate': float(np.mean(all_progress_rates)),
                            'model_monotonic_rate': float(np.mean(all_monotonic_rates)),
                            'model_initial_distance': float(np.mean(all_initial_distances)),
                            'model_final_distance': float(np.mean(all_final_distances)),
                            'n_trials': len(all_progress_rates)
                        }
                        logger.info(f"模型进度率: {goal_progress_metrics['model_progress_rate']:+.2%}")
                        logger.info(f"模型单调率: {goal_progress_metrics['model_monotonic_rate']:.2%}")
                    else:
                        goal_progress_metrics = {'enabled': False, 'error': 'No goal progress data in trials'}
            
            # 使用 policy_result 作为主要结果（向后兼容）
            result = policy_result or TaskResult(
                task_id=task_id,
                language=language,
                instruction=instruction or "",
                success_rate=0.0,
                avg_steps=0,
                avg_time=0.0,
                trials=[]
            )
            
            # 创建综合评估结果（用于 HTML 报告）
            combined_result = TaskEvaluationResult(
                task_id=task_id,
                instruction=instruction or "",
                language=language,
                category=task_config.get('category', 'unknown'),
                policy_result=result,
                prior_metrics=prior_metrics,
                action_similarity_metrics=action_similarity_metrics,
                goal_progress_metrics=goal_progress_metrics
            )
            
            # 保存任务结果到目录
            self._save_task_results(result, output_dir)
            
            # 保存综合评估结果
            self._save_combined_results(combined_result, output_dir)
            
            # 输出评估汇总表格
            self._print_task_summary_table(
                task_id=task_id,
                result=result,
                prior_metrics=prior_metrics,
                action_similarity_metrics=action_similarity_metrics,
                goal_progress_metrics=goal_progress_metrics
            )
            
            # 保存结果
            self.results.append(result)
            
            return result, output_dir
        finally:
            # 只清理环境，不关闭整个评估器（保留模型以便复用）
            logger.info(f"清理任务环境资源...")
            task_evaluator.cleanup_env_only()
            logger.info(f"  ✓ 环境资源已释放")
    
    def _print_task_summary_table(
        self,
        task_id: str,
        result: TaskResult,
        prior_metrics: Dict,
        action_similarity_metrics: Dict,
        goal_progress_metrics: Dict
    ):
        """
        输出任务评估汇总表格
        
        核心指标:
        - ① 目标嵌入基线 (MineCLIP)
        - ② Prior 目标嵌入
        - ③ 接近率基线
        - ④ Policy 接近率
        - ⑤ Policy 成功率
        
        辅助指标:
        - Prior 变体
        - Prior 区分度
        - 动作相似度
        - Camera 相似度
        - 动作熵
        - 时序平滑度
        - 动作覆盖率
        """
        logger.info(f"{'='*40}")
        logger.info(f"任务评估汇总: {task_id}")
        logger.info(f"{'='*40}")
        
        # ========== 核心指标（任务级别） ==========
        mineclip_baseline = prior_metrics.get('mineclip_baseline', 0)
        prior_output = prior_metrics.get('goal_accuracy', 0)
        expert_progress = action_similarity_metrics.get('expert_progress_rate', 0)
        expert_monotonic = action_similarity_metrics.get('expert_monotonic_rate', 0)
        model_progress = goal_progress_metrics.get('model_progress_rate', 0)
        model_monotonic = goal_progress_metrics.get('model_monotonic_rate', 0)
        success_rate = result.success_rate if result else 0
        
        # 核心指标汇总
        logger.info(f"【核心指标】")
        logger.info(f"{'指标':<20} {'值':<15}")
        logger.info(f"{'-'*35}")
        logger.info(f"{'① 目标嵌入基线':<20} {mineclip_baseline:.4f}")
        logger.info(f"{'② Prior 目标嵌入':<20} {prior_output:.4f}")
        logger.info(f"{'③ 接近率基线':<20} {expert_progress:+.1%} / {expert_monotonic:.1%}")
        logger.info(f"{'④ Policy 接近率':<20} {model_progress:+.1%} / {model_monotonic:.1%}")
        logger.info(f"{'⑤ Policy 成功率':<20} {success_rate:.1%}")
        
        # ========== Trial 级别表格 ==========
        if result and result.trials:
            logger.info(f"\n【Trial 级别详情】")
            # 表头
            header = f"{'Trial':<8} {'成功':<6} {'步数':<8} {'时间(s)':<10} {'进度率':<12} {'单调率':<10}"
            logger.info(header)
            logger.info(f"{'-'*len(header)}")
            
            for i, trial in enumerate(result.trials, 1):
                success_mark = "✅" if trial.success else "❌"
                progress_rate = getattr(trial, 'goal_progress_rate', 0)
                monotonic_rate = getattr(trial, 'goal_monotonic_rate', 0)
                logger.info(
                    f"{i:<8} {success_mark:<6} {trial.steps:<8} {trial.time_seconds:<10.1f} "
                    f"{progress_rate:+.1%}      {monotonic_rate:.1%}"
                )
        
        # ========== 辅助指标 ==========
        logger.info(f"【辅助指标】")
        logger.info(f"{'指标':<20} {'值':<15}")
        logger.info(f"{'-'*35}")
        
        # Prior 相关
        variant_alignment = prior_metrics.get('semantic_robustness', 0)
        discriminability = prior_metrics.get('discriminability', 0)
        logger.info(f"{'Prior 变体':<20} {variant_alignment:.4f}")
        logger.info(f"{'Prior 区分度':<20} {discriminability:.4f}")
        
        # Policy 相关
        action_sim = action_similarity_metrics.get('action_similarity', 0)
        camera_sim = action_similarity_metrics.get('camera_similarity', 0)
        action_entropy = action_similarity_metrics.get('action_entropy', 0)
        temporal_smooth = action_similarity_metrics.get('temporal_smoothness', 0)
        action_coverage = action_similarity_metrics.get('action_coverage', 0)
        
        logger.info(f"{'动作相似度':<20} {action_sim:.1%}")
        logger.info(f"{'Camera 相似度':<20} {camera_sim:.1%}")
        logger.info(f"{'动作熵':<20} {action_entropy:.2f}")
        logger.info(f"{'时序平滑度':<20} {temporal_smooth:.1%}")
        logger.info(f"{'动作覆盖率':<20} {action_coverage:.1%}")
        
        logger.info(f"{'='*40}")
    
    def _get_goal_embed_for_task(self, task_id: str, task_config: Dict) -> Optional[np.ndarray]:
        """
        获取任务的目标嵌入（从 train_samples 的 visual_embeds.pkl）
        
        目录结构: train_samples/{task_id}/trial{N}/visual_embeds.pkl
        
        Args:
            task_id: 任务ID
            task_config: 任务配置
            
        Returns:
            目标嵌入或 None
        """
        # 从 train_samples 目录加载
        samples_dir = Path(task_config.get('train_samples_dir', self.config.train_samples_dir))
        task_samples_dir = samples_dir / task_id
        
        if not task_samples_dir.exists():
            logger.debug(f"训练样本目录不存在: {task_samples_dir}")
            return None
        
        # 查找所有 trial 目录中的 visual_embeds.pkl
        trial_dirs = sorted([d for d in task_samples_dir.iterdir() if d.is_dir() and d.name.startswith('trial')])
        embeds = []
        
        for trial_dir in trial_dirs:
            embed_file = trial_dir / 'visual_embeds.pkl'
            if embed_file.exists():
                try:
                    with open(embed_file, 'rb') as f:
                        goal_embed = pickle.load(f)
                    # 确保是 numpy array
                    if hasattr(goal_embed, 'numpy'):
                        goal_embed = goal_embed.numpy()
                    embeds.append(np.squeeze(goal_embed))
                except Exception as e:
                    logger.debug(f"加载 {embed_file} 失败: {e}")
        
        if embeds:
            # 返回所有 trial 嵌入的平均值
            return np.mean(embeds, axis=0)
        
        return None
    
    def _save_combined_results(self, combined_result: TaskEvaluationResult, output_dir: Path):
        """
        保存综合评估结果（包含三阶段指标）
        
        Args:
            combined_result: 综合评估结果
            output_dir: 输出目录
        """
        combined_path = output_dir / "combined_evaluation.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_result.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug(f"  ✓ 综合评估结果已保存: {combined_path.name}")
    
    def _save_task_results(self, result: TaskResult, output_dir: Path):
        """
        保存任务结果到指定目录（JSON、TXT）
        
        注意：视频保存现在由 policy_evaluator 在 _run_single_trial 中完成
        
        Args:
            result: 任务结果
            output_dir: 输出目录
        """
        # 构建结果数据
        result_data = {
            "task_id": result.task_id,
            "language": result.language,
            "instruction": result.instruction,
            "success_rate": result.success_rate,
            "avg_steps": result.avg_steps,
            "avg_time": result.avg_time,
            "trials": [
                {
                    "trial_idx": i + 1,
                    "success": trial.success,
                    "steps": trial.steps,
                    "time_seconds": trial.time_seconds,
                    "has_video": (output_dir / f"trial_{i+1}.mp4").exists()  # 检查视频文件是否存在
                }
                for i, trial in enumerate(result.trials)
            ]
        }
        
        # 保存JSON
        json_path = output_dir / "result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        #logger.info(f"  ✓ 结果已保存: {json_path.name}")
        
        # 保存TXT（人类可读）
        txt_path = output_dir / "result.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"任务评估结果: {result.task_id}\n")
            f.write("="*80 + "\n\n")
            f.write(f"语言: {result.language}\n")
            f.write(f"指令: {result.instruction}\n")
            f.write(f"成功率: {result.success_rate*100:.1f}%\n")
            f.write(f"平均步数: {result.avg_steps:.1f}\n")
            f.write(f"平均时间: {result.avg_time:.1f}s\n\n")
            f.write("试验详情:\n")
            f.write("-"*80 + "\n")
            for i, trial in enumerate(result.trials, 1):
                status = "✅ 成功" if trial.success else "❌ 失败"
                video_status = "🎬" if (output_dir / f"trial_{i}.mp4").exists() else ""
                f.write(f"Trial {i}: {status} | 步数: {trial.steps:4d} | 时间: {trial.time_seconds:.1f}s {video_status}\n")
        #logger.info(f"  ✓ 报告已保存: {txt_path.name}")
    
    def evaluate_task_list(
        self,
        task_ids: List[str],
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None,
        task_set_name: Optional[str] = None  # 任务集名称（用于创建目录）
    ) -> List[TaskResult]:
        """
        批量评估任务列表
        
        Args:
            task_ids: 任务ID列表
            n_trials: 试验次数（应用于所有任务）
            max_steps: 最大步数（应用于所有任务）
            task_set_name: 任务集名称（如果提供，将创建专门的目录）
        
        Returns:
            List[TaskResult]: 任务结果列表
        """
        
        # 如果提供了 task_set_name，创建 task-set 目录
        task_set_dir = None
        if task_set_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_set_dir_name = f"{task_set_name}_{timestamp}"
            task_set_dir = self.base_output_dir / task_set_dir_name
            task_set_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"结果输出目录: {task_set_dir}")
            
            # 创建检查点目录（在task_set_dir下）
            checkpoint_dir = task_set_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"检查点目录: {checkpoint_dir}")
            
            # 为这个任务集创建专属的检查点管理器
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
            
            # 保存 task_set_dir 供后续 generate_report 使用
            self.current_task_set_dir = task_set_dir
        
        # Task-set级别的检查点恢复
        completed_task_ids = []
        remaining_task_ids = task_ids.copy()
        
        if task_set_name and self.checkpoint_manager and self.checkpoint_config.enabled and self.checkpoint_config.auto_resume:
            taskset_checkpoint = self.checkpoint_manager.load_taskset_checkpoint(task_set_name)
            if taskset_checkpoint:
                # 检查任务列表是否匹配
                if taskset_checkpoint['all_task_ids'] == task_ids:
                    completed_task_ids = taskset_checkpoint['completed_task_ids']
                    remaining_task_ids = [tid for tid in task_ids if tid not in completed_task_ids]
                    logger.info(f"📥 发现task-set检查点，恢复进度...")
                    logger.info(f"   已完成: {len(completed_task_ids)}/{len(task_ids)} tasks")
                    logger.info(f"   剩余: {len(remaining_task_ids)} tasks")
                    logger.info(f"   将从第{len(completed_task_ids)+1}个任务继续\n")
                else:
                    logger.warning(f"⚠️ Task-set检查点的任务列表不匹配，忽略检查点")
        
        results = []
        
        # 只评估剩余的任务
        for i, task_id in enumerate(task_ids, 1):
            # 检查是否已完成
            if task_id in completed_task_ids:
                logger.info(f"[{i}/{len(task_ids)}] ⏭️  跳过已完成任务: {task_id}")
                continue
            
            #logger.info(f"[{i}/{len(task_ids)}] 评估任务: {task_id}")
            
            try:
                # evaluate_single_task 返回 (TaskResult, output_dir)
                result, _ = self.evaluate_single_task(
                    task_id=task_id,
                    n_trials=n_trials,
                    max_steps=max_steps,
                    parent_dir=task_set_dir,  # 传递 task-set 目录
                    task_index=i,  # 传递任务索引
                    total_tasks=len(task_ids)  # 传递总任务数
                )
                results.append(result)  # 只保存 TaskResult
                
                # 打印任务摘要
                logger.info(f"{task_id} 完成: 成功率 {result.success_rate*100:.1f}%, "
                           f"平均步数 {result.avg_steps:.1f}")
                
                # 保存task-set检查点（每完成一个任务就保存）
                if task_set_name and self.checkpoint_manager and self.checkpoint_config.enabled:
                    completed_task_ids.append(task_id)
                    self.checkpoint_manager.save_taskset_checkpoint(
                        task_set_name=task_set_name,
                        all_task_ids=task_ids,
                        completed_task_ids=completed_task_ids,
                        metadata={
                            "n_trials": n_trials,
                            "max_steps": max_steps
                        }
                    )
                
            except Exception as e:
                logger.error(f"  ❌ 任务失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"批量评估完成: {len(results)}/{len(task_ids)} 个任务成功")
        
        # 完成后清理task-set检查点
        if task_set_name and self.checkpoint_manager and self.checkpoint_config.enabled and self.checkpoint_config.cleanup_on_complete:
            if len(completed_task_ids) == len(task_ids):  # 所有任务都完成了
                self.checkpoint_manager.delete_taskset_checkpoint(task_set_name)
                logger.info(f"  Task-set已全部完成，检查点已清理")
        
        # 注意：不要在这里重置 current_task_set_dir，因为 generate_report 还需要用它
        
        return results
    
    def evaluate_task_set(
        self,
        task_set_name: str,
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> List[TaskResult]:
        """
        评估任务集（从 YAML 配置中的 harvest_tasks, quick_test, baseline_test 等）
        
        Args:
            task_set_name: 任务集名称 ('harvest_tasks', 'quick_test', 'baseline_test')
            n_trials: 试验次数
            max_steps: 最大步数
        
        Returns:
            List[TaskResult]: 任务结果列表
        """
        # 从 YAML 加载任务集
        task_ids = self.task_loader.get_task_set(task_set_name)
        
        if not task_ids:
            raise ValueError(f"任务集不存在或为空: {task_set_name}")
        
        
        logger.info(f"评估任务集: {task_set_name}")
        logger.info(f"任务数量: {len(task_ids)}")
        logger.info(f"任务列表: {', '.join(task_ids)}")
        
        return self.evaluate_task_list(
            task_ids=task_ids,
            n_trials=n_trials,
            max_steps=max_steps,
            task_set_name=task_set_name  # 传递任务集名称
        )
    
    def print_summary(self, results: Optional[List[TaskResult]] = None):
        """
        打印评估结果摘要
        
        Args:
            results: 任务结果列表（如果None则使用self.results）
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("没有评估结果")
            return
        
        print(f"\n{'='*100}")
        print("评估结果汇总")
        print(f"{'='*100}\n")
        
        # 表头
        print(f"{'任务ID':<30} {'指令':<20} {'成功率':<10} {'平均步数':<12} {'平均时间'}")
        print("-" * 100)
        
        # 每个任务的结果
        for result in results:
            task_id = result.task_id[:28]  # 截断过长的ID
            instruction = result.instruction[:18] if result.instruction else "N/A"
            success_rate = f"{result.success_rate * 100:.1f}%"
            avg_steps = f"{result.avg_steps:.1f}"
            avg_time = f"{result.avg_time:.1f}s"
            
            print(f"{task_id:<30} {instruction:<20} {success_rate:<10} {avg_steps:<12} {avg_time}")
        
        # 总体统计
        print("\n" + "-" * 80)
        overall_success = sum(r.success_rate for r in results) / len(results)
        overall_steps = sum(r.avg_steps for r in results) / len(results)
        overall_time = sum(r.avg_time for r in results) / len(results)
        total_trials = sum(len(r.trials) for r in results)
        
        print(f"{'总体统计':<30} {'N/A':<20} {overall_success*100:.1f}% {overall_steps:<12.1f} {overall_time:.1f}s")
        print(f"\n总任务数: {len(results)}")
        print(f"总试验数: {total_trials}")
        print(f"{'='*100}\n")
    
    def generate_report(
        self,
        results: Optional[List[TaskResult]] = None,
        report_name: str = "evaluation_report"
    ):
        """
        生成评估报告
        
        Args:
            results: 任务结果列表（如果None则使用self.results）
            report_name: 报告名称
            
        Returns:
            Tuple[str, str]: JSON报告路径和TXT报告路径
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("没有评估结果，无法生成报告")
            return None, None
        
        # 构建报告数据
        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(results),
                "evaluator": "STEVE-1",
                "framework": "EvaluationFramework"
            },
            "tasks": []
        }
        
        for result in results:
            task_data = {
                "task_id": result.task_id,
                "instruction": result.instruction,
                "language": result.language,
                "success_rate": result.success_rate * 100,  # 转换为百分比
                "avg_steps": result.avg_steps,
                "avg_time": result.avg_time,
                "trials": [
                    {
                        "success": trial.success,
                        "steps": trial.steps,
                        "time_seconds": trial.time_seconds,
                        "final_inventory": trial.final_inventory
                    }
                    for trial in result.trials
                ]
            }
            report_data["tasks"].append(task_data)
        
        # 计算总体统计
        report_data["summary"] = {
            "overall_success_rate": np.mean([r.success_rate for r in results]) * 100,
            "total_trials": sum(len(r.trials) for r in results),
            "successful_trials": sum(sum(1 for t in r.trials if t.success) for r in results)
        }
        
        # 保存JSON报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{report_name}_{timestamp}.json"
        
        # 优先级：task-set 目录 > 单任务目录 > 全局目录
        if self.current_task_set_dir:
            # 多任务评估（task-set），保存到 task-set 目录
            json_path = self.current_task_set_dir / json_filename
            #logger.info(f"将报告保存到 task-set 目录: {self.current_task_set_dir.name}")
        elif len(results) == 1:
            # 单任务评估，保存到任务目录下
            task_id = results[0].task_id
            language = results[0].language
            # 查找匹配的目录（按时间倒序）
            pattern = f"{task_id}_{language}_*"
            matching_dirs = sorted(
                self.base_output_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if matching_dirs:
                json_path = matching_dirs[0] / json_filename
                #logger.info(f"将报告保存到任务目录: {matching_dirs[0].name}")
            else:
                json_path = self.base_output_dir / json_filename
        else:
            # 多任务但无 task-set，使用全局目录
            json_path = self.base_output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 生成文本报告
        txt_path = json_path.with_suffix('.txt')
        self._generate_text_report(report_data, txt_path)
        
        # 生成三维能力矩阵分析和HTML报告
        matrix_analysis, html_path = self._generate_matrix_report(results, json_path.parent)
        
        logger.info(f"报告已生成:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  TXT:  {txt_path}")
        if html_path:
            logger.info(f"  HTML: open {html_path}")
        return str(json_path), str(txt_path)
    
    def _generate_matrix_report(
        self, 
        results: List[TaskResult], 
        output_dir: Path
    ) -> Tuple[Optional[Dict], Optional[Path]]:
        """
        生成 Prior 和 Policy 综合评估 HTML 报告
        
        Args:
            results: 任务结果列表
            output_dir: 输出目录
            
        Returns:
            (analysis_data, html_path): 分析数据和HTML路径
        """
        try:
            # 收集综合评估结果
            logger.info(f"收集综合评估结果: {output_dir}")
            combined_results = self._collect_combined_results(output_dir)
            logger.info(f"收集到 {len(combined_results)} 个任务的综合评估结果")
            
            # 构建报告数据结构（兼容 PriorHTMLGenerator）
            report_data = self._build_report_data(results, combined_results)
            
            # 调试：检查 goal_progress_summary
            goal_progress_summary = report_data.get('summary', {}).get('goal_progress_summary', {})
            logger.debug(f"  goal_progress_summary: {goal_progress_summary}")
            
            if not report_data:
                logger.warning("没有可分析的任务数据")
                return None, None
            
            # 生成可视化图表（如果有多个任务）
            if len(results) >= 2:
                logger.info("生成可视化图表...")
                self._generate_visualizations(results, combined_results, output_dir)
            
            # 生成 HTML 报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_generator = PriorHTMLGenerator(str(output_dir))
            html_path = html_generator.generate_report(
                report_data,
                output_filename=f"evaluation_report_{timestamp}.html"
            )
            
            return report_data, html_path
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None, None
    
    def _generate_visualizations(
        self,
        results: List[TaskResult],
        combined_results: Dict[str, Dict],
        output_dir: Path
    ):
        """
        生成可视化图表
        
        图1: MineCLIP vs Prior 空间对比 (t-SNE)
        图2: Prior 输出相似度矩阵
        图3: Prior 输出方差分布
        图4: 目标接近度概览
        
        Args:
            results: TaskResult 列表
            combined_results: 综合评估结果
            output_dir: 输出目录
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import torch as th
            
            # 配置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 获取 Prior 评估器
            prior_evaluator = self._get_prior_evaluator()
            if prior_evaluator is None:
                logger.warning("  无法获取 Prior 评估器，跳过可视化")
                return
            
            # 收集嵌入数据
            task_ids = []
            prior_embeds = []
            text_embeds = []
            task_info = {}
            visual_embeds_by_task = {}  # {task_id: [visual_embeds]}
            variant_embeds_by_task = {}  # {task_id: [(embed, category_id), ...]}
            
            for result in results:
                task_id = result.task_id
                instruction = result.instruction
                
                if not instruction:
                    continue
                
                task_ids.append(task_id)
                
                # 获取任务配置
                task_config = self.task_loader.get_task(task_id)
                if task_config:
                    category = task_config.get('category', 'other')
                    if category in ['harvest', 'raw_resources', 'plants', 'animal_drops']:
                        cat_display = 'Harvest'
                    elif category == 'combat':
                        cat_display = 'Combat'
                    elif category == 'techtree':
                        cat_display = 'Techtree'
                    else:
                        cat_display = 'Other'
                    
                    tier = task_config.get('tier', 2)
                    task_info[task_id] = {'tier': tier, 'category': cat_display}
                    
                    # 获取指令变体（带分类信息）
                    instruction_variants = task_config.get('instruction_variants', {})
                    variant_embeds_by_task[task_id] = []
                    if instruction_variants:
                        if isinstance(instruction_variants, dict):
                            # 分类变体结构: {category_id: {variants: [...]}}
                            for cat_id, cat_config in instruction_variants.items():
                                if isinstance(cat_config, dict):
                                    variants = cat_config.get('variants', [])[:2]  # 每类取2个
                                elif isinstance(cat_config, list):
                                    variants = cat_config[:2]
                                else:
                                    continue
                                
                                for variant in variants:
                                    try:
                                        z_variant = prior_evaluator._get_prior_embed(variant)
                                        variant_embeds_by_task[task_id].append((z_variant, cat_id))
                                    except:
                                        pass
                        elif isinstance(instruction_variants, list):
                            # 简单列表格式
                            for variant in instruction_variants[:10]:
                                try:
                                    z_variant = prior_evaluator._get_prior_embed(variant)
                                    variant_embeds_by_task[task_id].append((z_variant, 'uncategorized'))
                                except:
                                    pass
                else:
                    task_info[task_id] = {'tier': 2, 'category': 'Other'}
                    variant_embeds_by_task[task_id] = []
                
                # Prior 嵌入
                try:
                    z_prior = prior_evaluator._get_prior_embed(instruction)
                    prior_embeds.append(z_prior)
                except Exception as e:
                    logger.debug(f"  获取 Prior 嵌入失败 ({task_id}): {e}")
                    prior_embeds.append(np.zeros(512))
                
                # MineCLIP 文本嵌入
                try:
                    with th.no_grad():
                        z_text = prior_evaluator._mineclip.encode_text([instruction])[0].cpu().numpy()
                    text_embeds.append(z_text)
                except Exception as e:
                    logger.debug(f"  获取 MineCLIP 嵌入失败 ({task_id}): {e}")
                    text_embeds.append(np.zeros(512))
                
                # 成功视频嵌入
                visual_embeds_by_task[task_id] = []
                if hasattr(prior_evaluator, 'success_visuals') and task_id in prior_evaluator.success_visuals:
                    task_visuals = prior_evaluator.success_visuals[task_id]
                    # success_visuals[task_id] 是一个字典，包含 'success_visual_embeds' 字段
                    if isinstance(task_visuals, dict) and 'success_visual_embeds' in task_visuals:
                        embeds = task_visuals['success_visual_embeds']
                        visual_embeds_by_task[task_id] = list(embeds[:3]) if hasattr(embeds, '__getitem__') else []
                    elif isinstance(task_visuals, (list, np.ndarray)):
                        visual_embeds_by_task[task_id] = list(task_visuals[:3])
            
            if len(task_ids) < 2:
                logger.warning("  任务数少于2个，跳过可视化")
                return
            
            prior_embeds = np.array(prior_embeds)
            text_embeds = np.array(text_embeds)
            
            # === 图1: MineCLIP vs Prior 空间对比 ===
            try:
                logger.info("  生成图1: MineCLIP vs Prior 空间对比...")
                self._plot_mineclip_vs_prior(text_embeds, prior_embeds, task_ids, task_info, output_dir)
            except Exception as e:
                logger.warning(f"    图1生成失败: {e}")
            
            # === 图2: 变体输出 vs 目标视频 ===
            try:
                if any(variant_embeds_by_task.values()) and any(visual_embeds_by_task.values()):
                    logger.info("  生成图2: 变体输出 vs 目标视频...")
                    self._plot_variants_vs_visual(variant_embeds_by_task, visual_embeds_by_task, task_ids, task_info, output_dir)
            except Exception as e:
                logger.warning(f"    图2生成失败: {e}")
            
            # === 图3: Prior vs 目标视频 ===
            try:
                if any(visual_embeds_by_task.values()):
                    logger.info("  生成图3: Prior vs 目标视频...")
                    self._plot_prior_vs_visual(prior_embeds, visual_embeds_by_task, task_ids, task_info, output_dir)
            except Exception as e:
                logger.warning(f"    图3生成失败: {e}")
            
            # === 辅助图1: Prior 输出相似度矩阵 ===
            try:
                logger.info("  生成辅助图1: Prior 输出相似度矩阵...")
                self._plot_similarity_matrix(prior_embeds, task_ids, output_dir)
            except Exception as e:
                logger.warning(f"    辅助图1生成失败: {e}")
            
            # === 辅助图2: Prior 输出方差分布 ===
            try:
                logger.info("  生成辅助图2: Prior 输出方差分布...")
                self._plot_variance_distribution(prior_embeds, output_dir)
            except Exception as e:
                logger.warning(f"    辅助图2生成失败: {e}")
            
            # === 图4: 目标接近度概览 ===
            goal_progress_data = {}
            for task_id, combined in combined_results.items():
                action_metrics = combined.get('action_similarity_metrics', {})
                goal_metrics = combined.get('goal_progress_metrics', {})
                if action_metrics.get('enabled') or goal_metrics.get('enabled'):
                    goal_progress_data[task_id] = {
                        'expert_progress_rate': action_metrics.get('expert_progress_rate', 0),
                        'expert_monotonic_rate': action_metrics.get('expert_monotonic_rate', 0),
                        'model_progress_rate': goal_metrics.get('model_progress_rate', 0),
                        'model_monotonic_rate': goal_metrics.get('model_monotonic_rate', 0),
                    }
            
            # === 图4: 目标接近度概览 ===
            try:
                if goal_progress_data:
                    logger.info("  生成图4: 目标接近度概览...")
                    self._plot_goal_progress_overview(goal_progress_data, output_dir)
            except Exception as e:
                logger.warning(f"    图4生成失败: {e}")
            
            # === Policy 可视化图表 ===
            # 收集动作分布和详细数据
            all_expert_dist = {}
            all_model_dist = {}
            all_expert_actions = []
            all_model_actions = []
            all_frame_similarities = {}
            all_camera_similarities = {}
            all_expert_distances = {}
            all_model_distances = {}
            
            for task_id, combined in combined_results.items():
                action_metrics = combined.get('action_similarity_metrics', {})
                goal_metrics = combined.get('goal_progress_metrics', {})
                
                # 动作分布
                if action_metrics.get('expert_action_distribution'):
                    for k, v in action_metrics['expert_action_distribution'].items():
                        all_expert_dist[k] = all_expert_dist.get(k, 0) + v
                if action_metrics.get('model_action_distribution'):
                    for k, v in action_metrics['model_action_distribution'].items():
                        all_model_dist[k] = all_model_dist.get(k, 0) + v
                
                # 动作列表（用于混淆矩阵）
                expert_list = action_metrics.get('expert_actions_list', [])
                model_list = action_metrics.get('model_actions_list', [])
                all_expert_actions.extend(expert_list)
                all_model_actions.extend(model_list)
                
                # 帧级别相似度
                if action_metrics.get('frame_similarities'):
                    all_frame_similarities[task_id] = action_metrics['frame_similarities']
                if action_metrics.get('camera_similarities'):
                    all_camera_similarities[task_id] = action_metrics['camera_similarities']
                
                # 专家和模型距离时间线
                if action_metrics.get('expert_distances'):
                    all_expert_distances[task_id] = action_metrics['expert_distances']
                
                # 从 trial 结果获取模型距离
                policy_result = combined.get('policy_result', {})
                if policy_result.get('trials'):
                    for trial in policy_result['trials']:
                        if trial.get('goal_distances'):
                            all_model_distances[task_id] = trial['goal_distances']
                            break
            
            # === 图5: 动作分布对比 ===
            try:
                if all_expert_dist or all_model_dist:
                    logger.info("  生成图5: 动作分布对比...")
                    self._plot_action_distribution(all_expert_dist, all_model_dist, output_dir)
            except Exception as e:
                logger.warning(f"    图5生成失败: {e}")
            
            # === 图6: 动作混淆矩阵 ===
            try:
                if len(all_expert_actions) > 10 and len(all_model_actions) > 10:
                    logger.info("  生成图6: 动作混淆矩阵...")
                    self._plot_confusion_matrix(all_expert_actions, all_model_actions, output_dir)
            except Exception as e:
                logger.warning(f"    图6生成失败: {e}")
            
            # === 图7: 逐帧相似度时间线（所有任务汇总）===
            try:
                if all_frame_similarities:
                    # 合并所有任务的数据
                    merged_frame_sim = []
                    merged_camera_sim = []
                    for task_id in all_frame_similarities:
                        merged_frame_sim.extend(all_frame_similarities[task_id])
                        if task_id in all_camera_similarities:
                            merged_camera_sim.extend(all_camera_similarities[task_id])
                    
                    n_tasks = len(all_frame_similarities)
                    logger.info(f"  生成图7: 逐帧相似度时间线（{n_tasks} 个任务汇总）...")
                    self._plot_similarity_timeline_aggregated(
                        all_frame_similarities,
                        all_camera_similarities,
                        output_dir
                    )
            except Exception as e:
                logger.warning(f"    图7生成失败: {e}")
            
            # === 图8: 目标接近度对比（所有任务汇总）===
            try:
                if all_expert_distances and all_model_distances:
                    common_tasks = set(all_expert_distances.keys()) & set(all_model_distances.keys())
                    if common_tasks:
                        logger.info(f"  生成图8: 目标接近度对比（{len(common_tasks)} 个任务汇总）...")
                        self._plot_goal_progress_comparison_aggregated(
                            all_expert_distances,
                            all_model_distances,
                            output_dir
                        )
            except Exception as e:
                logger.warning(f"    图8生成失败: {e}")
            
            logger.info(f"  ✓ 可视化图表已保存到: {output_dir}")
            
        except ImportError as e:
            logger.warning(f"  可视化依赖缺失: {e}")
        except Exception as e:
            import traceback
            logger.warning(f"  可视化生成失败: {e}")
            logger.warning(f"  详细错误: {traceback.format_exc()}")
    
    def _plot_mineclip_vs_prior(
        self,
        text_embeds: np.ndarray,
        prior_embeds: np.ndarray,
        task_ids: List[str],
        task_info: Dict,
        output_dir: Path
    ):
        """
        图1: MineCLIP vs Prior 空间对比 (t-SNE)
        生成两张图：按 Tier 和按 Category 着色
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # 颜色定义
        TIER_COLORS = {1: '#4CAF50', 2: '#FF9800', 3: '#F44336'}
        TIER_NAMES = {1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3'}
        CAT_COLORS = {'Harvest': '#4CAF50', 'Combat': '#F44336', 'Techtree': '#2196F3', 'Other': '#9E9E9E'}
        
        # 合并嵌入
        all_embeds = np.vstack([text_embeds, prior_embeds])
        n = len(task_ids)
        
        if len(all_embeds) < 3:
            return
        
        try:
            perplexity = min(30, max(2, len(all_embeds) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(all_embeds)
            
            # === 图1a: 按 Tier 着色 ===
            fig, ax = plt.subplots(figsize=(10, 8))
            drawn_tiers = set()
            for i, task in enumerate(task_ids):
                info = task_info.get(task, {'tier': 2})
                tier = info['tier']
                color = TIER_COLORS.get(tier, '#9E9E9E')
                tier_name = TIER_NAMES.get(tier, f'Tier {tier}')
                
                label_mc = f'○ MineCLIP ({tier_name})' if tier not in drawn_tiers else None
                ax.scatter(coords[i, 0], coords[i, 1], marker='o', s=100, 
                          c=[color], alpha=0.5, edgecolors='white', linewidth=1, label=label_mc)
                
                label_prior = f'▲ Prior ({tier_name})' if tier not in drawn_tiers else None
                ax.scatter(coords[n + i, 0], coords[n + i, 1], marker='^', s=130, 
                          c=[color], edgecolors='black', linewidth=1.5, label=label_prior)
                drawn_tiers.add(tier)
                
                ax.plot([coords[i, 0], coords[n + i, 0]], [coords[i, 1], coords[n + i, 1]], 
                       c=color, linestyle='--', alpha=0.3, linewidth=1)
            
            ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
            ax.set_title('MineCLIP vs Prior (by Tier, t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "viz_1a_mineclip_vs_prior_tier.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # === 图1b: 按 Category 着色 ===
            fig, ax = plt.subplots(figsize=(10, 8))
            drawn_cats = set()
            for i, task in enumerate(task_ids):
                info = task_info.get(task, {'category': 'Other'})
                cat = info['category']
                color = CAT_COLORS.get(cat, '#9E9E9E')
                
                label_mc = f'○ MineCLIP ({cat})' if cat not in drawn_cats else None
                ax.scatter(coords[i, 0], coords[i, 1], marker='o', s=100, 
                          c=[color], alpha=0.5, edgecolors='white', linewidth=1, label=label_mc)
                
                label_prior = f'▲ Prior ({cat})' if cat not in drawn_cats else None
                ax.scatter(coords[n + i, 0], coords[n + i, 1], marker='^', s=130, 
                          c=[color], edgecolors='black', linewidth=1.5, label=label_prior)
                drawn_cats.add(cat)
                
                ax.plot([coords[i, 0], coords[n + i, 0]], [coords[i, 1], coords[n + i, 1]], 
                       c=color, linestyle='--', alpha=0.3, linewidth=1)
            
            ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
            ax.set_title('MineCLIP vs Prior (by Category, t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "viz_1b_mineclip_vs_prior_category.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"    图1生成失败: {e}")
    
    def _plot_variants_vs_visual(
        self,
        variant_embeds_by_task: Dict[str, List],  # {task_id: [(embed, category_id), ...]}
        visual_embeds_by_task: Dict[str, List[np.ndarray]],
        task_ids: List[str],
        task_info: Dict,
        output_dir: Path
    ):
        """
        图2: 变体输出 vs 目标视频 (t-SNE)
        生成两张图：按变体类别着色 和 按任务着色
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # 颜色调色板
        PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 变体类别显示名称和颜色
        CATEGORY_DISPLAY = {
            'simple_direct': 'Simple Direct',
            'reddit_casual': 'Reddit Casual',
            'reddit_posts': 'Reddit Posts',
            'reddit_comments': 'Reddit Comments',
            'location_based': 'Location Based',
            'purpose_oriented': 'Purpose Oriented',
            'building_purpose': 'Building Purpose',
            'action_detailed': 'Action Detailed',
            'conversational': 'Conversational',
            'beginner_advice': 'Beginner Advice',
            'casual_slang': 'Casual Slang',
            'survival_urgency': 'Survival Urgency',
            'meme_culture': 'Meme Culture',
            'meme_humor': 'Meme Humor',
            'humor_meme': 'Humor Meme',
            'uncategorized': 'Uncategorized',
        }
        
        CATEGORY_COLORS = {
            'simple_direct': '#4CAF50',
            'reddit_casual': '#2196F3',
            'reddit_posts': '#03A9F4',
            'reddit_comments': '#00BCD4',
            'location_based': '#009688',
            'purpose_oriented': '#FF9800',
            'building_purpose': '#FF5722',
            'action_detailed': '#9C27B0',
            'conversational': '#E91E63',
            'beginner_advice': '#795548',
            'casual_slang': '#607D8B',
            'survival_urgency': '#F44336',
            'meme_culture': '#673AB7',
            'meme_humor': '#673AB7',
            'humor_meme': '#673AB7',
            'uncategorized': '#9E9E9E',
        }
        
        # 收集所有数据
        all_embeds = []
        embed_labels = []      # task_id
        embed_types = []       # 'variant' or 'visual'
        embed_categories = []  # 变体类别
        
        for task_id in task_ids:
            # 变体嵌入（带分类）
            variants = variant_embeds_by_task.get(task_id, [])
            for item in variants[:10]:
                if isinstance(item, tuple):
                    embed, cat_id = item
                else:
                    embed, cat_id = item, 'uncategorized'
                all_embeds.append(embed)
                embed_labels.append(task_id)
                embed_types.append('variant')
                embed_categories.append(cat_id)
            
            # 视觉嵌入
            visuals = visual_embeds_by_task.get(task_id, [])
            for v in visuals[:2]:
                all_embeds.append(v)
                embed_labels.append(task_id)
                embed_types.append('visual')
                embed_categories.append('_visual_')
        
        if len(all_embeds) < 3:
            logger.warning("    数据不足，跳过图2")
            return
        
        try:
            all_embeds = np.array(all_embeds)
            perplexity = min(30, max(2, len(all_embeds) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(all_embeds)
            
            # 统计有多少个真正的变体类别
            real_categories = [c for c in set(embed_categories) if c not in ['_visual_', 'uncategorized']]
            
            # === 图2a: 按变体类别着色（如果有分类变体）===
            if real_categories:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 绘制目标视频
                visual_mask = np.array([t == 'visual' for t in embed_types])
                visual_coords = coords[visual_mask]
                if len(visual_coords) > 0:
                    ax.scatter(visual_coords[:, 0], visual_coords[:, 1], marker='*', s=250,
                              c='#1565C0', edgecolors='white', linewidth=1, 
                              label='Target Video', zorder=10)
                
                # 按类别绘制变体
                all_cats = sorted(real_categories) + (['uncategorized'] if 'uncategorized' in embed_categories else [])
                for category in all_cats:
                    mask = np.array([(c == category and t == 'variant') for c, t in zip(embed_categories, embed_types)])
                    cat_coords = coords[mask]
                    
                    if len(cat_coords) == 0:
                        continue
                    
                    color = CATEGORY_COLORS.get(category, PALETTE[hash(category) % len(PALETTE)])
                    display_name = CATEGORY_DISPLAY.get(category, category)
                    
                    ax.scatter(cat_coords[:, 0], cat_coords[:, 1], marker='^', s=80,
                              c=[color], alpha=0.7, label=display_name)
                
                ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
                ax.set_title('Variants vs Target Video (by Category, t-SNE)', fontsize=14, fontweight='bold')
                ax.set_xlabel('t-SNE Dimension 1')
                ax.set_ylabel('t-SNE Dimension 2')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "viz_2a_variants_by_category.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # === 图2b: 按任务着色 ===
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制目标视频（统一颜色）
            visual_mask = np.array([t == 'visual' for t in embed_types])
            visual_coords = coords[visual_mask]
            if len(visual_coords) > 0:
                ax.scatter(visual_coords[:, 0], visual_coords[:, 1], marker='*', s=250,
                          c='#1565C0', edgecolors='white', linewidth=1, 
                          label='Target Video', zorder=10)
            
            # 按任务着色变体
            unique_tasks = sorted(set(l for l, t in zip(embed_labels, embed_types) if t == 'variant'))
            task_colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(unique_tasks)}
            
            for task_id in unique_tasks:
                mask = np.array([(l == task_id and t == 'variant') for l, t in zip(embed_labels, embed_types)])
                task_coords = coords[mask]
                
                if len(task_coords) == 0:
                    continue
                
                color = task_colors[task_id]
                short_name = task_id.split('_')[-1][:8]
                
                ax.scatter(task_coords[:, 0], task_coords[:, 1], marker='^', s=80,
                          c=[color], alpha=0.7, label=short_name)
            
            ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
            ax.set_title('Variants vs Target Video (by Task, t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "viz_2b_variants_by_task.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 复制为默认图（如果有分类图用分类图，否则用任务图）
            import shutil
            if real_categories and (output_dir / "viz_2a_variants_by_category.png").exists():
                shutil.copy(output_dir / "viz_2a_variants_by_category.png", 
                           output_dir / "viz_2_variants_vs_visual.png")
            else:
                shutil.copy(output_dir / "viz_2b_variants_by_task.png", 
                           output_dir / "viz_2_variants_vs_visual.png")
            
        except Exception as e:
            logger.warning(f"    图2生成失败: {e}")
    
    def _plot_prior_vs_visual(
        self,
        prior_embeds: np.ndarray,
        visual_embeds_by_task: Dict[str, List[np.ndarray]],
        task_ids: List[str],
        task_info: Dict,
        output_dir: Path
    ):
        """
        图3: Prior vs 目标视频 (t-SNE)
        生成两张图：按 Tier 和按 Category 着色
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        TIER_COLORS = {1: '#4CAF50', 2: '#FF9800', 3: '#F44336'}
        TIER_NAMES = {1: 'Tier 1', 2: 'Tier 2', 3: 'Tier 3'}
        CAT_COLORS = {'Harvest': '#4CAF50', 'Combat': '#F44336', 'Techtree': '#2196F3', 'Other': '#9E9E9E'}
        
        # 收集所有数据
        all_embeds = []
        embed_labels = []
        embed_types = []
        embed_tiers = []
        embed_cats = []
        
        for i, task_id in enumerate(task_ids):
            info = task_info.get(task_id, {'tier': 2, 'category': 'Other'})
            tier = info['tier']
            cat = info['category']
            
            # Prior 嵌入
            all_embeds.append(prior_embeds[i])
            embed_labels.append(task_id)
            embed_types.append('prior')
            embed_tiers.append(tier)
            embed_cats.append(cat)
            
            # 视觉嵌入
            visuals = visual_embeds_by_task.get(task_id, [])
            for v in visuals[:2]:
                all_embeds.append(v)
                embed_labels.append(task_id)
                embed_types.append('visual')
                embed_tiers.append(tier)
                embed_cats.append(cat)
        
        if len(all_embeds) < 3:
            logger.warning("    数据不足，跳过图3")
            return
        
        try:
            all_embeds = np.array(all_embeds)
            perplexity = min(30, max(2, len(all_embeds) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(all_embeds)
            
            # === 图3a: 按 Tier 着色 ===
            fig, ax = plt.subplots(figsize=(10, 8))
            for tier in sorted(set(embed_tiers)):
                color = TIER_COLORS.get(tier, '#9E9E9E')
                tier_name = TIER_NAMES.get(tier, f'Tier {tier}')
                
                prior_mask = np.array([(t == tier and et == 'prior') for t, et in zip(embed_tiers, embed_types)])
                prior_coords = coords[prior_mask]
                if len(prior_coords) > 0:
                    ax.scatter(prior_coords[:, 0], prior_coords[:, 1], marker='^', s=180,
                              c=[color], edgecolors='black', linewidth=1.5, 
                              label=f'Prior ({tier_name})', zorder=5)
                
                visual_mask = np.array([(t == tier and et == 'visual') for t, et in zip(embed_tiers, embed_types)])
                visual_coords = coords[visual_mask]
                if len(visual_coords) > 0:
                    ax.scatter(visual_coords[:, 0], visual_coords[:, 1], marker='o', s=50,
                              c=[color], alpha=0.4, label=f'Visual ({tier_name})')
            
            ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
            ax.set_title('Prior vs Target Video (by Tier, t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "viz_3a_prior_vs_visual_tier.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # === 图3b: 按 Category 着色 ===
            fig, ax = plt.subplots(figsize=(10, 8))
            for cat in sorted(set(embed_cats)):
                color = CAT_COLORS.get(cat, '#9E9E9E')
                
                prior_mask = np.array([(c == cat and et == 'prior') for c, et in zip(embed_cats, embed_types)])
                prior_coords = coords[prior_mask]
                prior_labels = [l for l, m in zip(embed_labels, prior_mask) if m]
                if len(prior_coords) > 0:
                    ax.scatter(prior_coords[:, 0], prior_coords[:, 1], marker='^', s=180,
                              c=[color], edgecolors='black', linewidth=1.5, 
                              label=f'Prior ({cat})', zorder=5)
                    # 添加标签
                    for j, (pc, label) in enumerate(zip(prior_coords, prior_labels)):
                        short_label = label.split('_')[-1][:5]
                        ax.annotate(short_label, (pc[0], pc[1]), fontsize=6, alpha=0.5,
                                   xytext=(3, 3), textcoords='offset points')
                
                visual_mask = np.array([(c == cat and et == 'visual') for c, et in zip(embed_cats, embed_types)])
                visual_coords = coords[visual_mask]
                if len(visual_coords) > 0:
                    ax.scatter(visual_coords[:, 0], visual_coords[:, 1], marker='o', s=50,
                              c=[color], alpha=0.4, label=f'Visual ({cat})')
            
            ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
            ax.set_title('Prior vs Target Video (by Category, t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "viz_3b_prior_vs_visual_category.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"    图3生成失败: {e}")
    
    def _plot_similarity_matrix(
        self,
        prior_embeds: np.ndarray,
        task_ids: List[str],
        output_dir: Path
    ):
        """
        图2: Prior 输出相似度矩阵
        """
        import matplotlib.pyplot as plt
        from scipy.spatial.distance import cosine
        
        n = len(task_ids)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = 1 - cosine(prior_embeds[i], prior_embeds[j])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
        
        # 设置标签
        short_labels = [t.split('_')[-1][:8] for t in task_ids]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        
        # 添加数值标注
        for i in range(n):
            for j in range(n):
                color = 'white' if sim_matrix[i, j] < 0.75 else 'black'
                ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center', 
                       fontsize=7, color=color)
        
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_title('Prior Output Similarity Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "task_similarity_matrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_variance_distribution(
        self,
        prior_embeds: np.ndarray,
        output_dir: Path
    ):
        """
        图3: Prior 输出方差分布
        """
        import matplotlib.pyplot as plt
        
        # 计算每个维度的方差
        variances = np.var(prior_embeds, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：方差分布直方图
        ax1 = axes[0]
        ax1.hist(variances, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
        ax1.axvline(np.mean(variances), color='red', linestyle='--', label=f'Mean: {np.mean(variances):.4f}')
        ax1.axvline(np.median(variances), color='orange', linestyle='--', label=f'Median: {np.median(variances):.4f}')
        ax1.set_xlabel('Variance')
        ax1.set_ylabel('Dimension Count')
        ax1.set_title('Prior Output Variance Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：方差累积分布
        ax2 = axes[1]
        sorted_vars = np.sort(variances)[::-1]
        cumsum = np.cumsum(sorted_vars)
        cumsum_ratio = cumsum / cumsum[-1]
        ax2.plot(range(len(sorted_vars)), cumsum_ratio, color='#4CAF50', linewidth=2)
        ax2.axhline(0.9, color='red', linestyle='--', alpha=0.7, label='90% Variance')
        ax2.set_xlabel('Dimensions (sorted by variance)')
        ax2.set_ylabel('Cumulative Variance Ratio')
        ax2.set_title('Prior Output Cumulative Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "variance_distribution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_goal_progress_overview(
        self,
        goal_progress_data: Dict,
        output_dir: Path
    ):
        """
        图4: 目标接近度概览
        左图：各任务进度率对比（专家 vs 模型）
        右图：进度率 vs 单调率散点图
        """
        import matplotlib.pyplot as plt
        
        task_ids = list(goal_progress_data.keys())
        n = len(task_ids)
        
        if n == 0:
            return
        
        expert_progress = [goal_progress_data[t].get('expert_progress_rate', 0) for t in task_ids]
        model_progress = [goal_progress_data[t].get('model_progress_rate', 0) for t in task_ids]
        expert_monotonic = [goal_progress_data[t].get('expert_monotonic_rate', 0) for t in task_ids]
        model_monotonic = [goal_progress_data[t].get('model_monotonic_rate', 0) for t in task_ids]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：进度率对比条形图
        ax1 = axes[0]
        x = np.arange(n)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, [p * 100 for p in expert_progress], width, 
                       label='Expert Baseline', color='#2196F3', alpha=0.8)
        bars2 = ax1.bar(x + width/2, [p * 100 for p in model_progress], width,
                       label='Policy', color='#4CAF50', alpha=0.8)
        
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Progress Rate (%)')
        ax1.set_title('Goal Progress Rate Comparison')
        short_labels = [t.split('_')[-1][:10] for t in task_ids]
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 右图：进度率 vs 单调率散点图
        ax2 = axes[1]
        ax2.scatter([p * 100 for p in expert_progress], [m * 100 for m in expert_monotonic],
                   c='#2196F3', s=100, alpha=0.7, label='Expert Baseline', marker='o')
        ax2.scatter([p * 100 for p in model_progress], [m * 100 for m in model_monotonic],
                   c='#4CAF50', s=100, alpha=0.7, label='Policy', marker='^')
        
        # 添加任务标签
        for i, task in enumerate(task_ids):
            short_name = task.split('_')[-1][:6]
            ax2.annotate(short_name, (expert_progress[i] * 100, expert_monotonic[i] * 100),
                        fontsize=7, alpha=0.7)
        
        ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Progress Rate (%)')
        ax2.set_ylabel('Monotonic Rate (%)')
        ax2.set_title('Progress Rate vs Monotonic Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "viz_5_goal_progress_overview.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_action_distribution(
        self,
        expert_dist: Dict[str, float],
        model_dist: Dict[str, float],
        output_dir: Path
    ):
        """
        图5: 动作分布对比
        显示专家和模型的动作类型分布
        """
        import matplotlib.pyplot as plt
        
        # 归一化分布
        expert_total = sum(expert_dist.values()) or 1
        model_total = sum(model_dist.values()) or 1
        
        expert_dist = {k: v / expert_total for k, v in expert_dist.items()}
        model_dist = {k: v / model_total for k, v in model_dist.items()}
        
        # 所有动作类型
        all_actions = sorted(set(expert_dist.keys()) | set(model_dist.keys()))
        
        if not all_actions:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(all_actions))
        width = 0.35
        
        expert_vals = [expert_dist.get(a, 0) for a in all_actions]
        model_vals = [model_dist.get(a, 0) for a in all_actions]
        
        ax.bar(x - width/2, expert_vals, width, label='Expert', color='#2196F3', alpha=0.8)
        ax.bar(x + width/2, model_vals, width, label='Model', color='#FF9800', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(all_actions, rotation=45, ha='right')
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_xlabel('Action Type', fontsize=12)
        ax.set_title('Action Distribution: Expert vs Model', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "viz_6_action_distribution.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_confusion_matrix(
        self,
        expert_actions: List[str],
        model_actions: List[str],
        output_dir: Path
    ):
        """
        图6: 动作混淆矩阵
        显示模型预测与专家动作的匹配情况
        """
        import matplotlib.pyplot as plt
        
        if not expert_actions or not model_actions:
            return
        
        # 确保长度一致
        min_len = min(len(expert_actions), len(model_actions))
        expert_actions = expert_actions[:min_len]
        model_actions = model_actions[:min_len]
        
        # 构建混淆矩阵
        all_types = sorted(set(expert_actions) | set(model_actions))
        n = len(all_types)
        
        if n < 2:
            return
        
        matrix = np.zeros((n, n))
        type_to_idx = {t: i for i, t in enumerate(all_types)}
        
        for e, m in zip(expert_actions, model_actions):
            matrix[type_to_idx[e], type_to_idx[m]] += 1
        
        # 归一化（按行）
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_normalized = np.divide(matrix, row_sums, where=row_sums > 0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix_normalized, cmap='Blues')
        
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(all_types, rotation=45, ha='right')
        ax.set_yticklabels(all_types)
        
        # 添加数值
        for i in range(n):
            for j in range(n):
                val = matrix_normalized[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
        
        ax.set_xlabel('Model Prediction', fontsize=12)
        ax.set_ylabel('Expert Action', fontsize=12)
        ax.set_title('Action Confusion Matrix (Row Normalized)', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Proportion')
        plt.tight_layout()
        
        plt.savefig(output_dir / "viz_7_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_similarity_timeline(
        self,
        frame_similarities: List[float],
        camera_similarities: List[float],
        task_id: str,
        output_dir: Path
    ):
        """
        图7: 逐帧相似度时间线
        显示动作相似度和 Camera 相似度随时间的变化
        """
        import matplotlib.pyplot as plt
        
        n = len(frame_similarities)
        if n < 5:
            return
        
        steps = list(range(n))
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # 1. Action 相似度
        ax1 = axes[0]
        ax1.fill_between(steps, frame_similarities, alpha=0.3, color='#2196F3')
        ax1.plot(steps, frame_similarities, color='#2196F3', linewidth=1.5, label='Action Similarity')
        
        # 添加移动平均线
        window = min(20, n // 5) if n > 20 else 5
        if n > window:
            action_ma = np.convolve(frame_similarities, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, n), action_ma, color='#1565C0', linewidth=2,
                    linestyle='--', label=f'Moving Avg ({window})')
        
        ax1.set_ylabel('Action Similarity', fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f'Frame-wise Similarity Timeline: {task_id}', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 计算并显示平均值
        avg_action = np.mean(frame_similarities)
        ax1.axhline(y=avg_action, color='#F44336', linestyle=':', linewidth=1.5, alpha=0.8)
        ax1.text(n * 0.02, avg_action + 0.03, f'Avg: {avg_action:.1%}', color='#F44336', fontsize=10)
        
        # 2. Camera 相似度
        ax2 = axes[1]
        if camera_similarities:
            camera_n = len(camera_similarities)
            ax2.fill_between(range(camera_n), camera_similarities, alpha=0.3, color='#4CAF50')
            ax2.plot(range(camera_n), camera_similarities, color='#4CAF50', linewidth=1.5, label='Camera Similarity')
            
            # 添加移动平均线
            if camera_n > window:
                camera_ma = np.convolve(camera_similarities, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, camera_n), camera_ma, color='#2E7D32', linewidth=2,
                        linestyle='--', label=f'Moving Avg ({window})')
            
            avg_camera = np.mean(camera_similarities)
            ax2.axhline(y=avg_camera, color='#F44336', linestyle=':', linewidth=1.5, alpha=0.8)
            ax2.text(camera_n * 0.02, avg_camera + 0.03, f'Avg: {avg_camera:.1%}', color='#F44336', fontsize=10)
        
        ax2.set_ylabel('Camera Similarity', fontsize=11)
        ax2.set_xlabel('Frame Index', fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"viz_8_similarity_timeline_{task_id}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_goal_progress_comparison(
        self,
        expert_distances: List[float],
        model_distances: List[float],
        task_id: str,
        output_dir: Path
    ):
        """
        图8: 目标接近度对比图
        对比专家和模型的目标距离变化
        """
        import matplotlib.pyplot as plt
        
        if not expert_distances or not model_distances:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 专家距离
        expert_n = len(expert_distances)
        expert_steps = np.linspace(0, 100, expert_n)  # 归一化到 0-100%
        ax.plot(expert_steps, expert_distances, color='#2196F3', linewidth=2, 
               label='Expert Baseline', alpha=0.9)
        ax.fill_between(expert_steps, expert_distances, alpha=0.15, color='#2196F3')
        
        # 模型距离
        model_n = len(model_distances)
        model_steps = np.linspace(0, 100, model_n)  # 归一化到 0-100%
        ax.plot(model_steps, model_distances, color='#4CAF50', linewidth=2,
               label='Policy Model', alpha=0.9)
        ax.fill_between(model_steps, model_distances, alpha=0.15, color='#4CAF50')
        
        # 标记起点和终点
        ax.scatter([0], [expert_distances[0]], color='#2196F3', s=80, marker='s', zorder=5, label='Expert Start')
        ax.scatter([100], [expert_distances[-1]], color='#2196F3', s=80, marker='D', zorder=5, label='Expert End')
        ax.scatter([0], [model_distances[0]], color='#4CAF50', s=80, marker='s', zorder=5, label='Model Start')
        ax.scatter([100], [model_distances[-1]], color='#4CAF50', s=80, marker='D', zorder=5, label='Model End')
        
        # 计算进度率
        expert_progress = (expert_distances[0] - expert_distances[-1]) / expert_distances[0] if expert_distances[0] > 1e-6 else 0
        model_progress = (model_distances[0] - model_distances[-1]) / model_distances[0] if model_distances[0] > 1e-6 else 0
        
        # 添加信息框
        info_text = f'Expert Progress: {expert_progress:+.1%}\nModel Progress: {model_progress:+.1%}'
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
               fontsize=11, fontweight='bold',
               ha='right', va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ddd'))
        
        ax.set_xlabel('Progress (%)', fontsize=11)
        ax.set_ylabel('Distance to Goal\n(lower is better)', fontsize=11)
        ax.set_title(f'Goal Progress Comparison: {task_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"viz_9_goal_comparison_{task_id}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_similarity_timeline_aggregated(
        self,
        all_frame_similarities: Dict[str, List[float]],
        all_camera_similarities: Dict[str, List[float]],
        output_dir: Path
    ):
        """
        图7: 逐帧相似度时间线（所有任务汇总）
        显示所有任务所有 trial 的平均相似度趋势
        """
        import matplotlib.pyplot as plt
        
        # 收集所有任务的数据长度
        all_lengths = [len(sims) for sims in all_frame_similarities.values()]
        if not all_lengths or max(all_lengths) < 5:
            return
        
        # 归一化到相同长度（100个点）进行平均
        NORM_LEN = 100
        
        def normalize_and_average(all_sims_dict: Dict[str, List[float]]) -> np.ndarray:
            """将不同长度的序列归一化到相同长度后平均"""
            normalized_all = []
            for task_id, sims in all_sims_dict.items():
                if len(sims) < 2:
                    continue
                # 插值到 NORM_LEN 个点
                x_old = np.linspace(0, 1, len(sims))
                x_new = np.linspace(0, 1, NORM_LEN)
                normalized = np.interp(x_new, x_old, sims)
                normalized_all.append(normalized)
            
            if not normalized_all:
                return np.array([])
            return np.mean(normalized_all, axis=0)
        
        # 计算平均趋势
        avg_action_sim = normalize_and_average(all_frame_similarities)
        avg_camera_sim = normalize_and_average(all_camera_similarities) if all_camera_similarities else np.array([])
        
        if len(avg_action_sim) < 5:
            return
        
        n_tasks = len(all_frame_similarities)
        total_frames = sum(len(sims) for sims in all_frame_similarities.values())
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        steps = np.arange(NORM_LEN)
        
        # 1. Action 相似度
        ax1 = axes[0]
        ax1.fill_between(steps, avg_action_sim, alpha=0.3, color='#2196F3')
        ax1.plot(steps, avg_action_sim, color='#2196F3', linewidth=2, label='Avg Action Similarity')
        
        # 添加移动平均线
        window = 10
        action_ma = np.convolve(avg_action_sim, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, NORM_LEN), action_ma, color='#1565C0', linewidth=2,
                linestyle='--', label=f'Moving Avg ({window})')
        
        ax1.set_ylabel('Action Similarity', fontsize=11)
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f'Frame-wise Similarity Timeline (All Tasks Aggregated: {n_tasks} tasks, {total_frames} frames)', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 显示整体平均值
        overall_avg = np.mean(avg_action_sim)
        ax1.axhline(y=overall_avg, color='#F44336', linestyle=':', linewidth=1.5, alpha=0.8)
        ax1.text(NORM_LEN * 0.02, overall_avg + 0.03, f'Overall Avg: {overall_avg:.1%}', 
                color='#F44336', fontsize=10)
        
        # 2. Camera 相似度
        ax2 = axes[1]
        if len(avg_camera_sim) > 0:
            ax2.fill_between(steps, avg_camera_sim, alpha=0.3, color='#4CAF50')
            ax2.plot(steps, avg_camera_sim, color='#4CAF50', linewidth=2, label='Avg Camera Similarity')
            
            camera_ma = np.convolve(avg_camera_sim, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, NORM_LEN), camera_ma, color='#2E7D32', linewidth=2,
                    linestyle='--', label=f'Moving Avg ({window})')
            
            camera_avg = np.mean(avg_camera_sim)
            ax2.axhline(y=camera_avg, color='#F44336', linestyle=':', linewidth=1.5, alpha=0.8)
            ax2.text(NORM_LEN * 0.02, camera_avg + 0.03, f'Overall Avg: {camera_avg:.1%}', 
                    color='#F44336', fontsize=10)
        
        ax2.set_ylabel('Camera Similarity', fontsize=11)
        ax2.set_xlabel('Normalized Progress (%)', fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "viz_8_similarity_timeline_aggregated.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_goal_progress_comparison_aggregated(
        self,
        all_expert_distances: Dict[str, List[float]],
        all_model_distances: Dict[str, List[float]],
        output_dir: Path
    ):
        """
        图8: 目标接近度对比图（所有任务汇总）
        对比所有任务的专家和模型平均目标距离变化
        """
        import matplotlib.pyplot as plt
        
        # 找到共同的任务
        common_tasks = set(all_expert_distances.keys()) & set(all_model_distances.keys())
        if not common_tasks:
            return
        
        # 归一化到相同长度（100个点）
        NORM_LEN = 100
        
        def normalize_to_length(distances: List[float], target_len: int) -> np.ndarray:
            """将序列插值到目标长度"""
            if len(distances) < 2:
                return np.array([distances[0]] * target_len if distances else [0] * target_len)
            x_old = np.linspace(0, 1, len(distances))
            x_new = np.linspace(0, 1, target_len)
            return np.interp(x_new, x_old, distances)
        
        # 收集并归一化所有数据
        expert_normalized = []
        model_normalized = []
        
        for task_id in common_tasks:
            expert_norm = normalize_to_length(all_expert_distances[task_id], NORM_LEN)
            model_norm = normalize_to_length(all_model_distances[task_id], NORM_LEN)
            expert_normalized.append(expert_norm)
            model_normalized.append(model_norm)
        
        # 计算平均值和标准差
        expert_mean = np.mean(expert_normalized, axis=0)
        expert_std = np.std(expert_normalized, axis=0)
        model_mean = np.mean(model_normalized, axis=0)
        model_std = np.std(model_normalized, axis=0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        progress = np.linspace(0, 100, NORM_LEN)
        
        # 专家平均距离（带置信区间）
        ax.plot(progress, expert_mean, color='#2196F3', linewidth=2.5, 
               label=f'Expert Baseline (n={len(common_tasks)})', alpha=0.9)
        ax.fill_between(progress, expert_mean - expert_std, expert_mean + expert_std,
                       alpha=0.15, color='#2196F3')
        
        # 模型平均距离（带置信区间）
        ax.plot(progress, model_mean, color='#4CAF50', linewidth=2.5,
               label=f'Policy Model (n={len(common_tasks)})', alpha=0.9)
        ax.fill_between(progress, model_mean - model_std, model_mean + model_std,
                       alpha=0.15, color='#4CAF50')
        
        # 标记起点和终点
        ax.scatter([0], [expert_mean[0]], color='#2196F3', s=100, marker='s', zorder=5)
        ax.scatter([100], [expert_mean[-1]], color='#2196F3', s=100, marker='D', zorder=5)
        ax.scatter([0], [model_mean[0]], color='#4CAF50', s=100, marker='s', zorder=5)
        ax.scatter([100], [model_mean[-1]], color='#4CAF50', s=100, marker='D', zorder=5)
        
        # 计算平均进度率
        expert_progress = (expert_mean[0] - expert_mean[-1]) / expert_mean[0] if expert_mean[0] > 1e-6 else 0
        model_progress = (model_mean[0] - model_mean[-1]) / model_mean[0] if model_mean[0] > 1e-6 else 0
        
        # 添加信息框
        info_text = (f'Avg Expert Progress: {expert_progress:+.1%}\n'
                    f'Avg Model Progress: {model_progress:+.1%}\n'
                    f'Tasks: {len(common_tasks)}')
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
               fontsize=11, fontweight='bold',
               ha='right', va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#ddd'))
        
        ax.set_xlabel('Progress (%)', fontsize=11)
        ax.set_ylabel('Distance to Goal\n(lower is better)', fontsize=11)
        ax.set_title(f'Goal Progress Comparison (All Tasks Aggregated)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(output_dir / "viz_9_goal_comparison_aggregated.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _collect_combined_results(self, output_dir: Path) -> Dict[str, Dict]:
        """
        从任务输出目录收集综合评估结果
        
        支持两种目录结构：
        1. 单任务: output_dir/combined_evaluation.json
        2. 多任务: output_dir/{task_dir}/combined_evaluation.json
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict[task_id, combined_result]
        """
        combined_results = {}
        
        # 首先检查根目录是否有 combined_evaluation.json（单任务评估）
        root_combined_file = output_dir / "combined_evaluation.json"
        if root_combined_file.exists():
            try:
                with open(root_combined_file, 'r', encoding='utf-8') as f:
                    combined = json.load(f)
                    task_id = combined.get('task_id', output_dir.name.split('_')[0])
                    combined_results[task_id] = combined
                    logger.debug(f"  加载 {task_id} 的综合评估结果 (根目录)")
            except Exception as e:
                logger.warning(f"  加载 {root_combined_file} 失败: {e}")
        
        # 查找所有子任务目录（多任务评估）
        for task_dir in output_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            # 跳过 checkpoints 等非任务目录
            if task_dir.name in ['checkpoints', '.DS_Store']:
                continue
            
            combined_file = task_dir / "combined_evaluation.json"
            if combined_file.exists():
                try:
                    with open(combined_file, 'r', encoding='utf-8') as f:
                        combined = json.load(f)
                        task_id = combined.get('task_id', task_dir.name.split('_')[0])
                        combined_results[task_id] = combined
                        logger.info(f"  加载 {task_id} 的综合评估结果: action={combined.get('action_similarity_metrics', {}).get('enabled')}, goal={combined.get('goal_progress_metrics', {}).get('enabled')}")
                except Exception as e:
                    logger.warning(f"  加载 {combined_file} 失败: {e}")
        
        logger.info(f"收集到 {len(combined_results)} 个任务的综合评估结果")
        return combined_results
    
    def _compute_goal_progress_summary(self, goal_progress_data: Dict) -> Dict:
        """
        计算 goal_progress 数据的汇总统计
        
        Args:
            goal_progress_data: 每个任务的 goal_progress 数据
            
        Returns:
            汇总统计字典
        """
        if not goal_progress_data:
            return {
                'avg_expert_progress_rate': 0,
                'avg_expert_monotonic_rate': 0,
                'avg_model_progress_rate': 0,
                'avg_model_monotonic_rate': 0,
                'avg_action_similarity': 0,
                'n_tasks': 0
            }
        
        expert_progress_rates = []
        expert_monotonic_rates = []
        model_progress_rates = []
        model_monotonic_rates = []
        action_similarities = []
        
        for task_id, data in goal_progress_data.items():
            # 专家指标
            if data.get('expert_progress_rate', 0) != 0 or data.get('expert_monotonic_rate', 0) != 0:
                expert_progress_rates.append(data.get('expert_progress_rate', 0))
                expert_monotonic_rates.append(data.get('expert_monotonic_rate', 0))
            
            # 模型指标
            if data.get('model_progress_rate', 0) != 0 or data.get('model_monotonic_rate', 0) != 0:
                model_progress_rates.append(data.get('model_progress_rate', 0))
                model_monotonic_rates.append(data.get('model_monotonic_rate', 0))
            
            # 动作相似度
            action_sim = data.get('action_similarity', 0)
            if action_sim > 0:
                action_similarities.append(action_sim)
        
        # 计算辅助指标
        camera_similarities = []
        action_entropies = []
        temporal_smoothnesses = []
        action_coverages = []
        
        for task_id, data in goal_progress_data.items():
            if data.get('camera_similarity', 0) > 0:
                camera_similarities.append(data.get('camera_similarity', 0))
            if data.get('action_entropy', 0) > 0:
                action_entropies.append(data.get('action_entropy', 0))
            if data.get('temporal_smoothness', 0) > 0:
                temporal_smoothnesses.append(data.get('temporal_smoothness', 0))
            if data.get('action_coverage', 0) > 0:
                action_coverages.append(data.get('action_coverage', 0))
        
        return {
            'avg_expert_progress_rate': np.mean(expert_progress_rates) if expert_progress_rates else 0,
            'avg_expert_monotonic_rate': np.mean(expert_monotonic_rates) if expert_monotonic_rates else 0,
            'avg_model_progress_rate': np.mean(model_progress_rates) if model_progress_rates else 0,
            'avg_model_monotonic_rate': np.mean(model_monotonic_rates) if model_monotonic_rates else 0,
            'avg_action_similarity': np.mean(action_similarities) if action_similarities else 0,
            # 辅助指标
            'avg_camera_similarity': np.mean(camera_similarities) if camera_similarities else 0,
            'avg_action_entropy': np.mean(action_entropies) if action_entropies else 0,
            'avg_temporal_smoothness': np.mean(temporal_smoothnesses) if temporal_smoothnesses else 0,
            'avg_action_coverage': np.mean(action_coverages) if action_coverages else 0,
            'n_tasks': len(goal_progress_data)
        }
    
    def _compute_auxiliary_metrics(
        self,
        combined_results: Dict[str, Dict],
        prior_gain_data: Dict,
        goal_progress_data: Dict
    ) -> Dict:
        """
        计算辅助指标
        
        Args:
            combined_results: 综合评估结果
            prior_gain_data: Prior 增益数据
            goal_progress_data: Goal Progress 数据
            
        Returns:
            辅助指标字典
        """
        # Prior 辅助指标
        prior_goal_accuracies = []
        prior_consistencies = []
        
        for task_id, combined in combined_results.items():
            prior_metrics = combined.get('prior_metrics', {})
            if prior_metrics.get('enabled'):
                prior_goal_accuracies.append(prior_metrics.get('goal_accuracy', 0))
                prior_consistencies.append(prior_metrics.get('consistency', 0))
        
        # 计算 Prior 输出的均值方差（跨任务）
        prior_mean_variance = 0.0
        if prior_goal_accuracies:
            prior_mean_variance = float(np.var(prior_goal_accuracies))
        
        # 计算 Prior 区分度（1 - 平均相似度）
        # 这里简化计算：使用 goal_accuracy 的标准差作为区分度的代理
        prior_discriminability = 0.0
        if len(prior_goal_accuracies) > 1:
            # 区分度：不同任务之间 goal_accuracy 的差异程度
            prior_discriminability = float(np.std(prior_goal_accuracies))
        
        # 构建每个任务的辅助指标
        task_auxiliary = {}
        for task_id, combined in combined_results.items():
            prior_metrics = combined.get('prior_metrics', {})
            action_metrics = combined.get('action_similarity_metrics', {})
            policy_result = combined.get('policy_result', {})
            
            task_auxiliary[task_id] = {
                # Prior 辅助指标
                'prior_variant_alignment': prior_metrics.get('semantic_robustness', 0) or 0,  # Prior 变体
                'prior_discriminability': prior_discriminability,  # 全局区分度（所有任务共享）
                'prior_goal_accuracy_std': prior_metrics.get('goal_accuracy_std', 0),
                'prior_goal_accuracy': prior_metrics.get('goal_accuracy', 0),
                'prior_consistency': prior_metrics.get('consistency', 0),
                # Policy 辅助指标
                'action_similarity': action_metrics.get('action_similarity', 0),
                'camera_similarity': action_metrics.get('camera_similarity', 0),
                'action_entropy': action_metrics.get('action_entropy', 0),
                'temporal_smoothness': action_metrics.get('temporal_smoothness', 0),
                'action_coverage': action_metrics.get('action_coverage', 0),
                # 成功率
                'success_rate': policy_result.get('success_rate', 0)
            }
        
        # Policy 辅助指标汇总
        goal_summary = self._compute_goal_progress_summary(goal_progress_data)
        
        # 计算平均 Prior 变体对齐度
        prior_variant_alignments = [
            combined.get('prior_metrics', {}).get('semantic_robustness', 0) or 0
            for combined in combined_results.values()
            if combined.get('prior_metrics', {}).get('semantic_robustness') is not None
        ]
        avg_prior_variant_alignment = float(np.mean(prior_variant_alignments)) if prior_variant_alignments else 0
        
        # 计算每个任务 goal_accuracy_std 的平均值
        prior_goal_accuracy_stds = [
            combined.get('prior_metrics', {}).get('goal_accuracy_std', 0)
            for combined in combined_results.values()
            if combined.get('prior_metrics', {}).get('enabled', False)
        ]
        avg_prior_goal_accuracy_std = float(np.mean(prior_goal_accuracy_stds)) if prior_goal_accuracy_stds else 0
        
        return {
            # Prior 辅助指标汇总
            'prior_mean_variance': avg_prior_goal_accuracy_std,  # 每个任务输出方差的平均
            'avg_prior_discriminability': prior_discriminability,  # 添加 avg_ 前缀
            'prior_discriminability': prior_discriminability,  # 保持兼容
            'avg_prior_variant_alignment': avg_prior_variant_alignment,
            'avg_prior_consistency': np.mean(prior_consistencies) if prior_consistencies else 0,
            
            # Policy 辅助指标汇总（从 goal_summary 获取）
            'avg_action_similarity': goal_summary.get('avg_action_similarity', 0),
            'avg_camera_similarity': goal_summary.get('avg_camera_similarity', 0),
            'avg_action_entropy': goal_summary.get('avg_action_entropy', 0),
            'avg_temporal_smoothness': goal_summary.get('avg_temporal_smoothness', 0),
            'avg_action_coverage': goal_summary.get('avg_action_coverage', 0),
            
            # 每个任务的辅助指标
            'task_auxiliary': task_auxiliary
        }
    
    def _build_report_data(
        self, 
        results: List[TaskResult],
        combined_results: Dict[str, Dict]
    ) -> Dict:
        """
        构建 PriorHTMLGenerator 兼容的报告数据结构
        
        Args:
            results: TaskResult 列表
            combined_results: 综合评估结果
            
        Returns:
            Dict 报告数据
        """
        config_filename = Path(self.config.task_config_path).name
        task_ids = [r.task_id for r in results]
        
        # 构建 output_quality 数据（Prior 指标）
        # 字段名要与 HTML 生成器匹配: alignment_text, alignment_prior
        prior_gain_data = {}
        for task_id, combined in combined_results.items():
            prior_metrics = combined.get('prior_metrics', {})
            if prior_metrics.get('enabled'):
                goal_accuracy = prior_metrics.get('goal_accuracy', 0)
                mineclip_baseline = prior_metrics.get('mineclip_baseline', 0)
                prior_gain_data[task_id] = {
                    'alignment_text': mineclip_baseline,  # MineCLIP 基线
                    'alignment_prior': goal_accuracy,  # Prior 输出与目标嵌入的相似度
                    'prior_sim': goal_accuracy,  # 兼容字段
                    'baseline_sim': mineclip_baseline,
                    'prior_gain': goal_accuracy - mineclip_baseline,  # Prior 增益
                }
        
        # 构建 intrinsic_quality 数据（一致性、鲁棒性）
        # 字段名要与 HTML 生成器匹配
        consistency_data = {}
        robustness_data = {}
        for task_id, combined in combined_results.items():
            prior_metrics = combined.get('prior_metrics', {})
            if prior_metrics.get('enabled'):
                consistency_data[task_id] = prior_metrics.get('consistency', 0)
                semantic_robustness = prior_metrics.get('semantic_robustness')
                if semantic_robustness is not None:
                    goal_accuracy = prior_metrics.get('goal_accuracy', 0)
                    robustness_data[task_id] = {
                        'robustness': semantic_robustness,
                        'n_variants': prior_metrics.get('n_variants', 0),
                        # 变体对齐度 = 主指令相似度 * 鲁棒性
                        'variant_alignment': goal_accuracy * semantic_robustness
                    }
        
        # 构建 goal_progress 数据（Policy 指标）
        goal_progress_data = {}
        for task_id, combined in combined_results.items():
            action_metrics = combined.get('action_similarity_metrics', {})
            goal_metrics = combined.get('goal_progress_metrics', {})
            
            if action_metrics.get('enabled') or goal_metrics.get('enabled'):
                goal_progress_data[task_id] = {
                    # 专家基线
                    'expert_progress_rate': action_metrics.get('expert_progress_rate', 0),
                    'expert_monotonic_rate': action_metrics.get('expert_monotonic_rate', 0),
                    'expert_initial_distance': action_metrics.get('expert_initial_distance', 0),
                    'expert_final_distance': action_metrics.get('expert_final_distance', 0),
                    # 模型接近度
                    'model_progress_rate': goal_metrics.get('model_progress_rate', 0),
                    'model_monotonic_rate': goal_metrics.get('model_monotonic_rate', 0),
                    'model_initial_distance': goal_metrics.get('model_initial_distance', 0),
                    'model_final_distance': goal_metrics.get('model_final_distance', 0),
                    # 动作相似度
                    'action_similarity': action_metrics.get('action_similarity', 0),
                    # 额外的 Policy 辅助指标
                    'camera_similarity': action_metrics.get('camera_similarity', 0),
                    'action_entropy': action_metrics.get('action_entropy', 0),
                    'temporal_smoothness': action_metrics.get('temporal_smoothness', 0),
                    'action_coverage': action_metrics.get('action_coverage', 0),
                }
        
        logger.debug(f"构建 goal_progress_data: {len(goal_progress_data)} 个任务")
        
        # 构建 task_info
        task_info = {}
        # difficulty 到 tier 的映射
        difficulty_to_tier = {'easy': 1, 'medium': 2, 'hard': 3, 'extreme': 4}
        
        for result in results:
            task_config = self.task_loader.get_task(result.task_id)
            if task_config:
                # 从 difficulty 推断 tier
                difficulty = task_config.get('difficulty', 'easy')
                tier = difficulty_to_tier.get(difficulty, 1)
                
                # 计算变体数量（instruction_variants 是嵌套字典）
                variants_config = task_config.get('instruction_variants', {})
                n_variants = 0
                n_variant_categories = 0
                if isinstance(variants_config, dict):
                    n_variant_categories = len(variants_config)
                    for cat_data in variants_config.values():
                        if isinstance(cat_data, dict):
                            n_variants += len(cat_data.get('variants', []))
                
                task_info[result.task_id] = {
                    'tier': tier,
                    'difficulty': difficulty,
                    'category': task_config.get('category', 'unknown'),
                    'n_variant_categories': n_variant_categories,
                    'n_variants': n_variants,
                }
            else:
                task_info[result.task_id] = {
                    'tier': 1,
                    'difficulty': 'unknown',
                    'category': 'unknown',
                    'n_variant_categories': 0,
                    'n_variants': 0,
                }
        
        # 构建完整报告数据
        report_data = {
            'config_file': config_filename,
            'n_tasks': len(results),
            'task_ids': task_ids,
            
            # 内在质量维度
            'intrinsic_quality': {
                'dimension_name': '内在质量',
                'enabled': bool(consistency_data),
                'metrics': {
                    'consistency': {
                        'task_consistency': consistency_data,
                        'n_samples': self.config.prior_n_samples
                    },
                    'semantic_robustness': {
                        'task_robustness': robustness_data
                    }
                },
                'visualizations': {}
            },
            
            # 输出质量维度
            'output_quality': {
                'dimension_name': '输出质量',
                'enabled': bool(prior_gain_data),
                'metrics': {
                    'prior_gain': {
                        'task_gains': prior_gain_data
                    }
                },
                'visualizations': {}
            },
            
            # 可控性维度（暂不使用）
            'controllability': None,
            
            # 任务级结果（包含 goal_progress）
            'task_results': {
                'goal_progress': {
                    'enabled': bool(goal_progress_data),
                    'task_progress': goal_progress_data,
                    'n_tasks_with_data': len(goal_progress_data)
                }
            },
            
            # 任务信息
            'task_info': task_info,
            
            # 总结
            'summary': {
                'total_tasks': len(results),
                'avg_success_rate': np.mean([r.success_rate for r in results]) if results else 0,
                'avg_prior_accuracy': np.mean([d.get('prior_sim', 0) for d in prior_gain_data.values()]) if prior_gain_data else 0,
                # goal_progress_summary for Average row in HTML
                'goal_progress_summary': self._compute_goal_progress_summary(goal_progress_data)
            },
            
            # 辅助指标（Prior 和 Policy）
            'auxiliary_metrics': self._compute_auxiliary_metrics(combined_results, prior_gain_data, goal_progress_data)
        }
        
        return report_data
    
    def _generate_text_report(self, report_data: Dict[str, Any], output_path: Path):
        """生成人类可读的文本报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("STEVE-1 评估报告\n")
            f.write("="*80 + "\n\n")
            
            # 元数据
            f.write(f"生成时间: {report_data['metadata']['timestamp']}\n")
            f.write(f"任务数量: {report_data['metadata']['total_tasks']}\n")
            f.write(f"评估框架: {report_data['metadata']['framework']}\n\n")
            
            # 总体统计
            summary = report_data['summary']
            f.write("总体统计:\n")
            f.write(f"  总成功率: {summary['overall_success_rate']:.1f}%\n")
            f.write(f"  总试验数: {summary['total_trials']}\n")
            f.write(f"  成功试验数: {summary['successful_trials']}\n\n")
            
            # ===== 添加表格汇总 =====
            f.write("="*80 + "\n")
            f.write("评估结果汇总\n")
            f.write("="*80 + "\n\n")
            
            # 表头
            f.write(f"{'任务ID':<30} {'指令':<20} {'成功率':<10} {'平均步数':<12} {'平均时间'}\n")
            f.write("-" * 80 + "\n")
            
            # 每个任务的汇总
            for task in report_data['tasks']:
                task_id = task['task_id'][:28]  # 截断过长的ID
                instruction = (task['instruction'][:18] if task['instruction'] else "N/A")
                success_rate = f"{task['success_rate']:.1f}%"
                avg_steps = f"{task['avg_steps']:.1f}"
                avg_time = f"{task['avg_time']:.1f}s"
                
                f.write(f"{task_id:<30} {instruction:<20} {success_rate:<10} {avg_steps:<12} {avg_time}\n")
            
            # 总体统计行
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"{'总体统计':<30} {'N/A':<20} {summary['overall_success_rate']:.1f}% ")
            
            # 计算平均步数和时间
            avg_steps_all = sum(task['avg_steps'] for task in report_data['tasks']) / len(report_data['tasks'])
            avg_time_all = sum(task['avg_time'] for task in report_data['tasks']) / len(report_data['tasks'])
            f.write(f"{avg_steps_all:<12.1f} {avg_time_all:.1f}s\n")
            
            f.write(f"\n总任务数: {report_data['metadata']['total_tasks']}\n")
            f.write(f"总试验数: {summary['total_trials']}\n")
            f.write("="*80 + "\n\n")
            
            # ===== 详细任务信息 =====
            f.write("="*80 + "\n")
            f.write("任务详情\n")
            f.write("="*80 + "\n\n")
            
            for task in report_data['tasks']:
                f.write(f"任务: {task['task_id']}\n")
                f.write(f"  指令: {task['instruction']}\n")
                f.write(f"  语言: {task['language']}\n")
                f.write(f"  成功率: {task['success_rate']:.1f}%\n")
                f.write(f"  平均步数: {task['avg_steps']:.0f}\n")
                f.write(f"  平均时间: {task['avg_time']:.1f}s\n")
                f.write(f"  试验详情:\n")
                for i, trial in enumerate(task['trials'], 1):
                    status = "✅ 成功" if trial['success'] else "❌ 失败"
                    f.write(f"    Trial {i}: {status} | 步数: {trial['steps']:4d} | 时间: {trial['time_seconds']:.1f}s")
                    
                    # 添加最终库存信息
                    if 'final_inventory' in trial and trial['final_inventory']:
                        inventory_items = [f"{item}×{count}" for item, count in trial['final_inventory'].items()]
                        f.write(f" | 库存: {', '.join(inventory_items)}")
                    
                    f.write("\n")
                f.write("\n")
    
    def close(self):
        """清理所有资源"""
        # 关闭共享的评估器（释放所有模型）
        if self._shared_evaluator:
            self._shared_evaluator.close()
            self._shared_evaluator = None
        
        # 向后兼容：如果有独立的 evaluator
        if self.evaluator:
            self.evaluator.close()
        
        logger.info("评估框架已关闭")


# 命令行接口
if __name__ == "__main__":
    import argparse
    import warnings
    from src.utils.logging_config import setup_evaluation_logging
    
    # 配置日志（使用统一的格式和过滤器）
    setup_evaluation_logging()
    
    # 过滤不必要的警告信息（在框架初始化之前）
    # 1. PyTorch 警告
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', message='.*CUDA is not available.*')
    warnings.filterwarnings('ignore', message='.*Implicit dimension choice for softmax.*')
    warnings.filterwarnings('ignore', message='.*has_cuda.*')
    
    # 2. 完全静默 MineRL/Malmo 日志（包括 ERROR）
    minerl_loggers = [
        'minerl.env.malmo.instance',
        'minerl.env._multiagent',
        'minerl.env.malmo',
        'process_watcher',
    ]
    for logger_name in minerl_loggers:
        minerl_logger = logging.getLogger(logger_name)
        minerl_logger.setLevel(logging.CRITICAL + 1)  # 完全静默
        minerl_logger.propagate = False  # 不传播到父 logger
    
    # 3. STEVE-1 警告
    warnings.filterwarnings('ignore', category=UserWarning, module='steve1')
    
    parser = argparse.ArgumentParser(description='STEVE-1 评估框架')
    parser.add_argument(
        '--config',
        type=str,
        default='config/eval_tasks.yaml',
        help='任务配置文件路径（默认: config/eval_tasks.yaml）'
    )
    parser.add_argument(
        '--task',
        type=str,
        help='评估单个任务（任务ID）'
    )
    parser.add_argument(
        '--task-set',
        type=str,
        help='评估任务集（如 harvest_tasks, quick_test, baseline_test）'
    )
    parser.add_argument(
        '--task-list',
        type=str,
        nargs='+',
        help='评估任务列表（多个任务ID）'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=None,  # None 表示使用配置文件的值
        help='每个任务的试验次数（默认使用配置文件中的值）'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,  # None 表示使用配置文件的值
        help='每个试验的最大步数（默认使用配置文件中的值）'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='启用游戏窗口渲染（显示画面）'
    )
    parser.add_argument(
        '--enable_video',
        action='store_true',
        help='启用视频录制（固定尺寸 640x360）'
    )
    parser.add_argument(
        '--enable_report',
        action='store_true',
        help='生成 HTML 报告'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认: data/evaluation/）'
    )
    
    args = parser.parse_args()
    
    # 视频录制：如果启用，使用固定尺寸 640x360
    video_size = (640, 360) if args.enable_video else None
    
    # 创建配置
    config = EvaluationConfig(
        task_config_path=args.config,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        enable_render=args.render,
        enable_report=args.enable_report,
        video_size=video_size,
        output_dir=args.output_dir  # 传递输出目录参数
    )
    
    # 创建评估框架
    framework = EvaluationFramework(config=config)
    
    try:
        results = []
        
        # 根据参数选择评估模式
        if args.task:
            # 单个任务（传递命令行参数，确保优先级）
            result, _ = framework.evaluate_single_task(
                args.task,
                n_trials=config.n_trials,
                max_steps=config.max_steps
            )
            results = [result]
        
        elif args.task_set:
            # 任务集（传递命令行参数，确保优先级）
            results = framework.evaluate_task_set(
                args.task_set,
                n_trials=config.n_trials,
                max_steps=config.max_steps
            )
        
        elif args.task_list:
            # 任务列表（传递命令行参数，确保优先级）
            results = framework.evaluate_task_list(
                args.task_list,
                n_trials=config.n_trials,
                max_steps=config.max_steps
            )
        
        else:
            # 默认：快速测试（传递命令行参数，确保优先级）
            logger.info("未指定任务，运行快速测试...")
            results = framework.evaluate_task_set(
                'quick_test',
                n_trials=config.n_trials,
                max_steps=config.max_steps
            )
        
        # 打印摘要
        framework.print_summary(results)
        
        # 生成报告
        framework.generate_report(results)
        
        # 重置 task-set 目录（避免影响后续评估）
        framework.current_task_set_dir = None
        
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        framework.close()
