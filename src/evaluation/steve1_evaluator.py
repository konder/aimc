"""
STEVE-1 评估器 (基于 MineRL 环境)
使用完整 pip 安装的 steve1 包
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import torch as th
import numpy as np
import cv2
from tqdm import tqdm

# 导入本地版本的工具函数（支持自定义环境）
from src.utils.steve1_mineclip_agent_env_utils import (
    load_mineclip_agent_env,
    load_mineclip_wconfig,
    load_vae_model,
    make_env  # 添加 make_env 导入
)
from src.utils.device import DEVICE

# 导入 steve1 官方工具
from steve1.utils.embed_utils import get_prior_embed
from steve1.config import PRIOR_INFO

from .metrics import TrialResult, TaskResult
from ..translation.translator import ChineseTranslator

logger = logging.getLogger(__name__)


class STEVE1Evaluator:
    """
    STEVE-1 评估器（执行器/Worker）
    
    职责:
    - 加载和管理 STEVE-1 模型、MineCLIP、Prior 和环境
    - 集成中文翻译器（自动检测和翻译中文指令）
    - 执行单个任务评估（run trials）
    - 返回任务结果（TaskResult）
    
    特性:
    - 使用官方 steve1 包 (pip install -e)
    - 基于 MineRL 环境（支持自定义环境）
    - 自动中文→英文翻译
    
    注意：
    - 报告生成由 EvaluationFramework 负责
    - 任务管理和调度由 EvaluationFramework 负责
    """
    
    def __init__(
        self,
        model_path: str = "data/weights/vpt/2x.model",
        weights_path: str = "data/weights/steve1/steve1.weights",
        prior_weights: str = "data/weights/steve1/steve1_prior.pt",
        text_cond_scale: float = 6.0,
        visual_cond_scale: float = 7.0,
        seed: int = 42,
        enable_render: bool = False,
        video_size: Optional[Tuple[int, int]] = None,
        env_name: str = 'MineRLHarvestEnv-v0',
        env_config: Optional[Dict] = None
    ):
        """
        初始化 STEVE-1 评估器（执行器/Worker）
        
        职责：
        - 加载 STEVE-1 模型和环境
        - 集成中文翻译器
        - 执行单个任务评估
        - 录制和保存视频（如果启用）
        
        Args:
            model_path: VPT 模型配置文件路径
            weights_path: STEVE-1 权重文件路径
            prior_weights: STEVE-1 Prior 权重文件路径（重要！）
            text_cond_scale: Text classifier-free guidance scale
            visual_cond_scale: Visual classifier-free guidance scale
            seed: 随机种子
            enable_render: 是否启用渲染
            video_size: 视频尺寸 (width, height)，None 表示不录制
            env_name: 环境名称（支持自定义环境，如 'MineRLHarvestEnv-v0'）
            env_config: 环境配置（传递给环境的参数，如 reward_config 等）
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.prior_weights = prior_weights
        self.text_cond_scale = text_cond_scale
        self.visual_cond_scale = visual_cond_scale
        self.seed = seed
        self.enable_render = enable_render
        self.video_size = video_size  # None 或 (width, height)
        self.env_name = env_name
        self.env_config = env_config
        
        # 延迟加载
        self._agent = None
        self._mineclip = None
        self._prior = None
        self._env = None
        
        # 初始化中文翻译器
        self.translator = ChineseTranslator(
            term_dict_path="data/chinese_terms.json",
            method="term_dict"  # 使用术语词典翻译
        )
        
        logger.info("STEVE-1 评估器初始化完成")
        if self.video_size:
            logger.info(f"  视频录制: 启用 (尺寸: {self.video_size[0]}x{self.video_size[1]})")
    
    def _load_components(self):
        """延迟加载 Agent, MineCLIP, Prior 和环境"""
        if self._agent is None:
            # 获取当前device信息
            import torch
            from src.utils.device import DEVICE
            
            logger.info(f"{'='*30}")
            logger.info(f"加载 STEVE-1 组件...")
            logger.info(f"{'='*30}")

            logger.info(f"Device 模式: {DEVICE}")
            if DEVICE == 'cuda':
                logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            elif DEVICE == 'mps':
                logger.info(f"  Apple Silicon GPU")
            else:
                logger.info(f"CPU 模式")
            
            logger.info(f"  模型: {self.model_path}")
            logger.info(f"  权重: {self.weights_path}")
            logger.info(f"  Prior: {self.prior_weights}")
            logger.info(f"  环境: {self.env_name}")
            if self.env_config:
                logger.info(f"  环境配置: {self.env_config}")
            logger.info(f"  Text CFG Scale: {self.text_cond_scale}")
            logger.info(f"  Visual CFG Scale: {self.visual_cond_scale}")
            
            # 1. 加载 Agent 和环境（支持自定义环境和配置）
            self._agent, self._mineclip, self._env = load_mineclip_agent_env(
                in_model=self.model_path,
                in_weights=self.weights_path,
                seed=self.seed,
                cond_scale=self.text_cond_scale,
                env_name=self.env_name,
                env_config=self.env_config  # 传递环境配置
            )
            
            # 🔧 包装agent的get_action方法，确保输入tensor是float32
            original_get_action = self._agent.get_action
            def get_action_float32(obs, goal_embed):
                """包装get_action，确保输入是float32"""
                # 如果goal_embed是numpy，确保是float32
                if isinstance(goal_embed, np.ndarray) and goal_embed.dtype == np.float16:
                    goal_embed = goal_embed.astype(np.float32)
                
                # 使用原始方法，但在禁用autocast的环境下
                with th.cuda.amp.autocast(enabled=False):
                    return original_get_action(obs, goal_embed)
            
            self._agent.get_action = get_action_float32
            logger.info("  ✓ Agent get_action 已包装为float32模式")
            
            # 2. 加载 Prior 模型（官方方式）
            logger.info(f"  加载 Prior CVAE...")
            prior_info = PRIOR_INFO.copy()
            prior_info['prior_weights'] = self.prior_weights
            self._prior = load_vae_model(prior_info)
            logger.info(f"  ✓ Prior 加载完成")
            
            logger.info(f"  ✓ STEVE-1 所有组件加载完成")
    
    def evaluate_task(
        self,
        task_id: str,
        language: str = "en",
        n_trials: int = 10,
        max_steps: int = 1000,
        instruction: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> TaskResult:
        """
        评估单个任务
        
        Args:
            task_id: 任务ID (如 "simple_survival", "chop_tree")
            language: 语言类型 ('en', 'zh_auto', 'zh_manual')
            n_trials: 试验次数
            max_steps: 最大步数
            instruction: 自定义指令（如果不提供，使用默认）
            output_dir: 输出目录（用于保存视频等）
            
        Returns:
            TaskResult: 任务评估结果
        """
        # 加载组件
        self._load_components()
        
        logger.info(f"{'='*30}")
        logger.info(f"开始评估任务: {task_id}")
        logger.info(f"{'='*30}")
        

        # 🔑 如果是中文指令，自动翻译成英文
        original_instruction = instruction
        if language in ["zh", "zh_auto", "zh_manual"]:
            logger.info(f"检测到中文指令，执行翻译...")
            logger.info(f"  原始指令: {instruction}")
            instruction = self.translator.translate(instruction)
            logger.info(f"  翻译结果: {instruction}")
        
        logger.info(f"  语言: {language}")
        logger.info(f"  指令: {original_instruction}")
        if original_instruction != instruction:
            logger.info(f"  翻译后: {instruction}")
        logger.info(f"  试验次数: {n_trials}")
        logger.info(f"  最大步数: {max_steps}")
        
        # 运行多次试验
        trials = []
        for trial_idx in range(n_trials):
            logger.info(f"  Trial {trial_idx + 1}/{n_trials}...")
            
            # ⚠️ 临时禁用：每次 trial 前重新加载组件（避免环境状态污染）
            # 取消注释以下代码块可启用环境重建
            """
            if trial_idx > 0:
                logger.info(f"  ♻️  重新创建环境...")
                try:
                    # 关闭旧环境
                    if self._env is not None:
                        self._env.close()
                    
                    # 清理 saves
                    self._clean_minedojo_saves()
                    
                    # 重新创建环境（保持 agent 和 mineclip）
                    from src.utils.steve1_mineclip_agent_env_utils import make_env
                    self._env = make_env(
                        seed=42,
                        env_name=self.env_name,
                        env_config=self.env_config
                    )
                    logger.info(f"  ✓ 环境已重新创建")
                except Exception as e:
                    logger.error(f"  ⚠️ 重新创建环境失败: {e}")
                    # 继续使用旧环境
            """
            
            trial_result = self._run_single_trial(
                task_id=task_id,
                instruction=instruction,
                max_steps=max_steps,
                trial_idx=trial_idx + 1,  # 1-based for display
                n_trials=n_trials,  # 传递总试验数
                output_dir=output_dir  # 传递输出目录
            )
            
            trials.append(trial_result)
            
            logger.info(f"    结果: {'✅ 成功' if trial_result.success else '❌ 失败'}, "
                       f"步数: {trial_result.steps}, "
                       f"时间: {trial_result.time_seconds:.1f}s")

        # 构建任务结果
        task_result = TaskResult(
            task_id=task_id,
            language=language,
            instruction=original_instruction,  # 保存原始指令
            trials=trials
        )
        
        logger.info(f"任务评估完成: 成功率 {task_result.success_rate*100:.1f}%")
        
        return task_result
    
    def _run_single_trial(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        trial_idx: int,
        n_trials: int,  # 总试验数
        output_dir: Optional[Path] = None  # 输出目录
    ) -> TrialResult:
        """
        运行单次试验，可选录制视频
        
        Args:
            task_id: 任务ID
            instruction: 指令文本
            max_steps: 最大步数
            trial_idx: 试验索引（从1开始）
            n_trials: 总试验数
            output_dir: 输出目录（用于保存视频）
            
        Returns:
            TrialResult: 试验结果（不包含frames）
        """
        start_time = time.time()
        frames = [] if self.video_size else None  # 只在需要时收集帧
        
        try:
            # 使用 Prior 编码指令（官方方式）
            logger.debug(f"  使用 Prior 编码指令: '{instruction}'")
            with th.no_grad():
                # 使用官方的 get_prior_embed 函数
                prompt_embed = get_prior_embed(
                    instruction,
                    self._mineclip,
                    self._prior,
                    DEVICE
                )
                # 🔧 修复dtype问题: 确保嵌入是float32（针对4090等支持混合精度的GPU）
                if hasattr(prompt_embed, 'dtype') and prompt_embed.dtype == th.float16:
                    logger.debug(f"  检测到 float16 嵌入，转换为 float32")
                    prompt_embed = prompt_embed.float()
                
                # 转换为 numpy（MineRLConditionalAgent 需要）
                prompt_embed_np = prompt_embed.cpu().numpy() if hasattr(prompt_embed, 'cpu') else prompt_embed
            
            # 重置环境
            obs = self._env.reset()
            
            # 注意: 官方实现中没有显式调用 agent.reset()
            # Agent 的内部状态（LSTM）会在第一次调用时自动初始化
            
            # 运行 episode
            done = False
            success = False
            steps = 0
            total_reward = 0.0
            
            # 创建 tqdm 进度条
            with tqdm(
                total=max_steps, 
                desc=f"Trial {trial_idx}/{n_trials}",
                unit="step",
                leave=False,
                ncols=100,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]'
            ) as pbar:
                while not done and steps < max_steps:
                    # 获取动作（使用 Prior 计算的嵌入）
                    # wrapper已经处理了dtype和autocast，直接调用即可
                    with th.no_grad():
                        action = self._agent.get_action(obs, prompt_embed_np)
                    
                    # 执行动作
                    obs, reward, done, info = self._env.step(action)
                    
                    # 累积奖励（环境自己计算奖励）
                    total_reward += reward
                    steps += 1
                    
                    # 更新进度条
                    pbar.update(1)
                    if reward > 0:
                        pbar.set_postfix({'reward': f'{total_reward:.1f}'})
                    
                    # 收集视频帧（如果启用录制）
                    if frames is not None and 'pov' in obs:
                        frame = obs['pov']
                        # 使用 video_size 调整大小
                        frame_resized = cv2.resize(frame, self.video_size)
                        frames.append(frame_resized)
                    
                    # 记录奖励（用于调试）
                    if reward > 0:
                        logger.debug(f"    Step {steps}: reward={reward:.3f}")
                    
                    # 可选渲染
                    if self.enable_render:
                        self._env.render()
            
            # 调试：任务结束时打印详细信息（无论done是True还是超时）
            logger.info("="*60)
            logger.info(f"任务结束调试信息 (Step {steps})")
            logger.info("="*60)
            
            # 打印基本信息
            logger.info(f"总奖励: {total_reward}")
            logger.info(f"最后done: {done}")
            
            # 打印所有非0库存
            non_zero_items = {}
            if 'inventory' in obs:
                for key, value in obs['inventory'].items():
                    # 处理 numpy array
                    if hasattr(value, 'item'):
                        value = value.item()
                    if value > 0:
                        non_zero_items[key] = value
            
            if non_zero_items:
                logger.info("库存中的物品:")
                for item, count in non_zero_items.items():
                    logger.info(f"  {item}: {count}")
            else:
                logger.info("库存为空")            # 打印结束原因
            if steps >= max_steps:
                logger.info(f"结束原因: 达到最大步数 ({steps})")
            elif done and total_reward > 0:
                logger.info(f"结束原因: 任务目标达成 (总奖励: {total_reward})")
            elif done:
                logger.info(f"结束原因: 任务提前结束但无奖励 (done=True)")
            else:
                logger.info(f"❓ 结束原因: 未知")
            
            logger.info("="*60)
            
            # 判断成功
            # 1. 如果 done=True 且有奖励，说明任务完成
            # 2. 对于 Survival 类任务，能持续运行较长时间即为成功
            if done and total_reward > 0:
                success = True
            # else:
            #     success = steps >= max_steps * 0.8
            
            time_seconds = time.time() - start_time
            
            # 保存视频（如果录制了）
            if frames and output_dir:
                try:
                    from steve1.utils.video_utils import save_frames_as_video
                    output_dir.mkdir(parents=True, exist_ok=True)
                    video_path = output_dir / f"trial_{trial_idx}.mp4"
                    logger.info(f"  保存视频: trial_{trial_idx}.mp4 ({len(frames)} 帧)")
                    save_frames_as_video(frames, str(video_path), 20, to_bgr=True)
                    logger.info(f"  ✓ 视频已保存: {video_path.name}")
                except Exception as e:
                    logger.warning(f"  ⚠ 视频保存失败: {e}")
                finally:
                    # 清空 frames 释放内存
                    frames.clear()
            
            return TrialResult(
                task_id=task_id,
                language="",  # 将在外层填充
                instruction=instruction,
                success=success,
                steps=steps,
                time_seconds=time_seconds
            )
            
        except Exception as e:
            logger.error(f"Trial {trial_idx} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果录制了视频但出错，也尝试保存（可能部分帧已收集）
            if frames and output_dir:
                try:
                    from steve1.utils.video_utils import save_frames_as_video
                    output_dir.mkdir(parents=True, exist_ok=True)
                    video_path = output_dir / f"trial_{trial_idx}.mp4"
                    if frames:
                        logger.info(f"  保存部分视频: trial_{trial_idx}.mp4 ({len(frames)} 帧)")
                        save_frames_as_video(frames, str(video_path), 20, to_bgr=True)
                    frames.clear()
                except Exception as save_error:
                    logger.warning(f"  ⚠ 视频保存失败: {save_error}")
            
            time_seconds = time.time() - start_time
            
            return TrialResult(
                task_id=task_id,
                language="",
                instruction=instruction,
                success=False,
                steps=0,
                time_seconds=time_seconds
            )
    
    def _clean_minedojo_saves(self):
        """清理 MineDojo 的 saves 目录"""
        import shutil
        import sys
        from pathlib import Path
        
        try:
            # MineDojo saves 目录位于其安装路径下的 Malmo/Minecraft/run/saves/
            minedojo_path = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "minedojo"
            saves_path = minedojo_path / "sim" / "Malmo" / "Minecraft" / "run" / "saves"
            
            if not saves_path.exists():
                return
            
            # 统计删除前的大小
            total_size = 0
            save_count = 0
            for save_dir in saves_path.iterdir():
                if save_dir.is_dir():
                    save_count += 1
                    for file in save_dir.rglob('*'):
                        if file.is_file():
                            total_size += file.stat().st_size
            
            if save_count == 0:
                return
            
            # 删除所有存档
            for save_dir in saves_path.iterdir():
                if save_dir.is_dir():
                    shutil.rmtree(save_dir)
            
            freed_mb = total_size / (1024 * 1024)
            logger.info(f"  ✓ 已清理 {save_count} 个 MineDojo 存档，释放 {freed_mb:.1f} MB 空间")
            
        except Exception as e:
            pass  # 静默失败
    
    def close(self):
        """清理资源，释放内存"""
        if self._env is not None:
            try:
                self._env.close()
                logger.debug("✓ 环境已关闭")
            except Exception as e:
                logger.warning(f"关闭环境时出错: {e}")
            finally:
                self._env = None
        
        # 清理 MineRL saves 存档（防止磁盘空间积累）
        try:
            from src.utils.minerl_cleanup import clean_minerl_saves
            removed_count, freed_mb = clean_minerl_saves()
            if removed_count > 0:
                logger.info(f"✓ 已清理 {removed_count} 个 MineRL 存档，释放 {freed_mb:.1f} MB 空间")
        except Exception as e:
            logger.warning(f"清理 MineRL 存档时出错: {e}")
        
        # 释放模型引用，帮助垃圾回收
        if self._agent is not None:
            self._agent = None
        if self._mineclip is not None:
            self._mineclip = None
        if self._prior is not None:
            self._prior = None
        
        # 清理 CUDA 缓存（如果使用GPU）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("✓ CUDA 缓存已清理")
        except Exception:
            pass
