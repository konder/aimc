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
        output_dir: Optional[Path] = None,
        enable_report: bool = False  # 启用详细报告（包含每步的动作和截图）
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
                output_dir=output_dir,  # 传递输出目录
                enable_report=enable_report  # 传递报告模式
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
        output_dir: Optional[Path] = None,  # 输出目录
        enable_report: bool = False  # 启用详细报告：保存动作和截图
    ) -> TrialResult:
        """
        运行单次试验，可选录制视频和生成详细报告
        
        Args:
            task_id: 任务ID
            instruction: 指令文本
            max_steps: 最大步数
            trial_idx: 试验索引（从1开始）
            n_trials: 总试验数
            output_dir: 输出目录（用于保存视频和报告）
            enable_report: 启用详细报告（保存动作、截图、生成HTML报告）
            
        Returns:
            TrialResult: 试验结果（不包含frames）
        """
        start_time = time.time()
        frames = [] if self.video_size else None  # 只在需要时收集帧
        
        # 报告模式：收集动作和帧
        report_actions = [] if enable_report else None
        report_frames = [] if enable_report else None
        
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
                    
                    # 📊 报告模式：收集动作
                    if enable_report:
                        report_actions.append(action.copy() if isinstance(action, dict) else action)
                    
                    # 执行动作
                    obs, reward, done, info = self._env.step(action)
                    
                    # 📊 报告模式：保存帧
                    if enable_report and 'pov' in obs:
                        report_frames.append(obs['pov'].copy())
                    
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
            
            # 📊 报告模式：保存动作和帧，生成HTML报告
            if enable_report and report_actions and report_frames:
                self._save_report_data(
                    report_actions, 
                    report_frames, 
                    output_dir or Path("/tmp/steve1_reports"), 
                    task_id, 
                    trial_idx
                )
            
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
    
    def _save_report_data(
        self, 
        actions: List[Dict], 
        frames: List[np.ndarray], 
        output_dir: Path, 
        task_id: str, 
        trial_idx: int
    ):
        """
        保存详细报告数据（动作序列、截图、HTML报告）
        
        Args:
            actions: 动作列表
            frames: 帧列表 (POV 图像)
            output_dir: 输出目录
            task_id: 任务ID
            trial_idx: Trial 索引
        """
        import json
        from PIL import Image
        
        # 创建报告目录
        report_dir = output_dir / f"report_{task_id}_trial{trial_idx}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  📊 保存报告数据到: {report_dir}")
        
        # 1. 保存动作序列为 JSON
        actions_file = report_dir / "actions.json"
        try:
            # 转换动作为可序列化格式
            actions_serializable = []
            for i, action in enumerate(actions):
                action_dict = {}
                for key, value in action.items():
                    if isinstance(value, np.ndarray):
                        action_dict[key] = value.tolist()
                    elif hasattr(value, 'item'):  # numpy scalar
                        action_dict[key] = value.item()
                    else:
                        action_dict[key] = value
                actions_serializable.append({
                    "step": i,
                    "action": action_dict
                })
            
            with open(actions_file, 'w', encoding='utf-8') as f:
                json.dump(actions_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"    ✓ 动作序列已保存: actions.json ({len(actions)} steps)")
        except Exception as e:
            logger.error(f"    ❌ 保存动作序列失败: {e}")
        
        # 2. 保存帧图像
        frames_dir = report_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        try:
            saved_count = 0
            for i, frame in enumerate(frames):
                # frame 是 (H, W, C) 的 numpy 数组
                img = Image.fromarray(frame)
                img_path = frames_dir / f"step_{i:04d}.png"
                img.save(img_path)
                saved_count += 1
            
            logger.info(f"    ✓ 帧图像已保存: frames/ ({saved_count} 张)")
        except Exception as e:
            logger.error(f"    ❌ 保存帧图像失败: {e}")
        
        # 3. 生成简单的 HTML 报告
        html_file = report_dir / "report.html"
        try:
            html_content = self._generate_report_html(actions, len(frames), task_id, trial_idx)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"    ✓ HTML 报告已生成: report.html")
            logger.info(f"    🌐 打开报告: open {html_file}")
        except Exception as e:
            logger.error(f"    ❌ 生成 HTML 报告失败: {e}")
    
    def _generate_report_html(
        self, 
        actions: List[Dict], 
        num_frames: int, 
        task_id: str, 
        trial_idx: int
    ) -> str:
        """
        生成精美的 HTML 详细报告（左右分栏：动作 | 图像）
        
        Args:
            actions: 动作列表
            num_frames: 帧数量
            task_id: 任务ID
            trial_idx: Trial 索引
            
        Returns:
            HTML 字符串
        """
        import json
        
        # 统计动作组合
        action_combo_stats = {}
        for action in actions:
            # 收集非零动作
            active_keys = []
            
            # 移动和功能键
            for key in ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 
                        'attack', 'use', 'drop', 'inventory']:
                val = action.get(key, 0)
                if val:
                    active_keys.append(key)
            
            # Camera
            camera = action.get('camera', [0, 0])
            if isinstance(camera, np.ndarray):
                camera_flat = camera.flatten()
                if len(camera_flat) >= 2 and (camera_flat[0] != 0 or camera_flat[1] != 0):
                    active_keys.append('camera')
            elif isinstance(camera, list) and len(camera) >= 2:
                if camera[0] != 0 or camera[1] != 0:
                    active_keys.append('camera')
            
            # 合成/装备
            for key in ['craft', 'equip', 'place']:
                val = action.get(key, 'none')
                if val != 'none':
                    active_keys.append(key)
            
            # 生成组合键
            if not active_keys:
                combo_key = 'noop'
            else:
                combo_key = '+'.join(sorted(active_keys))
            
            action_combo_stats[combo_key] = action_combo_stats.get(combo_key, 0) + 1
        
        # 按出现次数排序
        sorted_combos = sorted(action_combo_stats.items(), key=lambda x: x[1], reverse=True)
        
        # 生成统计表格
        stats_html = '<div class="stats-table">\n'
        stats_html += '  <h3 style="margin-top: 0; color: #667eea;">📊 动作组合统计</h3>\n'
        stats_html += '  <table>\n'
        stats_html += '    <tr><th>动作组合</th><th>次数</th></tr>\n'
        for combo, count in sorted_combos:
            stats_html += f'    <tr><td>{combo}</td><td>{count}</td></tr>\n'
        stats_html += '  </table>\n'
        stats_html += '</div>'
        
        # 生成 HTML 头部
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STEVE-1 详细报告 - {task_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            margin: 0; 
            padding: 15px; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            font-size: 13px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 20px 30px; 
            margin-bottom: 15px; 
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .header .meta {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .stats-table {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stats-table h3 {{
            color: #667eea;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
        }}
        .stats-table table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .stats-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-radius: 5px 5px 0 0;
        }}
        .stats-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .stats-table td:last-child {{
            text-align: right;
            font-weight: 600;
            color: #667eea;
        }}
        .stats-table tr:last-child td {{
            border-bottom: none;
        }}
        .stats-table tr:hover {{
            background: #f8f9ff;
        }}
        .step-row {{ 
            display: flex;
            background: white; 
            margin-bottom: 10px; 
            border-radius: 10px; 
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .step-row:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }}
        .step-left {{ 
            flex: 0 0 50%;
            padding: 10px;
            border-right: 2px solid #667eea;
            display: flex;
            flex-direction: column;
        }}
        .step-right {{ 
            flex: 0 0 50%;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
        }}
        .step-right img {{ 
            width: 100%;
            height: auto;
            display: block;
        }}
        .step-num {{ 
            font-weight: 600; 
            color: #667eea; 
            font-size: 15px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .action-readable {{ 
            background: linear-gradient(135deg, #e7f3ff 0%, #f0f8ff 100%);
            padding: 12px; 
            border-radius: 8px; 
            margin-bottom: 10px;
            font-size: 13px;
            line-height: 1.6;
            border-left: 3px solid #667eea;
        }}
        .action-raw {{ 
            background: #f8f9fa; 
            padding: 12px; 
            border-radius: 8px; 
            font-size: 11px;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            overflow-x: auto;
            flex-grow: 1;
            max-height: 180px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        .inventory {{ 
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 3px 8px; 
            border-radius: 4px; 
            color: #155724; 
            font-weight: 600;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .key {{ 
            color: #667eea; 
            font-weight: 600;
            padding: 2px 4px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 STEVE-1 详细报告</h1>
            <div class="meta">
                任务: {task_id} | Trial: {trial_idx} | 总步数: {len(actions)} | 帧数: {num_frames}
            </div>
        </div>
        
        {stats_html}
"""
        
        # 生成每一步的紧凑信息（左右分栏）
        for i, action in enumerate(actions):
            # 生成可读的动作描述
            readable_action = self._format_action_readable(action)
            
            # 将原始动作转换为 JSON 字符串
            action_json = self._action_to_json_str(action)
            
            html += f"""
    <div class="step-row">
        <div class="step-left">
            <div class="step-num">Step {i}</div>
            <div class="action-readable">{readable_action}</div>
            <div class="action-raw">{action_json}</div>
        </div>
        <div class="step-right">
            <img src="frames/step_{i:04d}.png" alt="Step {i}">
        </div>
    </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        return html
    
    def _format_action_readable(self, action: Dict[str, Any]) -> str:
        """
        将动作格式化为可读的 HTML 字符串
        
        Args:
            action: 动作字典
            
        Returns:
            HTML 格式的可读字符串
        """
        parts = []
        
        # 移动
        if action.get('forward', 0):
            parts.append('<span class="key">forward</span>')
        if action.get('back', 0):
            parts.append('<span class="key">back</span>')
        if action.get('left', 0):
            parts.append('<span class="key">left</span>')
        if action.get('right', 0):
            parts.append('<span class="key">right</span>')
        
        # 功能
        if action.get('jump', 0):
            parts.append('<span class="key">jump</span>')
        if action.get('sneak', 0):
            parts.append('<span class="key">sneak</span>')
        if action.get('sprint', 0):
            parts.append('<span class="key">sprint</span>')
        if action.get('attack', 0):
            parts.append('<span class="key">attack</span>')
        if action.get('use', 0):
            parts.append('<span class="key">use</span>')
        if action.get('drop', 0):
            parts.append('<span class="key">drop</span>')
        if action.get('inventory', 0):
            parts.append('<span class="inventory">📦 INVENTORY</span>')
        
        # 合成/装备
        if action.get('craft', 'none') != 'none':
            parts.append(f'<span class="key">craft({action["craft"]})</span>')
        if action.get('equip', 'none') != 'none':
            parts.append(f'<span class="key">equip({action["equip"]})</span>')
        if action.get('place', 'none') != 'none':
            parts.append(f'<span class="key">place({action["place"]})</span>')
        
        # Camera
        camera = action.get('camera', [0, 0])
        if isinstance(camera, np.ndarray):
            camera_flat = camera.flatten()
            if len(camera_flat) >= 2:
                camera_pitch = float(camera_flat[0])
                camera_yaw = float(camera_flat[1])
                if camera_pitch != 0 or camera_yaw != 0:
                    parts.append(f'<span class="key">camera=({camera_pitch:.2f}, {camera_yaw:.2f})</span>')
        
        if not parts:
            return '<span style="color: #999;">noop</span>'
        
        return ' + '.join(parts)
    
    def _action_to_json_str(self, action: Dict[str, Any]) -> str:
        """
        将动作转换为格式化的 JSON 字符串
        
        Args:
            action: 动作字典
            
        Returns:
            格式化的 JSON 字符串
        """
        import json
        
        # 转换为可序列化格式
        action_serializable = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action_serializable[key] = value.tolist()
            elif hasattr(value, 'item'):
                action_serializable[key] = value.item()
            else:
                action_serializable[key] = value
        
        return json.dumps(action_serializable, indent=2, ensure_ascii=False)
        
        # 清理 CUDA 缓存（如果使用GPU）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("✓ CUDA 缓存已清理")
        except Exception:
            pass
