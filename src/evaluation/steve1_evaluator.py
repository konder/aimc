"""
STEVE-1 评估器 (基于 MineRL 环境)
使用完整 pip 安装的 steve1 包
"""

import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch as th
import numpy as np

# 导入本地版本的工具函数（支持自定义环境）
from src.utils.steve1_mineclip_agent_env_utils import (
    load_mineclip_agent_env,
    load_mineclip_wconfig,
    load_vae_model
)
from src.utils.device import DEVICE

# 导入 steve1 官方工具
from steve1.utils.embed_utils import get_prior_embed
from steve1.config import PRIOR_INFO

from .metrics import TrialResult, TaskResult
from ..translation.translator import ChineseTranslator

logger = logging.getLogger(__name__)


def reset_env_with_retry(env, max_retries=3, retry_delay=2.0):
    """
    带重试机制的环境重置
    
    Args:
        env: MineRL 环境
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        
    Returns:
        obs: 重置后的观察
        
    Raises:
        RuntimeError: 如果所有重试都失败
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"重置环境 ({attempt + 1}/{max_retries})...")
            obs = env.reset()
            logger.info("✅ 环境重置成功")
            return obs
        except Exception as e:
            logger.warning(f"❌ 环境重置失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error(f"环境重置失败，已达到最大重试次数 ({max_retries})")
                raise RuntimeError(f"环境重置失败: {e}") from e


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
        env_name: str = 'MineRLHarvestEnv-v0'
    ):
        """
        初始化 STEVE-1 评估器（执行器/Worker）
        
        职责：
        - 加载 STEVE-1 模型和环境
        - 集成中文翻译器
        - 执行单个任务评估
        - 返回任务结果
        
        Args:
            model_path: VPT 模型配置文件路径
            weights_path: STEVE-1 权重文件路径
            prior_weights: STEVE-1 Prior 权重文件路径（重要！）
            text_cond_scale: Text classifier-free guidance scale
            visual_cond_scale: Visual classifier-free guidance scale
            seed: 随机种子
            enable_render: 是否启用渲染
            env_name: 环境名称（支持自定义环境，如 'MineRLHarvestEnv-v0'）
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.prior_weights = prior_weights
        self.text_cond_scale = text_cond_scale
        self.visual_cond_scale = visual_cond_scale
        self.seed = seed
        self.enable_render = enable_render
        self.env_name = env_name
        
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
    
    def _load_components(self):
        """延迟加载 Agent, MineCLIP, Prior 和环境"""
        if self._agent is None:
            logger.info("加载 STEVE-1 组件...")
            logger.info(f"  模型: {self.model_path}")
            logger.info(f"  权重: {self.weights_path}")
            logger.info(f"  Prior: {self.prior_weights}")
            logger.info(f"  环境: {self.env_name}")
            logger.info(f"  Text CFG Scale: {self.text_cond_scale}")
            logger.info(f"  Visual CFG Scale: {self.visual_cond_scale}")
            
            # 1. 加载 Agent 和环境（支持自定义环境）
            self._agent, self._mineclip, self._env = load_mineclip_agent_env(
                in_model=self.model_path,
                in_weights=self.weights_path,
                seed=self.seed,
                cond_scale=self.text_cond_scale,
                env_name=self.env_name  # 传递环境名称
            )
            
            # 2. 加载 Prior 模型（官方方式）
            logger.info(f"  加载 Prior CVAE...")
            prior_info = PRIOR_INFO.copy()
            prior_info['prior_weights'] = self.prior_weights
            self._prior = load_vae_model(prior_info)
            logger.info(f"  ✓ Prior 加载完成")
            
            logger.info("✅ STEVE-1 所有组件加载完成")
    
    def evaluate_task(
        self,
        task_id: str,
        language: str = "en",
        n_trials: int = 10,
        max_steps: int = 1000,
        instruction: Optional[str] = None
    ) -> TaskResult:
        """
        评估单个任务
        
        Args:
            task_id: 任务ID (如 "simple_survival", "chop_tree")
            language: 语言类型 ('en', 'zh_auto', 'zh_manual')
            n_trials: 试验次数
            max_steps: 最大步数
            instruction: 自定义指令（如果不提供，使用默认）
            
        Returns:
            TaskResult: 任务评估结果
        """
        # 加载组件
        self._load_components()
        
        # 🔑 如果是中文指令，自动翻译成英文
        original_instruction = instruction
        if language in ["zh", "zh_auto", "zh_manual"]:
            logger.info(f"检测到中文指令，执行翻译...")
            logger.info(f"  原始指令: {instruction}")
            instruction = self.translator.translate(instruction)
            logger.info(f"  翻译结果: {instruction}")
        
        logger.info(f"开始评估任务: {task_id}")
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
            
            trial_result = self._run_single_trial(
                task_id=task_id,
                instruction=instruction,
                max_steps=max_steps,
                trial_idx=trial_idx
            )
            
            trials.append(trial_result)
            
            logger.info(f"    结果: {'✅ 成功' if trial_result.success else '❌ 失败'}, "
                       f"步数: {trial_result.steps}, "
                       f"时间: {trial_result.time_seconds:.1f}s")
        
        # 构建任务结果
        task_result = TaskResult(
            task_id=task_id,
            language=language,
            instruction=instruction,
            trials=trials
        )
        
        logger.info(f"任务评估完成: 成功率 {task_result.success_rate*100:.1f}%")
        
        return task_result
    
   
    def _run_single_trial(
        self,
        task_id: str,
        instruction: str,
        max_steps: int,
        trial_idx: int
    ) -> TrialResult:
        """运行单次试验"""
        start_time = time.time()
        
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
                # 转换为 numpy（MineRLConditionalAgent 需要）
                prompt_embed_np = prompt_embed.cpu().numpy() if hasattr(prompt_embed, 'cpu') else prompt_embed
            
            # 重置环境（带重试机制）
            obs = reset_env_with_retry(self._env, max_retries=3, retry_delay=2.0)
            
            # 注意: 官方实现中没有显式调用 agent.reset()
            # Agent 的内部状态（LSTM）会在第一次调用时自动初始化
            
            # 运行 episode
            done = False
            success = False
            steps = 0
            total_reward = 0.0
            
            while not done and steps < max_steps:
                # 获取动作（使用 Prior 计算的嵌入）
                with th.no_grad():
                    action = self._agent.get_action(obs, prompt_embed_np)
                
                # 执行动作
                obs, reward, done, info = self._env.step(action)
                total_reward += reward
                steps += 1
                
                # 记录奖励（用于调试）
                if reward > 0:
                    logger.debug(f"    Step {steps}: reward={reward:.3f}")
                
                # 可选渲染
                if self.enable_render:
                    self._env.render()
            
            # 判断成功
            # 1. 如果 done=True 且有奖励，说明任务完成
            # 2. 对于 Survival 类任务，能持续运行较长时间即为成功
            if done and total_reward > 0:
                success = True
            # else:
            #     success = steps >= max_steps * 0.8
            
            time_seconds = time.time() - start_time
            
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
            
            time_seconds = time.time() - start_time
            
            return TrialResult(
                task_id=task_id,
                language="",
                instruction=instruction,
                success=False,
                steps=0,
                time_seconds=time_seconds
            )
    
    def close(self):
        """清理资源"""
        if self._env is not None:
            self._env.close()
            logger.info("环境已关闭")
