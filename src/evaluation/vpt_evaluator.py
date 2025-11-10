"""
VPT 基础模型评估器 (用于对比实验)
使用原始 VPT 模型，不加载 STEVE-1 的指令条件功能
"""

import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch as th
import numpy as np
import gym

# 导入 VPT 相关模块
from steve1.VPT.agent import MineRLAgent, resize_image, AGENT_RESOLUTION

from .task_loader import TaskLoader
from .metrics import TrialResult, TaskResult
from .report_generator import ReportGenerator

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
            logger.info(f"尝试重置环境 (尝试 {attempt + 1}/{max_retries})...")
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


class VPTEvaluator:
    """
    VPT 基础模型评估器
    
    特性:
    - 使用原始 VPT 模型（无指令条件）
    - 基于 MineRL 环境
    - 用于对比 STEVE-1 的性能
    """
    
    def __init__(
        self,
        model_path: str = "data/weights/vpt/2x.model",
        weights_path: str = "data/weights/vpt/rl-from-foundation-2x.weights",
        seed: int = 42,
        task_config_path: str = "config/eval_tasks.yaml",
        results_dir: str = "results/evaluation",
        enable_render: bool = False
    ):
        """
        初始化 VPT 评估器
        
        Args:
            model_path: VPT 模型配置文件路径
            weights_path: VPT 权重文件路径（使用基础 VPT 权重）
            seed: 随机种子
            task_config_path: 评估任务配置路径
            results_dir: 结果输出目录
            enable_render: 是否启用渲染
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.seed = seed
        self.enable_render = enable_render
        
        # 延迟加载
        self._agent = None
        self._env = None
        
        # 加载任务配置
        self.task_loader = TaskLoader(task_config_path)
        
        # 报告生成器
        self.report_generator = ReportGenerator(results_dir)
        
        logger.info("VPT 基础模型评估器初始化完成")
    
    def _load_components(self):
        """延迟加载 Agent 和环境"""
        if self._agent is None:
            logger.info("加载 VPT 基础组件...")
            logger.info(f"  模型: {self.model_path}")
            logger.info(f"  权重: {self.weights_path}")
            
            # 加载 VPT Agent
            self._agent = MineRLAgent(
                env=None,
                device="auto",
                policy_kwargs=self.model_path,
                pi_head_kwargs=self.weights_path
            )
            
            # 创建环境
            from src.envs import register_custom_envs
            register_custom_envs()
            
            self._env = gym.make('MineRLHarvestLog-v0')
            
            logger.info("✅ VPT 组件加载完成")
    
    def evaluate_task(
        self,
        task_id: str,
        language: str = "en",
        n_trials: int = 3,
        instruction: Optional[str] = None
    ) -> TaskResult:
        """
        评估单个任务
        
        Args:
            task_id: 任务 ID
            language: 语言（用于记录，VPT 不使用）
            n_trials: 试验次数
            instruction: 指令（用于记录，VPT 不使用）
            
        Returns:
            TaskResult: 任务评估结果
        """
        self._load_components()
        
        # 获取任务配置
        task_config = self.task_loader.get_task_details(task_id)
        if not task_config:
            raise ValueError(f"任务不存在: {task_id}")
        
        # VPT 不使用指令，但记录用于报告
        if instruction is None:
            instruction = task_config.get('instruction', 'no instruction (VPT baseline)')
        
        # 获取最大步数
        expected_steps = task_config.get('expected_steps', 2000)
        if isinstance(expected_steps, str):
            if '-' in expected_steps:
                expected_steps = int(expected_steps.split('-')[1])
            else:
                expected_steps = int(expected_steps)
        max_steps = expected_steps * 2
        
        trials = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始评估任务: {task_id} (VPT 基础模型)")
        logger.info(f"  注意: VPT 不使用指令，仅根据预训练行为探索")
        logger.info(f"  试验次数: {n_trials}")
        logger.info(f"  最大步数: {max_steps}")
        logger.info(f"{'='*60}")
        
        for trial_idx in range(n_trials):
            logger.info(f"\n  Trial {trial_idx + 1}/{n_trials}...")
            
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
            instruction=f"{instruction} (VPT无指令)",
            trials=trials
        )
        
        logger.info(f"\n任务评估完成: 成功率 {task_result.success_rate*100:.1f}%")
        
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
            # 重置环境（带重试机制）
            obs = reset_env_with_retry(self._env, max_retries=3, retry_delay=2.0)
            
            # 重置 Agent 状态
            self._agent.reset()
            
            # 运行 episode（VPT 不使用指令）
            done = False
            steps = 0
            total_reward = 0.0
            
            while not done and steps < max_steps:
                # 预处理观察（VPT 需要特定格式）
                pov = obs['pov']
                # Resize to VPT resolution
                pov = resize_image(pov, AGENT_RESOLUTION)
                
                # 获取动作（VPT 不需要条件输入）
                with th.no_grad():
                    action = self._agent.get_action(pov)
                
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
            
            # 判断成功（与 STEVE-1 相同的标准）
            success = steps >= max_steps * 0.8
            
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
    
    def evaluate_task_set(self, set_name: str, language: str = "en") -> List[TaskResult]:
        """评估一个任务集"""
        task_ids = self.task_loader.get_task_set(set_name)
        results = []
        for task_id in task_ids:
            result = self.evaluate_task(task_id, language)
            results.append(result)
        return results
    
    def generate_report(
        self,
        results: List[TaskResult],
        report_name: str = "vpt_baseline_report"
    ):
        """
        生成评估报告
        
        Args:
            results: 任务结果列表
            report_name: 报告名称
        """
        import json
        from datetime import datetime
        
        # 构建报告数据
        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(results),
                "evaluator": "VPT Baseline",
                "note": "VPT 基础模型不使用指令，仅根据预训练行为探索"
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
                        "time_seconds": trial.time_seconds
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
        json_path = Path(self.report_generator.output_dir) / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 生成文本报告
        txt_path = json_path.with_suffix('.txt')
        self._generate_text_report(report_data, txt_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"报告已生成:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  TXT:  {txt_path}")
        logger.info(f"{'='*60}")
        
        return str(json_path), str(txt_path)
    
    def _generate_text_report(self, report_data: Dict[str, Any], output_path: Path):
        """生成人类可读的文本报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("VPT 基础模型评估报告\n")
            f.write("="*60 + "\n\n")
            
            # 元数据
            f.write(f"生成时间: {report_data['metadata']['timestamp']}\n")
            f.write(f"任务数量: {report_data['metadata']['total_tasks']}\n")
            f.write(f"备注: {report_data['metadata']['note']}\n\n")
            
            # 总体统计
            summary = report_data['summary']
            f.write("总体统计:\n")
            f.write(f"  总成功率: {summary['overall_success_rate']:.1f}%\n")
            f.write(f"  总试验数: {summary['total_trials']}\n")
            f.write(f"  成功试验数: {summary['successful_trials']}\n\n")
            
            # 每个任务的详情
            f.write("="*60 + "\n")
            f.write("任务详情\n")
            f.write("="*60 + "\n\n")
            
            for task in report_data['tasks']:
                f.write(f"任务: {task['task_id']}\n")
                f.write(f"  成功率: {task['success_rate']:.1f}%\n")
                f.write(f"  平均步数: {task['avg_steps']:.0f}\n")
                f.write(f"  平均时间: {task['avg_time']:.1f}s\n")
                f.write(f"  试验详情:\n")
                for i, trial in enumerate(task['trials'], 1):
                    status = "✅ 成功" if trial['success'] else "❌ 失败"
                    f.write(f"    Trial {i}: {status} | 步数: {trial['steps']:4d} | 时间: {trial['time_seconds']:.1f}s\n")
                f.write("\n")
    
    def close(self):
        """清理资源"""
        if self._env is not None:
            self._env.close()

