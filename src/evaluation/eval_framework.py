"""
评估框架 - 任务管理与调度
Evaluation Framework - Task Management and Scheduling

职责:
- 管理 STEVE1Evaluator 实例
- 从 YAML 加载任务配置
- 单/批量任务调度
- 结果收集与聚合
- 生成报告和统计
"""

import sys
import logging
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入自定义环境（触发环境注册）
import src.envs

from src.evaluation.steve1_evaluator import STEVE1Evaluator
from src.evaluation.metrics import TaskResult
from src.evaluation.task_loader import TaskLoader
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.matrix_analyzer import MatrixAnalyzer
from src.evaluation.html_report_generator import HTMLReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置"""
    # STEVE-1 模型配置
    model_path: str = "data/weights/vpt/2x.model"
    weights_path: str = "data/weights/steve1/steve1.weights"
    prior_weights: str = "data/weights/steve1/steve1_prior.pt"
    text_cond_scale: float = 6.0
    seed: int = 42
    enable_render: bool = False
    enable_report: bool = False
    video_size: Optional[Tuple[int, int]] = None  # 视频尺寸 (width, height)，None 表示不录制
    
    # 评估配置
    n_trials: int = 3  # 默认每个任务运行次数
    max_steps: int = 2000  # 默认最大步数
    
    # 路径配置
    task_config_path: str = "config/eval_tasks.yaml"
    results_dir: str = "results/evaluation"


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
        
        # 初始化报告生成器
        self.report_generator = ReportGenerator(self.config.results_dir)
        self.matrix_analyzer = MatrixAnalyzer()
        self.html_generator = HTMLReportGenerator(self.config.results_dir)

        # 保留 evaluator 参数用于向后兼容，但不在初始化时创建
        # 每个任务会创建专用的 evaluator，避免环境配置冲突
        self.evaluator = evaluator  # 通常为 None
        if self.evaluator:
            logger.info("使用提供的评估器实例")
        
        # 结果存储
        self.results: List[TaskResult] = []
        
        # Task-set 目录（用于批量评估时组织结果）
        self.current_task_set_dir: Optional[Path] = None
        
        logger.info("评估框架初始化完成")
    
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
        
        # 2. 过滤 MineRL/Malmo 的 WARNING 日志
        # 设置 minerl 相关 logger 的级别为 ERROR
        minerl_loggers = [
            'minerl.env.malmo.instance',
            'minerl.env._multiagent',
            'minerl.env.malmo',
        ]
        for logger_name in minerl_loggers:
            minerl_logger = logging.getLogger(logger_name)
            minerl_logger.setLevel(logging.ERROR)  # 只显示 ERROR 及以上级别
        
        # 3. 过滤 STEVE-1 的 UserWarning
        warnings.filterwarnings('ignore', category=UserWarning, module='steve1')
        
        logger.debug("日志系统已配置：缩短模块名、过滤不需要的日志")
    
    def evaluate_single_task(
        self,
        task_id: str,
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None,
        parent_dir: Optional[Path] = None,  # 父目录（用于 task-set）
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
        
        # 确定参数（优先级：函数参数 > 全局配置 > 任务配置）
        # 注意：命令行参数（n_trials, max_steps）应该优先于任务配置
        n_trials = n_trials if n_trials is not None else (self.config.n_trials if self.config.n_trials != 3 else task_config.get('n_trials', 3))
        max_steps = max_steps if max_steps is not None else (self.config.max_steps if self.config.max_steps != 2000 else task_config.get('max_steps', 2000))
        
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
                    logger.info(f"  使用全局 image_size: {env_config['image_size']}")
                else:
                    env_config['image_size'] = global_image_size
                    logger.info(f"  使用全局 image_size: {env_config['image_size']}")
        
        # 获取动作序列文件路径（如果配置了）
        replay_actions_file = task_config.get('replay_actions_file', None)
        if replay_actions_file:
            logger.info(f"  检测到动作序列文件: {replay_actions_file}")
        
        # 为当前任务创建专用的 evaluator（确保环境配置正确）
        logger.info("创建任务专用评估器...")
        task_evaluator = STEVE1Evaluator(
            model_path=self.config.model_path,
            weights_path=self.config.weights_path,
            prior_weights=self.config.prior_weights,
            text_cond_scale=self.config.text_cond_scale,
            seed=self.config.seed,
            enable_render=self.config.enable_render,
            video_size=self.config.video_size,  # 视频尺寸，None 表示不录制
            env_name=env_name,
            env_config=env_config,  # 传递环境配置（包含 max_episode_steps）
            enable_report=self.config.enable_report,
            replay_actions_file=replay_actions_file  # 传递动作序列文件路径
        )
        
        logger.info(f"{'='*30}")
        logger.info(f"调度任务: {task_id}")
        logger.info(f"{'='*30}")
        logger.info(f"  描述: {task_config.get('description', 'N/A')}")
        logger.info(f"  类别: {task_config.get('category', 'N/A')}")
        logger.info(f"  难度: {task_config.get('difficulty', 'N/A')}")
        logger.info(f"  指令: {instruction}")
        logger.info(f"  语言: {language}")
        logger.info(f"  试验次数: {n_trials}")
        logger.info(f"  最大步数: {max_steps}")
        if env_config.get('specified_biome'):
            logger.info(f"  🌍 指定Biome: {env_config.get('specified_biome')}")
        if replay_actions_file:
            logger.info(f"  🎬 回放模式: {replay_actions_file}")
        
        # 创建任务输出目录（总是创建，不管是否保存视频）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{task_id}_{language}_{timestamp}"
        
        # 如果提供了父目录，在父目录下创建任务目录
        if parent_dir:
            output_dir = parent_dir / dir_name
        else:
            output_dir = Path(self.config.results_dir) / dir_name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  结果目录: {output_dir}")
        
        try:
            # 调用评估器执行
            result = task_evaluator.evaluate_task(
                task_id=task_id,
                language=language,
                n_trials=n_trials,
                max_steps=max_steps,
                instruction=instruction,
                output_dir=output_dir,  # 传递输出目录给evaluator
            )
            
            # 保存任务结果到目录
            self._save_task_results(result, output_dir)
            
            # 保存结果
            self.results.append(result)
            
            return result
        finally:
            # ⚠️ 重要：立即关闭任务评估器，释放资源
            logger.info(f"  关闭任务评估器，释放环境资源...")
            task_evaluator.close()
            logger.info(f"  ✓ 资源已释放")
    
    def _save_task_results(self, result: TaskResult, output_dir: Path):
        """
        保存任务结果到指定目录（JSON、TXT）
        
        注意：视频保存现在由 steve1_evaluator 在 _run_single_trial 中完成
        
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
        logger.info(f"\n{'='*80}")
        logger.info(f"批量评估开始: {len(task_ids)} 个任务")
        logger.info(f"{'='*80}\n")
        
        # 如果提供了 task_set_name，创建 task-set 目录
        task_set_dir = None
        if task_set_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_set_dir_name = f"{task_set_name}_{timestamp}"
            task_set_dir = Path(self.config.results_dir) / task_set_dir_name
            task_set_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Task-set 目录: {task_set_dir}")
            logger.info(f"{'='*80}\n")
            # 保存 task_set_dir 供后续 generate_report 使用
            self.current_task_set_dir = task_set_dir
        
        results = []
        
        for i, task_id in enumerate(task_ids, 1):
            logger.info(f"\n[{i}/{len(task_ids)}] 评估任务: {task_id}")
            
            try:
                # evaluate_single_task 现在返回 tuple
                result = self.evaluate_single_task(
                    task_id=task_id,
                    n_trials=n_trials,
                    max_steps=max_steps,
                    parent_dir=task_set_dir  # 传递 task-set 目录
                )
                results.append(result)  # 只保存 TaskResult
                
                # 打印任务摘要
                logger.info(f"  ✅ 完成: 成功率 {result.success_rate*100:.1f}%, "
                           f"平均步数 {result.avg_steps:.1f}")
                
            except Exception as e:
                logger.error(f"  ❌ 任务失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"批量评估完成: {len(results)}/{len(task_ids)} 个任务成功")
        logger.info(f"{'='*80}\n")
        
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
        
        logger.info(f"\n{'='*80}")
        logger.info(f"评估任务集: {task_set_name}")
        logger.info(f"任务数量: {len(task_ids)}")
        logger.info(f"任务列表: {', '.join(task_ids)}")
        logger.info(f"{'='*80}\n")
        
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
        
        print(f"\n{'='*80}")
        print("评估结果汇总")
        print(f"{'='*80}\n")
        
        # 表头
        print(f"{'任务ID':<30} {'指令':<20} {'成功率':<10} {'平均步数':<12} {'平均时间'}")
        print("-" * 80)
        
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
        print(f"{'='*80}\n")
    
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
            logger.info(f"  将报告保存到 task-set 目录: {self.current_task_set_dir.name}")
        elif len(results) == 1:
            # 单任务评估，保存到任务目录下
            task_id = results[0].task_id
            language = results[0].language
            # 查找匹配的目录（按时间倒序）
            pattern = f"{task_id}_{language}_*"
            matching_dirs = sorted(
                Path(self.config.results_dir).glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if matching_dirs:
                json_path = matching_dirs[0] / json_filename
                #logger.info(f"  将报告保存到任务目录: {matching_dirs[0].name}")
            else:
                json_path = Path(self.report_generator.output_dir) / json_filename
        else:
            # 多任务但无 task-set，使用全局目录
            json_path = Path(self.report_generator.output_dir) / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 生成文本报告
        txt_path = json_path.with_suffix('.txt')
        self._generate_text_report(report_data, txt_path)
        
        # 生成三维能力矩阵分析和HTML报告
        matrix_analysis, html_path = self._generate_matrix_report(results, json_path.parent)
        
        #logger.info(f"\n{'='*80}")
        #logger.info(f"报告已生成:")
        #logger.info(f"  JSON: {json_path}")
        #logger.info(f"  TXT:  {txt_path}")
        #if html_path:
        #    logger.info(f"  HTML: {html_path}")
        #logger.info(f"{'='*80}\n")
        
        return str(json_path), str(txt_path)
    
    def _generate_matrix_report(
        self, 
        results: List[TaskResult], 
        output_dir: Path
    ) -> Tuple[Optional[Dict], Optional[Path]]:
        """
        生成三维能力矩阵分析报告和HTML可视化报告
        
        Args:
            results: 任务结果列表
            output_dir: 输出目录
            
        Returns:
            (matrix_analysis, html_path): 矩阵分析结果和HTML路径
        """
        try:
            # 将TaskResult转换为矩阵分析器需要的格式
            analysis_data = []
            for result in results:
                # 从task_loader中获取原始任务配置
                task_config = self.task_loader.get_task(result.task_id)
                if not task_config:
                    logger.warning(f"无法找到任务配置: {result.task_id}")
                    continue
                
                task_data = {
                    'task_config': task_config,
                    'success_rate': result.success_rate,
                    'avg_steps': result.avg_steps,
                    'avg_time': result.avg_time,
                }
                analysis_data.append(task_data)
            
            if not analysis_data:
                logger.warning("没有可分析的任务数据")
                return None, None
            
            # 执行矩阵分析
            matrix_analysis = self.matrix_analyzer.analyze_results(analysis_data)
            
            # 保存JSON分析结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_json_path = output_dir / f"matrix_analysis_{timestamp}.json"
            self.matrix_analyzer.save_analysis(matrix_analysis, analysis_json_path)
            logger.info(f"✓ 矩阵分析已保存: {analysis_json_path.name}")
            
            # 打印分析摘要
            self.matrix_analyzer.print_summary(matrix_analysis)
            
            # 生成HTML报告
            config_filename = Path(self.config.task_config_path).name
            html_path = self.html_generator.generate(
                analysis=matrix_analysis,
                config_file=config_filename,
                output_filename=f"evaluation_report_{timestamp}.html"
            )
            
            return matrix_analysis, html_path
            
        except Exception as e:
            logger.error(f"生成矩阵报告失败: {e}", exc_info=True)
            return None, None
    
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
        """清理资源"""
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
    
    # 2. MineRL/Malmo 警告（设置 logger 级别）
    minerl_loggers = [
        'minerl.env.malmo.instance',
        'minerl.env._multiagent',
        'minerl.env.malmo',
    ]
    for logger_name in minerl_loggers:
        minerl_logger = logging.getLogger(logger_name)
        minerl_logger.setLevel(logging.ERROR)
    
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
        default=3,
        help='每个任务的试验次数（默认3次）'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=2000,
        help='每个试验的最大步数（默认2000）'
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
        video_size=video_size
    )
    
    # 创建评估框架
    framework = EvaluationFramework(config=config)
    
    try:
        results = []
        
        # 根据参数选择评估模式
        if args.task:
            # 单个任务
            result = framework.evaluate_single_task(args.task)
            results = [result]
        
        elif args.task_set:
            # 任务集
            results = framework.evaluate_task_set(args.task_set)
        
        elif args.task_list:
            # 任务列表
            results = framework.evaluate_task_list(args.task_list)
        
        else:
            # 默认：快速测试
            logger.info("未指定任务，运行快速测试...")
            results = framework.evaluate_task_set('quick_test')
        
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
