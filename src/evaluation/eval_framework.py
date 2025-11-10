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
from typing import List, Dict, Any, Optional
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
        self.config = config or EvaluationConfig()
        
        # 加载任务配置
        self.task_loader = TaskLoader(self.config.task_config_path)
        logger.info(f"加载任务配置: {len(self.task_loader.tasks)} 个任务")
        
        # 初始化报告生成器
        self.report_generator = ReportGenerator(self.config.results_dir)
        
        # 创建或使用提供的评估器
        if evaluator is None:
            logger.info("创建 STEVE-1 评估器...")
            self.evaluator = STEVE1Evaluator(
                model_path=self.config.model_path,
                weights_path=self.config.weights_path,
                prior_weights=self.config.prior_weights,
                text_cond_scale=self.config.text_cond_scale,
                seed=self.config.seed,
                enable_render=self.config.enable_render,
                env_name='MineRLHarvestEnv-v0'  # 默认使用自定义环境
            )
        else:
            logger.info("使用提供的评估器实例")
            self.evaluator = evaluator
        
        # 结果存储
        self.results: List[TaskResult] = []
        
        logger.info("评估框架初始化完成")
    
    def evaluate_single_task(
        self,
        task_id: str,
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> TaskResult:
        """
        评估单个任务
        
        Args:
            task_id: 任务ID
            n_trials: 试验次数（如果None则使用配置中的值）
            max_steps: 最大步数（如果None则使用配置中的值）
        
        Returns:
            TaskResult: 任务结果
        """
        # 从配置加载任务
        task_config = self.task_loader.get_task(task_id)
        if not task_config:
            raise ValueError(f"任务不存在: {task_id}")
        
        # 确定参数（优先级：函数参数 > 任务配置 > 全局配置）
        n_trials = n_trials or task_config.get('n_trials') or self.config.n_trials
        max_steps = max_steps or task_config.get('max_steps') or self.config.max_steps
        
        # 确定指令和语言
        instruction = None
        language = "en"
        
        if 'en_instruction' in task_config:
            instruction = task_config['en_instruction']
            language = "en"
        elif 'zh_instruction' in task_config:
            instruction = task_config['zh_instruction']
            language = "zh"
        
        # 如果任务配置了自定义环境，更新评估器
        if 'env_name' in task_config:
            env_name = task_config['env_name']
            if self.evaluator.env_name != env_name:
                logger.info(f"切换环境: {self.evaluator.env_name} → {env_name}")
                self.evaluator.env_name = env_name
                # 强制重新加载环境
                self.evaluator._env = None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"评估任务: {task_id}")
        logger.info(f"{'='*80}")
        logger.info(f"  描述: {task_config.get('description', 'N/A')}")
        logger.info(f"  类别: {task_config.get('category', 'N/A')}")
        logger.info(f"  难度: {task_config.get('difficulty', 'N/A')}")
        logger.info(f"  指令: {instruction}")
        logger.info(f"  语言: {language}")
        logger.info(f"  试验次数: {n_trials}")
        logger.info(f"  最大步数: {max_steps}")
        
        # 调用评估器执行
        result = self.evaluator.evaluate_task(
            task_id=task_id,
            language=language,
            n_trials=n_trials,
            max_steps=max_steps,
            instruction=instruction
        )
        
        # 保存结果
        self.results.append(result)
        
        return result
    
    def evaluate_task_list(
        self,
        task_ids: List[str],
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> List[TaskResult]:
        """
        批量评估任务列表
        
        Args:
            task_ids: 任务ID列表
            n_trials: 试验次数（应用于所有任务）
            max_steps: 最大步数（应用于所有任务）
        
        Returns:
            List[TaskResult]: 任务结果列表
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"批量评估开始: {len(task_ids)} 个任务")
        logger.info(f"{'='*80}\n")
        
        results = []
        
        for i, task_id in enumerate(task_ids, 1):
            logger.info(f"\n[{i}/{len(task_ids)}] 评估任务: {task_id}")
            
            try:
                result = self.evaluate_single_task(
                    task_id=task_id,
                    n_trials=n_trials,
                    max_steps=max_steps
                )
                results.append(result)
                
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
        
        return results
    
    def evaluate_test_set(
        self,
        test_set_name: str,
        n_trials: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> List[TaskResult]:
        """
        评估测试集（从 YAML 配置中的 quick_test, baseline_test 等）
        
        Args:
            test_set_name: 测试集名称 ('quick_test', 'baseline_test')
            n_trials: 试验次数
            max_steps: 最大步数
        
        Returns:
            List[TaskResult]: 任务结果列表
        """
        # 从 YAML 加载测试集
        task_ids = self.task_loader.get_task_set(test_set_name)
        
        if not task_ids:
            raise ValueError(f"测试集不存在或为空: {test_set_name}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"评估测试集: {test_set_name}")
        logger.info(f"任务数量: {len(task_ids)}")
        logger.info(f"任务列表: {', '.join(task_ids)}")
        logger.info(f"{'='*80}\n")
        
        return self.evaluate_task_list(
            task_ids=task_ids,
            n_trials=n_trials,
            max_steps=max_steps
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
        
        logger.info(f"\n{'='*80}")
        logger.info(f"报告已生成:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  TXT:  {txt_path}")
        logger.info(f"{'='*80}\n")
        
        return str(json_path), str(txt_path)
    
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
            
            # 每个任务的详情
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
                    f.write(f"    Trial {i}: {status} | 步数: {trial['steps']:4d} | 时间: {trial['time_seconds']:.1f}s\n")
                f.write("\n")
    
    def close(self):
        """清理资源"""
        if self.evaluator:
            self.evaluator.close()
        logger.info("评估框架已关闭")


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='STEVE-1 评估框架')
    parser.add_argument(
        '--task',
        type=str,
        help='评估单个任务（任务ID）'
    )
    parser.add_argument(
        '--test-set',
        type=str,
        choices=['quick_test', 'baseline_test'],
        help='评估测试集'
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
        help='启用渲染'
    )
    parser.add_argument(
        '--report-name',
        type=str,
        default='evaluation_report',
        help='报告名称'
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvaluationConfig(
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        enable_render=args.render
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
        
        elif args.test_set:
            # 测试集
            results = framework.evaluate_test_set(args.test_set)
        
        elif args.task_list:
            # 任务列表
            results = framework.evaluate_task_list(args.task_list)
        
        else:
            # 默认：快速测试
            logger.info("未指定任务，运行快速测试...")
            results = framework.evaluate_test_set('quick_test')
        
        # 打印摘要
        framework.print_summary(results)
        
        # 生成报告
        framework.generate_report(results, args.report_name)
        
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        framework.close()
