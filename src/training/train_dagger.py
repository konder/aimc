#!/usr/bin/env python3
"""
DAgger主循环 - 自动化迭代训练

整合状态收集、数据聚合和重新训练的完整DAgger流程。
可以手动逐步执行，也可以自动化运行多轮迭代。

Usage:
    # 手动模式: 单轮迭代
    python src/training/train_dagger.py \
        --iteration 1 \
        --base-data data/expert_demos/round_0/ \
        --new-data data/expert_labels/iter_1.pkl \
        --output checkpoints/dagger_iter_1.zip

    # 自动模式: 多轮迭代（需要人工标注）
    python src/training/train_dagger.py \
        --auto \
        --initial-model checkpoints/bc_round_0.zip \
        --iterations 3 \
        --output-dir checkpoints/dagger/
"""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_bc import load_expert_demonstrations, train_bc_with_ppo
from tools.run_policy_collect_states import collect_policy_states
from tools.evaluate_policy import evaluate_policy


def aggregate_data(base_data_path, new_data_path, output_path):
    """
    聚合基础数据和新标注数据
    
    Args:
        base_data_path: 基础数据路径（目录或.pkl）
        new_data_path: 新标注数据路径（.pkl）
        output_path: 输出聚合数据路径（.pkl）
    
    Returns:
        (observations, actions): 聚合后的数据
    """
    
    print(f"\n{'='*60}")
    print(f"数据聚合")
    print(f"{'='*60}")
    
    # 加载基础数据
    print(f"加载基础数据: {base_data_path}")
    base_obs, base_actions = load_expert_demonstrations(base_data_path)
    print(f"  基础数据: {len(base_obs)} 样本")
    
    # 加载新标注数据
    print(f"加载新标注数据: {new_data_path}")
    with open(new_data_path, 'rb') as f:
        new_labeled = pickle.load(f)
    
    new_obs = np.array([item['observation'] for item in new_labeled])
    new_actions = np.array([item['expert_action'] for item in new_labeled])
    print(f"  新标注: {len(new_obs)} 样本")
    
    # 聚合
    all_obs = np.concatenate([base_obs, new_obs], axis=0)
    all_actions = np.concatenate([base_actions, new_actions], axis=0)
    
    print(f"  聚合后: {len(all_obs)} 样本")
    print(f"{'='*60}\n")
    
    # 保存聚合数据
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        aggregated_data = []
        for obs, action in zip(all_obs, all_actions):
            aggregated_data.append({
                'observation': obs,
                'expert_action': action
            })
        
        with open(output_path, 'wb') as f:
            pickle.dump(aggregated_data, f)
        
        print(f"✓ 聚合数据已保存: {output_path}\n")
    
    return all_obs, all_actions


def run_dagger_iteration(
    iteration,
    current_model,
    base_data_path,
    output_dir,
    task_id="harvest_1_log",
    num_episodes=20,
    learning_rate=3e-4,
    epochs=30
):
    """
    运行单轮DAgger迭代
    
    流程:
    1. 评估当前策略
    2. 运行策略收集失败状态
    3. 等待人工标注
    4. 聚合数据
    5. 重新训练
    6. 评估新策略
    
    Args:
        iteration: 迭代轮次（1, 2, 3, ...）
        current_model: 当前策略模型路径
        base_data_path: 基础数据路径
        output_dir: 输出目录
        task_id: MineDojo任务ID
        num_episodes: 收集的episode数量
        learning_rate: 学习率
        epochs: 训练轮数
    """
    
    print(f"\n{'='*70}")
    print(f"DAgger 迭代 {iteration}")
    print(f"{'='*70}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    states_dir = os.path.join(output_dir, f"states_iter_{iteration}")
    labels_file = os.path.join(output_dir, f"labels_iter_{iteration}.pkl")
    combined_file = os.path.join(output_dir, f"combined_iter_{iteration}.pkl")
    new_model_file = os.path.join(output_dir, f"dagger_iter_{iteration}.zip")
    
    # 步骤1: 评估当前策略
    print(f"[步骤1/{6}] 评估当前策略")
    print(f"-" * 70)
    eval_results = evaluate_policy(
        model_path=current_model,
        num_episodes=10,
        task_id=task_id
    )
    
    if eval_results:
        current_success_rate = eval_results['success_rate']
        print(f"当前成功率: {current_success_rate*100:.1f}%\n")
        
        if current_success_rate >= 0.90:
            print(f"✓ 成功率已达到90%，无需继续迭代")
            return current_model, True  # 返回True表示已收敛
    
    # 步骤2: 运行策略收集状态
    print(f"[步骤2/6] 运行策略收集失败状态")
    print(f"-" * 70)
    collect_stats = collect_policy_states(
        model_path=current_model,
        num_episodes=num_episodes,
        output_dir=states_dir,
        task_id=task_id,
        save_failures_only=True,  # 只保存失败的episode
        max_steps=1000
    )
    
    if not collect_stats or collect_stats['saved_episodes'] == 0:
        print(f"⚠️  未收集到失败episode，迭代终止")
        return current_model, True
    
    # 步骤3: 等待人工标注
    print(f"[步骤3/6] 等待人工标注")
    print(f"-" * 70)
    print(f"请运行以下命令进行标注:")
    print(f"\n  python tools/label_states.py \\")
    print(f"      --states {states_dir} \\")
    print(f"      --output {labels_file} \\")
    print(f"      --smart-sampling\n")
    
    input(f"完成标注后，按Enter继续...")
    
    # 验证标注文件存在
    if not os.path.exists(labels_file):
        print(f"✗ 错误: 标注文件不存在: {labels_file}")
        print(f"  请先完成标注！")
        return current_model, False
    
    # 步骤4: 聚合数据
    print(f"\n[步骤4/6] 聚合数据")
    print(f"-" * 70)
    
    # 更新base_data_path: 如果是第一轮，使用原始数据；否则使用上一轮的聚合数据
    if iteration > 1:
        prev_combined = os.path.join(output_dir, f"combined_iter_{iteration-1}.pkl")
        if os.path.exists(prev_combined):
            base_data_path = prev_combined
    
    all_obs, all_actions = aggregate_data(
        base_data_path=base_data_path,
        new_data_path=labels_file,
        output_path=combined_file
    )
    
    # 步骤5: 重新训练
    print(f"[步骤5/6] 重新训练策略")
    print(f"-" * 70)
    new_model = train_bc_with_ppo(
        observations=all_obs,
        actions=all_actions,
        output_path=new_model_file,
        task_id=task_id,
        learning_rate=learning_rate,
        n_epochs=epochs
    )
    
    # 步骤6: 评估新策略
    print(f"[步骤6/6] 评估新策略")
    print(f"-" * 70)
    new_eval_results = evaluate_policy(
        model_path=new_model_file,
        num_episodes=20,
        task_id=task_id
    )
    
    if new_eval_results:
        new_success_rate = new_eval_results['success_rate']
        improvement = (new_success_rate - current_success_rate) * 100
        
        print(f"\n{'='*70}")
        print(f"迭代 {iteration} 完成")
        print(f"{'='*70}")
        print(f"成功率: {current_success_rate*100:.1f}% → {new_success_rate*100:.1f}% (+{improvement:.1f}%)")
        print(f"新模型: {new_model_file}")
        print(f"{'='*70}\n")
    
    return new_model_file, False  # 返回False表示未收敛


def main():
    parser = argparse.ArgumentParser(
        description="DAgger训练 - Dataset Aggregation迭代优化"
    )
    
    # 手动模式参数
    parser.add_argument(
        "--iteration",
        type=int,
        help="迭代轮次（手动模式）"
    )
    
    parser.add_argument(
        "--base-data",
        type=str,
        help="基础数据路径（手动模式）"
    )
    
    parser.add_argument(
        "--new-data",
        type=str,
        help="新标注数据路径（手动模式，.pkl文件）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="输出模型路径（手动模式）"
    )
    
    # 自动模式参数
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动模式（多轮迭代）"
    )
    
    parser.add_argument(
        "--initial-model",
        type=str,
        help="初始BC模型路径（自动模式）"
    )
    
    parser.add_argument(
        "--initial-data",
        type=str,
        help="初始专家数据路径（自动模式）"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="迭代次数（自动模式，默认: 3）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dagger",
        help="输出目录（自动模式，默认: data/dagger）"
    )
    
    # 通用参数
    parser.add_argument(
        "--task-id",
        type=str,
        default="harvest_1_log",
        help="MineDojo任务ID（默认: harvest_1_log）"
    )
    
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="每轮收集的episode数（默认: 20）"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="学习率（默认: 3e-4）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="训练轮数（默认: 30）"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        # 自动模式: 多轮迭代
        print(f"\n🔄 DAgger自动模式")
        print(f"迭代次数: {args.iterations}")
        print(f"输出目录: {args.output_dir}")
        
        if not args.initial_model or not args.initial_data:
            print(f"✗ 自动模式需要 --initial-model 和 --initial-data")
            sys.exit(1)
        
        current_model = args.initial_model
        
        for i in range(1, args.iterations + 1):
            current_model, converged = run_dagger_iteration(
                iteration=i,
                current_model=current_model,
                base_data_path=args.initial_data,
                output_dir=args.output_dir,
                task_id=args.task_id,
                num_episodes=args.num_episodes,
                learning_rate=args.learning_rate,
                epochs=args.epochs
            )
            
            if converged:
                print(f"✓ DAgger已收敛，停止迭代")
                break
        
        print(f"\n✓ DAgger训练完成！")
        print(f"最终模型: {current_model}\n")
    
    else:
        # 手动模式: 单轮迭代
        if not all([args.iteration, args.base_data, args.new_data, args.output]):
            print(f"✗ 手动模式需要: --iteration, --base-data, --new-data, --output")
            sys.exit(1)
        
        # 聚合数据
        all_obs, all_actions = aggregate_data(
            base_data_path=args.base_data,
            new_data_path=args.new_data,
            output_path=None  # 不保存中间文件
        )
        
        # 训练
        train_bc_with_ppo(
            observations=all_obs,
            actions=all_actions,
            output_path=args.output,
            task_id=args.task_id,
            learning_rate=args.learning_rate,
            n_epochs=args.epochs
        )
        
        print(f"✓ 训练完成！")
        print(f"模型: {args.output}\n")


if __name__ == "__main__":
    main()

