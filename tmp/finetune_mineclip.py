#!/usr/bin/env python3
"""
MineCLIP微调 - 主脚本

使用专家演示微调MineCLIP，使其能够识别砍树任务的进度

用法:
    python tmp/finetune_mineclip.py \
        --expert-dir data/tasks/harvest_1_log/expert_demos \
        --base-model data/mineclip/attn.pth \
        --output data/mineclip/attn_finetuned_harvest.pth \
        --epochs 20 \
        --batch-size 16
"""

import sys
import argparse
from pathlib import Path
import torch

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tmp.finetune_mineclip_data import create_dataloader
from tmp.finetune_mineclip_trainer_v2 import MineCLIPFineTunerV2 as MineCLIPFineTuner


def main():
    parser = argparse.ArgumentParser(description="微调MineCLIP用于砍树任务")
    
    # 数据参数
    parser.add_argument(
        "--expert-dir",
        type=str,
        default="data/tasks/harvest_1_log/expert_demos",
        help="专家演示目录"
    )
    
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多使用的episode数量（None=全部）"
    )
    
    # 模型参数
    parser.add_argument(
        "--base-model",
        type=str,
        default="data/mineclip/attn.pth",
        help="预训练MineCLIP模型路径"
    )
    
    parser.add_argument(
        "--freeze-ratio",
        type=float,
        default=0.0,
        help="冻结底层的比例（0.0=不冻结，0.5=冻结50%）"
    )
    
    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批次大小"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    
    parser.add_argument(
        "--temporal-gap",
        type=int,
        default=10,
        help="时序间隔（正负样本的帧数差距）"
    )
    
    # 输出参数
    parser.add_argument(
        "--output",
        type=str,
        default="data/mineclip/attn_finetuned_harvest.pth",
        help="输出模型路径"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="tmp/mineclip_checkpoints",
        help="检查点保存目录"
    )
    
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="每N个epoch保存一次检查点"
    )
    
    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="运行设备 (auto/cuda/mps/cpu)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载进程数"
    )
    
    args = parser.parse_args()
    
    # 打印配置
    print(f"\n{'='*70}")
    print(f"MineCLIP微调 - 砍树任务")
    print(f"{'='*70}")
    print(f"配置:")
    print(f"  专家演示: {args.expert_dir}")
    print(f"  最大episodes: {args.max_episodes or '全部'}")
    print(f"  基础模型: {args.base_model}")
    print(f"  输出模型: {args.output}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  冻结比例: {args.freeze_ratio*100:.0f}%")
    print(f"  时序间隔: {args.temporal_gap}")
    print(f"  设备: {args.device}")
    print(f"{'='*70}\n")
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器
    print("="*70)
    print("步骤1: 创建数据加载器")
    print("="*70)
    
    dataloader = create_dataloader(
        expert_demos_dir=args.expert_dir,
        batch_size=args.batch_size,
        temporal_gap=args.temporal_gap,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers
    )
    
    print(f"✓ 数据加载器创建完成")
    print(f"  总批次数: {len(dataloader)}")
    print()
    
    # 创建微调器
    print("="*70)
    print("步骤2: 创建微调器")
    print("="*70)
    
    trainer = MineCLIPFineTuner(
        base_model_path=args.base_model,
        device=args.device,
        freeze_ratio=args.freeze_ratio,
        learning_rate=args.learning_rate
    )
    
    print(f"✓ 微调器创建完成\n")
    
    # 训练
    print("="*70)
    print("步骤3: 开始训练")
    print("="*70)
    
    best_margin = -float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)
        
        # 训练一个epoch
        stats = trainer.train_epoch(dataloader, epoch)
        
        # 打印统计
        print(f"\nEpoch {epoch} 统计:")
        print(f"  Loss: {stats['loss']:.4f}")
        print(f"  Sim(Positive): {stats['sim_pos']:.3f}")
        print(f"  Sim(Negative): {stats['sim_neg']:.3f}")
        print(f"  Margin (Pos-Neg): {stats['margin']:.3f}")
        
        # 判断是否改善
        if stats['margin'] > best_margin:
            best_margin = stats['margin']
            print(f"  ✓ 最佳Margin更新: {best_margin:.3f}")
            
            # 保存最佳模型
            best_model_path = Path(args.checkpoint_dir) / "best_model.pth"
            trainer.save_checkpoint(best_model_path, epoch, stats)
        
        # 定期保存检查点
        if epoch % args.save_freq == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
            trainer.save_checkpoint(checkpoint_path, epoch, stats)
        
        print()
    
    # 训练完成
    print("="*70)
    print("训练完成！")
    print("="*70)
    
    # 保存最终模型
    print(f"\n保存最终模型...")
    final_stats = {
        'final_epoch': args.epochs,
        'best_margin': best_margin
    }
    trainer.save_checkpoint(args.output, args.epochs, final_stats)
    
    print(f"\n✓ 最终模型已保存: {args.output}")
    print(f"  最佳Margin: {best_margin:.3f}")
    
    # 使用建议
    print(f"\n{'='*70}")
    print(f"下一步:")
    print(f"{'='*70}")
    print(f"1. 测试微调后的MineCLIP:")
    print(f"   python tmp/test_mineclip_rewards.py \\")
    print(f"       --mineclip-model {args.output} \\")
    print(f"       --num-episodes 10")
    print()
    print(f"2. 如果效果好，在BC训练中使用:")
    print(f"   修改 src/training/train_bc.py")
    print(f"   使用模型: {args.output}")
    print()
    print(f"3. 查看训练日志和检查点:")
    print(f"   目录: {args.checkpoint_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

