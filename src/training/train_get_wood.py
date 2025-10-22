#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获得木头任务训练脚本 (MineCLIP 版本)

使用官方 MineCLIP 包和预训练权重
支持通过 YAML 配置文件配置训练参数
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

try:
    import minedojo
    MINEDOJO_AVAILABLE = True
except ImportError:
    MINEDOJO_AVAILABLE = False
    print("❌ MineDojo未安装")
    sys.exit(1)

from src.utils.realtime_logger import RealtimeLoggerCallback
from src.utils.env_wrappers import make_minedojo_env
from src.utils.mineclip_reward import MineCLIPRewardWrapper


def load_config(config_path):
    """
    加载 YAML 配置文件
    
    Args:
        config_path: YAML 配置文件路径
        
    Returns:
        config: 配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ 已加载配置文件: {config_path}")
    return config


def _detect_device(device_arg):
    """检测可用设备"""
    if device_arg != 'auto':
        return device_arg
    
    if torch.cuda.is_available():
        print("🚀 检测到CUDA GPU")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 检测到Apple Silicon MPS")
        return 'mps'
    else:
        print("💻 使用CPU")
        return 'cpu'


def create_harvest_log_env(task_id="harvest_1_log", use_mineclip=False, 
                          mineclip_model_path=None, mineclip_variant="attn", 
                          image_size=(160, 256), sparse_weight=10.0, 
                          mineclip_weight=10.0, use_dynamic_weight=True, 
                          weight_decay_steps=50000, min_weight=0.1, device='auto',
                          use_camera_smoothing=True, max_camera_change=12.0,
                          use_video_mode=True, num_frames=16, compute_frequency=4):
    """
    创建采集木头任务环境
    
    Args:
        task_id: MineDojo任务ID（如: harvest_1_log, harvest_1_log_forest）
        use_mineclip: 是否使用MineCLIP密集奖励
        mineclip_model_path: MineCLIP模型权重路径
        mineclip_variant: MineCLIP变体 ("attn" 或 "avg")
        image_size: 图像尺寸
        sparse_weight: 稀疏奖励权重
        mineclip_weight: MineCLIP奖励初始权重
        use_dynamic_weight: 是否使用动态权重调整（课程学习）
        weight_decay_steps: 权重衰减步数
        min_weight: MineCLIP权重最小值
        device: 运行设备 ('auto', 'cuda', 'mps', 'cpu')
        use_camera_smoothing: 是否启用相机平滑（减少抖动）
        max_camera_change: 相机最大角度变化（度/步）
        
    Returns:
        MineDojo环境
        
    Note:
        无头模式通过JAVA_OPTS环境变量在外部控制
    """
    import os
    
    # 任务显示名称映射
    task_names = {
        'harvest_1_log': '获得1个原木',
        'harvest_1_log_forest': '获得1个原木（森林）',
        'harvest_1_log_plains': '获得1个原木（平原）',
        'harvest_1_log_taiga': '获得1个原木（针叶林）',
    }
    task_name = task_names.get(task_id, task_id)
    
    print(f"创建环境: {task_id} ({task_name})")
    print(f"  图像尺寸: {image_size}")
    # 显示无头模式状态（从环境变量读取）
    java_opts = os.environ.get('JAVA_OPTS', '')
    headless_enabled = 'headless=true' in java_opts
    print(f"  无头模式: {'启用' if headless_enabled else '禁用'}")
    print(f"  MineCLIP: {'启用 (' + mineclip_variant + ')' if use_mineclip else '禁用'}")
    
    # 使用 env_wrappers 创建基础环境
    env = make_minedojo_env(
        task_id=task_id,
        image_size=image_size,
        use_frame_stack=False,
        use_discrete_actions=False,
        max_episode_steps=1000,  # 每回合最大1000步，防止无限运行
        use_camera_smoothing=use_camera_smoothing,
        max_camera_change=max_camera_change
    )
    
    # 如果启用MineCLIP
    if use_mineclip:
        # 任务描述优化：使用动作导向的描述for视频模式
        # 视频模式下，MineCLIP能理解动作和过程
        task_prompt = "punching tree" if use_video_mode else "tree"
        
        env = MineCLIPRewardWrapper(
            env,
            task_prompt=task_prompt,
            model_path=mineclip_model_path,
            variant=mineclip_variant,
            sparse_weight=sparse_weight,
            mineclip_weight=mineclip_weight,
            use_dynamic_weight=use_dynamic_weight,
            weight_decay_steps=weight_decay_steps,
            min_weight=min_weight,
            device=device,
            use_video_mode=use_video_mode,
            num_frames=num_frames,
            compute_frequency=compute_frequency
        )
    
    # 最后添加Monitor（必须在最外层！）
    # Monitor跟踪episode统计信息，必须能看到所有wrapper处理后的reward和done
    env = Monitor(env)
    print("  ✓ Monitor已添加（跟踪episode统计）")
    
    return env


def train(config):
    """
    主训练函数
    
    Args:
        config: 从 YAML 文件加载的配置字典
    """
    
    # 提取配置参数
    task_id = config['task']['task_id']
    image_size = tuple(config['task']['image_size'])
    
    total_timesteps = config['training']['total_timesteps']
    device_arg = config['training']['device']
    learning_rate = config['training']['learning_rate']
    resume = config['training']['resume']
    
    use_mineclip = config['mineclip']['use_mineclip']
    mineclip_model = config['mineclip']['model_path']
    mineclip_variant = config['mineclip']['variant']
    sparse_weight = config['mineclip']['sparse_weight']
    mineclip_weight = config['mineclip']['mineclip_weight']
    use_dynamic_weight = config['mineclip']['use_dynamic_weight']
    weight_decay_steps = config['mineclip']['weight_decay_steps']
    min_weight = config['mineclip']['min_weight']
    use_video_mode = config['mineclip']['use_video_mode']
    num_frames = config['mineclip']['num_frames']
    compute_frequency = config['mineclip']['compute_frequency']
    
    use_camera_smoothing = config['camera']['use_smoothing']
    max_camera_change = config['camera']['max_camera_change']
    
    save_freq = config['checkpointing']['save_freq']
    checkpoint_dir = config['checkpointing']['checkpoint_dir']
    
    log_dir = config['logging']['log_dir']
    tensorboard_dir = config['logging']['tensorboard_dir']
    save_frames = config['logging']['save_frames']
    frames_dir = config['logging']['frames_dir']
    
    # 检测设备
    device = _detect_device(device_arg)
    
    print("=" * 70)
    print(f"MineDojo 获得木头训练")
    print("=" * 70)
    print(f"配置:")
    print(f"  任务ID: {task_id}")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  设备: {device}")
    # 显示无头模式（从JAVA_OPTS环境变量读取）
    java_opts = os.environ.get('JAVA_OPTS', '')
    headless_enabled = 'headless=true' in java_opts
    print(f"  无头模式: {'启用 (JAVA_OPTS)' if headless_enabled else '禁用'}")
    print(f"  MineCLIP: {'启用' if use_mineclip else '禁用'}")
    if use_mineclip:
        print(f"  MineCLIP模型: {mineclip_model}")
        print(f"  MineCLIP变体: {mineclip_variant}")
        print(f"  稀疏权重: {sparse_weight}")
        print(f"  MineCLIP初始权重: {mineclip_weight}")
        print(f"  动态权重: {'启用' if use_dynamic_weight else '禁用'}")
        if use_dynamic_weight:
            print(f"    衰减步数: {weight_decay_steps:,}")
            print(f"    最小权重: {min_weight}")
        print(f"  视频模式: {'启用' if use_video_mode else '禁用'}")
        if use_video_mode:
            print(f"    帧数: {num_frames}")
            print(f"    计算频率: 每{compute_frequency}步")
    print(f"  学习率: {learning_rate}")
    print(f"  图像尺寸: {image_size}")
    print(f"  相机平滑: {'启用' if use_camera_smoothing else '禁用'}")
    if use_camera_smoothing:
        print(f"    最大变化: {max_camera_change}°/步")
    print("=" * 70)
    print()
    
    # 创建目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建环境
    print("[1/4] 创建环境...")
    env_instance = create_harvest_log_env(
        task_id=task_id,
        use_mineclip=use_mineclip,
        mineclip_model_path=mineclip_model if use_mineclip else None,
        mineclip_variant=mineclip_variant,
        image_size=image_size,
        sparse_weight=sparse_weight,
        mineclip_weight=mineclip_weight,
        use_dynamic_weight=use_dynamic_weight,
        weight_decay_steps=weight_decay_steps,
        min_weight=min_weight,
        device=device_arg,
        use_camera_smoothing=use_camera_smoothing,
        max_camera_change=max_camera_change,
        use_video_mode=use_video_mode,
        num_frames=num_frames,
        compute_frequency=compute_frequency
    )
    env = DummyVecEnv([lambda: env_instance])
    print("  ✓ 环境创建成功")
    print()
    
    # 创建或加载模型
    print("[2/4] 创建/加载PPO模型...")
    
    # 检查是否存在checkpoint（用于恢复训练）
    checkpoint_to_load = None
    
    if resume:
        # 自动检测最新checkpoint
        import glob
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "get_wood_*_steps.zip"))
        if checkpoints:
            # 按修改时间排序，取最新的
            checkpoint_to_load = max(checkpoints, key=os.path.getmtime)
            print(f"  🔄 检测到checkpoint: {os.path.basename(checkpoint_to_load)}")
        elif os.path.exists(os.path.join(checkpoint_dir, "get_wood_final.zip")):
            checkpoint_to_load = os.path.join(checkpoint_dir, "get_wood_final.zip")
            print(f"  🔄 检测到最终模型: get_wood_final.zip")
        elif os.path.exists(os.path.join(checkpoint_dir, "get_wood_interrupted.zip")):
            checkpoint_to_load = os.path.join(checkpoint_dir, "get_wood_interrupted.zip")
            print(f"  🔄 检测到中断模型: get_wood_interrupted.zip")
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        # 加载现有模型（继续训练）
        print(f"  ✅ 从checkpoint恢复训练...")
        model = PPO.load(
            checkpoint_to_load,
            env=env,
            device=device,
            tensorboard_log=tensorboard_dir
        )
        # 更新学习率（如果指定了新的）
        model.learning_rate = learning_rate
        print(f"  ✓ 模型加载成功，继续训练")
    else:
        # 创建新模型
        if resume:
            print(f"  ⚠️  未找到checkpoint，创建新模型")
        else:
            print(f"  🆕 创建新模型（从头开始）")
        
        # 重要：normalize_images=False 因为观察已经归一化到 [0, 1]
        policy_kwargs = dict(
            normalize_images=False
        )
        
        # 获取 PPO 超参数
        ppo_config = config['training']['ppo']
        
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=ppo_config['gamma'],
            device=device,
            verbose=1,
            tensorboard_log=tensorboard_dir,
            policy_kwargs=policy_kwargs
        )
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  ✓ 模型参数量: {total_params:,}")
    print()
    
    # 设置回调
    print("[3/4] 设置训练回调...")
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="get_wood"
    )
    
    logger_callback = RealtimeLoggerCallback(
        log_freq=100,
        verbose=1,
        save_frames=save_frames,
        frames_dir=frames_dir
    )
    
    callbacks = CallbackList([checkpoint_callback, logger_callback])
    print("  ✓ 回调设置完成")
    print()
    
    # 开始训练
    print("[4/4] 开始训练...")
    print("=" * 70)
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(checkpoint_dir, "get_wood_final.zip")
        model.save(final_model_path)
        print(f"\n最终模型已保存: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n训练被中断")
        interrupted_path = os.path.join(checkpoint_dir, "get_wood_interrupted.zip")
        model.save(interrupted_path)
        print(f"当前模型已保存: {interrupted_path}")
    
    finally:
        env.close()
        print("环境已关闭")


def main():
    """
    主函数：加载 YAML 配置文件并启动训练
    """
    parser = argparse.ArgumentParser(
        description="MineDojo 获得木头训练 (通过YAML配置)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置文件
  python src/training/train_get_wood.py config/get_wood_config.yaml
  
  # 使用预设配置（test/quick/standard/long）
  python src/training/train_get_wood.py config/get_wood_config.yaml --preset test
  
  # 覆盖特定参数
  python src/training/train_get_wood.py config/get_wood_config.yaml --preset quick --override mineclip.use_mineclip=true
        """
    )
    
    parser.add_argument('config', type=str,
                       help='YAML配置文件路径 (例如: config/get_wood_config.yaml)')
    parser.add_argument('--preset', type=str, choices=['test', 'quick', 'standard', 'long'],
                       help='使用预设配置场景')
    parser.add_argument('--override', type=str, action='append', dest='overrides',
                       help='覆盖配置参数 (格式: key.subkey=value，可多次使用)')
    
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ YAML解析错误: {e}")
        sys.exit(1)
    
    # 应用预设配置
    if args.preset:
        if 'presets' in config and args.preset in config['presets']:
            preset_config = config['presets'][args.preset]
            print(f"✓ 应用预设配置: {args.preset}")
            
            # 覆盖训练参数
            for key, value in preset_config.items():
                if key in config['training']:
                    config['training'][key] = value
                    print(f"  - {key}: {value}")
        else:
            print(f"⚠️  警告: 预设配置 '{args.preset}' 不存在，使用默认配置")
    
    # 应用命令行覆盖
    if args.overrides:
        print("✓ 应用命令行覆盖:")
        for override in args.overrides:
            try:
                key_path, value = override.split('=', 1)
                keys = key_path.split('.')
                
                # 解析值类型
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.replace('.', '').replace('-', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                
                # 设置嵌套配置值
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
                
                print(f"  - {key_path}: {value}")
            except ValueError:
                print(f"⚠️  警告: 无效的覆盖格式 '{override}'，应为 key.subkey=value")
    
    print()
    
    # 开始训练
    train(config)


if __name__ == "__main__":
    main()
