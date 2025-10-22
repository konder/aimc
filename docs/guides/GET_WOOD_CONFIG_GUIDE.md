# Get Wood 训练配置指南

## 概述

`train_get_wood` 训练脚本已重构为通过 YAML 配置文件管理所有训练参数，简化了使用流程并提高了配置的可维护性。

## 配置文件

### 默认配置文件

- 位置: `config/get_wood_config.yaml`
- 包含所有训练参数的默认值
- 支持预设配置场景 (test/quick/standard/long)

### 配置文件结构

```yaml
# 任务配置
task:
  task_id: "harvest_1_log"  # 任务ID
  image_size: [160, 256]     # 图像尺寸

# 训练配置
training:
  total_timesteps: 200000    # 总训练步数
  device: "auto"             # 设备
  learning_rate: 0.0005      # 学习率
  resume: true               # 自动恢复训练
  ppo:                       # PPO超参数
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99

# MineCLIP 配置
mineclip:
  use_mineclip: false        # 是否启用MineCLIP
  model_path: "data/mineclip/attn.pth"
  variant: "attn"
  sparse_weight: 10.0
  mineclip_weight: 10.0
  use_dynamic_weight: true
  weight_decay_steps: 50000
  min_weight: 0.1
  use_video_mode: true
  num_frames: 16
  compute_frequency: 4

# 相机控制配置
camera:
  use_smoothing: true        # 启用相机平滑
  max_camera_change: 12.0    # 最大角度变化

# 保存和日志配置
checkpointing:
  save_freq: 10000
  checkpoint_dir: "checkpoints/get_wood"

logging:
  log_dir: "logs/training"
  tensorboard_dir: "logs/tensorboard"
  save_frames: false
  frames_dir: "logs/frames"

# 预设配置
presets:
  test:
    total_timesteps: 10000
    save_freq: 5000
  quick:
    total_timesteps: 50000
    save_freq: 10000
  standard:
    total_timesteps: 200000
    save_freq: 10000
  long:
    total_timesteps: 500000
    save_freq: 20000
```

## 使用方法

### 1. 基础使用

#### 使用默认配置
```bash
bash scripts/train_get_wood.sh
```

#### 使用预设配置
```bash
# 快速测试 (10K步)
bash scripts/train_get_wood.sh test

# 快速训练 (50K步)
bash scripts/train_get_wood.sh quick

# 标准训练 (200K步)
bash scripts/train_get_wood.sh standard

# 长时间训练 (500K步)
bash scripts/train_get_wood.sh long
```

### 2. 启用MineCLIP加速训练

```bash
# 标准训练 + MineCLIP
bash scripts/train_get_wood.sh standard --mineclip

# 快速测试 + MineCLIP
bash scripts/train_get_wood.sh test --mineclip
```

### 3. 修改任务和生物群系

```bash
# 使用森林生物群系（树木更密集，推荐）
bash scripts/train_get_wood.sh --task-id harvest_1_log_forest --mineclip

# 使用平原生物群系
bash scripts/train_get_wood.sh --task-id harvest_1_log_plains
```

### 4. 高级用法：覆盖配置参数

```bash
# 覆盖学习率
bash scripts/train_get_wood.sh quick --override training.learning_rate=0.001

# 覆盖多个参数
bash scripts/train_get_wood.sh test \
  --override mineclip.use_mineclip=true \
  --override mineclip.mineclip_weight=20.0

# 禁用相机平滑
bash scripts/train_get_wood.sh --override camera.use_smoothing=false
```

### 5. 使用自定义配置文件

```bash
# 创建自定义配置文件
cp config/get_wood_config.yaml config/my_config.yaml
# 编辑 my_config.yaml...

# 使用自定义配置
bash scripts/train_get_wood.sh --config config/my_config.yaml
```

## 直接使用 Python 脚本

如果不需要 Shell 脚本的便利功能（如自动启动 TensorBoard），可以直接调用 Python 脚本：

```bash
# 基础用法
python src/training/train_get_wood.py config/get_wood_config.yaml

# 使用预设
python src/training/train_get_wood.py config/get_wood_config.yaml --preset test

# 覆盖参数
python src/training/train_get_wood.py config/get_wood_config.yaml \
  --preset quick \
  --override mineclip.use_mineclip=true

# 查看帮助
python src/training/train_get_wood.py --help
```

## 常见配置场景

### 场景1: 快速验证代码
```bash
bash scripts/train_get_wood.sh test
```
- 10K步，5-10分钟
- 验证环境配置是否正确

### 场景2: MineCLIP加速训练（推荐）
```bash
bash scripts/train_get_wood.sh standard \
  --task-id harvest_1_log_forest \
  --mineclip
```
- 200K步，2-4小时
- 使用MineCLIP密集奖励
- 森林生物群系，树木更密集

### 场景3: 长时间高性能训练
```bash
bash scripts/train_get_wood.sh long \
  --mineclip \
  --override training.learning_rate=0.001 \
  --override mineclip.mineclip_weight=15.0
```
- 500K步，5-10小时
- 更高的学习率和MineCLIP权重

### 场景4: 调试模式（显示游戏窗口）
```bash
bash scripts/train_get_wood.sh test --no-headless
```
- 禁用无头模式，可以看到游戏画面
- 适合调试和观察AI行为

## 配置参数说明

### 训练参数
- `total_timesteps`: 总训练步数，建议 50K-500K
- `learning_rate`: 学习率，默认 0.0005
- `device`: 设备选择 (auto/cpu/cuda/mps)
- `resume`: 是否自动恢复训练（默认 true）

### MineCLIP参数
- `use_mineclip`: 是否启用MineCLIP密集奖励
- `sparse_weight`: 稀疏奖励权重（默认 10.0）
- `mineclip_weight`: MineCLIP初始权重（默认 10.0）
- `use_dynamic_weight`: 是否使用动态权重衰减（课程学习）
- `weight_decay_steps`: 权重衰减步数（默认 50000）
- `use_video_mode`: 是否使用16帧视频模式（推荐，默认 true）

### 相机参数
- `use_smoothing`: 是否启用相机平滑（默认 true）
- `max_camera_change`: 相机最大角度变化（默认 12.0°/步）

## 迁移指南

如果你之前使用的是旧版本的命令行参数方式，可以参考以下对照表：

| 旧参数                    | 新配置位置                        |
|--------------------------|----------------------------------|
| `--task-id`              | `task.task_id`                   |
| `--total-timesteps`      | `training.total_timesteps`       |
| `--learning-rate`        | `training.learning_rate`         |
| `--device`               | `training.device`                |
| `--use-mineclip`         | `mineclip.use_mineclip`          |
| `--sparse-weight`        | `mineclip.sparse_weight`         |
| `--mineclip-weight`      | `mineclip.mineclip_weight`       |
| `--camera-smoothing`     | `camera.use_smoothing`           |
| `--max-camera-change`    | `camera.max_camera_change`       |
| `--save-freq`            | `checkpointing.save_freq`        |
| `--checkpoint-dir`       | `checkpointing.checkpoint_dir`   |

## 故障排除

### 问题1: 找不到配置文件
```
❌ 错误: 配置文件不存在: config/get_wood_config.yaml
```
**解决**: 确保在项目根目录运行脚本，或使用 `--config` 指定正确的配置文件路径。

### 问题2: YAML解析错误
```
❌ YAML解析错误: ...
```
**解决**: 检查YAML文件语法，确保缩进正确（使用空格，不是Tab）。

### 问题3: 配置键缺失
检查配置文件是否包含所有必需的字段，可以参考默认配置文件 `config/get_wood_config.yaml`。

## 最佳实践

1. **使用版本控制**: 为不同实验创建不同的配置文件，便于追踪
2. **预设优先**: 优先使用预设配置，通过 `--override` 调整个别参数
3. **记录实验**: 在配置文件顶部添加注释，说明实验目的
4. **渐进式调整**: 先用 `test` 预设验证配置，再切换到更长的训练

## 相关文档

- [训练快速开始指南](QUICK_START.md)
- [MineCLIP设置指南](MINECLIP_SETUP_GUIDE.md)
- [训练加速指南](TRAINING_ACCELERATION_GUIDE.md)

