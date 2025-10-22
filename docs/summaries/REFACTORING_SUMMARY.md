# Train Get Wood 脚本重构总结

## 重构概述

成功将 `train_get_wood` 训练脚本重构为基于 YAML 配置文件的方式，简化了参数管理和使用流程。

## 主要变更

### 1. 新增文件

- **`config/get_wood_config.yaml`**: 专门用于 get_wood 任务的 YAML 配置文件
- **`docs/guides/GET_WOOD_CONFIG_GUIDE.md`**: 详细的配置使用指南

### 2. 修改文件

#### `src/training/train_get_wood.py`
- ✅ 添加 `load_config()` 函数，支持从 YAML 文件加载配置
- ✅ 重构 `train()` 函数，接受配置字典而非 argparse 参数
- ✅ 重写 `main()` 函数，仅接受一个必需参数：配置文件路径
- ✅ 支持可选的 `--preset` 参数（test/quick/standard/long）
- ✅ 支持可选的 `--override` 参数，用于覆盖配置文件中的任何参数

#### `scripts/train_get_wood.sh`
- ✅ 简化命令行参数，移除大量具体训练参数
- ✅ 保留预设快捷方式（test/quick/standard/long）
- ✅ 添加 `--config` 参数，支持指定自定义配置文件
- ✅ 添加 `--override` 参数支持
- ✅ 添加常用快捷方式（`--mineclip`, `--task-id`）
- ✅ 更新帮助信息

## 使用方式对比

### 旧方式（命令行参数）
```bash
# 复杂的命令行
bash scripts/train_get_wood.sh \
    --task-id harvest_1_log_forest \
    --total-timesteps 200000 \
    --learning-rate 0.0005 \
    --use-mineclip \
    --sparse-weight 10.0 \
    --mineclip-weight 10.0 \
    --device auto \
    --save-freq 10000 \
    --checkpoint-dir checkpoints/get_wood
```

### 新方式（YAML配置）
```bash
# 简洁的命令行
bash scripts/train_get_wood.sh standard --mineclip --task-id harvest_1_log_forest

# 或直接使用 Python 脚本
python src/training/train_get_wood.py config/get_wood_config.yaml --preset standard
```

## 主要优势

1. **配置管理更清晰**: 所有参数集中在 YAML 文件中，易于查看和修改
2. **命令行更简洁**: 常用操作只需简单命令
3. **预设配置**: 内置 test/quick/standard/long 四种预设，开箱即用
4. **灵活覆盖**: 支持通过 `--override` 覆盖任意配置参数
5. **易于版本控制**: 可以为不同实验创建不同的配置文件
6. **向后兼容**: Shell 脚本仍支持常用参数的快捷方式

## 快速开始

### 1. 使用默认配置
```bash
bash scripts/train_get_wood.sh
```

### 2. 快速测试
```bash
bash scripts/train_get_wood.sh test
```

### 3. 启用 MineCLIP 训练
```bash
bash scripts/train_get_wood.sh standard --mineclip
```

### 4. 覆盖特定参数
```bash
bash scripts/train_get_wood.sh quick \
  --override training.learning_rate=0.001 \
  --override mineclip.use_mineclip=true
```

### 5. 使用自定义配置文件
```bash
bash scripts/train_get_wood.sh --config my_custom_config.yaml
```

## 配置文件结构

```yaml
task:           # 任务配置
training:       # 训练参数
  ppo:          # PPO超参数
mineclip:       # MineCLIP配置
camera:         # 相机控制
checkpointing:  # 检查点保存
logging:        # 日志配置
presets:        # 预设配置
```

## 测试验证

✅ YAML 配置文件语法正确  
✅ 配置文件可正确加载  
✅ Shell 脚本帮助信息正确  
✅ Python 脚本语法无错误  
✅ 参数覆盖功能正常  
✅ 预设配置功能正常  

## 迁移建议

如果你有现有的训练脚本或命令：

1. **查看默认配置**: 打开 `config/get_wood_config.yaml` 了解配置结构
2. **使用预设**: 优先使用 test/quick/standard/long 预设
3. **创建自定义配置**: 复制默认配置文件，修改需要的参数
4. **使用快捷方式**: 对于常用参数（如 `--mineclip`），使用快捷方式

## 相关文档

- 详细使用指南: `docs/guides/GET_WOOD_CONFIG_GUIDE.md`
- 默认配置文件: `config/get_wood_config.yaml`
- 训练脚本: `src/training/train_get_wood.py`
- Shell 脚本: `scripts/train_get_wood.sh`

## 技术细节

### Python 脚本改动
- 添加 `yaml` 模块导入
- 添加 `load_config()` 函数
- `train()` 函数从接受 args 对象改为接受 config 字典
- 支持嵌套配置覆盖（如 `mineclip.use_mineclip=true`）
- 自动类型推断（字符串、整数、浮点数、布尔值）

### Shell 脚本改动
- 移除大量具体参数的命令行选项
- 添加配置文件和预设支持
- 简化参数传递逻辑
- 保留常用功能的快捷方式

## 注意事项

1. **YAML 语法**: 确保使用空格缩进，不是 Tab
2. **路径**: 配置文件中的路径相对于项目根目录
3. **覆盖格式**: 使用点号分隔嵌套键，如 `section.subsection.key=value`
4. **布尔值**: 覆盖布尔值时使用 `true`/`false`（小写）

---

**重构完成日期**: 2025-10-21  
**兼容性**: 向后兼容，保留常用快捷方式

