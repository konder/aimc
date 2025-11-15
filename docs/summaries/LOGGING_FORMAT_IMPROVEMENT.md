# 日志格式优化总结

**日期**: 2025-11-14  
**目的**: 优化日志格式，缩短模块名，过滤不需要的日志，提高可读性

---

## 🎯 优化目标

### 问题 1: 模块名太长导致不对齐

**修复前**:
```
2025-11-14 16:04:18,144 - src.evaluation.steve1_evaluator - INFO - STEVE-1 评估器初始化完成
2025-11-14 16:04:18,145 - src.envs.minerl_harvest_flatworld - INFO - 🌍 使用 FlatWorldGenerator
2025-11-14 16:04:18,146 - __main__ - INFO - 评估框架初始化完成
```

**修复后**:
```
2025-11-14 16:04:18,144 - s.ev.steve1          - INFO    - STEVE-1 评估器初始化完成
2025-11-14 16:04:18,145 - s.en.flat           - INFO    - 🌍 使用 FlatWorldGenerator
2025-11-14 16:04:18,146 - main                - INFO    - 评估框架初始化完成
```

### 问题 2: 不需要的日志输出

**过滤的日志**:
- `process_watcher` - 进程监控日志（太频繁）
- `minerl.env.malmo.instance.*` - Malmo 实例日志（调试信息）

---

## ✅ 实现方案

### 1. 自定义日志格式化器

**文件**: `src/utils/logging_config.py`

**功能**: 缩短模块名

```python
class ShortModuleFormatter(logging.Formatter):
    """
    缩短模块名
    
    映射规则:
    - src.evaluation.steve1_evaluator → s.ev.steve1
    - src.envs.minerl_harvest_flatworld → s.en.flat
    - __main__ → main
    """
```

**缩写规则**:
| 原模块名 | 缩写 | 规则 |
|---------|------|------|
| `src.evaluation.steve1_evaluator` | `s.ev.steve1` | 预定义映射 |
| `src.envs.minerl_harvest_flatworld` | `s.en.flat` | 预定义映射 |
| `src.utils.steve1_mineclip_agent_env_utils` | `s.ut.env_utils` | 预定义映射 |
| `__main__` | `main` | 预定义映射 |
| 其他 `src.xxx.yyy` | `s.xx.yyy` | 自动缩写 |

### 2. 日志过滤器

**功能**: 过滤不需要的模块

```python
class ModuleFilter(logging.Filter):
    """
    过滤黑名单中的模块
    """
    BLOCKED_MODULES = [
        'process_watcher',
        'minerl.env.malmo.instance',
    ]
```

### 3. 统一日志配置函数

```python
def setup_evaluation_logging():
    """
    为评估框架配置日志
    
    - 固定宽度的模块名（20字符）
    - 自动缩短模块名
    - 过滤不需要的日志
    """
    format_string = '%(asctime)s - %(name)-20s - %(levelname)-7s - %(message)s'
    setup_logging(level=logging.INFO, format_string=format_string)
```

---

## 📊 模块名映射表

### 预定义映射

| 原模块名 | 缩写 | 说明 |
|---------|------|------|
| `src.evaluation.steve1_evaluator` | `s.ev.steve1` | Steve1 评估器 |
| `src.evaluation.eval_framework` | `s.ev.framework` | 评估框架 |
| `src.evaluation.task_loader` | `s.ev.task_ld` | 任务加载器 |
| `src.evaluation.report_generator` | `s.ev.report` | 报告生成器 |
| `src.envs.minerl_harvest_flatworld` | `s.en.flat` | FlatWorld 环境 |
| `src.envs.minerl_harvest_default` | `s.en.default` | Default 环境 |
| `src.utils.steve1_mineclip_agent_env_utils` | `s.ut.env_utils` | 环境工具 |
| `src.utils.minerl_cleanup` | `s.ut.cleanup` | 清理工具 |
| `src.translation.translator` | `s.tr.translator` | 翻译器 |
| `__main__` | `main` | 主程序 |

### 自动缩写规则

对于没有预定义映射的模块，使用自动缩写：

```
src.evaluation.xxx → s.ev.xxx
src.envs.xxx → s.en.xxx
src.utils.xxx → s.ut.xxx
src.translation.xxx → s.tr.xxx
```

---

## 🔧 集成

### 1. eval_framework.py

在 `_setup_log_filters()` 中配置：

```python
def _setup_log_filters(self):
    """配置日志系统：格式化、过滤器等"""
    from src.utils.logging_config import setup_evaluation_logging
    
    # 配置统一的日志格式和过滤器
    setup_evaluation_logging()
    
    # ... 其他过滤器配置 ...
```

### 2. utils/__init__.py

导出日志配置函数：

```python
from .logging_config import setup_evaluation_logging

__all__ = [
    ...,
    'setup_evaluation_logging',
]
```

---

## 📋 日志格式对比

### 修复前

```
2025-11-14 16:04:18,100 - __main__ - INFO - ==============================
2025-11-14 16:04:18,100 - __main__ - INFO - 调度器加载...
2025-11-14 16:04:18,100 - __main__ - INFO - ==============================
2025-11-14 16:04:18,143 - __main__ - INFO - 加载任务配置: 28 个任务
2025-11-14 16:04:18,144 - __main__ - INFO - 创建 STEVE-1 评估器...
2025-11-14 16:04:18,144 - src.evaluation.steve1_evaluator - INFO - STEVE-1 评估器初始化完成
2025-11-14 16:04:18,144 - src.evaluation.steve1_evaluator - INFO -   视频录制: 启用 (尺寸: 128x128)
2025-11-14 16:04:23,100 - src.envs.minerl_harvest_flatworld - INFO - 🌍 使用 FlatWorldGenerator
2025-11-14 16:04:23,100 - src.envs.minerl_harvest_flatworld - INFO -   generatorString: minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block;minecraft:plains
2025-11-14 16:05:01,451 - minerl.env.malmo.instance.2d8fc7 - ERROR - java.lang.IllegalStateException: Tried to release unknown channel
```

**问题**:
- ❌ 模块名长度不一，不对齐
- ❌ 不需要的 malmo 日志

### 修复后

```
2025-11-14 16:04:18,100 - main                - INFO    - ==============================
2025-11-14 16:04:18,100 - main                - INFO    - 调度器加载...
2025-11-14 16:04:18,100 - main                - INFO    - ==============================
2025-11-14 16:04:18,143 - main                - INFO    - 加载任务配置: 28 个任务
2025-11-14 16:04:18,144 - main                - INFO    - 创建 STEVE-1 评估器...
2025-11-14 16:04:18,144 - s.ev.steve1         - INFO    - STEVE-1 评估器初始化完成
2025-11-14 16:04:18,144 - s.ev.steve1         - INFO    -   视频录制: 启用 (尺寸: 128x128)
2025-11-14 16:04:23,100 - s.en.flat           - INFO    - 🌍 使用 FlatWorldGenerator
2025-11-14 16:04:23,100 - s.en.flat           - INFO    -   generatorString: minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block;minecraft:plains
```

**改进**:
- ✅ 模块名固定 20 字符宽度，完美对齐
- ✅ 模块名简洁易读
- ✅ 过滤掉不需要的 malmo 日志
- ✅ 过滤掉 process_watcher 日志

---

## 🎨 日志格式说明

### 格式字符串

```python
'%(asctime)s - %(name)-20s - %(levelname)-7s - %(message)s'
```

**字段说明**:
- `%(asctime)s`: 时间戳
- `%(name)-20s`: 模块名（左对齐，20字符宽度）
- `%(levelname)-7s`: 日志级别（左对齐，7字符宽度）
- `%(message)s`: 日志消息

### 宽度设置

| 字段 | 宽度 | 理由 |
|------|------|------|
| 模块名 | 20 字符 | 足够容纳缩写后的模块名 |
| 日志级别 | 7 字符 | 最长级别是 `WARNING`（7 字符） |

---

## 🧪 验证

### 测试日志输出

```bash
./scripts/run_minedojo_x86.sh python -m src.evaluation.eval_framework \
  --config config/eval_tasks.yaml \
  --task harvest_1_dirt \
  --n-trials 1
```

**预期效果**:
- ✅ 所有日志整齐对齐
- ✅ 模块名简短易读
- ✅ 无 `process_watcher` 日志
- ✅ 无 `minerl.env.malmo.instance` 日志

---

## 📝 扩展：添加新模块映射

如果需要为新模块添加缩写，编辑 `src/utils/logging_config.py`:

```python
MODULE_ABBREV = {
    # 现有映射...
    
    # 添加新映射
    'src.new_module.some_component': 's.nm.component',
}
```

---

## 🔍 调试：查看原始模块名

如果需要查看原始模块名（用于调试），可以临时使用标准格式：

```python
import logging

# 临时使用标准格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## ✅ 总结

**优化内容**:
1. ✅ 缩短模块名：`src.evaluation.steve1_evaluator` → `s.ev.steve1`
2. ✅ 固定宽度：模块名 20 字符，日志级别 7 字符
3. ✅ 过滤日志：`process_watcher` 和 `minerl.env.malmo.instance`
4. ✅ 统一配置：`setup_evaluation_logging()` 函数

**效果**:
- 📊 日志完美对齐，易于阅读
- 🎯 减少不必要的日志干扰
- 🚀 提高调试效率

**文件修改**:
- 新增：`src/utils/logging_config.py`
- 修改：`src/evaluation/eval_framework.py`
- 修改：`src/utils/__init__.py`


