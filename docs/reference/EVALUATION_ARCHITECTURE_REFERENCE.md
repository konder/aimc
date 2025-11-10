# 评估架构快速参考

## 🏗️ 架构图

```
┌─────────────────────────────────────────────────────┐
│           EvaluationFramework (Manager)              │
│  职责：任务管理、批量调度、报告生成                     │
├─────────────────────────────────────────────────────┤
│  • TaskLoader (加载 YAML 配置)                        │
│  • ReportGenerator (生成 JSON + TXT)                 │
│  • evaluate_single_task()                           │
│  • evaluate_task_list()                             │
│  • evaluate_test_set()                              │
│  • generate_report()                                │
│  • print_summary()                                  │
└────────────────┬────────────────────────────────────┘
                 │ 管理和调度
                 ↓
┌─────────────────────────────────────────────────────┐
│           STEVE1Evaluator (Worker)                   │
│  职责：任务执行、模型加载、自动翻译                     │
├─────────────────────────────────────────────────────┤
│  • ChineseTranslator (中文→英文)                     │
│  • evaluate_task() [唯一的公开评估方法]              │
│  • close()                                          │
└────────────────┬────────────────────────────────────┘
                 │ 执行环境
                 ↓
┌─────────────────────────────────────────────────────┐
│          Environment + Agent + MineCLIP              │
│  MineRL 环境 / 自定义环境                             │
└─────────────────────────────────────────────────────┘
```

---

## 📋 API 速查表

### STEVE1Evaluator（执行器）

#### 初始化
```python
evaluator = STEVE1Evaluator(
    model_path="data/weights/vpt/2x.model",
    weights_path="data/weights/steve1/steve1.weights",
    prior_weights="data/weights/steve1/steve1_prior.pt",
    text_cond_scale=6.0,
    visual_cond_scale=7.0,
    seed=42,
    enable_render=False,
    env_name='MineRLHarvestEnv-v0'
)
```

#### 公开方法（仅2个）

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `evaluate_task()` | `task_id`, `language`, `instruction`, `n_trials`, `max_steps` | `TaskResult` | **执行单个任务** |
| `close()` | 无 | 无 | 清理资源 |

#### 示例
```python
result = evaluator.evaluate_task(
    task_id="harvest_wood",
    language="zh",              # 自动翻译中文
    instruction="砍树",
    n_trials=3,
    max_steps=2000
)
print(f"成功率: {result.success_rate * 100:.1f}%")
evaluator.close()
```

---

### EvaluationFramework（管理器）

#### 初始化
```python
from src.evaluation.eval_framework import EvaluationFramework, EvaluationConfig

config = EvaluationConfig(
    model_path="...",
    weights_path="...",
    n_trials=3,
    max_steps=2000,
    enable_render=False,
    task_config_path="config/eval_tasks.yaml",
    results_dir="results/evaluation"
)

framework = EvaluationFramework(config=config)
# 或使用默认配置
framework = EvaluationFramework()
```

#### 公开方法

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `evaluate_single_task()` | `task_id`, `n_trials`, `max_steps` | `TaskResult` | 评估单个任务（从 YAML） |
| `evaluate_task_list()` | `task_ids`, `n_trials`, `max_steps` | `List[TaskResult]` | **批量评估任务** |
| `evaluate_test_set()` | `test_set_name`, `n_trials`, `max_steps` | `List[TaskResult]` | 评估测试集 |
| `generate_report()` | `results`, `report_name` | `(json_path, txt_path)` | **生成报告** |
| `print_summary()` | `results` | 无 | 打印统计摘要 |
| `close()` | 无 | 无 | 清理资源 |

#### 示例
```python
# 单个任务
result = framework.evaluate_single_task('harvest_wood_zh')

# 批量任务
results = framework.evaluate_task_list([
    'harvest_wood_en',
    'harvest_wood_zh'
])

# 测试集
results = framework.evaluate_test_set('quick_test')

# 打印摘要
framework.print_summary(results)

# 生成报告
framework.generate_report(results, "my_evaluation")

# 清理
framework.close()
```

---

## 🎯 使用场景选择

| 场景 | 推荐方式 | 理由 |
|------|---------|------|
| 调试单个任务 | `STEVE1Evaluator` | 直接控制，灵活 |
| 批量评估任务 | `EvaluationFramework` | 自动调度，生成报告 |
| 正式评估 | `EvaluationFramework` | 完整的配置管理和报告 |
| 自定义评估流程 | `STEVE1Evaluator` | 自由组合 |
| 中文指令评估 | 任意（自动翻译） | 两者都支持 |

---

## 📁 文件结构

```
src/evaluation/
├── steve1_evaluator.py      # ⚡ 执行器：任务执行
├── eval_framework.py        # 🎛️  管理器：任务管理
├── task_loader.py           # 📂 任务配置加载
├── metrics.py               # 📊 结果数据结构
└── report_generator.py      # 📄 报告生成工具

src/translation/
└── translator.py            # 🌐 中文翻译

config/
└── eval_tasks.yaml          # ⚙️  任务配置文件

scripts/
├── evaluate_harvest.py      # 🔧 单任务示例
└── evaluate_with_framework.py # 🎯 批量评估示例
```

---

## ⚙️ 配置文件格式

### config/eval_tasks.yaml

```yaml
basic_tasks:
  - task_id: "harvest_wood_en"
    category: "harvest"
    difficulty: "easy"
    description: "使用英文指令砍树"
    env_name: "MineRLHarvestEnv-v0"
    en_instruction: "chop tree"
    n_trials: 3
    max_steps: 2000

  - task_id: "harvest_wood_zh"
    category: "harvest"
    difficulty: "easy"
    description: "使用中文指令砍树"
    env_name: "MineRLHarvestEnv-v0"
    zh_instruction: "砍树"  # 自动翻译成 "chop tree"
    n_trials: 3
    max_steps: 2000

# 测试集定义
quick_test:
  - "harvest_wood_en"
  - "harvest_wood_zh"
```

---

## 🚀 命令行接口

### 直接运行 Framework

```bash
# 单个任务
python src/evaluation/eval_framework.py \
    --task harvest_wood_zh \
    --n-trials 3 \
    --max-steps 2000

# 批量任务
python src/evaluation/eval_framework.py \
    --task-list harvest_wood_en harvest_wood_zh \
    --n-trials 3

# 测试集
python src/evaluation/eval_framework.py \
    --test-set quick_test \
    --n-trials 5

# 启用渲染
python src/evaluation/eval_framework.py \
    --task harvest_wood_zh \
    --render
```

### 使用示例脚本

```bash
# 中文 vs 英文对比
python scripts/evaluate_with_framework.py

# 单任务评估（带渲染）
python scripts/evaluate_harvest.py
```

---

## 📊 输出格式

### 控制台输出
```
================================================================================
评估结果汇总
================================================================================

任务ID                       指令        成功率      平均步数      平均时间
--------------------------------------------------------------------------------
harvest_wood_en             chop tree    100.0%      839.3        56.9s
harvest_wood_zh             砍树         100.0%      852.1        58.3s

--------------------------------------------------------------------------------
总体统计                     N/A          100.0%      845.7        57.6s

总任务数: 2
总试验数: 6
================================================================================
```

### JSON 报告
```json
{
  "metadata": {
    "timestamp": "2025-11-08T17:12:34",
    "total_tasks": 2,
    "evaluator": "STEVE-1",
    "framework": "EvaluationFramework"
  },
  "tasks": [
    {
      "task_id": "harvest_wood_zh",
      "instruction": "砍树",
      "language": "zh",
      "success_rate": 100.0,
      "avg_steps": 852.1,
      "trials": [...]
    }
  ],
  "summary": {
    "overall_success_rate": 100.0,
    "total_trials": 6
  }
}
```

---

## 🔑 关键特性

### 1. 自动中文翻译
```python
# 中文指令自动翻译
result = evaluator.evaluate_task(
    instruction="砍树",  # 自动 → "chop tree"
    language="zh"
)
```

### 2. 参数优先级
```
函数参数 > 任务配置 > 全局配置
```

### 3. 自定义环境支持
```python
# 在 YAML 中指定
env_name: "MineRLHarvestEnv-v0"

# 或在代码中
evaluator = STEVE1Evaluator(env_name='CustomEnv-v0')
```

### 4. 延迟加载
```python
# 模型和环境只在首次 evaluate_task 时加载
evaluator = STEVE1Evaluator()  # 快速
result = evaluator.evaluate_task(...)  # 此时才加载
```

---

## 📚 相关文档

- **使用指南**: `docs/guides/EVALUATION_FRAMEWORK_GUIDE.md`
- **重构总结**: `docs/summaries/EVALUATION_FRAMEWORK_REFACTORING.md`
- **清理总结**: `docs/summaries/STEVE1_EVALUATOR_CLEANUP.md`
- **任务配置**: `config/eval_tasks.yaml`

---

**更新日期**: 2025-11-08  
**版本**: 2.0 (重构后)

