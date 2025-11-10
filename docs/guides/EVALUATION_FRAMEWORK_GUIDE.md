# 评估框架使用指南

## 🏗️ 架构重构

### 职责划分

```
EvaluationFramework (Manager/Scheduler)
    ↓ 管理和调度
STEVE1Evaluator (Worker/Executor)
    ↓ 集成翻译器
ChineseTranslator
    ↓ 执行环境
Environment + Agent
```

### 组件职责

| 组件 | 职责 | 关键功能 |
|------|------|---------|
| **EvaluationFramework** | 任务管理器/调度器 | • 从 YAML 加载任务<br>• 批量任务调度<br>• 结果收集与聚合<br>• **报告生成（JSON + TXT）** |
| **STEVE1Evaluator** | 执行器/工作节点 | • 加载 STEVE-1 模型和环境<br>• 集成 ChineseTranslator<br>• 自动翻译中文指令<br>• 执行单任务评估<br>• 返回 TaskResult |
| **ChineseTranslator** | 翻译服务 | • 术语词典翻译<br>• 中文→英文 |

### 职责清单

**EvaluationFramework（Manager）：**
- ✅ TaskLoader（任务配置加载）
- ✅ ReportGenerator（报告生成）
- ✅ 任务批量调度
- ✅ 结果聚合和统计

**STEVE1Evaluator（Worker）：**
- ✅ 模型/环境加载
- ✅ ChineseTranslator 集成
- ✅ 单任务执行
- ❌ ~~任务配置~~ → Framework
- ❌ ~~报告生成~~ → Framework

---

## 🚀 快速开始

### 方式1：使用命令行（推荐）

```bash
cd /Users/nanzhang/aimc
conda activate minedojo

# 评估单个任务
python src/evaluation/eval_framework.py --task harvest_wood_en --n-trials 3

# 评估测试集
python src/evaluation/eval_framework.py --test-set quick_test --n-trials 3

# 评估多个任务
python src/evaluation/eval_framework.py --task-list harvest_wood_en harvest_wood_zh --n-trials 3

# 启用渲染
python src/evaluation/eval_framework.py --task harvest_wood_zh --n-trials 3 --render
```

### 方式2：使用示例脚本

```bash
# 中文 vs 英文砍树任务对比
python scripts/evaluate_with_framework.py
```

### 方式3：Python API

```python
import src.envs
from src.evaluation.eval_framework import EvaluationFramework, EvaluationConfig

# 创建配置
config = EvaluationConfig(
    n_trials=3,
    max_steps=2000,
    enable_render=False
)

# 创建框架
framework = EvaluationFramework(config=config)

# 评估单个任务
result = framework.evaluate_single_task('harvest_wood_zh')

# 评估任务列表
results = framework.evaluate_task_list(['harvest_wood_en', 'harvest_wood_zh'])

# 打印摘要
framework.print_summary(results)

# 生成报告
framework.generate_report(results, "my_evaluation")

# 清理
framework.close()
```

---

## 📝 配置文件格式

### config/eval_tasks.yaml

```yaml
# 基础任务
basic_tasks:
  # 英文任务
  - task_id: "harvest_wood_en"
    category: "harvest"
    difficulty: "easy"
    description: "使用英文指令砍树获取木头"
    env_name: "MineRLHarvestEnv-v0"
    en_instruction: "chop tree"
    n_trials: 3
    max_steps: 2000
  
  # 中文任务
  - task_id: "harvest_wood_zh"
    category: "harvest"
    difficulty: "easy"
    description: "使用中文指令砍树获取木头"
    env_name: "MineRLHarvestEnv-v0"
    zh_instruction: "砍树"  # 🔑 自动翻译
    n_trials: 3
    max_steps: 2000

# 测试集
quick_test:
  - "harvest_wood_en"
  - "harvest_wood_zh"
```

---

## 🔑 核心特性

### 1. 自动中文翻译

**STEVE1Evaluator** 自动检测和翻译中文指令：

```python
# 检测中文
if self._is_chinese(instruction):
    logger.info(f"原始指令: {instruction}")
    instruction = self.translator.translate(instruction)  # 砍树 → chop tree
    logger.info(f"翻译结果: {instruction}")
```

**支持的语言标识：**
- `language="en"` - 英文指令
- `language="zh"` - 中文指令（自动翻译）
- `language="zh_auto"` - 中文指令（自动翻译）
- `language="zh_manual"` - 中文指令（自动翻译）

### 2. 灵活的环境配置

任务可以指定自定义环境：

```yaml
- task_id: "my_task"
  env_name: "MineRLHarvestEnv-v0"  # 🔑 自定义环境
  en_instruction: "do something"
```

框架会自动切换环境：

```python
if 'env_name' in task_config:
    framework.evaluator.env_name = task_config['env_name']
```

### 3. 参数优先级

```
函数参数 > 任务配置 > 全局配置
```

示例：
```python
# 全局配置: n_trials=10
config = EvaluationConfig(n_trials=10)

# 任务配置: n_trials=3
task_config = {'n_trials': 3, ...}

# 函数参数: n_trials=5
framework.evaluate_single_task('task_id', n_trials=5)

# 实际使用: 5 (函数参数优先级最高)
```

---

## 📊 输出示例

### 控制台输出

```
================================================================================
批量评估开始: 2 个任务
================================================================================

[1/2] 评估任务: harvest_wood_en
================================================================================
评估任务: harvest_wood_en
================================================================================
  描述: 使用英文指令砍树获取木头
  类别: harvest
  难度: easy
  指令: chop tree
  语言: en
  试验次数: 3
  最大步数: 2000

  Trial 1/3...
    结果: ✅ 成功, 步数: 772, 时间: 52.3s
  Trial 2/3...
    结果: ✅ 成功, 步数: 845, 时间: 57.1s
  Trial 3/3...
    结果: ✅ 成功, 步数: 901, 时间: 61.2s

  ✅ 完成: 成功率 100.0%, 平均步数 839.3

[2/2] 评估任务: harvest_wood_zh
  原始指令: 砍树
  翻译结果: chop tree

  ✅ 完成: 成功率 100.0%, 平均步数 852.1

================================================================================
评估结果汇总
================================================================================

任务ID                       指令                 成功率      平均步数      平均时间
--------------------------------------------------------------------------------
harvest_wood_en             chop tree            100.0%      839.3        56.9s
harvest_wood_zh             砍树                 100.0%      852.1        58.3s

--------------------------------------------------------------------------------
总体统计                     N/A                  100.0%      845.7        57.6s

总任务数: 2
总试验数: 6
================================================================================
```

### 报告文件

**JSON 报告** (`results/evaluation/harvest_wood_en_vs_zh_20251108_171234.json`):
```json
{
  "metadata": {
    "timestamp": "2025-11-08T17:12:34",
    "total_tasks": 2,
    "evaluator": "STEVE-1"
  },
  "tasks": [
    {
      "task_id": "harvest_wood_en",
      "instruction": "chop tree",
      "language": "en",
      "success_rate": 100.0,
      "avg_steps": 839.3,
      "avg_time": 56.9,
      "trials": [...]
    },
    {
      "task_id": "harvest_wood_zh",
      "instruction": "砍树",
      "language": "zh",
      "success_rate": 100.0,
      "avg_steps": 852.1,
      "avg_time": 58.3,
      "trials": [...]
    }
  ],
  "summary": {
    "overall_success_rate": 100.0,
    "total_trials": 6,
    "successful_trials": 6
  }
}
```

---

## 🔧 高级用法

### 1. 自定义评估器

```python
# 创建自定义配置的评估器
evaluator = STEVE1Evaluator(
    model_path="custom/model.model",
    weights_path="custom/weights.weights",
    env_name='CustomEnv-v0',
    text_cond_scale=8.0,
    enable_render=True
)

# 使用自定义评估器创建框架
framework = EvaluationFramework(evaluator=evaluator)
```

### 2. 只收集结果，稍后统一处理

```python
framework = EvaluationFramework()

# 运行多个评估
framework.evaluate_single_task('task1')
framework.evaluate_single_task('task2')
framework.evaluate_single_task('task3')

# 所有结果存储在 framework.results 中
print(f"共评估 {len(framework.results)} 个任务")

# 统一处理
framework.print_summary()
framework.generate_report(report_name="batch_results")
```

### 3. 按类别评估

```python
# 获取所有 harvest 类任务
harvest_tasks = framework.task_loader.get_tasks_by_category('harvest')

# 批量评估
results = framework.evaluate_task_list(harvest_tasks)
```

---

## 🆚 对比：三种方式

| 特性 | test_steve1_official_baseline.py | evaluate_harvest.py | eval_framework.py |
|------|----------------------------------|---------------------|-------------------|
| **定位** | 调试脚本 | 单任务脚本 | **生产框架** ✅ |
| **YAML 配置** | ❌ | ❌ | ✅ |
| **批量任务** | ❌ | ❌ | ✅ |
| **中文翻译** | ❌ | ❌ | ✅ 自动 |
| **报告生成** | ❌ | ❌ | ✅ |
| **适用场景** | 调试、验证 | 快速测试 | **正式评估** |

---

## 📚 相关文档

- **STEVE1Evaluator API**：`src/evaluation/steve1_evaluator.py`
- **任务配置**：`config/eval_tasks.yaml`
- **翻译器**：`src/translation/translator.py`
- **术语词典**：`data/chinese_terms.json`

---

## 日期

2025-11-08

