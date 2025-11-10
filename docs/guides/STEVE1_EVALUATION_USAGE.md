# STEVE-1评估系统使用指南

**更新日期**: 2025-11-06  
**适用版本**: Phase 1 完成版

---

## 📋 快速开始

### 方式1: 使用便捷脚本（推荐）

```bash
# 运行评估启动器
./scripts/run_steve1_evaluation.sh

# 选择评估模式:
#   1) 测试环境 (仅测试创建评估器)
#   2) 快速评估 (1个任务 × 2次trial)
#   3) 标准评估 (3个任务 × 3次trial)
#   4) 完整评估 (10个任务 × 10次trial)
```

### 方式2: 使用Python代码

```bash
# 通过run_minedojo_x86.sh运行
scripts/run_minedojo_x86.sh python -c "
from src.evaluation import STEVE1Evaluator

# 创建评估器
evaluator = STEVE1Evaluator(
    model_path='data/weights/steve1/steve1.weights',
    device='auto'  # 自动选择最优设备
)

# 运行快速测试
evaluator.quick_test(n_trials=3)
"
```

---

## 🎯 评估器配置

### STEVE1Evaluator 参数

```python
from src.evaluation import STEVE1Evaluator

evaluator = STEVE1Evaluator(
    model_path="data/weights/steve1/steve1.weights",  # STEVE-1权重路径
    mineclip_path="data/weights/mineclip/attn.pth",   # MineCLIP权重路径
    device="auto",                                     # 设备选择
    task_config_path="config/eval_tasks.yaml"         # 任务配置文件
)
```

### Device选项

- `"auto"` - 自动选择最优设备 (cuda > mps > cpu) **推荐**
- `"cuda"` - 使用NVIDIA GPU
- `"mps"` - 使用Apple Silicon GPU
- `"cpu"` - 使用CPU

---

## 📊 评估方法

### 1. 评估单个任务

```python
# 评估单个任务（英文指令）
result = evaluator.evaluate_task(
    task_id="harvest_1_log",
    lang="en",
    n_trials=10
)

print(f"成功率: {result['success_rate']:.1f}%")
```

### 2. 快速测试（3个任务）

```python
# 快速测试集：harvest_1_log, harvest_1_dirt, combat_cow_forest_barehand
evaluator.quick_test(n_trials=3)
```

### 3. 完整基线评估（10个任务）

```python
# 完整基线测试集（10个任务）
evaluator.run_baseline_evaluation(n_trials=10)
```

### 4. 自定义任务集

```python
# 评估自定义任务集
comparisons = evaluator.evaluate_task_set(
    set_name="baseline_test",  # 或 "quick_test"
    n_trials=10
)
```

---

## 📁 评估结果

### 报告位置

评估完成后，报告会自动保存到：

```
results/evaluation/
├── quick_test_report_2025-11-06T10-22-06.json     # JSON格式
└── quick_test_report_2025-11-06T10-22-06.txt      # 文本格式
```

### 报告内容

#### JSON报告结构

```json
{
  "generated_at": "2025-11-06T10:22:06",
  "total_tasks": 3,
  "summary": {
    "overall_en_success_rate": 83.3,
    "overall_zh_auto_success_rate": 83.3,
    "avg_gap": 33.3,
    "tasks_evaluated": 3
  },
  "detailed_results": [
    {
      "task_id": "harvest_1_log",
      "en_success_rate": 100.0,
      "zh_auto_success_rate": 50.0,
      "gap": 50.0,
      "semantic_variance": 40.8,
      "en_trials": [...],
      "zh_auto_trials": [...],
      ...
    }
  ]
}
```

#### 文本报告示例

```
====================================================================================================
评估结果对比表 (Evaluation Results Comparison)
====================================================================================================
Task ID                              EN   ZH(Auto)    ZH(Man)      Gap      Var
----------------------------------------------------------------------------------------------------
harvest_1_log                    100.0%      50.0%      50.0%    50.0%    40.8%
harvest_1_dirt                   100.0%     100.0%     100.0%     0.0%    23.6%
combat_cow_forest_barehand        50.0%     100.0%     100.0%    50.0%    23.6%
----------------------------------------------------------------------------------------------------
Overall Average                   83.3%      83.3%        N/A    33.3%      N/A
====================================================================================================
```

---

## 🔧 高级用法

### 使用Python脚本

创建自定义评估脚本 `my_evaluation.py`:

```python
#!/usr/bin/env python
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.evaluation import STEVE1Evaluator

def main():
    # 创建评估器
    evaluator = STEVE1Evaluator(
        model_path="data/weights/steve1/steve1.weights",
        device="auto"
    )
    
    # 自定义评估流程
    tasks = ["harvest_1_log", "harvest_1_dirt", "harvest_1_apple"]
    
    for task_id in tasks:
        print(f"\n评估任务: {task_id}")
        result = evaluator.evaluate_task(task_id, "en", n_trials=5)
        print(f"  成功率: {result['success_rate']:.1f}%")

if __name__ == "__main__":
    main()
```

运行:

```bash
scripts/run_minedojo_x86.sh python my_evaluation.py
```

### 评估中文指令

```python
# 评估中文指令（自动翻译）
result_zh = evaluator.evaluate_task(
    task_id="harvest_1_log",
    lang="zh_auto",
    n_trials=10,
    instruction_override="砍树"  # 使用中文指令
)

# 评估英文指令
result_en = evaluator.evaluate_task(
    task_id="harvest_1_log",
    lang="en",
    n_trials=10
)

# 计算Gap
gap = abs(result_en['success_rate'] - result_zh['success_rate'])
print(f"中英文Gap: {gap:.1f}%")
```

---

## 📋 任务列表

### Quick Test Set（快速测试集）

1. `harvest_1_log` - 砍树（简单）
2. `harvest_1_dirt` - 采集泥土（简单）
3. `combat_cow_forest_barehand` - 空手杀牛（中等）

### Baseline Test Set（基线测试集）

1. `harvest_1_log` - 砍树
2. `harvest_1_apple` - 采集苹果
3. `harvest_1_crafting_table` - 制作工作台
4. `harvest_1_iron_ingot` - 冶炼铁锭
5. `combat_zombie_forest_leather_armors_wooden_sword_shield` - 击杀僵尸
6. `techtree_from_barehand_to_wooden_pickaxe` - 制作木镐
7. `techtree_from_barehand_to_iron_sword` - 制作铁剑
8. `harvest_1_cooked_beef` - 烹饪熟牛肉
9. `harvest_1_torch` - 制作火把
10. `combat_spider_plains_iron_armors_iron_sword_shield` - 击杀蜘蛛

完整任务列表见: `config/eval_tasks.yaml`

---

## 🚨 常见问题

### Q1: 如何指定使用GPU?

```python
evaluator = STEVE1Evaluator(device="auto")  # 自动选择
# 或
evaluator = STEVE1Evaluator(device="cuda")  # 强制使用CUDA
```

### Q2: 评估很慢怎么办?

- 减少trials数量: `n_trials=3` 而不是 `10`
- 使用quick_test而不是baseline_evaluation
- 确保使用GPU加速

### Q3: MineDojo环境创建失败?

确保使用`run_minedojo_x86.sh`启动:

```bash
scripts/run_minedojo_x86.sh python your_script.py
```

### Q4: 如何查看详细日志?

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 设置为DEBUG级别
```

### Q5: 如何只测试中文指令?

```python
result = evaluator.evaluate_task(
    "harvest_1_log",
    lang="zh_auto",
    n_trials=10,
    instruction_override="砍树"
)
```

---

## 📊 评估指标说明

### 主要指标

1. **EN Success Rate** - 英文baseline成功率
2. **ZH(Auto) Success Rate** - 中文自动翻译成功率
3. **Language Gap** - 中英文成功率差距
4. **Semantic Variance** - 语义变体鲁棒性（方差）

### 指标解读

- **Gap越小越好** - 说明翻译质量好
- **Variance越小越好** - 说明对不同表述鲁棒
- **Success Rate** - 绝对成功率

---

## 🎯 下一步

评估完成后，可以:

1. 分析报告，识别问题任务
2. 优化术语词典 (`data/chinese_terms.json`)
3. 微调模型提高成功率
4. 扩展评估任务集

---

## 📚 相关文档

- **技术方案**: `docs/design/CHINESE_AIMC_AGENT_TECHNICAL_PLAN.md`
- **Phase 1总结**: `docs/status/PHASE1_STEVE1_INTEGRATION_COMPLETE.md`
- **任务配置**: `config/eval_tasks.yaml`
- **术语词典**: `data/chinese_terms.json`

---

**更新时间**: 2025-11-06  
**维护者**: AIMC Team

