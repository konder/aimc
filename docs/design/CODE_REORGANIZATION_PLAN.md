# AIMC代码重组方案

**日期**: 2025-11-06  
**目标**: 重组STEVE-1和VPT代码结构，使其更清晰、易于维护和评估

---

## 📊 当前结构分析

### 问题诊断

```
当前问题：
  ❌ src/training/steve1/ 包含了模型定义、Agent、训练、评估等混合内容
  ❌ src/training/steve1/VPT/ 是 src/models/vpt/ 的重复
  ❌ 模型推理代码和训练代码混在一起
  ❌ Agent封装没有统一的位置
  ❌ 评估代码分散在多个地方

核心问题：
  "training" 目录既包含训练脚本，又包含模型定义和Agent
  这导致评估时需要从training导入，语义不清晰
```

---

## 🎯 重组原则

### 关注点分离 (Separation of Concerns)

```
src/
├── models/           # 模型定义、权重加载、推理接口（无状态）
├── agents/           # Agent封装（有状态，连接模型和环境）
├── training/         # 训练脚本、数据处理、训练配置
├── evaluation/       # 评估框架、评估脚本
├── envs/             # 环境封装、任务定义
├── translation/      # 翻译模块
└── utils/            # 通用工具函数
```

### 职责划分

| 目录 | 职责 | 特点 |
|------|------|------|
| **models/** | 模型定义、权重加载、推理 | 无状态、可复用 |
| **agents/** | 连接模型和环境、管理状态 | 有状态、封装完整 |
| **training/** | 训练脚本、数据处理 | 训练时使用 |
| **evaluation/** | 评估框架、评估脚本 | 评估时使用 |
| **envs/** | 环境封装、任务wrapper | 环境相关 |

---

## 📋 详细重组方案

### 方案A：完全重组（推荐）⭐⭐⭐⭐⭐

```
src/
├── models/                          # 模型定义和推理
│   ├── __init__.py
│   │
│   ├── vpt/                         # VPT模型（保持不变）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── lib/                     # VPT核心库
│   │   └── ...
│   │
│   ├── steve1/                      # STEVE-1模型（新建）⭐
│   │   ├── __init__.py
│   │   ├── policy.py                # ← training/steve1/embed_conditioned_policy.py
│   │   ├── mineclip.py              # ← training/steve1/mineclip_code/
│   │   ├── lib/                     # STEVE-1核心库
│   │   │   ├── __init__.py
│   │   │   ├── impala_cnn.py        # ← 从VPT复用
│   │   │   ├── action_head.py       # ← 从VPT复用
│   │   │   └── ...
│   │   └── helpers.py               # ← training/steve1/helpers.py
│   │
│   └── mineclip/                    # MineCLIP模型（新建）⭐
│       ├── __init__.py
│       ├── load_mineclip.py         # ← training/steve1/mineclip_code/
│       └── ...
│
├── agents/                          # Agent封装（新建）⭐
│   ├── __init__.py
│   │
│   ├── base.py                      # Agent基类
│   │   # ← training/agent/agent_base.py
│   │
│   ├── vpt_agent.py                 # VPT Agent
│   │   # ← training/vpt/vpt_agent.py
│   │   # ← models/vpt/agent.py (合并)
│   │
│   ├── steve1_agent.py              # STEVE-1 Agent（新建）⭐
│   │   # ← training/steve1/MineRLConditionalAgent.py
│   │   # 封装: 模型加载、MineCLIP、推理接口
│   │
│   └── dagger_agent.py              # DAgger Agent
│       # ← training/dagger/ (部分)
│
├── training/                        # 训练相关（保留核心）
│   ├── __init__.py
│   │
│   ├── steve1/                      # STEVE-1训练
│   │   ├── __init__.py
│   │   ├── train.py                 # ← training/steve1/training/train.py
│   │   ├── config.py                # 保留
│   │   ├── data/                    # 数据处理（保留）
│   │   │   ├── minecraft_dataset.py
│   │   │   ├── generation/
│   │   │   └── sampling/
│   │   └── scripts/                 # 训练脚本
│   │       ├── 1_generate_dataset.sh
│   │       ├── 2_create_sampling.sh
│   │       ├── 3_train.sh
│   │       └── 3_train_finetune_template.sh
│   │
│   ├── vpt/                         # VPT训练
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── dagger/                      # DAgger训练
│   │   ├── train_dagger.py
│   │   ├── label_states.py
│   │   └── ...
│   │
│   └── bc/                          # BC训练
│       └── train_bc.py
│
├── evaluation/                      # 评估框架（已有）
│   ├── __init__.py
│   ├── eval_framework.py
│   ├── metrics.py
│   ├── task_loader.py
│   ├── report_generator.py
│   │
│   └── steve1_evaluator.py         # STEVE-1专用评估器（新建）⭐
│       # 集成STEVE-1Agent和MineDojo环境
│
├── envs/                            # 环境封装（保持）
│   ├── __init__.py
│   ├── env_wrappers.py
│   └── task_wrappers.py
│
├── translation/                     # 翻译模块（保持）
│   ├── __init__.py
│   └── translator.py
│
└── utils/                           # 工具函数（保持）
    ├── __init__.py
    └── ...
```

**优点**:
- ✅ 职责清晰：models（模型）、agents（封装）、training（训练）分离
- ✅ 易于评估：直接从agents导入，无需依赖training
- ✅ 可维护性强：每个模块职责单一
- ✅ 可复用性高：模型和Agent可以独立使用

**缺点**:
- ⚠️ 需要移动较多文件
- ⚠️ 需要更新import路径

---

### 方案B：最小改动（保守）⭐⭐⭐

```
src/
├── models/
│   ├── vpt/                         # 保持不变
│   └── steve1/                      # 新建，只放模型定义
│       ├── __init__.py
│       └── policy.py                # ← training/steve1/embed_conditioned_policy.py
│
├── agents/                          # 新建
│   ├── __init__.py
│   ├── steve1_agent.py              # ← training/steve1/MineRLConditionalAgent.py
│   └── vpt_agent.py                 # ← training/vpt/vpt_agent.py
│
├── training/
│   ├── steve1/                      # 大部分保持不变
│   │   ├── training/                # 训练脚本
│   │   ├── data/                    # 数据处理
│   │   └── ...
│   └── ...
│
└── evaluation/                      # 已有
    └── ...
```

**优点**:
- ✅ 改动最小
- ✅ 风险较低
- ✅ 可以快速完成

**缺点**:
- ❌ training/steve1还是很大，职责不够清晰
- ❌ 依然有部分重复代码

---

### 方案C：渐进式重组（折中）⭐⭐⭐⭐

```
Phase 1: 立即（今天）
  1. 创建 src/agents/steve1_agent.py
  2. 创建 src/models/steve1/__init__.py
  3. 移动关键模型定义
  4. 集成到评估框架

Phase 2: 本周
  1. 清理 training/steve1/VPT/ 重复代码
  2. 统一MineCLIP加载
  3. 优化import路径

Phase 3: 下周（可选）
  1. 完全重组到方案A
  2. 清理历史遗留代码
```

**优点**:
- ✅ 风险可控，每个阶段独立
- ✅ 可以边重组边测试
- ✅ 不影响当前评估工作

---

## 🎯 推荐方案

### **方案C（渐进式）+ Phase 1立即执行**

**理由**:
1. **风险可控**：不一次性改动太多
2. **可测试**：每个阶段都可以测试
3. **不阻塞**：今天就可以继续评估工作
4. **灵活性**：Phase 2/3可选，根据实际需求决定

---

## 📝 Phase 1 详细步骤（今天执行）

### Step 1: 创建Agent模块

```bash
mkdir -p src/agents
touch src/agents/__init__.py
```

### Step 2: 创建STEVE-1 Agent封装

**文件**: `src/agents/steve1_agent.py`

```python
"""
STEVE-1 Agent封装
连接STEVE-1模型、MineCLIP和MineDojo环境
"""

from pathlib import Path
import torch
import numpy as np

# 从training导入（临时，Phase 2会清理）
from ..training.steve1.embed_conditioned_policy import MinecraftAgentPolicy
from ..training.steve1.mineclip_code.load_mineclip import load as load_mineclip
from ..training.steve1.helpers import get_action_from_agent

class STEVE1Agent:
    """STEVE-1 Agent（用于评估）"""
    
    def __init__(
        self,
        model_path: str,
        mineclip_path: str = "data/weights/mineclip/attn.pth",
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 加载STEVE-1策略
        self.policy = self._load_policy(model_path)
        
        # 加载MineCLIP
        self.mineclip = load_mineclip(mineclip_path, device=self.device)
    
    def get_action(self, obs, instruction: str = None, goal_embed=None):
        """
        获取动作
        
        Args:
            obs: MineDojo观测
            instruction: 文本指令（可选）
            goal_embed: 目标嵌入（可选，如果提供则覆盖instruction）
        
        Returns:
            动作字典
        """
        # 如果提供了instruction，编码为MineCLIP嵌入
        if instruction and goal_embed is None:
            goal_embed = self.encode_instruction(instruction)
        
        # 调用策略获取动作
        action = get_action_from_agent(
            self.policy,
            obs,
            goal_embed,
            self.device
        )
        
        return action
    
    def encode_instruction(self, instruction: str):
        """使用MineCLIP编码文本指令"""
        with torch.no_grad():
            text_tokens = self.mineclip.tokenize([instruction])
            text_embed = self.mineclip.encode_text(text_tokens)
        return text_embed
    
    def _load_policy(self, model_path: str):
        """加载STEVE-1策略"""
        # TODO: 实现策略加载
        pass
```

### Step 3: 创建评估器集成

**文件**: `src/evaluation/steve1_evaluator.py`

```python
"""
STEVE-1评估器
集成STEVE-1 Agent和MineDojo环境
"""

import minedojo
from ..agents.steve1_agent import STEVE1Agent
from .eval_framework import ChineseAIMCEvaluator

class STEVE1Evaluator:
    """STEVE-1专用评估器"""
    
    def __init__(
        self,
        model_path: str,
        task_config_path: str = "config/eval_tasks.yaml"
    ):
        # 创建STEVE-1 Agent
        self.agent = STEVE1Agent(model_path)
        
        # 创建评估器
        self.evaluator = ChineseAIMCEvaluator(
            self.agent,
            task_config_path=task_config_path
        )
    
    def run_evaluation(self, task_set="quick_test", n_trials=10):
        """运行评估"""
        return self.evaluator.evaluate_task_set(task_set, n_trials)
```

### Step 4: 更新评估框架

修改 `src/evaluation/eval_framework.py` 的 `_run_single_trial` 方法，
使其可以实际运行MineDojo环境。

---

## 🔄 导入路径变化

### 当前导入 (Before)
```python
from src.training.steve1.MineRLConditionalAgent import MineRLConditionalAgent
from src.training.steve1.embed_conditioned_policy import MinecraftAgentPolicy
```

### Phase 1导入 (After)
```python
from src.agents.steve1_agent import STEVE1Agent
```

### Phase 3导入 (最终目标)
```python
from src.models.steve1 import MinecraftAgentPolicy
from src.agents import STEVE1Agent
from src.evaluation import STEVE1Evaluator
```

---

## ⚠️ 注意事项

### 1. VPT重复代码
```
问题:
  training/steve1/VPT/ 和 models/vpt/ 有重复

解决:
  Phase 1: 暂时保留（避免破坏训练代码）
  Phase 2: 统一使用 models/vpt/
  Phase 3: 删除 training/steve1/VPT/
```

### 2. 训练脚本
```
问题:
  训练脚本依赖当前目录结构

解决:
  Phase 1: 保持训练脚本不变
  Phase 2: 更新import路径
  Phase 3: 完全重组
```

### 3. 向后兼容
```
策略:
  1. 在 training/steve1/__init__.py 中添加兼容import
  2. 逐步迁移，保持旧路径可用
  3. 添加deprecation警告
```

---

## 📋 执行检查清单

### Phase 1 (今天)
- [ ] 创建 `src/agents/` 目录
- [ ] 创建 `src/agents/steve1_agent.py`
- [ ] 创建 `src/evaluation/steve1_evaluator.py`
- [ ] 实现MineDojo环境集成
- [ ] 测试基本推理流程
- [ ] 运行快速评估验证

### Phase 2 (本周)
- [ ] 清理VPT重复代码
- [ ] 统一MineCLIP加载
- [ ] 更新训练脚本import
- [ ] 添加向后兼容层

### Phase 3 (可选)
- [ ] 完全迁移到方案A结构
- [ ] 清理历史代码
- [ ] 更新所有文档

---

## 📊 影响范围评估

### 需要修改的文件数量

| 阶段 | 新建文件 | 移动文件 | 修改文件 | 风险 |
|------|---------|---------|---------|------|
| Phase 1 | 2-3 | 0 | 1-2 | 低 |
| Phase 2 | 1-2 | 5-10 | 5-10 | 中 |
| Phase 3 | 0 | 10-20 | 20-30 | 高 |

### 预计时间

| 阶段 | 时间 | 优先级 |
|------|------|--------|
| Phase 1 | 2-4小时 | 高（今天完成）|
| Phase 2 | 1-2天 | 中（本周完成）|
| Phase 3 | 2-3天 | 低（可选）|

---

## 🎯 决策建议

### 今天的目标
1. ✅ 执行Phase 1
2. ✅ 完成STEVE-1集成
3. ✅ 运行第一次真实评估

### 本周的目标
1. 清理重复代码
2. 统一导入路径
3. 完善文档

### 下周考虑
根据本周使用体验，决定是否执行Phase 3完全重组

---

**推荐方案**: Phase 1 (渐进式重组) ✅  
**执行时间**: 今天立即开始  
**风险等级**: 低  
**可回滚**: 是

