# MineRL 自定义任务实施总结

**日期**: 2025-11-06  
**状态**: ✅ 完成实现  
**目标**: 基于 MineRL 创建自定义任务，参考 MineDojo 设计

---

## 背景

用户决定：
> "保持使用 minerl 环境，不需要在这个阶段做转换的事情了"
> "参考 minedojo 的任务和实现，首先实现一个获得1个木头的自定义任务环境"

**核心需求**:
1. 在 MineRL 中创建自定义任务
2. 参考 MineDojo 的任务设计（奖励机制、成功条件）
3. 实现 `harvest_1_log` 任务
4. 编写评估脚本测试 STEVE-1 性能

---

## 实施方案

### 设计思路

由于 MineRL 本身不直接支持像 MineDojo 那样灵活的任务定义，我们采用 **包装器模式**：

```
MineRL 基础环境 
    ↓ 包装
MineRLCustomTaskWrapper
    ↓ 添加
• 自定义奖励函数
• 任务完成检测
• Inventory 检查
```

**优点**:
- ✅ 复用 MineRL 现有环境和基础设施
- ✅ 灵活定义奖励和成功条件
- ✅ 兼容 STEVE-1 评估框架
- ✅ 易于扩展新任务

---

## 核心组件

### 1. `MineRLCustomTaskWrapper`

**位置**: `src/envs/minerl_custom_tasks.py`

**功能**:
```python
class MineRLCustomTaskWrapper(gym.Wrapper):
    def __init__(
        self,
        base_env_name: str,      # 基础 MineRL 环境
        target_item: str,        # 目标物品
        target_count: int = 1,   # 目标数量
        max_steps: int = 6000,   # 最大步数
        reward_type: str = 'sparse',  # 奖励类型
    ):
```

**关键方法**:

1. **`step(action)`** - 执行动作并计算自定义奖励
   ```python
   def step(self, action):
       # 执行基础环境的 step
       obs, base_reward, terminated, truncated, info = self.env.step(action)
       
       # 获取当前物品数量
       current_count = self._get_item_count(obs, info)
       
       # 计算自定义奖励
       reward = self._compute_reward(current_count, base_reward)
       
       # 检查任务完成
       if current_count >= self.target_count:
           terminated = True
           info['success'] = True
       
       return obs, reward, terminated, truncated, info
   ```

2. **`_get_item_count()`** - 从 inventory 获取物品数量
   ```python
   def _get_item_count(self, obs, info):
       # 尝试从 obs['inventory'] 获取
       # 回退到 info['inventory']
       # 支持多种物品名称变体（如不同的木头类型）
   ```

3. **`_compute_reward()`** - 计算奖励
   ```python
   def _compute_reward(self, current_count, base_reward):
       if self.reward_type == 'sparse':
           # 稀疏奖励：完成时 +1.0
           return 1.0 if current_count >= self.target_count else 0.0
       
       elif self.reward_type == 'dense':
           # 密集奖励：根据进度给奖励
           progress = current_count / self.target_count
           return increment_reward
   ```

---

### 2. 已实现的任务

#### `HarvestLogEnv` - 获得木头

```python
class HarvestLogEnv(MineRLCustomTaskWrapper):
    def __init__(self):
        super().__init__(
            base_env_name='MineRLObtainDiamond-v0',
            target_item='log',
            target_count=1,
            max_steps=2000,
            reward_type='sparse',
        )
```

**注册ID**: `MineRLHarvestLog-v0`

**参考**: MineDojo 的 `harvest_1_log`

---

#### `HarvestIronIngotEnv` - 冶炼铁锭

```python
class HarvestIronIngotEnv(MineRLCustomTaskWrapper):
    def __init__(self):
        super().__init__(
            base_env_name='MineRLObtainIronPickaxe-v0',
            target_item='iron_ingot',
            target_count=1,
            max_steps=6000,
            reward_type='sparse',
        )
```

**注册ID**: `MineRLHarvestIronIngot-v0`

**参考**: MineDojo 的 `harvest_1_iron_ingot`

---

#### `HarvestDiamondEnv` - 获得钻石

```python
class HarvestDiamondEnv(MineRLCustomTaskWrapper):
    def __init__(self):
        super().__init__(
            base_env_name='MineRLObtainDiamond-v0',
            target_item='diamond',
            target_count=1,
            max_steps=18000,
            reward_type='sparse',
        )
```

**注册ID**: `MineRLHarvestDiamond-v0`

**参考**: MineDojo 的 `harvest_1_diamond`

---

### 3. 环境注册

**位置**: `src/envs/__init__.py`

```python
from .minerl_custom_tasks import (
    MineRLCustomTaskWrapper,
    HarvestLogEnv,
    HarvestIronIngotEnv,
    HarvestDiamondEnv,
    register_custom_envs,
)

# 自动注册
register_custom_envs()
```

**注册函数**:
```python
def register_custom_envs():
    """注册所有自定义环境到 gym"""
    envs = [
        ('MineRLHarvestLog-v0', 
         'src.envs.minerl_custom_tasks:HarvestLogEnv'),
        ('MineRLHarvestIronIngot-v0', 
         'src.envs.minerl_custom_tasks:HarvestIronIngotEnv'),
        ('MineRLHarvestDiamond-v0', 
         'src.envs.minerl_custom_tasks:HarvestDiamondEnv'),
    ]
    
    for env_id, entry_point in envs:
        gym.register(id=env_id, entry_point=entry_point)
```

---

## 评估脚本

### `test_steve1_evaluation.py`

**位置**: `scripts/test_steve1_evaluation.py`

**功能**:
1. ✅ 加载 STEVE-1 和 MineCLIP
2. ✅ 创建自定义 MineRL 环境
3. ✅ 运行多次评估试验
4. ✅ 计算成功率和平均步数
5. ✅ 生成 JSON 评估报告

**工作流程**:
```python
# 1. 加载组件
agent, mineclip, _ = load_mineclip_agent_env(...)
env = gym.make('MineRLHarvestLog-v0')

# 2. 编码指令
text_embed = mineclip.encode_text("chop tree")

# 3. 运行评估
for trial in range(n_trials):
    obs, _ = env.reset()
    agent.reset()
    
    while not done:
        action = agent.get_action(obs, text_embed)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward > 0:
            # 任务完成！
            break

# 4. 生成报告
save_results(results)
```

---

### 运行脚本

**位置**: `scripts/run_steve1_harvest_log.sh`

```bash
#!/bin/bash
# 运行 STEVE-1 Harvest Log 评估

cd "$PROJECT_ROOT"
python scripts/test_steve1_evaluation.py
```

**使用**:
```bash
bash scripts/run_steve1_harvest_log.sh
```

---

## 测试脚本

### `test_custom_env.py`

**位置**: `scripts/test_custom_env.py`

**功能**: 快速测试自定义环境是否能正常创建和运行

```bash
python scripts/test_custom_env.py
```

**测试内容**:
- ✅ 环境注册
- ✅ 环境创建
- ✅ Reset 功能
- ✅ Step 功能
- ✅ 环境关闭

---

## 与 MineDojo 的对比

### 任务定义方式

| 方面 | MineDojo | MineRL 自定义任务 |
|------|---------|----------------|
| 环境创建 | 直接指定 task_id | 包装 MineRL 环境 |
| 世界配置 | 灵活的 Malmo XML | 受限于基础环境 |
| 奖励函数 | 内置多种奖励 | 自定义实现 |
| Inventory | 直接访问 | 从 obs/info 获取 |
| 任务数量 | 1,572 个 | 可无限扩展 |

---

### 奖励机制

**MineDojo**:
```python
# harvest_1_log
reward = 1.0 if inventory['log'] >= 1 else 0.0
```

**MineRL 自定义任务**:
```python
# MineRLHarvestLog-v0
def _compute_reward(self, current_count, base_reward):
    if self.reward_type == 'sparse':
        return 1.0 if current_count >= 1 else 0.0
```

**完全一致！** ✅

---

### 成功条件

**MineDojo**:
```python
success = (inventory['log'] >= 1)
```

**MineRL 自定义任务**:
```python
if current_count >= self.target_count:
    terminated = True
    info['success'] = True
```

**完全一致！** ✅

---

## 优势和限制

### ✅ 优势

1. **复用 MineRL 基础设施**
   - 无需重新实现 Malmo 接口
   - 兼容 MineRL 生态系统

2. **灵活的任务定义**
   - 简单的包装器接口
   - 易于扩展新任务

3. **兼容 STEVE-1**
   - 直接使用 STEVE-1 官方 agent
   - 无需额外适配

4. **参考 MineDojo 设计**
   - 奖励机制一致
   - 任务定义清晰

---

### ⚠️ 限制

1. **基于现有 MineRL 环境**
   - 无法完全自定义世界生成
   - 受限于基础环境配置

2. **Inventory 访问**
   - 依赖 MineRL 的 inventory 信息
   - 部分环境可能不提供完整数据

3. **任务类型**
   - 主要支持 Harvest 类任务
   - Combat 和 TechTree 需要额外工作

---

## 文件清单

### 新增文件

```
src/envs/
├── __init__.py                     # ✅ 环境包初始化
├── minerl_custom_tasks.py          # ✅ 自定义任务包装器
└── harvest_log.py                  # （废弃）早期实现

scripts/
├── test_steve1_evaluation.py       # ✅ 评估脚本（正式版）
├── run_steve1_harvest_log.sh       # ✅ 运行脚本
└── test_custom_env.py              # ✅ 测试脚本

docs/guides/
└── MINERL_CUSTOM_TASKS_GUIDE.md    # ✅ 使用指南

docs/summaries/
└── MINERL_CUSTOM_TASKS_IMPLEMENTATION.md  # ✅ 本文档

docs/reference/
└── MINEDOJO_TASKS_REFERENCE.md     # 参考文档（已存在）
```

---

## 使用示例

### 创建自定义任务

```python
from src.envs import MineRLCustomTaskWrapper

# 创建：获得 5 个苹果
env = MineRLCustomTaskWrapper(
    base_env_name='MineRLObtainDiamond-v0',
    target_item='apple',
    target_count=5,
    max_steps=3000,
    reward_type='sparse',
)
```

---

### 运行评估

```bash
# 测试环境创建
python scripts/test_custom_env.py

# 运行 STEVE-1 评估
bash scripts/run_steve1_harvest_log.sh

# 或直接运行 Python 脚本
python scripts/test_steve1_evaluation.py
```

---

## 后续计划

### 短期（1-2 周）

- [ ] 测试 STEVE-1 在 HarvestLog 任务的成功率
- [ ] 添加更多 Harvest 任务（苹果、小麦等）
- [ ] 优化奖励函数

### 中期（1 个月）

- [ ] 实现 Combat 类任务
- [ ] 实现 TechTree 类任务
- [ ] 添加中文指令评估

### 长期（3 个月）

- [ ] 自定义 Malmo XML 配置
- [ ] 完整的 MineDojo 任务覆盖
- [ ] 模型 fine-tuning 和训练

---

## 总结

✅ **成功实现了基于 MineRL 的自定义任务系统**

**核心成果**:
1. ✅ 通用的任务包装器（`MineRLCustomTaskWrapper`）
2. ✅ 3 个参考 MineDojo 的 Harvest 任务
3. ✅ 完整的评估脚本和工具
4. ✅ 详细的文档和使用指南

**技术亮点**:
- 包装器模式，复用 MineRL 基础设施
- 参考 MineDojo 设计，奖励和成功条件一致
- 兼容 STEVE-1，无需额外适配
- 易于扩展，可快速添加新任务

**下一步**: 运行测试并验证 STEVE-1 的评估效果！

---

**文档版本**: 1.0  
**最后更新**: 2025-11-06  
**结论**: ✅ 实现完成，准备测试

