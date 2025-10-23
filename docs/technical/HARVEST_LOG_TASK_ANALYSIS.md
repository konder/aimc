# 🎯 harvest_1_log 任务判断条件分析

## 🐛 用户报告的问题

**问题**: 游戏中获得黑色的木头（Dark Oak）没有奖励

这是一个关键发现！说明MineDojo的 `harvest_1_log` 任务可能只识别特定类型的木头。

---

## 📋 MineDojo harvest_1_log 任务说明

### **任务定义**

`harvest_1_log` 是MineDojo的内置任务，目标是**获得1个原木（log）**。

```python
task_id = "harvest_1_log"
# 任务名称: Harvest 1 Log
# 目标: 获得1个原木方块
```

### **判断条件**

MineDojo内置任务的成功条件基于**库存物品检测**：
- 任务在每一步检查玩家的库存
- 当检测到目标物品数量达到要求时，返回 `done=True`
- 同时给予稀疏奖励（通常为+1）

---

## 🌳 Minecraft木头类型

### **所有原木（Log）类型**

Minecraft中有6种原木类型：

| 英文名 | 中文名 | 物品ID | 颜色 |
|--------|--------|--------|------|
| Oak Log | 橡木原木 | `minecraft:oak_log` | 浅棕色 |
| Birch Log | 白桦木原木 | `minecraft:birch_log` | 白色 |
| Spruce Log | 云杉木原木 | `minecraft:spruce_log` | 深棕色 |
| **Dark Oak Log** | **深色橡木原木** | `minecraft:dark_oak_log` | **黑褐色** ✅ |
| Jungle Log | 丛林木原木 | `minecraft:jungle_log` | 浅棕色 |
| Acacia Log | 金合欢木原木 | `minecraft:acacia_log` | 灰棕色 |

**你获得的"黑色木头"应该是 Dark Oak Log （深色橡木原木）**

---

## 🔍 问题分析

### **可能的原因**

#### **原因1: MineDojo任务判断逻辑限制** ⚠️

```python
# 可能的判断逻辑（推测）
success_item = "oak_log"  # 只识别橡木原木？
# 或者
success_item = ["oak_log", "birch_log", "spruce_log"]  # 部分木头？
```

**如果MineDojo只检测特定类型的log**，则Dark Oak Log不会被计入。

---

#### **原因2: 物品ID不匹配**

Dark Oak Log的物品ID是 `minecraft:dark_oak_log`，可能MineDojo的检测列表中没有包含这个。

---

#### **原因3: 生成世界的树种限制**

MineDojo可能在生成世界时只生成特定类型的树：
- 默认生成: Oak（橡木）和 Birch（白桦木）
- 不生成: Dark Oak（需要特殊生物群系）

如果你遇到了Dark Oak树，可能是：
1. 世界生成的随机因素
2. 你移动到了Dark Oak森林生物群系

---

## 🔧 解决方案

### **方案A: 修改任务判断条件（推荐）⭐**

创建一个**自定义任务**，接受所有类型的原木：

```python
# 文件: src/tasks/custom_harvest_log.py

import minedojo
from minedojo.tasks import HarvestTask

class CustomHarvestLogTask:
    """
    自定义harvest_log任务，接受所有类型的原木
    """
    
    def __init__(self):
        # 所有原木类型
        self.log_types = [
            "oak_log",
            "birch_log", 
            "spruce_log",
            "dark_oak_log",  # 深色橡木
            "jungle_log",
            "acacia_log"
        ]
    
    def check_success(self, obs, inventory):
        """
        检查是否获得任意类型的原木
        
        Args:
            obs: 观察
            inventory: 库存字典
        
        Returns:
            success (bool): 是否成功
            reward (float): 奖励
        """
        total_logs = 0
        
        # 检查所有原木类型
        for log_type in self.log_types:
            if log_type in inventory:
                total_logs += inventory[log_type]
        
        # 只要获得1个原木就成功
        if total_logs >= 1:
            return True, 1.0
        else:
            return False, 0.0

# 使用示例
def create_custom_harvest_env():
    # 创建基础环境
    env = minedojo.make(
        task_id="open_ended",  # 使用开放式任务
        image_size=(160, 256)
    )
    
    # 添加自定义判断逻辑
    task_checker = CustomHarvestLogTask()
    
    # 包装环境
    env = CustomHarvestWrapper(env, task_checker)
    
    return env
```

---

### **方案B: 使用MineDojo的回调修改判断**

```python
# 文件: src/utils/custom_task_wrappers.py

import gym
import numpy as np

class AllLogTypesWrapper(gym.Wrapper):
    """
    包装器: 接受所有类型的原木作为成功条件
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.log_types = [
            "oak_log",
            "birch_log",
            "spruce_log",
            "dark_oak_log",
            "jungle_log",
            "acacia_log"
        ]
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 检查库存中的所有原木
        if 'inventory' in info:
            total_logs = 0
            for log_type in self.log_types:
                if log_type in info['inventory']:
                    total_logs += info['inventory'][log_type]
            
            # 如果获得原木，设置done=True和reward
            if total_logs >= 1 and not done:
                done = True
                reward = 1.0
                info['success'] = True
                print(f"✓ 检测到{total_logs}个原木！任务成功！")
        
        return obs, reward, done, info
```

**使用方法**:

```python
# 在 src/utils/env_wrappers.py 中修改 make_minedojo_env

def make_minedojo_env(task_id, ...):
    env = minedojo.make(task_id=task_id, ...)
    
    # 如果是harvest_1_log，添加自定义包装器
    if task_id == "harvest_1_log":
        env = AllLogTypesWrapper(env)
    
    env = MinedojoWrapper(env)
    # ... 其他包装器
    
    return env
```

---

### **方案C: 指定生成世界类型（最简单）**

如果问题是世界生成了Dark Oak树，可以限制世界生成：

```python
# 在 minedojo.make 中指定生物群系
env = minedojo.make(
    task_id="harvest_1_log",
    world_seed=12345,  # 固定种子
    # 或指定生物群系（如果API支持）
)
```

**但这不是根本解决方案**，因为：
- 限制了探索多样性
- 不符合真实游戏场景

---

## 🧪 验证问题

### **步骤1: 检查MineDojo的任务定义**

```bash
# 创建测试脚本
cat > test_harvest_log.py << 'EOF'
import minedojo

# 创建环境
env = minedojo.make(task_id="harvest_1_log", image_size=(160, 256))

# 重置环境
obs = env.reset()

# 手动测试
print("=" * 70)
print("harvest_1_log 任务测试")
print("=" * 70)

# 模拟步骤
for step in range(10):
    # 随机动作
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # 打印库存信息
    if 'inventory' in info:
        inventory = info['inventory']
        logs = {k: v for k, v in inventory.items() if 'log' in k}
        if logs:
            print(f"Step {step}: 原木库存: {logs}")
            print(f"  Reward: {reward}, Done: {done}")
    
    if done:
        print(f"\n✓ 任务完成！总步数: {step+1}")
        break

env.close()
EOF

# 运行测试
bash scripts/run_minedojo_x86.sh python test_harvest_log.py
```

---

### **步骤2: 手动获得不同类型的木头**

```bash
# 使用创造模式测试
python << 'EOF'
import minedojo

env = minedojo.make(
    task_id="harvest_1_log",
    image_size=(160, 256),
    # 如果支持，设置创造模式或直接给予物品
)

obs = env.reset()

# 使用/give命令测试（如果MineDojo支持）
# /give @p minecraft:oak_log 1
# /give @p minecraft:dark_oak_log 1

env.close()
EOF
```

---

## 📊 验证结果分析

### **情况A: Dark Oak不被识别**

如果Dark Oak Log确实不被识别：

```
Oak Log获得 -> ✓ Reward=1.0, Done=True
Dark Oak Log获得 -> ✗ Reward=0.0, Done=False
```

**确认**: MineDojo的harvest_1_log只识别部分原木类型

**解决**: 使用**方案B (AllLogTypesWrapper)**

---

### **情况B: 所有Log都被识别**

如果所有原木都能被识别：

```
Oak Log获得 -> ✓ Reward=1.0, Done=True
Dark Oak Log获得 -> ✓ Reward=1.0, Done=True
```

**说明**: 问题可能是其他原因（例如库存检测延迟）

**解决**: 检查游戏内是否真的获得了物品（按E查看库存）

---

## 🛠️ 实施步骤

### **推荐实施: 方案B (AllLogTypesWrapper)**

#### **1. 创建包装器**

```bash
# 在 src/utils/env_wrappers.py 中添加
```

#### **2. 集成到环境创建**

```python
def make_minedojo_env(task_id, ...):
    env = minedojo.make(task_id=task_id, ...)
    
    # harvest相关任务统一处理
    if "harvest" in task_id and "log" in task_id:
        env = AllLogTypesWrapper(env)
        print(f"  ✓ 启用AllLogTypes包装器（支持所有原木类型）")
    
    env = MinedojoWrapper(env)
    # ... 其他包装器
    
    return env
```

#### **3. 测试验证**

```bash
# 手动录制，尝试获得不同类型的木头
bash scripts/run_minedojo_x86.sh \
python tools/dagger/record_manual_chopping.py \
    --base-dir data/test_all_logs \
    --episodes 3
```

---

## 📝 注意事项

### **1. Minecraft物品名称**

MineDojo使用的物品ID可能是：
- `oak_log` (简写)
- `minecraft:oak_log` (完整ID)

需要测试确认使用哪种格式。

---

### **2. 库存检测延迟**

MineDojo可能有1-2帧的延迟：
- 实际获得物品后
- 库存信息可能下一帧才更新

**解决**: 在包装器中缓存之前的库存状态

---

### **3. 木板(Planks) vs 原木(Log)**

注意区分：
- **Log (原木)**: 从树上直接获得的方块
- **Planks (木板)**: 原木合成的方块

`harvest_1_log` 只要求**Log**，不是Planks。

---

### **4. 不同树种的获取难度**

| 树种 | 常见程度 | 获取难度 |
|------|---------|---------|
| Oak | 极常见 | ✅ 易 |
| Birch | 常见 | ✅ 易 |
| Spruce | 中等 | 🟡 中 |
| Dark Oak | 较少 | 🔴 难（需要特殊生物群系）|
| Jungle | 稀有 | 🔴 难（需要丛林）|
| Acacia | 稀有 | 🔴 难（需要热带草原）|

**建议**: 如果Dark Oak经常出现，说明世界生成在Dark Oak森林，可能需要调整世界生成参数。

---

## 🎯 最终建议

### **立即行动**

1. ✅ **实施AllLogTypesWrapper** (最快解决方案)
   - 修改 `src/utils/env_wrappers.py`
   - 添加所有6种原木类型检测

2. ✅ **测试验证**
   - 手动录制，尝试获得不同类型木头
   - 确认奖励和Done信号

3. ✅ **更新文档**
   - 记录这个发现
   - 说明支持所有原木类型

---

### **长期优化**

1. 考虑创建多个harvest任务变体：
   - `harvest_oak_log`: 只要橡木
   - `harvest_any_log`: 任意原木（自定义）
   - `harvest_dark_oak_log`: 只要深色橡木（高难度）

2. 根据树种调整训练策略：
   - Oak/Birch: 基础训练
   - Dark Oak: 需要学习寻找特定生物群系

---

## 📚 参考资料

### **Minecraft Wiki**
- [Wood (Log)](https://minecraft.fandom.com/wiki/Log)
- [Dark Oak Log](https://minecraft.fandom.com/wiki/Dark_Oak_Log)

### **MineDojo文档**
- [Programmatic Tasks](https://docs.minedojo.org/sections/getting_started/sim.html#task-specification)
- [Inventory System](https://docs.minedojo.org/sections/getting_started/sim.html#observation-space)

---

**版本**: 1.0.0  
**创建日期**: 2025-10-22  
**关键发现**: Dark Oak Log (深色橡木) 可能不被harvest_1_log任务识别
**解决方案**: 使用AllLogTypesWrapper支持所有6种原木类型

