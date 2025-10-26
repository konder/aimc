# 🎯 Task Wrappers 设计指南

## 📋 概述

本项目采用**两层Wrapper架构**：

1. **通用Wrapper** (`env_wrappers.py`): 适用于所有任务的通用功能
2. **任务特定Wrapper** (`task_wrappers.py`): 针对特定任务的定制逻辑

---

## 🏗️ 架构设计

### **文件结构**

```
src/utils/
├── env_wrappers.py       # 通用环境包装器
│   ├── TimeLimitWrapper        (超时限制)
│   ├── MinedojoWrapper         (观察空间简化)
│   ├── FrameStack              (帧堆叠)
│   ├── ActionWrapper           (动作空间处理)
│   └── make_minedojo_env()     (环境创建函数)
│
└── task_wrappers.py      # 任务特定包装器
    ├── HarvestLogWrapper       (harvest_log任务)
    ├── HarvestWheatWrapper     (harvest_wheat任务)
    ├── CombatWrapper           (hunt/combat任务)
    ├── CraftWrapper            (craft任务)
    ├── get_task_wrapper()      (自动选择Wrapper)
    └── apply_task_wrapper()    (便捷应用函数)
```

---

## 🎯 设计原则

### **原则1: 职责分离**

| Wrapper类型 | 职责 | 示例 |
|------------|------|------|
| **通用Wrapper** | 所有任务通用的功能 | 图像归一化、超时限制、相机平滑 |
| **任务Wrapper** | 特定任务的判断条件、奖励 | harvest_log检测6种原木 |

---

### **原则2: 自动选择**

```python
# ❌ 不好：手动选择Wrapper
if task_id == "harvest_1_log":
    env = HarvestLogWrapper(env, required_logs=1)
elif task_id == "harvest_8_log":
    env = HarvestLogWrapper(env, required_logs=8)
# ... 很多if-else

# ✅ 好：自动选择Wrapper
from src.envs import apply_task_wrapper
env = apply_task_wrapper(env, task_id)  # 自动识别任务类型
```

---

### **原则3: 可扩展**

添加新任务Wrapper只需3步：

```python
# 1. 在 task_wrappers.py 中定义新的Wrapper类
class NavigateWrapper(gym.Wrapper):
    def __init__(self, env, target_coords, verbose=True):
        # ...

# 2. 在 get_task_wrapper() 中添加识别逻辑
if "navigate" in task_id:
    return NavigateWrapper, {'target_coords': (0, 0, 0)}

# 3. 完成！环境创建时自动应用
env = apply_task_wrapper(env, task_id="navigate_to_origin")
```

---

## 📦 已实现的Task Wrappers

### **1. HarvestLogWrapper**

**用途**: harvest_log 任务（获得原木）

**问题**: MineDojo可能只识别Oak Log，不识别Dark Oak等其他原木

**解决**: 检测所有6种原木类型

```python
# 支持的原木类型
log_types = [
    "oak_log",       # 橡木
    "birch_log",     # 白桦木
    "spruce_log",    # 云杉木
    "dark_oak_log",  # 深色橡木 ← 用户报告的"黑色木头"
    "jungle_log",    # 丛林木
    "acacia_log"     # 金合欢木
]
```

**适用任务**:
- `harvest_1_log` (获得1个原木)
- `harvest_8_log` (获得8个原木)
- `harvest_64_log` (获得64个原木)

**使用方法**:
```python
# 方法1: 手动应用
from src.envs import HarvestLogWrapper
env = minedojo.make(task_id="harvest_8_log")
env = HarvestLogWrapper(env, required_logs=8)

# 方法2: 自动应用（推荐）
from src.envs import apply_task_wrapper
env = minedojo.make(task_id="harvest_8_log")
env = apply_task_wrapper(env, "harvest_8_log")  # 自动识别需要8个
```

---

### **2. HarvestWheatWrapper** (TODO)

**用途**: harvest_wheat 任务（收获小麦）

**状态**: 框架已定义，逻辑待实现

**适用任务**:
- `harvest_1_wheat`
- `harvest_8_wheat`

---

### **3. CombatWrapper** (TODO)

**用途**: hunt/combat 任务（狩猎/战斗）

**状态**: 框架已定义，逻辑待实现

**适用任务**:
- `hunt_cow`
- `hunt_pig`
- `combat_spider`

**可扩展功能**:
- 击杀奖励
- 受伤惩罚
- 死亡处理
- 多目标支持

---

### **4. CraftWrapper** (TODO)

**用途**: craft 任务（合成）

**状态**: 框架已定义，逻辑待实现

**适用任务**:
- `craft_planks`
- `craft_stick`
- `craft_crafting_table`

**可扩展功能**:
- 合成表验证
- 材料检测
- 合成奖励

---

## 🔧 如何添加新的Task Wrapper

### **步骤1: 定义Wrapper类**

在 `src/envs/task_wrappers.py` 中添加：

```python
class NavigateWrapper(gym.Wrapper):
    """
    Navigate 任务专用包装器
    
    用于导航到指定坐标的任务。
    
    适用任务:
    - navigate_to_origin
    - navigate_to_coords
    """
    
    def __init__(self, env, target_coords=(0, 0, 0), distance_threshold=5.0, verbose=True):
        """
        Args:
            env: MineDojo环境实例
            target_coords: 目标坐标 (x, y, z)
            distance_threshold: 到达判定距离（默认5米）
            verbose: 是否打印详细信息
        """
        super().__init__(env)
        self.target_coords = target_coords
        self.distance_threshold = distance_threshold
        self.verbose = verbose
        
        if self.verbose:
            print(f"  ✓ NavigateWrapper已启用")
            print(f"    - 目标坐标: {target_coords}")
            print(f"    - 判定距离: {distance_threshold}米")
    
    def reset(self, **kwargs):
        """重置环境"""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """执行一步并检查是否到达目标"""
        obs, reward, done, info = self.env.step(action)
        
        # 获取当前位置
        if 'location_stats' in info:
            current_pos = info['location_stats']['pos']
            
            # 计算距离
            distance = self._calculate_distance(current_pos, self.target_coords)
            
            # 如果距离小于阈值，任务完成
            if distance <= self.distance_threshold and not done:
                done = True
                reward = 1.0
                info['success'] = True
                
                if self.verbose:
                    print(f"\n✓ 到达目标！距离: {distance:.2f}米\n")
        
        return obs, reward, done, info
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点间的欧氏距离"""
        import math
        return math.sqrt(
            (pos1[0] - pos2[0])**2 +
            (pos1[1] - pos2[1])**2 +
            (pos1[2] - pos2[2])**2
        )
```

---

### **步骤2: 注册到自动选择系统**

在 `get_task_wrapper()` 函数中添加：

```python
def get_task_wrapper(task_id, verbose=True):
    # ... 现有代码 ...
    
    # navigate 任务（新增）
    if "navigate" in task_id:
        # 解析目标坐标（如果任务ID中包含）
        # 或使用默认值
        return NavigateWrapper, {
            'target_coords': (0, 0, 0),  # 原点
            'distance_threshold': 5.0,
            'verbose': verbose
        }
    
    # ... 其他任务 ...
```

---

### **步骤3: 测试**

```python
# 测试新的Wrapper
from src.envs import apply_task_wrapper
import minedojo

env = minedojo.make(task_id="navigate_to_origin")
env = apply_task_wrapper(env, "navigate_to_origin")

obs = env.reset()
# ... 环境交互
```

---

### **步骤4: 更新文档**

在本文档中添加新Wrapper的说明：

```markdown
### **5. NavigateWrapper**

**用途**: navigate 任务（导航到指定坐标）

**功能**:
- 计算当前位置与目标的距离
- 距离小于阈值时判定任务完成
- 支持自定义目标坐标和判定距离

**适用任务**:
- `navigate_to_origin`
- `navigate_to_coords`
```

---

## 🎮 Wrapper应用顺序

在 `make_minedojo_env()` 中的应用顺序：

```python
def make_minedojo_env(task_id, ...):
    # 1. 创建基础环境
    env = minedojo.make(task_id=task_id, ...)
    
    # 2. 任务特定Wrapper（修改判断条件）← 最先应用！
    env = apply_task_wrapper(env, task_id)
    
    # 3. 通用Wrapper
    env = MinedojoWrapper(env)         # 简化观察空间
    env = TimeLimitWrapper(env, ...)   # 超时限制
    env = ActionWrapper(env)           # 动作空间处理
    env = CameraSmoothingWrapper(env)  # 相机平滑（可选）
    env = FrameStack(env)              # 帧堆叠（可选）
    
    # 4. Monitor（最后）
    # 由调用者在需要时添加
    
    return env
```

**为什么任务Wrapper最先应用？**

因为任务Wrapper需要直接访问MineDojo的原始`info`字典（包含`inventory`等），在`MinedojoWrapper`简化观察空间之前。

---

## 📊 通用 vs 任务 Wrapper 对比

| 特性 | 通用Wrapper (env_wrappers.py) | 任务Wrapper (task_wrappers.py) |
|------|-------------------------------|-------------------------------|
| **适用范围** | 所有任务 | 特定任务或任务类别 |
| **功能** | 观察、动作、时间等通用功能 | 任务判断、奖励、特殊逻辑 |
| **示例** | TimeLimitWrapper, FrameStack | HarvestLogWrapper, CombatWrapper |
| **修改频率** | 低（稳定） | 中（根据任务调整） |
| **依赖性** | 无任务依赖 | 依赖任务特性 |

---

## 🎯 最佳实践

### **1. 命名规范**

```python
# ✅ 好：清晰的任务相关命名
class HarvestLogWrapper(gym.Wrapper):
class CombatZombieWrapper(gym.Wrapper):
class CraftPlanksWrapper(gym.Wrapper):

# ❌ 不好：模糊或通用的命名
class CustomWrapper(gym.Wrapper):
class TaskWrapper(gym.Wrapper):
class MyWrapper(gym.Wrapper):
```

---

### **2. 参数化配置**

```python
# ✅ 好：通过参数配置
class HarvestLogWrapper(gym.Wrapper):
    def __init__(self, env, required_logs=1, verbose=True):
        # 可以处理 harvest_1_log, harvest_8_log 等

# ❌ 不好：硬编码
class Harvest1LogWrapper(gym.Wrapper):
    def __init__(self, env):
        self.required_logs = 1  # 硬编码，不灵活
```

---

### **3. 详细日志**

```python
# ✅ 好：提供有用的反馈
if total_logs >= self.required_logs:
    if self.verbose:
        log_info = ", ".join(obtained_log_types)
        print(f"✓ 获得原木！总数: {total_logs} | 类型: {log_info}")

# ❌ 不好：无反馈或过少反馈
if total_logs >= self.required_logs:
    done = True  # 用户不知道发生了什么
```

---

### **4. 向后兼容**

```python
# ✅ 好：不破坏原有行为
if total_logs >= self.required_logs and not done:
    done = True  # 只在原任务未完成时修改

# ❌ 不好：强制覆盖
done = (total_logs >= self.required_logs)  # 可能覆盖原任务的done
```

---

## 🔬 测试指南

### **单元测试模板**

```python
# tests/test_task_wrappers.py

import pytest
import minedojo
from src.envs import HarvestLogWrapper

def test_harvest_log_wrapper_oak():
    """测试HarvestLogWrapper识别Oak Log"""
    env = minedojo.make(task_id="harvest_1_log")
    env = HarvestLogWrapper(env, required_logs=1, verbose=False)
    
    obs = env.reset()
    
    # 模拟获得Oak Log
    # （需要Mock MineDojo的inventory）
    # ...
    
    assert done == True
    assert reward == 1.0
    assert info['success'] == True

def test_harvest_log_wrapper_dark_oak():
    """测试HarvestLogWrapper识别Dark Oak Log"""
    # 用户报告的"黑色木头"场景
    # ...

def test_harvest_log_wrapper_multiple():
    """测试HarvestLogWrapper识别多个原木"""
    # harvest_8_log场景
    # ...
```

---

### **集成测试**

```bash
# 手动测试
bash scripts/run_minedojo_x86.sh python << 'EOF'
from src.envs import make_minedojo_env

# 创建环境（会自动应用HarvestLogWrapper）
env = make_minedojo_env(task_id="harvest_1_log")

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        print(f"任务完成！Reward: {reward}")
        break
env.close()
EOF
```

---

## 📚 参考资料

### **相关文档**
- `env_wrappers.py` - 通用Wrapper实现
- `task_wrappers.py` - 任务Wrapper实现
- `HARVEST_LOG_TASK_ANALYSIS.md` - harvest_log任务详细分析

### **MineDojo任务列表**
- `docs/reference/MINEDOJO_TASKS_REFERENCE.md`

### **设计模式**
- Wrapper Pattern (装饰器模式)
- Strategy Pattern (策略模式)

---

## 🎯 未来扩展

### **计划中的Wrapper**

1. **NavigateWrapper**: 导航任务
2. **BuildWrapper**: 建造任务
3. **MineWrapper**: 挖矿任务
4. **TechTreeWrapper**: 科技树任务（需要特定道具）

### **高级功能**

1. **Wrapper组合**: 
   ```python
   # 组合多个任务Wrapper
   env = HarvestLogWrapper(env)
   env = CraftWrapper(env)  # 先砍树再合成
   ```

2. **动态奖励塑形**:
   ```python
   # 根据距离目标的距离给予连续奖励
   reward = -distance_to_tree * 0.01  # 越接近越好
   ```

3. **进度追踪**:
   ```python
   # 追踪任务完成进度
   info['progress'] = total_logs / required_logs
   ```

---

**版本**: 1.0.0  
**创建日期**: 2025-10-22  
**核心理念**: 职责分离、自动选择、可扩展

