# MineDojo Camera精度修改方案

**目标**: 将camera控制精度从15度/单位提升到1度/单位  
**状态**: 评估阶段，未实施  
**日期**: 2025-10-29

---

## 📊 问题分析

### 当前状态
- **Camera离散值**: 25个 (0-24)
- **精度**: 360°/24 = **15度/单位**
- **MineDojo定义**: 0=-180°, 12=0°, 24=+180°

### VPT需求
- **输出范围**: ±10度（实际观测）
- **典型值**: 0.62, 1.61, 3.22, 5.81等
- **精度需求**: 需要1-5度级别的精细控制

### 精度损失示例
```
VPT输出: 3.22度
当前转换: 3.22/15 = 0.21 → round(0) → noop (完全丢失!)
期望转换: 3.22度 → discrete=3 (实际移动3度)
```

---

## 🎯 修改方案概述

### 核心思路
将camera的离散值数量从25扩展到361（或721），实现1度（或0.5度）精度。

### 修改范围
- Action Space定义
- Action转换逻辑  
- Minecraft命令生成

---

## 📝 详细修改方案

### 方案A: 1度精度（推荐）⭐

#### 1. 修改Action Space定义

**文件**: `minedojo/sim/spaces.py` 或环境创建相关文件

**位置**: 查找 `MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])`

**修改**:
```python
# 原始
MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])

# 修改为 (1度精度)
MultiDiscrete([3, 3, 4, 361, 361, 8, 244, 36])
#                      ↑    ↑
#                   pitch  yaw
# 361 = 180*2 + 1 (从-180到+180，每度一个值)
```

#### 2. 修改Action解释逻辑

**文件**: `minedojo/sim/bridge/bridge.py` 或 `minedojo/sim/bridge/mc_bridge.py`

**位置**: 查找camera action的处理代码

**原始逻辑推测**:
```python
# 当前实现（推测）
def _convert_camera_action(discrete_value):
    """
    将离散值转换为Minecraft camera命令
    discrete_value: 0-24
    """
    # 0 -> -180度, 12 -> 0度, 24 -> +180度
    degrees = (discrete_value - 12) * 15
    return degrees
```

**修改为**:
```python
def _convert_camera_action(discrete_value):
    """
    将离散值转换为Minecraft camera命令
    discrete_value: 0-360 (1度精度)
    """
    # 0 -> -180度, 180 -> 0度, 360 -> +180度
    degrees = discrete_value - 180
    return degrees
```

#### 3. 查找并修改所有相关常量

**搜索关键词**:
```bash
# 在MineDojo源代码中搜索
grep -r "25" minedojo/sim/
grep -r "camera.*24" minedojo/sim/
grep -r "MultiDiscrete" minedojo/
grep -r "pitch.*yaw" minedojo/sim/bridge/
```

**可能需要修改的文件**:
```
minedojo/sim/
├── spaces.py          # Action space定义
├── wrappers/          # 可能有action space相关wrapper
├── bridge/
│   ├── bridge.py      # 桥接器主逻辑
│   ├── mc_bridge.py   # Minecraft通信
│   └── action.py      # Action转换
└── env.py             # 环境主类
```

#### 4. 更新文档和注释

修改所有提到camera精度的地方：
```python
# 旧注释
# Camera delta pitch: 0: -180 degree, 24: 180 degree (25 actions, 15 degree per unit)

# 新注释  
# Camera delta pitch: 0: -180 degree, 360: 180 degree (361 actions, 1 degree per unit)
```

---

### 方案B: 0.5度精度（高精度）

如果需要更高精度：

```python
# 0.5度精度
MultiDiscrete([3, 3, 4, 721, 721, 8, 244, 36])
# 721 = 360*2 + 1

def _convert_camera_action(discrete_value):
    degrees = (discrete_value - 360) * 0.5
    return degrees
```

---

## 🔍 需要验证的关键点

### 1. MineDojo底层实现

**检查点**:
```python
# 找到MineDojo如何定义action space
import minedojo
env = minedojo.make(task_id="harvest_1_log")

# 查看action space
print(env.action_space)
print(env.action_space.nvec)

# 查看实际执行
action = env.action_space.sample()
print(action)
```

**验证文件**:
```bash
# 查找MineDojo安装位置
python -c "import minedojo; print(minedojo.__file__)"

# 典型路径
# /usr/local/lib/python3.9/site-packages/minedojo/__init__.py
```

### 2. Malmo/Minecraft接口

MineDojo底层使用Malmo，需要确认：
- Malmo是否支持1度精度的camera控制
- Minecraft命令格式：可能是 `/tp` 或 `/camera` 命令

**参考**: 
- Malmo文档: https://github.com/microsoft/malmo
- Minecraft commands: `/teleport` 的rotation参数

### 3. 性能影响

**Action Space大小变化**:
```python
# 原始
total_actions = 3 * 3 * 4 * 25 * 25 * 8 * 244 * 36
# = 3,780,672,000

# 修改后 (1度精度)
total_actions = 3 * 3 * 4 * 361 * 361 * 8 * 244 * 36  
# = 219,634,204,992 (增加58倍)
```

**影响**:
- ✅ 对于VPT这种确定性策略：**无影响**（只是选择一个action）
- ⚠️ 对于RL算法：可能需要更多样本
- ⚠️ 对于actor-critic：critic输入维度增大

---

## 📋 修改步骤（执行计划）

### Step 1: 定位关键文件 (1小时)

```bash
# 1. 找到MineDojo安装目录
MINEDOJO_PATH=$(python -c "import minedojo; import os; print(os.path.dirname(minedojo.__file__))")
cd $MINEDOJO_PATH

# 2. 备份
cp -r $MINEDOJO_PATH $MINEDOJO_PATH.backup

# 3. 搜索关键代码
grep -rn "MultiDiscrete" .
grep -rn "25, 25" .
grep -rn "camera.*24" .
grep -rn "pitch.*yaw" sim/
```

### Step 2: 修改Action Space (30分钟)

```python
# 文件: minedojo/sim/XXX.py (具体路径需确认)

# 查找并修改
OLD: MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
NEW: MultiDiscrete([3, 3, 4, 361, 361, 8, 244, 36])
```

### Step 3: 修改转换逻辑 (1小时)

```python
# 文件: minedojo/sim/bridge/XXX.py

# 查找camera action处理
def process_action(action):
    # ...
    camera_pitch = action[3]
    camera_yaw = action[4]
    
    # OLD: 
    # pitch_degrees = (camera_pitch - 12) * 15
    # yaw_degrees = (camera_yaw - 12) * 15
    
    # NEW:
    pitch_degrees = camera_pitch - 180
    yaw_degrees = camera_yaw - 180
    # ...
```

### Step 4: 测试验证 (2小时)

```python
# 测试脚本
import minedojo
import numpy as np

env = minedojo.make(task_id="harvest_1_log")
print("Action Space:", env.action_space)
print("Camera nvec:", env.action_space.nvec[3:5])

# 测试不同的camera值
test_cases = [
    (180, 180),  # 中心 (0度)
    (181, 180),  # +1度 pitch
    (180, 181),  # +1度 yaw
    (190, 180),  # +10度 pitch
    (170, 180),  # -10度 pitch
]

for pitch, yaw in test_cases:
    action = np.array([0, 0, 0, pitch, yaw, 0, 0, 0])
    obs, reward, done, info = env.step(action)
    print(f"Action: pitch={pitch-180}°, yaw={yaw-180}° → executed")
    
env.close()
```

### Step 5: 集成VPT (30分钟)

```python
# 修改 src/training/vpt/vpt_agent.py

class MineRLActionToMineDojo:
    def convert(self, minerl_action):
        # ...
        
        # Camera转换 (1度精度)
        camera = np.asarray(minerl_action['camera']).flatten()
        pitch_degrees = float(camera[0])  # VPT输出的度数
        yaw_degrees = float(camera[1])
        
        # 直接映射到MineDojo的361个离散值
        # 0 -> -180度, 180 -> 0度, 360 -> +180度
        pitch_discrete = int(round(np.clip(pitch_degrees, -180, 180))) + 180
        yaw_discrete = int(round(np.clip(yaw_degrees, -180, 180))) + 180
        
        minedojo_action[3] = pitch_discrete
        minedojo_action[4] = yaw_discrete
        # ...
```

---

## ⚖️ 方案评估

### 优点 ✅

1. **精度大幅提升**: 15度 → 1度（提升15倍）
2. **完美匹配VPT**: VPT的±10度输出可以精确表达
3. **保持离散空间**: 训练稳定性好
4. **理论上可行**: Minecraft/Malmo支持精确的角度控制

### 缺点 ⚠️

1. **修改源代码**: 
   - 需要修改MineDojo库
   - 升级MineDojo时需要重新修改
   - 可能影响其他用户的代码

2. **Action Space变大**:
   - 从25 → 361（增加14倍）
   - 总action space增加约58倍
   - 对RL算法可能有影响（对BC/VPT无影响）

3. **兼容性**:
   - 需要测试与现有代码的兼容性
   - 可能影响MineDojo的其他功能

### 风险 🔥

1. **低风险**:
   - ✅ 只修改camera维度
   - ✅ 不影响其他动作空间
   - ✅ VPT是确定性策略，action space大小无影响

2. **中风险**:
   - ⚠️ 需要正确定位和修改代码
   - ⚠️ 需要充分测试

3. **高风险**:
   - ❌ 如果MineDojo底层有硬编码的25，可能需要多处修改
   - ❌ 如果Malmo不支持精确控制，修改无效

---

## 🔬 验证清单

在实施前，需要验证：

- [ ] MineDojo的action space定义位置
- [ ] Camera action的转换逻辑位置
- [ ] Malmo是否支持1度精度
- [ ] 修改后的性能影响
- [ ] 与VPT的集成测试

---

## 💡 替代方案对比

| 方案 | 精度 | 实施难度 | 维护成本 | 性能影响 |
|------|------|---------|---------|---------|
| **修改MineDojo** | ★★★★★ 1度 | ★★★ 中 | ★★ 中 | ✅ 无 |
| 累积策略 | ★★★ 累积精度 | ★ 低 | ★ 低 | ⚠️ 延迟 |
| 插值/平滑 | ★★ 视觉改善 | ★★ 中 | ★ 低 | ✅ 无 |
| 直接Malmo | ★★★★★ 任意 | ★★★★★ 高 | ★★★★ 高 | ❓ 未知 |

---

## 📚 参考资源

1. **MineDojo官方**:
   - GitHub: https://github.com/MineDojo/MineDojo
   - Docs: https://docs.minedojo.org/

2. **Malmo官方**:
   - GitHub: https://github.com/microsoft/malmo
   - Camera control: 查看 `teleport` 命令文档

3. **相关Issue**:
   - 搜索MineDojo GitHub Issues中关于camera precision的讨论

---

## 🚀 推荐执行策略

### 阶段1: 调研验证 (1天)
1. 安装MineDojo到本地可编辑模式
2. 定位所有相关代码
3. 小范围测试修改可行性

### 阶段2: 实施修改 (2-3天)
1. 备份原始代码
2. 修改action space和转换逻辑
3. 单元测试验证

### 阶段3: 集成测试 (1-2天)
1. 与VPT集成
2. 运行零样本评估
3. 对比修改前后的行为

### 阶段4: 长期维护
1. 文档化修改内容
2. 考虑向MineDojo提交PR
3. 建立修改的版本管理

---

**总结**: 这个方案技术上可行，风险可控，建议先进行调研验证阶段，确认可行后再实施。

**预估工作量**: 1-2周（包含调研、实施、测试）

---

**文档创建**: 2025-10-29  
**状态**: 待评估

