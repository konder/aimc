# MineDojo Camera精度提升方案 - 无需修改源代码！

**发现日期**: 2025-10-29  
**状态**: ✅ 可直接实施  
**难度**: ⭐ 极简单

---

## 🎉 重大发现

MineDojo **已经内置支持** 自定义camera精度！无需修改源代码！

### 关键代码位置

**文件**: `minedojo/tasks/__init__.py`

```python
def make(task_id: str, *args, cam_interval: int | float = 15, **kwargs):
    """
    Make a task. task_id can be one of the following:
    1. a task id for Programmatic tasks
    2. format "creative:{idx}" for the idx-th Creative task
    3. "playthrough" or "open-ended" for these two special tasks
    4. one of "harvest", "combat", "techtree", and "survival"
    """
```

**参数说明**:
- `cam_interval`: Camera的离散间隔（度数）
- 默认值: `15` (即25个离散值，360°/15°=24个间隔+1)
- **可以设置为任意值**，例如 `1` (实现1度精度)

### 内部实现

**文件**: `minedojo/sim/wrappers/ar_nn/nn_action_space_wrapper.py`

```python
class NNActionSpaceWrapper(gym.Wrapper):
    def __init__(
        self,
        env: Union[MineDojoSim, gym.Wrapper],
        discretized_camera_interval: Union[int, float] = 15,
        strict_check: bool = True,
    ):
        # 计算bins数量
        n_pitch_bins = math.ceil(360 / discretized_camera_interval) + 1
        n_yaw_bins = math.ceil(360 / discretized_camera_interval) + 1
        
        self.action_space = spaces.MultiDiscrete([
            3,  # forward/back
            3,  # left/right
            4,  # jump/sneak/sprint
            n_pitch_bins,   # camera pitch ⭐
            n_yaw_bins,     # camera yaw ⭐
            8,  # functional actions
            244,  # craft items
            36,  # inventory slots
        ])
        
        self._cam_interval = discretized_camera_interval
    
    def action(self, action: Sequence[int]):
        """离散动作 → 连续度数"""
        # 转换camera
        noop["camera"][0] = float(action[3]) * self._cam_interval + (-180)
        noop["camera"][1] = float(action[4]) * self._cam_interval + (-180)
```

---

## 🚀 实施方案

### 方案：直接使用 `cam_interval` 参数 ⭐⭐⭐⭐⭐

#### 1. 修改环境创建代码

**文件**: `src/training/vpt/evaluate_vpt_zero_shot.py`

**修改前**:
```python
env = minedojo.make(
    task_id="harvest_milk_1_bucket",
    image_size=(160, 256),
)
```

**修改后**:
```python
env = minedojo.make(
    task_id="harvest_milk_1_bucket",
    image_size=(160, 256),
    cam_interval=1,  # ⭐ 1度精度
)
```

#### 2. 更新VPT Agent的动作转换

**文件**: `src/training/vpt/vpt_agent.py`

**修改 `MineRLActionToMineDojo.convert` 方法**:

```python
class MineRLActionToMineDojo:
    def __init__(self, cam_interval: float = 1.0):
        """
        Args:
            cam_interval: MineDojo环境的camera间隔（度数）
                         1.0 = 1度精度（推荐）
                         15.0 = 15度精度（默认）
        """
        self.cam_interval = cam_interval
        
        # 计算离散值范围
        self.n_camera_bins = math.ceil(360 / cam_interval) + 1
        self.camera_center = (self.n_camera_bins - 1) // 2
    
    def convert(self, minerl_action, debug=False):
        # ... 其他代码 ...
        
        # Camera转换（新算法）
        camera = np.asarray(minerl_action['camera']).flatten()
        pitch_degrees = float(camera[0])  # MineRL输出的度数
        yaw_degrees = float(camera[1])
        
        # 度数 → 离散索引
        # 公式: discrete = round((degrees - (-180)) / cam_interval)
        pitch_discrete = int(round((pitch_degrees + 180) / self.cam_interval))
        yaw_discrete = int(round((yaw_degrees + 180) / self.cam_interval))
        
        # 限制范围
        pitch_discrete = np.clip(pitch_discrete, 0, self.n_camera_bins - 1)
        yaw_discrete = np.clip(yaw_discrete, 0, self.n_camera_bins - 1)
        
        minedojo_action[3] = pitch_discrete
        minedojo_action[4] = yaw_discrete
        
        if debug:
            print(f"  Camera: MineRL[{pitch_degrees:.2f}°, {yaw_degrees:.2f}°] "
                  f"→ MineDojo[{pitch_discrete}, {yaw_discrete}] "
                  f"(实际: {(pitch_discrete * self.cam_interval - 180):.1f}°, "
                  f"{(yaw_discrete * self.cam_interval - 180):.1f}°)")
        
        # ... 其他代码 ...
```

#### 3. 更新VPTAgent初始化

```python
class VPTAgent:
    def __init__(
        self,
        vpt_model_path: str,
        vpt_weights_path: str,
        device: str = "cuda",
        cam_interval: float = 1.0,  # ⭐ 新增参数
        debug_actions: bool = False,
    ):
        # ... 其他代码 ...
        
        # 创建action转换器
        self.action_converter = MineRLActionToMineDojo(
            cam_interval=cam_interval  # ⭐ 传递参数
        )
```

---

## 📊 精度对比

### 不同 `cam_interval` 值的效果

| cam_interval | 离散值数量 | 精度 | Action Space大小 | 推荐场景 |
|--------------|-----------|------|-----------------|---------|
| **15** (默认) | 25 | 15°/单位 | 3×3×4×25×25×8×244×36 = 3.78B | 快速原型 |
| **5** | 73 | 5°/单位 | ×8.5倍 | 平衡方案 |
| **1** ⭐ | 361 | 1°/单位 | ×208倍 | VPT推荐 |
| **0.5** | 721 | 0.5°/单位 | ×831倍 | 极致精度 |

### VPT典型输出示例

```
VPT输出: 3.22度

cam_interval=15:  3.22/15 = 0.21 → round(0) = 0 → 0°    (❌ 完全丢失!)
cam_interval=5:   3.22/5  = 0.64 → round(1) = 1 → 5°    (⚠️ 1.78°误差)
cam_interval=1:   3.22/1  = 3.22 → round(3) = 3 → 3°    (✅ 0.22°误差)
cam_interval=0.5: 3.22/0.5= 6.44 → round(6) = 6 → 3°    (✅ 0.22°误差)
```

**结论**: `cam_interval=1` 完全足够，继续降低增益不大。

---

## ⚡ 实施步骤

### Step 1: 修改代码 (15分钟)

1. **更新环境创建** (`evaluate_vpt_zero_shot.py`, `vpt_agent.py` 等):
   ```python
   env = minedojo.make(..., cam_interval=1)
   ```

2. **更新VPT Agent** (`vpt_agent.py`):
   - 添加 `cam_interval` 参数
   - 修改 `MineRLActionToMineDojo` 转换逻辑

3. **更新配置** (`get_wood_config.yaml`):
   ```yaml
   env:
     cam_interval: 1  # 新增配置
   ```

### Step 2: 测试验证 (30分钟)

```bash
# 1. 快速测试
scripts/evaluate_vpt_zero_shot.sh 1

# 2. 对比测试（15度 vs 1度）
python tools/compare_camera_precision.py

# 3. 详细调试
python tools/debug_vpt_detailed.py --cam-interval 1
```

### Step 3: 性能评估 (1小时)

- 运行10个episodes
- 对比成功率和行为质量
- 确认没有性能退化

---

## ✅ 优势分析

### 相比修改源代码方案

| 特性 | 修改源代码 | 使用参数 (本方案) |
|-----|----------|------------------|
| **实施难度** | ★★★ 中 | ⭐ 极简单 |
| **维护成本** | ★★★★ 高 | ⭐ 无 |
| **升级兼容** | ❌ 需重新修改 | ✅ 自动兼容 |
| **代码侵入** | ❌ 修改库代码 | ✅ 零侵入 |
| **精度提升** | ✅ 1度 | ✅ 1度 |
| **灵活性** | ⚠️ 固定 | ✅ 可配置 |
| **风险** | ⚠️ 中 | ✅ 无 |

### 关键优势

1. ✅ **官方支持**: MineDojo原生功能，稳定可靠
2. ✅ **零侵入**: 不修改任何库代码
3. ✅ **灵活配置**: 可根据需要调整精度
4. ✅ **易于维护**: MineDojo升级无影响
5. ✅ **立即可用**: 修改几行代码即可

---

## 🎯 推荐配置

### 对于VPT Zero-shot评估

```python
env = minedojo.make(
    task_id="harvest_milk_1_bucket",
    image_size=(160, 256),
    cam_interval=1,  # ⭐ 1度精度，完美匹配VPT
)

agent = VPTAgent(
    vpt_model_path="...",
    vpt_weights_path="...",
    cam_interval=1,  # ⭐ 必须与环境一致
    debug_actions=False,
)
```

### 对于训练

```python
# 如果关注训练速度，可使用5度
env = minedojo.make(..., cam_interval=5)

# 如果关注精度，使用1度
env = minedojo.make(..., cam_interval=1)
```

---

## 📝 完整修改清单

### 需要修改的文件

1. ✅ `src/training/vpt/vpt_agent.py`
   - `VPTAgent.__init__`: 添加 `cam_interval` 参数
   - `MineRLActionToMineDojo.__init__`: 添加 `cam_interval` 参数
   - `MineRLActionToMineDojo.convert`: 更新camera转换逻辑

2. ✅ `src/training/vpt/evaluate_vpt_zero_shot.py`
   - `create_env`: 添加 `cam_interval=1`

3. ✅ `config/get_wood_config.yaml`
   - 添加 `cam_interval: 1` 配置

4. ⚠️ 其他使用MineDojo的脚本（可选）
   - 根据需要更新

---

## 🔬 验证计划

### 测试用例

1. **单步验证**:
   ```python
   # VPT输出: 3.22度
   # MineDojo应该: 执行3度转动（而非0度）
   ```

2. **对比测试**:
   ```bash
   # 测试A: cam_interval=15 (默认)
   # 测试B: cam_interval=1 (新配置)
   # 预期: B的camera移动更平滑、精确
   ```

3. **完整评估**:
   ```bash
   # 10个episodes，记录:
   # - 成功率
   # - 平均步数
   # - 行为质量（视频录制）
   ```

---

## 💡 后续优化

### 可选优化项

1. **动态调整**: 根据任务复杂度自动选择 `cam_interval`
2. **多精度训练**: 使用不同精度训练多个模型
3. **精度感知**: 在观察空间中暴露当前精度信息

---

## 📋 总结

### 问题
- MineDojo默认camera精度为15度/单位
- VPT输出的小幅度camera移动（1-10度）被完全丢失

### 解决方案
- **使用 `cam_interval=1` 参数**（MineDojo原生支持）
- 修改VPT Agent的转换逻辑以匹配新精度

### 效果
- ✅ 精度提升15倍（15度 → 1度）
- ✅ 完美匹配VPT的输出范围（±10度）
- ✅ 零代码侵入，易于维护
- ✅ 预期显著改善VPT在MineDojo的行为质量

### 工作量
- ⏱️ **15-30分钟** 代码修改
- ⏱️ **30分钟** 测试验证
- ⏱️ **总计: < 1小时**

---

**状态**: ✅ 方案已确认，可立即实施  
**优先级**: 🔥 高（直接影响VPT性能）  
**风险**: ✅ 极低（官方支持功能）

---

**下一步**: 是否立即实施修改？

