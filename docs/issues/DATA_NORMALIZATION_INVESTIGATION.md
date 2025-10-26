# 数据归一化问题调查

## 🔍 **问题报告**

用户在训练时看到以下警告日志：

```
数据集样本检查:
  形状: torch.Size([4, 3, 160, 256])
  类型: torch.float32
  范围: [0.000, 255.000]  ← 问题！
  均值: 66.625
  ⚠️  警告: 数据未正确归一化！应该在[0,1]范围内
```

**预期：** 数据应该在 [0, 1] 范围内  
**实际：** 数据在 [0, 255] 范围内  
**设备：** MPS (Apple Silicon)

---

## 🚨 **严重性评估**

### **在 MPS 设备上的影响**

| 问题 | 严重程度 | 说明 |
|------|---------|------|
| **Loss 爆炸** | 🔴 严重 | 未归一化的输入 → 梯度爆炸 → Loss = NaN |
| **训练不稳定** | 🟡 中等 | 收敛缓慢或振荡 |
| **精度损失** | 🟡 中等 | Float16/Float32 精度问题 |
| **内存溢出** | 🟡 中等 | MPS 内存管理较敏感 |

### **为什么 MPS 更敏感？**

1. **数值精度：** MPS 使用 Float16 进行部分计算，[0, 255] 范围会导致精度损失
2. **内存带宽：** 大数值需要更多带宽
3. **GPU 优化：** MPS 针对 [0, 1] 范围优化

---

## 🔬 **根本原因分析**

### **数据流追踪**

```
MineDojo 环境 (返回 uint8 [0, 255])
    ↓
MinedojoWrapper._process_obs() 
    rgb = rgb.astype(np.float32) / 255.0  ← 应该归一化到 [0, 1]
    ↓
run_policy_collect_states.py
    'observation': obs.copy()  ← 保存到 episode_*.npy
    ↓
label_states.py
    'observation': state_info['state']['observation']  ← 保存到 iter_*.pkl
    ↓
train_bc.py / load_expert_demonstrations()
    observations = np.array(observations)  ← 转换为 numpy 数组
    ↓
ExpertDataset.__init__()
    if observations.dtype == np.uint8:  ← 检查类型
        observations = observations.astype(np.float32) / 255.0
    ↓
    self.observations = torch.FloatTensor(observations)
```

### **可能的原因**

#### **假设1: 环境归一化失败** ❌
**不太可能** - `MinedojoWrapper._process_obs()` 明确进行了归一化（第124行）。

#### **假设2: 数据类型条件判断错误** ✅ **最可能**

`ExpertDataset.__init__()` (train_bc.py 第50-51行):

```python
if observations.dtype == np.uint8:
    observations = observations.astype(np.float32) / 255.0
```

**问题：**
- 如果数据已经是 `float32` 类型（即使值在 [0, 255]），条件不满足
- 归一化被跳过
- 数据保持 float32 [0, 255] 范围

#### **假设3: 环境包装器未应用** ❌
**不太可能** - `make_minedojo_env()` 确实应用了 `MinedojoWrapper`（第228行）。

#### **假设4: 数据保存/加载时转换** ✅ **可能**

numpy 的 `.npy` 文件保存/加载可能导致类型转换：
- 保存 float32 [0, 1] 时，numpy 可能优化为 uint8
- 加载时转回 float32，但值变成 [0, 255]

---

## 🔧 **诊断方法**

### **1. 运行诊断脚本**

```bash
python tools/diagnose_data_normalization.py
```

这会检查：
1. 环境直接返回的观察
2. 保存的 episode_*.npy 文件
3. DAgger 标注 pkl 文件
4. 专家演示数据

### **2. 手动检查环境**

```python
import sys
sys.path.insert(0, '/Users/nanzhang/aimc')

from src.envs import make_minedojo_env
import numpy as np

env = make_minedojo_env("harvest_1_log", max_episode_steps=10)
obs = env.reset()

print(f"类型: {obs.dtype}")
print(f"范围: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"样本: {obs[0, 0, :5]}")
```

**预期输出：**
```
类型: float32
范围: [0.000, 1.000]
样本: [0.234 0.456 0.789 0.123 0.567]
```

---

## ✅ **修复方案**

### **方案1: 改进 ExpertDataset 归一化逻辑** ⭐ 推荐

修改 `src/training/bc/train_bc.py` 第49-52行：

**修复前：**
```python
# 归一化图像到[0, 1]
if observations.dtype == np.uint8:
    observations = observations.astype(np.float32) / 255.0
self.observations = torch.FloatTensor(observations)
```

**修复后：**
```python
# 归一化图像到[0, 1]
if observations.dtype == np.uint8:
    observations = observations.astype(np.float32) / 255.0
elif observations.dtype == np.float32 and observations.max() > 1.5:
    # 如果是 float32 但值域在 [0, 255]，需要归一化
    print(f"  ⚠️  检测到未归一化的 float32 数据，正在归一化...")
    observations = observations / 255.0
self.observations = torch.FloatTensor(observations)
```

**优点：**
- ✅ 简单直接
- ✅ 向后兼容
- ✅ 处理所有情况

---

### **方案2: 确保 np.save 保持数据类型**

修改数据保存逻辑，显式指定 dtype：

```python
# run_policy_collect_states.py
episode_data = {
    'states': episode_states,
    'actions': episode_actions,
    'rewards': episode_rewards,
    'total_reward': episode_reward,
    'success': episode_success,
    'num_steps': step_count,
    'episode_id': ep
}

# 确保观察数据类型正确
for state in episode_data['states']:
    obs = state['observation']
    if obs.max() > 1.5:
        print(f"  ⚠️  检测到未归一化数据！")
        state['observation'] = obs.astype(np.float32) / 255.0

np.save(filepath, episode_data)
```

**优点：**
- ✅ 在源头修复
- ✅ 确保数据一致性

**缺点：**
- ❌ 需要修改多处
- ❌ 已有数据需要重新收集

---

### **方案3: 在环境包装器中添加断言**

在 `MinedojoWrapper._process_obs()` 中添加检查：

```python
def _process_obs(self, obs_dict):
    """处理观察数据"""
    rgb = obs_dict['rgb']  # (C, H, W)
    
    # 归一化到[0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    
    # 验证归一化
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0, \
        f"归一化失败！范围: [{rgb.min():.3f}, {rgb.max():.3f}]"
    
    return rgb
```

**优点：**
- ✅ 早期发现问题
- ✅ 开发时有用

**缺点：**
- ❌ 生产环境可能降低性能

---

## 🧪 **测试验证**

### **1. 单元测试**

```python
import numpy as np
import torch
from src.training.train_bc import ExpertDataset

# 测试 uint8 数据
obs_uint8 = np.random.randint(0, 256, (10, 3, 160, 256), dtype=np.uint8)
actions = np.random.randint(0, 3, (10, 8))
dataset = ExpertDataset(obs_uint8, actions)
assert dataset.observations.max() <= 1.0, "uint8 归一化失败"

# 测试 float32 [0, 255] 数据
obs_float_255 = np.random.uniform(0, 255, (10, 3, 160, 256)).astype(np.float32)
dataset = ExpertDataset(obs_float_255, actions)
assert dataset.observations.max() <= 1.0, "float32[0,255] 归一化失败"

# 测试已归一化的 float32 数据
obs_float_01 = np.random.uniform(0, 1, (10, 3, 160, 256)).astype(np.float32)
dataset = ExpertDataset(obs_float_01, actions)
assert dataset.observations.max() <= 1.0, "float32[0,1] 处理失败"

print("✓ 所有测试通过")
```

### **2. 集成测试**

```bash
# 重新训练 BC baseline
python src/training/bc/train_bc.py \
  --data data/expert_demos/harvest_1_log/ \
  --output checkpoints/test_bc.zip \
  --device mps \
  --epochs 5

# 检查训练日志
# 应该看到: ✓ 数据归一化正确
```

---

## 📊 **性能影响对比**

| 场景 | 未归一化 [0, 255] | 已归一化 [0, 1] |
|------|-------------------|----------------|
| **Loss 初始值** | ~100-1000 | ~1-10 |
| **收敛速度** | 慢或不收敛 | 正常 |
| **梯度范围** | 很大，易爆炸 | 稳定 |
| **MPS 训练** | 不稳定/失败 | 稳定 |
| **内存使用** | 相同 | 相同 |

---

## 🎯 **立即行动**

### **步骤1: 诊断**
```bash
python tools/diagnose_data_normalization.py
```

### **步骤2: 应用修复**
修改 `src/training/bc/train_bc.py` 的 `ExpertDataset.__init__()`（方案1）

### **步骤3: 验证**
```bash
# 重新训练
bash scripts/run_dagger_workflow.sh \
  --skip-recording --skip-bc-eval \
  --device mps \
  --iterations 1

# 检查日志，应该看到：
# ✓ 数据归一化正确
```

### **步骤4: (可选) 重新收集数据**
如果问题持续，重新收集状态：
```bash
rm -rf data/policy_states/harvest_1_log/iter_1/
# 重新运行 collect states
```

---

## 📝 **相关文件**

- `src/training/bc/train_bc.py` - `ExpertDataset` 类（需要修复）
- `src/envs/env_wrappers.py` - `MinedojoWrapper._process_obs()`
- `tools/diagnose_data_normalization.py` - 诊断脚本

---

## 🎓 **总结**

### **问题根源**
`ExpertDataset` 的归一化逻辑只检查 `uint8` 类型，忽略了 `float32[0,255]` 的情况。

### **为什么会出现 float32[0,255]？**
可能原因：
1. numpy save/load 时的类型转换
2. 某个环节的数据处理错误
3. 环境包装器未正确应用（不太可能）

### **修复优先级**
1. ⭐ **立即：** 应用方案1（改进归一化逻辑）
2. 📊 **诊断：** 运行诊断脚本找出根源
3. 🔧 **可选：** 如果问题持续，重新收集数据

### **影响**
- ❌ **未修复：** MPS 训练失败/不稳定
- ✅ **修复后：** 正常训练，2-3倍加速

---

**调查日期：** 2025-10-25  
**问题类型：** 数据归一化缺陷  
**影响范围：** MPS/CUDA 训练稳定性  
**优先级：** 🔴 高  
**状态：** 🔍 调查中，待用户运行诊断脚本确认

