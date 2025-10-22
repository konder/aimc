# 环境参数统一修复总结

> **问题**: 多个脚本调用`make_minedojo_env`时使用了错误的参数

---

## 🐛 **问题根源**

所有脚本都使用了不存在的参数：
- ❌ `use_mineclip` - 不存在
- ❌ `max_steps` - 参数名错误（应该是`max_episode_steps`）

---

## ✅ **修复清单**

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/training/train_bc.py` | ✅ 已修复 | BC训练环境创建 |
| `tools/evaluate_policy.py` | ✅ 已修复 | 策略评估 + `fast_reset=False` |
| `tools/run_policy_collect_states.py` | ✅ 已修复 | DAgger状态收集 + `fast_reset=False` |
| `src/utils/env_wrappers.py` | ✅ 已更新 | 添加`fast_reset`参数支持 |

---

## 📝 **标准环境创建模式**

### **训练时（fast_reset=True，默认）**

```python
env = make_minedojo_env(
    task_id="harvest_1_log",
    use_camera_smoothing=True,  # 减少抖动
    max_episode_steps=1000,
    fast_reset=True  # 默认值，快速重置
)
```

**特点**:
- ✅ Reset快（1-2秒）
- ❌ 重用世界（数据多样性低）
- ✅ 适合训练（大量episode）

---

### **评估时（fast_reset=False）**

```python
env = make_minedojo_env(
    task_id="harvest_1_log",
    use_camera_smoothing=False,  # 评估时不需要
    max_episode_steps=1000,
    fast_reset=False  # 每个episode独立
)
```

**特点**:
- ❌ Reset慢（5-10秒）
- ✅ 新世界（数据多样性高）
- ✅ 适合评估（少量episode）

---

### **DAgger收集（fast_reset=False）**

```python
env = make_minedojo_env(
    task_id="harvest_1_log",
    use_camera_smoothing=False,
    max_episode_steps=1000,
    fast_reset=False  # 收集多样状态
)
```

**特点**:
- ✅ 每个episode独立环境
- ✅ 避免策略在同一环境反复失败
- ✅ 收集到的状态更具代表性

---

## 🔧 **修复的具体问题**

### **1. 参数名错误**

```python
# ❌ 错误
env = make_minedojo_env(
    use_mineclip=False,  # 不存在
    max_steps=1000       # 参数名错误
)

# ✅ 正确
env = make_minedojo_env(
    max_episode_steps=1000  # 正确的参数名
)
```

---

### **2. fast_reset参数缺失**

```python
# ❌ 之前（硬编码True）
def make_minedojo_env(...):
    env = minedojo.make(
        fast_reset=True,  # 硬编码
        ...
    )

# ✅ 现在（可配置）
def make_minedojo_env(..., fast_reset=True):
    env = minedojo.make(
        fast_reset=fast_reset,  # 参数传递
        ...
    )
```

---

### **3. 评估环境不独立**

```python
# ❌ 之前
# Episode 1: 新世界 → 成功
# Episode 2-20: 重用世界 → 失败（树已被砍）

# ✅ 现在
env = make_minedojo_env(fast_reset=False)
# Episode 1: 世界A → 独立测试
# Episode 2: 世界B → 独立测试
# ...
```

---

## 📊 **fast_reset对比**

| 场景 | fast_reset | Reset时间 | 环境多样性 | 推荐 |
|------|-----------|----------|-----------|------|
| **RL训练** | True | 1-2秒 | 低 | ✅ 默认 |
| **BC训练** | True | 1-2秒 | 低 | ✅ 不需要环境交互 |
| **策略评估** | False | 5-10秒 | 高 | ✅ 准确性优先 |
| **DAgger收集** | False | 5-10秒 | 高 | ✅ 多样性优先 |
| **手动录制** | False | 5-10秒 | 高 | ✅ 用户可配置 |

---

## 🎯 **修复影响**

### **BC训练**
- ✅ 正常训练（不需要环境交互）
- ✅ Loss正常下降（9.2 → 0.16）
- ✅ 可以保存模型

### **策略评估**
- ✅ 每个episode独立测试
- ✅ 评估结果准确（不会因环境相同而失真）
- ⏱️ 评估时间增加（~3-5分钟/20 episodes）

### **DAgger收集**
- ✅ 收集到多样化的失败状态
- ✅ 避免在同一环境反复失败
- ✅ 人类专家看到不同情况

---

## 📚 **相关Commits**

1. **`3a1cca2`** - BC训练参数修复
2. **`32e10b1`** - BC训练normalize_images修复
3. **`9349888`** - BC训练图像维度转置修复
4. **`9d4e02d`** - evaluate_policy参数修复
5. **`ee7cb3d`** - fast_reset参数支持
6. **`3dfa2b0`** - run_policy_collect_states参数修复

---

## ✅ **验证清单**

- [x] BC训练可以正常运行
- [x] 策略评估可以正常运行
- [x] DAgger收集可以正常运行
- [x] 所有脚本使用一致的参数
- [x] fast_reset在正确的场景使用
- [x] 文档已更新

---

## 🔗 **相关文档**

- [`BC_TRAINING_READY.md`](../status/BC_TRAINING_READY.md) - BC训练就绪状态
- [`FAST_RESET_PARAMETER_GUIDE.md`](../guides/FAST_RESET_PARAMETER_GUIDE.md) - fast_reset参数说明
- [`DAGGER_QUICK_START.md`](../guides/DAGGER_QUICK_START.md) - DAgger快速开始

---

**最后更新**: 2025-10-21  
**状态**: ✅ 所有环境参数已统一修复

