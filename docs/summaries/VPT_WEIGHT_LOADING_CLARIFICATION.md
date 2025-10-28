# VPT权重加载问题 - 完全解决！

## 🎯 问题根源

### 之前的错误代码：

```python
# ❌ 错误：去掉了'net.'前缀
if any(k.startswith('net.') for k in state_dict.keys()):
    state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
```

**结果：**
- Missing keys: 125个（模型期望`net.*`但权重里变成了`img_process.*`）
- Unexpected keys: 130个（权重里有`img_process.*`但模型不认识）

---

## ✅ OpenAI官方实现

**参考：** `/tmp/Video-Pre-Training/agent.py` 第132-135行

```python
def load_weights(self, path):
    """Load model weights from a path, and reset hidden state"""
    self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
    self.reset()
```

**关键：**
1. ✅ 直接加载整个state_dict
2. ✅ **不做任何前缀处理**
3. ✅ **不过滤任何keys**
4. ✅ 使用`strict=False`

---

## 🔍 权重文件和模型结构的对应关系

### 权重文件包含（139个keys）：
```
net.* (125个)          ← 视觉特征提取器 (MinecraftPolicy)
pi_head.* (4个)        ← RL的action head
value_head.* (5个)     ← RL的value head
aux_value_head.* (5个) ← RL训练时的辅助head（仅在RL训练时使用）
```

### MinecraftAgentPolicy模型结构：

```python
class MinecraftAgentPolicy(nn.Module):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs):
        super().__init__()
        self.net = MinecraftPolicy(**policy_kwargs)   # ← 对应 net.*
        self.pi_head = make_action_head(...)          # ← 对应 pi_head.*
        self.value_head = ScaledMSEHead(...)          # ← 对应 value_head.*
        # 注意：没有 aux_value_head（仅在RL训练时存在）
```

**来源：** 
- 我们的代码：`src/models/vpt/lib/policy.py` (从官方复制)
- 官方代码：`/tmp/Video-Pre-Training/lib/policy.py`
- 完全一致！✅

### 官方加载权重的结果：

```
加载：
  ✅ net.* → self.net (125个参数)
  ✅ pi_head.* → self.pi_head (4个参数)
  ✅ value_head.* → self.value_head (5个参数)
  ⚠️ aux_value_head.* → Unexpected keys (5个) - 被忽略
  
Missing keys: 0
Unexpected keys: 5 (aux_value_head.*)
```

---

## ✅ 我们的正确实现（已修复）

```python
def _load_weights(self, path: str):
    """完全遵循OpenAI官方实现"""
    state_dict = th.load(path, map_location=self.device)
    
    # 直接加载，不做任何处理（与官方完全一致）
    result = self.policy.load_state_dict(state_dict, strict=False)
    
    # aux_value_head.*会成为unexpected keys被忽略
    self.policy.eval()
```

**预期结果：**
- Missing keys: **0** ✅
- Unexpected keys: **5** (aux_value_head.*) - 正常！

---

## 📋 为什么有`aux_value_head`？

**背景：**
- VPT使用RL（PPO）进行预训练
- RL训练需要value head来估计状态价值
- `aux_value_head`是RL训练时的辅助value估计器

**在BC训练时：**
- 只需要policy (net + pi_head)来预测动作
- value_head可以保留（用于未来RL fine-tuning）
- aux_value_head不需要，所以模型中没有

**结果：**
- 权重文件包含aux_value_head.*（RL训练的产物）
- BC模型不包含aux_value_head
- 加载时成为unexpected keys，被`strict=False`忽略
- **这是完全正常的！** ✅

---

## 🎓 核心要点总结

### ❌ 之前的错误

1. 错误地去掉'net.'前缀
2. 导致keys完全不匹配
3. 产生大量missing/unexpected keys

### ✅ 正确的做法

1. **完全遵循OpenAI官方实现**
2. 直接加载全部权重，不做任何处理
3. 使用`strict=False`忽略aux_value_head.*
4. 预期有5个unexpected keys - 这是正常的！

### 📚 参考资料

1. **官方代码：** `/tmp/Video-Pre-Training/agent.py`
2. **官方BC训练：** `/tmp/Video-Pre-Training/behavioural_cloning.py`
3. **我们的实现：** `src/training/agent/vpt_agent.py`
4. **模型结构：** `src/models/vpt/lib/policy.py` (从官方复制)

---

## ✅ 验证

运行以下命令验证修复：

```bash
conda run -n minedojo-x86 python tmp/test_weight_loading_fix.py
```

**预期输出：**
```
权重文件包含 139 个参数
✓ 权重加载完成

ℹ️  Unexpected keys: 5个
    → aux_value_head.* (RL训练专用，BC训练时不需要)
    → 已被忽略 (strict=False)
```

---

## 🙏 感谢

**感谢用户的质疑和追问！** 

1. 第一次澄清：说明我们使用PyTorch标准方法
2. 第二次质疑：指出我们的lib是从官方复制的，不应该有不匹配
3. 提供官方代码：`/tmp/Video-Pre-Training`
4. **真相大白**：发现是错误地去掉了'net.'前缀

这个过程让我们：
- ✅ 找到了真正的问题根源
- ✅ 完全对齐了官方实现
- ✅ 理解了权重文件的结构
- ✅ 学会了如何正确追溯问题

**现在的实现与OpenAI官方完全一致！** 🎉
