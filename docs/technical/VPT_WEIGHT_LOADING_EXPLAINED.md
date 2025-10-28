# VPT权重加载详解

**📚 重要更新：** 本文档已更新为准确描述OpenAI官方VPT的实现方式。

**详细的问题发现和解决过程请参考：**
- `docs/summaries/VPT_WEIGHT_LOADING_FINAL_SUMMARY.md` - 完整的问题诊断和修复过程
- `docs/summaries/VPT_WEIGHT_LOADING_CLARIFICATION.md` - 问题根源和正确实现

---

## OpenAI官方实现

**参考：** [`/tmp/Video-Pre-Training/agent.py`](https://github.com/openai/Video-Pre-Training/blob/main/agent.py#L132-L135)

```python
def load_weights(self, path):
    """Load model weights from a path, and reset hidden state"""
    self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
    self.reset()
```

**关键特点：**
1. ✅ 直接加载整个state_dict
2. ✅ 不做任何前缀处理或过滤
3. ✅ 使用`strict=False`

---

## 权重文件和模型结构

### 权重文件包含（139个keys）：

```
net.* (125个)          - 视觉特征提取器 (MinecraftPolicy)
pi_head.* (4个)        - RL的action head
value_head.* (5个)     - RL的value head
aux_value_head.* (5个) - RL训练时的辅助head
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

### 官方加载结果：

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

## 为什么有Unexpected Keys？

### aux_value_head的来源

**VPT训练过程：**
1. **RL预训练（PPO）：** 使用强化学习在Minecraft中预训练
   - 需要value head估计状态价值
   - 使用aux_value_head作为辅助value估计器
   - 权重文件包含：net.*, pi_head.*, value_head.*, aux_value_head.*

2. **BC fine-tuning：** 使用行为克隆在特定任务上fine-tune
   - 只需要policy (net + pi_head)来预测动作
   - value_head保留（用于未来的RL fine-tuning）
   - **不需要aux_value_head**

### 结果

- 权重文件有`aux_value_head.*`（RL训练的产物）
- BC模型没有`aux_value_head`
- 加载时成为unexpected keys
- 使用`strict=False`忽略这些多余的参数

**这是完全正常的！** ✅

---

## 我们的实现（已对齐官方）

```python
def _load_weights(self, path: str):
    """完全遵循OpenAI官方实现"""
    state_dict = th.load(path, map_location=self.device)
    
    # 直接加载全部权重，不做任何处理（与官方完全一致）
    result = self.policy.load_state_dict(state_dict, strict=False)
    
    if self.verbose:
        print(f"  ✓ 权重加载完成")
        
        if len(result.unexpected_keys) > 0:
            print(f"\n  ℹ️  Unexpected keys: {len(result.unexpected_keys)}个")
            print(f"      → aux_value_head.* (RL训练专用，BC训练时不需要)")
            print(f"      → 已被忽略 (strict=False)")
    
    self.policy.eval()
```

**预期结果：**
- Missing keys: **0** ✅
- Unexpected keys: **5** (aux_value_head.*) ✅

---

## strict=False的作用

### PyTorch官方文档

参考：[torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)

```python
torch.nn.Module.load_state_dict(state_dict, strict=True)
```

**参数说明：**
- `strict` (bool) – 是否严格强制state_dict中的键与模型的state_dict()返回的键匹配

**当strict=False时：**
- 允许加载部分权重
- 忽略Missing keys（模型需要但权重中没有的参数会使用默认初始化）
- 忽略Unexpected keys（权重中有但模型不需要的参数会被丢弃）

### 使用场景

`strict=False`常用于：
- 迁移学习（修改了模型的部分结构）
- Fine-tuning（替换了分类头等）
- 加载预训练权重（权重文件比模型包含更多参数）

在VPT的情况下，`strict=False`用于忽略`aux_value_head.*`。

---

## 验证权重加载

### 方法1：检查模型参数

```python
# 打印模型的所有参数名
for name, param in agent.policy.named_parameters():
    print(name, param.shape)
```

### 方法2：验证输出

```python
# 创建dummy输入测试前向传播
import torch as th
dummy_obs = th.randn(1, 128, 128, 3)
agent_input = {"img": dummy_obs}
dummy_first = th.from_numpy(np.array((False,)))

# 测试
with th.no_grad():
    output = agent.policy(agent_input, dummy_first, agent.hidden_state)
    print(f"输出形状: {output}")
```

### 方法3：零样本评估

```bash
bash scripts/evaluate_vpt_zero_shot.sh 1 auto 100
```

如果agent能正常运行并产生合理的动作，说明权重加载正确。

---

## 常见问题

### Q1: 为什么之前有125个Missing keys和130个Unexpected keys？

**A:** 之前的代码错误地去掉了'net.'前缀：

```python
# ❌ 错误代码
if any(k.startswith('net.') for k in state_dict.keys()):
    state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
```

这导致：
- `net.img_process.xxx` → `img_process.xxx`
- 模型期望`net.*`但权重变成了`img_process.*`
- 完全不匹配！

**修复：** 移除这段错误的代码，直接加载原始权重。

### Q2: 是否应该过滤掉pi_head和value_head？

**A:** 不应该！官方的做法是加载所有可用的权重：

```python
# ✅ 正确：加载所有权重（包括pi_head和value_head）
result = self.policy.load_state_dict(state_dict, strict=False)
```

这样在BC fine-tuning时：
- 如果要fine-tune pi_head，从预训练权重开始
- 如果要从头训练pi_head，可以freeze其他层

保留预训练的pi_head权重给我们更多灵活性。

### Q3: 5个Unexpected keys会影响性能吗？

**A:** 不会！这些是`aux_value_head.*`，被`strict=False`忽略，不会加载到模型中。

核心的`net.*`、`pi_head.*`、`value_head.*`都已正确加载。

---

## 参考资料

1. **官方VPT代码**
   - GitHub: https://github.com/openai/Video-Pre-Training
   - 关键文件：`agent.py`, `lib/policy.py`, `behavioural_cloning.py`

2. **PyTorch文档**
   - load_state_dict: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict

3. **VPT论文**
   - Paper: Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos
   - arXiv: https://arxiv.org/abs/2206.11795

4. **我们的文档**
   - 问题诊断：`docs/summaries/VPT_WEIGHT_LOADING_FINAL_SUMMARY.md`
   - 问题澄清：`docs/summaries/VPT_WEIGHT_LOADING_CLARIFICATION.md`

---

**总结：** Missing keys和Unexpected keys不一定是错误，关键是要理解其原因并确认核心权重已正确加载。在VPT的情况下，5个unexpected keys (aux_value_head.*) 是完全正常的！✅
