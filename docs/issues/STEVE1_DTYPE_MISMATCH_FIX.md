# STEVE-1 Dtype 不匹配问题修复

> **问题**: 在4090等支持混合精度的GPU上运行STEVE-1评估时，出现 `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float` 错误  
> **创建日期**: 2025-11-12  
> **状态**: ✅ 已修复

---

## 🐛 问题描述

### 错误信息

```bash
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
```

### 错误位置

```python
File "/tmp/steve1/steve1/embed_conditioned_policy.py", line 215, in forward
    mineclip_embed = self.mineclip_embed_linear(mineclip_embed)
```

### 完整堆栈

```
Traceback (most recent call last):
  File "/root/autodl-tmp/aimc/src/evaluation/steve1_evaluator.py", line 276, in _run_single_trial
    action = self._agent.get_action(obs, prompt_embed_np)
  File "/tmp/steve1/steve1/MineRLConditionalAgent.py", line 87, in get_action
    agent_action, self.hidden_state, _ = self.policy.act(
  File "/tmp/steve1/steve1/embed_conditioned_policy.py", line 339, in act
    (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in)
  File "/tmp/steve1/steve1/embed_conditioned_policy.py", line 284, in forward
    (pi_h, v_h), state_out = self.net(obs, state_in, context={"first": first})
  File "/tmp/steve1/steve1/embed_conditioned_policy.py", line 215, in forward
    mineclip_embed = self.mineclip_embed_linear(mineclip_embed)
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float
```

### 触发条件

- 在4090等支持混合精度的GPU上运行
- 使用官方steve1包的`get_prior_embed`函数
- 该函数内部使用了`torch.cuda.amp.autocast()`

---

## 🔍 根本原因

### 1. 混合精度自动转换

`steve1/utils/embed_utils.py`中的`get_prior_embed`函数使用了自动混合精度（AMP）：

```python
def get_prior_embed(text, mineclip, prior, device):
    with torch.cuda.amp.autocast():  # 自动转换为float16
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_embed = mineclip.encode_text(text)
            # ...
```

这会导致某些张量被自动转换为`float16`（Half精度）。

### 2. 模型权重仍为float32

但是，STEVE-1策略网络中的某些层（特别是`mineclip_embed_linear`）的权重仍然是`float32`。

### 3. Dtype不匹配

在前向传播时：

```python
# mineclip_embed 是 float16 (由AMP自动转换)
# self.mineclip_embed_linear.weight 是 float32
mineclip_embed = self.mineclip_embed_linear(mineclip_embed)  # ❌ 类型不匹配
```

PyTorch在矩阵乘法时要求输入和权重的dtype必须一致，因此报错。

### 4. 为什么其他GPU没问题？

- **4090等新GPU**: 原生支持float16，PyTorch会积极使用混合精度
- **较老的GPU**: 可能不支持float16，或PyTorch不会自动启用混合精度

---

## ✅ 修复方案

### 方案1: 转换嵌入为float32 (已实施)

**位置**: `src/evaluation/steve1_evaluator.py` (第262-265行)

在获取Prior嵌入后，确保转换为float32：

```python
# 使用 Prior 编码指令（官方方式）
logger.debug(f"  使用 Prior 编码指令: '{instruction}'")
with th.no_grad():
    # 使用官方的 get_prior_embed 函数
    prompt_embed = get_prior_embed(
        instruction,
        self._mineclip,
        self._prior,
        DEVICE
    )
    # 🔧 修复dtype问题: 确保嵌入是float32（针对4090等支持混合精度的GPU）
    if hasattr(prompt_embed, 'dtype') and prompt_embed.dtype == th.float16:
        logger.debug(f"  检测到 float16 嵌入，转换为 float32")
        prompt_embed = prompt_embed.float()
    
    # 转换为 numpy（MineRLConditionalAgent 需要）
    prompt_embed_np = prompt_embed.cpu().numpy() if hasattr(prompt_embed, 'cpu') else prompt_embed
```

**优点**:
- ✅ 直接修复问题的根源（嵌入dtype）
- ✅ 不修改官方steve1包代码
- ✅ 兼容所有GPU

**局限**:
- ⚠️ 在agent内部forward时可能仍被autocast影响

### 方案2: 确保Agent模型权重为float32 (已实施)

**位置**: `src/utils/steve1_mineclip_agent_env_utils.py` (第105-109行)

在加载Agent时，显式转换模型为float32：

```python
def make_agent(in_model, in_weights, cond_scale):
    print(f'Loading agent with cond_scale {cond_scale}...')
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs,
                                   pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    
    # 🔧 修复dtype问题: 确保模型权重是float32（针对4090等支持混合精度的GPU）
    # 将agent的policy网络转为float32，避免与float16嵌入混用时出错
    if hasattr(agent, 'policy') and hasattr(agent.policy, 'float'):
        agent.policy.float()
        print('  Agent policy 已转换为 float32')
    
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent
```

**优点**:
- ✅ 确保模型权重一致性
- ✅ 双重保险（结合方案1）
- ✅ 不修改官方steve1包代码

**局限**:
- ⚠️ 在agent内部forward时可能仍被autocast影响

### 方案3: 禁用agent推理时的autocast (已实施) ⭐ **关键修复**

**位置**: `src/evaluation/steve1_evaluator.py` (第293-297行)

在调用agent.get_action时显式禁用autocast：

```python
while not done and steps < max_steps:
    # 获取动作（使用 Prior 计算的嵌入）
    # 🔧 在no_grad环境下禁用autocast，防止dtype自动转换
    with th.no_grad():
        # 禁用autocast以防止float16自动转换
        with th.cuda.amp.autocast(enabled=False):
            action = self._agent.get_action(obs, prompt_embed_np)
```

**优点**:
- ✅ **彻底解决问题**：防止agent内部forward时被autocast影响
- ✅ 不修改官方steve1包代码
- ✅ 兼容所有GPU
- ✅ 性能影响极小

**为什么这个修复是关键**:
- 方案1和2只处理了输入和权重的dtype
- 但在4090等GPU上，agent内部的forward过程仍可能被全局autocast影响
- 需要显式禁用autocast来确保整个推理过程保持float32

### 方案4: 禁用AMP (不推荐)

修改官方steve1包的`embed_utils.py`，移除`autocast`：

```python
# ❌ 不推荐：需要修改官方包代码
def get_prior_embed(text, mineclip, prior, device):
    with torch.no_grad():  # 移除 autocast
        text_embed = mineclip.encode_text(text)
        # ...
```

**缺点**:
- ❌ 需要修改官方包（维护困难）
- ❌ 可能影响性能
- ❌ 升级steve1包时修改会丢失

---

## 🧪 测试验证

### 测试环境

```bash
GPU: NVIDIA RTX 4090
CUDA: 12.1
PyTorch: 2.0+
Python: 3.9+
```

### 测试命令

```bash
# 测试单个任务
python src/evaluation/eval_framework.py \
  --task-set quick_harvest_tasks \
  --n-trials 3 \
  --max-steps 2000 \
  --report-name evaluation_report

# 预期输出（不应报错）
[INFO] Trial 1/3...
[INFO]   结果: ✅ 成功, 步数: 234, 时间: 45.2s
```

### 验证要点

1. **不报dtype错误**: 不再出现 `mat1 and mat2 must have the same dtype` 错误
2. **正常推理**: Agent能够正常生成动作
3. **性能无影响**: 推理速度和成功率不受影响

---

## 📊 相关信息

### PyTorch混合精度文档

- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [torch.cuda.amp.autocast](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast)

### STEVE-1相关文件

```bash
# 官方steve1包（不要修改）
/tmp/steve1/steve1/utils/embed_utils.py
/tmp/steve1/steve1/embed_conditioned_policy.py
/tmp/steve1/steve1/MineRLConditionalAgent.py

# 本地修复文件（可以修改）
src/evaluation/steve1_evaluator.py
src/utils/steve1_mineclip_agent_env_utils.py
```

---

## 🎓 经验教训

### 1. 混合精度的隐患

在支持float16的新GPU上，PyTorch会积极使用混合精度，可能导致dtype不匹配。

**建议**:
- 显式检查和转换dtype
- 避免在不同精度间传递张量
- 测试多种GPU环境

### 2. 第三方库的兼容性

官方steve1包在新GPU上可能存在兼容性问题。

**建议**:
- 在本地工具函数中添加兼容性处理
- 不直接修改第三方包（维护困难）
- 保持本地修复代码的清晰注释

### 3. 防御性编程

即使问题出现在第三方库，也可以在调用侧添加防护。

**建议**:
- 在关键点添加dtype检查和转换
- 添加调试日志（如检测到float16时记录）
- 提供清晰的错误信息和修复建议

---

## 📝 总结

### 修复要点

1. ✅ 在`steve1_evaluator.py`中，将Prior嵌入转换为float32
2. ✅ 在`steve1_mineclip_agent_env_utils.py`中，确保Agent模型为float32
3. ✅ 不修改官方steve1包代码
4. ✅ 兼容所有GPU环境

### 适用范围

- 4090等新GPU（原生支持float16）
- 使用混合精度训练的环境
- 任何出现dtype不匹配错误的场景

### 后续优化

- [ ] 监控其他可能的dtype不匹配点
- [ ] 考虑在全局设置torch默认dtype
- [ ] 向steve1官方提交issue/PR

---

**相关文档**:
- [STEVE-1 评估指南](../guides/STEVE1_EVALUATION_GUIDE.md)
- [STEVE-1 技术分析](../technical/STEVE1_TRAINING_ANALYSIS.md)

