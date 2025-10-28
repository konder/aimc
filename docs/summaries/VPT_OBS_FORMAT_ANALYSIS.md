# VPT观察格式分析与修复

## 📋 问题分析

### MineRL vs MineDojo观察格式

**MineRL环境**:
- `obs['pov']`: RGB图像
- 格式: `(H, W, C)` - HWC (Height, Width, Channels)
- 类型: `numpy.ndarray`, `uint8`, 范围 [0, 255]

**MineDojo环境**:
- `obs['rgb']`: RGB图像  
- 格式: `(H, W, C)` - HWC (Height, Width, Channels)
- 类型: `numpy.ndarray`, `uint8`, 范围 [0, 255]

**结论**: ✅ **MineDojo和MineRL使用相同的HWC格式**

## 🔍 官方VPT代码流程

### 1. agent._env_obs_to_agent (agent.py:141-149)

```python
def _env_obs_to_agent(self, minerl_obs):
    agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
    agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
    return agent_input
```

**输出**: `{"img": tensor of shape (1, 128, 128, 3)}`
- `(1, H, W, C)` - 只有Batch维度，**缺少时间维度T**

### 2. ImpalaCNN期望输入 (lib/impala_cnn.py:187-191)

```python
def forward(self, x):
    b, t = x.shape[:-3]  # 期望 (B, T, H, W, C)
    x = x.reshape(b * t, *x.shape[-3:])
    x = misc.transpose(x, "bhwc", "bchw")  # HWC -> CHW转换
    x = tu.sequential(self.stacks, x, diag_name=self.name)
    x = x.reshape(b, t, *x.shape[1:])
```

**期望**: `(B, T, H, W, C)` - 需要时间维度T
**内部处理**: `misc.transpose(x, "bhwc", "bchw")` 自动完成HWC->CHW转换

## ⚠️ 核心问题

官方`agent._env_obs_to_agent`输出:
- ❌ `(1, H, W, C)` - 缺少时间维度T

ImpalaCNN期望:
- ✅ `(B, T, H, W, C)` - 需要时间维度T

**错误信息**:
```
b, t = x.shape[:-3]
# 当x是(1, 128, 128, 3)时:
# b=1, t=128 ❌ 错误！把Height当成了Time
# 导致后续shape错乱
```

## ✅ 解决方案

### 在VPTAgent中添加时间维度

**修改位置**: `src/training/vpt/vpt_agent.py`

```python
# 包装官方_env_obs_to_agent，添加时间维度
original_env_obs_to_agent = self.vpt_agent._env_obs_to_agent

def fixed_env_obs_to_agent(minerl_obs):
    # 调用原始方法：(H,W,C) -> (1,H,W,C) tensor
    result = original_env_obs_to_agent(minerl_obs)
    
    # 添加时间维度T：(1,H,W,C) -> (1,1,H,W,C)
    # ImpalaCNN期望(B,T,H,W,C)格式
    result["img"] = result["img"].unsqueeze(1)
    
    return result

self.vpt_agent._env_obs_to_agent = fixed_env_obs_to_agent
```

### 完整流程

1. **MineDojo obs**: `{'rgb': (160, 256, 3)}`
2. **VPTAgent.predict**: 提取 `pov = obs['rgb']` -> `(160, 256, 3)`
3. **minerl_obs**: `{"pov": (160, 256, 3)}`
4. **fixed_env_obs_to_agent**:
   - resize: `(160, 256, 3)` -> `(128, 128, 3)`
   - [None]: `(128, 128, 3)` -> `(1, 128, 128, 3)`
   - tensor: `(1, 128, 128, 3)` numpy -> torch
   - **unsqueeze(1)**: `(1, 128, 128, 3)` -> `(1, 1, 128, 128, 3)` ✓
5. **ImpalaCNN**:
   - 输入: `(1, 1, 128, 128, 3)` = `(B, T, H, W, C)` ✓
   - `misc.transpose("bhwc", "bchw")`: `(1, 3, 128, 128)` ✓
   - CNN处理成功！

## 🎯 关键发现

### HWC vs CHW

- **不需要手动转换**: ImpalaCNN内部通过`misc.transpose`自动完成
- **官方代码正确**: MineRL和MineDojo都使用HWC格式
- **无需monkey patch官方转换逻辑**

### 时间维度T

- **官方代码缺陷**: `_env_obs_to_agent`输出缺少T维度
- **原因**: 官方代码可能假设某处会添加T维度
- **我们的修复**: 在适配层添加`.unsqueeze(1)`

## 📊 测试结果

```bash
✅ VPT Agent测试通过！

测试结果：
  ✓ VPT Agent正确创建（组合官方MineRLAgent）
  ✓ 权重加载正确
  ✓ 能够接受观察并输出MineDojo动作
  ✓ Hidden state正确维护
  ✓ 时间维度T正确添加
```

## 📝 总结

### MineDojo -> MineRL 转换

1. **观察格式**: 无需转换（都是HWC）
2. **obs键名**: `obs['rgb']` -> `obs['pov']`
3. **时间维度**: 添加T维度 `(B,H,W,C)` -> `(B,T,H,W,C)`

### 架构优势

- ✅ 不修改官方Video-Pre-Training/代码
- ✅ 只在VPTAgent适配层添加必要的维度转换
- ✅ 尊重ImpalaCNN的内部HWC->CHW转换逻辑
- ✅ 最小化侵入，最大化兼容性

---

**文档日期**: 2025-10-27
**验证状态**: ✅ 已通过测试
**适用版本**: VPT官方代码 + MineDojo 1.0
