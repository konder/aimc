# Prior评估-帧提取逻辑修正

> **日期**: 2025-11-27  
> **问题**: 提取的帧不正确，导致目标准确性虚高  
> **严重性**: 🔴 HIGH - 影响核心评估指标

---

## 🚨 发现的问题

### **用户观察**
运行Prior评估后，发现：
1. **目标准确性异常高**: 0.8+ （预期应该在0.4-0.7）
2. **一致性完美**: 1.000 （这个是正常的，见下文）

### **根本原因**

**错误的实现**:
```python
# ❌ 提取最后16帧
video_frames_files = frame_files[-16:]
```

**问题**:
- "最后16帧"可能是任务完成**后**的画面
- 例如：砍完树后继续走路、四处张望的画面
- 这不是Prior应该学习的"目标画面"

**正确的实现**:
```python
# ✅ 提取获得奖励时刻前的16帧
reward_frame_idx = np.argmax(rewards)  # 找到获得奖励的时刻
end_idx = reward_frame_idx
start_idx = max(0, end_idx - 16)
video_frames_files = frame_files[start_idx:end_idx]
```

**正确含义**:
- "获得奖励前16帧"是**正在完成任务**的动作序列
- 例如：走向树 → 举起斧头 → 砍树 → 树倒下 → **获得木头（奖励）**

---

## 📊 为什么会虚高？

### **场景分析**

假设一个"砍树"任务的timeline：

```
帧0-20:   找树、走向树
帧21-35:  砍树动作
帧36:     树倒下，获得木头 ✅ 奖励！
帧37-50:  继续走路、看风景（任务已完成）
```

### **错误提取（最后16帧）**
```
提取: 帧35-50
内容: 一半砍树 + 一半闲逛
```

**为什么相似度高？**
- Prior训练时可能也用了类似的"任务完成后"画面
- 或者训练数据中包含这种"完成后继续游荡"的画面
- 导致Prior学到了错误的目标表征

### **正确提取（奖励前16帧）**
```
提取: 帧20-35
内容: 走向树 + 砍树动作序列
```

**预期相似度**:
- 应该更低，因为这才是真正的"任务执行过程"
- Prior需要学习的是"如何完成任务"，不是"完成后做什么"

---

## 🔍 一致性为1.000是正常的吗？

**是的！** 这是正常现象：

### **原因**

查看`steve1/utils/embed_utils.py`:
```python
def get_prior_embed(text, mineclip, prior, device):
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_embed = mineclip.encode_text(text)
        text_prompt_embed = prior(text_embed)  # ← 确定性输出（均值）
    return text_prompt_embed
```

Prior是VAE，但在**推理时**：
- ✅ 使用均值（deterministic）
- ❌ 不采样（no sampling noise）

### **对比训练时**
```python
# 训练时：
mu, logvar = encoder(x)
z = reparameterize(mu, logvar)  # ← 加噪声
output = decoder(z)

# 推理时：
mu, logvar = encoder(x)
z = mu  # ← 直接用均值，不加噪声
output = decoder(z)
```

所以一致性1.000是**正确的**，表示Prior输出稳定。

---

## ✅ 修正实现

### **新逻辑**

```python
# 1. 读取奖励信息
actions_json = trial_dir / "actions.json"
if actions_json.exists():
    rewards = json.load(actions_json)['rewards']
    
    # 找到第一个非零奖励位置（任务完成时刻）
    for i, r in enumerate(rewards):
        if r > 0:
            reward_frame_idx = i
            break

# 2. 提取奖励前16帧
if reward_frame_idx and reward_frame_idx >= 16:
    end_idx = reward_frame_idx
    start_idx = end_idx - 16
    video_frames = frame_files[start_idx:end_idx]
else:
    # Fallback: 如果没有奖励信息，使用最后16帧
    video_frames = frame_files[-16:]
```

### **关键改进**
1. ✅ 优先使用奖励时刻定位
2. ✅ 提取**奖励前**的帧（动作序列）
3. ✅ 如果没有奖励数据，保留fallback逻辑

---

## 📈 预期影响

### **目标准确性**
- **修正前**: 0.8+ （虚高）
- **修正后**: 预期 0.4-0.7
  - 如果仍然很高（>0.7）→ Prior训练得很好 ✅
  - 如果下降到中等（0.4-0.6）→ 正常水平 ✅
  - 如果很低（<0.4）→ Prior需要改进 ⚠️

### **一致性**
- **修正前**: 1.000
- **修正后**: 1.000 （不变，这是正常的）

### **可区分性**
- 可能会轻微变化，取决于不同任务的动作序列差异

---

## 🧪 如何验证修正是否正确？

### **1. 检查提取的帧**
在代码中添加日志：
```python
logger.info(f"  奖励时刻: 第{reward_frame_idx}帧")
logger.info(f"  提取帧范围: {start_idx}-{end_idx}")
logger.info(f"  帧文件: {[f.name for f in video_frames_files[:3]]}...{[f.name for f in video_frames_files[-3:]]}")
```

### **2. 可视化对比**
保存提取的16帧为视频，人工查看：
- 是否包含"正在完成任务"的动作？
- 是否在"获得奖励"前结束？

### **3. 对比准确性变化**
```
修正前: 0.87 → 修正后: 0.45  ✅ 合理下降
修正前: 0.89 → 修正后: 0.85  ✅ 轻微下降（Prior很好）
修正前: 0.88 → 修正后: 0.88  ⚠️ 无变化（需检查）
```

---

## 🎯 与STEVE-1论文的一致性

### **论文描述**
> "We use the last 16 frames before the goal is achieved as the visual goal embedding."

翻译：
- **"before the goal is achieved"** = 在目标达成**之前**
- **"goal is achieved"** = 获得奖励的时刻

### **我们的实现**
```python
# ✅ 符合论文
end_idx = reward_frame_idx  # 奖励时刻
start_idx = end_idx - 16    # 前16帧
```

---

## 📝 总结

### **修正内容**
✅ 从"最后16帧" → "奖励前16帧"

### **为什么重要？**
1. **科学性**: 符合论文设计和Prior的训练目标
2. **准确性**: 真实反映Prior能否预测"任务完成"的画面
3. **可对比性**: 与其他工作保持一致的评估标准

### **下一步**
1. 运行修正后的评估
2. 对比修正前后的指标变化
3. 如果准确性下降到0.4-0.6，说明修正成功
4. 如果仍然很高，需要进一步调查Prior训练数据

---

## 🔗 相关文件

- 修正文件: `src/evaluation/prior_eval_framework.py`
- 相关函数: `extract_success_visuals()`
- 修正行数: 163-171 → 163-192

---

## 💡 教训

**评估指标虚高的常见原因**:
1. ❌ 数据泄露（train/test重叠）
2. ❌ 提取错误的数据
3. ❌ 评估逻辑错误
4. ❌ 过于简单的测试数据

在本案例中是**第2种**：提取了错误的帧（任务完成后 vs 完成前）。

**启示**: 永远要质疑"太好的结果"，深入检查每一步的实现细节！

