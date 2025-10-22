# MineCLIP 16帧视频模式 - 完整使用指南

> **更新时间**: 2025-10-21  
> **版本**: v2.0 - 16帧视频模式  
> **目标**: 建立通用MineCLIP奖励框架，支持任意技能自主学习

---

## 🎯 **核心改进**

### **从单帧到16帧视频**

| 模式 | 输入 | MineCLIP处理 | 区分度 | 适用场景 |
|------|------|-------------|--------|----------|
| **单帧模式** | 1张图像 `[1, 3, H, W]` | `forward_image_features()` | 0.007 ❌ | 静态场景识别 |
| **16帧视频模式** | 16帧序列 `[1, 16, 3, H, W]` | `forward_image_features()` + `forward_video_features()` | ??? ⏳ | 动作识别、过程理解 |

**关键区别**:
- **单帧**: 只能看到"树"
- **16帧**: 能理解"砍树"这个动作

---

## 📦 **已完成的改造**

### **1. `src/utils/mineclip_reward.py`**

✅ 添加16帧缓冲机制 (`deque`)  
✅ 新增 `_encode_video()` 方法（官方完整流程）  
✅ 新增 `_compute_video_similarity()` 方法  
✅ 支持稀疏计算（每N步计算一次，减少开销）  
✅ 使用MineCraft官方归一化参数

**核心代码**:
```python
# 初始化
self.frame_buffer = deque(maxlen=16)  # 滚动窗口
self.compute_frequency = 4            # 每4步计算一次

# step方法
self.frame_buffer.append(obs.copy())  # 添加帧

if step_count % compute_frequency == 0:
    # 16帧 -> MineCLIP encode_video -> 相似度
    similarity = self._compute_video_similarity(list(self.frame_buffer))
```

### **2. `src/training/train_get_wood.py`**

✅ 添加 `--use-video-mode` 参数（默认启用）  
✅ 添加 `--num-frames` 参数（默认16）  
✅ 添加 `--compute-frequency` 参数（默认4）  
✅ 动态任务描述：视频模式用 `"chopping a tree with hand"`

### **3. 验证工具**

✅ `record_chopping_sequence.py` - 录制砍树过程  
✅ `verify_mineclip_16frames.py` - 分析16帧视频效果

---

## 🚀 **使用流程**

### **方案A: 使用已有logs/frames（快速）**

如果你之前训练时保存了帧（`--save-frames`），可以直接验证：

```bash
# 验证logs/frames中的帧序列
python verify_mineclip_16frames.py \
    --sequence-dir logs/frames \
    --num-frames 16 \
    --stride 4 \
    --task-prompt "chopping a tree with hand"

# 查看结果
cat logs/frames/similarity_results.txt
open logs/frames/similarity_analysis.png  # macOS
```

**预期输出**:
```
📊 相似度统计:
  最小值: 0.xxxx
  最大值: 0.xxxx
  变化范围: 0.xxxx  ← 关键指标！

🎯 评估结论:
  [根据变化范围判断效果]
```

---

### **方案B: 录制新的砍树序列（推荐）** 🎬

你提到可以录制真实的砍树过程，这是最好的验证方式！

**步骤1: 录制**

```bash
# 激活环境
conda activate minedojo-x86

# 运行录制脚本
python record_chopping_sequence.py \
    --output-dir logs/my_chopping \
    --max-frames 500

# 游戏启动后：
# - 手动控制角色
# - 寻找树
# - 砍树
# - 收集木头
# - 按Ctrl+C停止
```

**步骤2: 验证**

```bash
python verify_mineclip_16frames.py \
    --sequence-dir logs/my_chopping \
    --num-frames 16 \
    --stride 4 \
    --task-prompt "chopping a tree with hand"
```

**步骤3: 分析结果**

检查以下文件：
- `logs/my_chopping/similarity_results.txt` - 详细数据
- `logs/my_chopping/similarity_analysis.png` - 可视化图表

**关键指标解读**:
```
变化范围 > 0.05:  ✅ 优秀，建议使用16帧模式
变化范围 0.02-0.05: ⚠️  可用，需调整提示词
变化范围 < 0.02:  ❌ 效果不佳，考虑其他方案
```

---

### **方案C: 直接训练测试（10000步快速验证）**

```bash
# 使用16帧视频模式训练
./scripts/train_get_wood.sh test \
    --timesteps 10000 \
    --use-mineclip \
    --use-video-mode \
    --num-frames 16 \
    --compute-frequency 4 \
    --mineclip-weight 10.0 \
    --sparse-weight 10.0 \
    --device cpu \
    --headless

# 观察TensorBoard
tensorboard --logdir logs/tensorboard

# 关键指标：
# - 相似度变化幅度
# - explained_variance > 0
# - 训练稳定性
```

---

## 🔬 **验证要点**

### **1. 任务描述对比**

测试不同的文本描述：

```bash
# 动作导向（推荐for视频模式）
--task-prompt "chopping a tree with hand"
--task-prompt "punching a tree trunk"
--task-prompt "mining wood with hand"

# 过程导向
--task-prompt "tree trunk breaking"
--task-prompt "collecting wood logs"

# 视觉导向（适合单帧）
--task-prompt "tree"
--task-prompt "oak tree in minecraft"
```

### **2. 关键时刻识别**

如果录制了完整的砍树过程，检查相似度在以下阶段的变化：

1. **寻找树** (帧0-100): 相似度应该较低
2. **靠近树** (帧100-200): 相似度上升
3. **砍树中** (帧200-400): 相似度最高 ⭐
4. **收集木头** (帧400-500): 相似度下降

理想情况下，"砍树中"的相似度应该明显高于其他阶段。

### **3. 对比单帧vs16帧**

```bash
# 单帧模式
python verify_mineclip_16frames.py \
    --sequence-dir logs/my_chopping \
    --num-frames 1 \  # 单帧
    --stride 1 \
    --task-prompt "tree"

# 16帧模式
python verify_mineclip_16frames.py \
    --sequence-dir logs/my_chopping \
    --num-frames 16 \
    --stride 4 \
    --task-prompt "chopping a tree with hand"

# 对比两者的变化范围
```

---

## ⚙️ **训练参数说明**

### **16帧视频模式专用参数**

```bash
--use-video-mode          # 启用16帧视频模式（默认）
--no-video-mode           # 禁用，回退到单帧模式

--num-frames 16           # 视频帧数（默认16，官方标准）
                          # 可选：8, 12, 16, 20, 24

--compute-frequency 4     # 每N步计算一次MineCLIP
                          # 值越小越频繁，但开销越大
                          # 推荐：2-8之间
```

### **性能优化**

```bash
# 高频计算（更密集的奖励信号）
--compute-frequency 2     # 每2步计算一次

# 低频计算（减少计算开销）
--compute-frequency 8     # 每8步计算一次

# 更少帧数（更快但可能效果差）
--num-frames 8            # 8帧模式
```

---

## 📊 **预期效果**

### **成功标志**

1. **验证阶段**:
   - 相似度变化范围 > 0.05
   - "砍树中"帧的相似度明显高于"寻找树"帧
   - 可视化图表呈现明显的峰值

2. **训练阶段**:
   - explained_variance > 0（不是负数）
   - 相似度在训练过程中有波动（不是静态的0.63）
   - 训练稳定，没有loss爆炸

### **失败情况**

1. **区分度仍然很低** (< 0.02):
   - 可能MineCLIP不适合harvest wood任务
   - 考虑任务分解或混合奖励方案

2. **训练发散**:
   - 降低MineCLIP权重
   - 增加compute_frequency（减少奖励更新频率）
   - 使用动态权重衰减

---

## 🎯 **下一步计划**

### **如果16帧效果好** ✅

```
1. 扩展到其他任务
   - hunt_1_cow → "killing a cow"
   - mine_1_iron_ore → "mining iron ore with pickaxe"

2. 建立通用任务配置
   tasks.yaml:
     harvest_1_log:
       prompt: "chopping a tree with hand"
       frames: 16
       freq: 4
     
     hunt_1_cow:
       prompt: "attacking a cow"
       frames: 16
       freq: 4

3. 实现自动任务学习
   python train_universal.py --task harvest_1_log
   python train_universal.py --task hunt_1_cow
```

### **如果16帧效果一般** ⚠️

```
1. 优化提示词（Prompt Engineering）
   - 测试100+种不同描述
   - 使用大模型生成候选描述

2. 混合奖励方案
   total_reward = (
       sparse_reward * 10.0 +
       inventory_reward * 1.0 +  # 简单库存变化
       mineclip_reward * 2.0     # 16帧视频辅助
   )

3. 任务分解 + 分层RL
   - 子任务1: find_tree
   - 子任务2: approach_tree
   - 子任务3: chop_tree
   - 高层策略决定切换
```

---

## 📝 **完整命令示例**

### **场景1: 录制并验证**

```bash
# 1. 录制砍树过程（手动控制）
python record_chopping_sequence.py \
    --output-dir logs/manual_chopping \
    --max-frames 500

# 2. 验证16帧效果
python verify_mineclip_16frames.py \
    --sequence-dir logs/manual_chopping \
    --num-frames 16 \
    --task-prompt "chopping a tree with hand"

# 3. 查看结果
cat logs/manual_chopping/similarity_results.txt
open logs/manual_chopping/similarity_analysis.png
```

### **场景2: 直接训练测试**

```bash
# 10000步快速测试
./scripts/train_get_wood.sh test \
    --timesteps 10000 \
    --use-mineclip \
    --use-video-mode \
    --num-frames 16 \
    --compute-frequency 4 \
    --mineclip-weight 10.0 \
    --device cpu \
    --headless

# 观察日志
tail -f logs/training/training_*.log

# TensorBoard可视化
tensorboard --logdir logs/tensorboard
```

### **场景3: 完整训练（100000步）**

```bash
# 如果验证效果好，运行完整训练
./scripts/train_get_wood.sh full \
    --timesteps 100000 \
    --use-mineclip \
    --use-video-mode \
    --num-frames 16 \
    --compute-frequency 4 \
    --mineclip-weight 10.0 \
    --sparse-weight 10.0 \
    --use-dynamic-weight \
    --weight-decay-steps 50000 \
    --device cpu \
    --headless
```

---

## ❓ **FAQ**

**Q1: 为什么默认compute_frequency=4而不是1？**

A: 每步计算MineCLIP开销很大。设置为4意味着：
- 每4步才计算一次相似度
- 其他3步重用上次的相似度
- 减少75%的计算量
- 仍能提供有效的奖励信号

**Q2: num_frames可以设置为8或32吗？**

A: 可以，但：
- 8帧：更快，但可能丢失时序信息
- 16帧：官方标准，推荐 ⭐
- 32帧：更多上下文，但开销大

**Q3: 如果验证结果显示区分度很低怎么办？**

A: 
1. 尝试不同的task_prompt
2. 确保录制的是完整的砍树过程（包含动作）
3. 检查是否在正确的时刻（砍树中）相似度最高
4. 如果仍然很低，考虑MineCLIP可能不适合这个任务

**Q4: 可以同时使用视频模式和库存奖励吗？**

A: 可以！修改`mineclip_reward.py`在`step`方法中添加库存检查：
```python
# 库存奖励
if 'log' in info.get('inventory', {}):
    inventory_reward = 10.0
else:
    inventory_reward = 0.0

total_reward = (
    sparse_reward * sparse_weight +
    mineclip_reward * mineclip_weight +
    inventory_reward * 1.0
)
```

---

## ✅ **检查清单**

在开始训练前，确认：

- [ ] MineCLIP模型已加载: `data/mineclip/attn.pth`
- [ ] CLIP tokenizer已下载: `data/clip_tokenizer/`
- [ ] 环境已激活: `conda activate minedojo-x86`
- [ ] 如果录制，确保有足够磁盘空间（500帧 ≈ 50MB）
- [ ] 如果训练，确认无头模式配置正确

---

## 📞 **总结**

你现在有三个选择：

1. **方案A**: 使用已有的logs/frames验证 → 5分钟
2. **方案B**: 录制新的砍树序列验证 → 15分钟 ⭐ 推荐
3. **方案C**: 直接训练10000步测试 → 30分钟

**我的建议**: 先录制一个真实的砍树过程（方案B），这样能最准确地评估16帧MineCLIP的效果。

如果验证结果好（变化范围 > 0.05），我们就全力推进16帧视频模式训练。  
如果结果一般，我们讨论混合奖励或任务分解方案。

**准备好录制了吗？** 🎬

