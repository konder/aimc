# MineCLIP + 密集奖励实施计划

> **目标**: 改善harvest wood训练效果  
> **策略**: 分阶段优化，从简单到复杂

---

## 📊 **问题总结**

1. ✅ **MineCLIP正确实现已验证**：归一化参数、temporal encoder
2. ❌ **单帧MineCLIP区分度太低**：0.007，无法有效引导
3. ❌ **纯稀疏奖励失败**：50万步训练崩溃
4. ✅ **需要密集奖励**：用户确认

---

## 🎯 **三阶段实施方案**

### **阶段1：简单库存奖励（1小时）** ⭐⭐⭐⭐⭐

**目标**: 快速验证密集奖励是否有效

**实现**:
```python
# 最简单的奖励：库存中出现木头 = +1.0
class InventoryBasedReward(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        inventory = info.get('inventory', {})
        if inventory.get('log', 0) > self.last_count:
            reward += 1.0  # 获得木头立即奖励
        return obs, reward, done, info
```

**优点**:
- ✅ 实现简单（已完成）
- ✅ 信号明确
- ✅ 不依赖MineCLIP

**测试方法**:
```bash
# 10000步快速测试
./scripts/train_get_wood.sh test \
    --timesteps 10000 \
    --learning-rate 0.0001 \
    --device cpu \
    --headless
# 使用简单库存奖励（在train_get_wood.py中集成）
```

**预期效果**:
- explained_variance > 0
- 训练稳定
- 可能在5000-10000步首次获得木头

---

### **阶段2：混合奖励（2小时）** ⭐⭐⭐⭐

**目标**: 结合库存奖励 + 动作鼓励

**实现**:
```python
class SimpleDenseRewardWrapper(gym.Wrapper):
    def step(self, action):
        dense_reward = 0.0
        
        # 1. 库存奖励（主要）
        if got_log:
            dense_reward += 10.0
        
        # 2. 攻击动作鼓励（次要）
        if action[5] == 1:  # 攻击
            dense_reward += 0.01
        
        # 3. 移动惩罚（可选）
        if action[0] != 0:
            dense_reward -= 0.001
        
        return obs, reward + dense_reward, done, info
```

**优点**:
- ✅ 多信号引导
- ✅ 鼓励探索攻击动作
- ✅ 减少无意义移动

**权重调整**:
- 库存奖励: 1.0 - 10.0（根据阶段1结果调整）
- 攻击奖励: 0.001 - 0.1
- 移动惩罚: -0.01 - 0.0

---

### **阶段3：16帧MineCLIP（1天）** ⭐⭐⭐

**目标**: 使用完整MineCLIP（如建议所示）

**需要实现**:

#### **3.1 帧缓存机制**

```python
class FrameBuffer:
    def __init__(self, max_frames=16):
        self.max_frames = max_frames
        self.frames = []
    
    def add(self, frame):
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
    
    def get_video(self):
        """返回 [1, T, 3, H, W] 格式"""
        if len(self.frames) < self.max_frames:
            return None  # 帧数不足
        return np.stack(self.frames)[np.newaxis, ...]
```

#### **3.2 MineCLIP奖励计算**

```python
class MineCLIPRewardWrapper(gym.Wrapper):
    def __init__(self, env, mineclip, task_prompt, frame_stack=16):
        self.frame_buffer = FrameBuffer(frame_stack)
        self.mineclip = mineclip
        self.task_prompt = task_prompt
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 添加帧到缓存
        self.frame_buffer.add(obs)  # 假设obs是图像
        
        # 每N步计算一次MineCLIP奖励
        if self.step_count % 16 == 0:
            video = self.frame_buffer.get_video()
            if video is not None:
                # 使用官方完整流程
                video_emb = self.mineclip.encode_video(video)
                text_emb = self.mineclip.encode_text(self.task_prompt)
                similarity = cosine_similarity(video_emb, text_emb)
                
                # MineCLIP奖励
                mineclip_reward = similarity * 2.0
                reward += mineclip_reward
        
        return obs, reward, done, info
```

#### **3.3 预处理函数**

```python
def preprocess_frames(frames):
    """
    Args:
        frames: [T, H, W, 3] numpy array, uint8, [0, 255]
    Returns:
        video: [1, T, 3, H, W] tensor, float32, normalized
    """
    processed = []
    for frame in frames:
        # 1. 调整大小
        frame = cv2.resize(frame, (256, 160))  # MineDojo标准
        
        # 2. 归一化到[0,1]
        frame = frame.astype(np.float32) / 255.0
        
        # 3. HWC -> CHW
        frame = frame.transpose(2, 0, 1)
        
        # 4. MineCraft归一化
        mean = np.array([0.3331, 0.3245, 0.3051]).reshape(3, 1, 1)
        std = np.array([0.2439, 0.2493, 0.2873]).reshape(3, 1, 1)
        frame = (frame - mean) / std
        
        processed.append(frame)
    
    # 堆叠并添加batch维度
    video = np.stack(processed)[np.newaxis, ...]  # [1, T, 3, H, W]
    return torch.from_numpy(video).float()
```

**挑战**:
- ⚠️ 性能开销（每16步计算一次MineCLIP）
- ⚠️ 内存占用（缓存16帧）
- ⚠️ 奖励延迟（16步才有MineCLIP信号）

**优化**:
- 使用GPU加速MineCLIP
- 降低帧率（如每4帧采样1帧，总共采样16帧）
- 异步计算MineCLIP（不阻塞训练）

---

## 📋 **推荐执行顺序**

### **Day 1: 阶段1快速验证**

```bash
# 1. 在train_get_wood.py中集成InventoryBasedReward
# 2. 测试10000步
./scripts/train_get_wood.sh test --timesteps 10000 --device cpu

# 3. 观察TensorBoard
tensorboard --logdir logs/tensorboard

# 4. 检查指标：
# - explained_variance > 0 ✅
# - 首次获得木头 < 5000步 ✅
# - 训练稳定不发散 ✅
```

**如果成功** → 继续训练100000步  
**如果失败** → 调整奖励权重或检查环境

---

### **Day 2: 阶段2优化**

```bash
# 添加动作鼓励和移动惩罚
# 测试不同权重组合：
# - 库存: 1.0, 攻击: 0.01, 移动: -0.001
# - 库存: 5.0, 攻击: 0.05, 移动: -0.01
# - 库存: 10.0, 攻击: 0.1, 移动: 0.0
```

---

### **Day 3-4: 阶段3（可选）**

只有在阶段1-2效果仍不理想时才考虑。

```python
# 实现16帧MineCLIP
# 结合库存奖励：
total_reward = (
    inventory_reward * 10.0 +      # 主要信号
    mineclip_reward * 2.0 +        # 辅助信号
    action_reward * 0.1            # 微调信号
)
```

---

## ⚡ **立即行动：集成阶段1**

修改 `train_get_wood.py`，添加简单库存奖励选项：

```python
# 在create_harvest_log_env中添加
def create_harvest_log_env(..., use_inventory_reward=False):
    env = make_minedojo_env(...)
    
    # 简单库存奖励（优先于MineCLIP）
    if use_inventory_reward:
        from src.utils.simple_dense_reward import InventoryBasedReward
        env = InventoryBasedReward(env, target_item='log', reward_per_item=1.0)
    
    # MineCLIP奖励（可选）
    elif use_mineclip:
        env = MineCLIPRewardWrapper(...)
    
    env = Monitor(env)
    return env
```

修改 `train_get_wood.sh`：

```bash
# 添加参数
INVENTORY_REWARD=""  # 默认不启用

# 解析参数
--inventory-reward)
    INVENTORY_REWARD="--inventory-reward"
    shift
    ;;

# 传递给Python
$INVENTORY_REWARD
```

---

## 📊 **预期效果对比**

| 方案 | 实现难度 | 训练时间 | 成功率预测 |
|------|---------|---------|-----------|
| **纯稀疏** | ⭐ | 慢 | 10% ❌ |
| **库存奖励** | ⭐ | 快 | 70% ✅ |
| **混合奖励** | ⭐⭐ | 中 | 85% ✅ |
| **16帧MineCLIP** | ⭐⭐⭐⭐ | 中 | 90% ⭐ |

---

## 🎯 **关键建议**

1. **先简单后复杂**：库存奖励可能已经足够
2. **快速迭代**：每个阶段只测试10000步
3. **观察TensorBoard**：explained_variance是关键指标
4. **权重调整**：根据实际效果调整，不要盲目照搬

---

## 💬 **下一步**

**立即测试阶段1？**
```bash
# 我可以帮你：
# 1. 修改train_get_wood.py集成库存奖励
# 2. 更新train_get_wood.sh添加参数
# 3. 运行10000步快速测试
# 4. 分析结果决定下一步
```

要不要现在就开始实施阶段1？这应该能在1小时内看到效果！🚀

