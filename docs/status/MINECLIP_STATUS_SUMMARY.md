# MineCLIP 当前状态总结

**时间**: 2025-10-21 17:00

---

## ✅ **已完成的工作**

### 1. **成功录制完整的砍树序列** 
- ✅ 272帧手动录制
- ✅ 包含完整的"寻找树木 → 接近树木 → 面对树木 → 攻击砍树"过程
- ✅ 修复了攻击动作映射（`functional=3`）
- 📂 位置: `logs/my_chopping/`

### 2. **MineCLIP集成已完成**
- ✅ 16帧视频模式已实现
- ✅ 正确的Minecraft归一化参数
- ✅ 正确的temporal encoder使用
- ✅ 训练代码已集成（`src/training/train_get_wood.py`）

### 3. **创建了优化测试工具**
- ✅ `tools/quick_optimize_mineclip.py` - Prompt优化测试
- ✅ `tools/optimize_mineclip_config.py` - 完整配置优化
- ✅ `tools/record_manual_chopping.py` - 手动录制工具
- ✅ `tools/verify_mineclip_16frames.py` - 相似度验证

---

## ⚠️ **核心问题**

### **MineCLIP相似度变化范围太小**

| 测试 | Prompt | 相似度范围 | 变化百分比 | 结论 |
|------|--------|-----------|-----------|------|
| 用户录制分析 | `chopping a tree with hand` | 0.0035 | 0.35% | ❌ 太小 |
| 单prompt测试 | `tree` | 0.0013 | 0.47% | ❌ 太小 |

**对比**:
- **当前**: 0.35-0.47% 变化范围
- **最低要求**: >1% 
- **理想目标**: >5%

**原因分析**:
1. ❓ **Prompt可能不够精确** - 需要测试更多prompt找到最佳描述
2. ❓ **MineCLIP对简单场景区分度低** - "面对树"vs"看天空"可能本身差异就小
3. ❓ **任务太简单** - 砍树任务在视觉上变化可能不如预期明显

---

## 🔍 **下一步选项**

### **选项 1: 运行完整Prompt优化测试** ⏱️ ~15-20分钟

**目标**: 找到相似度变化范围最大的prompt

**测试内容**: 15个不同prompt
```python
test_prompts = [
    "chopping a tree with hand",  # 当前使用
    "a tree", "tree", "oak tree", "tree trunk", "wood blocks",  # 简单物体
    "looking at tree", "facing a tree", "tree in front",  # 视觉位置
    "breaking tree", "punching tree", "mining wood", "cutting wood",  # 动作
    "forest", "trees in minecraft"  # 场景
]
```

**命令**:
```bash
cd /Users/nanzhang/aimc
conda run -n minedojo-x86 python tools/quick_optimize_mineclip.py
```

**预期输出**:
- `logs/mineclip_optimization/prompt_optimization.png` - 对比图
- `logs/mineclip_optimization/prompt_results.txt` - 详细结果

**预期结果**:
- **最好情况**: 找到变化范围>2%的prompt → 可以用于训练
- **一般情况**: 所有prompt都<1% → 需要混合奖励策略
- **最坏情况**: 所有prompt都<0.5% → MineCLIP对这个任务不适用

---

### **选项 2: 直接采用混合奖励策略** ⚡ 立即可行

基于当前结果（变化范围<0.5%），直接承认MineCLIP单独使用效果不佳，改用混合策略：

**方案**: MineCLIP (低权重) + 环境事件奖励 (高权重)

```python
# 伪代码
dense_reward = 0

# MineCLIP奖励（权重降低）
dense_reward += mineclip_similarity_diff * 5.0  # 从40降到5

# 环境事件奖励（新增）
if inventory.contains_wood:
    dense_reward += 100.0  # 收集到木头
elif is_attacking_tree:
    dense_reward += 5.0    # 正在砍树
elif facing_tree and distance < 3.0:
    dense_reward += 1.0    # 靠近树木
elif can_see_tree:
    dense_reward += 0.1    # 看到树木
```

**优点**:
- ✅ 立即可用
- ✅ 奖励信号更明确
- ✅ 更容易调试

**缺点**:
- ❌ 需要手动设计奖励
- ❌ 不够通用（每个任务都要设计）

---

### **选项 3: 切换到更复杂的任务测试MineCLIP**

**假设**: MineCLIP可能在更复杂的视觉任务上表现更好

**建议任务**:
- `harvest_1_iron_pickaxe` - 制作铁镐（需要找矿、熔炼、合成）
- `combat_spider_plains` - 战斗任务（动态场景）
- `navigate_1_diamond` - 寻找钻石（需要深度探索）

**原理**: 这些任务的视觉变化更丰富（洞穴vs地表、敌人vs无敌人），MineCLIP可能提供更大的相似度变化范围。

---

## 🎯 **我的建议**

基于已测试的两个prompt结果（都<0.5%），我建议：

1. **⏰ 如果有时间**: 
   - 先运行完整的Prompt优化测试（15-20分钟）
   - 如果找到>1%的prompt，继续使用Min eCLIP
   - 如果所有都<1%，切换到混合策略

2. **⚡ 如果想快速推进**:
   - 直接采用混合奖励策略（选项2）
   - MineCLIP作为辅助（低权重）
   - 环境事件奖励作为主导（高权重）

3. **🔬 如果想深入验证MineCLIP**:
   - 先用提供的3张图片（全天空、半泥土、全泥土）测试
   - 验证MineCLIP是否能区分极端场景
   - 然后决定是否值得继续优化

---

## 📊 **技术细节**

### **MineCLIP配置（已验证正确）**

```python
model = MineCLIP(
    arch="vit_base_p16_fz.v2.t2",
    pool_type="attn.d2.nh8.glusw",
    resolution=(160, 256),
    image_feature_dim=512,
    mlp_adapter_spec="v0-2.t0",
    hidden_dim=512
)

# 归一化参数（Minecraft特定）
MC_IMAGE_MEAN = [0.3331, 0.3245, 0.3051]
MC_IMAGE_STD = [0.2439, 0.2493, 0.2873]

# 编码流程
encode_video() = forward_image_features() + forward_video_features()
```

---

## 📝 **相关文档**

| 文档 | 说明 |
|------|------|
| `docs/MINECLIP_OPTIMIZATION_STRATEGY.md` | 详细的优化策略和测试计划 |
| `docs/MINEDOJO_ACTION_MAPPING.md` | MineDojo动作映射（已验证） |
| `docs/RECORDING_CONTROLS.md` | 手动录制控制说明 |
| `tools/README.md` | 工具使用说明 |

---

## 🚀 **快速命令**

```bash
# 1. 运行完整Prompt优化测试（15-20分钟）
cd /Users/nanzhang/aimc
conda run -n minedojo-x86 python tools/quick_optimize_mineclip.py

# 2. 查看结果
open logs/mineclip_optimization/prompt_optimization.png
cat logs/mineclip_optimization/prompt_results.txt

# 3. 如果找到好的prompt，更新训练脚本
# 编辑 src/training/train_get_wood.py
# 修改 task_prompt = "最佳prompt"

# 4. 开始训练
./scripts/train_get_wood.sh
```

---

**下一步由您决定**：
1. 运行完整优化测试？
2. 直接采用混合策略？
3. 还是先用极端场景验证MineCLIP？

