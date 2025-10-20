# 常见问题解答（FAQ）

## 关于加速训练

### Q1: MineRL和MineDojo的数据集在哪里获取？

**A**: 

**MineRL数据集**（推荐用于离线RL）：
```bash
# 安装MineRL
pip install minerl

# 数据会在首次使用时自动下载
python -c "import minerl; data = minerl.data.make('MineRLTreechop-v0')"

# 探索数据集
python scripts/explore_minerl_dataset.py --dataset MineRLTreechop-v0
```

**MineDojo数据集**：
- YouTube视频（73万个）：**不直接提供**，已用于训练MineCLIP模型
- Wiki知识库（6,735页）：可从 https://zenodo.org/records/6693745 下载

**详细文档**：`docs/guides/MINERL_DATASET_GUIDE.md`

---

### Q2: MineRL数据集的结构是什么样的？

**A**: MineRL数据是**轨迹数据**（trajectories），包含：

```python
{
    'state': {
        'pov': np.ndarray,        # 第一人称视角图像 (64, 64, 3)
        'inventory': dict,        # 物品栏状态
        'equipped_items': dict,   # 手持物品
    },
    'action': {
        'camera': [pitch, yaw],   # 摄像机移动
        'forward': 0/1,           # 前进
        'attack': 0/1,            # 攻击
        'jump': 0/1,              # 跳跃
        # ... 其他动作
    },
    'reward': float,              # 奖励值
    'done': bool,                 # episode是否结束
}
```

**关键点**：
- ✅ 自动标注的人类游戏轨迹
- ✅ 包含60M+状态-动作对
- ❌ 不是监督学习的"标记数据"（没有"最优动作"标签）
- ✅ 适合离线RL（CQL、IQL）和行为克隆（BC）

**可用数据集**：
- `MineRLTreechop-v0` - 砍树（15GB，简单）
- `MineRLNavigate-v0` - 导航（20GB，简单）
- `MineRLObtainDiamond-v0` - 获取钻石（45GB，困难）

---

### Q3: MineCLIP是什么？提供什么能力？

**A**: MineCLIP是一个**视觉-语言多模态模型**，在73万YouTube Minecraft视频上训练。

**核心能力**：

1. **视觉-文本匹配**
```python
similarity = mineclip.compute_similarity(
    image,  # 游戏截图
    "chop down a tree"  # 任务描述
)
# 输出: 0.85 (0到1之间，越高表示越匹配)
```

2. **视觉编码**
```python
features = mineclip.encode_image(image)
# 提取语义特征向量（512维或1024维）
```

3. **文本编码**
```python
features = mineclip.encode_text("chop down trees")
# 将任务描述转为特征向量
```

**详细文档**：`docs/guides/MINECLIP_EXPLAINED.md`

---

### Q4: MineCLIP如何参与训练过程？

**A**: MineCLIP作为**密集奖励函数**，将稀疏奖励转换为密集奖励。

**传统RL的问题**（稀疏奖励）：
```python
步骤1-500: 所有奖励都是0  ❌
步骤501: 获得木头，奖励=1  ✅
# 智能体不知道前500步哪些行为有用
```

**MineCLIP解决方案**（密集奖励）：
```python
步骤1: 看到树 → 奖励=0.05   ✅ 鼓励寻找树
步骤2: 靠近树 → 奖励=0.15   ✅ 鼓励接近
步骤3: 面向树 → 奖励=0.30   ✅ 鼓励对准
步骤4: 攻击树 → 奖励=0.50   ✅ 鼓励攻击
步骤5: 获得木头 → 奖励=1.00 ✅ 完成任务
# 每一步都有反馈！
```

**使用方式**：
```bash
# 一行命令启动MineCLIP训练
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000
```

**效果**：
- ⚡ 训练速度提升 **3-5倍**
- 🎯 首次成功快 **5倍**
- 📈 最终性能提升 **15-20%**

---

### Q5: 数据集是"标记数据"吗？

**A**: **不完全是**。

MineRL数据是：
- ✅ **自动标注的轨迹**：记录了状态、动作、奖励
- ❌ **不是监督学习标记**：没有"这个状态下最优动作是什么"的标签
- ⚠️ **人类水平**：数据来自人类玩家，可能包含错误

**可以用于**：
1. **行为克隆（BC）**：直接模仿人类动作
   ```python
   # 监督学习：输入状态，输出动作
   model.fit(states, actions)
   ```

2. **离线强化学习（Offline RL）**：从数据中学习更好的策略
   ```python
   # 使用CQL、IQL等算法
   # 可以超越数据集中的人类表现
   ```

3. **预训练**：先用数据预训练，再用在线RL微调
   ```python
   # 阶段1: 行为克隆
   model.pretrain_from_demos(minerl_data)
   
   # 阶段2: 强化学习微调
   model.finetune_with_rl(minedojo_env)
   ```

---

### Q6: 如何选择加速训练方法？

**A**: 根据你的情况选择：

**快速原型（1-2周）**：
```bash
# MineCLIP最简单最快
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000
```
- ✅ 一行命令
- ✅ 3-5倍加速
- ✅ 适合所有任务

**追求质量（2-4周）**：
```bash
# 课程学习 - 更稳定、性能更好
./scripts/train_curriculum.sh --skill chop_tree
```
- ✅ 渐进式训练
- ✅ 最终性能更高
- ⚠️ 需要设计课程

**有人类演示数据**：
```bash
# 行为克隆 + RL微调
python scripts/train_behavior_cloning.py --dataset MineRLTreechop-v0
```
- ✅ 5-10倍加速
- ✅ 利用人类知识
- ⚠️ 需要下载数据

**详细对比**：`docs/guides/TRAINING_METHODS_COMPARISON.md`

---

### Q7: MineCLIP和MineRL可以一起用吗？

**A**: **可以！而且效果更好！**

推荐组合策略：

```python
# 阶段1: 用MineRL数据预训练（行为克隆）
model = train_behavior_cloning(minerl_dataset)
# 学到基本技能

# 阶段2: 在MineDojo环境中用MineCLIP微调
env = MineCLIPRewardWrapper(minedojo_env, "chop down trees")
model.finetune(env)
# 超越人类水平

# 阶段3: 用稀疏奖励精调
model.finetune(minedojo_env_sparse)
# 确保任务完成
```

**效果**：
- 结合了人类知识（MineRL）
- 密集奖励引导（MineCLIP）
- 任务导向优化（稀疏奖励）
- 预期 **10-20倍** 加速

---

### Q8: 训练太慢怎么办？

**A**: 多种优化方法：

**1. 使用MineCLIP**（首选）
```bash
./scripts/train_with_mineclip.sh --task harvest_log
```
- 3-5倍加速

**2. 启用无头模式**
```bash
export JAVA_OPTS="-Djava.awt.headless=true"
```
- 20-40%速度提升

**3. 使用GPU**
```bash
# 安装CUDA版PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 训练时指定GPU
./scripts/train_with_mineclip.sh --task harvest_log --device cuda
```
- 2-3倍加速

**4. 并行环境**
```bash
./scripts/train_with_mineclip.sh --task harvest_log --n-envs 4
```
- 接近线性加速（需要更多内存）

**5. 减少图像尺寸**
```bash
# 修改训练脚本，添加：
--image-size 120 160  # 默认是160 256
```
- 30-50%速度提升（可能影响性能）

---

### Q9: 如何验证MineCLIP是否工作？

**A**: 查看训练日志和TensorBoard：

```bash
# 启动训练
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000

# 另一个终端启动TensorBoard
tensorboard --logdir logs/tensorboard
```

**检查指标**：

1. **Episode奖励**：`rollout/ep_rew_mean`
   - MineCLIP: 应该快速上升（几千步就有进步）
   - 纯RL: 长时间保持0或负值

2. **训练日志**：
```
[100步] ep_rew_mean: 0.05  ← MineCLIP开始工作
[1000步] ep_rew_mean: 0.15 ← 持续进步
[10000步] ep_rew_mean: 0.45 ← 接近成功
```

3. **Info字段**（如果实现了详细记录）：
```python
info = {
    'sparse_reward': 0.0,       # 原始奖励（还未成功）
    'mineclip_reward': 0.25,    # MineCLIP奖励（在进步）
    'mineclip_similarity': 0.65 # 与任务的相似度
}
```

---

### Q10: 我应该从哪里开始？

**A**: 推荐路线（2-3周）：

**第1周：快速验证**
```bash
# Day 1: 安装和测试
pip install minedojo stable-baselines3
python scripts/validate_install.py

# Day 2-3: 第一个MineCLIP训练
./scripts/train_with_mineclip.sh --task harvest_log --timesteps 200000

# Day 4-5: 训练2-3个简单技能
./scripts/train_with_mineclip.sh --task harvest_wool --timesteps 150000
./scripts/train_with_mineclip.sh --task harvest_milk --timesteps 150000

# Day 6-7: 评估和调试
python scripts/evaluate_skills.py
```

**第2周：深入优化**
```bash
# 使用课程学习训练核心技能
./scripts/train_curriculum.sh --skill chop_tree
./scripts/train_curriculum.sh --skill mine_stone
```

**第3周：组合应用**
```bash
# 构建技能库
./scripts/manage_skill_library.sh add chop_tree checkpoints/...

# 测试技能组合
python scripts/test_skill_combination.py
```

**关键文档**：
1. 先读：`docs/guides/QUICK_START_ACCELERATED_TRAINING.md`
2. 深入：`docs/guides/TRAINING_ACCELERATION_GUIDE.md`
3. 参考：`docs/guides/MINECLIP_EXPLAINED.md`
4. 参考：`docs/guides/MINERL_DATASET_GUIDE.md`

---

### Q11: MineRL数据集只有几个场景，我要训练的技能（如获得煤块）不在其中怎么办？

**A**: MineRL确实只有8个预定义任务，对于不在其中的技能，有5种解决方案：

**方案1：使用MineCLIP（推荐）⭐**
```bash
# 不需要任何数据！直接用MineCLIP训练
./scripts/train_with_mineclip.sh \
    --task open-ended \
    --task-description "mine coal ore and collect coal" \
    --timesteps 200000
```
- ✅ 最简单最快
- ✅ 不需要数据
- ✅ 支持任意任务
- ✅ 3-5倍加速

**方案2：迁移学习**
```python
# 使用相似任务的数据（如ObtainDiamond包含挖矿）
model = train_bc(minerl_data="MineRLObtainDiamond-v0")
model.finetune(coal_mining_env_with_mineclip)
```

**方案3：自己收集演示**
- 自己玩游戏录制15-30个演示
- 1-3小时即可完成
- 质量最高

**方案4：课程学习**
- 分解复杂技能为简单子技能
- 渐进式训练

**方案5：组合已有技能**
- 使用技能库组合基础技能
- 模块化、可复用

**推荐策略**：
- 90%情况用**MineCLIP**（不需要数据）
- 追求质量用**课程学习**
- 有时间可**收集少量演示**

**详细文档**：`docs/guides/ADVANCED_TRAINING_SOLUTIONS.md`

---

### Q12: MineCLIP是在线模型吗？本地训练时会在线请求吗？

**A**: **不是！MineCLIP是本地离线模型！**

**关键事实**：
- ✅ MineCLIP模型权重在**本地**（~250-350MB）
- ✅ 首次使用时会**自动下载**到 `~/.minedojo/models/`
- ✅ 之后**完全离线运行**，不需要网络
- ✅ 推理在你的**本地GPU/CPU**上执行
- ✅ 完全**免费开源**，无使用限制

**网络需求**：
```bash
# 只在首次使用时需要网络
pip install minedojo  # ← 需要网络
python -c "import minedojo; minedojo.make('harvest_log')"  # ← 首次下载模型

# 之后所有训练都是离线的
./scripts/train_with_mineclip.sh --task harvest_log  # ← 完全离线！
```

**验证方法**：
```bash
# 检查模型文件
ls -lh ~/.minedojo/models/
# 应该看到：
#   mineclip_attn.pth (50MB)
#   mineclip_vision.pth (150MB)
#   mineclip_text.pth (50MB)

# 断网测试：
# 1. 断开网络
# 2. 运行训练脚本
# 3. 如果能正常运行，说明是离线的！
```

**性能影响**：
- MineCLIP本地推理：15-30ms/次（CPU）
- 对训练速度影响：约10-20%
- 但收敛快3-5倍，总时间大幅缩短

**离线工作流**：
```bash
# 在有网络的机器上打包
tar -czf minedojo_models.tar.gz ~/.minedojo/models/

# 在离线机器上使用
tar -xzf minedojo_models.tar.gz -C ~/
./scripts/train_with_mineclip.sh --task harvest_log  # 完全离线训练
```

**详细文档**：`docs/guides/ADVANCED_TRAINING_SOLUTIONS.md`

---

## 其他常见问题

### Q13: MineDojo环境创建失败？

**A**: 
```bash
# 1. 检查Java版本
java -version  # 需要Java 8+

# 2. 设置无头模式
export JAVA_OPTS="-Djava.awt.headless=true"

# 3. 重新编译Minecraft
cd /path/to/minedojo/sim/Malmo/Minecraft
./gradlew shadowJar
```

### Q14: 内存不足？

**A**:
```bash
# 1. 减少并行环境
--n-envs 1

# 2. 减少批次大小
--batch-size 32

# 3. 使用技能库的延迟加载
# 不同时加载所有技能
```

### Q15: GPU不被识别？

**A**:
```bash
# 检查PyTorch GPU支持
python -c "import torch; print(torch.cuda.is_available())"

# 如果False，重新安装CUDA版本
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 获取帮助

- 📖 **完整文档**：`docs/guides/`
- 💡 **示例代码**：`src/training/`
- 🔧 **诊断工具**：`python scripts/diagnose_minedojo.py`
- 📊 **数据探索**：`python scripts/explore_minerl_dataset.py --list`

---

**有其他问题？** 查看文档或运行诊断工具！

