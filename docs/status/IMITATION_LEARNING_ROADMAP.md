# 模仿学习实施路线图

> **目标**: 通过人工录制专家演示，加速训练并提升最终性能

---

## 📊 **当前状态**

### ✅ 已具备条件
- ✅ 录制工具已完成（`tools/record_manual_chopping.py`）
- ✅ 动作映射已验证（`functional=3` = 攻击）
- ✅ 成功录制272帧砍树序列
- ✅ 数据格式与MineDojo兼容

### ⚠️ 需要补充
- ❌ 录制工具未保存动作序列
- ❌ 缺少数据预处理脚本
- ❌ 缺少BC训练脚本
- ❌ 缺少BC+PPO混合训练

---

## 🗺️ **实施路线**

### **Phase 1: 录制工具增强** ⏱️ 1-2小时

**目标**: 让录制工具保存完整的(observation, action)对

#### 任务清单
- [ ] 修改`tools/record_manual_chopping.py`
  - [ ] 保存每一帧的action到`actions.npy`
  - [ ] 保存episode metadata（总帧数、是否成功等）
  - [ ] 支持批量录制模式（自动编号episode）
  
- [ ] 创建验证脚本`tools/verify_expert_data.py`
  - [ ] 回放录制的序列
  - [ ] 验证action与observation对应
  - [ ] 统计数据质量

**验收标准**:
```
data/expert_demos/chop_wood/
├── episode_001/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── ...
│   ├── actions.npy          # 新增
│   └── metadata.json        # 新增
├── episode_002/
│   └── ...
```

---

### **Phase 2: 数据收集** ⏱️ 2-3小时

**目标**: 录制高质量的专家演示数据

#### 数据收集计划

| 任务 | 演示次数 | 预计时间 | 场景要求 |
|------|---------|---------|---------|
| 🪵 砍树 | 20次 | 1小时 | 森林、平原各10次 |
| 🏗️ 建造 | 10次 | 1小时 | 不同地形 |
| ⛏️ 挖矿 | 15次 | 1.5小时 | 洞穴、地表 |

#### 录制规范
1. **质量要求**:
   - ✅ 必须成功完成任务
   - ✅ 动作连贯自然（像人类玩家）
   - ✅ 避免长时间卡顿或失误
   - ✅ 完整流程（从开始到获得物品）

2. **多样性要求**:
   - 不同起始位置
   - 不同地形/生物群系
   - 不同时间（白天/黄昏/夜晚）
   - 不同树木类型（橡树/白桦/云杉）

**验收标准**:
- 至少20次成功的砍树演示
- 平均每次演示200-400帧
- 总数据量: 5K-10K frames

---

### **Phase 3: 数据预处理** ⏱️ 2-3小时

**目标**: 将原始录制数据转换为训练格式

#### 任务清单
- [ ] 创建`tools/prepare_expert_data.py`
  - [ ] 加载所有episode的observations和actions
  - [ ] 数据清洗（移除失败episode、异常帧）
  - [ ] 数据增强（可选：颜色抖动、随机裁剪）
  - [ ] 保存为PyTorch Dataset格式
  
- [ ] 创建`tools/analyze_expert_data.py`
  - [ ] 统计动作分布
  - [ ] 可视化轨迹
  - [ ] 检测数据偏差

**输出格式**:
```python
{
    'episodes': [
        {
            'observations': np.array([frames...]),  # (T, H, W, 3)
            'actions': np.array([actions...]),       # (T, 8)
            'success': True,
            'task': 'chop_wood'
        },
        ...
    ],
    'stats': {
        'total_episodes': 20,
        'total_frames': 8453,
        'avg_episode_length': 422.6
    }
}
```

**验收标准**:
- 生成`data/processed/chop_wood_expert.pkl`
- 数据统计报告
- 动作分布可视化

---

### **Phase 4: BC训练** ⏱️ 3-4小时

**目标**: 实现基础的行为克隆训练

#### 任务清单
- [ ] 创建`src/training/train_bc.py`
  - [ ] 实现ExpertDataset类
  - [ ] 实现BC训练循环
  - [ ] 支持MultiDiscrete动作空间
  - [ ] 添加训练监控（TensorBoard）
  
- [ ] 创建`src/training/eval_bc.py`
  - [ ] 在MineDojo环境中评估BC策略
  - [ ] 记录成功率、平均步数
  - [ ] 保存评估视频

**训练配置**:
```yaml
# config/bc_config.yaml
data:
  expert_file: data/processed/chop_wood_expert.pkl
  train_split: 0.9
  val_split: 0.1

model:
  backbone: CNN
  hidden_dim: 512
  
training:
  epochs: 50
  batch_size: 32
  learning_rate: 3e-4
  optimizer: Adam
  
evaluation:
  eval_episodes: 10
  eval_frequency: 5  # epochs
```

**验收标准**:
- BC模型训练完成
- 验证集准确率 > 80%
- MineDojo评估成功率 > 50%

---

### **Phase 5: BC+PPO混合训练** ⏱️ 4-5小时

**目标**: 用PPO微调BC策略，达到最佳性能

#### 任务清单
- [ ] 修改`src/training/train_get_wood.py`
  - [ ] 支持加载BC预训练模型
  - [ ] 实现warmstart机制
  - [ ] 调整PPO超参数（降低learning_rate）
  
- [ ] 实现对比实验
  - [ ] 纯RL baseline
  - [ ] BC-only
  - [ ] BC+PPO
  
- [ ] 创建训练报告
  - [ ] 对比学习曲线
  - [ ] 统计最终性能
  - [ ] 分析优劣势

**实验设置**:
```python
# 实验1: 纯RL（baseline）
python src/training/train_get_wood.py \
  --total-steps 100000 \
  --learning-rate 3e-4

# 实验2: BC预训练 + PPO微调
python src/training/train_bc.py --epochs 50
python src/training/train_get_wood.py \
  --pretrain-model checkpoints/bc_chop_wood.zip \
  --total-steps 50000 \
  --learning-rate 1e-4

# 实验3: BC-only（对照）
python src/training/eval_bc.py \
  --model checkpoints/bc_chop_wood.zip \
  --episodes 100
```

**验收标准**:
- BC+PPO在20K steps内达到>80%成功率
- 纯RL需要50K+ steps才能达到相同水平
- 训练时间减少2-3倍

---

## 📊 **预期成果**

### **定量指标**

| 指标 | 纯RL | BC+PPO | 提升 |
|------|------|--------|------|
| 首次成功step | ~50K | **~5K** | **10x** ⚡ |
| 达到80%成功率 | 100K | **20K** | **5x** ⚡ |
| 最终成功率 | 85% | **95%** | +10% |
| 总训练时间 | 4小时 | **1.5小时** | **2.7x** ⚡ |

### **定性收益**

1. ✅ **更快的探索**: 从好的策略开始，避免随机探索浪费
2. ✅ **更稳定的训练**: 减少训练方差，结果更可复现
3. ✅ **更好的最终性能**: 结合人类先验和RL优化
4. ✅ **可扩展到复杂任务**: 长序列任务（建造、探险）更容易学习

---

## 🚀 **快速开始（最小可行方案）**

如果只想快速验证效果，可以简化流程：

### **简化版路线图** ⏱️ 总计3-4小时

1. **录制5次高质量演示** (30分钟)
   ```bash
   for i in {1..5}; do
       python tools/record_manual_chopping.py \
         --output-dir data/expert_demos/episode_$i
   done
   ```

2. **手动整理数据** (30分钟)
   - 手动挑选最好的3次演示
   - 简单的numpy数组合并

3. **简单BC实现** (1.5小时)
   - 使用PyTorch实现最简单的监督学习
   - 只训练policy head（不训练value head）

4. **快速测试** (30分钟)
   - 在MineDojo中运行10次评估
   - 观察是否比随机策略好

5. **对比实验** (1小时)
   - 运行纯RL训练10K steps
   - 对比首次成功时间

**如果简化版有效** → 投入资源做完整版
**如果简化版无效** → 重新评估模仿学习可行性

---

## ⚠️ **风险和缓解**

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| 数据量不够 | 中 | 高 | 逐步增加演示，观察边际收益 |
| 过拟合 | 高 | 中 | 数据增强、early stopping |
| BC策略脆弱 | 中 | 中 | 用PPO微调纠正 |
| 实现困难 | 低 | 高 | 使用现有库（imitation） |

---

## 📚 **参考实现**

### **可以直接使用的库**:

1. **imitation** (推荐)
   ```bash
   pip install imitation
   ```
   - 提供BC、DAgger等现成实现
   - 与Stable-Baselines3兼容
   - 文档完善

2. **手动实现**
   - 更灵活
   - 更好理解原理
   - 适合定制化需求

---

## 🎯 **决策点**

### **现在需要决定：**

1. **是否投入资源做模仿学习？**
   - ✅ 是 → 开始Phase 1
   - ❌ 否 → 继续优化MineCLIP或混合奖励

2. **采用哪种实施方案？**
   - ⚡ 简化版（3-4小时）→ 快速验证
   - 🏗️ 完整版（15-20小时）→ 系统实施

3. **优先级排序**
   - 🥇 砍树任务（最简单，验证概念）
   - 🥈 建造任务（展示复杂任务优势）
   - 🥉 其他任务（扩展应用）

---

**下一步建议**: 

如果对模仿学习感兴趣，建议先做**简化版快速验证**（3-4小时），证明有效后再投入完整实施。

如果时间紧张，可以**先继续MineCLIP优化**（运行prompt测试），模仿学习作为Plan B。

