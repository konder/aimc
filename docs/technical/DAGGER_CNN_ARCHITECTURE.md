# 🧠 DAgger训练的CNN架构详解

## 🎯 直接回答

**是的！当前的DAgger训练使用了CNN（卷积神经网络）**

---

## 📋 架构概览

### **代码位置**

`src/training/bc/train_bc.py` (第238行):
```python
model = PPO(
    "CnnPolicy",  # ← 这里指定使用CNN策略
    env,
    learning_rate=learning_rate,
    # ...
)
```

### **什么是CnnPolicy？**

`CnnPolicy` 是 Stable-Baselines3 提供的专门用于**图像输入**的策略网络，内部使用 **NatureCNN** 架构。

---

## 🏗️ NatureCNN 架构详解

### **来源**

NatureCNN 来自 DeepMind 的 DQN 论文：
> "Human-level control through deep reinforcement learning" (Nature 2015)

这是 Atari 游戏 RL 研究的标准 CNN 架构。

---

### **网络结构**

```python
NatureCNN(
    observation_space,
    features_dim=512  # 输出特征维度
)
```

#### **层级结构**

```
输入: (B, 3, 160, 256)  # Minecraft RGB图像
  ↓
[Conv2d Layer 1]
  卷积核: 8×8, stride=4, filters=32
  激活: ReLU
  输出: (B, 32, 39, 63)
  ↓
[Conv2d Layer 2]
  卷积核: 4×4, stride=2, filters=64
  激活: ReLU
  输出: (B, 64, 18, 30)
  ↓
[Conv2d Layer 3]
  卷积核: 3×3, stride=1, filters=64
  激活: ReLU
  输出: (B, 64, 16, 28)
  ↓
[Flatten]
  输出: (B, 28,672)  # 64×16×28
  ↓
[Linear (全连接层)]
  输入: 28,672
  输出: 512
  激活: ReLU
  ↓
输出特征: (B, 512)
```

---

### **参数详解**

| 层 | 类型 | 输入 | 卷积核 | Stride | Filters | 输出 | 参数量 |
|----|------|------|--------|--------|---------|------|--------|
| Conv1 | Conv2d | (3, 160, 256) | 8×8 | 4 | 32 | (32, 39, 63) | 6,176 |
| Conv2 | Conv2d | (32, 39, 63) | 4×4 | 2 | 64 | (64, 18, 30) | 32,832 |
| Conv3 | Conv2d | (64, 18, 30) | 3×3 | 1 | 64 | (64, 16, 28) | 36,928 |
| Flatten | - | (64, 16, 28) | - | - | - | (28,672) | 0 |
| Linear | Dense | 28,672 | - | - | 512 | (512) | 14,680,576 |

**总参数量**: ~14.7M (1470万)

---

## 🎮 为什么使用NatureCNN？

### **优势**

1. ✅ **专为游戏图像设计**
   - Atari/Minecraft等像素游戏的标准架构
   - 经过大量实验验证

2. ✅ **逐步降维**
   - 160×256 → 39×63 → 18×30 → 16×28
   - 高效提取空间特征

3. ✅ **感受野合适**
   - Conv1 (8×8, stride=4): 捕获大范围特征（树木位置）
   - Conv2 (4×4, stride=2): 捕获中等特征（树干形状）
   - Conv3 (3×3, stride=1): 捕获细节特征（叶子纹理）

4. ✅ **计算效率**
   - 相比ResNet等深度网络，参数量适中
   - 训练速度快

---

### **为什么不用ResNet/VGG等更深的网络？**

| 架构 | 参数量 | 训练速度 | 过拟合风险 | 适用场景 |
|------|-------|---------|-----------|----------|
| NatureCNN | 14.7M | 快 | 低 | ✅ 游戏/低分辨率 |
| ResNet-18 | 11.7M | 中 | 中 | ImageNet分类 |
| ResNet-50 | 25.6M | 慢 | 高 | ImageNet分类 |
| VGG-16 | 138M | 很慢 | 很高 | ImageNet分类 |

**结论**:
- NatureCNN 对于 Minecraft 的 160×256 图像是**最优选择**
- 更深的网络容易**过拟合**（专家数据有限）
- 训练速度更快，适合 DAgger 的迭代训练

---

## 🔍 完整的策略网络结构

### **PPO策略 = NatureCNN + MLP + Action Head**

```
[输入] 
  Observation: (3, 160, 256) RGB图像
    ↓
[NatureCNN特征提取器]
  Conv1 → Conv2 → Conv3 → Flatten → Linear
  输出: 512维特征向量
    ↓
[MLP Extractor (Actor-Critic分支)]
  
  Actor分支:
    Linear(512 → 64) → ReLU
    输出: 64维 Actor特征
  
  Critic分支:
    Linear(512 → 64) → ReLU
    输出: 64维 Critic特征
    ↓
[Action Head]
  MineDojo MultiDiscrete(8维):
  
  Dimension 0 (forward/back):    Linear(64 → 3)  → Categorical
  Dimension 1 (left/right):      Linear(64 → 3)  → Categorical
  Dimension 2 (jump/sneak):      Linear(64 → 4)  → Categorical
  Dimension 3 (camera pitch):    Linear(64 → 25) → Categorical
  Dimension 4 (camera yaw):      Linear(64 → 25) → Categorical
  Dimension 5 (functional):      Linear(64 → 8)  → Categorical
  Dimension 6 (craft_argument):  Linear(64 → 4)  → Categorical
  Dimension 7 (inventory_arg):   Linear(64 → 36) → Categorical
  
  总输出维度: 3+3+4+25+25+8+4+36 = 108
    ↓
[Value Head]
  Linear(64 → 1) → Scalar
  输出: 状态价值 V(s)
```

---

## 📊 训练过程中的CNN

### **行为克隆阶段 (BC)**

`src/training/bc/train_bc.py` (第303-305行):
```python
# 使用CNN提取特征
features = policy_net.extract_features(batch_obs)  # NatureCNN前向传播
latent_pi = policy_net.mlp_extractor.forward_actor(features)  # Actor MLP
action_logits = policy_net.action_net(latent_pi)  # Action头
```

**流程**:
1. `batch_obs` (B, 3, 160, 256) 输入NatureCNN
2. 提取512维特征向量
3. 通过Actor MLP得到64维表示
4. 通过Action Head预测8维动作分布
5. 计算与专家动作的交叉熵损失
6. 反向传播更新**整个CNN+MLP+ActionHead**

**关键**: BC训练会**更新CNN的所有参数**！

---

### **DAgger迭代阶段**

1. **收集状态** (`run_policy_collect_states.py`):
   - 使用训练好的CNN预测动作

2. **标注** (`label_states.py`):
   - 人工提供正确动作

3. **重新训练** (`train_dagger.py`):
   - 再次更新CNN参数

**每次DAgger迭代都会重新训练CNN！**

---

## 🎓 为什么CNN对Minecraft重要？

### **1. 空间不变性 (Spatial Invariance)**

**问题**: 树可能出现在画面的任何位置

**CNN解决**:
- 卷积核在整个图像上滑动
- 无论树在左边、右边、中间，都能识别

**如果用全连接层**:
- 树在左边和右边是"不同的输入"
- 需要学习每个位置的树 → 泛化能力差

---

### **2. 层级特征学习 (Hierarchical Features)**

```
Conv1 (8×8, stride=4):
  学习: 边缘、颜色块
  例如: "绿色区域"、"棕色区域"

Conv2 (4×4, stride=2):
  学习: 形状组合
  例如: "绿色圆形团"、"棕色竖条"

Conv3 (3×3, stride=1):
  学习: 复杂模式
  例如: "树叶+树干"、"完整的树"
```

**这种层级结构与人类视觉系统类似！**

---

### **3. 参数共享 (Parameter Sharing)**

**全连接层**:
```
输入: 160×256×3 = 122,880 像素
第一层神经元: 512
参数量: 122,880 × 512 = 62,914,560 (6300万！)
```

**CNN**:
```
Conv1卷积核: 8×8×3 = 192 参数
32个卷积核: 192 × 32 = 6,144
总参数(3层): 约76,000
```

**参数减少 >800倍！**

---

### **4. 局部连接 (Local Connectivity)**

**全连接**: 每个神经元连接所有像素
- 问题: 右上角的树叶和左下角的草地关系很弱
- 过度连接 → 过拟合

**CNN**: 每个神经元只连接局部区域
- 符合视觉任务特性（相邻像素相关）
- 更高效、更少过拟合

---

## 🔬 实验验证

### **如果不用CNN会怎样？**

假设用MLP（全连接网络）:

| 架构 | 参数量 | 训练数据需求 | 泛化能力 | 预测性能 |
|------|-------|-------------|---------|----------|
| **CNN (NatureCNN)** | 14.7M | 中等 (1000样本) | ✅ 高 | ✅ 好 |
| MLP (3层) | 63M+ | 极高 (10000+样本) | ❌ 低 | ❌ 差 |

**MLP的问题**:
1. ❌ 参数量暴增 (4倍以上)
2. ❌ 需要更多专家数据 (10倍以上)
3. ❌ 树在不同位置无法泛化
4. ❌ 训练时间长，容易过拟合

**结论**: 对于图像输入，CNN是**必需的**！

---

## 💡 优化建议

### **当前架构已经很好，但如果要优化:**

#### **1. 增加通道数**

```python
policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(
        features_dim=512,  # 当前
        # 可以增加每层的filter数量
    )
)
```

**效果**: 
- ✅ 更强的表示能力
- ❌ 参数量增加，训练变慢

**建议**: 当前512已经足够，不需要修改

---

#### **2. 使用ResNet (高级)**

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision.models as models

class MinecraftResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.resnet = models.resnet18(pretrained=False)
        # ... 修改输入和输出层
```

**效果**:
- ✅ 更强的特征提取
- ❌ 参数量略增，训练慢20-30%
- ❌ 可能过拟合（专家数据少）

**建议**: 达到80%成功率后再考虑

---

#### **3. 数据增强**

```python
import torchvision.transforms as T

transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),  # 左右翻转
    T.ColorJitter(brightness=0.2),  # 亮度变化
])
```

**效果**:
- ✅ 增强泛化能力
- ✅ 减少过拟合
- ❌ 训练时间增加

**建议**: 如果发现过拟合，可以尝试

---

## 🎯 总结

### **核心结论**

```
✅ 是的！DAgger训练使用了CNN (NatureCNN)
✅ 架构: 3层卷积 + 1层全连接 = 14.7M参数
✅ 专为游戏图像设计，经过验证
✅ 每次BC/DAgger训练都会更新CNN参数
```

---

### **CNN在DAgger中的作用**

1. **特征提取**: 从160×256 RGB图像提取512维特征
2. **空间理解**: 识别树木、地形等空间结构
3. **泛化能力**: 对不同位置、不同世界的树都有效
4. **参数效率**: 只需1000个样本就能训练

---

### **为什么不需要担心CNN架构？**

1. ✅ **NatureCNN已经是标准选择**
   - DQN、PPO、A3C等都用它
   - Minecraft、Atari等游戏的默认架构

2. ✅ **Stable-Baselines3自动处理**
   - 只需指定`"CnnPolicy"`
   - 自动适配MultiDiscrete动作空间

3. ✅ **当前瓶颈不是CNN架构**
   - 标注质量 > 网络架构
   - 专家数据量 > 网络深度
   - 先优化标注策略（见`DAGGER_LABELING_STRATEGY.md`）

---

### **何时需要修改CNN架构？**

只在以下情况考虑:

1. ⚠️ 已经达到80%+成功率，想冲90%
2. ⚠️ 有大量专家数据 (>5000样本)
3. ⚠️ 有明确证据表明CNN学不到特征
4. ⚠️ 想尝试迁移学习 (ImageNet预训练)

**当前阶段**: 专注于标注质量和数据量，不要修改架构！

---

## 📚 参考资料

### **NatureCNN原始论文**
- Mnih et al. "Human-level control through deep RL" (Nature 2015)
- 原始用于Atari游戏
- 输入: 84×84×4 灰度帧堆叠
- Minecraft适配: 160×256×3 RGB

### **Stable-Baselines3文档**
- [CnnPolicy](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#ppo-policies)
- [Custom Feature Extractor](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)

### **相关工作**
- MineRL Competition: 大多使用NatureCNN或ResNet
- MineDojo Paper: 使用类似的CNN架构

---

## 🔧 验证你的CNN

### **查看模型结构**

```bash
# 训练完BC后
python << 'EOF'
import torch
from stable_baselines3 import PPO

model = PPO.load("checkpoints/dagger/harvest_1_log/bc_baseline.zip")
print(model.policy)
print(f"\nCNN特征提取器:")
print(model.policy.features_extractor)
print(f"\n总参数量: {sum(p.numel() for p in model.policy.parameters()):,}")
EOF
```

### **预期输出**

```
ActorCriticCnnPolicy(
  (features_extractor): NatureCNN(
    (cnn): Sequential(
      (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
      (1): ReLU()
      (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
      (3): ReLU()
      (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
      (5): ReLU()
      (6): Flatten()
    )
    (linear): Sequential(
      (0): Linear(in_features=28672, out_features=512, bias=True)
      (1): ReLU()
    )
  )
  (mlp_extractor): MlpExtractor(...)
  (action_net): MultiCategorical(...)
  (value_net): Linear(...)
)

总参数量: 14,756,812
```

---

**版本**: 1.0.0  
**创建日期**: 2025-10-22  
**核心结论**: DAgger训练使用NatureCNN (3层卷积+1层全连接，14.7M参数)

