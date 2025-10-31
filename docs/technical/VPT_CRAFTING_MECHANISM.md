# VPT合成机制：MineRL vs MineDojo

**日期**: 2025-10-29  
**问题**: VPT在MineRL中如何制作物品（如木斧）？

---

## 🔍 观察到的现象

在官方VPT演示中（MineRL环境），agent砍树后：
1. ✅ 打开了合成窗口
2. ✅ 制作了木斧
3. ✅ 继续游戏

**问题**: 这个动作序列如何实现？

---

## 📚 MineRL的合成机制

### 动作空间中的相关动作

根据[MineRL文档](https://minerl.readthedocs.io/en/v1.0.0/environments/index.html#action-space)：

```python
{
    "inventory": Discrete(2),  # 打开/关闭物品栏GUI
    "camera": Box(low=-180.0, high=180.0, shape=(2,)),  # 移动准星
    "use": Discrete(2),        # 使用/点击
    "attack": Discrete(2),     # 攻击/点击
    ...
}
```

**关键**: MineRL **没有直接的"craft"动作**！

### VPT的合成流程（GUI操作）

```python
# 步骤1: 打开物品栏/合成界面
action = {"inventory": 1, ...}
# → 观察变成GUI界面（显示物品栏、合成槽等）

# 步骤2: 移动准星到合成结果位置
action = {"camera": [pitch, yaw], ...}
# → 准星移动到木斧的位置

# 步骤3: 点击获取木斧
action = {"use": 1, ...}  # 或 "attack": 1
# → 从合成槽中取出木斧

# 步骤4: 关闭物品栏
action = {"inventory": 1, ...}
# → 返回正常游戏视角
```

**本质**: VPT通过**视觉引导的GUI操作**来合成物品，类似人类玩家！

### 观察空间的变化

```python
# 正常游戏时
obs["pov"] = [游戏世界的第一人称视角]

# 打开inventory后  
obs["pov"] = [物品栏GUI界面]
            # 可以看到：
            # - 物品栏格子
            # - 合成槽（2x2或3x3）
            # - 合成结果位置
```

VPT的视觉网络需要：
1. 识别当前是否在GUI界面
2. 定位物品和合成槽位置
3. 规划鼠标移动和点击序列

---

## 🎮 MineDojo的合成机制

### 动作空间中的craft

根据[MineDojo文档](https://docs.minedojo.org/sections/core_api/action_space.html)：

```python
MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
                                  ↑    ↑
                           index 5    index 6
```

| Index | 描述 | 值 |
|-------|------|---|
| 5 | Functional actions | 4 = craft |
| 6 | Craft argument | 配方ID (0-243) |

### MineDojo的合成流程（直接指令）

```python
# 一步完成！
action = [0, 0, 0, 12, 12, 4, wood_axe_id, 0]
         # index 5 = 4 (craft)
         # index 6 = wood_axe_id (假设是某个ID)

# → 直接获得木斧（前提：物品栏有足够材料）
```

**本质**: MineDojo提供**高级抽象**，一个动作完成合成！

---

## ⚖️ 两种机制对比

| 方面 | MineRL/VPT | MineDojo |
|------|-----------|----------|
| **合成方式** | GUI操作（打开→定位→点击） | 直接指令（一步完成） |
| **动作数量** | 多步（4-10步） | 单步 |
| **需要技能** | 视觉理解 + 鼠标控制 | 知道配方ID |
| **难度** | 高（接近人类玩家） | 低（程序化） |
| **`inventory`动作** | 打开/关闭GUI | 无此动作 |
| **`craft`动作** | 无 | 有（index 5=4） |
| **配方选择** | 视觉引导 | 参数指定（index 6） |
| **真实性** | ⭐⭐⭐⭐⭐（完全模拟人类） | ⭐⭐⭐（简化抽象） |

---

## 🔄 映射问题

### 当前映射的局限性

```python
# src/models/vpt/minedojo_agent.py
def convert(self, minerl_action: dict) -> np.ndarray:
    # ...
    if minerl_action.get('attack', 0):
        minedojo_action[5] = 3
    elif minerl_action.get('use', 0):
        minedojo_action[5] = 1
    # ...
    # ❌ inventory动作被忽略！
```

**问题**：
1. VPT的 `inventory=1` 被忽略
2. VPT在GUI中的 `use/attack` 点击被转换为MineDojo的use/attack
3. **无法正确映射合成操作**！

### 为什么这是可接受的？

对于**harvest_1_log**等简单任务：
- ✅ **不需要合成**物品
- ✅ VPT只需要移动、视角、攻击
- ✅ 即使VPT尝试打开inventory，MineDojo会忽略，不影响任务完成

对于**需要合成**的复杂任务：
- ❌ 直接运行VPT会**失败**
- ❌ VPT的GUI操作在MineDojo中无法执行
- 🔧 需要**任务特定的适配**或**fine-tune**

---

## 💡 解决方案

### 方案1: 任务限制（当前方案）

**适用场景**: harvest_1_log, 探索任务等不需要合成的任务

```python
# 直接使用VPT zero-shot
agent = VPTAgent(weights_path='...', cam_interval=0.01)
# 在harvest_1_log任务中正常工作
```

**优点**:
- ✅ 无需修改
- ✅ 简单任务表现良好

**缺点**:
- ❌ 无法处理需要合成的任务

---

### 方案2: Fine-tune适配MineDojo

**适用场景**: 需要合成的复杂任务

```python
# 1. 使用MineDojo环境fine-tune VPT
#    - MineDojo的craft动作更简单
#    - Agent学会使用craft而不是GUI操作

# 2. 或者收集MineDojo专家数据训练
#    - 使用MineDojo的craft机制
```

**优点**:
- ✅ 充分利用MineDojo的简化机制
- ✅ 更高效的合成

**缺点**:
- ❌ 需要重新训练
- ❌ 失去VPT的zero-shot能力

---

### 方案3: Hybrid Wrapper（高级）

```python
class VPTMineDojoCraftWrapper:
    """
    检测VPT的GUI操作序列，转换为MineDojo的craft指令
    """
    def __init__(self, agent):
        self.agent = agent
        self.in_inventory = False
        self.craft_buffer = []
    
    def predict(self, obs):
        action = self.agent.predict(obs)
        
        # 检测打开inventory
        if minerl_action.get('inventory', 0):
            self.in_inventory = not self.in_inventory
            if self.in_inventory:
                # 开始记录GUI操作
                self.craft_buffer = []
            else:
                # 关闭inventory，分析craft意图
                craft_id = self.infer_craft_intent(self.craft_buffer)
                if craft_id:
                    return self.create_craft_action(craft_id)
        
        # 在GUI中记录操作
        if self.in_inventory:
            self.craft_buffer.append(minerl_action)
            return noop_action  # 暂时不执行
        
        return action
    
    def infer_craft_intent(self, buffer):
        """从GUI操作序列推断合成意图"""
        # 分析camera移动和click位置
        # 推断要合成的物品
        ...
```

**优点**:
- ✅ 保留VPT的原始能力
- ✅ 适配MineDojo的craft机制

**缺点**:
- ❌ 实现复杂
- ❌ 需要GUI位置映射
- ❌ 可能不够准确

---

## 📊 实际影响评估

### 当前VPT在MineDojo中的表现

| 任务类型 | VPT表现 | 原因 |
|---------|--------|------|
| harvest_1_log | ✅ 良好 | 无需合成 |
| 探索任务 | ✅ 良好 | 无需合成 |
| combat任务 | ✅ 较好 | 基本动作为主 |
| 建造任务（需要工具） | ❌ 差 | 无法合成工具 |
| 复杂任务（需要多步合成） | ❌ 失败 | GUI操作无效 |

### VPT尝试打开inventory的频率

根据测试（100步）：
- `inventory=1` 出现次数: **需要实际测试**
- 预计：**较低**（VPT在early-game主要是探索和收集）

---

## ✅ 当前建议

### 短期（Zero-shot评估）

```python
# 使用VPT进行zero-shot评估
# ✅ 适用任务: harvest_1_log, explore等
# ❌ 不适用: 需要合成的任务

agent = VPTAgent(weights_path='...', cam_interval=0.01)
# inventory动作会被忽略，不影响简单任务
```

**动作映射保持现状**：
- ✅ 忽略 `inventory`（不会破坏简单任务）
- ✅ 正确映射其他动作

### 长期（Fine-tune训练）

```python
# 在MineDojo环境中fine-tune
# 学习使用MineDojo的craft机制
# 而不是MineRL的GUI操作

# 可以考虑：
# 1. BC训练，使用MineDojo专家数据
# 2. RL fine-tune，利用MineDojo的craft奖励
```

---

## 🎯 结论

1. **MineRL的`inventory`** = 打开GUI进行**视觉引导操作**
2. **MineDojo的`craft`** = 直接指令**程序化合成**
3. **两者无法直接映射**，语义完全不同
4. **当前方案**适用于不需要合成的简单任务
5. **复杂任务**需要fine-tune或wrapper来适配

**对于harvest_1_log任务**: 当前映射完全足够！✅

---

## 📚 参考

- [MineRL Action Space](https://minerl.readthedocs.io/en/v1.0.0/environments/index.html#action-space)
- [MineDojo Action Space](https://docs.minedojo.org/sections/core_api/action_space.html)
- VPT Paper: "Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos"

---

**最后更新**: 2025-10-29  
**状态**: ✅ 问题已分析，当前映射策略合理

