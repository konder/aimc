# FlatWorld GeneratorString 验证指南

**日期**: 2025-11-14  
**目的**: 如何验证 FlatWorldGenerator 的 generatorString 配置是否生效

---

## 🎯 验证方法总览

| 方法 | 难度 | 推荐度 | 说明 |
|------|------|--------|------|
| **方法 1: 视觉检查**  | 简单 | ⭐⭐⭐⭐⭐ | 查看 POV 图像，直接观察世界 |
| **方法 2: 日志分析** | 简单 | ⭐⭐⭐⭐ | 检查 MC 日志中的 MissionInit XML |
| **方法 3: Inventory 验证** | 中等 | ⭐⭐⭐⭐⭐ | 挖掘方块，检查物品栏 |
| **方法 4: 坐标检查** | 中等 | ⭐⭐⭐ | 检查 Y 坐标和位置 |

---

## 方法 1: 视觉检查（最直观）⭐⭐⭐⭐⭐

### 原理
FlatWorld 的层结构会直接反映在 POV 图像中。

### 步骤

#### 1. 创建测试脚本

```python
# test_flatworld.py
import gym
from src.envs import register_minerl_harvest_flatworld_env
import matplotlib.pyplot as plt

# 注册环境
register_minerl_harvest_flatworld_env()

# 创建环境
env = gym.make(
    'MineRLHarvestFlatWorldEnv-v0',
    generator_string="minecraft:bedrock,50*minecraft:stone;minecraft:plains",
    max_episode_steps=100
)

# 重置环境
obs = env.reset()

# 显示 POV
pov = obs['pov']  # shape: (320, 640, 3)
plt.imshow(pov)
plt.title("FlatWorld POV - Stone Layer")
plt.axis('off')
plt.savefig('/Users/nanzhang/aimc/test_flatworld_pov.png')
plt.show()

print(f"✅ POV 图像已保存: test_flatworld_pov.png")
print(f"   图像尺寸: {pov.shape}")

env.close()
```

#### 2. 运行测试

```bash
cd /Users/nanzhang/aimc
conda activate minedojo-x86
python test_flatworld.py
```

#### 3. 检查图像

打开 `test_flatworld_pov.png`，你应该看到：

| GeneratorString | 预期视觉效果 |
|----------------|------------|
| `50*minecraft:stone` | **灰色**石头地面 |
| `5*minecraft:sand` | **黄色**沙地 |
| `10*minecraft:gravel` | **深灰色**沙砾地面 |
| `2*minecraft:dirt,minecraft:grass_block` | **绿色**草地 |
| `5*minecraft:coal_ore` | 石头地面，**黑色**煤矿点缀 |

**示例对比**:

```
正确配置（石头）:      错误配置（未生效）:
┌────────────────┐    ┌────────────────┐
│  灰色石头地面  │    │ 绿色草地+树木  │
│  纹理明显      │    │ （默认世界）   │
└────────────────┘    └────────────────┘
```

---

## 方法 2: 日志分析 ⭐⭐⭐⭐

### 原理
MineRL 会将 MissionInit XML 打印到 MC 日志中，其中包含世界生成器配置。

### 步骤

#### 1. 运行环境

```python
import gym
from src.envs import register_minerl_harvest_flatworld_env

register_minerl_harvest_flatworld_env()

env = gym.make(
    'MineRLHarvestFlatWorldEnv-v0',
    generator_string="minecraft:bedrock,50*minecraft:stone;minecraft:plains",
    max_episode_steps=100
)

obs = env.reset()
print("✅ 环境已启动，检查日志...")
```

#### 2. 查找最新日志

```bash
# 找到最新的 MC 日志
ls -lt /Users/nanzhang/aimc/logs/mc_*.log | head -1

# 或直接查看最新日志
tail -f $(ls -t /Users/nanzhang/aimc/logs/mc_*.log | head -1)
```

#### 3. 检查 MissionInit XML

搜索 `<FlatWorldGenerator`：

```bash
grep -A 5 "FlatWorldGenerator" /Users/nanzhang/aimc/logs/mc_*.log | tail -20
```

**预期输出**:

```xml
<FlatWorldGenerator forceReset="true" generatorString="minecraft:bedrock,50*minecraft:stone;minecraft:plains"/>
```

**检查要点**:

| 检查项 | 正确值 | 错误值 |
|--------|--------|--------|
| `forceReset` | `"true"` | `"false"` 或缺失 |
| `generatorString` | 你配置的值 | 为空或不存在 |
| 是否有 `<DefaultWorldGenerator>` | ❌ 不应该有 | ✅ 说明配置错误 |

---

## 方法 3: Inventory 验证（最可靠）⭐⭐⭐⭐⭐

### 原理
通过挖掘方块并检查物品栏，确认世界结构。

### 步骤

#### 1. 创建交互式测试

```python
# test_flatworld_inventory.py
import gym
from src.envs import register_minerl_harvest_flatworld_env

register_minerl_harvest_flatworld_env()

env = gym.make(
    'MineRLHarvestFlatWorldEnv-v0',
    generator_string="minecraft:bedrock,50*minecraft:stone;minecraft:plains",
    initial_inventory=[{"type": "iron_pickaxe", "quantity": 1}],
    max_episode_steps=200
)

obs = env.reset()

print("=" * 60)
print("开始验证 FlatWorld 配置")
print("=" * 60)

# 执行一些向下挖掘的动作
for i in range(10):
    # 动作: 攻击（挖掘）
    action = {
        'camera': [0, 90],  # 向下看
        'buttons': {'attack': 1}  # 攻击
    }
    obs, reward, done, info = env.step(env.action_space.noop())
    
    if i == 5:
        # 开始挖掘
        obs, reward, done, info = env.step(action)

# 检查物品栏
inventory = obs.get('inventory', {})
print("\n📦 物品栏内容:")
for item, count in inventory.items():
    if hasattr(count, 'item'):
        count = count.item()
    if count > 0:
        print(f"  {item}: {count}")

# 预期结果
print("\n✅ 预期结果:")
print("  如果配置生效，应该有: cobblestone (圆石)")
print("  如果配置未生效，可能有: dirt, log, 等")

env.close()
```

#### 2. 运行验证

```bash
conda activate minedojo-x86
python test_flatworld_inventory.py
```

#### 3. 检查结果

| GeneratorString | 挖掘后预期物品 |
|----------------|--------------|
| `50*minecraft:stone` | ✅ `cobblestone` |
| `5*minecraft:sand` | ✅ `sand` |
| `10*minecraft:gravel` | ✅ `gravel` |
| `5*minecraft:coal_ore` | ✅ `coal` |
| 默认世界（未生效） | ❌ `dirt`, `log`, 等 |

---

## 方法 4: 坐标检查 ⭐⭐⭐

### 原理
FlatWorld 的 Y 坐标固定，可以通过坐标验证。

### 步骤

```python
import gym
from src.envs import register_minerl_harvest_flatworld_env

register_minerl_harvest_flatworld_env()

env = gym.make(
    'MineRLHarvestFlatWorldEnv-v0',
    generator_string="minecraft:bedrock,50*minecraft:stone;minecraft:plains",
    max_episode_steps=100
)

obs = env.reset()

# 获取玩家位置
location = obs.get('location_stats', {})
x = location.get('xpos', 0)
y = location.get('ypos', 0)
z = location.get('zpos', 0)

print(f"🌍 玩家位置: X={x:.1f}, Y={y:.1f}, Z={z:.1f}")

# FlatWorld 的 Y 坐标
# 基岩(1) + 石头(50) = Y=51（玩家站在上面，Y=53左右）
print(f"\n✅ FlatWorld 预期 Y 坐标: 51-53")
print(f"   实际 Y 坐标: {y:.1f}")

if 50 <= y <= 55:
    print("✅ Y 坐标正常，FlatWorld 配置可能生效")
else:
    print("⚠️ Y 坐标异常，可能是默认世界")

env.close()
```

**预期结果**:

| 世界类型 | Y 坐标范围 |
|---------|-----------|
| FlatWorld (50 层石头) | 51-53 |
| FlatWorld (5 层) | 6-8 |
| 默认世界 | 60-80（随机） |

---

## 🚨 常见问题排查

### 问题 1: POV 显示的是默认世界（有树、山）

**原因**: `generatorString` 未传递或环境类型错误

**解决**:

1. 检查是否使用了 `MineRLHarvestFlatWorldEnv-v0`
2. 确认 `generator_string` 参数拼写正确

```python
# ❌ 错误
env = gym.make('MineRLHarvestDefaultEnv-v0')  # 用了 Default 环境

# ✅ 正确
env = gym.make(
    'MineRLHarvestFlatWorldEnv-v0',  # FlatWorld 环境
    generator_string="minecraft:bedrock,50*minecraft:stone;minecraft:plains"
)
```

### 问题 2: 日志中没有 `<FlatWorldGenerator>`

**原因**: 环境未正确注册或使用了错误的环境

**解决**:

```python
# 确保先注册
from src.envs import register_minerl_harvest_flatworld_env
register_minerl_harvest_flatworld_env()

# 然后创建
env = gym.make('MineRLHarvestFlatWorldEnv-v0', ...)
```

### 问题 3: generatorString 语法错误

**常见错误**:

| 错误 | 正确 |
|------|------|
| `stone*50` | `50*minecraft:stone` |
| `minecraft:stone,bedrock` | `minecraft:bedrock,minecraft:stone` |
| `stone;plains` | `minecraft:stone;minecraft:plains` |
| `stone` | `minecraft:stone;minecraft:plains` （需要群系） |

**正确格式**:

```
格式: layer1,layer2,...,layerN;biome

示例:
- minecraft:bedrock,50*minecraft:stone;minecraft:plains
- minecraft:bedrock,10*minecraft:dirt,minecraft:grass_block;minecraft:forest
```

---

## 📊 完整验证清单

使用以下清单确保 FlatWorld 配置完全生效：

```
□ 1. 视觉检查
   □ POV 图像显示预期的方块类型
   □ 没有树木、山脉等自然地形

□ 2. 日志检查
   □ MC 日志中有 <FlatWorldGenerator>
   □ generatorString 值正确
   □ forceReset="true"

□ 3. Inventory 验证
   □ 挖掘方块后获得预期物品
   □ 没有获得非预期物品（如树木相关）

□ 4. 坐标验证
   □ Y 坐标在预期范围内
   □ 地形是平坦的

□ 5. 任务执行
   □ 任务可以完成
   □ 成功率符合预期
```

---

## 🎯 快速测试脚本

### 一键验证脚本

```bash
#!/bin/bash
# verify_flatworld.sh

echo "========================================"
echo "FlatWorld 配置验证"
echo "========================================"

# 1. 创建测试脚本
cat > /tmp/test_flatworld.py << 'EOF'
import gym
from src.envs import register_minerl_harvest_flatworld_env
import matplotlib.pyplot as plt

register_minerl_harvest_flatworld_env()

# 测试不同配置
configs = [
    ("stone", "minecraft:bedrock,50*minecraft:stone;minecraft:plains"),
    ("sand", "minecraft:bedrock,5*minecraft:sand;minecraft:desert"),
    ("coal", "minecraft:bedrock,20*minecraft:stone,5*minecraft:coal_ore,minecraft:grass_block;minecraft:mountains"),
]

for name, gen_string in configs:
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    env = gym.make(
        'MineRLHarvestFlatWorldEnv-v0',
        generator_string=gen_string,
        max_episode_steps=50
    )
    
    obs = env.reset()
    
    # 保存 POV
    pov = obs['pov']
    plt.imsave(f'/tmp/flatworld_{name}.png', pov)
    print(f"✅ POV 已保存: /tmp/flatworld_{name}.png")
    
    # 检查位置
    location = obs.get('location_stats', {})
    y = location.get('ypos', 0)
    print(f"   Y 坐标: {y:.1f}")
    
    env.close()

print("\n✅ 验证完成！检查 /tmp/flatworld_*.png 图像")
EOF

# 2. 运行测试
conda activate minedojo-x86
python /tmp/test_flatworld.py

# 3. 显示结果
echo ""
echo "========================================"
echo "验证结果"
echo "========================================"
ls -lh /tmp/flatworld_*.png
```

### 使用方法

```bash
cd /Users/nanzhang/aimc
chmod +x verify_flatworld.sh
./verify_flatworld.sh
```

---

## 📚 参考文档

- **FlatWorld 参数参考**: `docs/reference/FLATWORLD_GENERATOR_STRING_REFERENCE.md`
- **配置生成器**: `scripts/generate_flatworld_configs.py`
- **环境实现**: `src/envs/minerl_harvest_flatworld.py`

---

## ✅ 总结

| 验证方法 | 何时使用 | 可靠性 |
|---------|---------|--------|
| 视觉检查 | 快速验证 | ⭐⭐⭐⭐⭐ |
| 日志分析 | 配置调试 | ⭐⭐⭐⭐ |
| Inventory 验证 | 最终确认 | ⭐⭐⭐⭐⭐ |
| 坐标检查 | 辅助验证 | ⭐⭐⭐ |

**推荐验证流程**:
1. 先用**视觉检查**快速判断
2. 如有问题，用**日志分析**排查配置
3. 最后用**Inventory 验证**确认功能

**记住**: FlatWorld 的最大优势是**可预测性**，如果配置正确，世界应该完全符合你的预期！

