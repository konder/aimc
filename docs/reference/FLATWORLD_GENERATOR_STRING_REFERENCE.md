# FlatWorldGenerator 参数参考

**适用场景**: 完全平坦的地形，适合建筑/简单采集任务  
**限制**: 无法控制矿物生成、生物群系特征

---

## 📝 generatorString 格式

```
<layers>;<biome>;<structures>
```

### 1. Layers（地层）

格式: `<count>*<blockID>` 或 `<blockID>`

**示例**:
```python
# 基础平地
"minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block"

# 更厚的土层
"minecraft:bedrock,5*minecraft:dirt,minecraft:grass_block"

# 沙漠风格
"minecraft:bedrock,3*minecraft:sandstone,5*minecraft:sand"

# 沙砾层
"minecraft:bedrock,2*minecraft:stone,3*minecraft:gravel"
```

### 2. Biome（生物群系）

指定群系类型（影响生物生成、天气）

**示例**:
```python
"...; minecraft:plains"        # 平原（默认）
"...; minecraft:desert"        # 沙漠
"...; minecraft:snowy_tundra"  # 雪地
```

### 3. Structures（结构）

可选的结构生成（村庄等）

**示例**:
```python
"...; village"                 # 生成村庄
"...; stronghold"              # 生成要塞
```

---

## 🎨 常用配置示例

### 标准平原
```python
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block;minecraft:plains"
)
```

### 沙漠风格（适合采沙子）
```python
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,3*minecraft:sandstone,5*minecraft:sand;minecraft:desert"
)
```

### 沙砾层（适合采沙砾）
```python
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,2*minecraft:stone,10*minecraft:gravel;minecraft:plains"
)
```

### 圆石/石头层
```python
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,50*minecraft:stone;minecraft:plains"
)
```

### 粘土层
```python
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,2*minecraft:stone,5*minecraft:clay;minecraft:swamp"
)
```

### 雪地
```python
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block,minecraft:snow;minecraft:snowy_tundra"
)
```

---

## ⚠️ 重要限制

### ❌ 不能做的事

1. **矿物生成**: 无法生成煤矿、铁矿、金矿等
2. **树木**: 无法控制树木生成（需要依赖群系）
3. **动物**: 动物生成取决于群系，不是地层
4. **洞穴**: 无法生成洞穴系统
5. **资源分布**: 无法像真实世界那样随机分布资源

### ✅ 可以做的事

1. **地表方块**: 可以指定任意方块作为地层
2. **厚度控制**: 精确控制每层厚度
3. **完全平坦**: 无山脉、河流等障碍
4. **快速生成**: 生成速度极快

---

## 🎯 适用任务

### ✅ 适合 FlatWorldGenerator 的任务

| 任务 | 配置 | 说明 |
|------|------|------|
| `harvest_1_dirt` | 标准平原 | 地表就是泥土 |
| `harvest_1_sand` | 沙漠风格 | 地表全是沙子 |
| `harvest_1_gravel` | 沙砾层 | 地表全是沙砾 |
| `harvest_1_snow` | 雪地 + 雪层 | 地表有雪 |
| `harvest_1_clay` | 粘土层 | 地表是粘土 |

### ❌ 不适合 FlatWorldGenerator 的任务

| 任务 | 原因 | 推荐方案 |
|------|------|----------|
| `harvest_1_cobblestone` | 需要挖石头，但 FlatWorldGenerator 可以放石头层 | ⚠️ 可以用石头层 |
| `harvest_1_coal` | 需要煤矿（地下矿脉） | ❌ 必须用 BiomeGenerator |
| `harvest_1_iron_ore` | 需要铁矿（地下矿脉） | ❌ 必须用 BiomeGenerator |
| `harvest_1_log` | 需要树木 | ❌ 必须用 BiomeGenerator |
| `harvest_1_milk` | 需要牛（生物生成） | ⚠️ 可能需要 BiomeGenerator |

---

## 💡 最佳实践

### 方案 1: 混合使用（推荐）

**简单资源** → FlatWorldGenerator  
**复杂资源** → BiomeGenerator（需编译）

```python
def create_server_world_generators(self) -> List[Handler]:
    task_id = self.task_id  # 假设你传入了 task_id
    
    # 简单任务用 FlatWorldGenerator
    if task_id in ['harvest_1_dirt', 'harvest_1_sand', 'harvest_1_gravel']:
        flat_configs = {
            'harvest_1_dirt': 'minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block;minecraft:plains',
            'harvest_1_sand': 'minecraft:bedrock,3*minecraft:sandstone,5*minecraft:sand;minecraft:desert',
            'harvest_1_gravel': 'minecraft:bedrock,2*minecraft:stone,10*minecraft:gravel;minecraft:plains',
        }
        return [handlers.FlatWorldGenerator(force_reset=True, generatorString=flat_configs[task_id])]
    
    # 复杂任务用 BiomeGenerator
    else:
        biome_map = {
            'harvest_1_log': 4,      # Forest
            'harvest_1_coal': 3,     # Mountains
            'harvest_1_iron_ore': 3, # Mountains
            # ...
        }
        return [handlers.BiomeGenerator(biome_id=biome_map.get(task_id, 1), force_reset=True)]
```

### 方案 2: 预制石头层（圆石任务）

```python
# harvest_1_cobblestone 可以用石头层
handlers.FlatWorldGenerator(
    force_reset=True,
    generatorString="minecraft:bedrock,50*minecraft:stone;minecraft:plains"
)
# 玩家可以直接挖地面获得圆石
```

---

## 🔧 动态配置示例

```python
FLAT_WORLD_CONFIGS = {
    "dirt": "minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block;minecraft:plains",
    "sand": "minecraft:bedrock,3*minecraft:sandstone,5*minecraft:sand;minecraft:desert",
    "gravel": "minecraft:bedrock,2*minecraft:stone,10*minecraft:gravel;minecraft:plains",
    "stone": "minecraft:bedrock,50*minecraft:stone;minecraft:plains",
    "clay": "minecraft:bedrock,2*minecraft:stone,5*minecraft:clay;minecraft:swamp",
    "snow": "minecraft:bedrock,2*minecraft:dirt,minecraft:grass_block,minecraft:snow;minecraft:snowy_tundra",
}

def get_flat_world_config(resource_type: str) -> str:
    return FLAT_WORLD_CONFIGS.get(resource_type, FLAT_WORLD_CONFIGS["dirt"])
```

---

## 📚 完整方块列表

<details>
<summary>点击展开可用方块</summary>

```
minecraft:stone
minecraft:granite
minecraft:diorite
minecraft:andesite
minecraft:grass_block
minecraft:dirt
minecraft:cobblestone
minecraft:oak_planks
minecraft:sand
minecraft:red_sand
minecraft:gravel
minecraft:gold_ore
minecraft:iron_ore
minecraft:coal_ore
minecraft:oak_log
minecraft:glass
minecraft:sandstone
minecraft:wool
minecraft:clay
minecraft:ice
minecraft:snow
minecraft:snow_block
minecraft:obsidian
minecraft:bedrock
... (还有更多)
```

</details>

---

## 🎯 总结

**FlatWorldGenerator 的核心价值**:
- ✅ 地表资源（沙子、沙砾、泥土、雪）
- ✅ 完全平坦（训练稳定）
- ✅ 快速生成

**FlatWorldGenerator 的限制**:
- ❌ 地下矿物（煤炭、铁矿）
- ❌ 树木/植物（依赖群系）
- ❌ 复杂地形（山洞、河流）

**建议**: 
- 简单资源 → FlatWorldGenerator
- 复杂资源 → BiomeGenerator（需编译）

