# 世界生成器 Biome 参数诊断

## 问题描述

在配置文件中设置了 `generator_options: '{"biome":"extreme_hills"}'`，但创建的环境看起来仍然像平原。

## 可能的原因

### 1. MineRL 的 DefaultWorldGenerator 限制

MineRL 的 `DefaultWorldGenerator` 可能：
- 不支持所有 Minecraft 的 biome 名称
- 需要特定的 biome 名称格式（大小写、下划线等）
- 在某些版本中 biome 参数可能被忽略

### 2. 参数传递问题

虽然代码逻辑看起来正确，但需要确认：
- `generator_options` 是否正确从 YAML 传递到 `DefaultWorldGenerator`
- JSON 字符串格式是否正确

### 3. Minecraft 版本差异

不同版本的 Minecraft 可能：
- 使用不同的 biome ID 系统
- 某些 biome 名称在不同版本中不同

## 诊断步骤

### 步骤 1: 检查日志输出

运行评估任务时，查看日志中的世界生成器配置信息：

```bash
scripts/run_evaluation.sh --task harvest_1_cobblestone --n-trials 1
```

应该看到类似这样的日志：

```
============================================================
🌍 世界生成器配置:
  force_reset: True
  generator_options: {"biome":"extreme_hills"}
  generator_options 类型: <class 'str'>
  ✅ JSON 解析成功: {'biome': 'extreme_hills'}
  🏔️  生物群系: extreme_hills
============================================================
```

如果看到这些日志，说明参数传递是正确的。

### 步骤 2: 验证 MineRL 支持的 Biome

MineRL 可能只支持部分 biome。根据文档，常见的支持 biome 包括：

- `plains` - 平原（已验证可用）
- `forest` - 森林
- `desert` - 沙漠
- `swampland` - 沼泽
- `extreme_hills` - 山地（可能不支持或格式不对）
- `ice_plains` - 冰原
- `taiga` - 针叶林
- `jungle` - 丛林

### 步骤 3: 尝试替代方案

如果 `extreme_hills` 不工作，可以尝试：

1. **使用其他 biome**：
   ```yaml
   generator_options: '{"biome":"plains"}'  # 平原（已验证可用）
   generator_options: '{"biome":"desert"}'   # 沙漠
   generator_options: '{"biome":"forest"}'  # 森林
   ```

2. **检查 MineRL 版本**：
   ```bash
   pip show minerl
   ```
   不同版本的 MineRL 可能支持不同的 biome。

3. **使用坐标生成**：
   如果 biome 参数不工作，可能需要使用坐标或其他参数来强制生成特定地形。

## 当前代码状态

代码已经添加了详细的日志输出，可以帮助诊断问题：

```python
# src/envs/minerl_harvest.py
def create_server_world_generators(self):
    # ... 详细日志输出
    logger.info("🌍 世界生成器配置:")
    logger.info(f"  generator_options: {generator_options}")
    # ... JSON 解析和验证
```

## 建议的解决方案

### 方案 1: 使用已验证的 Biome

对于需要山地地形的任务（如采集圆石、铁矿），可以：

1. **使用 `plains` biome**：虽然地形平坦，但可以通过其他方式生成石头（如自然生成的石头结构）
2. **使用 `desert` biome**：沙漠中也有石头结构
3. **移除 biome 限制**：让世界自然生成，可能包含山地

### 方案 2: 检查 MineRL 源码

查看 MineRL 的 `DefaultWorldGenerator` 实现：

```python
# 可能需要检查 minerl 源码
from minerl.herobraine.hero.handlers import DefaultWorldGenerator
import inspect
print(inspect.getsource(DefaultWorldGenerator.__init__))
```

### 方案 3: 使用其他世界生成参数

`generator_options` 可能支持其他参数，如：

```json
{
  "biome": "extreme_hills",
  "structures": true,
  "generate_features": true
}
```

## 测试建议

1. **运行诊断**：
   ```bash
   scripts/run_evaluation.sh --task harvest_1_cobblestone --n-trials 1
   ```

2. **查看日志**：
   ```bash
   tail -f logs/mc_*.log | grep -E "世界生成器|biome|generator_options"
   ```

3. **对比不同 biome**：
   - 测试 `plains`（已知可用）
   - 测试 `extreme_hills`（问题 biome）
   - 测试 `desert`（另一个选项）

4. **检查实际生成的地形**：
   - 查看视频输出
   - 检查地形特征（高度、植被、石头分布）

## 相关文件

- `src/envs/minerl_harvest.py` - 环境定义和世界生成器配置
- `src/utils/steve1_mineclip_agent_env_utils.py` - 环境创建逻辑
- `config/eval_tasks.yaml` - 任务配置文件

## 下一步

1. ✅ 已添加详细日志
2. ⏳ 等待用户运行测试并查看日志输出
3. ⏳ 根据日志结果确定问题原因
4. ⏳ 实施相应的修复方案

