# Inventory 问题相关代码文件列表

## 问题描述
物品栏打开一次后就关闭了

## 1. 项目代码（你的代码库）

### `src/envs/minedojo_harvest.py`
**MineDojoBiomeWrapper 类（第 145-491 行）**

#### 关键方法：
- `__init__()`: 第 189 行
  - 初始化 `self._inventory_opened = False`

- `reset()`: 第 206 行
  - 重置 `self._inventory_opened = False`

- `step()`: 第 225-259 行
  - **Inventory 状态管理逻辑（第 245-256 行）**:
  ```python
  current_inventory = minerl_action.get('inventory', 0)
  
  if current_inventory == 1 and not self._inventory_opened:
      self._inventory_opened = True
      # inventory=1 保留，正常发送
  else:
      # 移除 inventory 键
      minerl_action = minerl_action.copy()
      if 'inventory' in minerl_action:
          del minerl_action['inventory']
  ```

- `_convert_action_to_minedojo()`: 第 297-397 行
  - **处理 inventory 动作（第 374-379 行）**:
  ```python
  # 5. Functional (index 5)
  # 优先级: inventory > attack > use > drop
  if minerl_action.get('inventory', 0):
      minedojo_action[5] = 8  # inventory
  elif minerl_action.get('attack', 0):
      minedojo_action[5] = 3  # attack
  ```

---

## 2. MineDojo 源码（已修改）

路径: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo`

### `sim/sim.py`
**第 228-236 行: common_actions 列表**
```python
common_actions = [
    "forward", "back", "left", "right",
    "jump", "sneak", "sprint",
    "use", "attack", "drop",
    "inventory",  # ← 我们添加的
]
```

### `sim/handlers/agent/action.py`
**第 35-56 行: KeybasedCommandAction.to_hero() 方法**

⚠️ **关键！** 这是生成 Malmo 命令的地方

```python
def to_hero(self, x):
    cmd = ""
    verb = self.command
    
    if isinstance(x, np.ndarray):
        flat = x.flatten().tolist()
        flat = [str(y) for y in flat]
        adjective = " ".join(flat)
    elif isinstance(x, Iterable) and not isinstance(x, str):
        adjective = " ".join([str(y) for y in x])
    else:
        adjective = str(x)
    cmd += "{} {}".format(verb, adjective)  # ← 总是生成命令
    
    return cmd
```

**注意**：
- `inventory=0` 会生成 `"inventory 0"`
- `inventory=1` 会生成 `"inventory 1"`

### `sim/wrappers/ar_nn/nn_action_space_wrapper.py`

**第 41-48 行: 定义 MultiDiscrete 动作空间**
```python
[
    3,      # forward/back
    3,      # left/right
    4,      # jump/sneak/sprint
    36001,  # camera pitch
    36001,  # camera yaw
    9,      # functional actions (0-8) ← 扩展为 9
    244,    # craft arg
    36,     # inventory/equip/place/destroy arg
]
```

**第 145-150 行: 处理 fn_action == 8 (inventory)**
```python
elif fn_action == 8:
    # inventory action - added for STEVE-1 compatibility
    noop["inventory"] = 1
```

### `sim/bridge/mc_ins/instance/instance.py`
- MineDojo 环境的核心实例类
- 负责与 Malmo 通信
- `step()` 方法会调用 `_action_obj_to_xml()`

---

## 3. Malmo Java 代码（JAR 中，不可修改）

路径: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build/`

### `sources/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java`
处理 keyboard 命令（如 `"inventory 1"`）

```java
if (press) {
    FakeKeyboard.press(key);
} else {
    FakeKeyboard.release(key);
}
```

### `sources/main/java/com/microsoft/Malmo/Client/FakeKeyboard.java`
⚠️ **关键！** 这是 toggle 行为的来源

- `press(keyCode)`: 将键添加到 `keysDown` 集合
- `release(keyCode)`: 从 `keysDown` 集合移除键

---

## 4. 测试脚本

### `a.py`（你的测试文件）
- 直接使用 MineDojo 原生 API
- `act[5] = 8` → 触发 inventory
- 测试结果显示 toggle 行为

### `scripts/test_inventory_key_removal.py`
- 测试 wrapper 的键移除逻辑
- 使用 `MineDojoHarvestEnv-v0`

### `scripts/debug_inventory_state_management.py`
- 调试脚本，打印 wrapper 状态

---

## 5. Patch 文件

### `docker/minedojo_inventory.patch`
定义了对 MineDojo 的修改，只修改：
- `sim.py`
- `nn_action_space_wrapper.py`

---

## 调试思路

### 可能原因

1. **wrapper 的键移除逻辑不生效**
   - 检查 `src/envs/minedojo_harvest.py` 第 251-256 行

2. **`del minerl_action['inventory']` 没有真正删除键**
   - 检查 Python dict 的 `copy()` 和 `del` 操作

3. **`_convert_action_to_minedojo` 仍然处理了 inventory**
   - 检查第 374-379 行的 `if` 条件

4. **MineDojo 的 `to_hero()` 被调用了**
   - 添加 `print()` 调试 `action.py` 第 52 行

5. **Malmo 收到了 `"inventory 0"` 或多个 `"inventory 1"`**
   - 添加 `print()` 调试 FakeKeyboard 调用

### 建议调试步骤

1. 在 `wrapper.step()` 中打印 `minerl_action`（删除前后）
2. 在 `_convert_action_to_minedojo()` 中打印 `minedojo_action[5]`
3. 在 `action.py` 的 `to_hero()` 中打印生成的命令
4. 观察游戏窗口，记录 GUI 显示/隐藏的时机

### 关键数据流

```
STEVE-1 输出
  ↓
MineRL action dict (inventory=1 或 0)
  ↓
wrapper.step() → 键移除逻辑
  ↓
_convert_action_to_minedojo() → 转换为 MultiDiscrete
  ↓
MineDojo env.step(minedojo_action)
  ↓
nn_action_space_wrapper → fn_action=8 → {"inventory": 1}
  ↓
action.py to_hero() → "inventory 1"
  ↓
Malmo CommandForKey → FakeKeyboard.press()
  ↓
Minecraft GUI toggle
```


