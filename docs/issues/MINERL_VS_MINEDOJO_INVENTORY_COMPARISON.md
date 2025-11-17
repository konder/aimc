# MineRL vs MineDojo Inventory 实现对比

## 概览

| 特性 | MineRL | MineDojo |
|------|--------|----------|
| Minecraft 版本 | 1.16.5 | 1.11.2 |
| Malmo 版本 | MCP-Reborn 6.13 | Malmo 0.37.0 |
| 安装路径 | `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/` | `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/` |
| Action Space | Dict | MultiDiscrete + Dict (wrapper) |
| Inventory 行为 | Press 后保持打开 | 每次 Press 触发 toggle |

---

## 1. Action Space 定义

### MineRL
**文件**: `minerl/herobraine/hero/spaces.py`

```python
# Dict-based action space
{
    "attack": Discrete(2),
    "back": Discrete(2),
    "camera": Box([-180, 180], shape=(2,)),
    "drop": Discrete(2),
    "forward": Discrete(2),
    "hotbar.1-9": Discrete(2),
    "inventory": Discrete(2),  # ← 原生支持
    "jump": Discrete(2),
    "left": Discrete(2),
    "right": Discrete(2),
    "sneak": Discrete(2),
    "sprint": Discrete(2),
    "use": Discrete(2)
}
```

### MineDojo
**文件**: `minedojo/sim/wrappers/ar_nn/nn_action_space_wrapper.py`

```python
# MultiDiscrete action space (第 41-48 行)
MultiDiscrete([
    3,      # [0] forward/back (0=noop, 1=forward, 2=back)
    3,      # [1] left/right (0=noop, 1=left, 2=right)
    4,      # [2] jump/sneak/sprint
    36001,  # [3] camera pitch
    36001,  # [4] camera yaw
    9,      # [5] functional actions (0-8)
            #     0=noop, 1=use, 2=drop, 3=attack, 
            #     4=craft, 5=equip, 6=place, 7=destroy, 8=inventory
    244,    # [6] craft arg
    36,     # [7] inventory/equip/place/destroy arg
])
```

**注意**: `inventory` 是我们后添加的（原本只有 0-7）

---

## 2. Action Handler 实现

### MineRL
**文件**: `minerl/herobraine/hero/handlers/agent/action.py`

#### `Action.to_hero()` 方法（第 35-56 行）

```python
def to_hero(self, x):
    cmd = ""
    verb = self.command
    adjective = str(x)
    
    # MineRL: 总是生成命令，不过滤 0
    cmd += "{} {}".format(verb, adjective)
    
    return cmd
```

**行为**:
- `inventory=0` → 生成 `"inventory 0"`
- `inventory=1` → 生成 `"inventory 1"`

#### 关键特性
- ✅ 原生支持 `inventory` 动作
- ✅ 不过滤任何值
- ✅ 所有命令都会发送到 Malmo

---

### MineDojo
**文件**: `minedojo/sim/handlers/agent/action.py`

#### `KeybasedCommandAction.to_hero()` 方法（第 35-56 行）

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
    
    # MineDojo: 总是生成命令（官方版本）
    cmd += "{} {}".format(verb, adjective)
    
    return cmd
```

**行为**:
- `inventory=0` → 生成 `"inventory 0"`
- `inventory=1` → 生成 `"inventory 1"`

#### 我们的修改
**文件**: `minedojo/sim/sim.py` (第 228-236 行)

```python
common_actions = [
    "forward", "back", "left", "right",
    "jump", "sneak", "sprint",
    "use", "attack", "drop",
    "inventory",  # ← 添加的
]
```

---

## 3. Wrapper 转换逻辑

### MineRL (无 wrapper，直接使用)
```python
# 直接使用 Dict action space
action = env.action_space.no_op()
action['inventory'] = 1
env.step(action)
```

### MineDojo (需要 wrapper)
**文件**: `src/envs/minedojo_harvest.py`

#### Wrapper 流程
```python
# 1. 接收 MineRL 格式的 action (Dict)
minerl_action = {
    "inventory": 1,
    "forward": 0,
    # ... 其他动作
}

# 2. Wrapper 状态管理（第 245-256 行）
current_inventory = minerl_action.get('inventory', 0)

if current_inventory == 1 and not self._inventory_opened:
    self._inventory_opened = True
    # inventory=1 保留
else:
    # 移除 inventory 键
    minerl_action = minerl_action.copy()
    if 'inventory' in minerl_action:
        del minerl_action['inventory']

# 3. 转换为 MineDojo MultiDiscrete（第 374-379 行）
if minerl_action.get('inventory', 0):
    minedojo_action[5] = 8  # functional action = inventory

# 4. MineDojo nn_action_space_wrapper 转换（第 145-150 行）
elif fn_action == 8:
    noop["inventory"] = 1  # 转回 Dict 格式
```

---

## 4. Malmo 命令执行

### 共同点（两者相同）
**文件**: `*/sim/Malmo/Minecraft/build/sources/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java`

```java
public void onExecute(String verb, String parameter, MalmoMod.safeTileEntityAccess accessHelper) {
    // Parse parameter as integer (0 or 1)
    int press = Integer.parseInt(parameter);
    
    if (press == 1) {
        FakeKeyboard.press(key);
    } else {
        FakeKeyboard.release(key);
    }
}
```

**文件**: `*/sim/Malmo/Minecraft/build/sources/main/java/com/microsoft/Malmo/Client/FakeKeyboard.java`

```java
public static void press(int keyCode) {
    if (!keysDown.contains(keyCode)) {
        keysDown.add(keyCode);
        // 发送 press 事件到 Minecraft
    }
}

public static void release(int keyCode) {
    if (keysDown.contains(keyCode)) {
        keysDown.remove(keyCode);
        // 发送 release 事件到 Minecraft
    }
}
```

---

## 5. Minecraft GUI 行为差异（关键！）

### MineRL (MC 1.16.5)
```
Step 1: "inventory 1" → FakeKeyboard.press() → GUI 打开
Step 2: "inventory 0" → FakeKeyboard.release() → GUI 保持打开 ✅
Step 3: "inventory 0" → FakeKeyboard.release() → GUI 保持打开 ✅
...
```

**原因**: MC 1.16.5 忽略 release 事件，GUI 保持状态

---

### MineDojo (MC 1.11.2)
```
Step 1: "inventory 1" → FakeKeyboard.press() → GUI 打开
Step 2: "inventory 0" → FakeKeyboard.release() → GUI 关闭 ❌
Step 3: "inventory 1" → FakeKeyboard.press() → GUI 打开
Step 4: "inventory 1" → FakeKeyboard.press() (已在 keysDown，忽略)
        但 GUI toggle 逻辑触发 → GUI 关闭 ❌
```

**原因**: MC 1.11.2 对 press/release 事件敏感，触发 toggle

---

## 6. 测试对比

### MineRL 测试
**文件**: `scripts/test_minerl_inventory_simple.py`

```python
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

for i in range(100):
    action = env.action_space.no_op()
    if i == 1:
        action['inventory'] = 1  # 打开
    else:
        action['inventory'] = 0  # 保持
    
    obs, reward, done, info = env.step(action)
```

**结果**: GUI 在 Step 2 打开后一直保持打开 ✅

---

### MineDojo 测试
**文件**: `a.py`

```python
env = minedojo.make(task_id="harvest_wool_with_shears_and_sheep")
obs = env.reset()

for i in range(50):
    act = env.action_space.no_op()
    if i == 1:
        act[5] = 8  # inventory
    if i == 7:
        act[5] = 8  # inventory
    # ...
    obs, reward, done, info = env.step(act)
```

**结果**:
```
Step 1:  act[5]=8 → GUI 显示 ✅
Step 2:  act[5]=0 → GUI 不显示
Step 7:  act[5]=8 → GUI 显示 ✅
Step 8:  act[5]=8 → GUI 不显示 (toggle) ❌
Step 22+: 交替 toggle
```

---

## 7. 关键文件路径对比

### MineRL
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/
├── herobraine/
│   └── hero/
│       └── handlers/
│           └── agent/
│               └── action.py          # Action.to_hero()
├── env/
│   └── malmo/
│       └── instance/
│           └── instance.py            # 环境实例
└── MCP-Reborn/
    └── (JAR 文件)
        └── com/microsoft/Malmo/
            ├── MissionHandlers/
            │   └── CommandForKey.java
            └── Client/
                └── FakeKeyboard.java
```

### MineDojo
```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/
├── sim/
│   ├── sim.py                          # common_actions 定义
│   ├── handlers/
│   │   └── agent/
│   │       └── action.py               # KeybasedCommandAction.to_hero()
│   ├── wrappers/
│   │   └── ar_nn/
│   │       └── nn_action_space_wrapper.py  # MultiDiscrete 转换
│   └── bridge/
│       └── mc_ins/
│           └── instance/
│               └── instance.py         # 环境实例
└── Malmo/
    └── Minecraft/
        └── build/
            └── sources/main/java/com/microsoft/Malmo/
                ├── MissionHandlers/
                │   └── CommandForKey.java
                └── Client/
                    └── FakeKeyboard.java
```

---

## 8. 调试建议

### 对比调试点

| 位置 | MineRL | MineDojo |
|------|--------|----------|
| Action 输入 | `action['inventory']` | `act[5]` (MultiDiscrete) |
| Handler | `Action.to_hero()` | `KeybasedCommandAction.to_hero()` |
| 命令生成 | `"inventory 0/1"` | `"inventory 0/1"` |
| FakeKeyboard | `press()/release()` | `press()/release()` (相同) |
| Minecraft | MC 1.16.5 (忽略 release) | MC 1.11.2 (响应 release) |

### 添加 print 调试

#### MineRL
```python
# minerl/herobraine/hero/handlers/agent/action.py
def to_hero(self, x):
    cmd = ""
    verb = self.command
    adjective = str(x)
    cmd += "{} {}".format(verb, adjective)
    
    if verb == "inventory":  # 添加
        print(f"[MineRL to_hero] inventory: x={x}, cmd='{cmd}'")
    
    return cmd
```

#### MineDojo
```python
# minedojo/sim/handlers/agent/action.py
def to_hero(self, x):
    cmd = ""
    verb = self.command
    adjective = str(x)
    cmd += "{} {}".format(verb, adjective)
    
    if verb == "inventory":  # 添加
        print(f"[MineDojo to_hero] inventory: x={x}, cmd='{cmd}'")
    
    return cmd
```

---

## 9. 总结

| 方面 | MineRL | MineDojo |
|------|--------|----------|
| **实现复杂度** | 简单（原生支持） | 复杂（需要 wrapper + patch） |
| **Inventory 行为** | 符合预期（保持打开） | 不符合预期（toggle） |
| **根本原因** | MC 1.16.5 GUI 系统 | MC 1.11.2 GUI 系统 |
| **解决难度** | 无需解决 | 需要 wrapper 状态管理 |
| **兼容性** | STEVE-1 训练环境 ✅ | 需要模拟 MineRL 行为 |

---

## 10. 当前 Wrapper 策略

**目标**: 让 MineDojo 模拟 MineRL 的 inventory 行为

**方法**: 只发送一次 `inventory=1`，之后移除 inventory 键

```python
# src/envs/minedojo_harvest.py (第 245-256 行)
if current_inventory == 1 and not self._inventory_opened:
    self._inventory_opened = True
    # inventory=1 保留，发送到 MineDojo
else:
    # 移除 inventory 键，不发送任何命令
    del minerl_action['inventory']
```

**预期效果**:
- Step 2: 发送 `"inventory 1"` → GUI 打开
- Step 3+: 不发送命令 → GUI 保持打开

**实际需要验证**: 键移除逻辑是否生效


