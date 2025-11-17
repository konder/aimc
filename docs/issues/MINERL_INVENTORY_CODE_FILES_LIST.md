# MineRL Inventory 实现代码文件列表

## 概述
MineRL 使用 Minecraft 1.16.5 + MCP-Reborn 6.13，inventory 行为正常（打开后保持打开）

## 环境信息
- **路径**: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/`
- **Minecraft 版本**: 1.16.5
- **Malmo 版本**: MCP-Reborn 6.13
- **JAR 文件**: `mcprec-6.13.jar` (444 MB)

---

## 1. Action Space 定义

### `minerl/herobraine/hero/spaces.py`
定义 MineRL 的动作空间

```python
# Dict-based action space
{
    "attack": Discrete(2),
    "back": Discrete(2),
    "camera": Box([-180, 180], shape=(2,)),  # 连续空间
    "drop": Discrete(2),
    "forward": Discrete(2),
    "hotbar.1": Discrete(2),
    "hotbar.2": Discrete(2),
    "hotbar.3": Discrete(2),
    "hotbar.4": Discrete(2),
    "hotbar.5": Discrete(2),
    "hotbar.6": Discrete(2),
    "hotbar.7": Discrete(2),
    "hotbar.8": Discrete(2),
    "hotbar.9": Discrete(2),
    "inventory": Discrete(2),  # ← 原生支持
    "jump": Discrete(2),
    "left": Discrete(2),
    "right": Discrete(2),
    "sneak": Discrete(2),
    "sprint": Discrete(2),
    "use": Discrete(2)
}
```

**特点**:
- ✅ 原生支持 `inventory` 动作
- ✅ Dict-based，直接访问 `action['inventory']`
- ✅ 不需要 wrapper 转换

---

## 2. Action Handler 实现

### `minerl/herobraine/hero/handlers/agent/action.py`

#### `Action` 基类

**关键方法: `to_hero()`**（第 35-60 行左右）

```python
class Action(TranslationHandler):
    """
    An action handler based on commands
    """
    
    def __init__(self, command: str, space: spaces.MineRLSpace):
        self._command = command
        super().__init__(space)
    
    @property
    def command(self):
        return self._command
    
    def to_hero(self, x):
        """
        Returns a command string for the multi command action.
        
        Args:
            x: Action value (0 or 1 for inventory)
        
        Returns:
            cmd: Malmo command string
        """
        cmd = ""
        verb = self.command
        adjective = str(x)
        
        # MineRL: 总是生成命令，不过滤任何值
        cmd += "{} {}".format(verb, adjective)
        
        return cmd
```

**行为**:
- `inventory=0` → 生成 `"inventory 0"`
- `inventory=1` → 生成 `"inventory 1"`
- ✅ 不进行任何过滤
- ✅ 所有命令都发送到 Malmo

#### `KeybasedCommandAction` 类

继承自 `Action`，用于键盘命令（如 inventory）

```python
class KeybasedCommandAction(Action):
    """
    Keyboard command action
    """
    
    def __init__(self, command: str, **kwargs):
        super().__init__(
            command=command,
            space=spaces.Discrete(2),  # 0 or 1
            **kwargs
        )
```

---

## 3. Environment Handler 注册

### `minerl/herobraine/env_specs/human_controls.py`

定义所有可用的动作

```python
def create_agent_handlers() -> List[Handler]:
    return [
        # ... 其他动作 ...
        
        # Inventory
        handlers.KeybasedCommandAction("inventory"),
        
        # ... 其他动作 ...
    ]
```

**注意**: `inventory` 是标准动作，无需额外配置

---

## 4. Environment 实例

### `minerl/env/malmo/instance/instance.py`

MineRL 环境的核心实例类

#### 关键方法

**`step(action)`**（处理动作执行）

```python
def step(self, action):
    """
    Execute action in the environment
    
    Args:
        action: Dict with action keys
    
    Returns:
        obs, reward, done, info
    """
    # 1. 转换 action dict 到 XML 命令
    xml_commands = self._action_obj_to_xml(action)
    
    # 2. 发送命令到 Malmo
    self._send_command(xml_commands)
    
    # 3. 获取观察
    obs = self._get_observation()
    
    return obs, reward, done, info
```

**`_action_obj_to_xml(action)`**（转换动作为 XML）

```python
def _action_obj_to_xml(self, action: Dict) -> str:
    """
    Convert action dict to Malmo XML commands
    
    Args:
        action: {"inventory": 1, "forward": 0, ...}
    
    Returns:
        xml_str: Malmo command XML
    """
    commands = []
    
    for handler in self._action_handlers:
        key = handler.to_string()  # e.g., "inventory"
        if key in action:
            value = action[key]
            # 调用 handler.to_hero() 生成命令
            cmd = handler.to_hero(value)
            if cmd:
                commands.append(cmd)
    
    return "\n".join(commands)
```

---

## 5. Malmo 底层实现（Java）

路径: `/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/`

### `CommandForKey.java`

处理键盘命令

```java
public class CommandForKey extends CommandBase {
    
    private KeyBinding key;
    
    @Override
    public void execute(String verb, String parameter) {
        // verb = "inventory"
        // parameter = "0" or "1"
        
        int press = Integer.parseInt(parameter);
        
        if (press == 1) {
            // 按下键
            FakeKeyboard.press(this.key.getKeyCode());
        } else {
            // 释放键
            FakeKeyboard.release(this.key.getKeyCode());
        }
    }
}
```

### `FakeKeyboard.java`

模拟键盘输入

```java
public class FakeKeyboard {
    
    private static Set<Integer> keysDown = new HashSet<>();
    
    public static void press(int keyCode) {
        if (!keysDown.contains(keyCode)) {
            keysDown.add(keyCode);
            // 发送 key press 事件到 Minecraft
            KeyBinding.setKeyBindState(keyCode, true);
        }
    }
    
    public static void release(int keyCode) {
        if (keysDown.contains(keyCode)) {
            keysDown.remove(keyCode);
            // 发送 key release 事件到 Minecraft
            KeyBinding.setKeyBindState(keyCode, false);
        }
    }
}
```

---

## 6. Minecraft 1.16.5 GUI 处理

### 关键差异（与 MC 1.11.2 对比）

**MC 1.16.5 的 Inventory GUI 行为**:

```
Step 1: "inventory 1" → FakeKeyboard.press() → GUI 打开 ✅
Step 2: "inventory 0" → FakeKeyboard.release() → **GUI 保持打开** ✅
Step 3: "inventory 0" → FakeKeyboard.release() → **GUI 保持打开** ✅
...
```

**原因**: 
- MC 1.16.5 在 GUI 打开后，**忽略** release 事件
- GUI 保持打开状态，直到玩家按 ESC 或其他关闭方式
- 这是 MC 1.13 "The Flattening" 更新后的新行为

---

## 7. 使用示例

### 基本使用

```python
import gym
import minerl

# 创建环境
env = gym.make('MineRLNavigateDense-v0')

# 重置
obs = env.reset()

# 执行动作
for i in range(100):
    action = env.action_space.no_op()
    
    if i == 1:
        # 打开 inventory
        action['inventory'] = 1
    else:
        # 保持关闭（或保持打开状态）
        action['inventory'] = 0
    
    obs, reward, done, info = env.step(action)
    env.render()
```

**结果**: Inventory GUI 在 Step 2 打开后一直保持打开 ✅

---

## 8. 测试脚本

### `scripts/test_minerl_inventory_simple.py`

简单测试脚本

```python
import gym
import minerl
import time

env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

print("测试 MineRL Inventory 行为")

for i in range(100):
    action = env.action_space.no_op()
    
    if i == 1:
        action['inventory'] = 1
        print(f"Step {i+1}: inventory=1 (打开)")
    else:
        action['inventory'] = 0
        # print(f"Step {i+1}: inventory=0")
    
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.1)
    
    if done:
        break

env.close()
```

### `scripts/debug_minerl_inventory_commands.py`

调试脚本（打印命令）

```python
import gym
import minerl

# Monkey-patch to_hero 方法
from minerl.herobraine.hero.handlers.agent.action import Action

original_to_hero = Action.to_hero

def patched_to_hero(self, x):
    cmd = original_to_hero(self, x)
    if self.command == "inventory":
        print(f"[MineRL to_hero] inventory: x={x}, cmd='{cmd}'")
    return cmd

Action.to_hero = patched_to_hero

# 运行测试
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()

for i in range(10):
    action = env.action_space.no_op()
    if i == 1:
        action['inventory'] = 1
    else:
        action['inventory'] = 0
    
    obs, reward, done, info = env.step(action)

env.close()
```

---

## 9. 与 MineDojo 的关键差异

| 方面 | MineRL | MineDojo |
|------|--------|----------|
| **Minecraft 版本** | 1.16.5 | 1.11.2 |
| **Action Space** | Dict (原生) | MultiDiscrete (需转换) |
| **Inventory 支持** | 原生支持 | 需要 patch |
| **GUI 行为** | 保持打开 ✅ | Toggle 开关 ❌ |
| **Wrapper 需求** | 不需要 | 需要状态管理 |
| **实现复杂度** | 简单 | 复杂 |

---

## 10. 关键文件路径总结

```
/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/

├── herobraine/
│   ├── hero/
│   │   ├── spaces.py                    # Action space 定义
│   │   └── handlers/
│   │       └── agent/
│   │           └── action.py            # Action.to_hero() 方法
│   └── env_specs/
│       └── human_controls.py            # 动作注册
│
├── env/
│   └── malmo/
│       └── instance/
│           └── instance.py              # 环境实例，step() 方法
│
└── MCP-Reborn/
    └── (JAR 文件 - mcprec-6.13.jar)
        └── com/microsoft/Malmo/
            ├── MissionHandlers/
            │   └── CommandForKey.java   # 命令处理
            └── Client/
                └── FakeKeyboard.java    # 键盘模拟
```

---

## 11. 数据流图

```
用户代码
  ↓
action = {"inventory": 1, "forward": 0, ...}
  ↓
env.step(action)
  ↓
instance._action_obj_to_xml(action)
  ↓
遍历 _action_handlers
  ↓
handler.to_hero(action["inventory"])  # 调用 Action.to_hero()
  ↓
生成 "inventory 1"
  ↓
发送到 Malmo XML
  ↓
Malmo 解析 XML
  ↓
CommandForKey.execute("inventory", "1")
  ↓
FakeKeyboard.press(inventoryKeyCode)
  ↓
KeyBinding.setKeyBindState(keyCode, true)
  ↓
Minecraft 1.16.5 接收 press 事件
  ↓
GUI 打开并保持打开 ✅
```

---

## 12. 调试建议

### 添加调试输出

1. **在 `action.py` 中打印命令**:
   ```python
   # minerl/herobraine/hero/handlers/agent/action.py
   def to_hero(self, x):
       cmd = ""
       verb = self.command
       adjective = str(x)
       cmd += "{} {}".format(verb, adjective)
       
       if verb == "inventory":  # 添加
           print(f"[MineRL] inventory command: '{cmd}'")
       
       return cmd
   ```

2. **在 `instance.py` 中打印 action**:
   ```python
   # minerl/env/malmo/instance/instance.py
   def step(self, action):
       if 'inventory' in action:  # 添加
           print(f"[MineRL] step action: inventory={action['inventory']}")
       
       # ... 原有代码 ...
   ```

### 观察要点

1. ✅ `inventory=0` 是否生成 `"inventory 0"` 命令？
2. ✅ Malmo 是否收到所有命令？
3. ✅ GUI 是否在第一次 `inventory=1` 后保持打开？
4. ✅ 连续发送 `inventory=0` 是否影响 GUI 状态？

---

## 总结

MineRL 的 inventory 实现非常简单直接：

1. ✅ **原生支持**: 不需要任何 patch 或 wrapper
2. ✅ **Dict Action Space**: 直接 `action['inventory'] = 1`
3. ✅ **无过滤**: 所有命令都发送到 Malmo
4. ✅ **MC 1.16.5**: GUI 行为正常（保持打开）
5. ✅ **与 STEVE-1 兼容**: 训练环境就是 MineRL

**这就是为什么 MineDojo 需要复杂的 wrapper 来模拟这个行为！**


