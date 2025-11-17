# Inventory Command Handler 调查指南

## 目标

定位 MCP-Reborn (MineRL) 和 Malmo (MineDojo) 在处理 `inventory` 命令时的差异，找出为什么：
- **MineRL (MC 1.16.5)**: inventory 打开后保持打开
- **MineDojo (MC 1.11.2)**: inventory 打开后自动关闭

---

## 调查路径

### 完整的消息处理链

```
Python 端发送 <inventory>1</inventory>
    ↓
Socket 接收 (TCP)
    ↓
XML 解析
    ↓
Command Handler 分发
    ↓
CommandForKey.java 处理
    ↓
FakeKeyboard.java (press/release)
    ↓
Minecraft KeyBinding 系统
    ↓
GUI 响应 (打开/关闭/Toggle)
```

---

## 第一步：提取和反编译 JAR 文件

### MCP-Reborn (MineRL)

```bash
# 1. 定位 JAR 文件
MCP_JAR="/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"

# 2. 创建工作目录
mkdir -p /tmp/mcp_reborn_decompile
cd /tmp/mcp_reborn_decompile

# 3. 解压 JAR (查看结构)
unzip -l "$MCP_JAR" | grep -i "command\|fake\|keyboard" > class_list.txt

# 4. 提取关键类
unzip "$MCP_JAR" "com/microsoft/Malmo/Client/*.class" -d ./
unzip "$MCP_JAR" "com/microsoft/Malmo/MissionHandlers/*.class" -d ./

# 5. 使用 javap 反编译 (查看方法签名)
find . -name "*.class" -type f | while read f; do
    echo "=== $f ===" >> disassembly.txt
    javap -c -private "$f" >> disassembly.txt 2>&1
    echo "" >> disassembly.txt
done

# 6. 查找关键方法
grep -A 20 "inventory\|press\|release\|onExecute" disassembly.txt > inventory_related.txt
```

### Malmo (MineDojo)

```bash
# 1. 定位 JAR 文件
MALMO_JAR="/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build/libs/MalmoMod-0.37.0-fat.jar"

# 2. 创建工作目录
mkdir -p /tmp/malmo_decompile
cd /tmp/malmo_decompile

# 3. 解压 JAR (查看结构)
unzip -l "$MALMO_JAR" | grep -i "command\|fake\|keyboard" > class_list.txt

# 4. 提取关键类
unzip "$MALMO_JAR" "com/microsoft/Malmo/Client/*.class" -d ./
unzip "$MALMO_JAR" "com/microsoft/Malmo/MissionHandlers/*.class" -d ./

# 5. 使用 javap 反编译
find . -name "*.class" -type f | while read f; do
    echo "=== $f ===" >> disassembly.txt
    javap -c -private "$f" >> disassembly.txt 2>&1
    echo "" >> disassembly.txt
done

# 6. 查找关键方法
grep -A 20 "inventory\|press\|release\|onExecute" disassembly.txt > inventory_related.txt
```

---

## 第二步：对比关键类的实现

### 需要重点对比的类

#### 1. CommandForKey.java

**关键方法**：
- `onExecute(String verb, String parameter)` - 接收并处理命令
- `executeCommand(String command)` - 执行具体命令

**需要检查**：
- `inventory 1` 是如何被转换为 `press` 调用的？
- `inventory 0` 是如何被转换为 `release` 调用的？
- 是否有任何特殊的 inventory 处理逻辑？

```bash
# 在反编译输出中查找
cd /tmp/mcp_reborn_decompile
grep -A 50 "class.*CommandForKey" disassembly.txt > CommandForKey_mcp.txt

cd /tmp/malmo_decompile
grep -A 50 "class.*CommandForKey" disassembly.txt > CommandForKey_malmo.txt

# 对比
diff /tmp/mcp_reborn_decompile/CommandForKey_mcp.txt \
     /tmp/malmo_decompile/CommandForKey_malmo.txt
```

#### 2. FakeKeyboard.java

**关键方法**：
- `press(int keyCode)` - 模拟按键按下
- `release(int keyCode)` - 模拟按键释放
- `keysDown` 集合的管理

**需要检查**：
- `press` 和 `release` 的实现是否相同？
- 调用的 MC API 是否不同？
- `keysDown` 集合的生命周期管理

```bash
cd /tmp/mcp_reborn_decompile
grep -A 50 "class.*FakeKeyboard" disassembly.txt > FakeKeyboard_mcp.txt

cd /tmp/malmo_decompile
grep -A 50 "class.*FakeKeyboard" disassembly.txt > FakeKeyboard_malmo.txt

# 对比
diff /tmp/mcp_reborn_decompile/FakeKeyboard_mcp.txt \
     /tmp/malmo_decompile/FakeKeyboard_malmo.txt
```

#### 3. ClientCommandHandler.java (MCP-Reborn) vs ClientStateMachine.java (Malmo)

**关键方法**：
- Socket 消息接收
- XML 解析
- 命令分发

```bash
cd /tmp/mcp_reborn_decompile
grep -A 50 "ClientCommandHandler\|handleMessage" disassembly.txt > ClientHandler_mcp.txt

cd /tmp/malmo_decompile
grep -A 50 "ClientStateMachine\|processMessage" disassembly.txt > ClientHandler_malmo.txt
```

---

## 第三步：添加调试日志

如果反编译不够清晰，可以通过添加日志来追踪实际执行路径。

### 方案 A: 使用 Java Agent 拦截方法调用

创建一个简单的 Java Agent 来打印方法调用：

```java
// LoggingAgent.java
import java.lang.instrument.Instrumentation;
import javassist.*;

public class LoggingAgent {
    public static void premain(String agentArgs, Instrumentation inst) {
        inst.addTransformer(new ClassFileTransformer() {
            public byte[] transform(...) {
                // 在 CommandForKey.onExecute 入口添加日志
                // 在 FakeKeyboard.press/release 入口添加日志
            }
        });
    }
}
```

### 方案 B: 修改 Python 端添加更详细的日志

在 Python 端捕获发送和接收的消息：

```python
# 在 minedojo_harvest.py 或 minerl 的相应位置

import sys

# Monkey patch socket send
original_send = socket.socket.send
def logged_send(self, data):
    msg = data.decode('utf-8', errors='ignore')
    if 'inventory' in msg:
        print(f"[Socket Send] {msg}", file=sys.stderr, flush=True)
    return original_send(self, data)
socket.socket.send = logged_send
```

### 方案 C: 分析 Minecraft 日志

MCP-Reborn 和 Malmo 可能会输出不同的日志。

```bash
# MineRL logs
ls -lht /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/logs/

# MineDojo logs
ls -lht /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/run/logs/
```

查找包含 "inventory", "command", "keyboard" 的日志条目。

---

## 第四步：创建最小复现案例

### 测试脚本：MineRL

```python
# test_minerl_inventory_minimal.py
import gym
import minerl
import time

env = gym.make('MineRLObtainDiamond-v0')
env.reset()

print("Step 1: Send inventory=1")
action = env.action_space.noop()
action['inventory'] = 1
env.step(action)
time.sleep(1)

for i in range(10):
    print(f"Step {i+2}: Send inventory=0")
    action = env.action_space.noop()
    action['inventory'] = 0
    env.step(action)
    time.sleep(0.5)

env.close()
```

### 测试脚本：MineDojo

```python
# test_minedojo_inventory_minimal.py
import minedojo
import time
import numpy as np

env = minedojo.make(task_id="harvest_wool_with_shears_and_sheep")
env.reset()

print("Step 1: Send inventory=1")
action = env.action_space.no_op()
action[5] = 8  # functional action: inventory
env.step(action)
time.sleep(1)

for i in range(10):
    print(f"Step {i+2}: Send inventory=0 (no-op)")
    action = env.action_space.no_op()
    action[5] = 0  # no-op
    env.step(action)
    time.sleep(0.5)

env.close()
```

### 观察要点

1. **游戏窗口**: inventory GUI 是否保持打开？
2. **Python 日志**: 发送的命令是什么？
3. **Java 日志**: 接收到的命令是什么？
4. **行为差异**: 在哪一步出现差异？

---

## 第五步：分析 Minecraft 源码（如果需要）

### MC 1.16.5 (MCP-Reborn)

```bash
# MCP-Reborn 可能包含反混淆的 MC 源码
find /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/ -name "*.java" | grep -i "keybind\|gui\|inventory"
```

### MC 1.11.2 (Malmo)

```bash
# Malmo 可能包含反混淆的 MC 源码
find /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/ -name "*.java" | grep -i "keybind\|gui\|inventory"
```

### 关键 MC 类

- **KeyBinding.java**: 键盘绑定管理
- **GuiScreen.java**: GUI 基类
- **GuiContainer.java**: 容器 GUI (包括 inventory)

查找这些类中关于 inventory 键的处理逻辑。

---

## 预期发现

基于已有的信息，我们预期会发现：

### 1. CommandForKey 层面

**可能相同**：
- `inventory 1` → `FakeKeyboard.press(inventoryKey)`
- `inventory 0` → `FakeKeyboard.release(inventoryKey)`

**可能不同**：
- 参数验证逻辑
- 错误处理
- 特殊情况处理

### 2. FakeKeyboard 层面

**可能相同**：
- `keysDown` 集合的概念
- press/release 的基本逻辑

**可能不同**：
- 调用的 MC API 方法名（MC 1.16.5 vs 1.11.2）
- 事件触发机制
- 错误处理

### 3. Minecraft GUI 层面（最关键！）

**MC 1.16.5 (MineRL)**：
```java
// 伪代码
public void onKeyPress(int keyCode) {
    if (keyCode == INVENTORY_KEY && currentScreen == null) {
        openInventoryScreen();
    }
}

public void onKeyRelease(int keyCode) {
    // 忽略 inventory 键的 release 事件
    if (keyCode == INVENTORY_KEY) {
        return; // ← 这就是为什么 MineRL 的 inventory 保持打开！
    }
}
```

**MC 1.11.2 (MineDojo)**：
```java
// 伪代码
public void onKeyPress(int keyCode) {
    if (keyCode == INVENTORY_KEY) {
        toggleInventoryScreen(); // ← Toggle 模式！
    }
}

public void onKeyRelease(int keyCode) {
    if (keyCode == INVENTORY_KEY) {
        closeInventoryScreen(); // ← 或者这里关闭
    }
}
```

---

## 实用脚本

### 一键执行完整调查

```bash
#!/bin/bash
# investigate_inventory_handlers.sh

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║         Inventory Command Handler 调查                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

# 1. 创建工作目录
WORK_DIR="/tmp/inventory_investigation"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 2. 定位 JAR 文件
MCP_JAR="/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minerl/MCP-Reborn/build/libs/mcprec-6.13.jar"
MALMO_JAR="/usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft/build/libs/MalmoMod-0.37.0-fat.jar"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 提取 MCP-Reborn 类"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p mcp_reborn
cd mcp_reborn

unzip -q "$MCP_JAR" "com/microsoft/Malmo/Client/*.class" 2>/dev/null || true
unzip -q "$MCP_JAR" "com/microsoft/Malmo/MissionHandlers/*.class" 2>/dev/null || true

echo "✓ 提取的类文件:"
find . -name "*.class" | head -20

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. 提取 Malmo 类"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$WORK_DIR"
mkdir -p malmo
cd malmo

unzip -q "$MALMO_JAR" "com/microsoft/Malmo/Client/*.class" 2>/dev/null || true
unzip -q "$MALMO_JAR" "com/microsoft/Malmo/MissionHandlers/*.class" 2>/dev/null || true

echo "✓ 提取的类文件:"
find . -name "*.class" | head -20

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 反编译关键类"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$WORK_DIR/mcp_reborn"
for class in $(find . -name "*FakeKeyboard*.class" -o -name "*CommandForKey*.class"); do
    echo "→ MCP-Reborn: $class"
    javap -c -private "$class" > "${class%.class}.txt" 2>/dev/null || echo "  (反编译失败)"
done

cd "$WORK_DIR/malmo"
for class in $(find . -name "*FakeKeyboard*.class" -o -name "*CommandForKey*.class"); do
    echo "→ Malmo: $class"
    javap -c -private "$class" > "${class%.class}.txt" 2>/dev/null || echo "  (反编译失败)"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. 对比结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$WORK_DIR"

# 查找配对的类文件
for mcp_txt in $(find mcp_reborn -name "*.txt"); do
    basename=$(basename "$mcp_txt")
    malmo_txt="malmo/${basename}"
    
    if [ -f "$malmo_txt" ]; then
        echo ""
        echo "对比: $basename"
        diff -u "$mcp_txt" "$malmo_txt" > "diff_${basename}" || true
        
        if [ -s "diff_${basename}" ]; then
            echo "  ✗ 发现差异 (见 diff_${basename})"
            head -30 "diff_${basename}"
        else
            echo "  ✓ 完全相同"
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "调查完成！结果保存在: $WORK_DIR"
echo ""
echo "下一步:"
echo "  1. 查看 diff_*.txt 文件了解具体差异"
echo "  2. 运行测试脚本观察实际行为"
echo "  3. 分析 Minecraft 日志"
echo "════════════════════════════════════════════════════════════════════════"
```

### 使用方法

```bash
chmod +x scripts/investigate_inventory_handlers.sh
./scripts/investigate_inventory_handlers.sh
```

---

## 总结

通过以上步骤，我们应该能够：

1. **确认 Java 层面的差异**（如果有）
   - CommandForKey 的实现差异
   - FakeKeyboard 的实现差异
   
2. **定位 Minecraft 层面的差异**（最关键）
   - MC 1.16.5 vs 1.11.2 的 GUI 处理逻辑
   - KeyBinding 的事件响应机制

3. **验证当前 wrapper 策略的有效性**
   - 打开一次后移除 `inventory` 键
   - 阻止发送 `inventory 0`

4. **为未来的优化提供依据**
   - 是否需要更复杂的状态管理？
   - 是否有更好的解决方案？

## 预期结果

最可能的发现：
- **Java Handler 层面**: 可能差异不大
- **Minecraft GUI 层面**: 这是差异的根源
- **解决方案**: Python wrapper 策略是正确的方向


