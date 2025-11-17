# 快速开始：添加 Malmo 命令日志

## TL;DR

```bash
cd /Users/nanzhang/aimc

# 1. 添加日志并重新编译（需要几分钟）
bash scripts/add_malmo_command_logs.sh

# 2. 运行测试（推荐方式）
python scripts/test_malmo_command_logs.py 2>&1 | grep --line-buffered -E '\[CommandForKey\]|\[FakeKeyboard\]|Step|━'
```

---

## 完整步骤

### 1. 添加日志到 Malmo Java 代码

```bash
bash scripts/add_malmo_command_logs.sh
```

这会：
- 备份原始 Java 文件
- 在 `CommandForKey.onExecute()` 添加日志
- 在 `FakeKeyboard.press()` 和 `release()` 添加日志
- 重新编译 Malmo（**需要 2-5 分钟**）

### 2. 运行测试

**推荐方式（只看关键日志）：**

```bash
python scripts/test_malmo_command_logs.py 2>&1 | grep --line-buffered -E '\[CommandForKey\]|\[FakeKeyboard\]|Step|━'
```

**查看所有日志（非常多！）：**

```bash
export MINEDOJO_DEBUG_LOG=1
python scripts/test_malmo_command_logs.py
```

**保存到文件：**

```bash
python scripts/test_malmo_command_logs.py 2>&1 | tee /tmp/malmo_full.log
grep -E '\[CommandForKey\]|\[FakeKeyboard\]' /tmp/malmo_full.log > /tmp/malmo_filtered.log
```

---

## 预期日志输出

### 第一次发送 action[5]=8

```
━━━ Step 1: action[5]=8 (inventory) ━━━
[CommandForKey] onExecute: verb=inventory, parameter=1
[FakeKeyboard] press() called with keyCode=18, keysDown.contains=false
```

### 第二次发送 action[5]=8（连续）

```
━━━ Step 2: action[5]=8 (inventory) ━━━
[CommandForKey] onExecute: verb=inventory, parameter=1
[FakeKeyboard] press() called with keyCode=18, keysDown.contains=true
```

注意：`keysDown.contains=true` 说明 E 键还在 `keysDown` 中，press 会被忽略。

### 发送 action[5]=0

```
━━━ Step 4: action[5]=0 (no-op) ━━━
[CommandForKey] onExecute: verb=inventory, parameter=0
[FakeKeyboard] release() called with keyCode=18, keysDown.contains=true
```

---

## 关键观察点

**验证"帧结束自动 release"假设：**

如果在 Step 1 和 Step 2 之间看到**额外的** `[FakeKeyboard] release()` 日志，说明 Malmo 在帧结束时自动调用了 release。

**正常情况（无自动 release）：**
```
Step 1: action[5]=8
  [CommandForKey] onExecute: verb=inventory, parameter=1
  [FakeKeyboard] press() called with keyCode=18, keysDown.contains=false
  
Step 2: action[5]=8
  [CommandForKey] onExecute: verb=inventory, parameter=1
  [FakeKeyboard] press() called with keyCode=18, keysDown.contains=true
```

**异常情况（有自动 release）：**
```
Step 1: action[5]=8
  [CommandForKey] onExecute: verb=inventory, parameter=1
  [FakeKeyboard] press() called with keyCode=18, keysDown.contains=false
  [FakeKeyboard] release() called with keyCode=18, keysDown.contains=true  ← 自动 release！
  
Step 2: action[5]=8
  [CommandForKey] onExecute: verb=inventory, parameter=1
  [FakeKeyboard] press() called with keyCode=18, keysDown.contains=true   ← 但仍然 true？
```

---

## 故障排除

### 编译失败

查看详细错误：

```bash
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
./gradlew clean shadowJar --stacktrace
```

### 看不到 Java 日志

确认 `MINEDOJO_DEBUG_LOG` 已设置：

```python
import os
print(os.environ.get('MINEDOJO_DEBUG_LOG'))  # 应该输出 '1'
```

### grep 无输出

可能是缓冲问题，添加 `--line-buffered` 选项：

```bash
python scripts/test_malmo_command_logs.py 2>&1 | grep --line-buffered -E '\[CommandForKey\]|\[FakeKeyboard\]'
```

---

## 清理/恢复

恢复原始代码：

```bash
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft

# 恢复备份
mv src/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java.backup \
   src/main/java/com/microsoft/Malmo/MissionHandlers/CommandForKey.java

mv src/main/java/com/microsoft/Malmo/Client/FakeKeyboard.java.backup \
   src/main/java/com/microsoft/Malmo/Client/FakeKeyboard.java

# 重新编译
./gradlew clean shadowJar
```

---

## 参考文档

- 完整指南：`docs/guides/MALMO_COMMAND_LOGGING_GUIDE.md`
- 假设分析：`docs/issues/MINEDOJO_INVENTORY_PRESS_RELEASE_HYPOTHESIS.md`


