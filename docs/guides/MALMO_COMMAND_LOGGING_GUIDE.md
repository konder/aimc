# Malmo CommandHandler 日志调试指南

## 目标

添加详细日志到 Malmo 的 Java 代码，追踪 `inventory` 命令从 Python 到 Minecraft 的完整处理流程。

---

## 步骤 1：添加日志并重新编译

```bash
cd /Users/nanzhang/aimc
bash scripts/add_malmo_command_logs.sh
```

**这个脚本会：**
1. 备份原始的 Java 文件
2. 在 `CommandForKey.java` 的 `onExecute()` 方法中添加日志
3. 在 `FakeKeyboard.java` 的 `press()` 和 `release()` 方法中添加日志
4. 重新编译 Malmo（需要几分钟）

**添加的日志：**

```java
// CommandForKey.java
public void onExecute(String verb, String parameter) {
    System.out.println("[CommandForKey] onExecute: verb=" + verb + ", parameter=" + parameter);
    // ... 原有代码 ...
}

// FakeKeyboard.java
public static void press(int keyCode) {
    System.out.println("[FakeKeyboard] press() called with keyCode=" + keyCode + ", keysDown.contains=" + keysDown.contains(keyCode));
    // ... 原有代码 ...
}

public static void release(int keyCode) {
    System.out.println("[FakeKeyboard] release() called with keyCode=" + keyCode + ", keysDown.contains=" + keysDown.contains(keyCode));
    // ... 原有代码 ...
}
```

---

## 步骤 2：启用 Java 日志输出

**重要！** MineDojo 默认**不显示** Minecraft 的 Java 日志！

需要设置环境变量：

```bash
export MINEDOJO_DEBUG_LOG=1
```

或者在 Python 脚本中设置（`test_malmo_command_logs.py` 已经包含）：

```python
import os
os.environ['MINEDOJO_DEBUG_LOG'] = '1'
```

---

## 步骤 3：运行测试并观察日志

### 方法 A：直接运行测试脚本

```bash
cd /Users/nanzhang/aimc
conda activate minedojo-x86
python scripts/test_malmo_command_logs.py 2>&1 | tee /tmp/malmo_command_logs.txt
```

**注意**：这会产生**大量日志**（Minecraft 的所有输出）！

这会：
- 运行一个简单的测试序列
- 捕捉所有输出到 `/tmp/malmo_command_logs.txt`
- 实时显示在终端

### 方法 B：使用 grep 过滤日志（**强烈推荐**）

```bash
python scripts/test_malmo_command_logs.py 2>&1 | grep --line-buffered -E '\[CommandForKey\]|\[FakeKeyboard\]|Step|━'
```

只显示我们添加的日志行和测试步骤，**过滤掉 Minecraft 的大量输出**。

`--line-buffered` 选项确保日志实时显示，不会等到缓冲区满才输出。

### 方法 C：查看 Minecraft 日志文件

```bash
# 找到最新的 MC 日志
ls -lt logs/mc_*.log | head -1

# 实时查看
tail -f logs/mc_XXXXX.log | grep -E '\[CommandForKey\]|\[FakeKeyboard\]'
```

---

## 步骤 3：分析日志

### 预期的日志输出

**场景 1：第一次发送 action[5]=8**

```
[CommandForKey] onExecute: verb=inventory, parameter=1
[FakeKeyboard] press() called with keyCode=18, keysDown.contains=false
[FakeKeyboard] press(18) - sending press event
```

- `keyCode=18` 是 'E' 键的代码（可能不同）
- `keysDown.contains=false` 说明这是第一次 press
- 发送了 press 事件

**场景 2：连续第二次发送 action[5]=8**

```
[CommandForKey] onExecute: verb=inventory, parameter=1
[FakeKeyboard] press() called with keyCode=18, keysDown.contains=true
```

- `keysDown.contains=true` 说明 E 键还在 `keysDown` 中
- **没有 "sending press event" 日志** → press 被忽略

**场景 3：发送 action[5]=0**

```
[CommandForKey] onExecute: verb=inventory, parameter=0
[FakeKeyboard] release() called with keyCode=18, keysDown.contains=true
[FakeKeyboard] release(18) - sending release event
```

- 发送了 release 事件
- `keysDown` 中的 E 键被移除

### 验证"帧结束自动 release"假设

**如果假设正确，应该看到：**

```
Step 1: action[5]=8
  [CommandForKey] onExecute: verb=inventory, parameter=1
  [FakeKeyboard] press() called with keyCode=18, keysDown.contains=false
  [FakeKeyboard] press(18) - sending press event
  
  ← 帧结束时，可能有额外的日志：
  [FakeKeyboard] release() called with keyCode=18, keysDown.contains=true
  [FakeKeyboard] release(18) - sending release event (auto-release)
```

**关键标志：**
- 在没有显式发送 `inventory=0` 的情况下
- 如果看到 `[FakeKeyboard] release()` 日志
- 说明 Malmo 自动调用了 `release()`

---

## 步骤 4：保存和分析

### 保存完整日志

```bash
python scripts/test_malmo_command_logs.py 2>&1 > /tmp/full_test_log.txt
```

### 提取关键部分

```bash
# 只看 Malmo 日志
grep -E '\[CommandForKey\]|\[FakeKeyboard\]' /tmp/full_test_log.txt > /tmp/malmo_only.txt

# 统计 press 和 release 次数
grep '\[FakeKeyboard\] press()' /tmp/malmo_only.txt | wc -l
grep '\[FakeKeyboard\] release()' /tmp/malmo_only.txt | wc -l
```

---

## 步骤 5：恢复原始代码（可选）

如果需要恢复到未修改的状态：

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

## 预期发现

根据你的观察（连续 3 次 action[5]=8 只显示 1 次 GUI），我们预期会发现：

1. **第一次 action[5]=8**:
   - ✅ CommandForKey 接收 inventory=1
   - ✅ FakeKeyboard.press() 被调用
   - ✅ 发送 press 事件
   - ❓ **可能：帧结束时自动调用 release()**

2. **第二次 action[5]=8**:
   - ✅ CommandForKey 接收 inventory=1
   - ✅ FakeKeyboard.press() 被调用
   - ❌ **但 keysDown.contains=true → 被忽略**

3. **第三次 action[5]=8**:
   - 同第二次

这将证实：
- Malmo 在每帧结束时**不会清空 `keysDown`**
- 但可能会**自动发送 release 事件**
- 导致 GUI 关闭，但 `keysDown` 仍保留 E 键

---

## 排查清单

- [ ] 编译成功
- [ ] 测试脚本运行成功
- [ ] 看到 `[CommandForKey]` 日志
- [ ] 看到 `[FakeKeyboard]` 日志
- [ ] 观察 `keysDown.contains` 的值变化
- [ ] 确认 press 被忽略的时机
- [ ] 寻找"额外的 release 事件"

---

## 故障排除

### 编译失败

```bash
# 查看错误日志
cd /usr/local/Caskroom/miniforge/base/envs/minedojo-x86/lib/python3.9/site-packages/minedojo/sim/Malmo/Minecraft
./gradlew clean shadowJar --stacktrace
```

常见问题：
- Java 语法错误：检查日志插入的位置
- 找不到类：确认文件路径正确

### 看不到日志

可能原因：
1. Java 的 `System.out.println` 被重定向
2. 日志级别设置过滤了 stdout

解决：
- 使用 Java logger 替代 System.out
- 或者将日志写入文件

### 测试脚本失败

```bash
# 检查 MineDojo 环境
python -c "import minedojo; print(minedojo.__version__)"

# 检查 Minecraft 是否能启动
python -c "import minedojo; env = minedojo.make('harvest'); env.reset(); env.close()"
```

---

## 下一步

根据日志分析结果：

1. **如果发现"帧结束自动 release"**:
   - 修改 Malmo 代码，禁用 inventory 键的自动 release
   - 参考 `docs/issues/MINEDOJO_INVENTORY_PRESS_RELEASE_HYPOTHESIS.md`

2. **如果没有自动 release**:
   - 需要重新分析为什么 GUI 关闭
   - 可能是 MC 1.11.2 的 GUI 行为

3. **如果 keysDown 被自动清空**:
   - 修改 wrapper 策略
   - 每帧都发送 inventory=1

