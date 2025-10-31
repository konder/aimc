# STEVE-1 渲染功能使用指南

## 功能说明

为 STEVE-1 的 `run_agent` 添加了 `--render` 参数，可以控制是否在运行时显示 Minecraft 游戏窗口。

## 使用方法

### 1. 命令行直接使用

```bash
# 不显示窗口（默认，速度快）
python run_agent/run_agent.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --gameplay_length 1000

# 显示游戏窗口（启用渲染）
python run_agent/run_agent.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --gameplay_length 1000 \
    --render
```

### 2. 在脚本中使用

所有脚本都已更新，支持通过修改 `RENDER_FLAG` 变量来启用/禁用渲染。

#### `1_gen_paper_videos.sh` - 生成论文演示视频

**启用渲染**:
```bash
# 编辑脚本，取消注释以下行：
RENDER_FLAG="--render"  # 取消注释这行以启用渲染
```

**禁用渲染**（默认）:
```bash
# 保持默认状态：
RENDER_FLAG=""
```

#### `2_gen_vid_for_text_prompt.sh` - 自定义文本提示

同样的方式，修改脚本中的 `RENDER_FLAG` 变量。

## 性能影响

### 不渲染（默认）
```
速度: ~3.5-4.0 it/s
3000 步耗时: ~14-15 分钟
CPU 占用: 高（主要在推理）
内存占用: 中等
```

### 渲染模式
```
速度: ~2.5-3.0 it/s（降低约 25-30%）
3000 步耗时: ~18-20 分钟
CPU 占用: 很高（推理 + 渲染）
内存占用: 较高
```

## 使用场景

### ❌ 不建议使用渲染的场景

1. **批量生成视频** - 论文演示的 13 个任务
   ```bash
   # 使用默认不渲染模式，速度快
   bash src/training/steve1/1_gen_paper_videos.sh
   ```

2. **无头服务器** - 没有显示器的 Linux 服务器
   ```bash
   # 服务器环境无法渲染，保持默认
   ```

3. **长时间运行** - gameplay_length > 5000
   ```bash
   # 长时间渲染窗口意义不大，且影响性能
   ```

### ✅ 建议使用渲染的场景

1. **调试和开发** - 查看 Agent 实时行为
   ```bash
   # 启用渲染，观察 Agent 是否按预期工作
   bash src/training/steve1/2_gen_vid_for_text_prompt.sh
   # (修改 RENDER_FLAG="--render")
   ```

2. **演示展示** - 给他人展示 STEVE-1 效果
   ```bash
   # 实时显示游戏画面
   python run_agent/run_agent.py \
       --custom_text_prompt "dig dirt" \
       --gameplay_length 500 \
       --render
   ```

3. **短视频测试** - 快速验证功能
   ```bash
   # 短视频（500-1000步）时渲染开销可接受
   python run_agent/run_agent.py \
       --gameplay_length 500 \
       --render
   ```

## 技术细节

### 实现方式

在游戏循环中添加了条件渲染：

```python
def run_agent(prompt_embed, gameplay_length, save_video_filepath,
              in_model, in_weights, seed, cond_scale, render=False):
    # ... 初始化代码 ...
    
    for _ in tqdm(range(gameplay_length)):
        # 1. Agent 决策
        minerl_action = agent.get_action(obs, prompt_embed)
        
        # 2. 环境更新
        obs, _, _, _ = env.step(minerl_action)
        
        # 3. 保存帧到视频
        frame = obs['pov']
        gameplay_frames.append(frame)
        
        # 4. 条件渲染（如果启用）
        if render:
            env.render()  # 显示游戏窗口
    
    # 5. 保存视频文件
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS)
```

### 渲染开销

渲染窗口的性能开销来自：

1. **图形渲染** - OpenGL/软件渲染 Minecraft 场景
2. **窗口管理** - 操作系统窗口更新
3. **帧同步** - 等待渲染完成
4. **显存/内存** - 额外的帧缓冲区

### 与视频保存的区别

- **`env.render()`**: 实时显示窗口（可选）
- **`gameplay_frames.append(frame)`**: 保存帧数据到内存（始终执行）
- **`save_frames_as_video()`**: 最后保存为 MP4 文件（始终执行）

**重要**: 即使不启用 `--render`，最终的 MP4 视频文件仍会正常生成！

## 常见问题

### Q: 为什么在 Mac 上渲染很慢？
A: Mac 使用 CPU 推理（没有 NVIDIA GPU），再加上渲染开销，速度会明显降低。建议：
   - 短视频测试时使用渲染
   - 长视频生成时禁用渲染

### Q: 渲染窗口卡顿或无响应？
A: 这是正常的。游戏循环在 Python 主线程中运行，渲染窗口可能无法响应交互。可以：
   - 不要点击窗口
   - 让程序自然运行完成
   - 或使用 Ctrl+C 中断

### Q: 无法显示窗口（报错 DISPLAY 未设置）？
A: 在无头服务器或 SSH 连接时会出现此问题。解决方案：
   ```bash
   # 不要使用 --render，或设置虚拟显示
   export DISPLAY=:0
   ```

### Q: 不渲染是否影响视频生成？
A: 完全不影响！无论是否渲染窗口，视频文件都会正常保存。`--render` 只是控制是否实时显示。

## 总结

| 特性 | 不渲染（默认） | 渲染模式 |
|------|---------------|---------|
| 速度 | ✅ 快 (~4 it/s) | ⚠️ 较慢 (~3 it/s) |
| CPU | ✅ 中等 | ❌ 很高 |
| 可视化 | ❌ 无 | ✅ 实时窗口 |
| 视频生成 | ✅ 正常 | ✅ 正常 |
| 适用场景 | 批量生成、服务器 | 调试、演示 |

**推荐**: 默认禁用渲染，仅在需要时启用。

## 示例命令

```bash
# 1. 快速测试（启用渲染，短视频）
python run_agent/run_agent.py \
    --custom_text_prompt "explore cave" \
    --gameplay_length 500 \
    --render

# 2. 生产运行（禁用渲染，长视频）
python run_agent/run_agent.py \
    --gameplay_length 3000

# 3. 批量生成（禁用渲染）
bash src/training/steve1/1_gen_paper_videos.sh
# 保持 RENDER_FLAG="" 不变

# 4. 演示展示（启用渲染）
# 编辑 2_gen_vid_for_text_prompt.sh，设置：
# RENDER_FLAG="--render"
bash src/training/steve1/2_gen_vid_for_text_prompt.sh
```

## 相关文档

- [STEVE-1 快速启动](STEVE1_QUICKSTART.md)
- [STEVE-1 集成修复](../summaries/STEVE1_INTEGRATION_FIXES.md)
- [STEVE-1 评估指南](STEVE1_EVALUATION_GUIDE.md)

