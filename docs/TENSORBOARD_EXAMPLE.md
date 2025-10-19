# TensorBoard 示例说明

## 🎯 当数据出现后，你会看到什么

### 1. Scalars（标量）标签页

这是最重要的标签页，显示训练过程中的各种指标曲线：

#### Rollout（回合统计）
```
rollout/ep_len_mean      - 平均episode长度（越长说明存活时间越长）
rollout/ep_rew_mean      - 平均episode奖励（应该逐渐上升）
```

#### Train（训练指标）
```
train/approx_kl          - KL散度（策略变化幅度，应该保持较小 <0.01）
train/clip_fraction      - 被裁剪的样本比例
train/clip_range         - PPO裁剪范围
train/entropy_loss       - 熵损失（探索程度）
train/explained_variance - 价值函数拟合质量（接近1最好）
train/learning_rate      - 当前学习率
train/loss               - 总损失（应该逐渐下降）
train/policy_gradient_loss - 策略梯度损失
train/value_loss         - 价值函数损失
```

#### Time（时间统计）
```
time/fps                 - 每秒处理的帧数
time/iterations          - 迭代次数
time/time_elapsed        - 已用时间（秒）
time/total_timesteps     - 总训练步数
```

### 2. 健康的训练曲线特征

✅ **好的迹象：**
- `ep_rew_mean` 整体趋势上升（即使有波动）
- `loss` 逐渐下降并趋于稳定
- `explained_variance` 逐渐接近 1
- `approx_kl` 保持在 0.01 以下
- `fps` 相对稳定

⚠️ **需要注意：**
- `ep_rew_mean` 长期不增长 → 学习率可能太高或太低
- `loss` 剧烈震荡 → 学习率可能太高
- `explained_variance` 接近 0 → 价值函数拟合不好
- `approx_kl` 持续很大 → 策略更新太激进，考虑降低学习率

### 3. harvest_milk 任务的预期表现

#### 初始阶段（0-50K steps）
- `ep_rew_mean`: 0-5（随机探索）
- `ep_len_mean`: 100-500（经常死亡或超时）
- 智能体在随机移动，尝试各种动作

#### 学习阶段（50K-200K steps）
- `ep_rew_mean`: 5-20（开始找到牛）
- `ep_len_mean`: 500-1000（学会基本生存）
- 智能体开始理解任务目标

#### 熟练阶段（200K+ steps）
- `ep_rew_mean`: 20-50+（高效收集牛奶）
- `ep_len_mean`: 1000+（稳定完成任务）
- 智能体能稳定地寻找牛并挤奶

### 4. 如何使用 TensorBoard

#### 平滑曲线
```
左侧面板 → Smoothing → 调整为 0.6-0.8
```
减少噪声，更清晰地看到趋势

#### 对比多次训练
```
勾选/取消勾选不同的 runs
```
比较不同超参数的效果

#### 放大查看
```
鼠标拖动选择区域 → 放大
双击 → 恢复
```

#### 下载数据
```
点击图表下方的下载按钮
导出为 CSV 或 SVG
```

### 5. 实际示例

假设训练10万步后，你可能看到：

```
rollout/ep_rew_mean:  从 0 增长到 15-20
rollout/ep_len_mean:  从 200 增长到 600-800
train/loss:           从 0.8 下降到 0.3-0.4
train/explained_variance: 从 0.2 增长到 0.7-0.9
time/fps:             约 2-5 fps（MineDojo较慢）
```

### 6. 为什么现在看不到数据？

**PPO的工作机制：**
1. 收集 `n_steps` (2048) 个样本
2. 使用这些样本训练 `n_epochs` (10) 轮
3. 记录指标到 TensorBoard
4. 重复步骤 1-3

**时间估算：**
- MineDojo 每步: ~0.5-1秒
- 2048 步: ~20-30 分钟
- 所以第一批数据需要等 20-30 分钟

### 7. 加速数据显示的方法

修改 `config/training_config.yaml`：
```yaml
ppo:
  n_steps: 256    # 从 2048 改为 256（更频繁更新）
  batch_size: 32  # 对应调整
```

或者命令行：
```bash
python src/training/train_harvest_paper.py \
    --n-steps 256 \
    --batch-size 32 \
    --total-timesteps 10000
```

这样约 3-5 分钟就能看到第一批数据。

### 8. TensorBoard 访问地址

```
http://localhost:6006
```

刷新频率：自动每 30 秒刷新一次

---

💡 **提示**：第一次看到"No dashboards are active"是正常的，耐心等待训练完成第一个 rollout 周期即可。

