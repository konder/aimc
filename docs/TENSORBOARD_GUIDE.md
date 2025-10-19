# TensorBoard 使用指南

## 🚀 启动TensorBoard

```bash
cd /Users/nanzhang/aimc
conda activate minedojo
tensorboard --logdir=logs/tensorboard --port=6006
```

## 📊 访问TensorBoard

在浏览器中打开：
```
http://localhost:6006
```

## 📈 可以看到的训练指标

### 1. Rollout（回合统计）
- **ep_len_mean** - 平均episode长度
- **ep_rew_mean** - 平均episode奖励

### 2. Train（训练损失）
- **loss** - 总损失
- **policy_loss** - 策略损失（Actor）
- **value_loss** - 价值损失（Critic）
- **entropy_loss** - 熵损失（探索程度）
- **explained_variance** - 价值函数拟合质量
- **approx_kl** - KL散度（策略更新幅度）
- **clip_fraction** - 被裁剪的样本比例

### 3. Time（时间统计）
- **fps** - 每秒帧数
- **iterations** - 迭代次数
- **time_elapsed** - 已用时间
- **total_timesteps** - 总训练步数

## 🔄 刷新数据

- **自动刷新**：TensorBoard每30秒自动更新
- **手动刷新**：点击右上角的刷新按钮 ⟳
- **调整刷新间隔**：Settings → Reload data → 设置秒数

## 📊 查看技巧

### 1. 平滑曲线
左侧面板 → Smoothing → 调整滑块（建议0.6-0.8）

### 2. 对比不同训练
- 勾选/取消勾选不同的runs
- 查看多次训练的对比

### 3. 下载数据
- 点击图表左下角的下载按钮
- 导出为CSV或JSON

### 4. 全屏查看
- 点击图表右上角的展开按钮
- 更清晰地查看趋势

## 🎯 关键指标解读

### 好的训练迹象
✅ **ep_rew_mean** 逐渐上升  
✅ **loss** 逐渐下降并稳定  
✅ **explained_variance** 接近1  
✅ **approx_kl** 保持较小（<0.01）

### 需要注意的情况
⚠️ **ep_rew_mean** 震荡剧烈 - 考虑降低学习率  
⚠️ **loss** 不下降 - 可能需要调整超参数  
⚠️ **approx_kl** 过大 - 策略更新太激进

## 🛠️ 常用命令

```bash
# 查看训练日志
tail -f logs/training/training_*.log

# 停止训练
pkill -f "train_harvest_paper"

# 停止TensorBoard
pkill -f "tensorboard"

# 清理旧日志
rm -rf logs/tensorboard/ppo_*
```

## 🌐 远程访问

如果需要从其他机器访问：
```bash
tensorboard --logdir=logs/tensorboard --port=6006 --bind_all --host=0.0.0.0
```

然后在浏览器访问：
```
http://<你的IP>:6006
```

## 📱 使用TensorBoard.dev（云端分享）

```bash
tensorboard dev upload --logdir logs/tensorboard --name "MineDojo Training"
```

会生成一个公开链接，可以分享给他人查看。

---

💡 **提示**：训练刚开始时，TensorBoard可能显示"No dashboards are active"，等待1-2分钟让数据写入即可。

