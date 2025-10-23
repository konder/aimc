# 国内云GPU平台部署指南

> **目标**: 将 AIMC 项目部署到云GPU平台进行训练  
> **适用场景**: DAgger训练、PPO训练、长时间训练任务  
> **更新时间**: 2025-10-23

---

## 📊 国内云GPU平台对比

### 1. AutoDL（推荐🌟🌟🌟🌟🌟）

**官网**: https://www.autodl.com/

#### 优势
- ✅ **价格最低**: RTX 4090 约 ¥2.5-3.5/小时
- ✅ **按秒计费**: 暂停即停止计费，非常灵活
- ✅ **无需实名**: 注册即用
- ✅ **镜像丰富**: 官方提供 PyTorch、TensorFlow 等镜像
- ✅ **数据盘独立**: 数据盘免费保存，不计费
- ✅ **Jupyter支持**: 内置 JupyterLab，方便开发
- ✅ **SSH支持**: 支持SSH连接，可以用VSCode远程开发

#### 价格参考（2025年10月）
```
RTX 3090 (24GB):  ¥2.0-2.8/小时
RTX 4090 (24GB):  ¥2.5-3.5/小时
A100 (40GB):      ¥6.0-8.0/小时
A100 (80GB):      ¥10.0-12.0/小时
```

#### 适用场景
- ✅ DAgger训练（需要频繁启停）
- ✅ PPO短期训练（<24小时）
- ✅ 调试和实验
- ✅ 预算有限的个人开发者

---

### 2. 恒源云

**官网**: https://www.gpushare.com/

#### 优势
- ✅ **价格适中**: RTX 4090 约 ¥3-4/小时
- ✅ **按分钟计费**: 比AutoDL稍差，但仍然灵活
- ✅ **稳定性好**: 机器可用率高
- ✅ **客服响应快**: 有问题可以快速解决
- ✅ **支持容器**: Docker镜像支持

#### 价格参考
```
RTX 3090 (24GB):  ¥2.5-3.2/小时
RTX 4090 (24GB):  ¥3.0-4.0/小时
A100 (40GB):      ¥7.0-9.0/小时
```

#### 适用场景
- ✅ 中等规模训练（24-72小时）
- ✅ 需要稳定性的项目
- ✅ 企业用户

---

### 3. 智星云

**官网**: https://www.ai-galaxy.cn/

#### 优势
- ✅ **学术优惠**: 学生和教师有折扣
- ✅ **高端卡多**: A100、H100 等高端卡资源充足
- ✅ **技术支持**: 提供AI训练优化建议

#### 价格参考
```
RTX 4090 (24GB):  ¥3.5-4.5/小时
A100 (40GB):      ¥8.0-10.0/小时
A100 (80GB):      ¥12.0-15.0/小时
```

#### 适用场景
- ✅ 学术研究
- ✅ 大规模训练
- ✅ 需要高端卡（A100/H100）

---

### 4. 阿里云PAI

**官网**: https://pai.aliyun.com/

#### 优势
- ✅ **企业级稳定性**: 99.9% SLA
- ✅ **安全性高**: 数据隔离、权限管理
- ✅ **生态完善**: 与OSS、NAS等服务集成
- ✅ **技术支持**: 7x24小时客服

#### 劣势
- ❌ **价格较高**: RTX 4090 约 ¥5-7/小时
- ❌ **按小时计费**: 不够灵活
- ❌ **需要实名**: 企业认证

#### 适用场景
- ✅ 企业级项目
- ✅ 需要高安全性
- ✅ 预算充足

---

### 5. 腾讯云TI

**官网**: https://cloud.tencent.com/product/ti

#### 优势
- ✅ **生态好**: 与腾讯云其他服务集成
- ✅ **稳定性**: 企业级SLA

#### 劣势
- ❌ **价格高**: 类似阿里云
- ❌ **按小时计费**

---

## 🎯 推荐方案

### 场景1: DAgger训练（需要频繁人工标注）
**推荐**: AutoDL

**理由**:
- DAgger需要频繁 暂停 → 人工标注 → 继续训练
- AutoDL **按秒计费**，暂停不计费
- 可以开着实例，人工标注时暂停，标注完继续

**预估成本** (harvest_1_log):
```
录制10轮: 2小时  → ¥0 (本地录制)
BC训练:   1小时  → ¥3 (RTX 4090)
DAgger迭代5轮:
  - 采集状态: 3小时 → ¥9
  - 人工标注: 5小时 → ¥0 (暂停不计费)
  - 训练更新: 5小时 → ¥15
总计: ¥27
```

---

### 场景2: PPO纯强化学习训练
**推荐**: AutoDL 或 恒源云

**理由**:
- PPO训练是连续运行，不需要频繁暂停
- 优先选便宜的

**预估成本** (harvest_1_log, 500K steps):
```
AutoDL RTX 4090:
  - 训练时间: 约 20-30 小时
  - 成本: ¥2.8/h × 25h = ¥70
  
恒源云 RTX 4090:
  - 成本: ¥3.5/h × 25h = ¥87.5
```

---

### 场景3: 大规模训练（>100万steps）
**推荐**: 智星云 A100 或 AutoDL A100

**理由**:
- A100 训练速度是 RTX 4090 的 2-3倍
- 总成本可能反而更低

**预估成本** (harvest_1_log, 2M steps):
```
RTX 4090:
  - 训练时间: 约 60 小时
  - 成本: ¥2.8/h × 60h = ¥168
  
A100 (40GB):
  - 训练时间: 约 25 小时
  - 成本: ¥7/h × 25h = ¥175
  
A100 更快完成，总成本接近，但时间省一半！
```

---

## 🚀 AutoDL 快速上手（推荐）

### 步骤1: 注册账号

1. 访问 https://www.autodl.com/
2. 注册账号（微信/手机号）
3. 充值 ¥50-100（首次充值通常有优惠）

---

### 步骤2: 创建实例

1. **选择GPU**:
   - 点击 "租用实例"
   - 筛选: `RTX 4090` 或 `RTX 3090`
   - 选择价格最低的（通常 ¥2.5-3.5/小时）

2. **选择镜像**:
   ```
   推荐镜像: PyTorch 2.0
   或: 自定义镜像（Miniconda3 + CUDA 11.8）
   ```

3. **配置存储**:
   ```
   系统盘: 30GB (免费)
   数据盘: 100GB (免费保存，推荐！)
   ```

4. **点击创建**

---

### 步骤3: 连接实例

#### 方式1: JupyterLab（推荐新手）
```
1. 实例启动后，点击 "打开JupyterLab"
2. 在浏览器中打开 JupyterLab
3. 打开 Terminal
```

#### 方式2: SSH（推荐熟练用户）
```bash
# AutoDL会显示SSH连接命令，类似：
ssh -p 12345 root@connect.autodl.com

# 或使用VSCode Remote SSH:
# 1. 安装 "Remote - SSH" 插件
# 2. 添加SSH配置：
#    Host autodl-gpu
#      HostName connect.autodl.com
#      User root
#      Port 12345
# 3. 连接到 autodl-gpu
```

---

### 步骤4: 部署项目

#### 4.1 上传代码

**方法A: Git克隆（推荐）**
```bash
cd /root/autodl-tmp  # AutoDL的工作目录
git clone https://github.com/YOUR_USERNAME/aimc.git
cd aimc
```

**方法B: 上传压缩包**
```bash
# 本地打包
cd /Users/nanzhang/aimc
tar -czf aimc.tar.gz --exclude=logs --exclude=checkpoints .

# 上传到AutoDL（通过JupyterLab的Upload按钮）
# 然后解压
cd /root/autodl-tmp
tar -xzf aimc.tar.gz -C aimc
```

---

#### 4.2 安装依赖

```bash
cd /root/autodl-tmp/aimc

# 创建conda环境
conda create -n minedojo python=3.9 -y
conda activate minedojo

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装MineDojo（x86架构）
pip install minedojo -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
python -c "import minedojo; import torch; print(f'MineDojo: {minedojo.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

#### 4.3 配置环境变量

```bash
# 创建 .env 文件（如果需要）
cat > .env << 'EOF'
JAVA_OPTS="-Djava.awt.headless=true"
EOF

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0
```

---

#### 4.4 下载数据（如果需要）

```bash
# MineCLIP模型（如果之前已下载）
# 方法1: 从本地上传
# 方法2: 从云端下载
cd /root/autodl-tmp/aimc/data
# 上传 mineclip/ 和 clip_tokenizer/ 目录
```

---

### 步骤5: 运行训练

#### DAgger训练
```bash
cd /root/autodl-tmp/aimc

# 注意：DAgger的录制步骤需要在本地完成
# 在云端只运行训练部分

# 1. 上传录制好的数据
# 将本地 data/expert_demos/harvest_1_log/ 上传到云端

# 2. BC训练
conda activate minedojo
python src/training/train_bc.py \
    --data-dir data/expert_demos/harvest_1_log \
    --output-path checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --epochs 100 \
    --batch-size 32 \
    --device cuda

# 3. 评估
python tools/dagger/evaluate_policy.py \
    --model checkpoints/dagger/harvest_1_log/bc_baseline.zip \
    --episodes 10 \
    --device cuda

# 4. DAgger迭代训练
# (采集状态 → 下载到本地 → 标注 → 上传 → 训练)
```

#### PPO训练
```bash
cd /root/autodl-tmp/aimc
conda activate minedojo

# 使用配置文件训练
bash scripts/train_get_wood.sh \
    --total-steps 500000 \
    --device cuda \
    --headless

# 或使用Python直接运行
python src/training/train_get_wood.py \
    --total-timesteps 500000 \
    --device cuda \
    --save-freq 10000 \
    --checkpoint-dir checkpoints/ppo/harvest_1_log
```

---

### 步骤6: 监控训练

#### 方法1: TensorBoard（推荐）
```bash
# 在AutoDL实例上启动TensorBoard
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006

# AutoDL会自动转发端口，点击实例页面的 "自定义服务" 查看地址
# 或在本地浏览器访问显示的URL
```

#### 方法2: 日志文件
```bash
# 实时查看训练日志
tail -f logs/training/training_*.log
```

#### 方法3: 本地同步（VSCode Remote）
```
使用VSCode Remote SSH连接，可以实时查看文件和日志
```

---

### 步骤7: 下载结果

#### 方法1: JupyterLab下载
```
1. 在JupyterLab文件浏览器中
2. 右键点击 checkpoints/ 目录
3. 选择 "Download"
```

#### 方法2: SCP下载（推荐大文件）
```bash
# 在本地执行
scp -P 12345 -r root@connect.autodl.com:/root/autodl-tmp/aimc/checkpoints ./

# 或使用rsync（断点续传）
rsync -avz -e "ssh -p 12345" root@connect.autodl.com:/root/autodl-tmp/aimc/checkpoints ./
```

---

### 步骤8: 暂停/关闭实例

```
暂停实例（推荐）:
- 保留数据盘
- 不计费
- 随时恢复

关闭实例:
- 完全停止
- 系统盘会被删除（数据盘保留）
- 下次需要重新创建
```

---

## 💡 AutoDL 使用技巧

### 技巧1: 使用数据盘保存重要数据
```bash
# 数据盘路径: /root/autodl-tmp
# 系统盘路径: /root

# ✅ 正确做法：所有数据放在数据盘
cd /root/autodl-tmp
git clone ...

# ❌ 错误做法：数据放在系统盘（关机会丢失）
cd /root
git clone ...
```

---

### 技巧2: 使用 tmux 保持训练
```bash
# 安装tmux（通常已预装）
apt-get install tmux -y

# 创建会话
tmux new -s train

# 运行训练
bash scripts/train_get_wood.sh --total-steps 500000

# 断开会话（Ctrl+B, 然后按 D）
# 训练会继续在后台运行

# 重新连接
tmux attach -t train
```

---

### 技巧3: 自动保存检查点
```bash
# 定期上传到云存储（避免实例意外关闭）
# 使用阿里云OSS或腾讯云COS

# 安装ossutil
wget http://gosspublic.alicdn.com/ossutil/1.7.14/ossutil64
chmod 755 ossutil64

# 配置OSS
./ossutil64 config

# 定时上传检查点
while true; do
    ./ossutil64 cp -r checkpoints/ oss://your-bucket/aimc/checkpoints/
    sleep 3600  # 每小时上传一次
done &
```

---

### 技巧4: 加速pip安装
```bash
# 使用清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或配置永久源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📊 成本优化建议

### 1. 选择合适的GPU
```
任务类型              推荐GPU          价格/小时     训练时间    总成本
------------------------------------------------------------------
调试/测试            RTX 3090         ¥2.0         2h          ¥4
DAgger训练           RTX 4090         ¥2.8         10h         ¥28
PPO中等训练(500K)    RTX 4090         ¥2.8         25h         ¥70
PPO大规模(2M)        A100 40GB        ¥7.0         25h         ¥175
```

### 2. 利用暂停功能
- DAgger训练时，人工标注期间暂停实例
- 调试代码时，本地完成后再上传运行

### 3. 批量训练
- 一次启动实例，运行多个实验
- 充分利用已付费的时间

### 4. 使用低峰时段
- 深夜/凌晨价格可能更低
- 机器可用率更高

---

## 🔧 常见问题

### Q1: MineDojo在云端能运行吗？
**A**: 可以，但需要 headless 模式：
```bash
export JAVA_OPTS="-Djava.awt.headless=true"
python src/training/train_get_wood.py --headless
```

### Q2: 录制数据需要在云端吗？
**A**: 不需要，录制需要图形界面，建议在本地完成：
```
本地录制 → 上传数据 → 云端训练 → 下载模型
```

### Q3: 如何处理大文件上传？
**A**: 
```bash
# 方法1: 使用Git LFS
git lfs install
git lfs track "*.zip"

# 方法2: 使用OSS/COS中转
# 本地 → OSS → 云端下载

# 方法3: 压缩后上传
tar -czf checkpoints.tar.gz checkpoints/
```

### Q4: 训练中断了怎么办？
**A**: 
```bash
# 使用 --resume 参数恢复训练
bash scripts/train_get_wood.sh --resume --total-steps 500000

# 会自动加载最新的检查点
```

### Q5: 如何多卡训练？
**A**: 
```bash
# 暂不支持，MineDojo环境难以并行化
# 建议使用单卡 + 更强的GPU（A100）
```

---

## 📚 参考资料

- [AutoDL 官方文档](https://www.autodl.com/docs/)
- [恒源云 使用指南](https://www.gpushare.com/docs/)
- [阿里云PAI 文档](https://help.aliyun.com/product/30347.html)
- [MineDojo Headless Mode](https://docs.minedojo.org/)

---

## ✅ 部署检查清单

部署前确认：
- [ ] 代码已提交到Git仓库
- [ ] requirements.txt 完整
- [ ] 录制数据已备份（DAgger）
- [ ] MineCLIP模型文件已准备
- [ ] .env 文件已配置（如需要）

部署后验证：
- [ ] CUDA 可用 (`torch.cuda.is_available()`)
- [ ] MineDojo 安装成功
- [ ] 可以创建环境 (`env = minedojo.make(...)`)
- [ ] TensorBoard 可访问
- [ ] 检查点正常保存

---

**最后更新**: 2025-10-23  
**推荐平台**: AutoDL (个人) / 恒源云 (企业)  
**预估成本**: ¥30-200 (取决于任务规模)

