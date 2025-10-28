# VPT Agent重构总结

## 概述

本次重构将完整版VPT Agent作为默认实现，移除了简化版Agent，统一了代码库。

## 完成的工作

### 1. ✅ Agent重构

**文件变更：**
- ❌ 移除：`src/training/agent/vpt_agent.py` (旧简化版)
  - 备份为：`src/training/agent/vpt_agent_old_backup.py`
- ✅ 重命名：`vpt_agent_complete.py` → `vpt_agent.py` 
- ✅ 更新：`src/training/agent/__init__.py`
  - 只导出：`VPTAgent`, `MineRLToMinedojoConverter`

**新VPTAgent特性：**
- ✅ Hidden State维护（Transformer memory）
- ✅ 使用policy.act()（完整VPT forward）
- ✅ Pi head智能决策
- ✅ 官方action_mapper和action_transformer
- ✅ First标志处理（episode边界）
- ✅ `device='auto'`支持（cuda > mps > cpu）

### 2. ✅ 零样本评估脚本统一

**清理：**
- ❌ 删除：`src/training/vpt/evaluate_vpt_true_zero_shot.py` (旧版，13K)
- ✅ 保留：`src/training/vpt/evaluate_vpt_zero_shot.py` (新版，使用新VPTAgent)

**新零样本评估特性：**
- 使用完整版VPTAgent
- 支持device='auto'
- 详细的统计输出
- 支持多种评估模式

### 3. ✅ 测试脚本清理

- ❌ 删除：`tools/test_vpt_agent_complete.py` (已通过测试)

### 4. ✅ 快速启动工具

**新增文件：**
- `scripts/evaluate_vpt_zero_shot.sh` - 一键启动脚本
- `docs/guides/VPT_ZERO_SHOT_QUICKSTART.md` - 快速入门指南

### 5. ✅ models/vpt/lib清理

- 保留所有lib文件（相互依赖较多）
- 主要使用：
  - `policy.py` - MinecraftAgentPolicy
  - `action_mapping.py` - CameraHierarchicalMapping
  - `actions.py` - ActionTransformer

### 6. ✅ Device自适应支持

**更新的文件：**
- `src/models/vpt/weights_loader.py`
- `src/training/agent/agent_base.py`
- `src/training/agent/vpt_agent.py`

**智能检测逻辑：**
```python
if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'      # 生产机
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'       # Mac开发机
    else:
        device = 'cpu'       # 备用
```

## 快速开始零样本评估

### 方法1: 使用启动脚本（推荐）

```bash
# 默认：10轮，auto设备
bash scripts/evaluate_vpt_zero_shot.sh

# 自定义
bash scripts/evaluate_vpt_zero_shot.sh 20            # 20轮
bash scripts/evaluate_vpt_zero_shot.sh 10 mps        # 指定设备
bash scripts/evaluate_vpt_zero_shot.sh 20 mps 1500   # 全部参数
```

### 方法2: 直接运行Python

```bash
python src/training/vpt/evaluate_vpt_zero_shot.py \
    --agent complete \
    --episodes 10 \
    --max_steps 1200 \
    --device auto
```

### 在Mac M1/M2上

```bash
# 使用x86模式运行MineDojo
bash scripts/run_minedojo_x86.sh python src/training/vpt/evaluate_vpt_zero_shot.py --episodes 5
```

## 文件结构

```
src/training/agent/
├── agent_base.py                    # Agent基类（支持device='auto'）
├── vpt_agent.py                     # VPTAgent（完整版，官方实现）
├── vpt_agent_old_backup.py         # 旧简化版备份
└── __init__.py                      # 导出VPTAgent

src/training/vpt/
├── evaluate_vpt_zero_shot.py       # ✨ 零样本评估（新版，使用VPTAgent）
├── train_bc_vpt.py                 # 📝 待更新：使用新VPTAgent
└── evaluate_bc_vpt.py              # 📝 待更新：使用新VPTAgent

scripts/
└── evaluate_vpt_zero_shot.sh       # ✨ 一键启动脚本

docs/guides/
└── VPT_ZERO_SHOT_QUICKSTART.md     # ✨ 快速入门指南

src/models/vpt/
├── weights_loader.py                # 支持device='auto'
└── lib/                             # VPT官方库（保留所有文件）
    ├── policy.py                    # MinecraftAgentPolicy
    ├── action_mapping.py            # CameraHierarchicalMapping
    ├── actions.py                   # ActionTransformer
    └── ...                          # 其他依赖
```

## 待完成任务

### 高优先级

1. **改写train_bc_vpt.py使用新VPTAgent**
   - 移除旧的MinedojoActionAdapter
   - 使用VPTAgent统一接口
   - 更新训练循环

2. **改写evaluate_bc_vpt.py使用新VPTAgent**
   - 统一评估接口
   - 对比fine-tune前后性能

### 中优先级

3. **实际零样本性能对比**
   - 运行评估获取实际成功率
   - 记录平均步数和奖励
   - 与baseline对比

4. **BC Fine-tuning**
   - 使用新VPTAgent进行BC训练
   - 冻结VPT参数，只训练decision head
   - 评估fine-tune效果

## 预期性能

### 零样本基线（harvest_1_log）

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 成功率 | 10-30% | 零样本，无任务训练 |
| 平均步数 | 800-1000 | 成功episodes |
| 平均奖励 | 0.1-0.3 | 每episode平均 |

### Fine-tune后预期

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 成功率 | 60-80% | BC训练后 |
| 平均步数 | 400-600 | 更高效 |
| 平均奖励 | 0.6-0.8 | 显著提升 |

## 技术亮点

### 1. 完整VPT实现

```python
class VPTAgent(AgentBase):
    """VPT Agent - 完整版实现，严格按照官方VPT"""
    
    def predict(self, obs, deterministic=True):
        # 1. 预处理观察（resize, normalize）
        agent_input = self._preprocess_obs(obs)
        
        # 2. VPT forward（使用hidden state）
        minerl_action_dict, self.hidden_state = self.vpt_policy.act(
            agent_input=agent_input,
            first=self.first_flag,
            state_in=self.hidden_state,
            deterministic=deterministic
        )
        
        # 3. 转换为MineDojo动作
        minedojo_action = self.action_converter.convert(minerl_action_dict)
        
        return minedojo_action
```

### 2. 智能设备检测

```python
# 自动适配不同平台
agent = VPTAgent(
    vpt_weights_path="data/pretrained/vpt/rl-from-early-game-2x.weights",
    device='auto',  # cuda (生产) > mps (开发) > cpu (备用)
    verbose=True
)
```

### 3. 官方动作转换

```python
# 完整的MineRL → MineDojo转换
action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
action_transformer = ActionTransformer(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law"
)
```

## 使用示例

### 基本使用

```python
from src.training.agent import VPTAgent
import minedojo

# 创建Agent
agent = VPTAgent(
    vpt_weights_path="data/pretrained/vpt/rl-from-early-game-2x.weights",
    device='auto',
    verbose=True
)

# 创建环境
env = minedojo.make("harvest_1_log", image_size=(160, 256))

# 运行episode
obs = env.reset()
agent.reset()  # 重置hidden state

done = False
total_reward = 0

while not done:
    action = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Episode reward: {total_reward}")
env.close()
```

### 零样本评估

```bash
# 一键评估
bash scripts/evaluate_vpt_zero_shot.sh 20

# 或
python src/training/vpt/evaluate_vpt_zero_shot.py \
    --agent complete \
    --episodes 20 \
    --device auto
```

## 相关文档

1. [VPT完整实现文档](./VPT_AGENT_COMPLETE_IMPLEMENTATION.md) - 技术细节
2. [零样本快速入门](../guides/VPT_ZERO_SHOT_QUICKSTART.md) - 使用指南
3. [MineDojo包装器](../guides/TASK_WRAPPERS_GUIDE.md) - 环境配置

## 常见问题

### Q: 旧代码如何迁移到新VPTAgent？

```python
# 旧代码（简化版）
from src.training.agent import VPTAgent  # 简化版，已移除
agent = VPTAgent(...)

# 新代码（完整版）
from src.training.agent import VPTAgent  # 完整版，自动使用
agent = VPTAgent(
    vpt_weights_path="...",
    device='auto',      # 新增：智能设备检测
    conflict_strategy='priority',
    verbose=True
)
```

### Q: 如何确认使用的是完整版？

```python
# 检查是否有hidden_state
agent = VPTAgent(...)
assert hasattr(agent, 'hidden_state'), "应该有hidden_state属性"
assert hasattr(agent, 'first_flag'), "应该有first_flag属性"
print("✓ 使用完整版VPTAgent")
```

### Q: 性能不如预期怎么办？

1. 先运行零样本评估建立基线
2. 检查device设置（使用GPU加速）
3. 确认VPT权重文件正确
4. 查看日志排查问题
5. 尝试BC fine-tuning提升性能

## 下一步计划

1. ✅ 完成VPT Agent重构
2. ✅ 统一零样本评估脚本
3. ✅ 创建快速启动工具
4. 📝 更新train_bc_vpt.py
5. 📝 更新evaluate_bc_vpt.py  
6. 🔄 运行零样本性能测试
7. 🚀 BC Fine-tuning实验

## 更新日志

- **2025-10-27**: 完成VPT Agent重构和零样本评估统一
  - 重命名vpt_agent_complete → vpt_agent
  - 移除旧简化版
  - 添加device='auto'支持
  - 创建快速启动脚本和文档
  - 清理测试文件和冗余脚本

