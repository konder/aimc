# STEVE-1 集成状态与下一步计划

**日期**: 2025-11-06  
**状态**: 需要进一步调试和修复  

---

## 当前状况总结

### 发现的关键问题

1. **图像格式差异**
   - **MineRL**: 返回 `(H, W, C)` 格式，键名 `obs['pov']`
   - **MineDojo**: 返回 `(C, H, W)` 格式，键名 `obs['rgb']`
   - **VPT网络**: 期望 `(N, H, W, C)` 格式（**非标准的 PyTorch 格式**）

2. **环境兼容性**
   - STEVE-1 官方实现基于 **MineRL 环境**
   - 我们的项目使用 **MineDojo 环境**
   - MineRL 环境在当前配置下无法正常启动（超时）

3. **数据类型要求**
   - `MineRLConditionalAgent.get_action()` 期望 `goal_embed` 是 **numpy 数组**
   - MineCLIP 返回的是 **PyTorch Tensor**
   - 需要转换：`text_embed.cpu().numpy()`

---

## 已修复的问题（共6个）

✅ 1. 嵌套调用 `run_minedojo_x86.sh`  
✅ 2. `expected_steps` 类型错误  
✅ 3. MineDojo 环境不兼容 `MineRLConditionalAgent`  
✅ 4. 键名错误（`prompt_embed` → `mineclip_embed`）  
✅ 5. 图像维度顺序错误（第一次尝试）  
✅ 6. 图像格式差异（MineDojo vs MineRL）  

---

## 核心技术发现

### VPT 网络的特殊性

VPT/STEVE-1 网络使用**非标准的图像格式**：

```python
# 标准 PyTorch CNN: (N, C, H, W)
# VPT 网络期望: (N, H, W, C)  ← 注意！

# 这是因为 VPT 最初为 MineRL 设计
# MineRL 返回的就是 (H, W, C) 格式
```

参考代码：
- `src/training/steve1/embed_conditioned_policy.py:206`
  ```python
  b, t = ob["img"].shape[:2]  # 取前两维作为 batch 和 time
  ```

### 官方实现的环境创建方式

```python
# src/training/steve1/utils/mineclip_agent_env_utils.py

def make_env(seed):
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
    env = HumanSurvival(**ENV_KWARGS).make()
    env.reset()
    if seed is not None:
        env.seed(seed)
    return env

def make_agent(in_model, in_weights, cond_scale):
    env = gym.make("MineRLBasaltFindCave-v0")  # 用于初始化
    agent = MineRLConditionalAgent(env, device=DEVICE, ...)
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()  # 初始化后关闭
    return agent
```

---

## 下一步方案

### 方案 A: 使用 MineDojo + 格式转换（推荐）

**优势**:
- MineDojo 在当前环境可以正常运行
- 提供更丰富的任务和评估功能
- 只需要做格式转换

**实现步骤**:

1. **观察格式转换**
   ```python
   def convert_minedojo_to_minerl(obs):
       """MineDojo (C,H,W) -> MineRL (H,W,C)"""
       minerl_obs = {'pov': np.transpose(obs['rgb'], (1, 2, 0))}
       return minerl_obs
   ```

2. **使用官方 MineRLConditionalAgent**
   ```python
   # 使用临时 MineRL 环境初始化 Agent
   temp_env = gym.make("MineRLBasaltFindCave-v0")
   agent = MineRLConditionalAgent(temp_env, device=device)
   agent.load_weights(weights_path)
   agent.reset(cond_scale=6.0)
   temp_env.close()
   
   # 在 MineDojo 环境中使用
   minedojo_env = minedojo.make('harvest_1_log')
   minedojo_obs = minedojo_env.reset()
   
   # 转换观察
   minerl_obs = convert_minedojo_to_minerl(minedojo_obs)
   
   # 编码指令（转换为numpy）
   text_embed = mineclip.encode_text("chop tree").cpu().numpy()
   
   # 获取动作
   action = agent.get_action(minerl_obs, text_embed)
   ```

3. **关键注意事项**
   - ✅ 使用 `gym.make("MineRLBasaltFindCave-v0")` 初始化（然后关闭）
   - ✅ `goal_embed` 必须是 numpy 数组：`.cpu().numpy()`
   - ✅ 观察转换：`(C,H,W)` → `(H,W,C)`
   - ✅ **不要** 做额外的 permute，VPT 期望 `(N,H,W,C)` 格式

### 方案 B: 修复 MineRL 环境（备选）

**劣势**:
- MineRL 环境在当前配置下无法启动
- 需要额外的 Minecraft/Malmo 配置
- 功能不如 MineDojo 丰富

**不推荐**，除非必须使用 MineRL。

---

## 代码修改建议

### 修改 `src/agents/steve1_agent.py`

```python
class STEVE1Agent:
    def _lazy_init(self, env):
        """延迟初始化"""
        if self._agent is None:
            # 使用临时 MineRL 环境初始化
            import gym
            temp_env = gym.make("MineRLBasaltFindCave-v0")
            
            from src.training.steve1.MineRLConditionalAgent import MineRLConditionalAgent
            self._agent = MineRLConditionalAgent(temp_env, device=str(self.device))
            
            if self.model_path:
                self._agent.load_weights(str(self.model_path))
            
            self._agent.reset(cond_scale=self.cond_scale)
            temp_env.close()
        
        if self._mineclip is None:
            self._mineclip = self._load_mineclip()
    
    def encode_instruction(self, instruction: str):
        """编码指令为 MineCLIP embedding（返回 numpy）"""
        if self._mineclip is None:
            self._mineclip = self._load_mineclip()
        
        with th.no_grad():
            text_embed = self._mineclip.encode_text(instruction)
        
        # 转换为 numpy（MineRLConditionalAgent 需要）
        return text_embed.cpu().numpy()
    
    def get_action(self, obs, instruction=None, goal_embed=None, env=None):
        """获取动作"""
        if self._agent is None:
            if env is None:
                raise ValueError("首次调用需要提供 env")
            self._lazy_init(env)
        
        # 编码指令
        if goal_embed is None:
            if instruction is None:
                raise ValueError("需要提供 instruction 或 goal_embed")
            goal_embed = self.encode_instruction(instruction)
        
        # 转换 MineDojo 观察为 MineRL 格式
        minerl_obs = self._convert_minedojo_to_minerl(obs)
        
        # 获取动作
        return self._agent.get_action(minerl_obs, goal_embed)
    
    def _convert_minedojo_to_minerl(self, obs):
        """MineDojo -> MineRL 格式转换"""
        import numpy as np
        
        if 'pov' in obs:  # 已经是 MineRL 格式
            return obs
        
        # 转换 MineDojo
        minerl_obs = {}
        if 'rgb' in obs:
            img = obs['rgb']
            # (C, H, W) -> (H, W, C)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            minerl_obs['pov'] = img
        
        return minerl_obs
```

---

## 测试计划

### Phase 1: 基础功能验证
1. ✅ 验证 MineRLConditionalAgent 可以创建（使用临时环境）
2. ✅ 验证权重可以加载
3. ⏳ 验证在 MineDojo 环境中可以生成动作
4. ⏳ 验证动作格式正确

### Phase 2: 评估集成
1. ⏳ 单任务评估（harvest_1_log, 2 trials）
2. ⏳ 多任务评估（quick_test）
3. ⏳ 中文指令测试

### Phase 3: 优化和文档
1. ⏳ 性能优化
2. ⏳ 完善文档
3. ⏳ 添加更多测试用例

---

## 关键文件清单

### 核心实现
- `src/agents/steve1_agent.py` - STEVE-1 Agent 封装
- `src/evaluation/steve1_evaluator.py` - 评估器
- `src/training/steve1/MineRLConditionalAgent.py` - 官方 Agent 实现

### 测试脚本
- `scripts/test_steve1_minerl.py` - MineRL 环境测试（有问题）
- `scripts/test_steve1_real_env.py` - 真实环境测试
- `scripts/run_steve1_evaluation.sh` - 评估启动脚本

### 配置文件
- `config/eval_tasks.yaml` - 评估任务配置
- `data/chinese_terms.json` - 中文术语词典

### 文档
- `docs/summaries/STEVE1_MINEDOJO_INTEGRATION_FIXES.md` - 修复总结
- `docs/guides/STEVE1_EVALUATION_USAGE.md` - 使用指南

---

## 总结

**当前状态**: 已经解决了大部分技术问题，核心是理解了 VPT 网络的非标准图像格式。

**主要障碍**: MineRL 环境无法在当前配置下启动。

**推荐方案**: 使用 MineDojo 环境 + 格式转换，直接使用官方的 `MineRLConditionalAgent`。

**下一步**: 按照"方案 A"修改代码并测试。

---

**修复完成时间**: 2025-11-06  
**总计修复问题**: 6个  
**状态**: 待继续调试  

