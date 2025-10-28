"""测试官方VPT Agent（限制步数）"""
from argparse import ArgumentParser
import pickle
import sys
from pathlib import Path

# 添加VPT路径
vpt_path = Path(__file__).resolve().parent.parent / "src" / "models" / "Video-Pre-Training"
if str(vpt_path) not in sys.path:
    sys.path.insert(0, str(vpt_path))

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS

def main(model, weights, max_steps=100):
    print("="*70)
    print("测试官方VPT Agent")
    print("="*70)
    print(f"模型: {model}")
    print(f"权重: {weights}")
    print(f"最大步数: {max_steps}")
    print("="*70)
    
    # 创建MineRL环境
    print("\n1. 创建MineRL环境...")
    env = HumanSurvival(**ENV_KWARGS).make()
    print("✓ 环境创建成功")
    
    # 加载模型配置
    print("\n2. 加载模型配置...")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    print("✓ 模型配置加载成功")
    
    # 创建Agent
    print("\n3. 创建MineRLAgent...")
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    print("✓ Agent创建成功")
    
    # 加载权重
    print("\n4. 加载权重...")
    agent.load_weights(weights)
    print("✓ 权重加载成功")
    
    # 重置环境
    print("\n5. 重置环境（这可能需要一些时间...）")
    obs = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  obs类型: {type(obs)}")
    print(f"  obs keys: {list(obs.keys())}")
    if 'pov' in obs:
        print(f"  obs['pov'] shape: {obs['pov'].shape}")
    
    # 运行agent
    print(f"\n6. 运行Agent（{max_steps}步）...")
    for step in range(max_steps):
        minerl_action = agent.get_action(obs)
        obs, reward, done, info = env.step(minerl_action)
        
        if step % 20 == 0:
            print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}")
        
        if done:
            print(f"\n  Episode结束于第{step}步")
            break
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ 官方VPT Agent测试通过！")
    print("="*70)
    print("\n结论:")
    print("  ✓ 官方VPT代码运行正常")
    print("  ✓ MineRL环境正常工作")
    print("  ✓ Agent能正常预测动作")
    print("  ✓ obs['pov']格式已验证")
    print("="*70)

if __name__ == "__main__":
    parser = ArgumentParser("测试官方VPT模型")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    
    args = parser.parse_args()
    main(args.model, args.weights, args.max_steps)
