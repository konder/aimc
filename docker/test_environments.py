#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MineDojo 和 MineRL 环境验证脚本
在 Docker 容器中运行，验证所有依赖和配置是否正确
"""

import sys
import os
import subprocess
from pathlib import Path

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")

# ============================================================================
# 测试 1: Python 环境检查
# ============================================================================
def test_python_environment():
    print_header("1. Python 环境检查")
    
    try:
        # Python 版本
        version = sys.version.split()[0]
        print_info(f"Python 版本: {version}")
        if version.startswith('3.9'):
            print_success("Python 版本正确 (3.9.x)")
        else:
            print_warning(f"Python 版本可能不匹配 (期望 3.9.x，当前 {version})")
        
        # Conda 环境
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'N/A')
        print_info(f"Conda 环境: {conda_env}")
        if conda_env == 'minedojo-x86':
            print_success("Conda 环境正确")
        else:
            print_error(f"Conda 环境错误 (期望 minedojo-x86，当前 {conda_env})")
        
        return True
    except Exception as e:
        print_error(f"Python 环境检查失败: {e}")
        return False

# ============================================================================
# 测试 2: 核心包导入测试
# ============================================================================
def test_core_imports():
    print_header("2. 核心包导入测试")
    
    packages = {
        'numpy': 'NumPy',
        'gym': 'OpenAI Gym',
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'PIL': 'Pillow',
    }
    
    all_success = True
    for module, name in packages.items():
        try:
            exec(f"import {module}")
            version = eval(f"{module}.__version__")
            print_success(f"{name:20} - {version}")
        except Exception as e:
            print_error(f"{name:20} - 导入失败: {e}")
            all_success = False
    
    return all_success

# ============================================================================
# 测试 3: MineDojo 导入和配置
# ============================================================================
def test_minedojo_import():
    print_header("3. MineDojo 导入和配置")
    
    try:
        import minedojo
        print_success(f"MineDojo 导入成功")
        
        # 检查 MineDojo 版本
        try:
            version = minedojo.__version__
            print_info(f"MineDojo 版本: {version}")
        except:
            print_warning("无法获取 MineDojo 版本")
        
        # 动态获取 Minecraft 路径
        import site
        site_packages = site.getsitepackages()[0]
        mc_path = Path(site_packages) / "minedojo/sim/Malmo/Minecraft"
        
        print_info(f"Minecraft 路径: {mc_path}")
        
        if mc_path.exists():
            print_success("Minecraft 目录存在")
            
            # 检查关键文件
            jar_file = mc_path / "build/libs/MalmoMod-0.37.0-fat.jar"
            if jar_file.exists():
                size_mb = jar_file.stat().st_size / (1024 * 1024)
                print_success(f"Minecraft JAR 已编译: {size_mb:.1f} MB")
            else:
                print_error("Minecraft JAR 未找到")
                return False
            
            # 检查 launchClient.sh
            launch_script = mc_path / "launchClient.sh"
            if launch_script.exists():
                print_success("launchClient.sh 存在")
                
                # 检查是否包含无头模式参数
                content = launch_script.read_text()
                if '-Djava.awt.headless=true' in content:
                    print_success("无头模式参数已配置")
                else:
                    print_warning("无头模式参数可能未配置")
            else:
                print_error("launchClient.sh 未找到")
        else:
            print_error(f"Minecraft 目录不存在: {mc_path}")
            return False
        
        return True
    except ImportError as e:
        print_error(f"MineDojo 导入失败: {e}")
        return False
    except Exception as e:
        print_error(f"MineDojo 配置检查失败: {e}")
        return False

# ============================================================================
# 测试 4: MineRL 导入和配置
# ============================================================================
def test_minerl_import():
    print_header("4. MineRL 导入和配置")
    
    try:
        import minerl
        print_success("MineRL 导入成功")
        
        # 检查 gym 版本
        import gym
        gym_version = gym.__version__
        print_info(f"Gym 版本: {gym_version}")
        
        if gym_version.startswith('0.19'):
            print_success("Gym 版本与 MineRL 兼容 (0.19.x)")
        else:
            print_warning(f"Gym 版本可能不兼容 (MineRL 需要 0.19.x，当前 {gym_version})")
        
        # 检查 MCP-Reborn
        site_packages = Path("/opt/conda/envs/minedojo-x86/lib/python3.9/site-packages")
        mcp_path = site_packages / "minerl/MCP-Reborn"
        
        if mcp_path.exists():
            print_success(f"MCP-Reborn 目录存在: {mcp_path}")
            
            # 检查 build.gradle 补丁
            build_gradle = mcp_path / "build.gradle"
            if build_gradle.exists():
                content = build_gradle.read_text()
                if 'DISABLED: apply plugin: \'org.spongepowered.mixin\'' in content:
                    print_success("MixinGradle 补丁已应用")
                else:
                    print_warning("MixinGradle 补丁可能未应用")
        else:
            print_warning(f"MCP-Reborn 目录不存在（可能在首次运行时创建）")
        
        return True
    except ImportError as e:
        print_error(f"MineRL 导入失败: {e}")
        return False
    except Exception as e:
        print_error(f"MineRL 配置检查失败: {e}")
        return False

# ============================================================================
# 测试 5: MineCLIP 导入
# ============================================================================
def test_mineclip_import():
    print_header("5. MineCLIP 导入测试")
    
    try:
        import mineclip
        print_success("MineCLIP 导入成功")
        return True
    except ImportError as e:
        print_error(f"MineCLIP 导入失败: {e}")
        return False
    except Exception as e:
        print_error(f"MineCLIP 测试失败: {e}")
        return False

# ============================================================================
# 测试 6: Java 环境检查
# ============================================================================
def test_java_environment():
    print_header("6. Java 环境检查")
    
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        java_output = result.stderr.split('\n')[0]
        print_info(f"Java 版本: {java_output}")
        
        if 'openjdk version "1.8' in java_output or 'java version "1.8' in java_output:
            print_success("Java 8 已安装")
        else:
            print_warning("Java 版本可能不是 1.8")
        
        # 检查 JAVA_HOME
        java_home = os.environ.get('JAVA_HOME', 'N/A')
        print_info(f"JAVA_HOME: {java_home}")
        if java_home != 'N/A':
            print_success("JAVA_HOME 已设置")
        else:
            print_warning("JAVA_HOME 未设置")
        
        return True
    except subprocess.TimeoutExpired:
        print_error("Java 版本检查超时")
        return False
    except FileNotFoundError:
        print_error("Java 未安装")
        return False
    except Exception as e:
        print_error(f"Java 环境检查失败: {e}")
        return False

# ============================================================================
# 测试 7: MineDojo 环境创建
# ============================================================================
def test_minedojo_tasks():
    print_header("7. MineDojo 环境创建")
    
    try:
        import minedojo
        
        print_info("测试 MineDojo 环境创建（参考官方测试）...")
        
        # 使用官方测试脚本的方式创建环境
        task_id = "combat_spider_plains_leather_armors_diamond_sword_shield"
        print_info(f"创建任务: {task_id}")
        
        env = minedojo.make(
            task_id=task_id,
            image_size=(288, 512),  # 必需参数
            world_seed=123,
            seed=42,
        )
        print_success(f"环境创建成功")
        
        # 打印任务提示
        try:
            print_info(f"任务提示: {env.task_prompt}")
        except:
            pass
        
        # Reset
        print_info("执行 reset...")
        obs = env.reset()
        print_success(f"Reset 成功")
        
        # 执行 20 步无动作（参考官方测试）
        print_info("执行 20 步无动作...")
        for i in range(20):
            obs, reward, done, info = env.step(env.action_space.no_op())
            if (i + 1) % 5 == 0:
                print_info(f"  步数 {i+1}/20 (reward={reward:.2f}, done={done})")
        
        print_success(f"完成 20 步测试")
        
        # 关闭环境
        env.close()
        print_success("环境已关闭")
        print_success("MineDojo 安装成功！")
        
        return True
    except Exception as e:
        print_error(f"MineDojo 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 测试 8: MineRL 环境创建
# ============================================================================
def test_minerl_envs():
    print_header("8. MineRL 环境创建")
    
    try:
        import gym
        import minerl
        
        print_info("测试 MineRL 环境创建（参考官方测试）...")
        
        # 使用官方测试脚本的环境
        env_id = "MineRLBasaltBuildVillageHouse-v0"
        print_info(f"创建环境: {env_id}")
        
        env = gym.make(env_id)
        print_success(f"环境创建成功")
        
        # Reset
        print_info("执行 reset...")
        obs = env.reset()
        print_success(f"Reset 成功")
        
        # 执行若干步无动作（参考官方测试，修改相机）
        print_info("执行测试步骤（旋转相机）...")
        done = False
        step_count = 0
        max_steps = 5  # 限制步数，避免运行太久
        
        while not done and step_count < max_steps:
            ac = env.action_space.noop()
            # 旋转相机查看周围（参考官方测试）
            ac["camera"] = [0, 3]
            obs, reward, done, info = env.step(ac)
            step_count += 1
            
            if step_count % 5 == 0:
                print_info(f"  步数 {step_count}/{max_steps} (reward={reward:.2f}, done={done})")
        
        print_success(f"完成 {step_count} 步测试")
        
        # 关闭环境
        env.close()
        print_success("环境已关闭")
        print_success("MineRL 安装成功！")
        
        return True
    except Exception as e:
        print_error(f"MineRL 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# 测试 9: 系统资源检查
# ============================================================================
def test_system_resources():
    print_header("9. 系统资源检查")
    
    try:
        import psutil
        
        # CPU
        cpu_count = psutil.cpu_count()
        print_info(f"CPU 核心数: {cpu_count}")
        
        # 内存
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_available_gb = mem.available / (1024**3)
        print_info(f"总内存: {mem_total_gb:.1f} GB")
        print_info(f"可用内存: {mem_available_gb:.1f} GB")
        
        if mem_available_gb < 2:
            print_warning("可用内存较少，可能影响 Minecraft 运行")
        else:
            print_success("内存充足")
        
        # 磁盘空间
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        print_info(f"可用磁盘空间: {disk_free_gb:.1f} GB")
        
        return True
    except ImportError:
        print_warning("psutil 未安装，跳过系统资源检查")
        return True
    except Exception as e:
        print_error(f"系统资源检查失败: {e}")
        return False

# ============================================================================
# 测试 10: 环境变量检查
# ============================================================================
def test_environment_variables():
    print_header("10. 环境变量检查")
    
    important_vars = {
        'JAVA_HOME': '/usr/lib/jvm/java-8-openjdk-amd64',
        'MINEDOJO_HEADLESS': '1',
        'CONDA_DEFAULT_ENV': 'minedojo-x86',
    }
    
    all_ok = True
    for var, expected in important_vars.items():
        actual = os.environ.get(var, 'N/A')
        if actual == expected:
            print_success(f"{var} = {actual}")
        elif actual == 'N/A':
            print_warning(f"{var} 未设置 (期望: {expected})")
            all_ok = False
        else:
            print_warning(f"{var} = {actual} (期望: {expected})")
    
    return all_ok

# ============================================================================
# 主函数
# ============================================================================
def main():
    print(f"\n{Colors.BOLD}MineDojo & MineRL Docker 环境验证{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
    
    results = {}
    
    # 执行所有测试
    tests = [
        ("Python 环境", test_python_environment),
        ("核心包导入", test_core_imports),
        ("MineDojo 导入", test_minedojo_import),
        ("MineRL 导入", test_minerl_import),
        ("MineCLIP 导入", test_mineclip_import),
        ("Java 环境", test_java_environment),
        ("MineDojo 环境", test_minedojo_tasks),
        ("MineRL 环境", test_minerl_envs),
        ("系统资源", test_system_resources),
        ("环境变量", test_environment_variables),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print_error(f"测试 '{name}' 出现异常: {e}")
            results[name] = False
    
    # 总结
    print_header("测试总结")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{Colors.BOLD}测试结果:{Colors.RESET}\n")
    for name, result in results.items():
        status = f"{Colors.GREEN}✓ 通过{Colors.RESET}" if result else f"{Colors.RED}✗ 失败{Colors.RESET}"
        print(f"  {name:20} {status}")
    
    print(f"\n{Colors.BOLD}总计: {passed}/{total} 通过{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 所有测试通过！环境配置正确。{Colors.RESET}\n")
        return 0
    elif passed >= total * 0.8:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  大部分测试通过，但有一些警告。{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ 多项测试失败，请检查环境配置。{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

