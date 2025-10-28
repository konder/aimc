#!/usr/bin/env python3
"""
环境验证脚本 - 完整验证 AIMC 部署环境

使用方法:
    python tools/validate_environment.py
    
    # M 芯片用户:
    ./scripts/run_minedojo_x86.sh python tools/validate_environment.py
"""

import sys
import os
import platform
import subprocess


def print_header(title):
    """打印章节标题"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def print_success(msg):
    """打印成功消息"""
    print(f"✓ {msg}")


def print_warning(msg):
    """打印警告消息"""
    print(f"⚠️  {msg}")


def print_error(msg):
    """打印错误消息"""
    print(f"✗ {msg}")


def check_python_version():
    """检查 Python 版本"""
    print_header("1. Python 环境")
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 9:
        print_success("Python 3.9 - 正确")
        return True
    elif version.major == 3 and version.minor >= 9:
        print_warning(f"Python {version.major}.{version.minor} - 推荐 3.9，但可能兼容")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} - 需要 Python 3.9")
        return False


def check_architecture():
    """检查系统架构"""
    print_header("2. 系统架构")
    
    machine = platform.machine()
    system = platform.system()
    
    print(f"操作系统: {system}")
    print(f"架构: {machine}")
    
    if machine == "x86_64":
        print_success("x86_64 - 正确")
        return True
    elif machine == "arm64" or machine == "aarch64":
        print_warning("ARM64 架构 - 请确保使用 x86_64 模式运行")
        print("提示: 使用 ./scripts/run_minedojo_x86.sh python tools/validate_environment.py")
        return False
    else:
        print_error(f"未知架构: {machine}")
        return False


def check_java():
    """检查 Java 环境"""
    print_header("3. Java 环境")
    
    # 检查 JAVA_HOME
    java_home = os.environ.get('JAVA_HOME', '')
    if java_home:
        print(f"JAVA_HOME: {java_home}")
    else:
        print_warning("JAVA_HOME 未设置")
    
    # 检查 Java 版本
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=5)
        version_output = result.stderr or result.stdout
        
        print(f"Java 输出:\n{version_output}")
        
        if '1.8.0' in version_output or 'version "8' in version_output:
            print_success("Java 8 - 正确")
            return True
        else:
            print_warning("Java 版本不是 8，可能影响 MineDojo")
            return True
    except FileNotFoundError:
        print_error("Java 未安装或不在 PATH 中")
        print("安装: brew install temurin@8  # macOS")
        print("      sudo apt install openjdk-8-jdk  # Ubuntu")
        return False
    except Exception as e:
        print_error(f"检查 Java 失败: {e}")
        return False


def check_core_packages():
    """检查核心 Python 包"""
    print_header("4. 核心 Python 包")
    
    packages = {
        'numpy': '1.24.3',
        'torch': '2.0+',
        'gym': '0.21.0',
        'stable_baselines3': '1.8.0+',
        'opencv-python': '4.8.1.78',
    }
    
    all_ok = True
    
    for package, expected in packages.items():
        try:
            if package == 'opencv-python':
                import cv2
                version = cv2.__version__
                name = 'OpenCV'
            elif package == 'stable_baselines3':
                import stable_baselines3
                version = stable_baselines3.__version__
                name = 'Stable-Baselines3'
            else:
                module = __import__(package)
                version = module.__version__
                name = package
            
            print(f"{name}: {version}", end=" ")
            
            # 检查版本
            if package == 'opencv-python' and version == expected:
                print_success("完全匹配")
            elif package == 'numpy' and version.startswith('1.'):
                print_success("正确（< 2.0）")
            elif package == 'gym' and version == '0.21.0':
                print_success("正确")
            elif '+' in expected:  # 版本要求 >=
                print_success("OK")
            else:
                print_warning(f"预期 {expected}")
                all_ok = False
                
        except ImportError:
            print_error(f"{package} 未安装")
            all_ok = False
        except AttributeError:
            print_warning(f"{package} 已安装但无版本信息")
    
    return all_ok


def check_minedojo():
    """检查 MineDojo"""
    print_header("5. MineDojo 环境")
    
    try:
        import minedojo
        print_success(f"MineDojo 已安装")
        
        # 尝试列出任务
        try:
            from minedojo.tasks import ALL_PROGRAMMATIC_TASK_IDS
            num_tasks = len(ALL_PROGRAMMATIC_TASK_IDS)
            print_success(f"可用任务: {num_tasks} 个")
        except:
            print_warning("无法列出任务")
        
        # 尝试创建环境
        try:
            print("尝试创建环境...")
            env = minedojo.make(
                task_id='harvest_1_log',
                image_size=(160, 256)
            )
            print_success("环境创建成功")
            
            # 测试 reset
            print("尝试 reset...")
            obs = env.reset()
            print_success(f"Reset 成功, 观测形状: {obs['rgb'].shape}")
            
            # 测试几步
            print("运行 3 步...")
            for i in range(3):
                obs, reward, done, info = env.step(env.action_space.no_op())
                if done:
                    print_warning(f"第 {i+1} 步任务完成")
                    break
            
            env.close()
            print_success("MineDojo 完整测试通过")
            return True
            
        except Exception as e:
            print_error(f"创建/运行环境失败: {e}")
            print("\n提示:")
            print("  1. 检查 JAVA_HOME 设置")
            print("  2. 检查 Java 版本 (需要 8)")
            print("  3. 查看日志: logs/mc_*.log")
            return False
            
    except ImportError:
        print_error("MineDojo 未安装")
        print("安装: pip install minedojo")
        return False


def check_optional_packages():
    """检查可选包"""
    print_header("6. 可选组件")
    
    optional = {
        'mineclip': 'MineCLIP 奖励模型',
        'pygame': '鼠标控制录制',
        'flask': 'Web 控制台',
        'tensorboard': '训练可视化',
        'minerl': 'BASALT 任务',
    }
    
    for package, desc in optional.items():
        try:
            __import__(package)
            print_success(f"{desc}: 已安装")
        except ImportError:
            print(f"  {desc}: 未安装（可选）")


def check_project_structure():
    """检查项目结构"""
    print_header("7. 项目结构")
    
    required_dirs = [
        'src',
        'scripts',
        'docs',
        'tools',
        'data',
        'logs',
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'DEPLOYMENT.md',
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print_success(f"目录存在: {dir_name}/")
        else:
            print_warning(f"目录缺失: {dir_name}/")
            all_ok = False
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print_success(f"文件存在: {file_name}")
        else:
            print_warning(f"文件缺失: {file_name}")
            all_ok = False
    
    # 检查脚本权限
    print("\n脚本权限:")
    script_dir = 'scripts'
    if os.path.isdir(script_dir):
        scripts = [f for f in os.listdir(script_dir) if f.endswith('.sh')]
        for script in scripts[:5]:  # 只检查前5个
            script_path = os.path.join(script_dir, script)
            if os.access(script_path, os.X_OK):
                print_success(f"可执行: {script}")
            else:
                print_warning(f"无执行权限: {script}")
                print(f"  修复: chmod +x {script_path}")
    
    return all_ok


def check_data_directories():
    """检查数据目录"""
    print_header("8. 数据目录")
    
    data_dirs = [
        'data/tasks',
        'data/pretrained/vpt',
        'data/mineclip',
        'data/clip_tokenizer',
        'logs/training',
        'logs/tensorboard',
    ]
    
    for dir_path in data_dirs:
        exists = os.path.isdir(dir_path)
        status = "✓" if exists else "○"
        desc = "存在" if exists else "将自动创建"
        print(f"{status} {dir_path}: {desc}")


def check_environment_variables():
    """检查环境变量"""
    print_header("9. 环境变量")
    
    env_vars = {
        'JAVA_HOME': '必需',
        'JAVA_OPTS': '推荐',
        'PYTHONPATH': '可选',
        'CUDA_VISIBLE_DEVICES': '可选（GPU）',
        'DISPLAY': '应该未设置（或清除）',
    }
    
    for var, desc in env_vars.items():
        value = os.environ.get(var, '')
        if value:
            if var == 'DISPLAY' and platform.system() == 'Darwin':
                print_warning(f"{var}={value} - 可能干扰窗口显示，建议 unset")
            else:
                print(f"✓ {var}={value}")
        else:
            if var == 'JAVA_HOME':
                print_error(f"{var} 未设置 - 必需")
            elif var == 'DISPLAY':
                print_success(f"{var} 未设置 - 正确")
            else:
                print(f"  {var} 未设置 - {desc}")


def print_summary(results):
    """打印总结"""
    print_header("验证总结")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n总计: {passed}/{total} 项通过\n")
    
    for check, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("🎉 所有检查通过！环境配置正确")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 开始 DAgger 训练:")
        print("     bash scripts/run_dagger_workflow.sh --task harvest_1_log --iterations 3")
        print("\n  2. 或查看文档:")
        print("     cat docs/guides/DAGGER_COMPREHENSIVE_GUIDE.md")
        return True
    else:
        print("\n" + "=" * 60)
        print("⚠️  部分检查未通过，请根据上述提示修复")
        print("=" * 60)
        print("\n帮助:")
        print("  - 完整部署指南: cat DEPLOYMENT.md")
        print("  - 常见问题: cat FAQ.md")
        print("  - 获取支持: https://github.com/your-repo/aimc/issues")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" AIMC 环境验证工具")
    print(" 版本: 2025-10-28")
    print("=" * 60)
    
    results = {}
    
    # 运行所有检查
    results['Python 版本'] = check_python_version()
    results['系统架构'] = check_architecture()
    results['Java 环境'] = check_java()
    results['核心包'] = check_core_packages()
    results['MineDojo'] = check_minedojo()
    
    # 可选检查（不影响总结）
    check_optional_packages()
    check_project_structure()
    check_data_directories()
    check_environment_variables()
    
    # 打印总结
    success = print_summary(results)
    
    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

