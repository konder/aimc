#!/bin/bash
#
# 检查点目录迁移脚本
# 
# 功能: 将旧的目录结构迁移到新的分层结构
# 
# 旧结构: checkpoints/TASK_ID/
# 新结构: checkpoints/METHOD/TASK_ID/
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "\n${BLUE}============================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}\n"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# ============================================================================
# 主函数
# ============================================================================

print_header "检查点目录迁移工具"

# 检查是否存在旧的目录结构
OLD_CHECKPOINTS_DIR="checkpoints"
if [ ! -d "$OLD_CHECKPOINTS_DIR" ]; then
    print_info "未找到 checkpoints 目录，无需迁移"
    exit 0
fi

# 查找旧格式的目录（直接在 checkpoints/ 下的任务目录）
OLD_TASK_DIRS=$(find "$OLD_CHECKPOINTS_DIR" -maxdepth 1 -type d -name "*_*" 2>/dev/null || true)

if [ -z "$OLD_TASK_DIRS" ]; then
    print_info "未找到需要迁移的旧格式目录"
    print_info "当前目录结构已经是新格式，或者没有模型文件"
    exit 0
fi

print_warning "发现旧格式的目录结构，需要迁移："
echo "$OLD_TASK_DIRS" | while read -r dir; do
    if [ -n "$dir" ]; then
        echo "  - $dir"
    fi
done
echo ""

# 询问用户是否继续
read -p "是否继续迁移？这将重新组织目录结构 (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    print_info "取消迁移"
    exit 0
fi

print_info "开始迁移..."

# 迁移每个任务目录
echo "$OLD_TASK_DIRS" | while read -r old_dir; do
    if [ -z "$old_dir" ] || [ ! -d "$old_dir" ]; then
        continue
    fi
    
    task_name=$(basename "$old_dir")
    print_info "迁移任务: $task_name"
    
    # 检查目录中的模型文件类型
    has_bc_models=$(find "$old_dir" -name "bc_*.zip" -o -name "*bc*.zip" | wc -l)
    has_dagger_models=$(find "$old_dir" -name "dagger_*.zip" | wc -l)
    has_ppo_models=$(find "$old_dir" -name "ppo_*.zip" -o -name "*ppo*.zip" | wc -l)
    
    # 根据模型类型决定迁移策略
    if [ "$has_dagger_models" -gt 0 ] || [ "$has_bc_models" -gt 0 ]; then
        # 有 DAgger 或 BC 模型，迁移到 dagger 目录
        new_dir="checkpoints/dagger/$task_name"
        mkdir -p "$new_dir"
        
        print_info "  迁移到 DAgger 目录: $new_dir"
        cp -r "$old_dir"/* "$new_dir/"
        
        # 列出迁移的文件
        find "$new_dir" -name "*.zip" | while read -r file; do
            echo "    ✓ $(basename "$file")"
        done
    fi
    
    if [ "$has_ppo_models" -gt 0 ]; then
        # 有 PPO 模型，迁移到 ppo 目录
        new_dir="checkpoints/ppo/$task_name"
        mkdir -p "$new_dir"
        
        print_info "  迁移到 PPO 目录: $new_dir"
        find "$old_dir" -name "*ppo*.zip" -exec cp {} "$new_dir/" \;
        
        # 列出迁移的文件
        find "$new_dir" -name "*.zip" | while read -r file; do
            echo "    ✓ $(basename "$file")"
        done
    fi
    
    # 如果没有识别出特定类型，默认迁移到 dagger
    if [ "$has_bc_models" -eq 0 ] && [ "$has_dagger_models" -eq 0 ] && [ "$has_ppo_models" -eq 0 ]; then
        # 检查是否有任何 .zip 文件
        zip_files=$(find "$old_dir" -name "*.zip" | wc -l)
        if [ "$zip_files" -gt 0 ]; then
            new_dir="checkpoints/dagger/$task_name"
            mkdir -p "$new_dir"
            
            print_warning "  未识别模型类型，默认迁移到 DAgger 目录: $new_dir"
            cp -r "$old_dir"/* "$new_dir/"
        fi
    fi
done

print_success "迁移完成！"

# 显示新的目录结构
print_info "新的目录结构："
if [ -d "checkpoints/dagger" ]; then
    echo "checkpoints/dagger/"
    find checkpoints/dagger -type d -name "*_*" | sort | while read -r dir; do
        task=$(basename "$dir")
        model_count=$(find "$dir" -name "*.zip" | wc -l)
        echo "  ├── $task/ ($model_count 个模型)"
    done
fi

if [ -d "checkpoints/ppo" ]; then
    echo "checkpoints/ppo/"
    find checkpoints/ppo -type d -name "*_*" | sort | while read -r dir; do
        task=$(basename "$dir")
        model_count=$(find "$dir" -name "*.zip" | wc -l)
        echo "  ├── $task/ ($model_count 个模型)"
    done
fi

echo ""
print_warning "旧目录仍然保留，请手动验证迁移结果后删除："
echo "$OLD_TASK_DIRS" | while read -r dir; do
    if [ -n "$dir" ] && [ -d "$dir" ]; then
        echo "  rm -rf $dir"
    fi
done

echo ""
print_info "迁移后的使用方法："
echo "  # DAgger 训练"
echo "  bash scripts/run_dagger_workflow.sh --task TASK_ID --method dagger"
echo ""
echo "  # PPO 训练"
echo "  bash scripts/train_get_wood.sh --task TASK_ID  # 会自动保存到 checkpoints/ppo/TASK_ID/"
echo ""
echo "  # 继续 DAgger 训练"
echo "  bash scripts/run_dagger_workflow.sh \\"
echo "    --task TASK_ID \\"
echo "    --method dagger \\"
echo "    --continue-from checkpoints/dagger/TASK_ID/dagger_iter_N.zip"

print_success "迁移工具执行完成！"
