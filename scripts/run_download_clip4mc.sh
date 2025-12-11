#!/bin/bash
# CLIP4MC YouTube 视频下载脚本
#
# 使用方法:
#   ./scripts/run_download_clip4mc.sh                    # 下载全部 test 集（360p，切片后删除原视频）
#   ./scripts/run_download_clip4mc.sh 100                # 下载 100 个样本
#   ./scripts/run_download_clip4mc.sh 100 --resolution 480   # 使用 480p 分辨率
#   ./scripts/run_download_clip4mc.sh 100 --keep-original    # 保留原始完整视频
#   ./scripts/run_download_clip4mc.sh all                # 下载并预处理
#   ./scripts/run_download_clip4mc.sh preprocess         # 仅预处理
#   ./scripts/run_download_clip4mc.sh restart            # 清除检查点重新开始
#
# 参数说明:
#   --resolution, -r  分辨率: 360, 480, 720, 1080 (默认: 360)
#   --fps             最大帧率 (默认: 不限制)
#   --keep-original   保留原始完整视频 (默认: 切片后删除以节省空间)

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"
COOKIES_FILE="${COOKIES_FILE:-data/www.youtube.com_cookies_2.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/raw_videos/clip4mc_youtube}"
RESOLUTION="${RESOLUTION:-360}"
KEEP_ORIGINAL="${KEEP_ORIGINAL:-false}"
FPS=""

cd "$PROJECT_DIR"

# 检查 cookies 文件
if [[ ! -f "$COOKIES_FILE" ]]; then
    echo "⚠️  警告: cookies 文件不存在: $COOKIES_FILE"
    echo "   建议: 从浏览器导出 YouTube cookies 到该文件以避免 403 错误"
    echo ""
fi

# 检查检查点
CHECKPOINT_FILE="$OUTPUT_DIR/.checkpoint.json"

# 解析参数
COMMAND=""
MAX_SAMPLES=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resolution|-r)
            RESOLUTION="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --keep-original)
            KEEP_ORIGINAL=true
            shift
            ;;
        --cookies|-c)
            COOKIES_FILE="$2"
            shift 2
            ;;
        --no-resume)
            EXTRA_ARGS="$EXTRA_ARGS --no-resume"
            shift
            ;;
        download|test|train|preprocess|all|restart|reset|status|help|--help|-h)
            COMMAND="$1"
            shift
            ;;
        [0-9]*)
            MAX_SAMPLES="$1"
            shift
            ;;
        *)
            # 未知参数，可能是命令
            if [[ -z "$COMMAND" ]]; then
                COMMAND="$1"
            fi
            shift
            ;;
    esac
done

# 默认命令
COMMAND="${COMMAND:-download}"

# 构建下载参数
DOWNLOAD_ARGS="--resolution $RESOLUTION"
if [[ -n "$FPS" ]]; then
    DOWNLOAD_ARGS="$DOWNLOAD_ARGS --fps $FPS"
fi
if [[ "$KEEP_ORIGINAL" == "true" ]]; then
    DOWNLOAD_ARGS="$DOWNLOAD_ARGS --keep-original"
fi
DOWNLOAD_ARGS="$DOWNLOAD_ARGS $EXTRA_ARGS"

# 显示检查点信息
if [[ -f "$CHECKPOINT_FILE" ]]; then
    echo "📂 发现检查点，将从上次中断处继续下载"
    echo "   如需重新开始，请运行: $0 restart"
    echo ""
fi

case "$COMMAND" in
    download|test)
        echo "============================================================"
        echo "📥 下载 CLIP4MC test 集视频"
        echo "============================================================"
        echo "  📐 分辨率: ${RESOLUTION}p"
        if [[ "$KEEP_ORIGINAL" == "true" ]]; then
            echo "  💾 原始视频: 保留"
        else
            echo "  💾 原始视频: 切片后删除"
        fi
        if [[ -n "$MAX_SAMPLES" ]]; then
            echo "  📊 最大样本数: $MAX_SAMPLES"
        else
            echo "  📊 最大样本数: 全部"
        fi
        echo "============================================================"
        
        CMD="$PYTHON -m src.utils.clip4mc_downloader download --dataset test --cookies $COOKIES_FILE $DOWNLOAD_ARGS"
        if [[ -n "$MAX_SAMPLES" ]]; then
            CMD="$CMD --max-samples $MAX_SAMPLES"
        fi
        echo "🚀 执行: $CMD"
        echo ""
        eval $CMD
        ;;
    
    train)
        echo "============================================================"
        echo "📥 下载 CLIP4MC train 集视频"
        echo "============================================================"
        echo "  📐 分辨率: ${RESOLUTION}p"
        if [[ "$KEEP_ORIGINAL" == "true" ]]; then
            echo "  💾 原始视频: 保留"
        else
            echo "  💾 原始视频: 切片后删除"
        fi
        echo "============================================================"
        
        CMD="$PYTHON -m src.utils.clip4mc_downloader download --dataset train --cookies $COOKIES_FILE $DOWNLOAD_ARGS"
        if [[ -n "$MAX_SAMPLES" ]]; then
            CMD="$CMD --max-samples $MAX_SAMPLES"
        fi
        eval $CMD
        ;;
    
    preprocess)
        echo "============================================================"
        echo "🔄 预处理视频为训练格式"
        echo "============================================================"
        $PYTHON -m src.utils.clip4mc_downloader preprocess
        ;;
    
    all)
        echo "============================================================"
        echo "📥 下载并预处理 (一键执行)"
        echo "============================================================"
        echo "  📐 分辨率: ${RESOLUTION}p"
        if [[ "$KEEP_ORIGINAL" == "true" ]]; then
            echo "  💾 原始视频: 保留"
        else
            echo "  💾 原始视频: 切片后删除"
        fi
        echo "============================================================"
        
        CMD="$PYTHON -m src.utils.clip4mc_downloader all --cookies $COOKIES_FILE $DOWNLOAD_ARGS"
        if [[ -n "$MAX_SAMPLES" ]]; then
            CMD="$CMD --max-samples $MAX_SAMPLES"
        fi
        eval $CMD
        ;;
    
    restart|reset)
        echo "============================================================"
        echo "🔄 清除检查点，重新开始下载"
        echo "============================================================"
        
        if [[ -f "$CHECKPOINT_FILE" ]]; then
            rm "$CHECKPOINT_FILE"
            echo "✓ 检查点已清除"
        else
            echo "没有检查点需要清除"
        fi
        
        CMD="$PYTHON -m src.utils.clip4mc_downloader download --dataset test --cookies $COOKIES_FILE $DOWNLOAD_ARGS --no-resume"
        if [[ -n "$MAX_SAMPLES" ]]; then
            CMD="$CMD --max-samples $MAX_SAMPLES"
        fi
        eval $CMD
        ;;
    
    status)
        echo "============================================================"
        echo "📊 下载状态"
        echo "============================================================"
        
        if [[ -f "$CHECKPOINT_FILE" ]]; then
            echo "📂 检查点文件: $CHECKPOINT_FILE"
            $PYTHON -c "
import json
with open('$CHECKPOINT_FILE') as f:
    cp = json.load(f)
print(f'   上次更新: {cp.get(\"timestamp\", \"未知\")}')
print(f'   已处理: {len(cp.get(\"processed_ids\", []))} 个')
print(f'   成功: {len(cp.get(\"processed_samples\", []))} 个')
print(f'   失败: {len(cp.get(\"failed_samples\", []))} 个')
"
        else
            echo "没有进行中的下载任务"
        fi
        
        if [[ -f "$OUTPUT_DIR/samples.json" ]]; then
            echo ""
            echo "📁 已下载样本:"
            $PYTHON -c "
import json
with open('$OUTPUT_DIR/samples.json') as f:
    samples = json.load(f)
print(f'   样本数: {len(samples)}')
"
        fi
        ;;
    
    help|--help|-h)
        echo "CLIP4MC YouTube 视频下载脚本"
        echo ""
        echo "使用方法:"
        echo "  $0 [命令] [样本数] [选项]"
        echo ""
        echo "命令:"
        echo "  download, test  下载 test 集视频 (默认，支持断点续传)"
        echo "  train           下载 train 集视频"
        echo "  preprocess      预处理已下载的视频"
        echo "  all             下载并预处理"
        echo "  restart         清除检查点重新开始"
        echo "  status          查看下载状态"
        echo "  help            显示帮助"
        echo ""
        echo "选项:"
        echo "  --resolution, -r  分辨率: 360, 480, 720, 1080 (默认: 360)"
        echo "  --fps             最大帧率 (默认: 不限制)"
        echo "  --keep-original   保留原始完整视频 (默认: 切片后删除)"
        echo "  --cookies, -c     cookies 文件路径"
        echo "  --no-resume       不从检查点恢复"
        echo ""
        echo "示例:"
        echo "  $0                              # 下载全部 test 集 (360p)"
        echo "  $0 100                          # 下载 100 个样本"
        echo "  $0 100 --resolution 480         # 使用 480p 分辨率"
        echo "  $0 100 --keep-original          # 保留原始完整视频"
        echo "  $0 100 --resolution 720 --fps 30  # 720p, 最大30fps"
        echo "  $0 all 50                       # 下载 50 个样本并预处理"
        echo "  $0 preprocess                   # 仅预处理"
        echo "  $0 restart 100                  # 清除进度，重新下载"
        echo "  $0 status                       # 查看当前下载状态"
        echo ""
        echo "环境变量:"
        echo "  COOKIES_FILE  cookies 文件路径 (默认: data/www.youtube.com_cookies.txt)"
        echo "  OUTPUT_DIR    输出目录 (默认: data/raw_videos/clip4mc_youtube)"
        echo "  RESOLUTION    默认分辨率 (默认: 360)"
        echo "  PYTHON        Python 解释器 (默认: python)"
        ;;
    
    *)
        # 如果第一个参数是数字，当作 max_samples
        if [[ "$COMMAND" =~ ^[0-9]+$ ]]; then
            echo "============================================================"
            echo "📥 下载 $COMMAND 个 test 样本"
            echo "============================================================"
            echo "  📐 分辨率: ${RESOLUTION}p"
            if [[ "$KEEP_ORIGINAL" == "true" ]]; then
                echo "  💾 原始视频: 保留"
            else
                echo "  💾 原始视频: 切片后删除"
            fi
            echo "============================================================"
            
            CMD="$PYTHON -m src.utils.clip4mc_downloader download --dataset test --max-samples $COMMAND --cookies $COOKIES_FILE $DOWNLOAD_ARGS"
            echo "🚀 执行: $CMD"
            echo ""
            eval $CMD
        else
            echo "❌ 未知命令: $COMMAND"
            echo "使用 '$0 help' 查看帮助"
            exit 1
        fi
        ;;
esac

echo ""
echo "完成!"

