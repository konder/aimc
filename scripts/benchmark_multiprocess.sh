#!/bin/bash
# 多进程性能基准测试脚本
# 用于测试不同 worker 数量下的性能

set -e

# 配置
TEST_JSON="${1:-data/training/dataset_test.json}"
DOWNLOAD_LOG="${2:-data/training/test2000_formatted.csv}"
VIDEOS_DIR="${3:-/root/autodl-tmp/TEST2000}"
OUTPUT_DIR="/tmp/benchmark_results"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "多进程性能基准测试"
echo "========================================"
echo "测试配置:"
echo "  数据集: $TEST_JSON"
echo "  下载日志: $DOWNLOAD_LOG"
echo "  视频目录: $VIDEOS_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 测试配置（workers 数量）
WORKERS_LIST="1 2 4 8 16 32"

# 结果文件
RESULTS_FILE="$OUTPUT_DIR/benchmark_results.txt"
echo "Workers,Time(s),Speed(clip/s),Mode" > "$RESULTS_FILE"

echo "开始测试..."
echo "========================================"

# 测试 1: 跳过 text 生成（纯视频匹配）
echo ""
echo "测试 1: 跳过 text 生成（纯视频匹配性能）"
echo "----------------------------------------"

for workers in $WORKERS_LIST; do
    echo ""
    echo "测试: $workers workers (无 text 生成)..."
    
    output_file="$OUTPUT_DIR/metadata_skip_${workers}w.json"
    unmatched_file="$OUTPUT_DIR/unmatched_skip_${workers}w.json"
    
    # 记录开始时间
    start_time=$(date +%s.%N)
    
    # 运行测试
    python src/utils/generate_clip4mc_metadata.py \
        --test-json "$TEST_JSON" \
        --download-log "$DOWNLOAD_LOG" \
        --videos-dir "$VIDEOS_DIR" \
        --text-inputs-dir "$OUTPUT_DIR/text_token_skip_${workers}w" \
        --output "$output_file" \
        --unmatched-output "$unmatched_file" \
        --loose-match \
        --num-workers "$workers" \
        --skip-text-generation \
        2>&1 | tail -15
    
    # 记录结束时间
    end_time=$(date +%s.%N)
    
    # 计算耗时
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # 读取匹配数量
    total=$(jq '. | length' "$output_file" 2>/dev/null || echo "0")
    
    # 计算速度
    if [ "$total" -gt 0 ] && [ "$(echo "$elapsed > 0" | bc)" -eq 1 ]; then
        speed=$(echo "scale=2; $total / $elapsed" | bc)
    else
        speed="0"
    fi
    
    echo "  ✅ 完成: ${elapsed}s, ${speed} clip/s"
    echo "$workers,$elapsed,$speed,skip_text" >> "$RESULTS_FILE"
    
    # 清理
    rm -rf "$OUTPUT_DIR/text_token_skip_${workers}w"
    
    # 等待 1 秒
    sleep 1
done

# 测试 2: 包含 text 生成（完整流程）
echo ""
echo ""
echo "测试 2: 包含 text 生成（完整流程性能）"
echo "----------------------------------------"

for workers in $WORKERS_LIST; do
    echo ""
    echo "测试: $workers workers (含 text 生成)..."
    
    output_file="$OUTPUT_DIR/metadata_full_${workers}w.json"
    unmatched_file="$OUTPUT_DIR/unmatched_full_${workers}w.json"
    
    # 记录开始时间
    start_time=$(date +%s.%N)
    
    # 运行测试
    python src/utils/generate_clip4mc_metadata.py \
        --test-json "$TEST_JSON" \
        --download-log "$DOWNLOAD_LOG" \
        --videos-dir "$VIDEOS_DIR" \
        --text-inputs-dir "$OUTPUT_DIR/text_token_full_${workers}w" \
        --output "$output_file" \
        --unmatched-output "$unmatched_file" \
        --loose-match \
        --num-workers "$workers" \
        2>&1 | tail -15
    
    # 记录结束时间
    end_time=$(date +%s.%N)
    
    # 计算耗时
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # 读取匹配数量
    total=$(jq '. | length' "$output_file" 2>/dev/null || echo "0")
    
    # 计算速度
    if [ "$total" -gt 0 ] && [ "$(echo "$elapsed > 0" | bc)" -eq 1 ]; then
        speed=$(echo "scale=2; $total / $elapsed" | bc)
    else
        speed="0"
    fi
    
    echo "  ✅ 完成: ${elapsed}s, ${speed} clip/s"
    echo "$workers,$elapsed,$speed,full" >> "$RESULTS_FILE"
    
    # 清理
    rm -rf "$OUTPUT_DIR/text_token_full_${workers}w"
    
    # 等待 1 秒
    sleep 1
done

# 生成报告
echo ""
echo ""
echo "========================================"
echo "性能测试报告"
echo "========================================"
echo ""

echo "1. 跳过 text 生成（纯视频匹配）:"
echo "----------------------------------------"
printf "%-10s %-12s %-15s\n" "Workers" "Time(s)" "Speed(clip/s)"
echo "----------------------------------------"
grep "skip_text" "$RESULTS_FILE" | while IFS=',' read -r workers time speed mode; do
    printf "%-10s %-12s %-15s\n" "$workers" "$time" "$speed"
done

echo ""
echo "2. 包含 text 生成（完整流程）:"
echo "----------------------------------------"
printf "%-10s %-12s %-15s\n" "Workers" "Time(s)" "Speed(clip/s)"
echo "----------------------------------------"
grep "full" "$RESULTS_FILE" | while IFS=',' read -r workers time speed mode; do
    printf "%-10s %-12s %-15s\n" "$workers" "$time" "$speed"
done

echo ""
echo "========================================"
echo "详细结果已保存到:"
echo "  $RESULTS_FILE"
echo "========================================"

