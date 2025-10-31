#!/bin/bash
# Interactive runs in text mode only, you can click on the window to pause and type in a new prompt

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# 设置 Hugging Face 缓存路径（避免在线下载）
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/data/huggingface_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "$SCRIPT_DIR/run_agent/run_interactive.py" \
--in_model "$PROJECT_ROOT/data/weights/vpt/2x.model" \
--in_weights "$PROJECT_ROOT/data/weights/steve1/steve1.weights" \
--prior_weights "$PROJECT_ROOT/data/weights/steve1/steve1_prior.pt" \
--output_video_dirpath "$PROJECT_ROOT/data/generated_videos/interactive_videos" \
--cond_scale 6.0
