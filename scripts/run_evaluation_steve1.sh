#!/bin/bash
# Change custom_text_prompt to whatever text prompt you want to generate a video for

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 设置 Hugging Face 缓存路径（避免在线下载）
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export HF_HUB_OFFLINE=1

# 是否显示游戏窗口（设置为 --render 启用，留空禁用）
# 启用会降低速度但可以看到实时画面
RENDER_FLAG="--render"  # 取消注释这行以启用渲染

python "$PROJECT_ROOT/src/training/steve1/run_agent/run_agent.py" \
--in_model "$PROJECT_ROOT/data/weights/vpt/2x.model" \
--in_weights "$PROJECT_ROOT/data/weights/steve1/steve1.weights" \
--prior_weights "$PROJECT_ROOT/data/weights/steve1/steve1_prior.pt" \
--text_cond_scale 6.0 \
--visual_cond_scale 7.0 \
--gameplay_length 2000 \
--save_dirpath "$PROJECT_ROOT/data/generated_videos/custom_text_prompt" \
--custom_text_prompt "craft wooden axe" \
$RENDER_FLAG