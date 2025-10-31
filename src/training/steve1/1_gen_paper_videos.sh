#!/bin/bash
# For some reason, MineRL kills the program unpredictably when we instantiate a couple of environments.
# A simple solution is to run the run_agent in an infinite loop and have the python script only generate videos
# that are not already present in the output directory. Then, whenever this error happens, the python script will
# exit with a non-zero exit code, which will cause the bash script to restart the python script.
# When it finishes all videos, it should exit with a zero exit code, which will cause the bash script to exit.

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# 设置 Hugging Face 缓存路径（避免在线下载）
export HF_HOME="$PROJECT_ROOT/data/huggingface_cache"
export HF_HUB_OFFLINE=1

# 是否显示游戏窗口（设置为 --render 启用，留空禁用）
# 启用会降低速度但可以看到实时画面
#RENDER_FLAG=""
RENDER_FLAG="--render"  # 取消注释这行以启用渲染

COMMAND="python $SCRIPT_DIR/run_agent/run_agent.py \
    --in_model $PROJECT_ROOT/data/weights/vpt/2x.model \
    --in_weights $PROJECT_ROOT/data/weights/steve1/steve1.weights \
    --prior_weights $PROJECT_ROOT/data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --visual_cond_scale 7.0 \
    --gameplay_length 3000 \
    --save_dirpath $PROJECT_ROOT/data/generated_videos/paper_prompts \
    $RENDER_FLAG"

# Run the command and get its exit status
$COMMAND
EXIT_STATUS=$?

# Keep running the command until the exit status is 0 (generates all videos)
while [ $EXIT_STATUS -ne 0 ]; do
    echo
    echo "Encountered an error (likely internal MineRL error), restarting (will skip existing videos)..."
    echo "NOTE: If not MineRL error, then there might be a bug or the parameters might be wrong."
    sleep 10
    $COMMAND
    EXIT_STATUS=$?
done
echo "Finished generating all videos."