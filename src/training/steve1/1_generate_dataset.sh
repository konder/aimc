#!/bin/bash
# This will download the dataset from OpenAI to a local directory and convert it to the format used by the code.

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

OUTPUT_DIR_CONTRACTOR="$PROJECT_ROOT/data/dataset_contractor/"
# IMPORTANT: Increase the value of N_EPISODES_CONTRACTOR if you want to
# download more contractor episodes from the publicly available VPT dataset.
N_EPISODES_CONTRACTOR=5

python "$SCRIPT_DIR/data/generation/convert_from_contractor.py" \
--batch_size 32 \
--num_episodes $N_EPISODES_CONTRACTOR \
--worker_id 0 \
--output_dir $OUTPUT_DIR_CONTRACTOR \
--index 8.x

python "$SCRIPT_DIR/data/generation/convert_from_contractor.py" \
--batch_size 32 \
--num_episodes $N_EPISODES_CONTRACTOR \
--worker_id 0 \
--output_dir $OUTPUT_DIR_CONTRACTOR \
--index 9.x

python "$SCRIPT_DIR/data/generation/convert_from_contractor.py" \
--batch_size 32 \
--num_episodes $N_EPISODES_CONTRACTOR \
--worker_id 0 \
--output_dir $OUTPUT_DIR_CONTRACTOR \
--index 10.x

# This will generate mixed agent episodes in the format used by the code.

OUTPUT_DIR_MIXED_AGENTS="$PROJECT_ROOT/data/dataset_mixed_agents/"
N_EPISODES_MIXED_AGENTS=5

python "$SCRIPT_DIR/data/generation/gen_mixed_agents.py" \
--output_dir $OUTPUT_DIR_MIXED_AGENTS \
--max_timesteps 7200 \
--min_timesteps 1000 \
--switch_agent_prob 0.001 \
--perform_spin_prob 0.00133 \
--batch_size 4 \
--num_episodes $N_EPISODES_MIXED_AGENTS

