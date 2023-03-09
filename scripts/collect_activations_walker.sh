 ### For collecting experts
# usage:
# bash scripts/collect_activations_walker.sh

## First, activate environment
source indvenv/bin/activate

## Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/talmolab/research/DSC180B-A08/"

# Collect le activations
python src/run/extract_activations.py --model_path data/models/sac_checkpoint_walker_walk_batch512_hidden1024_1123_500 --env_name Walker-v1 --num_episodes 100 --save_path data/activations_walker/walker_rl
