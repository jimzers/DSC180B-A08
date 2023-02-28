### For collecting experts
# usage:
# bash scripts/collect_activations.sh

## First, activate environment
source venv/bin/activate

## Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/jimzers/research/DSC180B-A08/"

# Collect le activations
python src/run/extract_activations.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000
