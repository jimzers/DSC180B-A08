### For collecting experts
# usage:
# bash scripts/collect_activations.sh

## First, activate environment
source venv/bin/activate

## Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/jimzers/research/DSC180B-A08/"

# Collect le activations

# No noise
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_nonoise --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_nonoise_bcmodel

# Action noise 0.05
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_actnoise005 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise005_bcmodel

# Action noise 0.1
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_actnoise010 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise010_bcmodel

# Action noise 0.2
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_actnoise020 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise020_bcmodel

# Action noise 0.4
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_actnoise040 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise040_bcmodel

# Action noise 0.8
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_actnoise080 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise080_bcmodel

# Action noise 1.6
python src/run/extract_activations_bc.py --model_path data/bc_model/cheetah_123456_10000_actnoise160 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise160_bcmodel
