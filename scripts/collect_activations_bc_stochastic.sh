### For collecting experts
# usage:
# bash scripts/collect_activations.sh

## First, activate environment
source indvenv/bin/activate

## Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/talmolab/research/DSC180B-A08/"

# Collect le activations

# No noise
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_nonoise_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_nonoise_bcmodel_stochastic

# Action noise 0.05
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_actnoise005_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise005_bcmodel_stochastic

# Action noise 0.1
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_actnoise010_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise010_bcmodel_stochastic

# Action noise 0.2
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_actnoise020_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise020_bcmodel_stochastic

# Action noise 0.4
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_actnoise040_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise040_bcmodel_stochastic

# Action noise 0.8
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_actnoise080_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise080_bcmodel_stochastic

# Action noise 1.6
python src/run/extract_activations_bc_stochastic.py --model_path data/bc_stochastic_model/bc_cheetah_123456_10000_actnoise160_mse_entropy.pt --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/activations/cheetah_123456_10000_actnoise160_bcmodel_stochastic
