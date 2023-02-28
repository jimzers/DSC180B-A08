### For collecting experts
# usage:
# bash scripts/collect_experts.sh

## First, activate environment
source venv/bin/activate

## Next, collect expert data
export PYTHONPATH="${PYTHONPATH}:/home/jimzers/research/DSC180B-A08/"

# no action noise
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_nonoise

# action noise 0.05
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise005 --act_noise 0.05

# action noise 0.1
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise010 --act_noise 0.1

# action noise 0.2
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise020 --act_noise 0.2

# action noise 0.4
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise040 --act_noise 0.4

# action noise 0.6
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise060 --act_noise 0.6

# action noise 0.8
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise080 --act_noise 0.8

# action noise 1.6
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 100 --save_path data/rollouts/cheetah_123456_10000_actnoise160 --act_noise 1.6
