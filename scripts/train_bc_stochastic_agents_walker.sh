### For training stochastic behavior cloning agents
# usage:
# bash scripts/train_bc_stochastic_agents_walker.sh

## First, activate environment
source indvenv/bin/activate

## Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/talmolab/research/DSC180B-A08/"


# no noise
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_nonoise/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_nonoise/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_nonoise/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0

# action noise 0.8
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise080/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise080/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise080/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0

# action noise 0.05
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise005/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise005/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise005/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0

# action noise 0.1
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise010/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise010/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise010/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0

# action noise 0.2
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise020/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise020/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise020/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0

# action noise 0.4
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise040/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise040/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise040/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0

# action noise 1.6
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise160/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise160/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses mse entropy --loss_scaling 1.0 0.001
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/walker_actnoise160/rollouts.pkl --save_path data/bc_stochastic_model_walker --env_name Walker-v1 --epochs 100 --batch_size 512 --lr 3e-4 --losses entropy --loss_scaling 1.0
