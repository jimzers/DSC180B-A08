### For training behavior cloning agents
# usage:
# bash scripts/train_bc_agents.sh

## First, activate environment
source venv/bin/activate

## Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/jimzers/research/DSC180B-A08/"

# Train BC agent
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_nonoise/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_nonoise --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 0.05
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise005/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise005 --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 0.1
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise010/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise010 --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 0.2
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise020/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise020 --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 0.4
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise040/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise040 --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 0.6
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise060/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise060 --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 0.8
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise080/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise080 --epochs 50 --batch_size 64 --lr 3e-4

# Train BC agent with action noise 1.6
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000_actnoise160/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000_actnoise160 --epochs 50 --batch_size 64 --lr 3e-4
