# DSC180B A08: Imitating Behavior to Understand the Brain

Scott Yang, Daniel Son, Akshay Murali, Adam Lee, Eric Leonardis, Talmo Pereira

#### Repos combined:

- https://github.com/danielcson/dsc_capstone_q1
- https://github.com/scott-yj-yang/180A-codebase
- https://github.com/akdec00/DSC-180A
- https://github.com/jimzers/dsc180-ma5

#### DSMLP Spawning Script

To train the model using UC San Diego's Data Science & Machine Learning Platform (DSMLP), you can setup a training
environment in the following commands

```bash
# on dsmlp's bash
IDENTITY_PROXY_PORTS=1 launch-scipy-ml.sh -b -j -g 1 -m 32 -c 10 -i scottyang17/dm:latest
```

Explanation: `IDENTITY_PROXY_PORTS=1` allow DSMLP's proxy port forwarding another empty port for the sake of record
keeping interfaces such as `tensorboard`. `-b` means run the pod in background mode, `-j` means launch Jupyter notebook
server within container (default), `-i` means custom docker image name.

#### Training Procedures

Entrypoint: Train expert SAC agent with `train_cheetah.py`

```bash
python train_cheetah.py --automatic_entropy_tuning=True
```

Extract activations

```bash
python src/run/extract_activations.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 1000 --save_path data/activations/cheetah_123456_10000
```

Collect expert data

```bash
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 15 --save_path data/rollouts/cheetah_123456_10000
```

Train behavioral cloning agent

```bash
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000 --epochs 10 --batch_size 32 --lr 3e-4
```

TODO: Run analysis

```bash
python run_analysis --policy=path/to/policy --analysis_path=path/to/analysis 
```
