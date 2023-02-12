# DSC180B A08: Imitating Behavior to Understand the Brain

Scott Yang, Daniel Son, Akshay Murali, Adam Lee, Eric Leonardis, Talmo Pereira

#### Repos combined:

- https://github.com/danielcson/dsc_capstone_q1
- https://github.com/scott-yj-yang/180A-codebase
- https://github.com/akdec00/DSC-180A
- https://github.com/jimzers/dsc180-ma5

#### DSMLP Spawning Script

To train the model using UC San Diego's Data Science & Machine Learning Platform (DSMLP), you can setup a training environment in the following commands

```bash
# on dsmlp's bash
IDENTITY_PROXY_PORTS=1 launch-scipy-ml.sh -b -j -g 1 -m 32 -c 10 -i scottyang17/dm:latest
```
Explanation: `IDENTITY_PROXY_PORTS=1` allow DSMLP's proxy port forwarding another empty port for the sake of record keeping interfaces such as `tensorboard`. `-b` means run the pod in background mode, `-j` means launch Jupyter notebook server within container (default), `-i` means custom docker image name.

#### Trainning Procedures

Entrypoint: Train expert SAC agent with `train_cheetah.py`
```bash
python train_cheetah.py --automatic_entropy_tuning=True
```

TODO: Extract activations
```bash
python extract_activations.py --policy=path/to/policy
```

TODO: Collect expert data
```bash
python collect_expert.py --expert_policy=path/to/expert_policy --dataset_path=path/to/dataset
```

TODO: Train behavioral cloning agent
```bash
python train_bc.py --dataset_path=path/to/dataset
```

TODO: Run analysis
```bash
python run_analysis --policy=path/to/policy --analysis_path=path/to/analysis 
```
