"""
Train a behavior cloning model

Sample usage:
python src/run/train_bc.py --rollout_path data/rollouts/cheetah_123456_10000/rollouts.pkl --save_path data/bc_model/cheetah_123456_10000 --epochs 10 --batch_size 32 --lr 3e-4
"""
import os
import pickle
import argparse
import torch
import torch.nn as nn

import tree
from acme import wrappers
from dm_control import suite

from src.environment import NormilizeActionSpecWrapper, MujocoActionNormalizer
from src.bc_net import BCNetworkContinuous
from src.bc_utils import evaluate_network_mujoco

if __name__ == '__main__':
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_path',
                        type=str,
                        default='data/rollouts/cheetah_123456_10000/rollouts.pkl',
                        help='path to rollouts')
    parser.add_argument('--save_path',
                        type=str,
                        default='data/bc_model/cheetah_123456_10000',
                        help='path to save bc model')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        help='learning rate')
    parser.add_argument('--env_name',
                        type=str,
                        default='HalfCheetah-v4',
                        help='name of environment')

    # parse args
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the environment
    if args.env_name == 'HalfCheetah-v4':
        env = suite.load(domain_name="cheetah", task_name="run")
    else:
        raise NotImplementedError
    # add wrappers onto the environment
    env = NormilizeActionSpecWrapper(env)
    env = MujocoActionNormalizer(environment=env, rescale='clip')
    env = wrappers.SinglePrecisionWrapper(env)

    # get the dimensionality of the observation_spec after flattening
    flat_obs = tree.flatten(env.observation_spec())
    # combine all the shapes
    obs_dim = sum([item.shape[0] for item in flat_obs])

    # load the rollouts
    with open(args.rollout_path, 'rb') as f:
        rollouts = pickle.load(f)

    ### Step 5: Train the network

    # initialize the network
    network = BCNetworkContinuous(obs_dim, env.action_spec().shape[0]).to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # define the loss function
    loss_fn = nn.MSELoss().to(device)

    # define the number of epochs
    num_epochs = args.epochs

    # define the batch size
    batch_size = args.batch_size

    # define the number of batches
    num_batches = len(rollouts.obs) // batch_size

    # convert the data to tensors
    obs = torch.tensor(rollouts.obs, dtype=torch.float32).squeeze().to(device)
    action = torch.tensor(rollouts.action, dtype=torch.float32).squeeze().to(device)

    # train the network
    for epoch in range(num_epochs):
        # accumulate loss
        epoch_loss = 0
        for batch in range(num_batches):
            # get the batch
            batch_obs = obs[batch * batch_size: (batch + 1) * batch_size]
            batch_action = action[batch * batch_size: (batch + 1) * batch_size]

            # forward pass
            logits = network(batch_obs)

            # compute the loss
            loss = loss_fn(logits, batch_action)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accumulate loss
            epoch_loss += loss.detach().cpu().item()

        # print the loss
        print("Epoch: {}, Loss: {}".format(epoch, epoch_loss / num_batches))

    # make the directory to save the model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # save the model
    torch.save(network.state_dict(), args.save_path)

    # evaluate the model
    mean_episode_reward = evaluate_network_mujoco(network, env, num_episodes=10)

    print("Done training behavior cloning model!")
