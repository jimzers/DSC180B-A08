"""
Train a behavior cloning model

Sample usage:
python src/run/train_bc_stochastic.py --rollout_path data/rollouts/cheetah_123456_10000_nonoise/rollouts.pkl --save_path data/bc_stochastic_model --epochs 10 --batch_size 32 --lr 3e-4 --losses mse kl entropy --loss_scaling 1.0 0.001 0.001
"""
import os
import pickle
import argparse
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import tree
from acme import wrappers
from dm_control import suite

from src.environment import NormilizeActionSpecWrapper, MujocoActionNormalizer
from src.sac import GaussianPolicy
from src.bc_net import BCNetworkContinuous
from src.bc_utils import evaluate_network_mujoco_stochastic

if __name__ == '__main__':
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollout_path',
                        type=str,
                        default='data/rollouts/cheetah_123456_10000_nonoise/rollouts.pkl',
                        help='path to rollouts')
    parser.add_argument('--save_path',
                        type=str,
                        default='data/bc_stochastic_model',
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
    parser.add_argument('--losses',
                        nargs='+',
                        type=str,
                        default=['mse', 'kl', 'entropy'],
                        help='losses to use')
    parser.add_argument('--loss_scaling',
                        nargs='+',
                        type=float,
                        default=[1.0, 0.001, 0.001],
                        help='scaling for each loss')


    # parse args
    args = parser.parse_args()

    loss_names = tuple(args.losses)
    loss_scaling = tuple(args.loss_scaling)
    assert len(loss_names) == len(loss_scaling), 'loss names and scaling must be the same length'
    loss_name_weight_dict = dict(zip(loss_names, loss_scaling))

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

    network = GaussianPolicy(obs_dim, env.action_spec().shape[0], hidden_dim=256).to(device)

    guide_dist = torch.distributions.Normal(torch.zeros(env.action_spec().shape[0]).to(device),
                                            torch.ones(env.action_spec().shape[0]).to(device))

    # define the optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    # define the loss function
    mse_loss_fn = nn.MSELoss().to(device)

    # define the number of epochs
    num_epochs = args.epochs

    # define the batch size
    batch_size = args.batch_size

    # define the number of batches
    num_batches = len(rollouts.obs) // batch_size

    # convert the data to tensors
    obs = torch.tensor(rollouts.obs, dtype=torch.float32).squeeze().to(device)
    action = torch.tensor(rollouts.action, dtype=torch.float32).squeeze().to(device)

    # train the network with reparametrization trick

    total_mse_loss_arr = []
    total_kl_div_arr = []
    total_entropy_loss_arr = []
    total_loss_arr = []

    for epoch in range(num_epochs):
        epoch_mse_loss_arr = []
        epoch_kl_div_arr = []
        epoch_entropy_loss_arr = []
        epoch_loss_arr = []

        for batch in range(num_batches):
            # get the batch
            batch_obs = obs[batch * batch_size:(batch + 1) * batch_size]
            batch_action = action[batch * batch_size:(batch + 1) * batch_size]

            # print(batch_obs.shape)

            # sample from the network
            sampled_action, log_prob, mean = network.sample(batch_obs)

            # compute the loss
            loss = torch.tensor(0.0, dtype=torch.float32).to(device)

            # compute the mse loss
            if 'mse' in loss_names:
                mse_loss = mse_loss_fn(sampled_action, batch_action)
                loss += mse_loss * loss_name_weight_dict['mse']

            # compute the kl divergence
            if 'kl' in loss_names:
                guide_log_prob = guide_dist.log_prob(sampled_action)
                kl_div = torch.mean(log_prob - guide_log_prob)
                loss += kl_div * loss_name_weight_dict['kl']

            # compute the entropy
            if 'entropy' in loss_names:
                entropy = torch.mean(-log_prob)
                loss += entropy * loss_name_weight_dict['entropy']


            # backpropagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the losses
            if 'mse' in loss_names:
                epoch_mse_loss_arr.append(mse_loss.detach().cpu().item())
            if 'kl' in loss_names:
                epoch_kl_div_arr.append(kl_div.detach().cpu().item())
            if 'entropy' in loss_names:
                epoch_entropy_loss_arr.append(entropy.detach().cpu().item())
            epoch_loss_arr.append(loss.detach().cpu().item())

        # log the losses
        total_mse_loss_arr.append(np.mean(epoch_mse_loss_arr))
        total_kl_div_arr.append(np.mean(epoch_kl_div_arr))
        total_entropy_loss_arr.append(np.mean(epoch_entropy_loss_arr))
        total_loss_arr.append(np.mean(epoch_loss_arr))

        # print the loss
        print(f'Epoch: {epoch + 1}, Loss: {np.mean(epoch_loss_arr).item():.4f}')

    # make the directory to save the model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # save the model
    loss_name_portion = '_'.join(loss_names)
    # from rollout_path, get the dataset name
    dataset_name = args.rollout_path.split('/')[-2]

    model_filename = f'{args.save_path}/bc_{dataset_name}_{loss_name_portion}.pt'
    print('saving model to: ', model_filename)
    # save the network
    torch.save(network.state_dict(), model_filename)

    # evaluate the model
    mean_episode_reward = evaluate_network_mujoco_stochastic(network, env, num_episodes=10, device=device)

    # plot the distribution of the kl_div, mse_loss, entropy_loss, loss with shared x axis but separate y axes
    fig, ax = plt.subplots(len(loss_names) + 1, 1, sharex=True, figsize=(8, 8*(len(loss_names) + 1)))
    # if len(loss_names) == 1:
    #     ax = [ax]
    for i, loss_name in enumerate(loss_names):
        if loss_name == 'mse':
            ax[i].plot(total_mse_loss_arr, label='mse loss')
        elif loss_name == 'kl':
            ax[i].plot(total_kl_div_arr, label='kl divergence')
        elif loss_name == 'entropy':
            ax[i].plot(total_entropy_loss_arr, label='entropy loss')
        ax[i].set_ylabel(loss_name)
        ax[i].legend()

    # plot the total loss in the last subplot
    ax[-1].set_xlabel('epoch')
    ax[-1].set_ylabel('loss')
    ax[-1].plot(total_loss_arr, label='total loss')
    ax[-1].legend()
    plt.savefig(f'{args.save_path}/bc_{dataset_name}_{loss_name_portion}_losscurves.png')
    plt.close()

    print("Done training stochastic behavior cloning model!")
