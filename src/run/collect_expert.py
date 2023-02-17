"""
Collect expert data for rollouts

Sample usage:
python src/run/collect_expert.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 15 --save_path data/rollouts/cheetah_123456_10000
"""
# imports
import os
import tree
import argparse
from acme import wrappers
from dm_control import suite

# in-house imports
from src.sac import SAC
from src.environment import NormilizeActionSpecWrapper, MujocoActionNormalizer, get_flat_obs

import pickle
from src.bc_utils import TransitionStorage

if __name__ == '__main__':
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default='data/models/sac_checkpoint_cheetah_123456_10000',
                        help='path to model')
    parser.add_argument('--env_name',
                        type=str,
                        default='HalfCheetah-v4',
                        help='name of environment')
    parser.add_argument('--num_episodes',
                        type=int,
                        default=15, help='number of episodes to collect')
    parser.add_argument('--save_path',
                        type=str,
                        default='data/rollouts/cheetah_123456_10000',
                        help='path to save rollouts')

    # parse args
    args = parser.parse_args()

    # load the environment
    if args.env_name == 'HalfCheetah-v4':
        env = suite.load(domain_name="cheetah", task_name="run")
    else:
        raise NotImplementedError
    # add wrappers onto the environment
    env = NormilizeActionSpecWrapper(env)
    env = MujocoActionNormalizer(environment=env, rescale='clip')
    env = wrappers.SinglePrecisionWrapper(env)


    class ModelArgs:
        env_name = 'whatever'
        policy = 'Gaussian'
        eval = True
        gamma = 0.99
        tau = 0.005
        lr = 0.0003
        alpha = 0.2
        automatic_entropy_tuning = True
        seed = 42
        batch_size = 256
        num_steps = 1000000
        hidden_size = 256
        updates_per_step = 1
        start_steps = 10000
        target_update_interval = 1
        replay_size = 1000000
        cuda = False


    model_args = ModelArgs()

    # get the dimensionality of the observation_spec after flattening
    flat_obs = tree.flatten(env.observation_spec())
    # combine all the shapes
    obs_dim = sum([item.shape[0] for item in flat_obs])

    # setup agent
    agent = SAC(obs_dim, env.action_spec(), model_args)
    # load checkpoint - UPLOAD YOUR FILE HERE!
    agent.load_checkpoint(args.model_path, evaluate=True)

    # setup storage
    storage = TransitionStorage()

    # run a few episodes just to collect activations
    num_episodes_to_run = args.num_episodes

    # run the loop
    for i in range(num_episodes_to_run):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():  # or env.get_termination()
            # get the state
            state = get_flat_obs(time_step)
            # sample an action
            action = agent.select_action(state)
            time_step = env.step(action)

            # record reward
            episode_reward += time_step.reward

            # store the transition
            storage.store_transition(state, action)

        print('Episode: {} Reward: {}'.format(i, episode_reward))

    # make folder from args.save_path
    os.makedirs(args.save_path, exist_ok=True)

    # pickle the storage
    with open(os.path.join(args.save_path, 'rollouts.pkl'), 'wb') as f:
        pickle.dump(storage, f)

    print('Done collecting rollouts!')
