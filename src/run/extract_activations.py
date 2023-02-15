"""
Extract activations

Sample usage:
python src/run/extract_activations.py --model_path data/models/sac_checkpoint_cheetah_123456_10000 --env_name HalfCheetah-v4 --num_episodes 1000 --save_path data/activations/cheetah_123456_10000
"""
# imports
import os
import tree
import argparse
import torch
from acme import wrappers
from dm_control import suite
import numpy as np

# in-house imports
from src.sac import SAC
from src.environment import NormilizeActionSpecWrapper, MujocoActionNormalizer, get_flat_obs
from src.activations import init_hook_dict, recordtodict_hook, get_kinematics, save_hook_dict

GEOM_NAMES = ['ground', 'torso', 'head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
JOINT_NAMES = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
ACTUATOR_NAMES = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']

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
                        default=1000, help='number of episodes to collect')
    parser.add_argument('--save_path',
                        type=str,
                        default='data/activations/cheetah_123456_10000',
                        help='path to save activations')

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
    agent = SAC(obs_dim, env.action_spec(), args)
    # load checkpoint - UPLOAD YOUR FILE HERE!
    agent.load_checkpoint(args.model_path, evaluate=True)

    # pull out model
    model = agent.policy
    # setup hook dict
    hook_dict = init_hook_dict(model)
    # add hooks
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(recordtodict_hook(name=name, hook_dict=hook_dict))

    # collect activations and kinematics

    # get the mapping of the geom names
    geom_names_to_idx = {geom_name: idx for idx, geom_name in enumerate(GEOM_NAMES)}
    # get the mapping of the joint names
    joint_names_to_idx = {joint_name: idx for idx, joint_name in enumerate(JOINT_NAMES)}
    # get the mapping of the actuator names
    actuator_names_to_idx = {actuator_name: idx for idx, actuator_name in enumerate(ACTUATOR_NAMES)}

    idx_to_joint_names = {idx: joint_name for joint_name, idx in joint_names_to_idx.items()}
    idx_to_actuator_names = {idx: actuator_name for actuator_name, idx in actuator_names_to_idx.items()}
    idx_to_geom_names = {idx: geom_name for geom_name, idx in geom_names_to_idx.items()}

    # run a few episodes just to collect activations
    num_episodes_to_run = 42

    # for recording kinematics
    total_kinematic_dict = {
        'geom_positions': [],
        'joint_angles': [],
        'joint_velocities': [],
        'actuator_forces': []
    }

    for i in range(num_episodes_to_run):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():  # or env.get_termination()
            # get the state
            state = get_flat_obs(time_step)
            # sample an action
            action = agent.select_action(state)
            time_step = env.step(action)

            # record kinematics
            kinematic_dict = get_kinematics(env.physics, GEOM_NAMES, JOINT_NAMES,
                                            ACTUATOR_NAMES)
            total_kinematic_dict['geom_positions'].append(kinematic_dict['geom_positions'])
            total_kinematic_dict['joint_angles'].append(kinematic_dict['joint_angles'])
            total_kinematic_dict['joint_velocities'].append(kinematic_dict['joint_velocities'])
            total_kinematic_dict['actuator_forces'].append(kinematic_dict['actuator_forces'])
            # record reward
            episode_reward += time_step.reward
        print('Episode: {} Reward: {}'.format(i, episode_reward))

    ### optional: save + load the hook_dict

    # make folder from args.save_path
    os.makedirs(args.save_path, exist_ok=True)

    save_path = os.path.join(args.save_path, 'hook_dict.npy')
    save_hook_dict(hook_dict, save_path)

    # process the kinematics - convert the kinematics to numpy arrays
    total_kinematic_dict['geom_positions'] = np.stack(total_kinematic_dict['geom_positions'],
                                                      axis=0)  # combine the geom_positions_arr into (t, n, 3)
    total_kinematic_dict['joint_angles'] = np.array(total_kinematic_dict['joint_angles'])
    total_kinematic_dict['joint_velocities'] = np.array(total_kinematic_dict['joint_velocities'])
    total_kinematic_dict['actuator_forces'] = np.array(total_kinematic_dict['actuator_forces'])

    # save total_kinematic_dict
    save_path = os.path.join(args.save_path, 'kinematics_dict.npy')
    np.save(save_path, total_kinematic_dict)

    print('Done!')
