import numpy as np
import torch
from src.environment import get_flat_obs


### Step 1: Make a data structure to store transitions.
### At the very least, you'll need observations and actions.

# example implementation
class TransitionStorage:
    def __init__(self):
        self.obs = []
        self.action = []

    def store_transition(self, obs, action):
        self.obs.append(obs)
        self.action.append(action)


### Step 2: Reuse the evaluate function, but now use it to collect data

def evaluate_and_collect(model, storage, num_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            next_obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            # store the transition
            storage.store_transition(obs, action)

            obs = next_obs

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


### Step 6: run the trained network on the environment, based on the evaluate function but using network instead of model
def evaluate_network(network, num_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action = network(torch.tensor(obs, dtype=torch.float32)).argmax().item()
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step([action])
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


def evaluate_network_mujoco(network, env, num_episodes=10):
    """
    Evaluate a RL agent in a MuJoCo environment
    """

    all_episode_rewards = []

    # run the loop
    for i in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():  # or env.get_termination()
            # get the state
            state = get_flat_obs(time_step)
            # sample an action
            action = network(torch.tensor(state, dtype=torch.float32)).detach().numpy()
            time_step = env.step(action)

            # record reward
            episode_reward += time_step.reward

        all_episode_rewards.append(episode_reward)

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


def evaluate_network_mujoco_stochastic(network, env, num_episodes=10):
    """
    Evaluate a RL agent in a MuJoCo environment
    """

    all_episode_rewards = []

    # run the loop
    for i in range(num_episodes):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():  # or env.get_termination()
            # get the state
            state = get_flat_obs(time_step)

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # sample an action
            _, _, action = network.sample(state)
            action = action.detach().numpy()

            time_step = env.step(action)

            # record reward
            episode_reward += time_step.reward

        all_episode_rewards.append(episode_reward)

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward