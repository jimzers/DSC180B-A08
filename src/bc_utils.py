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