### Step 4: Define a network that you'll train through behavioral cloning (supervised learning)

import torch
import torch.nn as nn
import torch.nn.functional as F


# BCNetwork but with a discrete action space
class BCNetworkDiscrete(nn.Module):
    def __init__(self, obs_dim, action_dim):
        # assumes that observation and action are one-dimensional
        super(BCNetworkDiscrete, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # map to between 0 and 1
        return F.softmax(x, dim=1)


# BCNetwork but with a continuous action space
class BCNetworkContinuous(nn.Module):
    def __init__(self, obs_dim, action_dim):
        # assumes that observation and action are one-dimensional
        super(BCNetworkContinuous, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # map to between -1 and 1
        return torch.tanh(x)
