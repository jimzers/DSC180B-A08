### Define a network that you'll train through behavioral cloning (supervised learning)

import torch
import torch.nn as nn
import torch.nn.functional as F

# BCNetwork with continuous action space
class BCNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        # assumes that observation and action are one-dimensional
        super(BCNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, obs, device="cpu"):
        x = F.relu(self.fc1(obs.to(device)))
        x = F.relu(self.fc2(x))
        x = self.mean_linear(x)
        return torch.tanh(x)