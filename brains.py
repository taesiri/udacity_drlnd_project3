import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MADDPGActor(nn.Module):
  """An Actor in MADDPG Algorithm"""
  def __init__(self, state_size=24, action_size=2, seed=0, hidden_sizes=[64, 64]):
    """Initialize parameters and build model.
    Params
    ======
    state_size (int): Dimension of each state
    action_size (int): Dimension of each action
    hidden_sizes (list): List of Hidden Layers' size
    seed (int): Random seed
    """
    super(MADDPGActor, self).__init__()
    self.seed = torch.manual_seed(seed)

    self.state_size = state_size
    self.action_size = action_size
    self.bn1 = nn.BatchNorm1d(hidden_sizes[0])

    # A Generic Fully Connection Network.
    layers = zip([state_size] + hidden_sizes, hidden_sizes + [action_size])
    self.fcs = [nn.Linear(h_size[0], h_size[1]) for h_size in layers]
    self.fcs = nn.ModuleList(self.fcs) 

  def forward(self, state):
    """Build a network that maps state -> action values."""
    x = state
    # x = self.bn1(state)
    x = F.leaky_relu(self.bn1(self.fcs[0](x))) # batchnorm only on first H output

    for layer in self.fcs[1:-1]:
      x = F.leaky_relu(layer(x))
    return torch.tanh(self.fcs[-1](x))

class MADDPGCentralCriticNetwork(nn.Module):
  """A Centeralized Q Network"""
  def __init__(self, state_size=24, action_size=2, seed=0, hidden_sizes = [128, 128, 64]):
    """Initialize parameters and build model.
    Params
    ======
      state_size (int): Dimension of each state
      action_size (int): Dimension of each action
      hidden_sizes (list): List of Hidden Layers' size
      number_of_agents (int): Number of Agents in the System
      seed (int): Random seed
    """
    super(MADDPGCentralCriticNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)

    self.state_size = state_size
    self.action_size = action_size
    # self.input_size = (state_size+action_size) * 2

    self.bn1 = nn.BatchNorm1d(128)
    # self.bn2 = nn.BatchNorm1d(state_size)

    self.preprocess_fcStates = nn.Linear(state_size*2, 128)
    # self.preprocess_fc12 = nn.Linear(action_size, 64)

    # self.preprocess_fc21 = nn.Linear(state_size, 64)
    # self.preprocess_fc22 = nn.Linear(action_size, 64)

    # A Generic Fully Connection Network.
    layers = zip([128+2+2] + hidden_sizes, hidden_sizes + [1])
    self.fcs = [nn.Linear(h_size[0], h_size[1]) for h_size in layers]
    self.fcs = nn.ModuleList(self.fcs) 

  def forward(self, states, actions):
    """Build a network that maps (X, A) to values."""
    # states = states.view(-1, 2, 24)
    states = states.view(-1, 48)
    # state1 = states[:, 0, :]
    # state2 = states[:, 1, :]

    # actions = actions.view(-1, 2, 2)
    actions = actions.view(-1, 4)
    # action1 = actions[:, 0, :]
    # action2 = actions[:, 1, :]

    # print('state1', state1.shape)
    # print('state2', state2.shape)
    # print('action1', action1.shape)
    # print('action2', action2.shape)

    # action1 = F.leaky_relu(self.preprocess_fc12(action1))

    # state2 = F.leaky_relu(self.preprocess_fc21(self.bn2(state2)))
    # action2 = F.leaky_relu(self.preprocess_fc22(action2))

    # print('AFTER')

    # print('state1', state1.shape)
    # print('state2', state2.shape)
    # print('action1', action1.shape)
    # print('action2', action2.shape)

    # x = torch.cat([state1, action1, state2, action2], axis = 1)

    states = F.leaky_relu(self.bn1(self.preprocess_fcStates(states)))

    x = torch.cat([states, actions], 1)

    for layer in self.fcs[:-1]:
      x = F.leaky_relu(layer(x))
    return self.fcs[-1](x)