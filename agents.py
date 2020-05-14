import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy 
import math
from collections import namedtuple, deque
from torch.utils.data import TensorDataset, DataLoader

## Hyper Parameters
BUFFER_SIZE  = int(1e6)  # replay buffer size
BATCH_SIZE   = 128       # minibatch size
GAMMA        = 0.99      # discount factor
TAU1         = 3e-2      # for soft update of target parameters
TAU2         = 3e-2      # for soft update of target parameters
LR_ACTOR     = 1e-3      # learning rate 
LR_CRITIC    = 1e-3      # learning rate 
UPDATE_EVERY = 1         # How often to update the network
EPS_START = 6            # OUNoise level start
EPS_END = 0              # OUNoise level end
EPS_DECAY = 250          # OUNoise Decay Length

# Default device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TwoAgentMADDPGSolver():
  """Controls other Agents and Manage them to compete and collaborate"""

  def __init__(self, agent1, agent2, state_size=24, action_size=2, seed=0):
    """Initialize the Solver.
    
    Params
    ======
      agent1 (DDPQAgent): first agent
      agent2 (DDPQAgent): second agent
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    
    # DDPG Agents, One for each racket
    self.agent1    = agent1
    self.agent2    = agent2

    self.train_mode = True

    # Replay memory, one Shared Memory for both agents
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    # OUNoise - This is very importance, cant' be replace with simple Normal Noise
    self.eps = EPS_START
    self.noise = OUNoise((2, action_size), seed)
  
  def step(self, states, actions, rewards, next_states, isDone):
    # Save experience in replay memory
    # states      2x24
    # actions     2x1x2
    # rewards     2x1
    # next_states 2x24
    # isDone      2x1
    self.memory.add(states, actions, rewards, next_states, isDone)

    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > BATCH_SIZE:
        # Sampling two different batches for each agent
        experiences = [self.memory.sample(), self.memory.sample()]
        self.learn(experiences, GAMMA)

  def reset(self):
    self.noise.reset()

  def act(self, states, add_noise=True):
    """Returns actions for each actor in given state as per current policy.
    
    Params
    ======
      state1 (array_like): current state for first agent
      state2 (array_like): current state for second agent
      add_noise (bool): wether should add noise or not
    """
    # get actions from each Actor!
    actions = []
    for actor, state in zip([self.agent1, self.agent2], states):
      state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      actions.append(actor.act(state))

    # some ugly reshaping!
    actions = np.asarray(actions).reshape(2, 2)

    if add_noise:
      noise   = np.asarray(self.eps * self.noise.sample()).reshape(2, 2)
      actions += noise

    return np.clip(actions, -1, 1)

  def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
      experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor
    """
    # s, a, r, sP, done = 

    # Compute Next Action for each exprience // for each agent
    current_actions_local = []
    next_actions_target = []

    for i, actor in enumerate([self.agent1, self.agent2]):
      S, _, _, SP, _ = experiences[i]
      actions        = actor.get_local_actions(S[:, i, :])
      current_actions_local.append(actions)

      next_action = actor.get_target_actions(SP[:, i, :])
      next_actions_target.append(next_action)

    # call Learn on each agent with s, a, r, s', next_state, done
    for i, actor in enumerate([self.agent1, self.agent2]):
      actor.learn(i, experiences[i], gamma, current_actions_local, next_actions_target)

    #  ----------------------- Update Noise Scale ----------------------- #
    self.eps = self.eps - (1/EPS_DECAY)
    if self.eps < EPS_END:
      self.eps=EPS_END


class DDPQAgent():
  """Interacts with and learns from the environment."""

  def __init__(self, actor_network_local, actor_network_target, critic_network_local, critic_network_target, state_size=24, action_size=2, seed=0):
    """ Deep Deterministic Policy Gradient Agent - Initialize an Agent object.
    
    Params
    ======
      state_size (int): dimension of each state
      action_size (int): dimension of each action
      seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)
    
    # Actor-Critic Networks, Local and Target
    self.actor_local   = actor_network_local
    self.actor_target  = actor_network_target

    self.critic_local  = critic_network_local
    self.critic_target = critic_network_target

    self.optimizerActor = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
    self.optimizerCritic = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
    # Replay memory
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0

  def act(self, state):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (tensor): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """
    self.actor_local.eval() # Set to Evaluation Mode
    with torch.no_grad(): # Dont Store Gradients
      action = self.actor_local(state)
    self.actor_local.train() # Set Back to Train Mode

    action = action.cpu().data.numpy()
    return action

  def get_local_actions(self, state):
    return self.actor_local(state)

  def get_target_actions(self, state):
    return self.actor_target(state)

  def learn(self, agentIndex, experience, gamma, current_actions_local, next_actions_target):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
      agentIndex
      experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor
      current_actions_local
      next_actions_target
    """
    states, actions, rewards, next_states, dones = experience

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print('rewards', rewards.shape)
    # print('next_states', next_states.shape)
    # print('dones', dones.shape)
    # print('current_actions_local', len(current_actions_local))
    # print('next_actions_target', len(next_actions_target))


    # ----------------------- Critic Loss ----------------------- #
    # y(r, s, d) = r + gmma*(1-d)* Q_target(S', Mu_target(S')) - TD TARGET
    # Loss = MSE( Q(S, A) - y(r, s', d) )

    next_actions_target = torch.cat(next_actions_target, dim=1).to(device)
    Q_target = self.critic_target(next_states, next_actions_target).detach()
    Y = rewards[:, agentIndex].unsqueeze(1) + (gamma * ( dones[:, agentIndex].unsqueeze(1) * Q_target))

    Q = self.critic_local(states, actions)
    critic_loss = F.smooth_l1_loss(Q, Y)

    # ----------------------- Critic Update ----------------------- #

    self.optimizerCritic.zero_grad() 
    critic_loss.backward()
    for param in self.critic_local.parameters():
      param.grad.data.clamp_(-1, 1)               # Gradient cliping
    self.optimizerCritic.step()

    # ----------------------- Actor  Loss ----------------------- #
    # Loss = Mean(Q(S, localActions))

    agent1_actions = None
    agent2_actions = None

    if agentIndex==1:
      agent1_actions = current_actions_local[0].clone().detach()
      agent2_actions = current_actions_local[1]
    if agentIndex==0:
      agent1_actions = current_actions_local[0]
      agent2_actions = current_actions_local[1].clone().detach()

    Mu_local = torch.cat([agent1_actions, agent2_actions], dim=1).to(device)

    actor_loss = self.critic_local(states, Mu_local)
    actor_loss = -actor_loss.mean()

    # ----------------------- Actor Update ----------------------- #
    self.optimizerActor.zero_grad()
    actor_loss.backward()
    for param in self.actor_local.parameters():
      param.grad.data.clamp_(-1, 1)               # Gradient cliping
    self.optimizerActor.step()

    # ----------------------- Soft Update Target Networks ----------------------- #
    self.soft_update(self.critic_local, self.critic_target, TAU1)
    self.soft_update(self.actor_local, self.actor_target, TAU2)                


  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
      local_model (PyTorch model): weights will be copied from
      target_model (PyTorch model): weights will be copied to
      tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed):
    """Initialize a ReplayBuffer object.

    Params
    ======
      action_size (int): dimension of each action
      buffer_size (int): maximum size of buffer
      batch_size (int): size of each training batch
      seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.tensor([e.state for e in experiences if e is not None]).float().to(device)
    actions = torch.tensor([e.action for e in experiences if e is not None]).float().to(device)
    rewards = torch.tensor([e.reward for e in experiences if e is not None]).float().to(device)
    next_states = torch.tensor([e.next_state for e in experiences if e is not None]).float().to(device)
    dones = torch.tensor([1-np.asarray(e.done) for e in experiences if e is not None]).float().to(device)

    actions = actions.view(-1, 2, 2)

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)


class OUNoise:
  """Ornstein-Uhlenbeck process."""

  def __init__(self, size, seed, mu=0.0, theta=0.13, sigma=0.2):
    """Initialize parameters and noise process."""
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.seed = random.seed(seed)
    self.size = size
    self.reset()

  def reset(self):
    """Reset the internal state (= noise) to mean (mu)."""
    self.state = copy.copy(self.mu)

  def sample(self):
    """Update internal state and return it as a noise sample."""
    x = self.state
    dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
    self.state = x + dx
    return self.state
  