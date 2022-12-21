import numpy as np
import random

from Actor import Actor
from Critic import Critic
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
FC1_unit = 64           # Number of neurons in the first hidden layer
FC2_unit = 128          # Number of neurons in the second hidden layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, agent_num, random_seed=2):

        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num
        self.seed = random.seed(random_seed)

        # Set the noise generator
        self.noise = OUNoise(action_size, random_seed)

        # Create the replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
 
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=FC1_unit, fc2_units=FC2_unit).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, fc1_units=FC1_unit, fc2_units=FC2_unit).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed,fcs1_units=FC1_unit, fc2_units=FC2_unit).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,fcs1_units=FC1_unit, fc2_units=FC2_unit).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Ensure weights are identical between actor/critic targets and locals
        for p1, p0 in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            p1.data.copy_(p0.data)

        for p1, p0 in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            p1.data.copy_(p0.data)
   
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()
        self.t_step = 0

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, done = experiences

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done))

        # Compute & minimize critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Compute & minimize actor loss with back propagation
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Copy weights from the actor/critic local to the actor/critic target using TAU as the interpolation parameter
        # This lets us reuse the model to serve both players
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy(), 
