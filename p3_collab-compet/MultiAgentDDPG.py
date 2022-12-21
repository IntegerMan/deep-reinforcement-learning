import numpy as np
from ReplayBuffer import ReplayBuffer
from Agent import Agent, BUFFER_SIZE, BATCH_SIZE, GAMMA

import random

class MultiAgentDDPG:
    
    def __init__(self, state_size, action_size, agent_num, random_seed=2):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num
        self.seed = random.seed(random_seed)
        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agents = [Agent(state_size, action_size, agent_num, random_seed) for _ in range(agent_num)]
        self.losses = (0., 0.)
        
    def reset(self):
        for agent in self.agents: agent.reset()
            
    def act(self, states, add_noise = True):
        return [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
    
    def step(self, states, actions, rewards, next_states, done):

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, done):
            self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) < BATCH_SIZE: 
            return
        critic_losses = []
        actor_losses = []

        for agent in self.agents:
            experiences = self.memory.sample()
            critic_loss, actor_loss = agent.learn(experiences, GAMMA)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

        self.losses = (np.mean(critic_losses), np.mean(actor_losses))
