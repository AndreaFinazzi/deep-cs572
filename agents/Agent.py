from agents.nets.QNet import QNet

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchsummary
from torchsummary import summary

#  In order to finalize the agent initialization, a call to ready() is required
class Agent(object):
    def __init__(self, state_size, action_size):
        self.is_ready = False
        self.name = "base_agent"
        self.state_size = state_size
        self.action_size = action_size

        self.memories = deque(maxlen = 10024)

        self.batch_size = 128 # for speed up
        self.gamma = 0.99

        self.model_online = None # Should throw exception if not initialized properly

    def ready(self):
        print(self.__class__)
        # summary(self.model_online, (self.state_size, ))
        self.optimizer_online = optim.Adam(self.model_online.parameters(), lr=0.0001)
        self.is_ready = True

    def store(self, state_torch, action, reward, next_state_torch, done):
        terminal = 1
        if done:
            terminal = 0

        transition = [state_torch, action, reward, next_state_torch, terminal]
        self.memories.append(transition)

    def train(self, episode=-1):
        if ~self.is_ready: 
            raise NotImplementedError("Agent.ready() not called")
        pass

    def act(self, state_torch):
        self.model_online.eval()
        Qfunc_s_a = self.model_online(state_torch)
        action = Qfunc_s_a.data.max(1)[1].item()
        return action

    def act_epsilon(self, state_torch, epsilon):
        
        if random.random() < epsilon:
            action = np.random.choice(range(self.action_size))
        else:
            self.model_online.eval()
            Qfunc_s_a = self.model_online(state_torch)
            action = Qfunc_s_a.data.max(1)[1].item()
        return action

    def get_name(self):
        return self.name