import random

from agents.Agent import Agent
from agents.nets.QNet import QNet

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_PARAMETERS_UPDATE_FREQ = 5

# Double Deep Q NEt
class D2QNAgent(Agent):
    def __init__(self, state_size, action_size, target_update_freq=TARGET_PARAMETERS_UPDATE_FREQ):
        super().__init__(state_size, action_size)

        self.model_online = QNet(state_size, action_size)

        self.target_online = QNet(state_size, action_size)
        self.target_update_freq=target_update_freq

        self.name = "D2QN_f" + str(self.target_update_freq)

    def train(self, episode=-1):
        if(len(self.memories) < self.batch_size):
            return

        batch_data = random.sample(self.memories, self.batch_size)
        state_torch = [data[0] for data in batch_data]
        action = [data[1] for data in batch_data]
        reward = [data[2] for data in batch_data]
        next_state_torch = [data[3] for data in batch_data]
        terminal = [data[4] for data in batch_data]

        batch_state_torch = torch.cat(state_torch)
        batch_next_state_torch = torch.cat(next_state_torch)
        batch_action = torch.tensor(action)
        reward_torch = torch.tensor(reward)
        terminal_torch = torch.tensor(terminal)

        # Get current q-values
        self.model_online.eval()
        result = self.model_online(batch_state_torch)
        state_action_torch = torch.gather(result, 1, batch_action.unsqueeze(1))

        # expected q-values from target network
        self.target_online.eval()
        next_state_action_torch = self.target_online(batch_next_state_torch)
        next_state_action_torch = torch.max(next_state_action_torch, 1)[0].detach()

        Y = (reward_torch + (self.gamma * next_state_action_torch * terminal_torch)).float()
        # Y = (reward_torch + (self.gamma * next_state_action_torch * terminal_torch)).float()

        self.model_online.train()
        loss = F.mse_loss(state_action_torch, Y.unsqueeze(1)) / self.batch_size
        self.optimizer_online.zero_grad()
        loss.backward()
        self.optimizer_online.step()

        self.update(episode)

    def update(self, episode):
        # Update target network weights
        if episode % self.target_update_freq == 0:
            # print("Updating target parameters")
            self.target_online.load_state_dict(self.model_online.state_dict())
