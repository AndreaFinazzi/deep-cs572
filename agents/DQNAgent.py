import random

from agents.Agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNAgent(Agent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.name = "DQN"

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

        self.model_online.eval()
        result = self.model_online(batch_state_torch)
        state_action_torch = torch.gather(result, 1, batch_action.unsqueeze(1))
        next_state_action_torch = self.model_online(batch_next_state_torch)
        next_state_action_torch = torch.max(next_state_action_torch, 1)[0].detach()

        Y = (reward_torch + (self.gamma * next_state_action_torch * terminal_torch)).float()

        self.model_online.train()
        loss = F.mse_loss(state_action_torch, Y.unsqueeze(1)) / self.batch_size
        self.optimizer_online.zero_grad()
        loss.backward()
        self.optimizer_online.step()