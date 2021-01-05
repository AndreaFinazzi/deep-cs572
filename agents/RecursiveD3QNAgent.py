import random
from agents.D3QNAgent import D3QNAgent
from agents.nets.RecursiveDuelingQNet import RecursiveDuelingQNet

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchsummary
from torchsummary import summary

TARGET_PARAMETERS_UPDATE_FREQ = 5

# Dueling Double Deep Q Net
class RecursiveD3QNAgent(D3QNAgent):
    def __init__(self, state_size, action_size, target_update_freq=TARGET_PARAMETERS_UPDATE_FREQ):
        super().__init__(state_size, action_size)
        self.model_online = RecursiveDuelingQNet(state_size, action_size)

        self.model_target = RecursiveDuelingQNet(state_size, action_size)
        self.target_update_freq=target_update_freq

        self.name = "RecursiveD3QN_f" + str(self.target_update_freq)

        self.ready()

    def ready(self):
        super().ready()
        summary(self.model_online, (4, self.state_size))
    # def train(self, episode=-1):
        # if(len(self.memories) < self.batch_size):
        #     return

        # batch_data = random.sample(self.memories, self.batch_size)
        # state_r = [data[0] for data in batch_data]
        # action = [data[1] for data in batch_data]
        # reward = [data[2] for data in batch_data]
        # next_state_r = [data[3] for data in batch_data]
        # terminal = [data[4] for data in batch_data]

        # batch_state_r = torch.cat(state_r)
        # batch_next_state_r = torch.cat(next_state_r)
        # batch_action = torch.tensor(action).unsqueeze(1)
        # batch_reward = torch.tensor(reward).unsqueeze(1)
        # batch_terminal = torch.tensor(terminal).unsqueeze(1)

        # # Get current q-values
        # self.model_online.eval()
        # online_Q = self.model_online(batch_state_r)
        # online_Q_next = self.model_online(batch_next_state_r)

        # # Current q-values on next state
        # online_max_action = torch.argmax(online_Q_next, 1, keepdim=True)

        # # expected q-values from target network
        # self.model_target.eval()
        # target_Q_next = self.model_target(batch_next_state)
        # target_Q_max_action = target_Q_next.gather(1, online_max_action)

        # Y = (batch_reward + self.gamma * target_Q_max_action * batch_terminal).float()
        # # Y = (reward_torch + (self.gamma * next_state_action_torch * terminal_torch)).float()

        # self.model_online.train()
        # loss = F.mse_loss(online_Q.gather(1, batch_action), Y) # / self.batch_size
        # self.optimizer_online.zero_grad()
        # loss.backward()
        # self.optimizer_online.step()

        # self.update(episode)

    # def update(self, episode):
    #     # Update target network weights
    #     if episode % self.target_update_freq == 0:
    #         # print("Updating target parameters")
    #         self.model_target.load_state_dict(self.model_online.state_dict())

    # OVERRIDE
    # def act_epsilon(self, state_torch, epsilon):
        
    #     if random.random() < epsilon:
    #         action = np.random.choice(range(self.action_size))
    #     else:
    #         self.model_online.eval()
    #         Qfunc_s_a = self.model_online(state_torch)
    #         action = Qfunc_s_a.data.max(1)[1].item()
    #     return action
