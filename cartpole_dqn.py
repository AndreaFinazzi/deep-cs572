# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import random
import gym
import numpy as np
import timeit
import matplotlib.pyplot as plt
import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import torchsummary
from torchsummary import summary
from collections import deque

torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def play(env, g):
    state = env.reset()
    step = 0
    done = False
    while done is not True:
        env.render()
        step += 1
        action = g.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            print('step = {}, reward = {}'.format(step, reward))

class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class agent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memories = deque(maxlen = 1024)

        self.batch_size = 32 # for speed up
        self.gamma = 0.99
        self.model_online = Qnet(state_size, action_size)

        print(self.model_online)
        summary(self.model_online, (state_size, ))

        self.optimizer_online = optim.Adam(self.model_online.parameters(), lr=0.0001)

    def store(self, state_torch, action, reward, next_state_torch, done):
        terminal = 1
        if done:
            terminal = 0

        transition = [state_torch, action, reward, next_state_torch, terminal]
        self.memories.append(transition)

    def train(self):
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
        #print(Y)

        self.model_online.train()
        loss = F.mse_loss(state_action_torch, Y.unsqueeze(1)) / self.batch_size
        self.optimizer_online.zero_grad()
        loss.backward()
        self.optimizer_online.step()


    def act(self, state):
        state_torch = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')
        self.model_online.eval()
        Qfunc_s_a = self.model_online(state_torch)
        action = Qfunc_s_a.data.max(1)[1].item()
        return action

    def act_epsilon(self, state_torch, epsilon):
        
        self.model_online.eval()
        Qfunc_s_a = self.model_online(state_torch)

        if random.random() < epsilon:
            action = np.random.choice(range(self.action_size))
        else:
            action = Qfunc_s_a.data.max(1)[1].item()
            #print(action)
        return action


output = subprocess.check_output("date +%y%m%d_%H%M%S", shell=True)
output = output.decode('utf-8').replace('\n','')
result_filename = "score_result_" + output + ".csv"
result_file = open(result_filename, mode='w')

env = gym.make('CartPole-v1')
state = env.reset()
score = 0
total_score = 0
episode = 0
state_size = len(state)
action_size = env.action_space.n

g = agent(state_size, action_size)
start_time = timeit.default_timer()
result_file.write("episode,score,total_score,eval_score\n")
epsilon = 0.5

while episode <= 3000:  # episode loop
    episode = episode + 1
    state = env.reset()
    score = 0
    done = False
    
    while not done:
        state_torch = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')
        action = g.act_epsilon(state_torch, epsilon * (0.998**episode)) #epsilon * (1 / episode))
        
        next_state, reward, done, info = env.step(action)
        next_state_torch = torch.from_numpy(next_state).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')

        g.store(state_torch, action, reward, next_state_torch, done)
        g.train()

        state = next_state

        score = score + reward
        total_score = total_score + reward

    eval_score = ((total_score + 554120) / 483370) * 100.
    result_file.write('{},{:.2f},{:.2f},{:.2f}\n'.format(episode, score, total_score, eval_score))

    if episode % 25 == 0:
        print('Episode: {} Score: {:.2f} Total score: {:.2f} Eval score : {:.2f}'.format(episode, score, total_score, eval_score))
        print('25 Episode time : {:.2f}s'.format((timeit.default_timer() - start_time)))
        start_time = timeit.default_timer()
        play(env, g)


# TEST     
episode = 0
state = env.reset()
step = 0
while episode < 10:  # episode loop
    play(env, g)
    episode += 1
env.close()

result_file.close()