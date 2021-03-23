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

MAX_STEP_SCORE = 500
PLAY_EVERY_X_EPISODES = 250
TEST_EPISODES_N = 5
TARGET_PARAMETERS_UPDATE_FREQ = 10

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
    return step

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
        self.target_online = Qnet(state_size, action_size)

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


    def act(self, state):
        state_torch = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
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
        return action


output = subprocess.check_output("date +%y%m%d_%H%M%S", shell=True)
output = output.decode('utf-8').replace('\n','')
result_folder = "./results/d2qn/lander/"
os.makedirs(os.path.dirname(result_folder), exist_ok=True)
result_filename = "score_result_d2qn_f" + str(TARGET_PARAMETERS_UPDATE_FREQ) + "_" + output + ".csv"
result_file = open(result_filename, mode='w')

env = gym.make('LunarLander-v2')

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

    # Update target network weights
    if episode % TARGET_PARAMETERS_UPDATE_FREQ == 0:
        print("Updating target parameters")
        g.target_online.load_state_dict(g.model_online.state_dict())

    while not done:
        state_torch = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
        action = g.act_epsilon(state_torch, epsilon * (0.998**episode))
        
        next_state, reward, done, info = env.step(action)
        next_state_torch = torch.from_numpy(next_state).type(torch.FloatTensor).unsqueeze(0)

        g.store(state_torch, action, reward, next_state_torch, done)
        g.train()

        state = next_state

        score = score + reward
        total_score = total_score + reward

    eval_score = ((total_score + 554120) / 483370) * 100.
    result_file.write('{},{:.2f},{:.2f},{:.2f}\n'.format(episode, score, total_score, eval_score))

    if episode % PLAY_EVERY_X_EPISODES == 0:
        print('Episode: {} Score: {:.2f} Total score: {:.2f} Eval score : {:.2f}'.format(episode, score, total_score, eval_score))
        print('100 Episode time : {:.2f}s'.format((timeit.default_timer() - start_time)))
        start_time = timeit.default_timer()
        step = play(env, g)
        # if step >= MAX_STEP_SCORE:
        #     break



# TEST   
# RECORD THE LAST TEST  
monitor_env = gym.wrappers.Monitor(env, "./recording/d2qn/lander",video_callable=lambda episode: True, force=True)
episode = 0
state = monitor_env.reset()
step = 0
while episode < TEST_EPISODES_N:  # episode loop
    play(monitor_env, g)
    episode += 1
env.close()
monitor_env.close()
result_file.close()