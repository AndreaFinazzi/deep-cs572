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

def play(env, g):
	state = env.reset()
	step = 0
	done = False
	score = 0	
	state_list = list()
	state_list.clear()
	while done is not True:
		env.render()
		state_list.append(state)
		if step < 4:
			action = g.act(state_list[step])
		else:
			action = g.act_r(state_list[step-3], state_list[step-2], state_list[step-1], state_list[step])
		next_state, reward, done, info = env.step(action)
		score += reward
		step += 1
		state = next_state
		if done:
			print('step = {}, score = {}'.format(step, score))


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

class RQnet(nn.Module):
	def __init__(self, state_size, action_size):
		super(RQnet, self).__init__()
		self.fc1_r = nn.Linear(state_size + action_size, 32)
		self.fc2_r = nn.Linear(32, 32)
		self.fc3_r = nn.Linear(32, action_size)
	
	def forward(self, x1, x2, x3, x4):
		t2 = torch.cat([x1, x2], dim=1)
		x = F.relu(self.fc1_r(t2))
		x = F.relu(self.fc2_r(x))
		out = self.fc3_r(x)

		t3 = torch.cat([out, x3], dim=1)
		x = F.relu(self.fc1_r(t3))
		x = F.relu(self.fc2_r(x))
		out = self.fc3_r(x)

		t4 = torch.cat([out, x4], dim=1)
		x = F.relu(self.fc1_r(t4))
		x = F.relu(self.fc2_r(x))
		out = self.fc3_r(x)

		return out

class agent(object):
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		
		self.memories = deque(maxlen = 2048)
		self.memories_r = deque(maxlen = 2048)

		self.batch_size = 128 # for speed up
		self.gamma = 0.99
		self.model_online = Qnet(state_size, action_size)#.to('cuda')
		self.model_target = Qnet(state_size, action_size)#.to('cuda')

		self.batch_size_R = 128
		self.model_R_online = RQnet(state_size, action_size)#.to('cuda')
		self.model_R_target = RQnet(state_size, action_size)#.to('cuda')
		
		print(self.model_online)
		#summary(self.model_online, (state_size, ))

		print(self.model_R_online)
		#summary(self.model_R_online, (state_size, ), (state_size+action_size, ), (state_size+action_size, ))

		self.optimizer_online = optim.Adam(self.model_online.parameters(), lr=0.0001)
		self.optimizer_R_online = optim.Adam(self.model_R_online.parameters(), lr=0.0001)		

	def store(self, state_torch, action, reward, next_state_torch, done):
		terminal = 1
		if done:
			terminal = 0

		transition = [state_torch, action, reward, next_state_torch, terminal]
		self.memories.append(transition)

	def store_r(self, state_torch, action, reward, next_state_torch, done):
		terminal = 1
		if done:
			terminal = 0

		transition = [state_torch[0], state_torch[1], state_torch[2], state_torch[3], 
					action, reward,
					next_state_torch[0], next_state_torch[1], next_state_torch[2],next_state_torch[3],
					terminal]
		self.memories_r.append(transition)

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
		batch_action = torch.tensor(action)#.to('cuda')
		reward_torch = torch.tensor(reward)#.to('cuda')
		terminal_torch = torch.tensor(terminal)#.to('cuda')

		self.model_online.eval()
		self.model_target.eval()
		result = self.model_online(batch_state_torch)
		state_action_torch = torch.gather(result, 1, batch_action.unsqueeze(1))

		next_state_action_torch = self.model_target(batch_next_state_torch)
		next_state_action_torch = torch.max(next_state_action_torch, 1)[0].detach()
	
		Y = reward_torch + (self.gamma * next_state_action_torch * terminal_torch)
		
		self.model_online.train()
		loss = F.mse_loss(state_action_torch, Y.unsqueeze(1))
		
		self.optimizer_online.zero_grad()
		loss.backward()
		self.optimizer_online.step()

	def train_r(self):
		if(len(self.memories) < self.batch_size_R):
			return
		
		batch_data = random.sample(self.memories_r, self.batch_size_R)
		state_torch_0 = [data[0] for data in batch_data]
		state_torch_1 = [data[1] for data in batch_data]
		state_torch_2 = [data[2] for data in batch_data]
		state_torch_3 = [data[3] for data in batch_data]
		action = [data[4] for data in batch_data]
		reward = [data[5] for data in batch_data]
		next_state_torch_0 = [data[6] for data in batch_data]
		next_state_torch_1 = [data[7] for data in batch_data]
		next_state_torch_2 = [data[8] for data in batch_data]
		next_state_torch_3 = [data[9] for data in batch_data]
		terminal = [data[10] for data in batch_data]
		
		batch_state_torch_0 = torch.cat(state_torch_0)
		batch_state_torch_1 = torch.cat(state_torch_1)
		batch_state_torch_2 = torch.cat(state_torch_2)
		batch_state_torch_3 = torch.cat(state_torch_3)
		batch_next_state_torch_0 = torch.cat(next_state_torch_0)
		batch_next_state_torch_1 = torch.cat(next_state_torch_1)
		batch_next_state_torch_2 = torch.cat(next_state_torch_2)
		batch_next_state_torch_3 = torch.cat(next_state_torch_3)

		batch_action = torch.tensor(action)#.to('cuda')
		reward_torch = torch.tensor(reward)#.to('cuda')
		terminal_torch = torch.tensor(terminal)#.to('cuda')

		self.model_online.eval()
		self.model_target.eval()

		x1 = self.model_online(batch_state_torch_3)
		state_action_torch = torch.gather(x1, 1, batch_action.unsqueeze(1))
		next_state_action_torch = self.model_target(batch_next_state_torch_3)
		next_state_action_torch = torch.max(next_state_action_torch, 1)[0].detach()
	
		Y = reward_torch + (self.gamma * next_state_action_torch * terminal_torch)
		
		self.model_online.train()
		loss = F.mse_loss(state_action_torch, Y.unsqueeze(1))
		#loss = F.smooth_l1_loss(state_action_torch, Y.unsqueeze(1))
		
		self.optimizer_online.zero_grad()
		loss.backward()
		self.optimizer_online.step()
		
#############.
		self.model_online.eval()
		x1_n = self.model_online(batch_state_torch_0)

		self.model_R_online.eval()
		self.model_R_target.eval()
		result_r = self.model_R_online(x1_n, batch_state_torch_1, batch_state_torch_2, batch_state_torch_3)
		state_action_torch_r = torch.gather(result_r, 1, batch_action.unsqueeze(1))
		
		x1_t = self.model_target(batch_next_state_torch_0)
		next_state_action_torch_r = self.model_R_target(x1_t, batch_next_state_torch_1,
									batch_next_state_torch_2,batch_next_state_torch_3)
		next_state_action_torch_r = torch.max(next_state_action_torch_r, 1)[0].detach()
	
		Y_r = reward_torch + (self.gamma * next_state_action_torch_r * terminal_torch)
		
		self.model_R_online.train()
		#loss = F.mse_loss(state_action_torch, Y.unsqueeze(1))
		loss_r = F.smooth_l1_loss(state_action_torch_r, Y_r.unsqueeze(1))
		
		self.optimizer_R_online.zero_grad()
		loss_r.backward()
		self.optimizer_R_online.step()

	def update_target(self):
		self.model_target.load_state_dict(self.model_online.state_dict())
		self.model_R_target.load_state_dict(self.model_R_online.state_dict())

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
		return action

	def act_r(self, state_0, state_1, state_2, state_3):
		state_torch_0 = torch.from_numpy(state_0).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')
		state_torch_1 = torch.from_numpy(state_1).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')
		state_torch_2 = torch.from_numpy(state_2).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')
		state_torch_3 = torch.from_numpy(state_3).type(torch.FloatTensor).unsqueeze(0)#.to('cuda')

		self.model_online.eval()
		x1 = self.model_online(state_torch_0)
		self.model_R_online.eval()
		Qfunc_s_a = self.model_R_online(x1, state_torch_1, state_torch_2, state_torch_3)

		action = Qfunc_s_a.data.max(1)[1].item()
		return action

	def act_epsilon_r(self, state_torch, epsilon):
		
		self.model_online.eval()
		x1 = self.model_online(state_torch[0])
		self.model_R_online.eval()
		Qfunc_s_a = self.model_R_online(x1, state_torch[1], state_torch[2], state_torch[3])

		if random.random() < epsilon:
			action = np.random.choice(range(self.action_size))
		else:
			action = Qfunc_s_a.data.max(1)[1].item()
		return action

output = subprocess.check_output("date +%y%m%d_%H%M%S", shell=True)
output = output.decode('utf-8').replace('\n','')
result_filename = "score_result_" + output + ".csv"
result_file = open(result_filename, mode='w')

env = gym.make('LunarLander-v2')
state = env.reset()
score = 0
total_score = 0
episode = 0
state_size = 8
action_size = env.action_space.n

g = agent(state_size, action_size)
start_time = timeit.default_timer()
result_file.write("episode,score,total_score,eval_score\n")
epsilon = 0.5
state_torch = list()
next_state_torch = list()
while episode <= 3000:  # episode loop
	episode = episode + 1
	state = env.reset()
	score = 0
	done = False
	step = 0
	state_torch.clear()
	next_state_torch.clear()
	while not done:
		state_torch.append(torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0))#.to('cuda'))
		if step >= 4 :
			state_torch_r = [state_torch[step-3], state_torch[step-2], state_torch[step-1], state_torch[step]]
			action = g.act_epsilon_r(state_torch_r, epsilon * (0.998**episode)) #epsilon * (1 / episode))
			next_state, reward, done, info = env.step(action)
			next_state_torch.append(torch.from_numpy(next_state).type(torch.FloatTensor).unsqueeze(0))#.to('cuda'))
			next_state_torch_r = [next_state_torch[step-3], next_state_torch[step-2], next_state_torch[step-1], next_state_torch[step]]
			g.store_r(state_torch_r, action, reward, next_state_torch_r, done)
			g.train_r()
		else :
			action = g.act_epsilon(state_torch[step], epsilon * (0.998**episode))
			next_state, reward, done, info = env.step(action)
			next_state_torch.append(torch.from_numpy(next_state).type(torch.FloatTensor).unsqueeze(0))#.to('cuda'))
			g.store(state_torch[step], action, reward, next_state_torch[step], done)
			g.train()
		state = next_state
		step += 1
		score = score + reward
		total_score = total_score + reward
	
	eval_score = ((total_score + 554120) / 483370) * 100.
	result_file.write('{},{:.2f},{:.2f},{:.2f}\n'.format(episode, score, total_score, eval_score))
	
	if episode % 2 == 0:
		g.update_target()

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
