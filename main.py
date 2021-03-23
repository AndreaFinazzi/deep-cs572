# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from agents.DQNAgent import DQNAgent
from agents.D2QNAgent import D2QNAgent
from agents.D3QNAgent import D3QNAgent
import torch
import timeit
import gym
import subprocess
import pathlib

torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = pathlib.Path(__file__).parent.absolute()

print(file_path)

MAX_STEP_SCORE = 500
PLAY_EVERY_X_EPISODES = 1000
TEST_EPISODES_N = 10
TARGET_PARAMETERS_UPDATE_FREQ = 5

def play(env, g):
    state = env.reset()
    step = 0
    done = False
    while done is not True:
        env.render()
        step += 1

        state_torch = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
        action = g.act(state_torch)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            print('step = {}, reward = {}'.format(step, reward))
    return step

# env = gym.make('CartPole-v1')
env = gym.make('LunarLander-v2')

state = env.reset()
score = 0
total_score = 0
episode = 0
state_size = len(state)
action_size = env.action_space.n

g = D3QNAgent(state_size, action_size, 10)
start_time = timeit.default_timer()
epsilon = 0.1

## Init file outputs
date_output = subprocess.check_output("date +%y%m%d_%H%M%S", shell=True)
datetime = date_output.decode('utf-8').replace('\n','')
result_dir = str(file_path) + "/results/" + env.spec.id + "_" + g.get_name() + "_" + datetime
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=False)

result_filename = result_dir + "/" + "result.csv"
result_file = open(result_filename, mode='w')
result_file.write("episode,score,total_score,eval_score\n")

recordings_folder =  result_dir + "/recordings"

while episode <= 3000:  # episode loop
    episode = episode + 1
    state = env.reset()
    score = 0
    done = False

    while not done:
        state_torch = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
        action = g.act_epsilon(state_torch, epsilon * (0.998**episode))
        
        next_state, reward, done, info = env.step(action)
        next_state_torch = torch.from_numpy(next_state).type(torch.FloatTensor).unsqueeze(0)

        g.store(state_torch, action, reward, next_state_torch, done)
        g.train(episode)

        state = next_state

        score = score + reward
        total_score = total_score + reward

    eval_score = ((total_score + 554120) / 483370) * 100.
    result_file.write('{},{:.2f},{:.2f},{:.2f}\n'.format(episode, score, total_score, eval_score))

    if episode % PLAY_EVERY_X_EPISODES == 0:
        print('Episode: {} Score: {:.2f} Total score: {:.2f} Eval score : {:.2f}'.format(episode, score, total_score, eval_score))
        print('100 Episode time : {:.2f}s'.format((timeit.default_timer() - start_time)))
        start_time = timeit.default_timer()
        result = play(env, g)

    if episode % 100 == 0:
        print('Episode: {} Score: {:.2f} Total score: {:.2f} Eval score : {:.2f}'.format(episode, score, total_score, eval_score))


# TEST   
# RECORD THE LAST TEST  
monitor_env = gym.wrappers.Monitor(env, recordings_folder ,video_callable=lambda episode: True, force=True)
episode = 0
state = monitor_env.reset()
step = 0
while episode < TEST_EPISODES_N:  # episode loop
    play(monitor_env, g)
    episode += 1
env.close()
monitor_env.close()
result_file.close()