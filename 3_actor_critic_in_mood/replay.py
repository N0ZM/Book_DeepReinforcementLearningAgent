from env import Env
from models import AgentBrain

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from render_hist import plot_hist
import copy
import os


args = {
    'n_in_act': 9,
    'n_mid_act': 128,
    'n_out_act': 2,
    'n_in_cri': 11,
    'n_mid_cri': 128,
    'n_out_cri': 1,
}

episodes = 10
env = Env()

agent = AgentBrain(**args)

map_models = {
    'pi': agent.pi,
    'v': agent.v,
}

base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir, 'params')

# 各モデルの学習済みパラメータを読み込み
for k, v in map_models.items():
    save_path = os.path.join(save_dir, f'{k}.prms')
    weights = torch.load(save_path)
    v.load_state_dict(weights)

history = []

idx_mood = 1
mood = np.zeros(3)
mood[idx_mood] = 1

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        mood_state = np.append(state, mood)
        action, log_prob = agent.get_action(mood_state)
        next_state, reward, done = env.step(action, idx_mood)

        state = next_state
        
        history.append([copy.deepcopy(env.agent), copy.deepcopy(env.target), copy.deepcopy(env.enemy)])

fig, ax = plt.subplots()
ani = animation.FuncAnimation(
    fig, plot_hist, fargs=(history, ax), interval=20
)
plt.show()

def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()
