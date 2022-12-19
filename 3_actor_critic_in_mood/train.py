from env import Env
from models import AgentBrain

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from render_hist import plot_hist
import copy
import os

np.random.seed(0)


args = {
    'n_in_act': 9,
    'n_mid_act': 128,
    'n_out_act': 2,
    'n_in_cri': 11,
    'n_mid_cri': 128,
    'n_out_cri': 1,
}

episodes = 500
env = Env()
agent = AgentBrain(**args)
reward_history = []
history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    idx_mood = np.random.choice([0, 1, 2])
    mood = np.zeros(3)
    mood[idx_mood] = 1
    while not done:
        mood_state = np.append(state, mood)
        action, log_prob = agent.get_action(mood_state)
        next_state, reward, done = env.step(action, idx_mood)
        next_mood_state = np.append(next_state, mood)

        agent.update(mood_state, action, log_prob, 
            reward, next_mood_state, done)

        state = next_state
        total_reward += reward

        # 全試行の最後の10%を履歴に残しておく
        if episode > episodes * 0.9:
            history.append([copy.deepcopy(env.agent), 
                copy.deepcopy(env.target), 
                copy.deepcopy(env.enemy)]
            )

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

map_models = {
    'pi': agent.pi,
    'v': agent.v,
}

base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir, 'params')
os.makedirs(save_dir, exist_ok=True)

# 各モデルの学習済みパラメータを保存
for k, v in map_models.items():
    save_path = os.path.join(save_dir, f'{k}.prms')
    torch.save(v.state_dict(), save_path)

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, plot_hist, fargs=(history, ax), interval=20)
plt.show()

def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

plot_total_reward(reward_history)
