from env import Env
from models import AgentBrain, FreeActor, RandomAgent

import numpy as np
import torch
from scipy import stats as st

import matplotlib.pyplot as plt
import os


args_actor_critic = {
    'n_in_act': 9,
    'n_mid_act': 128,
    'n_out_act': 2,
    'n_in_cri': 11,
    'n_mid_cri': 128,
    'n_out_cri': 1,
}

args_free_actor = {
    'n_in': 10,
    'n_mid': 128,
    'n_out': 2,
}

episodes = 500
env = Env()

agent = AgentBrain(**args_actor_critic)
free_actor = FreeActor(**args_free_actor)
dummy_agent = RandomAgent()

map_models = {
    'pi': agent.pi,
    'v': agent.v,
    'free_actor': free_actor,
}

base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir, 'params')

# 各モデルの学習済みパラメータを読み込み
for k, v in map_models.items():
    save_path = os.path.join(save_dir, f'{k}.prms')
    weights = torch.load(save_path)
    v.load_state_dict(weights)

idx_mood = 0
mood = np.zeros(3)
mood[idx_mood] = 1
arg_reward = 2
means_experimental = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    rewards = []
    while not done:
        action = free_actor(mood, state, arg_reward)
        next_state, reward, done = env.step(action, idx_mood)
        rewards.append(reward)

        state = next_state

    means_experimental.append(np.mean(rewards))

arg_reward = - 8
means_experimental_low = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    rewards = []
    while not done:
        action = free_actor(mood, state, arg_reward)
        next_state, reward, done = env.step(action, idx_mood)
        rewards.append(reward)

        state = next_state

    means_experimental_low.append(np.mean(rewards))

rewards = []
means_control = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    rewards = []
    while not done:
        action = dummy_agent.get_action()
        next_state, reward, done = env.step(action, idx_mood)
        rewards.append(reward)

        state = next_state

    means_control.append(np.mean(rewards))

mu_experimental = np.mean(means_experimental)
sigma_experimental = np.std(means_experimental)
rv_experimental = st.norm(mu_experimental, sigma_experimental)

min_experimental = np.min(means_experimental)
max_experimental = np.max(means_experimental)
x = np.linspace(min_experimental - 10, max_experimental + 10, 100)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.hist(means_experimental, bins=100, 
    range=(min_experimental - 10, max_experimental + 10), 
    density=True, alpha=0.5)
ax.plot(x, rv_experimental.pdf(x), label='Experimental High')

mu_control = np.mean(means_control)
sigma_control = np.std(means_control)
rv_control = st.norm(mu_control, sigma_control)

min_control = np.min(means_control)
max_control = np.max(means_control)
x = np.linspace(min_control - 10, max_control + 10, 100)

ax.hist(means_control, bins=100, 
    range=(min_control - 10, max_control + 10), density=True, alpha=0.5)
ax.plot(x, rv_control.pdf(x), label='Control', ls='--')

mu_experimental_low = np.mean(means_experimental_low)
sigma_experimental_low = np.std(means_experimental_low)
rv_experimental_low = st.norm(mu_experimental_low, sigma_experimental_low)

min_experimental_low = np.min(means_experimental_low)
max_experimental_low = np.max(means_experimental_low)
x = np.linspace(min_experimental_low - 10, max_experimental_low + 10, 100)

ax.hist(means_experimental_low, bins=100, 
    range=(min_experimental_low - 10, max_experimental_low + 10), 
    density=True, alpha=0.5)
ax.plot(x, rv_experimental_low.pdf(x), label='Experimental Low', ls=':')

ax.legend()
file_name = os.path.join(os.path.dirname(__file__), 'results/result.png')
plt.savefig(file_name, format="png", dpi=300)
plt.show()

print(f'P_value Experimental High to Control(Upper): \
    {1 - rv_control.cdf(mu_experimental)}')
print(f'P_value Experimental High to Experimental Low(Upper): \
    {1 - rv_experimental_low.cdf(mu_experimental)}')

print(f'Critical value of Control(Level of significance: Upper 5%): \
    {rv_control.isf(0.05)}')
print(f'Critical value of Experimental Low(Level of significance: Upper 5%): \
    {rv_experimental_low.isf(0.05)}')
print(f'Mean of Experimental High: {rv_experimental.mean()}')
