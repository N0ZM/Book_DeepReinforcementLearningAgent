from env import Env
from models import AgentBrain, ReasonableMotivator, ActionMaker

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from render_hist import plot_hist
import copy
import os


args_actor_critic = {
    'n_in_act': 9,
    'n_mid_act': 128,
    'n_out_act': 2,
    'n_in_cri': 11,
    'n_mid_cri': 128,
    'n_out_cri': 1,
}

args_reasonable_motivator = {
    'n_in': 10,
    'n_mid': 128,
    'n_out': 6,
}

args_action_maker = {
    'n_in': 12,
    'n_mid': 128,
    'n_out': 2,
}

episodes = 100
env = Env()

agent = AgentBrain(**args_actor_critic)
reasonable_motivator = ReasonableMotivator(**args_reasonable_motivator)
action_maker = ActionMaker(**args_action_maker)

map_models = {
    'pi': agent.pi,
    'v': agent.v,
    'reasonable_motivator': reasonable_motivator,
    'action_maker': action_maker
}

base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir, 'params')

# 各モデルの学習済みパラメータを読み込み
for k, v in map_models.items():
    save_path = os.path.join(save_dir, f'{k}.prms')
    weights = torch.load(save_path)
    v.load_state_dict(weights)

reward_history = []
history = []

idx_mood = 0
mood = np.zeros(3)
mood[idx_mood] = 1
arg_reward = 8

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        next_state = reasonable_motivator(mood, state, arg_reward)
        next_state = next_state.detach().numpy()[0]
        action = action_maker(state, next_state)

        next_state, reward, done = env.step(action, idx_mood)

        state = next_state

        history.append([copy.deepcopy(env.agent), 
            copy.deepcopy(env.target), 
            copy.deepcopy(env.enemy)]
        )

fig, ax = plt.subplots()
ani = animation.FuncAnimation(
    fig, plot_hist, fargs=(history, ax), interval=20
)
plt.show()
