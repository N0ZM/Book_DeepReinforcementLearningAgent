from env import Env
from models import AgentBrain, RandomAgent, FreeActor

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from render_hist import plot_hist
import copy
import os

np.random.seed(0)


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

episodes = 1000
env = Env()

agent = AgentBrain(**args_actor_critic)
free_actor = FreeActor(**args_free_actor)
dummy_agent = RandomAgent()

reward_history = []
history = []
value_list = []
gamma = 0.98

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    idx_mood = np.random.choice([0, 1, 2])
    mood = np.zeros(3)
    mood[idx_mood] = 1
    is_dummy_agent = np.random.choice([0, 1], p=[0.9, 0.1])
    # is_dummy_agent = 1
    while not done:
        mood_state = np.append(state, mood)

        if not is_dummy_agent:
            action, log_prob = agent.get_action(mood_state)
        else:
            action = dummy_agent.get_action()
        
        next_state, reward, done = env.step(action, idx_mood)
        # rewardの値を[-1, 1]に収めておく
        reward = np.array([[reward]])
        reward = F.tanh(torch.tensor(reward, dtype=torch.float32))
        reward = reward.detach().numpy()[0][0]

        next_mood_state = np.append(next_state, mood)

        if not is_dummy_agent:
            agent.update(mood_state, action, log_prob, 
                reward, next_mood_state, done)

        state_action = np.concatenate(
            [mood_state, action.detach().numpy()[0]]
        )
        state_action = torch.tensor(
            state_action[np.newaxis, :], dtype=torch.float32
        )
        q = agent.v(state_action).detach().numpy()[0]

        free_actor.update(mood, state, q, action)

        state = next_state

        # 行動価値の最小値、最大値、平均値確認用
        value_list.append(q)

        # 全試行の最後の10%を履歴に残しておく
        if episode > episodes * 0.9:
            history.append(
                [copy.deepcopy(env.agent), 
                copy.deepcopy(env.target), 
                copy.deepcopy(env.enemy)]
            )

        # 最適化確認用の報酬記録を別途出力
        # next_stateをもとに最大に近い価値を求めた場合に得られる報酬を求める
        acttion = free_actor(mood, next_state, 3)
        _, reward, _ = env.step_to_get_reward(action, idx_mood)
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {:.1f}".format(
            episode, total_reward))

print('max q: ', np.max(value_list))
print('min q: ', np.min(value_list))
print('average q: ', np.average(value_list))

map_models = {
    'pi': agent.pi,
    'v': agent.v,
    'free_actor': free_actor,
}

base_dir = os.path.dirname(__file__)
save_dir = os.path.join(base_dir, 'params')
os.makedirs(save_dir, exist_ok=True)

# 各モデルの学習済みパラメータを保存
for k, v in map_models.items():
    save_path = os.path.join(save_dir, f'{k}.prms')
    torch.save(v.state_dict(), save_path)

fig, ax = plt.subplots()
anim = animation.FuncAnimation(
    fig, plot_hist, fargs=(history, ax), interval=20
)
plt.show()

def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

plot_total_reward(reward_history)
