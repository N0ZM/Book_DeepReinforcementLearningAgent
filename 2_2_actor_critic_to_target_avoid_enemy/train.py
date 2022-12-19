from env import Env
from models import AgentBrain

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from render_hist import plot_hist
import copy


args = {
    'n_in_act': 6,
    'n_mid_act': 128,
    'n_out_act': 2,
    'n_in_cri': 8,
    'n_mid_cri': 128,
    'n_out_cri': 1,
}

episodes = 300
env = Env()
agent = AgentBrain(**args)
reward_history = []
history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, log_prob = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, log_prob, reward, next_state, done)

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

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, plot_hist, fargs=(history, ax), interval=20)
plt.show()

def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

plot_total_reward(reward_history)
