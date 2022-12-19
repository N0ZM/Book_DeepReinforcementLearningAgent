import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env import Env, WIDTH, HEIGHT


fig, ax = plt.subplots()
env = Env()

def plot(frame, env):
    plt.cla()
    plt.xlim(WIDTH)
    plt.ylim(HEIGHT)
    ax.invert_xaxis()

    env.step()
    plt.scatter(env.agent.x, env.agent.y)
    plt.scatter(env.target.x, env.target.y)

anim = animation.FuncAnimation(fig, plot, fargs=(env,), interval=20)
plt.show()
