import numpy as np

np.random.seed(0)


WIDTH = 400
HEIGHT = 200
SPEED = 5

class Agent:

    def __init__(self):
        self.x = np.random.rand() * WIDTH
        self.y = np.random.rand() * HEIGHT

    def move(self, target):
        dist = calc_dist(self, target)
        self.x += (target.x - self.x) / dist * SPEED
        self.y += (target.y - self.y) / dist * SPEED


class Target:

    def __init__(self):
        self.x = np.random.rand() * WIDTH
        self.y = np.random.rand() * HEIGHT

# オブジェクト同士の距離の計算
def calc_dist(obj_a, obj_b):
    dist = np.sqrt((obj_b.x - obj_a.x) ** 2 + (obj_b.y - obj_a.y) ** 2)
    return dist

# オブジェクト同士の当たり判定
def is_hit(obj_a, obj_b):
    # 当たり判定距離
    dist_collide = 5
    if calc_dist(obj_a, obj_b) < dist_collide:
        return True
    return False


class Env:

    def __init__(self):
        self.agent = Agent()
        self.target = Target()
        self.cnt = 0

    def step(self, action):
        prev_dist = calc_dist(self.target, self.agent)

        x = action[0][0].detach().numpy()
        self.agent.x += x
        y = action[0][1].detach().numpy()
        self.agent.y += y

        now_dist = calc_dist(self.target, self.agent)
        # next_stateはagentからtargetまでの単位方向ベクトルと距離
        next_state = np.array(
            [(self.target.x - self.agent.x) / now_dist, 
             (self.target.y - self.agent.y) / now_dist, 
             now_dist / WIDTH]
        )
        # rewardは行動前のagentとtargetの距離と行動後のagentとtargetの距離の差分
        reward = prev_dist - now_dist
        done = is_hit(self.target, self.agent)
        if done:
            reward = 100

        # agentがtargetに到達できなくても50actionで1試行終了
        self.cnt += 1
        if self.cnt >= 50:
            done = True
        return next_state, reward, done

    def reset(self):
        self.__init__()
        now_dist = calc_dist(self.target, self.agent)
        state = np.array(
            [(self.target.x - self.agent.x) / now_dist, 
             (self.target.y - self.agent.y) / now_dist, 
             now_dist / WIDTH]
        )
        return state