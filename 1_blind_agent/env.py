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

    def step(self):
        self.agent.move(self.target)
        if is_hit(self.agent, self.target):
            self.target = Target()
