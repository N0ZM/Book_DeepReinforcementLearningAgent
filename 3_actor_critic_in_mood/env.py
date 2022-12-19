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
        self.enemy = Agent()
        self.cnt = 0

    def step(self, action, mood):
        prev_dist_target = calc_dist(self.target, self.agent)
        prev_dist_enemy = calc_dist(self.enemy, self.agent)

        self.enemy.move(self.agent)

        x = action[0][0].detach().numpy()
        self.agent.x += x
        y = action[0][1].detach().numpy()
        self.agent.y += y

        now_dist_target = calc_dist(self.target, self.agent)
        now_dist_enemy = calc_dist(self.enemy, self.agent)

        # next_stateはagentからtargetおよびagentからenemyまでの単位方向ベクトルと距離
        next_state = self.get_next_state(now_dist_target, now_dist_enemy)

        reward, done = self.get_reward(mood,
                                       prev_dist_target,
                                       now_dist_target,
                                       prev_dist_enemy,
                                       now_dist_enemy)

        # agentがtargetに到達できなくても50actionで1試行終了
        self.cnt += 1
        if self.cnt >= 50:
            done = True
        return next_state, reward, done

    def reset(self):
        self.__init__()
        now_dist = calc_dist(self.target, self.agent)
        now_dist_enemy = calc_dist(self.enemy, self.agent)
        state = self.get_next_state(now_dist, now_dist_enemy)
        return state

    def get_next_state(self, dist_target, dist_enemy):
        next_state = np.array(
            [(self.target.x - self.agent.x) / dist_target, 
             (self.target.y - self.agent.y) / dist_target, 
             dist_target / WIDTH, 
             (self.enemy.x - self.agent.x) / dist_enemy, 
             (self.enemy.y - self.agent.y) / dist_enemy, 
             dist_enemy / WIDTH]
        )
        return next_state

    def get_reward(self, mood, prev_dist_target, now_dist_target,
                    prev_dist_enemy, now_dist_enemy):
        # 通常パターン(ターゲットに接近、敵から逃走)
        if mood == 0:
            # rewardは行動前のagentとtargetの距離と行動後のagentとtargetの距離の差分(縮んだ距離がそのまま報酬になる)
            reward = prev_dist_target - now_dist_target
            done = is_hit(self.target, self.agent)
            if done:
                reward = 100
                return reward, done
            # rewardは行動前のagentとenemyの距離と行動後のagentとenemyの距離の差分(開いた距離がそのまま報酬になる)
            reward += now_dist_enemy - prev_dist_enemy
            done = is_hit(self.enemy, self.agent)
            if done:
                reward = - 50
            return reward, done
        # 挑発パターン(ターゲットに接近、敵に接近[ただし接触しない])
        elif mood == 1:
            # rewardは行動前のagentとtargetの距離と行動後のagentとtargetの距離の差分(縮んだ距離がそのまま報酬になる)
            reward = prev_dist_target - now_dist_target
            done = is_hit(self.target, self.agent)
            if done:
                reward = 50
                return reward, done
            # enemyから一定範囲内にいれば報酬
            if 50 < now_dist_enemy < 100:
                reward += 5
            else:
                reward -= 5
            done = is_hit(self.enemy, self.agent)
            if done:
                reward = - 50
            return reward, done
        # 攻撃パターン(ターゲットに接近、敵に接近)
        if mood == 2:
            # rewardは行動前のagentとtargetの距離と行動後のagentとtargetの距離の差分(縮んだ距離がそのまま報酬になる)
            reward = prev_dist_target - now_dist_target
            done = is_hit(self.target, self.agent)
            if done:
                reward = 50
                return reward, done
            # rewardは行動前のagentとenemyの距離と行動後のagentとenemyの距離の差分(縮んだ距離がそのまま報酬になる)
            reward += prev_dist_enemy - now_dist_enemy
            done = is_hit(self.enemy, self.agent)
            if done:
                reward = 100
            return reward, done
        return False
