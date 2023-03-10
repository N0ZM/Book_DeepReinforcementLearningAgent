import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from env import SPEED

np.random.seed(0)
torch.manual_seed(0)


class Actor(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l2 = nn.Linear(n_mid, n_mid)
        self.mean_linear = nn.Linear(n_mid, n_out)
        self.log_std_linear = nn.Linear(n_mid, n_out)
        self.speed = SPEED
        self.eps = 1e-6

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, max=2, min=-20)
        
        std = log_std.exp()
        a = mean + std * torch.randn(std.shape)
        log_prob = - 1 / 2 * torch.log(2 * torch.pi * std ** 2) - \
            (a - mean) ** 2 / (2 * std ** 2)
        y = torch.tanh(a)
        action = y * self.speed
        log_prob -= torch.log(self.speed * (1 - y.pow(2)) + self.eps)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l2 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class AgentBrain:

    def __init__(self, n_in_act, n_mid_act, n_out_act, 
            n_in_cri, n_mid_cri, n_out_cri):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005

        self.pi = Actor(n_in_act, n_mid_act, n_out_act)
        self.v = Critic(n_in_cri, n_mid_cri, n_out_cri)
        self.optimizer_pi = optim.Adam(
            self.pi.parameters(), lr=self.lr_pi
        )
        self.optimizer_v = optim.Adam(
            self.v.parameters(), lr=self.lr_v
        )

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        action, log_prob = self.pi(state)
        return action, log_prob

    def update(self, state, action, log_prob, reward, next_state, done):
        action = action.detach().numpy()[0]
        state_action = np.concatenate([state, action])
        state_action = torch.tensor(
            state_action[np.newaxis, :], dtype=torch.float32
        )
        next_action, _ = self.pi(
            torch.tensor([next_state], dtype=torch.float32)
        )
        next_state_action = np.concatenate(
            [next_state, next_action.detach().numpy()[0]]
        )
        next_state_action = torch.tensor(
            next_state_action[np.newaxis, :], dtype=torch.float32
        )

        target = reward + self.gamma * \
            self.v(next_state_action) * (1 - done)
        target.detach()
        v = self.v(state_action)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = - log_prob * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


class RandomAgent:
    """
        ?????????????????????????????????
        ????????????????????????????????????

        Returns:
            action (tensor): ????????????????????????????????????
    """

    def __init__(self):
        self.x = (2 * np.random.rand() - 1) * SPEED * 2
        self.y = (2 * np.random.rand() - 1) * SPEED * 2
        self.cnt = 0

    def get_action(self):
        action = torch.tensor([[self.x, self.y]], dtype=torch.float32)
        self.cnt += 1
        if self.cnt > 10:
            self.__init__()
        return action


# ?????????(???????????????)??????????????????????????????
class FreeActor(nn.Module):
    """
    ????????????????????????????????????????????????????????????
    mood, state, reward???????????????action?????????

    Args:
        mood (int): agent?????????(??????)
        state (ndarray): state_t0
        reward (int): ????????????
    Returns:
        action (tensor): ????????????
    """

    def __init__(self, n_in, n_mid, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l2 = nn.Linear(n_mid, n_mid)
        self.l3 = nn.Linear(n_mid, n_out)

        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, mood, state, reward):
        x = np.array([mood])
        x = np.append(x, state)
        x = np.append(x, np.array([reward]))
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def update(self, mood, state, reward, t):
        t = t.detach()
        y = self(mood, state, reward)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ?????????????????????????????????
class ReasonableMotivator(nn.Module):
    """
    ?????????????????????????????????(?????????????????????????????????)?????????
    ??????(??????)????????????????????????(??????)???????????????
    mood, state, reward???????????????next_state?????????

    Args:
        mood (int): agent?????????(??????)
        state (ndarray): state_t0
        reward (int): ????????????
    Returns:
        next_state (tensor): state_t1
    """

    def __init__(self, n_in, n_mid, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l2 = nn.Linear(n_mid, n_mid)
        self.l3 = nn.Linear(n_mid, n_out)

        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, mood, state, reward):
        x = np.array(mood)
        x = np.append(x, state)
        x = np.append(x, [reward])
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def update(self, mood, state, reward, t):
        t = torch.tensor(t[np.newaxis, :], dtype=torch.float32)
        y = self(mood, state, reward)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ??????????????????????????????????????????(?????????????????????????????????????????????????????????)?????????
class ActionMaker(nn.Module):
    """
    state_t0??????state_t1???????????????????????????????????????????????????(?????????????????????????????????????????????)????????????????????????

    Args:
        state (ndarray): state_t0
        next_state (ndarray): state_t1
    Returns:
        action (tensor): ????????????
    """

    def __init__(self, n_in, n_mid, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, n_mid)
        self.l2 = nn.Linear(n_mid, n_out)

        self.lr = 0.001
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, state, next_state):
        x = np.append(state, next_state)
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)

        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def update(self, state, next_state, t):
        y = self(state, next_state)
        loss_fn = nn.MSELoss()
        t = t.detach()
        loss = loss_fn(y, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
