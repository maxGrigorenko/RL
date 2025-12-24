'''Реализуем DQN with Prioritized Experience Replay (по аналогии с 4-ой практикой)'''

import os
import sys
import gymnasium as gym

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import traci
from stable_baselines3.dqn.dqn import DQN

from sumo_rl import SumoEnvironment
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def softmax(xs, temp=1.0):
    exp_xs = np.exp((xs - xs.max()) / temp)
    return exp_xs / exp_xs.sum()


class PrioritizedReplayBuffer:
    def __init__(self, bufsize):
        self.buffer = deque(maxlen=bufsize)
        self.rng = np.random.default_rng()

    def push(self, state, action, reward, next_state, done, priority=1.0):
        transition = (priority, state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batchsize, temp=1.0):
        n_samples = min(batchsize, len(self.buffer))
        priorities = np.array([sample[0] for sample in self.buffer])
        probs = softmax(symlog(priorities), temp=temp)
        indices = self.rng.choice(len(self.buffer), size=n_samples, p=probs)
        samples = [self.buffer[i] for i in indices]
        _, states, actions, rewards, next_states, dones = zip(*samples)

        batch = (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
        return batch, indices

    def update_priorities(self, indices, batch, new_priorities):
        states, actions, rewards, next_states, dones = batch
        for i, idx in enumerate(indices):
            updated_transition = (
                new_priorities[i],
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            )
            self.buffer[idx] = updated_transition

    def sort_by_priority(self):
        sorted_items = sorted(self.buffer, key=lambda x: x[0])
        self.buffer = deque(sorted_items, maxlen=self.buffer.maxlen)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=1e-3,
            gamma=0.99,
            eps_begin=0.4,
            eps_end=0.01,
            eps_decay=0.99,
            bufsize=50000,
            batchsize=64,
            target_upd_freq=500,
            soft_t=1.0,
            buf_sort_freq=1000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batchsize = batchsize
        self.target_upd_freq = target_upd_freq
        self.soft_t = soft_t
        self.buf_sort_freq = buf_sort_freq
        self.epsilon = eps_begin
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = torch.device("mps") # MacBook M1
        print(f"Using device: {self.device}")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.buffer = PrioritizedReplayBuffer(bufsize)
        self.train_steps = 0

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done, priority=1.0):
        self.buffer.push(state, action, reward, next_state, done, priority)

    def train_step(self):
        if len(self.buffer) < self.batchsize:
            return None

        batch, indices = self.buffer.sample(self.batchsize, temp=self.soft_t)
        states, actions, rewards, next_states, dones = batch

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(1)
            next_q_value = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + self.gamma * next_q_value * (1 - dones_t)

        td_errors = torch.abs(q_value - target).detach().cpu().numpy()
        self.buffer.update_priorities(indices, batch, td_errors)

        loss = nn.functional.mse_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_upd_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_steps % self.buf_sort_freq == 0:
            self.buffer.sort_by_priority()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)
        print(f"Model saved to {path}")


def train():
    episodes = 10
    lr = 1e-3
    gamma = 0.99
    bufsize = 50000
    batchsize = 64
    target_upd_freq = 50
    lr_starts = 100
    eps_begin = 0.5
    eps_end = 0.01
    eps_decay = 0.99
    soft_t = 1.0
    buf_sort_freq = 100
    episode_steps = 500

    os.makedirs("outputs/big-intersection", exist_ok=True)

    env = SumoEnvironment(
        net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
        route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
        single_agent=True,
        out_csv_name="outputs/big-intersection/my_dqn",
        use_gui=False,
        num_seconds=5400,
        yellow_time=4,
        min_green=5,
        max_green=60,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"{state_dim=}, {action_dim=}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        eps_begin=eps_begin,
        eps_end=eps_end,
        eps_decay=eps_decay,
        bufsize=bufsize,
        batchsize=batchsize,
        target_upd_freq=target_upd_freq,
        soft_t=soft_t,
        buf_sort_freq=buf_sort_freq,
    )

    rewards_history = []
    best_reward = float('-inf')

    for episode in range(1, episodes + 1):
        print(f"{episode=}")
        total_steps = 0
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        done = False

        while not done and total_steps < episode_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, done, priority=1.0)

            if total_steps >= lr_starts:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                agent.decay_epsilon()

            state = next_state
            episode_reward += reward
            total_steps += 1

        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-10:])
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        print(f"{episode=}, {episode_reward=}, {avg_reward=}, {avg_loss=}, {agent.epsilon=}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("outputs/big-intersection/best_model.pt")

    agent.save("outputs/big-intersection/final_model.pt")
    env.close()

    print("\n" + "=" * 60)
    print(f"{best_reward=}")
    print("=" * 60)


if __name__ == "__main__":
    train()

'''
env = SumoEnvironment(
    net_file="sumo_rl/nets/big-intersection/big-intersection.net.xml",
    single_agent=True,
    route_file="sumo_rl/nets/big-intersection/routes.rou.xml",
    out_csv_name="outputs/big-intersection/dqn",
    use_gui=False,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = DQN(
    env=env,
    policy="MlpPolicy",
    lr=1e-3,
    lr_starts=0,
    bufsize=50000,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
)
model.learn(total_timesteps=10_000)
'''