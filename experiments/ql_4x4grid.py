import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

import numpy as np
from collections import defaultdict


class TheAgent:
    def __init__(
            self,
            starting_state,
            state_space,
            action_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1,
            epsilon_min=0.01,
            epsilon_decay=0.995
    ):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

        self.action = None
        self.acc_rew = 0
        self.steps = 0
        self.episode_rewards = []

    def act(self):
        if np.random.random() < self.epsilon:
            self.action = self.action_space.sample()
        else:
            q_values = self.q_table[self.state]
            max_q = np.max(q_values)
            best_act = np.where(q_values == max_q)[0]
            self.action = np.random.choice(best_act)
        return self.action

    def learn(self, next_state, reward, done=False):
        current_q = self.q_table[self.state][self.action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[self.state][self.action] = current_q + self.alpha * (target_q - current_q)
        self.state = next_state
        self.acc_rew += reward
        self.steps += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode_stats(self):
        self.episode_rewards.append(self.acc_rew)
        self.acc_rew = 0


if __name__ == "__main__":

    import os
    os.environ["LIBSUMO_AS_TRACI"] = "0"

    alpha = 0.1
    gamma = 0.99
    decay = 0.99
    runs = 2
    episodes = 4

    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=1000,
        min_green=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        '''
        ql_agents = {
            ts: TheAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                # exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)
                epsilon=0.05,
                epsilon_min=0.005,
                epsilon_decay=decay
            )
            for ts in env.ts_ids
        }
        '''
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            print(f"{episode=}")
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                    # ql_agents[agent_id].decay_epsilon()

            env.save_csv(f"outputs/4x4/baseline_ql-4x4grid_run{run}", episode)

    env.close()
