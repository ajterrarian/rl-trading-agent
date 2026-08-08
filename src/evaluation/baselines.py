import torch.optim as optim
import numpy as np

from src.data.pipeline import get_data, add_features, chronological_split
from src.env.trading_env import MarketTradingEnv

def run_do_nothing(env):
    #always hold -- networth remains flat
    state, _ = env.reset()
    while True:
        action = 0 # hold
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            return info["Net Worth"]

def run_buy_and_hold(env):
    #always buys and opens a long position on step 1
    state, _ = env.reset()
    while True:
        action = 1 #Buy/Long
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            return info["Net Worth"]

def run_random_agent(env, num_trials = 100):
    #takes a random action from the action space at each step
    #averages results across multiple trials
    final_net_worths = []

    for _ in range(num_trials):
        state, _ = env.reset()
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                final_net_worths.append(info["Net Worth"])
                break

    return np.mean(final_net_worths)