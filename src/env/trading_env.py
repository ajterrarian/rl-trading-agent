import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple, deque
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
from pipeline import get_data, add_features, chronological_split

class MarketTradingEnv(gym.Env):
    #custom env for market trading
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, df):
        super(MarketTradingEnv, self).__init__()

        #store the df internally
        self.df = df
        self.current_step = 0

        #transaction cost percent (broker fee)
        self.transaction_cost_pct = 0.001

        self.action_space = spaces.Discrete(3)
        self.action_mapping = {
            0: 0,   #hold
            1: 1,   #buy
            2: -1,  #short
        }
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (4,), dtype = np.float32)

        #portfolio attributes
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.shares_held = 0

        self.previous_net_worth = self.initial_balance

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.current_step = 0 
        self.balance = self.initial_balance
        self.shares_held = 0
        self.previous_net_worth = self.initial_balance

        return self._get_obs(), {}
    
    def _get_obs(self):
        current_row = self.df.iloc[self.current_step]

        ret = current_row['return']
        ma_10 = current_row['ma_10']
        volatility_10 = current_row['volatility_10']

        #determine position status
        if self.shares_held > 0:
            current_position = 1.0      #long

        elif self.shares_held < 0:
            current_position = -1.0     #short

        else:
            current_position = 0.0      #hold

        return np.array([ret, ma_10, volatility_10, current_position], dtype = np.float32)

    def step(self, action):
        current_row = self.df.iloc[self.current_step]
        current_price = current_row['Close']

        target_position = self.action_mapping[action]

        #track current positions before making a trade
        if self.shares_held > 0:
            old_position = 1
        elif self.shares_held < 0:
            old_position = -1
        else:
            old_position = 0

        #calculate transition cost if position changes
        transaction_cost = 0.0
        if target_position != old_position:
            transaction_cost = abs(self.shares_held) * current_price * self.transaction_cost_pct

            #execute trade transitions
            #1. liquidate old position first
            if old_position == 1:       #close long
                self.balance += self.shares_held * current_price
            elif old_position == -1:    #close short
                self.balance -= (abs(self.shares_held) * current_price)

            self.shares_held = 0



            #2. open new targeted position
            if target_position == 1:        #open long (buy as many whole shares as possible)
                max_shares = ((self.balance - transaction_cost) // current_price)
                if max_shares > 0:
                    self.shares_held = max_shares
                    self.balance -= (self.shares_held * current_price)
            elif target_position == -1:     #open short (short max whole shares)
                max_shares = ((self.balance - transaction_cost) // current_price)
                if max_shares > 0:
                    self.balance += (max_shares * current_price)
                    self.shares_held = -max_shares

            #deduct trading fee
            self.balance -= transaction_cost

        self.current_step += 1

        #fetch next day price to evaluate portfolio change
        next_row = self.df.iloc[self.current_step]
        next_price = next_row['Close']

        #calculate net worth
        net_worth = self.balance + (self.shares_held * next_price)

        #reward (incrmental) + cache the new networth
        reward = net_worth - self.previous_net_worth
        self.previous_net_worth = net_worth

        #termination rules for episode
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {"Net Worth": net_worth, "Cost Paid": transaction_cost}


if __name__ == "__main__":
    df = get_data(ticker="SPY", start="2001-01-01")
    df = add_features(df)
    train, val, test = chronological_split(df)

    env = MarketTradingEnv(train)
    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Initial balance:", env.balance, "| shares:", env.shares_held)

    # Deliberate, not random -- force one of each action so you're not
    # relying on random.sample() happening to hit every branch by luck.
    test_actions = [1, 1, 0, 2, 2, 0]  # buy, stay long, go flat, short, stay short, go flat
    for step, action in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {step} | action={action} | reward={reward:.2f} | "
              f"balance={env.balance:.2f} | shares={env.shares_held} | "
              f"net_worth={info['Net Worth']:.2f} | cost={info['Cost Paid']:.2f}")
        if terminated:
            break

    # Then a longer random smoke test, just checking nothing crashes over many steps
    obs, info = env.reset()
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Terminated at step", step)
            break
    print("Random smoke test completed without crashing.")