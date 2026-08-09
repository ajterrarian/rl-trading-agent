import sys
import os
import torch
import torch.optim as optim
import numpy as np
import collections

from src.agent.dqn import QNetwork, ReplayMemory, Agent
from src.data.pipeline import get_data, add_features, chronological_split
from src.env.trading_env import MarketTradingEnv

device = torch.device("cpu")

#load data and prepare environment
df = get_data(ticker = 'SPY', start = '2001-01-01')
df = add_features(df)
train_df, val_df, test_df = chronological_split(df)

env = MarketTradingEnv(train_df)

n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

#network/optimizer setup
policy_net = QNetwork(n_observations, n_actions).to(device)
target_net = QNetwork(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr = 3e-4, amsgrad = True)
memory = ReplayMemory(10000)

#create agent
agent = Agent(
    n_actions = n_actions,
    policy_net=policy_net,
    target_net=target_net,
    optimizer=optimizer,
    memory=memory,
    device=device,
)

#training loop
num_episodes = 20
episode_net_worths = []

for i_episode in range(num_episodes):
   state, _ = env.reset()
   state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

   current_net_worth = env.initial_balance

   while True:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated
        reward_tensor = torch.tensor([reward], device = device)
        current_net_worth = info["Net Worth"]

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)

        #store in memory
        agent.memory.push(state, action, next_state, reward_tensor)
        state = next_state

        #perform optimization step
        agent.optimize_model()

        #soft update target network
        agent.update_target_network()

        if done:
            episode_net_worths.append(current_net_worth)
            break

   print(f"Episode {i_episode + 1}/{num_episodes} | Final Net Worth: ${current_net_worth:,.2f}")


greedy_episodes = 1
greedy_episode_net_worths = []
counter = collections.Counter()

env.reset()
for i_episode in range(greedy_episodes):
   state, _ = env.reset()
   state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

   greedy_current_net_worth = env.initial_balance

   while True:
        action = agent.act_greedy(state)
        observation, reward, terminated, truncated, info = env.step(action.item())

        done = terminated or truncated
        reward_tensor = torch.tensor([reward], device = device)
        greedy_current_net_worth = info["Net Worth"]

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)

        if done:
            greedy_episode_net_worths.append(greedy_current_net_worth)
            break

        counter.update([action.item()])
   print(f"Greedy Episode {i_episode + 1}/{greedy_episodes} | Greedy Final Net Worth: ${greedy_current_net_worth:,.2f} | Counter = {counter}")

env.close()

#performance summary
final_avg_net_worth = np.mean(episode_net_worths)
print("--------------------------------------------------")
print(f"Training Complete!")
print(f"Initial Starting Balance: ${env.initial_balance:,.2f}")
print(f"Average Final Net Worth across all episodes: ${final_avg_net_worth:,.2f}")


# checkpointing
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok = True)
checkpoint_path = os.path.join(checkpoint_dir, "dqn_trading_v1.pt")

torch.save(policy_net.state_dict(), checkpoint_path)
print(f"Saved policy network checkpoint to {checkpoint_path}")
