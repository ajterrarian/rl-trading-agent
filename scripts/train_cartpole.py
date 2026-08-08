# train_cartpole.py
import gymnasium as gym
import torch
import torch.optim as optim
from src.agent.dqn import QNetwork, ReplayMemory, Agent

device = torch.device("cpu")

# 1. Environment Setup
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

# 2. Networks & Optimizer Setup
policy_net = QNetwork(n_observations, n_actions).to(device)
target_net = QNetwork(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=3e-4, amsgrad=True)
memory = ReplayMemory(10000)

# 3. Create Agent
agent = Agent(
    n_actions=n_actions,
    policy_net=policy_net,
    target_net=target_net,
    optimizer=optimizer,
    memory=memory,
    device=device
)

# 4. Training Loop
num_episodes = 600
episode_durations = []

for i_episode in range(num_episodes):
   state, _ = env.reset()
   state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


   for t in range(500):
      action = agent.select_action(state)
      observation, reward, terminated, truncated, _ = env.step(action.item())
        
      done = terminated or truncated
      reward = torch.tensor([reward], device=device)

      if done:
         next_state = None
      else:
         next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

      # Store in memory
      agent.memory.push(state, action, next_state, reward)
      state = next_state

      # Perform optimization step
      agent.optimize_model()

      # Soft update target network
      agent.update_target_network()

      if done:
         episode_durations.append(t+1)
         break

      if (i_episode + 1) % 50 == 0:
         recent = episode_durations[-100:]
         print(f"Episode {i_episode + 1}/{num_episodes} | avg duration (last {len(recent)}): {sum(recent)/len(recent):.1f}")

env.close()
final_avg = sum(episode_durations[-100:]) / len(episode_durations[-100:])
print(f"Training complete. Avg duration over last 100 episodes: {final_avg:.1f}")