import gymnasium as gym
import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple, deque
import random

env = gym.make("CartPole-v1")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
   def __init__(self, capacity):
      self.memory = deque([], maxlen = capacity)

   def push(self, *args):
      #save a transition
      self.memory.append(Transition(*args))

   def sample(self, batch_size):
      return random.sample(self.memory, batch_size)

   def __len__(self):
      return len(self.memory)

class QNetwork(torch.nn.Module):
   def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        #fc1 is input to hidden
        self.fc1 = torch.nn.Linear(state_dim, 128)

        #fc2 is hidden to hidden
        self.fc2 = torch.nn.Linear(128, 128)

        #fc3 is hidden to output
        self.fc3 = torch.nn.Linear(128, action_dim)

        self.relu = torch.nn.ReLU()

   def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.fc3(x)
      return x

#epsilon-greedy action selection
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
tau = 0.005
learning_rate = 3e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = QNetwork(n_observations, n_actions).to(torch.device("cpu"))
target_net = QNetwork(n_observations, n_actions).to(torch.device("cpu"))
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr = learning_rate, amsgrad = True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
   global steps_done
   sample = np.random.rand()
   eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1 * steps_done / epsilon_decay)
   steps_done +=1
   if sample > eps_threshold:
      with torch.no_grad():
      #return the policy with the highest reward
         return policy_net(state).argmax(dim = 1).view(1, 1)
   else:
      return torch.tensor([[env.action_space.sample()]], device = torch.device("cpu"), dtype = torch.long)


episode_durations = []

def plot_durations(show_result=False):
   plt.figure(1)
   durations_t = torch.tensor(episode_durations, dtype = torch.float)
   if show_result:
      plt.title('Result')
   else:
      plt.clf()
      plt.title('Training...')
   plt.xlabel('Episode')
   plt.ylabel('Duration')
   plt.plot(durations_t.numpy())

   #take 100 episodes averages and plot them too
   if len(durations_t) >= 100:
      means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
      means = torch.cat((torch.zeros(99), means))
      plt.plot(means.numpy())

   plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
   if len(memory) < batch_size:
      return
   transitions = memory.sample(batch_size)

   batch = Transition(*zip(*transitions))

   #compute a mask of non final states and concatenate the batch elements
   non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = torch.device("cpu"), dtype = torch.bool)
   non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

   state_batch = torch.cat(batch.state)
   action_batch = torch.cat(batch.action)
   reward_batch = torch.cat(batch.reward)

   state_action_values = policy_net(state_batch).gather(1, action_batch)

   next_state_values = torch.zeros(batch_size, device = torch.device("cpu"))
   with torch.no_grad():
      next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

   expected_state_action_values = (next_state_values * gamma) + reward_batch

   #compute Huber loss
   criterion = torch.nn.SmoothL1Loss()
   loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

   #optimize the model
   optimizer.zero_grad()
   loss.backward()

   #in-place gradient clipping
   torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
   optimizer.step()


#training loop
if torch.cuda.is_available():
   num_episodes = 600
else:
   num_episodes = 600

for i_episode in range(num_episodes):
   #initialize the environment and state
   state, info = env.reset()
   state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

   for t in count():
      #select and perform an action
      action = select_action(state)
      next_state, reward, terminated, truncated, info = env.step(action.item())
      done = terminated or truncated
      reward = torch.tensor([reward], device = torch.device("cpu"))

      if not done:
         next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
      else:
         next_state = None

      #store the transition in memory
      memory.push(state, action, next_state, reward)

      #move to the next state
      state = next_state

      #perform one step of the optimization (on the policy network)
      optimize_model()
      if done:
         episode_durations.append(t + 1)
         plot_durations()
         break

   #update the target network, copying all weights and biases in DQN
   if i_episode % 10 == 0:
      target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plot_durations(show_result = False)
plt.ioff()
plt.show()