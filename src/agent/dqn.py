import torch
import numpy as np
from torch import optim
from itertools import count
from collections import namedtuple, deque
import random

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


#Agent class
class Agent:
    def __init__(
        self,
        n_actions,
        policy_net,
        target_net,
        optimizer,
        memory,
        batch_size = 128,
        gamma = 0.99,
        epsilon_start = 1.0,
        epsilon_end = 0.01,
        epsilon_decay = 500,
        tau = 0.005,
        device = torch.device("cpu")
   ):
        self.n_actions = n_actions
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = memory

        #hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.device = device

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1 * self.steps_done / self.epsilon_decay
        )
        self.steps_done +=1

        if sample > eps_threshold:
            with torch.no_grad():
                #return the policy with the highest reward
                return self.policy_net(state).argmax(dim = 1).view(1, 1)
        else:
            #environment agnostic random action
            random_action = random.randrange(self.n_actions)
            return torch.tensor([[random_action]], device = self.device, dtype = torch.long)


    def optimize_model(self):
        #performs one step of the optimization on the policy network
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        #compute a mask of non final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device = self.device,
            dtype = torch.bool,
        )

        #filter non final next states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #Q values predicted by policy net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        #Value of next state for all next states
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        #expected q values: reward + gamma * max a
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        #compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        #optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        #in-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        #Soft update of target network parameters: theta_target = tau*theta_local + (1 - tau)*theta_target
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)