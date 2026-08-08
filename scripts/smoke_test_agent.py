# scripts/smoke_test_agent.py
import torch
from torch import optim
from src.agent.dqn import Agent, QNetwork, ReplayMemory

state_dim, action_dim = 4, 3
policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=3e-4)
memory = ReplayMemory(1000)

agent = Agent(action_dim, policy_net, target_net, optimizer, memory)

# fill memory with junk transitions so optimize_model has enough to sample
for _ in range(200):
    s = torch.rand(1, state_dim)
    a = torch.tensor([[agent.select_action(s).item()]])
    ns = torch.rand(1, state_dim)
    r = torch.tensor([1.0])
    memory.push(s, a, ns, r)

agent.optimize_model()   # should run without crashing
agent.update_target_network()
print("smoke test passed")