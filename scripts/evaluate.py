import os
import torch

from src.agent.dqn import QNetwork
from src.evaluation.baselines import run_do_nothing, run_buy_and_hold, run_random_agent
from src.data.pipeline import get_data, add_features, chronological_split
from src.env.trading_env import MarketTradingEnv

def run_dqn_policy(env, policy_net, device):
    #evaluate the loaded policy network w/ inline inference
    policy_net.eval()
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)

    while True:
        with torch.no_grad():
            action = policy_net(state).argmax(dim = 1).item()

        observation, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            return info["Net Worth"]

        state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)


def main():
    device = torch.device("cpu")

    #load data and prepare validation environment
    df = get_data(ticker = "SPY", start = "2001-01-01")
    df = add_features(df)
    _, val_df, _ = chronological_split(df)

    env = MarketTradingEnv(val_df)

    n_actions = env.action_space.n
    sample_state, _ = env.reset()
    n_observations = len(sample_state)


    #load checkpoint into q network
    checkpoint_path = os.path.join("checkpoints", "dqn_trading_v1.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first!")

    policy_net = QNetwork(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model checkpoint from: {checkpoint_path}")

    #evaluate strategies on validation data
    print("Evaluating strategies on validation set...")
    dqn_net_worth = run_dqn_policy(env, policy_net, device)
    do_nothing_nw = run_do_nothing(env)
    buy_hold_nw = run_buy_and_hold(env)
    random_avg_nw = run_random_agent(env, num_trials=50)

    #summary table
    print("\n" + "=" * 55)
    print(f"{'Strategy (Validation Set)':<30} | {'Final Net Worth':<20}")
    print("=" * 55)
    print(f"{'Initial Starting Balance':<30} | ${env.initial_balance:>18,.2f}")
    print(f"{'Do Nothing (Hold)':<30} | ${do_nothing_nw:>18,.2f}")
    print(f"{'Buy & Hold (Long)':<30} | ${buy_hold_nw:>18,.2f}")
    print(f"{'Random Agent (50 avg)':<30} | ${random_avg_nw:>18,.2f}")
    print(f"{'DQN Greedy Policy':<30} | ${dqn_net_worth:>18,.2f}")
    print("=" * 55)

    env.close()

if __name__ == "__main__":
    main()