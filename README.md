# RL Trading Agent

From-scratch Deep Q-Network (DQN) trading agent with honest train/validation/test separation and no lookahead bias compared against simple baselines (buy-and-hold, random, and do-nothing). The agent trains and validates offline on historical data and the live paper-trading window is a frozen-polcuy evaluation period.

## Status

- [x] RL fundamentals (MDP, Q-learning, Bellman update, replay buffer, target network) — done
- [x] DQN implemented from scratch, validated against `CartPole-v1` — done
- [x] Data pipeline + custom trading environment — done
- [x] Train on trading environment, validate vs. baselines – done
- [x] Risk layer, Alpaca paper execution, deploy frozen policy – done
- [ ] Weeks 3-5: Live paper-trading window (monitoring only, no further training) – in progress
- [ ] Final writeup: backtest vs. live, honest diagnosis of the gap

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For live execution (Alpaca paper trading), copy `.env.example` to `.env` and fill in your Alpaca paper trading API credentials:

\`\`\`
cp .env.example .env
\`\`\`

## Structure

```
src/
  agent/       DQN implementation (QNetwork, ReplayMemory, Agent) -- reused unchanged for CartPole and the trading env
  env/         Custom gymnasium.Env for trading
  data/        Historical data pipeline (yfinance, feature engineering, chronological splits)
  evaluation/  Baseline strategies (buy-and-hold, random, do-nothing) for comparison against the agent
  risk/        Kill switch (manual + automatic drawdown halt), data sanity checks, position sizing
  monitoring/  Structured logging of live execution outcomes
scripts/       Entry points: training, evaluation, Alpaca connection check, daily live execution, smoke tests
.github/workflows/  Scheduled GitHub Actions workflow running daily execution unattended
data/          Local data cache (gitignored)
```
