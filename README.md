# RL Trading Agent

From-scratch Deep Q-Network (DQN) trading agent. Full project spec (objectives, architecture, timeline, learning resources) lives in the project doc — see `PROJECT_SPEC.md`.

**What this proves, and what it doesn't (read before judging results):** this project is not trying to prove the agent is profitable. Two to three weeks of live paper trading cannot prove that, and it's not the point. The point is engineering rigor: a correctly-implemented DQN, honest train/validation/test separation with no lookahead bias, a fair comparison against simple baselines (buy-and-hold, random, do-nothing), and an honest writeup of the gap between backtest and live performance — including if the agent underperforms the baselines. That result is reportable, not a failure to hide.

**Training approach:** the agent trains and validates entirely offline on historical data. The live paper-trading window is a frozen-policy evaluation period, not additional training — there isn't enough live data in a few weeks to responsibly update weights without just fitting to noise.

## Status

- [ ] Day 1: RL fundamentals (MDP, Q-learning, Bellman update, replay buffer, target network) — done
- [ ] Day 2: DQN implemented from scratch, validated against `CartPole-v1`
- [ ] Days 3-4: Data pipeline + custom trading environment
- [ ] Days 5-7: Train on trading environment, validate vs. baselines
- [ ] Days 8-10: Risk layer, Alpaca paper execution, deploy frozen policy
- [ ] Weeks 3-5: Live paper-trading window (monitoring only, no further training)
- [ ] Final writeup: backtest vs. live, honest diagnosis of the gap

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
src/
  agent/        DQN implementation — reused unchanged for both CartPole and the real trading env
  env/           Custom gymnasium.Env for trading (built in Days 3-4)
  data/          Historical data pipeline (yfinance, feature engineering, chronological splits)
  risk/          Position limits, max daily loss, kill switch
  execution/     Alpaca paper trading integration
  persistence/   Trade/decision logging
scripts/         Entry-point scripts (training, deployment)
tests/           Unit tests
data/            Local data cache (gitignored)
```
