# From-Scratch RL Trading Agent — Project Spec

Successor project to `kalshi-trading-bot`. That project's infra lessons (data → risk → execution → logging → deployment → reporting) carry forward; the venue and the strategy layer change.

**Career context:** quant trading > ML/AI engineering > SWE > quant dev. This project sits at the intersection of the top two.

---

## 0. Framing — read this before you start building

Same honest framing as the arb project: **the point is not "make money."** RL-for-trading is a genre that experienced quant people are openly skeptical of — it's sample-inefficient, markets are non-stationary, and it's very easy to fool yourself with a backtest that looks great and means nothing. None of that makes it a bad project. It makes *rigor* the actual deliverable: correct train/val/test separation, an honest backtest-vs-live comparison, and a clear-eyed writeup of where the agent's learned behavior breaks down.

**Decision baked into this spec:** the agent trains and validates entirely offline on historical data. The live paper-trading window is a **frozen-policy evaluation period**, not additional training. Two to three weeks of live data is nowhere near enough to responsibly update model weights.

**Assumption on "from scratch":** you implement the RL algorithm's update logic yourself (Q-network architecture, the Bellman update, the training loop, epsilon-greedy exploration) using PyTorch for autograd — not a hand-rolled backprop engine, and not `stable-baselines3` or similar RL libraries.

## 1. Objectives

- Implement a Deep Q-Network (DQN) RL agent from scratch that learns a trading policy (buy / hold / sell) on historical equities data.
- Validate it properly: chronological train/validation/test split, comparison against baselines (buy-and-hold, random policy, do-nothing), no lookahead bias.
- Deploy the frozen, validated policy to live paper trading (Alpaca) for 2-3 weeks.
- Produce an honest writeup: backtest performance vs. live performance, the size and likely causes of the gap, and what would need to change for this to be a real strategy.

## 2. Prerequisites

- Python / numpy — have this already from the Kalshi project.
- Basic probability & stats.
- Gradient descent & backprop, conceptually (not hand-derived — using PyTorch autograd).
- PyTorch basics — tensors, `nn.Module`, `optimizer.step()`, autograd.
- RL fundamentals: MDP formalism, exploration-exploitation/epsilon-greedy, Q-learning, why DQN needs a replay buffer and a target network.

If RL is entirely new: spend 3-5 days first on RL fundamentals + implementing DQN against `CartPole-v1` before touching market data (see Day 1-2 below).

## 3. Recommended technologies

- Data: `yfinance`. Start with one liquid instrument (e.g. SPY).
- RL environment interface: `gymnasium`.
- Deep learning: PyTorch.
- Paper trading: Alpaca's paper trading API.
- Persistence: SQLite or CSV.
- Reference only, not a starting point: FinRL (`AI4Finance-Foundation/FinRL`) — check your own implementation against it after building, not before.

## 4. Architecture

1. Data pipeline — historical OHLCV, modest feature set, chronological train/val/test split.
2. Trading environment (`gymnasium.Env`) — state = market features + position; action = `{sell, hold, buy}`; reward = P&L net of transaction costs.
3. DQN agent (from scratch) — Q-network, replay buffer, target network, epsilon-greedy, training loop.
4. Toy-environment validation gate — solve `CartPole-v1` with the DQN before touching market data.
5. Training & offline validation — train, validate against baselines.
6. Risk layer — position limits, max daily loss, kill switch.
7. Paper execution + logging — Alpaca integration, persistence, alerting.
8. Reporting — daily summary live; final backtest-vs-live writeup.

## 5. Compressed 10-day build schedule

Live paper-trading (2-3 weeks) starts after Day 10 and runs separately/passively.

- **Day 1:** RL fundamentals (MDP, Q-learning, Bellman update, why DQN needs replay buffer + target network).
- **Day 2:** Implement DQN from scratch against `CartPole-v1`. If this slips to Day 3, let it slip.
- **Day 3:** Data pipeline — `yfinance` pull, minimal features, chronological split.
- **Day 4:** Build the `gymnasium.Env` trading environment, validate with a random-action agent first.
- **Days 5-6:** Train the DQN against the trading environment.
- **Day 7:** Validate vs. baselines (buy-and-hold, random, do-nothing). Go/no-go checkpoint.
- **Day 8:** Risk layer + Alpaca paper-trading integration.
- **Day 9:** Logging/persistence + end-to-end dry run.
- **Day 10:** Freeze the policy, deploy. Live monitoring window begins.

## 6. Testing strategy

- Unit tests on the environment's `step()` logic against hand-computed examples.
- Sanity check for degenerate policies (e.g. always holding to avoid transaction costs).
- Baseline comparison is itself a test — not beating "do nothing" is a real, reportable result.

## 7. Expected learning outcomes

- Genuine, first-principles understanding of the RL training loop.
- Direct experience with RL-for-trading failure modes (non-stationarity, reward hacking, degenerate policies, the backtest-live gap).
- A defensible, honestly-reported result that's genuinely yours end to end.

## 8. Possible extensions

- Continuous position sizing via PPO, once DQN is solid.
- True online learning during live trading (longer window, proper safeguards).
- Multi-asset portfolios.
- Returning to Kalshi with this same RL machinery, pooled across many concurrent markets.

## 9. Learning resources

**RL fundamentals**
- Sutton & Barto, *Reinforcement Learning: An Introduction* — Ch. 3 (MDPs), §6.5 (Q-learning). Free: incompleteideas.net/book/the-book-2nd.html
- Hugging Face Deep RL Course — Unit 2 (Q-Learning), Unit 3 (Deep Q-Learning).
- PyTorch official DQN tutorial (CartPole, replay buffer, target network) — docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

**Deep learning / PyTorch**
- PyTorch "60 Minute Blitz" tutorial.
- 3Blue1Brown's neural network series, if backprop intuition is shaky.

**Trading-specific rigor**
- Marcos López de Prado, *Advances in Financial Machine Learning* — lookahead bias, proper backtesting.
- FinRL (GitHub) — reference/comparison only.

**Tools**
- `yfinance`, `gymnasium`, Alpaca API docs.
