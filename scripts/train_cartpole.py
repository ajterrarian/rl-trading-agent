"""
Day 2 validation gate: implement DQN from scratch and confirm it reliably
solves CartPole-v1 before touching any market data.

This is intentionally a skeleton, not a working implementation -- the point
of this script is that you write it, per the project spec's "from scratch"
framing. Use it as a checklist, not a fill-in-the-blank.

Pieces to build (see src/agent/ -- the DQN class you write there should be
generic enough to reuse unchanged against the real trading environment later):

1. Sanity-check the environment first with random actions, before any
   network exists. `import gymnasium as gym; env = gym.make("CartPole-v1")`.
   State is 4 numbers (cart position, cart velocity, pole angle, pole
   angular velocity); 2 discrete actions (push left/right).

2. Q-network -- small MLP, input = 4-number state, output = one Q-value
   per action.

3. Epsilon-greedy action selection -- random action w.p. epsilon, else
   argmax of the Q-network's output. Decay epsilon over training.

4. Replay buffer -- fixed-size store of (state, action, reward,
   next_state, done) tuples; sample random minibatches to train on,
   instead of training on consecutive correlated frames.

5. Target network -- a synced-periodically copy of the Q-network, used
   only to compute the TD target, so training doesn't chase a constantly
   shifting target.

6. Training loop -- TD target = reward + gamma * max(Q_target(next_state))
   * (1 - done). Loss = MSE/Huber between Q(state, action) and that target.
   Backprop, step the optimizer, sync the target network periodically,
   decay epsilon.

Validation target: CartPole-v1 is considered solved around an average
reward of ~475-500 over 100 consecutive episodes (max episode length is
500). Plot or print reward per episode -- you want to see it actually
climb.
"""

# TODO: everything.
