# src/risk/position_sizing.py
"""
translates position into a # of shares to trade, capped at a fraction of account equity
"""

import math

MAX_POSITION_PCT = 0.50  # fraction of equity a single position may occupy on entry

ACTION_TARGET_DIRECTION = {
    0: 0,   # hold -> flat
    1: 1,   # buy -> long
    2: -1,  # short -> short
}


def compute_order_delta(action: int, equity: float, current_price: float, current_shares: float) -> int:
    """
    Returns the signed number of shares to trade to move from the current
    position to the target direction implied by `action`. Positive = buy,
    negative = sell/short, zero = no order needed.
    """
    old_direction = 0 if current_shares == 0 else (1 if current_shares > 0 else -1)
    new_direction = ACTION_TARGET_DIRECTION[action]

    if new_direction == old_direction:
        return 0  # no rebalancing just because price moved

    if new_direction == 0:
        target_shares = 0
    else:
        max_dollar_exposure = equity * MAX_POSITION_PCT
        target_shares = new_direction * math.floor(max_dollar_exposure / current_price)

    return int(target_shares - current_shares)