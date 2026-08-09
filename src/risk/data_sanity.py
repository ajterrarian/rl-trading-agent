# src/risk/data_sanity.py
"""
Validates a freshly fetched market reading before it's trusted enough to feed into the model.
Catches: missing/NaN values, implausible single-day returns, and a moving average that's diverged unreasonably far from the current price (more likely a data alignment bug than a real market condition).
"""

import numpy as np

MAX_ABS_DAILY_RETURN = 0.20   # SPY's worst historical single days (2008, 2020) were roughly -9% to -12%; 20% is a generous ceiling that still catches  glitches
MAX_MA_DEVIATION_PCT = 0.30   # a 10-day moving average shouldn't be more than ~30% from the current price under any real market condition


def check_data_sanity(current_price: float, ret: float, ma_10: float, volatility_10: float) -> bool:
    """
    Returns True if the data looks trustworthy enough to act on,
    False if it should be rejected (caller should skip trading, not guess).
    """
    values = [current_price, ret, ma_10, volatility_10]

    # 1. Missing values
    if any(v is None for v in values):
        print("[DATA SANITY] Missing value in fetched data. Rejecting.")
        return False

    # 2. NaN or inf
    if not all(np.isfinite(v) for v in values):
        print("[DATA SANITY] NaN or inf in fetched data. Rejecting.")
        return False

    # 3. Implausible single-day return.
    if abs(ret) > MAX_ABS_DAILY_RETURN:
        print(f"[DATA SANITY] Return {ret:.2%} exceeds plausible bound of {MAX_ABS_DAILY_RETURN:.0%}. Rejecting.")
        return False

    # 4. Non-positive price
    if current_price <= 0:
        print(f"[DATA SANITY] Non-positive price (${current_price}). Rejecting.")
        return False

    # 5. Negative volatility (a rolling std can't be negative)
    if volatility_10 < 0:
        print(f"[DATA SANITY] Negative volatility ({volatility_10}). Rejecting.")
        return False

    # 6. Moving average too far from current price
    ma_deviation = abs(current_price - ma_10) / current_price
    if ma_deviation > MAX_MA_DEVIATION_PCT:
        print(
            f"[DATA SANITY] ma_10 (${ma_10:.2f}) deviates {ma_deviation:.1%} from "
            f"current price (${current_price:.2f}), exceeds {MAX_MA_DEVIATION_PCT:.0%}. Rejecting."
        )
        return False

    return True