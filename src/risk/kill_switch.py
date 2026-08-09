"""
Manual and automatic kill triggers that soft-stop trading:
1. Manual -- operator controlled sentinel file
2. Automatic -- account equity has drawn more than max drawdown pct from its recoreded peak
"""

import json
import os

STATE_DIR = "state"
PEAK_EQUITY_FILE = os.path.join(STATE_DIR, "peak_equity.json")
MANUAL_KILL_FILE = os.path.join(STATE_DIR, "KILL_SWITCH")


def _load_peak_equity(current_equity: float) -> float:
    """Read the last recorded peak equity from disk. If no record exists
    yet (first run ever), initialize it to today's equity."""
    if not os.path.exists(PEAK_EQUITY_FILE):
        return current_equity
    with open(PEAK_EQUITY_FILE, "r") as f:
        data = json.load(f)
    return data["peak_equity"]


def _save_peak_equity(peak_equity: float) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(PEAK_EQUITY_FILE, "w") as f:
        json.dump({"peak_equity": peak_equity}, f)


def check_kill_switch(trading_client, max_drawdown_pct: float = 0.10) -> bool:
    """
    Returns True if trading should halt today (soft stop, no new orders),
    False if it's safe to proceed with today's decision.
    """
    # 1. Manual trigger -- cheapest check, no network call, always wins.
    if os.path.exists(MANUAL_KILL_FILE):
        print(f"[KILL SWITCH] Manual halt file found at {MANUAL_KILL_FILE}. Skipping trading today.")
        return True

    # 2. Automatic trigger -- drawdown from peak equity.
    account = trading_client.get_account()
    current_equity = float(account.equity)

    peak_equity = _load_peak_equity(current_equity)
    peak_equity = max(peak_equity, current_equity)
    _save_peak_equity(peak_equity)

    drawdown_pct = (peak_equity - current_equity) / peak_equity

    if drawdown_pct > max_drawdown_pct:
        print(
            f"[KILL SWITCH] Drawdown {drawdown_pct:.2%} exceeds max "
            f"{max_drawdown_pct:.2%} (peak equity ${peak_equity:,.2f}, "
            f"current ${current_equity:,.2f}). Skipping trading today."
        )
        return True

    return False