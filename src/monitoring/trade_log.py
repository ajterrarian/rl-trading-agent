import csv
import os
from datetime import date

LOG_FILE = os.path.join("state", "live_trading_log.csv")

FIELDNAMES = [
    "date", "status", "current_price", "return", "ma_10", "volatility_10",
    "action", "shares_before", "order_delta", "equity", "note",
]


def log_daily_result(status: str, current_price=None, ret=None, ma_10=None,
                      volatility_10=None, action=None, shares_before=None,
                      order_delta=None, equity=None, note: str = "") -> None:
    """status: one of 'kill_switch', 'no_data', 'data_rejected', 'no_trade', 'traded'.
    Fields unavailable at the logging point (e.g. price during a kill-switch
    halt) are left blank rather than guessed."""
    os.makedirs("state", exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "date": date.today().isoformat(), "status": status,
            "current_price": current_price, "return": ret, "ma_10": ma_10,
            "volatility_10": volatility_10, "action": action,
            "shares_before": shares_before, "order_delta": order_delta,
            "equity": equity, "note": note,
        })