import os
from datetime import datetime, timedelta

import torch
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, PositionSide, TimeInForce
from alpaca.common.exceptions import APIError

from src.agent.dqn import QNetwork
from src.data.pipeline import get_data, add_features
from src.risk.kill_switch import check_kill_switch
from src.risk.data_sanity import check_data_sanity
from src.risk.position_sizing import compute_order_delta
from src.monitoring.trade_log import log_daily_result

SYMBOL = "SPY"
CHECKPOINT_PATH = os.path.join("checkpoints", "dqn_trading_v1.pt")
DEVICE = torch.device("cpu")


def get_current_shares(trading_client, symbol: str) -> float:
    """Signed share count: positive = long, negative = short, zero = flat."""
    try:
        position = trading_client.get_open_position(symbol)
        sign = 1 if position.side == PositionSide.LONG else -1
        return float(position.qty) * sign
    except APIError:
        return 0.0


def load_policy_net(n_observations: int, n_actions: int) -> QNetwork:
    policy_net = QNetwork(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    policy_net.eval()
    return policy_net


def get_dqn_action(policy_net: QNetwork, state_vector: list) -> int:
    """One greedy inference step. Read-only -- no gradients, no training."""
    state = torch.tensor(state_vector, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).item()


def main():
    load_dotenv() 
    client = TradingClient(
        os.environ["APCA_API_KEY_ID"],
        os.environ["APCA_API_SECRET_KEY"],
        paper=True,
    )

    # 1. Kill switch
    if check_kill_switch(client):
        log_daily_result(status="kill_switch", note="halted by kill switch")
        return

    # 2. Fetch enough trailing history for 10-day rolling features.
    lookback_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    df = get_data(ticker=SYMBOL, start=lookback_start)
    df = add_features(df) 

    if df.empty:
        print("[EXECUTION] No feature rows available after fetch. Skipping today.")
        log_daily_result(status="no_data", note="no feature rows after fetch")
        return

    latest = df.iloc[-1]
    current_price = float(latest["Close"])
    ret = float(latest["return"])
    ma_10 = float(latest["ma_10"])
    volatility_10 = float(latest["volatility_10"])

    # 3. Data sanity check
    if not check_data_sanity(current_price, ret, ma_10, volatility_10):
        log_daily_result(status="data_rejected", current_price=current_price, ret=ret,
                          ma_10=ma_10, volatility_10=volatility_10, note="failed data sanity check")
        return

    # 4. Current position and account state
    current_shares = get_current_shares(client, SYMBOL)
    equity = float(client.get_account().equity)
    position_flag = 0.0 if current_shares == 0 else (1.0 if current_shares > 0 else -1.0)
    state_vector = [ret, ma_10, volatility_10, position_flag]

    # 5. Frozen policy inference
    policy_net = load_policy_net(n_observations=len(state_vector), n_actions=3)
    action = get_dqn_action(policy_net, state_vector)
    print(f"[EXECUTION] Model action: {action} (0=hold, 1=buy, 2=short)")

    # 6. Position sizing
    order_delta = compute_order_delta(action, equity, current_price, current_shares)
    if order_delta == 0:
        print("[EXECUTION] No trade needed -- already positioned correctly.")
        log_daily_result(status="no_trade", current_price=current_price, ret=ret, ma_10=ma_10,
                          volatility_10=volatility_10, action=action, shares_before=current_shares,
                          order_delta=0, equity=equity)
        return

    side = OrderSide.BUY if order_delta > 0 else OrderSide.SELL
    order = MarketOrderRequest(symbol=SYMBOL, qty=abs(order_delta), side=side, time_in_force=TimeInForce.DAY)
    submitted = client.submit_order(order_data=order)
    print(f"[EXECUTION] Submitted {side.value} order for {abs(order_delta)} shares. Order ID: {submitted.id}")

    log_daily_result(status="traded", current_price=current_price, ret=ret, ma_10=ma_10,
                      volatility_10=volatility_10, action=action, shares_before=current_shares,
                      order_delta=order_delta, equity=equity, note=f"order_id={submitted.id}")

if __name__ == "__main__":
    main()