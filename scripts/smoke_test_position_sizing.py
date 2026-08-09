from src.risk.position_sizing import compute_order_delta

delta = compute_order_delta(action=1, equity=100_000, current_price=500.0, current_shares=0)
print(f"Test 1 (flat -> buy): delta={delta}")
assert delta == 100  # floor(100000*0.5/500)

delta = compute_order_delta(action=1, equity=105_000, current_price=525.0, current_shares=100)
print(f"Test 2 (already long, buy again -- no rebalance): delta={delta}")
assert delta == 0

delta = compute_order_delta(action=0, equity=105_000, current_price=525.0, current_shares=100)
print(f"Test 3 (long -> flat): delta={delta}")
assert delta == -100

delta = compute_order_delta(action=2, equity=105_000, current_price=525.0, current_shares=100)
print(f"Test 4 (long -> short): delta={delta}")
assert delta == -200  # sell 100 to close long, sell 100 more to open short

delta = compute_order_delta(action=0, equity=100_000, current_price=500.0, current_shares=0)
print(f"Test 5 (flat -> hold, no-op): delta={delta}")
assert delta == 0

print("All position sizing smoke tests passed.")