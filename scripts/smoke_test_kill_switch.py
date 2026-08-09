# scripts/smoke_test_kill_switch.py
import os
import json
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from src.risk.kill_switch import check_kill_switch, STATE_DIR, PEAK_EQUITY_FILE, MANUAL_KILL_FILE

load_dotenv()
client = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=True)

# clean slate
for f in [PEAK_EQUITY_FILE, MANUAL_KILL_FILE]:
    if os.path.exists(f):
        os.remove(f)

# 1. Fresh state, no drawdown yet -- should NOT halt
halted = check_kill_switch(client)
print("Test 1 (fresh, no drawdown):", "HALTED" if halted else "OK")
assert halted == False

# 2. Manual trigger
os.makedirs(STATE_DIR, exist_ok=True)
open(MANUAL_KILL_FILE, "w").close()
halted = check_kill_switch(client)
print("Test 2 (manual flag present):", "HALTED" if halted else "OK")
assert halted == True
os.remove(MANUAL_KILL_FILE)

# 3. Simulated drawdown -- inflate the recorded peak to force a 33% "loss"
with open(PEAK_EQUITY_FILE, "r") as f:
    peak = json.load(f)["peak_equity"]
with open(PEAK_EQUITY_FILE, "w") as f:
    json.dump({"peak_equity": peak * 1.5}, f)

halted = check_kill_switch(client)
print("Test 3 (simulated 33% drawdown):", "HALTED" if halted else "OK")
assert halted == True

print("All kill switch smoke tests passed.")