# scripts/verify_alpaca_connection.py
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

api_key = os.environ.get("APCA_API_KEY_ID")
secret_key = os.environ.get("APCA_API_SECRET_KEY")

if not api_key or not secret_key:
    raise RuntimeError(
        "Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY. "
        "Check that .env exists at the repo root and contains both, filled in."
    )

client = TradingClient(api_key, secret_key, paper=True)
account = client.get_account()

print(f"Connected. Account status: {account.status}")
print(f"Buying power: ${account.buying_power}")
print(f"Cash: ${account.cash}")