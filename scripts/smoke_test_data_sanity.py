import numpy as np
from src.risk.data_sanity import check_data_sanity

assert check_data_sanity(current_price=550.0, ret=0.01, ma_10=548.0, volatility_10=0.012) == True
print("Test 1 (normal data): OK")

assert check_data_sanity(current_price=550.0, ret=None, ma_10=548.0, volatility_10=0.012) == False
print("Test 2 (missing value): OK")

assert check_data_sanity(current_price=550.0, ret=np.nan, ma_10=548.0, volatility_10=0.012) == False
print("Test 3 (NaN value): OK")

assert check_data_sanity(current_price=550.0, ret=0.45, ma_10=548.0, volatility_10=0.012) == False
print("Test 4 (implausible return): OK")

assert check_data_sanity(current_price=-5.0, ret=0.01, ma_10=548.0, volatility_10=0.012) == False
print("Test 5 (non-positive price): OK")

assert check_data_sanity(current_price=550.0, ret=0.01, ma_10=548.0, volatility_10=-0.01) == False
print("Test 6 (negative volatility): OK")

assert check_data_sanity(current_price=550.0, ret=0.01, ma_10=300.0, volatility_10=0.012) == False
print("Test 7 (ma_10 too far from price): OK")

print("All data sanity smoke tests passed.")