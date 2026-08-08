import yfinance as yf
import pandas as pd

def get_data(ticker = "SPY", start = "2001-01-01", end = None):
    df = yf.download(ticker, start = start, end = end)
    return df

def add_features(df):
    #calculate return, 10-day moving average, 10-day volatility
    df['return'] = df['Close'].pct_change()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['volatility_10'] = df['return'].rolling(10).std()
    df = df.dropna()
    print(df.tail())
    return df


def chronological_split(df, train_frac = 0.7, val_frac = 0.15):
    #chronological data split in order
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int (n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


if __name__ == "__main__":
    df = get_data(ticker = "SPY", start = "2001-01-01")
    print("Raw shape:", df.shape)

    df = add_features(df)
    print("After features/dropna shape:", df.shape)

    train, val, test = chronological_split(df)
    print("Train:", train.index.min(), "->", train.index.max(), f"({len(train)} rows)")
    print("Val:  ", val.index.min(), "->", val.index.max(), f"({len(val)} rows)")
    print("Test: ", test.index.min(), "->", test.index.max(), f"({len(test)} rows)")