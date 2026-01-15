import yfinance as yf
import pandas as pd

# Download historical weekly data for QQQ
ticker = yf.Ticker("QQQ")
df = ticker.history(period="max", interval="1wk")

# Drop any rows without valid close prices
df = df.dropna(subset=["Close"])

# Calculate weekly percent change
df["Weekly_Pct_Change"] = df["Close"].pct_change() * 100

# Filter weeks with >10% loss
big_losses = df[df["Weekly_Pct_Change"] < -5]

# Print results
if big_losses.empty:
    print("No weeks found with a greater than 5% loss.")
else:
    print("Weeks with greater than 5% loss in QQQ:")
    for date, row in big_losses.iterrows():
        pct = row["Weekly_Pct_Change"]
        print(f"Week ending {date.date()}: {pct:.2f}%")
