# ibtest_quote_fixed.py
from ib_insync import *

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=20, timeout=5)

print("Connected:", ib.isConnected())
print("Server time:", ib.reqCurrentTime())

# 🔑 THIS IS THE KEY LINE
ib.reqMarketDataType(3)  # delayed data

qqq = Stock("QQQ", "SMART", "USD")
ib.qualifyContracts(qqq)

ticker = ib.reqMktData(qqq)
ib.sleep(2)

print(
    "QQQ:",
    "bid=", ticker.bid,
    "ask=", ticker.ask,
    "last=", ticker.last,
    "close=", ticker.close
)

ib.cancelMktData(qqq)
ib.disconnect()

