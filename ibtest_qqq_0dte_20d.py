# ibtest_qqq_nearest_20d.py
from ib_insync import *
from datetime import datetime
from math import isnan

TARGET_DELTA = 0.20
N_STRIKES = 60
SLEEP_SEC = 3

def best_by_delta(tickers, target=0.20):
    best = None
    best_dist = 9e9
    for t in tickers:
        g = t.modelGreeks or t.bidGreeks or t.askGreeks
        if not g or g.delta is None:
            continue
        dist = abs(abs(g.delta) - target)
        if dist < best_dist:
            best = (t, g, dist)
            best_dist = dist
    return best

def onError(reqId, errorCode, errorString, contract):
    # Print subscription/permission errors clearly
    if errorCode in (10089, 10090, 10091, 10167, 10168, 10197):
        print(f"IB Error {errorCode} (reqId {reqId}): {errorString}")

ib = IB()
ib.errorEvent += onError
ib.connect("127.0.0.1", 7496, clientId=31, timeout=8)

ib.reqMarketDataType(3)  # delayed

qqq = Stock("QQQ", "SMART", "USD")
ib.qualifyContracts(qqq)

chains = ib.reqSecDefOptParams(qqq.symbol, "", qqq.secType, qqq.conId)
chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

today = datetime.now().strftime("%Y%m%d")
exp = sorted([e for e in chain.expirations if e >= today])[0]

ut = ib.reqMktData(qqq)
ib.sleep(1.5)
under_px = ut.last if ut.last and not isnan(ut.last) else ut.close
ib.cancelMktData(qqq)

print("Underlying: QQQ px=", under_px, "expiry=", exp)

strikes = sorted(chain.strikes)
near = sorted(strikes, key=lambda s: abs(s - under_px))[:N_STRIKES]

calls = [Option("QQQ", exp, s, "C", "SMART") for s in near]
puts  = [Option("QQQ", exp, s, "P", "SMART") for s in near]
ib.qualifyContracts(*(calls + puts))

call_ticks = ib.reqTickers(*calls)
put_ticks  = ib.reqTickers(*puts)
ib.sleep(SLEEP_SEC)

best_call = best_by_delta(call_ticks, TARGET_DELTA)
best_put  = best_by_delta(put_ticks, TARGET_DELTA)

def show(label, best):
    if not best:
        print(f"{label}: no greeks (likely needs OPRA/options data).")
        return
    t, g, dist = best
    print(
        f"{label}: strike={t.contract.strike} right={t.contract.right} "
        f"delta={g.delta:.3f} dist={dist:.3f} "
        f"bid={t.bid} ask={t.ask} last={t.last} iv={getattr(g,'impliedVol', None)}"
    )

show("BEST CALL ~20d", best_call)
show("BEST PUT  ~20d", best_put)

ib.disconnect()
