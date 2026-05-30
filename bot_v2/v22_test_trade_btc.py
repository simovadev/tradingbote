"""v22_test_trade_btc.py - Test rapide d'un trade BTCUSD.

Place 1 ordre BUY 0.01 lot BTCUSD avec SL/TP a +-200$ du prix actuel.
But : valider que le pipeline MT5Executor marche bout en bout.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import MetaTrader5 as mt5
from bot_v2.mt5_executor import MT5Executor

print("=" * 60)
print("V22 TEST TRADE - BTCUSD 0.01 lot BUY")
print("=" * 60)

# Init MT5
mt5_exec = MT5Executor()
if not mt5_exec.initialize():
    print("MT5 init FAILED")
    sys.exit(1)

info = mt5.account_info()
print(f"Compte : login={info.login} server={info.server} balance={info.balance:.2f}")

# Get BTC current price
tick = mt5.symbol_info_tick("BTCUSD")
if tick is None:
    print("BTCUSD : no tick"); sys.exit(1)

ask = tick.ask; bid = tick.bid
print(f"BTCUSD : ask={ask:.2f} bid={bid:.2f}")

# SL/TP : 200$ chacun = ~0.27% du prix BTC ~73000 (au-dessus du min 0.1% qu'on a hardcode)
sl = ask - 300.0       # 300$ sous l'ask
tp = ask + 600.0       # 600$ au-dessus = RR 2

print(f"BUY entry={ask:.2f} sl={sl:.2f} tp={tp:.2f} (SL=-300$ TP=+600$ RR=2)")

# Place ordre
result = mt5_exec.place_market_order(
    symbol="BTCUSD",
    direction="bullish",
    volume=0.01,
    sl=sl,
    tp=tp,
    comment="V22-TEST-BTC",
    magic=22260530,
)

if result is None:
    print("ORDER FAILED - voir log MT5")
    sys.exit(1)

print()
print(f"ORDER OK :")
print(f"  ticket : {result['ticket']}")
print(f"  price  : {result['price']}")
print(f"  volume : {result['volume']}")
print()

# Recupere la position pour confirmer
positions = mt5.positions_get(symbol="BTCUSD")
if positions:
    for p in positions:
        if p.magic == 22260530:
            print(f"Position confirmee : ticket={p.ticket} type={p.type} vol={p.volume} "
                  f"open={p.price_open:.2f} sl={p.sl:.2f} tp={p.tp:.2f} profit={p.profit:+.2f}")

mt5_exec.shutdown()
print()
print("Test termine. Tu peux fermer le trade manuellement depuis MT5 si tu veux.")
