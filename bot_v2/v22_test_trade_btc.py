"""v22_test_trade_btc.py - Test rapide d'un trade BTCUSD avec SPLIT.

Place 2 ordres BTCUSD pour valider la strategie split Vantage :
- Ordre 1 (50%) -> TP 1R
- Ordre 2 (50%) -> TP 2R (runner)

Volume : 0.02 lot total (= 2x0.01)
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

# SL/TP : risque 300$ = RR1 et RR2
sl = ask - 300.0       # 300$ sous l'ask
risk = ask - sl        # = 300
tp_1r = ask + 1.0 * risk   # +300$
tp_2r = ask + 2.0 * risk   # +600$

print(f"BUY split : entry={ask:.2f} sl={sl:.2f} tp1R={tp_1r:.2f} tp2R={tp_2r:.2f}")
print()

# Order 1 : PARTIAL (50%) - TP 1R
print(">> PARTIAL (0.01 lot TP 1R)")
r1 = mt5_exec.place_market_order(
    symbol="BTCUSD", direction="bullish", volume=0.01,
    sl=sl, tp=tp_1r,
    comment="V22-TEST-BTC-P", magic=22260530,
)
if r1: print(f"  OK ticket={r1['ticket']} price={r1['price']}")
else:  print(f"  FAILED")

# Order 2 : RUNNER (50%) - TP 2R
print(">> RUNNER  (0.01 lot TP 2R)")
r2 = mt5_exec.place_market_order(
    symbol="BTCUSD", direction="bullish", volume=0.01,
    sl=sl, tp=tp_2r,
    comment="V22-TEST-BTC-R", magic=22260530,
)
if r2: print(f"  OK ticket={r2['ticket']} price={r2['price']}")
else:  print(f"  FAILED")

print()
# Recupere les positions pour confirmer
positions = mt5.positions_get(symbol="BTCUSD")
if positions:
    print("Positions actuelles BTCUSD magic 22260530 :")
    for p in positions:
        if p.magic == 22260530:
            print(f"  ticket={p.ticket} type={p.type} vol={p.volume} "
                  f"open={p.price_open:.2f} sl={p.sl:.2f} tp={p.tp:.2f} profit={p.profit:+.2f} comment={p.comment}")

mt5_exec.shutdown()
print()
print("Test termine. Quand le partial touche son TP, le bot live va bouger le SL du runner a entry.")
