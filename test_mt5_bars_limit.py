"""Test combien de bougies M1 MT5 peut fournir au max."""
import sys
import time
sys.path.insert(0, ".")

import MetaTrader5 as mt5
from bot_v2.mt5_executor import MT5Executor, to_broker_symbol

mt5_exec = MT5Executor()
mt5_exec.initialize()

print(f"MT5 connecte | balance={mt5_exec.get_balance()}")
print(f"Broker offset: {mt5_exec.broker_utc_offset_sec}s")
print()

broker_sym = to_broker_symbol("XAUUSD")
print(f"Test XAUUSD ({broker_sym})")
print()

# Test plusieurs tailles directement avec copy_rates_from_pos
for n in [2000, 10000, 30000, 50000, 80000, 100000, 130000, 200000]:
    t0 = time.time()
    rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_M1, 0, n)
    elapsed = (time.time() - t0) * 1000
    if rates is None:
        err = mt5.last_error()
        print(f"  N={n:>6} : KO (None) | err={err} | {elapsed:.0f}ms")
    else:
        print(f"  N={n:>6} : OK ({len(rates):,} bougies) | {elapsed:.0f}ms")

print()
print("=== Test get_bars (avec force_sync) ===")
for n in [2000, 30000, 80000, 130000]:
    t0 = time.time()
    df = mt5_exec.get_bars("XAUUSD", "M1", n, force_sync=True)
    elapsed = (time.time() - t0) * 1000
    if df is None:
        err = mt5.last_error()
        print(f"  get_bars N={n:>6} : KO (None) | err={err} | {elapsed:.0f}ms")
    else:
        print(f"  get_bars N={n:>6} : OK ({len(df):,} bougies) | {elapsed:.0f}ms")
