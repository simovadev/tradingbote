"""Test combien de bougies M1 MT5 peut fournir au max."""
import sys
sys.path.insert(0, ".")

from bot_v2.mt5_executor import MT5Executor

mt5_exec = MT5Executor()
mt5_exec.initialize()

print(f"MT5 connecte | balance={mt5_exec.get_balance()}")
print(f"Broker offset: {mt5_exec.broker_utc_offset_sec}s")
print()

# Test plusieurs tailles
for n in [2000, 10000, 30000, 50000, 80000, 100000, 130000, 200000]:
    df = mt5_exec.get_bars("XAUUSD", "M1", n, force_sync=True)
    if df is None:
        print(f"  N={n:>6} : KO (None)")
        break
    else:
        print(f"  N={n:>6} : OK ({len(df):,} bougies) | first={df.index[0]} | last={df.index[-1]}")
