"""Test rapide : verifie que MT5 trouve XAGUSD et DXY (utilises en SMT)."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from bot_v2.mt5_executor import MT5Executor

mt5 = MT5Executor()
mt5.initialize()

print(f"\nBalance: {mt5.get_balance()}")
print(f"Broker offset: {mt5.broker_utc_offset_sec}s\n")

# Test des symboles SMT
SMT_SYMBOLS = ["XAGUSD", "DXY", "SPX500"]

for sym in SMT_SYMBOLS:
    print(f"--- {sym} ---")
    # 1. Test tick
    try:
        import MetaTrader5 as _mt5
        broker_sym = mt5.to_broker(sym) if hasattr(mt5, 'to_broker') else None
        from bot_v2.mt5_executor import to_broker_symbol
        broker_sym = to_broker_symbol(sym)
        print(f"  Broker symbol: {broker_sym}")
        tick = _mt5.symbol_info_tick(broker_sym)
        if tick is None:
            print(f"  TICK: KO (symbole non disponible)")
        else:
            print(f"  TICK: OK bid={tick.bid} ask={tick.ask} time={tick.time}")
    except Exception as e:
        print(f"  TICK: EXCEPTION {e}")

    # 2. Test fetch M1
    try:
        df = mt5.get_bars(sym, "M1", 100)
        if df is None:
            print(f"  M1 fetch: KO (None)")
        elif len(df) < 50:
            print(f"  M1 fetch: PARTIEL ({len(df)} bougies)")
        else:
            print(f"  M1 fetch: OK ({len(df)} bougies, last={df.index[-1]})")
    except Exception as e:
        print(f"  M1 fetch: EXCEPTION {e}")
    print()
