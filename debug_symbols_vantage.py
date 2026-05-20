"""Debug : verifie que TOUS les 14 actifs sont accessibles sur Vantage.

Pour chaque actif :
- Symbol broker (avec/sans +)
- Symbol info existe ?
- Tick courant (bid/ask)
- Lot min, lot max, lot step
- Stop level (distance min SL/TP)
- Visible dans Market Watch ?

Usage :
    python debug_symbols_vantage.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

import MetaTrader5 as mt5
from bot_v2.mt5_executor import MT5Executor, to_broker_symbol, BROKER_SYMBOL_MAP


# 14 actifs du portefeuille live V4
LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
    # SMT
    "XAGUSD", "DXY",
]


def main():
    print("=" * 80)
    print("  DEBUG SYMBOLES VANTAGE")
    print("=" * 80)

    m = MT5Executor()
    if not m.initialize():
        print("ERREUR : impossible d'initialiser MT5")
        return

    print(f"Compte : {m.account_info.login} | balance : {m.get_balance():.2f}€")
    print(f"Server : {m.account_info.server}")
    print()

    print(f"{'STD':<10}{'BROKER':<15}{'INFO':<6}{'TICK':<8}{'BID':<12}{'ASK':<12}{'LOT_MIN':<10}{'STOPS':<8}{'VISIBLE'}")
    print("-" * 100)

    ok_count = 0
    fail_count = 0
    for std_name in LIVE_ASSETS:
        broker_sym = to_broker_symbol(std_name)
        info = mt5.symbol_info(broker_sym)
        if info is None:
            # Fallback sans +
            broker_sym = std_name
            info = mt5.symbol_info(broker_sym)

        if info is None:
            print(f"{std_name:<10}{broker_sym:<15}{'NO':<6}{'-':<8}{'-':<12}{'-':<12}{'-':<10}{'-':<8}{'NO'}")
            fail_count += 1
            continue

        # Force visible
        if not info.visible:
            mt5.symbol_select(broker_sym, True)
            info = mt5.symbol_info(broker_sym)

        tick = mt5.symbol_info_tick(broker_sym)
        bid = tick.bid if tick else None
        ask = tick.ask if tick else None

        status_info = "OK"
        status_tick = "OK" if tick and bid > 0 else "NO"
        lot_min = info.volume_min
        stops_level = info.trade_stops_level
        visible = "OUI" if info.visible else "NON"

        bid_str = f"{bid:.5f}" if bid else "-"
        ask_str = f"{ask:.5f}" if ask else "-"

        print(f"{std_name:<10}{broker_sym:<15}{status_info:<6}{status_tick:<8}{bid_str:<12}{ask_str:<12}"
              f"{lot_min:<10}{stops_level:<8}{visible}")
        ok_count += 1

    print()
    print("=" * 80)
    print(f"  RESULTAT : {ok_count}/{len(LIVE_ASSETS)} OK | {fail_count} ECHECS")
    print("=" * 80)

    if fail_count > 0:
        print()
        print("⚠️  ACTIFS EN ECHEC : verifie qu'ils sont dans le Market Watch MT5")
        print("    Click droit Market Watch -> Show all -> active les actifs manquants")

    m.shutdown()


if __name__ == "__main__":
    main()
