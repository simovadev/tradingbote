"""Test rapide de connexion MT5.

Verifie que :
1. MT5 desktop est ouvert + connecte au compte
2. Le bot Python peut se connecter
3. Les 8 symboles sont accessibles
4. On peut lire des bougies
5. On voit la balance

A faire AVANT de lancer le bot live.
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

from bot_v2.mt5_executor import MT5Executor


SYMBOLS_TO_TEST = ["XAUUSD", "NAS100", "GER40", "BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]


def main():
    print("=" * 60)
    print("TEST CONNEXION MT5 VANTAGE")
    print("=" * 60)
    print()

    mt5_exec = MT5Executor()
    print("[1/5] Connexion MT5...")
    if not mt5_exec.initialize():
        print("  ECHEC : verifie que MT5 desktop est ouvert + connecte au compte")
        return
    print("  OK")
    print()

    print("[2/5] Info compte :")
    info = mt5_exec.account_info
    print(f"  Login    : {info.login}")
    print(f"  Server   : {info.server}")
    print(f"  Balance  : {info.balance:.2f} {info.currency}")
    print(f"  Equity   : {info.equity:.2f} {info.currency}")
    print(f"  Leverage : 1:{info.leverage}")
    print(f"  Trade allowed : {info.trade_allowed}")
    print(f"  Trade expert  : {info.trade_expert}")
    print()

    if not info.trade_allowed:
        print("  /!\\ ATTENTION : trade NON autorise sur ce compte")
    if not info.trade_expert:
        print("  /!\\ ATTENTION : AutoTrading DESACTIVE - active dans MT5 (bouton vert)")
    print()

    print("[3/5] Verification des symboles :")
    available = []
    missing = []
    for sym in SYMBOLS_TO_TEST:
        sym_info = mt5_exec.symbol_info(sym)
        if sym_info is None:
            missing.append(sym)
            print(f"  {sym:8s} : NON TROUVE")
            continue
        ok = mt5_exec.ensure_symbol_active(sym)
        status = "OK (active)" if ok else "OK (mais inactif)"
        print(f"  {sym:8s} : {status} | "
              f"vol_min={sym_info.volume_min} step={sym_info.volume_step} "
              f"digits={sym_info.digits} contract_size={sym_info.trade_contract_size}")
        available.append(sym)
    print()
    if missing:
        print(f"  /!\\ Symboles manquants : {missing}")
        print("  -> Verifie dans MT5 Market Watch que ces symboles existent.")
        print("  -> Sur Vantage, ils peuvent avoir un suffixe (.r, .raw, etc.)")
    print()

    print("[4/5] Test lecture bougies :")
    for sym in available[:3]:  # juste 3 pour le test
        df = mt5_exec.get_bars(sym, "M1", 10)
        if df is None or len(df) == 0:
            print(f"  {sym:8s} : ECHEC fetch bougies")
        else:
            print(f"  {sym:8s} : {len(df)} bougies M1 | last={df.index[-1]} close={df['close'].iloc[-1]:.5f}")
    print()

    print("[5/5] Positions actuellement ouvertes :")
    positions = mt5_exec.get_positions()
    if not positions:
        print("  Aucune position ouverte")
    else:
        for p in positions:
            print(f"  {p['symbol']} {p['type']} vol={p['volume']} "
                  f"entry={p['price_open']:.5f} SL={p['sl']:.5f} TP={p['tp']:.5f} "
                  f"profit={p['profit']:+.2f}")
    print()

    mt5_exec.shutdown()
    print("=" * 60)
    if not missing and info.trade_allowed and info.trade_expert:
        print("TOUT EST OK - le bot peut etre lance !")
    else:
        print("ATTENTION : voir messages /!\\ ci-dessus avant de lancer le bot")
    print("=" * 60)


if __name__ == "__main__":
    main()
