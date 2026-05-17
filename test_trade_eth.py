"""Test trade reel sur ETH avec lot minimum.

But : valider que Python peut placer un ordre MT5 avec SL/TP cote broker.
On utilise ETH car son lot minimum est probablement le plus accessible avec 1 EUR + levier 1:500.

SL/TP : risk minimal (~0.30 EUR de risk avec lot min)
"""
import sys
sys.path.insert(0, 'C:/Users/Administrator/tradingbote')

import MetaTrader5 as mt5
import time


def main():
    print("=" * 60)
    print("TEST TRADE ETH (BUY lot minimum)")
    print("=" * 60)

    if not mt5.initialize():
        print(f"MT5 init FAILED : {mt5.last_error()}")
        return

    # Liste les symboles ETH disponibles
    print("\n[1] Recherche symboles ETH...")
    syms = mt5.symbols_get("*ETH*")
    if not syms:
        print("  Aucun symbole ETH trouve. Essaie *eth*...")
        syms = mt5.symbols_get("*eth*")
    if not syms:
        print("  ERREUR : pas de symbole ETH chez ce broker")
        mt5.shutdown()
        return

    print(f"  Symboles ETH trouves :")
    for s in syms:
        print(f"    - {s.name} | vol_min={s.volume_min} step={s.volume_step} digits={s.digits}")

    # Choisis le premier qui contient ETH et USD
    eth_sym = None
    for s in syms:
        if "USD" in s.name.upper():
            eth_sym = s.name
            break
    if eth_sym is None:
        eth_sym = syms[0].name

    print(f"\n[2] Symbole retenu : {eth_sym}")
    info = mt5.symbol_info(eth_sym)

    # Active le symbole
    if not info.visible:
        mt5.symbol_select(eth_sym, True)
        time.sleep(1)
        info = mt5.symbol_info(eth_sym)

    # Recupere le tick
    tick = mt5.symbol_info_tick(eth_sym)
    if tick is None:
        print(f"  ERREUR : pas de tick pour {eth_sym}")
        mt5.shutdown()
        return

    print(f"  Prix ASK : {tick.ask}")
    print(f"  Prix BID : {tick.bid}")
    print(f"  Spread   : {(tick.ask - tick.bid):.2f}")
    print(f"  Volume min: {info.volume_min}")
    print(f"  Contract size: {info.trade_contract_size}")
    print(f"  Margin requis pour lot min: {info.margin_initial} (approx)")

    # Calcule SL/TP : risk tres serre car balance = 1 EUR
    # On veut SL = -0.30 EUR max
    # Pour ETH, 1 lot = trade_contract_size unites (varie selon broker, souvent 1 ETH = 1 unit)
    entry = tick.ask
    spread = tick.ask - tick.bid
    # SL a 0.5% (rapide a toucher mais risque limite vu lot min)
    sl = round(entry * 0.995, info.digits)
    tp = round(entry * 1.010, info.digits)  # TP 1% (RR=2)

    risk_pts = entry - sl
    print(f"\n[3] Setup :")
    print(f"  Entry (market) : {entry}")
    print(f"  SL             : {sl}  (-{risk_pts:.2f})")
    print(f"  TP             : {tp}  (+{tp - entry:.2f})")
    print(f"  Volume         : {info.volume_min}")

    # Confirmation
    confirm = input("\n>>> Place l'ordre BUY ? (oui/non) : ").strip().lower()
    if confirm not in ("oui", "o", "yes", "y"):
        print("Annulation.")
        mt5.shutdown()
        return

    # Place l'ordre
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": eth_sym,
        "volume": info.volume_min,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 20260517,
        "comment": "TestETH",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print(f"\n[4] Envoi de l'ordre...")
    result = mt5.order_send(request)

    if result is None:
        print(f"  ERREUR : order_send None - {mt5.last_error()}")
    elif result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  ORDRE REJETE : retcode={result.retcode} - {result.comment}")
        # Decode courants
        codes = {
            10004: "REQUOTE (prix change, refait)",
            10006: "REJECTED",
            10013: "INVALID REQUEST",
            10014: "INVALID VOLUME",
            10015: "INVALID PRICE",
            10016: "INVALID STOPS (SL/TP invalides ou trop proches)",
            10018: "MARKET CLOSED",
            10019: "NO MONEY (pas assez de margin)",
            10020: "PRICE CHANGED",
            10021: "OFF QUOTES",
            10025: "INVALID FILL TYPE",
            10027: "AUTOTRADING DISABLED",
        }
        print(f"  -> {codes.get(result.retcode, 'inconnu')}")
    else:
        print(f"  OK ! TICKET={result.order} prix={result.price} volume={result.volume}")
        print(f"  Va voir dans MT5 desktop, la position est ouverte.")

    mt5.shutdown()


if __name__ == "__main__":
    main()
