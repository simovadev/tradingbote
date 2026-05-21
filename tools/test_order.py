"""Test isole : place un ordre LIMIT minimal LOIN du marche.

But : verifier si place_limit_order() du bot fonctionne, sans dependre
du pipeline. On place un BUY LIMIT UK100 tres en-dessous du prix actuel
-> il ne sera jamais touche -> on peut l'annuler apres.

Si ca place OK -> le bug est dans le pipeline (entre SETUP et l'appel).
Si ca echoue -> le bug est dans place_limit_order / MT5.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import MetaTrader5 as mt5
from bot_v2.mt5_executor import MT5Executor

mt5x = MT5Executor()
if not mt5x.initialize():
    print("MT5 init KO")
    sys.exit(1)

print(f"Compte : {mt5x.account_info.login} balance {mt5x.get_balance()}")

# Prix actuel UK100
tick = mt5.symbol_info_tick("UK100")
print(f"UK100 bid={tick.bid} ask={tick.ask}")

# BUY LIMIT tres en dessous (jamais touche) : -500 points
entry = round(tick.bid - 500, 2)
sl = round(entry - 50, 2)
tp = round(entry + 100, 2)
print(f"\nTest : BUY LIMIT UK100 vol=0.1 (min) entry={entry} sl={sl} tp={tp}")
print("(entry tres en dessous du marche -> ne sera jamais fill)")

result = mt5x.place_limit_order(
    symbol="UK100",
    direction="bullish",
    volume=0.1,
    entry_price=entry,
    sl=sl,
    tp=tp,
    expiration_minutes=30,
    comment="TEST_ORDER",
    magic=99999,
)

print()
if result is None:
    print(">>> ECHEC : place_limit_order a retourne None")
    print(">>> Le bug est dans place_limit_order / MT5")
else:
    print(f">>> SUCCES : ordre place, ticket={result['ticket']}")
    print(">>> place_limit_order fonctionne -> le bug est dans le pipeline du bot")
    print()
    # On annule l'ordre de test tout de suite
    cancel_req = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": result["ticket"],
    }
    cr = mt5.order_send(cancel_req)
    if cr and cr.retcode == mt5.TRADE_RETCODE_DONE:
        print(f">>> Ordre de test {result['ticket']} ANNULE proprement")
    else:
        print(f">>> ATTENTION : annulation KO, annule manuellement le ticket {result['ticket']} dans MT5")

mt5.shutdown()
