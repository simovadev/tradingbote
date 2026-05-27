"""Configuration bot_v2 — strictement aligne avec VIZION_BIBLE.md.

Toute valeur ici doit etre tracee a une regle de la bible. Pas de magic number
sans source.
"""
from __future__ import annotations

from pathlib import Path

# ============ CHEMINS ============
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "cache"
DB_PATH = ROOT / "db" / "tradingbot.sqlite"
BIBLE_PATH = ROOT / "VIZION_BIBLE.md"


# ============ INSTRUMENTS (bible §0 — actifs tradés) ============
# Categorisation :
#   - "primary"  : on cherche des setups dessus.
#   - "smt_only" : utilise uniquement pour la divergence SMT (bible §7).
#
# Tick_value : PnL$ pour 1 unite de mouvement avec 1 lot.
# - XAU = 1 lot = 100 oz, donc $1 move = $100
# - Indices US = 1 lot ~ 1 contrat, $1 move = $1
# - Forex paires en USD : 1 lot standard = 100k unites, 1 pip = $10
#   tick_value = 100000 pour la decimale (1.0900 -> 1.0901 = 0.0001 * 100000 = $10)
#   Mais on travaille en POINTS = 1 pip = 0.0001 sur EUR/USD = 0.01 sur USD/JPY
INSTRUMENTS: dict[str, dict] = {
    # === COMMODITIES & INDICES ===
    "XAUUSD": {
        "duka": "INSTRUMENT_FX_METALS_XAU_USD",
        "tick_value": 100.0,
        "min_sl_points": 3.0,
        "label": "Or",
        "role": "primary",
        "type": "metal",
    },
    "NAS100": {
        "duka": "INSTRUMENT_IDX_AMERICA_E_NQ_100",
        "tick_value": 1.0,
        "min_sl_points": 25.0,
        "label": "Nasdaq 100",
        "role": "primary",
        "type": "index",
    },
    "GER40": {
        "duka": "INSTRUMENT_IDX_EUROPE_E_DAAX",
        "tick_value": 1.0,
        "min_sl_points": 20.0,
        "label": "DAX 40",
        "role": "primary",
        "type": "index",
    },
    "USOUSD": {
        "duka": "INSTRUMENT_CMD_ENERGY_E_LIGHT",
        "tick_value": 1000.0,
        "min_sl_points": 0.60,                       # user 2026-05-17 : 0.30 -> 0.60 (15x spread WTI typique 0.04)
        "label": "WTI Crude (Vantage USOUSD)",
        "role": "primary",
        "type": "energy",
    },
    "BTCUSD": {
        "duka": "INSTRUMENT_VCCY_BTC_USD",
        "tick_value": 1.0,
        "min_sl_points": 100.0,
        "label": "Bitcoin",
        "role": "primary",
        "type": "crypto",
    },
    # === NOUVEAUX INDICES (user 2026-05-18) ===
    "SP500": {
        "duka": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
        "tick_value": 1.0,
        "min_sl_points": 5.0,
        "label": "S&P 500",
        "role": "primary",
        "type": "index",
    },
    "DJ30": {
        "duka": "INSTRUMENT_IDX_AMERICA_E_D_J_IND",
        "tick_value": 1.0,
        "min_sl_points": 20.0,
        "label": "Dow Jones 30",
        "role": "primary",
        "type": "index",
    },
    "UK100": {
        "duka": "INSTRUMENT_IDX_EUROPE_E_FUTSEE_100",
        "tick_value": 1.0,
        "min_sl_points": 15.0,
        "label": "FTSE 100",
        "role": "primary",
        "type": "index",
    },
    "FRA40": {
        "duka": "INSTRUMENT_IDX_EUROPE_E_CAAC_40",
        "tick_value": 1.0,
        "min_sl_points": 15.0,
        "label": "CAC 40",
        "role": "primary",
        "type": "index",
    },
    "JP225": {
        "duka": "INSTRUMENT_IDX_ASIA_E_N225JAP",
        "tick_value": 1.0,
        "min_sl_points": 50.0,
        "label": "Nikkei 225",
        "role": "primary",
        "type": "index",
    },
    # === NOUVEAUX FOREX MAJORS + CROSSES (user 2026-05-18) ===
    "USDCAD": {
        "duka": "INSTRUMENT_FX_MAJORS_USD_CAD",
        "tick_value": 100000.0,
        "min_sl_points": 0.0010,
        "label": "USD/CAD",
        "role": "primary",
        "type": "forex",
    },
    "USDCHF": {
        "duka": "INSTRUMENT_FX_MAJORS_USD_CHF",
        "tick_value": 100000.0,
        "min_sl_points": 0.0010,
        "label": "USD/CHF",
        "role": "primary",
        "type": "forex",
    },
    # === FOREX MAJORS (smt_only desormais reactives) ===
    # === FOREX MAJORS ===
    # Decision user 2026-05-15 : DESACTIVES (role=smt_only) car WR catastrophique
    # en backtest (17-24% sur AUDUSD/USDJPY/EURUSD). Restent disponibles pour SMT.
    "EURUSD": {
        "duka": "INSTRUMENT_FX_MAJORS_EUR_USD",
        "tick_value": 100000.0,
        "min_sl_points": 0.0010,
        "label": "EUR/USD",
        "role": "primary",                  # user 2026-05-17 : reactive pour Phase 6 forex
        "type": "forex",
    },
    "GBPUSD": {
        "duka": "INSTRUMENT_FX_MAJORS_GBP_USD",
        "tick_value": 100000.0,
        "min_sl_points": 0.0010,
        "label": "GBP/USD",
        "role": "primary",                  # user 2026-05-17 : reactive Phase 6
        "type": "forex",
    },
    "USDJPY": {
        "duka": "INSTRUMENT_FX_MAJORS_USD_JPY",
        "tick_value": 1000.0,
        "min_sl_points": 0.10,
        "label": "USD/JPY",
        "role": "primary",                  # user 2026-05-17 : reactive Phase 6
        "type": "forex",
    },
    "AUDUSD": {
        "duka": "INSTRUMENT_FX_MAJORS_AUD_USD",
        "tick_value": 100000.0,
        "min_sl_points": 0.0010,
        "label": "AUD/USD",
        "role": "primary",                  # user 2026-05-17 : reactive Phase 6
        "type": "forex",
    },
    # === SMT ONLY ===
    "DXY": {
        "duka": "INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD",
        "tick_value": 1000.0,
        "min_sl_points": 0.10,
        "label": "Dollar Index",
        "role": "smt_only",
        "type": "index",
    },
    "XAGUSD": {
        "duka": "INSTRUMENT_FX_METALS_XAG_USD",
        "tick_value": 5000.0,
        "min_sl_points": 0.05,
        "label": "Argent",
        "role": "smt_only",
        "type": "metal",
    },
    "SPX500": {
        "duka": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
        "tick_value": 1.0,
        "min_sl_points": 5.0,
        "label": "S&P 500",
        "role": "smt_only",
        "type": "index",
    },
    "UKOIL": {
        "duka": "INSTRUMENT_CMD_ENERGY_E_BRENT",
        "tick_value": 1000.0,
        "min_sl_points": 0.30,
        "label": "Brent Crude",
        "role": "smt_only",
        "type": "energy",
    },
    # === V18 NOUVEAUX ACTIFS (Soufiane 2026-05-26) — pool 28 actifs decoreles ===
    # Indices Asia
    "Nikkei225": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 50.0,
        "label": "Nikkei 225 (Vantage)", "role": "primary", "type": "index",
    },
    "HK50": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 30.0,
        "label": "Hang Seng 50", "role": "primary", "type": "index",
    },
    "BVSPX": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 100.0,
        "label": "Bovespa (Bresil)", "role": "primary", "type": "index",
    },
    # FX EM / secondaires
    "USDMXN": {
        "duka": None, "tick_value": 5000.0, "min_sl_points": 0.0100,
        "label": "USD/MXN (Peso)", "role": "primary", "type": "forex",
    },
    "USDZAR": {
        "duka": None, "tick_value": 5500.0, "min_sl_points": 0.0100,
        "label": "USD/ZAR (Rand)", "role": "primary", "type": "forex",
    },
    "NZDUSD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.0010,
        "label": "NZD/USD", "role": "primary", "type": "forex",
    },
    # Energie
    "CL-OIL": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.30,
        "label": "WTI Crude (CL-OIL Vantage)", "role": "primary", "type": "energy",
    },
    "GAS-C": {
        "duka": None, "tick_value": 10000.0, "min_sl_points": 0.05,
        "label": "Natural Gas", "role": "primary", "type": "energy",
    },
    # Softs
    "Cocoa-C": {
        "duka": None, "tick_value": 10.0, "min_sl_points": 10.0,
        "label": "Cocoa", "role": "primary", "type": "soft",
    },
    "Coffee-C": {
        "duka": None, "tick_value": 375.0, "min_sl_points": 1.0,
        "label": "Coffee", "role": "primary", "type": "soft",
    },
    "Sugar-C": {
        "duka": None, "tick_value": 1120.0, "min_sl_points": 0.10,
        "label": "Sugar", "role": "primary", "type": "soft",
    },
    "Wheat-C": {
        "duka": None, "tick_value": 50.0, "min_sl_points": 5.0,
        "label": "Wheat", "role": "primary", "type": "soft",
    },
    # Crypto
    "ETHUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 5.0,
        "label": "Ethereum", "role": "primary", "type": "crypto",
    },
    # === V20 : 50 nouveaux actifs Vantage (added 2026-05-27) ===
    "EURJPY": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.1,
        "label": "EURJPY (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "EURGBP": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURGBP (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "EURCHF": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURCHF (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "EURAUD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURAUD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "EURCAD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURCAD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "EURNZD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURNZD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "GBPJPY": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.1,
        "label": "GBPJPY (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "GBPCHF": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "GBPCHF (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "GBPAUD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "GBPAUD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "GBPCAD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "GBPCAD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "GBPNZD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "GBPNZD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "AUDJPY": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.1,
        "label": "AUDJPY (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "AUDCHF": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "AUDCHF (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "AUDCAD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "AUDCAD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "AUDNZD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "AUDNZD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "NZDJPY": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.1,
        "label": "NZDJPY (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "NZDCHF": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "NZDCHF (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "NZDCAD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "NZDCAD (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "CADJPY": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.1,
        "label": "CADJPY (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "CADCHF": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "CADCHF (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "CHFJPY": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.1,
        "label": "CHFJPY (forex_crosses)", "role": "secondary", "type": "forex",
    },
    "USDTRY": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDTRY (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDSGD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDSGD (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDHKD": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDHKD (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDNOK": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDNOK (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDSEK": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDSEK (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDDKK": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDDKK (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDPLN": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDPLN (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "USDCNH": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "USDCNH (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "EURPLN": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURPLN (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "EURNOK": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURNOK (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "EURSEK": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURSEK (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "EURHUF": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURHUF (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "EURCZK": {
        "duka": None, "tick_value": 100000.0, "min_sl_points": 0.001,
        "label": "EURCZK (forex_exotic)", "role": "secondary", "type": "forex",
    },
    "XAUEUR": {
        "duka": None, "tick_value": 100.0, "min_sl_points": 0.40504,
        "label": "XAUEUR (metals)", "role": "secondary", "type": "metal",
    },
    "XAUAUD": {
        "duka": None, "tick_value": 100.0, "min_sl_points": 0.67645,
        "label": "XAUAUD (metals)", "role": "secondary", "type": "metal",
    },
    "XAUJPY": {
        "duka": None, "tick_value": 100.0, "min_sl_points": 75.14725,
        "label": "XAUJPY (metals)", "role": "secondary", "type": "metal",
    },
    "XPDUSD": {
        "duka": None, "tick_value": 5000.0, "min_sl_points": 0.1581,
        "label": "XPDUSD (metals)", "role": "secondary", "type": "metal",
    },
    "XPTUSD": {
        "duka": None, "tick_value": 5000.0, "min_sl_points": 0.20818,
        "label": "XPTUSD (metals)", "role": "secondary", "type": "metal",
    },
    "CHINA50": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 7.47425,
        "label": "CHINA50 (indices)", "role": "secondary", "type": "index",
    },
    "LTCUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.05458,
        "label": "LTCUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "XRPUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.00139,
        "label": "XRPUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "ADAUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.00025,
        "label": "ADAUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "BCHUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.44869,
        "label": "BCHUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "DOTUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.00126,
        "label": "DOTUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "LNKUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.00912,
        "label": "LNKUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "SOLUSD": {
        "duka": None, "tick_value": 1.0, "min_sl_points": 0.08518,
        "label": "SOLUSD (crypto)", "role": "secondary", "type": "crypto",
    },
    "UKOUSD": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.05,
        "label": "UKOUSD (energy)", "role": "secondary", "type": "energy",
    },
    "Cotton-C": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.01,
        "label": "Cotton-C (softs)", "role": "secondary", "type": "soft",
    },
    "Soybean-C": {
        "duka": None, "tick_value": 1000.0, "min_sl_points": 0.0115,
        "label": "Soybean-C (softs)", "role": "secondary", "type": "soft",
    },
}

DEFAULT_INSTRUMENT = "XAUUSD"


# ============ PAIRES SMT (bible §7) ============
# (primaire -> [(correle, type_correlation)])
# - "positive" : bougent dans le meme sens
# - "inverse"  : sens oppose
SMT_PAIRS: dict[str, list[tuple[str, str]]] = {
    # === Commodities / Indices ===
    # DXY = USDX sur Vantage (mappe dans mt5_executor.BROKER_SYMBOL_MAP)
    "XAUUSD": [("XAGUSD", "positive"), ("DXY", "inverse")],
    "NAS100": [("SPX500", "positive"), ("GER40", "positive")],
    "GER40":  [("SPX500", "positive"), ("NAS100", "positive")],
    "USOUSD": [],   # UKOIL pas dispo Vantage + USOUSD abandonne
    # === Forex (correlations ICT classiques) ===
    "EURUSD": [("GBPUSD", "positive"), ("DXY", "inverse")],
    "GBPUSD": [("EURUSD", "positive"), ("DXY", "inverse")],
    "USDJPY": [("DXY", "positive")],
    "AUDUSD": [("EURUSD", "positive"), ("DXY", "inverse")],
}


# ============ TIMEFRAMES (bible §2.5 — entonnoir TF Vizion strict) ============
# Chaine VIZION OFFICIELLE (rdeAjnVdfRM, 6IN5CNs5qrc, BKGoNf6vhRY) :
#   D1 -> H1 -> M15 -> M1
# Pas de M5 dans la chaine officielle Vizion.
# On garde M5/H4 en cache pour analyse hors-chaine si besoin.
ALL_TF = ["D1", "H4", "H1", "M15", "M5", "M1"]

# Chaine de trading Vizion : du plus haut au plus bas
VIZION_TF_CHAIN = ["D1", "H1", "M15", "M1"]

HTF_TF = ["D1", "H1"]      # contexte / bias
LTF_TF = ["M15", "M1"]     # entree

# Imbrication TF stricte (entonnoir) : chaque TF doit etre contenu dans son PARENT direct.
# Mapping enfant -> parent.
TF_PARENT: dict[str, str] = {
    "M1":  "M15",   # M1 doit etre dans un OB M15
    "M15": "H1",    # M15 doit etre dans un OB H1
    "H1":  "D1",    # H1 doit etre dans un OB D1
    # Hors chaine officielle (gardes au cas ou) :
    "M5":  "H1",
    "H4":  "D1",
}


# ============ KILLZONES NY (bible §8) ============
# Toutes les killzones en HEURE NEW YORK (gestion DST automatique dans killzones.py).
# Format : (nom, heure_debut, heure_fin, [actifs_concernes ou None pour tous])
KILLZONES_NY = [
    ("Asia",       20, 0),    # 20h NY -> 00h NY (overnight)
    ("London",      2, 5),    # 02-05h NY
    ("NY_AM",       9, 11),   # 9h-11h NY (decision user 2026-05-15 : 9h30 indices fixe)
    ("NY_Lunch",   12, 13),   # 12-13h NY
    ("NY_PM",      14, 16),   # 14-16h NY
]

# Heure d'ouverture cash market US (indices) — NY AM
INDICES_OPEN_NY_HOUR = 9
INDICES_OPEN_NY_MIN = 30


# ============ ORDER BLOCK (bible §2 + decision user) ============
# Decision user 2026-05-15 : OB STRICT.
# - OB bullish = N bougies BAISSIERES consecutives (N>=1), pas d'intruse haussiere.
# - Validation = cloture corps au-dessus du high du groupe (long) ou sous le low (short).
# - Doit prendre une liquidite externe (sweep d'un swing).
OB_MIN_BODY_PCT = 0.0       # pas de min strict, Vizion accepte les dojis BAISSIERS dans un OB bullish
OB_MAX_GROUP_SIZE = 5       # garde-fou : pas plus de 5 bougies dans un OB sinon c'est une range


# ============ STOP LOSS / TAKE PROFIT (bible §13 + decision user) ============
# Decision user 2026-05-15 : SL = MECHE TOUJOURS (OB, breaker, FVG).
SL_MODE = "wick"            # "wick" | "body" (figé sur wick)
# RR surchargeable via env var RR_OVERRIDE (ex: build V10 test RR=1.5).
# Par defaut RR=2.0 (config live inchangee). Si RR_OVERRIDE defini, RR_MIN
# et RR_TARGET prennent cette valeur (RR_MAX = max(3.0, override)).
import os as _os
_rr_override = _os.getenv("RR_OVERRIDE")
if _rr_override:
    RR_MIN = float(_rr_override)
    RR_TARGET = float(_rr_override)
    RR_MAX = max(3.0, float(_rr_override))
else:
    RR_MIN = 2.0            # decision user 2026-05-15 : RR mini 1:2 (PF baseline 1.34)
    RR_TARGET = 2.0         # cible identique baseline
    RR_MAX = 3.0            # decision user : RR max 1:3 (revert 2026-05-16 : 1:5 testé -66% WR)
RISK_PER_TRADE_PCT = 0.01   # 1% du capital, configurable, decision user 2026-05-15


# ============ FILTRES (decisions user 2026-05-15) ============
# Calendrier economique : IGNORE (pas de blackout news).
USE_ECONOMIC_CALENDAR = False
# Vendredi : on trade.
TRADE_FRIDAY = True
# SMT : BONUS de score, pas filtre eliminatoire.
SMT_IS_FILTER = False
SMT_BONUS_POINTS = 10


# ============ PO3 (bible §11 + decision user) ============
# Decision user 2026-05-15 : PO3 sur M5 utilisable comme signal (pas seulement confluence).
PO3_TIMEFRAMES_SIGNAL = ["D1", "H4", "H1", "M5"]


# ============ PORTEFEUILLE ============
INITIAL_BALANCE_USD = 10_000.0


# ============ SERVEUR ============
HOST = "127.0.0.1"
PORT = 8000


# ============ PARAMETRES PAR ACTIF (decision user 2026-05-15) ============
# Chaque actif a sa propre volatilite et son propre niveau de prix => les seuils
# en POURCENTAGE doivent etre adaptes. Un seuil unique pour tous = absurde.
#
# Cles :
# - parent_ob_tolerance_pct : tolerance prix pour qu'un OB HTF "contexte" soit accepte
#   comme parent (sans chevauchement strict). Plus large pour indices nerveux.
# - discount_premium_tol_pct : tolerance autour de l'equilibrium pour la zone "neutre".
# - min_displacement_atr : ratio corps bougie / ATR pour valider un displacement.
#   Plus eleve pour indices (NAS, GER) qui ont souvent des grosses bougies bruyantes.
# - swing_strength_m1 : nombre de bougies de chaque cote pour qualifier un swing M1.
#   Plus eleve pour indices qui ont beaucoup de bruit micro.
# - min_distance_tp_pct : distance minimale (en % du prix) pour qu'un swing soit pris
#   comme TP (evite les TP trop proches).
INSTRUMENT_PARAMS: dict[str, dict] = {
    "XAUUSD": {
        "parent_ob_tolerance_pct": 0.002,     # 0.2% = ~5$ sur or a 2500
        "discount_premium_tol_pct": 0.05,
        "min_displacement_atr": 0.8,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.001,
        # Filtres qualite par actif (cible 1-2 trades premium/jour)
        "min_score": 150,
        "min_quality": 58,
    },
    "NAS100": {
        "parent_ob_tolerance_pct": 0.005,     # 0.5% = ~115 pts sur NAS a 23000
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 2,               # V18.1 : 3->2 (capture +30% OB ICT)
        "min_distance_tp_pct": 0.001,
        "min_score": 140,                     # NAS plus rare, on prend les meilleurs
        "min_quality": 55,
        "rr_max": 2.0,                         # decision user 2026-05-16 : NAS100 cape a RR=2
    },
    "GER40": {
        "parent_ob_tolerance_pct": 0.005,     # 0.5% = ~110 pts sur DAX a 22000
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 2,               # V18.1 : 3->2
        "min_distance_tp_pct": 0.001,
        "min_score": 160,                     # GER tres actif, on serre fort
        "min_quality": 62,
        "rr_max": 2.0,                        # decision user 2026-05-17 : GER40 cape a RR=2
    },
    "USOUSD": {
        # USOUSD abandonne 2026-05-17 (cf BILAN_USOUSD_ABANDON.md).
        # Params conserves pour reference historique uniquement.
        "parent_ob_tolerance_pct": 0.005,
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0008,
        "min_score": 145,
        "min_quality": 52,
        "rr_max": 2.0,
    },
    "BTCUSD": {
        # Phase 5 (user 2026-05-17) : crypto 24/7, forte volatilite.
        # Params initiaux inspires de NAS/GER (indices volatils).
        "parent_ob_tolerance_pct": 0.005,     # 0.5% = ~285\$ sur BTC a 57000
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,          # BTC explose, on serre comme indices
        "swing_strength_m1": 2,               # V18.1 : 3->2 (+OB ICT)
        "min_distance_tp_pct": 0.001,
        "min_score": 145,                     # seuil moyen pour demarrer
        "min_quality": 55,
        "rr_max": 2.0,                        # user 2026-05-17 : revert a RR=2 (RR=3 = -64% volume en live)
    },
    # === FOREX MAJORS ===
    # Specificites forex (decision user 2026-05-15) :
    # - Tolerance OB parent ASSOUPLIE car le forex bouge en pips, les zones sont serrees
    # - Tolerance discount/premium ELARGIE car les ranges Fibo forex sont microscopiques
    #   sur M15 (parfois 20 pips), donc 5% = 1 pip = quasi rien
    # - Daily bias contraire moins penalisant (le forex a des biais macros forts qui
    #   contredisent souvent les setups intraday valides)
    # - is_forex flag pour appliquer des regles supplementaires (allow_against_daily_bias_if_kz)
    "EURUSD": {
        "parent_ob_tolerance_pct": 0.0025,
        "discount_premium_tol_pct": 0.20,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0003,
        "is_forex": True,
        "allow_against_daily_bias": True,
        "min_score": 138,                     # forex moins volatile, scores plus bas
        "min_quality": 52,
    },
    "GBPUSD": {
        "parent_ob_tolerance_pct": 0.0030,
        "discount_premium_tol_pct": 0.20,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0003,
        "is_forex": True,
        "allow_against_daily_bias": True,
        "min_score": 155,                     # GBPUSD tres actif, on serre
        "min_quality": 65,
    },
    "USDJPY": {
        "parent_ob_tolerance_pct": 0.0030,
        "discount_premium_tol_pct": 0.20,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0003,
        "is_forex": True,
        "allow_against_daily_bias": True,
        "min_score": 135,
        "min_quality": 50,
    },
    "AUDUSD": {
        "parent_ob_tolerance_pct": 0.0025,
        "discount_premium_tol_pct": 0.20,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0003,
        "is_forex": True,
        "allow_against_daily_bias": True,
        "min_score": 140,                     # AUDUSD WR catastrophique en mode large, durcir
        "min_quality": 55,
    },
    # === V15.1 FIX A12 (2026-05-25) : 6 actifs manquants ===
    # Avant : SP500/DJ30/UK100/FRA40/USDCAD/USDCHF tombaient sur _DEFAULT_PARAMS
    # (swing_strength=2, generic). Inadapte pour indices (besoin swing=3 + min_disp 1.0)
    # et pour forex (besoin is_forex=True + allow_against_daily_bias).
    "SP500": {
        "parent_ob_tolerance_pct": 0.004,     # 0.4% = ~30 pts sur SP a 7500
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 3,
        "min_distance_tp_pct": 0.001,
        "min_score": 140,
        "min_quality": 55,
        "rr_max": 2.0,
    },
    "DJ30": {
        "parent_ob_tolerance_pct": 0.004,     # ~205 pts sur DJ a 51000
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 3,
        "min_distance_tp_pct": 0.001,
        "min_score": 140,
        "min_quality": 55,
        "rr_max": 2.0,
    },
    "UK100": {
        "parent_ob_tolerance_pct": 0.004,     # ~40 pts sur FTSE a 10000
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 3,
        "min_distance_tp_pct": 0.001,
        "min_score": 140,
        "min_quality": 55,
        "rr_max": 2.0,
    },
    "FRA40": {
        "parent_ob_tolerance_pct": 0.004,     # ~33 pts sur CAC a 8200
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 3,
        "min_distance_tp_pct": 0.001,
        "min_score": 140,
        "min_quality": 55,
        "rr_max": 2.0,
    },
    "USDCAD": {
        "parent_ob_tolerance_pct": 0.0030,
        "discount_premium_tol_pct": 0.20,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0003,
        "is_forex": True,
        "allow_against_daily_bias": True,
        "min_score": 140,
        "min_quality": 55,
    },
    "USDCHF": {
        "parent_ob_tolerance_pct": 0.0030,
        "discount_premium_tol_pct": 0.20,
        "min_displacement_atr": 0.7,
        "swing_strength_m1": 2,
        "min_distance_tp_pct": 0.0003,
        "is_forex": True,
        "allow_against_daily_bias": True,
        "min_score": 140,
        "min_quality": 55,
    },
}

# Valeurs par defaut (si actif non liste)
_DEFAULT_PARAMS = {
    "parent_ob_tolerance_pct": 0.003,
    "discount_premium_tol_pct": 0.05,
    "min_displacement_atr": 0.8,
    "swing_strength_m1": 2,
    "min_distance_tp_pct": 0.0005,
}


def get_param(instrument: str, key: str, default=None):
    """Recupere un parametre specifique a l'actif, avec fallback default puis _DEFAULT_PARAMS.

    V10 : swing_strength_m1 surchargeable via env var SWS_OVERRIDE (test detection
    plus permissive). Par defaut, comportement live inchange.
    """
    if key == "swing_strength_m1":
        _sws = _os.getenv("SWS_OVERRIDE")
        if _sws:
            return int(_sws)
    if instrument in INSTRUMENT_PARAMS and key in INSTRUMENT_PARAMS[instrument]:
        return INSTRUMENT_PARAMS[instrument][key]
    if key in _DEFAULT_PARAMS:
        return _DEFAULT_PARAMS[key]
    return default


# ============ HELPERS ============
def primary_instruments() -> list[str]:
    return [k for k, v in INSTRUMENTS.items() if v["role"] == "primary"]


def smt_only_instruments() -> list[str]:
    return [k for k, v in INSTRUMENTS.items() if v["role"] == "smt_only"]
