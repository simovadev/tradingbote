"""Configuration globale du bot multi-actif."""
from pathlib import Path

ROOT = Path(__file__).parent

# Donnees
DATA_DIR = ROOT / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# DB
DB_PATH = ROOT / "db" / "tradingbot.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============ INSTRUMENTS ============
# Chaque actif a un ticker court (utilise en interne / DB) et un mapping
# vers le code Dukascopy.
# Le "tick_value" est le PnL$ pour 1 unite de mouvement avec 1 lot.
# Le "min_sl_points" est le SL minimum en points (couvre spread + bruit moyen).

INSTRUMENTS = {
    "XAUUSD": {
        "duka": "INSTRUMENT_FX_METALS_XAU_USD",
        "tick_value": 100.0,    # 1 lot XAU = 100 oz, $1 move = $100
        "min_sl_points": 3.0,
        "label": "Or",
    },
    "NAS100": {
        "duka": "INSTRUMENT_IDX_AMERICA_E_NQ_100",
        "tick_value": 1.0,
        "min_sl_points": 25.0,   # NAS bouge facile de 20-30pts, faut respirer
        "label": "Nasdaq 100",
    },
    "GER40": {
        "duka": "INSTRUMENT_IDX_EUROPE_E_DAAX",
        "tick_value": 1.0,
        "min_sl_points": 20.0,   # DAX bouge facile de 15-25pts
        "label": "DAX 40",
    },
    "USOIL": {
        "duka": "INSTRUMENT_CMD_ENERGY_E_LIGHT",
        "tick_value": 1000.0,
        "min_sl_points": 0.30,    # WTI bouge 0.20-0.50 facilement
        "label": "WTI Crude",
    },
    "DXY": {
        "duka": "INSTRUMENT_IDX_AMERICA_DOLLAR_IDX_USD",
        "tick_value": 1000.0, "min_sl_points": 0.10,
        "label": "Dollar Index", "smt_only": True,
    },
    "XAGUSD": {
        "duka": "INSTRUMENT_FX_METALS_XAG_USD",
        "tick_value": 5000.0, "min_sl_points": 0.05,
        "label": "Argent", "smt_only": True,
    },
    "SPX500": {
        "duka": "INSTRUMENT_IDX_AMERICA_E_SANDP_500",
        "tick_value": 1.0, "min_sl_points": 5.0,
        "label": "S&P 500", "smt_only": True,
    },
    "UKOIL": {
        "duka": "INSTRUMENT_CMD_ENERGY_E_BRENT",
        "tick_value": 1000.0, "min_sl_points": 0.30,
        "label": "Brent Crude", "smt_only": True,
    },
}

DEFAULT_INSTRUMENT = "XAUUSD"

# Paires SMT (actif_primaire -> [(actif_correle, type_correlation)])
# - "positive" : bougent dans le meme sens -> SMT = un fait extremum, pas l'autre
# - "inverse"  : bougent en sens oppose -> SMT = un fait extremum, autre fait l'oppose
#
# Vraies paires correlees ICT classiques :
SMT_PAIRS: dict[str, list[tuple[str, str]]] = {
    "XAUUSD": [
        ("XAGUSD", "positive"),    # Or / Argent : corrélation positive forte
        ("DXY", "inverse"),         # Or / Dollar : corrélation inverse
    ],
    "NAS100": [
        ("SPX500", "positive"),    # NAS / SP500 : corrélation positive très forte
        ("GER40", "positive"),     # 2 indices = positive
    ],
    "GER40":  [
        ("SPX500", "positive"),
        ("NAS100", "positive"),
    ],
    "USOIL":  [
        ("UKOIL", "positive"),     # WTI / Brent : corrélation positive très forte
    ],
}

# Timeframes utilises (du plus grand au plus petit)
# H4 = bias macro, H1 = HTF principal, M30/M15 = intermediaires, M5 = micro, M1 = signal
TIMEFRAMES = ["H4", "H1", "M30", "M15", "M5", "M1"]

# Timeframes pour l'analyse top-down
HTF_TIMEFRAMES = ["H4", "H1"]        # bias macro
MTF_TIMEFRAMES = ["M30", "M15"]      # confirmation intermediaire / setup zone
LTF_TIMEFRAMES = ["M5", "M1"]        # signal d'execution / micro

# Portefeuille fictif
INITIAL_BALANCE = 10_000.0  # USD
RISK_PER_TRADE = 0.01  # 1% du capital par trade

# Serveur
HOST = "127.0.0.1"
PORT = 8000
