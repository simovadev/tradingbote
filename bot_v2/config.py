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
    # === FOREX MAJORS ===
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
}

DEFAULT_INSTRUMENT = "XAUUSD"


# ============ PAIRES SMT (bible §7) ============
# (primaire -> [(correle, type_correlation)])
# - "positive" : bougent dans le meme sens
# - "inverse"  : sens oppose
SMT_PAIRS: dict[str, list[tuple[str, str]]] = {
    # === Commodities / Indices ===
    "XAUUSD": [("XAGUSD", "positive"), ("DXY", "inverse")],
    "NAS100": [("SPX500", "positive"), ("GER40", "positive")],
    "GER40":  [("SPX500", "positive"), ("NAS100", "positive")],
    "USOUSD": [("UKOIL", "positive")],
    # === Forex (correlations ICT classiques) ===
    "EURUSD": [("GBPUSD", "positive"), ("DXY", "inverse")],
    "GBPUSD": [("EURUSD", "positive"), ("DXY", "inverse")],
    "USDJPY": [("DXY", "positive")],         # JPY -> sens du dollar
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
RR_MIN = 2.0                # decision user 2026-05-15 : RR mini 1:2 (PF baseline 1.34)
RR_TARGET = 2.0             # cible identique baseline
RR_MAX = 3.0                # decision user : RR max 1:3 (revert 2026-05-16 : 1:5 testé donne -66% WR)
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
        "swing_strength_m1": 3,
        "min_distance_tp_pct": 0.001,
        "min_score": 140,                     # NAS plus rare, on prend les meilleurs
        "min_quality": 55,
        "rr_max": 2.0,                         # decision user 2026-05-16 : NAS100 cape a RR=2
    },
    "GER40": {
        "parent_ob_tolerance_pct": 0.005,     # 0.5% = ~110 pts sur DAX a 22000
        "discount_premium_tol_pct": 0.08,
        "min_displacement_atr": 1.0,
        "swing_strength_m1": 3,
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
        "swing_strength_m1": 3,               # 24/7 -> beaucoup de bruit micro
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
    """Recupere un parametre specifique a l'actif, avec fallback default puis _DEFAULT_PARAMS."""
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
