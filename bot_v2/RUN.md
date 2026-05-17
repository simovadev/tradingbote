# Bot_v2 — Guide d'utilisation

Bot autonome conforme à la méthodologie Vizion FR (VIZION_BIBLE.md).

## Chaîne TF Vizion stricte

```
D1 (bias daily)
 └── H1 (contexte)
      └── M15 (parent direct)
           └── M1 (entrée)
```

Imbrication double : un OB M1 doit être contenu dans un OB M15 ET un OB H1.

## Actifs supportés

**Primary (tradables) :**
- Métaux : XAUUSD (Or)
- Indices : NAS100, GER40
- Énergie : USOIL
- Forex : EURUSD, GBPUSD, USDJPY, AUDUSD

**SMT only (pour divergences) :** DXY, XAGUSD, SPX500, UKOIL

## Commandes

### Fetch des données
```bash
# Tous les actifs primaires, 30 jours, tous les TF
./venv/Scripts/python.exe -m bot_v2.fetch_data --primary-only --days 30

# Un actif spécifique
./venv/Scripts/python.exe -m bot_v2.fetch_data --instrument EURUSD --days 90

# Un TF spécifique (utile pour D1 long)
./venv/Scripts/python.exe -m bot_v2.fetch_data --tf D1 --days 365
```

### Vérifier l'état du cache
```bash
./venv/Scripts/python.exe -m bot_v2.data_loader
```

### Tester un concept
```bash
./venv/Scripts/python.exe -m bot_v2.concepts.order_block
./venv/Scripts/python.exe -m bot_v2.concepts.fvg
./venv/Scripts/python.exe -m bot_v2.concepts.killzones
# etc.
```

### Backtest
```bash
# Default : XAUUSD M1/M15/H1, 30 jours, min_quality=60, min_score=100
./venv/Scripts/python.exe -m bot_v2.backtest

# Autre instrument
./venv/Scripts/python.exe -m bot_v2.backtest EURUSD 30

# Autre fenêtre
./venv/Scripts/python.exe -m bot_v2.backtest XAUUSD 90
```

## Paramètres clés du pipeline

Dans `bot_v2/config.py` :

| Param | Defaut | Source |
|---|---|---|
| `OB_MAX_GROUP_SIZE` | 5 | bible : pas plus de 5 bougies sinon = range |
| `SL_MODE` | "wick" | décision user 2026-05-15 |
| `RR_MIN` | 1.5 | bible §13 |
| `RR_TARGET` | 2.0 | bible §0 |
| `RISK_PER_TRADE_PCT` | 0.01 | décision user 2026-05-15 |
| `USE_ECONOMIC_CALENDAR` | False | décision user 2026-05-15 |
| `TRADE_FRIDAY` | True | décision user 2026-05-15 |
| `SMT_IS_FILTER` | False (= bonus) | décision user 2026-05-15 |

## Architecture

```
bot_v2/
├── config.py                    Paramètres globaux + tes décisions
├── data_loader.py               Charge le cache parquet
├── fetch_data.py                Fetch dukascopy
├── pipeline.py                  ★ CASCADE Vizion (le cerveau)
├── trade_setup.py               Entry/SL/TP/lots
├── backtest.py                  Simulation historique
└── concepts/                    1 fichier = 1 concept
    ├── killzones.py             KZ NY + DST
    ├── liquidity.py             Swings/Sweeps/Equal HL
    ├── order_block.py           OB STRICT Vizion
    ├── fvg.py                   FVG/IFVG/Volume Imbalance
    ├── discount_premium.py      Fibo 50% + zones
    ├── structure.py             BOS/MSS/Modèle 2022
    ├── breaker.py               OB inversé + displacement
    ├── daily_bias.py            PDH/PDL + close strength
    ├── weekly_profile.py        3 profils Vizion
    ├── po3_amd.py               OLHC/OHLC + AMD
    ├── smt.py                   Divergence positive/inverse
    ├── htf_swings.py            Swings HTF pour TP fiables
    ├── setup_quality.py         Force OB, Unicorn, retests
    └── session_bias.py          London/NY play
```

## Logique de la cascade (pipeline.py)

1. **Daily Bias** — bias align (+25), neutral (+0), contraire (-15)
2. **Killzone** — élimine si hors KZ (+15)
3. **Session play** — bonus continuation/reversal selon mvt session (+5 à +15)
4. **Imbrication TF** — OB LTF dans OB HTF (+20)
5. **Imbrication TF² (chaîne complète)** — OB HTF dans OB grand-parent (+15)
6. **Discount/Premium** — élimine si mauvaise zone (+15)
7. **PO3 HTF** — distribution alignée (+10)
8. **PO3 grand-parent** — sens aligné (+8)
9. **SMT divergence** (bonus) — +10
10. **Breaker présent** — +10
11. **Setup quality** (compute_setup_quality) — élimine si < min_quality. Pondéré 50% du score (+0 à +50)
12. **RR/Setup** — élimine si RR < 1.5
13. **Score final** — élimine si < min_score

Score max théorique : ~200.

## Comment ajuster pour avoir 2-3 trades/jour

Si trop de trades :
- ↑ `min_quality` (defaut 60, essayer 70-80)
- ↑ `min_score` (defaut 100, essayer 120-150)

Si pas assez de trades :
- ↓ `min_quality` (essayer 40-50)
- ↓ `min_score` (essayer 80)
