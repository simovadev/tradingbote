# DIAGNOSTIC FINAL V12 — 4 jours tick ultra-logge (2026-05-23)

> Backtest 12-15 mai 2026, 14 actifs, 112 setups detectes, 21 fermes
> CSV : `bt_diag_trades.csv` (227 KB, 30+ colonnes)
> Script analyse : `analyse_diag.py`

## RESULTATS GLOBAUX

| Outcome | N | % |
|---|---|---|
| WIN | 7 | 6% |
| LOSS | 14 | 12% |
| NO_FILL | 56 | 50% |
| INVALID_PRICE | 35 | 31% |

**WR fermes : 33.3%  |  PnL : -2.17R  |  Setups non executes : 81%**

## LES 7 PATTERNS DECOUVERTS (chiffres a l'appui)

### Pattern 1 — Le ML V12 NE DISTINGUE PAS WIN vs LOSS

ML proba mediane :
- WIN  = 0.738
- LOSS = 0.742 (quasi identique, legerement plus haut !)
- NO_FILL = 0.768
- INVALID_PRICE = 0.734

**Conclusion** : le ML voit WIN et LOSS de la meme facon. Le seuil 0.70 + setup ICT
ne filtrent pas les LOSS. **Probleme fondamental de calibration.**

→ Fix possible : reentrainer V13 avec features qui distinguent (SL/ATR ratio,
killzone encodee, age OB, etc.)

### Pattern 2 — SL TROP SERRE SUR LES LOSS (40% plus serre que WIN)

| Outcome | SL median en % du prix |
|---|---|
| WIN | 0.0718% |
| LOSS | **0.0439%** (40% plus serre) |

**Conclusion** : Les LOSS ont des OB avec SL serres (typiquement OB intra-bougie).
Le SL est bouffe par le bruit tick avant que le mouvement directionnel ne se confirme.

→ **Fix critique : SL minimum = 0.5 x ATR M5** ou similaire. Probablement
le levier #1 pour passer de 33% a 60%+ WR.

### Pattern 3 — LIMIT MANQUE 81% DES SETUPS

- **NO_FILL (50%)** : le LIMIT n'a jamais ete touche (prix parti vers TP sans nous)
- **INVALID_PRICE (31%)** : le prix a depasse l'entry de **-0.285% en moyenne**
  au moment du placement → ordre rejete par MT5

→ **Fix : passer en MARKET au lieu de LIMIT.** Rejeu prouve : MARKET sur les
memes setups = 67% WR sur 122/124 trades (vs LIMIT 39% sur 23/124).

### Pattern 4 — LOSS MEURENT EN 4 MIN (bruit tick, pas vrai mouvement)

| Outcome | Hold time median |
|---|---|
| WIN | 1034s (17 min) — vrai mouvement directionnel |
| LOSS | **230s (4 min)** — bouffes par le bruit |

**Confirme Pattern 2** : SL serre + bruit tick = LOSS en quelques minutes.

### Pattern 5 — 00h UTC = POUBELLE, 15h UTC = OR

| Heure UTC | n | WR | PnL |
|---|---|---|---|
| **00h** | 4 | **0%** | -4R |
| 01h | 1 | 100% | +1.7R |
| 03h | 1 | 0% | -1R |
| 05h | 1 | 100% | +2.2R |
| 08-09h | 4 | 50% | +1.4R |
| 10h-14h | 4 | 0% | -4R |
| **15h** | **3** | **100%** | **+4.5R** |
| 19-20h | 2 | 0% | -2R |

→ **Filtrer 00h-01h, 10h-14h, 19h-21h** (= heures faible liquidite).
→ **NY morning (15h UTC)** = best zone.

### Pattern 6 — 36% DES LOSS ONT ETE EN PROFIT AVANT (sauvable par BE)

MFE/MAE en R :
- WIN  : MFE median = 1.52R, MAE = 0.70R
- LOSS : MFE median = 0.33R, MAE = 1.01R

**5 LOSS / 14 (36%)** sont alles en MFE >= 0.5R avant de revenir taper le SL.
**2 LOSS / 14 (14%)** sont alles en MFE >= 1.0R.

→ **Fix : stop loss a breakeven a +0.5R**. Aurait sauve ~5 LOSS = +5R sur 4 jours.

### Pattern 7 — KILLZONES : NY morning >> Asia

| Killzone | n | WR | PnL |
|---|---|---|---|
| asia (00-06h) | 7 | 28.6% | -1.1R |
| london (06-12h) | 6 | 33.3% | -0.6R |
| **ny_morning (12-16h)** | 6 | **50%** | **+1.5R** |
| ny_afternoon (16-21h) | 2 | 0% | -2.0R |

→ **Concentrer sur ny_morning (12h-16h UTC)** — best WR + best PnL.

## PROCHAINES ETAPES (par ordre de priorite)

### 1. SL minimum (Pattern 2) — IMPACT MAJEUR
Tester : SL = max(ob_high - ob_low, 0.5 × ATR_M5) pour eviter SL serres bouffes.
Backtester sur les 4 jours, comparer.

### 2. MARKET au lieu de LIMIT (Pattern 3) — DEJA PROUVE
Rejeu MARKET immediat = 67% WR (vs LIMIT 33%). Coder dans live_runner_v2.py.

### 3. Breakeven a +0.5R (Pattern 6) — Sauve 36% des LOSS
Modifier la logique de gestion : si MFE atteint 0.5R, deplacer SL a entry.
Sauve les LOSS qui sont alles en + avant de revenir.

### 4. Filtre heures (Pattern 5) — Elimine LOSS evidentes
Bloquer 00h-01h UTC, 19h-21h UTC. Garder 06-16h (london + ny_morning).

### 5. Reentrainer V13 (Pattern 1) — Long terme
Ajouter features : sl_to_atr_ratio, killzone_encoded, age_ob_min, mfe_recent.

## FICHIERS

- `bt_diag_trades.csv` — 112 setups complets (toutes dimensions)
- `analyse_diag.py` — analyse 11 dimensions, reproductible
- `backtest_v12_diagnostic.py` — code du backtest ultra-logge
- `SESSION_V12_DIAGNOSTIC_FINAL.md` — ce document

## ETAT INFRASTRUCTURE

- **Vast.ai** : ACTIVE ($1.346/h) — penser a destroy si pause longue
- **Bot live** : ARRETE depuis 22/05 soir
- **GitHub** : main = `0ad8abf` (commit script analyse) + ce CSV a venir
- **MT5 Vantage** : balance 84.92 EUR
