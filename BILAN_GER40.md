# 📊 BILAN — GER40 (Phase 3)

**Date** : 2026-05-17
**Asset** : GER40 (DAX 40)
**Période OOS testée** : 2025-05-15 → 2026-05-15 (12 mois)
**Statut** : ✅ **VALIDÉ — actif secondaire**

---

## 🎯 Configuration finale

### Pipeline détection
- **Stratégie** : ICT/SMC pur (même méthodologie que XAUUSD / NAS100)
- **Setup principal** : OB confirmés par MSS (intersection stricte)
- **Timeframe actif** : **M1 uniquement** (M5 non entraîné)
- **Chaîne TF** : M1 → HTF M15 → HTF2 H1

### Spécificités GER40 vs autres actifs
| Paramètre | XAUUSD | NAS100 | **GER40** |
|---|---|---|---|
| `parent_ob_tolerance_pct` | 0.002 | 0.005 | **0.005** |
| `discount_premium_tol_pct` | 0.05 | 0.08 | **0.08** |
| `min_displacement_atr` | 0.8 | 1.0 | **1.0** |
| `swing_strength_m1` | 2 | 3 | **3** (bruité comme NAS) |
| `min_score` | 150 | 140 | **160** (serré) |
| `min_quality` | 58 | 55 | **62** (serré) |
| **RR plafond** | 3.0 | 2.0 | **3.0** (default) |
| **Seuil ML** | 0.50 | 0.55 | **0.60** |
| `min_sl_points` | 3.0 | 25.0 | **20.0** |

### Filtres Vizion (identiques)
- ✅ Daily bias contraire → pénalité -15 pts
- ✅ Session play (continuation ou reversal)
- ✅ Phase Acc/Manip → rejet
- ✅ Displacement ≥ 1.0 × ATR
- ✅ Parent OB M15 obligatoire
- ✅ Discount/Premium élargi 0.08

### Money Management
- **Balance initiale** : 100€
- **Risk par trade** : 20% du capital courant (compound)
- **Cooldown** : 15 min séparé par TF
- **Recommandation live** : 10% (au lieu de 20%) — variance plus élevée

### SL / TP
- **SL** : extrême du groupe OB + 5 bougies AVANT (logique stop hunt)
- **TP** : dynamique priorité HTF (D1 > H4 > H1) → LTF → fallback RR=2
- **RR plafonné à 3** (default, pas de surcharge GER40)
- **Pas de trailing SL**

### Modèle ML
- **Fichier** : `bot_v2/ml_model_GER40.pkl`
- **Features** : `bot_v2/ml_features_GER40.json`
- **Dataset** : **4 ans** (2022-05 → 2026-05), 1122 candidats, 855 trades fermés
- **Type** : LightGBM (300 estimateurs, learning_rate=0.05, max_depth=6)

---

## 📈 RÉSULTATS

### Métriques ML (split temporel 60/20/20)

| Set | Trades | Baseline WR | Seuil 0.50 | Seuil 0.55 | **Seuil 0.60** | Seuil 0.65 | AUC |
|---|---|---|---|---|---|---|---|
| TRAIN | 513 | 37.8% | 92.8% (180t) | 94.1% (153t) | 97.0% (135t) | 99.1% (112t) | 0.972 |
| VAL | 171 | 38.6% | 60.3% (58t) | 61.2% (49t) | **63.9% (36t)** ✅ | 59.3% (27t) | 0.781 |
| **TEST** (OOS) | 171 | 34.5% | 49.0% (51t) | 50.0% (42t) | **51.4% (37t)** | 50.0% (32t) | **0.729** |

→ **Seuil retenu : 0.60** (compromis volume / WR)

### Évolution 2 ans → 4 ans

| Métrique | 2 ans | **4 ans** | Gain |
|---|---|---|---|
| Trades train | 246 | 513 | +108% |
| AUC val | 0.673 | **0.781** | **+0.108** énorme |
| VAL WR @0.60 | 36.4% ❌ | **63.9%** ✅ | +27 pts |
| TEST WR @0.50 | 56.5% | 49.0% | -7 pts |
| TEST WR @0.55 | 43.8% | 50.0% | +6 pts |
| TEST WR @0.60 | 55.6% | 51.4% | -4 pts |

→ Le passage à 4 ans **a stabilisé** le modèle (VAL est devenu monotone, gap train/val réduit).

### Backtest exhaustif 12 mois OOS (seuil 0.60, MM compound 20%)

| Métrique | Valeur |
|---|---|
| **Balance** | 100€ → **3754€** |
| **Performance** | **+3654%** |
| **Trades totaux** | 54 (~4.5/mois) |
| **Wins / Losses** | 22W / 20L |
| **WR global** | **52.4%** |
| **Mois positifs** | 9/12 (75%) |
| **Mois négatifs** | 2/12 |
| **Drawdown max** | **-36%** |

### Détail mois par mois

| # | Période | Trades | WR | Balance | PnL |
|---|---|---|---|---|---|
| 1 | 2025-05-15 → 06-14 | 1 | 0% (no_fill) | 100€ | +0€ |
| 2 | 06-14 → 07-14 | 3 | 50% | 113€ | +13€ |
| 3 | 07-14 → 08-13 | 3 | 33% ❌ | 102€ | -12€ |
| 4 | 08-13 → 09-12 | **10** | 62.5% ✅ | 402€ | **+300€** |
| 5 | 09-12 → 10-12 | 3 | 66.7% ✅ | 720€ | +318€ |
| 6 | 10-12 → 11-11 | 2 | 50% | 838€ | +118€ |
| 7 | 11-11 → 12-11 | **10** | 57% ✅ | 1647€ | **+810€** |
| 8 | 12-11 → 01-10 | 6 | 50% | 1845€ | +198€ |
| 9 | 01-10 → 02-09 | 4 | 50% | 2495€ | +650€ |
| 10 | 02-09 → 03-11 | 2 | 50% | 2795€ | +299€ |
| 11 | 03-11 → 04-10 | **7** | 57% ✅ | **5866€** | **+3071€** 🚀 |
| 12 | 04-10 → 05-10 | 3 | **0%** ❌ | 3754€ | **-2112€** |

### Top 15 Features (importance LightGBM)

1. `risk_points` (214) — dominant
2. `score` (165)
3. `retest_count` (99)
4. `quality` (96)
5. `sweep_strength` (88)
6. `bars_sweep_to_validation` (60)
7. `has_mss_fvg` (49)
8. `rr` (47)
9. `bars_group_to_validation` (38)
10. `ob_strength` (28)
11. `kz_asia` (22)
12. `has_breaker_kz` (18)
13. `kz_ny_pm` (16)
14. `has_phase_expansion` (16)
15. `daily_bias_neutral` (15)

---

## ⚠️ Points d'attention

### 🚨 Variance élevée
- **Mois 11** : +3071€ (un mois fait 80% du PnL)
- **Mois 12** : -2112€ en 3 trades = -36% DD brutal
- **3 mois "faibles"** (1, 3, 12) avec 0-33% WR → la stratégie peut traverser 2-3 mois de purgatoire

### Volume faible
- **4.5 trades/mois** en moyenne (vs 14 XAU / 7 NAS)
- 4 mois sur 12 à seulement 1-3 trades
- → effet variance amplifié

### WR test 50% = breakeven sans MM compound
- AUC test 0.729 (vs 0.757 NAS, 0.74 XAU) → **modèle moins fiable**
- WR ~50% avec RR=2 = EV neutre. **La perf vient du MM compound 20%**
- → réduire risk à 10% en live pour smooth le DD

---

## 🛡️ Recommandation live

| Phase | Balance | Risk GER40 | Raison |
|---|---|---|---|
| Initial | 100€ | **10%** | Variance élevée, ne pas exposer |
| Stable | 500€+ | **5%** | Actif secondaire, complément XAU/NAS |

→ GER40 = **actif d'appoint, pas principal**.

---

## ✅ Critères validés

| Critère | Cible | Réalité | ✅/❌ |
|---|---|---|---|
| WR OOS | ≥ 50% | 52.4% | ✅ |
| AUC test | ≥ 0.70 | 0.729 | ✅ |
| Mois positifs | ≥ 60% | 75% | ✅ |
| DD max | < 50% | -36% | ✅ |
| Volume trades/mois | ≥ 5 | 4.5 | ⚠️ |
| Performance | > +50% | +3654% | ✅ |

---

## 📋 Fichiers actifs (production)

### Modèles
- `bot_v2/ml_model_GER40.pkl` — modèle dédié (ACTIF)
- `bot_v2/ml_features_GER40.json` — features

### Configuration
- `bot_v2/config.py` : `INSTRUMENT_PARAMS["GER40"]` (swing=3, displacement=1.0, score=160, quality=62)
- `bot_v2/ml_filter.py` :
  - `_MODELS_BY_INSTRUMENT["GER40"]` → modèle dédié
  - `ML_THRESHOLDS["GER40"] = 0.60`

### Datasets
- `data/ml_dataset_GER40_4ans.parquet` (1122 candidats, 4 ans)
- `data/ml_partial_M1/GER40_M1_*.parquet` (17 chunks 3 mois)

### Scripts dédiés
- `bot_v2/ml_dataset_GER40.py` — build dataset 4 ans
- `bot_v2/ml_train_GER40.py` — training
- `backtest_yearly_ger40.py` — backtest 12 mois OOS

---

## 🐛 Bug corrigé pendant le développement

**Consolidation ml_dataset.py** : nom des chunks M1 incohérent entre write (`GER40_M1_<date>`) et read post-build (`GER40_<date>`) → dataset final vide.
Fix : harmonisé sur `{inst}_{ltf}_{date}` partout dans `ml_dataset.py:437`.

---

## 📝 Décisions clés utilisateur (GER40)

| Date | Décision | Justification |
|---|---|---|
| 2026-05-17 | Phase 3 GER40 démarrée | Diversification après XAU + NAS |
| 2026-05-17 | Dataset 4 ans (vs 2 ans NAS) | 2 ans donnait VAL non monotone (overfit) |
| 2026-05-17 | Seuil ML = 0.60 | VAL 63.9%, TEST 51.4% — compromis |
| 2026-05-17 | RR plafond = 3.0 (default) | Pas de spécificité testée |
| 2026-05-17 | **GER40 = actif secondaire** | WR 52%, variance élevée, volume bas |
| 2026-05-17 | Risk recommandé live : 10% | Smooth le DD vs 20% |

---

## 🎯 Verdict final

> **GER40 est validé mais en actif secondaire dans le portefeuille.**
>
> Configuration : RR plafond 3.0, seuil ML 0.60, MM 10% recommandé live (vs 20% backtest).
>
> Performance attendue : **~52% WR**, **4-5 trades/mois**, variance élevée (1 mois peut faire ±30%).
>
> ⚠️ Le mois 11 (+3071€) et le mois 12 (-2112€) montrent que GER40 traverse des phases extrêmes. Idéal en **complément** de XAU/NAS, pas en remplacement.

---

*Document généré le 2026-05-17 — TradingBot v3+ / Phase 3 GER40*
