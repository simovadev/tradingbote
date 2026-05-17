# 🚫 BILAN — USOUSD (Phase 4) — ABANDONNÉ

**Date** : 2026-05-17
**Asset** : USOUSD (WTI Crude, broker Vantage)
**Statut** : ❌ **ABANDONNÉ — non viable en live**

---

## 🎯 Contexte

USOUSD était la Phase 4 du déploiement multi-asset (après XAUUSD, NAS100, GER40).
Build dataset + training ML réalisés sur **4 ans** avec multiple itérations.

---

## 📊 Tests effectués

### Itération 1 — Filtres standards
- `min_displacement_atr=1.0`, `swing_strength_m1=2`
- 592 candidats, 427 trades fermés
- AUC test 0.705, WR test @0.55 = 63.6%

### Itération 2 — Filtres relâchés
- `min_displacement_atr=0.7`, `parent_ob_tolerance=0.005`, `discount_premium=0.08`
- **862 candidats, 636 trades fermés**
- **AUC test 0.789, WR test @0.55 = 75.7%** ← chiffres flatteurs mais trompeurs

### Itération 3 — Activation `min_sl_points` (15× spread)
- `min_sl_points=0.60$` activé
- **Effondrement à 172 candidats** (chunks récents = 0)

---

## 🚨 Pourquoi USOUSD ne marche pas

### 1. La structure ICT ne s'applique pas au WTI

Distribution des `risk_points` sur les 862 setups détectés (v2 relâchée) :

| Percentile | SL en points |
|---|---|
| P10 | 0.030$ (3 cents) |
| P25 | 0.053$ (5 cents) |
| **P50** | **0.093$ (9 cents)** |
| P75 | 0.155$ (15 cents) |
| P90 | 0.240$ (24 cents) |

**SL médian de 9 cents** = on détecte des micro-mouvements, pas des vrais Order Blocks ICT.
Seulement **2 setups sur 862** ont un SL >= 0.60$.

### 2. Spreads + slippage tuent l'EV

- Spread WTI typique : **4 cents** par côté = 8 cents aller-retour
- Slippage stops : 2-3 cents
- **Coût fixe par trade : ~10-11 cents**

Si SL = 10 cents et coût = 10 cents → le spread mange **100% du risk**.
Pour rendre USOUSD viable il faudrait SL ≥ 0.50$+, soit **moins de 1% des setups détectés**.

### 3. Le WTI bouge sur du macro, pas de l'ICT

Les drivers de prix du pétrole sont :
- EIA mercredi 10h30 NY
- Décisions OPEC+
- Géopolitique (Moyen-Orient, Russie, Iran)
- Réserves stratégiques US

**Aucun** de ces facteurs n'est dans les features ML. On prédisait du bruit avec un modèle conçu pour de la structure ICT propre.

### 4. Les WR backtest étaient un mirage

- WR 75.7% sur setups à SL 9 cents = facile en backtest sans coûts
- En live avec spread 4 cents : TP rarement atteint → WR effondre

### 5. Le screenshot user (19 Oct 2025)

Trade BUY ml=0.578 : la bougie de signal fait ~5 cents de range total. Entry/SL/TP tiennent dans **2 bougies M1**.
→ En live : impossible à trader sans se faire sortir par le spread ou un tick adverse.

---

## ✅ Décision finale

**Le portefeuille reste à 3 actifs validés** :

| Actif | WR OOS | Volume/mois | Statut |
|---|---|---|---|
| **XAUUSD** | 63.8% | ~14 | ⭐ pilier |
| **NAS100** | ~60% | 7-10 | ⭐ pilier |
| **GER40** | ~52% | 4-5 | secondaire |

3 actifs décorrélés (métal + indice US + indice EU) suffisent pour la diversification.

---

## 🧹 Nettoyage effectué

- USOUSD retiré de `ml_filter.py` (`_MODELS_BY_INSTRUMENT`, `_FEATURES_BY_INSTRUMENT`, `ML_THRESHOLDS`)
- USOUSD retiré du sélecteur dashboard `ml_dashboard.py`
- `min_sl_points` réinitialisé à 0.30 (valeur originale, non bloquant)
- Fichiers ML conservés au cas où : `ml_model_USOUSD.pkl`, `ml_features_USOUSD.json`
- Dataset conservé : `data/ml_dataset_USOUSD_4ans.parquet`
- Scripts conservés : `ml_dataset_USOUSD.py`, `ml_train_USOUSD.py`

---

## 📝 Leçons apprises

1. **AUC élevée ≠ tradable**. Toujours valider la distribution des `risk_points` et comparer aux spreads/slippage réels.
2. **Activer `min_sl_points` dès le début** sur tout nouvel actif. Si <5% des setups passent → l'actif ne respecte pas la structure ICT.
3. **Vérifier que l'actif respecte les principes ICT/SMC** avant de lancer un build de 4 ans. Le pétrole = macro driven, pas structure.
4. **Le spread du broker est un paramètre clé** ignoré jusqu'ici. À documenter par actif dans `INSTRUMENTS` (`typical_spread`).

---

## 🚀 Prochaine étape

**Phase 5 : BTCUSD** (24/7, sera intéressant car comportement différent : pas de killzones classiques, volatilité crypto).

---

*Document généré le 2026-05-17 — TradingBot v3+ / Abandon USOUSD documenté*
