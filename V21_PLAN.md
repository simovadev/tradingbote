# V21 — TradingBot Plan : Self-Learning ICT Agent

> **Statut** : Vision, NON démarré. Attend validation V20 en production.
> **Décision** : V20 d'abord en prod, comparaison perf live → si V20 OK, alors V21 vaut l'investissement.
> **Cible** : Saut générationnel — passer d'un classifieur ICT à un agent autonome complet.

---

## 1. Pré-requis V21 (à valider AVANT de commencer)

V21 ne démarre **QUE SI** ces critères V20 sont remplis en démo live :

| Critère | Seuil |
|---|---|
| WR live sur 100+ trades (sur 1-2 semaines démo) | ≥ 65% |
| Stabilité du WR week-over-week | écart < 10 points |
| Drawdown max | < 15% du compte |
| AUC OOS V20 (déjà mesuré training) | ≥ 0.70 |
| Bot stable en exécution | 0 crash / 48h |

Si tout ≥ seuils → V21 ON. Sinon → debug V20.

---

## 2. Différences fondamentales V20 → V21

V20 = **un modèle qui prédit**. V21 = **un agent qui apprend en continu**.

| Aspect | V20 (actuel) | V21 (objectif) |
|--------|--------------|----------------|
| **Features** | 52 ICT hand-crafted | **Bougies brutes seules** — le model découvre |
| **Architecture** | 235k params figée | **10-50M params** + genetic v5 sur archi |
| **Hyperparams** | Config médiane générique | **Genetic v5 / Bayesian** sur 100-500 variantes |
| **Apprentissage** | Offline (8 ans historique) | + **Online** (s'adapte aux trades live) |
| **Cross-asset** | 1 OB isolé | **Voit 5-10 actifs simultanément** (corrélations) |
| **Feedback** | Aucun | **Reinforcement Learning** (récompensé par PnL démo) |
| **Décisions** | "trade ou pas" + proba | + **sizing** + **timing** + **gestion portefeuille** |
| **Multi-objectif** | Juste WIN/LOSS | + RR optimal + risque global |

---

## 3. Les 3 piliers V21

### Pilier A — Self-discovery des features (le model découvre tout)

**Problème V20** : on a décidé manuellement les 52 features (ob_strength, sweep_strength, has_FVG_sync...). Si un signal ICT important n'est pas dans la liste = invisible.

**Solution V21** : input = bougies brutes uniquement
- 1000 bougies M1 (vs 240)
- 500 M15 + 200 H1 + 100 D1
- Le model apprend son propre encoder via Transformer profond (8-12 couches, self-attention multi-head)
- Il "découvre" lui-même que OB+sweep+FVG comptent
- Peut même trouver des patterns auxquels on n'a pas pensé

**Implémentation** :
- Remplacer `V20ICTNet` par `V21Transformer` (architecture style GPT-mini)
- Drop branch ICT features
- Input shape : `(B, 1000+500+200+100, 5 OHLCV)` = ~5800 timesteps
- Output : same proba + auxiliary heads (multi-task)

### Pilier B — Genetic v5 sur l'architecture entière

**Problème V20** : architecture choisie à la main (`embed_dim=128, hidden=64, n_layers=4`). Peut-être sub-optimale.

**Solution V21** : lancer 100-500 variantes en parallel
- Variations testées :
  - depth : 4 / 8 / 16 / 24 couches
  - hidden : 64 / 256 / 512 / 1024
  - attention heads : 4 / 8 / 16
  - dropout : 0.1 / 0.2 / 0.4
  - LR schedule : cosine / onecycle / warmup-decay
  - batch size : 256 / 512 / 1024 / 2048
- Critère de sélection : **OOS AUC** + gap train-OOS minimal
- Early stop si val AUC ne progresse pas après 5 epochs
- Genetic algo (mutation + crossover) sur 5-10 générations

**Compute** :
- ~12-24h sur Vast RTX PRO 6000 Blackwell
- Coût estimé : ~$30-60 (à $1.34/h)

### Pilier C — Reinforcement Learning online

**Problème V20** : apprend uniquement sur l'historique simulé. Pas de retour des trades réels.

**Solution V21** : agent RL qui s'adapte en continu
- État `s` : OB courant + features marché + état portefeuille (positions ouvertes, P&L journée, corrélations)
- Action `a` : `(trade?, direction, sizing %, timing offset)`
- Récompense `r` : PnL réalisé du trade + bonus stabilité (low DD) + penalty over-trading
- Algorithme : **PPO** (Proximal Policy Optimization) ou **SAC**
- Mode shadow d'abord (paper trading) → live progressif

**Apprend des situations cross-asset** :
- "Si EURUSD et GBPUSD sont corrélés à 0.9 ce moment, ne pas ouvrir les 2 en même temps"
- "Si déjà 5 positions ouvertes et drawdown -3%, refuser nouveaux trades faibles"
- "Si vendredi 21h UTC, fermer toutes positions" (auto-discovered)

---

## 4. Cross-asset multi-séquences (bonus important)

Tous les actifs sont **corrélés** :
- DXY ↑ → forex majeurs ↓
- VIX ↑ → indices ↓, USDJPY ↑
- BTC ↑ → ETH ↑ (high correlation)
- XAU ↑ → DXY ↓

**V21 input** : pas juste 1 actif isolé, mais **panel** :
- Actif en focus (celui à trader)
- 5-10 actifs de référence (DXY, SP500, VIX, XAU, BTC, EUR/USD, USD/JPY, BUND, OIL)
- Cross-attention entre eux dans le Transformer
- Le model apprend les corrélations dynamiques (pas figées)

---

## 5. Plan d'exécution V21

### Phase 1 : Préparation (1 semaine)
- [ ] Valider V20 en démo live (critères §1)
- [ ] Étendre dataset à 10 ans (vs 8 actuels) si dispo Vantage
- [ ] Ajouter 5-10 actifs "référence cross-asset" (DXY, VIX, BUND...)
- [ ] Coder `V21Transformer` architecture de base
- [ ] Setup pipeline data multi-asset aligné par timestamp

### Phase 2 : Genetic v5 (2-3 jours sur Vast)
- [ ] Define search space (100-500 archi variantes)
- [ ] Run en parallel sur GPU Blackwell
- [ ] Save top 5 architectures
- [ ] Train chacune à fond → choisir winner par OOS AUC

### Phase 3 : Training V21 final (2-3 jours sur Vast)
- [ ] Train winner architecture sur 10 ans + cross-asset
- [ ] Adversarial validation (CLEAN < 0.65)
- [ ] OOS WR @ thresholds
- [ ] Comparison directe V20 vs V21 (même OOS period)

### Phase 4 : RL online layer (1-2 semaines)
- [ ] PPO agent autour de V21 (V21 fournit features state)
- [ ] Reward shaping (PnL + risk penalties + diversity bonus)
- [ ] Train sur 1 an de trades simulés
- [ ] Shadow démo 1 semaine

### Phase 5 : Production V21 (1 semaine)
- [ ] Bot live V21 en shadow démo
- [ ] Comparison live V20 vs V21 (mêmes trades, sizing identique)
- [ ] Si V21 ≥ V20 sur 100+ trades → switch prod

---

## 6. Coût total V21

| Item | Estimation |
|---|---|
| Compute Vast (genetic + train) | ~$100-200 |
| Compute Vast (RL training) | ~$100-150 |
| Compute Vast (validation + comparison) | ~$50 |
| Temps total dev | 3-4 semaines effort plein |
| **Total** | **~$300-500 + 1 mois travail** |

---

## 7. Risques V21

| Risque | Mitigation |
|---|---|
| Overfit (gros model + plus de data demandée) | Dropout heavy, weight decay, embargo strict, adversarial check |
| RL instable (PPO sensible) | Reward clipping, KL penalty, warm start depuis V21 supervisé |
| Pas mieux que V20 | Phase 5 comparison live = stop if no gain |
| Compute trop cher | Stop genetic à 50 variantes au lieu de 500 si budget serré |
| Black-box (model trop complexe à debug) | Logs attention weights, saliency maps pour expliquer décisions |

---

## 8. Critères de succès V21 vs V20

V21 est validé si TOUS ces critères battent V20 sur le même OOS :

| Métrique | V20 cible | V21 minimum acceptable |
|---|---|---|
| AUC OOS | ~0.70 | **≥ 0.78** (+0.08) |
| WR @ 0.70 OOS | ~75% | **≥ 82%** (+7 pts) |
| Gap train-OOS | < 0.05 | **< 0.05** (pas pire) |
| WR live sur 100 trades démo | ~70% | **≥ 78%** |
| Drawdown max démo | < 15% | **< 12%** |
| Adversarial CLEAN | < 0.65 | **< 0.62** |

Si 4/6+ → V21 prod. Si 2-3/6 → debug. Si 0-1/6 → revert V20.

---

## 9. Hors-scope V21 (réservé V22+)

- Multi-broker (Vantage + IC Markets + Pepperstone)
- Multi-strategy ensemble (V19 + V20 + V21 vote)
- News/calendar economic integration
- Sentiment social (Twitter, news)
- Order book L2 (depth)
- Plateforme automatique de redéploiement zero-downtime

---

## 10. Notes diverses

- **V20 reste actif en parallèle** (au moins 1 mois) pour benchmark continu
- **Le live runner doit supporter cascade V21 → V20 → V19** pour rollback safety
- **Repos privé sur GitHub** pour V21 (code + weights), V20 reste public
- **Pas de re-train mensuel forcé** : V21 = self-learning, juste retrain quartely on full data

---

*Document créé le 2026-05-27. Auteur : session collaborative user + Claude. Statut : à valider V20 puis activer.*
