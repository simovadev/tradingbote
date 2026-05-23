# SESSION V12 TICK PAR TICK - ETAT FINAL AVANT COMPACT (2026-05-23 nuit)

> Document de reprise CRITIQUE - on est tres proche du resultat reel.
> Lire SESSION_V12_CONTEXT.md pour le contexte initial (bugs #1 a #5).

## OU ON EN EST EXACTEMENT

On simule le bot V12 **tick par tick** sur Vast.ai pour avoir le resultat
le plus proche possible du live reel. Apres avoir corrige le bug #6
(fill instantane), on compare maintenant LIMIT vs MARKET avec latence
et commission, en parallele sur 4 scenarios.

**Bot VPS = TOUJOURS ARRETE** (pas de trade en attendant).

## LES 6 BUGS DECOUVERTS (chronologie)

1. **Bougie M1 en cours** (CORRIGE commit 364c4e6) : `pos=1` au lieu de `pos=0`
2. **Data leakage V11 snapshots** (CORRIGE via V12 PUR AMONT)
3. **V11.1 stabilite** (DESACTIVE en V12)
4. **Scan pas sync close M1** (CORRIGE commit 3ced998 : sleep aligne sur xx:xx:03)
5. **Filtre overnight forex** (SUPPRIME commit 4638a98 : cassait le forex sans raison)
6. **Fill instantane** (CORRIGE commit d90e4da dans le backtest tick) :
   - SELL LIMIT doit etre AU-DESSUS du prix, BUY LIMIT EN DESSOUS
   - Si prix deja du mauvais cote au placement -> INVALID_PRICE (rejet MT5)
   - Decouvert via tick : 4 trades en ~100ms = fills instantanes faux

## DECOUVERTE MAJEURE 19/05 (analyse_nofill.py)

Sur les 13 NO_FILL du backtest tick 19/05 (LIMIT) :
- **13 RATE_WIN** : le prix a touche le TP **sans nous** (LIMIT a fait rater le mouvement)
- **0 EVITE_LOSS** : aucun cas ou le LIMIT a protege d'un loss
- **0 NEUTRE** : aucun cas de stagnation

**Conclusion ferme** : le LIMIT attend un retracement qui ne vient JAMAIS sur les bons setups.
**L'intuition du user (depuis le matin) est validee a 100%** : il faut entrer en MARKET, pas LIMIT.

## RESULTATS BACKTESTS (chronologie)

### Backtest M1 19/05 (filtre overnight enleve)
- WR 84.6%, +18.4R sur 13 trades fermes
- EURUSD 3x WIN 100%, XAUUSD 2x WIN 100%
- **TROP OPTIMISTE** : le M1 ne voit pas les fills instantanes (high/low cache la realite tick)

### Backtest TICK 19/05 v1 (avant fix #6)
- 8 trades fermes, WR 37.5%, +2.3R
- 13 NO_FILL, 4 LOSS = fills+SL en ~100ms (BUG)

### Backtest TICK 19/05 v2 (apres fix #6, mode LIMIT)
- **3 trades reels fermes, WR 67%, +3.5R**
- 13 NO_FILL (= 13 WIN rates, prouve via analyse_nofill.py)
- 5 INVALID_PRICE (prix deja mauvais cote = MT5 aurait rejete l'ordre)
- Sur 21 setups detectes, seulement 3 trades reels → **taux conversion tres faible**

### EN COURS : Compare 4 scenarios (LIMIT + MARKET 5s/18s/30s)
Lance via `compare_entry_modes.py`. 4 scenarios paralleles sur Vast (64 workers chacun = 256 cores) :
1. **LIMIT lat=0 commission=0.07** (baseline)
2. **MARKET lat=5s commission=0.07**
3. **MARKET lat=18s commission=0.07**
4. **MARKET lat=30s commission=0.07**

Commission = **7% du SL** par trade (= cout Vantage round-turn approxime).
Apres : ce qui sort le meilleur PnL gagne. On l'applique au live.

## INFRASTRUCTURE

### Vast.ai instance ACTIVE
- IP : `ssh -i ~/.ssh/vast_v8 -p 41297 root@31.13.223.140`
- AMD EPYC 9754, 512 threads / 128 cores physiques, 1.5 TB RAM
- $1.346/h - NE PAS OUBLIER DE LA DESTROY APRES
- `/workspace/TradingBot` = repo synchronise
- `data_vantage/` (8 ans bougies, 852 MB) + `data_ticks/` (19/05, 30 MB)

### REGLE SATURATION (en memoire)
- Tout run Vast doit utiliser ~128 workers, env BLAS=1
- Verifier load proche du nb workers + %idle <30%
- Decouper en >=128 taches si moins

### VPS prod Contabo (ARRETE)
- `ssh -i ~/.ssh/contabo_debug administrator@167.86.83.144`
- bot lance via schtasks `TradingBotV10`
- Balance ~135 EUR sur MT5 Vantage 28879819

## COMMITS RECENTS

- `364c4e6` fix bougie en cours
- `e184d49` build V12 sans leakage
- `4638a98` suppression filtre overnight forex
- `3ced998` scan sync close M1 live
- `d90e4da` fix #6 fill instantane (INVALID_PRICE)
- `72ce3d9` latence + commission tick
- `ef86103` compare 4 scenarios LIMIT vs MARKET

HEAD actuel : `ef86103`

## SCRIPTS CRITIQUES

| Script | Role |
|---|---|
| `bot_v2/build_v12_dataset.py` | Build dataset V12 (BUILD_V12_MODE=1) |
| `bot_v2/train_v12_vantage.py` | Train V12, OOS 6 mois → AUC 0.742 WR 77.4% |
| `backtest_v12_tick_vast.py` | Backtest tick par tick (LIMIT + MARKET) |
| `compare_entry_modes.py` | Lance 4 scenarios en parallele |
| `export_ticks.py` | Exporte ticks MT5 → data_ticks/*.parquet |
| `analyse_nofill.py` | Analyse NO_FILL : rate_win vs evite_loss |
| `rapport_backtest.py` | Genere graphiques (equity, WR, PnL, distrib) - matplotlib OK |

## CONFIG V12 FIGEE
- `swing_strength=1` (SWS_OVERRIDE), `RR=1.5` (RR_OVERRIDE)
- `ML_THRESHOLDS = 0.70` partout
- 59 features (pas de `snapshot_k`)
- Filtre overnight SUPPRIME
- Cap age OB = 30 min, expire pending = 60 min, risk = 2%
- Cascade `load_model V12 > V11 > V10 > ...`

## RESULTATS V12 OOS OFFICIEL (deja valides)
Split TRAIN 2018-03→2025-05, VAL 6m, **OOS 2025-11-22→2026-05-21** (6m jamais vus)
14/14 PASS. Moyennes : **AUC OOS 0.742, WR@0.65 77.4%**

## PROCHAINES ETAPES (immediates)

1. **Attendre fin des 4 scenarios** (~5-7 min apres lancement)
   Cmd suivi :
   ```
   ssh -i ~/.ssh/vast_v8 -p 41297 root@31.13.223.140 "tail -10 /workspace/compare.log"
   ```

2. **Lire le tableau comparatif** (4 lignes : LIMIT vs MARKET lat 5/18/30s)
   - PnL apres commission, WR, nb trades fermes, INVALID, NO_FILL
   - Le meilleur PnL R = gagnant

3. **Appliquer au live** :
   - Si MARKET gagne → modifier `execute_setup` dans `live_runner_v2.py` pour
     placer un ordre MARKET au lieu de LIMIT (ou STOP MARKET selon implementation)
   - Si LIMIT gagne avec commission → on garde le mecanisme actuel
   - Dans tous les cas, **respecter la latence reelle** (le user dit ~18s sur VPS)

4. **DESTROY l'instance Vast** une fois tout fini (couts $1.346/h)

5. **Question user en suspens** : *"avons nous les logs Railway sur la latence ?"*
   - Railway = dashboard hostng `tradingbote-production.up.railway.app`
   - Les logs Railway peuvent montrer la latence entre push du bot et reception
   - A creuser apres le compare en cours

## QUESTION OUVERTE NON RESOLUE

**Latence reelle a appliquer** : le user a dit 18s mais on teste 5/18/30s pour avoir le spectre.
Une fois le compare termine, on saura :
- Si MARKET-5s est rentable mais pas MARKET-30s → besoin de minimiser la latence
- Si MARKET-30s est rentable aussi → robuste, on peut deployer sans optimiser la latence

## CE QU'ON A APPRIS / VALIDE

1. **V12 sans leakage = OOS valide** (AUC 0.742, WR 77%)
2. **Filtre overnight bloquait le forex** sans raison (le ML filtre deja les heures creuses)
3. **Le M1 ment** (cache fills instantanes via high/low) - le tick est honnete
4. **Le LIMIT rate les bons mouvements** (13/13 NO_FILL = 13 WIN rates)
5. **Le MARKET avec commission/latence est probablement la solution** - validation imminente

## NOTES

- Bot live VPS = ARRETE depuis le 22/05 au soir
- Compte MT5 Vantage 28879819, balance ~135 EUR
- Repo GitHub : `https://github.com/simovadev/tradingbote.git`
- Dashboard : `https://tradingbote-production.up.railway.app/`
- User a etabli REGLE saturation Vast (en memoire dans vast_saturation_rule.md)
