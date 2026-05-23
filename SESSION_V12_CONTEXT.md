# SESSION V12 — CONTEXTE COMPLET (2026-05-22 → 23 nuit)

> Document de reprise apres compactage. Tout l'etat de la session de debug/dev V12.

## VUE D'ENSEMBLE : ou on en est

Le bot TradingBot (ICT/SMC, broker Vantage MT5 compte 28879819) a fait **13L/3W le 22/05 en live** (catastrophe). On a diagnostiqué, corrige, et on construit V12. **Le bot VPS est ARRETE** depuis (pas de trade non surveille).

### Etat des versions
- **V11** = EN PROD avant (snapshots K=[3,7,15]) → identifie comme **data leakage** (le ML voyait 3-15 bougies du futur)
- **V12** = nouvelle version PUR AMONT (ML ne voit que le passe de l'OB). **Buildee + entrainee + validee OOS.** Modeles sur PC + Vast.

## LES BUGS TROUVES ET CORRIGES (dans l'ordre)

### Bug #1 : bougie M1 en cours (CORRIGE, commit 364c4e6)
`mt5_executor.py:200` faisait `copy_rates_from_pos(symbol, M1, 0, N)`. Le `pos=0` inclut la **bougie en cours non fermee** qui change a chaque tick. Le ML calculait sur des closes partiels → probas gonflees (0.75) → trades fantomes → SL.
**Fix** : `pos=1` (commence a la bougie precedente fermee). Deploye VPS.

### Bug #2 : data leakage V11 snapshots (CORRIGE via V12)
Le build V11 (`ml_dataset.py`) avec `SNAPSHOTS_K=[3,7,15]` coupait le cache a `vi+1+K` → le ML voyait K bougies POST-validation. A K=15, il voyait 15 min de futur. En live le bot entrait trop tard (apres que le mouvement soit consomme) → SL.
**Fix V12** : `BUILD_V12_MODE=1` → 1 seule evaluation par OB a `vi+1` (cache strictement <= vi), pas de feature `snapshot_k`.
- Code : `bot_v2/ml_dataset.py` (variable `_v12_mode`), `bot_v2/build_v12_dataset.py`, `bot_v2/train_v12_vantage.py`
- Live : `bot_v2/live_runner_v2.py` cascade `load_model` V12 > V11 > ... + en V12, `compute_asset` tronque df+cache a `vi`

### Bug #3 : V11.1 stabilite inutile en V12 (CORRIGE, commit 6d3e869)
Le check stabilite V11.1 (proba stable sur 2 cycles) servait a compenser le leakage V11. En V12 la proba est figee des validation → check inutile qui retarde. Desactive si modele V12 (`snapshot_k not in features`).

### Bug #4 : scan pas synchronise sur close M1 (CORRIGE, commit 3ced998)
Le live faisait `time.sleep(5s)` → 3-4 scans sur la meme bougie. Maintenant : sleep aligne sur `xx:xx:03` (3s apres chaque close M1) → 1 scan par bougie fermee = exactement comme le backtest.

### Bug #5 : filtre overnight forex (SUPPRIME, commit 4638a98)
`pipeline.py:182` bloquait tout forex (sauf USDJPY) de 21h-02h NY. Le user voulait bloquer la session Asia, mais ce filtre cassait le forex/or sans raison (le ML donne deja des probas basses en heure creuse). **Supprime.** Resultat : EURUSD 0.835, USDCAD 0.848 en NY → forex trade enfin.

### Bug #6 : FILL INSTANTANE (PAS ENCORE FIXE — LE SUJET ACTUEL)
**Decouvert via le backtest tick par tick.** 4 trades sur 8 = fill + SL en ~100 millisecondes :
- Pending place a 18:15:00, premier tick a 18:15:00.103 : le prix est DEJA dans la zone → fill instantane
- Tick suivant : SL touche → LOSS en 0.1s
**Cause** : le pending LIMIT se remplit alors que le prix a deja depasse l'entry au placement. Un vrai LIMIT ne devrait se remplir QUE sur retracement (prix revient toucher entry par le bon cote).
**A FIXER** : rejeter le pending si le prix est deja du mauvais cote de l'entry au placement, OU exiger un vrai retracement (prix doit d'abord s'eloigner puis revenir).

## RESULTATS DES BACKTESTS

### Backtest M1 19/05 (sans filtre overnight) — TROP OPTIMISTE
- 13 trades, **WR 84.6%**, +18.4R
- EURUSD 3x 100%, XAUUSD 2x 100%, forex+or tradent
- MAIS le M1 (high/low) cache les fills instantanes → surestime

### Backtest TICK 19/05 (le plus fidele, ~98% live) — REVELE LA VERITE
- 8 trades, **WR 37.5%**, +2.3R, 13 NO_FILL
- Par actif : BTCUSD 0%, DJ30 0%, EURUSD 0%, USDJPY 100% (2), XAUUSD 50%
- **4 LOSS = fills instantanes en ~100ms** (bug #6)
- Les NO_FILL = le M1 surestimait les fills via high/low
**Le tick est severe mais honnete. La verite live est entre M1 (84%) et tick (37%), mais le tick revele un vrai bug exploitable.**

## V12 OOS OFFICIEL (train_v12_vantage.py, deja valide)
Split : TRAIN 2018-03→2025-05, VAL 6 mois, **OOS 2025-11-22→2026-05-21 (6 mois jamais vus)**
14/14 PASS. Moyennes : **AUC OOS 0.742, WR@0.65 77.4%**
| Actif | AUC | WR@0.65 |
|---|---|---|
| XAUUSD | 0.723 | 73.1% |
| NAS100 | 0.742 | 79.5% |
| EURUSD | 0.763 | 82.5% |
| AUDUSD | 0.743 | 83.8% |
| USDCAD | 0.757 | 80.6% |
| USDCHF | 0.757 | 82.5% |
(les 8 autres entre 69-79%)

## GRID SEARCH (fait, conclusions)
20 configs sur XAUUSD 3 mois. Conclusions :
- `16_v11_snapshots` ressort "meilleur" (212 trades 67% WR) MAIS c'est la TRICHE temporelle (voit le futur). A ignorer.
- Baseline V12 (RR=1.5, sws=1) = proche optimum
- **Filtre phase = AUCUN impact** (configs 1-5 identiques) → innocente
- sws=2/3 = pire. RR=1.0 = plus de volume mais WR plus bas.

## INFRASTRUCTURE

### Vast.ai (serveur de calcul)
- Instance : `ssh -i ~/.ssh/vast_v8 -p 41297 root@31.13.223.140`
- **AMD EPYC 9754, 512 threads / 128 cores physiques, 1.5 TB RAM**, $1.346/h
- `/workspace/TradingBot` = repo cloné
- `data_vantage/` = 8 ans de bougies (2018-03 → 2026-05-21), 852 MB, uploadé via tar
- `data_ticks/` = ticks 19/05 (2.93M ticks, 30.7 MB) uploadé via tar
- **REGLE SATURATION** (en memoire) : tout run doit utiliser ~128 workers, env BLAS=1, verifier load proche du nb workers + %idle <30%. Decouper en >=128 taches.

### VPS prod (Contabo)
- `ssh -i ~/.ssh/contabo_debug administrator@167.86.83.144`
- `C:\Users\Administrator\tradingbote\` (repo), bot lance via schtasks `TradingBotV10`
- **ARRETE actuellement** (pas de trade pendant le debug)
- Compte MT5 28879819 Vantage, balance ~135 EUR

### Repo
- GitHub : `https://github.com/simovadev/tradingbote.git`
- HEAD actuel : `c0b5d2b` (backtest tick sature)
- Dashboard : `https://tradingbote-production.up.railway.app/`

## SCRIPTS CLES
- `bot_v2/build_v12_dataset.py` — build dataset V12 (BUILD_V12_MODE=1)
- `bot_v2/train_v12_vantage.py` — train + verdict OOS
- `backtest_v12_realistic.py` — backtest live-style M1 (cycle par cycle)
- `run_v12_backtest_massive.py` — wrapper parallel (--period_hours, --workers)
- `backtest_v12_tick.py` — tick local (lit MT5)
- `backtest_v12_tick_vast.py` — tick Vast (lit data_ticks parquet, parallel par fenetre 2h)
- `export_ticks.py` — exporte ticks MT5 → data_ticks/*.parquet
- `grid_build_v12.py` / `grid_search_v12.py` — grid configs
- `rapport_backtest.py` — genere graphiques (equity, WR/PnL par actif, distrib proba, timeline) — matplotlib installe en local

## CONFIG V12 FIGEE
- swing_strength=1 (SWS_OVERRIDE), RR=1.5 (RR_OVERRIDE)
- ML_THRESHOLDS 0.70 partout (ml_filter.py)
- 59 features (pas de snapshot_k)
- Filtre overnight forex SUPPRIME
- Cap age OB 30min, expire pending 60min, risk 2%
- Cascade load_model V12 > V11 > V10 > ...

## PROCHAINE ETAPE IMMEDIATE (ce qu'on fait LA)
**Fixer le bug #6 (fill instantane).** Le pending LIMIT ne doit se remplir que sur un VRAI retracement :
- SELL LIMIT : le prix doit etre EN DESSOUS de l'entry au placement, puis MONTER toucher l'entry
- BUY LIMIT : le prix doit etre AU DESSUS de l'entry au placement, puis DESCENDRE toucher l'entry
Si au placement le prix est deja du mauvais cote → skip (ou attendre un eloignement puis retour).
A corriger dans : la logique de placement live (`live_runner_v2.execute_setup`) ET dans `simulate_on_ticks` du backtest tick (pour valider).

Apres le fix : re-run backtest tick 19/05 → le WR devrait remonter (on vire les fills instantanes perdants).
