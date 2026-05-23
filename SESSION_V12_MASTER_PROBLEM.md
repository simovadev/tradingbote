# SESSION V12 — DIAGNOSTIC PROBLEME RACINE (avant compact 2026-05-23)

> CE QUE TU DOIS LIRE EN PREMIER APRES COMPACT.
> Voir aussi SESSION_V12_CONTEXT.md et SESSION_V12_TICK_FINALE.md pour l'historique.

## LE CONSTAT BRUTAL (a tete reposee)

On a passe la nuit a corriger 6 bugs (bougie en cours, leakage, filtre overnight,
V11.1, fill instantane, etc.) et a developper un **backtest tick par tick ultra-realiste**
(98% fidele live, ticks bid/ask reels MT5, latence + commission 7% inclus).

Apres tout ce travail, **resultats reels sur 3 jours pris au hasard (14, 15, 18 mai)** :

| Jour | LIMIT (trades / WR / PnL) | MARKET 30s |
|---|---|---|
| 14/05 | 5 / 20% / **-2.2R** | 17 / 47% / **-6.2R** |
| 15/05 | 11 / 27% / **-4.0R** | 13 / 77% / **+1.9R** |
| 18/05 | 1 / 0% / -1.1R | 3 / 67% / **-0.4R** |
| **Total 3j** | **17 trades, WR 23%, -7.3R** | **33 trades, WR 60%, -4.7R** |

(19 et 20 mai non testes, runs annules. 19 deja teste seul : MARKET +5.3R, LIMIT +3.3R)

## CE QUE CA VEUT DIRE (analyse user 23/05)

> *"sa me rassure car sa veut dire qu'on peut debug jusqu'a qu'on a un bon winrate.
> car c'est pas un probleme de trade — si c'etait un probleme de trade le OOS marcherai pas
> et meme le semi-reel avec les M1."*

**Le user a raison** :
- Le OOS V12 a marche (AUC 0.742, WR 77.4% sur 6 mois jamais vus)
- Le backtest M1 marche aussi (WR 84.6% sur 19/05)
- Le backtest TICK (la realite vraie) **NE MARCHE PAS** (WR 23% LIMIT, 60% MARKET sur 3 jours)

→ **Il y a une divergence entre la simulation par bougies et la simulation tick reelle.**
Le probleme N'EST PAS le ML ni les setups. C'est dans la mecanique d'execution
(fill, SL/TP au tick, sequence intra-bougie).

## CE QU'ON SAIT POUR DEBUG ENSUITE

### 1. Le MARKET prend plus de trades MAIS perd plus aussi
- LIMIT : 17 trades, 23% WR sur 3 jours
- MARKET : 33 trades (×2), 60% WR mais plus de volume = plus de risque

→ Le MARKET n'est PAS la solution miracle. Il convertit les NO_FILL en trades,
mais ces trades ajoutes ont un WR ~50%, pas 77% comme l'OOS predit.

### 2. INVALID_PRICE eleve (3-26 par jour)
Le bot calcule des entries du MAUVAIS COTE du prix actuel. C'est un vrai
probleme : meme MT5 refuserait ces ordres (ou les fillerait au prix marche
au lieu de l'entry voulu = trade degenere).
→ A creuser : pourquoi le bot place-t-il entry=ob_high/ob_low quand le prix
a deja depasse ce niveau au moment du placement ?

### 3. Les fills LIMIT instantanes (~100ms) revelent une fragilite
Sur des OB tres proches du prix actuel, le LIMIT se remplit en 100ms puis SL.
C'est REEL en live aussi. Le fix actuel le compte INVALID_PRICE.

## HYPOTHESES A TESTER (par ordre de priorite)

### Hyp 1 : Le RR est mal calcule par OB → SL trop serre vs marche reel
Les OB ont des dimensions M1 (high-low de 1 a 5 bougies). En realite, le marche
fait des **mouvements intra-tick** plus grands. Le SL place au low/high de l'OB
est souvent touche par du bruit tick avant que le mouvement directionnel
ne se confirme. **A verifier** : taille moyenne du SL en pips vs ATR M1 reel.

### Hyp 2 : Le ML voit des features stables mais le LIVE n'a pas les memes prix
Hypothese du "broker different" : le V12 a ete entraine sur des bougies parquet.
Les ticks live sont... les memes ticks (on a verifie parquet = MT5 exact).
**Ecarte** sauf si le bot live et le backtest divergent encore.

### Hyp 3 : Le timing du scan (toutes les 5 min en backtest, sync close M1 en live)
- Backtest : scan toutes les 5 min, voit un OB qui pourrait dater de 4 min
- Live (post-fix #4) : scan a chaque close M1, voit l'OB des qu'il est valide
**Difference** : en backtest on rate des OB qui apparaissent et disparaissent
entre 2 scans 5 min. Le LIVE serait plus reactif → meilleurs trades.
**A tester** : backtest tick avec --step 1 (scan toutes les minutes).

### Hyp 4 : Le filtre overnight (qu'on a SUPPRIME) servait quand meme
On a vire le filtre forex overnight US (21h-02h NY). Mais le 14/05 a peut-etre
ete plombe par des trades overnight pourris. **A verifier** : sur les LOSS du
14/05, combien etaient en heure overnight ?

### Hyp 5 : Le ML V12 surestime sa precision OOS
Le OOS V12 a fait 77% WR sur 6 mois en simulant via simulate_trade (bougies M1).
Mais le **tick reel** divergent. Possible que la metric OOS du V12 elle-meme
est gonflee parce qu'elle utilise la meme logique M1 que le backtest M1 (qui
ment aussi en cachant les fills instantanes via high/low).

## CE QU'IL FAUT FAIRE EN PREMIER

1. **Analyser jour par jour les LOSS** : quel etait le contexte de chaque LOSS ?
   - Heure / session
   - Distance entry-prix au placement
   - Direction reelle du marche apres placement
   
2. **Tester scan toutes les minutes (--step 1)** au lieu de 5 min dans le backtest tick
   → Reproduire au max le live sync close M1

3. **Comparer les trades du backtest tick avec ce que le bot LIVE ferait**
   → Pour 1 OB precis : meme entry/sl/tp ? meme proba ML ?

4. **Verifier la taille des SL vs ATR**
   → Si SL << ATR moyen, c'est trop serre, normal d'etre touche par du bruit

5. **Re-tester avec --step 1 + step intelligent quand setup pendant**
   → Le live scanne chaque minute, le backtest doit faire pareil

## CE QU'IL NE FAUT PAS FAIRE

- ❌ Rebuilder un V13 immediatement : on ne sait pas encore POURQUOI les setups perdent
- ❌ Modifier le ML sans avoir compris : le V12 OOS est bon, le probleme est ailleurs
- ❌ Deployer le live en aveugle : on a la preuve tick que ca perdrait ~5R/jour

## INFRASTRUCTURE

- **Vast.ai instance ACTIVE** (`ssh -i ~/.ssh/vast_v8 -p 41297 root@31.13.223.140`)
  - $1.346/h - **PENSER A DESTROY** si on s'arrete
- **VPS Contabo bot = ARRETE** depuis 22/05 soir
- **Repo** : `https://github.com/simovadev/tradingbote.git` HEAD `62e6e1d`
- **MT5 Vantage 28879819**, balance ~135 EUR
- **Ticks 5 jours exportes** dans `data_ticks/` (PC) et uploades sur Vast
  - Dates : 14, 15, 18, 19, 20 mai 2026

## COMMITS RECENTS (a l'envers)

- `62e6e1d` compare 5 jours LIMIT vs MARKET (annule)
- `ef86103` compare 4 scenarios LIMIT + MARKET 5/18/30s
- `72ce3d9` latence + commission tick
- `d90e4da` fix #6 fill instantane (INVALID_PRICE)
- `c0b5d2b` saturation Vast (168 taches)
- `b306424` mode entree MARKET tick
- `4638a98` suppression filtre overnight forex
- `3ced998` scan sync close M1 live
- `e184d49` build V12 sans leakage
- `364c4e6` fix bougie en cours

## SCRIPTS DISPONIBLES

| Script | Role |
|---|---|
| `backtest_v12_tick_vast.py` | Backtest tick (LIMIT + MARKET, --latency_s, --commission_r) |
| `compare_5days.py` | Compare 2 modes sur 5 jours |
| `compare_entry_modes.py` | Compare 4 scenarios sur 1 jour |
| `export_ticks.py` | Exporter ticks MT5 → data_ticks/ |
| `analyse_nofill.py` | Diagnostic NO_FILL (rate_win vs evite_loss) |
| `rapport_backtest.py` | Generer graphiques (matplotlib OK) |
| `grid_build_v12.py` | Grid configs filtres pipeline |
| `bot_v2/build_v12_dataset.py` | Build dataset V12 PUR AMONT |
| `bot_v2/train_v12_vantage.py` | Train V12 + verdict OOS |

## CONFIG V12 ACTUELLE
- swing_strength=1, RR=1.5, ML_THRESHOLDS=0.70
- 59 features (pas snapshot_k)
- Filtre overnight SUPPRIME
- Cap age 30min, expire 60min, risk 2%
- Modeles : `bot_v2/ml_model_*_vantage_v12.pkl` (14 actifs)

## REGLE EN MEMOIRE

`vast_saturation_rule.md` : tout run Vast = 128 workers, env BLAS=1,
verifier load proche du nb workers, decouper en >=128 taches si moins.

## ETAT MENTAL AU MOMENT DU COMPACT

- Le tick par tick **DIT LA VERITE** (cache 0 par les bougies M1)
- La V12 OOS et le M1 mentent (surestiment ~3x)
- Le LIMIT n'est pas le probleme. Le MARKET non plus. C'est plus profond.
- On debug a partir d'ici avec les hypothes 1-5 ci-dessus.
- **Ne pas perdre confiance** : on a un outil de test ultra-fidele maintenant
  (tick par tick). C'est ENORME. On peut iterer.
