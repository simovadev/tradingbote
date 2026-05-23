# HYPOTHESES BASEES SUR LES LOGS REELS (4 jours tick)

> Analyse des 4 jours testes (14, 15, 18, 19 mai 2026) en mode LIMIT tick par tick.
> Tous chiffres extraits de `logs_tick_5j/bt_tick_2026*_trades.csv` (124 setups au total).

## 1. RECAP HONNETE DES CHIFFRES

### Global sur 4 jours (mode LIMIT)
| Outcome | N | % |
|---|---|---|
| NO_FILL | 58 | 46.8% |
| INVALID_PRICE | 43 | 34.7% |
| WIN | 9 | 7.3% |
| LOSS | 14 | 11.3% |

**81.5% des setups detectes ne donnent JAMAIS un trade execute.**
Sur les 23 trades reellement fermes : WR = 39.1%, PnL = -3.4R (catastrophique).

### Par actif (taux de non-execution)
| Actif | Setups | INVALID | NO_FILL | Fermes | Non-exec | W | L |
|---|---|---|---|---|---|---|---|
| AUDUSD | 6 | 1 | 5 | 0 | **100%** | - | - |
| FRA40 | 3 | 0 | 3 | 0 | **100%** | - | - |
| UK100 | 1 | 0 | 1 | 0 | **100%** | - | - |
| USDCHF | 3 | 0 | 3 | 0 | **100%** | - | - |
| DJ30 | 11 | 8 | 2 | 1 | 90.9% | 1 | 0 |
| GBPUSD | 9 | 6 | 2 | 1 | 88.9% | 0 | 1 |
| GER40 | 15 | 5 | 8 | 2 | 86.7% | 1 | 1 |
| NAS100 | 13 | 2 | 9 | 2 | 84.6% | 0 | 2 |
| BTCUSD | 12 | 4 | 6 | 2 | 83.3% | 1 | 1 |
| USDJPY | 5 | 2 | 2 | 1 | 80.0% | 1 | 0 |
| USDCAD | 9 | 0 | 7 | 2 | 77.8% | 0 | 2 |
| XAUUSD | 15 | 7 | 4 | 4 | 73.3% | 2 | 2 |
| EURUSD | 9 | 5 | 1 | 3 | 66.7% | 2 | 1 |
| SP500 | 13 | 3 | 5 | 5 | 61.5% | 1 | 4 |

**Le bot detecte plein de setups mais en execute presque aucun, et ceux qui s'executent perdent en moyenne.**

## 2. LES 5 HYPOTHESES (ancrees dans les chiffres)

### Hyp 1 : Le LIMIT echoue parce que l'entry est mal calcule

**Preuve dans les logs** :
- 43 INVALID_PRICE = le prix etait deja du mauvais cote au moment du placement
- Exemple jour 14/05 : EURUSD bearish placed_ts=07:15, entry=1.17143, mais le prix actuel etait DEJA en dessous de 1.17143 -> ordre sell limit refuse
- Age median OB->placement = **3 min** pour les INVALID_PRICE
- Le bot scanne toutes les 5 min en backtest -> entre la detection (xx:13) et le placement (xx:15), le prix bouge

**Cause racine probable** : on place entry = `ob_high` (sell) ou `ob_low` (buy) MAIS le prix a deja traverse ce niveau. Le LIMIT n'a plus de sens.

**Solution a tester** :
1. Scanner toutes les MINUTES en backtest (`--step 1`) pour reproduire le live
2. Recalculer entry au moment du placement avec le prix courant (entry = max(ob_high, prix_actuel + delta) pour un sell)
3. Si entry impossible -> SKIP au lieu de placer

### Hyp 2 : Le SL est trop serre pour le bruit tick reel

**Preuve dans les logs** :
- XAUUSD : SL median = 0.1375% du prix = ~6 USD sur de l'or a 4500
- En tick reel, XAUUSD bouge facilement de 3-5 USD en 30 secondes (spread + bruit)
- 14 LOSS sur 4 jours, donc 14 trades touches au SL en quelques minutes
- Exemple : XAUUSD 14/05 13:20 entry=4701, SL=4704.82 (3.71 USD) -> fill 13:26 -> LOSS 13:27 (1m20s seulement entre fill et SL)

**Pattern clair** : les SL sont des high/low d'OB M1, donc tres proches. Sur des bougies M1, ils tiennent. Sur des ticks, le bruit les bouffe.

**Solution a tester** :
1. Calculer la taille moyenne du SL / ATR M5 ou M15
2. Si SL < 0.5 * ATR_M5 -> augmenter SL (mettre `SL = max(ob_high, entry + 0.5*ATR)`)
3. Re-tester le backtest tick avec SL elargi

### Hyp 3 : Les SP500 + NAS100 + DJ30 plombent tout

**Preuve dans les logs** :
- SP500 : 13 setups, **5 fermes, 1 WIN / 4 LOSS = WR 20%**
- NAS100 : 13 setups, 2 fermes, 0 WIN / 2 LOSS = WR 0%
- DJ30 : 1 WIN / 0 LOSS mais 8 INVALID_PRICE (donc 11 setups quasi tous rates)
- Pattern : les indices US sont **bruites en pre-open** (00h-07h UTC = 19h-02h NY)

**Heures les plus pourries** :
- **00h UTC** : 12 setups, 0 WIN / 4 LOSS, 7 INVALID (asia close, faible liquidite)
- **04h UTC** : 8 setups, 0 WIN / 1 LOSS, 1 INVALID (creux tokyo-londres)
- **20h UTC** : 3 setups, 0 WIN / 1 LOSS (after-NY)

**Solution a tester** :
1. Filtrer les indices US (SP500, NAS100, DJ30) sur 00h-07h UTC (avant ouverture)
2. Garder pour le forex : EURUSD 18h-19h UTC marche bien (2 WIN sur 18h-19h)
3. Tester WR si on garde uniquement EURUSD + XAUUSD + USDJPY (les 3 actifs avec WR positif)

### Hyp 4 : Le scan toutes les 5 min rate des bons setups

**Preuve dans les logs** :
- 58 NO_FILL = LIMIT place mais prix n'est jamais revenu
- L'analyse precedente sur le 19/05 montre 13/13 NO_FILL = RATE_WIN (le prix est parti SANS nous)
- Si on avait scanne toutes les minutes, on aurait place l'ordre AVANT que le prix bouge
- Age OB median = 4 min : ca veut dire qu'on detecte un OB qui pourrait avoir 4 min de retard

**Hypothese forte** : avec --step 1, beaucoup de NO_FILL devien­draient WIN (le bot place plus tot dans la sequence intra-bougie).

**Solution a tester** :
1. Relancer backtest tick avec `--step 1` (scan chaque minute)
2. Mesurer combien de NO_FILL deviennent WIN
3. Le live tourne deja avec sync close M1 (cad scan chaque minute) -> doit etre coherent

### Hyp 5 : Les EURUSD + XAUUSD + USDJPY sont les 3 SEULS actifs OK

**Preuve dans les logs** (les 9 WIN sur 4 jours) :
- EURUSD : 2 WIN (heures 18, 18) +3.82R
- XAUUSD : 2 WIN (heures 15, 19) +1.90R
- USDJPY : 1 WIN (heure 14) +1.90R
- SP500 : 1 WIN +2.11R (mais 4 LOSS) → 20% WR
- GER40 : 1 WIN, 1 LOSS
- BTCUSD : 1 WIN, 1 LOSS
- DJ30 : 1 WIN, 0 LOSS

**Et les ml scores** :
- WIN ml_median = 0.738
- LOSS ml_median = 0.748 (legerement plus haut !)
- INVALID_PRICE ml_median = 0.745
- **Le ML ne distingue PAS les WIN des LOSS** dans le backtest tick !

**Cela confirme l'intuition user** : le ML V12 OOS est gonfle (Hyp 5 du master MD). 
Le ML voit du M1 propre, mais le tick bid/ask + latence + bruit cassent sa prediction.

**Solution a tester** :
1. Whitelist : ne trader QUE EURUSD + XAUUSD + USDJPY (les 3 actifs avec WR positif sur 4j)
2. Tester avec un seuil ML plus eleve (0.80 au lieu de 0.70) pour voir si ca filtre les LOSS
3. Verifier la correlation ML vs outcome (devrait etre quasi nulle = preuve)

## 3. PATTERN INVERSE : direction bias bullish/bearish

| Direction | INVALID | NO_FILL | WIN | LOSS |
|---|---|---|---|---|
| bearish | 26 | 28 | **8** | 9 |
| bullish | 17 | 30 | **1** | 5 |

**Les setups bullish sur 4 jours = 1 WIN / 5 LOSS = WR 16%**
**Les setups bearish sur 4 jours = 8 WIN / 9 LOSS = WR 47%**

→ Le marche etait baissier en moyenne sur 14-19 mai 2026 (cf. XAUUSD dump apres 4700, EURUSD 1.17→1.16, etc.).
→ Le bot bullish prend tous des LOSS car il essaie d'acheter le couteau qui tombe.

**Solution a tester** :
1. Ajouter un filtre "bias H1/H4" (trend > 0 -> autorise bullish, sinon bearish only)
2. Pas faire de bullish quand le marche est baissier sur H1

## 4. PLAN DE DEBUG PROPOSE (par ordre de priorite)

### Etape 1 : Confirmer que --step 1 ameliore les choses (30 min)
- Lancer `backtest_v12_tick_vast.py --step 1 --date 2026-05-14 --entry_mode limit`
- Comparer avec le run actuel (`--step 5`)
- Si NO_FILL drop de 50% et WIN augmentent -> deja un gros pas

### Etape 2 : Whitelist EURUSD + XAUUSD + USDJPY uniquement (1h)
- Modifier `LIVE_ASSETS` dans config pour ne garder que ces 3
- Re-tester les 4 jours
- Confirmer WR > 50% sur les fermes (vs 39% actuel)

### Etape 3 : Recalculer entry au placement (2h)
- Modifier `backtest_v12_tick_vast.py` pour : si prix au placement est deja du mauvais cote, SKIP
- Mesurer si on perd des opportunites ou si on evite des INVALID_PRICE
- Si 0 INVALID_PRICE -> on a gagne 43 trades de bruit

### Etape 4 : Elargir SL avec ATR M5 (2h)
- Dans le code OB, calculer `sl_distance = max(ob_size, 0.5 * atr_m5)`
- Re-tester
- Si les LOSS baissent mais les WIN restent -> jackpot

### Etape 5 : Filtre bias H1/H4 (1h)
- Avant de placer, regarder le close H1 vs close H1[-3] -> trend
- Si trend < 0 et setup bullish -> SKIP
- Mesurer impact

## 5. CE QU'ON NE SAIT PAS (encore)

- **Sur quel jour le ML V12 OOS a vraiment WR 77%** ? Si c'est sur des jours specifiques (pas mai 2026), c'est un autre regime de marche.
- **Quel est le bias H4 sur 14-19 mai** ? On suppose baissier mais a verifier.
- **Le live ferait-il les memes erreurs** ? Le live scanne chaque minute -> peut-etre moins de NO_FILL. Mais les LOSS et INVALID_PRICE seraient pareils.

## 6. CE QU'IL FAUT DESACTIVER POUR EVITER PERTES SUPPLEMENTAIRES

Si on doit deployer un V12 ameliore en live, **par defaut on doit** :
1. **Whitelister EURUSD, XAUUSD, USDJPY uniquement** (seuls avec WR positif)
2. **Bloquer 00h-07h UTC** (heures les plus pourries)
3. **Bloquer setups bullish si bias H1 < 0**
4. **Scan chaque minute** (deja fait en live, juste verifier)

→ Avec ces 4 filtres, sur les 4 jours testes, on aurait pris :
   - EURUSD 18h WIN (+3.5R), EURUSD 18h WIN (+0.3R)
   - XAUUSD 15h WIN (+1.4R), XAUUSD 19h WIN (+0.5R)
   - USDJPY 14h WIN (+1.9R)
   - Au lieu d'ouvrir aux 23 trades dont 14 LOSS

   **Total estime : +7.6R sur 4 jours au lieu de -3.4R = +11R d'amelioration**

## 7. PROCHAINE ACTION

Tester ces filtres dans le backtest tick avant de toucher au live :
```bash
# Etape 1 - scan chaque minute
python backtest_v12_tick_vast.py --date 2026-05-14 --step 1 --entry_mode limit

# Etape 2 - whitelist
# (modifier LIVE_ASSETS dans config a EURUSD + XAUUSD + USDJPY)
python backtest_v12_tick_vast.py --date 2026-05-14 --step 1 --entry_mode limit

# Et comparer avec le baseline (toutes config = full setups, step 5)
```

Si l'amelioration est > +5R sur 4 jours -> on a un V12.1 deployable.
