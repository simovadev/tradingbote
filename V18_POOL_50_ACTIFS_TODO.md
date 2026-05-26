# V18 — Pool 50 actifs vraiment décorrélés

**Date analyse** : 2026-05-26
**Statut** : EN ATTENTE (V17 doit d'abord prouver sa valeur)

## Contexte

User a demandé : "100 actifs indépendants, pas de doublons, beaucoup de volatilité".

Vérité documentée : **les marchés mondiaux n'ont que ~25-30 dimensions de risque vraiment indépendantes**. Au-delà = doublons (corr > 0.7).

Vantage offre 1018 symboles total sur le compte 28879819. Pool optimal = **50 actifs solides** (94% de la diversification mondiale capturée, sans complexité ingérable).

## Découverte critique sur les spreads

**Première analyse foirée** : j'ai jugé les spreads en pleine session Asia (dimanche 00:48 UTC, post-réouverture).
Spread réels en session NY (13-21 UTC) = **3-10× plus serrés** que ce que j'annonçais.

Exemples chiffrés (mesure 7j de ticks) :

| Actif | Spread now (Asia) | Spread NY moyen | Facteur |
|---|---|---|---|
| GER40 | 1166 pts | **188 pts** | 6.2× |
| FRA40 | 746 pts | **72 pts** | **10×** |
| UK100 | 210 pts | 77 pts | 2.7× |
| USDMXN | 733 pts | 327 pts | 2.2× |
| USDZAR | 1946 pts | 754 pts | 2.6× |
| USDTRY | 3490 pts | 1319 pts | 2.6× |

**Implication V18** : il faut un **filtre `tradable_hours` par actif** pour ne trader qu'en session active.

## Pool 50 actifs (corrigé avec spread NY)

### Indices US (3)
- NAS100 ★ — déjà
- SP500 ★ — déjà
- DJ30 ★ — déjà

### Indices EU (3)
- GER40 ★★★ — DAX, super tradable London/NY (était à 478% spread/range en Asia, 30% en NY)
- UK100 — FTSE
- FRA40 ★★★ — CAC, OK en NY

### Indices Asia/EM (7)
- HK50 — Hang Seng
- CHINAH — China H-shares
- Nikkei225 — Japon
- BVSPX — Brésil
- INDA — India ETF
- TWINDEX — Taiwan
- SGP20 — Singapour

### FX Majors USD (5)
- EURUSD+ ★ — déjà
- GBPUSD+ ★ — déjà
- USDCHF+ ★ — déjà
- AUDUSD+ ★ — déjà
- NZDUSD+ — kiwi (nouveau)

### FX Crosses volatiles (10)
- GBPJPY+ ★★★ "the beast"
- EURJPY+
- AUDJPY+ ★★★
- CHFJPY+ ★★★
- NZDJPY+
- CADJPY+
- GBPAUD+
- EURAUD+
- EURNZD+
- GBPCHF+

### FX EM/Exotiques (4)
- USDMXN+ ★★ — peso (en session NY)
- USDPLN+ ★★ — zloty (en session NY)
- USDSGD+ ★★★
- USDZAR ★★ — rand (en session NY)

### Métaux (3)
- XAUUSD+ ★ — or, déjà
- XAGUSD — argent
- XPDUSD — palladium

### Énergie (3)
- CL-OIL — WTI
- GAS-C — gaz naturel
- GASOIL-C — gasoil

### Softs (3)
- Cotton-C
- Soybean-C
- Cocoa-C — cycle Afrique très propre

### Crypto (5)
- BTCUSD ★ — déjà
- ETHUSD
- ADAUSD
- DOTUSD
- XRPUSD

### Stocks US ultra-liquides (4)
- AMD — le + volatile (range 0.166% M1)
- TSLA
- META
- AMAZON

**TOTAL** : 50 actifs

## Filtres horaires par actif à coder

```python
TRADABLE_HOURS_UTC = {
    "GER40":      (7, 18),    # London/NY overlap
    "FRA40":      (7, 18),
    "UK100":      (7, 18),
    "USDMXN+":    (12, 21),   # NY active
    "USDZAR":     (12, 21),
    "USDPLN+":    (8, 17),    # London + early NY
    "USDTRY+":    (8, 17),
    "Coffee-C":   (10, 19),   # ICE NY hours
    "Cocoa-C":    (10, 19),
    "Sugar-C":    (10, 19),
    "Nikkei225":  ((0, 7), (22, 24)),  # Asia
    "HK50":       (1, 8),
    "CHINAH":     (1, 8),
    "TSLA":       (13, 20),   # US market open
    "AMD":        (13, 20),
    "META":       (13, 20),
    "AMAZON":     (13, 20),
    # Crypto = 24/7 OK
    # Forex majeurs/crosses = 24/5 OK
}
```

## Risques sérieux à connaître

1. **3 classes d'actifs jamais entraînées** :
   - Stocks US : gaps overnight (M1 saute 22h→13h30). Modèle pas entraîné pour ça.
   - Softs : 60% du temps M1 mort hors heures CME.
   - Crypto : 24/7, features killzones toujours = 0.

2. **Spread/range élevé** sur certains : seuil ML à augmenter par actif (0.65+ au lieu de 0.55).

3. **Données historiques** : vérifier que chaque actif a 5+ ans M1 chez Vantage. Si AMD n'a que 2 ans = modèle fragile.

4. **Compute Vast** : ~18h build (vs 5h V17) = ~$25.

5. **Maintenance** : log live 30MB/jour, 50 modèles à monitorer.

## Plan d'exécution (2 étapes)

### Étape 1 — V18 à 30 actifs (priorité après validation V17)
14 actuels + 16 nouveaux du **même type** (indices, forex, métaux). Évite stocks/softs/crypto pour ne pas mélanger classes jamais entraînées.

### Étape 2 — V19 à 50 actifs (après V18 validé)
Ajoute 7 stocks US + 3 softs + 6 cryptos + 4 derniers. Avec période d'entraînement adaptée par actif.

## Pré-requis

- V17 doit prouver sa valeur sur 24-48h de live (5-10 trades à 60%+ WR)
- Sinon : DEBUG V17 d'abord, ne pas amplifier sur un système qui ne marche pas
