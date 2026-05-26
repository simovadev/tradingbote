# V18 — Définitions ICT/SMC strictes (Soufiane) — VERROUILLÉES

**Date** : 2026-05-26
**Statut** : OB + BB verrouillés. Reste à définir : Sweep, Swings, FVG, MSS/BOS, Killzones, Entry, SL, TP.

---

## 1. ORDER BLOCK (OB) — ✅ VERROUILLÉ

### OB HAUSSIER

```
ÉTAPE 1 — Sweep de liquidité
   • Mèche d'une bougie descend sous un swing low récent (= liquidité prise)
   • La bougie de sweep DOIT être BAISSIÈRE (close < open)
   • Si bougie de sweep haussière → PAS d'OB (ignoré)

ÉTAPE 2 — Construction du groupe
   • Groupe OB = bougie de sweep + bougies BAISSIÈRES consécutives juste avant
   • MIN_GROUP_SIZE = 2 bougies
   • MAX_GROUP_SIZE = illimité (tant que consécutivement baissières)

ÉTAPE 3 — Bornes
   • ob_low  = LOW de la mèche de la bougie de sweep (mèche INCLUSE en bas)
   • ob_high = MAX(open, close) de toutes les bougies du groupe
              (corps uniquement, mèches du haut EXCLUES)

ÉTAPE 4 — Pending indéfini (aucun timer)

ÉTAPE 5a — VALIDATION
   • Une bougie clôture > ob_high → OB validé, tradable
   • La bougie de validation peut arriver N bougies plus tard

ÉTAPE 5b — REJECTION
   • Si pendant le pending, une bougie a son LOW < ob_low (mèche suffit)
     → OB REJECTED À VIE
```

### OB BAISSIER (symétrique strict)

```
ÉTAPE 1 — Sweep d'un swing high (mèche dépasse un swing high récent)
   • Bougie de sweep DOIT être HAUSSIÈRE

ÉTAPE 2 — Groupe = bougie sweep + bougies HAUSSIÈRES consécutives (min 2, max ∞)

ÉTAPE 3 — Bornes
   • ob_high = HIGH de la mèche du sweep (mèche INCLUSE en haut)
   • ob_low  = MIN(open, close) du groupe (mèches du bas EXCLUES)

ÉTAPE 4 — Pending indéfini

ÉTAPE 5a — VALIDATION : close < ob_low → tradable
ÉTAPE 5b — REJECTION : mèche > ob_high → mort à vie
```

### Récap bornes (symétrie)

| | OB bullish | OB bearish |
|---|---|---|
| Côté sweep | `ob_low` = mèche sweep (INCLUSE en bas) | `ob_high` = mèche sweep (INCLUSE en haut) |
| Côté opposé | `ob_high` = MAX corps groupe (mèches hautes EXCLUES) | `ob_low` = MIN corps groupe (mèches basses EXCLUES) |

---

## 2. BREAKER BLOCK (BB) — ✅ VERROUILLÉ

### BB BULLISH (à partir d'un OB bearish cassé)

```
ÉTAPE 1 — Un OB bearish existe avec bornes [ob_low ; ob_high]
ÉTAPE 2 — Une bougie clôture > ob_high (cassure)
ÉTAPE 3 — L'OB bearish cassé devient un BB BULLISH
          • bornes héritées = [ob_low ; ob_high] de l'OB cassé
          • zone ex-résistance → support potentiel

ÉTAPE 4 — Tradable au retour dans la zone

ÉTAPE 5 — RÉINVALIDATION
   • Si une bougie a son LOW < BB_low (mèche suffit)
     → BB MORT À VIE (même règle que l'OB)
```

### BB BEARISH (symétrique)

```
À partir d'un OB bullish cassé (close < ob_low).
BB bearish hérite des bornes de l'OB cassé.
Meurt si mèche > BB_high.
```

---

## 3. SWEEP / LIQUIDITY GRAB — ⏳ À DÉFINIR

À clarifier :
- Critère exact de "mèche dépasse swing low/high"
- Bougie de sweep doit-elle clôturer dans le sens contraire (rejet) ou pas ?
- Ancienneté max du swing à sweep
- Distance min entre swing et bougie de sweep

---

## 4. SWINGS — ⏳ À DÉFINIR

À clarifier :
- Strength = 2, 3, ou différent par actif ?
- Égalité stricte (`<`) ou tolérée (`≤`) ?

---

## 5. FAIR VALUE GAP (FVG) — ⏳ À DÉFINIR

À clarifier :
- Définition 3-bougies standard ICT (low C3 > high C1 pour bullish) ?
- Direction implicite oui/non ?
- Quand mitigated (50% / 100% / touch) ?
- Tradable directement ou juste confluence ML ?

---

## 6. MSS / BOS — ⏳ À DÉFINIR

À clarifier :
- MSS vs BOS différents ou pareil ?
- Cassure par close ou mèche ?
- Quel swing high/low doit être cassé ?
- Rôle dans l'entrée (confirmation obligatoire ou feature ML) ?

---

## 7. KILLZONES — ⏳ À DÉFINIR

Heures actuelles V17 (à valider) :
- Asia : 00-04 UTC
- London : 07-11 UTC
- NY_AM : 13-16 UTC
- NY_Lunch : 16-17 UTC
- NY_PM : 17-20 UTC
- London_Close : 14-16 UTC

---

## 8. ENTRY — ⏳ À DÉFINIR

À clarifier :
- Touch / 50% / ob_low / MSS confirmé ?
- Limit ou Market ?
- Délai max après validation OB ?

---

## 9. STOP LOSS — ⏳ À DÉFINIR

Problème V17 : 70% des LOSS sortent en <10min → SL trop serré.

À clarifier :
- Sous ob_low / sous swing sweepé / ATR ?
- Marge fixe (pips/points/spread) ?
- Min SL ?

---

## 10. TAKE PROFIT — ⏳ À DÉFINIR

À clarifier :
- RR fixe (=2) / liquidité opposée / partiels ?
- RR min pour valider setup ?
- Trailing après 1er TP ?

---

## Plan code V18 (à exécuter)

1. ✅ OB strict (cette définition) — fichier `bot_v2/concepts/order_block_v18.py`
2. ✅ BB hérité d'OB cassé — fichier `bot_v2/concepts/breaker_block_v18.py`
3. ⏳ Sweep, swings, FVG, MSS, killzones, entry, SL, TP — DÉFINITIONS MANQUANTES
