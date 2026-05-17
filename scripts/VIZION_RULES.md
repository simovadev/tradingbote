# 📖 Règles Vizion FR extraites des transcriptions

Source : 34 vidéos transcrites de la playlist Vizion FR (les 24 autres en cours via Whisper).

---

## 🎯 LA CHECKLIST DE BASE (vidéo bible BKGoNf6vhRY)

### Pour prendre un trade, 4 étapes OBLIGATOIRES dans cet ordre :

#### Étape 1 — Crédibilité de l'OB (sur le TF d'entrée, ex: M5)
- L'OB **pousse le prix dans une zone de liquidité externe** (sweep d'un swing low/high)
- L'OB se forme en **zone Discount** (LONG, sous équilibrium 50%) ou **Premium** (SHORT, au-dessus)
- Le prix **rebalance les FVG** trouvés sur le chemin de validation de l'OB
- Optionnel : volume imbalance rebalancé

#### Étape 2 — Imbrication TF (Time Frame Alignment)
- Si trade M5 → vérifier H1
- L'**OB M5 doit être contenu dans un OB H1** de même direction
- L'**OB H1 doit avoir validé des FVG** sur son chemin

#### Étape 3 — Confirmation SMT Divergence
- Sur l'actif corrélé (EURUSD ↔ GBPUSD, XAU ↔ XAG, NAS ↔ SPX, etc.)
- L'actif primaire prend la liquidité, le corrélé NON → divergence confirmée
- C'est la "signature algo" institutionnelle

#### Étape 4 — Tenue (hold) après entrée
- Chercher Breaker Block validé
- FVG respectés (rebalancés)
- Nouveaux OB qui se forment dans le sens du trade

---

## 🕐 KILLZONES PRÉCISES (vidéo EHQzmAML6VI)

**IMPORTANT : Vizion utilise l'heure de NEW YORK comme référence, pas UTC !**

### Killzones Forex
- **London** : commence à 7h NY
- **NY AM** : 7h NY
- **NY Lunch** : pause médiane
- **NY PM** : après-midi NY
- **Asia** : nuit (Vizion ne trade pas Asia)

### Killzones Indices (plus précises)
- **8h30 NY** : ouverture pré-marché US
- **9h30 NY** : ouverture cash market US
- Le reste pareil que Forex

### Règle d'or : **Time + Price**
- OB qui se valide PENDANT killzone = **high probability**
- OB hors killzone = **trade déchet** (poubelle)
- Le **temps** valide le **prix** = signature algo

### Convertir NY → UTC (selon heure d'été/hiver)
- Heure d'hiver (Nov-Mar) : NY+5 = UTC. Donc 7h NY = 12h UTC
- Heure d'été (Mar-Nov) : NY+4 = UTC. Donc 7h NY = 11h UTC

---

## 📅 WEEKLY PROFILE - "Classic Expansion" (vidéo 0pjj5rM1M18)

Le plus fréquent et le plus puissant.

### Structure type :
- **Lundi** : hésitation (doji) ou faux mouvement contraire
- **Mardi** : reversal + prise de PDL/PDH du lundi → début expansion
- **Mardi → Jeudi** : expansion (= phase où on trade)
- **Vendredi** : profit-taking / clôture

### Application
- On cherche les setups **mardi-mercredi-jeudi**
- Lundi : on observe, on ne trade pas
- Vendredi : on évite les nouvelles positions

---

## 🔍 DAILY BIAS AVANCÉ (vidéo y2Fwp4T9sRM)

Sur la clôture daily, pour anticiper le lendemain :

1. **PDH ou PDL pris ?**
   - PDH pris + bougie haussière → bias bullish, direction = nouveau PDH
   - PDL pris + bougie baissière → bias bearish, direction = nouveau PDL
   - Aucun des deux → attendre

2. **OB daily validé ?**
   - On attend un pullback DANS cet OB pour entrer en LTF

3. **FVG (busy/sebi) à combler ?**
   - Le FVG agit comme un magnet → cible probable

---

## 📊 CONCEPTS ICT À CODER

### Définitions exactes Vizion

| Concept | Définition Vizion |
|---------|-------------------|
| **OB** | "Bougie/bloc qui pousse le prix dans une zone de liquidité" - fin de manipulation |
| **Liquidité externe** | Swing High (BSL) ou Swing Low (SSL) |
| **Liquidité interne** | FVG, Volume Imbalance, Inefficiency |
| **Discount** | Sous l'équilibrium (50%) du dernier range |
| **Premium** | Au-dessus de l'équilibrium |
| **Équilibrium** | 50% du dernier range/leg directionnelle |
| **PDR** (Pre-Determined Range) | Order Block, FVG, Breaker, etc. |
| **Breaker Block** | Ancien OB invalidé qui devient pertinent au retest |
| **Mèche puissante** | Petit corps + grande mèche = volatilité directionnelle révélée |
| **PDH/PDL** | Previous Day High/Low |
| **Time + Price** | Concept ICT qui se valide pendant killzone = haute proba |
| **Trade déchet** | OB sans HTF alignment + sans SMT + hors killzone |
| **AMD** | Accumulation - Manipulation - Distribution (cycle journalier/hebdo) |
| **Classic Expansion** | Weekly profile : Lun=hésit, Mar=reversal, Mar-Jeu=expansion |

---

## 🚨 CE QUI MANQUE DANS MON BOT vs VIZION

### Déjà fait ✓
- ✓ Sweep liquidité externe (swing H/L)
- ✓ Discount/Premium check
- ✓ SMT divergence (XAG, DXY, SPX, UKOIL)
- ✓ Killzones (mais à recalibrer en heure NY)
- ✓ FVG M15 confluence
- ✓ Multi-actif

### Manque ❌
1. **Imbrication TF stricte** : OB M5 doit être DANS un OB H1 (pas juste "M5 aligned with H1 bias")
2. **Rebalance FVG sur chemin de validation OB** : vérifier que le prix a comblé les FVG entre le sweep et la validation OB
3. **Daily bias avancé** : analyser la bougie daily précédente (PDH/PDL pris, OB daily, FVG daily)
4. **Weekly Profile Classic Expansion** : filtrer le lundi (pas de trade), favoriser mardi-jeudi
5. **Killzones en heure NY** (avec gestion été/hiver)
6. **Killzones différentes Forex vs Indices** (8h30 et 9h30 NY pour indices)
7. **Calendrier économique** : éviter trade pendant news (FOMC, NFP, CPI)
8. **Concept "Time + Price"** : OB DOIT se valider pendant killzone, sinon poubelle
9. **Bougie mèche puissante** : détecter petit corps + grande mèche comme signal d'entrée LTF
10. **Volume Imbalance** : pas que FVG, aussi les volume gaps

---

## 🛠 PRIORITÉS DE CODAGE

### Priorité 1 (le plus impactant)
1. **Imbrication TF stricte** : pour chaque OB M5, vérifier l'existence d'un OB H1 qui le contient
2. **Daily bias avancé** : analyser la bougie daily de la veille pour valider la direction du trade
3. **Killzones en heure NY** : recalibrer avec gestion DST (Daylight Saving Time)

### Priorité 2
4. **Weekly Profile filter** : pondérer le score selon jour de semaine + structure weekly
5. **Calendrier économique blackout** : skip ±30min autour des annonces majeures
6. **Rebalance FVG check** : valider que les FVG sont comblés sur le chemin OB

### Priorité 3
7. Mèches puissantes
8. Volume Imbalance détection
