# VIZION BIBLE — Methodologie Trading FR complete

> Source : 58 videos transcrites de la playlist Vizion FR (chaine YouTube Vizion Trading France).
> Objectif : reference exhaustive pour coder un bot qui suit EXACTEMENT cette methodologie ICT/SMC francaise.
> Toute regle est rattachee a une ou plusieurs videos sources (ID YouTube entre parentheses).

---

## 0. TL;DR — Le bot en une page

Le bot doit reproduire la chaine de decision suivante, du HAUT vers le BAS :

1. **Bias Weekly** (echelle weekly) — clair ou neutre (BKGoNf6vhRY, UU4MZRT4324, 0pjj5rM1M18).
   - Si neutre : on ne trade pas l'actif cette semaine.
2. **Weekly Profile** identifie (0pjj5rM1M18) : Classic Expansion / Midweek Reversal / Consolidation Reversal. Pas de profil clair => ne pas trader.
3. **Bias Daily** etabli sur la bougie daily de la veille (y2Fwp4T9sRM) : PDH/PDL pris, OB daily valide, FVG daily, breaker daily.
4. ~~Calendrier économique~~ — **ignoré en V1** (décision user 2026-05-15). Le bot trade pendant les news.
5. **Killzone active** (heure NY) (EHQzmAML6VI, TDYspBnIZOw) : London, NY AM (9h30 indices, fixe), NY Lunch, NY PM. Hors KZ = trash.
6. **Time Frame Alignment strict** :
   - Analyse Daily => setup H1
   - Analyse H1 => setup 5min
   - Analyse 15min => setup 1min
   - Pas de saut d'echelle (BKGoNf6vhRY, EHQzmAML6VI, 6IN5CNs5qrc, AE0K6W9uiSY, ZlFI7LgOrw0).
7. **Structure HTF validee** : OB HTF qui prend une liquidite externe pertinente (swing H/L time-based) + rebalance d'un FVG sur le chemin de validation (BKGoNf6vhRY).
8. **Zone Discount/Premium** : OB long en zone Discount, OB short en zone Premium (eS6adgOouqI, BKGoNf6vhRY).
9. **SMT Divergence** (BONUS, pas obligatoire — décision user 2026-05-15) sur paire correlee (uOv1znt2uAY, 0dzjqeTDXxU). Améliore le score si présente.
10. **PO3 / AMD** : Accumulation -> Manipulation -> Distribution. On trade la Distribution. Utilisable sur M5 comme signal (décision user 2026-05-15) en plus de Daily/H4/H1 (Pm29OIifOns, SBZvpF3FK2w, AE0K6W9uiSY).
11. **Entree LTF** : OB LTF (= 5min si daily/H1) valide PENDANT killzone, apres prise d'un low/high time-based, OB se forme avec displacement (FVG), idealement breaker block (setup Unicorn) (tNgYKSK3Vz0, BKGoNf6vhRY).
12. **Stop Loss** : **MÈCHE TOUJOURS** (OB, breaker, FVG). Décision user 2026-05-15. Sous le low pour long, au-dessus du high pour short (tNgYKSK3Vz0, 5fv2MjuPKE4).
13. **Take Profit** : prochain swing H/L pertinent (swing en HTF, pas LTF) (sjQW8l-Vi_s), ou Standard Deviation (tKEoXpb1FlE), RR min 1.5 (8DQz3QlWB5Q), RR cible 2-6 (BKGoNf6vhRY).
14. **Tenue** : breakers valides, FVG respectes, nouveaux OB dans le sens du trade (BKGoNf6vhRY, ZlFI7LgOrw0).
15. **Filtre trade-dechet** : pas de signature time+price => poubelle. Pas d'imbrication TF => poubelle (BKGoNf6vhRY, EHQzmAML6VI, 8DQz3QlWB5Q, eS6adgOouqI).

**Resume en une phrase :** une signature = bias + timing (killzone) + pidur (OB/FVG/Breaker) imbrique en HTF, valide par une prise de liquidite externe time-based, optionnellement confirme par SMT et un PO3 de la bougie HTF (ZlFI7LgOrw0).

---

## 1. Concepts fondamentaux — vocabulaire Vizion

| Terme | Definition Vizion (sources) |
|---|---|
| **OB (Order Block)** | "Bougie inverse qui pousse le prix dans une zone de liquidite". OB bullish = bougie(s) baissiere(s) qui pousse(nt) le prix vers le bas pour prendre une liquidite, puis cloture(nt) au-dessus de l'origine. OB bearish = symetrique haussier. PAS d'OB sans prise de liquidite. (5fv2MjuPKE4, BKGoNf6vhRY, EPDZiywdUyw, ZlFI7LgOrw0) |
| **OB high probability** | OB qui se valide PENDANT une killzone, apres prise d'un swing low/high time-based, en zone Discount (long) ou Premium (short), avec rebalance d'un FVG sur le chemin (BKGoNf6vhRY). |
| **PDR (Pre-Determined Reaction)** | Terme generique chez Vizion pour tout concept ICT pouvant generer une reaction du prix : OB, FVG, Breaker, IFVG, BPR, divergence SMT, etc. (ZlFI7LgOrw0). |
| **PIDR / pidiret / pidur** | Variante orthographique de PDR dans les transcriptions Whisper. Synonyme stricte. |
| **FVG (Fair Value Gap)** | Imbalance sur 3 bougies consecutives ; le corps de la bougie centrale forme une "fenetre" non couverte par les meches des bougies adjacentes. Traduit un desequilibre offre/demande, donc la presence d'un programme algo (TrTBuk2Ttl4). |
| **Busy (bullish FVG)** | FVG haussier. Lecture Whisper : "biz", "bizy", "bsy". Tous les memes. |
| **CBI (bearish FVG)** | FVG baissier (couleur souvent affichee differemment par Vizion). |
| **IFVG (Inverse FVG)** | FVG qui a ete traverse en CLOTURE (corps de bougie, pas meche) dans l'autre sens. Un CBI inverse devient un IFVG bullish ; un Busy inverse devient un IFVG bearish (TrTBuk2Ttl4). Signature d'inversion. |
| **BPR (Balance Price Range)** | Fusion d'un CBI et d'un Busy au meme niveau. BPR bullish = le dernier des deux est un Busy. BPR bearish = le dernier est un CBI. Confluence supplementaire (TrTBuk2Ttl4). |
| **Volume Imbalance** | "Trou de cotation" entre la cloture d'une bougie et l'ouverture de la suivante (gap intra-bougie). Doit etre rebalance comme un FVG (BKGoNf6vhRY). |
| **BSL (Buyside Liquidity)** | Liquidite au-dessus d'un swing high ou equal high : stops des shorts. |
| **SSL (Sellside Liquidity)** | Liquidite sous un swing low / equal low : stops des longs. |
| **Liquidite externe** | Swing high/low (BSL/SSL). Cherchee comme objectif et comme signature de manipulation (5fv2MjuPKE4, BKGoNf6vhRY). |
| **Liquidite interne** | FVG, Volume Imbalance, inefficience (BKGoNf6vhRY). |
| **Equal High / Equal Low** | Deux high (ou deux low) quasi au meme niveau ; magnet a liquidite. "Relative equal" quand le prix arrive +- quelques pips pres (uOv1znt2uAY). |
| **PDH / PDL** | Previous Daily High / Low. Toujours mis a jour bougie daily par bougie daily (y2Fwp4T9sRM). |
| **PWH / PWL** | Previous Weekly High / Low. Memes regles, sur l'echelle weekly (uOv1znt2uAY). |
| **Discount** | Sous l'equilibrium (50% Fibonacci) du dernier mouvement directionnel. Zone d'achat (eS6adgOouqI). |
| **Premium** | Au-dessus de l'equilibrium. Zone de vente. |
| **Equilibrium** | 50% Fibo du dernier swing low->swing high (ou inverse). Frontiere Discount/Premium. |
| **BOS (Break of Structure)** | Cassure d'un swing dans le sens de la tendance en cours. Continuation. |
| **MSS (Market Structure Shift)** | Schema "swing-low entre 2 swing-highs" (ou inverse) traverse en CORPS de bougie. Annonce un changement de tendance (7q1cQyvvQKI, 3RAGDN6TfTA, EPDZiywdUyw). |
| **SISD / Sised (Change in State of Delivery)** | Un OB qui acte un changement de PHASE de marche (retracement -> expansion, ou expansion -> reversal). Plus precoce que le MSS le plus souvent (7q1cQyvvQKI). |
| **Breaker Block** | Ancien OB qui a ete invalide (le prix a clos de l'autre cote). Si l'OB etait bearish et est traverse a la hausse => breaker bullish. Stop loss sous la MECHE du breaker (tNgYKSK3Vz0, ZlFI7LgOrw0). |
| **Stop Hunt** | Apres une prise de liquidite, une deuxieme baisse/hausse vient chercher les stops des early participants. Compose le setup Unicorn (tNgYKSK3Vz0). |
| **SMT Divergence (Smart Money Technique)** | Sur 2 actifs correles : l'un prend une liquidite que l'autre ne prend pas. Signature algorithmique. SMT bullish = divergence sur des lows ; SMT bearish = sur des highs. VALIDEE seulement quand un OB se valide (sinon "en cours") (0dzjqeTDXxU, BKGoNf6vhRY, uOv1znt2uAY). |
| **Killzone (KZ)** | Plage horaire NY ou les algos sont actifs. Box de couleurs sur TradingView via l'indicateur "ICT Killzones and Pivots" de TFO (TDYspBnIZOw, EHQzmAML6VI). |
| **Time + Price** | Concept central ICT chez Vizion : un evenement (OB, FVG, prise de liquidite) qui se produit DANS une killzone EST une signature. Hors KZ = sans valeur (EHQzmAML6VI, ZlFI7LgOrw0, 8DQz3QlWB5Q). |
| **AMD** | Accumulation - Manipulation - Distribution. Cycle algorithmique fractale (Pm29OIifOns, AE0K6W9uiSY, SBZvpF3FK2w). |
| **PO3 (Power of Three)** | Lecture du couple corps/meche d'une bougie HTF pour anticiper son sens (ouverture, prise de liquidite via meche, distribution, cloture). OLHC = bougie bullish ; OHLC = bougie bearish. Utilise sur D1, H4, H1 ; PAS sous H1 (SBZvpF3FK2w, EHQzmAML6VI). |
| **4 phases de marche** | Consolidation, Retracement, Expansion, Reversal. On trade UNIQUEMENT l'Expansion et le Reversal (Pm29OIifOns). |
| **Daily Profile** | Profil intra-journee (ex. London Reversal, NY Manipulation, NY Reversal) (XQ6A6w3brwU, 0pjj5rM1M18). |
| **Weekly Profile** | Classic Expansion / Midweek Reversal / Consolidation Reversal. Determine quel jour trader (0pjj5rM1M18). |
| **TGIF** | "Thank God It's Friday". Retracement du vendredi apres un Classic Expansion. Objectif = 20-30% du weekly range (0pjj5rM1M18). |
| **POI** | Point of Interest = high/low de killzone passee (TDYspBnIZOw). |
| **Setup Unicorn** | OB + Stop Hunt + FVG + Breaker. Entree sur le BREAKER block (tNgYKSK3Vz0). |
| **Trade dechet** | Tout trade pris sans signature time+price, hors killzone, sans TF alignment, sans bias HTF. A bannir (BKGoNf6vhRY, 8DQz3QlWB5Q). |
| **Narrative** | Histoire racontee par les PDR. Permet d'anticiper la page suivante du graphique (ZlFI7LgOrw0). |
| **Signature** | bias + timing + pidur. Tout trade pris doit etre une signature complete (ZlFI7LgOrw0). |

---

## 2. Order Block (OB) — Definition stricte Vizion

### 2.1 Definition (5fv2MjuPKE4, BKGoNf6vhRY, EPDZiywdUyw)

> **Un Order Block bullish est une bougie BAISSIERE (ou un groupe de bougies BAISSIERES consecutives) qui pousse le prix dans une zone de liquidite, puis le marche clos AU-DESSUS du HIGH de cette/ces bougie(s) baissiere(s).**
> Symetrique pour OB bearish.

### 2.2 Conditions de validite (5fv2MjuPKE4)

1. **Prise de liquidite obligatoire** : "Pas de manipulation, pas d'OB". L'OB doit faire prendre un SSL (pour OB bullish) ou un BSL (pour OB bearish). Sans cela, c'est seulement "une formation de bougies", pas un OB (EPDZiywdUyw).
2. **Bougies consecutives** : pour un OB bullish multi-bougies, toutes les bougies doivent etre baissieres (pas de bougie haussiere entre elles). La derniere bougie qui prend la liquidite doit etre baissiere (pour OB bullish) ; sinon ce n'est pas techniquement un OB (a verifier sur l'echelle superieure) (EPDZiywdUyw).
3. **Validation = cloture** : l'OB est valide quand une bougie cloture AU-DESSUS de l'origine (high de la bougie d'origine) de l'OB. Vizion prefere CORPS DE BOUGIE plutot que meches => +1 securite (5fv2MjuPKE4).
4. **Pas de bougie 1m / 2m** : Vizion utilise OB H1, 15min, 5min principalement ; les TF plus basses generent trop de faux signaux (5fv2MjuPKE4).

### 2.3 Pertinence (signature) (BKGoNf6vhRY, ZlFI7LgOrw0)

Un OB **doit marcher** seulement si :
- Il a pris une liquidite externe pertinente (swing low/high TF approprie ou high/low de killzone passee).
- Il se valide **PENDANT** une killzone (Time+Price) : London, NY AM (8h30/9h30 indices), NY PM.
- Il se forme en **zone Discount** (OB bullish) ou **Premium** (OB bearish) du dernier mouvement directionnel.
- Le prix **rebalance** au moins un FVG sur le chemin de validation.
- Optionnel : Volume Imbalance rebalance.
- Optionnel : SMT divergence sur l'actif correle.

### 2.4 Invalidation / Breaker (5fv2MjuPKE4, tNgYKSK3Vz0)

- Si le prix CLOS sous le low de l'OB bullish, l'OB est invalide.
- Cet OB invalide peut devenir un **Breaker** (voir section 16).
- Pour un OB bearish : si le prix CLOS au-dessus du high, l'OB est invalide.

### 2.5 OB par echelle de temps (5fv2MjuPKE4, BKGoNf6vhRY, rdeAjnVdfRM)

| TF analyse | TF setup |
|---|---|
| Weekly | H4 |
| Daily | H1 |
| H4 | 15min |
| H1 | 5min |
| 15min | 1min |

**Regle fractale** : un OB LTF (ex. 5min) qui est CONTENU dans un OB HTF de meme direction (H1) est high probability. C'est le **time frame alignment strict Vizion** (BKGoNf6vhRY).

**Methode "entonnoir" Vizion (rdeAjnVdfRM)** : on procede HTF -> LTF en exigeant **un feu vert HTF avant de descendre** :
1. **Verifier H1** : OB H1 valide dans la direction du bias ? Si NON => ne pas descendre en 5min.
2. Si OB H1 valide => descendre en **5min**. Si pas de setup 5min => attendre, ne pas sauter en 1min.
3. Si OB 5min valide en + de l'OB H1 => descendre en **1min** seulement pour affiner l'entree.

> "Time Frame Alignement = il faut TOUJOURS un feu vert en HTF pour se permettre de descendre. Sans feu vert, on ne descend pas." (rdeAjnVdfRM)

**Regle anti-FOMO (rdeAjnVdfRM)** : meme si on voit une bougie 1min haussiere a 9h30 NY qui correspond exactement a notre bias depuis la veille, **on ne rentre PAS** si :
- Pas de structure HTF (15min, H1) validee juste avant.
- RR < 1.5 (si SL sous le FVG 1min et TP au BSL, RR souvent < 1 => no trade).

**Phase de marche autorisee pour descendre (rdeAjnVdfRM)** :
- **Expansion** : on trade.
- **Retracement** : on attend, c'est l'opportunite d'entrer en suivant (busy rebalance).
- **Reversal** : signature contre, on ne trade plus dans le sens initial.
- **Consolidation** : on attend la manipulation puis le SISD (AMD).

### 2.6 Mode d'utilisation (5fv2MjuPKE4)

1. **Identifier la fleur de l'OB** : ensemble de signatures (FVG, breakers...) en faveur du OB.
2. **Niveau a proteger** : low de l'OB bullish = niveau de stop loss naturel.
3. **Identifier l'entree** : retest de l'OB dans une phase d'accumulation au sein du mouvement.

---

## 3. FVG, IFVG, Volume Imbalance, Liquidites internes

### 3.1 FVG — Definition (TrTBuk2Ttl4, NK6XSOZSPac)

3 bougies consecutives. La bougie centrale forme un "espace" non couvert par les meches des 2 bougies adjacentes. Cet espace = **imbalance** = inefficience du price action.

- **Busy** = FVG bullish (vert pour Vizion). Acronyme : **B**uyside imbalance + **S**ellside inefficiency. Traduit un prix qui s'oriente a la hausse (NK6XSOZSPac).
- **CBI** = FVG bearish (couleur differente). Acronyme : **C**ellside imbalance + **B**uyside inefficiency. Traduit un prix qui s'oriente a la baisse (NK6XSOZSPac).

**Tracage Vizion strict (NK6XSOZSPac)** : on utilise les **meches** des bougies 1 et 3 pour delimiter le FVG, PAS les corps. Le FVG est par nature la zone que le prix n'a PAS cherchee entre la bougie 1 et la bougie 3.

**Pas un gap de cotation** : un FVG est un gap intra-bougie (sur la bougie centrale), pas un trou de cotation entre 2 bougies (qui s'appelle Volume Imbalance).

### 3.2 Quand un FVG marche-t-il ? (TrTBuk2Ttl4)

3 situations principales d'utilisation chez Vizion :

1. **FVG couple a un OB** : OB valide + Busy/CBI forme sur le chemin de validation (= sur la jambe de displacement).
2. **FVG en zone Discount (long) ou Premium (short)** : ignorer les FVG en milieu de mouvement, attendre un FVG dans la zone Disc/Prem.
3. **FVG + SMT Divergence** : confluence multi-actifs.

En dehors de ces 3 cas, Vizion **ignore le FVG** (TrTBuk2Ttl4).

### 3.3 Rebalance vs invalidation

- **Rebalance** = le prix vient TOUCHER le FVG (idealement le corps). Le FVG reste valide.
- **Invalidation** = le prix CLOS de l'autre cote du FVG en corps de bougie. Devient IFVG.

**Les 3 niveaux d'utilisation du FVG (NK6XSOZSPac)** : on se concentre sur les CORPS de bougie a l'interieur du FVG (les meches peuvent depasser). Tant qu'un corps respecte l'un de ces niveaux, le FVG reste valide :

1. **Entree du FVG (haut du gap pour un Busy)** : le prix s'arrete pile a l'entree => signe le plus fort. "Le prix n'avait pas besoin de descendre plus bas pour chercher davantage de liquidite" => forte volonte algo.
2. **Mi-FVG** : le prix entre jusqu'a la moitie du FVG. Ce niveau s'appelle le **Mi-Busy** ou **Mi-Cibi** (ligne au 50% du gap).
3. **Extremite (fond du gap)** : le FVG est "filled" (rempli en anglais). C'est le seuil ultime — au-dela en corps de bougie, le FVG devient IFVG.

**Les 3 placements de Stop Loss possibles pour un FVG (NK6XSOZSPac)** — du plus risque au moins risque :
- (Plus risque) Sous l'extremite **en meche** de la bougie 2 du FVG (la bougie centrale ou est le gap).
- Sous l'extremite **en corps** de la bougie 1 (la bougie d'origine des 3 bougies).
- (Moins risque) Sous l'extremite **en meche** de la bougie 1.

> "Mettre le SL sous la meche de la bougie 1 est la moins risquee des possibilites." (NK6XSOZSPac)

### 3.4 IFVG (Inversed FVG) (TrTBuk2Ttl4)

- Quand un FVG est invalide en CORPS de bougie, il devient IFVG.
- IFVG bullish = ancien CBI inverse a la hausse.
- IFVG bearish = ancien Busy inverse a la baisse.
- L'IFVG est une **signature d'inversion** : si on est bullish, un Busy qui s'inverse en IFVG bearish est mauvais signe ; un CBI qui s'inverse en IFVG bullish est bon signe.
- L'IFVG sert souvent de zone d'entree apres rebalance.

### 3.5 BPR (Balance Price Range) (TrTBuk2Ttl4)

Quand un CBI puis un Busy se forment au meme niveau de prix (ou inversement) :
- **BPR bullish** : le dernier des deux est un Busy.
- **BPR bearish** : le dernier est un CBI.
- Utilise en confluence apres l'inversion d'un PDR.

### 3.6 Volume Imbalance (BKGoNf6vhRY)

Gap entre la cloture d'une bougie et l'ouverture de la suivante (trou de cotation). Doit etre rebalance pour completer le narrative.

### 3.7 Triple Time Frame Alignment avec FVG (TrTBuk2Ttl4)

Exemple : FVG bullish H4 rebalance => descendre 15min, attendre OB bullish valide => descendre 1min pour chercher le setup.

---

## 4. Liquidites externes : BSL / SSL / equal H-L / swing pertinent

### 4.1 Definitions (BKGoNf6vhRY, sjQW8l-Vi_s)

- **Swing high** = point haut entre 3 bougies (la bougie centrale a un high superieur a la bougie precedente et a la suivante).
- **Swing low** = symetrique.
- **BSL** = liquidite au-dessus des swing highs (stops des shorts).
- **SSL** = liquidite sous les swing lows.

### 4.2 Quel swing est PERTINENT ? (sjQW8l-Vi_s)

**Regle Vizion** : pour qualifier un swing, **monter d'un cran dans le TF alignment**.

| TF du swing | TF de validation |
|---|---|
| 5min | H1 |
| 1min | 15min |
| H1 | Daily |
| H4 | Weekly |

Un swing 5min est pertinent SEULEMENT si il correspond a un swing visible sur le H1. Sinon c'est du "bruit" qu'on filtre.

### 4.3 Niveaux time-based prioritaires (TDYspBnIZOw, EHQzmAML6VI)

Toujours regarder les highs/lows des KILLZONES PASSEES :
- Low/high de la **killzone asiatique** (precedente, hors trading direct).
- Low/high de la **killzone Londres**.
- Low/high de la **killzone NY AM**.
- Low/high de la **killzone NY PM** (de la veille pour les sessions du matin).

**PDH / PDL** et **PWH / PWL** sont aussi prioritaires (y2Fwp4T9sRM, uOv1znt2uAY).

### 4.4 Sweep / Manipulation

Lorsque le prix prend un swing time-based puis valide un OB dans la foulee, **la prise de ce swing devient la phase de manipulation** d'une AMD (en cours ou complete).

### 4.5 Equal Highs / Equal Lows (uOv1znt2uAY, Ry0mCKpAFD0)

Deux ou plusieurs highs ou lows au meme niveau (ou "relative equal" si quelques pips d'ecart) = aimant a liquidite. Cible TP de premier choix.

**Specifite Vizion : IQH / IQL (Ry0mCKpAFD0)** :
- **IQH (Inducement Quality High)** : niveau de prix touche **plusieurs fois en tant que resistance** (3+ points de contact). Pour le retail = zone de resistance ou shorter (avec SL au-dessus). Pour Vizion = cible de prise de liquidite par les algos.
- **IQL (Inducement Quality Low)** : symetrique, **plusieurs touches en tant que support**.

**Mecanique du piege (Ry0mCKpAFD0)** : les algos font CROIRE que le niveau marche (le prix part dans le sens souhaite par le retail) avant de venir prendre les SL au-dessus/sous le niveau. Donc **avant de prendre la liquidite IQH, le prix donne souvent un faux signal dans le sens du retail**.

**Hierarchie des cibles TP Vizion (Ry0mCKpAFD0)** :
1. **PDH / PDL** : prioritaire si bias daily clair (acheter sur PDL en direction de PDH).
2. **PWH / PWL** : cibles weekly pertinentes.
3. **IQH / IQL** : zones touchees plusieurs fois.
4. **Swing high / low HTF** : cible de fond.
5. **Equal high / low** : aimants intermediaires.

**Structure ideale d'entree (Ry0mCKpAFD0)** : si bias daily bullish, attendre **manipulation sur PDL** puis **distribution vers PDH**. Le couple PDL (entree) -> PDH (target) = setup high probability.

### 4.6 Mecanique des algorithmes et liquidites (QPQWlXQ-El4)

**3 types d'acteurs sur le marche (QPQWlXQ-El4)** :
- **Gros ordres institutionnels** : acheteurs/vendeurs **PASSIFS**. Font bouger le prix par leur masse.
- **Petits ordres retail** : acheteurs **ACTIFS**. N'ont pas l'impact pour bouger le prix.
- **Algorithmes des institutions** : font bouger le prix DE LIQUIDITE EN LIQUIDITE pour atteindre le niveau auquel les institutionnels veulent acheter/vendre.

**Pourquoi les algos cherchent les liquidites (QPQWlXQ-El4)** : pour faire bouger le prix, les algos puisent dans les ordres deja places (SL des retails) ou les ordres en attente (buy stops, sell stops).

**Mecanique news (QPQWlXQ-El4)** : a la publication d'une news (CPI 14h30 FR), la volatilite est generee par les algos qui activent leur programme **prepare a l'avance**. Les analyses publiees apres la news pour expliquer le mouvement = **pareidolie** (on voit ce qu'on veut voir). En tant que trader ICT, peu importe que la news soit bonne / mauvaise pour l'economie : on regarde si on est en manipulation ou en distribution.

**BSL / SSL en pratique (QPQWlXQ-El4)** :
- **BSL au-dessus d'une resistance** = stops des shorts (sell stops devenant buy orders = sell stops). Aussi : buy orders a la cassure de la resistance par les retails breakouters.
- **SSL sous un support** = stops des longs (sell stops). Aussi : sell orders a la cassure du support.

**Implication pratique** : Vizion ne trade JAMAIS les news majeures elles-memes (slippage). On observe pendant la news, on agit apres dans le sens de la phase identifiee.

---

## 5. Structure : BOS, MSS, SISD

### 5.1 BOS (Break of Structure)

Cassure d'un swing dans le sens de la tendance => continuation. Pas detaille intensement chez Vizion (la priorite est sur MSS et SISD).

### 5.2 MSS (Market Structure Shift) (7q1cQyvvQKI, 3RAGDN6TfTA, EPDZiywdUyw, kyk9Y3EYeE8, P66QVQNegvo)

**Schema MSS bearish** :
- Swing high 1 — Swing low (entre les 2 highs) — Swing high 2
- Le niveau du **swing low intermediaire** = niveau de "displacement".
- MSS valide quand le prix CLOS (corps de bougie) SOUS ce niveau de displacement.
- MSS bullish = symetrique.

**Le MSS est plus tardif que le SISD le plus souvent**. Il est utilise dans le **modele 2022** : trader la rebalance d'un FVG forme dans le displacement du MSS.

#### 5.2.1 Les 3 elements constitutifs stricts d'un MSS (kyk9Y3EYeE8)

**MSS bullish** :
1. **Swing point** : avoir d'abord un swing low (= 3 bougies dont la centrale a un low plus bas que voisines), suivi d'un swing high, suivi d'un nouveau swing low. Le 2eme swing low doit etre **plus bas que le premier** (dans le sens de la tendance bearish initiale).
2. **Cloture au-dessus du swing high intermediaire** : le prix doit clore (corps) au-dessus du swing high entre les 2 swing lows.
3. **Displacement** : le mouvement de remontee qui produit cette cloture doit etre un **displacement higher** — une vraie volonte algo (gros corps de bougie, peu de meche).

**MSS bearish** : symetrique avec swing high / swing low / swing high, cloture en corps sous le swing low intermediaire, displacement lower.

> "Sans displacement, il n'y a pas de volonte de changement de structure quantifiable sur les bougies, donc PAS DE MSS." (kyk9Y3EYeE8)

#### 5.2.2 Les 5 erreurs courantes a eviter sur le MSS (P66QVQNegvo)

**Erreur 1 — Pas de swing point correct** : si le 2eme swing high n'est PAS au-dessus du premier (pour un MSS bearish), ou inversement pour bullish, **ce n'est pas un MSS** meme si on a un displacement par la suite (P66QVQNegvo).

**Erreur 2 — MSS qui ne cloture pas correctement lors du displacement** : si la cloture au-dessus/sous le swing intermediaire se fait **avec de grandes meches** sans corps net, ce n'est PAS un displacement, donc PAS un MSS (P66QVQNegvo).

**Erreur 3 — Cloture nette mais pas de displacement** : on peut clore en corps au tiers au-dessus du swing, mais sans bougie d'impulsion claire (volume + grand corps), il n'y a pas displacement. Une **cloture en corps n'est PAS un displacement** automatiquement (P66QVQNegvo).

**Erreur 4 — MSS parfait mais sans niveau de liquidite pertinent en amont** : un MSS bullish qui se valide **AU-DESSUS d'un sellside liquidity non pris** est low probability. Le MSS doit idealement se valider **APRES la prise** de la liquidite pertinente (le SSL pour un MSS bullish, le BSL pour un MSS bearish). Sinon, le MSS n'annonce qu'un **simple retracement**, pas une vraie distribution (P66QVQNegvo).

**Erreur 5 — Confondre MSS et MSS Modele 2022 quand on veut TRADER le MSS** : si on veut entrer en position sur un MSS, il faut imperativement un **FVG forme dans la phase de displacement** (busy pour MSS bullish, cibi pour MSS bearish). Sans FVG dans le displacement, on ne peut PAS trader le MSS comme un setup d'entree — on a juste un MSS observable (P66QVQNegvo).

#### 5.2.3 MSS High Probability — la confluence parfaite (P66QVQNegvo)

Un MSS bullish "high probability" cumule :
- 3 elements constitutifs stricts (swing pattern + cloture + displacement).
- Validation en **zone Discount** d'un mouvement haussier majeur (apres retracement).
- **Prise de SSL** juste avant le MSS (sweep / manipulation).
- Presence d'un **FVG (busy) dans le displacement** => MSS = Modele 2022 tradable.
- (Bonus) **Divergence SMT** sur l'actif correle.

> "Cet ensemble du MSS valide sur un niveau bien particulier represente une confluence qui rend le MSS high probability." (P66QVQNegvo)

#### 5.2.4 Setup d'entree MSS Modele 2022 (kyk9Y3EYeE8, P66QVQNegvo)

- **Entree** : au rebalance du FVG (busy/cibi) forme dans le displacement.
- **Stop loss** : sous le swing low (bullish) ou au-dessus du swing high (bearish) qui a structure le MSS.
- **Take profit** : prochaine BSL/SSL pertinente, ou zone Discount/Premium opposee.
- Apres validation du modele 2022 sur HTF, on peut **descendre d'une echelle** pour chercher un setup imbrique (fractalite).

#### 5.2.5 Lien MSS et Market Maker Model (kyk9Y3EYeE8)

Les "vrais" MSS qui marchent sont ceux qui se forment :
- Au plus haut d'un mouvement haussier pour un **Market Maker Sell Model** (MSS bearish).
- Au plus bas d'un mouvement baissier pour un **Market Maker Buy Model** (MSS bullish).

Un Market Maker Model se declenche quand le prix atteint une PDR HTF (high time frame) ou un reversal est attendu. Concept a developper dans une future section dediee.

### 5.3 SISD (Change in State of Delivery) (7q1cQyvvQKI, 4qrCoscxmq8)

> Un SISD est un OB qui acte un CHANGEMENT DE PHASE de marche : retracement -> expansion, ou expansion -> reversal.

C'est concretement un OB place a la fin d'une phase qui demarre une nouvelle phase. **Plus precoce que le MSS** => meilleur RR.

**Definition complete (4qrCoscxmq8)** : "L'utilite du SISD reside dans le fait de determiner la fin d'une phase et le debut de la nouvelle. Ce qui delimite ces deux phases, c'est l'OB qui valide la fin de la premiere et le debut de la seconde."

**Echelles autorisees pour le SISD (4qrCoscxmq8)** : Daily, H4, H1, 30min, 15min et **maximum 5min**. Pas de SISD 1min ou 2min — c'est une regle stricte Vizion : meme si l'alerte TradingView sonne sur 1min apres prise d'un BSL/SSL, **NE PAS descendre directement en 1min**. Cela "est absolument a eviter" (4qrCoscxmq8).

**Le SISD = OUTIL DE CONFIRMATION (4qrCoscxmq8)** : le prix atteint un niveau d'interet, puis le price action manifeste une volonte de changement de direction via la validation d'un OB. Le SISD donne le "feu vert" qu'on n'attend plus la poursuite du sens precedent mais bien le nouveau sens.

**Conditions d'apparition (4qrCoscxmq8)** :
1. Le prix doit arriver sur un **niveau ou on anticipe une reaction** (BSL/SSL, equal H/L, FVG HTF).
2. Apres la prise, **un OB doit se valider** sur le TF approprie.
3. Cet OB devient le SISD : il marque la **fin de la phase precedente** et le **debut d'une nouvelle phase** (manipulation -> distribution dans un AMD, ou retracement -> expansion).

**SISD high probability = SISD pendant killzone (4qrCoscxmq8)** : un SISD qui se forme PENDANT London KZ, NY AM (8h30 si news, 9h30 toujours sur indices) est une signature **Time + Price**. Exemple cite : un OB bearish forme avec la bougie 8h30 (news US) et valide a 9h30 (open equity US) sur indices.

**SISD final d'une AMD (4qrCoscxmq8)** : sur une structure AMD (Accumulation -> Manipulation -> Distribution), c'est l'OB valide en fin de Manipulation qui est le SISD. Il acte le debut de la Distribution.

### 5.4 Combiner MSS + SISD (7q1cQyvvQKI)

Strategie classique : attendre un MSS valide, puis attendre un SISD au retour dans la zone du displacement, et entrer sur ce SISD.

### 5.5 Application au TF alignment (ZlFI7LgOrw0)

> Un OB H1 = un MSS 5min (fractalite). Un OB 15min = MSS 1min.

C'est cette fractalite qui justifie le tableau du TF alignment.

---

## 6. Discount / Premium / Equilibrium

### 6.1 Outil (eS6adgOouqI)

Fibonacci retracement entre le dernier swing low (palier 0) et le dernier swing high (palier 1) :
- **Equilibrium** = 50%.
- **Discount** = sous 50% (zone d'achat).
- **Premium** = au-dessus de 50% (zone de vente).

### 6.2 Application (eS6adgOouqI, BKGoNf6vhRY)

1. Toujours tracer Discount/Premium sur le dernier mouvement directionnel propre.
2. OB bullish n'est pertinent que s'il se forme en **Discount**.
3. OB bearish n'est pertinent que s'il se forme en **Premium**.
4. FVG en Discount/Premium = high probability (TrTBuk2Ttl4).

### 6.3 Important (eS6adgOouqI)

> "La phase d'expansion est l'unique phase pendant laquelle les pidurets fonctionnent." => si le prix ne montre pas d'engagement directionnel apres avoir touche le Discount/Premium, on n'attend PAS que les FVG marchent.

### 6.4 Avantages concrets (xUb364bilGQ)

**Filtre numero 1 (xUb364bilGQ)** :
- Repere en un clic si c'est le moment d'acheter / vendre.
- "Eviter les mauvaises operations est une part enorme du travail pour etre profitable" (xUb364bilGQ).

**Money management — amelioration du RR (xUb364bilGQ)** : exemple concret cite :
- Acheter au milieu d'une hausse : SL sous low recent, TP au BSL plus haut = RR ~0.5 (mauvais).
- Attendre la zone Discount : meme TP, SL plus serre = RR ~2.8 (excellent).

Le simple fait d'attendre la zone Discount/Premium **MULTIPLIE le RR par ~5x** sur le meme trade.

**Reduction du drawdown (xUb364bilGQ)** : entrer en zone Discount/Premium reduit drastiquement le temps passe en drawdown. Si on entre au milieu d'une hausse, on peut subir une perte longue et lourde jusqu'au SL.

### 6.5 Discount/Premium comme cible TP (xUb364bilGQ)

Dans une operation **contraire-tendance** (short dans une tendance haussiere), on ne vise PAS un TP enorme. **L'equilibrium** (palier 50% Fibo) est une cible pertinente pour TP. Au-dela : risque de reprise de la tendance.

### 6.6 Discount/Premium = filtre des PDR pertinents (xUb364bilGQ)

> "Quels sont les OB / FVG / Breakers a surveiller ? Reponse : ceux en zone Discount/Premium." (xUb364bilGQ)

C'est le filtre principal pour selectionner les "bonnes" PDR dans un mouvement.

### 6.7 Multi-TF (xUb364bilGQ)

Si une zone Discount n'a pas ete cherchee sur Daily, elle peut tout a fait avoir ete cherchee sur H4 ou H1. Toujours verifier plusieurs echelles avant de conclure qu'une zone n'est "pas cherchee".

---

## 7. SMT Divergence

### 7.1 Definition (0dzjqeTDXxU, BKGoNf6vhRY, uOv1znt2uAY)

Sur 2 actifs CORRELES, l'un prend un swing high/low que l'autre ne prend pas. Cet ecart anormal = signature algorithmique.

### 7.2 Paires correlees utilisees chez Vizion (0dzjqeTDXxU)

| Actif primaire | Actif correle |
|---|---|
| EUR/USD | GBP/USD |
| NAS100 | SP500 (ou Dow Jones) |
| XAU (or) | XAG (argent) |
| WTI | Brent |

Vizion privilegie : **NAS vs SPX**, **EURUSD vs GBPUSD**, **XAU vs XAG** (0dzjqeTDXxU).

### 7.3 Bullish vs Bearish SMT (0dzjqeTDXxU)

- **SMT bullish** : divergence sur les LOWS. Un actif prend un low, l'autre fait un higher low => potentiel haussier.
- **SMT bearish** : divergence sur les HIGHS. Un actif prend un high, l'autre fait un lower high => potentiel baissier.

### 7.4 SMT observee vs SMT validee (0dzjqeTDXxU)

- **Observee** = simple constat graphique. Pas un signal de trade.
- **Validee** = un OB s'est valide dans le sens attendu. C'est ce qui transforme la divergence en signature actionnable.

### 7.5 Force / Faiblesse (0dzjqeTDXxU)

- **Actif fort** = celui qui FAIT le nouveau high (en SMT bearish) ou low (en SMT bullish).
- **Actif faible** = celui qui n'a pas reussi.
- Vizion privilegie de **trader l'actif fort** (celui qui a "purge" la liquidite). C'est ce qui a moins de chance de re-prendre la liquidite.

### 7.6 SMT comme epice, pas comme plat (0dzjqeTDXxU)

> "La SMT est l'epice. Si votre plat (analyse) n'est pas bon, l'epice ne le sauvera pas."
- Ne JAMAIS trader un actif juste parce qu'il y a une SMT.
- Ne PAS exiger une SMT pour valider un setup deja complet.

### 7.7 Ne PAS confondre avec divergence RSI (0dzjqeTDXxU)

SMT = entre 2 actifs (prix vs prix). Pas un indicateur.

---

## 8. Killzones (heure NY)

### 8.1 Outil TradingView (TDYspBnIZOw)

Indicateur : **"ICT Killzones and Pivots"** par TFO. Parametrage Vizion :
- Timezone : **GMT-4** (heure NY).
- Time frame limit : H1 (afficher KZ jusqu'a H1 maximum).
- Sessions : 50 (50 dernieres).

### 8.2 Killzones precises (heure NY) (TDYspBnIZOw, EHQzmAML6VI)

| Killzone | Horaire NY |
|---|---|
| Asiatique | 20h00 — 00h00 NY |
| London (UK) | 02h00 — 05h00 NY |
| NY AM | 08h30 — 11h00 NY |
| NY Lunch | 12h00 — 13h00 NY |
| NY PM | 13h30 — 16h00 NY |

### 8.3 Forex vs Indices (EHQzmAML6VI)

- **Forex** : les killzones NY commencent a **7h00 NY**.
- **Indices** : on distingue **8h30 NY** (news US embargo) et **9h30 NY** (open cash US).
  - 8h30 = pertinent SEULEMENT s'il y a une news US a 8h30.
  - 9h30 = pertinent TOUS LES JOURS (open equity).

### 8.4 Killzones reellement tradees par Vizion (EHQzmAML6VI, TDYspBnIZOw)

- **London KZ** : tradee.
- **NY AM KZ** : tradee.
- **NY PM KZ** : tradee (en soiree heure FR, moins systematique).
- **Asia KZ** : non tradee (Vizion dort la nuit).
- **NY Lunch** : rarement productive.

### 8.5 Time + Price (EHQzmAML6VI, ZlFI7LgOrw0)

> Un OB ou FVG qui se valide DANS une killzone = high probability. Hors KZ = trade dechet.

### 8.6 POI = high/low de killzone (TDYspBnIZOw)

Sur une journee :
- Si bullish bias => le POI est le LOW de la killzone Asia ou London.
- Si bearish bias => le POI est le HIGH de la killzone Asia ou London.

### 8.7 Conversion NY -> UTC

- Heure d'ete US (Mars-Novembre) : NY+4 = UTC. 7h NY = 11h UTC.
- Heure d'hiver US (Novembre-Mars) : NY+5 = UTC. 7h NY = 12h UTC.

### 8.8 Open Midnight NY (TDYspBnIZOw)

Niveau d'ouverture daily a 00h00 NY. Sert de reference pour le PO3 daily (debut de la bougie).

### 8.9 Profils anti-trade : Sick'n'Destroy & Asia delivre (UVHltN8NewM)

**Sick'n'Destroy (UVHltN8NewM)** :
- Configuration ou Londres prend FORTEMENT le high ET le low de la KZ asiatique (les deux cotes).
- Aucune direction claire prise ensuite.
- Structure souvent en **megaphone** (range expansif avec volatilite anarchique).
- Apparait typiquement **la veille d'une news majeure (NFP, FOMC, CPI)** — c'est pourquoi on ne trade pas la veille des NFP.
- Sens du nom : **SICK** = "rechercher" (les liquidites), **DESTROY** = "detruire" (les participants).

> "Quand vous reperez cette structure, ne vous acharnez pas. Faire autre chose ce jour-la est imperatif." (UVHltN8NewM)

**Asia qui delivre (UVHltN8NewM)** :
- La KZ asiatique presente NORMALEMENT le moins de volume.
- Quand Asia delivre avec **gros volume + direction claire**, c'est ANORMAL.
- Annonce une journee **chaotique** : les SL seront cherches, les TP esquives.
- Signature courante : production d'**equal highs/lows** pendant Asia ou UK qui ne seront PAS pris ce jour-la (TP esquive) mais qui resteront des cibles pour les jours suivants.

**Manipulation a contre-sens du bias (UVHltN8NewM)** :
- Si bias bullish, on attend une **manipulation BAISSIERE** d'abord (puis distribution bullish).
- Si la KZ Londres part directement a la hausse SANS manipulation baissiere = signature contraire => journee a EVITER.
- Si bias bearish et London descend tout droit sans manipulation haute => meme regle, journee a eviter.

### 8.10 Enchainement classique des killzones (UVHltN8NewM)

Sequence typique (bougie Daily en cours, bias bullish) :
- **Asia** : accumulation (range, faible volume).
- **London** : manipulation a la baisse (prise du Asia low ou mi-busy/cibi pertinent).
- **NY AM** : SISD bullish ou MSS bullish 15min/5min => debut de la distribution.
- **NY PM** : completer l'extension jusqu'au BSL HTF cible.

Variantes (UVHltN8NewM) :
- Accumulation possible pendant **NY AM**, manipulation pendant **NY Lunch** (12-13h NY), distribution pendant **NY PM**.
- Apres MSS NY AM bullish, la suite de NY AM peut etre **retracement**, et NY PM finit l'extension.

---

## 9. Daily Bias avance

### 9.1 Procedure stricte (y2Fwp4T9sRM)

Etape par etape sur la cloture daily :

1. **PDH ou PDL pris ?**
   - PDH pris + bougie haussiere => bias bullish, cible = nouveau PDH (puis BSL plus haut).
   - PDL pris + bougie baissiere => bias bearish, cible = nouveau PDL.
   - Aucun des deux pris => **bias neutre, attendre.**
   - Les 2 pris dans la meme bougie => se fier au CORPS de cloture (cloture haute => bullish, cloture basse => bearish) ET au bias HTF.

2. **OB daily valide ?**
   - Si oui dans le sens du bias => attendre un pullback DANS cet OB en H1 pour entrer.
   - Tant que l'OB daily reste valide, le bias daily reste actif **meme si la bougie suivante est contre**.

3. **FVG daily a combler ?**
   - Agit comme magnet => cible probable.

4. **Breaker daily ?**
   - Si valide dans le sens du bias => "garde-fou" : le bias reste valide tant que le breaker n'est pas inverse.

### 9.2 Clotures fortes / faibles (y2Fwp4T9sRM)

- **Cloture forte** = grand corps de bougie + faible meche contre tendance.
- **Cloture faible** = petit corps, grandes meches => bias fragile, attendre le lendemain.
- **Doji** = hesitation => bias **neutre**, ne PAS trader le lendemain dans aucun sens jusqu'a clarification.

### 9.3 Tolerance de pullback (y2Fwp4T9sRM)

- Pour conserver un bias bullish, le pullback ne doit pas casser l'OB daily ni rebalancer/inverser un Busy daily majeur en zone Discount.
- Le Discount/Premium daily est calcule sur le DERNIER mouvement directionnel daily.

### 9.4 Bias intra-journee (y2Fwp4T9sRM)

Au sein d'une journee, le bias peut "se diviser" en deux phases :
- Si le PDL est pris en debut de journee dans un bias bullish daily => courte phase bearish jusqu'au PDL, puis bias bullish pour reprendre.

### 9.5 Methode simple PDH/PDL stricte (M4F4YYDVmMI, IR0MNTY2-rs)

**Algorithme decisionnel a la cloture daily 23h FR (M4F4YYDVmMI)** :
- **Cloture AU-DESSUS du PDH** => bias bullish demain, target = nouveau PDH (puis BSL plus haut).
- **Cloture SOUS le PDL** => bias bearish demain, target = nouveau PDL (puis SSL plus bas).
- **PDH pris en meche puis cloture sous PDH** = "echec de cloture" => bias INVERSE (bearish demain, target = PDL).
- **PDL pris en meche puis cloture au-dessus du PDL** = echec de cloture inverse => bias bullish demain.
- **Ni PDH ni PDL pris + cloture neutre (doji, toupie)** => bias **NEUTRE** => attendre demain.

**Echec de cloture (M4F4YYDVmMI)** : prise du niveau en meche puis cloture du mauvais cote = signature de force inverse. Forme grosse meche haute (echec en haut) ou grosse meche basse (echec en bas).

### 9.6 HTF prioritaires (M4F4YYDVmMI)

Vizion utilise uniquement **Weekly, Daily, H4** pour le bias. Les echelles **H1, 15min, 5min, 1min ne servent JAMAIS a construire un bias** (M4F4YYDVmMI).

> "Ne croyez pas que realiser un bias H1 vous permettra de savoir ce qui se passera l'heure suivante. C'est un piege a eviter imperativement." (M4F4YYDVmMI)

### 9.7 Workflow bias chronologique (M4F4YYDVmMI)

- **1x par mois** : verifier la cloture mensuelle (bias monthly).
- **1x en debut de semaine** : verifier la cloture weekly precedente (bias weekly).
- **1x par jour a 23h FR** : verifier la cloture daily du jour (bias pour demain).

### 9.8 Confluences a integrer au bias (M4F4YYDVmMI, IR0MNTY2-rs)

Apres le bias PDH/PDL de base, ajouter en hierarchie :
1. **FVG (busy / cibi)** rebalance ou inverse en cloture daily.
2. **OB daily** valide / invalide.
3. **Breaker daily** valide.
4. **Zones Discount / Premium** : cloture en zone Discount d'un mouvement haussier majeur = bias bullish renforce.

**Exemple cite (M4F4YYDVmMI)** : un trader regardant uniquement PDH/PDL aurait short en suivant un signal apparemment bearish. Mais en integrant le Busy daily respecte + zone Discount cherchee + fond haussier => bias bullish. Le bias PDH/PDL seul peut etre trompeur.

### 9.9 Bias double / split bias (M4F4YYDVmMI)

Au sein d'une journee, le bias peut etre split en 2 :
- Si bias bullish daily mais Busy bearish 15min respecte en debut de session => bias court terme bearish en direction du PDL.
- Une fois le PDL pris (manipulation), bias bullish reprend en direction du PDH.
- C'est typiquement la AMD intra-day.

---

## 10. Weekly Profile

### 10.1 Les 3 profils (0pjj5rM1M18)

#### A. Classic Expansion (le plus frequent et puissant)

- **Lundi** : doji ou faux mouvement contraire au bias hebdo.
- **Mardi** : reversal — prise du PDL/PDH du lundi, validation OB.
- **Mardi soir / Mercredi / Jeudi** : phase d'expansion = on trade.
- **Vendredi** : TGIF — retracement contraire. Objectif = 20-30% du weekly range. Peut etre trade dans le sens du retracement.

#### B. Midweek Reversal

- **Lundi et Mardi** : accumulation / faible volatilite.
- **Mercredi** : reversal — prise des low/high du lundi-mardi, validation OB.
- **Mercredi soir / Jeudi / Vendredi** : expansion.

#### C. Consolidation Reversal (AMD weekly)

- **Lundi-Mardi-Mercredi** : consolidation/dojis.
- **Jeudi** : manipulation (prise de liquidite contre le bias).
- **Jeudi-Vendredi** : distribution dans le sens du bias.

### 10.2 Filtres a appliquer (0pjj5rM1M18)

- **Toujours filtrer par le bias hebdo**. Un Classic Expansion bullish ne se trade qu'avec bias bullish.
- **Si aucun profil ne se valide la semaine, NE PAS TRADER**.
- "Ne pas trader, c'est aussi trader."

### 10.3 Combiner avec daily profile (0pjj5rM1M18, XQ6A6w3brwU)

Une fois le weekly profile attendu, on cherche un daily profile en confluence (London Reversal, NY Manipulation, NY Reversal).

### 10.4 Methode bias weekly express dimanche soir (rdrCs58XZfA)

Routine 5 minutes le dimanche, sur l'echelle Weekly uniquement :

1. **Tracer PWH et PWL** (Previous Weekly High / Low).
2. **Verifier les PDR weekly** sur la cloture :
   - Busy ou Cibi rebalance / inverse ?
   - OB weekly valide ?
   - Breaker weekly valide ?
3. **Verifier la position vs equilibrium** du dernier mouvement weekly directionnel.
4. **Conclusion bias weekly** :
   - Cloture sur Busy bullish respecte + en zone Discount (sous equilibrium) = bias weekly **bullish**.
   - Inverse = bias weekly **bearish**.
   - Aucun PDR pertinent + cloture en milieu de mouvement = bias weekly **neutre** => ne pas trader.

### 10.5 Application PO3 weekly (rdrCs58XZfA)

Apres bias weekly bullish defini :
- On attend une **bougie weekly bullish (OLHC)** : ouverture > meche basse (manipulation) > corps haussier > cloture haute.
- La meche basse = **manipulation weekly**, idealement sur le PWL (target manipulation).
- Le corps haussier = **distribution weekly**, idealement vers PWH (target distribution).

**Niveau ideal d'achat** : sur le PWL (apres manipulation), structure bullish sur LTF (15min/5min) pour entrer dans la distribution.

---

## 11. Power of Three (PO3) / AMD

### 11.1 AMD (Pm29OIifOns, AE0K6W9uiSY, h0Dc_28mZhY, Ad2rHrZGW5c)

3 phases : Accumulation -> Manipulation -> Distribution.

- **Accumulation** = consolidation visible (range). Pas tradable.
- **Manipulation** = prise de liquidite CONTRE le bias (faux mouvement). Pas tradable.
- **Distribution** = expansion dans le sens du bias. C'est ce qu'on TRADE.

L'entree se fait a la fin de la Manipulation, via la validation d'un OB.

**Ordre chronologique strict (h0Dc_28mZhY)** : les 3 phases DOIVENT s'observer dans l'ordre A -> M -> D. **On ne peut PAS sauter la phase d'accumulation**. Si on observe seulement une manipulation isolee, **ce n'est pas une AMD**.

**Mecanique liquidite (h0Dc_28mZhY)** : pendant l'Accumulation, les participants prennent position dans les deux sens, leurs SL se placent de part et d'autre du range = **engineered liquidity**. La Manipulation va chercher ces SL pour donner aux algos l'**elan en termes de liquidite** necessaire a la Distribution. Apres prise des SL, la Smart Money active le programme de distribution.

**Comment savoir si la Manipulation est terminee (h0Dc_28mZhY)** :
> "La fin de la manipulation est confirmee par la validation d'un OB (et optionnellement un MSS)." (h0Dc_28mZhY)

C'est l'OB valide en sens contraire de la manipulation qui acte la bascule en Distribution.

**AMD high probability (h0Dc_28mZhY)** : une AMD est high probability quand elle se produit sur **un gros niveau de liquidite passe** (PDH, PDL, swing HTF, equal H/L pertinent). Sans niveau pertinent, l'AMD est faible.

**Trois pieges classiques (h0Dc_28mZhY)** :
- Trader une fausse AMD ou la liquidite n'a JAMAIS ete prise pendant la "manipulation" => piege.
- Trader un mouvement qui part en ligne droite sans manipulation => **PAS d'AMD**, donc PAS d'elan suffisant, risque eleve.
- Se tromper sur quel mouvement est la manipulation : sur prises bilaterales (high ET low pris), c'est le **bias daily** qui tranche (bias bearish => la prise du high est la manipulation, la prise du low est la distribution).

**AMD HTF (h0Dc_28mZhY)** : sur Daily/Weekly/Monthly, l'AMD permet d'anticiper la direction sur plusieurs jours/semaines. Apres le SISD HTF, on attend des retracements sur l'OB ou FVG HTF pour entrer en continuation.

### 11.2 PO3 sur une bougie HTF (SBZvpF3FK2w, EHQzmAML6VI)

Une bougie peut etre nommee :
- **OLHC** (bullish) : Open, Low, High, Close. La meche basse = manipulation. Le corps = distribution.
- **OHLC** (bearish) : Open, High, Low, Close. La meche haute = manipulation. Le corps = distribution.

### 11.3 PO3 par TF (SBZvpF3FK2w)

| TF du PO3 | Setup attendu sur |
|---|---|
| Daily | H1 |
| H4 | 15min |
| H1 | 5min |

Vizion N'UTILISE PAS le PO3 sous H1 (trop de faux signaux).

### 11.4 PO3 H4 (SBZvpF3FK2w)

- Bougie H4 de **10h NY** = bougie de reference (souvent retrace puis distribue). Surtout sur indices US et matieres premieres.

### 11.5 PO3 H1 (SBZvpF3FK2w)

Bougies de **8h, 9h, 10h NY** = bougies high probability (en raison des news 8h30 et open 9h30 sur indices).

### 11.6 Application (SBZvpF3FK2w)

> "Quand le prix part DANS le mauvais sens en debut de bougie HTF, c'est NORMAL et c'est PARFAIT : c'est la mesh / manipulation. Attendre la validation d'un OB sur LTF pour entrer dans la direction de la cloture HTF attendue."

### 11.7 Quelle bougie HTF surveiller ? (SBZvpF3FK2w)

Bougies HTF qui CONTIENNENT au moins une killzone => high probability. Les autres = ignorees.

### 11.8 Indicateur TradingView PO3 (cZka3M1YGDg)

**Outil utilise quotidiennement chez Vizion (cZka3M1YGDg)** :
- Indicateur : **"PO3"** par **Toodegrees** sur TradingView.
- Permet d'afficher la bougie d'une echelle HTF (Daily, H4, Weekly...) directement sur le graphique en cours.
- Sur 15min, on voit la bougie Daily en cours dessinee en temps reel a droite => on suit l'evolution de la meche/corps.

**Parametrage Vizion (cZka3M1YGDg)** :
- **Plage horaire** : choisir le TF de la bougie a afficher (Day, H4, Week...).
- **COCHER "Use New York Midnight"** : la bougie Daily commencera bien a 00h00 NY (pas 18h NY par defaut TradingView).
- Possibilite d'afficher 2 bougies HTF simultanement (ex. Daily + Weekly) en ajoutant l'indicateur 2 fois et en decalant l'une via le parametre "decalage".
- Cacher temporairement : icone oeil sur l'indicateur (pas besoin de reparametrer).

**Utilite (cZka3M1YGDg)** : voir en direct si la bougie Daily est en phase de manipulation (mauvais signe pour entrer) ou de distribution (entrer maintenant). Conjugue au bias daily.

---

## 12. Checklist de setup parfait (etape par etape)

> Source principale : BKGoNf6vhRY (la "bible"), enrichi par EHQzmAML6VI, 8DQz3QlWB5Q, uOv1znt2uAY, AE0K6W9uiSY.

### Etape 0 — Pre-conditions globales (jour J)

- [ ] Bias daily defini (PDH/PDL pris la veille).
- [ ] Weekly profile defini ou semaine "neutre" -> ne pas trader.
- [ ] Calendrier economique verifie (pas de FOMC/NFP/CPI surprise non couverte).
- [ ] Jour de semaine compatible : Mardi-Mercredi-Jeudi de preference ; Lundi/Vendredi avec prudence.

### Etape 1 — Crediblite de l'OB (TF entree, ex. 5min) (BKGoNf6vhRY)

- [ ] L'OB a-t-il pris une **liquidite externe pertinente** (swing high/low, PDH/PDL, low/high de KZ passee) ?
- [ ] L'OB se forme-t-il en **zone Discount** (long) ou **Premium** (short) du dernier mouvement ?
- [ ] Le prix **rebalance-t-il des FVG** sur le chemin de validation de l'OB ?
- [ ] (Bonus) Y a-t-il un volume imbalance rebalance ?

Si l'une des 3 premieres conditions n'est pas remplie => **PAS DE TRADE**.

### Etape 2 — Imbrication TF (Time Frame Alignment) (BKGoNf6vhRY)

- [ ] Si trade 5min => verifier H1.
- [ ] L'**OB 5min doit etre CONTENU dans un OB H1** de meme direction.
- [ ] L'**OB H1 doit avoir valide des FVG** sur son chemin (idealement aussi PDH/PDL, ou un FVG daily).
- [ ] Si trade 1min => verifier 15min (memes regles).

### Etape 3 — Confirmation SMT Divergence (BKGoNf6vhRY)

- [ ] Sur l'actif correle (paire dans la table SMT).
- [ ] L'actif primaire prend la liquidite, le correle NON => divergence validee par l'OB.
- [ ] Si pas de SMT mais le reste est solide, c'est OK (la SMT est l'epice).

### Etape 4 — Time + Price (EHQzmAML6VI)

- [ ] L'OB se valide-t-il **PENDANT** une killzone (London, NY AM, NY PM) ?
- [ ] Pour indices : si validation a 9h30 NY, c'est ideal (open US).
- [ ] Si hors KZ => **PAS DE TRADE** ou trade tres reduit (trash potentiel).

### Etape 5 — PO3 / AMD (SBZvpF3FK2w, AE0K6W9uiSY)

- [ ] La validation de l'OB correspond-elle a la fin de la phase de Manipulation d'un AMD H1 ou H4 ?
- [ ] Sur la bougie HTF (H1 ou H4), a-t-on un debut de bougie qui forme la meche dans le sens contraire (manipulation) ?

### Etape 6 — Entree (BKGoNf6vhRY, tNgYKSK3Vz0)

- [ ] Entree au RETEST de l'OB LTF (pas a la validation).
- [ ] OU entree au retest du Breaker block si on a un setup Unicorn.
- [ ] OU entree au pullback FVG dans le displacement.

### Etape 7 — Risk management

- [ ] Stop loss sous le low de l'OB (long), au-dessus du high (short).
- [ ] Risk reward minimum **1.5** (8DQz3QlWB5Q), cible 2-6.
- [ ] TP sur prochaine BSL/SSL pertinente (swing HTF) ou standard deviation -1, -2.5, -4.

### Etape 8 — Tenue (BKGoNf6vhRY, ZlFI7LgOrw0)

- [ ] Breakers blocks valides dans le sens du trade => signature de continuation.
- [ ] FVG sur le chemin respectes (rebalance + bounce).
- [ ] Nouveaux OB qui se forment dans le sens du trade.
- [ ] Pas de SMT bearish en faveur du contraire.

---

## 13. Entree, Stop Loss, Take Profit

### 13.1 Entree (BKGoNf6vhRY, tNgYKSK3Vz0)

**3 modes d'entree principaux :**

1. **Direct sur OB LTF retest** (5min). Stop sous le low de l'OB 5min.
2. **Setup Unicorn** (tNgYKSK3Vz0) : entree sur le **Breaker block** (apres prise de liquidite + stop hunt + OB + FVG). Stop sous le breaker (preference mecanique Vizion : sous la MECHE) OU sous le low du stop hunt (plus safe).
3. **Rebalance FVG en zone Discount/Premium** (TrTBuk2Ttl4).

### 13.2 Stop Loss (tNgYKSK3Vz0, 5fv2MjuPKE4)

- Long : sous le **LOW** (corps + meche) de l'OB ou du stop hunt.
- Short : au-dessus du HIGH.
- **Toujours en meche, pas en corps** (sinon le marche prendra la meche).
- Pour un Breaker, sous la **MECHE** du breaker.

### 13.3 Take Profit (BKGoNf6vhRY, sjQW8l-Vi_s, tKEoXpb1FlE, TrTBuk2Ttl4)

**Methodes Vizion :**

1. **Prochain swing H/L pertinent** : swing visible UNE ECHELLE AU-DESSUS (regle TF alignment, sjQW8l-Vi_s).
2. **Equal Highs / Equal Lows** = magnets prioritaires.
3. **PDH / PDL / PWH / PWL**.
4. **Standard Deviation** (tKEoXpb1FlE) : Fibonacci a niveaux -0.5, -1, -1.5, -2, -2.5, -3, -4 trace sur l'OB. TP partiel sur chaque palier ; TP ultime sur -2.5 ou -4 (si -2.5 traverse en cloture).
5. **Clotures partielles** : sur chaque palier + breakeven progressif.

### 13.4 Risk Reward (8DQz3QlWB5Q, tNgYKSK3Vz0, BKGoNf6vhRY)

- **RR minimum : 1.5** (refuser un trade < 1.5).
- **RR ideal : 2** (RR2 rule pour Unicorn).
- Vizion mentionne souvent des RR 3 a 10 sur des setups parfaits (BKGoNf6vhRY mentionne RR=6).

### 13.5 Money management

Pas de regle stricte explicitee (Vizion dit "1.5 minimum", "rester safe"), mais le bot devra implementer :
- Risk fixe par trade (recommande 0.5-1%).
- Cloture partielle a RR=1, RR=2.
- Breakeven a RR=1.

---

## 14. Trades "dechets" (a EVITER)

### 14.1 La liste complete (BKGoNf6vhRY, 8DQz3QlWB5Q, eS6adgOouqI, CBwTRl5-DqU, wfSjER_Ixao, nSL8DQ3UIMg, Pm29OIifOns, tIeV0r-yko4)

1. **OB hors killzone** (EHQzmAML6VI). Time+Price absent => trash.
2. **OB seul, sans TF alignment** (BKGoNf6vhRY). Pas d'OB HTF de meme direction => trash.
3. **OB sans prise de liquidite** (EPDZiywdUyw, 5fv2MjuPKE4). Pas un OB.
4. **OB en zone "milieu"** (ni Discount ni Premium) (eS6adgOouqI).
5. **Trader le 1min/2min directement** sans descendre via le TF alignment (eS6adgOouqI, AE0K6W9uiSY).
6. **Trader pendant la consolidation** sans attendre l'expansion (AE0K6W9uiSY, Pm29OIifOns).
7. **Trader le Lundi** sans attendre signature (8DQz3QlWB5Q, 0pjj5rM1M18, uOv1znt2uAY). "Garder les cartouches".
8. **Trader autour du Vendredi soir** (sauf TGIF clairement valide).
9. **Trader contre le bias daily** (BKGoNf6vhRY). Bias bearish => ne pas chercher d'OB bullish 5min.
10. **Trader contre le weekly profile** attendu (0pjj5rM1M18).
11. **FVG isole sans OB ni Discount/Premium ni SMT** (TrTBuk2Ttl4, eS6adgOouqI). Le FVG seul n'est PAS un setup.
12. **Acheter "pile poil" a 9h30 NY** (EHQzmAML6VI). Slippage, volatilite, faux mouvement = trade detruit.
13. **Acheter parce que "ca part sans moi"** (FOMO) (eS6adgOouqI, uOv1znt2uAY).
14. **Shorter un retracement dans une expansion HTF** (Pm29OIifOns). Le retracement n'est pas un setup short si la tendance HTF est bullish.
15. **Ouvrir un trade sur OB juste apres une expansion qui a atteint son objectif** (eS6adgOouqI). Apres l'objectif HTF atteint, on ENTRE dans une phase d'incertitude (sick-n-destroy possible).
16. **Trader pendant les news high impact sans plan precis** (uOv1znt2uAY, wfSjER_Ixao). FOMC/NFP/CPI = blackout 30 min avant et apres.
17. **Trader hors heures de session pertinentes** (EHQzmAML6VI). Asia KZ = pas de trade pour Vizion.
18. **Imbriquer trop de concepts** (eS6adgOouqI, ZlFI7LgOrw0). Vizion utilise ~6-7 concepts seulement.
19. **Trader un breaker bloc qui n'a jamais ete un vrai OB avant** (EPDZiywdUyw). Pas d'OB initial = pas de breaker.
20. **Trader avec un RR < 1.5** (8DQz3QlWB5Q).

### 14.2 Anti-pattern psychologique (CBwTRl5-DqU, nSL8DQ3UIMg, Pm29OIifOns, tIeV0r-yko4)

- "Hier mes FVG marchaient, donc aujourd'hui ils marcheront" => FAUX (eS6adgOouqI).
- "Je vais faire un trade rapide pour 'me refaire'" => trade revenge => boucle de pertes.
- "Le marche c'est de l'arnaque" => non, c'est juste qu'on a saute des etapes.
- "Le scalping 1min c'est plus rapide d'enrichir" => non, c'est plus rapide de perdre (5fv2MjuPKE4, eS6adgOouqI).

---

## 15. Mindset & gestion

### 15.1 Discipline (BKGoNf6vhRY, 8DQz3QlWB5Q, uOv1znt2uAY, Teew5ZPctQo, GIRUY_tQEE8, lIICSXQIZ7Y)

- **Patience > action**. Une semaine peut passer sans setup = normal.
- **"Ne pas trader, c'est aussi trader."**
- **Garder les cartouches pour les vrais moments** (8DQz3QlWB5Q).
- **Comprendre AVANT de trader** : si vous ne savez pas POURQUOI un trade doit marcher, ne le prenez pas.

### 15.2 Gestion psychologique (eS6adgOouqI, uOv1znt2uAY)

- En drawdown : si l'analyse etait correcte (signature complete), TENIR. Ne pas paniquer sur la meche basse de la bougie H1 (PO3 normal).
- En profit : laisser courir avec TP partiels + breakeven progressif (BKGoNf6vhRY).

### 15.3 Money management

- Risque fixe par trade.
- RR mini 1.5.
- Diversifier (plusieurs paires Forex, indices, matieres premieres) pour avoir des opportunites tournantes (UU4MZRT4324).

### 15.4 Routine quotidienne (BKGoNf6vhRY, uOv1znt2uAY)

1. Verifier la cloture daily de la veille (bias daily).
2. Lire le calendrier economique.
3. Identifier le weekly profile en cours.
4. Definir le scenario "si baisse alors X, si hausse alors Y" AVANT l'ouverture.
5. Attendre les killzones, ne RIEN faire avant.

---

## 16. Breaker Blocks

### 16.1 Definition (tNgYKSK3Vz0, ZlFI7LgOrw0, 9yChQ3V7u_o)

Un **Breaker block bullish** = ancien OB bearish qui a ete invalide (cloture au-dessus du high de l'OB bearish). Apres l'invalidation, il devient un breaker bullish, et un retour du prix dessus est une zone d'achat.

Symetrique pour breaker bearish.

> "Un breaker, c'est un order block qui a ete inverse." (9yChQ3V7u_o)
> "Breaker en anglais = casseur. La tendance a ete cassee, renversee, et le breaker doit supporter la nouvelle tendance qui est en place." (9yChQ3V7u_o)

**Point cle (9yChQ3V7u_o)** : un breaker bullish est constitue des memes bougies qui formaient l'OB bearish d'origine. Ces bougies etaient haussieres dans l'OB bearish (qui etait constitue de bougies haussieres pousseuses), et ces memes bougies restent haussieres dans le breaker bullish. La conversion du statut bearish a bullish traduit le sens du breaker.

**Inversion de statut PDR (9yChQ3V7u_o)** : la conversion d'un PDR bearish en PDR bullish (ou inverse) lorsqu'il devient breaker est une **signature algorithmique**.

### 16.2 Conditions strictes (EPDZiywdUyw, 9yChQ3V7u_o)

- **L'OB initial doit etre un vrai OB** (avoir pris de la liquidite quand il a ete valide).
- **L'OB doit etre invalide en CLOTURE**, pas seulement en meche.
- Le retest du breaker en CORPS de bougie = entree de qualite.
- Apres validation du breaker, **il n'est plus question que le prix reparte dans le sens initial**. Le breaker doit supporter la nouvelle tendance (9yChQ3V7u_o).

### 16.3 Entree et Stop Loss sur breaker (9yChQ3V7u_o, tNgYKSK3Vz0)

**Convention Vizion stricte (9yChQ3V7u_o)** :
- **Entree** : en **CORPS de bougie** du breaker (pas en meche). Memes raisons que pour les OB : on prefere les corps de bougie pour l'entree.
- **Stop Loss** : sous la **MECHE** du breaker (pour le breaker bullish) ou au-dessus de la meche (pour le breaker bearish). On prend en consideration l'extremite en meche de bougie pour le SL = davantage de securite.
- Pour un setup Unicorn : SL alternatif sous le low du stop hunt.

### 16.4 Breaker high probability — les 3 conditions (9yChQ3V7u_o)

Un breaker peut etre qualifie de "high probability" (le plus de chance d'etre respecte) si **les 3 conditions suivantes** sont reunies :

**Condition 1 — Qualite de l'OB d'origine (9yChQ3V7u_o)**

Le breaker est dependant de l'OB. Plus l'OB initial est high probability, plus son inversion produit un breaker pertinent. L'OB d'origine doit donc :
- Avoir manipule un niveau de liquidite (high, low, equal high/low, ou imbalance dans un FVG).
- Avoir ete valide en zone Discount (OB bullish a l'origine) ou Premium (OB bearish a l'origine).
- Exemple cite : OB bearish forme en zone Discount qui inverse l'OB lui-meme un peu plus tard = breaker bullish HP en zone Discount.

**Condition 2 — Association des PDR sur le chemin de validation du breaker (9yChQ3V7u_o)**

Une fois l'OB inverse en breaker, observer dans le mouvement de validation **quels PDR bullish (ou bearish) se sont crees en parallele** :
- L'ancien Cibi (FVG bearish) qui a ete inverse devient un **IFVG bullish**.
- De **nouveaux OB bullish** se sont valides sur le chemin.
- De **nouveaux Busy** (FVG bullish) se sont formes et tiennent.
- Si ces PDR bullish sont tous respectes apres validation du breaker => le breaker d'origine est qualifie de high probability.

> "Surveiller la qualite de ces pidiere bullish permet de qualifier le breaker d'origine comme etant high probability." (9yChQ3V7u_o)

**Condition 3 — Displacement (9yChQ3V7u_o)**

Le breaker est high probability quand la cloture qui le valide se situe **sous la meche** du breaker (pour un breaker bearish — symetrique pour bullish), traduisant la force de la direction du prix. Quand on cloture fortement sous la meche, c'est ce qu'on appelle le **displacement**.

**Implication pratique** : un breaker avec displacement part souvent sans pullback => l'entree au retest peut etre manquee. Mais tant que le niveau de liquidite cible n'a pas ete cherche, **un pullback reste attendu** et c'est la qu'on peut rentrer. Le mouvement post-pullback est ensuite rapide et puissant.

### 16.5 Les 2 types de breakers : Continuation vs Reversal (9yChQ3V7u_o)

Vizion distingue deux types de breakers selon la phase de marche dans laquelle ils apparaissent. Cette distinction est cruciale pour anticiper la target.

**A. Breaker de Continuation (9yChQ3V7u_o)**

- Apparait **AU SEIN d'une tendance** deja claire (un breaker bullish dans une tendance haussiere ; un breaker bearish dans une tendance baissiere).
- Apparait idealement **apres un mouvement correctif** (retracement en zone Discount/Premium).
- Condition : **l'objectif HTF de la tendance n'est pas encore atteint**.
- Sert a relancer / supporter la tendance en cours, pas a la renverser.
- Exemple cite : breaker bullish apres pull-back en zone Discount d'un mouvement haussier > continuation jusqu'au buyside liquidity HTF.

**B. Breaker de Reversal (9yChQ3V7u_o)**

- Valide le **renversement de la tendance**.
- Apparait **APRES que l'objectif HTF a ete atteint** (la liquidite cible — BSL ou SSL — a ete prise).
- Bascule d'un statut haussier a un statut baissier (ou inverse).
- Cible suivante : le prochain niveau de liquidite en sens inverse (SSL si bascule en bearish, BSL si bascule en bullish).
- Lien fort avec le Market Maker Model (sera detaille dans la section dediee).

**Regle de decision Vizion (9yChQ3V7u_o)** :
- Si la tendance n'a pas encore atteint son objectif HTF => le breaker rencontre = breaker de continuation.
- Si la tendance a deja atteint son objectif HTF => le breaker rencontre = breaker de reversal.

### 16.6 Breaker comme garde-fou de bias (y2Fwp4T9sRM, UU4MZRT4324)

Un breaker daily ou weekly bullish reste valide => le bias bullish reste valide, meme si plusieurs bougies suivantes sont contre.

### 16.7 Fractalite du breaker (9yChQ3V7u_o)

Comme tous les concepts ICT, le breaker est fractal. Application Vizion :
- Detecter un breaker valide sur une echelle de temps (ex. H1).
- Attendre qu'il soit retravaille (retest).
- A l'interieur du breaker, sur une echelle inferieure (ex. 5min), chercher un nouveau setup (nouvel OB et/ou nouveau breaker valide).
- Avec seulement le concept du breaker, on peut imbriquer plusieurs setups.

### 16.8 Combinaison minimale Vizion (9yChQ3V7u_o)

> "Coupler le breaker avec l'order block, le FVG, les zones discount et premium represente une force de frappe puissante. Et je ne parle que de quatre concepts."

Ces 4 concepts cumulatifs (OB + FVG + Breaker + Discount/Premium) suffisent. Pas besoin de 15 concepts differents, pas besoin de se perdre en 1min. Privilegier 15min et 5min pour observer des OB et donc des breakers significatifs.

---

## 16-bis. Calendrier economique

> ⚠️ **Section informationnelle uniquement — NON appliquée par le bot V1.**
> Décision user 2026-05-15 : pas de blackout news. Le bot trade pendant les annonces. Cette section reste comme référence pour la stratégie manuelle / future V2.

### 16b.1 Outils utilises (QblIZjGov4I)

- **ForexFactory.com** (prefere Vizion). Filtre : icone Filter > monnaie = USD > importance = rouge (High Impact). Calendrier en anglais.
- **Investing.com** (alternative francais). Filtre : Filtre > pays = Etats-Unis > Etoiles = 3.

### 16b.2 Les 3 news majeures a noter (QblIZjGov4I)

1. **IPC (CPI / inflation US)** : 1x par mois, **14h30 heure FR (8h30 NY)**.
2. **NFP (Non-Farm Payroll)** : 1x par mois, **vendredi de la 1ere semaine du mois**, 14h30 FR (8h30 NY).
3. **FOMC** : decision sur les taux Fed + discours Jerome Powell. Generalement le **mercredi a 20h FR (14h NY) + Powell 30 min plus tard**.

### 16b.3 News medium impact (QblIZjGov4I)

- Generalement **16h FR (10h NY)**.
- Active sur ForexFactory en cochant "Medium" dans le filtre.
- Importantes pour les bougies H4 sur indices (sera detaille avec PO3 H4).

### 16b.4 Regle d'or news (QblIZjGov4I)

> "Ne JAMAIS trader lors de la publication d'une news high impact. Si la news est a 14h30, ne cherchez pas a trader a 14h29 ou 14h30. Slippage et volatilite detruisent le trade." (QblIZjGov4I)

Blackout recommande : **30 min avant et apres** une news high impact.

### 16b.5 Lecture de la semaine via le calendrier (QblIZjGov4I)

Application du PO3 weekly avec calendrier :
- **Lundi + Mardi sans news majeure** = **range / accumulation** => NE PAS TRADER. Les algos sont en "Sick'n'Destroy" pour ces deux jours, ils chassent les SL sans direction. Trader Lundi-Mardi = se faire detruire.
- **Mercredi = FOMC** = phase de **manipulation weekly** typique. La grosse bougie de mercredi peut etre la meche basse (si bias weekly bullish) ou haute (si bearish) de la bougie weekly.
- **Jeudi + Vendredi** = phase de **distribution weekly**. C'est la qu'on trade.

> "Pendant que plein de particuliers tentent des choses Lundi et Mardi, vous n'avez encore rien fait. A mercredi, le FOMC valide votre bias et vous savez quoi faire jeudi-vendredi." (QblIZjGov4I)

### 16b.6 Verification dimanche soir (QblIZjGov4I)

Routine Vizion :
1. Dimanche soir, ouvrir le calendrier eco filtre US 3 etoiles.
2. Reperer la prochaine FOMC / NFP / CPI de la semaine.
3. Si Lundi-Mardi sans news => prevoir 0 trade ces jours.
4. Si news mercredi => prevoir manipulation weekly mercredi, distribution jeudi-vendredi.
5. Le vendredi NFP est une journee a part : forte volatilite, on peut s'attendre a un Sick'n'Destroy le jeudi.

---

## 17. Daily Profiles

### 17.1 London Reversal (XQ6A6w3brwU)

- Durant la KZ London, le prix prend un high/low time-based (typiquement le high de NY PM precedent).
- OB se valide dans la KZ London.
- Expansion sur le reste de la session, jusqu'au low/high de la KZ asiatique.
- Condition : bias daily doit correspondre.

### 17.2 NY Manipulation (mentionne, XQ6A6w3brwU)

Manipulation sur l'open US, puis distribution dans le sens du bias daily.

### 17.3 NY Reversal

Reversal majeur dans la KZ NY AM (typiquement a 9h30 NY ou 10h NY).

### 17.4 Marche futur, types d'ordres (cL6xgy6T8o8)

Vizion explicite pour le trading futures (NQ, MNQ) :
- **Market order** : execution directe.
- **Limit order** : execution conditionnelle (Buy Limit sous le prix actuel, Sell Limit au-dessus).
- **Stop order** : devient market a un niveau (Buy Stop au-dessus, Sell Stop en dessous).

Pour un short futur :
- TP = Buy Limit (en dessous).
- SL = Buy Stop (au-dessus).

---

## 18. Differences ICT classique vs Vizion FR

| Concept | ICT classique | Vizion FR |
|---|---|---|
| OB definition | Bougie inverse au mouvement | Bougie qui pousse le prix dans une LIQUIDITE (plus strict, exige prise de liquidite) (5fv2MjuPKE4) |
| TF privilegies | Tous | **H1, 15min, 5min uniquement** pour OB. Pas de 1min direct (5fv2MjuPKE4) |
| Entree OB | Souvent meche | **Corps** prefere chez Vizion (plus de marge + RR) (5fv2MjuPKE4) |
| Killzones | Plusieurs definitions | Vizion = NY/London/PM/Asia avec horaires precis NY (TDYspBnIZOw) |
| Indices 8h30/9h30 | Peu detaille | Vizion : 8h30 seulement si news, 9h30 toujours (EHQzmAML6VI) |
| Setup Unicorn | Mentionne sans nom | Vizion : entree sur BREAKER (pas sur OB), avec stop hunt obligatoire (tNgYKSK3Vz0) |
| SMT | Operee sur multiples paires | Vizion : 3 paires principales (NAS/SPX, EUR/GBP, XAU/XAG) (0dzjqeTDXxU) |
| PO3 | Daily principalement | Vizion : Daily + H4 + H1 (pas sous H1) (SBZvpF3FK2w) |
| RR | Variable | Vizion : RR mini 1.5, "RR2 rule" sur Unicorn (tNgYKSK3Vz0, 8DQz3QlWB5Q) |
| Weekly profile | ICT 5 profils | Vizion : 3 profils (Classic Expansion, Midweek Reversal, Consolidation Reversal) (0pjj5rM1M18) |
| Daily bias | PDH/PDL simple | Vizion : PDH/PDL + OB daily + FVG daily + Breaker daily + Discount/Premium (y2Fwp4T9sRM) |

---

## 19. Spécifications pour le bot

### Priorite 1 — INDISPENSABLE

1. **Detection OB strict Vizion** :
   - OB bullish = N bougies BAISSIÈRES **consécutives** (N≥1) qui prennent une liquidité externe. Aucune bougie haussière "intruse" tolérée au milieu du groupe.
   - OB bearish = symétrique avec N bougies haussières consécutives.
   - Validation = clôture corps de bougie au-dessus (long) / en-dessous (short) de l'origine du groupe.
   - Sortie : si pas de prise de liquidité => `is_valid_OB = False`.
   - Pas de fusion TF automatique en V1 (si bougie intruse, le candidat est rejeté ; on ne remonte pas en TF supérieur). Décision user 2026-05-15.

2. **Time Frame Alignment strict** :
   - Si OB 5min, exiger OB H1 contenant dans la meme direction.
   - Si OB 15min, exiger OB H4 ou H1 ; OB 1min => exiger OB 15min ; etc.
   - Une fonction `tf_aligned(ob_ltf, ob_htf)` qui retourne True/False.

3. **Killzones en HEURE NY avec DST** :
   - Implementer 5 killzones : Asia (20-00 NY), London (02-05), NY AM (08:30-11), NY Lunch (12-13), NY PM (13:30-16).
   - **Distinguer Forex (KZ NY a 7h NY) vs Indices (8h30 et 9h30)**.
   - Gestion automatique heure ete/hiver US (mars-novembre / novembre-mars).

4. **Daily Bias automatique** :
   - Analyse de la bougie daily de la veille : PDH/PDL pris ? Cloture forte/faible/doji ?
   - OB daily valide ? FVG daily ? Breaker daily ?
   - Sortie : `bias = {bullish, bearish, neutral}` + cibles.

5. **Filtre weekly profile** :
   - Identifier Classic Expansion / Midweek Reversal / Consolidation Reversal.
   - Si profil ne se valide pas => **scoring -> 0**.
   - Tous les jours tradables (lundi à vendredi). Décision user 2026-05-15 : pas de restriction par jour de semaine, le bias HTF prime.

6. **Discount / Premium / Equilibrium** :
   - Fibonacci automatique entre les 2 derniers swing high/low pertinents.
   - Verifier que l'OB est en Discount (long) ou Premium (short).

7. **SMT Divergence (BONUS, pas filtre)** :
   - 3 paires correlees : NAS/SPX, EURUSD/GBPUSD, XAU/XAG.
   - Detecter divergence sur swing low/high time-based.
   - VALIDEE seulement quand un OB de meme direction se valide.
   - Décision user 2026-05-15 : SMT = **bonus de score**, jamais filtre éliminatoire. Un trade reste valide sans SMT.

8. ~~Calendrier économique blackout~~ — **RETIRÉ** (décision user 2026-05-15).
   - Pas de blackout news. Le bot trade pendant les news.
   - Killzones standard : NY AM = 9h30 NY pour indices (pas de switch 8h30 selon news).

9. **Time + Price gate** :
   - Un OB valide doit etre dans une killzone. Sinon => trash.

10. **Stop Loss et TP automatiques** :
    - **SL = MÈCHE TOUJOURS** (OB, breaker, FVG). Décision user 2026-05-15.
    - SL sous la mèche du low de l'OB (long) ou au-dessus de la mèche du high (short).
    - TP1 = prochaine BSL/SSL HTF.
    - TP2 = standard deviation -2.5 a -4 si pertinent.
    - RR mini = 1.5, sinon skip.
    - **Risque par trade = 1% par défaut** (configurable, à revoir avec backtest).

### Priorite 2 — IMPORTANT

11. **Rebalance FVG check** :
    - Verifier que les FVG presents sur le chemin de validation de l'OB ont ete rebalances (au moins partiellement).

12. **Setup Unicorn complet** :
    - Prise liquidite + stop hunt + OB + FVG + Breaker.
    - Entree sur le breaker, pas sur l'OB.

13. **PO3 daily / H4 / H1 / M5** :
    - Detecter la phase courante de la bougie (manipulation vs distribution).
    - Filtrer : ne trader que pendant la phase de distribution.
    - Décision user 2026-05-15 : **PO3 sur M5 utilisable comme signal d'entrée** (pas seulement confluence). Notamment bougie M5 de 9h30 NY = signal valide.

14. **AMD H1 / H4** :
    - Detecter Accumulation -> Manipulation -> Distribution sur les 3 a 5 dernieres bougies.

15. **Daily profile** :
    - London Reversal, NY Manipulation, NY Reversal.
    - Filtrer en fonction du bias.

16. **Equal high / equal low detection** :
    - Tolerance "relative equal" (±X pips).

17. **Volume Imbalance** :
    - Detecter les gaps entre cloture et ouverture de bougies (intra-bougie).

18. **Breaker block detection** :
    - Tracking d'OB invalides en cloture, conversion automatique en breaker.

### Priorite 3 — NICE-TO-HAVE

19. **IFVG detection + BPR**.
20. **Standard Deviation tool** (Fibonacci -0.5, -1, -1.5, -2, -2.5, -3, -4 trace sur OB).
21. **Mèche puissante** (petit corps + grande meche) detectee comme signal LTF.
22. **Score de confluence agrege** par trade : OB strict + TF align + KZ + SMT + Discount/Premium + FVG rebalance + AMD + bias daily = score sur 8.
23. **Backtest module** avec replay bougie par bougie (style 'exercice' EPDZiywdUyw).
24. **Money management** : Kelly partiel, ou simplement risque fixe (0.5-1% par trade), avec breakeven progressif et cloture partielle.

---

## Annexe : Index des 58 videos

| ID | Theme principal | Citation marquante |
|---|---|---|
| BKGoNf6vhRY | **Checklist bible (4 etapes)** | "L'OB pousse le prix dans une zone de liquidite. C'est la premiere etape." |
| EHQzmAML6VI | **Killzones + PO3 + 830/930 indices** | "Time + Price : OB qui se valide pendant killzone = signature." |
| ZlFI7LgOrw0 | **Signatures = bias + timing + pidur** | "Une signature, c'est l'alignement du facteur temps et du facteur prix." |
| 5fv2MjuPKE4 | **Order Block defini en detail** | "Pas de manipulation, pas d'OB." |
| TDYspBnIZOw | **Indicateur Killzones TradingView (TFO)** | "Mettre GMT-4. Open midnight, 8h30, 9h30." |
| y2Fwp4T9sRM | **Daily bias avance** | "PDH pris + bougie haussiere => bias bullish demain." |
| 0pjj5rM1M18 | **3 Weekly Profiles** | "Classic Expansion = mardi reversal puis expansion 3 jours." |
| 8DQz3QlWB5Q | **Trades dechets / Routine semaine** | "Lundi-Mardi : pas de news, on ne trade pas." |
| P4ZJMKea3_w | Setup parfait exemple | "L'analyse commence en daily, jamais en 5min." |
| AE0K6W9uiSY | **Trader la consolidation (AMD)** | "Ne tradez pas la consolidation, attendez la distribution." |
| 7RWT0TOgQ5A | Setup parfait suite | "Imbriquer les concepts, pas en multiplier." |
| ZAfa52jxPDA | Setup parfait suite | "Une signature, pas un setup isole." |
| XQ6A6w3brwU | **London Reversal (daily profile)** | "Prise high NY PM veille -> OB London -> distribution." |
| wfSjER_Ixao | **Erreurs analyse + facteur temps** | "Avant le graphique, regardez le calendrier." |
| sjQW8l-Vi_s | **Choisir le bon swing pour TP** | "Le swing 5min n'est pertinent que s'il est un swing H1." |
| 7q1cQyvvQKI | **MSS vs SISD** | "MSS = changement de structure. SISD = changement de PHASE." |
| 3RAGDN6TfTA | MSS + modele 2022 | "Le MSS se valide en CLOTURE de bougie." |
| EPDZiywdUyw | **Exercices PDR (OB/FVG/Breaker)** | "Pas d'OB avant, pas de breaker apres." |
| uOv1znt2uAY | **SMT divergence en pratique** | "L'actif fort prend la liquidite, l'actif faible non." |
| eS6adgOouqI | **Discount/Premium + pourquoi les FVG ne marchent pas** | "Les pidurets marchent dans l'expansion, pas en consolidation." |
| tKEoXpb1FlE | **Standard Deviation pour TP** | "Tracer le standard dev sur l'OB. -2.5 traverse = -4 attendu." |
| UU4MZRT4324 | **Bias weekly + diversification** | "Diversifier sur 30 paires pour avoir des opportunites." |
| fb6oMRSxheU | Session London / NY interaction | "London prepare, NY delivre." |
| oP1RdY9Losc | Sessions detaillees | "La KZ NY PM peut piéger, restez prudent en soiree." |
| 6IN5CNs5qrc | **Utiliser concepts ensemble** | "Une poignee de concepts, bien utilises, suffit." |
| cL6xgy6T8o8 | **Marches futures + types d'ordres** | "NQ = 20$ du point. Buy stop = ordre devient market au declencheur." |
| CBwTRl5-DqU | Erreurs / pieges psychologiques | "L'envie de cliquer detruit votre compte." |
| nSL8DQ3UIMg | Pieges en LTF | "Le 1min sans HTF context, c'est du gambling." |
| Pm29OIifOns | **4 phases de marche** | "Consolidation, Retracement, Expansion, Reversal." |
| tIeV0r-yko4 | Erreurs courantes | "Une analyse structuree elimine 80% des trades dechets." |
| TrTBuk2Ttl4 | **FVG / IFVG / BPR + Standard Dev (TP)** | "Le FVG sans contexte n'est pas un setup." |
| tNgYKSK3Vz0 | **Setup Unicorn** | "L'entree d'un Unicorn se fait sur le BREAKER, pas sur l'OB." |
| SBZvpF3FK2w | **PO3 H4 et H1** | "Sous H1, le PO3 n'est pas fiable. 8h-9h-10h NY = bougies cles." |
| f5i0BC8M91o | Base ICT | "ICT c'est un cadre, pas une recette." |
| 0dzjqeTDXxU | **SMT divergence complete** | "SMT observee + OB valide = SMT validee." |
| bVW9NPO0zYw | Exercice | "Entrainez-vous a identifier la phase de marche." |
| i7uB5vprRDA | Exercice | "Combine OB + FVG + SMT sur exemple reel." |
| Teew5ZPctQo | Mindset/coach | "La discipline est la cle, pas la strategie." |
| GIRUY_tQEE8 | Mindset | "Le trading est un marathon." |
| lIICSXQIZ7Y | Mindset | "Acceptez les pertes, elles font partie du jeu." |
| 4qrCoscxmq8 | **SISD complet (definition + OB de transition + KZ)** | "L'utilite du SISD reside dans le fait de determiner la fin d'une phase et le debut de la nouvelle." |
| 9yChQ3V7u_o | **Breakers complet : definition, 3 conditions HP, continuation vs reversal** | "Un breaker, c'est un order block qui a ete inverse. La tendance a ete cassee." |
| Ad2rHrZGW5c | **PO3 complet (AMD, bougie OLHC/OHLC, fractalite weekly/daily/H1)** | "Pas de manipulation, pas de trade." |
| HT8ux8pww-0 | Annonce / presentation ICT (vraie annonce) | "ICT permet d'apprendre a lire les marches grace au facteur temps PUIS facteur prix." |
| IR0MNTY2-rs | **Exercice biais daily PDH/PDL en serie (Forex, or, indices)** | "Bougie baissiere qui cloture sous le low precedent = bias bearish vers le PDL." |
| M4F4YYDVmMI | **Daily Bias complet methode simple + workflow mensuel/weekly/daily** | "A 23h FR a la cloture daily, je sais deja ou va le marche demain dans la plupart des cas." |
| NK6XSOZSPac | **FVG (Busy/Cibi) complet : 3 niveaux d'utilisation + 3 placements SL** | "Le Busy = Buyside imbalance + Sellside inefficiency. Tracer les meches, pas les corps." |
| P66QVQNegvo | **5 erreurs MSS + MSS High Probability + Modele 2022** | "Confondre une cloture nette avec un displacement est l'erreur 3." |
| QPQWlXQ-El4 | **Liquidites + mecanique algos + news macro** | "Les algos bougent le prix DE LIQUIDITE EN LIQUIDITE pour atteindre le niveau institutionnel." |
| QblIZjGov4I | **Calendrier economique complet (ForexFactory, NFP/CPI/FOMC, lundi-mardi range)** | "Ne pas trader, c'est aussi trader. Lundi-Mardi sans news = sick'n'destroy garantis." |
| Ry0mCKpAFD0 | **IQH/IQL + PDH/PDL comme cibles TP prioritaires** | "L'IQH est cherche apres avoir fait croire au retail que la resistance marche." |
| UVHltN8NewM | **Killzones complet + Sick'n'Destroy + Asia qui delivre + enchainements** | "Quand Asia delivre, la journee va etre dangereuse. Megaphone = pieges, faites autre chose." |
| cZka3M1YGDg | **Indicateur PO3 (Toodegrees) sur TradingView** | "Cocher 'Use New York Midnight' pour avoir l'open daily a minuit NY et non 18h NY." |
| h0Dc_28mZhY | **AMD setup complet (mecanique liquidite + OB de fin de manipulation + HTF)** | "La fin de la manipulation est confirmee par la validation d'un OB en sens inverse." |
| kyk9Y3EYeE8 | **MSS complet : 3 elements + displacement strict + setup Modele 2022** | "Sans displacement, il n'y a pas de MSS, point barre." |
| rdrCs58XZfA | **Bias weekly express + PO3 weekly + PWH/PWL** | "Dimanche soir, 5 minutes suffisent pour anticiper toute la semaine." |
| xUb364bilGQ | **Discount/Premium complet (RR x5, retracement Fibo, filtre PDR)** | "Discount = pas cher = on achete. Premium = cher = on vend. Equilibrium = 50% Fibo." |
| rdeAjnVdfRM | **Time Frame Alignement strict (methode entonnoir, anti-FOMO 1min)** | "Pas de feu vert HTF, pas de descente en LTF. Sans structure 15min validee, ne pas trader 1min." |

---

## NOTES FINALES — Décisions user validées (2026-05-15)

Les 7 points ambigus identifiés ont été tranchés par le user. Ces décisions priment sur toute interprétation contraire dans les sections précédentes.

| # | Question | **Décision retenue** | Source Vizion |
|---|---|---|---|
| 1 | Stop Loss : mèche ou corps ? | **MÈCHE partout** (OB, breaker, FVG) | tNgYKSK3Vz0, NK6XSOZSPac |
| 2 | OB peut-il contenir une bougie "intruse" du mauvais sens ? | **STRICT** : que des bougies consécutives du sens opposé au mouvement (OB bullish = N bougies BAISSIÈRES consécutives, sans intruse haussière). Pas de fusion TF auto en V1. | 5fv2MjuPKE4 ("consécutivement baissières") |
| 3 | PO3 sur M5 ? | **UTILISABLE comme signal** (pas seulement confluence). Bougie M5 = 9h30 NY, ouverture KZ. | EHQzmAML6VI |
| 4 | SMT requise ? | **BONUS de score**, jamais filtre éliminatoire. Trade valide sans SMT si reste OK. | 0dzjqeTDXxU |
| 5 | Trader le vendredi ? | **OUI**, vendredi tradable comme tout autre jour. Pas de restriction TGIF. | (décision user) |
| 6 | Calendrier économique / blackout news ? | **IGNORÉ**. Pas de fenêtre de blackout news. Killzones standard : NY AM = 9h30 NY indices (pas de switch 8h30 selon news). | (décision user) |
| 7 | Risk per trade ? | **1% par défaut** (configurable). À revoir avec backtest. | (décision user) |

### Implications pour le bot

- §2 Order Block : implémentation stricte (bougies consécutives, pas de fusion TF auto).
- §11 PO3 : actif sur M5 comme signal d'entrée valide (pas juste confluence).
- §13 SL : règle unique = mèche.
- §16-bis Calendrier éco : **section informationnelle uniquement**, NON appliquée par le bot.
- §19 Specs bot : retirer "calendrier éco blackout" et "restriction vendredi". Conserver killzones standard.

### Points encore ouverts

- **Market Maker Model** : laissé de côté en V1. Concept ICT standard déjà couvert implicitement par PO3 + AMD + Weekly Profile.
- **Nom exact indicateur PO3 TradingView** : "Toodegrees" probable (transcription Whisper "Tout D'Yggris"). À vérifier sur ton TradingView quand utile.

---

Fin du document.
