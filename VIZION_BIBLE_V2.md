# VIZION BIBLE V2 — Enrichissement playlist 2026 (8 videos)

> Source : playlist `PL-Mtj712CaBIFVCi2xnAknmXDRTZxkr-t` (chaine Vizion FR).
> Methode : transcript Whisper + extraction frames 1/min + analyse multimodale.
> Ce document COMPLETE `c:/Users/Shadow/TradingBot/VIZION_BIBLE.md` (V1). Il ne reproduit PAS V1, il l'enrichit.
> **ADAPTE INTRADAY (decision user 2026-05-15)** : la METHODOLOGIE Vizion est conservee mais les regles de FREQUENCE/PATIENCE (2-3 trades/semaine, mode passif strict, "pas trader lundi-mercredi") NE s'appliquent PAS. Le bot scanne en continu ~5-10 setups/jour.

---

## §0. Cadre de lecture

### 0.1 Sources

| # | ID | Titre court | Apport principal |
|---|----|-------------|------------------|
| 01 | `YAhGt8tmfCY` | 2 tips (TFA + signatures) | Tenue par signatures, KZ obligatoire BB |
| 02 | `WywX6DX72u8` | ICT explained simply | Breakaway gap, switch d'actif, cascade D->H1->M5->M1 |
| 03 | `VXnCchk8NTM` | 5min strat (no daily bias) | Mode "Feu Vert H1" formalise |
| 04 | `k1EXTR1bJ0k` | 1min strategy Fridays | Setup OB+BISI, Open Midnight NY, TP safe SMT |
| 05 | `likFjNWr6Rk` | Jours d'expansion | Propulsion Block, Daily Open vs Midnight, re-entree post-PDH |
| 06 | `tjkJoBmT2Gs` | OB + FVG | Synchro OB+FVG, filtre "pourquoi ca marcherait" |
| 07 | `P43KEkyv6C8` | Methode complete 5 etoiles | Methode Maitre, RR gate binaire, FVG obligatoire pour MSS |
| 08 | `Zaqk9WkRQPM` | Strategie simplifiee | Noyau minimal, "8h30 manipule -> 9h30 distribue" |

### 0.2 Regles Vizion ECARTEES pour le bot intraday

- "2-3 trades par semaine" (video 07) -> **ECARTEE**, ~5-10 setups/jour vises.
- "Mode passif strict entre validation HTF et open US" (videos 02, 07) -> **ECARTEE comme regle bloquante**.
- "Lundi/mardi/mercredi pas de trade" (video 05, calendrier eco) -> **ECARTEE**.
- "Filtre calendrier economique" (videos 05, 07) -> **DEJA ECARTEE en V1**.

### 0.3 Ce qui RESTE valable en intraday

Cascade TF rigide, concept entonnoir, toutes les regles structurelles (OB, FVG, MSS, SMT, KZ, Time+Price), filtres de qualite (RR, premium/discount, KZ obligatoire BB), sequence imperative **manipulation -> retournement -> setup -> entree**.

---

## Sommaire

§1 Cascade revisee • §2 Feu Vert H1 • §3 Synchro OB+FVG • §4 Propulsion Block • §5 Breakaway Gap • §6 Open Midnight vs Daily Open • §7 KZ obligatoire BB • §8 Setup OB+BISI • §9 Pattern 8h30/9h30 • §10 Pas de FVG = pas de MSS • §11 RR gate binaire • §12 Calendrier eco (reference) • §13 "Pourquoi ca marcherait" • §14 Time+Price stricte • §15 Noyau minimal • §16 Sequence imperative • §17 Switch d'actif • §18 Re-entree post-PDH • §19 Bougie d'expansion • §20 TP safe SMT • §21 Plancher TF (clarification) • §22 Tenue par signatures • §23 3 modes intraday • §24 Contradictions • §25 Implications bot_v2 • Annexe

---

## §1. Cascade Vizion REVISEE : Methode Maitre vs Feu Vert

V1 propose une cascade unique `D1 -> H1 -> M15 -> M1`. Les videos 03 et 07 montrent **deux cascades distinctes** selon le contexte daily.

**Methode Maitre (biais daily clair)** — video 07 :
```
Daily (biais) -> M15 (MSS + CB premium) -> M1 (MSS imbrique + entree CB 1min)
```
H1 absent comme niveau decisionnel. SMT obligatoire au M1.

**Feu Vert (biais daily neutre)** — video 03 :
```
Daily (cadrage neutre) -> H1 (OB H1 post-sweep PDH/PDL) -> M5 (entree)
```
M15 absent. SMT NQ vs DJ (et non NQ vs ES si correles).

**Implication bot intraday** : garder la cascade V1 `D1 -> H1 -> M15 -> M1` comme **squelette**. H1 = contexte secondaire dans Methode Maitre, decisionnel dans Feu Vert. Ne pas attendre LA fenetre unique 9h30 NY — rejouer le template a chaque KZ (Asia, London, NY AM, Lunch, NY PM).

---

## §2. Le "Feu Vert H1" - mode biais daily neutral

Concept FORMALISE en video 03 (~5:00). C'est l'evenement H1 qui debloque la chasse de trade quand le daily bias est neutre.

**Conditions strictes** :
1. Sweep PDH ou PDL sur la bougie H1 courante.
2. Cloture H1 forte dans le sens oppose (~70% amplitude).
3. OB H1 valide par cette cloture.

**Boosts (non obligatoires)** : OB time-based (forme en KZ), divergence SMT confirmee.

**Apres feu vert** : descendre en M5 (jamais M1 direct), panier de signatures imbriquees (breaker M5 + SiBi/iFVG + OB M5), entree sur retest OB M5. TP = origine du LRLR si en KZ.

**Application intraday** : le bot doit detecter automatiquement `daily_bias == neutral` (grosse bougie d'expansion daily + 1-2 bougies range) et **assouplir le filtre daily_bias** pour autoriser la chasse via H1. Sans cela, paralysie plusieurs jours par mois.

---

## §3. Synchronisation OB + FVG (setup high-probability)

Source : video 06 (~11:25, ~12:35).

> "Order block bullish valide avec creation d'un busy lors de la validation de l'order block, ce sont des conditions high probability."

**La bougie qui valide l'OB DOIT etre la bougie centrale d'un FVG du meme sens.**

**Difference V1** : V1 §2.3 dit "rebalance FVG sur le chemin de validation" (plus large). Video 06 exige le FVG NAISSANT pendant la validation.

**Algorithme** :
```python
def is_high_proba_OB(ob, seq3):
    validation = ob.validation_candle
    fvg = detect_fvg(seq3)
    if fvg and fvg.middle_candle == validation and fvg.direction == ob.direction:
        return "high_proba_OB_FVG_sync"
    return "OB_seul"  # tradable mais score reduit
```

Boost `setup_quality_score += 2` si synchro. OB seul reste tradable avec RR plus eleve.

---

## §4. Propulsion Block (OB-sur-OB)

Source : video 05 (~17:50).

**Definition** : nouvel OB qui se forme **directement par-dessus** un premier OB valide, meme direction. Empilement.

**Usage** : 2eme entree avec SL plus serre (sous le Propulsion Block, pas sous l'OB original). TP identique a l'entree initiale.

**Implication bot** : apres 1ere entree validee, continuer a scanner pour Propulsion Block. Si detecte ET position en profit -> ADD-IN autorise (limite a 1x taille initiale). Si position en perte -> NE PAS faire moyenne (RE-ENTREE distincte avec SL propre).

**Distinct du Breaker (V1 §16)** : Propulsion = OB additionnel meme direction. Breaker = OB inverse.

---

## §5. Breakaway Gap (FVG garde-fou)

Sources : videos 02 (~24:42) et 04 (~20:00).

**Definition** : FVG (busy/CB) qui se forme dans la **phase d'expansion** juste apres validation OB. Ne doit JAMAIS etre rebalance/inverse.

**Role** :
- **Garde-fou runtime** : inverse en IFVG du cote oppose -> cloture immediate du trade.
- **SL serre** : SL place sous le breakaway gap (pas sous l'OB) -> RR ameliore.

**Critere visuel** : FVG cree par bougie de displacement (corps > 70% range), entre l'OB et la cible, apres accumulation->expansion AMD.

**Code bot** :
```python
# Runtime invalidation
if trade.breakaway_gap.is_inverted_to_ifvg():
    close_trade(reason="breakaway_gap_inverted")
```

**iOFED** (mention rapide video 04) : reaction au tick pres sur breakaway gap + nouvel OB bullish 1min = boost confiance (a logger).

---

## §6. Open Midnight NY vs Daily Open NY (deux niveaux distincts)

Source : video 05 (~297s).

> "C'est a 18h, heure de New York, a ne pas confondre avec l'Open Midnight, qui est minuit, heure de New York."

| Niveau | Heure NY | Heure Paris (hiver) | Role |
|---|---|---|---|
| Daily Open | 18h00 | 00h00 | Ref D1 pour PO3 daily, separe les sessions daily |
| Open Midnight NY | 00h00 | 06h00 | Pivot intraday, niveau de retracement minimal |

**Usage** :
- **Daily Open** -> PO3 H1 (meche basse sous + cloture H1 au-dessus = manipulation finie).
- **Open Midnight NY** -> pivot intraday, retracement bullish doit s'arreter A ou AU-DESSUS. Cassure en dessous = biais affaibli.

**V1 vs V2** : V1 §8.8 mentionne "Open Midnight NY" sans le distinguer du Daily Open 18h. **A clarifier dans le code** : DEUX variables distinctes.

---

## §7. Killzone obligatoire pour Breaker Block

Source : video 01 (~13:43).

> "Cet [breaker block bullish] s'est fait sur la killzone asiatique, tandis que ce breaker block ici s'est fait entre deux killzones. On n'est donc pas dans une fourchette de temps high probability."

**Regle stricte** : un BB hors killzone est INVALIDE meme structurellement parfait.

**Difference V1** : V1 §16 detaille le BB (continuation/reversal, 3 conditions high-proba) sans specifier la KZ comme condition stricte BINAIRE. Video 01 erige ca en filtre eliminatoire.

**Code** :
```python
def validate_breaker_block(bb):
    if not bb.in_killzone(["asia", "london", "ny_am", "ny_pm", "ny_lunch"]):
        return False, "BB_hors_KZ_invalide"
    # ... V1 §16.4 checks
```

---

## §8. Setup OB + BISI (different du Unicorn)

Source : video 04 (~19:00, ~21:00).

**Difference avec Unicorn (V1 §13.1)** :
- Unicorn = OB + stop hunt (2eme prise) + FVG + Breaker. Entree sur le breaker.
- **OB+BISI** = OB valide + BISI (FVG bullish) cree par l'OB. Entree sur **rebalance BISI**. SL **sous l'OB** (pas sous le BISI - meche-cherche frequente).

**Quand l'utiliser** : pas de 2eme stop hunt apres premier sweep (donc pas de Breaker).

**Mecanique** :
```
1. OB bullish 1min valide (apres sweep + KZ + SMT)
2. BISI nait de la validation OB (synchro §3)
3. Ordre limite acheteur au top du BISI
4. SL = sous l'OB
5. TP1 = premier BSL intraday (cloture partielle)
6. TP final = BSL avec SMT bearish observable (§20)
```

Ajouter un mode d'entree explicite `mode = "OB_BISI"`.

---

## §9. Pattern "8h30 manipule -> 9h30 distribue" (indices US)

Source : video 08 (~10:00).

> "Si 8h30 manipule, donc 14h30 heure francaise, alors 9h30, donc 15h30, distribue. C'est ce que ICT nous explique."

**Lecture** : quand news 8h30 NY cree un mouvement violent, c'est la **manipulation**. La VRAIE direction se delivre a 9h30 NY.
- News fait MONTER violemment a 8h30 -> attendre 9h30 prenne BSL puis SHORT.
- News fait BAISSER violemment a 8h30 -> attendre 9h30 prenne SSL puis LONG.

**V1** : §16-bis traite calendrier eco mais cette regle Time+Price n'est pas codifiee. Pattern intraday recurrent a coder.

**Implication bot** :
- Marquer fenetres 8h30 et 9h30 NY.
- Si grand mouvement a 8h30 (range > 1.5x ATR moyenne) : activer mode "watch 8h30 manipulation".
- A 9h30, attendre sweep des extremes 8h30 + reversal + OB inverse + SMT contre direction 8h30.

**Note user** : compatible avec decision user (filtre calendrier eco desactive) car cette regle ne dit pas d'EVITER les news mais comment les TRADER.

---

## §10. Regle "Pas de FVG = pas de MSS"

Source : video 07 (~11:30).

> "S'il n'y a pas de FVG, et bien il n'y a pas de market structure shift."

**Un MSS sans FVG genere par le displacement EST INVALIDE.** Le FVG (CB/SiBi) est la signature physique du displacement.

**V1** : §5.2 definit le MSS (swing-low entre 2 swing-highs + cassure en corps) sans mentionner le FVG comme condition. **Precision stricte a coder** :
```python
def validate_mss(mss):
    fvg = check_fvg_around(mss.displacement_candle, window=3)
    if fvg is None or fvg.direction != mss.direction:
        return False, "MSS_sans_FVG_invalide"
    return True
```

Impact : reduit ~30-50% des faux MSS. Compatible avec besoin intraday (moins de faux signaux).

---

## §11. RR = GATE BINAIRE avant entree

Source : video 07 (~10:00, ~10:45).

> "Que le setup soit magnifique ou pas vous ne le prenez pas si le ratio n'est pas bon."

**Difference V1** : V1 §13.4 dit "RR min 1.5, ideal 2" comme seuil GLOBAL. Video 07 va plus loin : **simuler RR a chaque niveau cascade**, attendre palier suivant si RR < 1.

**Application bot intraday** :
```python
def cascade_with_RR_gate(setup):
    rr_m15 = simulate_rr(entry=mss_15m.high, sl=mss_15m.high+buffer, tp=PDL)
    if rr_m15 < 1.0:
        wait_for_retracement_to_cb()
    rr_m1 = simulate_rr(entry=cb_1m.mid, sl=cb_1m.high+buffer, tp=asian_low)
    if rr_m1 < 2.0:
        return "skip_no_trade"
    return "execute"
```

**Seuils intraday** : RR M15 >= 1.0, RR M1 >= 2.0 (vs 3.0 chez Vizion swing). **NE JAMAIS ajuster SL/TP pour faire passer le RR**.

---

## §12. Predictibilite jour d'expansion via calendrier eco (REFERENCE)

Source : video 05 (~50s-150s).

Regle Vizion : news US concentrees jeudi/vendredi -> 3 jours range (lun-mer) puis 2 jours expansion (jeu-ven).

**Statut bot** : **REFERENCE SEULEMENT - PAS APPLIQUEE**. Filtre calendrier eco desactive en V1. Section conservee pour comprehension future si le user veut reactiver.

---

## §13. Filtre "pourquoi ca devrait marcher ?" (cible algo deja prise)

Source : video 06 (~05:00, ~06:30, ~07:55).

> "Si vous voyez qu'un swing low a ete pris, c'est que le mouvement de baisse provoque n'a plus de raison de se poursuivre."

**Regle** : avant d'attendre qu'un OB/FVG/BB soit respecte, verifier si la cible (SSL/BSL majeure) du leg parent est deja prise. Si oui -> programme algo termine -> ignorer tous les setups intermediaires de ce leg.

**V1** : §2.2 dit "pas de manipulation -> pas d'OB" (condition positive). Video 06 ajoute la condition NEGATIVE : "manipulation deja accomplie -> setups intermediaires invalides".

**Algorithme** :
```python
def filter_OB_FVG(setup):
    if setup.parent_leg.target_liquidity.has_been_taken():
        return None  # ignorer
    return setup
```

Va eliminer beaucoup de faux setups en intraday.

---

## §14. Filtre Time+Price erige en regle stricte

Source : video 04 (~13:30, ~14:00).

> "Les pidures qui se validate au milieu de rien et dans un contexte qui n'a pas de sens n'ont aucune raison de marcher."

**Difference V1** : V1 §14 mentionne "OB hors KZ = trash" comme exclusion simple. Video 04 le formalise comme **prerequis positif** : on doit POUVOIR REPONDRE a "pourquoi le prix devrait atteindre CE niveau a CE moment ?"

**Check-list Time+Price avant entree** :
- [ ] Niveau aligne avec liquidite reconnue (PDH/PDL, equal H/L, FVG/OB HTF) ?
- [ ] Moment dans une killzone ?
- [ ] Raison pour que le prix aille chercher ce niveau MAINTENANT (manipulation, SMT non-resolue) ?

Si une condition negative -> NO TRADE. Score < 2/3 -> ignore, 3/3 -> high-proba.

---

## §15. NOYAU MINIMAL Vizion

Source : video 08 (~19:00).

> "J'utilise les order block, les breakers, les divergences SMT, les elements time plus price, killzone, 8h30, 9h30 et ca me suffit."

**5 piliers minimaux** :
1. Order Block (M5/M15 en intraday).
2. Breakers (incluant Mitigation Block pour add-in).
3. Divergence SMT (NQ vs ES systematiquement).
4. Time + Price (KZ 8h30, 9h30, Asia/London/NY).
5. MSS comme setup d'entree.

**BONUS optionnel** : PO3, Weekly/Daily Profile, Calendrier Eco, equilibrium/discount, FVG seul.

**Implication bot** : feature flag `simple_mode = TRUE` par defaut. Le reste = scoring boost, jamais conditions necessaires. Maximise la robustesse.

---

## §16. Sequence imperative : manip -> retournement -> setup -> entree

Source : video 08 (~07:00).

> "Pas de manipulation, pas de trade."

Sequence non-negociable :
```
1. Manipulation observable (sweep liquidite) -> oui ? continuer
2. Structure de retournement (MSS, OB, CISD) -> oui ? continuer
3. Setup d'entree visible (OB+FVG sync, BISI, CB) -> oui ? continuer
4. Entree
```

**V1** : a une checklist (§12) mais pas une machine d'etat sequentielle. La sequence Vizion impose chaque etape ATTEINTE avant la suivante.

**Implication bot** : machine d'etat `WAIT_HTF_CONTEXT -> WAIT_MANIPULATION -> WAIT_RETOURNEMENT -> WAIT_SETUP -> WAIT_PULLBACK -> ENTER_TRADE -> MONITOR`. Setup detecte sans manipulation observee = IGNORE. Reset a chaque nouvelle KZ ou changement daily bias.

---

## §17. Switch d'actif obligatoire entre indices US correles

Source : video 02 (~03:20).

> "Quand on voit que ca ne marche pas sur l'actif qu'on avait en reference, on va devoir simplement changer de referentiel."

Si Nasdaq ne valide pas son OB Daily mais SP500/Dow le fait -> switcher l'analyse.

**Implication bot** :
- Le bot ne doit PAS etre attache a UN actif.
- A chaque scan, evaluer NQ, ES, YM en parallele.
- Si un seul valide un setup high-proba -> trader celui-la.

**V1** : §7 SMT inter-marches comme outil de **divergence**. Video 02 ajoute la **selection** : prendre l'actif valide vs ignorer celui qui ne valide pas.

---

## §18. Re-entree apres PDH pris (modele agressif)

Source : video 05 (~520s-740s).

**Concept** : normalement "PDH pris -> trade fini". Video 05 propose modele agressif assume :

**Conditions** :
1. PDH du NQ deja pris.
2. SP500 n'a PAS pris son high equivalent.
3. Retracement attendu vers Open Midnight NY (pas plus bas).
4. Lot de KZ valide la divergence.
5. PO3 sur l'Open Midnight (meche basse + cloture au-dessus).

**UTILE pour intraday** : evite de "louper" la 2eme expansion apres PDH matin. Compatible avec philosophie intraday (2-3 trades/jour potentiels apres prise PDH si SMT confirme).

**V1** : §14 dit "acheter sur prise de liquidite seule sans pidieres bullish = sommet" (vrai). Video 05 ajoute la NUANCE : SMT + retracement Open Midnight + KZ -> re-entree possible.

---

## §19. Bougie d'expansion (definition visuelle)

Source : video 05 (~1245s).

> "Grand corps de bougie, petite meche."

**Ratio corps/range > 70%, meche sens contraire < 30%.**

```python
def is_expansion_candle(c):
    rng = c.high - c.low
    return rng > 0 and (abs(c.close - c.open) / rng) > 0.70

def is_reversal_candle(c):  # video 02 ~14:40
    rng = c.high - c.low
    return rng > 0 and (abs(c.close - c.open) / rng) < 0.30
```

**Application** :
- Distribution PO3 doit etre une bougie d'expansion.
- **Bougie reversal NE PEUT PAS etre la bougie de validation d'un OB** (video 02 ~14:40).
- Jour d'expansion : bougie daily de cloture doit etre une bougie d'expansion.

---

## §20. TP safe via SMT bearish (cloture partielle obligatoire)

Source : video 04 (~26:00).

> "Pourquoi j'ai pris mon TP final sur ce buy-side liquidity et pas celui la ? C'est la-dessus qu'on a une divergence SMT avec le SP500."

**Regle** : le TP final n'est PAS le plus haut BSL accessible. C'est le BSL ou une **divergence SMT bearish** est observable sur le correle.

**Logique** : marche "fatigue" sur ce BSL si correle ne suit pas -> probabilite plus faible d'atteindre le BSL suivant -> hit rate maximise.

```python
def find_safe_TP(direction, correlated_asset):
    candidates = identify_BSL_levels(direction)
    for tp in candidates:
        if has_SMT_against_at_level(correlated_asset, tp, direction):
            return tp
    return candidates[-1]
```

**Cloture partielle OBLIGATOIRE** (video 04 ~22:00) : "Pay the trader". TP1 = premier BSL intraday, cloture 30-50%. Reste = BE jusqu'au TP safe.

**V1** : §13.3 propose "Standard Deviation" et "swing H/L pertinent". Video 04 ajoute methode FILTRANTE basee sur SMT pour CHOISIR PARMI candidats.

---

## §21. Plancher TF "OB" vs plancher TF "entree" - clarification CONTRADICTION

**La contradiction** :
- Video 06 (~09:00) : "Ne descendez pas en dessous du 15 minutes" pour les OB.
- Video 07 (~16:30) : Entree sur CB 1 minute, MSS imbrique 1min.

**Resolution** : les deux disent la meme chose a des niveaux differents :
- **OB de STRUCTURE** : minimum M15. La zone attendue.
- **OB d'ENTREE** : peut etre M1, **DANS la zone d'imbalance de l'OB/FVG M15+**.

C'est l'entonnoir final : structure M15 -> entree M1 dans l'imbalance.

**Code bot** :
```python
def is_valid_entry_OB(ob, parent_structure):
    if ob.timeframe in ["M1", "M5"]:
        return parent_structure.imbalance_zone.contains(ob.price_range)
    return ob.timeframe in ["M15", "H1", "H4", "D1"]
```

**Application** : bot trace OB structure sur M15+ uniquement. Quand OB structure valide, "ouvre" une zone d'entree LTF. Tant que prix pas dans la zone d'imbalance, M1 ignore.

---

## §22. Methode "2 tips" - Tenue par signatures

Source : video 01.

**Les 2 tips** :
1. Timeframe Alignment (deja V1 + §1, §2).
2. **Lecture par signatures** : tenue du trade par accumulation de signatures.

**Regle de tenue NOUVELLE** :
> "Si des signatures bullish s'effacent et laissent place a des signatures bearish, passer break-even, couper une partie, voire couper tout."

**Pour un trade LONG** :
- Signatures bullish entrantes (accumuler) : OB bullish valide, FVG bullish respecte EN CORPS, OB bearish INVERSE -> BB bullish, FVG bearish INVERSE -> IFVG bullish (= "NIVG bullish" dans la transcription).
- Signatures bearish entrantes (declenchent action) : OB bullish casse en cloture, FVG bullish NON respecte en corps, OB bearish valide pendant le trade.

**Algorithme** :
```python
def manage_open_position(trade):
    bearish_count = 0
    while trade.is_open:
        signature = detect_new_signature()
        if not signature.is_aligned_with(trade.direction):
            bearish_count += 1
            if bearish_count == 1: trade.move_sl_to_breakeven()
            elif bearish_count == 2: trade.close_partial(0.5)
            elif bearish_count >= 3: trade.close_all()
```

**Respect en CORPS** (precision video 01 ~02:14) : un FVG respecte uniquement par la meche n'est PAS une signature valide. Le corps doit reposer SUR/AU-DESSUS du FVG.

**NIVG = IFVG (synonyme)** : Vizion utilise "NIVG bullish" pour IFVG bullish. Memes concepts. A logger comme synonyme.

---

## §23. Synthese - 3 modes intraday

**Mode A : Methode Maitre (biais daily clair)** — video 07
- Cascade `D1 -> M15 -> M1`. Trigger : MSS 15m post-sweep + CB premium. Entree : retest CB 1m apres MSS 1m + SMT NQ/ES. TP : Asian Low.

**Mode B : Feu Vert (biais daily neutre)** — video 03
- Cascade `D1 -> H1 -> M5`. Trigger : OB H1 post-sweep PDH/PDL + SMT. Entree : OB M5 dans zone H1 (signatures imbriquees). TP : origine LRLR.

**Mode C : Vendredi sans weekly profile** — video 04
- Cascade `D1 -> M1` (compact). Trigger : sweep Open Midnight NY + SMT NQ/ES + OB 1m + BISI. Entree : retest BISI, SL sous OB. TP : BSL safe avec SMT bearish, cloture partielle obligatoire. Heure limite : 17h30 Paris.

**Selection mode (bot)** :
```python
def select_mode():
    if is_friday() and not weekly_profile_validated():
        return "Mode_C_Vendredi"
    if daily_bias == "neutral":
        return "Mode_B_FeuVert"
    return "Mode_A_MethodeMaitre"
```

**Frequence intraday** : independamment du mode, le bot continue de scanner en continu. Vizion 2-3/semaine NON APPLICABLE. Le bot peut prendre 5-10 trades/jour si chacun satisfait son mode.

---

## §24. Contradictions et clarifications

| Sujet | Position 1 | Position 2 | Resolution |
|---|---|---|---|
| Plancher TF OB | Video 06 : pas en dessous M15 | Video 07 : entree CB M1 | §21 — Structure M15+, entree M1/M5 OK dans imbalance |
| Cascade TF | Video 03 : D -> H1 -> M5 | Video 07 : D -> M15 -> M1 | §23 — Deux modes selon biais daily |
| SMT obligatoire ? | V1 : bonus | Video 04 : non-negociable M1 | Bonus en M5/M15, **obligatoire en M1** |
| OB "pur" puriste | V1 §0 : strict consecutif | Video 05 : OB qui pousse en liq | Garder V1 strict. Nuance video 05 ECARTEE |
| Daily bias produit | Video 07 : le matin | V1 §9 : a partir bougie veille (18h NY) | Compatible (bougie 18h dispo le matin EU) |
| RR minimum | V1 : 1.5 | Video 07 : refus < 1, cible 3.8 | M15+ >= 1.0 prerequis, entree finale >= 2.0 intraday |
| Cloture journee | Video 02 : prop firm cloture obligatoire | V1 : non specifie | Parametre `prop_firm_mode = False` par defaut |

---

## §25. Implications pour le bot bot_v2

### A. Corrections IMMEDIATES (haute priorite)

1. **Synchroniser OB + FVG dans detecteur** (§3) : si bougie de validation OB est centrale d'un FVG meme sens -> `setup_quality_score += 2`.
2. **Filtre "FVG obligatoire pour MSS"** (§10) : MSS sans FVG associe -> invalide.
3. **Killzone obligatoire pour Breaker Block** (§7) : `bb.in_killzone()`. Hors KZ = invalide.
4. **Distinguer Open Midnight NY (00h NY) et Daily Open NY (18h NY)** (§6) : deux variables distinctes.
5. **Filtre "programme algo termine"** (§13) : cible SSL/BSL du leg parent deja prise -> ignorer setups intermediaires.
6. **RR gate binaire en cascade** (§11) : simuler RR a chaque palier TF, skip/wait selon resultat.

### B. Nouvelles features a coder

| # | Feature | Source | Localisation |
|---|---------|--------|--------------|
| 1 | Mode "Feu Vert H1" auto-active si daily neutre | §2 / v03 | `bot_v2/modes/feu_vert.py` |
| 2 | Detection Propulsion Block (OB-sur-OB) | §4 / v05 | `bot_v2/detectors/propulsion_block.py` |
| 3 | Detection Breakaway Gap + invalidation runtime | §5 / v02,04 | `bot_v2/detectors/breakaway_gap.py` |
| 4 | Setup "OB + BISI" (distinct du Unicorn) | §8 / v04 | `bot_v2/setups/ob_bisi.py` |
| 5 | Pattern "8h30 manipule -> 9h30 distribue" | §9 / v08 | `bot_v2/patterns/news_manipulation.py` |
| 6 | Detection bougie expansion (corps/range > 70%) | §19 / v05 | `bot_v2/utils/candle_classification.py` |
| 7 | TP safe via SMT bearish | §20 / v04 | `bot_v2/exit/tp_smt_filter.py` |
| 8 | Switch d'actif inter-correles auto | §17 / v02 | `bot_v2/scanner/multi_asset_scanner.py` |
| 9 | Re-entree post-PDH si SMT + KZ | §18 / v05 | `bot_v2/modes/agressive_reentry.py` |
| 10 | Tenue de trade par signatures | §22 / v01 | `bot_v2/position_management/signature_manager.py` |
| 11 | Machine d'etat sequentielle | §16 / v08 | `bot_v2/state_machine.py` |
| 12 | Filtre Time+Price comme score positif | §14 / v04 | `bot_v2/scoring/time_price_score.py` |

### C. Filtres a renforcer

1. **Validation OB Daily = cloture daily au-dessus** (pas seulement meche). V1 implicite, video 02 explicite.
2. **OB seul vs OB+FVG sync** : champ `setup_quality` (`high_proba` si sync, `acceptable` sinon).
3. **Reversal candle detection** : rejeter bougie reversal comme bougie de validation OB.
4. **SMT obligatoire pour M1** : si entree M1 sans SMT -> NO TRADE (different du SMT bonus en M5/M15).
5. **Respect FVG en CORPS** (pas en meche seule) — video 01.

### D. Roadmap proposee

**Phase 1 (semaine en cours)** : implementer A.1-6 (6 corrections immediates). Tests unitaires.

**Phase 2 (semaine suivante)** : Mode Feu Vert (B.1) + Breakaway Gap (B.3) + Tenue par signatures (B.10). Backtest comparatif 1 mois.

**Phase 3 (2 semaines)** : Propulsion Block, OB+BISI, Pattern 8h30/9h30. Multi-asset scanner (B.8) : NQ-only -> NQ/ES/YM.

**Phase 4 (3 semaines)** : Machine d'etat sequentielle (B.11) — refactor flow principal. TP safe SMT, re-entree agressive.

**Phase 5 (mois 2)** : calibration seuils RR/scores via backtest. Forward-test 2 semaines. Doc utilisateur.

### E. Tests regression a prevoir

Pour chaque modification §25.A :
- Nombre de trades/jour >= 3 en moyenne.
- Winrate ne baisse pas (si > -5%, audit).
- RR moyen ne chute pas sous 1.8.

---

## Annexe : mapping video -> nouveautes

| Video | ID | Apports principaux |
|-------|----|-------------------|
| 01 | YAhGt8tmfCY | §7 KZ obligatoire BB, §22 Methode 2 tips / tenue par signatures / respect corps / NIVG=IFVG |
| 02 | WywX6DX72u8 | §5 Breakaway Gap, §17 Switch d'actif, §19 bougie reversal != OB, validation OB Daily en cloture |
| 03 | VXnCchk8NTM | §2 Feu Vert H1, §23 Mode B, §1 cascade D->H1->M5 |
| 04 | k1EXTR1bJ0k | §5 Breakaway Gap, §8 Setup OB+BISI, §14 Time+Price stricte, §20 TP safe SMT, §23 Mode C |
| 05 | likFjNWr6Rk | §4 Propulsion Block, §6 Open Midnight vs Daily Open, §12 calendrier eco ref, §18 re-entree post-PDH, §19 bougie expansion |
| 06 | tjkJoBmT2Gs | §3 OB+FVG sync, §13 "pourquoi ca marcherait", §21 plancher TF |
| 07 | P43KEkyv6C8 | §1 Methode Maitre, §10 FVG obligatoire MSS, §11 RR gate binaire, §23 Mode A |
| 08 | Zaqk9WkRQPM | §9 8h30 manipule / 9h30 distribue, §15 Noyau minimal, §16 Sequence imperative |

---

Fin du document V2.

> V2 complete V1, ne la remplace pas. V1 reste la reference pour les CONCEPTS DE BASE (OB, FVG, MSS, SMT, KZ). V2 apporte les PRECISIONS et NUANCES des 8 nouvelles videos, adaptees au trading intraday.
