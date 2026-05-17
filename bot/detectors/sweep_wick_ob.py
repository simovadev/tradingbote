"""Detection Order Block ICT correct.

VRAIE DEFINITION ICT :
- OB HAUSSIER (setup LONG) :
  * Une SEQUENCE de bougies BAISSIERES (rouges) consecutives
  * Suivie d'un push HAUSSIER impulsif (vertes) qui casse la structure
  * L'OB = ENSEMBLE des bougies rouges (le bloc qui sera retrace)
  * Au retour du prix dans cette zone -> opportunite LONG

- OB BAISSIER (setup SHORT) :
  * Une SEQUENCE de bougies HAUSSIERES (vertes) consecutives
  * Suivie d'un push BAISSIER impulsif (rouges)
  * L'OB = ENSEMBLE des bougies vertes
  * Au retour du prix dans cette zone -> opportunite SHORT

Pipeline :
1. Identifier un push IMPULSIF
2. Remonter en arriere : trouver la sequence de bougies OPPOSEES qui le precede
3. Verifier sweep de liquidite par le push
4. Verifier que le prix revient dans la zone OB (mitigation)
5. Definir entry au mid de la zone OB, SL au-dela, TP vers liquidite opposee
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot.detectors.swings import Swing, detect_swings

PushDirection = Literal["bullish", "bearish"]


@dataclass
class SweepWickOB:
    """Un OB ICT correct.

    L'OB = ENSEMBLE des bougies de meme couleur que le push, dans le push.
    - Push bullish (trade SHORT) : OB = bougies vertes du push
    - Push bearish (trade LONG) : OB = bougies rouges du push
    zone_high/low = high max / low min de ce BLOC de bougies.
    """
    direction: PushDirection           # direction du TRADE (oppose au push)
    push_start_index: int              # premiere bougie du push (range)
    push_end_index: int                # derniere bougie du push (range)
    push_start_time: pd.Timestamp
    push_end_time: pd.Timestamp

    # Le bloc OB (peut faire 1 a 5 bougies)
    ob_candle_index: int               # bougie principale (= derniere du bloc)
    ob_candle_time: pd.Timestamp
    ob_block_start: int                # premiere bougie du bloc
    ob_block_end: int                  # derniere bougie du bloc
    zone_high: float                   # high max du bloc
    zone_low: float                    # low min du bloc

    # Moment ou le prix revient toucher le mid OB (= VRAI entry time)
    fill_index: int
    fill_time: pd.Timestamp

    # Liquidite sweepee
    swept_level: float
    swept_swing: Swing | None

    # Push pour reference (toujours utile pour scoring)
    push_high: float                   # max high du push
    push_low: float                    # min low du push
    push_strength: float               # taille du push en pts

    @property
    def zone_mid(self) -> float:
        return (self.zone_high + self.zone_low) / 2

    @property
    def zone_size(self) -> float:
        return self.zone_high - self.zone_low


def detect_sweep_wick_obs(
    df: pd.DataFrame,
    swings: list[Swing],
    min_push_candles: int | None = None,
    max_push_candles: int | None = None,
    min_push_atr_mult: float | None = None,
    min_push_points: float | None = None,
    sweep_buffer_pct: float = 0.0,
    max_return_lookforward: int | None = None,
    dedup_window_candles: int | None = None,
    use_tuning: bool = True,
) -> list[SweepWickOB]:
    # Si use_tuning, on charge les params de la generation courante
    if use_tuning:
        try:
            from bot.retrain import load_current_params
            p = load_current_params()
            if min_push_candles is None: min_push_candles = p.get("min_push_candles", 2)
            if max_push_candles is None: max_push_candles = p.get("max_push_candles", 8)
            if min_push_atr_mult is None: min_push_atr_mult = p.get("min_push_atr_mult", 2.0)
            if max_return_lookforward is None: max_return_lookforward = p.get("max_return_lookforward", 30)
            if dedup_window_candles is None: dedup_window_candles = p.get("dedup_window_candles", 20)
        except Exception:
            pass
    # Defaults si toujours None
    if min_push_candles is None: min_push_candles = 2
    if max_push_candles is None: max_push_candles = 8
    if min_push_atr_mult is None: min_push_atr_mult = 2.0
    if max_return_lookforward is None: max_return_lookforward = 30
    if dedup_window_candles is None: dedup_window_candles = 20
    """Detecte tous les Sweep Wick OBs dans le df.

    Args:
        df: OHLC.
        swings: swings deja detectes (pour identifier les liquidites sweepees).
        min_push_candles / max_push_candles: taille du push acceptable.
        min_push_atr_mult: minimum impulsivity du push.
    """
    if len(df) < 30:
        return []

    # Si pas de min_push_points fourni, on derive du prix moyen :
    # On exige un push d'au moins 0.10% du prix (= 5pts a 5000, 0.10$ a 100 pour USOIL)
    if min_push_points is None:
        avg_price = float(df["close"].mean())
        min_push_points = avg_price * 0.0010

    obs: list[SweepWickOB] = []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    times = df.index

    # Pre-calcul des SWING significatifs (left/right >= 5) pour exiger un VRAI sweep
    # (pas juste un mini-swing local de 3 bougies)
    major_swings = [s for s in swings if hasattr(s, 'kind')]

    # ATR rapide (14 bougies)
    def atr_at(i: int) -> float:
        if i < 14:
            return float(df.iloc[:max(1, i)]["high"].sub(df.iloc[:max(1, i)]["low"]).mean() or 0)
        return float(df.iloc[i - 14:i]["high"].sub(df.iloc[i - 14:i]["low"]).mean() or 0)

    # Scan : pour chaque bougie i, regarder si on a un push qui finit la
    for i in range(min_push_candles, len(df) - 5):
        atr = atr_at(i)
        if atr <= 0:
            continue

        # Tente plusieurs tailles de push (de min a max)
        for push_size in range(min_push_candles, min(max_push_candles + 1, i + 1)):
            push_start = i - push_size + 1
            push_end = i
            if push_start < 0:
                continue

            sub_highs = highs[push_start:push_end + 1]
            sub_lows = lows[push_start:push_end + 1]
            push_high = float(sub_highs.max())
            push_low = float(sub_lows.min())
            push_range = push_high - push_low
            if push_range < min_push_atr_mult * atr:
                continue
            if push_range < min_push_points:
                continue  # filtre absolu en points (evite micro-bruit)

            # Direction du push : net move haussier ou baissier
            net_move = closes[push_end] - opens[push_start]
            if abs(net_move) < min_push_atr_mult * atr * 0.7:
                continue
            if abs(net_move) < min_push_points * 0.5:
                continue

            push_dir: PushDirection = "bullish" if net_move > 0 else "bearish"
            # Trade dans le sens OPPOSE
            trade_dir: PushDirection = "bearish" if push_dir == "bullish" else "bullish"

            # REGLE STRICTE ICT : SWEEP OBLIGATOIRE d'une VRAIE liquidite VISIBLE
            # Le swing a sweep doit etre :
            # - structurel (left/right >= 5)
            # - age >= 30 bougies (a vraiment "tenu" et accumule des stops)
            # - VISIBLE : dans les 150 dernieres bougies (~2h30 M1)
            # - PROCHE EN PRIX : pas plus loin que 2x ATR de l'extreme du push
            target_kind = "high" if push_dir == "bullish" else "low"

            major_swings_local = detect_swings(df.iloc[:push_start], left=5, right=5)
            recent_swings = [s for s in major_swings_local
                             if s.kind == target_kind
                             and s.index >= max(0, push_start - 150)  # plus restrictif : visible recent
                             and (push_start - s.index) >= 30]
            if not recent_swings:
                continue

            # Filtre PROXIMITE en prix : le swing doit etre proche du push
            # (sinon ce n'est pas la liquidite que le push visait reellement)
            push_extremum = push_high if push_dir == "bullish" else push_low
            visible_swings = []
            for s in recent_swings:
                price_distance = abs(s.price - push_extremum)
                if price_distance <= atr * 2.5:   # max 2.5x ATR
                    visible_swings.append(s)
            if not visible_swings:
                continue
            recent_swings = visible_swings

            # On veut le swing le plus PROCHE en prix du push (= celui qui a ete sweep)
            swept_level = None
            swept_swing = None
            for s in recent_swings:
                if push_dir == "bullish":
                    # Le push doit DEPASSER ce swing high
                    if push_high <= s.price:
                        continue
                    # Et la meche du push doit etre DERRIERE le swing (sweep visible)
                    if push_high - s.price < atr * 0.2:
                        continue   # depassement insignifiant
                else:
                    if push_low >= s.price:
                        continue
                    if s.price - push_low < atr * 0.2:
                        continue

                # On prend le swing le plus proche
                if swept_level is None:
                    swept_level = s.price
                    swept_swing = s
                else:
                    if push_dir == "bullish":
                        if s.price < swept_level and s.price > push_high - atr * 3:
                            swept_level = s.price
                            swept_swing = s
                    else:
                        if s.price > swept_level and s.price < push_low + atr * 3:
                            swept_level = s.price
                            swept_swing = s

            if swept_level is None:
                continue

            # EXIGENCE STRICTE : sweep distance >= 50% ATR (= sweep VISIBLE, pas du bruit)
            if push_dir == "bullish":
                sweep_dist = push_high - swept_level
            else:
                sweep_dist = swept_level - push_low
            if sweep_dist < atr * 0.5:
                continue

            # EXIGENCE STRICTE : le sweep doit etre suivi d'un RETOURNEMENT
            # = la bougie qui sweep doit cloturer du COTE OPPOSE (sous le swing pour bullish push)
            # MECHE qui depasse mais CLOSE en-dessous = vrai liquidity grab
            if push_dir == "bullish":
                idx_max = int(highs[push_start:push_end + 1].argmax()) + push_start
                # La bougie au high max doit avoir une meche superieure significative
                wick_top = highs[idx_max] - max(opens[idx_max], closes[idx_max])
                body_size = abs(closes[idx_max] - opens[idx_max])
                if wick_top < body_size * 0.5:   # mèche superieure < 50% du corps
                    continue
            else:
                idx_min = int(lows[push_start:push_end + 1].argmin()) + push_start
                wick_bottom = min(opens[idx_min], closes[idx_min]) - lows[idx_min]
                body_size = abs(closes[idx_min] - opens[idx_min])
                if wick_bottom < body_size * 0.5:
                    continue

            # (rejet visible deja verifie ci-dessus via la meche)

            # VRAIE LOGIQUE ICT : OB = SEQUENCE CONSECUTIVE de bougies AVANT le push impulsif
            #
            # - Trade LONG (push bearish a sweep le low) :
            #   OB haussier = bougies ROUGES consecutives JUSTE AVANT que le push haussier post-sweep commence
            #   Mais ici notre "push" est ce qui a fait le sweep, donc bearish.
            #   Apres le sweep, on attend un MSS haussier dont l'origine = OB rouge.
            #
            # - Trade SHORT (push bullish a sweep le high) :
            #   OB baissier = bougies VERTES consecutives juste AVANT le retournement baissier
            #
            # Approche pragmatique : on identifie le bloc de bougies CONSECUTIVES de la
            # couleur OPPOSEE au TRADE, JUSTE AVANT l'extremum du push.
            # - Trade LONG : on cherche les bougies ROUGES consecutives qui precedent
            #   immediatement le LOW du push (= le bottom)
            # - Trade SHORT : on cherche les bougies VERTES consecutives qui precedent
            #   immediatement le HIGH du push (= le top)

            # Trouve l'extremum du push (= bougie pivot du retournement)
            if push_dir == "bullish":
                # Trade SHORT : pivot = bougie avec le high max (le sommet)
                pivot_idx = int(highs[push_start:push_end + 1].argmax()) + push_start
                # OB SHORT = bougies VERTES consecutives JUSTE AVANT le pivot (ou incluant le pivot si vert)
                target_color_is_bull = True
            else:
                # Trade LONG : pivot = bougie avec le low min (le creux)
                pivot_idx = int(lows[push_start:push_end + 1].argmin()) + push_start
                # OB LONG = bougies ROUGES consecutives JUSTE AVANT le pivot
                target_color_is_bull = False

            # Trouve TOUT le bloc de bougies consecutives de la bonne couleur
            # autour du pivot (avant ET apres le pivot dans la limite du push).
            #
            # 1. Si le pivot n'est pas de la bonne couleur, on cherche la bougie
            #    de la bonne couleur la plus proche dans le push.
            # 2. Une fois trouve, on remonte AVANT et on descend APRES pour capturer
            #    toute la sequence consecutive.

            # Trouve la bougie pivot effective (de la bonne couleur)
            pivot_color_match = (closes[pivot_idx] > opens[pivot_idx]) == target_color_is_bull
            start_idx = pivot_idx
            if not pivot_color_match:
                # Cherche la bougie de la bonne couleur la plus proche dans le push
                best = None
                for k in range(push_start, push_end + 1):
                    if (closes[k] > opens[k]) == target_color_is_bull:
                        if best is None or abs(k - pivot_idx) < abs(best - pivot_idx):
                            best = k
                if best is None:
                    continue
                start_idx = best

            # Remonte AVANT le start_idx pour chercher les bougies consecutives de meme couleur
            # ON ETEND AU-DELA du push_start si on a encore des bougies de bonne couleur
            block_start = start_idx
            # Limite remontee : on accepte 3 bougies avant push_start aussi
            min_k = max(0, push_start - 3)
            for k in range(start_idx - 1, min_k - 1, -1):
                if (closes[k] > opens[k]) == target_color_is_bull:
                    block_start = k
                else:
                    break

            # Descend APRES le start_idx - on accepte aussi 3 bougies apres push_end
            max_k = min(len(df) - 1, push_end + 3)
            block_end = start_idx
            for k in range(start_idx + 1, max_k + 1):
                if (closes[k] > opens[k]) == target_color_is_bull:
                    block_end = k
                else:
                    break

            block = list(range(block_start, block_end + 1))

            if not block:
                continue   # pas de bougies OB consecutives = pas valide

            # REGLE STRICTE 1 : OB doit faire AU MOINS 3 bougies consecutives
            # (Vizion : un vrai OB c'est une SEQUENCE, pas 1-2 bougies isolees)
            if len(block) < 3:
                continue

            # REGLE STRICTE 2 : OB doit avoir une VOLATILITE minimum
            # La zone OB doit faire au moins 40% de l'ATR pour ne pas etre du bruit
            ob_zone_size = float(max(highs[i] for i in block) - min(lows[i] for i in block))
            if ob_zone_size < atr * 0.4:
                continue   # OB trop plate = pas de vraie absorption

            # REGLE STRICTE 3 : corps moyen significatif
            # La moyenne des corps des bougies OB >= 25% de l'ATR
            avg_body = sum(abs(closes[i] - opens[i]) for i in block) / len(block)
            if avg_body < atr * 0.25:
                continue   # bougies trop molles, pas d'impulsion

            ob_idx = block[-1]   # bougie principale = derniere du bloc
            ob_high_precise = float(max(highs[i] for i in block))
            ob_low_precise = float(min(lows[i] for i in block))
            ob_mid_precise = (ob_high_precise + ob_low_precise) / 2

            # REGLE ICT : l'OB doit etre COMBLE PAR UNE BOUGIE POSTERIEURE au push.
            # On demarre la recherche apres push_end (pas apres ob_idx car push peut
            # avoir des bougies apres l'OB candle).
            search_start = max(ob_idx + 1, push_end + 1)
            after = df.iloc[search_start:search_start + max_return_lookforward]
            if after.empty:
                continue

            ob_size = ob_high_precise - ob_low_precise
            avg_price = float(closes[max(0, ob_idx - 20):ob_idx + 1].mean() or 1)
            # Zone OB "tres petite" = < 0.05% du prix (typique XAU)
            ob_is_small = ob_size < avg_price * 0.0005

            comble = False
            fill_index = None
            fill_time = None
            outside_ob = False
            ob_broken = False    # REGLE ICT VIZION : OB casse = setup invalide
            # On tolere un petit buffer pour les meches qui touchent juste l'extreme
            # (sinon trop strict, beaucoup de faux rejets sur le bruit)
            tol = max(ob_high_precise - ob_low_precise, 0.0001) * 0.05   # 5% de la zone

            for ts, row in after.iterrows():
                # === REGLE INVALIDATION OB (MECHE OU CLOSE) ===
                # Trade LONG : OB invalide si LOW de la bougie descend sous ob_low (meche test)
                # Trade SHORT : OB invalide si HIGH de la bougie depasse ob_high (meche test)
                if trade_dir == "bullish":
                    if row["low"] < ob_low_precise - tol:
                        ob_broken = True
                        break
                else:
                    if row["high"] > ob_high_precise + tol:
                        ob_broken = True
                        break

                # Si OB tres petite, on skip l'exigence "sortir d'abord"
                if not outside_ob and not ob_is_small:
                    if push_dir == "bullish":
                        if row["close"] < ob_low_precise:
                            outside_ob = True
                    else:
                        if row["close"] > ob_high_precise:
                            outside_ob = True
                    continue
                # OB touchee au mid
                if row["low"] <= ob_mid_precise <= row["high"]:
                    comble = True
                    fill_time = ts
                    fill_index = int(df.index.get_loc(ts))
                    break

            if ob_broken:
                continue   # OB casse (meche ou corps) avant comblement -> setup invalide

            if not comble:
                continue

            # On a deja calcule ob_idx (derniere bougie du bloc OB), ob_high/low du bloc
            ob_high = ob_high_precise
            ob_low = ob_low_precise
            block_start_idx = block[0]
            block_end_idx = block[-1]

            obs.append(SweepWickOB(
                direction=trade_dir,
                push_start_index=push_start,
                push_end_index=push_end,
                push_start_time=times[push_start],
                push_end_time=times[push_end],
                ob_candle_index=ob_idx,
                ob_candle_time=times[ob_idx],
                ob_block_start=block_start_idx,
                ob_block_end=block_end_idx,
                zone_high=ob_high,
                zone_low=ob_low,
                fill_index=fill_index,
                fill_time=fill_time,
                swept_level=swept_level,
                swept_swing=swept_swing,
                push_high=push_high,
                push_low=push_low,
                push_strength=push_range,
            ))
            break  # on trouve un push pour cette bougie i, on passe

    # Dedup AGGRESSIF : 1 seul OB par fenetre temporelle (par direction)
    # On garde le push le plus FORT dans chaque fenetre.
    obs.sort(key=lambda o: (o.push_end_index, -o.push_strength))
    unique: list[SweepWickOB] = []
    for ob in obs:
        too_close = False
        for kept in unique:
            if kept.direction == ob.direction and \
               abs(kept.push_end_index - ob.push_end_index) < dedup_window_candles:
                # Si l'autre est plus fort, on skip celui-ci
                if kept.push_strength >= ob.push_strength:
                    too_close = True
                    break
                else:
                    # Sinon on retire l'autre (on remplacera par celui-ci)
                    unique.remove(kept)
                    break
        if not too_close:
            unique.append(ob)

    unique.sort(key=lambda o: o.push_end_index)
    return unique
