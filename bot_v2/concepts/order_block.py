"""Order Block — detection STRICTE Vizion (bible §2 + decision user 2026-05-15).

Definition stricte :
- **OB bullish** = N bougies BAISSIERES CONSECUTIVES (N>=1) qui prennent une
  liquidite externe (sweep d'un swing low). Validation = cloture corps de bougie
  AU-DESSUS du HIGH du groupe.
  ⚠ Aucune bougie haussiere "intruse" toleree au milieu du groupe (decision user).

- **OB bearish** = symetrique avec N bougies HAUSSIERES consecutives, sweep d'un
  swing high, cloture corps SOUS le LOW du groupe.

Stop Loss : MECHE (decision user 2026-05-15).
- OB bullish : SL sous la meche basse du groupe (low du groupe).
- OB bearish : SL au-dessus de la meche haute du groupe (high du groupe).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from bot_v2.concepts.liquidity import Sweep, Swing, find_sweeps, find_swings


OBDirection = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class OrderBlock:
    """Un Order Block valide selon les regles Vizion strictes."""
    direction: OBDirection                # "bullish" | "bearish"
    # Bornes du GROUPE de bougies inverses (l'OB lui-meme)
    group_start_index: int                # iloc premiere bougie du groupe
    group_end_index: int                  # iloc derniere bougie du groupe (inclusif)
    group_start_ts: pd.Timestamp
    group_end_ts: pd.Timestamp
    # Niveaux de prix de l'OB
    ob_high: float                        # plus haut du groupe (meches incluses)
    ob_low: float                         # plus bas du groupe (meches incluses)
    ob_open: float                        # open de la 1ere bougie du groupe
    ob_close: float                       # close de la derniere bougie du groupe
    # La bougie de VALIDATION (clos corps au-dessus du high pour bullish)
    validation_index: int
    validation_ts: pd.Timestamp
    validation_close: float
    # Sweep qui a precede / declenche l'OB (liquidite externe prise)
    sweep: Sweep
    # Taille du groupe en bougies
    group_size: int


def _is_bearish(row) -> bool:
    """Une bougie est baissiere si close < open."""
    return row["close"] < row["open"]


def _is_bullish(row) -> bool:
    return row["close"] > row["open"]


def detect_order_blocks(
    df: pd.DataFrame,
    swings: list[Swing] | None = None,
    sweeps: list[Sweep] | None = None,
    swing_strength: int = 2,
    max_group_size: int = 5,
    max_bars_after_sweep: int = 10,
    min_sweep_depth_atr: float = 0.0,
) -> list[OrderBlock]:
    """Detecte tous les OB valides dans le DataFrame.

    Logique :
    1. Pour chaque sweep (prise de liquidite) :
       - Si sweep BULLISH (low pris) -> on cherche un OB bullish
         = groupe de bougies BAISSIERES consecutives juste avant le retournement,
           puis cloture corps au-dessus du HIGH du groupe.
       - Si sweep BEARISH (high pris) -> symetrique.
    2. Le groupe doit etre 100% du bon sens (decision user : strict, pas d'intruse).
    3. La validation doit arriver dans les `max_bars_after_sweep` bougies apres le sweep.

    Args:
        df: DataFrame avec colonnes open, high, low, close.
        swings, sweeps: optionnels, recalcules sinon.
        swing_strength: passage a find_swings si recalcul.
        max_group_size: garde-fou (bible : pas plus de 5 bougies sinon = range).
        max_bars_after_sweep: validation doit arriver vite, sinon OB ignore.
    """
    if swings is None:
        swings = find_swings(df, strength=swing_strength)
    if sweeps is None:
        sweeps = find_sweeps(df, swings, min_depth_atr=min_sweep_depth_atr)

    obs: list[OrderBlock] = []

    for sweep in sweeps:
        if sweep.direction == "bullish":
            ob = _try_bullish_ob(df, sweep, max_group_size, max_bars_after_sweep)
        else:
            ob = _try_bearish_ob(df, sweep, max_group_size, max_bars_after_sweep)
        if ob is not None:
            obs.append(ob)

    return deduplicate_obs(obs)


def deduplicate_obs(obs: list[OrderBlock]) -> list[OrderBlock]:
    """Supprime les OB doublons (meme groupe de bougies, meme direction).

    On considere 2 OB doublons s'ils ont :
    - Meme direction
    - Meme group_start_index et group_end_index
    OU
    - Chevauchement complet et meme validation_index

    On garde le premier (= celui qui vient du sweep le plus pertinent : le 1er detecte).
    """
    seen: set[tuple] = set()
    out: list[OrderBlock] = []
    for ob in obs:
        # Cle unique = direction + bornes du groupe + validation
        key = (ob.direction, ob.group_start_index, ob.group_end_index, ob.validation_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(ob)
    return out


def _try_bullish_ob(
    df: pd.DataFrame,
    sweep: Sweep,
    max_group_size: int,
    max_bars_after_sweep: int,
) -> OrderBlock | None:
    """Cherche un OB bullish autour d'un sweep bullish (low pris).

    Structure attendue :
        [N bougies baissieres consecutives] [bougie qui sweep le low] [validation : close > high du groupe]
    OU
        [N bougies baissieres incluant la sweep] [validation]

    Note Vizion 5fv2MjuPKE4 : c'est souvent LA bougie qui sweep qui FORME la fin
    du groupe baissier. Sa meche depasse le swing low, mais elle CLOS au-dessus
    (= rejet). Cette bougie peut etre baissiere ou haussiere.

    On retient : le GROUPE = bougies baissieres consecutives terminant la
    descente. La bougie de sweep peut etre dedans (si baissiere) ou juste apres.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    sweep_idx = sweep.sweep_index

    # 1. Trouver le GROUPE baissier qui se termine a (ou juste avant) sweep_idx
    # On remonte depuis sweep_idx tant que les bougies sont baissieres.
    group_end = sweep_idx
    # La bougie de sweep est-elle baissiere ?
    if closes[sweep_idx] >= opens[sweep_idx]:
        # Bougie de sweep haussiere : le groupe finit a sweep_idx - 1
        group_end = sweep_idx - 1
    if group_end < 0:
        return None
    if closes[group_end] >= opens[group_end]:
        # Pas de bougie baissiere juste avant -> pas de groupe -> pas d'OB
        return None

    # Remonte les baissieres consecutives
    group_start = group_end
    while group_start > 0 and closes[group_start - 1] < opens[group_start - 1]:
        if (group_end - (group_start - 1) + 1) > max_group_size:
            break
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < 1 or group_size > max_group_size:
        return None

    # 2. Niveaux du groupe (meches incluses)
    ob_high = float(highs[group_start:group_end + 1].max())
    ob_low = float(lows[group_start:group_end + 1].min())

    # 3. Cherche la validation : close > ob_high dans les `max_bars_after_sweep` bougies
    # apres le sweep (en partant de sweep_idx, ou apres si sweep est dans le groupe).
    val_start = max(sweep_idx, group_end) + 1
    val_end = min(val_start + max_bars_after_sweep, len(df))
    validation_idx = None
    for j in range(val_start, val_end):
        if closes[j] > ob_high:
            validation_idx = j
            break

    if validation_idx is None:
        return None

    return OrderBlock(
        direction="bullish",
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        ob_high=ob_high,
        ob_low=ob_low,
        ob_open=float(opens[group_start]),
        ob_close=float(closes[group_end]),
        validation_index=int(validation_idx),
        validation_ts=df.index[validation_idx],
        validation_close=float(closes[validation_idx]),
        sweep=sweep,
        group_size=int(group_size),
    )


def _try_bearish_ob(
    df: pd.DataFrame,
    sweep: Sweep,
    max_group_size: int,
    max_bars_after_sweep: int,
) -> OrderBlock | None:
    """Symetrique : sweep d'un high -> OB bearish (bougies haussieres consecutives)."""
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    sweep_idx = sweep.sweep_index

    group_end = sweep_idx
    if closes[sweep_idx] <= opens[sweep_idx]:
        group_end = sweep_idx - 1
    if group_end < 0:
        return None
    if closes[group_end] <= opens[group_end]:
        return None

    group_start = group_end
    while group_start > 0 and closes[group_start - 1] > opens[group_start - 1]:
        if (group_end - (group_start - 1) + 1) > max_group_size:
            break
        group_start -= 1

    group_size = group_end - group_start + 1
    if group_size < 1 or group_size > max_group_size:
        return None

    ob_high = float(highs[group_start:group_end + 1].max())
    ob_low = float(lows[group_start:group_end + 1].min())

    val_start = max(sweep_idx, group_end) + 1
    val_end = min(val_start + max_bars_after_sweep, len(df))
    validation_idx = None
    for j in range(val_start, val_end):
        if closes[j] < ob_low:
            validation_idx = j
            break

    if validation_idx is None:
        return None

    return OrderBlock(
        direction="bearish",
        group_start_index=int(group_start),
        group_end_index=int(group_end),
        group_start_ts=df.index[group_start],
        group_end_ts=df.index[group_end],
        ob_high=ob_high,
        ob_low=ob_low,
        ob_open=float(opens[group_start]),
        ob_close=float(closes[group_end]),
        validation_index=int(validation_idx),
        validation_ts=df.index[validation_idx],
        validation_close=float(closes[validation_idx]),
        sweep=sweep,
        group_size=int(group_size),
    )


# ============ HELPERS ============

SL_LOOKBACK_BARS = 5  # decision user 2026-05-16 : 5 bougies AVANT le groupe pour SL protecteur


def ob_stop_loss(ob: OrderBlock, df: 'pd.DataFrame | None' = None) -> float:
    """SL = extreme depuis le SWEEP jusqu'a la VALIDATION (user 2026-05-17).

    BUG CRITIQUE FIX : avant on s'arretait a group_end, mais les bougies entre
    group_end et validation_index peuvent atteindre des extremes BIEN PLUS HAUTS
    (ex: BTCUSD SELL 2025-08-20 : group_end high=113316, validation+1 high=113378).
    Si SL ne couvre pas cette zone, il est PRIS en live avant la validation.

    Logique ICT pro :
    - Bougie de SWEEP (point ou la liquidite a ete prise)
    - + toutes les bougies entre sweep et VALIDATION (incl. group + bougies de
      poussee/manipulation avant cassure)
    - + padding SL_LOOKBACK_BARS=5 bougies AVANT le sweep
    - SL = pire extreme dans cette fenetre

    - OB bullish  : SL = MIN des lows
    - OB bearish  : SL = MAX des highs

    Si df n'est pas fourni : fallback meche du groupe seul (compat ancien).
    """
    if df is None:
        # Fallback : meche groupe seul (compat)
        if ob.direction == "bullish":
            return ob.ob_low
        return ob.ob_high

    # Point de depart : le sweep (si dispo) sinon group_start
    sweep_idx = getattr(ob.sweep, "sweep_index", None) if ob.sweep is not None else None
    if sweep_idx is None:
        sweep_idx = ob.group_start_index

    # Fenetre : 5 bougies avant sweep -> validation_index (inclusif)
    # Couvre group + bougies de manipulation entre group et validation
    lookback_start = max(0, sweep_idx - SL_LOOKBACK_BARS)
    end_idx = max(ob.group_end_index, ob.validation_index)
    sub = df.iloc[lookback_start:end_idx + 1]
    if len(sub) == 0:
        # Fallback meche
        if ob.direction == "bullish":
            return ob.ob_low
        return ob.ob_high

    if ob.direction == "bullish":
        return float(sub["low"].min())
    return float(sub["high"].max())


def ob_entry(ob: OrderBlock) -> float:
    """Entree de l'OB : prix d'entree quand le price revient dans la zone.

    Vizion privilégie l'entree au CORPS de bougie (open du groupe = milieu de l'OB).
    Pour l'instant on prend le HIGH (pour un bullish OB) ou le LOW (pour un bearish OB)
    comme niveau d'entree limite : le price doit retomber dans la zone OB.
    """
    if ob.direction == "bullish":
        return ob.ob_high
    return ob.ob_low


def ob_is_invalidated(ob: OrderBlock, df: pd.DataFrame) -> bool:
    """Verifie si l'OB a ete invalide apres sa validation.

    Bullish : invalide si une bougie clos SOUS le low du groupe (corps, pas meche).
    Bearish : invalide si une bougie clos AU-DESSUS du high du groupe.
    """
    after = df.iloc[ob.validation_index + 1:]
    if ob.direction == "bullish":
        return bool((after["close"] < ob.ob_low).any())
    return bool((after["close"] > ob.ob_high).any())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from bot_v2.data_loader import load

    df = load("XAUUSD", "M5")
    print(f"XAUUSD M5 : {len(df)} bougies")

    # Test sur 7 derniers jours
    mask = df.index >= (df.index.max() - pd.Timedelta(days=7))
    df_week = df[mask]
    print(f"Derniers 7 jours : {len(df_week)} bougies")

    obs = detect_order_blocks(df_week)
    print(f"\nOrder Blocks detectes : {len(obs)}")
    print(f"  - Bullish : {sum(1 for o in obs if o.direction == 'bullish')}")
    print(f"  - Bearish : {sum(1 for o in obs if o.direction == 'bearish')}")

    print(f"\nGroup size repartition :")
    from collections import Counter
    sizes = Counter(o.group_size for o in obs)
    for s, c in sorted(sizes.items()):
        print(f"  {s} bougies : {c} OB")

    print(f"\n=== 5 derniers OB ===")
    for ob in obs[-5:]:
        sl = ob_stop_loss(ob)
        entry = ob_entry(ob)
        risk = abs(entry - sl)
        print(
            f"  {ob.validation_ts} | {ob.direction:8s} | "
            f"group={ob.group_size}b | entry={entry:.3f} | sl={sl:.3f} | risk={risk:.3f}"
        )

    # Verification stricte : tous les OB ont des bougies du bon sens
    print("\n=== Verification stricte (decision user 2026-05-15) ===")
    n_strict_ok = 0
    for ob in obs:
        group = df_week.iloc[ob.group_start_index:ob.group_end_index + 1]
        if ob.direction == "bullish":
            # Toutes les bougies du groupe doivent etre baissieres
            all_bearish = (group["close"] < group["open"]).all()
            if all_bearish:
                n_strict_ok += 1
        else:
            all_bullish = (group["close"] > group["open"]).all()
            if all_bullish:
                n_strict_ok += 1
    print(f"OB strictement valides : {n_strict_ok}/{len(obs)}")
    if n_strict_ok != len(obs):
        print("⚠ ATTENTION : certains OB ont des bougies intruses (bug)")
