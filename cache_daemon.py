"""Cache Daemon — calcule en continu les structures lourdes (OB, FVG, breakers,
swings, structure_breaks) pour les 14 actifs et les ecrit dans data_cache/<ASSET>.pkl.

Le live_runner_v2 lit ce pickle au lieu de recalculer -> cache_build passe de
1.6s a ~10ms par actif (read seul). Latence cycle live : 24.8s -> ~3s attendu.

Le calcul est STRICTEMENT IDENTIQUE a celui du live actuel (memes fonctions,
memes parametres). Aucun risque de desalignement V12 OOS.

Architecture :
- Boucle : pour chaque actif, fetch les bougies (via DataBuffer ou MT5),
  calcule les 8 structures du cache, ecrit le pickle de maniere atomique.
- Sleep aligne sur close M1 (xx:00:02) pour minimiser le delai entre close
  et disponibilite du cache (le live lit a xx:00:03).
- Logging pour suivre la latence par actif et detecter les ralentissements.

Lancement :
    python cache_daemon.py                  # tous les 14 actifs
    python cache_daemon.py --assets XAUUSD  # 1 actif (debug)

Le live_runner_v2 doit etre modifie pour lire data_cache/<ASSET>.pkl avec
fallback (si pickle manquant ou > 2 min vieux -> recalcul direct).
"""
from __future__ import annotations
import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Chemin portable : racine = dossier du script (marche sur PC local et VPS)
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from bot_v2.concepts.order_block import detect_order_blocks
from bot_v2.concepts.liquidity import find_swings
from bot_v2.concepts.fvg import detect_fvg
from bot_v2.concepts.breaker import detect_breakers
from bot_v2.concepts.structure import detect_structure_breaks, detect_trend
from bot_v2.config import get_param
from bot_v2.mt5_executor import MT5Executor

LIVE_ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

N_BARS_M1 = 88000
N_BARS_M15 = 11000
N_BARS_H1 = 2800

CACHE_DIR = _ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cache_daemon")


def build_cache_for_asset(asset: str, mt5_exec: MT5Executor) -> dict | None:
    """Reproduit EXACTEMENT le cache_build de live_runner_v2.compute_asset.

    Returns un dict picklable, ou None si pas assez de donnees.
    """
    df_m1 = mt5_exec.get_bars(asset, "M1", N_BARS_M1, force_sync=True)
    if df_m1 is None or len(df_m1) < 200:
        return None
    df_m1 = df_m1.iloc[:-1]  # exclut bougie en cours (V12 fix bougie en cours)

    df_m15 = mt5_exec.get_bars(asset, "M15", N_BARS_M15)
    if df_m15 is None:
        return None
    df_m15 = df_m15.iloc[:-1]

    df_h1 = mt5_exec.get_bars(asset, "H1", N_BARS_H1)
    if df_h1 is None:
        return None
    df_h1 = df_h1.iloc[:-1]

    sws = get_param(asset, "swing_strength_m1", 2)

    # Bloc cache_build STRICTEMENT IDENTIQUE a live_runner_v2:431-442
    obs = detect_order_blocks(df_m1, swing_strength=sws)
    swings_ltf = find_swings(df_m1, strength=sws)
    fvgs_ltf = detect_fvg(df_m1)
    breakers_ltf = detect_breakers(df_m1)
    obs_htf = detect_order_blocks(df_m15)
    structure_breaks = detect_structure_breaks(df_m1, swings=swings_ltf, fvgs=fvgs_ltf)
    htf_trend = detect_trend(swings_ltf, lookback=6)
    obs_htf2 = detect_order_blocks(df_h1)

    return {
        # Metadata pour validation par le live
        "asset": asset,
        "computed_at": datetime.now(timezone.utc),
        "df_m1_last_ts": df_m1.index[-1],
        "df_m1_len": len(df_m1),
        "df_m15_last_ts": df_m15.index[-1],
        "df_h1_last_ts": df_h1.index[-1],
        "sws": sws,
        # Cache content (utilise par compute_asset)
        "obs": obs,
        "swings_ltf": swings_ltf,
        "fvgs_ltf": fvgs_ltf,
        "breakers_ltf": breakers_ltf,
        "obs_htf": obs_htf,
        "structure_breaks": structure_breaks,
        "htf_trend": htf_trend,
        "obs_htf2": obs_htf2,
    }


def write_cache_atomic(asset: str, data: dict, retries: int = 40, delay_s: float = 0.025) -> None:
    """Ecriture atomique : ecrit dans un fichier tmp puis rename.

    Sur Windows, os.replace(tmp, final) leve WinError 32/5 si `final` est ouvert
    en lecture par le bot a cet instant precis (Windows verrouille le fichier
    ouvert, contrairement a POSIX). La lecture du bot ne dure que quelques ms et
    n'a lieu qu'une fois/minute, donc on retry le rename avec backoff jusqu'a ce
    que la fenetre de lecture se libere. Plafond : ~retries*delay_s = ~1s, large
    devant la duree d'une lecture bot. Sur Linux ce retry n'est jamais utilise.
    """
    final = CACHE_DIR / f"{asset}.pkl"
    # tmp unique par PID pour eviter qu'un autre process ecrase notre tmp
    tmp = CACHE_DIR / f"{asset}.pkl.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(tmp, final)
            return
        except OSError as e:  # WinError 32/5 : fichier en cours de lecture
            last_err = e
            time.sleep(delay_s)
    # Echec apres tous les retries : nettoyer le tmp et propager
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise last_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", nargs="+", default=LIVE_ASSETS,
                   help="Actifs a calculer (defaut: les 14)")
    p.add_argument("--sleep_align", action="store_true",
                   help="Aligner sur close M1 (sleep jusqu'a xx:00:02)")
    p.add_argument("--once", action="store_true",
                   help="Un seul cycle puis exit (debug)")
    args = p.parse_args()

    log.info(f"=== Cache Daemon ===")
    log.info(f"Actifs : {args.assets}")
    log.info(f"Output : {CACHE_DIR}")

    mt5_exec = MT5Executor()
    if not mt5_exec.initialize():
        log.error("MT5 init FAIL — ouvrir MT5 et se connecter")
        sys.exit(1)
    log.info(f"MT5 OK, offset broker = {mt5_exec.broker_utc_offset_sec}s")

    cycle = 0
    try:
        while True:
            cycle += 1
            t_cycle = time.time()
            log.info(f"--- Cycle {cycle} ---")

            for asset in args.assets:
                t_asset = time.time()
                try:
                    cache = build_cache_for_asset(asset, mt5_exec)
                    if cache is None:
                        log.warning(f"  {asset:<8} skip (pas de bougies)")
                        continue
                    write_cache_atomic(asset, cache)
                    dt = (time.time() - t_asset) * 1000
                    log.info(f"  {asset:<8} OK {dt:>5.0f}ms  obs={len(cache['obs'])} "
                             f"fvg={len(cache['fvgs_ltf'])} brk={len(cache['breakers_ltf'])} "
                             f"sw={len(cache['swings_ltf'])} sb={len(cache['structure_breaks'])} "
                             f"last={cache['df_m1_last_ts']}")
                except Exception as e:
                    log.exception(f"  {asset:<8} FAIL : {e}")

            dt_cycle = time.time() - t_cycle
            log.info(f"--- Cycle {cycle} fini en {dt_cycle:.1f}s ---")

            if args.once:
                break

            # Sleep aligne sur close M1 : on veut etre PRET a xx:00:02
            # (le live lit a xx:00:03). Si on a fini avant la prochaine minute,
            # on attend. Sinon on enchaine direct.
            if args.sleep_align:
                now = datetime.now(timezone.utc)
                next_min = now.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
                # Cible xx:00:03 : assez tard pour que MT5 ait propage la bougie
                # close a xx:00, assez tot pour que le calcul (~2.5s) + ecriture
                # finisse avant que le live runner ne LISE le cache a xx:00:06.
                # Garantit que daemon et bot voient la MEME bougie -> cache hit.
                target = next_min + pd.Timedelta(seconds=3)
                sleep_s = (target - now).total_seconds()
                if sleep_s > 0:
                    log.info(f"sleep {sleep_s:.1f}s (target = {target.strftime('%H:%M:%S')})")
                    time.sleep(sleep_s)
            else:
                # Mode non aligne : on attend juste 5s entre cycles pour pas saturer MT5
                time.sleep(5)

    except KeyboardInterrupt:
        log.info("Arret manuel (Ctrl+C)")
    finally:
        mt5_exec.shutdown()
        log.info("Daemon arrete proprement")


if __name__ == "__main__":
    main()
