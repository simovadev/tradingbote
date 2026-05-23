"""Launcher : spawn 1 cache_daemon par actif (14 process paralleles).

Pourquoi : un daemon mono-process calcule les 14 actifs sequentiellement
(~39s/cycle), ce qui fait que les 13 derniers actifs ont un cache "vieux d'1
bougie" au moment ou le bot scanne a xx:06 -> fallback recalcul cote bot.

Avec 14 daemons independants, chacun calcule SON actif a xx:03 en parallele,
fini a xx:05.5 -> bot lit 14 caches frais a xx:06 -> hit ~100%.

Resilience :
  - poll() chaque enfant toutes les 3s
  - si un daemon meurt, on le relance (auto-restart)
  - cap : max 5 restarts en 60s par actif (sinon log error et on abandonne
    cet actif -> le bot retombera en recalcul ; pas de crash global)
  - Ctrl+C tue tous les enfants proprement

Logs : daemon_<ASSET>.log par actif (separes pour diagnostic).
"""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # meme interpreteur que le launcher
DAEMON_SCRIPT = str(ROOT / "cache_daemon.py")

ASSETS = [
    "XAUUSD", "NAS100", "GER40", "BTCUSD",
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY",
    "SP500", "DJ30", "UK100", "FRA40",
    "USDCAD", "USDCHF",
]

# Cap anti-boucle : si un daemon crash plus de N fois en WINDOW_S, on abandonne
MAX_RESTARTS_IN_WINDOW = 5
WINDOW_S = 60.0
POLL_INTERVAL_S = 3.0


def spawn(asset: str) -> subprocess.Popen:
    log_path = ROOT / f"daemon_{asset}.log"
    f = open(log_path, "ab", buffering=0)
    p = subprocess.Popen(
        [PY, "-u", DAEMON_SCRIPT, "--assets", asset, "--sleep_align"],
        stdout=f, stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    print(f"[{time.strftime('%H:%M:%S')}] spawn {asset:<8} pid={p.pid}  -> {log_path.name}")
    return p


def kill_all(procs: dict[str, subprocess.Popen]) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] arret de {len(procs)} daemons...")
    for asset, p in procs.items():
        if p.poll() is None:
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()
            except Exception:
                pass
    # Grace period puis kill -9 ceux qui resistent
    deadline = time.time() + 5.0
    while time.time() < deadline and any(p.poll() is None for p in procs.values()):
        time.sleep(0.2)
    for asset, p in procs.items():
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
    print(f"[{time.strftime('%H:%M:%S')}] tous arretes")


def main() -> None:
    print(f"=== launch_daemons : {len(ASSETS)} daemons paralleles ===")
    procs: dict[str, subprocess.Popen] = {a: spawn(a) for a in ASSETS}
    restart_history: dict[str, deque[float]] = {a: deque() for a in ASSETS}
    abandoned: set[str] = set()

    try:
        while True:
            time.sleep(POLL_INTERVAL_S)
            now = time.time()
            for asset in ASSETS:
                if asset in abandoned:
                    continue
                p = procs[asset]
                rc = p.poll()
                if rc is None:
                    continue  # vivant
                # Mort : evaluer si on relance
                hist = restart_history[asset]
                while hist and (now - hist[0]) > WINDOW_S:
                    hist.popleft()
                if len(hist) >= MAX_RESTARTS_IN_WINDOW:
                    print(f"[{time.strftime('%H:%M:%S')}] {asset:<8} ABANDONNE "
                          f"({MAX_RESTARTS_IN_WINDOW} crashes en {WINDOW_S:.0f}s) "
                          f"-> bot retombera en recalcul pour cet actif")
                    abandoned.add(asset)
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] {asset:<8} mort (rc={rc}) -> restart "
                      f"({len(hist)+1}/{MAX_RESTARTS_IN_WINDOW})")
                hist.append(now)
                procs[asset] = spawn(asset)
    except KeyboardInterrupt:
        kill_all(procs)


if __name__ == "__main__":
    main()
