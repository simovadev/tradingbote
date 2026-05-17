"""Point d'entree TradingBot Trainer.

Usage :
    python main.py            # lance le serveur de training (mode par defaut)
    python main.py --reset    # reset la DB (efface tous les trades reviewes)
"""
from __future__ import annotations

import argparse

from db.models import init_db, reset_db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Drop + recreate la DB")
    args = ap.parse_args()

    if args.reset:
        print("Reset de la DB...")
        reset_db()
        print("OK")
        return

    init_db()
    from web.server import run
    run()


if __name__ == "__main__":
    main()
