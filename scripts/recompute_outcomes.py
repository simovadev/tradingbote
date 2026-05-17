"""Recalcule les outcomes des trades labels humains avec la nouvelle logique."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from db.models import HumanLabeledTrade, SessionLocal, init_db
from web.server import _compute_outcome


def main():
    init_db()
    with SessionLocal() as s:
        trades = s.execute(select(HumanLabeledTrade)).scalars().all()
        print(f"\nRecalcul de {len(trades)} trades...\n")
        for t in trades:
            # Si on a un OB candles, on devine entry_time = derniere bougie + 1 min
            entry_time_iso = None
            if t.ob_candles and t.ob_candles.get("candles"):
                last = t.ob_candles["candles"][-1]
                from datetime import datetime, timezone, timedelta
                entry_time_iso = datetime.fromtimestamp(last["time"] + 60, tz=timezone.utc).isoformat()

            old_outcome = t.outcome
            new_outcome, exit_time, exit_price = _compute_outcome(
                t.instrument, t.day.isoformat(), t.direction,
                t.entry_price, t.stop_loss, t.take_profit,
                entry_time_iso,
            )

            change = "" if old_outcome == new_outcome else f"  [CHANGE {old_outcome} -> {new_outcome}]"
            print(f"  Trade #{t.id} {t.direction:<8} entry={t.entry_price:.2f}  "
                  f"old={old_outcome:<8} new={new_outcome}{change}")

            t.outcome = new_outcome
            if exit_time:
                t.exit_time = exit_time.replace(tzinfo=None) if hasattr(exit_time, 'replace') else exit_time
            t.exit_price = exit_price

        s.commit()
        print(f"\nOK, {len(trades)} trades mis a jour")


if __name__ == "__main__":
    main()
