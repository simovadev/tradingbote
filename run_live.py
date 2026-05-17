"""Lanceur du bot live.

Usage:
    python run_live.py           # Mode reel : place les ordres MT5
    python run_live.py --dry-run # Mode test : detecte les setups sans placer ordre
"""
import sys
sys.path.insert(0, 'c:/Users/Shadow/TradingBot')

from bot_v2.live_runner import run_live


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    print("=" * 60)
    print("BOT LIVE VIZION ICT/SMC - Multi-Asset")
    print("=" * 60)
    print(f"Mode : {'DRY RUN (no orders)' if dry_run else 'LIVE (real orders)'}")
    print("Ctrl+C pour arreter proprement")
    print("=" * 60)
    print()

    run_live(test_dry_run=dry_run)
