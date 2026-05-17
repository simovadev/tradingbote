"""Detection de sweeps multiples / consecutifs.

Quand plusieurs liquidites sont prises d'affilee (par ex. asian range low
PUIS london low) = liquidation forte = signal plus probabilisant.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.detectors.liquidity import Sweep


@dataclass
class MultiSweepCluster:
    sweeps: list[Sweep]
    side: str
    span_candles: int        # combien de bougies entre 1er et dernier sweep

    @property
    def count(self) -> int:
        return len(self.sweeps)


def find_sweep_clusters(sweeps: list[Sweep], max_gap: int = 30) -> list[MultiSweepCluster]:
    """Groupe les sweeps proches (meme cote, ecart <= max_gap bougies)."""
    if not sweeps:
        return []

    clusters: list[MultiSweepCluster] = []
    sorted_sweeps = sorted(sweeps, key=lambda s: s.candle_index)

    current: list[Sweep] = [sorted_sweeps[0]]
    for sweep in sorted_sweeps[1:]:
        last = current[-1]
        if sweep.side == last.side and (sweep.candle_index - last.candle_index) <= max_gap:
            current.append(sweep)
        else:
            if len(current) >= 2:
                clusters.append(MultiSweepCluster(
                    sweeps=current,
                    side=current[0].side,
                    span_candles=current[-1].candle_index - current[0].candle_index,
                ))
            current = [sweep]

    if len(current) >= 2:
        clusters.append(MultiSweepCluster(
            sweeps=current,
            side=current[0].side,
            span_candles=current[-1].candle_index - current[0].candle_index,
        ))

    return clusters


def is_in_cluster(sweep: Sweep, clusters: list[MultiSweepCluster]) -> MultiSweepCluster | None:
    """Retourne le cluster qui contient ce sweep, ou None."""
    for c in clusters:
        if sweep in c.sweeps:
            return c
    return None
