"""Detection de confluence multi-OB.

Si 2 OB se chevauchent ou sont imbriques sur la meme zone, le setup est
considere "premium" (exemple 4 du trader).
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.detectors.order_blocks import OrderBlock


@dataclass
class OBConfluence:
    main_ob: OrderBlock
    confluent_obs: list[OrderBlock]   # OB qui chevauchent main_ob

    @property
    def count(self) -> int:
        return 1 + len(self.confluent_obs)


def _overlap(a: OrderBlock, b: OrderBlock) -> bool:
    return not (a.zone_high < b.zone_low or a.zone_low > b.zone_high)


def find_confluences(obs: list[OrderBlock]) -> list[OBConfluence]:
    """Pour chaque OB, cherche d'autres OB du meme sens qui chevauchent."""
    out: list[OBConfluence] = []
    used: set[int] = set()

    for i, main in enumerate(obs):
        if i in used:
            continue
        confluent: list[OrderBlock] = []
        for j in range(i + 1, len(obs)):
            if j in used:
                continue
            other = obs[j]
            if other.kind != main.kind:
                continue
            if abs(other.candle_index - main.candle_index) > 50:
                continue  # trop eloignes en temps
            if _overlap(main, other):
                confluent.append(other)
                used.add(j)
        if confluent:
            out.append(OBConfluence(main, confluent))
            used.add(i)

    return out


def confluence_for(ob: OrderBlock, confluences: list[OBConfluence]) -> OBConfluence | None:
    for c in confluences:
        if c.main_ob is ob or ob in c.confluent_obs:
            return c
    return None
