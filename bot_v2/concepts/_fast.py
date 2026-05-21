"""Helpers numpy partages pour les optims O(N) des fonctions ICT.

Pattern : remplacer les boucles `arr[start:] op seuil` (qui materialisent
toute la queue par iteration = O(N^2) en total) par une recherche en
fenetres croissantes (galloping). Le sweep/breaker/rebalance arrive
generalement vite apres l'evenement -> les 1-2 premieres fenetres
suffisent.
"""
from __future__ import annotations

import numpy as np

__all__ = ["first_breach"]


def first_breach(arr, start: int, threshold: float, cmp) -> int:
    """Trouve le 1er index i >= start ou cmp(arr[i], threshold) est True.

    Equivalent strict de :
        for i in range(start, len(arr)):
            if cmp(arr[i], threshold):
                return i
        return -1

    Args:
        arr: numpy array (lecture seule).
        start: index de depart (inclusif).
        threshold: valeur comparee.
        cmp: ufunc numpy a 2 arguments — np.greater / np.less /
             np.greater_equal / np.less_equal. STRICT (>) ou non-strict (>=)
             selon le besoin du caller.

    Returns:
        Index dans `arr` (>= start), ou -1 si aucun.
    """
    n = len(arr)
    if start >= n:
        return -1
    lo = start
    win = 256
    while lo < n:
        hi = min(lo + win, n)
        sub = arr[lo:hi]
        hits = cmp(sub, threshold)
        if hits.any():
            k = int(np.argmax(hits))
            return lo + k
        lo = hi
        win *= 2
    return -1
