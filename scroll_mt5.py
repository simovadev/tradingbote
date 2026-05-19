"""Auto-scroll MT5 chart vers le passe pour forcer le download historique.

Objectif user 2026-05-19 : 6 ans d'historique (2020 -> 2026) pour ML

Usage :
1. Ouvre MT5 desktop avec le chart M1 de l'actif a charger
2. Zoom out a fond ('-' x 20 fois)
3. Lance le script : python scroll_mt5.py [N_SCROLLS]
4. Tu as 5s pour cliquer sur le chart MT5 (le mettre au premier plan)
5. Le script scroll automatiquement (~1-2 min par 5000 scrolls)

Necessite : pip install pyautogui
"""
import pyautogui
import time
import sys

# Nombre de scrolls (chaque scroll = ~30-50 bougies en arriere selon le zoom)
# User 2026-05-19 : objectif 6 ans (2020 -> 2026) au lieu de 4
# 6 ans M1 = ~3.1M bougies | si chaque scroll charge ~1000 bougies -> 3100 scrolls min
N_SCROLLS = 5000
# Delai entre scrolls (sinon MT5 peut bug)
DELAY = 0.015

if len(sys.argv) > 1:
    try:
        N_SCROLLS = int(sys.argv[1])
    except ValueError:
        pass

print("=" * 50)
print("AUTO-SCROLL MT5 - Force download historique")
print("=" * 50)
print()
print("INSTRUCTIONS :")
print("1. Ouvre MT5 desktop")
print("2. Ouvre le chart M1 de l'actif a charger (ex: XAUUSD+)")
print("3. Zoom out a fond (touche '-' x 20 fois)")
print("4. Apres ce message, tu as 5s pour cliquer sur le chart")
print()
print(f"Nombre de scrolls : {N_SCROLLS} (~{N_SCROLLS * DELAY:.0f}s)")
print(f"Objectif : 6 ans d'historique (2020 -> 2026)")
print()
input("Appuie sur Entree pour demarrer le compte a rebours...")

for i in range(5, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

print(f"\nScroll en cours ({N_SCROLLS} steps, ~{N_SCROLLS * DELAY:.0f}s)...")
print("NE TOUCHE PLUS A LA SOURIS / CLAVIER")
print()

start = time.time()
try:
    for i in range(N_SCROLLS):
        pyautogui.scroll(20)  # scroll up = vers le passe
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"  Scroll {i+1}/{N_SCROLLS}  ({elapsed:.0f}s)")
        time.sleep(DELAY)
except KeyboardInterrupt:
    print(f"\nInterrompu apres {i} scrolls")

print(f"\nTermine en {time.time()-start:.0f}s")
print("Va voir dans MT5 si l'historique va loin maintenant (regarde la date la plus a gauche du chart)")
