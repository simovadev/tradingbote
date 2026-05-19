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
# User 2026-05-19 : objectif jusqu'a 10 ans (2016 -> 2026)
N_SCROLLS = 8000
# Delai entre scrolls : 0.003s = 3000 scrolls/s max, mais MT5 peut suivre 0.005s
DELAY = 0.003
# Direction scroll : NEGATIF = scroll vers le PASSE dans MT5
SCROLL_DIR = -20

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
print(f"Objectif : jusqu'a 10 ans d'historique (2016 -> 2026 ideal)")
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
        pyautogui.scroll(SCROLL_DIR)  # negatif = vers le passe
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  Scroll {i+1}/{N_SCROLLS}  ({elapsed:.0f}s)")
        time.sleep(DELAY)
except KeyboardInterrupt:
    print(f"\nInterrompu apres {i} scrolls")

print(f"\nTermine en {time.time()-start:.0f}s")
print("Va voir dans MT5 si l'historique va loin maintenant (regarde la date la plus a gauche du chart)")
