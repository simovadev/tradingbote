# TradingBot Trainer — Mode entrainement humain ICT/SMC

Bot algo qui pioche un jour aleatoire dans le passe, trouve son meilleur trade,
et te le montre pour que **tu corriges**. Tes feedbacks + dessins sont logges,
exportables en markdown, et servent a calibrer manuellement le bot.

## Stack

- Python 3.12 + FastAPI + Lightweight Charts (TradingView OSS)
- SQLite local (training_trades + feedbacks)
- Donnees historiques : Dukascopy gratuit (XAUUSD)
- Pas de LLM, pas d'auto-learning : c'est TOI qui dis au dev quoi ajuster.

## Premiere installation

```bash
cd c:\Users\Shadow\TradingBot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Telecharge 30 jours de XAUUSD sur 6 timeframes (premiere fois = 15-30 min)
python -m data.fetch --days 30
```

## Lancer le serveur

```bash
python main.py
# -> http://localhost:8000
```

Clique sur **"Pioche un jour aleatoire"**, le bot :
1. Choisit un jour au hasard dans les donnees
2. Scanne ce jour avec son algo ICT/SMC complet
3. Affiche son meilleur trade + tout ce qu'il a vu (60+ facteurs scores)
4. Tu dessines (zone OB violette, lignes, fleches) si tu veux montrer le vrai setup
5. Tu ecris ton feedback texte
6. Tu cliques **Sauve + Suivant** -> nouveau jour

## Exporter les feedbacks

Bouton **"Exporter logs (markdown)"** en bas a gauche, ou directement :
http://localhost:8000/api/training/export/markdown

Le markdown contient tous les trades + ce que le bot a vu + tes feedbacks.
Tu peux le coller dans une conversation pour qu'on calibre le code ensemble.

## Reset complet

```bash
python main.py --reset   # efface tous les trades reviewes
```

## Regles dures encodees (le bot REJETTE si une seule casse)

1. OB forme dans une killzone (London / NY / Asia / Overlap)
2. Entree dans une killzone
3. Sweep de liquidite confirme
4. Break of Structure confirme
5. HTF non contradictoire
6. RR >= 2

## Facteurs scores (le bot voit TOUT, pondere)

- 🕐 Killzone exacte (Asia 8pts, London 15, NY AM 15, Overlap 20, NY PM 10)
- 📈 Alignement multi-TF H4/H1/M30/M15/M3/M1
- 💧 Sweep unique vs cluster multi-sweep
- 🔨 Displacement (strong/moderate/weak)
- 🟧 OB simple / OB de meche / Breaker Block
- 🟧 Confluence multi-OB
- 📊 FVG / iFVG en confluence
- 📍 Premium / Discount zone
- 🎯 1er touch / 2eme touch
- 📊 Volume du sweep vs moyenne
