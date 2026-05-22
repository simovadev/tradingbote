# Guide build ML sur Vast.ai — comment exploiter au max

Ce document est le **journal opérationnel** des builds ML (V11, V12, etc.) sur Vast.ai. À chaque build, mettre à jour les sections "Setup", "Pièges rencontrés", "Perf observée".

## TL;DR — commandes prêtes à coller

### 1. Setup nouvelle instance (5 min)

```bash
# Sur ton PC : verifier la cle SSH publique (qu'il faut donner a Vast)
cat ~/.ssh/vast_v8.pub

# Une fois l'instance lancee, recuperer l'IP/port (ex: ssh -p 41297 root@31.13.223.140)
# Test de connexion :
ssh -i ~/.ssh/vast_v8 -p <PORT> -o StrictHostKeyChecking=no root@<IP> "uname -a; nproc; free -g | head -2"

# Clone repo + install deps (uniquement les libs build, pas le live)
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "mkdir -p /workspace && cd /workspace && git clone https://github.com/simovadev/tradingbote.git TradingBot && cd TradingBot && git log -1 --oneline && pip install --quiet pandas==2.2.3 numpy==2.1.3 pyarrow==18.0.0 lightgbm scikit-learn"
```

### 2. Upload data_vantage (8 ans, ~850 MB, ~2 min)

Le repo GitHub contient une version **réduite** de `data_vantage/` (115 MB, 7 mois). Pour un build 8 ans il faut uploader la vraie version depuis le PC.

```bash
# Sur ton PC :
cd /c/Users/Shadow/TradingBot && tar -cf /c/Users/Shadow/data_vantage_full.tar data_vantage/
scp -i ~/.ssh/vast_v8 -P <PORT> /c/Users/Shadow/data_vantage_full.tar root@<IP>:/workspace/

# Sur Vast :
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "cd /workspace/TradingBot && rm -rf data_vantage && tar -xf /workspace/data_vantage_full.tar -C . && du -sh data_vantage/"
# -> doit dire ~850 MB
```

### 3. Lancer le build V12 en parallèle (tous les actifs en même temps)

**Le truc important** : ne pas faire `build_one(a)` en boucle (séquentiel = lent). Passer **les 14 actifs en une seule liste** à `build_dataset()` pour avoir 1 grande queue de ~5600 tâches que les workers piochent librement.

```bash
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "cd /workspace/TradingBot && cat > /workspace/run_v12_parallel.py <<'PYEOF'
import os, sys
os.environ['BUILD_DATA_DIR']='data_vantage'
os.environ['BUILD_V12_MODE']='1'
sys.path.insert(0, '.')
import pandas as pd
from bot_v2.ml_dataset import build_dataset

ALL_ASSETS = ['XAUUSD','NAS100','GER40','BTCUSD','EURUSD','GBPUSD','AUDUSD','USDJPY','SP500','DJ30','UK100','FRA40','USDCAD','USDCHF']
TRAIN_START = pd.Timestamp('2018-03-01', tz='UTC')
TRAIN_END = pd.Timestamp('2026-05-22', tz='UTC')
df = build_dataset(TRAIN_START, TRAIN_END, ALL_ASSETS, output_path=None, chunk_months=0.25, ltf='M1', version_suffix='_V12_VANTAGE')
print(f'TOTAL {len(df):,} lignes sur {len(ALL_ASSETS)} actifs')
PYEOF
nohup env N_WORKERS=256 BUILD_DATA_DIR=data_vantage BUILD_V12_MODE=1 python3 -u /workspace/run_v12_parallel.py > /workspace/build_v12.log 2>&1 < /dev/null & disown"
```

### 4. Suivre l'avancement

```bash
# Tableau de bord (a relancer a la main quand tu veux)
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "echo === TERMINES ===; grep '>>> RECAP' /workspace/build_v12.log 2>/dev/null || echo aucun; echo === CHUNKS / ACTIF ===; ls /workspace/TradingBot/data/ml_partial_M1_V12_VANTAGE/ 2>/dev/null | sed 's/_M1_.*//' | sort | uniq -c | sort -rn; echo === TOTAL ===; ls /workspace/TradingBot/data/ml_partial_M1_V12_VANTAGE/ 2>/dev/null | wc -l; echo === CPU ===; uptime; top -bn1 | head -3 | tail -1"

# Suivi log en direct
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "tail -f /workspace/build_v12.log"
```

### 5. Train + récupération modèles

Une fois le build fini (`>>> RECAP` pour les 14 actifs) :

```bash
# Train V12 sur Vast (~5 min)
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "cd /workspace/TradingBot && python3 -u -m bot_v2.train_v12_vantage --all 2>&1 | tee /workspace/train_v12.log"

# Recuperer modeles + recap
scp -i ~/.ssh/vast_v8 -P <PORT> 'root@<IP>:/workspace/TradingBot/bot_v2/ml_model_*_vantage_v12.pkl' /c/Users/Shadow/TradingBot/bot_v2/
scp -i ~/.ssh/vast_v8 -P <PORT> 'root@<IP>:/workspace/TradingBot/bot_v2/ml_features_*_vantage_v12.json' /c/Users/Shadow/TradingBot/bot_v2/
scp -i ~/.ssh/vast_v8 -P <PORT> root@<IP>:/workspace/TradingBot/ml_metrics_v12_vantage_recap.json /c/Users/Shadow/TradingBot/
```

---

## Règles pour saturer la machine

### Choisir N_WORKERS

| CPU cores physiques | N_WORKERS optimal | Pourquoi |
|---|---|---|
| 8-16 | nb_cores - 1 | Garde 1 core pour l'OS |
| 32-64 | nb_cores | OK, peu de contention |
| 96-128 | **nb_cores** (pas plus) | Au-delà = thrashing |
| 256-512 (EPYC) | **128-256 max** | Plus = sur-souscription, workers à 12% CPU au lieu de 100% |

**Règle d'or** : `1 worker = 1 core physique`. Les "logical cores" (hyperthreading) ne donnent pas un gain x2 sur du Python pandas — le scheduler Linux time-slice et chaque worker tourne à ~50% au lieu de 100%.

**Symptôme de sur-souscription** :
- `top` montre `%Cpu(s): 80%+ id` malgré 500 workers
- Chaque python3 à 10-25% CPU au lieu de 90%+
- `load average` faible (< nb_cores / 2)

**Fix** : réduire `N_WORKERS` via env var (`env N_WORKERS=256 python3 ...`).

### Sizing chunk_months

Dans `build_v12_dataset.py:build_one()` :

```python
if cpu_count >= 256:
    chunk_months = 0.25   # ~7.5 jours/chunk -> ~400 chunks/actif
elif cpu_count >= 128:
    chunk_months = 0.5    # ~15 jours/chunk -> ~200 chunks/actif
elif cpu_count >= 64:
    chunk_months = 1
else:
    chunk_months = 1.5
```

**Pourquoi 0.25 mois sur EPYC ?** Beaucoup de petits chunks = la queue de tâches reste bien remplie pour 256 workers, pas de creux entre chunks. Trop gros (1 mois+) = 100 chunks pour 256 workers = workers en idle.

### Bug connu : `_WORKER_CACHE` non utilisé

Le code `_get_global_data()` dans `ml_dataset.py` (lignes 110-168) **n'est jamais appelé** par `_process_instrument()`. Conséquence : chaque chunk recharge les parquets (M1+M15+H1+H4+D1 = ~250 MB) depuis disque. Avec 256 workers en parallèle, on lit ~64 GB/s = **disque saturé, workers à 25% CPU**.

**TODO** (à faire avant le prochain build complet) : refactorer `_process_instrument` pour utiliser `_get_global_data(inst, ...)` au lieu de `load(inst, tf, start, end)` direct. Gain attendu : **×3 à ×5** sur la vitesse de build.

---

## Pièges rencontrés (à éviter)

### 1. Parquets corrompus après kill brutal
Quand on `pkill -9` un build en cours, le worker qui était en train d'écrire un parquet laisse un fichier **tronqué** (footer Parquet manquant). Au redémarrage, `pd.read_parquet(partial_path)` crash sur `pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer`.

**Fix** : avant de relancer après un kill, purger les fichiers corrompus :

```bash
ssh -i ~/.ssh/vast_v8 -p <PORT> root@<IP> "cd /workspace/TradingBot/data/ml_partial_M1_V12_VANTAGE/ && python3 -c \"
import os, glob
from concurrent.futures import ProcessPoolExecutor
import pyarrow.parquet as pq
def check(p):
    try: pq.ParquetFile(p).metadata; return None
    except: return p
files = sorted(glob.glob('*.parquet'))
bad = []
with ProcessPoolExecutor(max_workers=64) as ex:
    for r in ex.map(check, files):
        if r: bad.append(r)
for b in bad: os.remove(b)
print(f'Deleted {len(bad)} corrupted files')\""
```

### 2. Repo GitHub a une data_vantage réduite
Le `data_vantage/` du repo fait 115 MB (7 mois). La vraie version 8 ans (852 MB) n'est PAS sur GitHub (.gitignore probable, ou trop gros pour push).

**Symptôme** : log dit `Plage data dispo : 2025-10-23 -> 2026-05-21` au lieu de `2018-03 -> 2026-05`.

**Fix** : toujours faire l'étape 2 (upload tar) avant le build.

### 3. SSH déconnecté coupe le build
Sans `nohup ... & disown`, le build meurt quand le SSH se déconnecte.

**Fix** : toujours `nohup ... > log 2>&1 < /dev/null & disown`.

### 4. stdout non flushé en background
Python bufferise stdout quand redirigé vers fichier. Le log apparaît par paquets.

**Fix** : `python3 -u` (unbuffered) + `print(..., flush=True)` dans le code.

### 5. SSH connect failed au début
Au démarrage de l'instance, SSH peut prendre 30-60s à être ready. Si le premier `ssh` rate, retry après 30s.

---

## Format pour logger un build (à remplir à chaque build)

```markdown
### Build V12 — 2026-05-22

- **Instance** : 1× RTX PRO 6000 S (mais on n'utilise pas le GPU), AMD EPYC 9754 128c/512t, 1.5 TB RAM, SSD 9100 PRO
- **Setup time** : 5 min (clone + pip install + upload data_vantage)
- **N_WORKERS** : 256 (testé 480 → sur-souscription, CPU 80% idle)
- **chunk_months** : 0.25 (5614 tâches au total)
- **Mode** : 14 actifs en parallèle (1 grande queue, pas séquentiel)
- **Build time** : XX min
- **Train time** : XX min
- **Coût** : $XX (instance à $1.346/h)
- **Verdict V12** : XX (AUC OOS moy, WR@0.70 moy)
- **Action prise** : XX (bascule live / on garde V11 / on rebuild)
```

---

## Cleanup après build

Une fois les modèles téléchargés sur le PC, **détruire l'instance Vast** pour ne plus payer :
- Bouton "X" rouge dans Vast.ai → confirme destroy
- Pas juste "stop" — l'instance stoppée continue à coûter pour le stockage

---

## Liens utiles

- Repo : https://github.com/simovadev/tradingbote
- Dashboard live : https://tradingbote-production.up.railway.app/
- VPS prod : Contabo 167.86.83.144 (cle `~/.ssh/contabo_debug`)
