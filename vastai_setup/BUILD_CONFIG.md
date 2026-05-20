# Configuration Build V5 sur Vast.ai

Documentation des paramètres optimaux pour build V5 (dataset ML 8 ans) sur Vast.ai.

## Setup recommandé

### Instance Vast.ai

**Specs minimum** :
- CPU : **96+ cores attribués** (EPYC 9B14 / 9654 / Xeon 6517P+)
- RAM : 200+ GB (le build M1 + cache HTF = ~500MB par worker)
- Disk : 50+ GB libre (data Admiral 8 ans = 1.2GB, parquets V5 = ~250MB)
- Network : 10+ Mbps download (pour upload data 1.2GB depuis PC)

**Recommandation** : EPYC 9B14 96 cores / 192 threads, 386 GB RAM, ~$1.5/h (testé OK).

**À éviter** :
- Instances avec moins de 32 cores attribués
- "X.0/Y CPU" où X << Y (cores limités)

### Filtre Vast.ai

Dans l'interface : `Machine Resources → CPU Cores min = 64+`

## Pipeline complet (build V5 14 actifs)

### 1. Setup serveur (1ère fois)

```bash
ssh -p PORT root@IP "wget -q -O setup.sh https://raw.githubusercontent.com/simovadev/tradingbote/main/vastai_setup/setup_server.sh && bash setup.sh"
```

Installe : python3, venv, requirements.txt, clone repo `/workspace/TradingBot`.

### 2. Upload data Admiral 8 ans (1.2 GB)

```powershell
scp -P PORT -r "C:\Users\Shadow\TradingBot\data\cache" "root@IP:/workspace/TradingBot/data/"
```

Durée : ~3-5 min (10 MB/s upload typique).

### 3. Build V5 14 actifs (séquentiel, chunks adaptatifs)

```bash
ssh -p PORT root@IP "cd /workspace/TradingBot && source venv/bin/activate && N_WORKERS=192 bash vastai_setup/build_v5_all.sh 2>&1 | tee build_v5_all.log"
```

**Important** : `N_WORKERS=192` (= nombre de threads). Le code `build_v5_dataset.py` adapte automatiquement `chunk_months` :
- 192+ threads → `chunk_months = 0.5` (14j) → **192 chunks** = sature toutes les cores
- 64+ threads → `chunk_months = 1` (1 mois) → 91 chunks
- Default → `chunk_months = 3` (3 mois) → 31 chunks

**Durée mesurée** (EPYC 9B14, 192 threads, chunks 14j) :
- XAUUSD seul : ~2 min
- 14 actifs total : **~28-30 min**

### 4. Train V5 14 modèles

```bash
ssh -p PORT root@IP "cd /workspace/TradingBot && source venv/bin/activate && bash vastai_setup/train_v5_all.sh 2>&1 | tee train_v5_all.log"
```

Durée : **~5 min** (LightGBM rapide).

### 5. Download tout en local

**Modèles + features + metrics + datasets** :

```powershell
scp -P PORT "root@IP:/workspace/TradingBot/bot_v2/ml_model_*_admiral_v5.pkl" "C:\Users\Shadow\TradingBot\bot_v2\"
scp -P PORT "root@IP:/workspace/TradingBot/bot_v2/ml_features_*_admiral_v5.json" "C:\Users\Shadow\TradingBot\bot_v2\"
scp -P PORT "root@IP:/workspace/TradingBot/ml_metrics_*_admiral_v5.txt" "C:\Users\Shadow\TradingBot\"
scp -P PORT "root@IP:/workspace/TradingBot/data/ml_dataset_*_admiral_8ans_V5.parquet" "C:\Users\Shadow\TradingBot\data\"
```

Taille totale : ~250 MB (14 × ~15-20 MB par dataset).

## Monitoring

### Watch progression (autre terminal)

```bash
ssh -p PORT root@IP "bash /workspace/TradingBot/vastai_setup/watch_v5.sh"
```

Refresh toutes les 30s, montre chunks par actif.

### CPU usage en direct

```bash
ssh -p PORT root@IP "top -bn1 | head -15"
```

Pendant build sain : load average proche du nombre de cores, CPU 50-90% utilisé.

### Log brut

```bash
ssh -p PORT root@IP "tail -f /workspace/TradingBot/build_v5_all.log"
```

## Anti-patterns à éviter

### ❌ Build 14 actifs en parallèle

`build_v5_parallel.sh` lance 14 builds × 13 workers chacun = **182 workers concurrents**.
Sur 192 threads → contention CPU → workers à 58% CPU au lieu de 100%. **Plus lent que séquentiel**.

### ❌ Trop peu de chunks

3 mois × 31 chunks sur 192 threads → seulement 31 workers actifs. **161 cores chôment**.
Solution : chunks adaptatifs (✅ déjà dans le code V5.1).

### ❌ N_WORKERS > nombre de chunks

Inutile, le `ProcessPoolExecutor` plafonne à `min(N_WORKERS, n_chunks)`.

### ❌ Oublier de tuer le bot live pendant build/train

```bash
pkill -9 python3   # avant chaque relance
rm -rf /workspace/TradingBot/data/ml_partial_M1_V5  # pour repartir clean
```

## Performances mesurées

| Config | Temps XAUUSD | 14 actifs |
|--------|--------------|-----------|
| EPYC 96c, 32 workers, chunks 3 mois | 135s | ~30 min |
| EPYC 192t, 64 workers, chunks 3 mois | 90s | ~21 min |
| EPYC 192t, 192 workers, chunks 14j | **120s** | **~28 min** |

Note : chunks 14j n'est pas forcément le plus rapide pour XAUUSD seul, mais c'est le **meilleur compromis** pour les 14 actifs (BTCUSD 24/7 = + de M1 que les indices = + de chunks utiles).

## Datasets V5 attendus

| Actif | Candidats | WR brut | AUC | WR @0.75 |
|-------|-----------|---------|-----|----------|
| XAUUSD | 127,566 | 29.2% | 0.818 | 86.5% |
| NAS100 | 82,654 | 33.4% | 0.828 | 79.7% |
| GER40 | 62,442 | 32.9% | 0.829 | 80.6% |
| BTCUSD | 120,461 | 33.0% | 0.828 | 78.1% |
| EURUSD | 67,137 | 30.3% | 0.836 | 81.5% |
| GBPUSD | 75,648 | 30.9% | 0.826 | 80.1% |
| AUDUSD | 65,728 | 30.9% | 0.829 | 78.3% |
| USDJPY | 72,197 | 30.7% | 0.825 | 81.2% |
| SP500 | 70,723 | 31.0% | 0.819 | 76.5% |
| DJ30 | 109,722 | 30.2% | 0.824 | 82.5% |
| UK100 | 91,998 | 30.5% | 0.820 | 82.1% |
| FRA40 | 58,465 | 30.7% | 0.827 | 82.1% |
| USDCAD | 82,395 | 28.5% | 0.826 | 81.3% |
| USDCHF | 77,092 | 29.2% | 0.828 | 84.7% |

(Référence du build V5 du 2026-05-20)

## Cleanup serveur (avant fin instance)

Toujours avant d'arrêter une instance Vast.ai :
1. Vérifier que tous les datasets/modèles sont bien downloadés en local
2. Tuer les process : `pkill -9 python3`

Pas besoin de cleanup data, l'instance est détruite quand stop.
