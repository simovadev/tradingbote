import os, sys
os.environ['BUILD_DATA_DIR']='data_vantage'
# V13 reutilise le mode V12 (PUR AMONT, 1 ligne par OB) — seul change la feature atr_regime.
os.environ['BUILD_V12_MODE']='1'
sys.path.insert(0, '.')
import pandas as pd
from bot_v2.ml_dataset import build_dataset

ALL_ASSETS = ['XAUUSD','NAS100','GER40','BTCUSD','EURUSD','GBPUSD','AUDUSD','USDJPY','SP500','DJ30','UK100','FRA40','USDCAD','USDCHF']
TRAIN_START = pd.Timestamp('2018-03-01', tz='UTC')
TRAIN_END = pd.Timestamp('2026-05-22', tz='UTC')
print(f'=== V13 BUILD : {len(ALL_ASSETS)} actifs EN PARALLELE (atr_regime par killzone) ===', flush=True)
print(f'Cores : {os.cpu_count()}, N_WORKERS env: {os.environ.get("N_WORKERS", "auto")}', flush=True)
df = build_dataset(TRAIN_START, TRAIN_END, ALL_ASSETS, output_path=None, chunk_months=0.25, ltf='M1', version_suffix='_V13_VANTAGE')
print(f'=== TOTAL : {len(df):,} lignes sur {len(ALL_ASSETS)} actifs ===', flush=True)
if 'instrument' in df.columns:
    print(df.groupby('instrument').size().to_string(), flush=True)
