"""Verifier les ranges des parquets sur le VPS."""
import pandas as pd
import os
assets = ['XAUUSD','NAS100','DJ30','SP500','GER40','BTCUSD','XAGUSD','DXY','SPX500']
for a in assets:
    p = f'data_vantage/{a}_M1.parquet'
    if not os.path.exists(p):
        print(f'{a:8s} MISSING')
        continue
    df = pd.read_parquet(p)
    years = (df.index[-1] - df.index[0]).days / 365.25
    print(f'{a:8s} {len(df):>10,} rows  {df.index[0].date()} -> {df.index[-1].date()}  ({years:.1f} ans)')
