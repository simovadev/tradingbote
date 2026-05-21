"""Test chargement modele V8 + prediction. Pour diagnostiquer le crash VPS."""
import pickle
import json
import sys
import traceback

try:
    import lightgbm
    print(f"lightgbm version : {lightgbm.__version__}")
except Exception as e:
    print(f"lightgbm import KO : {e}")

try:
    import sklearn
    print(f"sklearn version  : {sklearn.__version__}")
except Exception as e:
    print(f"sklearn import KO : {e}")

import pandas as pd
print(f"pandas version   : {pd.__version__}")
print(f"python           : {sys.version}")
print("-" * 50)

try:
    with open("bot_v2/ml_model_XAUUSD_vantage_v8.pkl", "rb") as f:
        model = pickle.load(f)
    print(f"V8 model loaded OK : {type(model).__name__}")
    features = json.load(open("bot_v2/ml_features_XAUUSD_vantage_v8.json"))["features"]
    print(f"V8 features : {len(features)}")
    X = pd.DataFrame([[0.0] * len(features)], columns=features)
    proba = model.predict_proba(X)[0, 1]
    print(f"predict_proba test : {proba:.4f}")
    print("=== V8 MODEL OK ===")
except Exception as e:
    print(f"!! ERREUR : {e}")
    traceback.print_exc()
