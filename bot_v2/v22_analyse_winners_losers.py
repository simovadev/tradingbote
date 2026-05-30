"""v22_analyse_winners_losers.py - Comprendre POURQUOI certains OB gagnent et d'autres non.

Au lieu de trainer un DL aveugle, on fait du rule mining interpretable :
1. Pour chaque OB, on calcule ~25 features descriptives ICT
2. On compare statistiquement WIN vs LOSS sur chaque feature (Mann-Whitney, ratio, etc.)
3. On entraine un Decision Tree (depth=4) qui sort des regles type "IF X AND Y THEN WR=Z%"
4. On ressort les TOP regles + suggestions pour ameliorer l'algorithme OB

Output : un rapport texte clair avec les VRAIES regles qui discriminent.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

from bot_v2.concepts.safe import (
    find_obs_safe, daily_bias_safe, htf_value_at_t, atr_safe,
)
from bot_screener.time_utils import killzone_name, current_macro

DATA = ROOT / "data_vantage"

ASSET = "XAUUSD"
START_DATE = "2023-01-01"
END_DATE = "2026-04-01"

# Plan trade pour le label (RR 2)
SL_ATR_MULT = 1.0
TP_RR = 2.0
MAX_HOLD_BARS = 60

DISPLACEMENT_MIN = 1.5
SWING_LOOKBACK = 20


def compute_descriptive_features(df_m5, df_h1, df_d1, ob, asset):
    """Calcule ~25 features descriptives pour un OB.
    Retourne un dict feature_name -> value.
    """
    val_idx = ob.validation_index
    val_ts = ob.validation_ts
    f = {}

    # ------ TIMING ------
    f["hour"] = val_ts.hour + val_ts.minute / 60
    f["day_of_week"] = val_ts.weekday()   # 0=lundi
    f["is_monday"] = int(val_ts.weekday() == 0)
    f["is_friday"] = int(val_ts.weekday() == 4)

    kz = killzone_name(val_ts)
    f["kz_ny_am"] = int(kz == "NY AM")
    f["kz_london"] = int(kz == "London")
    f["kz_ny_pm"] = int(kz == "NY PM")
    f["kz_asia"] = int(kz == "Asia")
    f["kz_main"] = int(kz in ("NY AM", "London", "NY PM"))
    f["kz_hors"] = int(kz == "Hors KZ")

    macro = current_macro(val_ts)
    f["in_macro"] = int(bool(macro))

    # ------ DIRECTION ------
    f["direction_bullish"] = int(ob.direction == "bullish")

    # ------ DISPLACEMENT ------
    f["displacement_atr"] = float(ob.displacement_atr)
    f["displacement_strong"] = int(ob.displacement_atr >= 2.0)
    f["displacement_extreme"] = int(ob.displacement_atr >= 3.0)

    # ------ OB BODY ------
    ob_height = ob.ob_high - ob.ob_low
    ob_body = abs(ob.ob_close - ob.ob_open)
    f["ob_body_ratio"] = ob_body / ob_height if ob_height > 0 else 0
    f["ob_has_wick"] = int(ob_body / ob_height < 0.7) if ob_height > 0 else 0

    # ------ ATR ------
    atr = atr_safe(df_m5, val_idx)
    price = float(df_m5["close"].iloc[val_idx])
    f["atr_pct"] = atr / price * 100 if price > 0 else 0   # ATR en % du prix
    f["atr_high_vol"] = int((atr / price * 100) > 0.05) if price > 0 else 0

    # ------ DAILY BIAS ------
    d1 = daily_bias_safe(df_d1, val_ts)
    if d1["ok"]:
        d1_haussier = d1["bias"] == "haussier"
        f["d1_haussier"] = int(d1_haussier)
        f["d1_aligned"] = int(
            (ob.direction == "bullish" and d1_haussier) or
            (ob.direction == "bearish" and not d1_haussier)
        )
        # Magnitude du dernier daily move
        if d1["j2_close"] > 0:
            f["d1_move_pct"] = (d1["j1_close"] - d1["j2_close"]) / d1["j2_close"] * 100
        else:
            f["d1_move_pct"] = 0
    else:
        f["d1_haussier"] = 0
        f["d1_aligned"] = 0
        f["d1_move_pct"] = 0

    # ------ H1 TREND ------
    h1_close = htf_value_at_t(df_h1, val_ts, "close")
    h1_close_10ago = None
    if h1_close is not None:
        pos = df_h1.index.searchsorted(val_ts, side="left") - 1 - 10
        if pos >= 0:
            h1_close_10ago = float(df_h1["close"].iloc[pos])
    if h1_close and h1_close_10ago:
        h1_momentum = (h1_close - h1_close_10ago) / h1_close_10ago * 100
        h1_haussier = h1_momentum > 0
        f["h1_momentum_pct"] = h1_momentum
        f["h1_aligned"] = int(
            (ob.direction == "bullish" and h1_haussier) or
            (ob.direction == "bearish" and not h1_haussier)
        )
        f["h1_strong_trend"] = int(abs(h1_momentum) > 0.5)
    else:
        f["h1_momentum_pct"] = 0
        f["h1_aligned"] = 0
        f["h1_strong_trend"] = 0

    # ------ DISTANCE PDH/PDL ------
    if df_d1 is not None and len(df_d1) >= 2:
        target_day = pd.Timestamp(val_ts).normalize()
        df_d1_past = df_d1[df_d1.index < target_day]
        if len(df_d1_past) >= 1:
            pdh = float(df_d1_past["high"].iloc[-1])
            pdl = float(df_d1_past["low"].iloc[-1])
            day_range = pdh - pdl
            if atr > 0:
                if ob.direction == "bullish":
                    # Distance jusqu'a la cible PDH en ATR
                    f["dist_to_target_atr"] = (pdh - price) / atr
                else:
                    f["dist_to_target_atr"] = (price - pdl) / atr
            else:
                f["dist_to_target_atr"] = 0
            # Position dans le range journalier (0=PDL, 1=PDH)
            if day_range > 0:
                f["price_in_range_pct"] = (price - pdl) / day_range
            else:
                f["price_in_range_pct"] = 0.5
        else:
            f["dist_to_target_atr"] = 0
            f["price_in_range_pct"] = 0.5
    else:
        f["dist_to_target_atr"] = 0
        f["price_in_range_pct"] = 0.5

    # ------ SWEEP RECENT (mesure simple : volatilite avant l'OB) ------
    if val_idx >= 30:
        recent_high = df_m5["high"].iloc[val_idx - 30:val_idx].max()
        recent_low = df_m5["low"].iloc[val_idx - 30:val_idx].min()
        recent_range = recent_high - recent_low
        f["recent_range_atr"] = recent_range / atr if atr > 0 else 0
        # OB est-il pres d'un extreme recent ?
        f["ob_near_recent_high"] = int(abs(ob.ob_high - recent_high) < 0.3 * atr) if atr > 0 else 0
        f["ob_near_recent_low"] = int(abs(ob.ob_low - recent_low) < 0.3 * atr) if atr > 0 else 0
    else:
        f["recent_range_atr"] = 0
        f["ob_near_recent_high"] = 0
        f["ob_near_recent_low"] = 0

    return f


def simulate_label(df_m5, ob, atr):
    """Triple-barrier RR 2. Retourne 1=WIN, 0=LOSS, None=skip."""
    val_idx = ob.validation_index
    if val_idx + 1 + MAX_HOLD_BARS >= len(df_m5):
        return None
    entry = float(df_m5["close"].iloc[val_idx])
    if ob.direction == "bullish":
        sl = entry - SL_ATR_MULT * atr; tp = entry + TP_RR * SL_ATR_MULT * atr
    else:
        sl = entry + SL_ATR_MULT * atr; tp = entry - TP_RR * SL_ATR_MULT * atr
    if abs(entry - sl) < 1e-9: return None

    fh = df_m5["high"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    fl = df_m5["low"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    fo = df_m5["open"].values[val_idx + 1:val_idx + 1 + MAX_HOLD_BARS]
    for i in range(len(fh)):
        if ob.direction == "bullish":
            hit_sl = fl[i] <= sl; hit_tp = fh[i] >= tp
        else:
            hit_sl = fh[i] >= sl; hit_tp = fl[i] <= tp
        if hit_sl and hit_tp:
            if abs(fo[i] - sl) < abs(fo[i] - tp): return 0
            return 1
        if hit_sl: return 0
        if hit_tp: return 1
    return 0


def build_features_df(asset):
    print(f"Loading {asset} M5/H1/D1 ...")
    df_m5 = pd.read_parquet(DATA / f"{asset}_M5.parquet")[["open", "high", "low", "close"]]
    df_h1 = pd.read_parquet(DATA / f"{asset}_H1.parquet")[["open", "high", "low", "close"]]
    df_d1 = pd.read_parquet(DATA / f"{asset}_D1.parquet")[["open", "high", "low", "close"]]
    start = pd.Timestamp(START_DATE, tz="UTC"); end = pd.Timestamp(END_DATE, tz="UTC")
    ctx = start - pd.Timedelta(days=10)
    df_m5 = df_m5[(df_m5.index >= ctx) & (df_m5.index < end)]
    df_h1 = df_h1[(df_h1.index >= ctx) & (df_h1.index < end)]
    df_d1 = df_d1[(df_d1.index >= start - pd.Timedelta(days=30)) & (df_d1.index < end)]

    print(f"Detect OBs ...")
    obs = find_obs_safe(df_m5, as_of_index=len(df_m5) - 1,
                        sweep_lookback=SWING_LOOKBACK,
                        displacement_min_atr=DISPLACEMENT_MIN)
    obs = [ob for ob in obs if start <= ob.validation_ts < end]
    print(f"OBs : {len(obs)}")

    rows = []
    for i, ob in enumerate(obs):
        if i % 200 == 0: print(f"  {i}/{len(obs)}")
        atr = atr_safe(df_m5, ob.validation_index)
        if atr <= 0: continue
        label = simulate_label(df_m5, ob, atr)
        if label is None: continue
        feats = compute_descriptive_features(df_m5, df_h1, df_d1, ob, asset)
        feats["label"] = label
        feats["asset"] = asset
        feats["ts"] = ob.validation_ts.timestamp()
        rows.append(feats)
    df = pd.DataFrame(rows)
    return df


def analyse_feature_discrimination(df, feature_cols):
    """Pour chaque feature numerique, mesure le pouvoir discriminant WIN vs LOSS."""
    wins = df[df["label"] == 1]
    losses = df[df["label"] == 0]
    print(f"\nDataset: {len(df)} OBs ({len(wins)} WIN / {len(losses)} LOSS, WR={len(wins)/len(df)*100:.1f}%)")
    print()

    print("=" * 100)
    print(f"{'feature':<30} {'WIN_mean':<12} {'LOSS_mean':<12} {'diff':<10} {'WIN_med':<10} {'LOSS_med':<10} {'discrim'}")
    print("=" * 100)
    discrim_scores = []
    for col in feature_cols:
        if df[col].nunique() < 2: continue
        w_mean = wins[col].mean()
        l_mean = losses[col].mean()
        w_med = wins[col].median()
        l_med = losses[col].median()
        diff = w_mean - l_mean
        # discriminant : |diff| / std combine
        std = df[col].std()
        if std > 1e-9:
            score = abs(diff) / std
        else:
            score = 0
        discrim_scores.append((col, score, w_mean, l_mean, diff))

    discrim_scores.sort(key=lambda x: x[1], reverse=True)
    for col, score, w_mean, l_mean, diff in discrim_scores[:20]:
        marker = "***" if score > 0.2 else ("**" if score > 0.1 else "")
        print(f"{col:<30} {w_mean:<12.3f} {l_mean:<12.3f} {diff:<+10.3f} {'':<10} {'':<10} score={score:.3f} {marker}")

    return discrim_scores


def analyse_binary_filters(df, feature_cols):
    """Pour chaque feature binaire (0/1), calcule le WR quand =1 vs =0."""
    binary_feats = [c for c in feature_cols if df[c].nunique() == 2 and set(df[c].unique()).issubset({0, 1})]
    print("\n=== EFFET DES FILTRES BINAIRES SUR LE WR ===")
    print(f"{'filtre':<30} {'n=1':<8} {'WR(=1)':<8} {'WR(=0)':<8} {'gain':<8} {'verdict'}")
    print("-" * 80)
    rows = []
    for col in binary_feats:
        sel = df[col] == 1
        n_sel = sel.sum()
        n_nsel = (~sel).sum()
        if n_sel < 20 or n_nsel < 20:
            continue
        wr_sel = df[sel]["label"].mean() * 100
        wr_nsel = df[~sel]["label"].mean() * 100
        gain = wr_sel - wr_nsel
        rows.append((col, n_sel, wr_sel, wr_nsel, gain))
    rows.sort(key=lambda x: abs(x[4]), reverse=True)
    for col, n_sel, wr_sel, wr_nsel, gain in rows:
        marker = "+++" if gain > 5 else ("++" if gain > 3 else ("--" if gain < -3 else ""))
        print(f"{col:<30} {n_sel:<8} {wr_sel:<8.1f} {wr_nsel:<8.1f} {gain:<+8.1f} {marker}")


def train_decision_tree(df, feature_cols, max_depth=4):
    """Entrainement d'un Decision Tree interpretable."""
    X = df[feature_cols].values
    y = df["label"].values

    # Split walk-forward par ts (pas random)
    df_sorted = df.sort_values("ts")
    split_idx = int(len(df_sorted) * 0.7)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]
    X_train = X[train_idx]; y_train = y[train_idx]
    X_test = X[test_idx]; y_test = y[test_idx]

    print(f"\n=== DECISION TREE (depth={max_depth}) ===")
    print(f"Train : {len(X_train)} OBs ({y_train.mean()*100:.1f}% WR)")
    print(f"Test  : {len(X_test)} OBs ({y_test.mean()*100:.1f}% WR)")

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20,
                                   class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    train_pred = clf.predict_proba(X_train)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]

    # AUC simple
    def auc(y_true, y_pred):
        from sklearn.metrics import roc_auc_score
        try: return roc_auc_score(y_true, y_pred)
        except: return 0.5

    print(f"AUC train : {auc(y_train, train_pred):.3f}")
    print(f"AUC test  : {auc(y_test, test_pred):.3f}")

    # Rules en texte
    print("\n--- ARBRE DE DECISION (regles) ---")
    rules = export_text(clf, feature_names=feature_cols, max_depth=max_depth)
    print(rules)

    # WR par seuil sur le test
    print("\n--- WR test par seuil de proba ---")
    for thr in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = test_pred >= thr
        if mask.sum() > 0:
            wr = y_test[mask].mean() * 100
            print(f"  thr {thr:.2f} : {mask.sum()} OBs, WR {wr:.1f}%")

    # Feature importance
    print("\n--- IMPORTANCES FEATURES ---")
    importances = list(zip(feature_cols, clf.feature_importances_))
    importances.sort(key=lambda x: -x[1])
    for name, imp in importances[:15]:
        if imp > 0.001:
            print(f"  {name:<30} {imp:.4f}")

    return clf


def main():
    print(f"\n=== V22 ANALYSE WINNERS vs LOSERS ===")
    print(f"Actif : {ASSET}")
    print(f"Periode : {START_DATE} -> {END_DATE}")

    df = build_features_df(ASSET)
    print(f"\nTotal OBs avec label : {len(df)}")

    # Features cols (tout sauf label/asset/ts)
    feature_cols = [c for c in df.columns if c not in ("label", "asset", "ts")]
    print(f"Features : {len(feature_cols)}")

    # 1. Discrimination
    analyse_feature_discrimination(df, feature_cols)

    # 2. Filtres binaires
    analyse_binary_filters(df, feature_cols)

    # 3. Decision tree
    clf = train_decision_tree(df, feature_cols, max_depth=4)

    # Sauvegarde dataset descriptif
    out = ROOT / f"v22_features_{ASSET}.csv"
    df.to_csv(out, index=False)
    print(f"\nSauvegarde features : {out}")

    print("\n=== CONCLUSION ===")
    print("Si l'arbre trouve des regles avec AUC test > 0.55 ET un seuil donne > 50% WR :")
    print("  -> on a des INSIGHTS exploitables pour ameliorer l'algo OB")
    print("Si tout reste autour 50% AUC :")
    print("  -> pas de signal discriminant dans les features actuelles")
    print("  -> il faut ajouter d'autres features (sweep precis, momentum quality, etc.)")


if __name__ == "__main__":
    main()
