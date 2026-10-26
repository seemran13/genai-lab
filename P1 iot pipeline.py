"""P1: Industrial IoT - RUL regression + 24h failure classifier.

Uses synthetic CNC sensor data so you can run it immediately.
Swap `make_data()` for your real CSV loader later.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             precision_recall_curve, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = ["vibration", "spindle_temp", "motor_voltage", "acoustic_noise"]


def make_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "vibration": rng.normal(5, 1.5, n),
        "spindle_temp": rng.normal(70, 8, n),
        "motor_voltage": rng.normal(220, 6, n),
        "acoustic_noise": rng.normal(60, 5, n),
    })
    wear = (0.5 * df.vibration + 0.05 * df.spindle_temp
            + 0.02 * df.acoustic_noise + rng.normal(0, 0.3, n))
    # Mean RUL of 70 h puts roughly 8% of samples inside the 24 h failure window.
    df["rul_hours"] = np.clip(70 - 40 * (wear - wear.mean()) + rng.normal(0, 10, n), 1, None)
    df["fail_24h"] = (df.rul_hours <= 24).astype(int)
    # simulate sensor dropouts to practice cleaning
    df.loc[rng.choice(n, 100, replace=False), "vibration"] = np.nan
    return df


def clean(df):
    df = df.copy()
    df[FEATURES] = df[FEATURES].interpolate(limit_direction="both")
    return df


def build_pipeline(model, degree=2):
    pre = ColumnTransformer([
        ("poly", Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                           ("scale", StandardScaler())]), FEATURES)
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = clean(make_data())
    X = df[FEATURES]
    X_tr, X_te, y_rul_tr, y_rul_te, y_f_tr, y_f_te = train_test_split(
        X, df.rul_hours, df.fail_24h, test_size=0.2, random_state=0, stratify=df.fail_24h)

    # --- Part 1: RUL regression ---
    reg = build_pipeline(LinearRegression()).fit(X_tr, y_rul_tr)
    pred = reg.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_rul_te, pred))
    print(f"RUL  RMSE={rmse:.2f} h   MAE={mean_absolute_error(y_rul_te, pred):.2f} h")

    # --- Part 2: failure classifier with recall-oriented threshold ---
    clf = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_tr, y_f_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_f_te, proba)
    # highest threshold that still gives >= 95% recall (fewest false alarms)
    ok = np.where(rec[:-1] >= 0.95)[0]
    threshold = thr[ok[-1]] if len(ok) else 0.5
    print(f"Chosen threshold for >=95% recall: {threshold:.3f}")
    print(classification_report(y_f_te, (proba >= threshold).astype(int)))


if __name__ == "__main__":
    main()