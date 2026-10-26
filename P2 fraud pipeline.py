"""P2: Fintech fraud scoring (Decision Tree vs Random Forest) + K-Means user segmentation.

Uses synthetic data so it runs immediately. Swap `make_transactions()` and
`make_user_profiles()` for your real data loaders later.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_text

TXN_FEATURES = ["amount", "location_delta_km", "new_device", "velocity_1h", "account_age_days"]
USER_FEATURES = ["avg_amount", "txns_per_day", "avg_location_delta_km", "night_txn_share"]


# ----------------------------------------------------------------- data
def make_transactions(n=30000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "amount": rng.lognormal(3.5, 1.0, n),
        "location_delta_km": rng.exponential(30, n),
        "new_device": rng.binomial(1, 0.08, n),
        "velocity_1h": rng.poisson(1.2, n),
        "account_age_days": rng.integers(1, 2500, n),
    })
    # Fraud pattern: large amounts, far-from-home, new device, bursts, young accounts.
    z = (-6.0
         + 0.6 * np.log1p(df.amount)
         + 0.02 * df.location_delta_km
         + 2.2 * df.new_device
         + 0.8 * df.velocity_1h
         - 0.0015 * df.account_age_days)
    p = 1 / (1 + np.exp(-z))
    df["is_fraud"] = rng.binomial(1, p)
    return df


def make_user_profiles(n=3000, seed=7):
    """Three hidden behaviour groups so clustering has real structure to find."""
    rng = np.random.default_rng(seed)
    group = rng.choice(3, n, p=[0.55, 0.30, 0.15])
    centers = {  # avg_amount, txns/day, avg location delta, night share
        0: (25, 1.0, 10, 0.05),     # low-spend, local, daytime
        1: (90, 3.5, 40, 0.15),     # active mainstream spenders
        2: (400, 6.0, 250, 0.45),   # high-value, travelling, night-heavy (riskier)
    }
    rows = [np.array(centers[g]) * rng.normal(1, 0.18, 4) for g in group]
    return pd.DataFrame(rows, columns=USER_FEATURES)


# ------------------------------------------------------------ fraud models
def evaluate(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "ROC-AUC": roc_auc_score(y_te, proba),
        "PR-AUC": average_precision_score(y_te, proba),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
    }


def run_fraud_models():
    df = make_transactions()
    print(f"Fraud rate: {df.is_fraud.mean():.2%} ({df.is_fraud.sum()} of {len(df)})\n")
    X, y = df[TXN_FEATURES], df.is_fraud
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                  class_weight="balanced", random_state=0).fit(X_tr, y_tr)
    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=0).fit(X_tr, y_tr)

    print("=== Decision tree rules (depth 4) ===")
    print(export_text(tree, feature_names=TXN_FEATURES, max_depth=3))

    results = pd.DataFrame([evaluate("Decision Tree", tree, X_te, y_te),
                            evaluate("Random Forest", forest, X_te, y_te)]).set_index("model")
    print("=== Comparison matrix (test set, threshold 0.5) ===")
    print(results.round(3).to_string(), "\n")

    imp = pd.Series(forest.feature_importances_, index=TXN_FEATURES).sort_values(ascending=False)
    print("=== Random Forest feature importance ===")
    print(imp.round(3).to_string(), "\n")


# ------------------------------------------------------------- clustering
def run_clustering():
    users = make_user_profiles()
    # Spend/distance are right-skewed, so log-transform before scaling; otherwise the
    # few very high-value users dominate the distance metric.
    X = StandardScaler().fit_transform(np.log1p(users[USER_FEATURES]))

    print("=== Choosing K (elbow + silhouette) ===")
    scores = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
        print(f"K={k}  inertia={km.inertia_:9.1f}  silhouette={scores[k]:.3f}")
    best_k = max(scores, key=scores.get)
    print(f"\nBest K by silhouette: {best_k}\n")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(X)
    users["cluster"] = km.labels_
    profile = users.groupby("cluster").agg(
        size=("avg_amount", "size"),
        avg_amount=("avg_amount", "mean"),
        txns_per_day=("txns_per_day", "mean"),
        avg_location_delta_km=("avg_location_delta_km", "mean"),
        night_txn_share=("night_txn_share", "mean"),
    ).round(2)
    print("=== Cluster profiles ===")
    print(profile.to_string())

    try:  # optional plots; script still works without matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(scores)
        inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(ks, inertias, marker="o"); ax[0].set(title="Elbow", xlabel="K", ylabel="Inertia")
        ax[1].plot(ks, list(scores.values()), marker="o"); ax[1].set(title="Silhouette", xlabel="K")
        fig.tight_layout(); fig.savefig("p2_k_selection.png", dpi=120)
        print("\nSaved p2_k_selection.png")
    except ImportError:
        print("\n(matplotlib not installed; skipped k-selection plot)")


if __name__ == "__main__":
    run_fraud_models()
    run_clustering()