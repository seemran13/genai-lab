"""P3: Demand forecasting with a PyTorch MLP, Optuna tuning, and a Random Forest baseline.

Uses synthetic e-commerce demand data so it runs immediately.
Swap `make_data()` for your real order-history table later.

Run:  python p3_demand_mlp.py            (about 20 trials)
      python p3_demand_mlp.py --trials 40
Needs: numpy pandas scikit-learn torch optuna matplotlib
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_COLS = ["discount_pct", "weather_index", "shipping_delay_days",
            "regional_index", "day_of_year_sin", "day_of_year_cos", "is_weekend"]
N_CATEGORIES = 6


# ----------------------------------------------------------------- data
def make_data(n=20000, seed=42):
    """Demand with real non-linear interactions (discount x weather, delay penalty,
    seasonality) that a plain linear model cannot capture."""
    rng = np.random.default_rng(seed)
    day = rng.integers(0, 365, n)
    cat = rng.integers(0, N_CATEGORIES, n)
    df = pd.DataFrame({
        "discount_pct": rng.uniform(0, 50, n),
        "weather_index": rng.normal(0, 1, n),            # e.g. rain / heat severity
        "shipping_delay_days": rng.exponential(1.5, n),
        "regional_index": rng.uniform(0.6, 1.4, n),      # regional purchasing power
        "day_of_year_sin": np.sin(2 * np.pi * day / 365),
        "day_of_year_cos": np.cos(2 * np.pi * day / 365),
        "is_weekend": (day % 7 >= 5).astype(float),
        "category": cat,
    })
    base = np.array([40, 80, 25, 120, 60, 15])[cat]
    promo_lift = 1 + 0.035 * df.discount_pct * (1 + 0.5 * np.tanh(df.weather_index * df.discount_pct / 20))
    season = 1 + 0.45 * df.day_of_year_sin + 0.2 * df.day_of_year_cos
    delay_penalty = np.exp(-0.18 * df.shipping_delay_days ** 1.2)
    demand = (base * df.regional_index * promo_lift * season * delay_penalty
              * (1 + 0.15 * df.is_weekend))
    df["units"] = np.clip(demand + rng.normal(0, 0.06 * demand + 2), 0, None)
    return df


def to_features(df, scaler=None):
    X_num = df[NUM_COLS].to_numpy(dtype=np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(X_num)
    X_num = scaler.transform(X_num)
    X_cat = np.eye(N_CATEGORIES, dtype=np.float32)[df["category"].to_numpy()]
    return np.hstack([X_num, X_cat]).astype(np.float32), scaler


# ---------------------------------------------------------------- model
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, activation, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       ACTIVATIONS[activation](), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(cfg, data, max_epochs=80, patience=8, seed=SEED):
    """Train with early stopping on validation loss. Returns model, history, best val RMSE."""
    torch.manual_seed(seed)
    Xtr, ytr, Xva, yva = (torch.tensor(a, device=DEVICE) for a in
                          (data["Xtr"], data["ytr_s"], data["Xva"], data["yva_s"]))
    model = MLP(Xtr.shape[1], [cfg["width"]] * cfg["n_layers"],
                cfg["activation"], cfg["dropout"]).to(DEVICE)
    # weight_decay is the L2 penalty
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])
    loss_fn = nn.MSELoss()
    best, best_state, bad, hist = float("inf"), None, 0, {"train": [], "val": []}

    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        run = 0.0
        for i in range(0, len(Xtr), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            if len(idx) < 2:          # BatchNorm needs more than one sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), yva).item()
        hist["train"].append(run / len(Xtr))
        hist["val"].append(val)
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist, best


def predict_units(model, X, y_mean, y_std):
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X, device=DEVICE)).cpu().numpy()
    return p * y_std + y_mean


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# ----------------------------------------------------------------- main
def main(n_trials):
    df = make_data()
    train, rest = train_test_split(df, test_size=0.4, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED)

    Xtr, scaler = to_features(train)
    Xva, _ = to_features(val, scaler)
    Xte, _ = to_features(test, scaler)
    y_mean, y_std = train.units.mean(), train.units.std()
    data = {"Xtr": Xtr, "Xva": Xva,
            "ytr_s": ((train.units - y_mean) / y_std).to_numpy(np.float32),
            "yva_s": ((val.units - y_mean) / y_std).to_numpy(np.float32)}
    print(f"Device: {DEVICE} | train/val/test = {len(train)}/{len(val)}/{len(test)}\n")

    # ---- baseline: Random Forest ----
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                               random_state=SEED).fit(Xtr, train.units)
    rf_pred = rf.predict(Xte)
    print(f"Random Forest baseline  test RMSE={rmse(test.units, rf_pred):.2f}  "
          f"MAE={mean_absolute_error(test.units, rf_pred):.2f}\n")

    # ---- Optuna search (validation RMSE in original units) ----
    def objective(trial):
        cfg = {
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "width": trial.suggest_categorical("width", [32, 64, 128, 256]),
            "activation": trial.suggest_categorical("activation", list(ACTIVATIONS)),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "l2": trial.suggest_float("l2", 1e-7, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
        model, _, _ = train_mlp(cfg, data)
        return rmse(val.units, predict_units(model, Xva, y_mean, y_std))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    print(f"Tuning MLP with Optuna ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best val RMSE: {study.best_value:.2f}")
    print("Best params:", {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in study.best_params.items()}, "\n")

    # ---- final model with best params ----
    model, hist, _ = train_mlp(study.best_params, data)
    mlp_pred = predict_units(model, Xte, y_mean, y_std)
    results = pd.DataFrame({
        "RMSE": [rmse(test.units, rf_pred), rmse(test.units, mlp_pred)],
        "MAE": [mean_absolute_error(test.units, rf_pred),
                mean_absolute_error(test.units, mlp_pred)],
    }, index=["Random Forest (baseline)", "Tuned MLP"])
    print("=== Test-set comparison (units per SKU-day) ===")
    print(results.round(2).to_string())
    winner = results.RMSE.idxmin()
    print(f"\nLower RMSE: {winner}")

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals = [t.value for t in study.trials]
    ax[0].plot(vals, "o", alpha=0.5, label="trial")
    ax[0].plot(np.minimum.accumulate(vals), "r-", label="best so far")
    ax[0].set(title="Optuna convergence", xlabel="trial", ylabel="validation RMSE")
    ax[0].legend()
    ax[1].plot(hist["train"], label="train")
    ax[1].plot(hist["val"], label="validation")
    ax[1].set(title="Best MLP loss curves (standardised MSE)", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("p3_results.png", dpi=120)
    print("Saved p3_results.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    main(ap.parse_args().trials)