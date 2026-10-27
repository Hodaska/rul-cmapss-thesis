"""Модели: baseline, Random Forest, XGBoost, LSTM."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

from data import rmse

SEED = 42


def fit_baseline_mean(F, y):
    return DummyRegressor(strategy="mean").fit(F, y)


def fit_ridge(F, y):
    return Ridge(alpha=1.0).fit(F, y)


def fit_rf(F, y, groups, n_iter=8):
    grid = {
        "max_depth": [10, 15, 25, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "max_features": [0.3, 0.5, 1.0],
    }
    base = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED)
    s = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=GroupKFold(3), scoring="neg_root_mean_squared_error",
                           random_state=SEED, n_jobs=1)
    s.fit(F, y, groups=groups)
    return s.best_estimator_, s.best_params_, -s.best_score_


def fit_xgb(F, y, groups, n_iter=10):
    grid = {
        "n_estimators": [300, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.06, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.6, 1.0],
    }
    base = XGBRegressor(n_jobs=-1, random_state=SEED, tree_method="hist")
    s = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=GroupKFold(3), scoring="neg_root_mean_squared_error",
                           random_state=SEED, n_jobs=1)
    s.fit(F, y, groups=groups)
    return s.best_estimator_, s.best_params_, -s.best_score_


class LSTMNet(nn.Module):
    def __init__(self, n_in, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def fit_lstm(X, y, units, seed=0, epochs=40, batch=256, lr=1e-3, val_frac=0.15, verbose=False,
             hidden=64, layers=1, dropout=0.2, weight_decay=1e-4, patience=8, val_units=None):
    """Обучение с отбором лучшей эпохи по валидации на отложенных двигателях;
    остановка, если валидация не улучшалась patience эпох подряд.
    val_units, если задан, фиксирует состав валидации явно (например, для GroupKFold)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    uniq = np.unique(units)
    if val_units is None:
        val_units = rng.choice(uniq, size=max(1, int(len(uniq) * val_frac)), replace=False)
    vm = np.isin(units, val_units)
    Xt, yt = torch.tensor(X[~vm]), torch.tensor(y[~vm], dtype=torch.float32)
    Xv, yv = torch.tensor(X[vm]), y[vm]
    model = LSTMNet(X.shape[2], hidden=hidden, layers=layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    best, best_state, hist, bad = 1e9, None, [], 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xv[i:i + 2048]) for i in range(0, len(Xv), 2048)]).numpy()
        v = rmse(yv, pv)
        sched.step(v)
        hist.append(v)
        if v < best:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
        if verbose:
            print(ep, round(v, 2))
        if patience and bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def predict_lstm(model, X):
    with torch.no_grad():
        Xt = torch.tensor(X)
        return torch.cat([model(Xt[i:i + 2048]) for i in range(0, len(Xt), 2048)]).numpy()
