"""Загрузка и предобработка C-MAPSS."""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# датчики, которые несут информацию о деградации (остальные почти константны)
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
N_COND = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
CAP = 125
WINDOW = 30


def load_raw(fd):
    train = pd.read_csv(DATA / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(DATA / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul_last = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None, names=["rul"])["rul"].values
    train[SENSORS] = train[SENSORS].astype(float)
    test[SENSORS] = test[SENSORS].astype(float)
    return train, test, rul_last


def add_rul(train, test, rul_last):
    """RUL в каждом цикле (без ограничения сверху)."""
    train = train.copy()
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    test = test.copy()
    n = test.groupby("unit")["cycle"].transform("max")
    final = test["unit"].map(dict(zip(sorted(test["unit"].unique()), rul_last)))
    test["rul"] = final + (n - test["cycle"])
    return train, test


def normalize(train, test, n_cond):
    """Стандартизация датчиков отдельно по режимам работы (для FD001/FD003 режим один)."""
    train, test = train.copy(), test.copy()
    if n_cond == 1:
        tr_c = np.zeros(len(train), dtype=int)
        te_c = np.zeros(len(test), dtype=int)
    else:
        km = KMeans(n_clusters=n_cond, n_init=10, random_state=0).fit(train[["op1", "op2", "op3"]])
        tr_c = km.predict(train[["op1", "op2", "op3"]])
        te_c = km.predict(test[["op1", "op2", "op3"]])
    for c in range(n_cond):
        m = tr_c == c
        mu = train.loc[m, SENSORS].mean()
        sd = train.loc[m, SENSORS].std().replace(0, 1)
        train.loc[m, SENSORS] = (train.loc[m, SENSORS] - mu) / sd
        mt = te_c == c
        test.loc[mt, SENSORS] = (test.loc[mt, SENSORS] - mu) / sd
    return train, test


def make_windows(df, window=WINDOW):
    """Окна по каждому двигателю. Начало ряда дополняется первой строкой.
    Возвращает X (N, window, n_sensors), rul (N,), unit (N,), cycle (N,)."""
    Xs, ys, us, cs = [], [], [], []
    for u, g in df.groupby("unit"):
        a = g[SENSORS].values.astype(np.float32)
        pad = np.repeat(a[:1], window - 1, axis=0)
        a = np.vstack([pad, a])
        w = sliding_window_view(a, (window, a.shape[1]))[:, 0]
        Xs.append(w)
        ys.append(g["rul"].values)
        us.append(g["unit"].values)
        cs.append(g["cycle"].values)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(us), np.concatenate(cs)


def window_features(X):
    """Признаки для RF/XGBoost: последнее значение, среднее, СКО и наклон по окну для каждого датчика."""
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    t = np.arange(X.shape[1], dtype=np.float32)
    t = t - t.mean()
    slope = (X * t[None, :, None]).sum(axis=1) / (t ** 2).sum()
    return np.hstack([last, mean, std, slope])


def feature_names():
    return [f"{s}_{k}" for k in ("last", "mean", "std", "slope") for s in SENSORS]


def last_window_idx(unit, cycle):
    """Индексы последних окон каждого двигателя (для тестовой оценки)."""
    idx = []
    for u in np.unique(unit):
        m = np.where(unit == u)[0]
        idx.append(m[np.argmax(cycle[m])])
    return np.array(idx)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def nasa_score(y, p):
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def prepare(fd, window=WINDOW, cap=CAP):
    """Всё вместе: окна для train и test, цель с ограничением cap."""
    train, test, rul_last = load_raw(fd)
    train, test = add_rul(train, test, rul_last)
    train, test = normalize(train, test, N_COND[fd])
    Xtr, rtr, utr, ctr = make_windows(train, window)
    Xte, rte, ute, cte = make_windows(test, window)
    return dict(Xtr=Xtr, rul_tr=rtr, ytr=np.minimum(rtr, cap) if cap else rtr, utr=utr, ctr=ctr,
                Xte=Xte, rul_te=rte, ute=ute, cte=cte, last_idx=last_window_idx(ute, cte))
