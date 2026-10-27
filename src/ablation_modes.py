"""Абляция по режимам работы (замечание 1.3): добавляются сведения о режиме (номер кластера режима в каждом цикле)
и сравнивается точность с вариантом без них на FD002 и FD004.
  - LSTM: к 14 датчикам добавляются 6 бинарных каналов (режим цикла в формате one-hot), окно 50;
  - XGBoost и Random Forest: к 56 признакам добавляются 6 долей режимов внутри окна и режим последнего цикла.
Режимы определяются теми же кластерами k-средних, что и при нормализации (по трём параметрам режима, обучающая выборка).
Запуск: python src/ablation_modes.py [FD002 FD004]
Результат: results/ablation_modes_FDxxx.json"""
import json
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from numpy.lib.stride_tricks import sliding_window_view
from xgboost import XGBRegressor

from data import N_COND, ROOT, SENSORS, add_rul, last_window_idx, load_raw, mae, make_windows, nasa_score, normalize, rmse, window_features
from models import fit_lstm, predict_lstm

RES = ROOT / "results"
WINDOW = 50
SEEDS = [0, 1, 2, 3, 4]
CFG = dict(hidden=64, layers=1, dropout=0.2, weight_decay=1e-4, lr=1e-3, epochs=100, batch=64, patience=15)


def mode_windows(df, labels, k, window):
    """Окна one-hot режимов, согласованные с make_windows (та же дополнительная первая строка)."""
    oh = np.eye(k, dtype=np.float32)[labels]
    out = []
    for u in np.unique(df["unit"].values):
        a = oh[(df["unit"].values == u)]
        a = np.vstack([np.repeat(a[:1], window - 1, axis=0), a])
        out.append(sliding_window_view(a, (window, k))[:, 0])
    return np.concatenate(out)


def metrics(y, p):
    return dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p))


def run(fd):
    k = N_COND[fd]
    tr, te, rl = load_raw(fd)
    tr, te = add_rul(tr, te, rl)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(tr[["op1", "op2", "op3"]])
    ltr, lte = km.predict(tr[["op1", "op2", "op3"]]), km.predict(te[["op1", "op2", "op3"]])
    trn, ten = normalize(tr, te, k)
    Xtr, rtr, utr, ctr = make_windows(trn, WINDOW)
    Xte, rte, ute, cte = make_windows(ten, WINDOW)
    Mtr, Mte = mode_windows(trn, ltr, k, WINDOW), mode_windows(ten, lte, k, WINDOW)
    ytr = np.minimum(rtr, 125)
    last = last_window_idx(ute, cte)
    y = rte[last]
    F, Ft = window_features(Xtr), window_features(Xte)
    Fm = np.hstack([F, Mtr.mean(axis=1), Mtr[:, -1]])
    Ftm = np.hstack([Ft, Mte.mean(axis=1), Mte[:, -1]])
    Xtr_m, Xte_m = np.concatenate([Xtr, Mtr], axis=2), np.concatenate([Xte, Mte], axis=2)
    tp = json.load(open(RES / f"trees_aligned_{fd}.json"))["w50"]
    out = {"dataset": fd, "variants": {}}
    for name, (A, At, XA, XAt) in {"без режимов": (F, Ft, Xtr, Xte), "с режимами": (Fm, Ftm, Xtr_m, Xte_m)}.items():
        t0 = time.time()
        res = {"xgb": [], "rf": [], "lstm": []}
        for s in SEEDS:
            xg = XGBRegressor(n_jobs=-1, random_state=s, tree_method="hist", **tp["xgb_params"]).fit(A, ytr)
            res["xgb"].append(metrics(y, xg.predict(At)[last]))
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=s, **tp["rf_params"]).fit(A, ytr)
            res["rf"].append(metrics(y, rf.predict(At)[last]))
            model, _ = fit_lstm(XA, ytr, utr, seed=s, **CFG)
            res["lstm"].append(metrics(y, predict_lstm(model, XAt)[last]))
            print(fd, name, "seed", s, {m: round(res[m][-1]["rmse"], 2) for m in res}, flush=True)
        for m in res:
            res[m] = dict(runs=res[m], rmse_mean=float(np.mean([r["rmse"] for r in res[m]])),
                          rmse_std_ddof1=float(np.std([r["rmse"] for r in res[m]], ddof=1)),
                          mae_mean=float(np.mean([r["mae"] for r in res[m]])))
        res["seconds"] = time.time() - t0
        out["variants"][name] = res
    json.dump(out, open(RES / f"ablation_modes_{fd}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    for fd in (sys.argv[1:] or ["FD002", "FD004"]):
        run(fd)
