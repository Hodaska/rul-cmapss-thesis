"""Деревья в условиях, выровненных с LSTM: окно 50 (и 30 для сравнения), 5 начальных значений генератора.
Гиперпараметры подбираются один раз на обучающей выборке (GroupKFold по двигателям, тест не используется),
затем модель переобучается с 5 начальными значениями. Также считаются внеблочные прогнозы XGBoost
на обучающих траекториях (для раздела 3.10).
Запуск: python src/run_trees_aligned.py [FD001 ...]
Результаты: results/trees_aligned_FDxxx.json, results/oof_xgb_FDxxx.npz"""
import json
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from data import ROOT, mae, nasa_score, prepare, rmse, window_features
from models import fit_rf, fit_xgb

RES = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
WINDOWS = [30, 50]


def metrics_last(d, pred_all):
    i = d["last_idx"]
    y, p = d["rul_te"][i], pred_all[i]
    return dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p))


def run(fd):
    out = {"dataset": fd}
    for w in WINDOWS:
        t0 = time.time()
        d = prepare(fd, window=w)
        Ftr, Fte = window_features(d["Xtr"]), window_features(d["Xte"])
        rf0, rf_par, rf_cv = fit_rf(Ftr, d["ytr"], d["utr"])
        xg0, xg_par, xg_cv = fit_xgb(Ftr, d["ytr"], d["utr"])
        xg_par = {k: (v.item() if hasattr(v, "item") else v) for k, v in xg_par.items()}
        print(fd, w, "tuning done", round(time.time() - t0), "s", rf_par, xg_par, flush=True)
        res = {"rf_params": rf_par, "rf_cv_rmse": float(rf_cv), "xgb_params": xg_par, "xgb_cv_rmse": float(xg_cv),
               "rf": [], "xgb": [], "rf_fit_s": [], "xgb_fit_s": []}
        preds = {"rf": [], "xgb": []}
        for s in SEEDS:
            t1 = time.time()
            rf = RandomForestRegressor(**{**rf0.get_params(), "random_state": s}).fit(Ftr, d["ytr"])
            res["rf_fit_s"].append(time.time() - t1)
            t1 = time.time()
            xg = XGBRegressor(**{**xg0.get_params(), "random_state": s}).fit(Ftr, d["ytr"])
            res["xgb_fit_s"].append(time.time() - t1)
            for name, m in (("rf", rf), ("xgb", xg)):
                p = m.predict(Fte)
                preds[name].append(p)
                res[name].append(metrics_last(d, p))
            print(fd, w, "seed", s, res["rf"][-1]["rmse"], res["xgb"][-1]["rmse"], flush=True)
        for name in ("rf", "xgb"):
            res[name + "_ensemble"] = metrics_last(d, np.mean(preds[name], axis=0))
            res[name + "_mean"] = {k: float(np.mean([r[k] for r in res[name]])) for k in ("rmse", "mae", "score")}
            res[name + "_std_ddof1"] = {k: float(np.std([r[k] for r in res[name]], ddof=1)) for k in ("rmse", "mae", "score")}
        out[f"w{w}"] = res
        np.savez_compressed(RES / f"trees_preds_{fd}_w{w}.npz", rul_te=d["rul_te"], last_idx=d["last_idx"],
                            rf=np.array(preds["rf"]), xgb=np.array(preds["xgb"]))
        if w == 50:  # внеблочные прогнозы XGBoost на обучающих траекториях для 3.10
            oof = np.zeros(len(Ftr))
            for tr, va in GroupKFold(5).split(Ftr, d["ytr"], d["utr"]):
                m = XGBRegressor(**{**xg0.get_params(), "random_state": 0}).fit(Ftr[tr], d["ytr"][tr])
                oof[va] = m.predict(Ftr[va])
            np.savez_compressed(RES / f"oof_xgb_{fd}.npz", oof=oof, unit=d["utr"], cycle=d["ctr"], rul=d["rul_tr"])
        print(fd, w, "done", round(time.time() - t0), "s", flush=True)
        json.dump(out, open(RES / f"trees_aligned_{fd}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    for fd in (sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"]):
        run(fd)
