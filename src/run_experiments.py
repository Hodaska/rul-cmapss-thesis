"""Основной эксперимент: baseline, Ridge, RF, XGBoost, LSTM на FD001..FD004.
Запуск: python src/run_experiments.py [FD001 FD002 ...]
Результаты: results/metrics_FDxxx.json, results/preds_FDxxx.npz"""
import json
import os
import sys
import time

import numpy as np

from data import ROOT, mae, nasa_score, prepare, rmse, window_features
from models import (SEED, fit_baseline_mean, fit_lstm, fit_rf, fit_ridge, fit_xgb, predict_lstm)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
N_LSTM_SEEDS = 3
SKIP_LSTM = os.environ.get("SKIP_LSTM") == "1"  # LSTM считается отдельно: src/run_lstm.py


def score_last(d, pred_all):
    """Метрики по последнему окну каждого тестового двигателя (эталон - RUL из файла RUL_FDxxx)."""
    i = d["last_idx"]
    y, p = d["rul_te"][i], pred_all[i]
    return dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p))


def run(fd):
    t0 = time.time()
    d = prepare(fd)
    Ftr, Fte = window_features(d["Xtr"]), window_features(d["Xte"])
    out, preds = {"dataset": fd, "n_train_windows": int(len(Ftr)), "n_test_engines": int(len(d["last_idx"]))}, {}

    m = fit_baseline_mean(Ftr, d["ytr"])
    preds["baseline_mean"] = m.predict(Fte)
    m = fit_ridge(Ftr, d["ytr"])
    preds["ridge"] = m.predict(Fte)

    m, params, cv = fit_rf(Ftr, d["ytr"], d["utr"])
    preds["rf"] = m.predict(Fte)
    out["rf_params"], out["rf_cv_rmse"] = params, float(cv)
    out["rf_importance"] = m.feature_importances_.tolist()
    print(fd, "RF done", round(time.time() - t0), "s", flush=True)

    m, params, cv = fit_xgb(Ftr, d["ytr"], d["utr"])
    preds["xgb"] = m.predict(Fte)
    out["xgb_params"], out["xgb_cv_rmse"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}, float(cv)
    out["xgb_importance"] = m.feature_importances_.tolist()
    print(fd, "XGB done", round(time.time() - t0), "s", flush=True)

    lstm_runs, hists = [], []
    if not SKIP_LSTM:
        for s in range(N_LSTM_SEEDS):
            model, hist = fit_lstm(d["Xtr"], d["ytr"], d["utr"], seed=s)
            p = predict_lstm(model, d["Xte"])
            preds[f"lstm_seed{s}"] = p
            lstm_runs.append(score_last(d, p))
            hists.append(hist)
            print(fd, "LSTM seed", s, lstm_runs[-1], flush=True)
        preds["lstm"] = np.mean([preds[f"lstm_seed{s}"] for s in range(N_LSTM_SEEDS)], axis=0)
        out["lstm_seeds"] = lstm_runs
        out["lstm_val_history"] = hists

    out["metrics"] = {k: score_last(d, preds[k]) for k in ["baseline_mean", "ridge", "rf", "xgb"]}
    if not SKIP_LSTM:
        out["metrics"]["lstm_mean_of_seeds"] = {k: float(np.mean([r[k] for r in lstm_runs])) for k in ("rmse", "mae", "score")}
        out["metrics"]["lstm_std_of_seeds"] = {k: float(np.std([r[k] for r in lstm_runs])) for k in ("rmse", "mae", "score")}
        out["metrics"]["lstm_ensemble"] = score_last(d, preds["lstm"])

    out["seconds"] = time.time() - t0

    json.dump(out, open(RES / f"metrics_{fd}.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(RES / f"preds_{fd}.npz", rul_te=d["rul_te"], ute=d["ute"], cte=d["cte"],
                        last_idx=d["last_idx"], **preds)
    print(fd, json.dumps(out["metrics"], indent=1), flush=True)


if __name__ == "__main__":
    for fd in (sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"]):
        run(fd)
