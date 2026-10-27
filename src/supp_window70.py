"""Дополнительная проверка окна 70 (раздел 3.7): вложенная перекрёстная проверка показала, что RMSE LSTM продолжает
снижаться при росте окна с 50 до 70, а у XGBoost выходит на плато. Для оценки влияния на сравнение на тесте LSTM
и деревья (гиперпараметры из подбора для окна 50, без повторной настройки) обучаются с окном 70, пять запусков.
Результат описательный: окно 70 не использовалось для выбора основной конфигурации.
Запуск: python src/supp_window70.py. Результат: results/supp_window70.json"""
import json
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data import ROOT, mae, nasa_score, prepare, rmse, window_features
from models import fit_lstm, predict_lstm

RES = ROOT / "results"
CFG = dict(hidden=64, layers=1, dropout=0.2, weight_decay=1e-4, lr=1e-3, epochs=100, batch=64, patience=15)
out = {}
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    d = prepare(fd, window=70)
    i = d["last_idx"]
    y = d["rul_te"][i]
    F, Ft = window_features(d["Xtr"]), window_features(d["Xte"])
    tp = json.load(open(RES / f"trees_aligned_{fd}.json"))["w50"]
    res = {"rf": [], "xgb": [], "lstm": []}
    for s in range(5):
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=s, **tp["rf_params"]).fit(F, d["ytr"])
        xg = XGBRegressor(n_jobs=-1, random_state=s, tree_method="hist", **tp["xgb_params"]).fit(F, d["ytr"])
        model, hist = fit_lstm(d["Xtr"], d["ytr"], d["utr"], seed=s, **CFG)
        for name, p in (("rf", rf.predict(Ft)[i]), ("xgb", xg.predict(Ft)[i]), ("lstm", predict_lstm(model, d["Xte"])[i])):
            res[name].append(dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p)))
        print(fd, s, {k: round(v[-1]["rmse"], 2) for k, v in res.items()}, flush=True)
    out[fd] = {k: dict(runs=v, rmse_mean=float(np.mean([r["rmse"] for r in v])),
                       rmse_std_ddof1=float(np.std([r["rmse"] for r in v], ddof=1)),
                       mae_mean=float(np.mean([r["mae"] for r in v])), score_mean=float(np.mean([r["score"] for r in v])))
               for k, v in res.items()}
    json.dump(out, open(RES / "supp_window70.json", "w"), ensure_ascii=False, indent=1)
