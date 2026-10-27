"""Время обучения моделей (раздел 3.9). Запускать на свободном процессоре, когда другие расчёты не идут.
Для FD001 и FD004 (окно 50) измеряется: подбор гиперпараметров RF и XGBoost (случайный поиск по GroupKFold),
одно обучение RF и XGBoost с подобранными параметрами, одно обучение LSTM итоговой конфигурации (один запуск).
Запуск: python src/benchmark_time.py. Результат: results/timing.json"""
import json
import time

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data import ROOT, prepare, window_features
from models import fit_lstm, fit_rf, fit_xgb

RES = ROOT / "results"
CFG = dict(hidden=64, layers=1, dropout=0.2, weight_decay=1e-4, lr=1e-3, epochs=100, batch=64, patience=15)
out = {}
for fd in ("FD001", "FD004"):
    d = prepare(fd, window=50)
    F = window_features(d["Xtr"])
    t = {}
    t0 = time.time()
    rf0, rf_par, _ = fit_rf(F, d["ytr"], d["utr"])
    t["rf_tuning_s"] = time.time() - t0
    t0 = time.time()
    xg0, xg_par, _ = fit_xgb(F, d["ytr"], d["utr"])
    t["xgb_tuning_s"] = time.time() - t0
    t0 = time.time()
    RandomForestRegressor(**rf0.get_params()).fit(F, d["ytr"])
    t["rf_fit_s"] = time.time() - t0
    t0 = time.time()
    XGBRegressor(**xg0.get_params()).fit(F, d["ytr"])
    t["xgb_fit_s"] = time.time() - t0
    t0 = time.time()
    _, hist = fit_lstm(d["Xtr"], d["ytr"], d["utr"], seed=0, **CFG)
    t["lstm_fit_s"] = time.time() - t0
    t["lstm_epochs"] = len(hist)
    out[fd] = t
    print(fd, t, flush=True)
json.dump(out, open(RES / "timing.json", "w"), ensure_ascii=False, indent=1)
