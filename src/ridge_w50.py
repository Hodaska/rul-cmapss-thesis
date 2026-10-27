"""Базовые модели (константа и Ridge) с окном 50 для единых условий сравнения.
Запуск: python src/ridge_w50.py. Результат: results/ridge_w50.json"""
import json

from data import ROOT, mae, nasa_score, prepare, rmse, window_features
from models import fit_baseline_mean, fit_ridge

out = {}
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    d = prepare(fd, window=50)
    F, Ft = window_features(d["Xtr"]), window_features(d["Xte"])
    i = d["last_idx"]
    y = d["rul_te"][i]
    out[fd] = {}
    for name, m in (("baseline_mean", fit_baseline_mean(F, d["ytr"])), ("ridge", fit_ridge(F, d["ytr"]))):
        p = m.predict(Ft)[i]
        out[fd][name] = dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p))
    print(fd, out[fd], flush=True)
json.dump(out, open(ROOT / "results" / "ridge_w50.json", "w"), ensure_ascii=False, indent=1)
