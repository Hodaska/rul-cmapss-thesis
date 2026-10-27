"""Чувствительность LSTM к процедуре обучения (разделы 3.8, замечания 1.7 и 2.8): таблица 2 × 2,
где меняются независимо длина окна (30 / 50) и процедура обучения (конфигурация 1: мини-пакет 256, до 40 эпох,
8 эпох без улучшения; конфигурация 2: мини-пакет 64, до 100 эпох, 15 эпох без улучшения).
Ячейки (окно 30, конфигурация 1) и (окно 50, конфигурация 2) уже получены в основных запусках, здесь добавляются
две недостающие по три запуска (начальные значения 0–2). Оценка на тесте, описательная (тест не используется для выбора).
Запуск: python src/lstm_procedure_grid.py [FD001 ...]
Результат: results/lstm_procedure_grid.json"""
import json
import sys
import time

import numpy as np

from data import ROOT, mae, nasa_score, prepare, rmse
from models import fit_lstm, predict_lstm

RES = ROOT / "results"
ARCH = dict(hidden=64, layers=1, dropout=0.2, weight_decay=1e-4)
CFG1 = dict(lr=1e-3, epochs=40, batch=256, patience=8)
CFG2 = dict(lr=1e-3, epochs=100, batch=64, patience=15)
CELLS = {"окно 50, конфигурация 1": (50, CFG1), "окно 30, конфигурация 2": (30, CFG2)}
SEEDS = [0, 1, 2]


def main(fds):
    out = json.load(open(RES / "lstm_procedure_grid.json")) if (RES / "lstm_procedure_grid.json").exists() else {}
    for fd in fds:
        out[fd] = {}
        for name, (w, cfg) in CELLS.items():
            d = prepare(fd, window=w)
            i = d["last_idx"]
            y = d["rul_te"][i]
            runs = []
            for s in SEEDS:
                t0 = time.time()
                model, hist = fit_lstm(d["Xtr"], d["ytr"], d["utr"], seed=s, **ARCH, **cfg)
                p = predict_lstm(model, d["Xte"])[i]
                runs.append(dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p), epochs=len(hist),
                                 fit_s=time.time() - t0))
                print(fd, name, s, runs[-1], flush=True)
            out[fd][name] = dict(runs=runs, rmse_mean=float(np.mean([r["rmse"] for r in runs])),
                                 rmse_std_ddof1=float(np.std([r["rmse"] for r in runs], ddof=1)))
        json.dump(out, open(RES / "lstm_procedure_grid.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"])
