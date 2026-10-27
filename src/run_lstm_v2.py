"""Итоговый запуск LSTM: окно 50, мини-пакет 64, до 100 эпох, 15 эпох без улучшения, 5 запусков на FD001..FD004.
Окно 50 принято при пересмотре конфигурации после первого запуска (раздел 3.8); вложенная перекрёстная проверка
(src/cv_lstm_nested.py) используется как независимая повторная проверка этого выбора. Валидация для ранней
остановки – случайные 15 % двигателей (GroupKFold здесь не применяется).
Результаты добавляются в results/metrics_FDxxx.json под ключами lstm_v2_*,
исходные результаты (src/run_experiments.py) не перезаписываются.
Запуск: python src/run_lstm_v2.py [FD001 ...]"""
import json
import sys
import time

import numpy as np

from data import ROOT, prepare
from models import fit_lstm, predict_lstm
from run_experiments import score_last

RES = ROOT / "results"
WINDOW = 50
ARCH = dict(hidden=64, layers=1, dropout=0.2, weight_decay=1e-4)
TRAIN_CFG = dict(lr=1e-3, epochs=100, batch=64, patience=15)
CFG = {**ARCH, **TRAIN_CFG}
N_SEEDS = 5


def run(fd):
    t0 = time.time()
    d = prepare(fd, window=WINDOW)
    mp, pp = RES / f"metrics_{fd}.json", RES / f"preds_{fd}.npz"
    out = json.load(open(mp))
    old = dict(np.load(pp))
    runs, hists, preds_seeds = [], [], {}
    for s in range(N_SEEDS):
        model, hist = fit_lstm(d["Xtr"], d["ytr"], d["utr"], seed=s, **CFG)
        p = predict_lstm(model, d["Xte"])
        preds_seeds[f"lstm_v2_seed{s}"] = p
        runs.append(score_last(d, p))
        hists.append(hist)
        print(fd, "LSTM v2 seed", s, runs[-1], flush=True)
    ens = np.mean([preds_seeds[f"lstm_v2_seed{s}"] for s in range(N_SEEDS)], axis=0)
    for k, v in preds_seeds.items():
        old[k] = v
    old["lstm_v2"] = ens
    out["lstm_v2_config"] = CFG
    out["lstm_v2_window"] = WINDOW
    out["lstm_v2_seeds"] = runs
    out["lstm_v2_val_history"] = hists
    out["metrics"]["lstm_v2_mean_of_seeds"] = {k: float(np.mean([r[k] for r in runs])) for k in ("rmse", "mae", "score")}
    out["metrics"]["lstm_v2_std_of_seeds"] = {k: float(np.std([r[k] for r in runs])) for k in ("rmse", "mae", "score")}
    out["metrics"]["lstm_v2_ensemble"] = score_last(d, ens)
    out["lstm_v2_seconds"] = time.time() - t0
    json.dump(out, open(mp, "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(pp, **old)
    print(fd, json.dumps(out["metrics"]["lstm_v2_mean_of_seeds"]), flush=True)


if __name__ == "__main__":
    for fd in (sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"]):
        run(fd)
