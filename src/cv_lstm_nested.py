"""Вложенная перекрёстная проверка для LSTM и XGBoost на FD001 (замечания 1.4, 1.5, 1.8).

Внешний уровень: GroupKFold(3) по двигателям обучающей выборки. Внутри каждого обучающего блока 15 % двигателей
случайно отделяются для ранней остановки (отдельная отложенная часть, не совпадающая с внешним блоком).
Метрика внешнего блока считается как на тесте: по последней точке усечённой траектории, истинный RUL без
ограничения, лежащий в диапазоне тестовых значений FD001 (7–145 циклов); для каждого двигателя блока берётся N_CUTS случайных точек усечения (одинаковых для всех
конфигураций). Для сравнения считается и RMSE по всем окнам блока (с ограниченным RUL) на выбранной эпохе и
оптимистичная оценка min(hist) по внутренней части, как в прежнем листинге.
Конфигурации: длина окна при пороге 125 и порог ограничения RUL при окне 50 (итоговая конфигурация сети).
Правило решения (зафиксировано до получения результатов): окно 50, принятое при первом пересмотре, сохраняется, если его
средний RMSE проверки не превышает лучшее значение среди окон более чем на одну стандартную ошибку (стандартное
отклонение по 6 запускам блок × начальное значение, делённое на корень из 6). Проверка независима от выбора окна на
тесте, но не от самого пересмотра. Ограничение: параметры XGBoost взяты из подбора по всем обучающим двигателям.
Запуск: python src/cv_lstm_nested.py
Результат: results/cv_nested.json"""
import json
import time

import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from data import ROOT, prepare, rmse, window_features
from models import fit_lstm, predict_lstm

RES = ROOT / "results"
FD = "FD001"
ARCH = dict(hidden=64, layers=1, dropout=0.2, weight_decay=1e-4)
TRAIN_CFG = dict(lr=1e-3, epochs=100, batch=64, patience=15)
SEEDS = [0, 1]
N_FOLDS = 3
N_CUTS = 10
MIN_CUT = 31
RUL_MIN, RUL_MAX = 7, 145
CONFIGS = [(w, 125) for w in (10, 20, 30, 50, 70)] + [(50, c) for c in (None, 100, 150)]


def cut_rows(d, units_block, rng):
    """Индексы окон, соответствующих случайным точкам усечения двигателей блока."""
    rows = []
    for u in units_block:
        idx = np.where(d["utr"] == u)[0]
        cyc, r = d["ctr"][idx], d["rul_tr"][idx]
        cand = idx[(cyc >= MIN_CUT) & (r >= RUL_MIN) & (r <= RUL_MAX)]  # как в тесте FD001: RUL от 7 до 145
        rows.extend(rng.choice(cand, size=N_CUTS, replace=False))
    return np.array(rows)


def run_config(window, cap, xgb_params):
    d = prepare(FD, window=window, cap=cap)
    uniq = np.unique(d["utr"])
    F = window_features(d["Xtr"])
    res = {"window": window, "cap": cap, "lstm": [], "xgb": []}
    for fold, (_, val_pos) in enumerate(GroupKFold(N_FOLDS).split(uniq, groups=uniq)):
        val_units = uniq[val_pos]
        trm = ~np.isin(d["utr"], val_units)
        rows = cut_rows(d, val_units, np.random.RandomState(1000 + fold))
        allrows = np.where(~trm)[0]
        y_cut = d["rul_tr"][rows]
        xg = XGBRegressor(n_jobs=-1, random_state=0, tree_method="hist", **xgb_params).fit(F[trm], d["ytr"][trm])
        px = xg.predict(F[rows])
        res["xgb"].append(dict(fold=fold, rmse_last=rmse(y_cut, px),
                               rmse_all=rmse(d["ytr"][allrows], xg.predict(F[allrows]))))
        for s in SEEDS:
            t0 = time.time()
            model, hist = fit_lstm(d["Xtr"][trm], d["ytr"][trm], d["utr"][trm], seed=s, **ARCH, **TRAIN_CFG)
            p = predict_lstm(model, d["Xtr"][rows])
            r = dict(fold=fold, seed=s, rmse_last=rmse(y_cut, p),
                     rmse_all=rmse(d["ytr"][allrows], predict_lstm(model, d["Xtr"][allrows])),
                     inner_min_hist=float(min(hist)), epochs=len(hist), fit_s=time.time() - t0)
            res["lstm"].append(r)
            print(window, cap, r, flush=True)
    for k in ("lstm", "xgb"):
        for m in ("rmse_last", "rmse_all"):
            res[f"{k}_{m}_mean"] = float(np.mean([r[m] for r in res[k]]))
    return res


def main():
    xgb_params = json.load(open(RES / f"trees_aligned_{FD}.json"))["w50"]["xgb_params"]
    out = []
    for w, c in CONFIGS:
        out.append(run_config(w, c, xgb_params))
        json.dump(dict(arch=ARCH, train_cfg=TRAIN_CFG, seeds=SEEDS, n_folds=N_FOLDS, n_cuts=N_CUTS, configs=out),
                  open(RES / "cv_nested.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
