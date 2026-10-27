"""Графики главы 3 и раздела 4 по results/*. Запуск: python src/plots.py
Требует results/final_compare.json, results/cv_nested.json, results/repair_cost.json (см. приложения В и Г)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data import ROOT, feature_names, prepare, window_features

FIG = ROOT / "figures"
RES = ROOT / "results"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13, "axes.grid": True, "grid.alpha": 0.3})
NAMES = {"ridge": "Ridge", "rf50": "Random Forest", "xgb50": "XGBoost", "lstm50": "LSTM"}
COL = {"ridge": "#999999", "rf50": "#228833", "xgb50": "#ee7733", "lstm50": "#4477aa"}
FDS = ["FD001", "FD002", "FD003", "FD004"]

R = json.load(open(RES / "ridge_w50.json"))
M = {fd: json.load(open(RES / f"metrics_{fd}.json")) for fd in FDS}
C = json.load(open(RES / "final_compare.json"))["datasets"]
P = {fd: np.load(RES / f"preds_{fd}.npz") for fd in FDS}
T = {fd: np.load(RES / f"trees_preds_{fd}_w50.npz") for fd in FDS}


def ens(fd, k):
    """Прогноз ансамбля пяти запусков по всем окнам тестовых траекторий."""
    if k == "lstm50":
        return np.mean([P[fd][f"lstm_v2_seed{s}"] for s in range(5)], axis=0)
    return T[fd][k[:-2]].mean(axis=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# 1. сравнение RMSE и MAE по подмножествам (среднее по пяти запускам, усы – стандартное отклонение, n − 1)
for metric, fname in [("rmse", "cmp_rmse.png"), ("mae", "cmp_mae.png")]:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for i, k in enumerate(["ridge", "rf50", "xgb50", "lstm50"]):
        vals, errs = [], []
        for fd in FDS:
            if k == "ridge":
                vals.append(R[fd]["ridge"][metric])
                errs.append(0)
            else:
                vals.append(C[fd]["models"][k]["mean"][metric])
                errs.append(C[fd]["models"][k]["std_ddof1"][metric])
        ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, yerr=errs, capsize=3, label=NAMES[k], color=COL[k])
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel(metric.upper() + ", циклов")
    ax.legend(ncol=4, fontsize=11, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    save(fig, fname)

# 2. кривые обучения LSTM (FD001)
fig, ax = plt.subplots(figsize=(7, 4))
for s, h in enumerate(M["FD001"]["lstm_v2_val_history"]):
    ax.plot(range(1, len(h) + 1), h, label=f"Запуск {s + 1}")
ax.set_xlabel("Эпоха")
ax.set_ylabel("RMSE на валидации, циклов")
ax.set_ylim(10, 35)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(fontsize=11)
save(fig, "lstm_curves.png")

d = P["FD001"]
idx = d["last_idx"]
y = d["rul_te"][idx]
KS = ["rf50", "xgb50", "lstm50"]

# 3. предсказано против истины (ансамбль пяти запусков)
fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
for ax, k in zip(axs, KS):
    ax.scatter(y, ens("FD001", k)[idx], s=16, color=COL[k], alpha=0.8)
    ax.plot([0, 150], [0, 150], "k--", lw=1)
    ax.set_title(NAMES[k])
    ax.set_xlabel("Истинный RUL, циклов")
axs[0].set_ylabel("Предсказанный RUL, циклов")
save(fig, "scatter_fd001.png")

# 4. распределение ошибок: отдельные панели, ступенчатые гистограммы
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharex=True, sharey=True)
bins = np.linspace(-60, 60, 25)
for ax, k in zip(axs, KS):
    e = ens("FD001", k)[idx] - y
    ax.hist(e, bins=bins, histtype="stepfilled", alpha=0.35, color=COL[k])
    ax.hist(e, bins=bins, histtype="step", lw=1.6, color=COL[k])
    ax.axvline(0, color="k", lw=1)
    ax.set_title(NAMES[k])
    ax.set_xlabel("Ошибка, циклов")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylabel("Число двигателей")
save(fig, "err_hist_fd001.png")

# 5. траектории четырёх тестовых двигателей (все окна, ансамбль пяти запусков)
units = np.unique(d["ute"])
order = np.argsort(y)
pick = [units[order[5]], units[order[35]], units[order[65]], units[order[95]]]
fig, axs = plt.subplots(2, 2, figsize=(11, 6.8), sharey=True)
for ax, u in zip(axs.ravel(), pick):
    m = d["ute"] == u
    c = d["cte"][m]
    ax.plot(c, d["rul_te"][m], "k-", lw=2, label="Истинный RUL")
    ax.plot(c, np.minimum(d["rul_te"][m], 125), "k:", lw=1.2, label="RUL с порогом 125")
    for k in KS:
        ax.plot(c, ens("FD001", k)[m], color=COL[k], lw=1.3, label=NAMES[k])
    ax.set_title(f"Двигатель № {u}")
    ax.set_xlabel("Цикл")
axs[0, 0].set_ylabel("RUL, циклов")
axs[1, 0].set_ylabel("RUL, циклов")
axs[0, 0].set_ylim(-5, 165)
axs[0, 0].legend(fontsize=10, loc="lower left")
save(fig, "traj_fd001.png")

# 6. ошибка по стадиям жизни (все окна тестовых двигателей FD001, среднее по запускам)
labels = ["0–25", "25–50", "50–75", "75–100", "100–125", ">125"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
w = 0.25
st = C["FD001"]["stages_all_windows"]
for i, k in enumerate(KS):
    ax.bar(np.arange(6) + (i - 1) * w, [v[k]["mae"] for v in st.values()], w, label=NAMES[k], color=COL[k])
ax.set_xticks(range(6))
ax.set_xticklabels(labels)
ax.set_xlabel("Истинный RUL, циклов")
ax.set_ylabel("MAE относительно RUL\nс порогом 125, циклов")
ax.legend(fontsize=11)
save(fig, "stage_error_fd001.png")

# 7. важность признаков (окно 50, FD001; параметры деревьев из results/trees_aligned_FD001.json)
dd = prepare("FD001", window=50)
F = window_features(dd["Xtr"])
tp = json.load(open(RES / "trees_aligned_FD001.json"))["w50"]
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, **tp["rf_params"]).fit(F, dd["ytr"])
xg = XGBRegressor(n_jobs=-1, random_state=0, tree_method="hist", **tp["xgb_params"]).fit(F, dd["ytr"])
names = feature_names()
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, m, t in zip(axs, [rf, xg], ["Random Forest", "XGBoost"]):
    imp = m.feature_importances_
    top = np.argsort(imp)[::-1][:12][::-1]
    ax.barh([names[i] for i in top], imp[top], color="#4477aa")
    ax.set_title(t)
save(fig, "importance_fd001.png")

# 8. чувствительность по результатам вложенной проверки (FD001): порог ограничения RUL и длина окна
cv = json.load(open(RES / "cv_nested.json"))["configs"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
caps = [c for c in cv if c["window"] == 50]
caps.sort(key=lambda c: (c["cap"] is None, c["cap"] or 0))
wins = sorted([c for c in cv if c["cap"] == 125], key=lambda c: c["window"])
for ax, rows, xs, lab in [(axs[0], caps, [("без" if c["cap"] is None else str(c["cap"])) for c in caps],
                           "Порог ограничения RUL, циклов"),
                          (axs[1], wins, [str(c["window"]) for c in wins], "Длина окна, циклов")]:
    for k, col, n in [("xgb", "#ee7733", "XGBoost"), ("lstm", "#4477aa", "LSTM")]:
        v = np.array([[r["rmse_last"] for r in c[k]] for c in rows], dtype=object)
        mean = [np.mean([r["rmse_last"] for r in c[k]]) for c in rows]
        sd = [np.std([r["rmse_last"] for r in c[k]], ddof=1) for c in rows]
        ax.errorbar(xs, mean, yerr=sd, fmt="o-", color=col, capsize=3, label=n)
    ax.set_xlabel(lab)
    ax.set_ylabel("RMSE, циклов (проверка)")
    ax.legend(fontsize=11)
save(fig, "sensitivity_fd001.png")

# 9. затраты правила ремонта (раздел 4.3): FD001, время упреждения 10 и 20 циклов
rc = json.load(open(RES / "repair_cost.json"))["FD001"]["scenarios"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, lead in zip(axs, [10, 20]):
    for s in [x for x in rc if x["lead"] == lead and x["f_ratio"] in (5, 20, 50)]:
        taus = sorted(int(t) for t in s["mean_cost"])
        mean = np.array([s["mean_cost"][str(t)] for t in taus])
        ax.plot(taus, mean, label=f"F = {int(s['F'])}")
        ax.plot(s["best_tau"], s["mean_cost"][str(s["best_tau"])], "k*", ms=9)
    ax.set_title(f"Время упреждения L = {lead}")
    ax.set_xlim(5, 80)
    ax.set_ylim(0, 200)
    ax.set_xlabel("Порог срабатывания τ, циклов")
    ax.set_ylabel("Средние затраты на двигатель")
    ax.legend(fontsize=11)
save(fig, "repair_cost_fd001.png")
print("ok")
