"""Разведочный анализ FD001: рисунки и числа для главы 2. Запуск: python src/eda.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from data import COLS, ROOT, SENSORS, add_rul, load_raw

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13, "axes.grid": True, "grid.alpha": 0.3})

stats = {}
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    tr, te, rl = load_raw(fd)
    life = tr.groupby("unit")["cycle"].max()
    stats[fd] = dict(train_units=int(tr.unit.nunique()), test_units=int(te.unit.nunique()),
                     train_rows=len(tr), test_rows=len(te),
                     life_min=int(life.min()), life_mean=float(life.mean()), life_max=int(life.max()),
                     test_len_min=int(te.groupby("unit")["cycle"].max().min()),
                     rul_test_min=int(rl.min()), rul_test_max=int(rl.max()))

tr, te, rl = load_raw("FD001")
tr, te = add_rul(tr, te, rl)
sens_all = [f"s{i}" for i in range(1, 22)]
sd = tr[sens_all].std()
stats["constant_sensors_FD001"] = [s for s in sens_all if sd[s] < 1e-6]
stats["low_var_sensors_FD001"] = [s for s in sens_all if 1e-6 <= sd[s] and s not in SENSORS]

# рис. 1: распределение длин жизни
life = tr.groupby("unit")["cycle"].max()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(life, bins=20, color="#4477aa", edgecolor="white")
ax.set_xlabel("Число циклов до отказа")
ax.set_ylabel("Число двигателей")
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(FIG / "eda_life_hist.png", dpi=200)
plt.close(fig)

# рис. 2: траектории нескольких датчиков для 5 двигателей
show = ["s2", "s4", "s7", "s11", "s12", "s15"]
fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))
for ax, s in zip(axs.ravel(), show):
    for u in [1, 2, 3, 4, 5]:
        g = tr[tr.unit == u]
        ax.plot(g["cycle"], g[s], lw=1)
    ax.set_title(s)
    ax.set_xlabel("Цикл")
fig.tight_layout()
fig.savefig(FIG / "eda_sensor_trends.png", dpi=200)
plt.close(fig)

# рис. 3: те же датчики против RUL (общая картина деградации)
fig, axs = plt.subplots(2, 3, figsize=(10, 5.5))
for ax, s in zip(axs.ravel(), show):
    ax.scatter(tr["rul"], tr[s], s=1, alpha=0.15, color="#4477aa")
    ax.set_title(s)
    ax.set_xlabel("RUL")
    ax.invert_xaxis()
fig.tight_layout()
fig.savefig(FIG / "eda_sensor_vs_rul.png", dpi=200)
plt.close(fig)

# рис. 4: корреляция датчиков с RUL (с ограничением 125) - столбцы
tr["rul_c"] = tr["rul"].clip(upper=125)
corr = tr[sens_all + ["rul_c"]].corr()["rul_c"].drop("rul_c")
stats["corr_with_rul_FD001"] = {k: (None if pd.isna(v) else float(v)) for k, v in corr.items()}
fig, ax = plt.subplots(figsize=(8, 3.5))
c = corr.fillna(0)
ax.bar(c.index, c.values, color=["#cc6677" if s in SENSORS else "#bbbbbb" for s in c.index])
ax.set_ylabel("Корреляция с RUL")
ax.tick_params(axis="x", rotation=60)
fig.tight_layout()
fig.savefig(FIG / "eda_corr.png", dpi=200)
plt.close(fig)

# рис. 5: кусочно-линейный RUL для одного двигателя
g = tr[tr.unit == 1]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(g["cycle"], g["rul"], label="Линейный RUL")
ax.plot(g["cycle"], g["rul"].clip(upper=125), label="Ограниченный RUL (125)")
ax.set_xlabel("Цикл")
ax.set_ylabel("RUL, циклов")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "eda_rul_cap.png", dpi=200)
plt.close(fig)

json.dump(stats, open(ROOT / "results" / "eda_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in stats.items() if k != "corr_with_rul_FD001"}, ensure_ascii=False, indent=1))
