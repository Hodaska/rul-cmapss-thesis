"""Минимальный прототип сервиса прогноза RUL (раздел 4.4): принимает окно измерений датчиков и возвращает
прогноз RUL, интервал и признак тревоги.

Модель – XGBoost на оконных признаках (окно 50), гиперпараметры из results/trees_aligned_FD001.json.
Интервал – эмпирический 80 %-й, построенный по внеблочным остаткам на обучающих двигателях
(results/oof_xgb_FD001.npz) отдельно для диапазонов прогноза. Тревога: прогноз не выше порога τ.

Режимы работы:
  python src/rul_service.py build                    – обучить модель и сохранить results/proto_FD001.json
  python src/rul_service.py demo [номер_двигателя]   – прогон по циклам тестового двигателя FD001,
                                                       рисунок figures/proto_demo.png
  python src/rul_service.py coverage                 – доля тестовых двигателей FD001 (последняя точка), попавших в интервал
  python src/rul_service.py predict файл.csv         – окно из CSV (50 строк × 14 датчиков, без заголовка)"""
import json
import sys

import numpy as np
from xgboost import XGBRegressor

from data import ROOT, SENSORS, load_raw, add_rul, normalize, make_windows, window_features, prepare

RES = ROOT / "results"
FIG = ROOT / "figures"
FD = "FD001"
WINDOW = 50
TAU = 35  # по модели затрат (раздел 4.3): время упреждения L = 20, стоимость отказа F = 20·P
BINS = [0, 25, 50, 75, 100, 1e9]
LEVEL = (10, 90)  # 80 %-й интервал


def build():
    d = prepare(FD, window=WINDOW)
    F = window_features(d["Xtr"])
    params = json.load(open(RES / f"trees_aligned_{FD}.json"))["w50"]["xgb_params"]
    model = XGBRegressor(n_jobs=-1, random_state=0, tree_method="hist", **params).fit(F, d["ytr"])
    model.save_model(str(RES / "proto_FD001_xgb.json"))
    tr, _, _ = load_raw(FD)
    mu, sd = tr[SENSORS].mean(), tr[SENSORS].std().replace(0, 1)
    o = np.load(RES / f"oof_xgb_{FD}.npz")
    res = o["rul"] - o["oof"]
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (o["oof"] >= lo) & (o["oof"] < hi)
        bins.append(dict(lo=lo, hi=hi, q_low=float(np.percentile(res[m], LEVEL[0])),
                         q_high=float(np.percentile(res[m], LEVEL[1])), n=int(m.sum())))
    json.dump(dict(window=WINDOW, sensors=SENSORS, mu=mu.tolist(), sd=sd.tolist(), bins=bins, tau=TAU),
              open(RES / "proto_FD001.json", "w"), ensure_ascii=False, indent=1)
    print("saved", bins)


class RulService:
    def __init__(self):
        cfg = json.load(open(RES / "proto_FD001.json"))
        self.cfg = cfg
        self.mu, self.sd = np.array(cfg["mu"]), np.array(cfg["sd"])
        self.model = XGBRegressor()
        self.model.load_model(str(RES / "proto_FD001_xgb.json"))

    def predict(self, window_raw, tau=None):
        """window_raw – массив (W, 14) сырых показаний датчиков в порядке SENSORS (W ≥ 1, короткое окно дополняется)."""
        tau = self.cfg["tau"] if tau is None else tau
        a = (np.asarray(window_raw, dtype=np.float32) - self.mu) / self.sd
        w = self.cfg["window"]
        if len(a) < w:
            a = np.vstack([np.repeat(a[:1], w - len(a), axis=0), a])
        a = a[-w:]
        p = float(self.model.predict(window_features(a[None]))[0])
        b = next(b for b in self.cfg["bins"] if b["lo"] <= p < b["hi"])
        lo, hi = max(0.0, p + b["q_low"]), p + b["q_high"]
        return dict(rul=round(p, 1), interval=[round(lo, 1), round(hi, 1)], alarm=bool(p <= tau), tau=tau)


def demo(unit=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr, te, rul_last = load_raw(FD)
    _, te = add_rul(tr, te, rul_last)
    g = te[te["unit"] == unit].sort_values("cycle")
    svc = RulService()
    raw = g[SENSORS].values
    out = [svc.predict(raw[max(0, i - WINDOW + 1):i + 1]) for i in range(len(g))]
    cyc = g["cycle"].values
    pred = np.array([o["rul"] for o in out])
    lo = np.array([o["interval"][0] for o in out])
    hi = np.array([o["interval"][1] for o in out])
    alarm = np.array([o["alarm"] for o in out])
    print("двигатель", unit, "циклов", len(g), "первая тревога на цикле", cyc[alarm][0] if alarm.any() else None,
          "истинный RUL в этот момент", g["rul"].values[alarm][0] if alarm.any() else None)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(cyc, lo, hi, color="#4477aa", alpha=0.25, label="80 %-й интервал")
    ax.plot(cyc, pred, color="#4477aa", label="Прогноз RUL")
    ax.plot(cyc, g["rul"].values, "k--", label="Истинный RUL")
    ax.axhline(TAU, color="#cc3311", lw=1, label=f"Порог тревоги τ = {TAU}")
    if alarm.any():
        ax.axvline(cyc[alarm][0], color="#cc3311", ls=":", lw=1)
    ax.set_xlabel("Номер цикла")
    ax.set_ylabel("RUL, циклов")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "proto_demo.png", dpi=200)


def coverage():
    tr, te, rul_last = load_raw(FD)
    _, te = add_rul(tr, te, rul_last)
    svc = RulService()
    hit, hit125, n125, widths = 0, 0, 0, []
    alarms = 0
    units = sorted(te["unit"].unique())
    for u in units:
        g = te[te["unit"] == u].sort_values("cycle")
        o = svc.predict(g[SENSORS].values[-WINDOW:])
        true = g["rul"].values[-1]
        inside = o["interval"][0] <= true <= o["interval"][1]
        hit += inside
        widths.append(o["interval"][1] - o["interval"][0])
        alarms += o["alarm"]
        if true <= 125:
            n125 += 1
            hit125 += inside
    print(f"двигателей {len(units)}; интервал накрыл истинный RUL у {hit} ({100 * hit / len(units):.0f} %), "
          f"при истинном RUL не выше 125: {hit125} из {n125} ({100 * hit125 / n125:.0f} %); ширина интервала в среднем "
          f"{np.mean(widths):.1f} цикла; тревога в последней точке у {alarms} двигателей")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "build":
        build()
    elif cmd == "demo":
        demo(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    elif cmd == "coverage":
        coverage()
    elif cmd == "predict":
        print(json.dumps(RulService().predict(np.loadtxt(sys.argv[2], delimiter=",")), ensure_ascii=False))
