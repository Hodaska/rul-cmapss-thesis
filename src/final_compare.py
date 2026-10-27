"""Итоговое сравнение моделей в выровненных условиях (замечания 1.2, 1.6, 2.1, 2.2, 2.4, 2.7).

Все модели: окно 50, пять начальных значений генератора (0–4), без пересмотра по тесту после финального запуска.
Деревья окна 30 приводятся для справки. Для каждой модели считаются метрики по последней точке тестовых двигателей
для каждого запуска (среднее и стандартное отклонение с ddof=1) и для ансамбля запусков.
Двухуровневый парный бутстреп (5000 повторов): в каждом повторе выбираются двигатели (одни и те же для обеих моделей)
и запуски каждой модели (с возвращением), метрика считается по запускам и усредняется (сравнение «среднее со средним»),
так что учитывается и разброс обучения; p-значения двусторонние, поправка Холма на 12 сравнений.
Запуск: python src/final_compare.py
Результат: results/final_compare.json"""
import json

import numpy as np
from scipy import stats

from data import ROOT, mae, nasa_score, rmse

RES = ROOT / "results"
FDS = ["FD001", "FD002", "FD003", "FD004"]
N_BOOT = 5000
STAGES = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 1e9)]


def load(fd):
    P = np.load(RES / f"preds_{fd}.npz")
    T50 = np.load(RES / f"trees_preds_{fd}_w50.npz")
    T30 = np.load(RES / f"trees_preds_{fd}_w30.npz")
    idx = P["last_idx"]
    preds = {
        "rf50": T50["rf"], "xgb50": T50["xgb"], "rf30": T30["rf"], "xgb30": T30["xgb"],
        "lstm50": np.array([P[f"lstm_v2_seed{s}"] for s in range(5)]),
    }
    return dict(P=P, idx=idx, y=P["rul_te"][idx], preds=preds)


def last_metrics(y, p):
    return dict(rmse=rmse(y, p), mae=mae(y, p), score=nasa_score(y, p))


def bootstrap(y, a, b, metric, rng):
    """Разность метрики a − b: среднее по запускам, парный бутстреп по двигателям."""
    n = len(y)
    if metric == "rmse":
        ea, eb = (a - y) ** 2, (b - y) ** 2  # (запуски, двигатели)
        f = lambda e, s: np.sqrt(e[:, s].mean(axis=1)).mean()
    elif metric == "mae":
        ea, eb = np.abs(a - y), np.abs(b - y)
        f = lambda e, s: e[:, s].mean(axis=1).mean()
    else:
        def sc(p):
            d = p - y
            return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
        ea, eb = sc(a), sc(b)
        f = lambda e, s: e[:, s].sum(axis=1).mean()
    diffs = np.empty(N_BOOT)
    sa, sb = len(ea), len(eb)
    for i in range(N_BOOT):
        s = rng.randint(0, n, n)
        # двухуровневый бутстреп: выборка двигателей и выборка запусков (с возвращением) для каждой модели
        ia, ib = rng.randint(0, sa, sa), rng.randint(0, sb, sb)
        diffs[i] = f(ea[ia], s) - f(eb[ib], s)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return dict(diff=float(f(ea, np.arange(n)) - f(eb, np.arange(n))), lo=float(lo), hi=float(hi), p=float(max(p, 1 / N_BOOT)))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    out = {"datasets": {}}
    rng = np.random.RandomState(0)
    comps, pvals = [], []
    for fd in FDS:
        D = load(fd)
        y, preds = D["y"], {k: v[:, D["idx"]] for k, v in D["preds"].items()}
        d = {"n_test": int(len(y)), "models": {}, "bootstrap": {}}
        for k, p in preds.items():
            runs = [last_metrics(y, p[s]) for s in range(len(p))]
            d["models"][k] = dict(
                runs=runs,
                mean={m: float(np.mean([r[m] for r in runs])) for m in ("rmse", "mae", "score")},
                std_ddof1={m: float(np.std([r[m] for r in runs], ddof=1)) for m in ("rmse", "mae", "score")},
                ensemble=last_metrics(y, p.mean(axis=0)),
                bias=float(np.mean(p.mean(axis=0) - y)), late_share=float(np.mean(p.mean(axis=0) > y)))
        for a, b in [("lstm50", "xgb50"), ("lstm50", "rf50"), ("rf50", "xgb50")]:
            r = {m: bootstrap(y, preds[a], preds[b], m, rng) for m in ("rmse", "mae", "score")}
            # сравнение разброса запусков: критерий Уэлча по RMSE отдельных запусков (n = 5 против 5)
            ra = [x["rmse"] for x in d["models"][a]["runs"]]
            rb = [x["rmse"] for x in d["models"][b]["runs"]]
            r["welch_seed_p"] = float(stats.ttest_ind(ra, rb, equal_var=False).pvalue)
            d["bootstrap"][f"{a}-{b}"] = r
            comps.append((fd, f"{a}-{b}"))
            pvals.append(r["rmse"]["p"])
        # ансамбль с ансамблем
        ens = {k: p.mean(axis=0)[None] for k, p in preds.items()}
        d["bootstrap_ensemble"] = {f"{a}-{b}": bootstrap(y, ens[a], ens[b], "rmse", rng)
                                   for a, b in [("lstm50", "xgb50"), ("lstm50", "rf50"), ("rf50", "xgb50")]}
        # группы по истинному RUL (среднее по запускам)
        groups = {"до 50": y <= 50, "50–100": (y > 50) & (y <= 100), "свыше 100": y > 100}
        d["groups"] = {g: {"n": int(m.sum()), **{k: float(np.mean([rmse(y[m], preds[k][s][m]) for s in range(5)]))
                                                for k in ("rf50", "xgb50", "lstm50")}} for g, m in groups.items()}
        d["groups_mae"] = {g: {k: float(np.mean([mae(y[m], preds[k][s][m]) for s in range(5)]))
                               for k in ("rf50", "xgb50", "lstm50")} for g, m in groups.items()}
        yc = np.minimum(y, 125)
        d["capped_rmse"] = {k: float(np.mean([rmse(yc, preds[k][s]) for s in range(5)])) for k in ("rf50", "xgb50", "lstm50")}
        d["n_above_125"] = int(np.sum(y > 125))
        # крупные запоздалые промахи (вклад в функцию оценки), по среднему прогнозу запусков
        units = D["P"]["ute"][D["idx"]]
        big = {}
        for k in ("xgb50", "rf50", "lstm50"):
            pm = preds[k].mean(axis=0)
            dd = pm - y
            sc = np.where(dd < 0, np.exp(-dd / 13) - 1, np.exp(dd / 10) - 1)
            top = np.argsort(-sc)[:5]
            big[k] = [dict(unit=int(units[i]), true=float(y[i]), pred=float(pm[i]), score=float(sc[i]),
                           share=float(sc[i] / sc.sum())) for i in top]
        d["top_score_engines"] = big
        # по всем окнам тестовых траекторий: ошибки по стадиям (истинный RUL с порогом 125), только для FD001 и FD003 в тексте
        P = D["P"]
        cap = np.minimum(P["rul_te"], 125)
        st = {}
        for lo, hi in STAGES:
            m = (cap >= lo) & (cap < hi) if hi < 1e8 else (P["rul_te"] > 125)
            if lo == 100:
                m = (cap >= 100) & (P["rul_te"] <= 125)
            st[f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"] = dict(
                n=int(m.sum()),
                **{k: dict(mae=float(np.mean([mae(cap[m], D["preds"][k][s][m]) for s in range(5)])),
                           rmse=float(np.mean([rmse(cap[m], D["preds"][k][s][m]) for s in range(5)])))
                   for k in ("rf50", "xgb50", "lstm50")})
        d["stages_all_windows"] = st
        out["datasets"][fd] = d
    ens_keys = [(fd, k) for fd in FDS for k in out["datasets"][fd]["bootstrap_ensemble"]]
    ens_p = np.array([out["datasets"][fd]["bootstrap_ensemble"][k]["p"] for fd, k in ens_keys])
    out["holm_ensemble"] = {f"{fd}:{k}": dict(p=float(p), p_holm=float(a)) for (fd, k), p, a in zip(ens_keys, ens_p, holm(ens_p))}
    adj = holm(np.array(pvals))
    out["holm"] = {f"{fd}:{name}": dict(p=float(p), p_holm=float(a)) for (fd, name), p, a in zip(comps, pvals, adj)}
    json.dump(out, open(RES / "final_compare.json", "w"), ensure_ascii=False, indent=1)
    for fd in FDS:
        d = out["datasets"][fd]
        print(fd, {k: (round(v["mean"]["rmse"], 2), round(v["std_ddof1"]["rmse"], 2), round(v["ensemble"]["rmse"], 2))
                   for k, v in d["models"].items()})
        for k, v in d["bootstrap"].items():
            h = out["holm"][f"{fd}:{k}"]
            print("  ", k, {m: round(v["rmse"][m], 2) for m in ("diff", "lo", "hi")}, "p", round(h["p"], 4),
                  "p_holm", round(h["p_holm"], 4), "welch", round(v["welch_seed_p"], 4))


if __name__ == "__main__":
    main()
