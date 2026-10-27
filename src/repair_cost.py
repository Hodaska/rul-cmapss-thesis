"""Имитация правила «ремонт при прогнозе RUL не выше τ» с моделью затрат (раздел 3.10 пересмотрен по замечанию 1.9).

Прогнозы внеблочные: XGBoost обучен по GroupKFold(5) по двигателям, прогноз для каждого цикла двигателя
получен моделью, не видевшей этот двигатель (results/oof_xgb_FDxxx.npz). Для всех обучающих двигателей момент
отказа известен, поэтому исход правила определён для каждого двигателя.
Исходы для двигателя с длиной жизни T и первым циклом срабатывания t (прогноз ≤ τ), r = T − t – истинный RUL:
  - r ≥ L: плановый ремонт, затраты P + c·(r − L), где L – время упреждения на подготовку ремонта;
  - r < L или правило не сработало: внеплановый отказ, затраты F.
Затраты выражены в условных единицах: c = 1 за каждый недоиспользованный цикл, P = 20, F = k·P.
Запуск: python src/repair_cost.py
Результат: results/repair_cost.json"""
import json

import numpy as np

from data import ROOT

RES = ROOT / "results"
TAUS = list(range(5, 101, 5))
LEADS = [0, 10, 20]
P = 20.0
F_RATIO = [5, 10, 20, 50]
N_BOOT = 2000


def outcomes(oof, unit, cycle, rul, tau, lead):
    """Для каждого двигателя: (r при срабатывании или NaN, длина жизни)."""
    res = []
    for u in np.unique(unit):
        m = np.where(unit == u)[0]
        order = np.argsort(cycle[m])
        p, y = oof[m][order], rul[m][order]
        hit = np.where(p <= tau)[0]
        res.append(y[hit[0]] if len(hit) else np.nan)
    return np.array(res)


def costs(r, lead, f):
    planned = r >= lead
    return np.where(planned, P + (r - lead), f)  # NaN >= lead даёт False -> отказ


def fixed_age(life, f):
    """Плановая замена в фиксированном возрасте a (одинаковом для всех двигателей), лучший a по тем же траекториям."""
    best = None
    for a in range(50, int(life.max()) + 1, 5):
        c = np.where(life >= a, P + (life - a), f).mean()
        if best is None or c < best[1]:
            best = (a, float(c))
    return dict(fixed_age_best=best[0], cost_fixed_age=best[1])


def run(fd):
    d = np.load(RES / f"oof_xgb_{fd}.npz")
    oof, unit, cycle, rul = d["oof"], d["unit"], d["cycle"], d["rul"]
    n = len(np.unique(unit))
    table = {tau: outcomes(oof, unit, cycle, rul, tau, 0) for tau in TAUS}
    rng = np.random.RandomState(0)
    boots = [rng.randint(0, n, n) for _ in range(N_BOOT)]
    life = np.array([cycle[unit == u].max() + 1 for u in np.unique(unit)], dtype=float)  # длина жизни T
    out = {"dataset": fd, "n_engines": n, "scenarios": []}
    for lead in LEADS:
        for k in F_RATIO:
            f = k * P
            c_tau = {tau: costs(table[tau], lead, f) for tau in TAUS}
            mean = {tau: float(c_tau[tau].mean()) for tau in TAUS}
            best = min(mean, key=mean.get)
            boot_best = [min(TAUS, key=lambda t: c_tau[t][b].mean()) for b in boots]
            se = {tau: float(c_tau[tau].std(ddof=1) / np.sqrt(n)) for tau in TAUS}
            flat = [t for t in TAUS if mean[t] <= mean[best] + se[best]]
            late = {tau: float(np.mean(~(table[tau] >= lead))) for tau in TAUS}
            unused = {tau: (float(np.mean(table[tau][table[tau] >= lead] - lead)) if np.any(table[tau] >= lead) else None)
                      for tau in TAUS}
            out["scenarios"].append(dict(
                lead=lead, f_ratio=k, F=f, mean_cost=mean, se_cost=se, best_tau=best,
                best_tau_boot_quantiles=[float(np.percentile(boot_best, q)) for q in (5, 50, 95)],
                taus_within_1se=flat, late_share=late, mean_unused_life=unused,
                cost_run_to_failure=f, cost_perfect_forecast=P, **fixed_age(life, f)))
    return out


def main():
    res = {fd: run(fd) for fd in ("FD001", "FD002", "FD003", "FD004") if (RES / f"oof_xgb_{fd}.npz").exists()}
    json.dump(res, open(RES / "repair_cost.json", "w"), ensure_ascii=False, indent=1)
    for fd, r in res.items():
        for s in r["scenarios"]:
            print(fd, "L", s["lead"], "F/P", s["f_ratio"], "tau*", s["best_tau"], "boot 5/50/95",
                  s["best_tau_boot_quantiles"], "1se", s["taus_within_1se"], flush=True)


if __name__ == "__main__":
    main()
