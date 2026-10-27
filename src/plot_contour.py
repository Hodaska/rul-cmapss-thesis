"""Схема места модели прогноза RUL в контуре управления обслуживанием (рисунок к разделу 4.1).
Запуск: python src/plot_contour.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data import ROOT

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13})
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

boxes = {
    "sens": (0.2, 4.1, "Датчики\nоборудования", "#dddddd"),
    "plc": (2.7, 4.1, "ПЛК / SCADA\n(сбор данных)", "#dddddd"),
    "arch": (5.2, 4.1, "Архив данных\n(историан)", "#dddddd"),
    "svc": (7.7, 4.1, "Сервис\nпрогноза RUL", "#bcd4ea"),
    "arm": (7.5, 1.6, "АРМ инженера\n(решение о ремонте)", "#dddddd"),
    "mes": (4.2, 1.6, "Система ТОиР / MES\n(заявки, запчасти)", "#dddddd"),
}
for key, (x, y, text, col) in boxes.items():
    w = 2.1 if key not in ("mes", "arm") else 2.4
    ax.add_patch(FancyBboxPatch((x, y), w, 1.3, boxstyle="round,pad=0.05", fc=col, ec="black", lw=1.2))
    ax.text(x + w / 2, y + 0.65, text, ha="center", va="center")


def arrow(p, q, label=None, lab_xy=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.3, color="black"))
    if label:
        ax.text(*lab_xy, label, ha="center", va="center", fontsize=11)


arrow((2.3, 4.75), (2.7, 4.75))
arrow((4.8, 4.75), (5.2, 4.75))
arrow((7.3, 4.75), (7.7, 4.75))
arrow((8.75, 4.1), (8.75, 2.9))
ax.text(8.6, 3.5, "RUL, интервал,\nсигнал тревоги", ha="right", va="center", fontsize=11)
arrow((7.5, 2.25), (6.7, 2.25))
ax.text(7.1, 2.65, "заявка", ha="center", va="center", fontsize=11)
arrow((4.2, 2.25), (3.5, 2.25))
ax.text(1.9, 2.25, "Плановый ремонт\nпри достаточном\nвремени упреждения", ha="center", va="center", fontsize=11)
arrow((5.9, 2.9), (5.9, 4.1))
ax.text(5.8, 3.5, "история ремонтов", ha="right", va="center", fontsize=11)
ax.text(5.0, 0.7, "Пересчёт прогноза раз в цикл (смену) работы агрегата; переобучение модели по мере накопления отказов",
        ha="center", va="center", fontsize=11, style="italic")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "contour.png", dpi=200)
