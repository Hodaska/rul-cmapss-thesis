# Прогнозирование остаточного ресурса оборудования на данных C-MAPSS

Код и результаты экспериментов выпускной квалификационной работы «Прогнозирование остаточного ресурса оборудования
методами машинного обучения для поддержки решений о техническом обслуживании» (направление 27.03.04).

Сравниваются константная модель, Ridge, Random Forest, XGBoost и LSTM на четырёх подмножествах NASA C-MAPSS
(FD001–FD004): окно 50 циклов, пять начальных значений генератора случайных чисел, парный бутстреп с поправкой Холма.
Отдельно реализованы интервал прогноза, модель затрат для порога ремонта и минимальный прототип сервиса прогноза.

## Данные
Набор C-MAPSS берётся со страницы NASA Prognostics Center of Excellence
(https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
и распаковывается в каталог `data/` (файлы `train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt`).

## Установка
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Версии библиотек в `requirements.txt` соответствуют запуску на Python 3.13. Расчёты идут на центральном процессоре.

## Порядок запуска
```
python src/run_experiments.py      # первоначальный эксперимент (деревья с окном 30)
python src/ridge_w50.py            # константа и Ridge с окном 50
python src/run_trees_aligned.py    # деревья: окна 30 и 50, пять начальных значений
python src/run_lstm_v2.py          # итоговый запуск LSTM, пять начальных значений
python src/cv_lstm_nested.py       # вложенная проверка окна и порога ограничения RUL (FD001)
python src/ablation_modes.py       # абляция по режимам работы (FD002, FD004)
python src/lstm_procedure_grid.py  # окно × процедура обучения LSTM
python src/supp_window70.py        # окно 70 для сети и деревьев
python src/final_compare.py        # итоговые таблицы, бутстреп -> results/final_compare.json
python src/repair_cost.py          # правило ремонта с моделью затрат
python src/rul_service.py build    # обучение модели прототипа
python src/rul_service.py demo 24  # демонстрация на тестовом двигателе № 24
python src/benchmark_time.py       # время обучения (запускать на свободном процессоре)
python src/eda.py                  # разведочный анализ
python src/plots.py                # рисунки
python src/plot_contour.py         # схема контура управления
```
Полный прогон занимает несколько часов. Готовые результаты лежат в `results/`, рисунки — в `figures/`.

## Структура
- `src/data.py` — загрузка, нормализация по режимам, окна, признаки для деревьев, метрики;
- `src/models.py` — Ridge, Random Forest, XGBoost, LSTM;
- `src/rul_service.py` — прототип сервиса прогноза (RUL, интервал, тревога);
- `results/` — json и npz с результатами и журналы запусков.
