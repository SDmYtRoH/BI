# RehaPiano EXACT_FORCE: ознаки тремору для кожного пальця

Pipeline: зчитування → заповнення пропусків → зниження частоти → обрізання → критерії валідності → ознаки кожного пальця → агрегація → CSV.
Далі — статистичний аналіз (`scripts/run_analysis.py`) і візуалізація (`notebooks/analyza_prstov.ipynb`).

## Встановлення

```bash
git clone <url-репозиторію> && cd <тека>
conda create -n rehapiano python=3.12 && conda activate rehapiano
pip install -r requirements.txt
```

Дані в репозиторій не входять: як їх розкласти, описано в [data/README.md](data/README.md).
Готові результати вже лежать в `outputs/` (CSV) і в ноутбуках, тож переглядати їх можна без даних.

## Запуск

```bash
conda activate rehapiano                 # numpy, pandas, scipy, matplotlib, jupyter
python scripts/build_dataset.py          # усі параметри за замовчуванням з src/rehapiano/config.py
python scripts/build_dataset.py --limit 5    # швидка перевірка на 5 файлах
python scripts/run_analysis.py           # статистика → outputs/stats/
jupyter lab notebooks/analyza_prstov.ipynb   # графіки (словацькою)
python -m unittest discover tests
```

## Структура

```
data/raw/RehaPiano/          сирі файли results_* (1912 шт., усі ігри)
data/metadata/               dataset_full.csv, dataset_patients.csv (використовуються);
                             dataset_tuke.csv, dataset_99.csv (підмножини dataset_full, не використовуються)
src/rehapiano/
  config.py                  УСІ шляхи, step_id, частоти, смуги, пороги
  io/metadata.py             читання CSV, зіставлення файлів, inventory, вікові групи
  io/reader.py               декодування JSON → base64 → zlib → pickle → пакети
  preprocessing/polarity.py  виправлення знаку каналу 9 (right_sensor_2)
  preprocessing/gaps.py      перевірка часу, усереднення дублікатів, інтерполяція на сітку 1 мс
  preprocessing/resample.py  1000 → 100 Гц (resample_poly з anti-alias фільтром)
  preprocessing/crop.py      обрізання 3–15 с
  preprocessing/pipeline.py  load_measurement_data(): усі кроки разом
  quality.py                 критерії валідності, прапорці неактивного пальця і викидів
  features/spectral.py       спектрограма, частотні смуги
  features/finger.py         ознаки пальця (+ список можливих додаткових ознак)
  aggregation.py             спроба → вимірювання → людина, неактивна рука
  stats/                     тести, демографія, активність рук, тремор пальців
scripts/build_dataset.py     CLI всього pipeline
scripts/run_analysis.py      CLI статистичного аналізу
tests/                       unit-тести
notebooks/analyza_prstov.ipynb  графіки й таблиці аналізу (словацькою)
notebooks/task_1.ipynb       перше завдання (самостійний код)
outputs/                     результати (генеруються скриптом)
```

## Дані

- **Вправа EXACT_FORCE:** людина стискає всі пальці й утримує силу на цільовому рівні. 4 спроби по 16 с (крок `d475…` — 1 спроба).
- **`step_id` збігається з групою:** здорові — `5867ebbc-…` і `d475bc4a-…`, пацієнти — лише `d2e6b7bd-…`. Різниця між групами може частково пояснюватися налаштуваннями гри.
- **Зіставлення:** ключ `(measurement_id, step_id)`; UUID з імені файла = `payload.examination_id` = `measurement_id`. `is_patient = 1`, якщо ключ є в `dataset_patients.csv`.
- **Сенсори:** канали `[0–4]` — ліва рука, `[8–12]` — права; 5–7 і 13–15 завжди 0. Канал 9 має протилежний знак (підтверджено калібруванням у `25_03_2025_calibration`) і виправляється. Без навантаження сенсори показують 0 ± 0.002, шум ~1e-4.
- **Пальці:** `sensor_1…5` = мізинець…великий — **не перевірено**, змінюється в `config.FINGER_NAMES`.

## Попередня обробка (перевірено на сирих даних)

Частота пакетів 1000–1010/с. ~25 % міток часу дублюються → усереднюються. NaN у 0.1–0.2 % рядків (найдовша серія 12 мс) → лінійна інтерполяція. Канал, у якому > 1 % NaN або розрив > 50 мс, лишається NaN. Розриви > 50 мс ділять спробу на сегменти; аналізується лише сегмент 1. Після обрізання 3–15 с — 1200 семплів при 100 Гц.

## Ознаки пальця (`features/finger.py`)

Спектрограма: Hann 5 с (0.2 Гц), крок 0.01 с, `detrend="constant"`, magnitude, ≤ 15 Гц. Рух 0–4 Гц (без DC), тремор 4–15 Гц (з межею 3 Гц 63 % «піків» лежали на краю смуги).

| Ознака | Значення |
|---|---|
| `tremor_mean_amplitude` | середня амплітуда в 4–15 Гц |
| `tremor_mean_frequency_hz` | центроїд спектра 4–15 Гц |
| `tremor_peak_frequency_hz` | частота найвищого **локального максимуму** спектра (не край смуги); NaN, якщо максимуму немає |
| `tremor_peak_amplitude` | амплітуда цієї складової |
| `mean_force` | середня сила пальця (знак виправлено) |
| `tremor_to_movement_ratio` | амплітуда 4–15 Гц / амплітуда 0–4 Гц (тремор відносно нормальної корекції сили) |
| `tremor_peak_prominence` | пік / медіана спектра 4–15 Гц (≈1–2 шум, більше — ритмічний тремор) |

## Валідність і викиди (`quality.py`)

| Рівень | Критерій | Наслідок |
|---|---|---|
| файл | не декодується, UUID не збігається, немає метаданих, дублікат | `recordings_inventory.csv`, `status = excluded` |
| спроба | час назад, неможлива частота пакетів, немає сегмента ≥ 3 с | файл/спроба не обробляється |
| спроба × палець | < 500 семплів, NaN, `std < 1e-5` (`flat_signal`) | `status = invalid` |
| спроба × палець | `mean_force < 0.002` | `finger_inactive = True`, рядок залишається |
| спроба × рука | середнє по пальцях `mean_force` або `tremor_mean_amplitude` > 3× або < 1/3 від медіани спроб цього вимірювання (≥ 3 спроби) | `attempt_outlier`, виключається з середніх |
| вимірювання × рука | те саме відносно інших вимірювань людини (≥ 3 вимірювання) | `measurement_outlier`, виключається з середніх |

**Неактивна рука:** `activity_ratio` = середня сила пальців слабшої руки / сильнішої; `< 0.10` → `inactive_hand = left/right`, інакше `none`; `unknown` — немає даних або жодна рука не тисне.

## Вихідні файли (`outputs/`)

| Файл | Рядок |
|---|---|
| `recordings_inventory.csv` | кожен файл `results_*`: matched / excluded + причина |
| `features_by_attempt_finger.csv` | спроба × рука × палець: контекст, статус, прапорці, 7 ознак |
| `features_by_measurement_finger.csv` | вимірювання × рука × палець: середнє валідних спроб без викидів |
| `features_by_person_finger.csv` | людина × step_id × рука × палець: середнє вимірювань без викидів |
| `person_summary.csv` | людина × step_id: вік, вікова група (10 років), кількість даних, сила рук, `activity_ratio`, `inactive_hand` |
| `processing_summary.csv` | скільки записів на кожному рівні, валідні / невалідні / викиди |

Здорові мають обидва контрольні кроки, тому на рівні людини вони представлені двома рядками (по одному на `step_id`).

## Статистичний аналіз (`scripts/run_analysis.py` → `outputs/stats/`)

Порівняння груп: контрольні — крок `5867…` (4 спроби, як у пацієнтів), пацієнти — `d2e6…`. Тести непараметричні, поправка Холма.
Тремор-ознаки неактивних пальців не використовуються. Додатково — підгрупа вік ≥ 50 (контрольні значно молодші).

| Файл | Що містить |
|---|---|
| `demographics_by_group`, `age_histogram`, `age_test`, `gender_by_age_group` | вік (інтервали 10 років), стать, Манн-Вітні для віку |
| `hand_activity`, `weak_hand_counts` | `force_ratio` (сила) і `fluctuation_ratio` (коливання) слабшої/сильнішої руки; кількість людей під порогами 0.10/0.25/0.50 |
| `finger_descriptives` | медіана, квартилі для групи × руки × пальця × ознаки |
| `finger_friedman`, `finger_posthoc` | чи відрізняються пальці однієї людини (Фрідман + Kendall W, попарний Вілкоксон) |
| `most_tremulous_finger` | у скількох людей який палець має найбільший тремор |
| `finger_spread_by_person`, `finger_spread_group_test` | CV і max/min між пальцями людини; порівняння груп |
| `group_comparison_by_finger` | пацієнти vs здорові для кожного пальця (r > 0 — у пацієнтів більше) |
| `left_right_by_finger` | ліва vs права рука для того самого пальця |
