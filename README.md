# RehaPiano EXACT_FORCE: ukazovatele tremoru pre každý prst

**Zjednotený pipeline pre všetky hry (EXACT_FORCE + RHYTHM, štatistika, klasifikátor): [PIPELINE.md](PIPELINE.md).**

Pipeline: načítanie → doplnenie chýbajúcich hodnôt → zníženie vzorkovacej frekvencie → orezanie → kritériá platnosti → ukazovatele pre každý prst → agregácia → CSV.
Ďalej nasleduje štatistická analýza (`scripts/run_analysis.py`) a vizualizácia (`notebooks/week3/analyza_prstov.ipynb`).

## Inštalácia

```bash
git clone https://github.com/SDmYtRoH/BI.git && cd BI
conda create -n rehapiano python=3.12 && conda activate rehapiano
pip install -r requirements.txt
```

Dáta nie sú súčasťou repozitára – ako ich uložiť, je opísané v [data/README.md](data/README.md).
Hotové výsledky sú už v `outputs/` (CSV) a v notebookoch, takže ich možno prezerať aj bez dát.

## Spustenie

```bash
conda activate rehapiano                 # numpy, pandas, scipy, matplotlib, jupyter
python scripts/build_dataset.py          # všetky predvolené parametre zo src/rehapiano/config.py
python scripts/build_dataset.py --limit 5    # rýchla kontrola na 5 súboroch
python scripts/run_analysis.py           # štatistika → outputs/stats/
jupyter lab notebooks/week3/analyza_prstov.ipynb   # grafy
python -m unittest discover tests
```

## Štruktúra

```
data/raw/RehaPiano/          surové súbory results_* (1912 ks, všetky hry)
data/metadata/               dataset_full.csv, dataset_patients.csv (používajú sa);
                             dataset_tuke.csv, dataset_99.csv (podmnožiny dataset_full, nepoužívajú sa)
src/rehapiano/
  config.py                  VŠETKY cesty, step_id, frekvencie, pásma, prahy
  io/metadata.py             čítanie CSV, priradenie súborov, inventár, vekové skupiny
  io/reader.py               dekódovanie JSON → base64 → zlib → pickle → pakety
  preprocessing/polarity.py  oprava znamienka kanála 9 (right_sensor_2)
  preprocessing/gaps.py      kontrola času, priemerovanie duplicít, interpolácia na mriežku 1 ms
  preprocessing/resample.py  1000 → 100 Hz (resample_poly s anti-aliasingovým filtrom)
  preprocessing/crop.py      orezanie 3–15 s
  preprocessing/pipeline.py  load_measurement_data(): všetky kroky spolu
  quality.py                 kritériá platnosti, príznaky neaktívneho prsta a odľahlých hodnôt
  features/spectral.py       spektrogram, frekvenčné pásma
  features/finger.py         ukazovatele prsta (+ zoznam možných ďalších ukazovateľov)
  aggregation.py             pokus → meranie → osoba, neaktívna ruka
  stats/                     testy, demografia, aktivita rúk, tremor prstov
scripts/build_dataset.py     CLI celého pipeline
scripts/run_analysis.py      CLI štatistickej analýzy
tests/                       unit testy
notebooks/README.md          prehľad úloh podľa týždňov (week1–week3)
notebooks/week1/ … week3/    notebooky 1.–3. úlohy (week3/analyza_prstov.ipynb = grafy tejto analýzy)
outputs/                     výsledky (generuje ich skript)
```

## Dáta

- **Cvičenie EXACT_FORCE:** osoba stláča všetky prsty a udržiava silu na cieľovej úrovni. 4 pokusy po 16 s (krok `d475…` – 1 pokus).
- **`step_id` sa zhoduje so skupinou:** zdraví – `5867ebbc-…` a `d475bc4a-…`, pacienti – iba `d2e6b7bd-…`. Rozdiely medzi skupinami môžu čiastočne vyplývať z nastavenia hry.
- **Priradenie:** kľúč `(measurement_id, step_id)`; UUID z názvu súboru = `payload.examination_id` = `measurement_id`. `is_patient = 1`, ak je kľúč v `dataset_patients.csv`.
- **Snímače:** kanály `[0–4]` – ľavá ruka, `[8–12]` – pravá; 5–7 a 13–15 sú vždy 0. Kanál 9 má opačné znamienko (potvrdené kalibráciou v `25_03_2025_calibration`) a opravuje sa. Bez zaťaženia ukazujú snímače 0 ± 0,002, šum ~1e-4.
- **Prsty:** `sensor_1…5` = malíček…palec – **neoverené**, mení sa v `config.FINGER_NAMES`.

## Predspracovanie (overené na surových dátach)

Frekvencia paketov 1000–1010/s. ~25 % časových značiek je duplicitných → priemerujú sa. NaN v 0,1–0,2 % riadkov (najdlhšia séria 12 ms) → lineárna interpolácia. Kanál s > 1 % NaN alebo s medzerou > 50 ms zostáva NaN. Medzery > 50 ms rozdelia pokus na segmenty; analyzuje sa iba segment 1. Po orezaní na 3–15 s zostáva 1200 vzoriek pri 100 Hz.

## Ukazovatele prsta (`features/finger.py`)

Spektrogram: Hann 5 s (0,2 Hz), krok 0,01 s, `detrend="constant"`, magnitúda, ≤ 15 Hz. Pohyb 0–4 Hz (bez DC), tremor 4–15 Hz (pri hranici 3 Hz ležalo 63 % „vrcholov“ na okraji pásma).

| Ukazovateľ | Význam |
|---|---|
| `tremor_mean_amplitude` | priemerná amplitúda v pásme 4–15 Hz |
| `tremor_mean_frequency_hz` | ťažisko spektra 4–15 Hz |
| `tremor_peak_frequency_hz` | frekvencia najvyššieho **lokálneho maxima** spektra (nie okraj pásma); NaN, ak maximum neexistuje |
| `tremor_peak_amplitude` | amplitúda tejto zložky |
| `mean_force` | priemerná sila prsta (znamienko opravené) |
| `tremor_to_movement_ratio` | amplitúda 4–15 Hz / amplitúda 0–4 Hz (tremor voči bežnej korekcii sily) |
| `tremor_peak_prominence` | vrchol / medián spektra 4–15 Hz (≈1–2 šum, viac – rytmický tremor) |

## Platnosť a odľahlé hodnoty (`quality.py`)

| Úroveň | Kritérium | Dôsledok |
|---|---|---|
| súbor | nedá sa dekódovať, UUID nesedí, chýbajú metadáta, duplicita | `recordings_inventory.csv`, `status = excluded` |
| pokus | čas ide dozadu, nemožná frekvencia paketov, chýba segment ≥ 3 s | súbor/pokus sa nespracuje |
| pokus × prst | < 500 vzoriek, NaN, `std < 1e-5` (`flat_signal`) | `status = invalid` |
| pokus × prst | `mean_force < 0,002` | `finger_inactive = True`, riadok zostáva |
| pokus × ruka | priemer cez prsty `mean_force` alebo `tremor_mean_amplitude` > 3× alebo < 1/3 mediánu pokusov tohto merania (≥ 3 pokusy) | `attempt_outlier`, vylúčené z priemerov |
| meranie × ruka | to isté voči ostatným meraniam osoby (≥ 3 merania) | `measurement_outlier`, vylúčené z priemerov |

**Neaktívna ruka:** `activity_ratio` = priemerná sila prstov slabšej ruky / silnejšej; `< 0,10` → `inactive_hand = left/right`, inak `none`; `unknown` – chýbajú dáta alebo netlačí ani jedna ruka.

## Výstupné súbory (`outputs/`)

| Súbor | Riadok |
|---|---|
| `recordings_inventory.csv` | každý súbor `results_*`: matched / excluded + dôvod |
| `features_by_attempt_finger.csv` | pokus × ruka × prst: kontext, stav, príznaky, 7 ukazovateľov |
| `features_by_measurement_finger.csv` | meranie × ruka × prst: priemer platných pokusov bez odľahlých hodnôt |
| `features_by_person_finger.csv` | osoba × step_id × ruka × prst: priemer meraní bez odľahlých hodnôt |
| `dataset_wide.csv` | **osoba (138 riadkov): skupina, vek, aktivita rúk a všetkých 7 ukazovateľov pre každý z 10 prstov v stĺpcoch `<ruka>_<prst>_<ukazovateľ>`**; zdraví iba s krokom `5867…` |
| `person_summary.csv` | osoba × step_id: vek, veková skupina (10 rokov), počty dát, sila rúk, `activity_ratio`, `inactive_hand` |
| `sensor_noise.csv` | ukazovatele tremoru nezaťažených snímačov z kalibračných meraní (úroveň šumu) |
| `processing_summary.csv` | počet záznamov na každej úrovni: platné / neplatné / odľahlé |

Zdraví majú oba kontrolné kroky, preto sú na úrovni osoby zastúpení dvoma riadkami (jeden na `step_id`).

## Štatistická analýza (`scripts/run_analysis.py` → `outputs/stats/`)

Porovnanie skupín: kontrolná skupina – krok `5867…` (4 pokusy ako u pacientov), pacienti – `d2e6…`. Neparametrické testy, Holmova korekcia.
Prsty sa neporovnávajú priamo (líšia sa už u zdravých silou aj kolísaním), ale každý prst s normou toho istého prsta u zdravých ≥ 50 rokov. Tremorové ukazovatele neaktívnych prstov sa nepoužívajú. Navyše sa analyzuje podskupina s vekom ≥ 50 (kontrolná skupina je výrazne mladšia).

| Súbor | Obsah |
|---|---|
| `demographics_by_group`, `age_histogram`, `age_test`, `gender_by_age_group` | vek (intervaly po 10 rokov), pohlavie, Mann-Whitneyho test veku |
| `hand_activity`, `weak_hand_counts` | `force_ratio` (sila) a `fluctuation_ratio` (kolísanie) slabšej/silnejšej ruky; počet osôb pod prahmi 0,10/0,25/0,50 |
| `weak_hand_patients` | **odpoveď na otázku 2:** osoby s pomerom sily nižším ako 97,5 % zdravých (prah z kontrolnej skupiny) |
| `finger_signal_to_noise` | tremor prstov v pomere k šumu nezaťaženého snímača (`outputs/sensor_noise.csv` z kalibrácie) |
| `finger_ratio_to_reference` | koľkokrát je tremor pacientov vyšší ako norma zdravých ≥ 50 rokov pre ten istý prst (95 % IS, bootstrap) |
| `finger_z_scores`, `tremor_pattern` | z-skóre každého prsta voči norme; typ tremoru osoby: bez zvýšenia / lokálny (2–4 prsty) / generalizovaný (≥ 5 prstov) |
| `finger_descriptives` | medián, kvartily pre skupinu × ruku × prst × ukazovateľ |
| `finger_friedman`, `finger_posthoc` | či sa líšia prsty jednej osoby (Friedmanov test + Kendallovo W, párový Wilcoxonov test) |
| `most_tremulous_finger` | u koľkých osôb má ktorý prst najväčší tremor |
| `finger_spread_by_person`, `finger_spread_group_test` | CV a max/min medzi prstami osoby; porovnanie skupín |
| `group_comparison_by_finger` | pacienti vs. zdraví pre každý prst (r > 0 – u pacientov vyššie) |
| `left_right_by_finger` | ľavá vs. pravá ruka pre ten istý prst |
