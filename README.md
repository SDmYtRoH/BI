# RehaPiano: analýza testov EXACT_FORCE a rytmus (RHYTHM)

Spoločný projekt (D. Sholtys, M. Kuznietsov) pre dve hry zariadenia RehaPiano:

| Test | Čo osoba robí | Analýza |
|---|---|---|
| **EXACT_FORCE** | drží všetkými prstami rovnomernú silu, 4 pokusy × 16 s | tremor každého prsta, slabo používaná ruka (3. týždeň); hypotézy, normy, klasifikátor (D. Sholtys) |
| **RHYTHM** | čo najrýchlejšie rytmicky stláča a uvoľňuje oboma rukami, 4 pokusy × 31 s | rovnováha rúk vs. EXACT_FORCE (4. týždeň); hypotézy, normy, klasifikátor (M. Kuznietsov) |
| **oba testy** | – | spoločný klasifikátor zdravý / pacient a ablácia testov a faktorov |

Úloha z 5. týždňa – *hypotézy a testy významnosti zdraví vs. chorí, zmeny parametrov s vekom a normy podľa dekád,
klasifikátor a ablačná analýza* – je spracovaná rovnakým kódom (`src/rehapiano/common/stats/group_analysis.py`) pre oba
testy aj pre ich spojenie.

## Inštalácia

```bash
git clone https://github.com/SDmYtRoH/BI.git && cd BI
conda create -n rehapiano python=3.12 && conda activate rehapiano
pip install -r requirements.txt
```

Dáta nie sú súčasťou repozitára – ako ich uložiť, je opísané v [data/README.md](data/README.md).
Hotové výsledky sú už v `outputs/` (CSV) a v notebookoch, takže ich možno prezerať aj bez dát.

## Spustenie (aj po aktualizácii dát)

```bash
conda activate rehapiano
python scripts/run_all.py                  # všetko: surové dáta → tabuľky → štatistika (~10 min)
python scripts/run_all.py --notebooks      # ... a znovu spustí notebooky (grafy, tabuľky, texty záverov)
python scripts/run_all.py --quick          # rýchla kontrola (menej opakovaní krížovej validácie)
python scripts/run_all.py --only rhythm    # len jeden test (exact_force / rhythm / combined)
python scripts/run_all.py --skip-build     # len štatistika, surové dáta sa znovu nečítajú
python -m unittest discover tests
```

Jednotlivé kroky (v tomto poradí, `run_all.py` ich spúšťa za sebou):

| Skript | Vstup → výstup |
|---|---|
| `scripts/build_exact_force.py` | surové dáta → `outputs/exact_force/*.csv` (ukazovatele prstov) |
| `scripts/run_exact_force_stats.py` | → `outputs/exact_force/stats/` (3. týždeň: vek, slabá ruka, tremor prstov) |
| `scripts/run_exact_force_analysis.py` | → `outputs/exact_force/analysis/` (hypotézy, normy, klasifikátor, ablácia) |
| `scripts/build_rhythm.py` | surové dáta → `outputs/rhythm/*.csv` (stlačenia, parametre pokusov) |
| `scripts/run_rhythm_hand_balance.py` | → `outputs/rhythm/hand_balance/` (4. týždeň: rovnováha rúk vs. EXACT_FORCE) |
| `scripts/run_rhythm_analysis.py` | → `outputs/rhythm/analysis/` (hypotézy, normy, klasifikátor, ablácia) |
| `scripts/run_combined_analysis.py` | → `outputs/combined/` (klasifikátor z oboch testov) |

**Po aktualizácii dát:** nové súbory `results_*` a riadky metadát stačí pridať do `data/` a spustiť
`python scripts/run_all.py --notebooks`. Noví zdraví vstúpia do hlavnej analýzy, až keď sú v `data/participant_labels.csv`
(dotazník – bez ťažkostí s rukami); chýbajúcich vypíše skript ako varovanie. Čísla v tabuľkách, grafoch a vo väčšine
záverov notebookov sa prepočítajú automaticky; slovný komentár medzi nimi opisuje dáta z 1. 11. 2026.

## Štruktúra

```
data/raw/RehaPiano/          surové súbory results_* (všetky hry; nie sú v gite)
data/metadata/               dataset_full.csv, dataset_patients.csv (nie sú v gite)
data/participant_labels.csv  zdraví bez ťažkostí podľa dotazníka (2. týždeň, M. Holovcyk)
src/rehapiano/
  common/                    SPOLOČNÉ pre všetky testy
    config.py                cesty, snímače, predspracovanie, vekové dekády, nastavenia štatistiky
    io/metadata.py           čítanie CSV, priradenie súborov k metadátam (step_type, step_id), inventár
    io/reader.py             dekódovanie JSON → base64 → zlib → pickle → pakety
    preprocessing/           znamienko kanála 9, medzery, 1000 → 100 Hz, orezanie (okno určuje test)
    stats/tests.py           Mann-Whitney, Wilcoxon, Friedman, Holm
    stats/models.py          lineárny model s robustnými chybami (HC3)
    stats/demographics.py    vek a pohlavie skupín
    stats/group_analysis.py  hypotézy, trendy s vekom, normy podľa dekád, klasifikátor, ablácia
  exact_force/               test EXACT_FORCE
    config.py                step_id, orezanie 3–15 s, pásma spektra, prahy kvality
    quality.py, aggregation.py, features/   platnosť, pokus → meranie → osoba, ukazovatele prsta
    stats/                   3. týždeň: slabá ruka, tremor prstov
    parameters.py            10 parametrov osoby a 8 faktorov pre úlohu 5. týždňa
  rhythm/                    test RHYTHM (M. Kuznietsov)
    config.py                step_id, orezanie 2–30 s, detekcia stlačení, 1. pokus = skúšobný
    presses.py               stlačenia, rovnováha rúk a 10 parametrov každého pokusu
    hand_balance.py          4. týždeň: rovnováha rúk vs. EXACT_FORCE, vek vs. ochorenie
    parameters.py            parametre osoby a 8 faktorov
  combined.py                spojenie parametrov oboch testov
scripts/                     skripty vyššie + run_all.py
notebooks/
  exact_force/analyza_prstov.ipynb                 3. týždeň: vek, slabá ruka, tremor prstov
  exact_force/statisticka_analyza_exact_force.ipynb   hypotézy, normy, klasifikátor, ablácia (EXACT_FORCE)
  exact_force/task_1.ipynb                         1. úloha (samostatný kód)
  rhythm/rovnovaha_ruk.ipynb                       4. týždeň: rovnováha rúk (M. Kuznietsov)
  rhythm/statisticka_analyza_rytmus.ipynb          hypotézy, normy, klasifikátor, ablácia (rytmus, M. Kuznietsov)
  combined/klasifikator_oba_testy.ipynb            klasifikátor z oboch testov
  */figures/<notebook>/                            grafy uložené notebookmi
outputs/                     výsledky (generujú ich skripty)
tests/                       unit testy
archive/maksym_week1-3/      pôvodná práca M. Kuznietsova na 1.–3. týždni (vlastný kód, len na prezeranie)
```

## EXACT_FORCE – dáta a pipeline

- **Cvičenie EXACT_FORCE:** osoba stláča všetky prsty a udržiava silu na cieľovej úrovni. 4 pokusy po 16 s (krok `d475…` – 1 pokus).
- **`step_id` sa zhoduje so skupinou:** zdraví – `5867ebbc-…` a `d475bc4a-…`, pacienti – iba `d2e6b7bd-…`. Rozdiely medzi skupinami môžu čiastočne vyplývať z nastavenia hry.
- **Priradenie:** kľúč `(measurement_id, step_id)`; UUID z názvu súboru = `payload.examination_id` = `measurement_id`. `is_patient = 1`, ak je kľúč v `dataset_patients.csv`.
- **Snímače:** kanály `[0–4]` – ľavá ruka, `[8–12]` – pravá; 5–7 a 13–15 sú vždy 0. Kanál 9 má opačné znamienko (potvrdené kalibráciou v `25_03_2025_calibration`) a opravuje sa. Bez zaťaženia ukazujú snímače 0 ± 0,002, šum ~1e-4.
- **Prsty:** `sensor_1…5` = malíček…palec – **neoverené**, mení sa v `common/config.py: FINGER_NAMES`.

## Predspracovanie (overené na surových dátach)

Frekvencia paketov 1000–1010/s. ~25 % časových značiek je duplicitných → priemerujú sa. NaN v 0,1–0,2 % riadkov (najdlhšia séria 12 ms) → lineárna interpolácia. Kanál s > 1 % NaN alebo s medzerou > 50 ms zostáva NaN. Medzery > 50 ms rozdelia pokus na segmenty; analyzuje sa iba segment 1. Po orezaní na 3–15 s zostáva 1200 vzoriek pri 100 Hz.

## Ukazovatele prsta (`exact_force/features/finger.py`)

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
| `movement_mean_amplitude` | priemerná amplitúda v pásme 0–4 Hz (pomalé korekcie sily – ako kolíše držaná sila) |

## Platnosť a odľahlé hodnoty (`exact_force/quality.py`)

| Úroveň | Kritérium | Dôsledok |
|---|---|---|
| súbor | nedá sa dekódovať, UUID nesedí, chýbajú metadáta, duplicita | `recordings_inventory.csv`, `status = excluded` |
| pokus | čas ide dozadu, nemožná frekvencia paketov, chýba segment ≥ 3 s | súbor/pokus sa nespracuje |
| pokus × prst | < 500 vzoriek, NaN, `std < 1e-5` (`flat_signal`) | `status = invalid` |
| pokus × prst | `mean_force < 0,002` | `finger_inactive = True`, riadok zostáva |
| pokus × ruka | priemer cez prsty `mean_force` alebo `tremor_mean_amplitude` > 3× alebo < 1/3 mediánu pokusov tohto merania (≥ 3 pokusy) | `attempt_outlier`, vylúčené z priemerov |
| meranie × ruka | to isté voči ostatným meraniam osoby (≥ 3 merania) | `measurement_outlier`, vylúčené z priemerov |

**Neaktívna ruka:** `activity_ratio` = priemerná sila prstov slabšej ruky / silnejšej; `< 0,10` → `inactive_hand = left/right`, inak `none`; `unknown` – chýbajú dáta alebo netlačí ani jedna ruka.

## Výstupné súbory EXACT_FORCE (`outputs/exact_force/`)

| Súbor | Riadok |
|---|---|
| `recordings_inventory.csv` | každý súbor `results_*`: matched / excluded + dôvod |
| `features_by_attempt_finger.csv` | pokus × ruka × prst: kontext, stav, príznaky, 8 ukazovateľov |
| `features_by_measurement_finger.csv` | meranie × ruka × prst: priemer platných pokusov bez odľahlých hodnôt |
| `features_by_person_finger.csv` | osoba × step_id × ruka × prst: priemer meraní bez odľahlých hodnôt |
| `dataset_wide.csv` | **osoba (138 riadkov): skupina, vek, aktivita rúk a všetkých 8 ukazovateľov pre každý z 10 prstov v stĺpcoch `<ruka>_<prst>_<ukazovateľ>`**; zdraví iba s krokom `5867…` |
| `person_summary.csv` | osoba × step_id: vek, veková skupina (10 rokov), počty dát, sila rúk, `activity_ratio`, `inactive_hand` |
| `sensor_noise.csv` | ukazovatele tremoru nezaťažených snímačov z kalibračných meraní (úroveň šumu) |
| `processing_summary.csv` | počet záznamov na každej úrovni: platné / neplatné / odľahlé |

Zdraví majú oba kontrolné kroky, preto sú na úrovni osoby zastúpení dvoma riadkami (jeden na `step_id`).

## Štatistika 3. týždňa (`scripts/run_exact_force_stats.py` → `outputs/exact_force/stats/`)

Porovnanie skupín: kontrolná skupina – krok `5867…` (4 pokusy ako u pacientov), pacienti – `d2e6…`. Neparametrické testy, Holmova korekcia.
Prsty sa neporovnávajú priamo (líšia sa už u zdravých silou aj kolísaním), ale každý prst s normou toho istého prsta u zdravých ≥ 50 rokov. Tremorové ukazovatele neaktívnych prstov sa nepoužívajú. Navyše sa analyzuje podskupina s vekom ≥ 50 (kontrolná skupina je výrazne mladšia).

| Súbor | Obsah |
|---|---|
| `demographics_by_group`, `age_histogram`, `age_test`, `gender_by_age_group` | vek (intervaly po 10 rokov), pohlavie, Mann-Whitneyho test veku |
| `hand_activity`, `weak_hand_counts` | `force_ratio` (sila) a `fluctuation_ratio` (kolísanie) slabšej/silnejšej ruky; počet osôb pod prahmi 0,10/0,25/0,50 |
| `weak_hand_patients` | **odpoveď na otázku 2:** osoby s pomerom sily nižším ako 97,5 % zdravých (prah z kontrolnej skupiny) |
| `finger_signal_to_noise` | tremor prstov v pomere k šumu nezaťaženého snímača (`outputs/exact_force/sensor_noise.csv` z kalibrácie) |
| `finger_ratio_to_reference` | koľkokrát je tremor pacientov vyšší ako norma zdravých ≥ 50 rokov pre ten istý prst (95 % IS, bootstrap) |
| `finger_z_scores`, `tremor_pattern` | z-skóre každého prsta voči norme; typ tremoru osoby: bez zvýšenia / lokálny (2–4 prsty) / generalizovaný (≥ 5 prstov) |
| `finger_descriptives` | medián, kvartily pre skupinu × ruku × prst × ukazovateľ |
| `finger_friedman`, `finger_posthoc` | či sa líšia prsty jednej osoby (Friedmanov test + Kendallovo W, párový Wilcoxonov test) |
| `most_tremulous_finger` | u koľkých osôb má ktorý prst najväčší tremor |
| `finger_spread_by_person`, `finger_spread_group_test` | CV a max/min medzi prstami osoby; porovnanie skupín |
| `group_comparison_by_finger` | pacienti vs. zdraví pre každý prst (r > 0 – u pacientov vyššie) |
| `left_right_by_finger` | ľavá vs. pravá ruka pre ten istý prst |

## Úloha 5. týždňa: hypotézy, normy, klasifikátor, ablácia

Rovnaká metóda pre oba testy (`common/stats/group_analysis.py`), výsledky v `outputs/<test>/analysis/` a `outputs/combined/`:

1. **Hypotézy:** H0 = rovnaké rozdelenie parametra u zdravých a pacientov; Mann-Whitney (všetci a len ≥ 50 rokov), AUC,
   lineárny model parameter ~ vek + ochorenie (HC3) a ten istý model s kovariátom (EXACT_FORCE: sila, rytmus: tempo);
   Holmova korekcia. Zdraví = bez ťažkostí podľa dotazníka (kontrola so všetkými zdravými).
2. **Vek u zdravých:** Spearman a zmena za 10 rokov; **normy podľa dekád** – 2,5. / 50. / 97,5. percentil všetkých zdravých
   (pre parameter závislý od veku kvantilová regresia na vek), empiricky po dekádach len pri ≥ 20 osobách.
3. **Klasifikátor:** logistická regresia (hlavný) a náhodný les, 5-násobná krížová validácia × 20; porovnanie s klasifikátorom
   len podľa veku a len na osobách ≥ 50 rokov; permutačný test.
4. **Ablácia:** pre každý faktor AUC bez neho a AUC iba z neho (aj s vekom ako ďalším faktorom); v spojenom modeli aj
   celé testy.

| Tabuľka | Obsah |
|---|---|
| `people` | parametre každej osoby (`screened` = v hlavnej analýze) |
| `hypotheses`, `hypotheses_all_healthy` (, `hypotheses_with_first_attempt`) | testy hypotéz; kontroly |
| `age_trends`, `decade_norms`, `norms_by_age_model`, `outside_norms` | vek u zdravých, normy podľa dekád, pacienti mimo normy |
| `classifier`, `classifier_oof`, `classifier_coefficients`, `permutation_test` | klasifikátor (`main` = hlavná množina) |
| `ablation_parameters`, `ablation_with_age` (combined: `ablation_exercises`, `ablation_factors*`) | ablácia |

## RHYTHM (`outputs/rhythm/`, M. Kuznietsov)

Podrobný opis metódy a výsledkov: [notebooks/rhythm/README_rovnovaha_ruk.md](notebooks/rhythm/README_rovnovaha_ruk.md)
a [notebooks/rhythm/README_statisticka_analyza.md](notebooks/rhythm/README_statisticka_analyza.md).
Stlačenia sa hľadajú priamo v signáli (vrcholy súčtu síl oboch rúk); časy z hry (`press_offset`) nie sú zarovnané so signálom.
