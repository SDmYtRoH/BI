# Dáta (nie sú súčasťou repozitára)

Surové merania a metadáta obsahujú údaje pacientov a majú ~6,6 GB, preto sú vylúčené cez `.gitignore`.
Na spustenie pipeline ich ulož takto:

```
data/
├── raw/RehaPiano/          priečinky so súbormi results_<uuid>_<YYYYMMDD-HHMMSS>
│   ├── 13_02_2025/ ...
│   ├── 22_08_2025/ ...
│   └── 25_03_2025_calibration/   (kalibrácia; potvrdzuje opačné znamienko kanála 9)
└── metadata/
    ├── dataset_full.csv              (povinný)
    ├── dataset_patients.csv          (povinný)
    ├── dataset_09_09_2026_en.csv     (povinný; opravuje dataset_full.csv pre meranie 9. 9. 2026)
    ├── dataset_13_11_2026_en.csv     (povinný; zdraví z meracieho centra 9.–13. 11. 2026)
    └── dotazni1311.csv               (dotazník k meraniu 13. 11. 2026, len na kontrolu)
```

Riadky súborov `dataset_*_en.csv` nahrádzajú riadky `dataset_full.csv` s rovnakým kľúčom (`measurement_id`, `step_id`).
V `dataset_full.csv` sú osoby z 9. 9. 2026 od štvrtej posunuté o jednu (iné ID a vek), preto je oprava povinná.
Surové súbory z 13. 11. 2026 sú v `raw/RehaPiano/13_11_2026_meranie_centrum/meracka/` (len 308 súborov, ku ktorým
existujú metadáta; 3 merania bez metadát a merania `06_11_2026_nemocnica`, `23_10_2026_nemocnica_pacienti` zatiaľ
nie sú pridané).

Súbory sa hľadajú rekurzívne, takže štruktúra podpriečinkov v `RehaPiano/` môže byť ľubovoľná.
Cesty sú nastavené v `src/rehapiano/common/config.py`. Súbor `data/participant_labels.csv` (zdraví bez ťažkostí podľa
dotazníka) je v gite; pri nových zdravých osobách ho treba doplniť.
