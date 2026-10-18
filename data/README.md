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
    ├── dataset_full.csv       (povinný)
    └── dataset_patients.csv   (povinný)
```

Súbory sa hľadajú rekurzívne, takže štruktúra podpriečinkov v `RehaPiano/` môže byť ľubovoľná.
Cesty sú nastavené v `src/rehapiano/config.py`.

## Local layout on the `mykyta` branch (English note)

The archive extracts to `data/RehaPiano/` with the metadata indices in `data/RehaPiano/datasets/`.
The week-3 pipeline (`src/rehapiano/config.py`) expects `data/raw/RehaPiano/` and `data/metadata/`, so two local symlinks bridge the layouts:

```sh
mkdir -p data/raw
ln -s ../RehaPiano data/raw/RehaPiano
ln -s RehaPiano/datasets data/metadata
```

`data/sources/` holds read-only snapshots used by week 2: `questionnaire.csv` (control-group questionnaire with self-reported health answers – never commit it), `protocol.txt`, `document.txt`.
