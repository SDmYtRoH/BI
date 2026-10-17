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
