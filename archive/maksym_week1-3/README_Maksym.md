> **Archív.** Pôvodný README priečinka `Maksym/` (stav pred zlúčením 1. 11. 2026). Týždne 4–5 sú teraz súčasťou
> projektu: kód v `src/rehapiano/rhythm/`, skripty v `scripts/`, notebooky v `notebooks/rhythm/`, tabuľky v
> `outputs/rhythm/`. Tu zostávajú len týždne 1–3 (vlastný kód, spustené v pôvodnom priečinku s dátami).

# Maksym Kuznietsov – RehaPiano

Moja časť práce na projekte RehaPiano (úlohy 1–5). Týždne 1–3 sú môj vlastný pipeline (hra EXACT_FORCE),
týždne 4–5 stavajú na spoločnom balíku `src/rehapiano` (D. Sholtys) a analyzujú rytmický test (RHYTHM).

| Priečinok | Úloha | Hlavný súbor |
|---|---|---|
| `week1-3/` | 1. Fourierova analýza jedného pacienta a jedného zdravého (okná 3/5/10 s, spektrogramy) | `fourier_exact_force.ipynb` |
| | 2. oddelenie pomalej zložky sily od trasu (≤ 15 Hz), príznaky pre každého človeka (`features.csv`) | `features_tremor.ipynb` |
| | 3. vek, slabo používaná ruka, tras jednotlivých prstov (`tras_prstov.csv`, `uloha3_ludia.csv`) | `tremor_prsty.ipynb` |
| `week4/` | 4. rovnováha síl rúk v rytmickom teste vs. EXACT_FORCE, vekové kategórie, vplyv veku a ochorenia | `rovnovaha_ruk.ipynb` |
| `week5/` | 5. hypotézy a testy, normy podľa dekád, klasifikátor zdravý/pacient, ablačná analýza (len rytmický test) | `statisticka_analyza_rytmus.ipynb` |
| `src/maksym_rehapiano/` | kód týždňov 4–5: `rhythm.py` (stlačenia, parametre pokusu), `hand_balance.py`, `rhythm_parameters.py`, `config.py` | |
| `scripts/` | `build_rhythm.py`, `run_rhythm_analysis.py` (4. týždeň), `run_rhythm_week5.py` (5. týždeň) | |
| `tests/` | unit testy kódu týždňov 4–5 | |
| `data/participant_labels.csv` | zdraví bez ťažkostí podľa dotazníka – kópia z vetvy `mykyta` (M. Holovcyk) | |

Notebooky sú uložené aj s výsledkami a grafmi, takže sa dajú prezerať bez dát.

## Spustenie týždňov 4–5 (z koreňa repozitára)

Dáta sa ukladajú podľa `data/README.md` (`data/raw/RehaPiano/…`, `data/metadata/…`).

```bash
pip install -r requirements.txt scikit-learn
python scripts/build_dataset.py && python scripts/run_analysis.py   # spoločný pipeline (EXACT_FORCE, 3. týždeň)
python Maksym/scripts/build_rhythm.py                                # RHYTHM: hodnoty pre každý pokus
python Maksym/scripts/run_rhythm_analysis.py                         # 4. týždeň → Maksym/week4/outputs/
python Maksym/scripts/run_rhythm_week5.py                            # 5. týždeň → Maksym/week5/outputs/ (~40 min)
python -m unittest discover Maksym/tests
```

Zmena v spoločnom kóde: `src/rehapiano/io/metadata.py` – `build_manifest` a `filter_target_steps` majú nový voliteľný
parameter `step_type` (predvolene EXACT_FORCE, správanie pre EXACT_FORCE sa nemení) a cesty k súborom v prepojených
priečinkoch s dátami zostávajú relatívne.

## Týždne 1–3

Vlastné moduly `rehapiano_io.py` (dekódovanie súborov; časová značka je `int64`), `rehapiano_features.py`
(predspracovanie, rozdelenie zložiek, príznaky) a `rehapiano_fingers.py` (úloha 3). Očakávajú priečinky s dátami
(`9_9_2026`, `22_08_2025`, …, `datasets/`) vedľa seba, alebo cestu v premennej prostredia `REHAPIANO_DATA_ROOT`.
Notebooky boli spustené v pôvodnom priečinku s dátami; grafy sú v `week1-3/figures/` (01–04 úloha 1, 10–17 úloha 2,
18–24 úloha 3).
