# Rytmus – 5. týždeň (M. Kuznietsov): štatistická analýza rytmického testu – hypotézy, normy podľa veku, klasifikátor, ablácia

Zadanie: *Pripraviť štatistickú analýzu – stanoviť hypotézy a urobiť testy štatistickej významnosti rozdielu medzi zdravými
a chorými. Urobiť analýzu zmien parametrov u zdravých ľudí s vekom, stanoviť „normálne" hodnoty a rozsahy podľa dekád.
Navrhnúť klasifikátor na oddelenie zdravých a chorých a ablačnú analýzu vplyvu jednotlivých faktorov na výsledok klasifikácie.*

Použitá je **iba rytmická hra (RHYTHM)**, pokusy 2–4 (1. pokus je skúšobný).

| Súbor | Obsah |
|---|---|
| `statisticka_analyza_rytmus.ipynb` | analýza, grafy, závery (po slovensky) |
| `figures/statisticka_analyza_rytmus/` | 5 grafov (`01_parametre_zdravi_vs_pacienti` … `05_ablacia`) |

## Spustenie (z koreňa repozitára)

```bash
pip install -r requirements.txt          # vrátane scikit-learn (klasifikátor)
python scripts/build_rhythm.py                 # parametre každého pokusu → outputs/rhythm/rhythm_parameters_by_attempt.csv
python scripts/run_rhythm_analysis.py          # testy, normy, klasifikátor, ablácia → outputs/rhythm/analysis/*.csv (~5 min)
jupyter lab notebooks/rhythm/statisticka_analyza_rytmus.ipynb
```

Kód: `src/rehapiano/rhythm/presses.py` (`attempt_rhythm_parameters`), `src/rehapiano/rhythm/parameters.py`,
`src/rehapiano/common/stats/group_analysis.py`, testy `tests/test_rhythm.py`. Zdraví bez ťažkostí podľa dotazníka:
`data/participant_labels.csv` (z vetvy `mykyta`, M. Holovcyk).

## Parametre (8 faktorov)

tempo · pravidelnosť rytmu · sila stlačenia (silnejšia / slabšia ruka) · rovnováha rúk · stálosť sily · súčasnosť rúk ·
uvoľnenie · únava (sila a tempo v 2. vs. 1. polovici pokusu).

## Hlavné výsledky (76 zdravých bez ťažkostí, 48 pacientov)

- H0 zamietnutá (Mann-Whitney, Holm) pre 5 z 10 parametrov; najstabilnejšie: **rovnováha rúk** (aj pri rovnakom veku a tempe) a **tempo**.
- U zdravých sa po Holmovej korekcii s vekom nemení žiadny parameter → norma (2,5.–97,5. percentil) je rovnaká pre všetky dekády;
  empiricky ju možno určiť len pre 20–29 rokov (ostatné dekády 1–8 osôb).
- Logistická regresia (5-násobná CV, 20×): AUC 0,85 z parametrov rytmu, 0,90 s vekom; samotný vek 0,84; len ≥ 50 rokov 0,80.
- Ablácia: najviac prispievajú tempo, rovnováha rúk a sila stlačenia; parametre sa navzájom zastupujú; vek je silný mätúci faktor.
- Obmedzenia: iná verzia hry (`step_id`) u pacientov, len 17 zdravých ≥ 50 rokov, žiadne nezávislé overenie.
