# Rytmus – 4. týždeň (M. Kuznietsov): rovnováha síl rúk – rytmický test vs. EXACT_FORCE, vek a ochorenie

Zadanie: *Vyhodnotiť rovnováhu síl rúk aj pre rytmický test, porovnať s exact force, analýzu robiť aj po rozdelení na
vekové kategórie, odlíšiť vplyv veku a ochorenia.*

| Súbor | Obsah |
|---|---|
| `rovnovaha_ruk.ipynb` | analýza, grafy, závery (po slovensky) |
| `figures/rovnovaha_ruk/` | 6 grafov (`01_rytmicky_test_priklady`, `02_tempo_stlacani` … `06_vek_a_rovnovaha`) |

## Spustenie (z koreňa repozitára)

```bash
python scripts/build_rhythm.py                # RHYTHM: rovnováha rúk pre každý pokus → outputs/rhythm/rhythm_by_attempt.csv
python scripts/run_rhythm_hand_balance.py     # štatistika → outputs/rhythm/hand_balance/*.csv (potrebuje outputs/exact_force/stats/hand_activity.csv)
jupyter lab notebooks/rhythm/rovnovaha_ruk.ipynb
```

Kód: `src/rehapiano/rhythm/presses.py` (hľadanie stlačení, amplitúda stlačenia ruky), `src/rehapiano/rhythm/hand_balance.py`
(porovnanie testov, vekové kategórie, model vek + ochorenie), testy `tests/test_rhythm.py`.

## Metóda v skratke

- **Rovnováha** = slabšia / silnejšia ruka (1 = rovnako, 0 = jedna ruka netlačí); prah slabo používanej ruky = 2,5 % kvantil zdravých.
- **RHYTHM:** sila ruky = priemer 5 snímačov; stlačenia = vrcholy súčtu síl oboch rúk; amplitúda ruky = medián (max − min) cez
  stlačenia. Časy `press_offset` / `release_offset` z hry nie sú zarovnané so signálom, preto sa nepoužívajú.
- **EXACT_FORCE:** `force_ratio` z 3. týždňa (`outputs/exact_force/stats/hand_activity.csv`).
- **Vekové kategórie:** 10-ročné skupiny (10–19 … 80–89); testujú sa skupiny, kde majú zdraví aj pacienti ≥ 3 osoby,
  a všetci ≥ 50 rokov spolu (zdravých ≥ 50 je len 24).
- **Vek vs. ochorenie:** vek u samotných zdravých; pacienti vs. zdraví v tej istej kategórii; lineárny model
  rovnováha = b₀ + b_vek · vek/10 + b_ochorenie · pacient (HC3).

## Hlavné výsledky

- RHYTHM: slabo používaná ruka u **18 zo 48 pacientov** (EXACT_FORCE: 6), u zdravých 3 z 90.
- Testy sa zhodujú len čiastočne (pacienti ρ = 0,40); slabá ruka je takmer vždy tá istá (88 %). Rytmický test je citlivejší a opakovateľný (ρ = 0,76).
- Vek u zdravých rovnováhu nemení; pri rovnakom veku majú pacienti v rytme o 0,20 nižšiu rovnováhu (p < 0,001), v EXACT_FORCE o 0,06.
- Tempo stláčania: zdraví ~2,5, pacienti ~1,5 stlačenia za sekundu; u zdravých nezávisí od veku.
- Obmedzenia: iná verzia hry (`step_id`) a pomalšie tempo u pacientov, len 24 zdravých ≥ 50 rokov.
