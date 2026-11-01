"""Spoločné funkcie pre analýzu zložiek sily (hra EXACT_FORCE) na všetkých ľuďoch.

Krok 0: tabuľka vyšetrení (kto je kto + cesta k súboru EXACT_FORCE).
Krok 1: predspracovanie (rovnomerná mriežka 1 kHz -> úsek 3-15 s -> 100 Hz) s cache na disku.
Krok 3: rozdelenie na pomalú zložku a tras, ustálené úseky.
Krok 4: aktivita prstov a rúk.
Krok 5: príznaky pre prst × iteráciu -> vyšetrenie -> človek.
"""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label
from scipy.signal import butter, decimate, sosfiltfilt, welch

from rehapiano_io import FINGER_CHANNELS, load_measurement

# Data folders (9_9_2026, 22_08_2025, ...) and datasets/ are expected next to this file; set REHAPIANO_DATA_ROOT
# to use them from another folder.
ROOT = Path(os.environ.get("REHAPIANO_DATA_ROOT", Path(__file__).resolve().parent))
DATA_FOLDERS = ["9_9_2026", "22_08_2025", "24_4_2025_pacienti", "13_02_2025", "14_02_2025",
                "17_02_2025", "18_02_2025", "19_02_2025", "21_02_2025", "28_02_2025"]
# hlavný krok EXACT_FORCE (4 iterácie po 16 s) v oboch verziách scenára
EXACT_FORCE_STEPS = {
    "5867ebbc-a463-4d20-8b20-50df7c64ace7": "scenár zdravých (2025-02, 2026-09)",
    "d2e6b7bd-a952-4652-afe3-9888e29ac205": "scenár pacientov (2025-03 až 08)",
}
LABEL_FILES = [("dataset_patients.csv", "pacient"), ("dataset_tuke.csv", "zdravý"), ("dataset_99.csv", "zdravý")]

FS_OUT = 100.0          # Hz po decimácii (analýza do 15 Hz)
T0, T1 = 3.0, 15.0      # analyzovaný úsek každej iterácie [s]

_rx_exam = re.compile(rb'"examination_id":\s*"([0-9a-f-]+)"')
_rx_step = re.compile(rb'"step_id":\s*"([0-9a-f-]+)"')


def scan_files(root=ROOT, folders=DATA_FOLDERS):
    """Prečíta z hlavičky každého súboru examination_id a step_id (bez dekódovania signálu)."""
    rows = []
    for folder in folders:
        for path in glob.glob(str(Path(root) / folder / "results_*")):
            with open(path, "rb") as fh:
                head = fh.read(400)
            rows.append(dict(folder=folder, path=path,
                             exam=_rx_exam.search(head).group(1).decode(),
                             step_id=_rx_step.search(head).group(1).decode()))
    return pd.DataFrame(rows)


def load_labels(root=ROOT):
    """Jeden riadok na vyšetrenie EXACT_FORCE z CSV súborov (duplicitné riadky odstránené)."""
    parts = []
    for name, group in LABEL_FILES:
        df = pd.read_csv(Path(root) / "datasets" / name, sep=";", dtype={"patient_id": str})
        df = df[df.step_id.isin(EXACT_FORCE_STEPS)]
        df = df.assign(skupina=group, zdroj=name)
        parts.append(df[["patient_id", "skupina", "zdroj", "age", "gender", "timestamp",
                         "measurement_id", "step_id"]])
    labels = pd.concat(parts).drop_duplicates(subset=["measurement_id", "step_id"])
    return labels.rename(columns={"age": "vek", "gender": "pohlavie", "timestamp": "datum"})


def subgroup(row):
    if row.skupina == "pacient":
        return "pacienti"
    return "zdraví 9_9_2026 (starší)" if row.zdroj == "dataset_99.csv" else "zdraví TUKE 2025-02"


def build_exam_table(root=ROOT):
    """Krok 0: tabuľka vyšetrení s cestou k súboru. Vráti (tabuľka, neoznačené súbory)."""
    files = scan_files(root)
    ef = files[files.step_id.isin(EXACT_FORCE_STEPS)]
    # rovnaký súbor môže byť vo viacerých priečinkoch (24_4_2025_pacienti je podmnožina 22_08_2025)
    ef = ef.sort_values("folder", key=lambda s: s != "22_08_2025").drop_duplicates(["exam", "step_id"])
    labels = load_labels(root)
    exams = labels.merge(ef, left_on=["measurement_id", "step_id"], right_on=["exam", "step_id"], how="left")
    exams["podskupina"] = exams.apply(subgroup, axis=1)
    unlabelled = ef[~ef.exam.isin(labels.measurement_id)]
    cols = ["patient_id", "skupina", "podskupina", "vek", "pohlavie", "datum", "measurement_id",
            "step_id", "folder", "path"]
    return exams[cols].reset_index(drop=True), unlabelled.reset_index(drop=True)


def preprocess_measurement(path, fs_out=FS_OUT, t0=T0, t1=T1):
    """Krok 1: pole tvaru (iterácie, vzorky, 10 prstov) na 100 Hz, úsek t0-t1 s, + počet NaN.

    Decimácia (FIR filter proti aliasingu + každá 10. vzorka) sa robí na celej 16 s iterácii,
    až potom sa vyreže úsek, aby okrajové efekty filtra nezasiahli analyzovaný úsek.
    """
    m = load_measurement(path, fs=1000.0)
    q = int(round(1000.0 / fs_out))
    i0, i1 = int(t0 * fs_out), int(t1 * fs_out)
    its = [decimate(it.x, q, axis=0, ftype="fir", zero_phase=True)[i0:i1] for it in m.iterations]
    return np.stack(its).astype(np.float32), sum(it.n_nan for it in m.iterations)


def load_all(exams, cache=ROOT / "cache" / "exact_force_100hz.npz", force=False):
    """Predspracuje všetky vyšetrenia (s cache). Vráti pole (vyšetrenia, 4, 1200, 10) a počty NaN."""
    cache = Path(cache)
    key = exams.measurement_id.to_numpy().astype(str)
    if cache.exists() and not force:
        z = np.load(cache)
        if list(z["exam"]) == list(key):
            return z["x"], z["n_nan"]
    xs, nans = [], []
    for path in exams.path:
        x, n = preprocess_measurement(path)
        xs.append(x); nans.append(n)
    x, n_nan = np.stack(xs), np.array(nans)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, x=x, n_nan=n_nan, exam=key)
    return x, n_nan


def finger_psd(x, fs=FS_OUT, seg_s=4.0):
    """Welchova PSD pre každý prst. x: (vzorky, 10) -> f, P (frekvencie, 10)."""
    return welch(x, fs=fs, nperseg=int(seg_s * fs), detrend="linear", axis=0)


FINGER_NAMES = [f"k{ch}" for ch in FINGER_CHANNELS]

# --- Krok 3: rozdelenie na zložky a ustálené úseky -------------------------------------------
SLOW_MAX = 3.0                 # Hz, pomalá zložka (ovládanie hry)
TREMOR_BAND = (3.5, 15.0)      # Hz, tras
JUMP_K = 8.0                   # skok = zmena rýchlejšia ako JUMP_K × typická rýchlosť daného prsta
JUMP_FLOOR = 0.01              # a zároveň aspoň 0,01 a.u./s
PAD_S, MIN_STEADY_S = 0.25, 2.0

_SOS_SLOW = butter(4, SLOW_MAX, "low", fs=FS_OUT, output="sos")
_SOS_TREMOR = butter(4, TREMOR_BAND, "band", fs=FS_OUT, output="sos")


def decompose(x, fs=FS_OUT):
    """Rozdelí signál (…, vzorky, …) pozdĺž osi vzoriek -2 na pomalú zložku a tras (Butterworth 4. rádu,
    obojsmerná filtrácia bez fázového posunu). Vráti (pomalá, tras, rýchlosť zmeny pomalej zložky [a.u./s])."""
    slow = sosfiltfilt(_SOS_SLOW, x, axis=-2)
    tremor = sosfiltfilt(_SOS_TREMOR, x, axis=-2)
    rate = np.abs(np.gradient(slow, 1.0 / fs, axis=-2))
    return slow, tremor, rate


def steady_mask(rate, fs=FS_OUT, k=JUMP_K, floor=JUMP_FLOOR, pad_s=PAD_S, min_len_s=MIN_STEADY_S):
    """Maska ustáleného držania sily pre jeden prst a jednu iteráciu (1D pole rýchlosti zmeny).

    Skok = vzorka, kde sa pomalá zložka mení rýchlejšie ako max(floor, k × medián rýchlosti tohto prsta).
    Okolo skokov sa vynechá pad_s (dozvuk filtra) a ponechajú sa len súvislé úseky dlhé aspoň min_len_s.
    Prah je relatívny k človeku: pri pacientovi s trasom v zhlukoch by pevný prah označil celý signál ako skok.
    """
    thr = max(floor, k * float(np.median(rate)))
    pad = int(round(pad_s * fs))
    bad = binary_dilation(rate > thr, structure=np.ones(2 * pad + 1, dtype=bool))
    labels, n = label(~bad)
    out = np.zeros(rate.shape, dtype=bool)
    for i in range(1, n + 1):
        seg = labels == i
        if seg.sum() >= min_len_s * fs:
            out |= seg
    return out


# --- Krok 4: aktivita prstov a rúk ---------------------------------------------------------
# Kanál 9 je zapojený obrátene: jeho pomalá zložka ide proti ostatným prstom ruky (korelácia ≈ −0,7)
# a úroveň je záporná v 95 % záznamov. Pre úroveň sily mu preto otočíme znamienko.
CHANNEL_SIGN = np.array([-1.0 if ch == 9 else 1.0 for ch in FINGER_CHANNELS])
LEFT_IDX, RIGHT_IDX = list(range(0, 5)), list(range(5, 10))
FINGER_ACTIVE_MIN = 0.003      # prst s menšou priemernou silou sa do trasu nezapočíta
HAND_UNUSED_MAX = 0.01         # ruka so súčtom síl pod touto hodnotou = nepoužívaná


def finger_levels(x):
    """Priemerná sila každého prsta (so správnym znamienkom). x: (…, vzorky, 10) -> (…, 10)."""
    return (x * CHANNEL_SIGN).mean(axis=-2)


def hand_levels(levels):
    """Súčet síl prstov ľavej a pravej ruky. levels: (…, 10) -> (…, 2) [ľavá, pravá]."""
    return np.stack([levels[..., LEFT_IDX].sum(-1), levels[..., RIGHT_IDX].sum(-1)], axis=-1)


def hand_status(hand, ratio, weak_level_thr, weak_ratio_thr):
    """Stav slabšej ruky: 'nepoužívaná' / 'málo používaná' / 'v norme'.

    málo používaná = sila slabšej ruky aj pomer rúk sú pod 2,5. percentilom zdravých (prahy z dát)."""
    weak = hand.min(axis=-1)
    status = np.where(weak < HAND_UNUSED_MAX, "nepoužívaná",
                      np.where((weak < weak_level_thr) & (ratio < weak_ratio_thr), "málo používaná", "v norme"))
    return status


# --- Krok 5: príznaky ------------------------------------------------------------------------
STFT_WIN_S, STFT_STEP_S = 2.0, 0.25    # okná na stabilitu trasu v čase
PEAK_HALF_WIDTH = 0.5                  # Hz, okolie vrcholu pre ostrosť vrcholu
EPISODE_PCT = 95                       # okno s trasom = nad 95. percentilom okien zdravých (tras upravený na silu)


def _band(f, lo, hi):
    return (f >= lo) & (f <= hi)


def steady_segments(mask):
    """Zoznam (začiatok, koniec) súvislých úsekov masky."""
    labels, n = label(mask)
    out = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(labels == k)
        out.append((idx[0], idx[-1] + 1))
    return out


def segment_psd(x, segments, fs=FS_OUT, nperseg=200, nfft=800):
    """Welchova PSD z ustálených úsekov (vážený priemer podľa dĺžky úseku)."""
    acc, weight, f = None, 0, None
    for a, b in segments:
        if b - a < nperseg:
            continue
        f, p = welch(x[a:b], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, nfft=nfft, detrend="linear")
        acc = p * (b - a) if acc is None else acc + p * (b - a)
        weight += b - a
    return (f, acc / weight) if weight else (None, None)


def slow_features(hand_force, fs=FS_OUT):
    """Príznaky pomalej zložky pre signál sily jednej ruky (súčet prstov) v jednej iterácii."""
    slow = sosfiltfilt(_SOS_SLOW, hand_force)
    rate = np.abs(np.gradient(slow, 1.0 / fs))
    thr = max(JUMP_FLOOR, JUMP_K * float(np.median(rate)))
    _, n_jumps = label(rate > thr)
    f, p = welch(hand_force, fs=fs, nperseg=int(4 * fs), detrend="linear")
    m = _band(f, 0.25, SLOW_MAX)
    duration_min = len(hand_force) / fs / 60
    return dict(pomaly_uroven=float(hand_force.mean()), pomaly_SD=float(slow.std()),
                pomaly_f_priemer=float(np.sum(f[m] * p[m]) / np.sum(p[m])),
                skoky_za_min=n_jumps / duration_min)


def tremor_features(x, tremor, steady, fs=FS_OUT):
    """Príznaky trasu pre jeden prst v jednej iterácii (len ustálené úseky). Bez ustálených úsekov -> None."""
    segs = [(a, b) for a, b in steady_segments(steady) if b - a >= MIN_STEADY_S * fs]
    if not segs:
        return None
    f, p = segment_psd(x, segs, fs)
    m = _band(f, *TREMOR_BAND)
    fb, pb = f[m], p[m]
    k = int(np.argmax(pb))
    near = np.abs(fb - fb[k]) <= PEAK_HALF_WIDTH
    # okná 2 s v ustálených úsekoch: frekvencia vrcholu a SD trasu v každom okne
    win, step = int(STFT_WIN_S * fs), int(STFT_STEP_S * fs)
    hann, t = np.hanning(win), np.arange(win)
    fw = np.fft.rfftfreq(800, 1 / fs)
    mw = _band(fw, *TREMOR_BAND)
    peak_f, win_sd = [], []
    for a, b in segs:
        for s in range(a, b - win + 1, step):
            seg = x[s:s + win]
            seg = (seg - np.polyval(np.polyfit(t, seg, 1), t)) * hann
            spec = np.abs(np.fft.rfft(seg, 800)) ** 2
            peak_f.append(fw[mw][np.argmax(spec[mw])])
            win_sd.append(tremor[s:s + win].std())
    return dict(tras_SD=float(tremor[steady].std()), tras_f_vrchol=float(fb[k]),
                tras_f_priemer=float(np.sum(fb * pb) / np.sum(pb)),
                tras_ostrost=float(pb[near].sum() / pb.sum()),
                tras_f_SD_cas=float(np.std(peak_f)), _win_sd=np.array(win_sd),
                ustaleny_cas_s=float(steady.sum() / fs))


def tremor_reference(levels, tremor_sd, n_bins=10):
    """Typická SD trasu zdravých podľa sily prsta (mediány v kvantilových skupinách, log-log interpolácia)."""
    lv, ts = np.log10(levels), np.log10(tremor_sd)
    edges = np.quantile(lv, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(lv, edges[1:-1]), 0, n_bins - 1)
    centers = np.array([np.median(lv[idx == b]) for b in range(n_bins)])
    medians = np.array([np.median(ts[idx == b]) for b in range(n_bins)])
    return lambda level: 10 ** np.interp(np.log10(np.maximum(level, 1e-6)), centers, medians)


def compute_finger_features(X, exams):
    """Príznaky trasu pre každý aktívny prst v každej iterácii (+ tras upravený na silu). Vráti (tabuľka, ref)."""
    _, tremor, rate = decompose(X)
    levels = finger_levels(X)
    rows = []
    for e in range(X.shape[0]):
        for i in range(X.shape[1]):
            for c in range(X.shape[3]):
                if levels[e, i, c] <= FINGER_ACTIVE_MIN:
                    continue
                st = steady_mask(rate[e, i, :, c])
                tf = tremor_features(X[e, i, :, c], tremor[e, i, :, c], st)
                if tf is None:
                    continue
                slow_sd = float(sosfiltfilt(_SOS_SLOW, X[e, i, :, c]).std())
                rows.append(dict(vysetrenie=e, iteracia=i + 1, prst=FINGER_NAMES[c],
                                 ruka="ľavá" if c in LEFT_IDX else "pravá", sila_prsta=float(levels[e, i, c]),
                                 tras_k_pomaly=tf["tras_SD"] / slow_sd, **tf))
    fing = pd.DataFrame(rows)
    healthy = exams.skupina.to_numpy()[fing.vysetrenie] == "zdravý"
    ref = tremor_reference(fing.sila_prsta[healthy].to_numpy(), fing.tras_SD[healthy].to_numpy())
    expected = ref(fing.sila_prsta.to_numpy())
    fing["tras_relativny"] = fing.tras_SD / expected
    # okno s trasom = upravený tras okna nad 95. percentilom okien všetkých prstov zdravých
    win_rel = [w / ex for w, ex in zip(fing._win_sd, expected)]
    episode_thr = float(np.percentile(np.concatenate([w for w, h in zip(win_rel, healthy) if h]), EPISODE_PCT))
    fing["tras_cas_podiel"] = [float(np.mean(w > episode_thr)) for w in win_rel]
    fing.attrs["episode_thr"] = episode_thr
    return fing.drop(columns="_win_sd"), ref


TREMOR_COLS = ["tras_SD", "tras_relativny", "tras_relativny_median", "tras_f_vrchol", "tras_f_priemer",
               "tras_ostrost", "tras_k_pomaly", "tras_f_SD_cas", "tras_cas_podiel"]
SLOW_COLS = ["pomaly_uroven", "pomaly_SD", "pomaly_f_priemer", "skoky_za_min", "podiel_ustaleny"]


def compute_iteration_features(X, fing):
    """Jeden riadok na vyšetrenie x iteráciu.

    Tras: v každej iterácii prst s najväčším trasom upraveným na silu (teda aj ruka s väčším trasom).
    Pomalá zložka: zo súčtu síl ruky, priemer oboch rúk.
    """
    xs = X * CHANNEL_SIGN
    hand_force = np.stack([xs[..., LEFT_IDX].sum(-1), xs[..., RIGHT_IDX].sum(-1)], -1)
    _, _, rate = decompose(X)
    rows = []
    for e in range(X.shape[0]):
        for i in range(X.shape[1]):
            sf = [slow_features(hand_force[e, i, :, h]) for h in range(2)]
            row = dict(vysetrenie=e, iteracia=i + 1, **{k: float(np.mean([s[k] for s in sf])) for k in sf[0]})
            act = finger_levels(X[e, i]) > FINGER_ACTIVE_MIN
            row["podiel_ustaleny"] = (float(np.mean([steady_mask(rate[e, i, :, c]).mean() for c in np.flatnonzero(act)]))
                                      if act.any() else np.nan)
            f = fing[(fing.vysetrenie == e) & (fing.iteracia == i + 1)]
            if len(f):
                best = f.loc[f.tras_relativny.idxmax()]
                row.update({k: best[k] for k in TREMOR_COLS if k in best})
                row["tras_relativny_median"] = float(f.tras_relativny.median())   # všetky aktívne prsty
                row["ruka_trasu"], row["prst_trasu"] = best.ruka, best.prst
            rows.append(row)
    return pd.DataFrame(rows)


def _mode(s):
    return s.mode().iat[0] if s.notna().any() else np.nan


def compute_person_features(exams, iters):
    """Iterácie -> vyšetrenie (priemer + SD cez iterácie) -> človek (priemer cez vyšetrenia).

    exams musí obsahovať stĺpce sila_lava, sila_prava, pomer_ruk, stav_slabsej_ruky (krok 4)."""
    main = TREMOR_COLS + SLOW_COLS
    sd_cols = ["tras_SD", "tras_relativny", "tras_f_vrchol", "pomaly_SD"]
    agg = iters.groupby("vysetrenie").agg(
        **{c: (c, "mean") for c in main}, **{f"{c}_SD_iter": (c, "std") for c in sd_cols},
        ruka_trasu=("ruka_trasu", _mode), prst_trasu=("prst_trasu", _mode))
    ex = exams.drop(columns=[c for c in agg.columns if c in exams.columns]).join(agg)
    ex["sila_slabsej_ruky"] = ex[["sila_lava", "sila_prava"]].min(axis=1)
    ex["obdobie"] = np.where(ex.podskupina == "zdraví TUKE 2025-02", "2025-02", "2025-03 a neskôr")
    num = main + [f"{c}_SD_iter" for c in sd_cols] + ["sila_lava", "sila_prava", "pomer_ruk", "sila_slabsej_ruky"]
    order = {"v norme": 0, "málo používaná": 1, "nepoužívaná": 2}
    person = ex.groupby("patient_id").agg(
        skupina=("skupina", "first"), podskupina=("podskupina", "first"), obdobie=("obdobie", "first"),
        vek=("vek", "first"), pohlavie=("pohlavie", "first"), pocet_vysetreni=("measurement_id", "size"),
        **{c: (c, "mean") for c in num},
        ruka_trasu=("ruka_trasu", _mode), prst_trasu=("prst_trasu", _mode),
        stav_slabsej_ruky=("stav_slabsej_ruky", lambda s: max(s, key=order.get)))
    return ex, person.reset_index()
