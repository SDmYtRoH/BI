"""Úloha 3: vek, slabo používaná ruka a tras jednotlivých prstov (hra EXACT_FORCE).

Stavia na rehapiano_features (cache 100 Hz, ustálené úseky, tras upravený na silu).
Tras prsta = `tras_relativny`: SD trasu (3,5–15 Hz, len ustálené úseky) / typická SD zdravých pri rovnakej
sile prsta. 1 = ako zdraví, 3 = trikrát viac.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu

import rehapiano_features as rf

ROOT = rf.ROOT  # data root, see rehapiano_features.ROOT
CALIBRATION_DIR = ROOT / "25_03_2025_calibration"

# Poradie prstov podľa sily: kanály 0/8 tlačia najslabšie, 4/12 najsilnejšie v oboch rukách (zdraví aj pacienti),
# preto pravdepodobne 0/8 = malíček ... 4/12 = palec. Overené len zo síl, nie z dokumentácie zariadenia.
FINGER_ORDER = ["malíček", "prstenník", "prostredník", "ukazovák", "palec"]
FINGER_LABEL = {name: f"{'ľ' if i < 5 else 'p'}. {FINGER_ORDER[i % 5]}" for i, name in enumerate(rf.FINGER_NAMES)}
FINGER_HAND = {name: ("ľavá" if i < 5 else "pravá") for i, name in enumerate(rf.FINGER_NAMES)}

AGE_BINS = np.arange(10, 100, 10)         # 10–19, 20–29, ..., 80–89


def holm(p):
    """Holmova korekcia p-hodnôt."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        out[idx] = running
    return out


# --- 1. Vek -----------------------------------------------------------------------------------------------
def person_table(exams):
    """Jeden riadok na človeka: skupina, podskupina, vek a pohlavie z prvého vyšetrenia, počet vyšetrení."""
    first = exams.sort_values("datum").groupby("patient_id")
    out = first.agg(skupina=("skupina", "first"), podskupina=("podskupina", "first"), vek=("vek", "first"),
                    pohlavie=("pohlavie", "first"), pocet_vysetreni=("measurement_id", "size"))
    out["vekova_skupina"] = pd.cut(out.vek, AGE_BINS, right=False,
                                   labels=[f"{a}–{a + 9}" for a in AGE_BINS[:-1]])
    return out.reset_index()


# --- 2. Slabo používaná ruka ---------------------------------------------------------------------------------
def hand_table(X, exams, pct=2.5):
    """Sila rúk pre každé vyšetrenie (medián cez iterácie) a stav slabšej ruky podľa prahov zo zdravých.

    Rovnaké pravidlo ako krok 4 úlohy 2: málo používaná = sila slabšej ruky aj pomer rúk pod `pct`.
    percentilom zdravých. Vráti (tabuľka vyšetrení, prah sily, prah pomeru)."""
    hand = rf.hand_levels(np.median(rf.finger_levels(X), axis=1))
    ratio = hand.min(axis=1) / hand.max(axis=1)
    healthy = (exams.skupina == "zdravý").to_numpy()
    level_thr = float(np.percentile(hand[healthy].ravel(), pct))
    ratio_thr = float(np.percentile(ratio[healthy], pct))
    out = exams[["patient_id", "skupina", "podskupina", "vek", "datum", "measurement_id"]].copy()
    out["sila_lava"], out["sila_prava"], out["pomer_ruk"] = hand[:, 0], hand[:, 1], ratio
    out["slabsia_ruka"] = np.where(hand[:, 0] < hand[:, 1], "ľavá", "pravá")
    out["stav"] = rf.hand_status(hand, ratio, level_thr, ratio_thr)
    return out, level_thr, ratio_thr


def weak_hand_people(hands, level_thr, ratio_thr):
    """Zhrnutie na človeka: v koľkých vyšetreniach je ruka málo používaná, a stav z priemeru vyšetrení
    (tak to počíta kolega: najprv priemer síl cez vyšetrenia, potom prah)."""
    h = hands.copy()
    h["slaba"] = h.stav != "v norme"
    g = h.groupby("patient_id")
    out = g.agg(skupina=("skupina", "first"), podskupina=("podskupina", "first"), vek=("vek", "first"),
                vysetreni=("slaba", "size"), slabych_vysetreni=("slaba", "sum"),
                pomer_min=("pomer_ruk", "min"), pomer_max=("pomer_ruk", "max"),
                sila_lava=("sila_lava", "mean"), sila_prava=("sila_prava", "mean"))
    weak_side = h[h.slaba].groupby("patient_id").slabsia_ruka.agg(lambda s: "/".join(sorted(set(s))))
    out["slabsia_ruka"] = weak_side.reindex(out.index).fillna("")
    out["aspon_raz"] = out.slabych_vysetreni > 0
    out["vo_vacsine"] = out.slabych_vysetreni > out.vysetreni / 2
    mean_ratio = out[["sila_lava", "sila_prava"]].min(axis=1) / out[["sila_lava", "sila_prava"]].max(axis=1)
    out["pomer_z_priemeru"] = mean_ratio
    out["z_priemeru"] = (out[["sila_lava", "sila_prava"]].min(axis=1) < level_thr) & (mean_ratio < ratio_thr)
    return out.reset_index()


# --- 3A. Šum snímača z kalibrácie ----------------------------------------------------------------------------
UNLOADED_MAX_LEVEL = 0.003     # |priemer| ako nezaťažený snímač
UNLOADED_MAX_SLOW_SD = 0.0005  # žiadna pomalá zmena sily (nikto netlačí)


def calibration_noise(folder=CALIBRATION_DIR):
    """SD trasovej zložky (3,5–15 Hz) nezaťažených kanálov v kalibračných meraniach – úroveň šumu snímača.

    Rovnaké spracovanie ako pri hre (1 kHz -> 100 Hz, úsek 3–15 s, rovnaký filter). Vráti tabuľku
    kanál × iterácia s príznakom `nezatazeny`."""
    rows = []
    for path in sorted(glob.glob(str(Path(folder) / "results_*"))):
        x, _ = rf.preprocess_measurement(path)
        slow, tremor, _ = rf.decompose(x.astype(float))
        for i in range(x.shape[0]):
            for c, name in enumerate(rf.FINGER_NAMES):
                rows.append(dict(subor=Path(path).name, iteracia=i + 1, prst=name, uroven=float(x[i, :, c].mean()),
                                 pomaly_SD=float(slow[i, :, c].std()), tras_SD=float(tremor[i, :, c].std())))
    d = pd.DataFrame(rows)
    d["nezatazeny"] = (d.uroven.abs() < UNLOADED_MAX_LEVEL) & (d.pomaly_SD < UNLOADED_MAX_SLOW_SD)
    return d


# --- 3B–D. Tras prstov na úrovni človeka ---------------------------------------------------------------------
def person_finger_table(fing, exams):
    """Človek × prst: medián cez všetky iterácie (aj viacero vyšetrení) aktívneho prsta s ustálenými úsekmi."""
    f = fing.assign(patient_id=exams.patient_id.to_numpy()[fing.vysetrenie])
    out = f.groupby(["patient_id", "prst"]).agg(tras_relativny=("tras_relativny", "median"),
                                                tras_SD=("tras_SD", "median"), sila=("sila_prsta", "median"),
                                                iteracii=("tras_SD", "size")).reset_index()
    info = person_table(exams).set_index("patient_id")[["skupina", "podskupina", "vek"]]
    return out.join(info, on="patient_id")


def wide(pf, value="tras_relativny"):
    """Človek × prst (stĺpce v poradí kanálov); chýbajúci prst = neaktívny alebo bez ustálených úsekov."""
    return pf.pivot(index="patient_id", columns="prst", values=value).reindex(columns=rf.FINGER_NAMES)


def healthy_finger_test(pf):
    """Friedmanov test: líšia sa prsty u toho istého zdravého človeka? (ľudia so všetkými 10 prstami)."""
    w = wide(pf[pf.skupina == "zdravý"]).dropna()
    stat, p = friedmanchisquare(*[w[c] for c in w.columns])
    kendall_w = stat / (len(w) * (w.shape[1] - 1))
    return dict(ludi=len(w), chi2=float(stat), p=float(p), kendall_W=float(kendall_w))


def finger_group_ratio(pf, ref_mask=None, n_boot=2000, seed=0):
    """Pre každý prst: medián pacientov / medián zdravých (95 % CI bootstrap cez ľudí), Mann-Whitney, Holm."""
    rng = np.random.default_rng(seed)
    pat = pf[pf.skupina == "pacient"]
    ref = pf[pf.skupina == "zdravý"] if ref_mask is None else pf[ref_mask]
    rows = []
    for c in rf.FINGER_NAMES:
        a = pat.loc[pat.prst == c, "tras_relativny"].to_numpy()
        b = ref.loc[ref.prst == c, "tras_relativny"].to_numpy()
        boots = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(prst=c, nazov=FINGER_LABEL[c], ruka=FINGER_HAND[c], n_pacienti=len(a), n_zdravi=len(b),
                         median_pacienti=np.median(a), median_zdravi=np.median(b),
                         pomer=np.median(a) / np.median(b), CI_lo=lo, CI_hi=hi,
                         p=mannwhitneyu(a, b, alternative="two-sided").pvalue))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    return out


def elevated_fingers(pf, q):
    """Počet prstov nad normou: prst je nad normou, ak je jeho tras > q-kvantil zdravých pre ten istý prst.

    Pre zdravých sa prah počíta bez daného človeka (leave-one-out), aby sa sám nezapočítal do normy –
    podiel zdravých s ≥ 2 prstami nad normou je potom čestný odhad falošných poplachov."""
    w = wide(pf)
    healthy_ids = pf.loc[pf.skupina == "zdravý", "patient_id"].unique()
    hw = w.loc[healthy_ids]
    thr = hw.quantile(q)
    above = w.gt(thr) & w.notna()
    for pid in healthy_ids:                                  # leave-one-out pre zdravých
        thr_loo = hw.drop(index=pid).quantile(q)
        above.loc[pid] = w.loc[pid].gt(thr_loo) & w.loc[pid].notna()
    return above, thr


def tremor_pattern(above, w):
    """Typ trasu človeka: 0–1 prst nad normou = bez zvýšeného trasu; ≥ 2 prsty a menej ako polovica aktívnych
    prstov = lokálny (niekoľko prstov); ≥ 2 prsty a aspoň polovica aktívnych = rozsiahly (väčšina prstov)."""
    n_up = above.sum(axis=1)
    n_act = w.notna().sum(axis=1)
    pattern = np.where(n_up < 2, "bez zvýšeného trasu",
                       np.where(n_up < n_act / 2, "lokálny (niekoľko prstov)", "rozsiahly (väčšina prstov)"))
    left = above[rf.FINGER_NAMES[:5]].sum(axis=1)
    right = above[rf.FINGER_NAMES[5:]].sum(axis=1)
    hand = np.where(n_up == 0, "", np.where(left > right, "ľavá", np.where(right > left, "pravá", "obe rovnako")))
    return pd.DataFrame({"prstov_nad_normou": n_up, "aktivnych_prstov": n_act, "typ_trasu": pattern,
                         "ruka_s_trasom": hand}, index=w.index)
