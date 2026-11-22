"""Obrázky do dokumentácie (docs/latex/obrazky/*.pdf).

Kreslí sa priamo z tabuliek v outputs/, v skutočnej veľkosti strany A4 (šírka sadzby 16 cm),
aby boli popisy v PDF čitateľné. Spustenie z koreňa repozitára:

    python docs/latex/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = Path(__file__).resolve().parent / "obrazky"
FIG.mkdir(exist_ok=True)

CM = 1 / 2.54
WIDTH = 16 * CM
HEALTHY, PATIENT = "#2F6DB5", "#D9622B"
EF_COLOR, RH_COLOR = "#6A51A3", "#D9622B"

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})

LABELS = {
    "tremor_amplitude": "amplitúda trasu",
    "tremor_amplitude_worse_hand": "amplitúda trasu horšej ruky",
    "movement_amplitude": "pomalé korekcie sily",
    "tremor_to_movement": "podiel trasu",
    "tremor_frequency_hz": "frekvencia trasu",
    "tremor_prominence": "rytmickosť trasu",
    "mean_force": "sila prstov",
    "force_balance": "rovnováha síl rúk",
    "tremor_balance": "rovnováha trasu rúk",
    "finger_spread": "rozdiely medzi prstami",
    "press_rate_hz": "tempo",
    "interval_cv": "nepravidelnosť rytmu",
    "amplitude_strong": "sila silnejšej ruky",
    "amplitude_weak": "sila slabšej ruky",
    "balance": "rovnováha rúk",
    "amplitude_cv": "nestálosť sily",
    "sync_lag_ms": "oneskorenie medzi rukami",
    "release_ratio": "neúplné uvoľnenie",
    "fatigue_amplitude": "únava – sila",
    "fatigue_rate": "únava – tempo",
}


def people(test):
    table = pd.read_csv(OUT / test / "analysis" / "people.csv")
    return table[table["screened"]]


def strip(ax, table, column, title, ylabel, scale=1.0, log=False, seed=0):
    rng = np.random.default_rng(seed)
    for x, group, color, name in ((0, "control", HEALTHY, "zdraví"), (1, "patient", PATIENT, "pacienti")):
        values = table.loc[table["group"] == group, column].dropna().to_numpy() * scale
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values, s=12, color=color, alpha=0.65,
                   linewidths=0)
        ax.hlines(np.median(values), x - 0.3, x + 0.3, color="black", linewidth=2)
    ax.set_xticks([0, 1], ["zdraví", "pacienti"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    else:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def figure_parameters():
    ef = people("exact_force")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 11 * CM))
    axes = axes.ravel()
    strip(axes[0], ef, "movement_amplitude", "pomalé korekcie sily", "amplitúda 0–4 Hz [×10⁻⁵]", 1e5, log=True)
    strip(axes[1], ef, "tremor_amplitude", "amplitúda trasu", "amplitúda 4–15 Hz [×10⁻⁵]", 1e5, log=True)
    strip(axes[2], ef, "mean_force", "sila prstov", "priemerná sila [×10⁻³]", 1e3)
    strip(axes[3], ef, "tremor_to_movement", "podiel trasu", "tras / pomalé korekcie")
    fig.tight_layout(h_pad=1.0, w_pad=2.0)
    fig.savefig(FIG / "obr_ef_parametre.pdf")
    plt.close(fig)

    rh = people("rhythm")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 11 * CM))
    axes = axes.ravel()
    strip(axes[0], rh, "press_rate_hz", "tempo", "stlačenia za sekundu")
    strip(axes[1], rh, "balance", "rovnováha rúk", "slabšia / silnejšia ruka")
    strip(axes[2], rh, "sync_lag_ms", "oneskorenie medzi rukami", "ms", log=True)
    strip(axes[3], rh, "amplitude_strong", "sila silnejšej ruky", "relatívna sila")
    fig.tight_layout(h_pad=1.0, w_pad=2.0)
    fig.savefig(FIG / "obr_rytmus_parametre.pdf")
    plt.close(fig)


def norm_panel(ax, test, column, title, ylabel, scale=1.0, log=False):
    table = people(test)
    norms = pd.read_csv(OUT / test / "analysis" / "norms_by_age_model.csv")
    norms = norms[norms["parameter"] == column].copy()
    norms["start"] = norms["decade"].str.slice(0, 2).astype(int)
    edges = np.r_[norms["start"].to_numpy(), norms["start"].iloc[-1] + 10]
    low = np.r_[norms["q2.5"].to_numpy(), norms["q2.5"].iloc[-1]] * scale
    high = np.r_[norms["q97.5"].to_numpy(), norms["q97.5"].iloc[-1]] * scale
    mid = np.r_[norms["q50"].to_numpy(), norms["q50"].iloc[-1]] * scale
    ax.fill_between(edges, low, high, step="post", color=HEALTHY, alpha=0.13, linewidth=0,
                    label="norma (95 % zdravých)")
    ax.step(edges, mid, where="post", color=HEALTHY, linewidth=1.4, label="medián zdravých")
    for group, color, name in (("control", HEALTHY, "zdraví"), ("patient", PATIENT, "pacienti")):
        part = table[table["group"] == group]
        ax.scatter(part["age"], part[column] * scale, s=11, color=color, alpha=0.65, linewidths=0, label=name)
    ax.set_xlim(10, 90)
    ax.set_xticks(range(10, 91, 10))
    ax.set_xlabel("vek [roky]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.5)
    if log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def figure_norms():
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 11.5 * CM))
    norm_panel(axes[0, 0], "exact_force", "movement_amplitude", "Exact Force: pomalé korekcie sily",
               "amplitúda 0–4 Hz [×10⁻⁵]", 1e5, log=True)
    norm_panel(axes[0, 1], "exact_force", "mean_force", "Exact Force: sila prstov",
               "priemerná sila [×10⁻³]", 1e3)
    norm_panel(axes[1, 0], "rhythm", "press_rate_hz", "Rhythm: tempo", "stlačenia za sekundu")
    norm_panel(axes[1, 1], "rhythm", "balance", "Rhythm: rovnováha rúk", "slabšia / silnejšia ruka")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=1.2, w_pad=1.5, rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "obr_normy.pdf")
    plt.close(fig)


def figure_classifier():
    table = pd.read_csv(OUT / "combined" / "classifier.csv")
    table = table[table["model"] == "logistická regresia"]
    all_sets = list(dict.fromkeys(table["set"]))
    sets = [(next(s for s in all_sets if "len" not in s), "všetci ľudia"),
            (next(s for s in all_sets if "len" in s), "len ľudia nad 50 rokov")]
    features = [("EXACT_FORCE", "Exact Force", EF_COLOR), ("rytmus", "Rhythm", RH_COLOR),
                ("EXACT_FORCE + rytmus", "obe cvičenia", "#2E8B57")]
    fig, ax = plt.subplots(figsize=(11 * CM, 6.3 * CM))
    width = 0.26
    for i, (key, name) in enumerate(sets):
        for j, (feat, label, color) in enumerate(features):
            row = table[(table["set"] == key) & (table["features"] == feat)].iloc[0]
            x = i + (j - 1) * width
            ax.bar(x, row["auc"], width * 0.92, color=color, label=label if i == 0 else None)
            ax.errorbar(x, row["auc"], yerr=[[row["auc"] - row["auc_low"]], [row["auc_high"] - row["auc"]]],
                        color="black", linewidth=1, capsize=2)
            ax.text(x, row["auc_high"] + 0.008, f"{row['auc']:.2f}".replace(".", ","), ha="center",
                    va="bottom", fontsize=8.5)
    age_only = table[(table["set"] == sets[0][0]) & (table["features"] == "len vek")]["auc"].iloc[0]
    ax.axhline(age_only, color="#444444", linestyle="--", linewidth=1)
    ax.text(1.42, age_only, " model iba\n z veku", ha="left", va="center", fontsize=8.5, color="#444444")
    ax.set_xlim(-0.45, 1.75)
    ax.set_xticks([0, 1], [name for _, name in sets])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC (krížová validácia)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0, 1.13))
    fig.savefig(FIG / "obr_klasifikator.pdf")
    plt.close(fig)


def figure_ablation():
    table = pd.read_csv(OUT / "combined" / "ablation_factors_with_age.csv")
    table = table.sort_values("auc_alone")
    names = table["group"].str.replace("rytmus:", "Rhythm:").str.replace("EF:", "Exact Force:").to_list()
    colors = ["#444444" if n == "vek" else (EF_COLOR if n.startswith("Exact") else RH_COLOR) for n in names]
    fig, ax = plt.subplots(figsize=(WIDTH, 8.4 * CM))
    y = np.arange(len(table))
    ax.hlines(y, 0.38, table["auc_alone"], color="#DDDDDD", linewidth=1)
    ax.errorbar(table["auc_alone"], y, xerr=[table["auc_alone"] - table["auc_alone_low"],
                                             table["auc_alone_high"] - table["auc_alone"]],
                fmt="none", ecolor="black", linewidth=0.8)
    ax.scatter(table["auc_alone"], y, color=colors, s=36, zorder=3)
    ax.set_yticks(y, names)
    ax.set_xlim(0.38, 0.88)
    ax.axvline(0.5, color="#555555", linestyle=":", linewidth=1)
    ax.set_xlabel("AUC modelu, ktorý používa iba tento faktor (0,5 = hádanie)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "obr_ablacia.pdf")
    plt.close(fig)


def figure_repeatability():
    table = pd.read_csv(OUT / "comparison" / "repeatability.csv")
    table["label"] = table["parameter"].map(LABELS)
    table = pd.concat([table[table["test"] == "EXACT_FORCE"].sort_values("icc"),
                       table[table["test"] == "RYTMUS"].sort_values("icc")])
    fig, ax = plt.subplots(figsize=(WIDTH, 10 * CM))
    y = np.arange(len(table))
    colors = [EF_COLOR if t == "EXACT_FORCE" else RH_COLOR for t in table["test"]]
    ax.barh(y, table["icc"], color=colors, height=0.7)
    ax.set_yticks(y, table["label"])
    for value in (0.5, 0.75):
        ax.axvline(value, color="#555555", linestyle=":", linewidth=1)
    ax.text(0.5, len(table) - 0.2, " slabá | stredná", ha="center", va="bottom", fontsize=8.5)
    ax.text(0.75, len(table) - 0.2, " stredná | dobrá", ha="center", va="bottom", fontsize=8.5)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("ICC – zhoda prvého a druhého merania")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RH_COLOR, label="Rhythm"), Patch(color=EF_COLOR, label="Exact Force")],
              loc="lower right", frameon=False)
    fig.savefig(FIG / "obr_opakovatelnost.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_parameters()
    figure_norms()
    figure_classifier()
    figure_ablation()
    figure_repeatability()
    print("obrázky uložené do", FIG)
