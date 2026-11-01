"""Figures of the analysis stage (``outputs/pipeline/analysis/figures``). Blue = healthy, orange = sick."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from rehapiano.pipeline import catalog, settings  # noqa: E402
from rehapiano.pipeline.participants import HEALTHY, SICK  # noqa: E402

COLORS = {HEALTHY: "#2377a4", SICK: "#cf5c36"}
INK, MUTED = "#222222", "#777777"
KEY_FEATURES = ["ef_tremor_amp", "ef_slow_amp", "rh_press_rate_hz", "rh_balance_press"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white",
                     "axes.titleweight": "bold", "axes.titlesize": 10.5})


def _label(feature: str) -> str:
    return f"{feature} – {catalog.FEATURES[feature].description}"


def group_differences(h: pd.DataFrame, path: Path) -> None:
    h = h.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(h))
    for column, p_column, offset, name in [("auc", "p_holm", .15, "all ages"), ("auc_50", "p_50_holm", -.15, "age 50+")]:
        significant = h[p_column] < .05
        color = INK if name == "all ages" else MUTED
        ax.scatter(h.loc[significant, column], y[significant] + offset, s=34, color=color, label=f"{name}, p (Holm) < 0.05")
        ax.scatter(h.loc[~significant, column], y[~significant] + offset, s=34, facecolor="white", edgecolor=color,
                   label=f"{name}, not significant")
    ax.axvline(.5, color=MUTED, lw=.8, ls=":")
    ax.set_yticks(y, [_label(f) for f in h["feature"]], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC = probability that a random sick person has a higher value than a random healthy person\n"
                  "(0.5 = no difference; below 0.5 = lower in the sick group)")
    ax.set_title("Healthy vs sick: which measurements differ?")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def norms(people: pd.DataFrame, norm_table: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(1, len(KEY_FEATURES), figsize=(15, 3.9))
    rng = np.random.default_rng(0)
    for ax, feature in zip(axs, KEY_FEATURES):
        n = norm_table.loc[norm_table["feature"] == feature].copy()
        n["start"] = n["decade"].str.split("–").str[0].astype(int)
        for _, row in n.iterrows():
            ax.fill_between([row["start"], row["start"] + 10], row["norm_low"], row["norm_high"], color=COLORS[HEALTHY],
                            alpha=.12, lw=0)
            ax.plot([row["start"], row["start"] + 10], [row["norm_median"]] * 2, color=COLORS[HEALTHY], lw=1.4)
        for label in [HEALTHY, SICK]:
            part = people.loc[people["label"] == label]
            ax.scatter(part["age"] + rng.uniform(-.4, .4, len(part)), part[feature], s=12, color=COLORS[label], alpha=.7,
                       edgecolor="none", label=f"{label} group")
        ax.set_title(f"{feature}\n{n['method'].iloc[0]}", fontsize=9)
        ax.set_xlabel("Age (years)")
        if catalog.FEATURES[feature].log:
            ax.set_yscale("log")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Normal range of healthy people per decade (band = 2.5–97.5 %, line = median)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def roc(analysis_dir: Path, classifier: pd.DataFrame, path: Path) -> None:
    sets = ["All game features + age", "All game features", "Rhythm features", "EXACT_FORCE features", "Age only"]
    styles = dict(zip(sets, [("-", INK), ("-", "#cf5c36"), ("--", "#cf5c36"), (":", "#cf5c36"), ("-", MUTED)]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, subset, file in [(axs[0], "all ages", "all_ages"), (axs[1], f"age {settings.OLDER_AGE}+",
                                                                   f"age_{settings.OLDER_AGE}plus")]:
        oof = pd.read_csv(analysis_dir / f"classifier_oof_{file}.csv")
        y = oof["label"].eq(SICK).astype(int)
        for name in sets:
            row = classifier.loc[(classifier["people"] == subset) & (classifier["features"] == name)
                                 & (classifier["model"] == "logistic regression")].iloc[0]
            fpr, tpr, _ = roc_curve(y, oof[f"logistic regression | {name}"])
            style, color = styles[name]
            ax.plot(fpr, tpr, ls=style, color=color, lw=1.8, label=f"{name} (AUC {row['auc']:.2f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls=":")
        ax.set(xlabel="Healthy people wrongly flagged (1 − specificity)", ylabel="Sick people found (sensitivity)",
               title=f"{subset}: {int(row['n_healthy'])} healthy, {int(row['n_sick'])} sick")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Logistic regression, repeated 5-fold cross-validation (people never in train and test at once)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ablation(table: pd.DataFrame, path: Path) -> None:
    part = table.loc[table["model_includes_age"]].sort_values("auc_alone")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    y = np.arange(len(part))
    colors = [MUTED if f == "Age" else COLORS[SICK] for f in part["factor"]]
    axs[0].hlines(y, part["auc_alone_low"], part["auc_alone_high"], color=colors, lw=2.2, alpha=.6)
    axs[0].scatter(part["auc_alone"], y, color=colors, s=36, zorder=3)
    axs[0].axvline(.5, color=MUTED, lw=.8, ls=":")
    axs[0].set(xlabel="AUC with this factor alone (line = 95 % range over CV repeats)", title="What a factor can do alone")
    axs[0].set_yticks(y, part["factor"])
    axs[1].barh(y, part["auc_drop"], color=colors)
    axs[1].axvline(0, color=INK, lw=.8)
    axs[1].set(xlabel="AUC lost when the factor is removed from the full model (with age)",
               title="What only this factor adds")
    for ax in axs:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(output_dir: Path = settings.OUTPUT_DIR, analysis_dir: Path = settings.ANALYSIS_DIR) -> None:
    figures = analysis_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(analysis_dir / f"{name}.csv")  # noqa: E731
    people = pd.read_csv(output_dir / "person_features.csv", dtype={"patient_id": str})
    group_differences(read("hypotheses"), figures / "01_group_differences.png")
    norms(people, read("decade_norms"), figures / "02_norms_by_decade.png")
    roc(analysis_dir, read("classifier"), figures / "03_roc.png")
    ablation(read("ablation"), figures / "04_ablation.png")
