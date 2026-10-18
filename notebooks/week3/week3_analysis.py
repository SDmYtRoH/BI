"""Week 3 (task sheet column F): age structure, weakly used hands and per-finger tremor.

Builds on the per-finger pipeline in ``<repo>/src/rehapiano`` (tables in
``<repo>/outputs``, produced by ``scripts/build_dataset.py``) and on the
questionnaire screening from week 2. Differences from the Slovak notebook
``analyza_prstov.ipynb`` in this folder:

* the healthy group is the 76 questionnaire-screened controls, not all 90;
* the age-matched reference (healthy, >= 50 years) is scored leave-one-out;
* fingers are also compared at equal force, because force differs ~7x between fingers;
* age, recording batch and the detected tremor peak frequency are checked explicitly.

Run from this folder:  python week3_analysis.py
Tables go to ./outputs, figures to ./figures. Nothing outside this folder is written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from rehapiano import config  # noqa: E402
from rehapiano.features.spectral import band_mask, compute_spectrogram  # noqa: E402
from rehapiano.io.metadata import resolve_project_path  # noqa: E402
from rehapiano.preprocessing.pipeline import load_measurement_data  # noqa: E402
from rehapiano.stats.finger_tremor import active_finger_table, noise_comparison  # noqa: E402
from rehapiano.stats.hand_activity import hand_activity_table  # noqa: E402
from rehapiano.stats.selection import primary_step  # noqa: E402
from rehapiano.stats.tests import friedman, holm_adjust, mann_whitney  # noqa: E402

PIPELINE = REPO / "outputs"
LABELS_CSV = REPO / "notebooks" / "week2" / "outputs" / "participant_labels.csv"
HAND_USAGE_CSV = REPO / "notebooks" / "week2" / "outputs" / "hand_usage.csv"
OUT = HERE / "outputs"
FIG = HERE / "figures"

HEALTHY, SICK, OTHER = "Healthy", "Sick", "Other control"
GROUPS = [HEALTHY, SICK]
COLORS = {HEALTHY: "#2377a4", SICK: "#cf5c36", OTHER: "#9a9a9a"}
BATCH = {"dataset_tuke.csv": "Feb 2025", "dataset_99.csv": "Sep 2026", "dataset_patients.csv": "Mar-Aug 2025"}
HANDS = ["left", "right"]
FINGERS = ["thumb", "index", "middle", "ring", "little"]  # display order; sensor 5 -> 1, mapping unverified

WEAK_HAND_QUANTILE = 0.025  # below 97.5 % of healthy people
FIXED_RATIO_THRESHOLDS = (0.10, 0.25, 0.50)
REFERENCE_MIN_AGE = 50
ELEVATED_Z = 1.96
BOOTSTRAP_SAMPLES = 2000
SEED = 0

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                     "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 11})


# --------------------------------------------------------------------------- inputs
def load_inputs() -> dict:
    """Pipeline tables restricted to the primary step, with the plain-language group labels."""

    labels = pd.read_csv(LABELS_CSV, dtype={"patient_id": str})
    labels["label"] = np.select([labels["cohort"].eq("patient"), labels["label_status"].eq("no_reported_condition")],
                                [SICK, HEALTHY], OTHER)
    labels["batch"] = labels["source_cohort"].map(BATCH)
    labels = labels[["patient_id", "label", "label_status", "batch"]]

    summary = pd.read_csv(PIPELINE / "person_summary.csv", dtype={"patient_id": str})
    by_person = pd.read_csv(PIPELINE / "features_by_person_finger.csv", dtype={"patient_id": str})
    by_attempt = pd.read_csv(PIPELINE / "features_by_attempt_finger.csv", dtype={"patient_id": str})

    people = primary_step(summary).merge(labels, on="patient_id", how="left", validate="1:1")
    if people["label"].isna().any():
        raise ValueError("people without a week-2 label: " + ", ".join(people.loc[people["label"].isna(), "patient_id"]))
    fingers = active_finger_table(by_person).merge(labels, on="patient_id", how="left")
    fingers["log_amplitude"] = np.log(fingers["tremor_mean_amplitude"])
    fingers["log_force"] = np.log(fingers["mean_force"].where(fingers["mean_force"] > 0))
    activity = hand_activity_table(summary, by_person).merge(labels, on="patient_id", how="left")
    activity = activity.merge(people[["patient_id", "metadata_hand"]], on="patient_id", how="left")
    attempts = primary_step(by_attempt).merge(labels, on="patient_id", how="left")
    return {"people": people, "fingers": fingers, "activity": activity, "attempts": attempts,
            "noise": pd.read_csv(PIPELINE / "sensor_noise.csv"),
            "hand_usage": pd.read_csv(HAND_USAGE_CSV, dtype={"patient_id": str})}


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _bootstrap_ratio(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95 % bootstrap interval of median(b) / median(a)."""

    ratios = [np.median(rng.choice(b, len(b))) / np.median(rng.choice(a, len(a))) for _ in range(BOOTSTRAP_SAMPLES)]
    return tuple(np.percentile(ratios, [2.5, 97.5]))


# --------------------------------------------------------------------------- 1. age
def age_analysis(people: pd.DataFrame) -> dict:
    width = config.AGE_GROUP_WIDTH_YEARS
    starts = range(int(people["age"].min() // width * width), int(people["age"].max() // width * width) + 1, width)
    bins = [f"{start}-{start + width - 1}" for start in starts]
    counts = pd.crosstab(people["age_group"], people["label"]).reindex(index=bins, columns=[HEALTHY, SICK, OTHER],
                                                                       fill_value=0)
    for label in GROUPS:
        counts[f"{label} %"] = 100 * counts[label] / counts[label].sum()
    counts.index.name = "Age group"

    rows = []
    for label in [HEALTHY, SICK, OTHER]:
        frame = people.loc[people["label"] == label]
        age = frame["age"]
        rows.append({"Group": label, "People": len(frame), "Median age": age.median(),
                     "Q1": age.quantile(.25), "Q3": age.quantile(.75), "Youngest": age.min(), "Oldest": age.max(),
                     "Aged 50+": int((age >= REFERENCE_MIN_AGE).sum()),
                     "Female": int((frame["gender"] == "female").sum()), "Male": int((frame["gender"] == "male").sum()),
                     "Handedness recorded": int(frame["metadata_hand"].notna().sum()),
                     "Recording period": ", ".join(sorted(frame["batch"].unique()))})
    summary = pd.DataFrame(rows)
    healthy, sick = (people.loc[people["label"] == g, "age"] for g in GROUPS)
    test = mann_whitney(healthy, sick)
    gender = pd.crosstab(people.loc[people["label"].isin(GROUPS), "label"],
                         people.loc[people["label"].isin(GROUPS), "gender"])
    gender_p = stats.chi2_contingency(gender)[1]

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(bins))
    for offset, label in zip((-0.2, 0.2), GROUPS):
        bars = ax.bar(x + offset, counts[label], width=0.38, color=COLORS[label], edgecolor="white", linewidth=1,
                      label=f"{label} group (n = {int(counts[label].sum())})")
        ax.bar_label(bars, labels=[str(v) if v else "" for v in counts[label]], fontsize=8, padding=2, color="#333")
    ax.set_xticks(x, [b.replace("-", "–") for b in bins])
    ax.set_xlabel("Age (years, 10-year groups)")
    ax.set_ylabel("Number of people")
    ax.set_title("The two groups barely overlap in age")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    savefig(fig, "01_age_histogram")
    return {"age_groups": counts.reset_index(), "age_summary": summary,
            "age_test": pd.DataFrame([{"comparison": "age, healthy (a) vs sick (b)", **test}]),
            "gender_p": gender_p}


# --------------------------------------------------------------------------- 2. weakly used hand
def weak_hand_analysis(activity: pd.DataFrame, hand_usage: pd.DataFrame) -> dict:
    """Force symmetry (weaker / stronger hand) against the healthy group's 2.5th percentile."""

    compared = activity.loc[activity["label"].isin(GROUPS)].copy()
    healthy = compared.loc[compared["label"] == HEALTHY, "force_ratio"].to_numpy()
    sick = compared.loc[compared["label"] == SICK, "force_ratio"].to_numpy()
    threshold = float(np.quantile(healthy, WEAK_HAND_QUANTILE))

    rng = np.random.default_rng(SEED)
    boot_counts = [(sick < np.quantile(rng.choice(healthy, len(healthy)), WEAK_HAND_QUANTILE)).sum()
                   for _ in range(BOOTSTRAP_SAMPLES)]

    low_use = (hand_usage.loc[(hand_usage["window_s"] == 5) & ~hand_usage["usable"].astype(bool)]
               .groupby(["patient_id", "hand"]).size().rename("week2_low_use_windows").reset_index())
    compared["weaker_hand"] = compared["force_weaker_hand"]
    compared = compared.merge(low_use.rename(columns={"hand": "weaker_hand"}), on=["patient_id", "weaker_hand"],
                              how="left")
    compared["week2_low_use_windows"] = compared["week2_low_use_windows"].fillna(0).astype(int)
    compared["inactive_fingers_weaker_hand"] = np.where(compared["weaker_hand"] == "left",
                                                        compared["left_n_inactive_fingers"],
                                                        compared["right_n_inactive_fingers"])
    flagged = compared.loc[compared["force_ratio"] < threshold, [
        "patient_id", "label", "age", "weaker_hand", "force_ratio", "fluctuation_ratio", "left_mean_force",
        "right_mean_force", "inactive_fingers_weaker_hand", "week2_low_use_windows"]].sort_values(
        ["label", "force_ratio"])

    rows = []
    for name, value in [*[(f"ratio < {t:.2f}", t) for t in FIXED_RATIO_THRESHOLDS],
                        (f"ratio < {threshold:.3f} (healthy 2.5th percentile)", threshold)]:
        row = {"Rule": name}
        for label, values in [(HEALTHY, healthy), (SICK, sick)]:
            row[f"{label}: people below"] = int((values < value).sum())
            row[f"{label}: % below"] = 100 * (values < value).mean()
        rows.append(row)
    counts = pd.DataFrame(rows)

    healthy_frame = compared.loc[compared["label"] == HEALTHY]
    age_rho = stats.spearmanr(healthy_frame["age"], healthy_frame["force_ratio"])
    dominance = pd.crosstab(healthy_frame["metadata_hand"].rename("Writing hand (questionnaire)"),
                            healthy_frame["force_weaker_hand"].rename("Hand that pressed less"))
    checks = {"threshold": threshold, "sick_below": int((sick < threshold).sum()), "n_sick": len(sick),
              "healthy_below": int((healthy < threshold).sum()), "n_healthy": len(healthy),
              "sick_below_bootstrap_95": [int(v) for v in np.percentile(boot_counts, [2.5, 97.5])],
              "age_rho_healthy": float(age_rho[0]), "age_p_healthy": float(age_rho[1]),
              "group_test": mann_whitney(healthy, sick),
              "sick_below_010": int((sick < 0.10).sum())}

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for label in GROUPS:
        frame = compared.loc[compared["label"] == label]
        ax.scatter(frame["left_mean_force"], frame["right_mean_force"], s=26, color=COLORS[label], alpha=.8,
                   edgecolor="white", linewidth=.6, label=f"{label} group")
    lim = np.array([2e-3, 8e-2])
    ax.plot(lim, lim, color="#555", lw=1, label="Both hands equal")
    ax.plot(lim, lim * threshold, color="#555", lw=1, ls="--", label=f"One hand at {threshold:.0%} of the other")
    ax.plot(lim, lim / threshold, color="#555", lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Left hand: average finger force (device units)",
           ylabel="Right hand: average finger force", title="Left vs right hand force")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axs[1]
    jitter = np.random.default_rng(1)
    for k, label in enumerate(GROUPS):
        values = compared.loc[compared["label"] == label, "force_ratio"]
        ax.scatter(k + jitter.uniform(-.17, .17, len(values)), values, s=22, color=COLORS[label], alpha=.75,
                   edgecolor="white", linewidth=.5)
        ax.plot([k - .25, k + .25], [values.median()] * 2, color="#222", lw=2)
    ax.axhline(threshold, color="#555", ls="--", lw=1)
    ax.text(1.45, threshold + .015, f"healthy lower limit ({threshold:.2f})", fontsize=8, color="#333")
    label_y = -1.0
    for _, row in flagged.loc[flagged["label"] == SICK].sort_values("force_ratio").iterrows():
        label_y = max(row["force_ratio"], label_y + .035)  # keep labels apart
        ax.annotate(row["patient_id"], (1.0, row["force_ratio"]), xytext=(1.3, label_y), fontsize=7, color="#333",
                    va="center", arrowprops={"arrowstyle": "-", "color": "#bbb", "lw": .6})
    ax.set_xticks([0, 1], [f"Healthy group\n(n = {len(healthy)})", f"Sick group\n(n = {len(sick)})"])
    ax.set_xlim(-.6, 2.0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weaker hand ÷ stronger hand (1 = symmetric)")
    ax.set_title("Hand symmetry per person")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savefig(fig, "02_hand_symmetry")
    return {"weak_hand_people": flagged, "weak_hand_counts": counts, "weak_hand_dominance": dominance.reset_index(),
            "weak_hand_checks": checks}


def _first_measurement_path(attempts: pd.DataFrame, patient_id: str) -> Path:
    rows = attempts.loc[attempts["patient_id"] == patient_id].sort_values(["timestamp", "attempt"])
    return resolve_project_path(rows["file_path"].iloc[0])


def weak_hand_example(attempts: pd.DataFrame, activity: pd.DataFrame) -> dict:
    """Whole-attempt hand force of the most asymmetric sick person and a typical older healthy person."""

    sick = activity.loc[activity["label"] == SICK].nsmallest(1, "force_ratio").iloc[0]
    healthy_pool = activity.loc[(activity["label"] == HEALTHY) & (activity["age"] >= REFERENCE_MIN_AGE)]
    healthy = healthy_pool.iloc[(healthy_pool["force_ratio"] - healthy_pool["force_ratio"].median()).abs().argsort()
                                ].iloc[0]
    people = [(HEALTHY, healthy), (SICK, sick)]
    fig, axs = plt.subplots(2, 4, figsize=(11, 4.8), sharex=True, sharey=True)
    for row, (label, person) in enumerate(people):
        data = load_measurement_data(_first_measurement_path(attempts, person["patient_id"]), start_s=0, end_s=16)
        for col, (attempt, frame) in enumerate(data.groupby("attempt")):
            if col >= 4:
                break
            ax = axs[row, col]
            ax.axvspan(config.CROP_START_S, config.CROP_END_S, color="#eee", lw=0)
            for hand, style in [("left", "-"), ("right", "--")]:
                columns = [f"{hand}_sensor_{k}" for k in range(1, 6)]
                ax.plot(frame["time_s"], frame[columns].mean(axis=1), ls=style, lw=1.3, color=COLORS[label],
                        label=f"{hand.capitalize()} hand")
            ax.set_title(f"Attempt {attempt}", fontsize=9, fontweight="normal")
            if col == 0:
                ax.set_ylabel(f"{label} participant\nforce (device units)")
        axs[row, 0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs[1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Average finger force of each hand; grey = the 3–15 s part used for the analysis", fontsize=10)
    fig.tight_layout()
    savefig(fig, "03_weak_hand_example")
    return {"weak_hand_example_ids": {HEALTHY: healthy["patient_id"], SICK: sick["patient_id"],
                                      "healthy_ratio": float(healthy["force_ratio"]),
                                      "sick_ratio": float(sick["force_ratio"]), "healthy_age": int(healthy["age"]),
                                      "sick_age": int(sick["age"])}}


# --------------------------------------------------------------------------- 3. fingers
def _finger_label(hand: str, finger: str) -> str:
    return f"{hand[0].upper()} {finger}"


def noise_analysis(fingers: pd.DataFrame, noise: pd.DataFrame) -> dict:
    table = noise_comparison(fingers.loc[fingers["label"].isin(GROUPS)].assign(group=lambda t: t["label"]), noise)
    table = table.rename(columns={"group": "label"})
    noise_median = float(noise["tremor_mean_amplitude"].median())
    noise_p95 = float(noise["tremor_mean_amplitude"].quantile(.95))

    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    positions = {(h, f): i + (6 if h == "right" else 0) for h in HANDS for i, f in enumerate(FINGERS)}
    for offset, label in zip((-.17, .17), GROUPS):
        frame = fingers.loc[fingers["label"] == label]
        for (hand, finger), x in positions.items():
            values = frame.loc[(frame["hand"] == hand) & (frame["finger_name"] == finger), "tremor_mean_amplitude"]
            q1, med, q3 = values.quantile([.25, .5, .75])
            ax.plot([x + offset] * 2, [q1, q3], color=COLORS[label], lw=2.2, alpha=.55)
            ax.scatter(x + offset, med, s=34, color=COLORS[label], edgecolor="white", linewidth=.6, zorder=3,
                       label=f"{label} group" if (hand, finger) == ("left", "thumb") else None)
    ax.axhspan(noise["tremor_mean_amplitude"].min(), noise_p95, color="#bbb", alpha=.35, lw=0,
               label="Unloaded sensor (noise only)")
    ax.set_yscale("log")
    ax.set_xticks(list(positions.values()), [f for _ in HANDS for f in FINGERS])
    for hand, center in [("Left hand", 2), ("Right hand", 8)]:
        ax.text(center, 1.02, hand, transform=ax.get_xaxis_transform(), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("4–15 Hz oscillation amplitude\n(device units, log scale)")
    ax.set_title("Finger oscillations compared with the sensor's own noise", pad=18)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax.grid(axis="x", visible=False)
    savefig(fig, "04_finger_vs_noise")
    return {"finger_noise": table, "noise_level": {"median": noise_median, "p95": noise_p95}}


def _within_hand_rho(fingers: pd.DataFrame, column: str) -> pd.Series:
    def rho(frame: pd.DataFrame) -> float:
        frame = frame.dropna(subset=["mean_force", column])
        return stats.spearmanr(frame["mean_force"], frame[column])[0] if len(frame) >= 4 else np.nan

    return fingers.groupby(["patient_id", "label", "hand"])[["mean_force", column]].apply(rho).rename(column)


def finger_force_analysis(fingers: pd.DataFrame) -> dict:
    """Does a finger oscillate more than the others, and is that only because it presses harder?"""

    compared = fingers.loc[fingers["label"].isin(GROUPS)].dropna(subset=["log_amplitude", "log_force"]).copy()
    keys = ["patient_id", "hand"]
    centred_force = compared["log_force"] - compared.groupby(keys)["log_force"].transform("mean")
    centred_amp = compared["log_amplitude"] - compared.groupby(keys)["log_amplitude"].transform("mean")
    healthy = compared["label"] == HEALTHY
    slope = float((centred_force[healthy] * centred_amp[healthy]).sum() / (centred_force[healthy] ** 2).sum())
    compared["profile_raw"] = centred_amp  # log of (finger amplitude / hand's geometric mean)
    compared["profile_force_adjusted"] = centred_amp - slope * centred_force

    rows, tests = [], []
    for label in GROUPS:
        for hand in HANDS:
            frame = compared.loc[(compared["label"] == label) & (compared["hand"] == hand)]
            for kind in ["profile_raw", "profile_force_adjusted"]:
                wide = frame.pivot_table(index="patient_id", columns="finger_name", values=kind)[FINGERS]
                tests.append({"label": label, "hand": hand, "measure": kind, **friedman(wide)})
            for finger in FINGERS:
                part = frame.loc[frame["finger_name"] == finger]
                rows.append({"label": label, "hand": hand, "finger": finger, "people": len(part),
                             "median_force": part["mean_force"].median(),
                             "median_amplitude": part["tremor_mean_amplitude"].median(),
                             "relative_amplitude": float(np.exp(part["profile_raw"].median())),
                             "relative_amplitude_equal_force": float(np.exp(part["profile_force_adjusted"].median()))})
    profile = pd.DataFrame(rows)

    group_tests = []
    for hand in HANDS:
        for finger in FINGERS:
            part = compared.loc[(compared["hand"] == hand) & (compared["finger_name"] == finger)]
            for kind in ["profile_raw", "profile_force_adjusted"]:
                group_tests.append({"hand": hand, "finger": finger, "measure": kind,
                                    **mann_whitney(part.loc[part["label"] == HEALTHY, kind],
                                                   part.loc[part["label"] == SICK, kind])})
    group_tests = pd.DataFrame(group_tests)
    group_tests["p_holm"] = group_tests.groupby("measure")["p"].transform(lambda p: holm_adjust(p))

    rho_amp = _within_hand_rho(compared, "tremor_mean_amplitude")
    rho_ratio = _within_hand_rho(compared, "tremor_to_movement_ratio")
    winners = compared.loc[compared.groupby(keys)["tremor_to_movement_ratio"].idxmax()]
    weakest = compared.loc[compared.groupby(keys)["mean_force"].idxmin()]
    same = (winners.set_index(keys)["finger_name"] == weakest.set_index(keys)["finger_name"]).mean()
    checks = {"force_slope": slope, "rho_force_amplitude_median": float(rho_amp.median()),
              "rho_force_ratio_median": float(rho_ratio.median()),
              "ratio_winner_is_weakest_finger": float(same)}

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=False)
    x = np.arange(len(FINGERS))
    for ax, column, title, ylabel in [
            (axs[0], "median_force", "How hard each finger presses", "Median finger force (device units)"),
            (axs[1], "relative_amplitude", "Oscillation, raw", "Oscillation relative to the\nperson's own hand average"),
            (axs[2], "relative_amplitude_equal_force", "Oscillation at equal force",
             "Oscillation relative to the\nhand average, force-adjusted")]:
        for label in GROUPS:
            for hand, marker, style in [("left", "o", "-"), ("right", "s", "--")]:
                values = profile.loc[(profile["label"] == label) & (profile["hand"] == hand)].set_index("finger")
                ax.plot(x, values.loc[FINGERS, column], marker=marker, ls=style, lw=1.4, ms=5, color=COLORS[label],
                        label=f"{label}, {hand} hand")
        if column != "median_force":
            ax.axhline(1, color="#555", lw=.8)
            ax.set_yscale("log")
            ax.set_yticks([.5, .7, 1, 1.5, 2], ["0.5×", "0.7×", "1×", "1.5×", "2×"])
            ax.minorticks_off()
        ax.set_xticks(x, FINGERS)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "05_finger_profiles")
    return {"finger_profile": profile, "finger_profile_friedman": pd.DataFrame(tests),
            "finger_profile_group_tests": group_tests, "finger_force_checks": checks}


def _robust_reference(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    median = float(values.median())
    return median, float(1.4826 * np.median(np.abs(values - median)))


def group_by_finger(fingers: pd.DataFrame) -> dict:
    """Sick vs healthy for each finger, both groups restricted to age >= 50."""

    rng = np.random.default_rng(SEED)
    older = fingers.loc[fingers["label"].isin(GROUPS) & (fingers["age"] >= REFERENCE_MIN_AGE)]
    rows = []
    for hand in HANDS:
        for finger in FINGERS:
            part = older.loc[(older["hand"] == hand) & (older["finger_name"] == finger)]
            healthy = part.loc[part["label"] == HEALTHY, "tremor_mean_amplitude"].dropna().to_numpy()
            sick = part.loc[part["label"] == SICK, "tremor_mean_amplitude"].dropna().to_numpy()
            low, high = _bootstrap_ratio(healthy, sick, rng)
            rows.append({"hand": hand, "finger": finger, "ratio_sick_to_healthy": np.median(sick) / np.median(healthy),
                         "ci_low": low, "ci_high": high, **mann_whitney(healthy, sick)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    labels = [_finger_label(h, f) for h, f in zip(table["hand"], table["finger"])]
    y = np.arange(len(table))[::-1]
    ax.hlines(y, table["ci_low"], table["ci_high"], color=COLORS[SICK], lw=2.2, alpha=.6)
    ax.scatter(table["ratio_sick_to_healthy"], y, s=40, color=COLORS[SICK], edgecolor="white", zorder=3)
    ax.axvline(1, color="#555", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xticks([.75, 1, 1.5, 2, 3], ["0.75×", "1×", "1.5×", "2×", "3×"])
    ax.minorticks_off()
    ax.set_xlabel("Sick group ÷ healthy group (median 4–15 Hz amplitude)\nline = 95 % bootstrap interval")
    n_h = older.loc[older["label"] == HEALTHY, "patient_id"].nunique()
    n_s = older.loc[older["label"] == SICK, "patient_id"].nunique()
    ax.set_title(f"Every finger oscillates more in the sick group\n(people aged 50+: {n_s} sick, {n_h} healthy)")
    ax.grid(axis="y", visible=False)
    savefig(fig, "06_finger_group_ratio")
    return {"finger_group_ratio": table}


def age_amplitude(fingers: pd.DataFrame) -> dict:
    """Person-level amplitude (geometric mean over active fingers) against age, by group and batch."""

    person = (fingers.loc[fingers["label"].isin(GROUPS)]
              .groupby(["patient_id", "label", "batch", "age"])["log_amplitude"].mean().reset_index())
    person["amplitude"] = np.exp(person["log_amplitude"])
    rows = []
    for (label, batch), frame in [*person.groupby(["label", "batch"]), *[((l, "all"), person[person["label"] == l])
                                                                          for l in GROUPS]]:
        rho, p = stats.spearmanr(frame["age"], frame["amplitude"]) if len(frame) >= 5 else (np.nan, np.nan)
        rows.append({"label": label, "batch": batch, "people": len(frame), "median_age": frame["age"].median(),
                     "median_amplitude": frame["amplitude"].median(), "rho_age_amplitude": rho, "p": p})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for label, batch, marker in [(HEALTHY, "Feb 2025", "o"), (HEALTHY, "Sep 2026", "^"), (SICK, "Mar-Aug 2025", "o")]:
        frame = person.loc[(person["label"] == label) & (person["batch"] == batch)]
        ax.scatter(frame["age"], frame["amplitude"], s=30, marker=marker, color=COLORS[label], alpha=.8,
                   edgecolor="white", linewidth=.6, label=f"{label} group, recorded {batch.replace('-', '–')}")
    for label in GROUPS:
        frame = person.loc[person["label"] == label]
        bands = pd.cut(frame["age"], [15, 30, 50, 65, 75, 95])
        medians = frame.groupby(bands, observed=True).agg(age=("age", "median"), amp=("amplitude", "median"),
                                                          n=("age", "size"))
        medians = medians.loc[medians["n"] >= 3]
        ax.plot(medians["age"], medians["amp"], color=COLORS[label], lw=2, alpha=.9)
    ax.set_yscale("log")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("4–15 Hz oscillation amplitude\n(person average, log scale)")
    ax.set_title("The sick group oscillates more at most ages;\nin healthy people amplitude does not rise with age")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, "07_age_vs_amplitude")
    return {"age_amplitude": table}


def tremor_patterns(fingers: pd.DataFrame) -> dict:
    """Elevated fingers (z > 1.96 vs healthy aged 50+, leave-one-out) and the resulting pattern, age 50+ only."""

    older = fingers.loc[fingers["label"].isin(GROUPS) & (fingers["age"] >= REFERENCE_MIN_AGE)].copy()
    older["z"] = np.nan
    for (hand, finger), part in older.groupby(["hand", "finger_name"]):
        reference = part.loc[part["label"] == HEALTHY]
        for index, row in part.iterrows():
            pool = reference.loc[reference["patient_id"] != row["patient_id"], "log_amplitude"]
            median, mad = _robust_reference(pool)
            older.loc[index, "z"] = (row["log_amplitude"] - median) / mad if mad > 0 else np.nan
    scored = older.dropna(subset=["z"]).assign(elevated=lambda t: t["z"] > ELEVATED_Z)
    scored["finger"] = [_finger_label(h, f) for h, f in zip(scored["hand"], scored["finger_name"])]
    person = scored.groupby(["patient_id", "label", "age"]).agg(
        fingers_scored=("z", "size"), fingers_elevated=("elevated", "sum"), median_z=("z", "median")).reset_index()
    person["elevated_fingers"] = scored.groupby(["patient_id", "label", "age"])[["elevated", "finger"]].apply(
        lambda g: ", ".join(g.loc[g["elevated"], "finger"])).to_numpy()
    person["pattern"] = np.select([person["fingers_elevated"] < config.LOCALIZED_MIN_FINGERS,
                                   person["fingers_elevated"] < config.GENERALIZED_MIN_FINGERS],
                                  ["none", "localized (2–4 fingers)"], "generalized (5+ fingers)")
    counts = pd.crosstab(person["pattern"], person["label"]).reindex(
        ["none", "localized (2–4 fingers)", "generalized (5+ fingers)"], fill_value=0)
    counts.index.name = "Pattern"
    return {"tremor_pattern_people": person.sort_values(["label", "fingers_elevated"], ascending=[True, False]),
            "tremor_pattern_counts": counts.reset_index()}


def _mean_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies, _, magnitude = compute_spectrogram(values)
    return frequencies, magnitude.mean(axis=1)


def peak_frequency_check(attempts: pd.DataFrame, noise: pd.DataFrame) -> dict:
    """Is the ~5 Hz 'tremor peak' a rhythm, or the highest bump on a falling spectrum?"""

    used = attempts.loc[attempts["label"].isin(GROUPS) & (attempts["status"] == "valid")
                        & ~attempts["finger_inactive"].astype(bool) & ~attempts["attempt_outlier"].astype(bool)]
    sources = [*[(label, used.loc[used["label"] == label]) for label in GROUPS], ("Unloaded sensor", noise)]
    rows = []
    for name, frame in sources:
        peaks = frame["tremor_peak_frequency_hz"].dropna()
        rows.append({"source": name, "finger_attempts": len(frame), "with_peak": len(peaks),
                     "peak_hz_q25": peaks.quantile(.25), "peak_hz_median": peaks.median(),
                     "peak_hz_q75": peaks.quantile(.75), "share_peaks_below_6hz": (peaks < 6).mean(),
                     "prominence_median": frame["tremor_peak_prominence"].median()})
    table = pd.DataFrame(rows)

    example = used.loc[(used["label"] == HEALTHY) & (used["patient_id"] == "992601") & (used["hand"] == "right")
                       & (used["finger_name"] == "index")].sort_values("attempt").iloc[0]
    data = load_measurement_data(resolve_project_path(example["file_path"]))
    finger_values = data.loc[(data["attempt"] == example["attempt"]) & (data["segment"] == 1), example["sensor"]]
    calibration = noise.iloc[0]
    calib = load_measurement_data(resolve_project_path(calibration["file_path"]))
    noise_values = calib.loc[(calib["attempt"] == calibration["attempt"]) & (calib["segment"] == 1),
                             calibration["sensor"]]

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axs[0]
    bins = np.arange(4, 15.01, .5)
    for name, frame in sources:
        color = COLORS.get(name, "#777")
        ax.hist(frame["tremor_peak_frequency_hz"].dropna(), bins=bins, density=True, histtype="step", lw=1.8,
                color=color, ls="-" if name in GROUPS else ":", label=f"{name}{' group fingers' if name in GROUPS else ''}")
    ax.set_xlabel("Frequency of the detected 'tremor peak' (Hz)")
    ax.set_ylabel("Share of finger-attempts (density)")
    ax.set_title("Detected peaks pile up at the lower band edge")
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    for values, label, color, style in [(finger_values, "Healthy participant, right index finger", COLORS[HEALTHY], "-"),
                                        (noise_values, "Unloaded sensor (calibration)", "#777", ":")]:
        f, spectrum = _mean_spectrum(values.to_numpy())
        keep = f > 0
        ax.plot(f[keep], spectrum[keep], color=color, ls=style, lw=1.5, label=label)
    ax.axvline(example["tremor_peak_frequency_hz"], color=COLORS[HEALTHY], lw=.8, ls="--")
    ax.text(example["tremor_peak_frequency_hz"], .93, f" detected peak {example['tremor_peak_frequency_hz']:.1f} Hz",
            transform=ax.get_xaxis_transform(), fontsize=8, color="#333")
    ax.axvspan(*config.TREMOR_BAND[:2], color="#f3e3dc", alpha=.5, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz); shaded = 4–15 Hz band")
    ax.set_ylabel("Oscillation strength (log scale)")
    ax.set_title("A smoothly falling spectrum, not a tremor rhythm")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1, .86))
    fig.tight_layout()
    savefig(fig, "08_peak_frequency_check")
    return {"peak_frequency_check": table,
            "peak_example": {"patient_id": example["patient_id"], "attempt": int(example["attempt"]),
                             "finger": "right index", "peak_hz": float(example["tremor_peak_frequency_hz"]),
                             "prominence": float(example["tremor_peak_prominence"])}}


# --------------------------------------------------------------------------- run
def run(write: bool = True) -> dict:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    inputs = load_inputs()
    results = {}
    results.update(age_analysis(inputs["people"]))
    results.update(weak_hand_analysis(inputs["activity"], inputs["hand_usage"]))
    results.update(weak_hand_example(inputs["attempts"], inputs["activity"]))
    results.update(noise_analysis(inputs["fingers"], inputs["noise"]))
    results.update(finger_force_analysis(inputs["fingers"]))
    results.update(group_by_finger(inputs["fingers"]))
    results.update(age_amplitude(inputs["fingers"]))
    results.update(tremor_patterns(inputs["fingers"]))
    results.update(peak_frequency_check(inputs["attempts"], inputs["noise"]))
    if write:
        scalars = {}
        for name, value in results.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(OUT / f"{name}.csv", index=False)
            else:
                scalars[name] = value
        (OUT / "summary.json").write_text(json.dumps(scalars, indent=2, default=float))
    return results


if __name__ == "__main__":
    for key, value in run().items():
        print(f"\n== {key}")
        print(value.to_string(max_rows=30) if isinstance(value, pd.DataFrame) else value)
