"""Week 5 (task sheet column G), short version: hand balance in the rhythm test vs EXACT_FORCE by 10-year age group.

Task: evaluate the balance of hand forces also for the rhythm test, compare it with EXACT_FORCE, analyse it
by age categories and separate the effect of age from the effect of the illness.

Balance of one person = weaker hand / stronger hand (1 = both equal, 0 = one hand does nothing):

* EXACT_FORCE: mean finger force of each hand (week-3 pipeline, ``outputs/stats/hand_activity.csv``);
* RHYTHM: force range of each hand during the attempt (95th - 5th percentile of the hand's mean finger
  force, 2-30 s). It needs no press detection, so it also works when a person alternates hands.
  The press-based measure (median press amplitude, presses found in the summed force) is kept as a check.

Age is grouped in 10-year bins. Healthy = 76 questionnaire-screened controls (week 2); sick = 48 patients.
Run from this folder:  python week5_analysis.py   (~2 min the first time, then cached)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from rehapiano.io.metadata import read_metadata, resolve_project_path  # noqa: E402
from rehapiano.preprocessing.pipeline import load_measurement_data  # noqa: E402
from rehapiano.stats.tests import holm_adjust, mann_whitney  # noqa: E402

PIPELINE = REPO / "outputs"
DATASETS = REPO / "data" / "RehaPiano" / "datasets"
LABELS_CSV = REPO / "notebooks" / "week2" / "outputs" / "participant_labels.csv"
OUT = HERE / "outputs"
FIG = HERE / "figures"
ATTEMPTS_CACHE = OUT / "rhythm_attempts.csv"

HEALTHY, SICK, OTHER = "Healthy", "Sick", "Other control"
GROUPS = [HEALTHY, SICK]
COLORS = {HEALTHY: "#2377a4", SICK: "#cf5c36"}
TESTS = {"exact_force_ratio": "EXACT_FORCE (hold a steady force)", "rhythm_ratio": "Rhythm test (press repeatedly)"}

RHYTHM_STEP_IDS = ("1e177cdd-0baf-42fe-b297-fbeba73ea828",  # controls
                   "65a5a4b1-907c-4717-b47b-cd8f31eb3e4b")  # patients
CROP_S = (2.0, 30.0)  # skip the start; attempts last ~31 s
MIN_ATTEMPT_S = 20.0
PRESS_MIN_DISTANCE_S = 0.15  # faster than ~6.7 presses/s is not a separate press
PRESS_MIN_PROMINENCE = 0.5  # of the attempt's 5-95 % range of the summed force
DECADE = 10
WEAK_QUANTILE = 0.025
MIN_GROUP_N = 3
PERMUTATIONS = 10000
SEED = 0

# Maksym's notebook (notebooks/week5/rovnovaha_ruk.ipynb), all 90 controls, press-amplitude measure.
MAKSYM = {"rhythm median, healthy": 0.87, "rhythm median, sick": 0.70, "EXACT_FORCE median, healthy": 0.94,
          "EXACT_FORCE median, sick": 0.89, "weak hand in rhythm, sick": 18, "weak hand in rhythm, healthy": 3,
          "weak hand in EXACT_FORCE, sick": 6, "weak hand in EXACT_FORCE, healthy": 3}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                     "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 11})


# --------------------------------------------------------------------------- rhythm recordings
def rhythm_index() -> pd.DataFrame:
    """One file per (measurement, rhythm step); byte-identical copies are dropped, conflicting copies excluded."""

    inventory = pd.read_csv(PIPELINE / "recordings_inventory.csv", dtype=str)
    files = inventory.loc[inventory["step_id"].isin(RHYTHM_STEP_IDS)].sort_values("file_path")
    rows = []
    for (uuid, step), group in files.groupby(["file_uuid", "step_id"]):
        digests = {hashlib.md5(resolve_project_path(p).read_bytes()).hexdigest() for p in group["file_path"]}
        rows.append({"measurement_id": uuid, "step_id": step, "file_path": group["file_path"].iloc[0],
                     "copies": len(group), "conflicting_copies": len(digests) > 1})
    index = pd.DataFrame(rows)

    metadata = pd.concat([read_metadata(DATASETS / name) for name in ("dataset_full.csv", "dataset_patients.csv")])
    metadata = metadata.loc[metadata["step_type"] == "RHYTHM",
                            ["measurement_id", "step_id", "patient_id", "timestamp"]].drop_duplicates()
    return index.merge(metadata, on=["measurement_id", "step_id"], how="inner", validate="1:1")


def hand_forces(frame: pd.DataFrame) -> dict:
    return {hand: frame[[f"{hand}_sensor_{k}" for k in range(1, 6)]].mean(axis=1).to_numpy()
            for hand in ("left", "right")}


def find_presses(total: np.ndarray, fs: float = 100.0) -> np.ndarray:
    spread = np.percentile(total, 95) - np.percentile(total, 5)
    peaks, _ = find_peaks(total, distance=int(PRESS_MIN_DISTANCE_S * fs), prominence=PRESS_MIN_PROMINENCE * spread)
    return peaks


def _ratio(a: float, b: float) -> float:
    high = max(a, b)
    return min(a, b) / high if high > 0 else np.nan


def attempt_features(frame: pd.DataFrame) -> dict:
    """Balance and tempo of one attempt (first continuous segment, 2-30 s)."""

    force = hand_forces(frame)
    span = {hand: np.percentile(v, 95) - np.percentile(v, 5) for hand, v in force.items()}
    total = force["left"] + force["right"]
    peaks = find_presses(total)
    duration = float(frame["time_s"].iloc[-1] - frame["time_s"].iloc[0])
    bounds = [0, *[int(np.argmin(total[a:b])) + a for a, b in zip(peaks[:-1], peaks[1:])], len(total) - 1]
    amplitude = {hand: np.median([np.ptp(v[bounds[i]:bounds[i + 1] + 1]) for i in range(len(peaks))])
                 if len(peaks) else np.nan for hand, v in force.items()}
    return {"duration_s": duration, "presses": len(peaks), "press_rate_hz": len(peaks) / duration,
            "left_range": span["left"], "right_range": span["right"],
            "range_ratio": _ratio(span["left"], span["right"]),
            "weaker_hand": "left" if span["left"] < span["right"] else "right",
            "press_ratio": _ratio(amplitude["left"], amplitude["right"]) if len(peaks) >= 5 else np.nan}


def rhythm_attempts(recompute: bool = False) -> pd.DataFrame:
    if ATTEMPTS_CACHE.exists() and not recompute:
        return pd.read_csv(ATTEMPTS_CACHE, dtype={"patient_id": str})
    rows = []
    for record in rhythm_index().itertuples():
        data = load_measurement_data(resolve_project_path(record.file_path), *CROP_S)
        for attempt, frame in data.loc[data["segment"] == 1].groupby("attempt"):
            row = {"patient_id": record.patient_id, "measurement_id": record.measurement_id,
                   "timestamp": record.timestamp, "file_path": record.file_path, "copies": record.copies,
                   "conflicting_copies": record.conflicting_copies, "attempt": int(attempt)}
            row.update(attempt_features(frame))
            rows.append(row)
    attempts = pd.DataFrame(rows)
    attempts["valid"] = (attempts["duration_s"] >= MIN_ATTEMPT_S) & ~attempts["conflicting_copies"]
    OUT.mkdir(exist_ok=True)
    attempts.to_csv(ATTEMPTS_CACHE, index=False)
    return attempts


# --------------------------------------------------------------------------- people
def decade_label(age: float) -> str:
    start = int(age // DECADE * DECADE)
    return f"{start}–{start + DECADE - 1}"


def people_table(attempts: pd.DataFrame) -> pd.DataFrame:
    """One row per person: rhythm balance (median over attempts, mean over sessions) and EXACT_FORCE balance."""

    valid = attempts.loc[attempts["valid"]]
    per_session = valid.groupby(["patient_id", "measurement_id"]).agg(
        rhythm_ratio=("range_ratio", "median"), rhythm_press_ratio=("press_ratio", "median"),
        press_rate_hz=("press_rate_hz", "median"), attempts=("attempt", "size"))
    per_person = per_session.groupby("patient_id").agg(
        rhythm_ratio=("rhythm_ratio", "mean"), rhythm_press_ratio=("rhythm_press_ratio", "mean"),
        press_rate_hz=("press_rate_hz", "mean"), rhythm_sessions=("attempts", "size"),
        rhythm_attempts=("attempts", "sum")).reset_index()
    weaker = valid.groupby("patient_id")["weaker_hand"].agg(lambda v: v.mode().iloc[0]).rename("rhythm_weaker_hand")

    labels = pd.read_csv(LABELS_CSV, dtype={"patient_id": str})
    labels["label"] = np.select([labels["cohort"].eq("patient"), labels["label_status"].eq("no_reported_condition")],
                                [SICK, HEALTHY], OTHER)
    exact = pd.read_csv(PIPELINE / "stats" / "hand_activity.csv", dtype={"patient_id": str})
    exact = exact[["patient_id", "force_ratio", "force_weaker_hand"]].rename(
        columns={"force_ratio": "exact_force_ratio", "force_weaker_hand": "exact_force_weaker_hand"})
    people = (labels[["patient_id", "label", "age", "gender"]]
              .merge(exact, on="patient_id", how="left").merge(per_person, on="patient_id", how="left")
              .merge(weaker.reset_index(), on="patient_id", how="left"))
    people["age"] = people["age"].astype(float)
    people["decade"] = people["age"].map(decade_label)
    return people


# --------------------------------------------------------------------------- analysis
def thresholds(people: pd.DataFrame, reference: pd.Series) -> dict:
    return {test: float(people.loc[reference, test].quantile(WEAK_QUANTILE)) for test in TESTS}


def decade_summary(people: pd.DataFrame, limits: dict) -> pd.DataFrame:
    rows = []
    for test in TESTS:
        for decade, frame in people.loc[people["label"].isin(GROUPS)].groupby("decade"):
            row = {"test": test, "decade": decade}
            for label in GROUPS:
                values = frame.loc[frame["label"] == label, test].dropna()
                row.update({f"{label} n": len(values), f"{label} median": values.median(),
                            f"{label} below limit": int((values < limits[test]).sum())})
            healthy, sick = (frame.loc[frame["label"] == g, test].dropna() for g in GROUPS)
            row["p"] = (mann_whitney(healthy, sick)["p"] if min(len(healthy), len(sick)) >= MIN_GROUP_N else np.nan)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["p (Holm)"] = table.groupby("test")["p"].transform(lambda p: holm_adjust(p))
    return table


def stratified_disease_effect(people: pd.DataFrame) -> pd.DataFrame:
    """Sick minus healthy balance within the same 10-year group, pooled; labels permuted only within decades.

    Only decades containing both groups count. The pooled effect weights every decade by n_h * n_s / (n_h + n_s),
    so a decade with 2 sick people cannot dominate. Shuffling labels inside each decade keeps the age structure
    fixed, so the p-value answers: is the illness associated with balance among people of the same decade?
    """

    rng = np.random.default_rng(SEED)
    rows = []
    for test in TESTS:
        frame = people.loc[people["label"].isin(GROUPS)].dropna(subset=[test])
        strata = [g for _, g in frame.groupby("decade") if g["label"].nunique() == 2]
        values = [g[test].to_numpy() for g in strata]
        sick = [g["label"].eq(SICK).to_numpy() for g in strata]
        weights = np.array([s.sum() * (~s).sum() / len(s) for s in sick])

        def effect(masks):
            return float(np.sum(weights * [v[m].mean() - v[~m].mean() for v, m in zip(values, masks)]) / weights.sum())

        observed = effect(sick)
        null = np.array([effect([rng.permutation(m) for m in sick]) for _ in range(PERMUTATIONS)])
        rows.append({"test": test, "decades_used": ", ".join(g["decade"].iloc[0] for g in strata),
                     "healthy_n": int(sum((~m).sum() for m in sick)), "sick_n": int(sum(m.sum() for m in sick)),
                     "sick_minus_healthy": observed,
                     "p_permutation": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (PERMUTATIONS + 1))})
    return pd.DataFrame(rows)


def age_effect_in_healthy(people: pd.DataFrame) -> pd.DataFrame:
    healthy = people.loc[people["label"] == HEALTHY]
    rows = []
    for test in TESTS:
        frame = healthy.dropna(subset=[test])
        rho, p = stats.spearmanr(frame["age"], frame[test])
        young, old = frame.loc[frame["age"] < 50, test], frame.loc[frame["age"] >= 50, test]
        rows.append({"test": test, "people": len(frame), "spearman_rho_age": rho, "p": p,
                     "median_under_50": young.median(), "median_50_plus": old.median(),
                     "p_under_50_vs_50_plus": mann_whitney(young, old)["p"]})
    return pd.DataFrame(rows)


def maksym_comparison(people: pd.DataFrame) -> pd.DataFrame:
    """Same summaries with all 90 controls as reference, as in Maksym's notebook."""

    controls = people["label"].isin([HEALTHY, OTHER])
    limits = thresholds(people, controls)
    sick = people["label"] == SICK
    mine = {"rhythm median, healthy": people.loc[controls, "rhythm_ratio"].median(),
            "rhythm median, sick": people.loc[sick, "rhythm_ratio"].median(),
            "EXACT_FORCE median, healthy": people.loc[controls, "exact_force_ratio"].median(),
            "EXACT_FORCE median, sick": people.loc[sick, "exact_force_ratio"].median(),
            "weak hand in rhythm, sick": int((people.loc[sick, "rhythm_ratio"] < limits["rhythm_ratio"]).sum()),
            "weak hand in rhythm, healthy": int((people.loc[controls, "rhythm_ratio"] < limits["rhythm_ratio"]).sum()),
            "weak hand in EXACT_FORCE, sick": int((people.loc[sick, "exact_force_ratio"]
                                                   < limits["exact_force_ratio"]).sum()),
            "weak hand in EXACT_FORCE, healthy": int((people.loc[controls, "exact_force_ratio"]
                                                      < limits["exact_force_ratio"]).sum())}
    return pd.DataFrame({"quantity": list(MAKSYM), "Maksym (press amplitude)": list(MAKSYM.values()),
                         "this version (force range)": [mine[k] for k in MAKSYM]})


# --------------------------------------------------------------------------- figures
def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def example_figure(attempts: pd.DataFrame, people: pd.DataFrame) -> None:
    examples = [("1902202505", "Healthy participant: both hands press together"),
                ("2102202501", "Healthy participant: hands alternate"),
                ("JMPV-JTBP", "Sick participant: only one hand presses")]
    fig, axs = plt.subplots(len(examples), 1, figsize=(11, 7.4), sharex=True)
    for ax, (pid, title) in zip(axs, examples):
        record = attempts.loc[(attempts["patient_id"] == pid) & attempts["valid"]].sort_values(
            ["timestamp", "attempt"]).iloc[0]
        data = load_measurement_data(resolve_project_path(record["file_path"]), *CROP_S)
        frame = data.loc[(data["attempt"] == record["attempt"]) & (data["segment"] == 1)]
        label = people.set_index("patient_id").loc[pid, "label"]
        for hand, style in [("left", "-"), ("right", "--")]:
            ax.plot(frame["time_s"], hand_forces(frame)[hand], ls=style, lw=1.2, color=COLORS[label],
                    label=f"{hand.capitalize()} hand")
        ax.set_xlim(2, 12)
        ax.set_title(f"{title} – balance in this attempt {record['range_ratio']:.2f}", fontsize=10)
        ax.set_ylabel("Hand force\n(device units)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axs[-1].set_xlabel("Time from the start of the attempt (s)")
    fig.tight_layout()
    savefig(fig, "01_rhythm_examples")


def decade_figure(people: pd.DataFrame, limits: dict) -> None:
    compared = people.loc[people["label"].isin(GROUPS)]
    decades = sorted(compared["decade"].unique(), key=lambda d: int(d.split("–")[0]))
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    rng = np.random.default_rng(1)
    for ax, (test, title) in zip(axs, TESTS.items()):
        for i, decade in enumerate(decades):
            for offset, label in zip((-.2, .2), GROUPS):
                values = compared.loc[(compared["decade"] == decade) & (compared["label"] == label), test].dropna()
                ax.scatter(i + offset + rng.uniform(-.07, .07, len(values)), values, s=20, color=COLORS[label],
                           alpha=.7, edgecolor="white", linewidth=.5,
                           label=f"{label} group" if i == 0 or (decade == decades[1] and label == SICK) else None)
                if len(values):
                    ax.hlines(values.median(), i + offset - .14, i + offset + .14, color="#222", lw=2)
                ax.text(i + offset, -.1, f"{len(values)}", ha="center", fontsize=7.5, color="#555")
        ax.axhline(limits[test], color="#555", ls="--", lw=1)
        ax.text(-.45, limits[test] + .02, f"healthy lower limit {limits[test]:.2f}", fontsize=8, color="#333")
        ax.set_ylim(-.16, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Balance\n(weaker ÷ stronger hand)")
        ax.grid(axis="x", visible=False)
    handles, names = axs[0].get_legend_handles_labels()
    keep = dict(zip(names, handles))
    fig.legend(keep.values(), keep.keys(), frameon=False, fontsize=9, loc="upper center", ncol=2,
               bbox_to_anchor=(.5, 1.03))
    axs[-1].set_xticks(range(len(decades)), decades)
    axs[-1].set_xlabel("Age group (years); small numbers = people in the group, bar = median")
    fig.tight_layout()
    savefig(fig, "02_balance_by_decade")


def weak_share_figure(summary: pd.DataFrame) -> None:
    decades = sorted(summary["decade"].unique(), key=lambda d: int(d.split("–")[0]))
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    x = np.arange(len(decades))
    for ax, (test, title) in zip(axs, TESTS.items()):
        part = summary.loc[summary["test"] == test].set_index("decade").reindex(decades)
        for offset, label in zip((-.2, .2), GROUPS):
            n, k = part[f"{label} n"].fillna(0), part[f"{label} below limit"].fillna(0)
            share = np.where(n > 0, 100 * k / n.where(n > 0, 1), np.nan)
            bars = ax.bar(x + offset, np.nan_to_num(share), width=.38, color=COLORS[label], edgecolor="white",
                          label=f"{label} group")
            ax.bar_label(bars, labels=[f"{int(a)}/{int(b)}" if b else "" for a, b in zip(k, n)], fontsize=7,
                         padding=2, color="#333")
        ax.set_xticks(x, decades)
        ax.set_title(title)
        ax.set_xlabel("Age group (years)")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("People below the healthy\nlower limit (%)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "03_weak_hand_share_by_decade")


# --------------------------------------------------------------------------- run
def run(recompute: bool = False, write: bool = True) -> dict:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    attempts = rhythm_attempts(recompute)
    people = people_table(attempts)
    limits = thresholds(people, people["label"] == HEALTHY)
    summary = decade_summary(people, limits)
    results = {
        "people": people, "limits": limits, "decade_summary": summary,
        "age_effect_healthy": age_effect_in_healthy(people),
        "stratified_disease_effect": stratified_disease_effect(people),
        "maksym_comparison": maksym_comparison(people),
        "checks": {
            "rhythm_sessions": int(attempts[["patient_id", "measurement_id"]].drop_duplicates().shape[0]),
            "rhythm_attempts": len(attempts), "valid_attempts": int(attempts["valid"].sum()),
            "files_with_copies": int((attempts.drop_duplicates("measurement_id")["copies"] > 1).sum()),
            "conflicting_copies": int(attempts.drop_duplicates("measurement_id")["conflicting_copies"].sum()),
            "people_with_rhythm": int(people["rhythm_ratio"].notna().sum()),
            "rho_range_vs_press": float(stats.spearmanr(people["rhythm_ratio"], people["rhythm_press_ratio"],
                                                        nan_policy="omit")[0]),
            "same_weaker_hand": float((people["rhythm_weaker_hand"] == people["exact_force_weaker_hand"])
                                      [people["label"] == SICK].mean()),
            "rho_tests_sick": float(stats.spearmanr(*people.loc[people["label"] == SICK,
                                                                ["exact_force_ratio", "rhythm_ratio"]].T.to_numpy())[0]),
            "press_rate_median": people.groupby("label")["press_rate_hz"].median().to_dict()},
    }
    example_figure(attempts, people)
    decade_figure(people, limits)
    weak_share_figure(summary)
    if write:
        for name, value in results.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(OUT / f"{name}.csv", index=False)
        (OUT / "summary.json").write_text(json.dumps({"limits": limits, "checks": results["checks"]}, indent=2,
                                                     default=float))
    return results


if __name__ == "__main__":
    for key, value in run(recompute="--recompute" in sys.argv).items():
        print(f"\n== {key}")
        print(value.to_string(max_rows=40) if isinstance(value, pd.DataFrame) else value)
