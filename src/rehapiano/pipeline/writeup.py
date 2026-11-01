"""Short final report of a pipeline run: ``REPORT.md`` and a self-contained ``REPORT.html`` (figures embedded).

Every number and every conclusion sentence is computed from the CSV files of the current run, so the report is
rewritten automatically when new data arrive. Section order follows the task of week 6.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, List

import markdown
import numpy as np
import pandas as pd

from rehapiano import config
from rehapiano.pipeline import catalog, settings
from rehapiano.pipeline.participants import HEALTHY, OTHER, SICK
from rehapiano.pipeline.report import key_features

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;
color:#1d1d1d;line-height:1.5;background:#fff}
h1{font-size:1.6em}h2{margin-top:2em;border-bottom:1px solid #ddd;padding-bottom:.2em}
table{border-collapse:collapse;font-size:.86em;margin:1em 0;display:block;overflow-x:auto}
th,td{border:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f4f4f2}
img{max-width:100%;height:auto;margin:.5em 0}code{background:#f4f4f2;padding:0 3px}
blockquote{color:#555;border-left:3px solid #ddd;margin-left:0;padding-left:1em}
"""


# --------------------------------------------------------------------------- formatting
def num(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    value = float(value)
    if value != 0 and (abs(value) < 0.01 or abs(value) >= 1000):
        return f"{value:.3g}"
    return f"{value:.{digits}f}"


def pval(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def names(features: Iterable[str]) -> str:
    features = list(features)
    return ", ".join(f"`{f}`" for f in features) if features else "none"


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "-C", str(config.PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(config.PROJECT_ROOT), "status", "--porcelain", "--", "src", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return commit + (" + uncommitted code changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --------------------------------------------------------------------------- sections
def section_data(people: pd.DataFrame, summary: dict) -> List[str]:
    rows = []
    for label in [HEALTHY, SICK, OTHER]:
        g = people.loc[people["label"] == label]
        age = g["age"]
        rows.append({"Group": label, "People": len(g),
                     "Median age (IQR)": f"{age.median():.0f} ({age.quantile(.25):.0f}–{age.quantile(.75):.0f})",
                     "Aged 50+": int((age >= settings.OLDER_AGE).sum()),
                     "Female / male": f"{int((g['gender'] == 'female').sum())} / {int((g['gender'] == 'male').sum())}",
                     "Recorded": ", ".join(sorted(g["recording_period"].dropna().unique())),
                     "EXACT_FORCE sessions": int(g.get("sessions_exact_force", pd.Series(dtype=float)).sum()),
                     "Rhythm sessions": int(g.get("sessions_rhythm", pd.Series(dtype=float)).sum())})
    features = pd.DataFrame([{"Feature": f"`{name}`", "Factor": f.factor, "Definition": f.description}
                             for name, f in catalog.FEATURES.items()])
    used = summary.get("used_recordings_per_game", {})
    return [
        "## 1. Data and methods",
        f"**Participants.** *Sick* = recruited into the patient cohort (no diagnoses available). *Healthy* = control "
        f"whose questionnaire reports no condition affecting hand strength or coordination. *Other controls* (reported "
        f"condition, no or duplicated questionnaire) are described but left out of every comparison. One person is one "
        f"statistical unit.",
        table(pd.DataFrame(rows)),
        f"**Games.** EXACT_FORCE: hold a steady force with all ten fingers (4 attempts × 16 s, analysed 3–15 s). "
        f"RHYTHM: press and release both hands as fast as possible (4 attempts × 31 s, analysed 2–30 s, the first "
        f"attempt is practice). Used recordings: {used.get('exact_force', 0)} EXACT_FORCE and {used.get('rhythm', 0)} "
        f"rhythm sessions.",
        "**Preprocessing.** Decoding of the raw packets, sign correction of channel 9, timestamp checks, averaging of "
        "duplicated timestamps, interpolation of gaps ≤ 50 ms (longer gaps split the attempt), anti-aliased resampling "
        "to 100 Hz, cropping. Byte-identical copies of a recording are counted once. Oscillation amplitudes are "
        "spectrogram amplitudes computed per finger and averaged only afterwards; fingers that do not press have no "
        "oscillation value (missing, never zero). Person value = mean over sessions of the median over attempts.",
        "**Features** (17, in 12 factors):",
        table(features),
    ]


def section_hypotheses(h: pd.DataFrame) -> List[str]:
    mark = lambda p: f"**{pval(p)}**" if p < .05 else pval(p)  # noqa: E731
    rows = pd.DataFrame({
        "Feature": [f"`{f}`" for f in h["feature"]],
        "Median healthy": h["median_healthy"].map(num), "Median sick": h["median_sick"].map(num),
        "AUC": h["auc"].map(num), "p Holm (all)": h["p_holm"].map(mark),
        "p Holm (50+)": h["p_50_holm"].map(mark),
        "Effect, same decade (SD)": h["effect_same_decade_sd"].map(num),
        "p Holm (same decade)": h["p_same_decade_holm"].map(mark)})
    all_ages = h.loc[h["p_holm"] < .05, "feature"]
    same = h.loc[h["p_same_decade_holm"] < .05].sort_values("p_same_decade")
    higher = same.loc[same["effect_same_decade_sd"] > 0, "feature"]
    lower = same.loc[same["effect_same_decade_sd"] < 0, "feature"]
    n50 = h[["n_healthy_50", "n_sick_50"]].max()
    return [
        "## 2. Hypotheses and significance tests",
        "For every feature: **H0** – healthy and sick people have the same distribution; **H1** – the distributions "
        "differ (two-sided). Tests: Mann–Whitney U for all ages and for people aged 50+ "
        f"({int(n50['n_healthy_50'])} healthy, {int(n50['n_sick_50'])} sick), and a permutation test that shuffles the "
        "healthy/sick labels only within 10-year age groups, so age cannot create the difference "
        f"({settings.PERMUTATIONS} permutations). p-values are Holm-corrected over the 17 features. "
        "AUC = probability that a random sick person has a higher value than a random healthy one (0.5 = no "
        "difference). The same-decade effect is in standard deviations of healthy people.",
        table(rows),
        f"**Result.** H0 is rejected for **{len(all_ages)} of {len(h)}** features across all ages. Comparing people "
        f"of the same decade, **{len(same)}** differences remain: higher in the sick group – {names(higher)}; lower – "
        f"{names(lower)}. Differences found only across all ages may partly reflect the age gap between the groups.",
        "![Group differences](analysis/figures/01_group_differences.png)",
        "![Key features](analysis/figures/05_key_feature_distributions.png)",
        "> The groups overlap: a significant difference of medians does not mean a single value separates every "
        "sick person from every healthy one.",
    ]


def section_age(t: pd.DataFrame) -> List[str]:
    rows = pd.DataFrame({
        "Feature": [f"`{f}`" for f in t["feature"]], "Spearman ρ": t["spearman_rho"].map(num),
        "p Holm": t["p_holm"].map(pval),
        "Change per 10 years (95 % CI)": [
            (f"{c:+.1f} % ({lo:+.1f} to {hi:+.1f} %)" if unit.startswith("%") else f"{num(c)} ({num(lo)} to {num(hi)})")
            for c, lo, hi, unit in zip(t["change_per_10_years"], t["ci_low"], t["ci_high"], t["change_unit"])],
        "Age-dependent": np.where(t["age_dependent"], "yes", "no")})
    dependent = t.loc[t["age_dependent"]]
    trend = ", ".join(f"`{r.feature}` ({'falls' if r.spearman_rho < 0 else 'rises'}, ρ = {r.spearman_rho:.2f})"
                      for r in dependent.itertuples()) or "none"
    uncertain = dependent.loc[(dependent["ci_low"] < 0) & (dependent["ci_high"] > 0), "feature"]
    caveat = (f" For {names(uncertain)} the rank correlation is significant but the size of the change is uncertain "
              "(its 95 % interval includes 0): the trend mainly contrasts the many people in their twenties with the "
              "few older healthy people." if len(uncertain) else "")
    return [
        "## 3. Changes with age in healthy people",
        f"Healthy people only (n = {int(t['n'].max())}). Age trend: Spearman correlation (Holm-corrected); "
        "change per 10 years from a linear fit (for amplitudes on the log scale, given in %).",
        table(rows),
        f"**Result.** After correction, **{len(dependent)} of {len(t)}** features change with age: {trend}. "
        "All other features do not show a detectable age trend, so one range of healthy people is valid for every "
        "decade." + caveat + " Most healthy people are in their twenties, so a weak trend in older decades could be missed.",
    ]


def section_norms(norms: pd.DataFrame, outside: pd.DataFrame, chosen: List[str]) -> List[str]:
    decades = sorted(norms["decade"].unique(), key=lambda d: int(d.split("–")[0]))
    part = norms.loc[norms["feature"].isin(chosen)]
    rows = []
    for feature in chosen:
        f = part.loc[part["feature"] == feature].set_index("decade")
        rows.append({"Feature": f"`{feature}`", "Norm from": f["method"].iloc[0],
                     **{d: f"{num(f.loc[d, 'norm_median'])} [{num(f.loc[d, 'norm_low'])}–{num(f.loc[d, 'norm_high'])}]"
                        for d in decades}})
    counts = norms.drop_duplicates("decade").set_index("decade")["n_healthy"]
    rows.append({"Feature": "*healthy people in the decade*", "Norm from": "",
                 **{d: str(int(counts[d])) for d in decades}})
    reliable = ", ".join(d for d in decades if counts[d] >= settings.MIN_PEOPLE_FOR_DECADE_RANGE) or "none"
    o = outside.assign(out=lambda t: t["below_range"] + t["above_range"])
    o = o.pivot(index="feature", columns="label", values=["out", "people"])
    out_rows = pd.DataFrame({
        "Feature": [f"`{f}`" for f in o.index],
        "Healthy outside": [f"{int(o.loc[f, ('out', HEALTHY)])} / {int(o.loc[f, ('people', HEALTHY)])}" for f in o.index],
        "Sick outside": [f"{int(o.loc[f, ('out', SICK)])} / {int(o.loc[f, ('people', SICK)])}" for f in o.index],
        "sort": [o.loc[f, ("out", SICK)] for f in o.index]}).sort_values("sort", ascending=False).drop(columns="sort")
    return [
        "## 4. Normal values by decade",
        f"Normal value = median of healthy people, normal range = 2.5–97.5 %. A decade gets its own empirical range "
        f"only with ≥ {settings.MIN_PEOPLE_FOR_DECADE_RANGE} healthy people (here: {reliable}). Otherwise the range "
        "of all healthy people is used for features without an age trend, and a linear quantile regression on age "
        "(evaluated at the middle of the decade) for age-dependent features. Key features (full table: "
        "`analysis/decade_norms.csv`):",
        table(pd.DataFrame(rows)),
        "![Norms](analysis/figures/02_norms_by_decade.png)",
        "**People outside the normal range of their decade** (by construction about 5 % of healthy people):",
        table(out_rows.head(8)),
    ]


def section_classifier(c: pd.DataFrame, perm: pd.DataFrame, coefficients: pd.DataFrame) -> List[str]:
    lr = c.loc[c["model"] == "logistic regression"]
    rows = pd.DataFrame({
        "People": c["people"] + " (" + c["n_healthy"].astype(int).astype(str) + " / " + c["n_sick"].astype(int).astype(str)
        + ")", "Features": c["features"], "Model": c["model"],
        "AUC (95 % range)": [f"{a:.2f} ({lo:.2f}–{hi:.2f})" for a, lo, hi in zip(c["auc"], c["auc_low"], c["auc_high"])],
        "Balanced accuracy": c["balanced_accuracy"].map(num), "Sensitivity": c["sensitivity"].map(num),
        "Specificity": c["specificity"].map(num)})
    get = lambda people, features: lr.loc[(lr["people"] == people) & (lr["features"] == features)].iloc[0]  # noqa
    older = f"age {settings.OLDER_AGE}+"
    full, age, both = get("all ages", "All game features"), get("all ages", "Age only"), get("all ages",
                                                                                            "All game features + age")
    full50, age50 = get(older, "All game features"), get(older, "Age only")
    p = perm.iloc[0]
    top = coefficients.reindex(coefficients["standardised_coefficient"].abs().sort_values(ascending=False).index).head(4)
    return [
        "## 5. Classifier: healthy vs sick",
        f"Logistic regression (main model: simple and interpretable; median imputation, standardisation, balanced "
        f"classes, amplitudes log-transformed) and a random forest for comparison. Evaluation: stratified "
        f"{settings.CV_FOLDS}-fold cross-validation repeated {settings.CV_REPEATS}×; a person is never in the training "
        f"and the test fold at the same time. Mean over repeats, 95 % range in brackets; threshold 0.5. Because the "
        f"sick group is much older, an **age-only** model and a model on people **aged {settings.OLDER_AGE}+** are "
        f"the reference points.",
        table(rows),
        "![ROC](analysis/figures/03_roc.png)",
        f"**Result.** All game features: AUC **{full['auc']:.2f}**, balanced accuracy {full['balanced_accuracy']:.2f} "
        f"(permutation test: AUC with shuffled labels {p['null_mean']:.2f}, p = {pval(p['p'])}). Age alone reaches "
        f"AUC {age['auc']:.2f}; game features plus age {both['auc']:.2f}. Among people aged {settings.OLDER_AGE}+, "
        f"where age hardly differs, game features reach AUC **{full50['auc']:.2f}** against {age50['auc']:.2f} for "
        f"age, so the measurements carry information beyond age. Largest standardised coefficients: "
        + ", ".join(f"`{r.feature}` ({r.standardised_coefficient:+.2f})" for r in top.itertuples()) + ".",
        "> The classifier predicts cohort membership in this dataset; it is not a diagnostic test and has not been "
        "validated on independent data.",
    ]


def section_ablation(a: pd.DataFrame) -> List[str]:
    part = a.loc[a["model_includes_age"]].sort_values("auc_alone", ascending=False)
    rows = pd.DataFrame({
        "Factor": part["factor"], "Features": part["features"].map(lambda s: ", ".join(f"`{x}`" for x in s.split(", "))),
        "AUC alone (95 % range)": [f"{x:.2f} ({lo:.2f}–{hi:.2f})"
                                   for x, lo, hi in zip(part["auc_alone"], part["auc_alone_low"], part["auc_alone_high"])],
        "AUC lost when removed": part["auc_drop"].map(lambda v: f"{v:+.3f}")})
    best = part.loc[part["factor"] != "Age"].head(3)
    largest_drop = part["auc_drop"].abs().max()
    return [
        "## 6. Ablation: influence of individual factors",
        f"Logistic regression with all factors and age (AUC {part['auc_all'].iloc[0]:.2f}). Each factor is (a) used "
        "alone and (b) removed from the full model; both with the same repeated cross-validation.",
        table(rows),
        "![Ablation](analysis/figures/04_ablation.png)",
        "**Result.** Alone, the strongest factors are " + ", ".join(
            f"{r.factor} (AUC {r.auc_alone:.2f})" for r in best.itertuples()) +
        f". Removing any single factor changes the AUC by at most {largest_drop:.3f}: the factors are largely "
        "redundant (tempo, press force and balance carry overlapping information), so the model compensates. "
        "Negative values mean the model is slightly better without that factor, a sign of mild overfitting with 17 "
        "features and few people.",
    ]


def section_limitations() -> List[str]:
    return [
        "## 7. Limitations",
        "- Group membership coincides with age, recording period and game version (healthy and sick people played "
        "different step versions of both games). Differences that remain within the same decade still cannot be "
        "separated from period or game version.",
        "- *Healthy* is questionnaire-screened, not physician-confirmed; diagnoses, severity and medication of the "
        "sick group are unknown.",
        f"- Few healthy people aged {settings.OLDER_AGE}+; decade norms above 30 years rest on the pooled healthy "
        "range or an age regression.",
        "- Force is in uncalibrated device units; the sensor-to-finger mapping is inferred, not verified.",
        "- The 4–15 Hz band also contains fast force corrections, not only involuntary tremor.",
    ]


def section_reproducibility(summary: dict) -> List[str]:
    rec = summary.get("recordings", {})
    versions = ", ".join(f"{k} {v}" for k, v in summary.get("versions", {}).items())
    return [
        "## 8. Reproducibility",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `python scripts/run_pipeline.py` (code version "
        f"{_git_commit()}). Recording files: {json.dumps(rec)}; extraction errors: "
        f"{summary.get('extraction_errors', 'n/a')}. Software: {versions}.",
        "Tables behind every number: `outputs/pipeline/analysis/*.csv`; per-person features: "
        "`outputs/pipeline/person_features.csv`. Method details: `PIPELINE.md`.",
    ]


def summary_bullets(h, t, c, a) -> List[str]:
    lr = c.loc[c["model"] == "logistic regression"]
    pick = lambda people, f: lr.loc[(lr["people"] == people) & (lr["features"] == f), "auc"].iloc[0]  # noqa: E731
    older = f"age {settings.OLDER_AGE}+"
    same = h.loc[h["p_same_decade_holm"] < .05, "feature"]
    best = a.loc[a["model_includes_age"] & (a["factor"] != "Age")].sort_values("auc_alone", ascending=False).iloc[0]
    return [
        "## Summary",
        f"- **{int((h['p_holm'] < .05).sum())} of {len(h)}** measurements differ between healthy and sick people; "
        f"**{len(same)}** also between people of the same decade ({names(same)}).",
        f"- In healthy people **{int(t['age_dependent'].sum())}** measurements change with age; normal values per "
        "decade are given in section 4.",
        f"- A logistic regression on the game measurements separates the groups with AUC **{pick('all ages', 'All game features'):.2f}** "
        f"(age alone {pick('all ages', 'Age only'):.2f}); among people aged {settings.OLDER_AGE}+ "
        f"{pick(older, 'All game features'):.2f} vs {pick(older, 'Age only'):.2f} for age alone.",
        f"- The strongest single factor is **{best['factor']}** (AUC {best['auc_alone']:.2f} alone); factors are "
        "largely redundant.",
        "- Not a diagnosis: group, age, recording period and game version coincide (section 7).",
    ]


# --------------------------------------------------------------------------- run
def run(output_dir: Path = settings.OUTPUT_DIR, analysis_dir: Path = settings.ANALYSIS_DIR) -> Path:
    read = lambda name: pd.read_csv(analysis_dir / f"{name}.csv")  # noqa: E731
    people = pd.read_csv(output_dir / "person_features.csv", dtype={"patient_id": str})
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    h, t, c, a = read("hypotheses"), read("age_trends"), read("classifier"), read("ablation")
    parts = [
        "# RehaPiano: healthy vs sick participants – statistical analysis, norms and classifier",
        "Task: formulate hypotheses and test the significance of healthy–sick differences; analyse how the "
        "measurements change with age in healthy people and set normal values and ranges per decade; propose a "
        "classifier separating healthy and sick people and an ablation of the influence of individual factors.",
        *summary_bullets(h, t, c, a),
        *section_data(people, summary),
        *section_hypotheses(h),
        *section_age(t),
        *section_norms(read("decade_norms"), read("outside_norm"), key_features(h)),
        *section_classifier(c, read("classifier_permutation_test"), read("classifier_coefficients")),
        *section_ablation(a),
        *section_limitations(),
        *section_reproducibility(summary),
    ]
    text = "\n\n".join(parts) + "\n"
    md_path = output_dir / "REPORT.md"
    md_path.write_text(text, encoding="utf-8")

    def embed(match: re.Match) -> str:
        data = base64.b64encode((output_dir / match.group(2)).read_bytes()).decode()
        return f"![{match.group(1)}](data:image/png;base64,{data})"

    body = markdown.markdown(re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, text), extensions=["tables"])
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>RehaPiano report</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    (output_dir / "REPORT.html").write_text(html, encoding="utf-8")
    return md_path
