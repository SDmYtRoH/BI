# Week 2 — findings for the presentation

**Completed:** analysis of all 138 indexed people: 48 in the sick group and 90 recruited as controls. Of the 90 control records, 76 pass the questionnaire screen and are labelled the healthy group for the exploratory comparison. The data cover 173 Balanced Force sessions and 692 attempts. All selected recordings were found; no processing failures or divergent duplicate copies occurred.

## The proposed analysis

- Slow force variation: 0.2 to below 3 Hz.
- Ambiguous boundary: 3 to below 4 Hz, retained separately.
- Candidate tremor: 4 through 15 Hz, with 4–12 and 12–15 Hz sub-bands and alternative lower cutoffs.
- Extract band RMS amplitude, power-weighted frequency, spectral spread, peak frequency, across-window SD and usable-window counts for each person and hand.
- Calculate spectra per finger, then average powers to avoid phase cancellation. Aggregate repetitions and sessions within people.
- Flag low-use windows; missing evidence must not become zero tremor. The default screening flags 3 of 2,768 primary hand-windows, while every participant retains both hand summaries. These flags are heuristics, not confirmed non-use.

## Results worth explaining

| Result | Value / meaning |
|---|---|
| Median faster-oscillation amplitude, sick group | 0.000292 device units |
| Median faster-oscillation amplitude, healthy/control recruitment group | 0.000144 device units |
| Group-summary unit | One mean-of-hands value per participant; all 138 people |
| Primary comparison pair | Across available recordings, 992601 and UQIS-DXIH have similar mean hand RMS values |
| Previously selected illustrative case | ILHC-BZCP has a visible channel-10 peak around 4.0–4.4 Hz; this pair was selected for illustration |
| Amplitude-only held-out balanced accuracy | 80.2% |
| Age-only held-out balanced accuracy | 80.2% |
| Amplitude accuracy using only 8–13 s windows | 74.2% |
| Classification sample | 48 sick-group people and 76 questionnaire-screened healthy people |

## Explicit healthy and sick examples

The main comparison contains one **healthy participant** and one **sick participant**. The figure titles now use those plain-language labels; their IDs are kept only in this explanation:

- Healthy participant: ID `992601` (technical source label: reference cohort; no relevant condition reported in the questionnaire).
- Sick participant: ID `UQIS-DXIH` (technical source label: patient cohort; exact diagnosis is not available in the analysis table).

Their average 4–15 Hz amplitudes are similar, so we know their health group from the supplied labels rather than from reliable separation by that spectrogram.

The clearer teaching example is:

- Healthy participant: ID `992612`.
- Sick participant: ID `ILHC-BZCP`.

The sick participant in this second pair has a repeatable peak around 4.0–4.4 Hz. This pair was deliberately selected earlier for a visible difference, so it is an illustration rather than validation accuracy.

The distributions overlap. **The same 80.2% score from age alone is a central limitation**, reinforced by different scenario IDs and collection periods. The signal may contain useful information, but this result does not establish diagnosis or separation of involuntary tremor from intentional movement.

Early force corrections contribute to the faster band. Removing early windows lowers amplitude-rule performance to 74.2%. Band-pass filtering can also ring after an abrupt force change; a burst in the filtered trace is not automatically tremor.

## Five-minute presentation order

1. State the research question: describe force-control variation and candidate tremor separately, up to 15 Hz.
2. Show the decomposition, explaining the uncertain 3–4 Hz band and the response to force corrections.
3. Explain RMS, mean frequency and the two meanings of standard deviation (spectral spread versus variation across windows).
4. Show the 3/5/10 s spectra and shared-scale spectrograms. Use the strong case to explain a ridge, with its selection caveat.
5. Show the participant table and low-use check. Finish with the age/transient confounding result and the need for verified labels and comparable protocols.

## Files to open

- `REPORT.html`: complete report with embedded plots.
- `week2_tremor_analysis.ipynb`: executed notebook.
- `outputs/participant_features.csv`: all people, one row each.
- `GUIDE.md`: explanation, formulas and likely teacher questions.

The existing project notebooks were preserved. The local archive was already extracted with matching file sizes; no newer remote recording download was made. Google source documents were read, and no messages were sent or cloud files changed.
