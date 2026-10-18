# Understanding the RehaPiano analysis

## 1. What are we measuring?

Each finger produces a force signal: a sequence of numbers over time. During Balanced Force, the participant tries to control that force. The recording can contain a slowly changing force level, deliberate corrections, physiological or pathological oscillations, sensor noise and brief disturbances.

A useful working model is:

`measured force = slow force control + oscillations + other variation/noise`

The terms overlap. A mathematical filter separates frequencies; it cannot read intention. We therefore say **candidate tremor**, not confirmed involuntary tremor. A sick participant can have no measurable tremor, and a healthy participant can have normal physiological tremor. The original RehaPiano study supports using finger-force measurements for action-tremor analysis, but does not validate these new groups or thresholds. [Original study](https://www.mdpi.com/1424-8220/20/3/663).

## 2. What does Fourier analysis add?

A time plot answers “when did the force change?” A spectrum answers “at which repetition rates does the force vary?” An oscillation repeated six times per second produces a peak near 6 Hz. A gradual change appears at low frequencies. A sharp correction can spread energy across many frequencies, including the candidate-tremor band.

Our working bands are:

| Band | Working interpretation | Important qualification |
|---|---|---|
| 0.2 to below 3 Hz | Slow force variation | Includes drift and other slow movement, not only game control |
| 3 to below 4 Hz | Uncertain boundary | Kept visible instead of forcing it into one interpretation |
| 4 through 15 Hz | Candidate tremor | May include physiological tremor, correction transients and noise |
| 4 to below 12 Hz; 12 through 15 Hz | Sub-bands | Help show where the faster signal is concentrated |

The lower limits 3, 3.5 and 4 Hz are compared in the sensitivity tables. These are analysis choices, not clinical cutoffs. In an isometric-force experiment, low-frequency force fluctuations below 3 Hz were associated with more than one mechanism, illustrating why frequency alone does not establish cause. [Force-control experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC3202062/).

## 3. Why 3, 5 and 10 seconds?

The frequency-bin spacing is `1 / window duration`:

| Window | Bin spacing | Trade-off |
|---|---:|---|
| 3 seconds | 0.333 Hz | Better localization of short episodes; fewer cycles |
| 5 seconds | 0.200 Hz | Primary compromise; two complete windows per repetition |
| 10 seconds | 0.100 Hz | Finer frequency bins; averages changes over longer time |

These are bin spacings, not the exact resolving power. A Hann window broadens spectral peaks while reducing leakage. Zero padding would draw more spectral points but would not add information, so we do not use it.

The cohort uses complete non-overlapping windows: 3–6, 6–9, 9–12 and 12–15 seconds for 3 s; 3–8 and 8–13 for 5 s; 3–13 for 10 s. The demonstration compares windows sharing the same 3 s start. Consequently, the cohort window-length sensitivity also changes how time is averaged; it is not a pure resolution experiment.

## 4. Why RMS instead of averaging FFT heights?

The **power spectral density** (PSD) describes how signal power is distributed over frequency. Its units are device-unit squared per Hz. Summing `PSD × frequency-bin spacing` across a band estimates that band's power. Its square root is the band RMS amplitude, in the original device units.

For example, a clean sinusoid with peak amplitude 0.002 has RMS amplitude approximately `0.002 / sqrt(2) = 0.001414`. The automated checks verify this recovery.

A mean of FFT-bin amplitudes depends strongly on the number and width of bins. Band RMS is more suitable for comparing windows of different lengths. It is not the same as peak-to-peak amplitude. Since the mean and linear trend are removed before each PSD, RMS describes fluctuations, not the average force being held. [SciPy PSD definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html).

## 5. Which frequency and standard deviation?

There are several different quantities:

- **Peak frequency:** the frequency bin with the highest PSD in 4–15 Hz. Every nonzero noisy spectrum has a highest bin; that alone is not evidence of tremor.
- **Power-weighted mean frequency (centroid):** `sum(frequency × PSD) / sum(PSD)` within the band. Stronger components count more.
- **Spectral spread:** the power-weighted standard deviation of frequencies around the centroid. This measures how broad the spectrum is.
- **Window standard deviation:** variation of the RMS, centroid or peak across the person's valid windows. This measures change across observations. It is not uncertainty in a population estimate.

The feature table deliberately keeps these concepts separate. A near-zero signal has no meaningful dominant frequency; it must not be assigned 0 Hz as though a real oscillation were measured.

## 6. Why not average the fingers first?

Imagine two fingers producing the same 6 Hz oscillation with opposite signs. Their average waveform can be almost flat even though both oscillate strongly. Instead, calculate PSD for each eligible finger, then average those PSDs within a hand. This preserves power even when phases differ.

The resulting hand RMS is the square root of the mean finger band powers. It describes an average eligible finger, not the total mechanical hand force. Left and right hands remain separate. Individual-finger results are also saved.

## 7. How do we handle a hand that is barely used?

For each window we inspect finite values, median absolute force, the fraction of samples above a small force floor, flatness, clipping and the force ratio between hands. The default absolute floor is 0.002 relative units; the hand-to-stronger-hand ratio must be at least 0.10. These are transparent heuristics because a verified unloaded-sensor baseline is unavailable.

A flagged hand-window contributes **missing values**, not zero tremor. The other hand can still contribute. We retain the reason and the usable-window count. A stable force is not rejected simply for having little variability; the flatness threshold only catches effectively constant channels.

Absolute force uses magnitude because channel signs vary. Sensor offsets could still make an unused hand look active. Low force does not prove non-use, and an offset does not prove contact. Threshold sensitivity (0.001/0.002/0.005 and hand ratios 0.05/0.10/0.20) is included. A confirmed no-load recording and supervised hand-use annotation would improve this step.

## 8. How should I read a spectrogram?

The horizontal axis is time, the vertical axis is frequency, and colour represents PSD. Brighter means more power. All panels in a comparison use one colour scale.

- A persistent horizontal ridge suggests a relatively stable oscillation.
- A broad vertical bright region can be a force correction or disturbance.
- Low-frequency brightness can reflect changing force control.
- An intermittent ridge can indicate an episode, but needs the raw signal and clinical context.

Our spectrogram uses 2-second Hann windows with a 0.2-second hop. Adjacent columns overlap heavily and are not independent observations. We display individual repetitions separately; joining them would create artificial jumps.

Start with the primary pair from `task_1.ipynb`. Then inspect the additional pair from `fourier_exact_force.ipynb`: that notebook selected the sick participant for prominent oscillations. It is useful for explaining a visible ridge, but cannot demonstrate population accuracy.

In the report figures, the pair labels are deliberately written in plain language as **Healthy participant** and **Sick participant**. Their pseudonymous IDs are moved to the descriptions below the figures. For the primary pair, healthy means reference-cohort membership plus a negative self-report for a relevant hand condition; sick means patient-cohort membership. These labels come from the supplied data and were not inferred from the spectrogram.

## 9. What does one participant's row represent?

The pipeline first averages valid windows within a repetition, then repetitions within a session, then sessions within a participant. This prevents a participant with more visits from dominating the group comparison. `participant_features.csv` has one row per person and separate left/right measurements.

For a first pass, inspect: slow RMS, candidate-tremor RMS, centroid, spectral spread, RMS window SD and usable fraction. Then inspect that person's spectrum and raw signal. Do not interpret many windows as many independent people.

The classifier uses the larger of the two assessable hand RMS values so one affected side is not diluted. This is distinct from the mean-of-hands value used in the cohort scatter plot; both definitions are stated where used.

## 10. Can we classify healthy and sick people now?

We can explore **healthy-versus-sick group separation**, but cannot claim validated tremor diagnosis. Some control-recruitment records have questionnaire exceptions, diagnoses and tremor severity are unavailable, and the groups differ in age, recording period and scenario identifiers. A classifier may learn those differences.

The included baseline learns a single amplitude threshold on four training folds and predicts the held-out fifth fold. People are split, not windows or sessions. The direction and threshold are selected only from training participants. An age-only threshold is evaluated on exactly the same split. The reported AUC is a descriptive AUC of the raw feature, not a calibrated probability or an independent validation score. Bootstrap intervals resample the fixed out-of-fold predictions and do not capture all training/split variability.

A next experiment should obtain verified tremor labels, confirm that game settings are comparable, use age-matched groups or adjustment, and reserve people and ideally a collection site/protocol for final testing. Any learned scaling, feature selection and tuning must remain inside training folds. Spectrogram models would also need participant-separated splits and enough independent people; thousands of overlapping image patches do not create thousands of participants.

## A five-minute explanation to the teacher

1. “We extended the two-person Fourier example to the full indexed cohort and retained separate hands.”
2. “We separated frequency ranges as a working model, with a boundary band and alternative cutoff checks.”
3. “We extracted band RMS, mean frequency, spectral spread and across-window variability for each person.”
4. “We averaged finger powers to avoid cancellation and marked low-use windows as missing evidence.”
5. “The exploratory classification is confounded: age alone performs similarly. The next step is validation with comparable protocols and verified labels.”

## Questions you should be able to answer

**Does a 4 Hz peak mean Parkinson's disease?** No. It is an oscillation rate; diagnosis needs more evidence.

**Why stop at 15 Hz?** That is the requested scope. Higher-frequency sensor content is not used for the reported features.

**Why resample?** FFT assumes regularly spaced samples. The packet timestamps have duplicates and jitter; we average duplicates, fill only short gaps, then anti-alias before reducing to 100 Hz. The packet rate is not necessarily the ADC conversion rate.

**Does filtering perfectly reconstruct two sources?** No. The displayed filters have transition bands, and there are ambiguous frequencies and residual noise. The low-pass trace includes the held force level; the PSD-based slow RMS excludes DC and linear trend, so they are related but not identical quantities.

**Why can the tremor fraction be smaller in the sick group?** It is a ratio: very large slow variation can increase its denominator. Absolute tremor-band RMS and the fraction answer different questions.
