# Feature-table dictionary

`participant_features.csv`: one row per participant; each side-specific column begins with `left_` or `right_`. All amplitudes are in relative device units, powers in unit², frequencies in Hz. Blank numeric fields mean missing/unassessable, never zero tremor.

- `patient_id`: pseudonymous participant key; read as text.
- `cohort`, `source_cohort`: recruitment/source membership; not tremor diagnosis.
- `label_status`: questionnaire screening status for reference participants, or patient-cohort membership.
- `eligible_comparison`: whether the person is eligible for the exploratory patient/reference baseline.
- `number_of_sessions`: deduplicated selected Balanced Force sessions.
- `slow_rms`: RMS from PSD bins 0.2 ≤ f < 3 Hz.
- `boundary_rms`: RMS from 3 ≤ f < 4 Hz.
- `tremor_rms`: RMS from 4 ≤ f ≤ 15 Hz.
- `tremor_4_12_rms`, `upper_rms`: 4 ≤ f < 12 Hz and 12 ≤ f ≤ 15 Hz.
- `centroid_hz`: power-weighted frequency in the 4–15 Hz band.
- `spectral_spread_hz`: weighted SD of frequencies in that band, within windows, then aggregated.
- `peak_hz`: highest PSD bin in 4–15 Hz, then averaged over the aggregation hierarchy; it need not itself be a bin or a coherent physiological oscillator.
- `tremor_fraction`: tremor-band power / (slow + boundary + tremor power), averaged across windows and hierarchy, not a ratio of final averaged powers.
- `tremor_rms_window_sd`: sample SD across all valid hand-windows; includes repeat/session variation.
- `centroid_window_sd_hz`, `peak_window_sd_hz`: sample SD of those window descriptors.
- `valid_windows`, `total_windows`, `usable_fraction`: observed evidence after/before activity screening.
- `status`: usable (≥4 windows), limited_windows (1–3), unassessable (0).

Primary results use 5 s windows and windows→repetition→session→participant equal-weight aggregation. The long tables contain 3/5/10 s variants and more diagnostic fields. The standard deviations pool valid windows and are descriptive; they do not apply the hierarchical averaging weights. `mean_signal_to_mean_power_ratio` in the long table compares band power after waveform averaging to band power after PSD averaging; low values quantify potential cancellation, not disease.

Detailed output naming and sources are in README.md. Exact parameters and versions are in run_summary.json. Four independent repetitions are expected per selected session and were observed in this run.
