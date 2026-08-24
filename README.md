# VitalDB early post-induction alpha and subsequent SR

This repository runs a checkpointable, outcome-blinded candidate-window
extraction pipeline on the public VitalDB API. It is designed for GitHub
Actions so collection continues when the investigator's computer is off.

## Scientific boundary

The automated pipeline identifies the earliest **technically eligible**
120-second two-channel BIS EEG window after the first recorded BIS <=60 after
propofol TCI initiation. It does **not** claim that the window has passed the
prespecified visual assessment of stable EEG dynamics. The final primary
cohort must be restricted to windows accepted by a reviewer blinded to alpha,
SR, and clinical outcomes.

VitalDB labels the waveforms `BIS/EEG1_WAV` and `BIS/EEG2_WAV`. The public
metadata does not establish that these are left and right bilateral channels;
therefore outputs use the term `two_channel` rather than `bilateral`.

## Automated eligibility rule

- Adult cases with ASA I-IV and the required public VitalDB tracks.
- Propofol TCI start: first positive `Orchestra/PPF20_CE` observation.
- Search anchor: first recorded `BIS/BIS <= 60` at or after TCI start.
- Search ends at surgical incision (`opstart`).
- 120 seconds = 60 consecutive non-overlapping 2-second epochs.
- In each channel and epoch: >=98% finite samples, absolute amplitude <=500 uV,
  peak-to-peak amplitude <=800 uV, filtered SD >=0.5 uV.
- SQI: >=90 with at least one observation in every 2-second epoch.
- No automatically detected suppression-like segment (absolute 0.5-45 Hz
  filtered amplitude <=5 uV for >=0.5 seconds) in either channel.
- Alpha magnitude, prominence, slope, and shape are never used for selection.
- Absolute spectral power is reported for delta (0.5-4 Hz), theta (4-8 Hz),
  alpha (8-12 Hz), and beta (12-25 Hz), separately for EEG1, EEG2, and the
  two-channel linear-power average, expressed in dB.

Amplitude and suppression thresholds are operational rules and require
validation against the blinded review set. They are not presented as a
substitute for expert EEG adjudication.

## Blinded review files

Each shard produces:

- `blinded_review/images/REV-*.png`: raw two-channel EEG, spectrograms, SQI and
  an extended context view. No case ID, alpha result, SR, or clinical outcome.
- `blinded_review/reviewer_form.csv`: acceptance and rejection fields.
- `restricted_key.csv`: review ID to case ID, alpha, pre-window SR and
  post-window SR mapping. Do not give this file to reviewers.

The primary reviewer reviews every candidate used in the primary analysis.
A second blinded reviewer should independently review a random validation
sample, with agreement reported separately.

## GitHub Actions

Run **VitalDB cloud EEG analysis** from the Actions tab. The matrix jobs process
deterministic shards in parallel and upload artifacts. Raw VitalDB waveforms are
not committed to GitHub and are deleted with the runner.

The workflow is intentionally `workflow_dispatch` only. This avoids repeatedly
downloading the public dataset on a timer. Failed shards can be re-run from the
Actions interface.

## Final analysis

After blinded forms are completed, add the decisions to the merged review form
and run:

```bash
python scripts/analyze_accepted.py \
  --cohort merged/cohort.csv \
  --reviews merged/reviewer_form_completed.csv \
  --key merged/restricted_key.csv \
  --out results
```

The primary model treats alpha power continuously. Quartile-specific SR
occurrence/recurrence and an incident-SR sensitivity analysis excluding cases
with pre-window SR are secondary outputs.
