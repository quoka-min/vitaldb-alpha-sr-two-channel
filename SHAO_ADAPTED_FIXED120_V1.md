# Shao-adapted BIS≤60+10 min fixed 120 s protocol v1

Protocol ID: `shao-adapted-bis60plus10m-fixed120-v1`

## Primary exposure window

1. Restrict to adults receiving general anesthesia with synchronized
   `BIS/EEG1_WAV`, `BIS/EEG2_WAV`, `BIS/BIS`, `BIS/SQI`, `BIS/SR`, and
   `Orchestra/PPF20_RATE` tracks.
2. Define propofol TCI start as the first positive PPF20_RATE.
3. Define the database landmark as the first BIS≤60 after that TCI start.  This
   is an operational proxy for entry into a post-induction state, not observed
   behavioral loss of consciousness.
4. The only primary candidate starts at the first 2-second grid point at or
   after landmark+600 s and ends 120 s later.  A rejected or unavailable window
   is not replaced by a later window.
5. Blinded visual review of both raw frontal EEG channels and both spectrograms
   determines whether both channels are interpretable and whether artifact,
   physiologic burst suppression, or a clear transition is present.  The
   reviewer is blinded to band power, manufacturer SR, and clinical outcomes.
6. SQI is summarized as a device signal-quality indicator; it is not treated as
   BIS stability and is not the primary alpha-window eligibility gate.

This is an adaptation of Shao et al. rather than a direct replication because
Shao anchored the 2-minute segment about 10 minutes after surgery start and
used four frontal channels.  The present public dataset uses a BIS landmark
and two recorded BIS EEG tracks.

## Spectrum

- Sampling: 128 Hz
- Filter: fourth-order zero-phase Butterworth 0.5–45 Hz; no 60-Hz notch
- Epochs: sixty non-overlapping 2-second epochs
- DPSS multitaper: TW=3, K=5
- Primary absolute bands: delta 0.5–4, theta 4–8, alpha 8–12, beta 13–30 Hz
- Sensitivity alpha band: 8–13 Hz
- Primary two-channel signal: samplewise `(EEG1+EEG2)/2`
- Montage/polarity sensitivity: average the two channel-level linear band
  powers before the dB transform
- Units: dB relative to 1 µV²; channel and linear-unit outputs are retained

Alpha amplitude, prominence, slope, or apparent spectral clarity is never used
to move or choose the window.

## Manufacturer SR outcomes

The primary SR observation begins 63 seconds after the fixed alpha-window end
so the manufacturer's rolling SR does not mathematically reuse the alpha
window.  It ends at operation end.

- `sr_gt10`: at least one finite manufacturer SR value strictly greater than 10
- `sr_gt20`: at least one finite manufacturer SR value strictly greater than 20
- A positive case is classifiable when the threshold is reached.
- A negative case additionally requires at least 600 observed seconds and ≥90%
  coverage of the planned observation period.
- SR AUC is integrated without bridging gaps longer than 2.5 seconds and is
  expressed as percent-minutes.
- Time-weighted mean SR equals AUC divided by integrated observation time.
- AUC above 10 and above 20 are retained.

Manufacturer SR is an electronic device endpoint.  It is not relabeled as an
independently adjudicated physiologic burst-suppression event, and it does not
require whole-cohort manual EEG review.

## Clinical outcomes

- ICU admission proxy: `icu_days>0`
- ICU length of stay: original nonnegative `icu_days`
- Postoperative hospital stay: integer calendar-day proxy from operation end
  to discharge.  Admission/discharge markers are day-level rather than exact
  clock times; small negative same-calendar-day differences are assigned zero.
- Total hospital stay: `(dis-adm)/86400`
- In-hospital death: `death_inhosp` 0/1

Source values and QC flags are retained.  VitalDB cannot distinguish planned
from unplanned ICU admission and does not provide ICU entry/exit timestamps,
death timing, or death cause.

## Analysis

Primary alpha is continuous.  Results include effects per 1-dB lower alpha and
per SD lower alpha.  Descriptive alpha quartiles report SR>10 and SR>20
incidence with Wilson 95% confidence intervals.  Minimal models include alpha
alone and alpha plus operation duration; post-index observation duration is
also evaluated because longer observation mechanically increases the chance of
detecting a binary SR threshold.  Repeated operations use patient-clustered
robust standard errors when possible.

Band-power correlations with SR maximum, AUC, and time-weighted mean use
Spearman coefficients.  ICU admission, ICU LOS among ICU users, postoperative
hospital LOS, total hospital LOS, and in-hospital death are exploratory
clinical outcomes.  Sparse events and model separation are explicitly warned.

All pre-adjudication analyses are labelled provisional.  Automated screening
must never be represented as completed blinded manual review.
