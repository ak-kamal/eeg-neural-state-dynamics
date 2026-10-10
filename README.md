# Latent Neural-State Dynamics in High-Density EEG During Working Memory

An investigation of how the brain moves between recurring electrophysiological
states during a Sternberg working-memory task, using unsupervised TDE-HMM
discovery of latent states and post-hoc characterization against cognitive load,
task phase, and behavior.

## Status

**In progress.** Dataset audited. Preprocessing pipeline being implemented.
See "Future Work" for what this project does not cover.

## Research Question

How does the brain move between recurring electrophysiological states as
working-memory demand increases, and how does that organization relate to
cognitive load, task phase, and behavioral performance?

## Motivation

Conventional frequency-band analyses of working-memory EEG examine isolated
sensor–frequency–phase combinations (e.g., frontal theta at FCz during
retention). This project asks a different question: whether the brain
occupies recurring, whole-scalp, spatio-spectral configurations, and how
the temporal structure of transitions between those configurations changes
with cognitive demand.

The approach is deliberately analysis-first. States are discovered from
EEG alone, without experimental labels; load, phase, accuracy, and reaction
time are revealed only after state discovery, so that downstream tests are
confirmatory rather than circular.

## Dataset

ds005095 ("Sternberg Difficult"; Zhozhikashvili et al., 2024), available
on NEMAR. 48 participants, 63-channel BrainVision EEG at 1000 Hz,
reference FCz (not recorded among the 63 channels), ground FPz,
50 Hz mains. Participants completed 200 trials of a Sternberg
working-memory task at set sizes 3/6/9/12/15 letters (40 trials each).
Encoding, retention, and retrieval phases are 1.5 s, 2.0 s, and 1.5 s
respectively; the probe disappears after 1.5 s but response time is
unlimited. Triggers mark trial onset only; retention and retrieval onsets
are reconstructed from the fixed timing. Each event is duplicated in the
events TSV at identical onsets with durations 0.0 and 0.001 s and is
deduplicated before use. Raw data under `data/raw/` are never modified.

## Method

raw BIDS EEG
↓
bad-channel detection
↓
robust reference handling
↓
ICA on a wider-band training copy
↓
ICLabel-based artifact identification
↓
apply ICA solution to analysis copy
↓
interpolate bad channels
↓
zero-phase FIR filter (1–45 Hz)
↓
downsample 1000 → 250 Hz
↓
trial reconstruction (each trial its own HMM sequence)
↓
time-delay embedding (±7 samples, ~56 ms window at 250 Hz)
↓
PCA
↓
TDE-HMM (Gaussian emissions, Baum–Welch)
↓
state characterization (occupancy, dwell, switching, transitions)
↓
post-hoc relation to load / phase / accuracy / RT

### Batch preprocessing and cohort QC

After the pipeline was frozen on the single-subject walkthrough
(`notebooks/02_preprocessing_qc.ipynb`), it was applied to the eight
subjects in `config.DEV_SUBJECTS` via `scripts/batch_preprocess.py`. The
script writes one cleaned derivative per subject to
`data/derivatives/<subject>/` and a cohort-level QC table to
`results/metrics/batch_preprocessing_qc.csv`. The QC table records, per
subject: number of channels flagged as bad by the variance detector, data
rank after average referencing, number of ICA components fitted and
excluded, trial count and accuracy, and the 95th percentile of absolute
amplitude in the cleaned signal.

Eight of eight subjects processed successfully. Three subjects (`sub-17`,
`sub-34`, `sub-45`) had bad channels flagged (4, 1, and 2 respectively),
which were excluded from average referencing and interpolated after ICA;
the remaining five had no channels flagged. In every subject, ICA retained
the full rank of the referenced data — no silent dimensionality truncation
occurred. See the QC table for per-subject values.

## Key Design Decisions

*(This section records choices where the pipeline deviates from the most
obvious default, or where a specific parameter needs justification.)*

### Filter bands and ICA copy architecture

The analysis copy is filtered to 1–45 Hz. The ICA training copy is filtered
to 1–100 Hz and resampled to 250 Hz before Picard is fitted. Rationale:

- **1 Hz high-pass** follows Winkler et al. (2015, *Psychophysiology*),
  who showed that higher cutoffs distort ERPs while lower cutoffs leave
  enough drift to destabilize ICA.
- **100 Hz upper bound on the ICA copy** retains muscle energy (muscle
  artifacts carry substantial power above 45 Hz), improving their
  separation from neural sources.
- **250 Hz sampling of the ICA copy** matches the MNE-ICALabel reference
  workflow and ICLabel's own training pipeline (Pion-Tonachini et al.
  2019, *NeuroImage*). It also reduces Picard's per-iteration cost by ~4×.
- The unmixing matrix learned on the 250 Hz copy is applied to the full-rate
  analysis copy. ICA learns a spatial unmixing matrix that is
  sample-rate independent, so this preserves the analysis band intact.

### Notch filtering before ICLabel

The ICA training copy is notch-filtered at 50 Hz and its second harmonic
(100 Hz) before ICLabel classification. This deviates from the MNE-ICALabel
tutorial (which does not notch), but is supported by the EEGLAB community's
long-standing guidance that unremoved line noise can occupy multiple ICA
dimensions, starving the decomposition of the degrees of freedom needed to
separate neural sources. Empirically, on `sub-04`, notching reduced
line-noise-classified components from 12 to 5 (19.4% → 8.1% of the
decomposition) and produced a distribution of component categories closer
to published benchmarks.

### ICLabel rejection threshold

Components are excluded when the predicted probability of a non-brain class
is ≥ 0.8. This follows the threshold used in recent published pipelines
using the same stack (BrainVision recordings → Picard ICA → ICLabel) and is
consistent with the reference implementation from the label's original
authors (who use 0.7 for the brain-probability side of the symmetric
criterion).

### Bad-channel detection is conservative

Bad channels are flagged by a robust MAD-based z-score on log-variance with
threshold 4.0. This will not produce false positives, but it also does not
detect subtle defects — channels whose variance is within normal range but
whose signal is not coherent with their neighbours. Two subjects in the
development subset (`sub-05`, `sub-10`) contain such channels; their ICs are
identified as channel noise by ICLabel and excluded downstream. See
"What This Project Does Not Claim" below for the implications.

## Repository Structure

*(To be filled at end, mirroring Project 2.)*

## What This Project Does Not Claim

- **Not a source-localization study.** EEG topographies are described in
  sensor space. Statements such as "State 4 shows a frontocentral scalp
  pattern" are supported; statements such as "State 4 reflects a
  frontoparietal cortical network" are not, and are avoided throughout.

- **Bad-channel detection is not exhaustive.** A conservative
  variance-based detector is used; subtle defects (channels with normal
  variance but poor coherence with neighbouring electrodes) are not
  flagged. In the development subset this affected `sub-05` and `sub-10`,
  where the affected channels' ICs were identified as channel noise by
  ICLabel and excluded before the state model. Per-channel amplitude
  comparisons across subjects should be interpreted with this in mind.

- **A small average-reference bias is present.** The dataset was recorded
  against FCz, which is not among the 63 recorded channels. The average
  reference is therefore computed over 63 electrodes rather than the 64
  that the original montage would imply. The resulting bias is a DC offset
  of order |V_FCz|/64 ≈ 0.15–0.5 µV applied uniformly to every channel.
  Because the pipeline high-passes at 1 Hz, this DC component is removed
  before any analysis step that consumes the data. The bias is therefore
  noted for completeness rather than corrected.

- **The result is a methodological demonstration, not a definitive finding
  about working memory.** Full-cohort analysis, source localization, and
  direct comparison with published Sternberg findings are outside the scope
  of this project.

## Reproducibility

*(To be filled at end.)*