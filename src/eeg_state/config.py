"""Central configuration for the EEG neural-state dynamics pipeline.

Every methodological parameter lives here so changes propagate consistently
across notebooks and modules. See docs/decisions.md for the justification
behind each value. Values marked OPEN are provisional pending literature
review and QC on a development subset.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_DERIV = REPO_ROOT / "data" / "derivatives"
RESULTS = REPO_ROOT / "results"

# --- Subjects --------------------------------------------------------------
ALL_SUBJECTS = [
    "sub-04", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-15", "sub-16", "sub-17",
    "sub-18", "sub-19", "sub-21", "sub-22", "sub-23", "sub-24", "sub-25",
    "sub-26", "sub-27", "sub-28", "sub-29", "sub-30", "sub-31", "sub-32",
    "sub-33", "sub-34", "sub-35", "sub-36", "sub-37", "sub-38", "sub-39",
    "sub-40", "sub-41", "sub-42", "sub-43", "sub-44", "sub-45", "sub-46",
    "sub-47", "sub-48", "sub-49", "sub-50", "sub-51", "sub-52",
]
# Development subset for pipeline validation and the HMM fit within the time
# budget. Chosen to span load and accuracy extremes from participants.tsv.
DEV_SUBJECTS = ["sub-04", "sub-05", "sub-10", "sub-11", "sub-17",
                "sub-24", "sub-34", "sub-45"]

SESSION = "ses-01"
TASK = "STERNBERG"

# --- Preprocessing ---------------------------------------------------------
RAW_SFREQ = 1000.0
TARGET_SFREQ = 250.0

# 1 Hz high-pass: standard recommendation to stabilize ICA and remove drift
# (Winkler et al. 2015, Psychophysiology). Also the low end of our band of
# interest for working-memory EEG (theta ~ 4-7 Hz).
HIGHPASS_HZ = 1.0

# 45 Hz low-pass: preserves theta through low gamma, well above the analysis
# band. Nyquist at 250 Hz is 125 Hz, so 45 Hz is comfortably sampled.
LOWPASS_HZ = 45.0

# 50 Hz notch: dataset reports PowerLineFrequency = 50.0. Applied only if
# the PSD on the dev subset shows a visible peak. OPEN.
NOTCH_HZ: float | None = 50.0   
NOTCH_HARMONIC_HZ: float | None = 100.0   # second harmonic, visible in PSD

# Bad channel detection: robust z-score on broadband variance.
BAD_CHANNEL_Z_THRESHOLD = 4.0

# Category-specific ICLabel rejection thresholds. A component is excluded
# when the predicted probability of its ICLabel category exceeds the
# threshold for that category. This follows the per-class design of the
# original ICLabel paper (Pion-Tonachini et al. 2019, NeuroImage) and
# Miyakoshi's EEGLAB pipeline: stereotyped artifact categories (eye,
# heart, line noise, channel noise) are rejected aggressively; muscle is
# moderately rejected; the "other" category, which absorbs ambiguous
# components, is rejected only when the classifier is very confident;
# brain components are never rejected.
ICLABEL_REJECT_PROBS = {
    "brain": 1.1,
    "eye blink": 0.5,
    "muscle artifact": 0.7,
    "heart beat": 0.5,
    "line noise": 0.5,
    "channel noise": 0.5,
    "other": 0.9,
}

# ICA: 99% explained variance, Picard with extended=True. Picard is a fast
# equivalent to extended Infomax (Ablin et al. 2018, IEEE TSP); extended mode
# handles sub- and super-Gaussian sources, which is appropriate for EEG
# (eye blinks and muscle are super-Gaussian; ongoing neural activity is
# closer to Gaussian/sub-Gaussian).
ICA_N_COMPONENTS: float | int | None = 0.999999
ICA_METHOD = "picard"
ICA_FIT_PARAMS = {"extended": True, "ortho": False}
ICA_LOWPASS_HZ = 100.0   # ICA copy band; see docs/decisions.md

# --- Model representation --------------------------------------------------
# Time-delay embedding: +/-25 samples at 250 Hz = ~200 ms window (Vidaurre 2016; Quinn 2018; Rossi 2023 all use 100–250 ms windows)
TDE_LAGS = list(range(-25, 26))

# PCA dimensionality retained before HMM fitting. OPEN.
N_PCS = 20

# Number of latent states. OPEN, will be selected via held-out likelihood.
K_STATES = 6

RANDOM_SEED = 42