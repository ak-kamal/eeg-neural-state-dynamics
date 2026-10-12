"""Preprocessing pipeline for the Sternberg EEG dataset.

Order of operations (see docs/decisions.md for literature justification):

  1.  Load raw at 1000 Hz.
  2.  High-pass at 1 Hz (Winkler et al. 2015, Psychophysiology).
  3.  Detect bad channels via robust z-score on log-variance.
  4.  Set average reference; marked bad channels are excluded from the
      average automatically by MNE.
  5.  Fit ICA on a wider-band copy (1-100 Hz) so muscle artifacts are
      visible; Picard with extended=True, the fast equivalent of
      extended Infomax (Ablin et al. 2018, IEEE TSP).
  6.  Classify components with ICLabel (Pion-Tonachini et al. 2019,
      NeuroImage); reject any component whose non-brain probability
      exceeds the configured threshold.
  7.  Apply the ICA solution to the analysis copy.
  8.  Low-pass at 45 Hz.
  9.  Interpolate bad channels (spherical spline).
  10. Downsample to 250 Hz (MNE anti-aliases internally).
  11. Save to data/derivatives/<subject>/.

The pipeline runs on a single subject and is designed to be inspected
stage-by-stage in notebooks/02_preprocessing_qc.ipynb. The batch
function `preprocess_subject` is what scripts and later notebooks call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components

from . import config
from .io import load_raw


# --- Individual stages -----------------------------------------------------

def highpass(raw: mne.io.BaseRaw, l_freq: float = config.HIGHPASS_HZ) -> mne.io.BaseRaw:
    """Zero-phase FIR high-pass. Applied to a copy; caller keeps the original."""
    return raw.copy().filter(
        l_freq=l_freq, h_freq=None, fir_design="firwin", verbose="ERROR",
    )


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    z_threshold: float = config.BAD_CHANNEL_Z_THRESHOLD,
) -> list[str]:
    """Flag channels whose log-variance deviates from the median by more
    than `z_threshold` scaled MADs.

    Input should already be high-pass filtered (>=1 Hz) so slow drift does
    not artificially inflate variance. The MAD-based z-score is robust to a
    small number of extreme channels.
    """
    eeg = raw.copy().pick("eeg")
    data = eeg.get_data()
    log_var = np.log(np.var(data, axis=1))
    median = float(np.median(log_var))
    mad = float(np.median(np.abs(log_var - median)))
    if mad == 0.0:
        return []
    z = (log_var - median) / (1.4826 * mad)
    bad_idx = np.where(np.abs(z) > z_threshold)[0]
    return [eeg.ch_names[i] for i in bad_idx]


def set_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Average reference on a copy; bad channels in info['bads'] are excluded
    from the mean automatically (PREP guidance; Bigdely-Shamlo et al. 2015)."""
    out = raw.copy()
    out.set_eeg_reference("average", projection=False, verbose="ERROR")
    return out


def prepare_ica_copy(
    raw: mne.io.BaseRaw,
    bad_channels: list[str],
    target_sfreq: float = config.TARGET_SFREQ,
) -> mne.io.BaseRaw:
    """Return a filtered, decimated copy for ICA fitting.

    Stage order: notch → low-pass → resample. Notching first is the EEGLAB
    convention and matters mechanistically: line noise is a single physical
    artifact, but if left in place it can occupy multiple ICA dimensions
    (one per channel-adjacent projection), starving the decomposition of the
    dimensions needed to separate neural sources. Removing it before band-
    limiting prevents that.

    Filtering to 1-100 Hz and decimating to 250 Hz reduces Picard's per-
    iteration cost by ~4x with no meaningful loss of ICA-relevant
    information: the Nyquist frequency at 250 Hz (125 Hz) exceeds the top
    of the ICA band (100 Hz), and ICLabel operates on features extracted
    near 100 Hz internally (Pion-Tonachini et al. 2019, NeuroImage). The ICA
    solution is a spatial unmixing matrix and is applied unchanged to the
    full-rate analysis copy, so analysis-band fidelity is preserved.
    """
    ica_raw = raw.copy()

    # 1. Notch out mains interference (and its harmonic) if configured.
    if config.NOTCH_HZ is not None:
        freqs = [config.NOTCH_HZ]
        if config.NOTCH_HARMONIC_HZ is not None:
            freqs.append(config.NOTCH_HARMONIC_HZ)
        ica_raw.notch_filter(
            freqs=freqs,
            picks="eeg",
            fir_design="firwin",
            verbose="ERROR",
        )

    # 2. Low-pass to the top of the ICA band.
    ica_raw.filter(
        l_freq=None,
        h_freq=config.ICA_LOWPASS_HZ,
        fir_design="firwin",
        verbose="ERROR",
    )

    # 3. Decimate to the target rate for ICA fitting.
    ica_raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    # 4. Mark (but keep) bad channels; MNE excludes them from the ICA fit.
    ica_raw.info["bads"] = list(bad_channels)
    return ica_raw


def fit_ica(ica_copy: mne.io.BaseRaw) -> ICA:
    """Fit Picard-extended ICA on the EEG channels only."""
    ica = ICA(
        n_components=config.ICA_N_COMPONENTS,
        method=config.ICA_METHOD,
        fit_params=config.ICA_FIT_PARAMS,
        rng=config.RANDOM_SEED,
        max_iter=500,
    )
    ica.fit(ica_copy, picks="eeg", verbose="ERROR")
    return ica


def classify_ica_components(
    ica: ICA,
    ica_copy: mne.io.BaseRaw,
    reject_probs: dict[str, float] | None = None,
) -> tuple[dict[str, Any], list[int]]:
    """Run ICLabel and return (labels_dict, component indices to exclude).

    Category-specific thresholds are used. A component is excluded when
    its predicted ICLabel class is not "brain" and the classifier's
    probability for that class exceeds the threshold for that class.
    Thresholds default to ``config.ICLABEL_REJECT_PROBS``.
    """
    if reject_probs is None:
        reject_probs = config.ICLABEL_REJECT_PROBS

    labels = label_components(ica_copy, ica, method="iclabel")

    exclude = []
    for i, lab in enumerate(labels["labels"]):
        threshold = reject_probs.get(lab, 1.1)  # unknown category -> keep
        if labels["y_pred_proba"][i] >= threshold:
            exclude.append(i)

    return labels, exclude


def apply_ica(raw: mne.io.BaseRaw, ica: ICA) -> mne.io.BaseRaw:
    """Apply the ICA solution to a copy. `ica.exclude` selects components."""
    out = raw.copy()
    ica.apply(out, verbose="ERROR")
    return out


def lowpass(raw: mne.io.BaseRaw, h_freq: float = config.LOWPASS_HZ) -> mne.io.BaseRaw:
    """Zero-phase FIR low-pass on a copy."""
    out = raw.copy()
    out.filter(l_freq=None, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
    return out


def interpolate_bad_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Spherical-spline interpolation of channels flagged in info['bads'].
    Operates on a copy."""
    out = raw.copy()
    out.interpolate_bads(reset_bads=True, verbose="ERROR")
    return out


def downsample(raw: mne.io.BaseRaw, sfreq: float = config.TARGET_SFREQ) -> mne.io.BaseRaw:
    """Resample a copy. MNE applies an anti-aliasing filter internally."""
    out = raw.copy()
    out.resample(sfreq, npad="auto", verbose="ERROR")
    return out


# --- QC --------------------------------------------------------------------

def qc_metrics(
    raw: mne.io.BaseRaw,
    *,
    n_bad_channels: int,
    n_ica_components: int,
    n_excluded_components: int,
) -> dict[str, Any]:
    """Summarize the cleaned recording for the QC table.

    Amplitudes are reported as the 95th percentile of absolute EEG values
    in microvolts. A common heuristic for clean task EEG is <100 uV;
    higher values suggest residual artifacts.
    """
    eeg = raw.copy().pick("eeg")
    data = eeg.get_data() * 1e6  # V -> uV
    return {
        "n_channels": len(eeg.ch_names),
        "n_bad_channels": n_bad_channels,
        "n_ica_components": n_ica_components,
        "n_excluded_components": n_excluded_components,
        "sfreq_hz": float(raw.info["sfreq"]),
        "duration_s": float(raw.times[-1]),
        "p95_amplitude_uv": float(np.percentile(np.abs(data), 95)),
    }


# --- Full pipeline ---------------------------------------------------------

def preprocess_subject(
    subject: str,
    *,
    save: bool = True,
    overwrite: bool = False,
    verbose: bool = False,
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Run the full pipeline on one subject.

    Returns the cleaned Raw object and a QC dict. If `save` is True, the
    cleaned Raw is written to data/derivatives/<subject>/ as a FIF file.
    """
    if verbose:
        print(f"[{subject}] loading")
    raw = load_raw(subject, preload=True)

    if verbose:
        print(f"[{subject}] high-pass @ {config.HIGHPASS_HZ} Hz")
    raw = highpass(raw)

    if verbose:
        print(f"[{subject}] detecting bad channels")
    bad_channels = detect_bad_channels(raw)
    raw.info["bads"] = bad_channels

    if verbose:
        print(f"[{subject}] average reference (excluding {len(bad_channels)} bad)")
    raw = set_average_reference(raw)

    if verbose:
        print(f"[{subject}] preparing ICA copy")
    ica_copy = prepare_ica_copy(raw, bad_channels)

    if verbose:
        print(f"[{subject}] fitting ICA")
    ica = fit_ica(ica_copy)

    if verbose:
        print(f"[{subject}] ICLabel")
    labels, exclude = classify_ica_components(ica, ica_copy)
    ica.exclude = exclude

    if verbose:
        print(f"[{subject}] applying ICA ({len(exclude)} excluded)")
    raw = apply_ica(raw, ica)

    if verbose:
        print(f"[{subject}] low-pass @ {config.LOWPASS_HZ} Hz")
    raw = lowpass(raw)

    if verbose:
        print(f"[{subject}] interpolating {len(bad_channels)} bad channels")
    raw = interpolate_bad_channels(raw)

    if verbose:
        print(f"[{subject}] downsampling to {config.TARGET_SFREQ} Hz")
    raw = downsample(raw)

    qc = qc_metrics(
        raw,
        n_bad_channels=len(bad_channels),
        n_ica_components=ica.n_components_,
        n_excluded_components=len(exclude),
    )

    if save:
        out_dir = config.DATA_DERIV / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject}_preprocessed_raw.fif"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"{out_path} exists; pass overwrite=True")
        raw.save(out_path, overwrite=True, verbose="ERROR")
        if verbose:
            print(f"[{subject}] saved -> {out_path}")

    return raw, qc