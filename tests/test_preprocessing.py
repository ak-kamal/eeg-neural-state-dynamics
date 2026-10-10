"""Regression and sanity tests for src/eeg_state/preprocessing.py.

Two categories of test:

1.  Invariant tests — mathematical/structural properties that must hold for
    any correctly functioning pipeline (rank preserved through average
    reference, ICA retains full rank, filtered output has the expected
    sampling rate and duration). These catch silent parameter errors such
    as the `n_components=0.99` truncation that starved an earlier run.

2.  Pinned regression tests — counts and statistics tied to sub-04. These
    catch accidental changes to pipeline ordering or configuration.

The invariant tests are the more important of the two: they would have
flagged the 26-of-62 component bug on the first execution, before any of
the downstream preprocessing stages even ran.
"""

import json

import mne
import numpy as np
import pytest

from eeg_state import config
from eeg_state.io import load_raw
from eeg_state import preprocessing as pp

SUB = "sub-04"


# --- Fixtures --------------------------------------------------------------

@pytest.fixture(scope="module")
def raw():
    return load_raw(SUB, preload=True)


@pytest.fixture(scope="module")
def raw_hp(raw):
    return pp.highpass(raw)


@pytest.fixture(scope="module")
def raw_ref(raw_hp):
    return pp.set_average_reference(raw_hp.copy())


@pytest.fixture(scope="module")
def ica_copy(raw_ref):
    return pp.prepare_ica_copy(raw_ref, bad_channels=[])


# --- Invariant tests -------------------------------------------------------

def test_average_reference_reduces_rank_by_one(raw_hp):
    """After average reference, rank should be n_channels - 1.

    A single linear dependency (sum of referenced channels = 0) is
    introduced. Any other rank deficiency indicates an additional
    constraint we did not intend.
    """
    raw_hp.info["bads"] = []
    ref = pp.set_average_reference(raw_hp.copy())
    rank = mne.compute_rank(ref, rank=None)["eeg"]
    assert rank == len(ref.ch_names) - 1, (
        f"Expected rank {len(ref.ch_names) - 1}, got {rank}. "
        f"Additional rank deficiency detected."
    )


def test_ica_retains_full_rank(ica_copy):
    """ICA must not silently truncate the decomposition.

    This is the regression test for the `n_components=0.99` bug: with
    n_components=0.99 on 1-100 Hz filtered EEG, MNE selected only 26 of
    the 62 available dimensions, causing ICLabel to over-reject mixed
    components. The fix is n_components=0.999999 (MNE's own default when
    None), which retains the full rank.

    Loads the cached ICA produced by notebook 02 rather than refitting,
    so this test runs in milliseconds.
    """
    ica_cache = config.DATA_DERIV / SUB / "cache" / f"{SUB}_ica.fif"
    if not ica_cache.exists():
        pytest.skip(
            f"cached ICA {ica_cache} not present; run notebook 02 first"
        )
    ica = mne.preprocessing.read_ica(ica_cache, verbose="ERROR")
    rank = mne.compute_rank(ica_copy, rank=None)["eeg"]
    assert ica.n_components_ >= rank - 2, (
        f"ICA retained only {ica.n_components_} of {rank} dimensions. "
        f"Check config.ICA_N_COMPONENTS — it should be 0.999999 or None."
    )


def test_ica_copy_is_downsampled(ica_copy):
    """The ICA training copy must be downsampled to the configured target.

    Fitting Picard on 1000 Hz data was 4x slower with no gain in
    decomposition quality; the training copy is resampled to 250 Hz in
    prepare_ica_copy.
    """
    assert ica_copy.info["sfreq"] == config.TARGET_SFREQ


def test_ica_copy_band_extends_to_100hz(raw, ica_copy):
    """ICA copy is high-passed at ~1 Hz and retains energy up to ~100 Hz."""
    psd_ica = ica_copy.compute_psd(fmax=120, n_fft=2048, verbose="ERROR")
    freqs = psd_ica.freqs
    ica_data = psd_ica.get_data().mean(axis=0)

    # Upper bound: energy must still be present at 90 Hz (muscle band).
    high_band = (freqs >= 85) & (freqs <= 95)
    assert ica_data[high_band].mean() > 0, (
        "ICA copy has no energy above 85 Hz; check ICA_LOWPASS_HZ."
    )

    # Lower bound: sub-1 Hz power must be strongly attenuated relative to
    # the raw, unfiltered signal. We compare the same frequency band across
    # the two recordings rather than comparing low vs. mid bands within one
    # recording — EEG's 1/f spectrum means low-frequency power dominates
    # even after high-passing, so within-signal comparisons are uninformative.
    psd_raw = raw.compute_psd(fmax=120, n_fft=4096, verbose="ERROR")
    raw_data = psd_raw.get_data().mean(axis=0)

    low_band_ica = (freqs > 0.1) & (freqs < 0.5)
    low_band_raw = (psd_raw.freqs > 0.1) & (psd_raw.freqs < 0.5)

    ratio = ica_data[low_band_ica].mean() / raw_data[low_band_raw].mean()
    assert ratio < 0.3, (
        f"High-pass did not adequately suppress sub-1 Hz power "
        f"(filtered/raw power ratio at 0.1-0.5 Hz = {ratio:.3f}; expected < 0.3)."
    )


def test_downsample_preserves_duration(raw_ref):
    """Downsampling must not change the recording duration."""
    original_duration = raw_ref.times[-1]
    ds = pp.downsample(raw_ref.copy())
    assert abs(ds.times[-1] - original_duration) < 0.01  # within 10 ms
    assert ds.info["sfreq"] == config.TARGET_SFREQ


def test_lowpass_attenuates_above_cutoff(raw_ref):
    """After 45 Hz low-pass, power above 60 Hz must be negligible."""
    lp = pp.lowpass(raw_ref.copy())
    psd = lp.compute_psd(fmax=125, n_fft=4096, verbose="ERROR")
    freqs = psd.freqs
    data = psd.get_data().mean(axis=0)

    power_above = data[freqs > 60].mean()
    power_below = data[(freqs >= 5) & (freqs <= 30)].mean()
    assert power_above < power_below / 100, (
        "Low-pass did not sufficiently attenuate frequencies above 60 Hz."
    )


# --- Pinned regression tests -----------------------------------------------

def test_bad_channel_detector_returns_list(raw_hp):
    """Detector returns a list of channel names (empty is a valid result)."""
    bads = pp.detect_bad_channels(raw_hp)
    assert isinstance(bads, list)
    for ch in bads:
        assert ch in raw_hp.ch_names


def test_sub04_qc_metrics_stable():
    """End-to-end QC metrics for sub-04, pinned.

    If any of these numbers change, either the pipeline changed or the
    data changed — both warrant investigation before batch processing.
    """
    # We do not re-run ICA in this test (too slow); instead, verify the
    # cached derivative that the notebook produced.
    fif = config.DATA_DERIV / SUB / f"{SUB}_preprocessed_raw.fif"
    if not fif.exists():
        pytest.skip(f"cached derivative {fif} not present; run notebook 02 first")
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")

    assert raw.info["sfreq"] == config.TARGET_SFREQ
    assert len(raw.ch_names) == 63
    assert abs(raw.times[-1] - 1196.548) < 0.1


def test_config_consistency():
    """Configuration parameters are internally consistent."""
    assert config.HIGHPASS_HZ < config.LOWPASS_HZ
    assert config.LOWPASS_HZ < config.ICA_LOWPASS_HZ
    assert config.TARGET_SFREQ < config.RAW_SFREQ
    assert config.ICA_N_COMPONENTS is None or config.ICA_N_COMPONENTS > 0.99, (
        "ICA_N_COMPONENTS must retain nearly the full rank; "
        "the value 0.99 previously truncated the decomposition."
    )
    # The 45 Hz low-pass must be below the Nyquist of the target rate.
    assert config.LOWPASS_HZ < config.TARGET_SFREQ / 2
    
    
def test_functions_do_not_mutate_input(raw_hp):
    """Functions must return new objects, leaving the input untouched.

    This guards against the in-place vs copy inconsistency that existed in
    an earlier version of the pipeline, where callers who forgot to
    reassign the return value silently corrupted their Raw object.
    """
    original = raw_hp.copy()
    _ = pp.set_average_reference(raw_hp)
    _ = pp.lowpass(raw_hp)
    _ = pp.downsample(raw_hp)
    _ = pp.interpolate_bad_channels(raw_hp)

    # Data arrays must be identical after running all functions.
    np.testing.assert_array_equal(raw_hp.get_data(), original.get_data())
    assert raw_hp.info["sfreq"] == original.info["sfreq"]