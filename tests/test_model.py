"""Tests for src/eeg_state/model.py.

Three stages can silently produce wrong results: time-delay embedding
(indexing bugs), PCA (shape/orientation bugs), and HMM fitting (lengths
handling). Synthetic-data tests use a known 3-state HMM whose state means
are well separated, so the fitted model should recover the true structure
up to permutation — the strongest class of test for a latent-variable
model, because it validates the whole pipeline end to end.
"""

import numpy as np
import pytest
from hmmlearn import hmm

from eeg_state import config, model


# --- Time-delay embedding --------------------------------------------------

def test_tde_trims_correct_number_of_samples():
    data = np.arange(100, dtype=float)[np.newaxis, :]
    lags = list(range(-7, 8))
    embedded = model.time_delay_embed(data, lags=lags)
    assert embedded.shape == (15, 86)   # 100 - 7 - 7


def test_tde_linear_ramp_reproduces_expected_values():
    """On x[t] = t, the TDE columns must contain exact consecutive integers."""
    n = 50
    data = np.arange(n, dtype=float)[np.newaxis, :]
    embedded = model.time_delay_embed(data, lags=[-2, -1, 0, 1, 2])
    # Column 0 corresponds to (lag=-2 at t=0, lag=-1 at t=1, ..., lag=+2 at t=4)
    np.testing.assert_array_equal(embedded[:, 0], [0, 1, 2, 3, 4])


def test_tde_block_ordering():
    """Channel blocks are nested inside lag blocks, not the other way around."""
    ch0 = np.arange(20, dtype=float)
    ch1 = np.arange(20, dtype=float) + 100
    data = np.stack([ch0, ch1])          # (2, 20)
    embedded = model.time_delay_embed(data, lags=[0, 1])
    assert embedded.shape == (4, 19)
    np.testing.assert_array_equal(embedded[0], ch0[0:19])   # lag=0, ch0
    np.testing.assert_array_equal(embedded[1], ch1[0:19])   # lag=0, ch1
    np.testing.assert_array_equal(embedded[2], ch0[1:20])   # lag=1, ch0
    np.testing.assert_array_equal(embedded[3], ch1[1:20])   # lag=1, ch1


def test_tde_sinusoid_is_low_rank():
    """A pure sinusoid embeds to a rank-2 manifold."""
    fs = 250.0
    t = np.arange(500) / fs
    sig = np.sin(2 * np.pi * 10.0 * t)[np.newaxis, :]
    embedded = model.time_delay_embed(sig)
    _, s, _ = np.linalg.svd(embedded, full_matrices=False)
    assert s[0] > 10 * s[2], f"Unexpected singular values: {s[:5]}"


def test_tde_raises_on_short_signal():
    data = np.arange(10, dtype=float)[np.newaxis, :]
    with pytest.raises(ValueError, match="too short"):
        model.time_delay_embed(data, lags=list(range(-7, 8)))


# --- PCA -------------------------------------------------------------------

def test_pca_returns_requested_components():
    X = np.random.default_rng(0).standard_normal((100, 50))
    pca = model.fit_pca(X, n_components=10)
    assert pca.n_components_ == 10


def test_pca_reduces_dimension_correctly():
    # API convention: X is (n_features, n_samples), matching TDE output.
    X = np.random.default_rng(0).standard_normal((50, 200))
    pca = model.fit_pca(X, n_components=15)
    X_red = model.transform_pca(X, pca)
    assert X_red.shape == (200, 15)


def test_pca_matches_sklearn_directly():
    """Our wrapper must be numerically identical to calling sklearn on X.T."""
    from sklearn.decomposition import PCA
    X = np.random.default_rng(0).standard_normal((100, 40))
    pca_ours = model.fit_pca(X, n_components=10, random_state=42)
    X_ours = model.transform_pca(X, pca_ours)
    X_ref = PCA(n_components=10, random_state=42).fit_transform(X.T)
    np.testing.assert_allclose(X_ours, X_ref, atol=1e-10)


# --- Pooled data preparation ----------------------------------------------

def test_pooled_data_lengths_sum_to_total_samples():
    rng = np.random.default_rng(0)
    subject_trials = {
        "sub-A": [rng.standard_normal((3, 100)) for _ in range(4)],
        "sub-B": [rng.standard_normal((3, 150)) for _ in range(3)],
    }
    X, lengths, _ = model.prepare_pooled_data(subject_trials)
    assert len(lengths) == 7
    assert sum(lengths) == X.shape[0]


def test_pooled_lengths_match_tde_trimming():
    """Each trial contributes n_times - window_size samples after TDE."""
    rng = np.random.default_rng(0)
    n_times = 100
    subject_trials = {"sub-A": [rng.standard_normal((3, n_times))]}
    _, lengths, _ = model.prepare_pooled_data(subject_trials)
    expected = n_times - max(config.TDE_LAGS) - (-min(config.TDE_LAGS))
    # For lags spanning -7..7, expected = 100 - 14 = 86
    assert lengths[0] == 86


def test_pooled_pca_is_reused_when_provided():
    rng = np.random.default_rng(0)
    subject_trials = {"sub-A": [rng.standard_normal((3, 100)) for _ in range(4)]}
    X1, _, pca1 = model.prepare_pooled_data(subject_trials)
    X2, _, pca2 = model.prepare_pooled_data(subject_trials, pca=pca1)
    np.testing.assert_allclose(X1, X2, atol=1e-10)
    assert pca2 is pca1


# --- HMM fitting -----------------------------------------------------------

def _make_synthetic_3state_data(n_samples=3000, n_features=5, seed=0):
    """Generate a 3-state Gaussian HMM sequence with well-separated means.

    States have orthogonal means at distance 15 from the origin (15 sigma
    separation given unit covariance), so any reasonable EM procedure will
    recover them. Data is generated directly with numpy rather than
    hmmlearn.sample() because the latter's behaviour on manually-populated
    but unfitted models depends on version internals.
    """
    rng = np.random.default_rng(seed)
    means = np.zeros((3, n_features))
    means[0, 0] = 15.0
    means[1, 1] = 15.0
    means[2, 2] = 15.0

    transmat = np.full((3, 3), 0.02)
    np.fill_diagonal(transmat, 0.96)

    states = np.zeros(n_samples, dtype=int)
    states[0] = int(rng.integers(0, 3))
    for t in range(1, n_samples):
        states[t] = int(rng.choice(3, p=transmat[states[t - 1]]))

    X = means[states] + rng.standard_normal((n_samples, n_features))
    return X, states, means


def test_fit_hmm_recovers_synthetic_state_means():
    """The fitted model must recover the true state means up to permutation.

    Baum-Welch converges to local optima from random initializations, so we
    try multiple restarts and check the best by log-likelihood. This mirrors
    standard practice in HMM fitting (Vidaurre et al. 2016 fit each subject's
    model from multiple initializations for the same reason).
    """
    X, _, true_means = _make_synthetic_3state_data(n_samples=5000)

    best_model = None
    best_ll = -np.inf
    for seed in range(5):
        fitted = model.fit_hmm(
            X, lengths=[len(X)], n_states=3, n_iter=200, random_state=seed,
        )
        ll = fitted.score(X)
        if ll > best_ll:
            best_ll = ll
            best_model = fitted

    for true_mean in true_means:
        dists = np.linalg.norm(best_model.means_ - true_mean, axis=1)
        assert dists.min() < 1.0, (
            f"Failed to recover true mean {true_mean}. "
            f"Closest fitted mean distance: {dists.min():.3f}. "
            f"Fitted means:\n{best_model.means_}"
        )


def test_fit_hmm_parameters_have_correct_shapes():
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    fitted = model.fit_hmm(X, lengths=[500], n_states=4, n_iter=20)
    assert fitted.means_.shape == (4, 5)
    assert fitted.transmat_.shape == (4, 4)
    assert fitted.startprob_.shape == (4,)
    assert fitted.covars_.shape == (4, 5, 5)


def test_fit_hmm_transition_matrix_is_stochastic():
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    fitted = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=50)
    np.testing.assert_allclose(fitted.transmat_.sum(axis=1), np.ones(3), atol=1e-6)
    assert (fitted.transmat_ >= 0).all()
    assert (fitted.transmat_ <= 1).all()


def test_fit_hmm_reproducible_with_seed():
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    m1 = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=20, random_state=42)
    m2 = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=20, random_state=42)
    np.testing.assert_array_equal(m1.means_, m2.means_)
    np.testing.assert_array_equal(m1.transmat_, m2.transmat_)


# --- State time course -----------------------------------------------------

def test_state_time_course_returns_valid_states():
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    fitted = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=50)
    viterbi, posteriors = model.get_state_time_course(fitted, X)
    assert viterbi.shape == (500,)
    assert viterbi.min() >= 0 and viterbi.max() < 3
    assert posteriors.shape == (500, 3)


def test_state_time_course_posteriors_sum_to_one():
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    fitted = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=50)
    _, posteriors = model.get_state_time_course(fitted, X)
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(500), atol=1e-6)


def test_state_time_course_with_lengths_preserves_total_length():
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    fitted = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=50)
    v_split, p_split = model.get_state_time_course(fitted, X, lengths=[100] * 5)
    assert v_split.shape == (500,)
    assert p_split.shape == (500, 3)


def test_state_time_course_uses_trial_boundaries(monkeypatch):
    """get_state_time_course must call model.decode once per trial when
    lengths is provided, not once for the whole sequence.

    Directly counting decode calls is more reliable than trying to construct
    synthetic data where the two decoders give different Viterbi paths,
    which turns out to be very hard to do (data evidence dominates the
    boundary effect in most configurations).
    """
    X, _, _ = _make_synthetic_3state_data(n_samples=500)
    fitted = model.fit_hmm(X, lengths=[500], n_states=3, n_iter=20)

    call_count = {"n": 0}
    original_decode = fitted.decode

    def counting_decode(*args, **kwargs):
        call_count["n"] += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(fitted, "decode", counting_decode)

    model.get_state_time_course(fitted, X, lengths=[100, 100, 100, 100, 100])
    assert call_count["n"] == 5, (
        f"model.decode called {call_count['n']} times; expected 5 "
        f"(one per trial). The lengths argument appears to be ignored."
    )

    # Sanity: with lengths=None, decode should be called exactly once.
    call_count["n"] = 0
    model.get_state_time_course(fitted, X)
    assert call_count["n"] == 1, (
        f"model.decode called {call_count['n']} times without lengths; "
        f"expected 1."
    )


# --- Integration (slow) ----------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_on_preprocessed_subject():
    """Load a preprocessed subject and run TDE -> PCA -> HMM end to end."""
    import mne
    from eeg_state.io import load_events, reconstruct_trials

    subject = "sub-04"
    fif = config.DATA_DERIV / subject / f"{subject}_preprocessed_raw.fif"
    if not fif.exists():
        pytest.skip(f"{fif} not present; run scripts/batch_preprocess.py first")

    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    events = reconstruct_trials(load_events(subject))
    sfreq = raw.info["sfreq"]
    data = raw.get_data()

    trial_arrays = []
    for _, trial in events.iterrows():
        start = int(trial["encoding_onset_s"] * sfreq)
        end = int(trial["retrieval_onset_s"] * sfreq)
        if end <= data.shape[1] and (end - start) > 2 * max(config.TDE_LAGS):
            trial_arrays.append(data[:, start:end])

    assert len(trial_arrays) > 0

    X, lengths, _ = model.prepare_pooled_data({subject: trial_arrays})
    assert X.shape[1] == config.N_PCS
    assert sum(lengths) == X.shape[0]

    fitted = model.fit_hmm(X, lengths, n_states=4, n_iter=30)
    viterbi, posteriors = model.get_state_time_course(fitted, X, lengths)

    assert viterbi.shape == (X.shape[0],)
    assert posteriors.shape == (X.shape[0], 4)
    assert (viterbi >= 0).all() and (viterbi < 4).all()
    

def test_pooled_data_is_standardized():
    """Pooled PCA output must have zero mean and unit variance per feature.

    Full-covariance HMMs are numerically unstable on unstandardized
    volt-scale EEG PCA outputs; this test guards against removing the
    standardize step by accident.
    """
    rng = np.random.default_rng(0)
    subject_trials = {"sub-A": [rng.standard_normal((3, 100)) * 1e-6 for _ in range(4)]}
    X, _, _ = model.prepare_pooled_data(subject_trials)
    np.testing.assert_allclose(X.mean(axis=0), np.zeros(X.shape[1]), atol=1e-10)
    np.testing.assert_allclose(X.std(axis=0), np.ones(X.shape[1]), atol=1e-10)